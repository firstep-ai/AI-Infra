# 导入所需的库
import os  # 用于访问环境变量
import torch  # PyTorch 核心库
import torch.nn as nn  # PyTorch 神经网络模块
import torch.nn.functional as F  # PyTorch 神经网络函数库
import torch.distributed as dist  # PyTorch 分布式通信库

import process_group_manager as pgm  # 导入进程组管理器

### begin PP communications ###
# --- 流水线并行 (PP) 通信原语 ---

# STEP 用于调试追踪，VERBOSE 控制日志详细程度
STEP, VERBOSE = 0, os.environ.get("VERBOSE", "0") == "1"

# 单向通信函数 (与 Step 7 相同)
def pipeline_communicate(operation, device, dtype, tensor=None, shapes=None):
    """处理流水线阶段间的单向点对点通信（发送或接收）。"""
    global STEP
    global VERBOSE

    if operation == 'recv_forward':
        if pgm.process_group_manager.pp_is_first_stage: return None
        tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        src = pgm.process_group_manager.pp_prev_rank
    elif operation == 'send_forward':
        if pgm.process_group_manager.pp_is_last_stage: return
        dest = pgm.process_group_manager.pp_next_rank
    elif operation == 'recv_backward':
        if pgm.process_group_manager.pp_is_last_stage: return None
        tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        src = pgm.process_group_manager.pp_next_rank
    elif operation == 'send_backward':
        if pgm.process_group_manager.pp_is_first_stage: return
        dest = pgm.process_group_manager.pp_prev_rank

    is_send = operation.startswith('send')
    peer_rank = dest if is_send else src
    op = dist.P2POp(dist.isend if is_send else dist.irecv, tensor, peer_rank)

    if VERBOSE:
        print(f"{operation} | {'sending' if is_send else 'receiving'} {operation.split('_')[1]} "
              f"{pgm.process_group_manager.pp_rank} {'→' if is_send else '←'} {peer_rank} | "
              f"STEP:{STEP} | RANK:{pgm.process_group_manager.pp_rank}", flush=True)

    # 执行并等待 P2P 操作完成 (同步操作)
    [req.wait() for req in dist.batch_isend_irecv([op])]
    torch.cuda.synchronize()

    if VERBOSE: STEP += 1
    return tensor if not is_send else None

# 新增：双向通信函数，用于 1F1B 调度
def bidirectional_pipeline_communicate(operation, send_tensor, recv_shapes, device, dtype):
    """
    处理流水线阶段间的双向通信，允许同时发送和接收操作。
    主要用于 1F1B 调度中的稳态阶段。

    Args:
        operation (str): 双向操作类型 ('send_fwd_recv_bwd' 或 'send_bwd_recv_fwd')
                         'send_fwd_recv_bwd': 发送前向激活，接收后向梯度
                         'send_bwd_recv_fwd': 发送后向梯度，接收前向激活
        send_tensor: 要发送的张量
        recv_shapes: 期望接收的张量的形状
        device: 设备
        dtype: 数据类型

    Returns:
        torch.Tensor or None: 接收到的张量，如果在流水线终端阶段则为 None
    """
    global STEP
    global VERBOSE

    # 判断操作方向是前向还是后向
    is_fwd = (operation == 'send_fwd_recv_bwd')

    # 如果是在流水线的终端阶段执行无效操作，则跳过
    # 例如：最后一个 stage 不能 send_fwd_recv_bwd；第一个 stage 不能 send_bwd_recv_fwd
    if (is_fwd and pgm.process_group_manager.pp_is_last_stage) or \
       (not is_fwd and pgm.process_group_manager.pp_is_first_stage):
        return None

    # 根据操作方向确定通信对方的 rank
    peer_rank = pgm.process_group_manager.pp_next_rank if is_fwd else pgm.process_group_manager.pp_prev_rank

    # 创建用于接收数据的空张量
    recv_tensor = torch.empty(recv_shapes, requires_grad=True, device=device, dtype=dtype)

    # --- 关键步骤：设置同时进行的发送和接收操作 ---
    # 创建两个 P2POp 对象：一个用于发送 (isend)，一个用于接收 (irecv)
    # 将这两个操作对象放入一个列表中，传递给 dist.batch_isend_irecv
    # 这个函数会同时启动这两个操作
    reqs = dist.batch_isend_irecv([
        dist.P2POp(dist.isend, send_tensor, peer_rank), # 发送 send_tensor 给 peer_rank
        dist.P2POp(dist.irecv, recv_tensor, peer_rank)  # 从 peer_rank 接收数据到 recv_tensor
    ])

    # 可选的详细日志
    if VERBOSE:
        print(f"{operation} | sending {'next' if is_fwd else 'prev'} "
              f"{pgm.process_group_manager.pp_rank} -> {peer_rank} | "
              f"receiving {'next' if is_fwd else 'prev'} {peer_rank} -> "
              f"{pgm.process_group_manager.pp_rank} | STEP {STEP=} | " # STEP= 是 Python 3.8+ 的 f-string 调试语法
              f"RANK:{pgm.process_group_manager.pp_rank}", flush=True)

    # 等待两个操作都完成 (同步点)
    [req.wait() for req in reqs]
    torch.cuda.synchronize() # 确保 GPU 操作完成

    if VERBOSE: STEP += 1 # 增加调试步骤计数

    # 返回接收到的张量
    return recv_tensor
### end PP communications ###


### begin Pipeline Parallel ###
# --- 流水线并行模块 (与 Step 7 相同) ---
class PipelineParallel(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        layer_distribution = self.distribute_layers(config.num_hidden_layers)
        self.embedding = model.embedding if pgm.process_group_manager.pp_is_first_stage else nn.Identity()
        self.decoder_layers = nn.ModuleDict({str(i): model.decoder_layers[i] for i in layer_distribution})
        self.final_norm = model.final_norm if pgm.process_group_manager.pp_is_last_stage else nn.Identity()
        self.final_proj = model.final_proj if pgm.process_group_manager.pp_is_last_stage else nn.Identity()

    def distribute_layers(self, num_layers):
        layers_per_gpu = [num_layers // pgm.process_group_manager.pp_world_size + (1 if i < num_layers % pgm.process_group_manager.pp_world_size else 0) for i in range(pgm.process_group_manager.pp_world_size)]
        start_layer = sum(layers_per_gpu[:pgm.process_group_manager.pp_rank])
        return list(range(start_layer, start_layer + layers_per_gpu[pgm.process_group_manager.pp_rank]))

    def forward(self, input_ids, position_ids, hidden_states):
        x = hidden_states if hidden_states is not None else input_ids
        x = self.embedding(x)
        for layer in self.decoder_layers.values():
            x = layer(x, position_ids=position_ids)
        x = self.final_norm(x)
        return self.final_proj(x)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        if input_tensor is not None: input_tensor.retain_grad()
        if output_tensor_grad is None:
            output_tensor_grad = torch.ones_like(output_tensor, memory_format=torch.preserve_format)
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad, retain_graph=False, create_graph=False)
        return input_tensor.grad if input_tensor is not None else None

# AFAB 训练步骤函数 (与 Step 7 相同)
def train_step_pipeline_afab(model, data_loader, tensor_shapes, device, dtype):
    """使用 AFAB 流水线并行执行一个训练步骤。"""
    logging_loss: torch.float32 = 0.0
    input_tensors, output_tensors = [], []
    requires_grad_sync = pgm.process_group_manager.dp_world_size > 1

    # === 前向传播阶段 ===
    for _ in range(data_loader.grad_acc_steps):
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=dtype)
        batch = next(data_loader)
        batch["hidden_states"] = input_tensor.to(device) if input_tensor is not None else input_tensor
        output_tensor = model.forward(input_ids=batch["input_ids"].to(device), position_ids=batch["position_ids"].to(device), hidden_states=batch["hidden_states"])
        pipeline_communicate(operation='send_forward', tensor=output_tensor, device=device, dtype=dtype)

        if pgm.process_group_manager.pp_is_last_stage:
            # 注意：这里直接修改了 output_tensor，使其在最后一个 stage 变成 loss 标量
            output_tensor_loss = F.cross_entropy(output_tensor.transpose(1, 2), batch["target_ids"].to(device), reduction='mean')
            logging_loss += output_tensor_loss.item() / data_loader.grad_acc_steps
            output_tensor = output_tensor_loss # 在最后一个 stage，保存 loss 以进行反向传播

        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    # === 反向传播阶段 ===
    for ith_microbatch in range(data_loader.grad_acc_steps):
        if requires_grad_sync:
            is_last_iteration = (ith_microbatch == data_loader.grad_acc_steps - 1)
            model.require_backward_grad_sync = is_last_iteration
        # 接收后向梯度，注意最后一个 stage 接收到的是 None
        output_tensor_grad = pipeline_communicate(operation='recv_backward', shapes=tensor_shapes, device=device, dtype=dtype)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        # 执行反向计算
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        # 发送梯度给上一个 stage
        pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=dtype)

    return logging_loss

# 新增：使用 1F1B (1 Forward - 1 Backward) 调度策略的流水线训练步骤
def train_step_pipeline_1f1b(model, data_loader, tensor_shapes, device, dtype):
    """使用 1F1B 流水线并行执行一个训练步骤。"""
    # --- 计算 Warmup 和 Steady 阶段的微批次数 ---
    # Warmup 阶段的微批次数：等于需要填充当前 rank 之后所有 ranks 的流水线所需的步数
    # 计算方法：(总 PP stages - 当前 rank - 1)
    # 同时不能超过总的梯度累积步数
    num_warmup_microbatches = min(pgm.process_group_manager.pp_world_size - pgm.process_group_manager.pp_rank - 1, data_loader.grad_acc_steps)
    # 1F1B 稳态 + Cooldown 阶段的微批次数
    num_microbatches_remaining = data_loader.grad_acc_steps - num_warmup_microbatches

    # 初始化损失记录、输入/输出张量存储列表
    logging_loss, input_tensors, output_tensors  = 0.0, [], []
    # 是否需要 DP 同步梯度
    requires_grad_sync = pgm.process_group_manager.dp_world_size > 1

    # --- 定义一个内部辅助函数来执行单个微批次的前向步骤 ---
    def _forward_step(input_tensor):
        # 从数据加载器获取数据
        batch = next(data_loader)
        # 设置输入激活值 (如果不是第一个 stage)
        batch["hidden_states"] = input_tensor.to(device) if input_tensor is not None else input_tensor
        # 执行模型前向计算
        output_tensor = model.forward(input_ids=batch["input_ids"].to(device), position_ids=batch["position_ids"].to(device), hidden_states=batch["hidden_states"])

        # 如果是最后一个 stage，计算损失
        if pgm.process_group_manager.pp_is_last_stage:
            # 计算交叉熵损失
            output_tensor_loss = F.cross_entropy(output_tensor.transpose(1, 2), batch["target_ids"].to(device), reduction='mean')
            # 使用 nonlocal 关键字修改外部函数的 logging_loss 变量
            nonlocal logging_loss
            # 累加损失
            logging_loss += output_tensor_loss.item() / data_loader.grad_acc_steps
            # 返回损失标量作为反向传播的起点
            return output_tensor_loss
        else:
            # 如果不是最后一个 stage，返回激活值张量
            return output_tensor

    # === 1. Warmup 阶段: 只有前向传播 ===
    # 执行 num_warmup_microbatches 次纯前向计算来填充流水线
    for _ in range(num_warmup_microbatches):
        # 接收前向输入
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=dtype)
        # 执行前向计算
        output_tensor = _forward_step(input_tensor)
        # 发送前向输出
        pipeline_communicate(operation='send_forward', tensor=output_tensor, device=device, dtype=dtype)
        # 保存输入和输出张量，用于后续的 Cooldown backward 阶段
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    # 如果还有剩余的微批次需要处理 (即进入稳态和 cooldown 阶段)
    if num_microbatches_remaining > 0:
        # 预先接收第一个稳态阶段所需的前向输入
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=dtype)

    # 在进入稳态阶段前，确保 DP 梯度同步标志为 False
    if requires_grad_sync:
        model.require_backward_grad_sync = False

    # === 2. 1F1B 稳态阶段: 同时进行前向和反向传播 ===
    # 循环处理剩余的微批次
    for ith_microbatch in range(num_microbatches_remaining):
        # 标记是否是稳态阶段的最后一次迭代
        is_last_iteration = (ith_microbatch == num_microbatches_remaining - 1)
        # 执行当前微批次的前向计算
        output_tensor = _forward_step(input_tensor)
        # --- 关键的 1F1B 通信步骤 ---
        # 同时：发送当前计算出的前向激活 (output_tensor) 给下一个 stage
        #       接收上一个微批次的反向梯度 (output_tensor_grad) 从下一个 stage
        output_tensor_grad = bidirectional_pipeline_communicate(operation='send_fwd_recv_bwd', send_tensor=output_tensor, recv_shapes=tensor_shapes, device=device, dtype=dtype)

        # 保存当前前向计算的输入和输出，用于未来的反向计算
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        # 取出对应于刚收到的梯度 (output_tensor_grad) 的那个微批次的前向输入和输出
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)

        # --- 控制 DP 梯度同步时机 ---
        # 只有当最后一个 stage (其 num_warmup_microbatches 为 0)
        # 并且处理的是稳态阶段的最后一个微批次时，才允许 DP 同步
        # 这确保了只有在全局最后一个微批次的反向传播完成后才进行 DP 同步
        if num_warmup_microbatches == 0 and is_last_iteration:
             if requires_grad_sync: model.require_backward_grad_sync = True

        # 执行对应微批次的反向计算
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)

        # --- 根据是否最后一次迭代决定下一步通信 ---
        if is_last_iteration:
            # 如果是稳态阶段最后一次迭代，后面不再有前向计算
            input_tensor = None # 不再需要接收前向输入
            # 只需单向发送计算出的反向梯度给上一个 stage
            pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=dtype)
        else:
            # 如果不是最后一次迭代，还需要为下一个前向计算做准备
            # --- 关键的 1F1B 通信步骤 ---
            # 同时：发送当前计算出的反向梯度 (input_tensor_grad) 给上一个 stage
            #       接收下一个微批次的前向激活 (input_tensor) 从上一个 stage
            input_tensor = bidirectional_pipeline_communicate(operation='send_bwd_recv_fwd', send_tensor=input_tensor_grad, recv_shapes=tensor_shapes, device=device, dtype=dtype)

    # === 3. Cooldown 阶段: 只有反向传播 ===
    # 处理在 Warmup 阶段保存的张量
    for ith_warmup_microbatches in range(num_warmup_microbatches):
        # 控制 DP 梯度同步时机
        if requires_grad_sync:
            # 只有在 Cooldown 阶段的最后一次迭代（即全局最后一次反向传播）时才允许 DP 同步
            is_last_iteration = (ith_warmup_microbatches == num_warmup_microbatches - 1)
            model.require_backward_grad_sync = is_last_iteration # 简化写法：只有最后一次为 True
        # 取出 Warmup 阶段保存的输入和输出
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        # 接收反向梯度
        output_tensor_grad = pipeline_communicate(operation='recv_backward', shapes=tensor_shapes, device=device, dtype=dtype)
        # 执行反向计算
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        # 发送反向梯度给上一个 stage
        pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=dtype)

    # 返回累加的损失值
    return logging_loss
### end Pipeline Parallel ###