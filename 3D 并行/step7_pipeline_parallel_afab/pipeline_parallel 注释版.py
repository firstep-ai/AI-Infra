# 导入所需的库
import os  # 用于访问环境变量
import torch  # PyTorch 核心库
import torch.nn as nn  # PyTorch 神经网络模块
import torch.nn.functional as F  # PyTorch 神经网络函数库
import torch.distributed as dist  # PyTorch 分布式通信库

import process_group_manager as pgm  # 导入进程组管理器，用于获取流水线并行 (PP) 相关信息

### begin PP communications ###
# --- 定义流水线并行 (Pipeline Parallelism, PP) 所需的通信操作 ---

# STEP 用于调试时追踪通信步骤，VERBOSE 控制是否打印调试信息
STEP, VERBOSE = 0, os.environ.get("VERBOSE", "0") == "1"

# 定义一个通用的流水线通信函数
def pipeline_communicate(operation, device, dtype, tensor=None, shapes=None):
    """
    处理流水线阶段之间用于前向和反向传播的点对点通信。

    Args:
        operation (str): 通信操作的类型 ('recv_forward', 'send_forward',
                        'recv_backward', 'send_backward')
        device: 张量操作的目标设备 (例如 'cuda')
        dtype: 张量的数据类型 (例如 torch.bfloat16)
        tensor: 用于发送操作的输入张量 (默认为 None)
        shapes: 用于接收操作的张量形状规格 (默认为 None)

    Returns:
        torch.Tensor or None: 对于接收操作，返回接收到的张量；对于发送操作，返回 None。
    """
    # 声明使用全局变量 STEP 和 VERBOSE
    global STEP
    global VERBOSE

    # --- 根据操作类型确定源/目标 rank 和张量 ---
    if operation == 'recv_forward':
        # 接收前向传播的激活值
        # 如果是流水线的第一个阶段，则无需接收，直接返回 None
        if pgm.process_group_manager.pp_is_first_stage: return None
        # 创建一个空的张量用于接收数据，形状由 shapes 指定
        # requires_grad=True 是必须的，因为接收到的激活值需要参与后续的反向传播
        tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        # 源 rank 是流水线中的上一个阶段
        src = pgm.process_group_manager.pp_prev_rank

    elif operation == 'send_forward':
        # 发送前向传播的激活值
        # 如果是流水线的最后一个阶段，则无需发送，直接返回
        if pgm.process_group_manager.pp_is_last_stage: return
        # 目标 rank 是流水线中的下一个阶段
        dest = pgm.process_group_manager.pp_next_rank

    elif operation == 'recv_backward':
        # 接收反向传播的梯度
        # 如果是流水线的最后一个阶段，则无需接收（它是梯度计算的起点），返回 None
        if pgm.process_group_manager.pp_is_last_stage: return None
        # 创建一个空的张量用于接收梯度，形状由 shapes 指定
        # requires_grad 可以为 False，因为梯度本身不需要再计算梯度，但设为 True 通常也无害
        tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        # 源 rank 是流水线中的下一个阶段
        src = pgm.process_group_manager.pp_next_rank

    elif operation == 'send_backward':
        # 发送反向传播的梯度
        # 如果是流水线的第一个阶段，则无需发送（梯度到此为止），直接返回
        if pgm.process_group_manager.pp_is_first_stage: return
        # 目标 rank 是流水线中的上一个阶段
        dest = pgm.process_group_manager.pp_prev_rank

    # --- 执行通信操作 ---
    # 判断是发送操作还是接收操作
    is_send = operation.startswith('send')
    # 确定通信对方的 rank
    peer_rank = dest if is_send else src

    # 创建点对点操作对象 (P2POp)
    # 如果是发送，操作是 dist.isend (异步发送)
    # 如果是接收，操作是 dist.irecv (异步接收)
    op = dist.P2POp(dist.isend if is_send else dist.irecv, tensor, peer_rank)

    # 如果 VERBOSE 标志为 True，则打印详细的通信日志
    if VERBOSE:
        print(f"{operation} | {'sending' if is_send else 'receiving'} {operation.split('_')[1]} "
              f"{pgm.process_group_manager.pp_rank} {'→' if is_send else '←'} {peer_rank} | "
              f"STEP:{STEP} | RANK:{pgm.process_group_manager.pp_rank}", flush=True)

    # 执行批处理的发送/接收操作并等待其完成
    # dist.batch_isend_irecv 接收一个 P2POp 列表，可以批量处理多个 P2P 操作
    # 这里虽然列表里只有一个 op，但使用了这个接口
    # 返回一个请求对象列表 (reqs)，列表推导式 `[req.wait() for req in ...]` 会立即对每个请求调用 .wait()
    # 这使得这里的通信实际上是同步（阻塞）的，函数会等到通信完成才继续执行
    [req.wait() for req in dist.batch_isend_irecv([op])]
    # 在 CUDA 环境下，确保 GPU 操作（包括数据传输）已完成
    torch.cuda.synchronize()

    # 如果 VERBOSE 标志为 True，增加通信步骤计数器
    if VERBOSE: STEP += 1

    # 如果是接收操作，返回接收到的张量；如果是发送操作，返回 None
    return tensor if not is_send else None
### end PP communications ###


### begin Pipeline Parallel ###
# --- 定义流水线并行模块和训练步骤 ---

# 定义流水线并行模块，包装原始模型
class PipelineParallel(nn.Module):
    # 构造函数
    def __init__(self, model, config):
        # model: 原始的、完整的模型对象
        # config: 模型配置对象

        super().__init__() # 调用父类初始化
        # 计算当前 PP rank 应该负责哪些层
        layer_distribution = self.distribute_layers(config.num_hidden_layers)

        # --- 根据当前 PP rank 分配模型层 ---
        # Embedding 层：只有第一个 stage (pp_rank == 0) 需要
        self.embedding = model.embedding if pgm.process_group_manager.pp_is_first_stage else nn.Identity()
        # Decoder 层：只保留当前 rank 负责的那些层，使用 ModuleDict 存储
        self.decoder_layers = nn.ModuleDict({str(i): model.decoder_layers[i] for i in layer_distribution})
        # Final Normalization 层：只有最后一个 stage 需要
        self.final_norm = model.final_norm if pgm.process_group_manager.pp_is_last_stage else nn.Identity()
        # Final Projection 层 (输出层)：只有最后一个 stage 需要
        self.final_proj = model.final_proj if pgm.process_group_manager.pp_is_last_stage else nn.Identity()
        # nn.Identity() 是一个占位模块，它不对输入做任何操作，直接返回输入

    # 定义一个方法来计算层如何在 PP ranks 之间分配
    def distribute_layers(self, num_layers):
        # num_layers: 模型总的 Decoder 层数
        # 计算每个 PP rank 平均分配多少层，并将余数分配给前面的 ranks
        layers_per_gpu = [num_layers // pgm.process_group_manager.pp_world_size + \
                          (1 if i < num_layers % pgm.process_group_manager.pp_world_size else 0) \
                          for i in range(pgm.process_group_manager.pp_world_size)]
        # 计算当前 rank 的起始层索引 (前面所有 ranks 的层数之和)
        start_layer = sum(layers_per_gpu[:pgm.process_group_manager.pp_rank])
        # 返回一个包含当前 rank 负责的所有层索引的列表
        return list(range(start_layer, start_layer + layers_per_gpu[pgm.process_group_manager.pp_rank]))

    # 定义流水线并行模块的前向传播逻辑
    def forward(self, input_ids, position_ids, hidden_states):
        # input_ids: 原始输入 (只在第一个 stage 使用)
        # position_ids: 位置 ID (传递给所有需要它的层)
        # hidden_states: 从上一个 stage 接收到的激活值 (如果不是第一个 stage)

        # 确定当前 stage 的实际输入
        x = hidden_states if hidden_states is not None else input_ids
        # 应用 Embedding 层 (只在第一个 stage 有效)
        x = self.embedding(x)
        # 依次通过分配给当前 stage 的 Decoder 层
        for layer in self.decoder_layers.values():
            x = layer(x, position_ids=position_ids) # 注意：原始 Llama 层可能不需要 position_ids，但这里传递了
        # 应用 Final Normalization 层 (只在最后一个 stage 有效)
        x = self.final_norm(x)
        # 应用 Final Projection 层 (只在最后一个 stage 有效) 并返回结果
        return self.final_proj(x)

    # 定义流水线并行模块的反向传播逻辑 (手动触发)
    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        # input_tensor: 当前 stage 在前向传播时的输入激活值
        # output_tensor: 当前 stage 在前向传播时的输出激活值
        # output_tensor_grad: 从下一个 stage 接收到的梯度

        # 如果存在来自上一 stage 的输入张量，需要保留它的梯度，以便回传
        if input_tensor is not None: input_tensor.retain_grad()
        # 如果没有从下一 stage 接收到梯度 (说明是最后一个 stage，梯度由此产生)
        if output_tensor_grad is None:
            # 创建一个与输出张量形状相同、值为 1 的梯度张量作为反向传播的起点
            output_tensor_grad = torch.ones_like(output_tensor, memory_format=torch.preserve_format)
        # 调用 PyTorch 的 autograd.backward 函数
        # 它会计算 output_tensor 相对于计算图中叶子节点（模型参数和设置了 retain_grad=True 的中间张量）的梯度
        # grad_tensors 指定了起始梯度
        # retain_graph=False 表示计算完梯度后释放计算图，节省内存
        # create_graph=False 表示不创建用于二阶导数计算的图
        # 参考: https://pytorch.org/docs/stable/generated/torch.autograd.backward.html
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad, retain_graph=False, create_graph=False)
        # 返回 input_tensor 的梯度 (如果 input_tensor 存在)
        # 这个梯度需要被发送回上一个 stage
        return input_tensor.grad if input_tensor is not None else None

# 定义使用 AFAB (Activation Forward - Activation Backward) 调度策略的流水线训练步骤
def train_step_pipeline_afab(model, data_loader, tensor_shapes, device, dtype):
    """
    使用 Activation Forward - Activation Backward (AFAB) 流水线并行执行一个训练步骤。
    实现分离的前向和反向传递以优化内存使用。
    """
    # 初始化用于记录损失的变量 (类型注解为 float32)
    logging_loss: torch.float32 = 0.0
    # 用于存储每个微批次在前向传播时的输入和输出张量 (激活值)
    input_tensors, output_tensors = [], []
    # 检查是否需要与数据并行 (DP) 组同步梯度
    requires_grad_sync = pgm.process_group_manager.dp_world_size > 1

    # === 所有微批次的前向传播阶段 ===
    # 遍历梯度累积的步数 (即微批次的数量)
    for _ in range(data_loader.grad_acc_steps):
        # 1. 接收来自上一个 stage 的激活值 (input_tensor)
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=dtype)
        # 2. 从数据加载器获取下一个微批次的数据
        batch = next(data_loader)
        # 3. 如果不是第一个 stage，将接收到的 input_tensor 放入 batch 字典中
        batch["hidden_states"] = input_tensor.to(device) if input_tensor is not None else input_tensor
        # 4. 执行当前 stage 的前向计算
        output_tensor = model.forward(input_ids=batch["input_ids"].to(device), position_ids=batch["position_ids"].to(device), hidden_states=batch["hidden_states"])
        # 5. 将计算得到的激活值 (output_tensor) 发送给下一个 stage
        pipeline_communicate(operation='send_forward', tensor=output_tensor, device=device, dtype=dtype)

        # 6. 如果是最后一个 stage，计算损失
        if pgm.process_group_manager.pp_is_last_stage:
            # 使用交叉熵损失函数
            # 注意：F.cross_entropy 期望的 logits 形状是 [N, C]，target 形状是 [N]
            # 模型输出是 [batch, seq_len, vocab_size]，target 是 [batch, seq_len]
            # 因此需要 transpose(1, 2) 变为 [batch, vocab_size, seq_len] (?) -> 检查损失计算是否需要调整 view/reshape
            # (常见的做法是 view(-1, vocab_size) 和 target.view(-1))
            output_tensor_loss = F.cross_entropy(output_tensor.transpose(1, 2), batch["target_ids"].to(device), reduction='mean')
            # 累加损失值（需要除以梯度累积步数）
            logging_loss += output_tensor_loss.item() / data_loader.grad_acc_steps
            # 保存损失张量本身作为反向传播的起点 (覆盖掉原来的 output_tensor)
            output_tensor = output_tensor_loss

        # 7. 保存当前微批次的输入和输出张量，用于后续的反向传播阶段
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    # === 所有微批次的反向传播阶段 ===
    # 再次遍历梯度累积的步数
    for ith_microbatch in range(data_loader.grad_acc_steps):
        # 如果使用了数据并行 (DP)，需要控制梯度同步的时机
        if requires_grad_sync:
            # 只有在处理最后一个微批次时，才允许 DP 进行梯度同步
            is_last_iteration = (ith_microbatch == data_loader.grad_acc_steps - 1)
            # 设置模型（通常是 DP 包装器）的标志位
            model.require_backward_grad_sync = is_last_iteration
        # 1. 接收来自下一个 stage 的梯度 (output_tensor_grad)
        output_tensor_grad = pipeline_communicate(operation='recv_backward', shapes=tensor_shapes, device=device, dtype=dtype)
        # 2. 从之前保存的列表中取出对应微批次的输入和输出张量 (按顺序 FIFO)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        # 3. 执行当前 stage 的反向计算，得到输入张量的梯度 (input_tensor_grad)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        # 4. 将计算得到的输入梯度发送给上一个 stage
        pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=dtype)

    # 返回累加的损失值 (主要在最后一个 stage 有意义，并且通常需要经过 DP all-reduce)
    return logging_loss

### end Pipeline Parallel ###