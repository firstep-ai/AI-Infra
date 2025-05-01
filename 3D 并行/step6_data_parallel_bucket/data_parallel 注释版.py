# 导入所需的库
import contextlib  # 提供用于创建上下文管理器的实用程序 (在此代码段中未使用)
from typing import List  # 用于类型注解
import torch  # PyTorch 核心库
import torch.distributed as dist  # PyTorch 分布式通信库
from torch import nn  # PyTorch 神经网络模块

import process_group_manager as pgm  # 导入进程组管理器

### begin Data Parallel (naive) ###
# --- 基础数据并行实现 (来自 Step 5) ---
class DataParallelNaive(nn.Module):
    # 注释：这个简单的实现不应该与梯度累积一起使用（因为它会在 bfloat16 而不是 float32 中累积梯度）
    # （译注：如果模型是 bfloat16，梯度也是 bfloat16，多次累加可能导致精度损失，
    #   更健壮的方法是用 float32 累积梯度，Bucket DP 会这样做）
    def __init__(self, module):
        super().__init__()
        self.module = module
        # 控制是否在反向传播时同步梯度，用于梯度累积
        self.require_backward_grad_sync = True
        self.register_backward_hook(self._allreduce_grads)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def register_backward_hook(self, hook):
        """为模型中所有需要计算梯度的参数注册一个反向传播钩子。"""
        for p in self.module.parameters():
            if p.requires_grad is True:
                p.register_hook(hook)

    def _allreduce_grads(self, grad):
        """执行 All-Reduce 操作以同步（平均）多个进程间的梯度。"""
        # 在梯度累积期间不需要同步，除非是最后一步
        if self.require_backward_grad_sync:
            # 在 DP 组内对梯度求和
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.dp_group)
            # 求平均
            grad /= pgm.process_group_manager.dp_world_size
        return grad

### end Data Parallel (naive) ###


### begin Data Parallel (bucket) ###
# --- 使用 Bucket 技术优化的数据并行实现 (Step 6) ---

# 定义 Bucket 类，管理一小部分参数的梯度同步
class Bucket:
    # 构造函数
    def __init__(self, params: List[torch.nn.Parameter], grad_data: torch.Tensor, process_group: torch.distributed.ProcessGroup) -> None:
        # params: 这个 bucket 包含的参数列表
        # grad_data: 预先分配好的、用于存储这个 bucket 中所有参数梯度的连续内存块 (Tensor)
        # process_group: 用于梯度同步的分布式通信组 (通常是 DP group)

        # 将参数列表转换为集合，方便快速查找
        self.params = set(params)
        # 用于存储已经计算好梯度的参数的集合
        # 当这个集合的大小等于 self.params 的大小时，触发 all-reduce
        self.params_with_grad_ready = set()
        # 保存对梯度存储内存块的引用
        self.grad_data = grad_data
        # 保存通信组
        self.process_group = process_group
        # 获取通信组的大小
        self.process_group_size = dist.get_world_size(group=self.process_group)
        # 用于保存异步 all-reduce 操作句柄 (handle)
        self.handle = None

        # 初始化 bucket 状态
        self.reset()

    # 启动异步梯度同步的方法
    def sync_gradient(self) -> None:
        """启动一个异步 all-reduce 操作来同步梯度。"""
        # 确保当前没有正在进行的 all-reduce 操作
        assert self.handle is None
        # 在 all-reduce 之前先进行本地除法 (pre-division)，减少通信量？ (存疑，通常是 post-division)
        # (译注：这里可能是为了避免在 all_reduce 后再做一次除法操作，
        # 或者假设梯度累加在 main_grad 上进行，这里先除了再 all_reduce)
        # 更新：更常见的做法是在 all_reduce(SUM) 之后再除以 world_size。
        # 如果这里 grad_data 是 float32，也许是为了保持精度？但注释说 grad_size=2 (bf16)。
        # 假设这里的目的是为了让 all_reduce 直接得到平均值（如果 ReduceOp.AVG 不可用或效率低）。
        self.grad_data /= self.process_group_size
        # 启动异步 all-reduce 操作 (求和)
        # async_op=True 表示该操作不会阻塞，会立即返回一个句柄
        self.handle = dist.all_reduce(self.grad_data, group=self.process_group, async_op=True)

    # 重置 bucket 状态的方法
    def reset(self) -> None:
        """将 bucket 重置到初始状态。通常在梯度同步完成后调用。"""
        self.handle = None  # 清空操作句柄
        # 清空已就绪参数的集合
        self.params_with_grad_ready.clear()
        # 将梯度存储内存块清零，为下一轮梯度累积做准备
        self.grad_data.zero_()

    # 等待梯度同步完成的方法
    def wait(self) -> None:
        """等待 all-reduce 操作完成。"""
        # 确保已经启动了 all-reduce 操作
        assert self.handle is not None, "You should launch an allreduce operation before waiting for it to finish"
        # 阻塞当前进程，直到句柄对应的异步操作完成
        self.handle.wait()

    # 标记一个参数梯度已就绪的方法
    def mark_param_as_ready(self, param: torch.nn.Parameter) -> None:
        """标记一个参数的梯度已准备好进行同步。当桶内所有参数都准备好时，启动同步。"""
        # 断言检查：确保该参数属于这个 bucket，并且之前未被标记为就绪
        assert param in self.params and param not in self.params_with_grad_ready
        # 将参数添加到已就绪集合中
        self.params_with_grad_ready.add(param)
        # 检查是否 bucket 中所有参数的梯度都已就绪
        if len(self.params_with_grad_ready) == len(self.params):
            # 如果是，则调用 sync_gradient 启动该 bucket 的梯度同步
            self.sync_gradient()

# 定义 Bucket 管理器类
class BucketManager:
    # 构造函数
    def __init__(self, params: List[torch.nn.Parameter], process_group: torch.distributed.ProcessGroup, bucket_size: int, grad_type: torch.dtype = torch.float32) -> None:
        # params: 模型的所有参数 (通常是 module.parameters())
        # process_group: 用于梯度同步的通信组
        # bucket_size: 每个 bucket 的容量大小 (通常是元素数量)
        # grad_type: 用于存储累积梯度的 Tensor 类型 (推荐 float32 以保证精度)

        # 将参数迭代器转换为列表
        self.params = list(params)
        # 用于存储所有 Bucket 对象的列表
        self.buckets = []
        # 保存通信组及其大小
        self.process_group = process_group
        self.process_group_size = dist.get_world_size(group=self.process_group)
        # 用于存储每个参数到其所属 bucket 位置信息的字典
        # key: parameter, value: (start_index, end_index, bucket_id)
        self.params_to_bucket_location = {}
        # 保存 bucket 容量设置
        self.bucket_size = bucket_size
        self.bucket_sizes = None # 实际每个 bucket 的大小，在 _initialize_buckets 中计算
        # 用于存储每个 bucket 对应的梯度内存块的列表
        self.grad_data_list = []
        # 保存梯度存储类型
        self.grad_type = grad_type
        # 初始化 buckets，分配参数，创建梯度存储区
        self._initialize_buckets()

    # 初始化 Bucket 的方法
    def _initialize_buckets(self) -> None:
        """根据 bucket_size 将模型参数划分到不同的 bucket 中。"""
        cur_bucket_size = 0   # 当前 bucket 已使用的元素数量
        cur_bucket_idx = 0   # 当前 bucket 的索引

        # 遍历所有模型参数，将它们分配到 buckets 中
        for param in self.params:
            # 跳过不需要梯度的参数
            if not param.requires_grad:
                continue

            # 如果当前 bucket 是空的，直接将参数放入
            if cur_bucket_size == 0:
                # 记录参数在 bucket 0 中的位置 (起始索引 0, 结束索引 param.numel(), bucket 索引 0)
                self.params_to_bucket_location[param] = (0, param.numel(), cur_bucket_idx)
                # 更新当前 bucket 的大小
                cur_bucket_size = param.numel()
                continue

            # 如果将当前参数放入后，当前 bucket 超出容量限制
            if cur_bucket_size + param.numel() > self.bucket_size:
                # 切换到下一个 bucket
                cur_bucket_idx += 1
                # 将参数放入新的 bucket，位置从 0 开始
                self.params_to_bucket_location[param] = (0, param.numel(), cur_bucket_idx)
                # 更新新 bucket 的大小
                cur_bucket_size = param.numel()
            else:
                # 如果可以放入当前 bucket
                # 记录参数在当前 bucket 中的位置 (起始索引 cur_bucket_size, 结束索引 cur_bucket_size + param.numel(), bucket 索引 cur_bucket_idx)
                self.params_to_bucket_location[param] = (cur_bucket_size, cur_bucket_size + param.numel(), cur_bucket_idx)
                # 累加当前 bucket 的大小
                cur_bucket_size += param.numel()

        # ---- 分配完参数后，创建实际的 Bucket 对象和梯度存储区 ----

        # 计算每个 bucket 实际需要的大小，并收集每个 bucket 包含的参数
        num_buckets = cur_bucket_idx + 1
        bucket_sizes = [0] * num_buckets           # 存储每个 bucket 的实际大小
        buckets_to_params = [[] for _ in range(num_buckets)] # 存储每个 bucket 包含的参数列表

        # 遍历 params_to_bucket_location 字典，填充 bucket_sizes 和 buckets_to_params
        for param, (_, end, idx) in self.params_to_bucket_location.items():
            # bucket 的实际大小取决于最后一个参数的结束索引
            bucket_sizes[idx] = max(bucket_sizes[idx], end)
            # 将参数添加到对应 bucket 的参数列表中
            buckets_to_params[idx].append(param)

        # 为每个 bucket 创建梯度存储张量 (grad_data) 和 Bucket 对象
        for i in range(len(bucket_sizes)):
            # 创建一个全零张量用于存储梯度，类型为 self.grad_type (推荐 float32)，设备为 CUDA
            self.grad_data_list.append(torch.zeros(bucket_sizes[i], dtype=self.grad_type, device='cuda'))
            # 创建 Bucket 对象，传入该 bucket 的参数列表、对应的梯度存储张量和通信组
            self.buckets.append(Bucket(buckets_to_params[i], self.grad_data_list[i], self.process_group))

        # --- 为每个参数创建指向其在对应 bucket 梯度存储区中的视图 (view) ---
        # 反向遍历参数列表（通常与反向传播计算梯度的顺序一致或接近，有助于重叠计算和通信）
        for param in self.params[::-1]:
            # 跳过不需要梯度的参数
            if not param.requires_grad:
                continue
            # 获取参数在 bucket 中的位置信息
            data_start_index, data_end_index, bucket_id = self.params_to_bucket_location[param]
            # 创建一个视图 (view) 指向对应 bucket 的 grad_data 张量中的特定区域
            # 将这个视图赋值给参数的一个新属性 param.main_grad
            # 后续梯度累积将直接在这个视图上进行，也就是直接在 bucket 的内存上进行
            param.main_grad = self._get_view_from_tensor(self.grad_data_list[bucket_id], param.shape, data_start_index, data_end_index)

    # 辅助函数：从一维张量中获取一个特定形状的视图
    def _get_view_from_tensor(self, tensor: torch.Tensor, shape: torch.Size, start: int, end: int) -> torch.Tensor:
        # tensor: 一维的梯度存储张量 (例如 self.grad_data_list[bucket_id])
        # shape: 参数的原始形状
        # start: 视图在 tensor 中的起始索引
        # end: 视图在 tensor 中的结束索引
        # 返回一个形状为 shape 的张量，其数据与 tensor[start:end] 共享内存
        return tensor[start:end].view(shape)

    # 重置所有 bucket 的状态
    def reset(self) -> None:
        # 遍历所有 Bucket 对象并调用它们的 reset 方法
        for bucket in self.buckets:
            bucket.reset()

    # 等待所有 bucket 的梯度同步完成
    def wait(self) -> None:
        # 遍历所有 Bucket 对象并调用它们的 wait 方法
        for bucket in self.buckets:
            bucket.wait() # 这会阻塞，直到对应 bucket 的 all-reduce 完成

    # 标记某个参数的梯度已准备好
    def mark_param_as_ready(self, param: torch.nn.Parameter) -> None:
        # 找到该参数所属的 bucket 索引
        bucket_idx = self.params_to_bucket_location[param][2]
        # 调用对应 Bucket 对象的 mark_param_as_ready 方法
        self.buckets[bucket_idx].mark_param_as_ready(param)

# 定义使用 Bucket 技术的 DataParallel 模块
class DataParallelBucket(nn.Module):
    # 构造函数
    def __init__(self, module, bucket_cap_mb=25, grad_type = torch.float32):
        # module: 被包装的原始模型
        # bucket_cap_mb: 每个 bucket 的容量上限（单位：MB）
        # grad_type: 存储和累积梯度的类型

        super().__init__()
        self.module = module # 保存原始模型
        # 控制是否同步梯度的标志位，用于梯度累积
        self.require_backward_grad_sync = True
        # 假设梯度类型为 bfloat16 (2字节)，用于计算 bucket 大小
        # (注意：grad_type 参数默认为 float32，这里可能需要根据实际情况调整或使 grad_size 动态计算)
        grad_size = 2 # bfloat16 gradient: 2 bytes
        # 计算 bucket_size (元素数量)
        # bucket 容量 (MB) * 1024 * 1024 转换为字节
        # 再除以每个梯度元素的大小 (字节)
        bucket_size = bucket_cap_mb * 1024 * 1024 // grad_size
        # 创建 BucketManager 实例来管理参数和梯度桶
        self.bucket_manager = BucketManager(module.parameters(), pgm.process_group_manager.dp_group, bucket_size, grad_type)
        # 注册特殊的反向传播钩子
        self.register_backward_hook()
        # 标志位，用于确保 post_backward 回调在每次 backward 中只注册一次
        self._post_backward_callback_set = False

    # 前向传播：直接调用内部模块
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    # 定义 backward 方法（主要用于流水线并行场景，这里简单传递给内部模块）
    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        return self.module.backward(input_tensor, output_tensor, output_tensor_grad)

    # 定义 get_flops 方法（用于计算 FLOPs，简单传递给内部模块）
    def get_flops(self, *args, **kwargs):
        return self.module.get_flops(*args, **kwargs)

    # 注册反向传播钩子的核心方法
    def register_backward_hook(self):
        """
        注册反向传播钩子以手动累积和同步梯度。
        主要目的：
        1. 支持混合精度下的梯度累积（将低精度梯度累积到高精度 buffer 中）。
        2. 在梯度计算完成后，标记参数已准备好进行同步。
        梯度累积函数 (grad_acc_fn) 需要被存储起来以防被垃圾回收。
        参考资料：
        - Megatron-LM issue: https://github.com/NVIDIA/Megatron-LM/issues/690
        - PyTorch hook 文档: https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.register_hook.html
        - 相关论文 (ZeRO): https://arxiv.org/abs/2006.15704 (page 5)
        """
        self.grad_accs = [] # 用于存储梯度累积函数对象
        # 遍历模型中所有需要梯度的参数
        for param in self.module.parameters():
            if param.requires_grad:
                # 技巧：通过 expand_as 获取参数对应的 grad_fn (梯度函数)
                # .grad_fn 指向计算该张量（参数本身是叶子节点，没有 grad_fn）的操作
                # 这里需要找到负责累积参数梯度的那个内部函数节点
                param_tmp = param.expand_as(param)
                # grad_fn 的 next_functions[0][0] 通常是梯度累积函数 (AccumulateGrad object)
                grad_acc_fn = param_tmp.grad_fn.next_functions[0][0]
                # 在这个梯度累积函数上注册我们自定义的钩子 (_make_param_hook 返回的函数)
                grad_acc_fn.register_hook(self._make_param_hook(param, self.bucket_manager))
                # 保存 grad_acc_fn 对象，防止其被 Python 垃圾回收
                self.grad_accs.append(grad_acc_fn)

    # 创建参数钩子函数的工厂方法
    def _make_param_hook(self, param: torch.nn.Parameter, bucket_manager: BucketManager):
        """为每个参数创建一个钩子，用于处理梯度累积和同步。"""
        # 返回一个闭包函数 param_hook
        def param_hook(*unused): # 钩子函数接收一些未使用的参数
            """
            在梯度计算完成后被调用的钩子。执行：
            1. 将计算出的梯度 (param.grad) 累积到主梯度 (param.main_grad，即 bucket 内存)。
            2. 添加一个 post-backward 回调 (_post_backward) 来等待梯度同步完成。
            3. 标记该参数已准备好进行同步。
            """
            # 确保参数需要梯度
            if param.requires_grad:
                # 此时 PyTorch 已经计算好了 param.grad
                assert param.grad is not None
                # 1. 将 param.grad 的数据累加到 param.main_grad (bucket 内存视图)
                param.main_grad.add_(param.grad.data)
                # 2. 清空 param.grad，释放内存，并防止优化器错误地使用它
                param.grad = None

                # 3. 检查是否需要进行梯度同步 (考虑梯度累积)
                if self.require_backward_grad_sync:
                    # 4. 注册 post-backward 回调（如果本次 backward 还没注册过）
                    #    这个回调会在整个 backward pass 完成后执行
                    if not self._post_backward_callback_set:
                        # 使用 PyTorch 内部执行引擎注册回调
                        torch.autograd.Variable._execution_engine.queue_callback(self._post_backward)
                        # 设置标志位，防止重复注册
                        self._post_backward_callback_set = True

                    # 5. 标记该参数的梯度已在 bucket 中准备好
                    bucket_manager.mark_param_as_ready(param)
        # 返回创建的钩子函数
        return param_hook

    # 在整个 backward pass 完成后执行的回调函数
    def _post_backward(self):
        """
        Post-backward 回调函数：等待梯度同步完成，然后将同步好的梯度复制回参数的 .grad 属性。
        在 backward pass 之后、optimizer step 之前调用。
        """
        # 1. 等待所有 bucket 的异步 all-reduce 操作完成
        self.bucket_manager.wait()
        # 2. 重置回调注册标志位，为下一次 backward 做准备
        self._post_backward_callback_set = False
        # 3. 将 bucket 中同步好的、高精度的梯度复制回每个参数的 .grad 属性
        for p in self.module.parameters():
            if p.requires_grad:
                # 将 param.main_grad (bucket 视图) 的数据转换为参数的原始数据类型 (p.dtype)
                # 然后赋值给 p.grad，这样优化器就可以使用它了
                # 注意：不能直接赋值不同类型的梯度
                p.grad = p.main_grad.to(p.dtype)

    # 重置梯度的方法（在每个训练 step 开始时调用 optimizer.zero_grad() 之前或之后）
    def reset(self):
        # 调用 bucket 管理器的 reset 方法，该方法会调用每个 bucket 的 reset 方法
        # 清空所有 bucket 中的梯度累积值 (grad_data.zero_())
        self.bucket_manager.reset()

### end Data Parallel (bucket) ###