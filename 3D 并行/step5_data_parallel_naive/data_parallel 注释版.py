# 导入所需的库
import contextlib  # 提供用于创建上下文管理器的实用程序 (在此代码段中未使用，但可能在完整文件中其他地方使用)
from typing import List  # 用于类型注解，例如 List[torch.nn.Parameter]
import torch  # PyTorch 核心库
import torch.distributed as dist  # PyTorch 分布式通信库
from torch import nn  # PyTorch 神经网络模块

import process_group_manager as pgm  # 导入进程组管理器，用于获取数据并行 (DP) 的信息

### begin Data Parallel (naive) ###
# --- 定义一个基础的（朴素的）数据并行实现 ---

# 定义 DataParallelNaive 类，继承自 PyTorch 的基础模块 nn.Module
class DataParallelNaive(nn.Module):
    # 类的构造函数 (初始化方法)
    def __init__(self, module):
        # module: 需要进行数据并行的原始 PyTorch 模型 (nn.Module)

        super().__init__()  # 调用父类 nn.Module 的构造函数
        self.module = module  # 将传入的原始模型保存为类的属性

        # 定义一个标志位，用于控制在反向传播时是否同步梯度。
        # 当使用梯度累积时，通常只在最后一次累积时才需要同步，其他累积步骤可以设为 False 来跳过同步。
        self.require_backward_grad_sync = True  # 默认初始化为 True，表示需要同步

        # 调用 register_backward_hook 方法，为模型中所有需要梯度的参数注册 _allreduce_grads 这个钩子函数。
        self.register_backward_hook(self._allreduce_grads)

    # 定义模型的前向传播方法
    def forward(self, *inputs, **kwargs):
        # *inputs: 接收任意数量的位置参数
        # **kwargs: 接收任意数量的关键字参数
        # 直接调用被包装的原始模型 (self.module) 的 forward 方法，并将所有参数传递给它
        # 数据并行本身不改变模型单次前向计算的逻辑
        return self.module(*inputs, **kwargs)

    # 定义一个方法，用于给模型参数注册反向传播钩子 (backward hook)
    def register_backward_hook(self, hook):
        """为模型中所有需要计算梯度的参数注册一个反向传播钩子。"""
        # hook: 要注册的钩子函数 (这里是 _allreduce_grads)

        # 遍历被包装模型 (self.module) 的所有参数
        for p in self.module.parameters():
            # 检查参数是否需要计算梯度 (即是否是可训练参数)
            if p.requires_grad is True:
                # 如果需要梯度，则调用参数的 register_hook 方法注册钩子函数
                # 这个钩子会在该参数的梯度计算完成之后、优化器使用梯度之前被自动调用
                p.register_hook(hook)

    # 定义实际执行梯度 All-Reduce 的钩子函数
    # 这个函数会在其注册到的参数梯度计算完成后被调用
    def _allreduce_grads(self, grad):
        """执行 All-Reduce 操作以同步（平均）多个进程间的梯度。"""
        # grad: PyTorch 自动计算出的原始梯度张量

        # 检查 require_backward_grad_sync 标志位。
        # 在梯度累积场景下，只有在最后一步需要同步时，这个标志才为 True。
        if self.require_backward_grad_sync:
            # 如果需要同步：
            # 1. 使用 dist.all_reduce 在数据并行组 (dp_group) 内对梯度张量 (grad) 执行 All-Reduce 操作。
            #    op=dist.ReduceOp.SUM 表示将所有进程的梯度加起来。
            #    操作是就地的 (in-place)，结果会覆盖原始的 grad 张量。
            #    执行后，每个进程上的 grad 张量都包含了所有 DP 进程对应参数的梯度之和。
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.dp_group)
            # 2. 将求和后的梯度除以数据并行组的大小 (dp_world_size)，得到平均梯度。
            #    这也是一个就地操作。
            grad /= pgm.process_group_manager.dp_world_size
        # 返回处理后的梯度 (可能是原始梯度，也可能是同步并平均后的梯度)
        # 钩子函数必须返回梯度张量
        return grad
### end Data Parallel (naive) ###