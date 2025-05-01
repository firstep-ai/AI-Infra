# 导入所需的库
import math  # 用于数学运算，例如初始化中的 sqrt
from typing import Optional  # 用于类型注解，例如 Optional[int] 表示可以是整数或 None
import torch  # PyTorch 核心库
import torch.nn as nn  # PyTorch 神经网络模块
import torch.distributed as dist  # PyTorch 分布式通信库
import torch.nn.functional as F  # PyTorch 神经网络函数库 (如 F.linear, F.embedding)
import process_group_manager as pgm  # 导入进程组管理器，用于获取分布式信息 (TP rank, TP world size, TP group)

### begin TP communications ###
# --- 定义张量并行 (Tensor Parallelism, TP) 所需的通信操作 ---

# 辅助函数：沿着张量的最后一个维度将其切分成 N 份
def split_tensor_along_last_dim(tensor, num_partitions):
    """将一个张量沿着最后一个维度切分成 num_partitions 个块。"""
    last_dim = tensor.dim() - 1  # 获取最后一个维度的索引
    # 检查最后一个维度的大小是否能被分区数整除
    assert tensor.size()[last_dim] % num_partitions == 0, f"{tensor.size()[last_dim]} is not divisible by {num_partitions}"
    # 计算每个分区的大小
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # 使用 torch.split 进行切分
    return torch.split(tensor, last_dim_size, dim=last_dim)

# 自定义 Autograd Function 实现 All-Reduce 操作
# 用于 RowParallelLinear 的前向传播
class Reduce(torch.autograd.Function):
    """前向传播执行 All-Reduce (SUM) 操作，反向传播是 Identity (恒等) 操作。"""
    @staticmethod
    def forward(ctx, input):
        # ctx: 上下文对象，用于在反向传播中传递信息（这里未使用）
        # input: 输入张量
        # 如果张量并行大小为 1，则无需通信，直接返回输入
        if pgm.process_group_manager.tp_world_size == 1:
            return input
        # 在张量并行组 (tp_group) 内对输入张量执行 All-Reduce 求和操作
        dist.all_reduce(input, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        # 返回 All-Reduce 后的结果
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: 从后续层传来的梯度
        # 在反向传播中，此操作是恒等操作，直接返回梯度
        # 因为前向是求和，梯度应该直接传递给所有参与求和的分支
        return grad_output

# 自定义 Autograd Function 实现 All-Gather 操作
# 用于 ColumnParallelLinear 的前向传播（当 gather_output=True 时）
class Gather(torch.autograd.Function):
    """前向传播执行 All-Gather 操作，反向传播执行 Split 操作。"""
    @staticmethod
    def forward(ctx, input):
        # 如果张量并行大小为 1，直接返回输入
        if pgm.process_group_manager.tp_world_size == 1:
            return input
        last_dim = input.dim() - 1  # 获取最后一个维度
        # 分布式集合通信操作通常需要连续的张量
        # 参考: https://github.com/pytorch/pytorch/blob/main/torch/distributed/nn/functional.py#L321
        input = input.contiguous()
        # 创建一个列表，用于存放从各个 rank 收集来的张量
        tensor_list = [torch.empty_like(input) for _ in range(pgm.process_group_manager.tp_world_size)]
        # 将当前 rank 的输入张量放在列表的对应位置 (此步骤对于 all_gather 可能非必需，但明确)
        tensor_list[pgm.process_group_manager.tp_rank] = input
        # 在张量并行组内执行 All-Gather 操作，将所有 rank 的 input 收集到 tensor_list 中
        dist.all_gather(tensor_list, input, group=pgm.process_group_manager.tp_group)
        # 将收集到的张量列表沿着最后一个维度拼接起来
        output = torch.cat(tensor_list, dim=last_dim).contiguous()
        # 返回拼接后的完整张量
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: 从后续层传来的、对应于拼接后完整张量的梯度
        # 如果张量并行大小为 1，直接返回梯度
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output
        # 将梯度按照张量并行的大小，沿着最后一个维度切分
        chunks = split_tensor_along_last_dim(grad_output, pgm.process_group_manager.tp_world_size)
        # 只返回属于当前 rank 的那部分梯度
        return chunks[pgm.process_group_manager.tp_rank].contiguous()

# 自定义 Autograd Function 实现 Identity 操作，但在反向传播时执行 All-Reduce
# 用于 ColumnParallelLinear 的前向传播
class Copy(torch.autograd.Function):
    """前向传播是 Identity (恒等) 操作，反向传播执行 All-Reduce (SUM) 操作。"""
    @staticmethod
    def forward(ctx, input):
        # 前向传播直接返回输入
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: 从后续层传来的梯度
        # 如果张量并行大小为 1，直接返回梯度
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output
        # 在反向传播时，对梯度执行 All-Reduce 求和操作
        # 这是因为前向传播是 Identity，相当于输入被复制（广播）到了所有 TP 进程
        # 因此反向传播时需要将各个进程上的梯度加起来
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        # 返回求和后的梯度
        return grad_output

### end TP communications ###

# 函数：将张量并行应用于模型
def apply_tensor_parallel(model):

    # 内部辅助函数，用于替换模型中的特定模块
    def _replace_module(_module, _linear_proj_name, _style, args={}):
        # _module: 要修改的父模块 (例如 Attention 层或 MLP 层)
        # _linear_proj_name: 父模块中要被替换的原始层名称 (例如 "q_proj", "embedding")
        # _style: 并行化的类型 ("column", "row", "vocab")
        # args: 额外参数，例如 {"gather_output": True}

        # 检查并行类型是否有效
        assert _style in ["column", "row", 'vocab']
        # 获取原始的 nn.Linear 或 nn.Embedding 层
        linear_layer = getattr(_module, _linear_proj_name)

        # 根据并行类型创建对应的并行层
        if _style == "column":
            new_linear_layer = ColumnParallelLinear(
                in_features=linear_layer.in_features,      # 输入特征数不变
                out_features=linear_layer.out_features,    # 输出特征数不变 (并行层内部会处理切分)
                bias=linear_layer.bias is not None,       # 是否有偏置
                gather_output=args.get("gather_output", False) # 是否在输出时进行 gather 操作
            )
        elif _style == "row":
            new_linear_layer = RowParallelLinear(
                in_features=linear_layer.in_features,      # 输入特征数不变 (并行层内部会处理切分)
                out_features=linear_layer.out_features,    # 输出特征数不变
                bias=linear_layer.bias is not None,       # 是否有偏置
            )
        else: # _style == "vocab"
            new_linear_layer = VocabParallelEmbedding(
                num_embeddings=linear_layer.num_embeddings, # 词表大小 (并行层内部会处理切分)
                embedding_dim=linear_layer.embedding_dim,   # 嵌入维度不变
            )
        # 使用新的并行层替换掉原始层
        setattr(_module, _linear_proj_name, new_linear_layer)

    # 定义需要进行张量并行的层及其并行方式
    module_linear_name_stype_mapping_list = [
        # Attention 块中的层
        ("attention", "q_proj", "column"), # Query 投射，按列并行 (输出维度切分)
        ("attention", "k_proj", "column"), # Key 投射，按列并行
        ("attention", "v_proj", "column"), # Value 投射，按列并行
        ("attention", "out_proj", "row"),  # 输出投射，按行并行 (输入维度切分)
        # MLP 块中的层
        ("mlp", "up_proj", "column"),    # MLP 上采样投射，按列并行
        ("mlp", "gate_proj", "column"),  # MLP 门控投射，按列并行
        ("mlp", "down_proj", "row"),   # MLP 下采样投射，按行并行
    ]

    # 遍历模型中的所有 Decoder 层
    for layer in model.decoder_layers:
        # 遍历定义好的映射列表
        for module_name, linear_proj_name, style in module_linear_name_stype_mapping_list:
            # 对每个 Decoder 层中的 Attention 和 MLP 子模块应用替换
            _replace_module(getattr(layer, module_name), linear_proj_name, style)

    # 对模型顶层的 Embedding 层应用词表并行
    _replace_module(model, "embedding", "vocab")
    # 对模型最后的输出 Projection 层应用列并行，并设置 gather_output=True
    # 因为损失计算通常需要完整的 logits 输出
    _replace_module(model, "final_proj", "column", args={"gather_output": True})

    # 返回修改后的模型
    return model

# 列并行线性层 (替换 nn.Linear)
# 将权重矩阵 W 按列（输出维度）切分：W = [W_1, W_2, ..., W_N]
# 每个 rank 计算 Y_i = X * W_i^T + b_i
# 输入 X 需要是完整的 (通过 Copy 操作保证梯度正确回传)
class ColumnParallelLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool, gather_output: bool = False):
        # in_features: 原始输入的特征维度
        # out_features: 原始输出的特征维度
        # bias: 是否使用偏置
        # gather_output: 是否在前向传播结束时收集所有 rank 的输出

        super(ColumnParallelLinear, self).__init__() # 调用父类初始化

        # 获取张量并行的 world_size 和当前 rank
        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank

        self.in_features = in_features    # 保存输入特征数
        self.out_features = out_features  # 保存原始输出特征数
        # 检查输出特征数是否能被 TP world_size 整除
        assert out_features % self.tp_world_size == 0, "Hidden dimension must be divisible by the tensor parallel world size"
        # 计算每个 rank 上的输出分区大小
        self.output_size_per_partition = out_features // self.tp_world_size
        self.gather_output = gather_output # 保存是否需要 gather 输出

        # 注意: F.linear(X, W, b) 计算的是 X * W^T + b
        # 创建当前 rank 的权重分区 W_i，形状为 [output_size_per_partition, in_features]
        self.weight = nn.Parameter(torch.Tensor(self.output_size_per_partition, self.in_features))
        if bias:
            # 如果需要偏置，创建当前 rank 的偏置分区 b_i
            self.bias = nn.Parameter(torch.Tensor(self.output_size_per_partition))
            with torch.no_grad(): # 初始化偏置为 0
                self.bias.zero_()
        else:
            # 如果不需要偏置，注册为 None
            self.register_parameter("bias", None)

        # 初始化权重参数
        self.reset_parameters()

    def reset_parameters(self):
        """初始化权重。"""
        # 如果 TP world_size 为 1 (即不进行张量并行)，则使用 PyTorch 默认的 nn.Linear 初始化方法
        if self.tp_world_size == 1:
            # 计算初始化范围 k = 1 / in_features
            k = 1 / self.weight.size(1)
            bound = math.sqrt(k)
            # 使用均匀分布 U(-sqrt(k), sqrt(k)) 初始化
            torch.nn.init.uniform_(self.weight, -bound, bound)
            return

        # 如果 TP world_size > 1:
        # 1. 先创建一个完整的 "主权重" (master_weight)，但不计算梯度
        master_weight = torch.empty(self.out_features, self.in_features, dtype=self.weight.dtype, requires_grad=False)
        # 2. 仍然根据完整的输入维度计算初始化范围
        k = 1 / master_weight.size(1)
        bound = math.sqrt(k)
        # 3. 初始化完整的主权重
        torch.nn.init.uniform_(master_weight, -bound, bound)

        # 4. 将主权重按列 (dim=0，对应输出维度) 切分成 TP world_size 份
        weight_list = torch.split(master_weight, self.output_size_per_partition, dim=0)
        # 5. 将属于当前 rank 的那一份权重赋值给 self.weight
        # .data 表示直接修改权重张量的数据，不影响梯度追踪
        # .contiguous() 确保内存连续
        self.weight.data = weight_list[self.tp_rank].contiguous()

    def forward(self, input):
        """前向传播逻辑。"""
        # 1. 对输入应用 Copy 操作。
        #    前向：Identity，输入被“复制”到所有 TP 进程。
        #    反向：All-Reduce，确保回传给输入的梯度是所有 TP 进程上的梯度之和。
        input_parallel = Copy.apply(input)
        # 2. 执行线性计算：X * W_i^T + b_i，得到当前 rank 的输出分区 Y_i。
        output = F.linear(input_parallel, self.weight, self.bias)
        # 3. 如果需要 gather 输出 (通常是模型最后一层)
        if self.gather_output:
            # 对输出分区 Y_i 应用 Gather 操作。
            # 前向：All-Gather，将所有 rank 的 Y_i 拼接成完整的 Y = [Y_1, ..., Y_N]。
            # 反向：Split，将回传给 Y 的梯度切分，只把对应 Y_i 的梯度传回。
            output = Gather.apply(output)
        # 4. 返回计算结果 (可能是分区 Y_i 或完整 Y)
        return output

# 行并行线性层 (替换 nn.Linear)
# 权重矩阵 W 按行（输入维度）切分：W = [W_1; W_2; ...; W_N] (分号表示垂直堆叠)
# 输入 X 来自上一层的 ColumnParallelLinear，已经是切分好的 X = [X_1, ..., X_N]
# 每个 rank 计算 Y_i = X_i * W_i^T
# 最后对所有 Y_i 进行 All-Reduce 求和得到最终输出 Y = sum(Y_i)
class RowParallelLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool):
        # in_features: 原始输入的特征维度
        # out_features: 原始输出的特征维度
        # bias: 是否使用偏置 (偏置通常加在 All-Reduce 之后，所以不需要切分)

        super(RowParallelLinear, self).__init__() # 调用父类初始化

        # 获取张量并行的 world_size 和当前 rank
        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank

        self.in_features = in_features    # 保存原始输入特征数
        self.out_features = out_features  # 保存输出特征数
        # 检查输入特征数是否能被 TP world_size 整除
        assert in_features % self.tp_world_size == 0, "Hidden dimension must be divisible by the tensor parallel world size"
        # 计算每个 rank 上的输入分区大小
        self.input_size_per_partition = in_features // self.tp_world_size

        # 创建当前 rank 的权重分区 W_i，形状为 [out_features, input_size_per_partition]
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.input_size_per_partition))
        if bias:
            # 偏置不切分，每个 rank 都持有完整的偏置 b
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
            # 初始化偏置为 0
            with torch.no_grad():
                self.bias.zero_()
        else:
            # 注册偏置为 None
            self.register_parameter("bias", None)

        # 初始化权重参数
        self.reset_parameters()

    def reset_parameters(self):
        """初始化权重。"""
        # 如果 TP world_size 为 1
        if self.tp_world_size == 1:
            # 使用 PyTorch 默认初始化，注意 k 是基于权重的第二维度 (输入维度)
            k = 1 / self.weight.size(1)
            bound = math.sqrt(k)
            torch.nn.init.uniform_(self.weight, -bound, bound)
            return

        # 如果 TP world_size > 1:
        # 1. 创建完整的主权重
        master_weight = torch.empty(self.out_features, self.in_features, dtype=self.weight.dtype, requires_grad=False)
        # 2. 根据完整权重的输入维度计算初始化范围
        k = 1 / master_weight.size(1)
        bound = math.sqrt(k)
        # 3. 初始化完整的主权重
        torch.nn.init.uniform_(master_weight, -bound, bound)

        # 4. 将主权重按行 (dim=1，对应输入维度) 切分成 TP world_size 份
        weight_list = torch.split(master_weight, self.input_size_per_partition, dim=1)
        # 5. 将属于当前 rank 的那一份权重赋值给 self.weight
        self.weight.data = weight_list[self.tp_rank].contiguous()

    def forward(self, input):
        """前向传播逻辑。"""
        # input 已经是上一层 ColumnParallelLinear 输出的分区 X_i
        # 1. 执行分区线性计算：X_i * W_i^T，得到当前 rank 的部分输出 Y_i
        output_parallel = F.linear(input, self.weight)
        # 2. 对部分输出 Y_i 应用 Reduce 操作
        #    前向：All-Reduce (Sum)，将所有 rank 的 Y_i 加起来得到最终输出 Y = sum(Y_i)
        #    反向：Identity，梯度直接传递给 Y_i
        output = Reduce.apply(output_parallel)
        # 3. 如果存在偏置，在 All-Reduce 之后加上偏置
        return output if self.bias is None else output + self.bias

# 词表并行嵌入层 (替换 nn.Embedding)
# 将词表权重矩阵 E 按行（词表维度）切分：E = [E_1; E_2; ...; E_N]
# 输入 token ids X
# 每个 rank 查找自己负责的那部分词表 E_i，得到部分嵌入 Y_i
# 对所有 Y_i 进行 All-Reduce 求和得到最终嵌入 Y = sum(Y_i)
class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,          # 原始词表大小
        embedding_dim: int,         # 嵌入维度
        padding_idx: Optional[int] = None, # padding token 的索引 (如果有)
        max_norm: Optional[float] = None,  # 最大范数约束 (如果有)
        norm_type: float = 2.0,         # 计算范数的类型
        scale_grad_by_freq: bool = False, # 是否根据词频缩放梯度
        sparse: bool = False            # 是否使用稀疏梯度
    ):

        super(VocabParallelEmbedding, self).__init__() # 调用父类初始化

        # 获取 TP world_size 和 rank
        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank

        # 保存原始参数
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        # 计算当前 rank 负责的词表范围 [start_index, end_index)
        self.vocab_start_index, self.vocab_end_index = self._vocab_range_from_global_vocab_size(
            self.num_embeddings, pgm.process_group_manager.tp_rank, pgm.process_group_manager.tp_world_size
        )
        # 计算当前 rank 负责的词表分区大小
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        # 创建当前 rank 的词表权重分区 E_i
        self.weight = nn.Parameter(torch.Tensor(self.num_embeddings_per_partition, self.embedding_dim))

        # 初始化权重
        self.reset_parameters()

    def _vocab_range_from_global_vocab_size(self, global_vocab_size: int, rank: int, world_size: int):
        """计算给定 rank 对应的词表范围。"""
        # 检查词表大小是否能被 TP world_size 整除
        assert global_vocab_size % world_size == 0, f"{global_vocab_size} is not divisible by {world_size}"
        # 计算每个分区的词表大小
        per_partition_vocab_size = global_vocab_size // world_size
        # 计算起始索引
        index_f = rank * per_partition_vocab_size
        # 计算结束索引 (不包含)
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    def reset_parameters(self):
        """初始化嵌入权重。"""
        # 如果 TP world_size 为 1
        if self.tp_world_size == 1:
            # 使用正态分布 N(0, 1) 初始化
            torch.nn.init.normal_(self.weight, mean=0.0, std=1.0)
            return

        # 如果 TP world_size > 1:
        # 1. 创建完整的主嵌入权重表
        master_weight = torch.empty(self.num_embeddings, self.embedding_dim, dtype=self.weight.dtype, requires_grad=False)
        # 2. 初始化完整的主嵌入表
        torch.nn.init.normal_(master_weight, mean=0.0, std=1.0)

        # 3. 将主嵌入表按行 (dim=0，对应词表维度) 切分成 TP world_size 份
        weight_list = torch.split(master_weight, self.num_embeddings_per_partition, dim=0)
        # 4. 将属于当前 rank 的那一份权重赋值给 self.weight
        self.weight.data = weight_list[self.tp_rank].contiguous()

    def forward(self, input):
        """
        前向传播逻辑：执行并行化的 Embedding 查找。
        1. 屏蔽掉不属于当前 rank 词表范围的 token ID。
        2. 对调整后的 token ID 执行本地 Embedding 查找。
        3. 将所有 rank 的本地 Embedding 结果通过 All-Reduce 求和。
        """
        # 1. 找出输入中哪些 token ID 不在当前 rank 的 [vocab_start_index, vocab_end_index) 范围内
        input_mask = (input < self.vocab_start_index) | (input >= self.vocab_end_index)
        # 2. 创建输入的副本，并将所有 token ID 减去当前 rank 的起始索引，使其变为本地索引
        masked_input = input.clone() - self.vocab_start_index
        #    将不在范围内的 token 的本地索引设为 0 (或其他有效索引)，避免查找时出错
        masked_input[input_mask] = 0
        # 3. 使用 F.embedding 执行本地查找，输入是调整后的 masked_input，权重是本地的 self.weight
        output_parallel = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx, # 传递其他 Embedding 参数
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # 4. 将那些原本不在范围内的 token 对应的嵌入向量置为 0
        #    因为它们的真实嵌入向量是由其他 rank 计算的
        output_parallel[input_mask, :] = 0.0
        # 5. 对所有 rank 的本地嵌入结果 (output_parallel) 应用 Reduce 操作
        #    前向：All-Reduce (Sum)，将每个 rank 计算的部分嵌入（对应其词表范围）加起来，得到最终完整的嵌入。
        #    反向：Identity，梯度直接传回给相应的本地 Embedding 层。
        output = Reduce.apply(output_parallel)
        # 6. 返回最终的嵌入向量
        return output