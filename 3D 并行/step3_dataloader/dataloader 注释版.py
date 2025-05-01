# 导入所需的库
import torch  # PyTorch 核心库
from torch.utils.data import DataLoader  # PyTorch 数据加载器基类
import numpy as np  # 用于数值计算，特别是在数据预处理中
from functools import partial  # 用于创建偏函数，方便地预设函数参数
from datasets import Features, Sequence, Value, load_dataset  # Hugging Face datasets 库，用于加载和处理数据集
from transformers import AutoTokenizer  # Hugging Face transformers 库，用于自动加载预训练模型的分词器

import process_group_manager as pgm  # 导入之前定义的进程组管理器，用于获取分布式训练信息

# 定义一个名为 MicroBatchDataLoader 的类，它继承自 PyTorch 的 DataLoader
class MicroBatchDataLoader(DataLoader):
    # 类的构造函数（初始化方法）
    def __init__(self, seq_len, micro_batch_size, grad_acc_steps, dataset_name, tokenizer_name, max_tokens, num_workers, num_proc, split="train"):
        # 参数说明：
        # seq_len: 模型处理的序列长度
        # micro_batch_size: 每个设备单次处理的微批次大小
        # grad_acc_steps: 梯度累积步数
        # dataset_name: 要加载的数据集名称 (例如 "roneneldan/TinyStories")
        # tokenizer_name: 使用的分词器名称 (例如 "HuggingFaceTB/SmolLM-360M-Instruct")
        # max_tokens: 训练所需的最小总 token 数，用于提前检查数据集大小
        # num_workers: PyTorch DataLoader 使用的子进程数，用于数据加载
        # num_proc: datasets 库在进行 map 操作时使用的进程数，用于加速数据预处理
        # split: 要加载的数据集划分 (例如 "train", "validation")

        # 将传入的参数保存为类的属性
        self.micro_batch_size = micro_batch_size  # 保存微批次大小
        self.grad_acc_steps = grad_acc_steps      # 保存梯度累积步数
        self.seq_len = seq_len                  # 保存序列长度

        # 计算全局批次大小 (Global Batch Size)
        # 这是在一个优化器步骤 (optimizer step) 中实际处理的总样本数
        # 计算方法：微批次大小 * 梯度累积步数 * 数据并行组的大小
        self.global_batch_size = micro_batch_size * grad_acc_steps * pgm.process_group_manager.dp_world_size

        # 加载预训练分词器
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # 加载指定的数据集和划分
        self.dataset = load_dataset(dataset_name, split=split)

        # 对加载的原始数据集进行分词和分块处理
        # 调用类自身的 tokenize_dataset 方法完成此操作
        # "text" 是假设数据集中包含文本的列名
        # num_proc 用于并行处理以加速
        self.tokenized_dataset = self.tokenize_dataset(self.dataset, "text", self.seq_len, num_proc)

        # 计算处理后的数据集包含的总 token 数（大约值）
        # 每个样本长度为 seq_len + 1 (因为要包含目标 token)
        total_tokens = self.tokenized_dataset.num_rows * (self.seq_len + 1)
        # 断言检查：确保数据集中的总 token 数足够用于计划的训练
        assert total_tokens >= max_tokens, f"Not enough tokens. Have {total_tokens} tokens but need {max_tokens} tokens"

        # 调用父类 (DataLoader) 的构造函数
        super().__init__(
            self.tokenized_dataset,       # 第一个参数：要加载的数据集 (这里是已经分词和分块处理过的)
            batch_size=micro_batch_size,  # DataLoader 每次产出的批次大小（即微批次大小）
            collate_fn=self.collate_batch,# 指定一个自定义函数 (collate_batch) 来整理批次数据
            pin_memory=True,              # 如果为 True，数据加载器会将张量复制到 CUDA 固定内存中，可以加速 GPU 数据传输
            num_workers=num_workers,      # 数据加载时使用的子进程数量
            shuffle=False,                # 不打乱数据顺序 (通常在大型数据集上，打乱在预处理阶段完成或使用 Sampler 控制)
                                          # 注意：在后续步骤中，这里会加入 DistributedSampler
        )

    # 定义一个静态方法（或类方法），用于将一批文本分词并组合成固定长度的块
    def tokenizer_group_text(self, examples, tokenizer, sequence_length):
        """将一批文本进行分词，并将它们组合成长度为 sequence_length + 1 的块"""
        # 使用分词器对输入的文本列表 (examples) 进行批量编码
        tokenized_text_batch = tokenizer.batch_encode_plus(
            examples,                     # 输入的文本列表
            return_attention_mask=False,  # 不需要返回 attention mask
            return_token_type_ids=False, # 不需要返回 token type ids
            return_tensors='np'           # 返回 NumPy 数组
        )
        # 将所有样本的 token ID 连接成一个长的一维 NumPy 数组
        concatenated_tokens = {'input_ids': np.concatenate(tokenized_text_batch['input_ids'])}
        # 获取连接后 token 的总长度
        total_length = len(concatenated_tokens['input_ids'])

        # 确保总长度至少为一个块的长度 (sequence_length + 1)
        if total_length >= sequence_length + 1:
            # 计算可以形成多少个完整的块，丢弃末尾不足一个块长度的 token
            # (total_length - 1) // sequence_length 计算出块的数量
            # 再乘以 sequence_length 得到用于分块的总 token 数
            # 最后加 1 是因为每个块需要 sequence_length + 1 个 token
            total_length = ((total_length - 1) // sequence_length) * sequence_length + 1

        # 使用列表推导式将连接后的 tokens 切分成多个长度为 sequence_length + 1 的块
        result = {
            'input_ids': [
                concatenated_tokens['input_ids'][i : i + sequence_length + 1]  # 从 i 开始切片，长度为 sequence_length + 1
                # 步长为 sequence_length，确保每个块不重叠地接续
                for i in range(0, total_length - sequence_length, sequence_length)
            ]
        }
        # 返回包含分块后 input_ids 的字典
        return result

    # 定义一个方法，用于对整个数据集应用分词和分块操作
    def tokenize_dataset(self, dataset, text_column_name, sequence_length, num_proc):
        """对数据集进行分词，并将文本分组为长度为 sequence_length + 1 的块"""
        # 使用 functools.partial 创建一个预设了 tokenizer 和 sequence_length 参数的函数
        # 这个函数签名符合 dataset.map 的要求
        tokenizer_func = partial(
            self.tokenizer_group_text,  # 要调用的函数
            tokenizer=self.tokenizer,   # 预设 tokenizer 参数
            sequence_length=sequence_length # 预设 sequence_length 参数
        )

        # 使用 dataset.map 方法将 tokenizer_func 应用到数据集的每一批数据上
        tokenized_dataset = dataset.map(
            tokenizer_func,             # 应用的函数
            input_columns=text_column_name, # 指定输入数据的列名 (例如 "text")
            remove_columns=dataset.column_names, # 处理后移除原始数据集的所有列
            # 定义输出数据的结构和类型
            features=Features({
                # "input_ids" 是一个序列，元素类型为 int64，固定长度为 sequence_length + 1
                "input_ids": Sequence(feature=Value(dtype="int64"), length=sequence_length + 1)
            }),
            batched=True,               # 以批处理模式运行 map，效率更高
            num_proc=num_proc,          # 使用指定数量的进程并行处理
            load_from_cache_file=True,  # 如果之前处理过并生成了缓存文件，则直接加载缓存，避免重复处理
            desc=f"Grouping texts in chunks of {sequence_length+1}", # 显示处理进度条时的描述信息
        )

        # 返回处理后的数据集
        return tokenized_dataset

    # 定义自定义的批次整理函数 (collate function)
    def collate_batch(self, batch):
        # batch 是一个列表，列表中的每个元素是 tokenized_dataset 中的一个样本 (字典)

        # 从批次中的每个样本 (item) 取出 'input_ids'，转换为 PyTorch 张量，然后将它们堆叠成一个新的张量
        # batch_input_ids 的形状是 [batch_size, sequence_length + 1]
        batch_input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
        # 获取当前微批次的大小
        batch_size = batch_input_ids.size(0)
        # 创建模型的输入 IDs (input_ids): 取每个样本的前 sequence_length 个 token
        # contiguous() 确保存储是连续的，有时是某些 PyTorch 操作所必需的
        input_ids = batch_input_ids[:, :-1].contiguous()
        # 创建模型的目标 IDs (target_ids): 取每个样本从第二个 token 开始到最后的 token (即向左移动一位)
        target_ids = batch_input_ids[:, 1:].contiguous()
        # 创建位置 IDs (position_ids): 生成一个从 0 到 seq_len-1 的序列
        # unsqueeze(0) 增加一个维度，expand(batch_size, -1) 将其复制 batch_size 次
        position_ids = torch.arange(self.seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).contiguous()
        # 创建注意力掩码 (attn_mask): 创建一个下三角矩阵（causal mask），用于 Transformer 解码器
        # 形状是 [seq_len, seq_len]
        attn_mask = torch.tril(torch.ones((self.seq_len, self.seq_len), dtype=torch.bool))
        # 增加批次维度并扩展，使其形状变为 [batch_size, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

        # 返回一个包含所有准备好的模型输入的字典
        return {
            "input_ids": input_ids,         # 模型输入 token
            "target_ids": target_ids,       # 训练时的目标 token
            "position_ids": position_ids,   # 位置编码信息
            "attn_mask": attn_mask,         # 注意力掩码
            "hidden_states": None           # 隐藏状态，初始化为 None，可能在流水线并行中使用
        }

    # 实现迭代器协议的 __iter__ 方法
    def __iter__(self):
        # 检查内部迭代器 _iterator 是否已创建
        if self._iterator is None:
            # 如果未创建，则调用父类 DataLoader 的 __iter__ 方法来创建实际的数据迭代器
            self._iterator = super().__iter__()
        # 返回 DataLoader 对象本身，使其可迭代
        return self

    # 实现迭代器协议的 __next__ 方法
    def __next__(self):
        # 检查内部迭代器是否已创建，如果直接调用 next 而没有先调用 iter，则先创建
        if self._iterator is None:
            self._iterator = super().__iter__()
        # 尝试从内部迭代器获取下一个批次
        try:
            batch = next(self._iterator)
        # 如果内部迭代器耗尽，捕获 StopIteration 异常
        except StopIteration:
            # 重置内部迭代器状态为 None
            self._iterator = None
            # 重新引发 StopIteration 异常，以通知外部循环迭代结束
            raise StopIteration
        # 如果成功获取批次，则返回该批次
        return batch