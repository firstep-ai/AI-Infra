import os                     # 导入 os 模块，用于访问环境变量，比如 "LOCAL_RANK"。
import torch                  # 导入 PyTorch 库。
import torch.distributed as dist # 导入 PyTorch 的分布式通信库，简写为 dist。

class ProcessGroupManager:     # 定义一个名为 ProcessGroupManager 的类。
                              # 这个类的目的是管理分布式训练环境中的不同进程组和相关信息。

    def __init__(self, dp_size, pp_size, tp_size): 
                              # 类的构造函数（初始化方法）。
                              # 当创建 ProcessGroupManager 对象时被调用。
                              # 参数：
                              #   dp_size: 数据并行（Data Parallelism）的大小。
                              #   pp_size: 流水线并行（Pipeline Parallelism）的大小。
                              #   tp_size: 张量并行（Tensor Parallelism）的大小。

        self.global_rank = dist.get_rank() 
                              # 获取当前进程在所有进程中的全局唯一排名（rank），从 0 开始。
                              # 例如，如果有 8 个进程，它们的 global_rank 分别是 0 到 7。
        self.world_size = dist.get_world_size() 
                              # 获取分布式环境中的总进程数。
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.global_rank % self.world_size))
                              # 获取当前进程在其所在计算节点（机器/GPU卡）上的本地排名。
                              # 它首先尝试从环境变量 "LOCAL_RANK" 获取（通常由 torchrun 等启动器设置）。
                              # 如果环境变量不存在，则尝试用 global_rank 对 world_size 取模来估算（这种估算方式不一定在所有环境中都准确）。
        
        assert self.world_size == dp_size * pp_size * tp_size, f"World size ({self.world_size}) != DP ({dp_size}) * PP ({pp_size}) * TP ({tp_size})"
                              # 断言检查：确保传入的并行维度大小 (dp_size, pp_size, tp_size) 的乘积等于总进程数 (world_size)。
                              # 如果不相等，说明并行配置有误，程序会报错并退出。

        self.grid = torch.arange(self.world_size).view(dp_size, pp_size, tp_size)  
                              # 创建一个 3D 的网格 (grid)，将 0 到 world_size-1 的全局排名映射到这个 (DP, PP, TP) 坐标系中。
                              # 这有助于确定每个进程在不同并行维度上的位置。
                              # 例如，如果有 8 个进程，dp=2, pp=2, tp=2，这个 grid 可能看起来像：
                              # [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]

        # Find the position of the current process in the grid
        self.dp_rank, self.pp_rank, self.tp_rank = (self.grid == self.global_rank).nonzero().flatten().tolist()
                              # 在 self.grid 中查找当前进程 (self.global_rank) 对应的 3D 坐标。
                              # (self.grid == self.global_rank) 会产生一个布尔张量，目标位置为 True。
                              # .nonzero() 获取 True 值的索引（即坐标）。
                              # .flatten().tolist() 将坐标转换为 [dp_rank, pp_rank, tp_rank] 的列表。

        # Process group creation - Update indexing to match new grid order
        # 使用 dist.new_subgroups_by_enumeration 根据 grid 中的全局 rank 列表创建新的通信子组。
        # 这个函数会为每个传入的 rank 列表创建一个子组，并返回一个包含这些子组的列表。
        # 这里我们只取返回列表中的第一个元素 [0]，因为每次调用都只基于一个特定的并行维度模式创建组。
        
        self.tp_group = dist.new_subgroups_by_enumeration([self.grid[d, p, :].tolist() for d in range(dp_size) for p in range(pp_size)])[0]
                              # 创建张量并行（TP）通信组。
                              # 对于每个 DP 和 PP 组合 (d, p)，将该组合下所有 TP 维度的进程（`self.grid[d, p, :]`）作为一个列表传入。
                              # `dist.new_subgroups_by_enumeration` 会为每个这样的列表创建一个通信组。当前进程会获取其所属的那个 TP 通信组。
                              # 例如，所有 dp_rank=0, pp_rank=0 的进程属于同一个 TP 组。

        self.pp_group = dist.new_subgroups_by_enumeration([self.grid[d, :, t].tolist() for d in range(dp_size) for t in range(tp_size)])[0]
                              # 创建流水线并行（PP）通信组。
                              # 逻辑类似 TP 组，但这次是固定 DP 和 TP 维度，变化 PP 维度。
                              # 所有 dp_rank=0, tp_rank=0 的进程属于同一个 PP 组。

        self.dp_group = dist.new_subgroups_by_enumeration([self.grid[:, p, t].tolist() for p in range(pp_size) for t in range(tp_size)])[0]
                              # 创建数据并行（DP）通信组。
                              # 固定 PP 和 TP 维度，变化 DP 维度。
                              # 所有 pp_rank=0, tp_rank=0 的进程属于同一个 DP 组。

        self.pp_dp_group = dist.new_subgroups_by_enumeration([self.grid[:, :, t].flatten().tolist() for t in range(tp_size)])[0]
                              # 创建一个结合了 DP 和 PP 维度的通信组，但保持 TP 维度固定。
                              # 对于每个 TP rank (t)，将所有 DP 和 PP 维度的进程（`self.grid[:, :, t]`）组成一个列表。
                              # 这个组可能用于某些需要在固定 TP rank 内跨 DP 和 PP rank 进行通信的操作。

        self.world_group = dist.group.WORLD 
                              # 获取默认的全局通信组，包含所有进程。

        # Update group IDs with new grid ordering
        self.tp_group_ids = self.grid[self.dp_rank, self.pp_rank, :].tolist()
                              # 获取当前进程所在 TP 组中所有进程的全局 rank 列表。
        self.pp_group_ids = self.grid[self.dp_rank, :, self.tp_rank].tolist()
                              # 获取当前进程所在 PP 组中所有进程的全局 rank 列表。
        self.dp_group_ids = self.grid[:, self.pp_rank, self.tp_rank].tolist()
                              # 获取当前进程所在 DP 组中所有进程的全局 rank 列表。
               
        # Tensor parallelism
        self.tp_world_size = dist.get_world_size(group=self.tp_group)
                              # 获取当前进程所在 TP 组的大小。
        self.tp_first_rank = self.tp_group_ids[0]
                              # 获取当前进程所在 TP 组中第一个进程（rank 0）的全局 rank。
        self.tp_last_rank = self.tp_group_ids[-1]
                              # 获取当前进程所在 TP 组中最后一个进程的全局 rank。

        # Pipeline parallelism
        self.pp_world_size = dist.get_world_size(group=self.pp_group)
                              # 获取当前进程所在 PP 组的大小（即流水线深度）。
        self.pp_first_rank = self.pp_group_ids[0]
                              # 获取当前进程所在 PP 组中第一个 stage (rank 0) 的全局 rank。
        self.pp_last_rank = self.pp_group_ids[-1]
                              # 获取当前进程所在 PP 组中最后一个 stage 的全局 rank。
        self.pp_is_first_stage = self.pp_rank == 0
                              # 布尔值，判断当前进程是否是流水线的第一个 stage。
        self.pp_is_last_stage = self.pp_rank == self.pp_world_size - 1
                              # 布尔值，判断当前进程是否是流水线的最后一个 stage。
        self.pp_next_rank = None if self.pp_rank == self.pp_world_size - 1 else int(self.grid[self.dp_rank, self.pp_rank + 1, self.tp_rank].item())
                              # 获取流水线中下一个 stage 的全局 rank。
                              # 如果当前是最后一个 stage，则为 None。
                              # 否则，在 grid 中找到相同 DP 和 TP rank，但 PP rank 加 1 的那个进程的全局 rank。
        self.pp_prev_rank = None if self.pp_rank == 0 else int(self.grid[self.dp_rank, self.pp_rank - 1, self.tp_rank].item())
                              # 获取流水线中上一个 stage 的全局 rank。
                              # 如果当前是第一个 stage，则为 None。
                              # 否则，在 grid 中找到相同 DP 和 TP rank，但 PP rank 减 1 的那个进程的全局 rank。

        # Data parallelism
        self.dp_world_size = dist.get_world_size(group=self.dp_group)
                              # 获取当前进程所在 DP 组的大小。
        self.dp_first_rank = self.dp_group_ids[0]
                              # 获取当前进程所在 DP 组中第一个进程（rank 0）的全局 rank。
        self.dp_last_rank = self.dp_group_ids[-1]
                              # 获取当前进程所在 DP 组中最后一个进程的全局 rank。
        
    def __str__(self):        # 定义当打印 ProcessGroupManager 对象时如何显示。
        return f"DP({self.dp_world_size})-PP({self.pp_world_size})-TP({self.tp_world_size})-Rank({self.global_rank})"
                              # 返回一个包含 DP, PP, TP 大小和当前进程全局 Rank 的格式化字符串。

def setup_process_group_manager(dp_size, pp_size, tp_size): 
                              # 定义一个辅助函数，用于创建并设置一个全局的 ProcessGroupManager 实例。
    global process_group_manager 
                              # 声明将要修改全局变量 process_group_manager。
    process_group_manager = ProcessGroupManager(dp_size, pp_size, tp_size)
                              # 创建 ProcessGroupManager 类的实例，并将其赋值给全局变量。
                              # 这样其他模块可以通过 `import process_group_manager as pgm` 然后使用 `pgm.process_group_manager` 来访问这个全局实例。