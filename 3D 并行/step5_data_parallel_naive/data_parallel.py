import contextlib
from typing import List
import torch
import torch.distributed as dist
from torch import nn

import process_group_manager as pgm

### begin Data Parallel (naive)
class DataParallelNaive(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        # whether to synchronize gradients during backward pass. Set to False when using gradient accumulation
        self.require_backward_grad_sync = True
        self.register_backward_hook(self._allreduce_grads)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def register_backward_hook(self, hook):
        """Registers a backward hook for all parameters of the model that require gradients.""" 
        for p in self.module.parameters():
            if p.requires_grad is True:
                p.register_hook(hook)
                
    def _allreduce_grads(self, grad):
        """Performs an all-reduce operation to synchronize gradients across multiple processes."""
        # No synchronization needed during gradient accumulation, except at the final accumulation step.
        if self.require_backward_grad_sync:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.dp_group)
            grad /= pgm.process_group_manager.dp_world_size
        return grad
### end Data Parallel (naive)