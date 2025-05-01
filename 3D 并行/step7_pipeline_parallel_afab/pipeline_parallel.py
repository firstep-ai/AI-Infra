import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import process_group_manager as pgm

### begin PP communications
STEP, VERBOSE = 0, os.environ.get("VERBOSE", "0") == "1"
def pipeline_communicate(operation, device, dtype, tensor=None, shapes=None):
    """
    Handles point-to-point communication between pipeline stages for forward and backward passes.
    
    Args:
        operation (str): Type of communication operation ('recv_forward', 'send_forward', 
                        'recv_backward', 'send_backward')
        device: Target device for tensor operations (e.g., CPU, GPU)
        dtype: Data type for tensors
        tensor: Input tensor for send operations (default: None)
        shapes: Shape specifications for receiving tensors (default: None)
    
    Returns:
        torch.Tensor or None: Received tensor for receive operations, None for send operations
    """
    global STEP
    global VERBOSE
    
    if operation == 'recv_forward':
        # Skip if this is the first pipeline stage (nothing to receive)
        if pgm.process_group_manager.pp_is_first_stage: return None
        # Create empty tensor to receive data
        tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        src = pgm.process_group_manager.pp_prev_rank
    
    elif operation == 'send_forward':
        # Skip if this is the last pipeline stage (nothing to send forward)
        if pgm.process_group_manager.pp_is_last_stage: return
        dest = pgm.process_group_manager.pp_next_rank
    
    elif operation == 'recv_backward':
        # Skip if this is the last pipeline stage (nothing to receive from backward)
        if pgm.process_group_manager.pp_is_last_stage: return None
        tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        src = pgm.process_group_manager.pp_next_rank
    
    elif operation == 'send_backward':
        # Skip if this is the first pipeline stage (nothing to send backward)
        if pgm.process_group_manager.pp_is_first_stage: return
        dest = pgm.process_group_manager.pp_prev_rank

    # Determine if this is a send operation and set peer rank
    is_send = operation.startswith('send')
    peer_rank = dest if is_send else src
    
    # Create P2P operation (send or receive)
    op = dist.P2POp(dist.isend if is_send else dist.irecv, tensor, peer_rank)
    
    if VERBOSE: 
        print(f"{operation} | {'sending' if is_send else 'receiving'} {operation.split('_')[1]} "
              f"{pgm.process_group_manager.pp_rank} {'→' if is_send else '←'} {peer_rank} | "
              f"STEP:{STEP} | RANK:{pgm.process_group_manager.pp_rank}", flush=True)
    
    # Execute communication operation and wait for completion
    [req.wait() for req in dist.batch_isend_irecv([op])]
    torch.cuda.synchronize()
    
    if VERBOSE: STEP += 1
    
    # Return received tensor for receive operations, None for send operations
    return tensor if not is_send else None
### end PP communications

### begin Pipeline Parallel
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
        # torch.autograd.backward will automatically accumulates gradients in the leaves (cf: https://pytorch.org/docs/stable/generated/torch.autograd.backward.html)
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad, retain_graph=False, create_graph=False)
        return input_tensor.grad if input_tensor is not None else None

def train_step_pipeline_afab(model, data_loader, tensor_shapes, device, dtype):
    """
    Executes a training step using Activation Forward - Activation Backward (AFAB) pipeline parallelism.
    Implements separate forward and backward passes to optimize memory usage.
    """
    logging_loss: torch.float32 = 0.0
    input_tensors, output_tensors = [], []
    requires_grad_sync = pgm.process_group_manager.dp_world_size > 1

    # === All Forward Pass Phase ===
    for _ in range(data_loader.grad_acc_steps):
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=dtype)
        batch = next(data_loader)
        batch["hidden_states"] = input_tensor.to(device) if input_tensor is not None else input_tensor
        output_tensor = model.forward(input_ids=batch["input_ids"].to(device), position_ids=batch["position_ids"].to(device), hidden_states=batch["hidden_states"])
        pipeline_communicate(operation='send_forward', tensor=output_tensor, device=device, dtype=dtype)
        
        # calculate loss on the last stage
        if pgm.process_group_manager.pp_is_last_stage:
            output_tensor = F.cross_entropy(output_tensor.transpose(1, 2), batch["target_ids"].to(device), reduction='mean')
            logging_loss += output_tensor.item() / data_loader.grad_acc_steps

        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    # === All Backward Pass Phase ===
    for ith_microbatch in range(data_loader.grad_acc_steps):
        if requires_grad_sync:
            is_last_iteration = (ith_microbatch == data_loader.grad_acc_steps - 1)
            model.require_backward_grad_sync = is_last_iteration
        output_tensor_grad = pipeline_communicate(operation='recv_backward', shapes=tensor_shapes, device=device, dtype=dtype)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=dtype)

    return logging_loss

### end Pipeline Parallel