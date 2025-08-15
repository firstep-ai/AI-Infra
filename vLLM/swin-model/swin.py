# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM implementation of Swin Transformer, adapted for vision language models."""

import collections.abc
import math
from collections.abc import Iterable
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.utils import torch_int

from vllm.attention.layer import MultiHeadAttention
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

# Define a placeholder config if transformers is not installed or for simplicity
class SwinConfig(PretrainedConfig):
    def __init__(self,
                 image_size=224,
                 patch_size=4,
                 num_channels=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 hidden_dropout_prob=0.0,
                 attention_probs_dropout_prob=0.0,
                 drop_path_rate=0.1,
                 hidden_act="gelu",
                 layer_norm_eps=1e-5,
                 **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps

# Copied from transformers.models.swin.modeling_swin.window_partition
def window_partition(input_feature: torch.Tensor,
                     window_size: int) -> torch.Tensor:
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = input_feature.view(batch_size, height // window_size,
                                       window_size, width // window_size,
                                       window_size, num_channels)
    windows = input_feature.permute(0, 1, 3, 2, 4,
                                    5).contiguous().view(-1, window_size,
                                                         window_size,
                                                         num_channels)
    return windows

# Copied from transformers.models.swin.modeling_swin.window_reverse
def window_reverse(windows: torch.Tensor, window_size: int, height: int,
                   width: int) -> torch.Tensor:
    """
    Merges windows to produce higher resolution features.
    """
    num_channels = windows.shape[-1]
    windows = windows.view(-1, height // window_size, width // window_size,
                           window_size, window_size, num_channels)
    windows = windows.permute(0, 1, 3, 2, 4,
                              5).contiguous().view(-1, height, width,
                                                   num_channels)
    return windows

# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor,
              drop_prob: float = 0.0,
              training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0], ) + (1, ) * (input.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output

class SwinDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

# Adapted from transformers.models.swin.modeling_swin.SwinPatchEmbeddings
class SwinPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: SwinConfig):
        super().__init__()
        image_size = (config.image_size if isinstance(config.image_size,
                                                      collections.abc.Iterable)
                      else (config.image_size, config.image_size))
        patch_size = (config.patch_size if isinstance(config.patch_size,
                                                      collections.abc.Iterable)
                      else (config.patch_size, config.patch_size))
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size[0],
                          image_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.projection = nn.Conv2d(config.num_channels,
                                    config.embed_dim,
                                    kernel_size=patch_size,
                                    stride=patch_size)

    def maybe_pad(self, pixel_values: torch.Tensor, height: int,
                  width: int) -> torch.Tensor:
        pad_right = (self.patch_size[1] - width % self.patch_size[1]
                     ) % self.patch_size[1]
        pad_bottom = (self.patch_size[0] - height % self.patch_size[0]
                      ) % self.patch_size[0]
        if pad_right > 0 or pad_bottom > 0:
            pixel_values = nn.functional.pad(
                pixel_values, (0, pad_right, 0, pad_bottom))
        return pixel_values

    def forward(
        self, pixel_values: torch.FloatTensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        _, _, height, width = pixel_values.shape
        pixel_values = self.maybe_pad(pixel_values, height, width)

        embeddings = self.projection(pixel_values)
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        return embeddings, output_dimensions


class SwinEmbeddings(nn.Module):
    """Construct the patch and position embeddings."""
    def __init__(self, config: SwinConfig):
        super().__init__()
        self.patch_embeddings = SwinPatchEmbeddings(config)
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, output_dimensions


class SwinAttention(nn.Module):
    """Swin Self-Attention, adapted for vLLM."""
    def __init__(self,
                 config: SwinConfig,
                 dim: int,
                 num_heads: int,
                 window_size: int,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of "
                f"attention heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = (window_size if isinstance(
            window_size, collections.abc.Iterable) else
                            (window_size, window_size))
        self.scale = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            bias=config.qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
        )
        self.out_proj = RowParallelLinear(
            input_size=dim,
            output_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) *
                        (2 * self.window_size[1] - 1),
                        self.num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w],
                                            indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:,
                                                                       None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index",
                             relative_position_index,
                             persistent=False)
        
        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(self.num_heads, tp_size)
        self.attn = MultiHeadAttention(self.num_heads_per_partition,
                                       self.head_dim, self.scale)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.FloatTensor] = None
                ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)
        combined_mask = self._get_rel_pos_bias()
        if attention_mask is not None:
            combined_mask = combined_mask + attention_mask.unsqueeze(1)
        
        attn_output = self.attn(query, key, value, attn_mask=combined_mask)
        output, _ = self.out_proj(attn_output)
        return output

class SwinMLP(nn.Module):
    """Swin MLP block, adapted for vLLM."""
    def __init__(self,
                 config: SwinConfig,
                 dim: int,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(
            dim,
            int(config.mlp_ratio * dim),
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            int(config.mlp_ratio * dim),
            dim,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SwinLayer(nn.Module):
    """Swin Transformer Layer, adapted for vLLM."""
    def __init__(self,
                 config: SwinConfig,
                 dim: int,
                 input_resolution: Tuple[int, int],
                 num_heads: int,
                 quant_config: Optional[QuantizationConfig] = None,
                 drop_path_rate: float = 0.0,
                 shift_size: int = 0,
                 prefix: str = ""):
        super().__init__()
        self.shift_size = shift_size
        self.window_size = config.window_size
        self.input_resolution = input_resolution

        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.attention = SwinAttention(
            config,
            dim,
            num_heads,
            window_size=self.window_size,
            quant_config=quant_config,
            prefix=f"{prefix}.attention",
        )
        self.drop_path = (SwinDropPath(drop_path_rate)
                          if drop_path_rate > 0.0 else nn.Identity())
        
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.mlp = SwinMLP(config, dim, quant_config=quant_config, prefix=f"{prefix}.mlp")

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
    
    def get_attn_mask(self, height: int, width: int, dtype: torch.dtype,
                      device: torch.device) -> Optional[torch.Tensor]:
        if self.shift_size > 0:
            img_mask = torch.zeros((1, height, width, 1),
                                   dtype=dtype,
                                   device=device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def maybe_pad(
        self, hidden_states: torch.Tensor, height: int, width: int
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        pad_right = (self.window_size - width % self.window_size
                     ) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size
                      ) % self.window_size
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
    ) -> torch.Tensor:
        height, width = input_dimensions
        batch_size, _, channels = hidden_states.shape
        shortcut = hidden_states

        hidden_states = self.layernorm_before(hidden_states)
        hidden_states = hidden_states.view(batch_size, height, width, channels)

        hidden_states, pad_values = self.maybe_pad(hidden_states, height,
                                                   width)
        _, height_pad, width_pad, _ = hidden_states.shape

        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states,
                                               shifts=(-self.shift_size,
                                                       -self.shift_size),
                                               dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        hidden_states_windows = window_partition(shifted_hidden_states,
                                                 self.window_size)
        hidden_states_windows = hidden_states_windows.view(
            -1, self.window_size * self.window_size, channels)
        attn_mask = self.get_attn_mask(height_pad,
                                       width_pad,
                                       dtype=hidden_states.dtype,
                                       device=hidden_states.device)

        attention_output = self.attention(hidden_states_windows, attn_mask)

        attention_windows = attention_output.view(-1, self.window_size,
                                                  self.window_size, channels)
        shifted_windows = window_reverse(attention_windows, self.window_size,
                                         height_pad, width_pad)

        if self.shift_size > 0:
            hidden_states = torch.roll(shifted_windows,
                                       shifts=(self.shift_size,
                                               self.shift_size),
                                       dims=(1, 2))
        else:
            hidden_states = shifted_windows

        if pad_values[3] > 0 or pad_values[5] > 0:
            hidden_states = hidden_states[:, :height, :width, :].contiguous()

        hidden_states = hidden_states.view(batch_size, height * width,
                                           channels)
        hidden_states = shortcut + self.drop_path(hidden_states)
        
        shortcut = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = shortcut + self.drop_path(hidden_states)
        
        return hidden_states


class SwinPatchMerging(nn.Module):
    """Swin Patch Merging Layer, adapted for vLLM."""
    def __init__(self,
                 dim: int,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = RowParallelLinear(4 * dim,
                                           2 * dim,
                                           bias=False,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.reduction")

    def maybe_pad(self, input_feature: torch.Tensor, height: int,
                  width: int) -> torch.Tensor:
        if (height % 2 == 1) or (width % 2 == 1):
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            input_feature = nn.functional.pad(input_feature, pad_values)
        return input_feature

    def forward(
        self, input_feature: torch.Tensor, input_dimensions: Tuple[int, int]
    ) -> torch.Tensor:
        height, width = input_dimensions
        batch_size, _, num_channels = input_feature.shape
        input_feature = input_feature.view(batch_size, height, width,
                                           num_channels)

        input_feature = self.maybe_pad(input_feature, height, width)

        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        input_feature = torch.cat(
            [input_feature_0, input_feature_1, input_feature_2, input_feature_3],
            -1)
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)

        input_feature = self.norm(input_feature)
        input_feature, _ = self.reduction(input_feature)
        return input_feature


class SwinStage(nn.Module):
    """Swin Transformer Stage, adapted for vLLM."""
    def __init__(self,
                 config: SwinConfig,
                 dim: int,
                 input_resolution: Tuple[int, int],
                 depth: int,
                 num_heads: int,
                 drop_path: list[float],
                 downsample: bool,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinLayer(
                config=config,
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                quant_config=quant_config,
                drop_path_rate=drop_path[i],
                shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                prefix=f"{prefix}.blocks.{i}",
            ) for i in range(depth)
        ])

        if downsample:
            self.downsample = SwinPatchMerging(
                dim=dim,
                quant_config=quant_config,
                prefix=f"{prefix}.downsample")
        else:
            self.downsample = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        height, width = input_dimensions
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, input_dimensions)

        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states, input_dimensions)
            output_dimensions = ((height + 1) // 2, (width + 1) // 2)
        else:
            output_dimensions = (height, width)

        return hidden_states, output_dimensions


class SwinEncoder(nn.Module):
    """Swin Transformer Encoder, adapted for vLLM."""
    def __init__(self,
                 config: SwinConfig,
                 grid_size: Tuple[int, int],
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.num_layers = config.num_layers
        dpr = [
            x.item()
            for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))
        ]
        
        self.layers = nn.ModuleList([
            SwinStage(
                config=config,
                dim=int(config.embed_dim * 2**i_layer),
                input_resolution=(grid_size[0] // (2**i_layer),
                                  grid_size[1] // (2**i_layer)),
                depth=config.depths[i_layer],
                num_heads=config.num_heads[i_layer],
                drop_path=dpr[sum(config.depths[:i_layer]):sum(
                    config.depths[:i_layer + 1])],
                downsample=(i_layer < self.num_layers - 1),
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{i_layer}",
            ) for i_layer in range(self.num_layers)
        ])

    def forward(self, hidden_states: torch.Tensor,
                input_dimensions: Tuple[int, int]) -> torch.Tensor:
        for layer_module in self.layers:
            hidden_states, input_dimensions = layer_module(
                hidden_states, input_dimensions)
        return hidden_states


class SwinVisionModel(nn.Module):
    """The main Swin Transformer model for vLLM."""
    config_class = SwinConfig
    main_input_name = "pixel_values"

    def __init__(self,
                 config: SwinConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config
        self.embeddings = SwinEmbeddings(config)
        self.encoder = SwinEncoder(config,
                                   self.embeddings.patch_embeddings.grid_size,
                                   quant_config=quant_config,
                                   prefix="encoder")
        
        num_features = int(config.embed_dim * 2**(self.num_layers - 1))
        self.norm = nn.LayerNorm(num_features, eps=config.layer_norm_eps)
        # We don't implement the pooling layer as it's not always used in VLMs.
        # The final hidden state is returned.

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states, input_dimensions = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states, input_dimensions)
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        # Mapping from HF weight names to vLLM's fused QKV layer
        stacked_params_mapping = [
            ("qkv.weight", "attention.self.query.weight", "q"),
            ("qkv.weight", "attention.self.key.weight", "k"),
            ("qkv.weight", "attention.self.value.weight", "v"),
            ("qkv.bias", "attention.self.query.bias", "q"),
            ("qkv.bias", "attention.self.key.bias", "k"),
            ("qkv.bias", "attention.self.value.bias", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        
        def rename_hf_weight(name: str) -> str:
            name = name.replace("attention.output.dense", "attention.proj")
            name = name.replace("intermediate.dense", "mlp.fc1")
            name = name.replace("output.dense", "mlp.fc2")
            return name

        for name, loaded_weight in weights:
            # Rename based on HF model structure
            if name.startswith("swin."): # Assuming a prefix like 'swin' from a VLM
                name = name[len("swin."):]
            
            # The top-level `norm` in HF SwinModel corresponds to our final norm
            if name.startswith("layernorm."):
                name = name.replace("layernorm.", "norm.")
            
            # This handles the main encoder blocks
            name = rename_hf_weight(name)

            is_stacked = False
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name in name:
                    vllm_name = name.replace(weight_name, param_name)
                    if vllm_name in params_dict:
                        param = params_dict[vllm_name]
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight, shard_id)
                        loaded_params.add(vllm_name)
                        is_stacked = True
                    break
            
            if not is_stacked and name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params