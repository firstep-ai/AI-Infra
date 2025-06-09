from typing import Optional, List

import torch
from torch import nn
from transformers.utils import logging

from vllm.config import VllmConfig


from .bart import BartDecoder

from .interfaces import SupportsQuant

# 导入 timm 库中的 SwinTransformer
try:
    from timm.models.swin_transformer import SwinTransformer
except ImportError:
    raise ImportError(
        "Please install timm to use SwinEncoder: pip install timm")

logger = logging.get_logger(__name__)


def get_bsz_seq_len(input_ids):
    shp = input_ids.shape
    ndim = len(shp)
    if ndim == 1:
        return 1, input_ids.numel()
    else:
        return shp[:2]

class SwinEncoder(nn.Module):
    r"""
    Encoder based on SwinTransformer
    Set the initial weights and configuration with a pretrained SwinTransformer and then
    modify the detailed configurations

    Args:
        input_size: Input image size (width, height)
        align_long_axis: Whether to rotate image if height is greater than width
        window_size: Window size(=patch size) of SwinTransformer
        encoder_layer: Number of layers of SwinTransformer encoder
        name_or_path: Name of a pretrained model name either registered in huggingface.co. or saved in local.
                      otherwise, `swin_base_patch4_window12_384` will be set (using `timm`).
    """

    def __init__(
        self,
        input_size,
        align_long_axis: bool = False,
        window_size: int = 7,
        encoder_layer: List[int] = [2, 2, 14, 2],
        patch_size: int = [4, 4],
        embed_dim: int = 128,
        num_heads: List[int] = [4, 8, 16, 32],
    ):
        super().__init__()
        if isinstance(input_size, int):
            input_size = [input_size, input_size]
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.model = SwinTransformer(
            img_size=self.input_size,
            depths=self.encoder_layer,
            window_size=self.window_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_classes=0,  # Set to 0 to remove the final classification head
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_channels, height, width)
        Returns:
            encoder_output: (batch_size, sequence_length, hidden_size)
        """
        # SwinTransformer forward pass
        # The output of self.model.layers is typically (B, H*W, C)
        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x)
        for layer in self.model.layers:
            x = layer(x)

        # The output `x` from SwinTransformer layers is already in (B, H*W, C)
        # where H*W is the sequence length and C is the hidden_size (embed_dim).
        # We need to ensure it matches the expected input format for BartDecoder.
        # Here, `C` should be equal to config.d_model for seamless integration.
        # If not, a linear projection might be needed.
        # Assuming embed_dim of Swin is compatible with d_model of Bart.
        return x


class DonutModel(nn.Module, SupportsQuant):
    _tied_weights_keys = ["decoder.embed_tokens.weight"]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        encoder_config = config.encoder
        decoder_config = config.decoder
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = decoder_config.vocab_size + lora_vocab
        self.org_vocab_size = decoder_config.vocab_size
        self.encoder = SwinEncoder(
            input_size=encoder_config.image_size,
            window_size=encoder_config.window_size,
            encoder_layer=encoder_config.depths,
            patch_size=encoder_config.patch_size,
            embed_dim=encoder_config.embed_dim,
            num_heads=encoder_config.num_heads,
        )
        self.decoder = BartDecoder(decoder_config,
                                   cache_config,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.decoder")

    def forward(self, 
                pixel_values: torch.Tensor,
                input_ids: torch.Tensor, 
                positions: torch.Tensor, 
                ) -> torch.Tensor:
        encoder_hidden_states = self.encoder(pixel_values)
        decoder_outputs = self.decoder(
            decoder_input_ids=input_ids,
            decoder_positions=positions,
            encoder_hidden_states=encoder_hidden_states)
        return decoder_outputs

    @classmethod
    def get_config(cls, vllm_config: VllmConfig):
        return vllm_config.model_config.hf_config

    @classmethod
    def get_name(cls):
        return "Donut"

    # ===================================================================
    #   ↓↓↓   这部分是本次修正的核心   ↓↓↓
    # ===================================================================
    def load_weights(self, weights: "Iterable[tuple[str, torch.Tensor]]") -> "set[str]":
            """
            [调试版本]
            这个版本加入了大量的打印语句，用于诊断为什么权重没有被加载。
            """
            # 1. 打印出模型自身认为存在的所有参数名
            print("\n" + "="*50)
            print("DEBUG: Inside load_weights. All parameters in the model:")
            params_dict = dict(self.named_parameters())
            model_param_names = set(params_dict.keys())
            # 为了方便查看，排序后打印
            for p_name in sorted(list(model_param_names)):
                print(f"  - {p_name}")
            print(f"Total parameters in model: {len(model_param_names)}")
            print("="*50 + "\n")

            loaded_params: set[str] = set()
            
            print("\n" + "="*50)
            print("DEBUG: Starting to process weights from checkpoint...")
            print("="*50)
            
            # 2. 遍历vLLM传入的权重列表
            processed_count = 0
            for name, loaded_weight in weights:
                processed_count += 1
                target_name = name
                
                # 打印正在处理的权重信息
                print(f"\n[{processed_count}] Processing checkpoint weight: '{name}'")

                if name.startswith("encoder."):
                    target_name = name.replace("encoder.", "encoder.model.", 1)
                    print(f"    - Name starts with 'encoder.', mapping to: '{target_name}'")

                # 3. 检查映射后的参数是否存在于我们的模型中
                if target_name in params_dict:
                    print(f"    - SUCCESS: Found corresponding parameter '{target_name}' in model.")
                    param = params_dict[target_name]
                    
                    if loaded_weight.shape != param.shape:
                        raise ValueError(
                            f"Shape mismatch for parameter {target_name} (from {name}): "
                            f"loaded_weight.shape={loaded_weight.shape}, "
                            f"param.shape={param.shape}")

                    param.data.copy_(loaded_weight)
                    loaded_params.add(target_name)
                else:
                    # 4. 如果没找到，这是一个关键的线索！
                    print(f"    - FAILURE: Could not find parameter '{target_name}' in the model.")

            print("\n" + "="*50)
            print("DEBUG: Weight processing finished.")
            print(f"Total weights processed from checkpoint: {processed_count}")
            print(f"Total weights successfully loaded into model: {len(loaded_params)}")
            print("="*50 + "\n")

            return loaded_params