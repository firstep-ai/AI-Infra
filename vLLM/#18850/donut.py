from typing import Optional, List

import torch
from torch import nn
from transformers.utils import logging

from vllm.config import VllmConfig

from vllm.model_executor.model_loader.weight_utils import default_weight_loader

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
    # _tied_weights_keys 应该被更新，因为 encoder.embed_tokens.weight 不再存在
    _tied_weights_keys = ["decoder.embed_tokens.weight"]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config

        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        # 使用 SwinEncoder 替换 BartEncoder
        # 这里需要根据你的具体情况调整 input_size, embed_dim, num_heads 等参数
        # 这些参数应该与 SwinTransformer 预训练模型保持一致，并确保 SwinEncoder 的输出
        # embed_dim 与 BartConfig 的 d_model 匹配，以便解码器可以正确处理。
        # 假设 BartConfig 的 d_model 与 SwinEncoder 的最终输出维度匹配。
        # 举例参数，请根据实际Donut模型使用的Swin配置进行调整
        self.encoder = SwinEncoder(
            input_size=(config.encoder_layers, config.encoder_attention_heads), # 示例参数，需要替换为图像实际大小
            embed_dim=config.d_model,  # SwinEncoder的输出维度应与Bart的d_model匹配
            num_heads=[config.encoder_attention_heads // (2**i) for i in range(len(config.encoder_layers))], # 示例
            encoder_layer=[config.encoder_layers // 4] * 4, # 示例
        )

        self.decoder = BartDecoder(config,
                                   cache_config,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.decoder")

    def forward(self, 
                pixel_values: torch.Tensor, # 图像输入
                input_ids: torch.Tensor, 
                positions: torch.Tensor, 
                ) -> torch.Tensor:
        r"""
        Args:
            pixel_values:
                Input image tensor. Shape: (batch_size, num_channels, height, width)
            input_ids
                Indices of *decoder* input sequence tokens in the vocabulary.
                Padding will be ignored by default should you
                provide it.
            positions
                Positions of *decoder* input sequence tokens.
        Returns:
            Model output torch.Tensor
        """

        # 通过 SwinEncoder 处理图像输入
        encoder_hidden_states = self.encoder(pixel_values)

        # decoder outputs consists of
        # (dec_features, past_key_value, dec_hidden, dec_attn)
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
        return "Donut" # 更改模型名称为 Donut

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in default_weight_loader(
                model_name_or_path, self._tied_weights_keys, cache_dir,
                load_format, revision):
            # 将 BartEncoder 的权重加载路径映射到 SwinEncoder 的路径
            # 这部分需要特别注意，因为SwinTransformer的内部结构和BartEncoder完全不同。
            # 通常，你会加载一个预训练的SwinTransformer模型权重，然后加载BartDecoder的权重。
            # 这里的 `default_weight_loader` 假设了基于 Hugging Face 的权重命名约定。
            if "encoder." in name:
                # SwinEncoder的权重加载需要独立处理，这里只是一个占位符。
                # 你需要根据实际情况，加载 SwinTransformer 的预训练权重到 self.encoder.model 中
                # 例如：
                # if "model." in name: # 假设 SwinEncoder 内部的 SwinTransformer 叫 model
                #     new_name = name.replace("encoder.model.", "encoder.model.")
                #     if new_name in params_dict:
                #         params_dict[new_name].data.copy_(loaded_weight)
                logger.warning(
                    f"Skipping loading weights for encoder. {name} "
                    "as SwinEncoder has a different structure. "
                    "You might need to load SwinTransformer weights separately."
                )
                continue
            
            # 对于decoder部分，保持原有的加载逻辑
            if name not in params_dict:
                logger.warning(f"Skipping loading parameter {name}.")
                continue
            param = params_dict[name]
            if loaded_weight.shape != param.shape:
                raise ValueError(
                    f"Shape mismatch for parameter {name}: "
                    f"loaded_weight.shape={loaded_weight.shape}, "
                    f"param.shape={param.shape}")
            param.data.copy_(loaded_weight)