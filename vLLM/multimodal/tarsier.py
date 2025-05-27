# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from collections.abc import Iterable, Mapping, Sequence
import math # Added for sqrt
from typing import (Final, Literal, Optional, Protocol, TypedDict, TypeVar,
                    Union, cast)

import torch
import torch.nn as nn
import torch.nn.functional as F # Added for F.embedding

from packaging.version import Version
from transformers import (BatchFeature, CLIPVisionConfig, LlavaConfig as HfLlavaConfig, # Renamed to avoid clash
                          PixtralVisionConfig, PretrainedConfig,
                          SiglipVisionConfig)
from transformers import __version__ as TRANSFORMERS_VERSION
# Tarsier specific processor and config might be different,
# but we'll adapt from LlavaProcessor for now.
# If Tarsier uses a custom processor, that needs to be handled.
from transformers.models.llava import LlavaProcessor
from transformers.models.pixtral import PixtralProcessor

from vllm.config import VllmConfig
from vllm.inputs import InputProcessingContext
from vllm.jsontree import json_map_leaves
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalInputs, MultiModalKwargs)
from vllm.multimodal.parse import (ImageEmbeddingItems, ImageProcessorItems,
                                   ImageSize, MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, ProcessingCache,
                                        PromptReplacement, PromptUpdate,
                                        PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

# Assuming these are correctly pathed from your vLLM LLaVA implementation
from .clip import CLIPVisionModel
from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .pixtral import PixtralHFEncoderInfo, PixtralHFVisionModel
from .siglip import SiglipVisionModel
from .utils import (AutoWeightsLoader, flatten_bn, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings)
from .vision import get_vision_encoder_info, VisionEncoderInfo


# --- Type Definitions (largely reusable from vLLM LLaVA) ---
class TarsierImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor


class PixtralHFImagePixelInputs(TypedDict): # Reusing from LLaVA
    type: Literal["pixel_values_pixtral"]
    pixel_values: Union[torch.Tensor, list[torch.Tensor]]


class TarsierImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor


TarsierImageInputs = Union[TarsierImagePixelInputs, PixtralHFImagePixelInputs,
                           TarsierImageEmbeddingInputs]


# --- Tarsier Specific Config Protocol ---
# We need to ensure LlavaConfig from transformers.models.llava.configuration_llava
# or a similar structure is used, which includes image_newline_idx and image_new_idx
class TarsierHfConfig(Protocol): # Based on the provided Tarsier's LlavaConfig
    vision_config: Final[PretrainedConfig]
    text_config: Final[PretrainedConfig] # Added from Tarsier's LlavaConfig
    image_token_index: Final[int]
    vision_feature_select_strategy: Final[str]
    vision_feature_layer: Final[Union[int, list[int]]]
    projector_hidden_act: Final[str]
    # Tarsier specific fields from its LlavaConfig
    image_newline_idx: Final[int]
    image_new_idx: Final[int]
    # Optional, for multimodal projector bias, might be in LlavaConfig or a new one
    multimodal_projector_bias: bool = True # Defaulting, adjust if needed


class TarsierLikeProcessor(Protocol): # Reusing LLaVA's structure
    image_token: Final[str]


# --- Tarsier MultiModal Projector ---
class TarsierMultiModalProjector(nn.Module):
    def __init__(self,
                 vision_hidden_size: int,
                 text_hidden_size: int,
                 projector_hidden_act: str,
                 multimodal_projector_bias: bool, # Added from LLaVA vLLM
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()

        # Using vLLM's parallel layers
        self.linear_1 = ColumnParallelLinear(vision_hidden_size,
                                             text_hidden_size,
                                             bias=multimodal_projector_bias,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.linear_1")
        self.act = get_act_fn(projector_hidden_act) # get_act_fn from vLLM
        self.linear_2 = RowParallelLinear(text_hidden_size,
                                          text_hidden_size,
                                          bias=multimodal_projector_bias,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.linear_2")

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


# --- Tarsier Processing Info ---
class TarsierProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self) -> TarsierHfConfig:
        # Load Tarsier's specific LlavaConfig.
        # This might require ensuring HfLlavaConfig can load Tarsier's custom fields
        # or using a custom config class that Tarsier provides if it's not HfLlavaConfig.
        # For this example, we assume HfLlavaConfig is flexible enough or adapted.
        return self.ctx.get_hf_config(HfLlavaConfig) # Use the HF LlavaConfig

    def get_vision_encoder_info(self) -> VisionEncoderInfo: # Explicit type hint
        return get_vision_encoder_info(self.get_hf_config())

    def get_hf_processor(self, **kwargs: object) -> TarsierLikeProcessor:
        # Assuming Tarsier uses a standard LlavaProcessor or compatible
        hf_processor = self.ctx.get_hf_processor(LlavaProcessor, **kwargs)
        # Patch for patch_size if needed (copied from vLLM LLaVA)
        if hasattr(hf_processor, 'patch_size') and hf_processor.patch_size is None:
            patch_size = self.get_vision_encoder_info().get_patch_size()
            hf_processor.patch_size = patch_size
        return hf_processor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None} # Typically no hard limit for images from processor

    def _apply_feature_select_strategy(
        self,
        strategy: str,
        encoder_num_image_tokens: int,
    ) -> int:
        if strategy == "default": # Usually removes CLS token
            return encoder_num_image_tokens -1 # check Tarsier's specific strategy
        if strategy == "full":
            return encoder_num_image_tokens
        # Add other strategies if Tarsier uses them
        msg = f"Unexpected feature select strategy: {strategy!r}"
        raise NotImplementedError(msg)

    def get_num_image_tokens(
        self,
        *,
        image_width: int, # image_width/height might be -1 for Mantis-like calc
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        vision_encoder_info = self.get_vision_encoder_info()

        # Number of patches from vision encoder after selection
        # This is num_image_patches in Tarsier's _merge_input_ids_with_image_features
        # It's the number of tokens output by the projector for a single image *before* add_split_tokens
        num_projected_patches = self._apply_feature_select_strategy(
            hf_config.vision_feature_select_strategy,
            vision_encoder_info.get_num_image_tokens(
                image_width=image_width,
                image_height=image_height,
            ),
        )

        # Now, account for Tarsier's add_split_tokens logic
        # num_height_patches = int(math.sqrt(num_projected_patches))
        # This assumes num_projected_patches is a perfect square, which is common.
        # If image_width/height is -1, we might need a default calculation
        # For now, let's assume we get a valid num_projected_patches
        if num_projected_patches <= 0: # Should not happen with valid inputs
            # Fallback or error for invalid patch count, e.g. if width/height was -1
            # and vision_encoder_info couldn't determine a default.
            # For robust calculation when image_width/height is -1 (like in Mantis example),
            # we might need to get default image size first.
            default_size = self.get_image_size_with_most_features()
            num_projected_patches_default = self._apply_feature_select_strategy(
                hf_config.vision_feature_select_strategy,
                vision_encoder_info.get_num_image_tokens(
                    image_width=default_size.width,
                    image_height=default_size.height,
                ),
            )
            if num_projected_patches_default <=0:
                 raise ValueError("Could not determine a valid number of image patches.")
            num_projected_patches = num_projected_patches_default


        if not (num_projected_patches > 0 and math.isqrt(num_projected_patches)**2 == num_projected_patches):
            # This can happen if vision_feature_select_strategy="full" and CLS is token 0,
            # and total patches (including CLS) isn't N*N+1. Or if strategy is default and N*N+1-1 isn't M*M.
            # Tarsier's `add_split_tokens` assumes a grid.
            # We need to be careful here. If the number of patches AFTER selection is not a perfect square,
            # Tarsier's `sqrt` logic for `num_height_patches` will be problematic.
            # For now, let's assume the common case where selected patches form a square grid (e.g., 24x24=576 for ViT-L/14 336px).
            # If Tarsier's `selected_image_feature` is always (N_images, H_patches * W_patches, embed_dim), then this is fine.
            # Let's print a warning or raise an error if it's not a perfect square, as Tarsier's logic depends on it.
             pass # Allow non-perfect square, Tarsier's math.sqrt will truncate.
                  # Or, one might need to adjust how num_height_patches is derived.
                  # The original Tarsier code uses `int(math.sqrt(num_image_patches))`.
                  # This implies `num_image_patches` refers to the count *after* projector and *before* split tokens.


        num_height_patches = int(math.sqrt(num_projected_patches))

        # Total tokens for one image after Tarsier's add_split_tokens:
        # projected_patches + newline_tokens (one per row of patches) + one_new_image_token
        total_image_tokens_for_llm = num_projected_patches + num_height_patches + 1
        return total_image_tokens_for_llm

    def get_image_size_with_most_features(self) -> ImageSize:
        vision_encoder_info = self.get_vision_encoder_info()
        width = height = vision_encoder_info.get_image_size()
        return ImageSize(width=width, height=height)

    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()
        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
        )

    # Tarsier specific accessors
    def get_image_newline_idx(self) -> int:
        return self.get_hf_config().image_newline_idx

    def get_image_new_idx(self) -> int:
        return self.get_hf_config().image_new_idx


_I_Tarsier = TypeVar("_I_Tarsier", bound=TarsierProcessingInfo)


# --- Tarsier Dummy Inputs Builder (can reuse LLaVA's if compatible) ---
class TarsierDummyInputsBuilder(BaseDummyInputsBuilder[_I_Tarsier]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        processor = self.info.get_hf_processor()
        image_token = processor.image_token
        return image_token * num_images # One <image> token per image in text

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }


# --- Tarsier MultiModal Processor ---
class TarsierMultiModalProcessor(BaseMultiModalProcessor[_I_Tarsier]):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # Similar to LLaVA, assuming "pixel_values" for images from processor
        # and "image_embeds" if providing pre-computed embeddings
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        image_token_id = hf_config.image_token_index # The <IMAGE> token ID

        # The replacement for each <IMAGE> token in the prompt needs to be
        # a sequence of `image_token_id` repeated `N` times, where `N` is
        # the *final* number of tokens one image expands to for the LLM.
        # This count is determined by `TarsierProcessingInfo.get_num_image_tokens`.

        def get_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems))

            if isinstance(images, ImageEmbeddingItems):
                # If pre-computed embeddings are provided, their feature size
                # must already match the final expected size after Tarsier's processing.
                # This is complex because `add_split_tokens` uses LLM's embedder.
                # For simplicity, let's assume pre-computed embeds are not yet split.
                # Or, the user must provide them already split.
                # This part might need more careful design if supporting pre-split embeds.
                # For now, let's assume image_embeds are like projector output.
                num_projected_patches = images.get_feature_size(item_idx)
                # This assumes num_projected_patches is a perfect square
                num_height_patches = int(math.sqrt(num_projected_patches))
                num_final_image_tokens = num_projected_patches + num_height_patches + 1

            else: # ImageProcessorItems (pixels)
                image_size = images.get_image_size(item_idx)
                # This will use the full logic including add_split_tokens effect
                num_final_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                )

            return [image_token_id] * num_final_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id], # Replace each single <IMAGE> token
                replacement=get_replacement,
            ),
        ]


# --- Helper for choosing processor/info (similar to LLaVA's) ---
def _build_tarsier_hf_info(ctx: InputProcessingContext) -> TarsierProcessingInfo:
    # For Tarsier, it's not differentiating like Llava vs Pixtral for the base.
    # It always uses its own logic on top of a vision model.
    return TarsierProcessingInfo(ctx)

def _build_tarsier_hf_processor(
    info: _I_Tarsier, # Type should be TarsierProcessingInfo
    dummy_inputs: BaseDummyInputsBuilder[_I_Tarsier],
    *,
    cache: Optional[ProcessingCache] = None,
) -> BaseMultiModalProcessor:
    if isinstance(info, TarsierProcessingInfo):
        return TarsierMultiModalProcessor(
            info,
            dummy_inputs,
            cache=cache,
        )
    raise NotImplementedError(type(info))


# --- Vision Tower Initialization (reusable from vLLM LLaVA if Tarsier's tower is compatible) ---
def init_vision_tower_for_tarsier(
    hf_config: TarsierHfConfig, # Use the Tarsier specific config protocol
    quant_config: Optional[QuantizationConfig],
    *,
    require_post_norm: Optional[bool] = None,
    prefix: str = "",
) -> Union[CLIPVisionModel, SiglipVisionModel, PixtralHFVisionModel]: # Add other vision models if Tarsier supports
    vision_config = hf_config.vision_config

    # Determine num_hidden_layers (reusing LLaVA's logic)
    feature_layers = hf_config.vision_feature_layer
    base_num_hidden_layers = vision_config.num_hidden_layers # Corrected: vision_config, not hf_config

    def _get_layer_index(feature_layer_index: int, num_hidden_layers_total: int) -> int:
        if feature_layer_index < 0:
            return num_hidden_layers_total + feature_layer_index + 1
        return feature_layer_index

    if isinstance(feature_layers, int):
        num_hidden_layers_to_init = _get_layer_index(feature_layers, base_num_hidden_layers)
    elif isinstance(feature_layers, (list, tuple)):
        num_hidden_layers_to_init = max(
            _get_layer_index(idx, base_num_hidden_layers) for idx in feature_layers)
    else:
        raise TypeError(f"vision_layer_feature type: {type(feature_layers)} is not supported")


    if isinstance(vision_config, CLIPVisionConfig):
        return CLIPVisionModel(
            vision_config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_to_init,
            require_post_norm=require_post_norm,
            prefix=prefix,
        )
    elif isinstance(vision_config, SiglipVisionConfig):
        return SiglipVisionModel(
            vision_config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_to_init,
            require_post_norm=require_post_norm,
            prefix=prefix,
        )
    elif isinstance(vision_config, PixtralVisionConfig): # If Tarsier supports Pixtral
        return PixtralHFVisionModel(
            vision_config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_to_init,
            require_post_norm=require_post_norm,
            prefix=prefix,
        )

    msg = f"Unsupported vision config for Tarsier: {type(vision_config)}"
    raise NotImplementedError(msg)


# --- Tarsier Model Implementation ---
@MULTIMODAL_REGISTRY.register_processor(_build_tarsier_hf_processor,
                                        info=_build_tarsier_hf_info,
                                        dummy_inputs=TarsierDummyInputsBuilder)
class TarsierForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):

    # Packed modules mapping if Tarsier's LLM uses it (e.g. Llama based)
    # This would come from the specific LLM architecture being used.
    # Assuming a Llama-like LLM for now.
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        # We expect vllm_config.model_config.hf_config to be Tarsier's LlavaConfig
        # or a compatible one that includes Tarsier's specific fields.
        config: TarsierHfConfig = vllm_config.model_config.hf_config # type: ignore
        quant_config = vllm_config.quant_config
        # multimodal_config = vllm_config.model_config.multimodal_config # Not directly used here

        self.config = config # Storing the Tarsier-specific HF config

        # Initialize Vision Tower
        self.vision_tower = init_vision_tower_for_tarsier(
            config,
            quant_config,
            require_post_norm=False, # Adjust if Tarsier needs post_norm
            prefix=maybe_prefix(prefix, "vision_tower"))

        # Initialize Multi-Modal Projector
        # Ensure projector_hidden_act and multimodal_projector_bias are in TarsierHfConfig
        projector_bias = getattr(config, "multimodal_projector_bias", True) # Default if not in config

        self.multi_modal_projector = TarsierMultiModalProjector(
            vision_hidden_size=config.vision_config.hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            projector_hidden_act=config.projector_hidden_act,
            multimodal_projector_bias=projector_bias,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "multi_modal_projector"))

        # Initialize Language Model
        # Tarsier's text_config should define the LLM.
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config, # Use text_config from Tarsier's main config
            prefix=maybe_prefix(prefix, "language_model"),
        )

        # Register Tarsier specific token indices as buffers
        # These will be used in _add_tarsier_split_tokens
        self.register_buffer('image_newline_idx_tensor',
                             torch.tensor([config.image_newline_idx], dtype=torch.long),
                             persistent=False)
        self.register_buffer('image_new_idx_tensor',
                             torch.tensor([config.image_new_idx], dtype=torch.long),
                             persistent=False)

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)


    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        # Standard validation from vLLM LLaVA
        # Tarsier might have different image size expectations,
        # config.vision_config.image_size should reflect that.
        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w) # Assuming 3 channels
        actual_dims = tuple(data.shape[1:])

        if actual_dims != expected_dims:
            expected_expr = ("batch_size", *map(str, expected_dims))
            raise ValueError(
                f"The expected shape of pixel values is {expected_expr}. "
                f"You supplied {tuple(data.shape)}.")
        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[TarsierImageInputs]:
        # Adapted from vLLM LLaVA, Tarsier might not support Pixtral directly
        # but the structure is reusable.
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            # Assuming Tarsier doesn't have a special Pixtral-like case here
            # unless its vision_config.model_type implies it.
            # For now, default to TarsierImagePixelInputs
            return TarsierImagePixelInputs(
                type="pixel_values",
                pixel_values=self._validate_pixel_values(
                    flatten_bn(pixel_values, concat=True)), # type: ignore
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")
            # Tarsier's `add_split_tokens` makes direct use of image_embeds complex
            # if they aren't already structured correctly.
            # Assuming image_embeds are projector outputs before splitting.
            return TarsierImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds, concat=True), # type: ignore
            )

        raise AssertionError("This line should be unreachable.")

    def _select_image_features(self, image_features: torch.Tensor, *,
                               strategy: str) -> torch.Tensor:
        # Copied from vLLM LLaVA, Tarsier's strategies should match
        if strategy == "default": # Typically, remove [CLS] token
            return image_features[:, 1:]
        elif strategy == "full":
            return image_features
        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    def _image_pixels_to_features(
        self,
        vision_tower: Union[CLIPVisionModel, SiglipVisionModel, PixtralHFVisionModel],
        pixel_values: Union[torch.Tensor, list[torch.Tensor]],
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        # From vLLM LLaVA, vision tower output handling
        image_hidden_states = vision_tower(pixel_values) # This is a tuple if multiple layers are returned

        # If vision_tower returns multiple layers based on config, select the correct one(s).
        # The init_vision_tower_for_tarsier already ensures only necessary layers are computed
        # and vision_tower should ideally return the features from the configured vision_feature_layer.
        # If image_hidden_states is a tuple (from multiple feature layers),
        # Tarsier original code implies selecting ONE layer: image_outputs.hidden_states[vision_feature_layer]
        # Our vision_tower (e.g. CLIPVisionModel) is modified to return only the last layer it computed.
        # So, image_hidden_states here should be the tensor from the desired layer.
        if not isinstance(image_hidden_states, torch.Tensor):
             # This case might occur if the vision tower is not adapted to return a single tensor
             # or if json_map_leaves is used before selection. For now, assume it's a tensor.
             # If it's a tuple from PixtralHFVisionModel, handle it.
            if isinstance(self.vision_tower, PixtralHFVisionModel) and isinstance(image_hidden_states, tuple):
                # Pixtral might return a tuple of tensors if handling lists of images
                pass # It will be handled by json_map_leaves below
            else:
                raise TypeError(f"Expected tensor from vision tower, got {type(image_hidden_states)}")


        def select_features_fn(leaf: torch.Tensor):
            return self._select_image_features(
                leaf,
                strategy=self.config.vision_feature_select_strategy,
            )
        # If pixel_values was a list (e.g. for Pixtral), image_hidden_states might be a tuple/list of tensors.
        # json_map_leaves handles this.
        selected_features = cast(
            Union[torch.Tensor, tuple[torch.Tensor, ...]],
            json_map_leaves(select_features_fn, image_hidden_states),
        )
        return selected_features


    def _add_tarsier_split_tokens(
        self,
        projected_image_features: torch.Tensor # Shape: (num_images, num_projected_patches, embed_dim)
    ) -> torch.Tensor:
        """
        Implements Tarsier's `add_split_tokens` logic.
        """
        num_images, num_projected_patches, embed_dim = projected_image_features.shape

        # This must be an integer for Tarsier's logic to hold.
        # num_projected_patches is after feature selection (e.g., 576 for 24x24 grid)
        if not (num_projected_patches > 0 and math.isqrt(num_projected_patches)**2 == num_projected_patches):
            # This indicates an issue with either the vision model output patch count
            # or the feature selection strategy not resulting in a perfect square grid.
            # Tarsier original code `int(math.sqrt(num_image_patches))` would proceed with a truncated int.
            # For robustness, one might prefer an error or a more adaptive grid logic.
            # Following original Tarsier's behavior:
            # logger.warning(f"num_projected_patches ({num_projected_patches}) is not a perfect square. "
            #                "Tarsier's split token logic might behave unexpectedly.")
            pass # Allow, as int(sqrt()) will truncate.

        num_height_patches = int(math.sqrt(num_projected_patches))
        # This implies num_width_patches would also be num_height_patches for a square grid.
        # If not square, Tarsier's original code still uses sqrt(total_patches) for height_patches.
        num_width_patches = num_projected_patches // num_height_patches # Assuming grid

        # Get embeddings for special tokens using the LLM's input embedding layer
        # Ensure tensors are on the same device as image_features
        device = projected_image_features.device
        # 假设 self.language_model 是 vLLM 的 LlamaForCausalLM (或类似模型)
        # 它有一个 'model' 属性 (LlamaModel)
        # LlamaModel 有一个 'embed_tokens' 属性 (这是 nn.Embedding 层)
        embedding_layer = self.language_model.model.embed_tokens
        image_newline_emb = embedding_layer(self.image_newline_idx_tensor.to(device)).squeeze(0)
        image_new_emb = embedding_layer(self.image_new_idx_tensor.to(device)).squeeze(0)


        # Reshape features to (num_images, num_height_patches, num_width_patches, embed_dim)
        # This assumes num_projected_patches = num_height_patches * num_width_patches
        # If num_projected_patches wasn't a perfect square, num_width_patches calculation above ensures this.
        try:
            current_image_features_grid = projected_image_features.view(
                num_images, num_height_patches, num_width_patches, embed_dim
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"Cannot reshape projected_image_features with shape {projected_image_features.shape} "
                f"to ({num_images}, {num_height_patches}, {num_width_patches}, {embed_dim}). "
                "Ensure num_projected_patches is compatible with a grid structure. "
                f"num_projected_patches={num_projected_patches}, derived num_height_patches={num_height_patches}, "
                f"derived num_width_patches={num_width_patches}. Original error: {e}"
            )


        # Add image_newline tokens (one per row of patches)
        # Expand image_newline_emb: (embed_dim) -> (num_images, num_height_patches, 1, embed_dim)
        image_newline_expanded = image_newline_emb.expand(
            (num_images, num_height_patches, 1, embed_dim)
        )
        features_with_newlines = torch.cat(
            [current_image_features_grid, image_newline_expanded], dim=2 # Concatenate along width dim
        )

        # Reshape back to (num_images, new_num_patches_after_newline, embed_dim)
        # new_num_patches_after_newline = num_height_patches * (num_width_patches + 1)
        #                                = num_projected_patches + num_height_patches
        new_num_patches_after_newline = num_projected_patches + num_height_patches
        features_with_newlines_flat = features_with_newlines.view(
            num_images, new_num_patches_after_newline, embed_dim
        )

        # Add image_new token (one per image)
        # Expand image_new_emb: (embed_dim) -> (num_images, 1, embed_dim)
        image_new_expanded = image_new_emb.expand((num_images, 1, embed_dim))
        final_image_features = torch.cat(
            [features_with_newlines_flat, image_new_expanded], dim=1 # Concatenate along patch sequence dim
        )
        # Final shape: (num_images, num_projected_patches + num_height_patches + 1, embed_dim)
        return final_image_features


    def _process_image_pixels(
        self,
        inputs: Union[TarsierImagePixelInputs, PixtralHFImagePixelInputs],
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        assert self.vision_tower is not None
        pixel_values = inputs["pixel_values"]
        # 1. Get features from vision tower & select
        image_features_selected = self._image_pixels_to_features(self.vision_tower, pixel_values) # type: ignore

        # 2. Pass through projector
        # This needs to handle if image_features_selected is a tuple (e.g. from Pixtral list input)
        if isinstance(image_features_selected, torch.Tensor):
            projected_features = self.multi_modal_projector(image_features_selected)
            # 3. Apply Tarsier's split token logic
            final_features = self._add_tarsier_split_tokens(projected_features)
            return final_features
        elif isinstance(image_features_selected, tuple): # e.g. Pixtral with list of images
            projected_list = [self.multi_modal_projector(feat) for feat in image_features_selected]
            # Apply split tokens to each tensor in the list
            final_list = [self._add_tarsier_split_tokens(proj_feat) for proj_feat in projected_list]
            return tuple(final_list)
        else:
            raise TypeError(f"Unexpected type from _image_pixels_to_features: {type(image_features_selected)}")


    def _process_image_input(
        self,
        image_input: TarsierImageInputs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]: # Can return tuple for list inputs
        if image_input["type"] == "image_embeds":
            # If image_embeds are provided, they are assumed to be projector outputs
            # *before* Tarsier's split token logic.
            projected_features = image_input["data"]
            if isinstance(projected_features, torch.Tensor):
                # Apply Tarsier's split token logic
                return self._add_tarsier_split_tokens(projected_features)
            else: # Should not happen with current flatten_bn logic
                raise ValueError("image_embeds data should be a single tensor after flatten_bn.")

        # Process pixel inputs
        assert self.vision_tower is not None # Should have been caught by _parse_and_validate
        return self._process_image_pixels(image_input) # type: ignore


    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        return self._process_image_input(image_input)


    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        # This is standard vLLM merging logic.
        # `multimodal_embeddings` here are the *final* image features after Tarsier's processing.
        # `self.config.image_token_index` is the ID of the <IMAGE> placeholder token.
        # `merge_multimodal_embeddings` will replace these placeholders with the actual features.
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings, # These are already Tarsier-processed
                self.config.image_token_index,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # Standard vLLM forward pass structure
        if intermediate_tensors is not None:
            inputs_embeds = None # Use intermediate_tensors if provided (pipeline parallel)
        elif inputs_embeds is None:
            # This path is typically taken by the model runner which calls
            # get_multimodal_embeddings and get_input_embeddings.
            # If called directly, ensure kwargs contain pixel_values or image_embeds.
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, vision_embeddings)
            input_ids = None # Embeddings are now primary input to LLM

        # Forward pass through the language model
        hidden_states = self.language_model.model( # type: ignore
            input_ids=input_ids, # Will be None if inputs_embeds is provided
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states, sampling_metadata)
    
    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)