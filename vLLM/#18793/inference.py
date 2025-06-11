# -*- coding: utf-8 -*-
import numpy as np
import os
import torch
import random
import argparse
from PIL import Image

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

# 导入项目自身的模块
from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer

# --- 1. 参数解析 ---
# 修改了 argparse 以适应命令行操作
parser = argparse.ArgumentParser(description="BAGEL 模型纯后台推理脚本")
# [修改点 1]：在 choices 中增加了 'txt2txt'
parser.add_argument("--task", type=str, required=True, choices=["txt2img", "img2txt", "edit", "txt2txt"],
                    help="要执行的任务: 'txt2img' (文生图), 'img2txt' (图像理解), 'edit' (图像编辑), 'txt2txt' (文本生成)")
parser.add_argument("--prompt", type=str, required=True, help="输入的文本提示词")
parser.add_argument("--input_image", type=str, help="输入图片的路径 (对于 'img2txt' 和 'edit' 任务是必需的)")
parser.add_argument("--output_path", type=str, default="output.png", help="生成图片的保存路径 (用于 'txt2img' 和 'edit' 任务)")
parser.add_argument("--model_path", type=str, default="models/BAGEL-7B-MoT", help="模型文件所在的路径")
parser.add_argument("--mode", type=int, default=1, choices=[1, 2, 3], help="模型加载模式: 1 for bfloat16, 2 for NF4, 3 for INT8")
parser.add_argument("--seed", type=int, default=42, help="随机种子，用于保证结果可复现")
args = parser.parse_args()


# --- 2. 模型初始化和加载 (与原脚本相同) ---

print(f"正在加载模型: {args.model_path}")
print(f"使用加载模式: {args.mode} (1:bf16, 2:NF4, 3:INT8)")

# 模型配置
llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers -= 1

vae_model, vae_config = load_ae(local_path=os.path.join(args.model_path, "ae.safetensors"))

config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config, 
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
    latent_patch_size=2,
    max_latent_size=64,
)

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model      = SiglipVisionModel(vit_config)
    model          = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)

# 多GPU推理和模型加载
device_map = infer_auto_device_map(
    model,
    max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
)

same_device_modules = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'llm2vae',
    'connector',
    'vit_pos_embed'
]

if torch.cuda.device_count() == 1:
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device
        else:
            device_map[k] = "cuda:0"
else:
    first_device = device_map.get(same_device_modules[0])
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device

if args.mode == 1: # bfloat16
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(args.model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        offload_folder="offload",
        dtype=torch.bfloat16,
        force_hooks=True,
    ).eval()
elif args.mode == 2: # NF4
    bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False, bnb_4bit_quant_type="nf4")
    model = load_and_quantize_model(
        model, 
        weights_location=os.path.join(args.model_path, "ema.safetensors"), 
        bnb_quantization_config=bnb_quantization_config,
        device_map=device_map,
        offload_folder="offload",
    ).eval()
elif args.mode == 3: # INT8
    bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, torch_dtype=torch.float32)
    model = load_and_quantize_model(
        model, 
        weights_location=os.path.join(args.model_path, "ema.safetensors"), 
        bnb_quantization_config=bnb_quantization_config,
        device_map=device_map,
        offload_folder="offload",
    ).eval()
else:
    raise NotImplementedError

# 推理器准备
inferencer = InterleaveInferencer(
    model=model,
    vae_model=vae_model,
    tokenizer=tokenizer,
    vae_transform=vae_transform,
    vit_transform=vit_transform,
    new_token_ids=new_token_ids,
)

print("模型加载完成！")

# --- 3. 核心推理函数 ---

def set_seed(seed):
    """设置随机种子以保证可复现性"""
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

def text_to_image(prompt, seed=0, image_ratio="1:1"):
    """文生图函数"""
    set_seed(seed)
    
    if image_ratio == "1:1":
        image_shapes = (1024, 1024)
    elif image_ratio == "4:3":
        image_shapes = (768, 1024)
    # ...可以根据需要添加更多比例
    else:
        image_shapes = (1024, 1024)

    inference_hyper = dict(
        cfg_text_scale=4.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        image_shapes=image_shapes,
    )
    
    result = inferencer(text=prompt, **inference_hyper)
    return result["image"]


def image_understanding(image: Image.Image, prompt: str):
    """图像理解函数"""
    image = pil_img2rgb(image)
    
    inference_hyper = dict(
        do_sample=False,
        text_temperature=0.3,
        max_think_token_n=512,
    )
    
    result = inferencer(image=image, text=prompt, understanding_output=True, **inference_hyper)
    return result["text"]


def edit_image(image: Image.Image, prompt: str, seed=0):
    """图像编辑函数"""
    set_seed(seed)
    image = pil_img2rgb(image)
    
    inference_hyper = dict(
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
    )
    
    result = inferencer(image=image, text=prompt, **inference_hyper)
    return result["image"]

def text_to_text(prompt: str):
    """纯文本生成函数"""
    # 这里可以设置文本生成的超参数
    inference_hyper = dict(
        do_sample=True,          # 开启采样，让回答更多样
        text_temperature=0.7,    # 设置温度，0.7是一个常用值
        max_think_token_n=1024,  # 设置最大生成长度
    )
    
    # 调用推理器时，只传入文本，并设置 understanding_output=True 来确保它执行文本生成
    result = inferencer(text=prompt, understanding_output=True, **inference_hyper)
    return result["text"]


# --- 4. 主执行逻辑 ---
if __name__ == "__main__":
    
    # 检查任务所需的参数是否提供
    if args.task in ["img2txt", "edit"] and not args.input_image:
        raise ValueError(f"任务 '{args.task}' 需要提供 '--input_image' 参数。")
    
    print(f"\n正在执行任务: {args.task}")
    print(f"提示词: {args.prompt}")

    if args.task == "txt2img":
        # 执行文生图
        generated_image = text_to_image(prompt=args.prompt, seed=args.seed)
        # 保存图片
        generated_image.save(args.output_path)
        print(f"图像生成成功，已保存至: {args.output_path}")

    elif args.task == "img2txt":
        # 执行图像理解
        input_image = Image.open(args.input_image)
        understanding_result = image_understanding(image=input_image, prompt=args.prompt)
        print("\n--- 模型输出 ---")
        print(understanding_result)
        print("--- 输出结束 ---")

    elif args.task == "edit":
        # 执行图像编辑
        input_image = Image.open(args.input_image)
        edited_image = edit_image(image=input_image, prompt=args.prompt, seed=args.seed)
        # 保存图片
        edited_image.save(args.output_path)
        print(f"图像编辑成功，已保存至: {args.output_path}")

    # [修改点 3]：增加处理 txt2txt 任务的逻辑分支
    elif args.task == "txt2txt":
        # 执行文本生成
        text_result = text_to_text(prompt=args.prompt)
        print("\n--- 模型输出 ---")
        print(text_result)
        print("--- 输出结束 ---")