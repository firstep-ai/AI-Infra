import os
from PIL import Image
# from huggingface_hub import snapshot_download # 如果模型已在本地，则不需要
import torch
import re
from typing import List

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import cv2
import numpy as np
# --- 全局配置 ---
MAX_VIDEO_FRAMES = 4  # 视频最多采样帧数
DO_IMAGE_PADDING = False # 图像预处理时是否进行填充

# --- 视觉数据采样辅助函数 (极致简化) ---
def get_visual_type(path: str) -> str:
    path_lower = path.lower()
    # 假设只会测试这两种已知类型的文件
    if path_lower.endswith(('.mp4', '.avi', '.mov', '.mkv')): # 根据您的 EXAMPLE_VIDEO_PATH 调整
        return 'video'
    # else if path_lower.endswith(('.jpg', '.jpeg', '.png')): # 根据您的 EXAMPLE_IMAGE_PATH 调整
    # return 'image'
    return 'image' # 默认为图像，因为我们只测试这两个文件

def sample_image(image_path: str, n_frames: int = 1, start_time: int = 0, end_time: int = -1) -> List[Image.Image]:
    return [Image.open(image_path).convert('RGB')] # 直接打开，无错误处理

def sample_video(video_path: str, n_frames: int = 8, start_time: int = 0, end_time: int = -1) -> List[Image.Image]:
    """
    从视频中均匀采样指定数量的帧。

    Args:
        video_path (str): 视频文件的路径。
        n_frames (int): 要采样的帧数。默认为 8。
        start_time (int): 采样开始时间（秒）。默认为 0。
        end_time (int): 采样结束时间（秒）。默认为 -1，表示到视频末尾。

    Returns:
        List[Image.Image]: 包含采样帧的 PIL Image 对象列表。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 无法打开视频文件 {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) # 视频的帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 视频总帧数
    
    # 将时间转换为帧索引
    start_frame = int(start_time * fps)
    # 如果 end_time 为 -1，则采样到视频的最后一帧
    end_frame = int(end_time * fps) if end_time != -1 else total_frames - 1

    # 确保采样范围在视频的有效帧范围内
    start_frame = max(0, start_frame)
    end_frame = min(total_frames - 1, end_frame)

    if start_frame >= end_frame:
        print(f"Warning: 采样范围无效。起始帧 {start_frame} 必须小于结束帧 {end_frame}。")
        return []

    sampled_images: List[Image.Image] = []
    
    # 计算采样帧的间隔
    # 如果只采 1 帧，直接取中间帧
    if n_frames == 1:
        frame_indices = [ (start_frame + end_frame) // 2 ]
    else:
        # 均匀采样 n_frames 帧，包括起始和结束附近的帧
        # 使用 np.linspace 确保均匀分布
        frame_indices = np.linspace(start_frame, end_frame, n_frames, dtype=int)
    
    for frame_idx in frame_indices:
        # 设置视频当前帧的位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read() # 读取帧

        if ret:
            # OpenCV 默认读取的是 BGR 格式，需要转换为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 将 NumPy 数组转换为 PIL Image 对象
            pil_image = Image.fromarray(frame_rgb)
            sampled_images.append(pil_image)
        else:
            print(f"Warning: 无法读取帧 {frame_idx}。可能已到达视频末尾或帧损坏。")
            # 如果无法读取所有请求的帧，我们仍返回已读取的部分
            break 

    cap.release() # 释放视频文件
    return sampled_images

ext2sampler = {
    'image': sample_image,
    'video': sample_video
    # 其他采样器已移除
}

def load_visual_frames(
    visual_data_path: str,
    n_frames_to_sample: int,
    max_total_frames: int
) -> List[Image.Image]: # 假定总能成功加载
    sampler = ext2sampler[get_visual_type(visual_data_path)] # 直接访问，无检查
    return sampler(visual_data_path, n_frames=min(n_frames_to_sample, max_total_frames))

# --- 图像预处理器类 (PIL to PIL) (极致简化初始化) ---
class CustomImageProcessor:
    def __init__(self, processor_hf) -> None:
        img_proc = processor_hf.image_processor
        # 强假设: image_processor.crop_size 是一个包含 'height' 和 'width' 的字典
        # 并且 image_processor.image_mean 存在。
        # 如果您的模型 processor 结构不同，这里会直接报错。
        crop_config = img_proc.crop_size 
        self.target_size = (crop_config['height'], crop_config['width'])
        self.image_mean = img_proc.image_mean

    def __call__(self, images: List[Image.Image], do_padding: bool) -> List[Image.Image]:
        processed_images = []
        for img in images:
            if do_padding:
                # 简化：expand2square 内部会检查是否已经是方形
                processed_images.append(self.expand2square(
                    img, tuple(int(x * 255) for x in self.image_mean)))
            else:
                # 简化：resize2square 内部会检查是否已经是目标尺寸
                processed_images.append(self.resize2square(img))
        return processed_images

    def expand2square(self, pil_img: Image.Image, background_color: tuple) -> Image.Image:
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def resize2square(self, pil_img: Image.Image) -> Image.Image:
        if pil_img.size == self.target_size: # 保留这个小优化
            return pil_img
        return pil_img.resize(self.target_size)

# --- 提示词格式化函数 (核心逻辑保留) ---
def format_multimodal_prompt(
    original_prompt: str,
    num_images: int,
    image_token_str: str
) -> str:
    image_placeholders = (image_token_str * num_images + "\n") if num_images > 0 else ""
    clean_prompt = original_prompt.replace("：", ":")
    question_text = ""
    
    match = re.search(r"USER:(.*?)ASSISTANT:", clean_prompt, re.IGNORECASE | re.DOTALL)
    
    if match:
        question_text = match.group(1).strip()
    else:
        last_user_idx = clean_prompt.upper().rfind("USER:")
        if last_user_idx != -1:
            question_text = clean_prompt[last_user_idx + len("USER:"):].strip()
        else:
            question_text = original_prompt.strip()

        if question_text.upper().endswith("ASSISTANT:"):
            question_text = question_text[:-len("ASSISTANT:")].strip()

    final_question_part = f"{image_placeholders}{question_text}".strip()
    return f"USER: {final_question_part} ASSISTANT:"

# --- 主脚本 (极致简化) ---
if __name__ == "__main__":
    LOCAL_MODEL_DIR = "./tarsier-7b" # 必须存在
    EXAMPLE_IMAGE_PATH = "kitty.jpg"   # 必须存在
    EXAMPLE_VIDEO_PATH = "kitchen.mp4" # 必须存在 (将使用 mock sample_video)

    # 环境变量可按需保留
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # os.environ["VLLM_CPU_ONLY"] = "1"

    print(f"加载vLLM模型: {LOCAL_MODEL_DIR}")
    try: # 关键步骤的加载保留 try-except
        llm = LLM(model=LOCAL_MODEL_DIR, trust_remote_code=True)
    except Exception as e:
        print(f"vLLM 模型加载失败: {e}")
        exit()
    print("vLLM 模型加载成功.")

    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=150, stop=["Human:", "Assistant:", "USER:", "ASSISTANT:"])

    print("加载Hugging Face Processor...")
    try: # 关键步骤的加载保留 try-except
        hf_processor = AutoProcessor.from_pretrained(LOCAL_MODEL_DIR, trust_remote_code=True)
    except Exception as e:
        print(f"Hugging Face Processor 加载失败: {e}")
        exit()
        
    tokenizer = hf_processor.tokenizer
    image_token = getattr(tokenizer, 'image_token', None) or "<image>"
    print(f"Processor 加载成功. 图像标记: {image_token}")
    
    custom_image_handler = CustomImageProcessor(hf_processor) # 初始化基于强假设

    # --- 测试场景 (移除所有条件跳过和大部分错误捕获) ---

    # 场景 1: 纯文本测试
    # print(f"\n--- 纯文本测试 ---")
    # question_text_only = "Please introduce yourself."
    # final_prompt_text_only = format_multimodal_prompt(question_text_only, 0, image_token)
    # vllm_inputs_text_only = {"prompt": final_prompt_text_only}
    # print(f"提示: {final_prompt_text_only}")
    # outputs = llm.generate(vllm_inputs_text_only, sampling_params) # 直接生成
    # for output_item in outputs:
    #     print(f"生成: {output_item.outputs[0].text}\n" + "-" * 20)

    # 场景 2: 文本和单张图像测试
    # print(f"\n--- 文本和单张图像测试 ---")
    # question_single_image = "Please describe the image."
    # raw_frames_single = load_visual_frames(EXAMPLE_IMAGE_PATH, 1, 1)
    # pil_images_single = custom_image_handler(raw_frames_single, do_padding=DO_IMAGE_PADDING)
    # num_images_single = len(pil_images_single)
    # print(f"从 '{EXAMPLE_IMAGE_PATH}' 加载处理了 {num_images_single} 帧.")

    # final_prompt_single_image = format_multimodal_prompt(question_single_image, num_images_single, image_token)
    # vllm_inputs_single_image = {
    #     "prompt": final_prompt_single_image,
    #     "multi_modal_data": {"image": pil_images_single}
    # }
    # outputs = llm.generate(vllm_inputs_single_image, sampling_params) # 直接生成
    # for output_item in outputs:
    #     print(f"生成: {output_item.outputs[0].text}\n" + "-" * 20)

    # 场景 3: 文本和视频（多帧图像）测试
    print(f"\n--- 文本和视频（作为多帧图像）测试 ---")
    question_video = "Please describe the video."
    raw_frames_video = load_visual_frames(EXAMPLE_VIDEO_PATH, MAX_VIDEO_FRAMES, MAX_VIDEO_FRAMES)
    pil_images_video = custom_image_handler(raw_frames_video, do_padding=DO_IMAGE_PADDING)
    num_images_video = len(pil_images_video)
    print(f"从 '{EXAMPLE_VIDEO_PATH}' 加载处理了 {num_images_video} 帧.")
        
    final_prompt_video = format_multimodal_prompt(question_video, num_images_video, image_token)
    vllm_inputs_video = {
        "prompt": final_prompt_video,
        "multi_modal_data": {"image": pil_images_video}
    }
    outputs = llm.generate(vllm_inputs_video, sampling_params) # 直接生成
    for output_item in outputs:
        print(f"生成: {output_item.outputs[0].text}\n" + "-" * 20)

    print("\n所有测试完成。")