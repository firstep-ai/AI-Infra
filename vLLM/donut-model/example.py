# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import copy
import os
import re
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from datasets import load_dataset
from PIL import Image
from transformers import DonutProcessor
from vllm import LLM, SamplingParams
from vllm.inputs import ExplicitEncoderDecoderPrompt, TextPrompt, TokensPrompt


# --- 以下是你提供的辅助函数，保持不变 ---

# Copied from https://github.com/bytedance/Dolphin/utils/utils.py
@dataclass
class ImageDimensions:
    original_w: int
    original_h: int
    padded_w: int
    padded_h: int

# Copied from https://github.com/bytedance/Dolphin/utils/utils.py
def map_to_original_coordinates(x1, y1, x2, y2, dims: ImageDimensions) -> Tuple[int, int, int, int]:
    try:
        top = (dims.padded_h - dims.original_h) // 2
        left = (dims.padded_w - dims.original_w) // 2
        orig_x1 = max(0, x1 - left)
        orig_y1 = max(0, y1 - top)
        orig_x2 = min(dims.original_w, x2 - left)
        orig_y2 = min(dims.original_h, y2 - top)
        if orig_x2 <= orig_x1:
            orig_x2 = min(orig_x1 + 1, dims.original_w)
        if orig_y2 <= orig_y1:
            orig_y2 = min(orig_y1 + 1, dims.original_h)
        return int(orig_x1), int(orig_y1), int(orig_x2), int(orig_y2)
    except Exception as e:
        print(f"map_to_original_coordinates error: {str(e)}")
        return 0, 0, min(100, dims.original_w), min(100, dims.original_h)

# Copied from https://github.com/bytedance/Dolphin/utils/utils.py
def adjust_box_edges(image, boxes: list[list[float]], max_pixels=15, threshold=0.2):
    if isinstance(image, str):
        image = cv2.imread(image)
    img_h, img_w = image.shape[:2]
    new_boxes = []
    for box in boxes:
        best_box = copy.deepcopy(box)

        def check_edge(img, current_box, i, is_vertical):
            edge = current_box[i]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            if is_vertical:
                line = binary[current_box[1] : current_box[3] + 1, edge]
            else:
                line = binary[edge, current_box[0] : current_box[2] + 1]
            transitions = np.abs(np.diff(line))
            return np.sum(transitions) / len(transitions)

        edges = [(0, -1, True), (2, 1, True), (1, -1, False), (3, 1, False)]
        current_box = copy.deepcopy(box)
        current_box[0] = min(max(current_box[0], 0), img_w - 1)
        current_box[1] = min(max(current_box[1], 0), img_h - 1)
        current_box[2] = min(max(current_box[2], 0), img_w - 1)
        current_box[3] = min(max(current_box[3], 0), img_h - 1)

        for i, direction, is_vertical in edges:
            best_score = check_edge(image, current_box, i, is_vertical)
            if best_score <= threshold:
                continue
            for step in range(max_pixels):
                current_box[i] += direction
                if i == 0 or i == 2:
                    current_box[i] = min(max(current_box[i], 0), img_w - 1)
                else:
                    current_box[i] = min(max(current_box[i], 0), img_h - 1)
                score = check_edge(image, current_box, i, is_vertical)
                if score < best_score:
                    best_score = score
                    best_box = copy.deepcopy(current_box)
                if score <= threshold:
                    break
        new_boxes.append(best_box)
    return new_boxes

# Copied from https://github.com/bytedance/Dolphin/utils/utils.py
def process_coordinates(coords, padded_image, dims: ImageDimensions, previous_box=None):
    try:
        x1, y1 = int(coords[0] * dims.padded_w), int(coords[1] * dims.padded_h)
        x2, y2 = int(coords[2] * dims.padded_w), int(coords[3] * dims.padded_h)
        x1, y1, x2, y2 = max(0, min(x1, dims.padded_w - 1)), max(0, min(y1, dims.padded_h - 1)), max(0, min(x2, dims.padded_w)), max(0, min(y2, dims.padded_h))
        if x2 <= x1: x2 = min(x1 + 1, dims.padded_w)
        if y2 <= y1: y2 = min(y1 + 1, dims.padded_h)
        new_boxes = adjust_box_edges(padded_image, [[x1, y1, x2, y2]])
        x1, y1, x2, y2 = new_boxes[0]
        x1, y1, x2, y2 = max(0, min(x1, dims.padded_w - 1)), max(0, min(y1, dims.padded_h - 1)), max(0, min(x2, dims.padded_w)), max(0, min(y2, dims.padded_h))
        if x2 <= x1: x2 = min(x1 + 1, dims.padded_w)
        if y2 <= y1: y2 = min(y1 + 1, dims.padded_h)
        if previous_box is not None:
            prev_x1, prev_y1, prev_x2, prev_y2 = previous_box
            if (x1 < prev_x2 and x2 > prev_x1) and (y1 < prev_y2 and y2 > prev_y1):
                y1 = prev_y2
                y1 = min(y1, dims.padded_h - 1)
                if y2 <= y1: y2 = min(y1 + 1, dims.padded_h)
        new_previous_box = [x1, y1, x2, y2]
        orig_x1, orig_y1, orig_x2, orig_y2 = map_to_original_coordinates(x1, y1, x2, y2, dims)
        return x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, new_previous_box
    except Exception as e:
        print(f"process_coordinates error: {str(e)}")
        orig_x1, orig_y1, orig_x2, orig_y2 = 0, 0, min(100, dims.original_w), min(100, dims.original_h)
        return 0, 0, 100, 100, orig_x1, orig_y1, orig_x2, orig_y2, [0, 0, 100, 100]

def prepare_image(image) -> Tuple[np.ndarray, ImageDimensions]:
    try:
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        original_h, original_w = image_cv.shape[:2]
        max_size = max(original_h, original_w)
        top = (max_size - original_h) // 2
        bottom = max_size - original_h - top
        left = (max_size - original_w) // 2
        right = max_size - original_w - left
        padded_image = cv2.copyMakeBorder(image_cv, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padded_h, padded_w = padded_image.shape[:2]
        dimensions = ImageDimensions(original_w=original_w, original_h=original_h, padded_w=padded_w, padded_h=padded_h)
        return padded_image, dimensions
    except Exception as e:
        print(f"prepare_image error: {str(e)}")
        h, w = image.height, image.width
        dimensions = ImageDimensions(original_w=w, original_h=h, padded_w=w, padded_h=h)
        return np.zeros((h, w, 3), dtype=np.uint8), dimensions

def parse_layout_string(bbox_str):
    """Parse layout string using regular expressions"""
    pattern = r"\[(\d*\.?\d+),\s*(\d*\.?\d+),\s*(\d*\.?\d+),\s*(\d*\.?\d+)\]\s*(\w+)"
    matches = re.finditer(pattern, bbox_str)

    parsed_results = []
    for match in matches:
        coords = [float(match.group(i)) for i in range(1, 5)]
        label = match.group(5).strip()
        parsed_results.append((coords, label))

    return parsed_results

# --- 主要执行逻辑 ---

# 1. 设置模型和参数
model_id = "ByteDance/Dolphin"
encoder_prompt = "".join(["0"] * 783)
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=2048,
    logprobs=0,
    prompt_logprobs=None,
    skip_special_tokens=False,
)

print("正在初始化 vLLM 引擎...")
processor = DonutProcessor.from_pretrained(model_id)
llm = LLM(
    model=model_id,
    dtype="float32",
    enforce_eager=True,
    max_num_seqs=16, # 增加并发数以处理更多页面元素
    hf_overrides={"architectures": ["DonutForConditionalGeneration"]},
)
print("vLLM 引擎初始化完成。")


# 2. 解析命令行参数以获取图片路径
parser = argparse.ArgumentParser(description="使用 vLLM 和 Dolphin 从文档图像中提取文本。")
parser.add_argument("--image_path", type=str, default=None, help="本地图片文件路径。")
args = parser.parse_args()

# 3. 加载图像
if args.image_path:
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"错误: 找不到文件 {args.image_path}")
    print(f"从本地路径加载图片: {args.image_path}")
    image = Image.open(args.image_path).convert("RGB")
else:
    print("加载 Hugging Face datasets 的默认图片。")
    dataset = load_dataset("hf-internal-testing/example-documents", split="test")
    image = dataset[0]["image"] # 默认使用数据集的第一张图片

# 4. 阶段一：布局分析
print("\n--- 阶段一：分析文档布局 ---")
prompt = "Parse the reading order of this document."
decoder_prompt = f"<s>{prompt}<Answer/>"
decoder_prompt_tokens = TokensPrompt(prompt_token_ids=processor.tokenizer(
    decoder_prompt, add_special_tokens=False)["input_ids"])
enc_dec_prompt = ExplicitEncoderDecoderPrompt(
    encoder_prompt=TextPrompt(prompt=encoder_prompt, multi_modal_data={"image": image}),
    decoder_prompt=decoder_prompt_tokens,
)

layout_outputs = llm.generate(prompts=enc_dec_prompt, sampling_params=sampling_params)
layout_result_str = layout_outputs[0].outputs[0].text
print(f"布局分析原始输出:\n{layout_result_str}")

# 5. 准备阶段二的输入 (裁剪图片和生成提示)
padded_image, dims = prepare_image(image)
layout_results = parse_layout_string(layout_result_str)

text_table_elements = []
previous_box = None
reading_order = 0

for bbox_coords, label in layout_results:
    if label == "fig":  # 忽略图片元素
        continue
    try:
        x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, previous_box = process_coordinates(
            bbox_coords, padded_image, dims, previous_box
        )
        cropped = padded_image[y1:y2, x1:x2]
        if cropped.size > 0 and cropped.shape[0] > 3 and cropped.shape[1] > 3:
            pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            prompt = "Parse the table in the image." if label == "tab" else "Read text in the image."
            text_table_elements.append({
                "crop": pil_crop,
                "prompt": prompt,
                "reading_order": reading_order,
            })
        reading_order += 1
    except Exception as e:
        print(f"处理 bbox (标签: {label}) 时出错: {str(e)}")
        continue

# 6. 阶段二：批量 OCR 推理
if text_table_elements:
    print(f"\n--- 阶段二：对 {len(text_table_elements)} 个元素进行批量 OCR ---")
    batch_prompts = []
    for elem in text_table_elements:
        decoder_prompt_str = f"<s>{elem['prompt']}<Answer/>"
        decoder_prompt_tokens = TokensPrompt(
            prompt_token_ids=processor.tokenizer(decoder_prompt_str, add_special_tokens=False)["input_ids"]
        )
        enc_dec_prompt = ExplicitEncoderDecoderPrompt(
            encoder_prompt=TextPrompt(prompt=encoder_prompt, multi_modal_data={"image": elem["crop"]}),
            decoder_prompt=decoder_prompt_tokens,
        )
        batch_prompts.append(enc_dec_prompt)

    # 对所有裁剪的图片进行一次性批量推理
    batch_outputs = llm.generate(prompts=batch_prompts, sampling_params=sampling_params)

    # 将识别结果添加回元素列表
    for i, output in enumerate(batch_outputs):
        text_table_elements[i]["text"] = output.outputs[0].text.strip()

# 7. 按阅读顺序输出最终结果
print("\n" + "------" * 8)
print(" OCR 提取的最终文本 (按阅读顺序) ".center(48, " "))
print("------" * 8 + "\n")

# 按 reading_order 排序
text_table_elements.sort(key=lambda x: x["reading_order"])

# 打印排序后的文本
for elem in text_table_elements:
    print(elem.get("text", ""))