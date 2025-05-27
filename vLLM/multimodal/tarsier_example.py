import os
from PIL import Image
from huggingface_hub import snapshot_download

from vllm import LLM, SamplingParams
# ImagePixelData 可以用来更精细地控制图像输入，但对于简单场景，直接传递 PIL Image 对象通常也有效
# from vllm.multimodal.utils import ImagePixelData

# --- 步骤 1: 配置和准备 (离线模式和模型路径) ---
MODEL_HF_NAME = "omni-research/Tarsier-7b"
LOCAL_MODEL_DIR = "./tarsier-7b"  # 你希望将模型下载到的本地目录

# 设置环境变量以强制离线模式 (防止不必要的网络请求)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["VLLM_CPU_ONLY"] = "1" # 如果你没有GPU或者想在CPU上测试，可以加上这行，但会非常慢

# 确保本地模型目录存在，如果不存在则尝试下载
if not os.path.exists(LOCAL_MODEL_DIR):
    print(f"本地模型目录 {LOCAL_MODEL_DIR} 不存在。")
    print(f"请先手动下载模型 {MODEL_HF_NAME} 到该目录，或取消注释下面的代码来尝试下载。")
    # print(f"正在尝试从 Hugging Face Hub 下载模型 {MODEL_HF_NAME} 到 {LOCAL_MODEL_DIR}...")
    # try:
    #     snapshot_download(MODEL_HF_NAME, local_dir=LOCAL_MODEL_DIR, local_dir_use_symlinks=False)
    #     print("模型下载完成。")
    # except Exception as e:
    #     print(f"模型下载失败: {e}")
    #     print("请确保你有网络连接，或者手动下载模型。")
    #     exit()
elif not os.listdir(LOCAL_MODEL_DIR): # 目录存在但是空的
    print(f"本地模型目录 {LOCAL_MODEL_DIR} 是空的。")
    print(f"请先手动下载模型 {MODEL_HF_NAME} 到该目录。")
    exit()


# --- 步骤 2: 准备一个示例图像 ---
# 创建一个虚拟图像用于测试，或替换为你的本地图像路径
EXAMPLE_IMAGE_PATH = "kitty.jpg"
if not os.path.exists(EXAMPLE_IMAGE_PATH):
    try:
        print(f"创建虚拟图像: {EXAMPLE_IMAGE_PATH}")
        temp_image = Image.new('RGB', (336, 336), color='blue') # Tarsier 可能对输入尺寸有要求
        temp_image.save(EXAMPLE_IMAGE_PATH)
    except Exception as e:
        print(f"创建虚拟图像失败: {e}")
        exit()

# --- 步骤 3: 初始化 vLLM引擎 ---
# 对于多模态模型，vLLM 通常能自动处理。
# trust_remote_code=True 允许执行模型仓库中的自定义代码（如果存在）。
# Tarsier-7B 的 config.json 指向 "LlavaLlamaForCausalLM"，vLLM应能识别。
print(f"正在从本地路径加载模型: {LOCAL_MODEL_DIR}")
try:
    llm = LLM(
        model=LOCAL_MODEL_DIR,
        trust_remote_code=True,  # 对于非官方支持或有自定义代码的模型通常需要
        # tensor_parallel_size=1, # 根据你的GPU数量调整
        # max_model_len=2048,     # 根据模型和你的需求调整
        # enforce_eager=True, # 有些模型可能需要eager模式
        # limit_mm_per_prompt={"image": 1} # 明确限制每条prompt的图片数量
    )
    print("模型加载成功!")
except Exception as e:
    print(f"模型加载失败: {e}")
    print("可能的原因:")
    print("  1. 模型文件不完整或损坏。")
    print("  2. 如果模型依赖特定的自定义代码 (类似你之前提供的 TarsierForConditionalGeneration.py)，")
    print("     确保该代码对 vLLM 可见 (例如在 PYTHONPATH 中)，并且模型的 config.json 指向正确的自定义类。")
    print("     然而，omni-research/Tarsier-7b 在HF上似乎使用标准的 LlavaLlamaForCausalLM 架构。")
    print("  3. 硬件资源不足 (如显存)。")
    exit()

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)

# --- 步骤 4: 纯文本测试 ---
# print("\n--- 纯文本测试 ---")
# text_prompt = "Human: 你好，请介绍一下你自己。 Assistant:"
# try:
#     outputs = llm.generate(text_prompt, sampling_params)
#     for output in outputs:
#         prompt_out = output.prompt
#         generated_text = output.outputs[0].text
#         print(f"提示: {prompt_out}")
#         print(f"生成: {generated_text}")
#         print("-" * 20)
# except Exception as e:
#     print(f"纯文本生成失败: {e}")

# --- 步骤 5: 文本和图像测试 ---
print("\n--- 文本和图像测试 ---")
# 根据 Tarsier 的模型卡, 提示格式通常包含特殊标记如 <image>
# "The format of the model input is `Human: <image>\n{image_input} {text_input} Assistant: {response}`.
#  `<image>` is a placeholder for the image token."
#  tokenizer_config.json 中 "<image>" 标记为 id 32000
image_question = "这张图片里有什么内容？"
image_prompt = f"Human: <image>\n{image_question} Assistant:"

try:
    # 加载图像
    pil_image = Image.open(EXAMPLE_IMAGE_PATH).convert("RGB")

    # 使用 multi_modal_data 参数传递图像
    # vLLM 会使用模型的处理器来处理 PIL Image 对象
    inputs={
        "prompt": image_prompt,
        "multi_modal_data": {"image": pil_image}
    }
    outputs = llm.generate(
        inputs,
        sampling_params=sampling_params
    )

    for output in outputs:
        prompt_out = output.prompt
        generated_text = output.outputs[0].text
        print(f"提示 (图像部分由 <image> 占位): {prompt_out}")
        print(f"生成: {generated_text}")
        print("-" * 20)

except Exception as e:
    print(f"文本+图像生成失败: {e}")
    print("可能的原因:")
    print("  1. 图像占位符 `<image>` 或提示格式不正确。")
    print("  2. 图像预处理问题或图像数据格式不符合模型期望。")
    print("  3. 模型的视觉部分或多模态融合层存在问题。")


# --- 步骤 6: 关于视频的说明 ---
print("\n--- 文本和视频测试 ---")
print("注意：`omni-research/Tarsier-7b` 模型 (以及 LLaVA 类模型) 主要设计用于处理静态图像。")
print("直接支持视频输入需要模型架构本身支持视频特征提取和处理（例如视频帧的序列化处理）。")
print("标准的 Tarsier-7b 模型预计不支持直接的视频输入。")
print("因此，此处不提供文本+视频的直接运行示例。")


print("\n测试完成。")