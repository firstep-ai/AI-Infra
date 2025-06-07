from vllm import LLM, SamplingParams
from vllm.multimodal.image import convert_image_mode
from PIL import Image

def main():
    image_path = "kitty.jpg"
    question = "Please describe the image."
    
    # 您可以使用 "llava-hf/llava-1.5-7b-hf" 或 "llava-hf/llava-v1.6-mistral-7b-hf"
    # 注意：LLaVA v1.6 (Mistral) 可能需要不同的提示格式: f"[INST] <image>\n{question} [/INST]"
    llava_model_name = "llava-hf/llava-1.5-7b-hf"
    prompt_text = f"USER: <image>\n{question}\nASSISTANT:"

    try:
        pil_image = Image.open(image_path)
        pil_image_rgb = convert_image_mode(pil_image, "RGB") # LLaVA 通常需要 RGB 格式
    except FileNotFoundError:
        print(f"错误：找不到图像文件 '{image_path}'。请确保它在正确的路径。")
        return
    except Exception as e:
        print(f"加载图像时出错: {e}")
        return

    # 初始化 vLLM 引擎
    # trust_remote_code=True 对于从 HuggingFace Hub 加载某些模型是必需的
    # limit_mm_per_prompt 指定了每个提示可以处理的图像数量
    llm = LLM(
        model=llava_model_name,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1}, 
        max_model_len=2048  # LLaVA 1.5 通常的上下文长度，可根据模型调整
    )

    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.2, #较低的温度使输出更具确定性
        max_tokens=150   # 为图像描述生成足够的 token
    )

    # 准备 vLLM 的输入
    vllm_input = {
        "prompt": prompt_text,
        "multi_modal_data": {"image": pil_image_rgb}
    }

    try:
        # 生成输出
        outputs = llm.generate(vllm_input, sampling_params)
    except Exception as e:
        print(f"vLLM 生成过程中出错: {e}")
        return

    # 打印结果
    if outputs and outputs[0].outputs:
        generated_text = outputs[0].outputs[0].text
        print(f"\n对 '{image_path}' 的描述:")
        print(generated_text)
    else:
        print("未能生成描述。")

if __name__ == "__main__":
    main()