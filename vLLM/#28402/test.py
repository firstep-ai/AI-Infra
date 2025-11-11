# reproduce_llava_onevision_bug.py
# 复现测试失败: test_custom_inputs_models[llava_onevision-multiple-images-test_case5]
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import rescale_image_size
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

if __name__ == '__main__':
    # 准备测试图像 (与测试中使用的相同)
    stop_sign = ImageAsset("stop_sign").pil_image
    cherry_blossom = ImageAsset("cherry_blossom").pil_image

    # test_case5 对应的是第一个测试用例：2张图片 [stop_sign, cherry_blossom]
    # Prompt: "<image><image>\nDescribe 2 images."
    prompt = "<|im_start|>user\n<image><image>\nDescribe 2 images.<|im_end|>\n<|im_start|>assistant\n"
    images = [stop_sign, cherry_blossom]  # 注意：这是一个列表，包含2张图片

    # ========== vLLM 推理 ==========
    print("Running vLLM inference...")
    llm = LLM(
        model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        max_model_len=16384,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": 4},
        gpu_memory_utilization=0.1,  # 添加这一行：只使用 10% 的 GPU 显存
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        logprobs=5,  # 获取 top-5 logprobs
    )

    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"image": images},  # 传入图片列表
        },
        sampling_params=sampling_params,
    )

    vllm_text = outputs[0].outputs[0].text
    vllm_logprobs = outputs[0].outputs[0].logprobs

    # ========== HuggingFace 推理 ==========
    print("\nRunning HuggingFace inference...")
    model = AutoModelForImageTextToText.from_pretrained(
        "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    )

    inputs = processor(text=prompt, images=images, return_tensors="pt").to(model.device)  # 传入图片列表
    prompt_length = inputs.input_ids.shape[1]  # 记录 prompt 的长度

    with torch.no_grad():
        hf_outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # 完整输出（包含 prompt）
    hf_full_text = processor.decode(hf_outputs.sequences[0], skip_special_tokens=True)
    
    # 只取生成的部分（跳过 prompt）
    hf_generated_ids = hf_outputs.sequences[0][prompt_length:]
    hf_text = processor.decode(hf_generated_ids, skip_special_tokens=True)

    # 比较第一个生成的 token
    hf_first_token_logprobs = {}
    if hf_outputs.scores:
        first_token_scores = hf_outputs.scores[0][0]
        top5_indices = torch.topk(first_token_scores, 5).indices
        for rank, idx in enumerate(top5_indices, 1):
            token = processor.tokenizer.decode([idx])
            logprob = first_token_scores[idx].item()
            hf_first_token_logprobs[idx.item()] = {
                'token': token,
                'logprob': logprob,
                'rank': rank
            }

    # ========== 打印完整输出 ==========
    print("\n" + "="*80)
    print("COMPLETE OUTPUT COMPARISON")
    print("="*80)
    
    print("\n【vLLM 输出】")
    print("-" * 80)
    print(f"生成的文本: {vllm_text}")
    print(f"\n第一个 token 的 logprobs (top-5):")
    if vllm_logprobs:
        for token_id, logprob_obj in list(vllm_logprobs[0].items())[:5]:
            print(f"  Rank {logprob_obj.rank}: '{logprob_obj.decoded_token}' "
                  f"(token_id={token_id}, logprob={logprob_obj.logprob:.4f})")
    
    print("\n【HuggingFace 输出】")
    print("-" * 80)
    print(f"完整输出（含 prompt）: {hf_full_text}")
    print(f"\n生成的文本（跳过 prompt）: {hf_text}")
    print(f"\n第一个 token 的 logprobs (top-5):")
    for token_id, info in list(hf_first_token_logprobs.items())[:5]:
        print(f"  Rank {info['rank']}: '{info['token']}' "
              f"(token_id={token_id}, logprob={info['logprob']:.4f})")
    
    print("\n" + "="*80)
    print("COMPARISON RESULT")
    print("="*80)
    print(f"生成的文本是否完全匹配: {vllm_text.strip() == hf_text.strip()}")
    print(f"生成的文本是否匹配（忽略大小写）: {vllm_text.strip().lower() == hf_text.strip().lower()}")
    print(f"\nvLLM 生成: '{vllm_text.strip()}'")
    print(f"HF 生成:   '{hf_text.strip()}'")
    print("="*80)