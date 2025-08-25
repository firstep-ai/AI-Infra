from vllm import LLM, SamplingParams
from vllm.assets.video import VideoAsset
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode

if __name__ == "__main__":
    # model_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    model_name = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    llm = LLM(model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        tensor_parallel_size=2,
        limit_mm_per_prompt={"image": 1},
        ignore_patterns=["consolidated.safetensors"],)
    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=500)

    # 场景 1: 纯文本测试
    print(f"\n--- 纯文本测试 ---")
    vllm_inputs_text_only = {"prompt": "Please introduce yourself."}
    outputs = llm.generate(vllm_inputs_text_only, sampling_params)
    for output_item in outputs:
        print(f"生成: {output_item.outputs[0].text}\n" + "-" * 20)

    # 场景 2: 文本和单张图像测试
    print(f"\n--- 文本和单张图像测试 ---")
    vllm_inputs_single_image = {
        "prompt": "<s>[INST][IMG][IMG_END]Please describe the image.\n[/INST]",
        "multi_modal_data": {"image": convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")}
    }
    outputs = llm.generate(vllm_inputs_single_image, sampling_params) # 直接生成
    for output_item in outputs:
        print(f"生成: {output_item.outputs[0].text}\n" + "-" * 20)