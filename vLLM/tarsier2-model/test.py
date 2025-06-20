from vllm import LLM, SamplingParams
from vllm.assets.video import VideoAsset
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode

if __name__ == "__main__":
    EXAMPLE_IMAGE_PATH = "kitty.jpg"
    llm = LLM(
        model="omni-research/Tarsier2-Recap-7b",
        hf_overrides={"architectures": ["Tarsier2ForConditionalGeneration"]},
        trust_remote_code=True
    )
    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=500)

    # 场景 1: 纯文本测试
    print(f"\n--- 纯文本测试 ---")
    vllm_inputs_text_only = {"prompt": "USER: Please introduce yourself. ASSISTANT:"}
    outputs = llm.generate(vllm_inputs_text_only, sampling_params)
    for output_item in outputs:
        print(f"生成: {output_item.outputs[0].text}\n" + "-" * 20)

    # 场景 2: 文本和单张图像测试
    print(f"\n--- 文本和单张图像测试 ---")
    vllm_inputs_single_image = {
        "prompt": "USER: <|image_pad|>\nPlease describe the image. ASSISTANT:",
        "multi_modal_data": {"image": convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")}
    }
    outputs = llm.generate(vllm_inputs_single_image, sampling_params) # 直接生成
    for output_item in outputs:
        print(f"生成: {output_item.outputs[0].text}\n" + "-" * 20)

    # 场景 3: 文本和视频测试
    print(f"\n--- 文本和视频测试 ---")
    vllm_inputs_single_image = {
        "prompt": "USER: <|video_pad|>\nPlease describe the image. ASSISTANT:",
        "multi_modal_data": {"video": VideoAsset(name="baby_reading", num_frames=16).np_ndarrays}
    }
    outputs = llm.generate(vllm_inputs_single_image, sampling_params) # 直接生成
    for output_item in outputs:
        print(f"生成: {output_item.outputs[0].text}\n" + "-" * 20)

# python examples/offline_inference/vision_language.py -m tarsier2 --modality image
# python examples/offline_inference/vision_language.py -m tarsier2 --modality video
# python examples/offline_inference/vision_language_multi_image.py -m tarsier2