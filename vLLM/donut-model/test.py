# transformers/src/transformers/models/donut/convert_donut_to_pytorch.py

from vllm import LLM, SamplingParams
from PIL import Image
import os

os.environ["VLLM_USE_V1"] = "0"
if __name__ == "__main__":
    EXAMPLE_IMAGE_PATH = "para_1.jpg"
    llm = LLM(
        model="ByteDance/Dolphin",
        hf_overrides={"architectures": ["DonutForConditionalGeneration"]},
    )
    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=500)

    vllm_inputs_single_image = {
        "prompt": "<s>Read text in the image. <Answer/>",
        "multi_modal_data": {"image": [Image.open(EXAMPLE_IMAGE_PATH).convert('RGB')]}
    }
    outputs = llm.generate(vllm_inputs_single_image, sampling_params) # 直接生成
    for output_item in outputs:
        print(f"生成: {output_item.outputs[0].text}\n" + "-" * 20)

    print("\n所有测试完成。")