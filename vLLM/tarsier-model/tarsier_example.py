from pathlib import Path
from vllm import LLM, SamplingParams
from PIL import Image
from vllm.multimodal.video import VideoMediaIO, ImageMediaIO

def extract_frames_from_video(video_filepath: str, num_frames: int):
    image_io = ImageMediaIO()
    video_io = VideoMediaIO(image_io=image_io, num_frames=num_frames)
    frames = video_io.load_file(Path(video_filepath))
    return frames

if __name__ == "__main__":
    EXAMPLE_IMAGE_PATH = "kitty.jpg"
    EXAMPLE_VIDEO_PATH = "kitchen.mp4"
    MAX_VIDEO_FRAMES = 4
    llm = LLM(model="omni-research/Tarsier-7b", trust_remote_code=True)
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
        "prompt": "USER: <image>\nPlease describe the image. ASSISTANT:",
        "multi_modal_data": {"image": [Image.open(EXAMPLE_IMAGE_PATH).convert('RGB')]}
    }
    outputs = llm.generate(vllm_inputs_single_image, sampling_params) # 直接生成
    for output_item in outputs:
        print(f"生成: {output_item.outputs[0].text}\n" + "-" * 20)

    # 场景 3: 文本和视频（多帧图像）测试
    vllm_inputs_video = {
        "prompt": f"USER: {'<image>'*MAX_VIDEO_FRAMES}\nPlease describe the video. ASSISTANT:",
        "multi_modal_data": {"image": extract_frames_from_video(EXAMPLE_VIDEO_PATH, MAX_VIDEO_FRAMES)}
    }
    outputs = llm.generate(vllm_inputs_video, sampling_params) # 直接生成
    for output_item in outputs:
        print(f"生成: {output_item.outputs[0].text}\n" + "-" * 20)

    print("\n所有测试完成。")