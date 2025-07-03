from vllm import LLM, SamplingParams
from vllm.assets.video import VideoAsset
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode

if __name__ == "__main__":
    llm = LLM(
        # model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        model="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        tensor_parallel_size=2,
        trust_remote_code=True,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral"
    )
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

# import requests
# import json

# url = "http://localhost:8000/v1/chat/completions"
# headers = {'Content-Type': 'application/json'}
# data = {
#     "model": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
#     "messages": [
#         {"role": "user", "content": [
#         {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/500px-Cat_November_2010-1a.jpg"}},
#         {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/020_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg/500px-020_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg"}},
#         {"type": "text", "text": "What are the animals in these images ?"}
#     ]}
#     ],
# }

# try:
#     response = requests.post(url, headers=headers, data=json.dumps(data))
#     response.raise_for_status()  # 如果响应状态码不是 200 OK，会抛出 HTTPError 异常
#     print("请求成功！")
#     print("响应内容:")
#     print(response.json())
# except requests.exceptions.RequestException as e:
#     print(f"请求失败: {e}")
#     if response is not None:
#         print(f"响应状态码: {response.status_code}")
#         print(f"响应内容: {response.text}")