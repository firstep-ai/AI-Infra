# 导入必要的类
from vllm import LLM, SamplingParams

llm = LLM(model="CognitiveComputations/Qwen3-30B-A3B-AWQ", 
          quantization='awq_marlin',
          tensor_parallel_size=4,
          gpu_memory_utilization=0.9,
          max_model_len=2048)

simple_prompt = ["你好，世界！"]  # 使用列表包含单个字符串
minimal_sampling_params = SamplingParams(max_tokens=10)  # 仅指定最大生成 token 数

outputs = llm.generate(simple_prompt, minimal_sampling_params)

for output in outputs:
    generated_text = output.outputs[0].text
    print(f"生成的文本: {generated_text}")