import os
os.environ["VLLM_USE_V1"] = "0"
os.environ["HF_TOKEN"] = "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"
from vllm import LLM, SamplingParams
chats = [[{"role": "user", "content": "What is the capital of France?"}], [{"role": "user", "content": "What is the capital of China?"}]]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
prompts = [tokenizer.apply_chat_template(chat, tokenize=False) for chat in chats]
llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")