from vllm import LLM

llm = LLM(model="./BAGEL-7B-MoT", trust_remote_code=True)