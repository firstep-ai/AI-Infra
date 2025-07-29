from vllm import LLM

llm = LLM(model="./BAGEL-7B-MoT", 
    hf_overrides={"architectures": ["BagelForConditionalGeneration"]},
    trust_remote_code=True)