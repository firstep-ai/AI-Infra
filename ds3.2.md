```
VLLM_TORCH_PROFILER_DIR="ds32_profile" VLLM_USE_DEEP_GEMM=0 vllm serve /tmp/scratch-space/DeepSeek-V3.2 --tensor-parallel-size 8 --max-model-len 51200
```

```
vllm bench serve --model /tmp/scratch-space/DeepSeek-V3.2 --num-prompts 1 --random-input-len 50176 --random-output-len 1024 --ignore-eos --profile
```

## Download Profile

Use `kubectl cp` to download the profile directory to your local machine:

```bash
# Syntax: kubectl cp <namespace>/<pod-name>:<remote-path> <local-path>
# Assuming the command was run in the root directory inside the pod:
kubectl cp llm-serving/deepseek-v3-2-vllm-77f648c586-whnq6:ds32_profile ./ds32_profile
```