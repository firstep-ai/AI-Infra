```
VLLM_TORCH_PROFILER_DIR="ds32_profile" vllm serve /tmp/scratch-space/DeepSeek-V3.2 --tensor-parallel-size 8
```

```
vllm bench serve --model tmp/scratch-space/DeepSeek-V3.2 --num-prompts 1 --random-input-len 120000 --random-output-len 10000 --ignore-eos --profile
```