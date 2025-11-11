## 使用 Ray 进行多节点部署（推荐）

vLLM 的多节点部署应该使用 Ray，而不是手动的 Data Parallel：

```bash
# Master 节点
ray start --head --port=6379

# Worker 节点
ray start --address=<MASTER_IP>:6379

# 在Master 节点查看ray状态
ray status

# 然后在 Master 节点启动 vLLM
VLLM_ALL2ALL_BACKEND=deepep_low_latency vllm serve \
    /proj-tango-pvc/users/frd_eng/models/meti/distillation/large_part0_run0/bck/checkpoint-1550/safetensors_test_johanes \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --port 8000 

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/proj-tango-pvc/users/frd_eng/models/meti/distillation/large_part0_run0/bck/checkpoint-1550/safetensors_test_johanes",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_completion_tokens": 30
  }'
```