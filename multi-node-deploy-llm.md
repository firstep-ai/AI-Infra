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
CUDA_LAUNCH_BLOCKING=1 VLLM_ALL2ALL_BACKEND=deepep_low_latency vllm serve \
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

vllm bench serve --model /proj-tango-pvc/users/frd_eng/models/meti/distillation/large_part0_run0/bck/checkpoint-1550/safetensors_test_johanes --num-prompts 1000 --random-input-len 1238 --random-output-len 231 --ignore-eos
```

VLLM_ALL2ALL_BACKEND=deepep_low_latency vllm serve \
    /proj-tango-pvc/users/akramusman01/models/dsrnn/x7_run4/checkpoint-32000/safetensors \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --port 8000 

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/proj-tango-pvc/users/akramusman01/models/dsrnn/x7_run4/checkpoint-32000/safetensors",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_completion_tokens": 30
  }'

vllm bench serve --model /proj-tango-pvc/users/akramusman01/models/dsrnn/x7_run4/checkpoint-32000/safetensors --num-prompts 1000 --random-input-len 653 --random-output-len 113 --ignore-eos


9B Benchmark

============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  29.06     
Total input tokens:                      652000    
Total generated tokens:                  113000    
Request throughput (req/s):              34.41     
Output token throughput (tok/s):         3888.48   
Peak output token throughput (tok/s):    7992.00   
Peak concurrent requests:                1000.00   
Total Token throughput (tok/s):          26324.69  
---------------Time to First Token----------------
Mean TTFT (ms):                          8954.25   
Median TTFT (ms):                        8811.07   
P99 TTFT (ms):                           15809.88  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          140.17    
Median TPOT (ms):                        142.19    
P99 TPOT (ms):                           152.97    
---------------Inter-token Latency----------------
Mean ITL (ms):                           140.21    
Median ITL (ms):                         129.70    
P99 ITL (ms):                            185.05    
==================================================