## 使用 Ray 进行多节点部署（推荐）

vLLM 的多节点部署应该使用 Ray，而不是手动的 Data Parallel：

```bash
# Master 节点
ray start --head --port=6379

# Worker 节点
ray start --address=<MASTER_IP>:6379

# 在Master 节点查看ray状态
ray status

```
VLLM_TORCH_PROFILER_DIR="dsrnn_profile" CUDA_LAUNCH_BLOCKING=1 VLLM_ALL2ALL_BACKEND=deepep_low_latency vllm serve \
    /proj-tango-pvc/users/frd_eng/models/meti/distillation/large_part0_run0/bck/checkpoint-1550/safetensors \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --port 8000 \
    --enforce-eager
```

```
vllm bench serve --backend openai-chat --model /proj-tango-pvc/users/frd_eng/models/meti/distillation/large_part0_run0/bck/checkpoint-1550/safetensors --endpoint /v1/chat/completions --dataset-name hf --dataset-path mgoin/mlperf-inference-llama2-data --hf-split train --num-prompts 2 --profile
```

```
============ Serving Benchmark Result ============
Successful requests:                     2         
Benchmark duration (s):                  64.74     
Total input tokens:                      169       
Total generated tokens:                  377       
Request throughput (req/s):              0.03      
Output token throughput (tok/s):         5.82      
Peak output token throughput (tok/s):    8.00      
Peak concurrent requests:                2.00      
Total Token throughput (tok/s):          8.43      
---------------Time to First Token----------------
Mean TTFT (ms):                          506.68    
Median TTFT (ms):                        506.68    
P99 TTFT (ms):                           649.32    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          262.09    
Median TPOT (ms):                        262.09    
P99 TPOT (ms):                           267.59    
---------------Inter-token Latency----------------
Mean ITL (ms):                           258.80    
Median ITL (ms):                         253.48    
P99 ITL (ms):                            441.73    
==================================================
```


```
VLLM_TORCH_PROFILER_DIR="ds_profile" CUDA_LAUNCH_BLOCKING=1 VLLM_ALL2ALL_BACKEND=deepep_low_latency vllm serve \
    /proj-tango-pvc/models/DeepSeek-V3-0324-BF16 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --port 8000 \
    --enforce-eager
```

```
vllm bench serve --backend openai-chat --model /proj-tango-pvc/models/DeepSeek-V3-0324-BF16 --endpoint /v1/chat/completions --dataset-name hf --dataset-path mgoin/mlperf-inference-llama2-data --hf-split train --num-prompts 2 --profile
```

```
============ Serving Benchmark Result ============
Successful requests:                     5         
Benchmark duration (s):                  7.55      
Total input tokens:                      521       
Total generated tokens:                  924       
Request throughput (req/s):              0.66      
Output token throughput (tok/s):         122.42    
Peak output token throughput (tok/s):    160.00    
Peak concurrent requests:                5.00      
Total Token throughput (tok/s):          191.45    
---------------Time to First Token----------------
Mean TTFT (ms):                          499.68    
Median TTFT (ms):                        529.28    
P99 TTFT (ms):                           530.14    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          28.92     
Median TPOT (ms):                        28.55     
P99 TPOT (ms):                           30.84     
---------------Inter-token Latency----------------
Mean ITL (ms):                           28.39     
Median ITL (ms):                         27.43     
P99 ITL (ms):                            33.27     
==================================================
```