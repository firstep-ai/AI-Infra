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
VLLM_TORCH_PROFILER_DIR="dsrnn_profile" VLLM_ALL2ALL_BACKEND=deepep_low_latency vllm serve \
    /proj-tango-pvc/users/frd_eng/models/meti/distillation/large_part0_run0/bck/checkpoint-1550/safetensors \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --port 8000
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

## 20次

```
============ Serving Benchmark Result ============
Successful requests:                     20        
Benchmark duration (s):                  15.51     
Total input tokens:                      1751      
Total generated tokens:                  3165      
Request throughput (req/s):              1.29      
Output token throughput (tok/s):         204.05    
Peak output token throughput (tok/s):    460.00    
Peak concurrent requests:                20.00     
Total Token throughput (tok/s):          316.95    
---------------Time to First Token----------------
Mean TTFT (ms):                          739.43    
Median TTFT (ms):                        812.14    
P99 TTFT (ms):                           815.97    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          42.59     
Median TPOT (ms):                        42.89     
P99 TPOT (ms):                           52.40     
---------------Inter-token Latency----------------
Mean ITL (ms):                           40.06     
Median ITL (ms):                         42.16     
P99 ITL (ms):                            51.27     
==================================================
```

## 40次



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
Successful requests:                     2         
Benchmark duration (s):                  51.98     
Total input tokens:                      169       
Total generated tokens:                  377       
Request throughput (req/s):              0.04      
Output token throughput (tok/s):         7.25      
Peak output token throughput (tok/s):    12.00     
Peak concurrent requests:                2.00      
Total Token throughput (tok/s):          10.50     
---------------Time to First Token----------------
Mean TTFT (ms):                          2648.08   
Median TTFT (ms):                        2648.08   
P99 TTFT (ms):                           3554.73   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          198.63    
Median TPOT (ms):                        198.63    
P99 TPOT (ms):                           200.21    
---------------Inter-token Latency----------------
Mean ITL (ms):                           198.12    
Median ITL (ms):                         195.17    
P99 ITL (ms):                            204.25    
==================================================
```

```
VLLM_TORCH_PROFILER_DIR="dsrnn_profile_graph" CUDA_LAUNCH_BLOCKING=1 VLLM_ALL2ALL_BACKEND=deepep_low_latency vllm serve \
    /proj-tango-pvc/users/frd_eng/models/meti/distillation/large_part0_run0/bck/checkpoint-1550/safetensors \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --port 8000
```

```
vllm bench serve --backend openai-chat --model /proj-tango-pvc/users/frd_eng/models/meti/distillation/large_part0_run0/bck/checkpoint-1550/safetensors --endpoint /v1/chat/completions --dataset-name hf --dataset-path mgoin/mlperf-inference-llama2-data --hf-split train --num-prompts 2 --profile
```

```
============ Serving Benchmark Result ============
Successful requests:                     2         
Benchmark duration (s):                  11.12     
Total input tokens:                      169       
Total generated tokens:                  377       
Request throughput (req/s):              0.18      
Output token throughput (tok/s):         33.89     
Peak output token throughput (tok/s):    48.00     
Peak concurrent requests:                2.00      
Total Token throughput (tok/s):          49.08     
---------------Time to First Token----------------
Mean TTFT (ms):                          581.20    
Median TTFT (ms):                        581.20    
P99 TTFT (ms):                           762.40    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          42.57     
Median TPOT (ms):                        42.57     
P99 TPOT (ms):                           42.74     
---------------Inter-token Latency----------------
Mean ITL (ms):                           42.40     
Median ITL (ms):                         41.39     
P99 ITL (ms):                            48.59     
==================================================
```

```
VLLM_TORCH_PROFILER_DIR="ds_profile_graph" VLLM_ALL2ALL_BACKEND=deepep_low_latency vllm serve \
    /proj-tango-pvc/models/DeepSeek-V3-0324-BF16 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --port 8000
```

```
vllm bench serve --backend openai-chat --model /proj-tango-pvc/models/DeepSeek-V3-0324-BF16 --endpoint /v1/chat/completions --dataset-name hf --dataset-path mgoin/mlperf-inference-llama2-data --hf-split train --num-prompts 2 --profile
```
## With Torch Compile
```
============ Serving Benchmark Result ============
Successful requests:                     2         
Benchmark duration (s):                  5.92      
Total input tokens:                      169       
Total generated tokens:                  377       
Request throughput (req/s):              0.34      
Output token throughput (tok/s):         63.70     
Peak output token throughput (tok/s):    84.00     
Peak concurrent requests:                2.00      
Total Token throughput (tok/s):          92.26     
---------------Time to First Token----------------
Mean TTFT (ms):                          222.86    
Median TTFT (ms):                        222.86    
P99 TTFT (ms):                           222.94    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          23.39     
Median TPOT (ms):                        23.39     
P99 TPOT (ms):                           24.07     
---------------Inter-token Latency----------------
Mean ITL (ms):                           23.03     
Median ITL (ms):                         23.77     
P99 ITL (ms):                            25.17     
==================================================
```
## Without Torch Compile

```
============ Serving Benchmark Result ============
Successful requests:                     2         
Benchmark duration (s):                  9.67      
Total input tokens:                      169       
Total generated tokens:                  377       
Request throughput (req/s):              0.21      
Output token throughput (tok/s):         38.99     
Peak output token throughput (tok/s):    54.00     
Peak concurrent requests:                2.00      
Total Token throughput (tok/s):          56.47     
---------------Time to First Token----------------
Mean TTFT (ms):                          364.58    
Median TTFT (ms):                        364.58    
P99 TTFT (ms):                           364.74    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          37.51     
Median TPOT (ms):                        37.51     
P99 TPOT (ms):                           37.94     
---------------Inter-token Latency----------------
Mean ITL (ms):                           37.16     
Median ITL (ms):                         37.15     
P99 ITL (ms):                            43.49     
==================================================
```

## 20次

```
============ Serving Benchmark Result ============
Successful requests:                     20        
Benchmark duration (s):                  25.14     
Total input tokens:                      1751      
Total generated tokens:                  2971      
Request throughput (req/s):              0.80      
Output token throughput (tok/s):         118.19    
Peak output token throughput (tok/s):    400.00    
Peak concurrent requests:                20.00     
Total Token throughput (tok/s):          187.84    
---------------Time to First Token----------------
Mean TTFT (ms):                          12135.69  
Median TTFT (ms):                        12763.94  
P99 TTFT (ms):                           12767.09  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          57.43     
Median TPOT (ms):                        51.14     
P99 TPOT (ms):                           139.93    
---------------Inter-token Latency----------------
Mean ITL (ms):                           55.20     
Median ITL (ms):                         51.42     
P99 ITL (ms):                            70.33     
==================================================
```