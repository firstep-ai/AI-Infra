如何避免MoE架构下的DP死锁 

扩散LLM的Infra方案

Truncated Prefill (TRF)（优先处理p，后处理d。   pd分离（适合异构GPU））是否严格优于其他方法？

投机解码中大模型和小模型的因并行策略不同导致资源浪费，A靠B，B靠A，小模型的kv cache进行舍弃，跨kv cache block
