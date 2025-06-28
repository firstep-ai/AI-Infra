# AI-Infra

## vLLM 多模态
VLLM 多模态支持演进：从 V0 到 V1 的优化改进
V0 版本存在的问题
VLLM 的 V0 版本在多模态模型支持方面遇到了以下三个核心问题：

1. Prefill 不兼容问题（Chunked Prefill Incompatibility）
VLLM 的 Prefill 机制通过将请求分块处理提升文本处理性能。

图像编码器（如 ViT）使用 全注意力机制（Full Attention），要求图像嵌入 一次性生成，不能分块。

问题示例：

若视频含 64 帧，每帧 48x48 分辨率，图像嵌入可能超过 16,000 tokens。

若 Prefill 预算仅为 2048 tokens，需要运行图像编码器 9 次，引发性能浪费和重复计算。

2. 前缀缓存（Prefix Caching）问题
V0 的前缀缓存通过 Token ID 实现共享 KVC（Key-Value Cache）块。

多模态模型中的图像占位符 Token ID 相同（如均为 32000）。

若两个请求使用不同图像但相同文本提示，可能错误共享缓存，导致 推理结果错误。

为避免错误，V0 强制 关闭多模态前缀缓存功能。

3. 输入预处理 CPU 开销大
原假设图像预处理（如将 Pillow Image 转 Tensor）CPU 开销极小。

实际上，该过程常常 比编码器运行还耗时，造成 CPU 成为 性能瓶颈，GPU 利用率降低。

V1 版本的改进与解决方案
V1 针对 V0 的瓶颈进行了多项重构与优化：

1. 引入编码器缓存（Encoder Cache）
图像经编码器生成的嵌入存入 GPU 内存缓存。

Prefill 直接从缓存中读取，避免重复编码器计算。

当前支持 单请求缓存，未来将支持 跨请求共享。

调度器（Scheduler）现在同时关注 解码器与编码器预算，避免阻塞。

2. 前缀缓存机制重构
新机制引入额外标识（如图像哈希或用户定义 UID）配合 Token ID 区分 KVC。

即使文本提示相同，不同图像也能正确区分，保证缓存正确性。

此机制同时提升了对 LoRA 等其他模块的支持。

3. 分离引擎循环与图像特征缓存
引擎进程分离：

一个专注于 CPU 密集任务（如图像预处理）。

一个处理 GPU 推理任务，提升 GPU 利用率。

图像特征缓存（Feature Caching）：

在 CPU 端缓存预处理后的图像特征。

遇到重复图像时，跳过重复预处理，直接使用缓存。

使用与前缀缓存相同的标识机制，结合 镜像缓存 技术减少进程间数据传输。

性能提升
整体服务吞吐量显著提高，尤其在高 QPS 和数据重复率高的场景下效果更明显。

即使数据无重复，新增机制（如哈希计算）也仅带来 约 1% 的性能损失，影响极小。

## Linger 项目简介：为 LLM 训练与推理打造高效算子优化方案
Linger 项目的核心动机是解决 大语言模型（LLM）训练与推理中的 GPU 瓶颈问题，主要集中在以下两个方面：

一、GPU 面临的主要瓶颈
1. GPU 内存压力与 OOM 问题
激活内存巨大：

序列长度和 batch size 增加时，激活（activation）内存占用急剧上升。

即便使用 Zero、FSDP 等优化技术，激活仍需占用每个 GPU 的大量内存。

大词表导致的 logits 内存激增：

LLaMA2 起，开源模型的词表通常超过 100K（如：LLaMA2 为 128K，Qwen 为 150K）。

以序列长度 8000 为例，仅 logits 的 FP32 内存需求就达 8GB。

后续 softmax、log 操作也在 FP32 下进行并复制数据，导致 GPU 内存峰值可能从 24GB 飙升至 60GB。

后果：

用户被迫缩短上下文长度、减小 batch size 或启用 gradient checkpointing，降低了训练效率。

2. GPU 多处理器利用率低
碎片化 Kernel 启动：

使用 PyTorch 且未编译优化（如未使用 torch.compile）时，会频繁启动小核函数。

Python → PyTorch 调度器开销大：

每次 kernel 启动都涉及 Python 到底层调度器的跳转，存在显著 overhead。

LLM I/O 与计算比高：

放大了上述调度开销。

二、Linger 项目的关键设计理念与实践
1. 需求驱动的开发模式（Demand-Driven Development）
从实际训练负载出发，识别瓶颈并逐一击破。

非从底层算子技术栈“自上而下”设计，而是“自下而上”解决痛点。

示例：C-Loss（Trunk Loss） 就是为了解决 logits 内存过大的问题而诞生的。

2. 使用 Triton 开发算子的动机
优于 CUDA 的开发体验：

学习曲线更低、开发效率更高、依赖更简洁。

便于社区贡献与维护。

适合训练场景：

Triton 是 Python JIT 编译语言，几秒的编译时间可忽略不计。

3. 现有算子库的不足
GitHub 上的算子库多数 零散、不可组合、集成难。

很多库 不兼容 Hugging Face，需迁移到自定义训练器，代价高。

Linger 的目标是 支持 Hugging Face 代码的原生集成，仅需少量代码改动即可享受优化效果。

三、Linger 提出的关键优化技术
1. 重计算（Recomputation）
借鉴 Flash Attention 思想，在反向传播阶段 重新计算部分激活。

用计算换内存，显著节省空间。

示例：RMSNorm 中通过重新加载输入并重新归一化，节省了激活内存。

2. 原地执行（In-place Execution）
将计算结果直接写回输入张量，避免中间张量创建。

适用于只有一个生产者和一个消费者的张量。

示例：在 RoPE 实现中，直接将旋转后的 QK 写入原地址。

3. 核函数融合（Coalescing / Chaining）
在 一个核函数中合并多个操作，提升效率，降低调度开销。

示例：RoPE 中同时处理 Q 和 K 的旋转操作。

4. Trunk Loss（C-Loss）
为了解决 logits 占用过多显存的问题：

避免 materialize 巨大的 batch_size × sequence_length × vocab_size logits 张量。

将 token 划分成小块（chunks），分别计算 partial loss 和 gradient。

使用定制算子在一个核函数中同时计算正向与反向，并 原地写回 logits 梯度，彻底消除 logits 带来的显存峰值。

四、优化重点领域选择
Linger 不再重复优化已有优秀实现的模块（如 MLP 和 Attention，已由 cuBLAS、Flash Attention 支持）。

重点优化以下常见但缺乏优化的模块：

Rotary Embedding

LayerNorm / RMSNorm

Gate Activation

五、总结
Linger 项目是应对 LLM 训练两大瓶颈（内存压力、计算效率低）的实际产物。
通过：

使用 Triton 进行高效算子开发，

引入重计算、原地执行、核函数融合与 Trunk Loss 等优化策略，

提供与 Hugging Face 深度兼容、低集成成本的解决方案，

Linger 成为大模型训练优化领域中 高效、易用、社区友好 的代表性项目。

## PyTorch 2.0 编译器体系中的 Dynamo：前端与“大脑”
🧠 什么是 Dynamo？
Dynamo 是 PyTorch 2.0 编译器（torch.compile）的 前端组件，负责 安全地捕获 Python 代码，并将其转换为 PyTorch 可优化的中间表示 —— FX Graph。

🗺️ 类比理解：一个顶级翻译团队
想象你有一份充满俚语和复杂句式的文学手稿（纯 Python 代码），希望将它：

快速翻译

适合大规模印刷（GPU/CPU 上高效运行）

这个“翻译团队”就是 torch.compile，包括三个专家：

🧑‍💼 1. Dynamo（首席口译官 / 前端）
职责：

读取你的 Python 代码，提取出可标准化的计算部分（如张量操作），生成 FX Graph。

碰到不能翻译的部分（如复杂 Python 控制流），会触发 Graph Break，让原生 Python 解释器执行，再继续捕获。

关键能力：Graph Break 技术

遇到不可处理部分时：

停止捕获

执行原生 Python

恢复捕获（继续构建下一个 FX Graph）

目标：尽可能多地捕获可优化图，同时保持 100% 安全和正确性。

🧠 2. AOTAutograd（自动微分专家）
职责：

接手 Dynamo 生成的前向图，自动推导反向传播图，为训练阶段准备梯度信息。

🛠️ 3. Inductor（后端优化专家）
职责：

将 FX Graph 编译为底层高效代码（如 Triton 内核/C++），适用于 CPU/GPU。

应用 算子融合、内存布局优化 等技巧，以最大化性能。

🔍 Dynamo 的核心意义与工作机制
🟦 1. 它是什么？
Dynamo 是一个 Python → FX Graph 的编译器。

是 torch.compile(model) 执行流程的 入口点。

🟨 2. 它解决了什么问题？
旧方案痛点（如 torch.jit.script）：

要求使用 静态子集的 Python 代码。

灵活性差，修改成本高。

Dynamo 的革命性：

支持 几乎所有 Python 特性。

无需用户修改代码 即可进行加速优化。

🟩 3. 它是如何工作的？——Graph Capture + Graph Break
✅ 捕获（Capture）
分析 Python 字节码，自动识别连续、支持的 PyTorch 操作，转为 FX Graph。

❌ 中断（Graph Break）
遇到无法安全捕获的代码时（如原生控制流、不支持的库、Python 动态特性）：

停止捕获

使用原生解释器执行

尝试从下一句恢复捕获

这让 torch.compile 兼顾灵活性与性能，是其成功的关键机制。

🧾 总结：Dynamo 与 Inductor 的分工
角色	职责
Dynamo（前端）	将灵活的 Python 转换为结构化 FX Graph，并通过 Graph Break 技术保持兼容性
Inductor（后端）	接收 FX Graph，生成底层高性能代码，实现实际加速效果

当你执行：

`model = torch.compile(model)`

你首先调用的就是 Dynamo —— 它是 PyTorch 2.0 编译器成功的“第一块基石”。

