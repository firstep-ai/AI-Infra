### 故事一：YonGPT（企业级AI助手的开发与复杂知识库构建）

这个故事可以用来回答关于**解决问题、技术挑战、团队合作、沟通能力、目标设定与达成、适应性**等方面的问题。

#### 中文版

**S - 情境 (Situation):**
**“我在永友全球研发部门负责开发核心的海外AI产品——YonGPT。这个项目旨在将AI助手深度集成到我们公司的业务软件生态系统YonBIP中，为全球用户提供企业级的AI解决方案 **^^^^^^^^。我们面临的挑战是，需要构建一个能够处理复杂企业数据并提供准确、全面的AI回复的知识库，这涉及到多源数据集成和复杂的检索策略。”

**T - 任务 (Task):**
**“我的任务是主导设计和构建一个强大的AI知识库，它必须支持多种检索策略（向量搜索、倒排索引、知识图谱），以确保AI助手能够基于我们专有的企业数据提供高度准确和个性化的响应 **^^。同时，我还负责开发核心交互模块，比如ChatBI（自然语言数据库查询）和‘深度研究’模块（自动化网络爬取和报告生成）^^。”

**A - 行动 (Action):**
**“为了解决复杂知识库的挑战，我首先进行了深入的需求分析，与产品经理和业务团队紧密沟通，明确了不同业务场景对知识检索的精度和广度要求。基于此，我**

**架构并构建**了一个多路径检索策略的知识库，创新性地结合了向量搜索（处理语义相似性）、倒排索引（处理关键词匹配）和知识图谱（处理实体关系和复杂逻辑）^^。在开发ChatBI功能时，我与数据库专家紧密协作，将自然语言转化为SQL查询，并确保其准确性和效率。针对‘深度研究’模块，我设计并实现了自动化的网络爬取和信息抽取流程。在整个过程中，我

 **积极与团队成员分享我的设计思路和实现细节** **，确保大家对复杂的架构有清晰的理解，并根据团队反馈迭代优化方案。为了确保最终产品的稳定性和安全性，我采用了Django作为后端并与Next.js前端集成，并与公司的中央用户系统对接以实现安全的个性化访问控制 **^^。”

**R - 结果 (Result):**
**“通过我的努力，YonGPT项目的核心AI知识库得以成功构建并投入使用，它能够有效处理和检索复杂的企业数据，显著提升了AI助手的响应准确性和全面性。ChatBI和‘深度研究’等核心交互模块也顺利上线，极大地增强了产品的用户体验和实用性。这个项目帮助公司为全球用户提供了**

 **企业级的AI解决方案** **，并****深度融入了公司的核心业务软件生态** ^^^^^^^^。从这个经历中，我不仅提升了复杂系统架构和多源数据集成的能力，也深刻体会到在复杂项目中与多方高效沟通和协作的重要性。”

#### English Version

**S - Situation:**
"At Yonyou's Global R&D Department, I was a key contributor to developing YonGPT, our flagship overseas AI product. **This initiative aimed to deeply integrate an AI assistant into the company's core business software ecosystem, YonBIP, to deliver enterprise-grade AI solutions to a global user base**^^^^^^^^. A significant challenge was to build a sophisticated AI knowledge base capable of handling complex enterprise data and providing accurate, comprehensive AI responses, requiring multi-source data integration and advanced retrieval strategies."

**T - Task:**
**"My primary task was to spearhead the architecture and construction of this robust AI knowledge base, which needed to leverage a multi-path retrieval strategy (Vector Search, Inverted Index, Knowledge Graph) to ensure highly accurate and personalized responses based on our proprietary enterprise data**^^. **Additionally, I was responsible for developing core interactive modules, including ChatBI for natural language database querying and a 'Deep Research' module for automated web-crawling and report generation**^^."

**A - Action:**
"To address the complexities of the knowledge base, I began by conducting in-depth requirements analysis, collaborating closely with product managers and business teams to understand the specific demands for knowledge retrieval precision and breadth across various business scenarios. **Based on this, I **

**architected and built** a multi-path retrieval strategy, innovatively combining Vector Search (for semantic similarity), Inverted Index (for keyword matching), and Knowledge Graph (for entity relationships and complex logic)^^. For the ChatBI feature, I collaborated closely with database experts to transform natural language queries into efficient SQL, ensuring accuracy and performance. For the 'Deep Research' module, I designed and implemented automated web-crawling and information extraction pipelines. Throughout the process, I

 **proactively shared my design rationale and implementation details with the team** , ensuring a clear understanding of the complex architecture and iterating on solutions based on their feedback. **To ensure the final product's stability and security, I engineered the backend using Django and integrated it with a Next.js frontend, ensuring secure, personalized access control through the company's central user system**^^."

**R - Result:**
"Through my efforts, the core AI knowledge base for YonGPT was successfully built and deployed, effectively processing and retrieving complex enterprise data, which significantly enhanced the AI assistant's response accuracy and comprehensiveness. Key interactive modules like ChatBI and 'Deep Research' were also successfully launched, greatly improving the product's user experience and utility. **This project enabled the company to deliver **

**enterprise-grade AI solutions** to a global user base and was **deeply integrated into the company's core business software ecosystem**^^^^^^^^. From this experience, I not only strengthened my capabilities in complex system architecture and multi-source data integration but also deeply appreciated the importance of efficient communication and collaboration with multiple stakeholders in complex projects."

---

### 故事二：VLLM项目 (Multi-modal LLM集成与核心引擎优化)

这个故事非常适合回答关于**技术深度、创新、解决复杂技术问题、开源贡献、学习新技能、影响力和适应性**等方面的问题。

#### 中文版

**S - 情境 (Situation):**
**“我在VLLM项目组担任关键贡献者，VLLM是一个面向LLM的高吞吐量和内存高效推理和 serving 引擎 **^^^^^^^^。在项目中，我面临的主要挑战是需要将字节跳动的Tarsier多模态LLM（一个文本-图像-视频模型）端到端地集成到vLLM生态系统中 ^^^^。这是一个复杂的任务，因为它要求我们支持多模态推理、张量并行和流水线并行，并且要兼容vLLM的新旧引擎版本。”

**T - 任务 (Task):**
**“我的核心任务是独立主导Tarsier模型的完整集成 **^^^^。这包括实现对张量并行(TP)和流水线并行(PP)的全面支持 ^^^^、添加对量化模型版本(AWQ, GPTQ)的支持 ^^、确保与vLLM v0和v1引擎的兼容性 ^^，以及开发全面的单元测试以保证模型的稳定性和正确性 ^^。此外，我还负责强化vLLM引擎的稳定性与多功能性，解决诸如Mistral系列模型长上下文处理、CPU-only执行以及Qwen2-VL模型多图像/视频任务推理的准确性等关键问题 ^^^^^^^^。”

**A - 行动 (Action):**
**“为了集成Tarsier模型，我首先深入研究了Tarsier的模型架构和推理流程，特别是其处理多模态输入和输出的机制。我独立设计并实现了vLLM中对Tarsier的张量并行和流水线并行支持 **^^^^^^^^，这需要对vLLM底层的并行计算逻辑进行精细调整。我还负责添加了对量化模型（AWQ、GPTQ）的支持，这对于提高推理效率至关重要 ^^。在整个过程中，我

 **与开源社区保持紧密沟通** **，积极参与讨论，并在开发过程中不断进行单元测试和回归测试，以确保代码的鲁棒性 **^^。此外，在核心引擎增强方面，我

 **主动排查并解决了多个关键性问题** **，例如优化Mistral模型长上下文输入的处理流程，使其能够稳定运行；我还成功实现了vLLM v1引擎的CPU-only执行，极大地拓展了部署选项；并修复了Qwen2-VL模型在多图像和视频任务上的推理准确性问题 **^^^^^^^^。我的工作确保了开源社区能够高效地服务这个复杂模型 ^^。”

**R - 结果 (Result):**
**“通过我的领导和贡献，Tarsier多模态LLM成功且高效地集成到了vLLM生态系统中，使开源社区能够有效地服务这一复杂模型 **^^。我对vLLM核心引擎的增强显著提升了其稳定性、多功能性和部署灵活性，特别是在长上下文处理、CPU执行和多模态推理方面 ^^^^^^^^。这不仅解决了关键的技术挑战，也显著拓展了vLLM的应用范围。这次经历极大地提升了我对大型语言模型推理引擎底层机制的理解，以及在复杂开源项目中独立解决问题和贡献代码的能力，也让我体会到作为核心贡献者，我的工作对整个社区产生了实际的影响。”

#### English Version

**S - Situation:**
**"As a key contributor to the VLLM project, a high-throughput and memory-efficient inference and serving engine for LLMs **^^^^^^^^, my main challenge was to lead the end-to-end integration of ByteDance's Tarsier, a text-image-video multi-modal LLM, into the vLLM ecosystem^^^^. This was a complex task as it required supporting multi-modal inference, tensor and pipeline parallelism, and ensuring compatibility with both older and newer vLLM engine versions."

**T - Task:**
**"My core task was to independently lead the full integration of the Tarsier model**^^^^. **This involved implementing comprehensive support for Tensor Parallelism (TP) and Pipeline Parallelism (PP) **^^^^, adding support for quantized model versions (AWQ, GPTQ) ^^, ensuring compatibility with both vLLM v0 and v1 engines ^^, and developing comprehensive unit tests to guarantee stability and correctness^^. **Furthermore, I was responsible for strengthening the vLLM engine's stability and versatility by resolving critical issues such as stable processing of long-context inputs for Mistral-family models, CPU-only execution for the VLLM v1 Engine, and correct inference for multi-image and video tasks on Qwen2-VL models**^^^^^^^^."

**A - Action:**
"To integrate the Tarsier model, I began by deeply studying its architecture and inference pipeline, particularly its mechanisms for handling multi-modal inputs and outputs. **I independently designed and implemented support for Tarsier's Tensor Parallelism and Pipeline Parallelism within vLLM**^^^^^^^^, which required fine-tuning vLLM's underlying parallel computation logic. **I also took charge of adding support for quantized models (AWQ, GPTQ), which was crucial for improving inference efficiency**^^. **Throughout the process, I **

 **maintained close communication with the open-source community** **, actively participating in discussions, and continuously conducting unit and regression tests during development to ensure code robustness**^^. Additionally, for core engine enhancements, I

 **proactively identified and resolved several critical issues** , such as optimizing the long-context input processing for Mistral models to ensure stable operation; **I also successfully enabled CPU-only execution for the vLLM v1 engine, significantly expanding deployment options; and fixed inference accuracy issues for multi-image and video tasks on Qwen2-VL models**^^^^^^^^. **My work enabled the open-source community to efficiently serve this complex model**^^."

**R - Result:**
**"Through my leadership and contributions, the Tarsier multi-modal LLM was successfully and efficiently integrated into the vLLM ecosystem, enabling the open-source community to effectively serve this complex model**^^. **My enhancements to the vLLM core engine significantly improved its stability, versatility, and deployment flexibility, particularly in areas like long-context handling, CPU execution, and multi-modal inference**^^^^^^^^. This not only resolved critical technical challenges but also significantly expanded vLLM's application scope. This experience greatly deepened my understanding of the underlying mechanisms of large language model inference engines and my ability to independently solve problems and contribute code in complex open-source projects, demonstrating the tangible impact of my work as a key contributor to the entire community."
