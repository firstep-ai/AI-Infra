# 安装最新版vLLM和编译后的CUDA

git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .

# pre-commit 检测

pip install pre-commit==4.0.1 pytest tensorizer>=2.9.0 pytest-forked pytest-asyncio pytest-rerunfailures pytest-shard pytest-timeout
pre-commit install
pre-commit run --files vllm/transformers_utils/config.py
pytest models/multimodal/processing/test_common.py -k "test_processing_correctness[1.0-32-0.3-omni-research/Tarsier-7b]"

vllm/model_executor/layers/fused_moe/fused_moe.py : invoke_fused_moe_kernel
