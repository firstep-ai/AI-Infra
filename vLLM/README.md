# 安装最新版vLLM和编译后的CUDA

git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .

# pre-commit 检测

`pip install pre-commit`
`pre-commit install`
`pre-commit run --files vllm/transformers_utils/config.py`

vllm/model_executor/layers/fused_moe/fused_moe.py : invoke_fused_moe_kernel
