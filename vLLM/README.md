# 安装最新版vLLM和编译后的CUDA

git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .

vllm/model_executor/layers/fused_moe/fused_moe.py : invoke_fused_moe_kernel