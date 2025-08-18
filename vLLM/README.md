## 安装最新版vLLM和编译后的CUDA

git clone https://github.com/vllm-project/vllm.git

cd vllm

VLLM_USE_PRECOMPILED=1 pip install --editable .

## pre-commit 检测

pip install pre-commit==4.0.1

pre-commit install

pre-commit run --files vllm/transformers_utils/config.py

## 单元测试

pip install pytest tensorizer>=2.9.0 pytest-forked pytest-asyncio pytest-rerunfailures pytest-shard pytest-timeout

pytest -v -s tests/models/multimodal/processing/test_common.py -k "omni-research/Tarsier2-Recap-7b"

## benchmark 测试

vllm serve Qwen/Qwen3-1.7B --enable-reasoning --reasoning-parser qwen3

vllm bench serve --model Qwen/Qwen3-1.7B --num-prompts 100 --random-input-len 1024 --random-output-len 1024 --ignore-eos
