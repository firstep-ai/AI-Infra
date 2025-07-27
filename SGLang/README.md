pip install -U "sglang[all]>=0.4.6.post1"

apt-get update && apt-get install -y libnuma-dev

pip install --upgrade "filelock>=3.13.1"

python -m sglang.launch_server --model-path Qwen/Qwen3-1.7B

python -m sglang.bench_serving --backend sglang --model Qwen/Qwen3-1.7B --dataset-name random --num-prompts 1000 --random-input-len 256 --random-output-len 100 --random-range-ratio 0.0 --profile
