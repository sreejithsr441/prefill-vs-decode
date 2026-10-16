#!/usr/bin/env bash
# Optional: your Mac's row for the llama.cpp Apple Silicon table (#4167), plus the depth sweep.
#   brew install llama.cpp          # provides llama-bench
#   ./llama_bench.sh /path/to/model.gguf
# For a row comparable with #4167, use the same Llama 2 7B Q4_0 file the discussion uses
# (see its first post: https://github.com/ggml-org/llama.cpp/discussions/4167).
set -euo pipefail
MODEL="${1:?usage: ./llama_bench.sh /path/to/model.gguf}"
mkdir -p results
echo "== #4167-style row: reading = pp512, writing = tg128"
llama-bench -m "$MODEL" -ngl 99 -fa 1 -p 512 -n 128 -o md | tee results/llama_bench_4167.md
echo "== depth sweep: reading and writing speed as the prompt grows"
llama-bench -m "$MODEL" -ngl 99 -fa 1 -p 512 -n 128 -d 0,4096,16384,32768 -o md | tee results/llama_bench_depth.md
llama-bench --version 2>&1 | head -2 | tee results/llama_bench_version.txt
