# Tested on

Fill this in on test day. It goes in the video description ("Tested on: …").

| | |
| --- | --- |
| Date | |
| Mac | e.g. MacBook Pro 14" (2024) |
| Chip / GPU cores | e.g. M4 Pro, 20-core GPU |
| Memory / bandwidth | e.g. 48 GB, 273 GB/s |
| macOS | |
| Ollama version | (bench.py saves it in results/summary.json) |
| Model tag + file size | e.g. `qwen3.5:9b` (GGUF Q4_K_M, 6.6 GB) |
| Engine | llama.cpp (GGUF) or Ollama MLX — check the server log |
| Context (num_ctx) | 32768 |
| Thinking | off (`"think": false`) — bench.py warns if it sees thinking |
| Power | plugged in, Low Power Mode off, lid open, other apps closed |
| Runs | median of 5, one warm-up, unique first line per run |

## Notes

- Anything odd (a run thrown out, a model that stopped early, cache not reused on the MLX tag):
