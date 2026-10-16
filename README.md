# Prefill vs Decode: why local AI pauses before it answers

Code for the Ring Zero video **Prefill vs Decode Explained: Why Local AI Pauses Before It Answers** (1 Nov 2026).
Measure on your own Mac the two numbers the video is about: how fast your Mac *reads* a prompt and how fast it *writes* an answer.

Every reply from a local AI has two jobs:

- **Prefill (reading):** your whole message goes through the model at once. That's the silent pause, measured as *time to first token*. It's limited by your chip's **compute**.
- **Decode (writing):** the answer comes out one token at a time, and the whole model is read from memory for each one. That's the typing speed, in *tokens per second*. It's limited by **memory bandwidth**.

Your wait ≈ **tokens in ÷ reading speed + tokens out ÷ writing speed**. This folder measures both on your Mac.

## Run it on your Mac

You need [Ollama](https://ollama.com) running and Python 3 (no packages to install).

```bash
ollama pull qwen3.5:9b
python3 bench.py all --model qwen3.5:9b --chip "M4 Pro"
```

It takes about 15–25 minutes on an M4 Pro. Plug in, switch off Low Power Mode and close other apps first.
To run one test: `python3 bench.py hook` (or `depth`, `quiz`, `cache`, `load`).

## What each test measures

| Test | What it does | What it shows (the idea from the video) |
| --- | --- | --- |
| `hook` | Pastes a 5,000-word report and asks one question. Logs the wait before the first word, reading time and speed, writing time and speed, tokens in and out | The two bars from the opening: the silent pause vs the typing, and how many times faster reading is per word |
| `depth` | Reads prompts of about 500, 2,000, 8,000 and 16,000 tokens | Reading slows down per word as the prompt gets longer |
| `quiz` | Reads about 2,000 words, then separately writes about 200 | The quiz: on your Mac, which takes longer? |
| `cache` | Same document, then a follow-up question (prompt cache hit), then the same chat with the first line changed (cache miss) | The ending: the second question starts almost instantly, until you change the start |
| `load` | Unloads the model, then times a cold start and a warm one | "Three things can hide in that pause" |

Each test uses Ollama's own timings (`prompt_eval_duration` for reading, `eval_duration` for writing) plus a wall-clock
time to the first visible word. Thinking is switched off (`"think": false`); the script warns if it sees thinking tokens anyway.

## Results

- `results/<test>.csv` — every run
- `results/summary.json` — medians, the Ollama version and the date
- `results/TEST_block.py` — the same numbers in the form the video's script uses (for re-rendering the video with your own Mac)

Post yours in the video's pinned comment: your chip, your **prompt eval rate** (reading) and **eval rate** (writing).

## Optional: compare with the llama.cpp community table

```bash
brew install llama.cpp
./llama_bench.sh /path/to/llama-2-7b.Q4_0.gguf
```

This gives you a row comparable with the
[llama.cpp Apple Silicon benchmark table](https://github.com/ggml-org/llama.cpp/discussions/4167)
(reading = pp512, writing = tg128) and a depth sweep.

## Fair-test rules

1. Plugged in, Low Power Mode off, other apps closed.
2. One untimed warm-up; a unique first line on every run, so the prompt cache can't help (except in the `cache` test, where that's the point).
3. Thinking off.
4. Median of 5 runs, 5 seconds' rest between runs.
5. Write your setup in `TESTED.md`.

## Things that can trip you up

- **Ollama's prompt eval count includes cached tokens.** After a cache hit it still reports the whole prompt; the *duration* is what drops. Use the duration (this script does).
- **MLX tags and the cache.** One integration guide reports that Ollama's MLX engine doesn't yet reuse the prompt cache for Qwen 3.x. Run the `cache` test on both the GGUF and the MLX tag if you use both.
- **Long prompts need room.** The script sets `num_ctx` to 32768 so the 16,000-token prompt fits; lower it only if memory is tight.

## The numbers in the video

The M4 Pro figures shown in the video (about 713 tokens/s reading, 54 tokens/s writing, about 23 s before the first word on a 16K-token prompt, GPT-OSS 20B 4-bit) come from Hardware Corner's lab test, [Is the 48 GB Mac Mini M4 Pro Good for Local LLMs?](https://www.hardware-corner.net/is-mac-mini-m4-pro-good-for-llms/) (Aug 2026). Run `bench.py` to get yours.

## Sources

- Apple ML Research, [Exploring LLMs with MLX and the M5 Neural Accelerators](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [llama.cpp Apple Silicon benchmarks (#4167)](https://github.com/ggml-org/llama.cpp/discussions/4167)
- EXO Labs, [DGX Spark + Mac Studio](https://blog.exolabs.net/nvidia-dgx-spark/)
- [DistServe (OSDI 2024)](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) and [SARATHI](https://arxiv.org/pdf/2308.16369)
- [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) (prompt cache)
