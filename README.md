# ai-engineering-lab

A small Python harness for recording model inference results. It uses LiteLLM so
the request interface can also support other providers later.

## Run locally

1. Install [uv](https://docs.astral.sh/uv/) and run `uv python install 3.12`.
2. Copy `.env.example` to `.env` and replace `YOUR_DESKTOP_IP` with the Ollama
   computer's address. Keep `.env` out of Git.
3. Make sure Ollama is serving the `qwen3:8b` model and is reachable from this
   computer.
4. Run `uv sync`, then `uv run python main.py`.

The script prints the response, latency, and token counts. It also appends each
run to `results/runs.jsonl`, which Git ignores.
