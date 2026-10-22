# ai-engineering-lab

A small Python inference harness for running prompt batches through Ollama,
Claude, or Gemini with one LiteLLM interface. Every model and prompt attempt is
appended to a local JSONL file with a unique run ID, latency, token counts,
success status, and any error.

## Setup

1. Install [uv](https://docs.astral.sh/uv/) and run `uv python install 3.12`.
2. Copy `.env.example` to `.env`. Set `OLLAMA_BASE_URL` to the computer serving
   Ollama. Keep `.env` out of Git.
3. Run `uv sync`.

## Run experiments

`uv run python main.py` runs `experiments/default.yaml` and writes to
`results/runs.jsonl`. The default experiment calls `local_qwen`, which maps to
`ollama/qwen3:8b` in `configs/models.yaml`.

Edit the `models` list and `prompts` list in an experiment YAML file to run
multiple prompts against one or more configured models. Add or change model IDs
and provider settings in `configs/models.yaml`; no Python changes are needed.
For example, use `models: [local_qwen, claude, gemini]` to run each prompt against
all three. Cloud calls require `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` in `.env`
and may incur provider charges. You can leave those values empty while using
Ollama.

Run a different file with `uv run python main.py --experiment experiments/my-run.yaml`.
Use `--models-config` and `--results` to choose alternate model definitions or
output paths. Failed calls are saved as records too; the CLI exits with code 1
if any call fails.

## Verify

Run `uv run pytest` and `uv run ruff check .`. Tests use fake model responses
and do not require Ollama or cloud API keys.
