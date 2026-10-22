"""Run one inference request and save its result as JSON Lines."""

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from litellm import completion

PROJECT_ROOT = Path(__file__).resolve().parent
PROMPT = "Explain what an LLM inference harness is in two sentences."


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    model = os.getenv("DEFAULT_MODEL", "ollama/qwen3:8b")
    base_url = os.environ["OLLAMA_BASE_URL"]

    start = time.perf_counter()
    response = completion(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        api_base=base_url,
    )
    latency = time.perf_counter() - start

    usage = getattr(response, "usage", None)
    result = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": model,
        "prompt": PROMPT,
        "response": response.choices[0].message.content,
        "latency_seconds": round(latency, 3),
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
    }

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    with (results_dir / "runs.jsonl").open("a", encoding="utf-8") as runs_file:
        runs_file.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
