"""Append inference run records to a JSON Lines file."""

import json
from pathlib import Path

from pydantic import BaseModel


class RunRecord(BaseModel):
    run_id: str
    timestamp: str
    experiment: str
    model_alias: str
    model: str
    provider: str
    prompt: str
    response: str | None
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    success: bool
    error: str | None


def append_run(path: Path, record: RunRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as runs_file:
        runs_file.write(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n"
        )
