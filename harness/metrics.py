"""Measure a completion call and normalize its response metrics."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CallMeasurement:
    response: str | None
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    success: bool
    error: str | None


def measure_call(call: Callable[[], Any]) -> CallMeasurement:
    start = time.perf_counter()
    try:
        completion = call()
        response = completion.choices[0].message.content
        usage = getattr(completion, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
    except Exception as exc:  # noqa: BLE001
        # Preserve any provider failure as a record and continue the batch.
        return CallMeasurement(
            response=None,
            latency_seconds=round(time.perf_counter() - start, 3),
            input_tokens=None,
            output_tokens=None,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    return CallMeasurement(
        response=response,
        latency_seconds=round(time.perf_counter() - start, 3),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        success=True,
        error=None,
    )
