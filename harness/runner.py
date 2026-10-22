"""Load and execute prompt batches against configured models."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import yaml
from pydantic import BaseModel, Field

from harness.metrics import measure_call
from harness.models import CompletionClient, ModelConfig
from harness.storage import RunRecord, append_run


class ExperimentConfig(BaseModel):
    name: str = Field(min_length=1)
    models: list[str] = Field(min_length=1)
    prompts: list[str] = Field(min_length=1)


def load_experiment(path: Path) -> ExperimentConfig:
    with path.open(encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file)
    return ExperimentConfig.model_validate(data)


class ExperimentRunner:
    def __init__(self, client: CompletionClient, results_path: Path) -> None:
        self.client = client
        self.results_path = results_path

    def run(
        self, experiment: ExperimentConfig, models: dict[str, ModelConfig]
    ) -> list[RunRecord]:
        missing = set(experiment.models) - models.keys()
        if missing:
            raise ValueError(f"Unknown model aliases: {', '.join(sorted(missing))}")

        records = []
        for model_alias in experiment.models:
            model = models[model_alias]
            for prompt in experiment.prompts:
                measurement = measure_call(
                    lambda model=model, prompt=prompt: self.client.complete(
                        model, prompt
                    )
                )
                record = RunRecord(
                    run_id=str(uuid4()),
                    timestamp=datetime.now(UTC).isoformat(),
                    experiment=experiment.name,
                    model_alias=model_alias,
                    model=model.model,
                    provider=model.provider,
                    prompt=prompt,
                    response=measurement.response,
                    latency_seconds=measurement.latency_seconds,
                    input_tokens=measurement.input_tokens,
                    output_tokens=measurement.output_tokens,
                    success=measurement.success,
                    error=measurement.error,
                )
                append_run(self.results_path, record)
                records.append(record)
        return records
