"""Command-line entry point for configured inference experiments."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from harness.models import LiteLLMClient, load_model_catalog
from harness.runner import ExperimentRunner, load_experiment

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a configured inference experiment"
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "models.yaml",
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "default.yaml",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs.jsonl",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    models = load_model_catalog(args.models_config)
    experiment = load_experiment(args.experiment)
    records = ExperimentRunner(LiteLLMClient(), args.results).run(experiment, models)

    for record in records:
        print(json.dumps(record.model_dump(mode="json"), ensure_ascii=False))
    return 1 if any(not record.success for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
