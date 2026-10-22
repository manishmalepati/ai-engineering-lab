"""Exercise configuration, provider routing, and run persistence without a server."""

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from harness.models import LiteLLMClient, ModelConfig, load_model_catalog
from harness.runner import ExperimentConfig, ExperimentRunner, load_experiment


def fake_response(text: str = "answer") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5),
    )


def test_multiple_prompts_and_models_are_saved_as_jsonl(tmp_path: Path) -> None:
    class FakeClient:
        def complete(self, model: ModelConfig, prompt: str) -> SimpleNamespace:
            return fake_response(f"{model.model}: {prompt}")

    models = {
        "local": ModelConfig(
            provider="ollama", model="ollama/qwen3:8b", api_base_env="OLLAMA_BASE_URL"
        ),
        "cloud": ModelConfig(
            provider="gemini",
            model="gemini/gemini-2.5-flash",
            api_key_env="GEMINI_API_KEY",
        ),
    }
    experiment = ExperimentConfig(
        name="batch", models=["local", "cloud"], prompts=["one", "two"]
    )
    results_path = tmp_path / "results" / "runs.jsonl"

    records = ExperimentRunner(FakeClient(), results_path).run(experiment, models)
    saved = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) == len(saved) == 4
    assert len({record.run_id for record in records}) == 4
    assert all(UUID(record.run_id).version == 4 for record in records)
    assert [(record["model_alias"], record["prompt"]) for record in saved] == [
        ("local", "one"),
        ("local", "two"),
        ("cloud", "one"),
        ("cloud", "two"),
    ]
    assert all(record["success"] and record["input_tokens"] == 3 for record in saved)
    assert all(
        record["output_tokens"] == 5 and record["error"] is None for record in saved
    )


def test_failure_is_recorded_and_next_prompt_runs(tmp_path: Path) -> None:
    class SometimesFailingClient:
        def complete(self, model: ModelConfig, prompt: str) -> SimpleNamespace:
            if prompt == "bad":
                raise RuntimeError("simulated provider failure")
            return fake_response("good answer")

    model = ModelConfig(
        provider="ollama", model="ollama/qwen3:8b", api_base_env="OLLAMA_BASE_URL"
    )
    experiment = ExperimentConfig(
        name="failures", models=["local"], prompts=["bad", "good"]
    )
    results_path = tmp_path / "runs.jsonl"

    records = ExperimentRunner(SometimesFailingClient(), results_path).run(
        experiment, {"local": model}
    )

    assert len(records) == 2
    assert records[0].success is False
    assert records[0].error == "RuntimeError: simulated provider failure"
    assert records[0].response is None
    assert records[1].success is True
    assert records[1].response == "good answer"
    assert len(results_path.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.parametrize(
    ("alias", "environment_name", "expected_option"),
    [
        ("local_qwen", "OLLAMA_BASE_URL", "api_base"),
        ("claude", "ANTHROPIC_API_KEY", "api_key"),
        ("gemini", "GEMINI_API_KEY", "api_key"),
    ],
)
def test_litellm_uses_configured_provider_and_environment(
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
    environment_name: str,
    expected_option: str,
) -> None:
    models = load_model_catalog(
        Path(__file__).resolve().parents[1] / "configs" / "models.yaml"
    )
    monkeypatch.setenv(environment_name, "test-value")
    calls = []

    def mocked_completion(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return fake_response()

    monkeypatch.setattr("harness.models.completion", mocked_completion)
    LiteLLMClient().complete(models[alias], "test prompt")

    assert len(calls) == 1
    assert calls[0]["model"] == models[alias].model
    assert calls[0]["messages"] == [{"role": "user", "content": "test prompt"}]
    assert calls[0][expected_option] == "test-value"


def test_switching_models_uses_experiment_yaml(tmp_path: Path) -> None:
    experiment_file = tmp_path / "experiment.yaml"
    experiment_file.write_text(
        "name: switched\nmodels: [claude, gemini]\nprompts: [hello]\n", encoding="utf-8"
    )
    experiment = load_experiment(experiment_file)
    models = load_model_catalog(
        Path(__file__).resolve().parents[1] / "configs" / "models.yaml"
    )

    class FakeClient:
        def complete(self, model: ModelConfig, prompt: str) -> SimpleNamespace:
            return fake_response(model.model)

    records = ExperimentRunner(FakeClient(), tmp_path / "runs.jsonl").run(
        experiment, models
    )

    assert [record.model_alias for record in records] == ["claude", "gemini"]
    assert all(record.success for record in records)
