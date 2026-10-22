"""Model configuration and the LiteLLM provider adapter."""

import os
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from litellm import completion
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    provider: Literal["ollama", "anthropic", "gemini"]
    model: str = Field(min_length=1)
    api_base_env: str | None = None
    api_key_env: str | None = None

    @model_validator(mode="after")
    def validate_provider_settings(self) -> "ModelConfig":
        if not self.model.startswith(f"{self.provider}/"):
            raise ValueError(f"model must start with '{self.provider}/'")
        if self.provider == "ollama" and not self.api_base_env:
            raise ValueError("Ollama models require api_base_env")
        if self.provider != "ollama" and not self.api_key_env:
            raise ValueError("cloud models require api_key_env")
        return self


class ModelCatalog(BaseModel):
    models: dict[str, ModelConfig] = Field(min_length=1)


def load_model_catalog(path: Path) -> dict[str, ModelConfig]:
    with path.open(encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file)
    return ModelCatalog.model_validate(data).models


class CompletionClient(Protocol):
    def complete(self, model: ModelConfig, prompt: str) -> Any: ...


class LiteLLMClient:
    def complete(self, model: ModelConfig, prompt: str) -> Any:
        options: dict[str, Any] = {
            "model": model.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if model.api_base_env:
            options["api_base"] = _required_environment_value(model.api_base_env)
        if model.api_key_env:
            options["api_key"] = _required_environment_value(model.api_key_env)
        return completion(**options)


def _required_environment_value(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Environment variable {name} is required for this model")
    return value
