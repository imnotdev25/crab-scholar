"""Configuration management for ResearchClaw using pydantic-settings.

Loads settings from: CLI flags > env vars > .env > rclaw.yaml > defaults.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

logger = logging.getLogger(__name__)

# Mapping from rclaw.yaml keys to config field names
_YAML_TO_FIELD = {
    "base_url": "base_url",
    "api_key": "api_key",
    "default_model": "default_model",
    "fallback_models": "fallback_models",
    "citation_depth": "citation_depth",
    "max_papers": "max_papers",
    "scholar_api_key": "scholar_api_key",
    "prompts_dir": "prompts_dir",
    "output": "output_dir",
    "concurrency": "concurrency",
}


class _ProjectYamlSource(PydanticBaseSettingsSource):
    """Read project config from rclaw.yaml (lower priority than env vars)."""

    def get_field_value(self, field, field_name):
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        yaml_path = Path.cwd() / "rclaw.yaml"
        if not yaml_path.exists():
            return {}

        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            logger.warning(f"Failed to parse rclaw.yaml: {e}")
            return {}

        result: dict[str, Any] = {}
        for yaml_key, field_name in _YAML_TO_FIELD.items():
            if yaml_key in data:
                result[field_name] = data[yaml_key]

        return result


class RClawConfig(BaseSettings):
    """Configuration settings for ResearchClaw.

    Settings are loaded from:
    1. Environment variables (with RCLAW_ prefix)
    2. .env file in current directory
    3. rclaw.yaml project config
    4. Built-in defaults
    """

    model_config = {
        "env_prefix": "RCLAW_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    # --- LLM ---
    base_url: str | None = Field(default=None, description="Custom API base URL")
    api_key: str | None = Field(default=None, description="Primary API key")
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    gemini_api_key: str | None = Field(default=None, description="Google Gemini API key")
    default_model: str = Field(
        default="openai/gpt-4o-mini", description="Default LLM model"
    )
    fallback_models: list[str] = Field(
        default_factory=list,
        description="Fallback models in priority order",
    )

    # --- Crawling ---
    citation_depth: int = Field(default=3, ge=1, le=10, description="Citation crawl depth")
    max_papers: int = Field(default=50, ge=1, le=500, description="Max papers to crawl")
    scholar_api_key: str | None = Field(default=None, description="Semantic Scholar API key")
    crawl_direction: str = Field(
        default="references",
        description="Crawl direction: references, citations, or both",
    )

    # --- Analysis ---
    prompts_dir: Path | None = Field(
        default=None, description="Custom prompts directory"
    )
    output_dir: Path = Field(default=Path("output"), description="Output directory")
    concurrency: int = Field(default=4, ge=1, le=20, description="Concurrent LLM calls")
    rpm: int = Field(default=40, ge=1, description="Rate limit (requests per minute)")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _ProjectYamlSource(settings_cls),
            file_secret_settings,
        )

    @field_validator("crawl_direction")
    @classmethod
    def validate_crawl_direction(cls, v: str) -> str:
        valid = {"references", "citations", "both"}
        if v not in valid:
            raise ValueError(f"crawl_direction must be one of {valid}, got {v!r}")
        return v

    @field_validator("fallback_models", mode="before")
    @classmethod
    def parse_fallback_models(cls, v):
        if isinstance(v, str):
            return [m.strip() for m in v.split(",") if m.strip()]
        return v

    @field_validator("output_dir")
    @classmethod
    def resolve_output_dir(cls, v: Path | str) -> Path:
        p = Path(v).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def export_api_keys(self) -> None:
        """Export API keys to environment so LiteLLM can find them."""
        if self.api_key:
            os.environ.setdefault("OPENAI_API_KEY", self.api_key)
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        if self.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key
        if self.gemini_api_key:
            os.environ["GEMINI_API_KEY"] = self.gemini_api_key

    def get_model_chain(self) -> list[str]:
        """Return the full model chain: [default_model] + fallback_models."""
        return [self.default_model] + list(self.fallback_models)
