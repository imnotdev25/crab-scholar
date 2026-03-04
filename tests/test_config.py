"""Tests for configuration module."""

import os
from pathlib import Path

import pytest
import yaml

from research_claw.config import RClawConfig


class TestRClawConfig:
    def test_defaults(self):
        config = RClawConfig()
        assert config.default_model == "openai/gpt-4o-mini"
        assert config.citation_depth == 3
        assert config.max_papers == 50
        assert config.fallback_models == []
        assert config.concurrency == 4

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("RCLAW_DEFAULT_MODEL", "anthropic/claude-3-haiku")
        monkeypatch.setenv("RCLAW_CITATION_DEPTH", "5")
        config = RClawConfig()
        assert config.default_model == "anthropic/claude-3-haiku"
        assert config.citation_depth == 5

    def test_fallback_models_from_string(self, monkeypatch):
        # pydantic-settings parses list env vars as JSON arrays
        monkeypatch.setenv("RCLAW_FALLBACK_MODELS", '["model-a", "model-b", "model-c"]')
        config = RClawConfig()
        assert config.fallback_models == ["model-a", "model-b", "model-c"]

    def test_fallback_models_from_comma_string(self):
        # Comma-separated works via init kwargs / YAML
        config = RClawConfig(fallback_models="model-a, model-b, model-c")
        assert config.fallback_models == ["model-a", "model-b", "model-c"]

    def test_model_chain(self):
        config = RClawConfig(
            default_model="primary",
            fallback_models=["fallback1", "fallback2"],
        )
        assert config.get_model_chain() == ["primary", "fallback1", "fallback2"]

    def test_crawl_direction_validation(self):
        assert RClawConfig(crawl_direction="references").crawl_direction == "references"
        assert RClawConfig(crawl_direction="citations").crawl_direction == "citations"
        assert RClawConfig(crawl_direction="both").crawl_direction == "both"
        with pytest.raises(ValueError, match="crawl_direction"):
            RClawConfig(crawl_direction="invalid")

    def test_yaml_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        yaml_content = {
            "default_model": "test/model",
            "citation_depth": 7,
            "output": "my_output",
        }
        (tmp_path / "rclaw.yaml").write_text(yaml.dump(yaml_content))
        config = RClawConfig()
        assert config.default_model == "test/model"
        assert config.citation_depth == 7

    def test_export_api_keys(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = RClawConfig(api_key="test-key-123")
        config.export_api_keys()
        assert os.environ.get("OPENAI_API_KEY") == "test-key-123"
