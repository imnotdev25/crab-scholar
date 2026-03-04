"""Tests for configuration module."""

import os

import pytest
import yaml

from crab_scholar.config import CrabConfig


class TestCrabConfig:
    def test_defaults(self):
        config = CrabConfig()
        assert config.default_model == "openai/gpt-4o-mini"
        assert config.citation_depth == 3
        assert config.max_papers == 50
        assert config.fallback_models == []
        assert config.concurrency == 4

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("CRAB_DEFAULT_MODEL", "anthropic/claude-3-haiku")
        monkeypatch.setenv("CRAB_CITATION_DEPTH", "5")
        config = CrabConfig()
        assert config.default_model == "anthropic/claude-3-haiku"
        assert config.citation_depth == 5

    def test_fallback_models_from_string(self, monkeypatch):
        # pydantic-settings parses list env vars as JSON arrays
        monkeypatch.setenv("CRAB_FALLBACK_MODELS", '["model-a", "model-b", "model-c"]')
        config = CrabConfig()
        assert config.fallback_models == ["model-a", "model-b", "model-c"]

    def test_fallback_models_from_comma_string(self):
        # Comma-separated works via init kwargs / YAML
        config = CrabConfig(fallback_models="model-a, model-b, model-c")
        assert config.fallback_models == ["model-a", "model-b", "model-c"]

    def test_model_chain(self):
        config = CrabConfig(
            default_model="primary",
            fallback_models=["fallback1", "fallback2"],
        )
        assert config.get_model_chain() == ["primary", "fallback1", "fallback2"]

    def test_crawl_direction_validation(self):
        assert CrabConfig(crawl_direction="references").crawl_direction == "references"
        assert CrabConfig(crawl_direction="citations").crawl_direction == "citations"
        assert CrabConfig(crawl_direction="both").crawl_direction == "both"
        with pytest.raises(ValueError, match="crawl_direction"):
            CrabConfig(crawl_direction="invalid")

    def test_yaml_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        yaml_content = {
            "default_model": "test/model",
            "citation_depth": 7,
            "output": "my_output",
        }
        (tmp_path / "rclaw.yaml").write_text(yaml.dump(yaml_content))
        config = CrabConfig()
        assert config.default_model == "test/model"
        assert config.citation_depth == 7

    def test_export_api_keys(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = CrabConfig(api_key="test-key-123")
        config.export_api_keys()
        assert os.environ.get("OPENAI_API_KEY") == "test-key-123"
