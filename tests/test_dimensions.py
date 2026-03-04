"""Tests for the dimension loader."""


import yaml

from crab_scholar.analyze.dimensions import (
    DEFAULT_DIMENSIONS,
    list_available_dimensions,
    load_dimensions,
)


class TestLoadDimensions:
    def test_load_defaults(self):
        dims = load_dimensions()
        assert len(dims) == 5
        names = {d.name for d in dims}
        assert names == set(DEFAULT_DIMENSIONS)

    def test_all_defaults_have_prompts(self):
        dims = load_dimensions()
        for dim in dims:
            assert dim.extraction_prompt, f"{dim.name} has empty extraction_prompt"
            assert dim.system_message, f"{dim.name} has empty system_message"
            assert dim.display_name, f"{dim.name} has empty display_name"

    def test_include_filter(self):
        dims = load_dimensions(include=["analysis", "metrics"])
        assert len(dims) == 2
        names = {d.name for d in dims}
        assert names == {"analysis", "metrics"}

    def test_exclude_filter(self):
        dims = load_dimensions(exclude=["statistical_tests"])
        assert len(dims) == 4
        names = {d.name for d in dims}
        assert "statistical_tests" not in names

    def test_custom_dimension(self, tmp_path):
        custom_yaml = {
            "name": "custom_test",
            "display_name": "Custom Test",
            "description": "A custom dimension",
            "system_message": "Test system message",
            "extraction_prompt": "Analyze: {paper_text}",
        }
        (tmp_path / "custom_test.yaml").write_text(yaml.dump(custom_yaml))

        dims = load_dimensions(prompts_dir=tmp_path)
        names = {d.name for d in dims}
        assert "custom_test" in names
        assert len(dims) == 6  # 5 defaults + 1 custom

    def test_override_default(self, tmp_path):
        override_yaml = {
            "name": "analysis",
            "display_name": "Overridden Analysis",
            "system_message": "Custom system",
            "extraction_prompt": "Custom prompt: {paper_text}",
        }
        (tmp_path / "analysis.yaml").write_text(yaml.dump(override_yaml))

        dims = load_dimensions(prompts_dir=tmp_path)
        analysis_dim = next(d for d in dims if d.name == "analysis")
        assert analysis_dim.display_name == "Overridden Analysis"
        assert "Custom prompt" in analysis_dim.extraction_prompt


class TestListAvailable:
    def test_list(self):
        dims = list_available_dimensions()
        assert len(dims) == 5
        for d in dims:
            assert d["source"] == "bundled"
            assert "name" in d
            assert "display_name" in d
