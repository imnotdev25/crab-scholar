"""Tests for the LLM client."""

import json

from crab_scholar.llm_client import parse_llm_json


class TestParseLLMJson:
    def test_plain_json(self):
        result = parse_llm_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_fenced(self):
        text = '```json\n{"key": "value"}\n```'
        result = parse_llm_json(text)
        assert result == {"key": "value"}

    def test_markdown_fenced_no_lang(self):
        text = '```\n{"key": "value"}\n```'
        result = parse_llm_json(text)
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self):
        text = 'Here is the result:\n{"key": "value"}\nDone.'
        result = parse_llm_json(text)
        assert result == {"key": "value"}

    def test_invalid_json_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Could not parse"):
            parse_llm_json("not json at all")

    def test_nested_json(self):
        data = {"entities": [{"name": "X", "type": "Y"}], "count": 1}
        result = parse_llm_json(json.dumps(data))
        assert result == data
