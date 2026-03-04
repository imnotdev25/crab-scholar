"""Tests for core data models."""

from crab_scholar.models import (
    AnalysisDimension,
    DimensionResult,
    ExtractedEntity,
    ExtractedRelation,
    Paper,
    PaperAnalysis,
)


class TestPaper:
    def test_create(self, sample_paper):
        assert sample_paper.paper_id == "abc123"
        assert sample_paper.title == "Attention Is All You Need"
        assert len(sample_paper.authors) == 3

    def test_display_name(self, sample_paper):
        assert sample_paper.display_name == "Attention Is All You Need (2017)"

    def test_display_name_no_year(self):
        p = Paper(paper_id="x", title="Test")
        assert p.display_name == "Test"

    def test_short_authors(self, sample_paper):
        assert sample_paper.short_authors == "Ashish Vaswani, Noam Shazeer, Niki Parmar"

    def test_short_authors_many(self):
        p = Paper(paper_id="x", title="Test", authors=["A", "B", "C", "D"])
        assert p.short_authors == "A et al."

    def test_short_authors_empty(self):
        p = Paper(paper_id="x", title="Test")
        assert p.short_authors == "Unknown"

    def test_serialization(self, sample_paper):
        data = sample_paper.model_dump()
        restored = Paper.model_validate(data)
        assert restored.paper_id == sample_paper.paper_id
        assert restored.title == sample_paper.title


class TestAnalysisDimension:
    def test_render_prompt(self, sample_paper, sample_dimension):
        prompt = sample_dimension.render_prompt(sample_paper)
        assert "Attention Is All You Need" in prompt
        assert "recurrent" in prompt  # from abstract

    def test_render_prompt_no_text(self, sample_dimension):
        p = Paper(paper_id="x", title="Empty Paper")
        prompt = sample_dimension.render_prompt(p)
        assert "No text available" in prompt


class TestExtractedEntity:
    def test_create(self):
        e = ExtractedEntity(
            name="Transformer",
            entity_type="METHOD",
            confidence=0.95,
        )
        assert e.name == "Transformer"
        assert e.entity_type == "METHOD"

    def test_confidence_bounds(self):
        e = ExtractedEntity(name="X", entity_type="Y", confidence=0.5)
        assert 0.0 <= e.confidence <= 1.0


class TestPaperAnalysis:
    def test_create(self, sample_analysis):
        assert sample_analysis.paper.paper_id == "abc123"
        assert "analysis" in sample_analysis.dimensions
        assert len(sample_analysis.dimensions["analysis"].entities) == 2
        assert len(sample_analysis.dimensions["analysis"].relations) == 1
