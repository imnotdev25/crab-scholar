"""Shared test fixtures for CrabScholar."""

import pytest

from crab_scholar.models import (
    AnalysisDimension,
    DimensionResult,
    ExtractedEntity,
    ExtractedRelation,
    Paper,
    PaperAnalysis,
)


@pytest.fixture
def sample_paper():
    """A sample paper for testing."""
    return Paper(
        paper_id="abc123",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        abstract="The dominant sequence transduction models are based on complex recurrent...",
        year=2017,
        doi="10.5555/3295222.3295349",
        url="https://arxiv.org/abs/1706.03762",
        venue="NeurIPS",
        citation_count=90000,
        reference_count=40,
        source="scholar",
        open_access_pdf="https://arxiv.org/pdf/1706.03762.pdf",
    )


@pytest.fixture
def sample_dimension():
    """A sample analysis dimension."""
    return AnalysisDimension(
        name="test_dimension",
        display_name="Test Dimension",
        description="A test dimension",
        system_message="You are a test analyst.",
        extraction_prompt=(
            "Analyze: {title}\n"
            "Text: {paper_text}\n"
            "Return JSON."
        ),
    )


@pytest.fixture
def sample_analysis(sample_paper):
    """A sample paper analysis."""
    return PaperAnalysis(
        paper=sample_paper,
        dimensions={
            "analysis": DimensionResult(
                dimension_name="analysis",
                paper_id="abc123",
                content='{"summary": "A transformer paper"}',
                entities=[
                    ExtractedEntity(
                        name="Transformer",
                        entity_type="METHOD",
                        confidence=0.95,
                        context="We propose the Transformer",
                        source_paper_id="abc123",
                    ),
                    ExtractedEntity(
                        name="BLEU",
                        entity_type="METRIC",
                        confidence=0.9,
                        context="measured by BLEU score",
                        source_paper_id="abc123",
                    ),
                ],
                relations=[
                    ExtractedRelation(
                        relation_type="PROPOSES",
                        source_entity="Attention Is All You Need",
                        target_entity="Transformer",
                        confidence=0.95,
                        evidence="We propose the Transformer",
                        source_paper_id="abc123",
                    ),
                ],
            ),
        },
        model_used="openai/gpt-4o-mini",
    )


@pytest.fixture
def tmp_output(tmp_path):
    """Temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out
