"""Core data models for ResearchClaw."""

from typing import Any

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """A research paper with metadata and optional full text."""

    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    venue: str = ""
    citation_count: int = 0
    reference_count: int = 0
    source: str = ""  # "scholar", "url", "text", "pdf"
    full_text: str = ""
    crawl_depth: int = 0
    open_access_pdf: str | None = None

    @property
    def display_name(self) -> str:
        year_str = f" ({self.year})" if self.year else ""
        return f"{self.title}{year_str}"

    @property
    def short_authors(self) -> str:
        if not self.authors:
            return "Unknown"
        if len(self.authors) <= 3:
            return ", ".join(self.authors)
        return f"{self.authors[0]} et al."


class ExtractedEntity(BaseModel):
    """An entity extracted from paper analysis."""

    name: str
    entity_type: str  # PAPER, DATASET, METRIC, METHOD, etc.
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    context: str = ""  # Quote from source text
    source_paper_id: str = ""


class ExtractedRelation(BaseModel):
    """A relation between two extracted entities."""

    relation_type: str
    source_entity: str
    target_entity: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: str = ""
    source_paper_id: str = ""


class AnalysisDimension(BaseModel):
    """Definition of a pluggable analysis dimension (loaded from YAML)."""

    name: str  # e.g. "analysis"
    display_name: str  # e.g. "Paper Analysis"
    description: str = ""
    system_message: str = ""
    extraction_prompt: str = ""  # Template with {paper_text}, {title}, {abstract}

    def render_prompt(self, paper: Paper) -> str:
        """Render the extraction prompt with paper data."""
        text = paper.full_text or paper.abstract or "(No text available)"
        return self.extraction_prompt.format(
            paper_text=text,
            title=paper.title,
            abstract=paper.abstract,
            authors=paper.short_authors,
            year=paper.year or "Unknown",
            venue=paper.venue or "Unknown",
        )


class DimensionResult(BaseModel):
    """Result of analyzing a paper with one dimension."""

    dimension_name: str
    paper_id: str
    content: str = ""  # Raw LLM output
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


class PaperAnalysis(BaseModel):
    """Complete analysis of a single paper across all dimensions."""

    paper: Paper
    dimensions: dict[str, DimensionResult] = Field(default_factory=dict)
    cost_usd: float = 0.0
    model_used: str = ""
    error: str | None = None
