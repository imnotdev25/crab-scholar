"""Pipeline orchestration — library API for CrabScholar.

Provides high-level functions for the full analysis workflow:
analyze → build_graph → export. Can be used as a Python library
or called from the CLI.
"""

import json
import logging
from pathlib import Path

from crab_scholar.analyze.analyzer import analyze_all
from crab_scholar.analyze.dimensions import load_dimensions
from crab_scholar.config import CrabConfig
from crab_scholar.export import export_graph
from crab_scholar.graph.builder import build_graph
from crab_scholar.graph.knowledge_graph import KnowledgeGraph
from crab_scholar.ingest.crawler import crawl
from crab_scholar.ingest.fetcher import fetch_paper_text, fetch_paper_text_from_path
from crab_scholar.ingest.scholar import ScholarClient
from crab_scholar.llm_client import LLMClient
from crab_scholar.models import Paper, PaperAnalysis

logger = logging.getLogger(__name__)


def _make_llm_client(config: CrabConfig) -> LLMClient:
    """Create an LLM client from config."""
    config.export_api_keys()
    return LLMClient(
        model=config.default_model,
        fallback_models=config.fallback_models,
        rpm=config.rpm,
        base_url=config.base_url,
        api_key=config.api_key,
    )


def run_search(
    query: str,
    config: CrabConfig | None = None,
    limit: int = 10,
) -> list[Paper]:
    """Search for papers by keywords.

    Args:
        query: Keywords (comma-separated or natural language)
        config: Optional config (for Scholar API key)
        limit: Max results

    Returns:
        List of matching Papers
    """
    config = config or CrabConfig()
    with ScholarClient(api_key=config.scholar_api_key) as scholar:
        return scholar.search(query, limit=limit)


def run_analyze(
    input_query: str | None = None,
    pdf_path: Path | None = None,
    raw_text: str | None = None,
    config: CrabConfig | None = None,
    dimensions: list[str] | None = None,
    depth: int | None = None,
) -> list[PaperAnalysis]:
    """Full analysis pipeline: search/fetch → crawl → analyze.

    Exactly one of input_query, pdf_path, or raw_text must be provided.

    Args:
        input_query: Paper title, DOI, URL, or keywords
        pdf_path: Path to a local PDF
        raw_text: Raw text to analyze directly
        config: Configuration
        dimensions: Optional list of dimension names to use
        depth: Override citation depth

    Returns:
        List of PaperAnalysis results
    """
    config = config or CrabConfig()
    llm = _make_llm_client(config)
    crawl_depth = depth or config.citation_depth

    # Load dimensions
    dims = load_dimensions(
        prompts_dir=config.prompts_dir,
        include=dimensions,
    )

    # Step 1: Get seed papers
    papers: list[Paper] = []

    if raw_text:
        import hashlib
        text_hash = hashlib.md5(raw_text.encode()).hexdigest()[:12]
        papers = [Paper(
            paper_id=f"text:{text_hash}",
            title="User-provided text",
            source="text",
            full_text=raw_text,
        )]
    elif pdf_path:
        text = fetch_paper_text_from_path(Path(pdf_path))
        papers = [Paper(
            paper_id=f"pdf:{Path(pdf_path).stem}",
            title=Path(pdf_path).stem.replace("_", " ").replace("-", " ").title(),
            source="pdf",
            full_text=text,
        )]
    elif input_query:
        papers = _resolve_input(input_query, config)
    else:
        raise ValueError("Must provide input_query, pdf_path, or raw_text")

    if not papers:
        logger.warning("No papers found to analyze")
        return []

    logger.info(f"Found {len(papers)} seed paper(s)")

    # Step 2: Crawl citations/references
    if crawl_depth > 1 and papers[0].source == "scholar":
        with ScholarClient(api_key=config.scholar_api_key) as scholar:
            papers = crawl(
                seed_papers=papers,
                scholar=scholar,
                depth=crawl_depth,
                max_papers=config.max_papers,
                direction=config.crawl_direction,
            )
        logger.info(f"After crawl: {len(papers)} papers total")

    # Step 3: Fetch full text for each paper
    cache_dir = config.output_dir / ".pdf_cache"
    for paper in papers:
        if not paper.full_text:
            try:
                paper.full_text = fetch_paper_text(paper, cache_dir=cache_dir)
            except Exception as e:
                logger.warning(f"Could not fetch text for '{paper.title[:50]}': {e}")

    # Step 4: Run analysis
    analyses = analyze_all(
        papers=papers,
        dimensions=dims,
        llm=llm,
        concurrency=config.concurrency,
    )

    # Step 5: Save results
    _save_analyses(analyses, config.output_dir)

    logger.info(
        f"Pipeline complete: {len(analyses)} papers analyzed, "
        f"${llm.total_cost:.4f} total LLM cost"
    )
    return analyses


def run_build(
    output_dir: Path | None = None,
    config: CrabConfig | None = None,
) -> KnowledgeGraph:
    """Build knowledge graph from saved analysis results.

    Args:
        output_dir: Directory containing analysis JSON files
        config: Optional config

    Returns:
        Populated KnowledgeGraph
    """
    config = config or CrabConfig()
    out = Path(output_dir) if output_dir else config.output_dir

    # Load analyses from saved JSON
    analyses = _load_analyses(out)
    if not analyses:
        raise FileNotFoundError(f"No analysis results found in {out}/analyses/")

    # Build graph
    kg = build_graph(analyses)

    # Save graph
    graph_path = out / "graph.json"
    kg.save(graph_path)

    return kg


def run_export(
    fmt: str = "json",
    output_dir: Path | None = None,
    config: CrabConfig | None = None,
) -> Path:
    """Export graph to a specific format.

    Args:
        fmt: Export format (json, graphml, gexf, csv)
        output_dir: Output directory
        config: Optional config

    Returns:
        Path to exported file
    """
    config = config or CrabConfig()
    out = Path(output_dir) if output_dir else config.output_dir

    # Load graph
    graph_path = out / "graph.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}. Run `rclaw build` first.")

    kg = KnowledgeGraph.load(graph_path)
    return export_graph(kg, fmt=fmt, output_dir=out / "exports")


def run_pipeline(
    input_query: str | None = None,
    pdf_path: Path | None = None,
    raw_text: str | None = None,
    config: CrabConfig | None = None,
    export_formats: list[str] | None = None,
) -> KnowledgeGraph:
    """Run the full pipeline: analyze → build → export.

    Args:
        input_query: Paper query
        pdf_path: Local PDF path
        raw_text: Raw text
        config: Configuration
        export_formats: List of export formats (default: ["json"])

    Returns:
        Built KnowledgeGraph
    """
    config = config or CrabConfig()

    # Analyze
    analyses = run_analyze(
        input_query=input_query,
        pdf_path=pdf_path,
        raw_text=raw_text,
        config=config,
    )

    if not analyses:
        raise ValueError("No papers were analyzed")

    # Build graph
    kg = build_graph(analyses)
    kg.save(config.output_dir / "graph.json")

    # Export
    for fmt in (export_formats or ["json"]):
        export_graph(kg, fmt=fmt, output_dir=config.output_dir / "exports")

    return kg


def _resolve_input(query: str, config: CrabConfig) -> list[Paper]:
    """Resolve user input (query, DOI, URL, keywords) to papers."""
    with ScholarClient(api_key=config.scholar_api_key) as scholar:
        # Try as paper ID/DOI/URL first
        paper_id = scholar.resolve_paper_id(query)
        if paper_id:
            try:
                paper = scholar.get_paper(paper_id)
                return [paper]
            except Exception:
                pass

        # Treat as keyword search
        # Handle comma-separated keywords
        keywords = [k.strip() for k in query.split(",")]
        all_papers: list[Paper] = []
        seen_ids: set[str] = set()

        for keyword in keywords:
            keyword = keyword.strip()
            if not keyword:
                continue
            results = scholar.search(keyword, limit=10)
            for paper in results:
                if paper.paper_id not in seen_ids:
                    seen_ids.add(paper.paper_id)
                    all_papers.append(paper)

        return all_papers


def _save_analyses(analyses: list[PaperAnalysis], output_dir: Path) -> None:
    """Save analysis results to JSON files."""
    analyses_dir = output_dir / "analyses"
    analyses_dir.mkdir(parents=True, exist_ok=True)

    for analysis in analyses:
        safe_id = analysis.paper.paper_id.replace("/", "_").replace(":", "_")
        path = analyses_dir / f"{safe_id}.json"
        path.write_text(
            analysis.model_dump_json(indent=2),
            encoding="utf-8",
        )


def _load_analyses(output_dir: Path) -> list[PaperAnalysis]:
    """Load analysis results from JSON files."""
    analyses_dir = output_dir / "analyses"
    if not analyses_dir.exists():
        return []

    analyses = []
    for path in sorted(analyses_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            analyses.append(PaperAnalysis.model_validate(data))
        except Exception as e:
            logger.warning(f"Failed to load analysis {path.name}: {e}")

    return analyses
