"""Analysis orchestrator — runs pluggable dimensions against papers.

Sends each paper through every configured analysis dimension,
extracts entities and relations from the LLM output, and
collects results into PaperAnalysis objects.
"""

import asyncio
import json
import logging
from typing import Any

from crab_scholar.llm_client import LLMClient
from crab_scholar.models import (
    AnalysisDimension,
    DimensionResult,
    ExtractedEntity,
    ExtractedRelation,
    Paper,
    PaperAnalysis,
)

logger = logging.getLogger(__name__)


def analyze_paper(
    paper: Paper,
    dimensions: list[AnalysisDimension],
    llm: LLMClient,
    concurrency: int = 4,
) -> PaperAnalysis:
    """Analyze a single paper across all dimensions.

    Args:
        paper: Paper with text content
        dimensions: Analysis dimensions to apply
        llm: LLM client for extraction
        concurrency: Max concurrent LLM calls

    Returns:
        PaperAnalysis with results for each dimension
    """
    return asyncio.run(
        _aanalyze_paper(paper, dimensions, llm, concurrency)
    )


async def _aanalyze_paper(
    paper: Paper,
    dimensions: list[AnalysisDimension],
    llm: LLMClient,
    concurrency: int = 4,
) -> PaperAnalysis:
    """Async implementation of paper analysis."""
    semaphore = asyncio.Semaphore(concurrency)
    analysis = PaperAnalysis(paper=paper, model_used=llm.model)

    async def _run_dimension(dim: AnalysisDimension) -> tuple[str, DimensionResult]:
        async with semaphore:
            try:
                result = await _aanalyze_dimension(paper, dim, llm)
                return dim.name, result
            except Exception as e:
                logger.warning(f"Dimension '{dim.name}' failed for '{paper.title[:50]}': {e}")
                return dim.name, DimensionResult(
                    dimension_name=dim.name,
                    paper_id=paper.paper_id,
                    content=f"Error: {e}",
                )

    tasks = [_run_dimension(dim) for dim in dimensions]
    results = await asyncio.gather(*tasks)

    for name, result in results:
        analysis.dimensions[name] = result

    analysis.cost_usd = llm.total_cost
    return analysis


async def _aanalyze_dimension(
    paper: Paper,
    dimension: AnalysisDimension,
    llm: LLMClient,
) -> DimensionResult:
    """Run a single analysis dimension against a paper."""
    prompt = dimension.render_prompt(paper)
    system_msg = dimension.system_message or None

    response_text = await llm.acall(prompt, system_message=system_msg)
    entities, relations = _parse_extraction_response(response_text, paper.paper_id)

    return DimensionResult(
        dimension_name=dimension.name,
        paper_id=paper.paper_id,
        content=response_text,
        entities=entities,
        relations=relations,
    )


def analyze_all(
    papers: list[Paper],
    dimensions: list[AnalysisDimension],
    llm: LLMClient,
    concurrency: int = 4,
    max_cost: float | None = None,
) -> list[PaperAnalysis]:
    """Analyze multiple papers across all dimensions.

    Args:
        papers: Papers with text content
        dimensions: Analysis dimensions to apply
        llm: LLM client
        concurrency: Max concurrent calls
        max_cost: Optional cost cap in USD

    Returns:
        List of PaperAnalysis results
    """
    return asyncio.run(
        _aanalyze_all(papers, dimensions, llm, concurrency, max_cost)
    )


async def _aanalyze_all(
    papers: list[Paper],
    dimensions: list[AnalysisDimension],
    llm: LLMClient,
    concurrency: int = 4,
    max_cost: float | None = None,
) -> list[PaperAnalysis]:
    """Async implementation for multi-paper analysis."""
    semaphore = asyncio.Semaphore(concurrency)
    results: list[PaperAnalysis] = []

    for i, paper in enumerate(papers, 1):
        if max_cost and llm.total_cost >= max_cost:
            logger.warning(
                f"Cost cap reached (${llm.total_cost:.4f} >= ${max_cost:.4f}), "
                f"stopping after {i - 1}/{len(papers)} papers"
            )
            break

        logger.info(
            f"[{i}/{len(papers)}] Analyzing: {paper.title[:60]}... "
            f"(depth={paper.crawl_depth}, ${llm.total_cost:.4f} spent)"
        )

        analysis = await _aanalyze_paper(paper, dimensions, llm, concurrency)
        results.append(analysis)

        # Save intermediate results
        logger.debug(
            f"  → {sum(len(r.entities) for r in analysis.dimensions.values())} entities, "
            f"{sum(len(r.relations) for r in analysis.dimensions.values())} relations"
        )

    logger.info(
        f"Analysis complete: {len(results)} papers, "
        f"${llm.total_cost:.4f} total cost"
    )
    return results


def _parse_extraction_response(
    text: str, paper_id: str
) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
    """Parse entities and relations from LLM JSON response."""
    entities: list[ExtractedEntity] = []
    relations: list[ExtractedRelation] = []

    try:
        from crab_scholar.llm_client import parse_llm_json
        data = parse_llm_json(text)
    except (ValueError, json.JSONDecodeError):
        logger.debug("Could not parse JSON from dimension response, skipping extraction")
        return entities, relations

    # Parse entities
    for item in data.get("entities", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name", "").strip()
        if not name:
            continue
        entities.append(ExtractedEntity(
            name=name,
            entity_type=item.get("entity_type", "CONCEPT").upper(),
            confidence=min(1.0, max(0.0, float(item.get("confidence", 0.5)))),
            context=item.get("context", ""),
            source_paper_id=paper_id,
            attributes={
                k: v for k, v in item.items()
                if k not in {"name", "entity_type", "confidence", "context"}
            },
        ))

    # Parse relations
    for item in data.get("relations", []):
        if not isinstance(item, dict):
            continue
        source = item.get("source_entity", "").strip()
        target = item.get("target_entity", "").strip()
        rel_type = item.get("relation_type", "").strip()
        if not (source and target and rel_type):
            continue
        relations.append(ExtractedRelation(
            relation_type=rel_type.upper(),
            source_entity=source,
            target_entity=target,
            confidence=min(1.0, max(0.0, float(item.get("confidence", 0.5)))),
            evidence=item.get("evidence", ""),
            source_paper_id=paper_id,
        ))

    return entities, relations
