"""Build knowledge graph from paper analysis results.

Creates nodes for papers, authors, and extracted entities.
Creates edges for citations, authorship, and extracted relations.
Deduplicates entities across papers by normalized name + type.
"""

import logging
from typing import Any

from unidecode import unidecode

from crab_scholar.graph.knowledge_graph import KnowledgeGraph
from crab_scholar.models import PaperAnalysis, Paper

logger = logging.getLogger(__name__)


def _make_entity_id(name: str, entity_type: str) -> str:
    """Generate a stable entity ID from name + type."""
    normalized = unidecode(name).lower().strip()
    normalized = normalized.replace(" ", "_").replace("-", "_")
    # Remove non-alphanumeric except underscores
    normalized = "".join(c for c in normalized if c.isalnum() or c == "_")
    return f"{entity_type.lower()}:{normalized}"


def build_graph(
    analyses: list[PaperAnalysis],
    papers: list[Paper] | None = None,
) -> KnowledgeGraph:
    """Build knowledge graph from analysis results.

    Args:
        analyses: List of paper analyses with extracted entities/relations
        papers: Optional full paper list (includes uncrawled papers for citation edges)

    Returns:
        Populated KnowledgeGraph
    """
    kg = KnowledgeGraph()

    # Track entity name normalization for dedup
    name_to_id: dict[str, str] = {}

    # Step 1: Add paper nodes
    all_papers: dict[str, Paper] = {}
    if papers:
        for p in papers:
            all_papers[p.paper_id] = p
    for a in analyses:
        all_papers[a.paper.paper_id] = a.paper

    for paper in all_papers.values():
        paper_entity_id = f"paper:{paper.paper_id}"
        kg.add_entity(
            entity_id=paper_entity_id,
            entity_type="PAPER",
            name=paper.title,
            confidence=1.0,
            source_papers=[paper.paper_id],
            year=paper.year,
            venue=paper.venue,
            citation_count=paper.citation_count,
            doi=paper.doi,
            crawl_depth=paper.crawl_depth,
        )

        # Add author nodes and edges
        for author_name in paper.authors:
            author_id = _make_entity_id(author_name, "AUTHOR")
            kg.add_entity(
                entity_id=author_id,
                entity_type="AUTHOR",
                name=author_name,
                confidence=1.0,
                source_papers=[paper.paper_id],
            )
            kg.add_relation(
                source_id=paper_entity_id,
                target_id=author_id,
                relation_type="AUTHORED_BY",
                confidence=1.0,
                source_paper=paper.paper_id,
            )
            name_to_id[author_name.lower()] = author_id

        name_to_id[paper.title.lower()] = paper_entity_id

    # Step 2: Add citation edges between papers
    # We infer these from the crawl structure — if paper B is a reference of paper A,
    # then A CITES B
    if papers:
        _add_citation_edges(kg, papers)

    # Step 3: Add extracted entities and relations from analyses
    for analysis in analyses:
        paper_id = analysis.paper.paper_id
        paper_entity_id = f"paper:{paper_id}"

        for dim_name, dim_result in analysis.dimensions.items():
            # Add entities from this dimension
            for entity in dim_result.entities:
                entity_id = _make_entity_id(entity.name, entity.entity_type)
                kg.add_entity(
                    entity_id=entity_id,
                    entity_type=entity.entity_type,
                    name=entity.name,
                    confidence=entity.confidence,
                    source_papers=[paper_id],
                    dimension=dim_name,
                    **entity.attributes,
                )
                name_to_id[entity.name.lower()] = entity_id

                # Link entity to its source paper
                kg.add_relation(
                    source_id=paper_entity_id,
                    target_id=entity_id,
                    relation_type="MENTIONS",
                    confidence=entity.confidence,
                    source_paper=paper_id,
                )

            # Add relations from this dimension
            for rel in dim_result.relations:
                src_id = _resolve_entity(rel.source_entity, name_to_id)
                tgt_id = _resolve_entity(rel.target_entity, name_to_id)

                if src_id and tgt_id:
                    kg.add_relation(
                        source_id=src_id,
                        target_id=tgt_id,
                        relation_type=rel.relation_type,
                        confidence=rel.confidence,
                        evidence=rel.evidence,
                        source_paper=paper_id,
                    )

    logger.info(
        f"Graph built: {kg.entity_count} entities, {kg.relation_count} relations "
        f"from {len(analyses)} papers"
    )
    return kg


def _resolve_entity(name: str, name_to_id: dict[str, str]) -> str | None:
    """Resolve an entity name to its graph ID."""
    lower = name.lower().strip()
    if lower in name_to_id:
        return name_to_id[lower]
    # Try case-insensitive partial match
    for key, eid in name_to_id.items():
        if lower in key or key in lower:
            return eid
    return None


def _add_citation_edges(kg: KnowledgeGraph, papers: list[Paper]) -> None:
    """Add CITES edges between papers based on crawl structure.

    Papers at depth N+1 are references found by expanding papers at depth N.
    We use Semantic Scholar reference data to create citation edges.
    """
    paper_ids = {p.paper_id for p in papers if p.paper_id}
    for paper in papers:
        paper_entity_id = f"paper:{paper.paper_id}"
        # If this paper was found as a reference of another paper at lower depth,
        # the citation relationship is implicit in the crawl structure.
        # We rely on the graph builder's explicit relation extraction for now.
        # Citation edges will be added when we can query reference lists.
        pass

    logger.debug(f"Citation edge pass: {len(paper_ids)} papers in graph")
