"""BFS citation/reference crawler with configurable depth.

Crawls the citation graph starting from seed papers up to a
configurable depth. Deduplicates by paper_id and respects
a global max_papers cap.
"""

import logging

from crab_scholar.ingest.scholar import ScholarClient
from crab_scholar.models import Paper

logger = logging.getLogger(__name__)


def crawl(
    seed_papers: list[Paper],
    scholar: ScholarClient,
    depth: int = 3,
    max_papers: int = 50,
    direction: str = "references",
) -> list[Paper]:
    """BFS crawl through citation/reference graph.

    Args:
        seed_papers: Starting papers (depth 0)
        scholar: Semantic Scholar API client
        depth: Max crawl depth (1=seeds only, 2=seeds+refs, 3=+refs-of-refs)
        max_papers: Global cap on total papers collected
        direction: "references" (what paper cites), "citations" (what cites paper), or "both"

    Returns:
        All collected papers (deduped), sorted by crawl depth then citation count
    """
    if depth < 1:
        return seed_papers[:max_papers]

    seen_ids: set[str] = set()
    all_papers: list[Paper] = []

    # Initialize with seeds at depth 0
    current_frontier: list[Paper] = []
    for paper in seed_papers:
        if paper.paper_id and paper.paper_id not in seen_ids:
            paper.crawl_depth = 0
            seen_ids.add(paper.paper_id)
            all_papers.append(paper)
            current_frontier.append(paper)

    if len(all_papers) >= max_papers:
        return all_papers[:max_papers]

    # BFS through depth levels
    for current_depth in range(1, depth):
        if not current_frontier or len(all_papers) >= max_papers:
            break

        next_frontier: list[Paper] = []
        logger.info(
            f"Crawling depth {current_depth}/{depth - 1}: "
            f"{len(current_frontier)} papers to expand, "
            f"{len(all_papers)}/{max_papers} total"
        )

        for paper in current_frontier:
            if len(all_papers) >= max_papers:
                break

            new_papers: list[Paper] = []

            try:
                if direction in ("references", "both"):
                    refs = scholar.get_references(paper.paper_id, limit=20)
                    new_papers.extend(refs)

                if direction in ("citations", "both"):
                    cites = scholar.get_citations(paper.paper_id, limit=20)
                    new_papers.extend(cites)
            except Exception as e:
                logger.warning(
                    f"Failed to fetch {'refs/cites' if direction == 'both' else direction} "
                    f"for '{paper.title[:50]}': {e}"
                )
                continue

            for new_paper in new_papers:
                if len(all_papers) >= max_papers:
                    break
                if not new_paper.paper_id or new_paper.paper_id in seen_ids:
                    continue

                new_paper.crawl_depth = current_depth
                seen_ids.add(new_paper.paper_id)
                all_papers.append(new_paper)
                next_frontier.append(new_paper)

        current_frontier = next_frontier

    # Sort: seeds first, then by citation count within each depth
    all_papers.sort(key=lambda p: (p.crawl_depth, -(p.citation_count or 0)))

    logger.info(
        f"Crawl complete: {len(all_papers)} papers across "
        f"{max(p.crawl_depth for p in all_papers) + 1 if all_papers else 0} depth levels"
    )
    return all_papers
