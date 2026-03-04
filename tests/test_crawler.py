"""Tests for the BFS citation crawler."""

from crab_scholar.ingest.crawler import crawl
from crab_scholar.models import Paper


class FakeScholar:
    """Mock Scholar client for testing."""

    def __init__(self, reference_graph: dict[str, list[Paper]] | None = None):
        self.reference_graph = reference_graph or {}
        self.call_count = 0

    def get_references(self, paper_id: str, limit: int = 100) -> list[Paper]:
        self.call_count += 1
        return self.reference_graph.get(paper_id, [])

    def get_citations(self, paper_id: str, limit: int = 100) -> list[Paper]:
        self.call_count += 1
        return []


def _make_paper(pid: str, title: str = "Test") -> Paper:
    return Paper(paper_id=pid, title=f"{title} {pid}")


class TestCrawl:
    def test_depth_1_returns_seeds(self):
        seeds = [_make_paper("A")]
        result = crawl(seeds, FakeScholar(), depth=1, max_papers=50)
        assert len(result) == 1
        assert result[0].paper_id == "A"

    def test_depth_2_expands_references(self):
        ref_graph = {
            "A": [_make_paper("B"), _make_paper("C")],
        }
        seeds = [_make_paper("A")]
        result = crawl(seeds, FakeScholar(ref_graph), depth=2, max_papers=50)
        ids = {p.paper_id for p in result}
        assert ids == {"A", "B", "C"}

    def test_depth_3(self):
        ref_graph = {
            "A": [_make_paper("B")],
            "B": [_make_paper("C")],
        }
        seeds = [_make_paper("A")]
        result = crawl(seeds, FakeScholar(ref_graph), depth=3, max_papers=50)
        ids = {p.paper_id for p in result}
        assert ids == {"A", "B", "C"}

    def test_dedup(self):
        ref_graph = {
            "A": [_make_paper("B"), _make_paper("C")],
            "B": [_make_paper("C")],  # C already seen via A
        }
        seeds = [_make_paper("A")]
        result = crawl(seeds, FakeScholar(ref_graph), depth=3, max_papers=50)
        ids = [p.paper_id for p in result]
        assert ids.count("C") == 1

    def test_max_papers_cap(self):
        ref_graph = {
            "A": [_make_paper(f"R{i}") for i in range(20)],
        }
        seeds = [_make_paper("A")]
        result = crawl(seeds, FakeScholar(ref_graph), depth=2, max_papers=5)
        assert len(result) <= 5

    def test_crawl_depth_tracking(self):
        ref_graph = {
            "A": [_make_paper("B")],
            "B": [_make_paper("C")],
        }
        seeds = [_make_paper("A")]
        result = crawl(seeds, FakeScholar(ref_graph), depth=3, max_papers=50)
        depths = {p.paper_id: p.crawl_depth for p in result}
        assert depths["A"] == 0
        assert depths["B"] == 1
        assert depths["C"] == 2

    def test_empty_seeds(self):
        result = crawl([], FakeScholar(), depth=3, max_papers=50)
        assert result == []
