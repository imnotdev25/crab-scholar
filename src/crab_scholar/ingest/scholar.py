"""Semantic Scholar API client.

Provides search, paper lookup, citations, and references via the
Semantic Scholar Academic Graph API (https://api.semanticscholar.org).
"""

import logging
import time
from typing import Any

import httpx

from crab_scholar.models import Paper

logger = logging.getLogger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1"

# Fields to request from Semantic Scholar
PAPER_FIELDS = (
    "paperId,title,abstract,year,venue,citationCount,referenceCount,"
    "authors,externalIds,openAccessPdf,url"
)


class ScholarClient:
    """Client for the Semantic Scholar Academic Graph API."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
        rate_limit_wait: float = 1.0,
    ):
        headers = {"Accept": "application/json"}
        if api_key:
            headers["x-api-key"] = api_key
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
        )
        self._rate_limit_wait = rate_limit_wait
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Simple rate limiter — wait between requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._rate_limit_wait:
            time.sleep(self._rate_limit_wait - elapsed)
        self._last_request_time = time.monotonic()

    def _get(self, path: str, params: dict | None = None) -> dict:
        """Make a GET request with rate limiting and error handling."""
        self._rate_limit()
        try:
            response = self._client.get(path, params=params)
            if response.status_code == 429:
                wait = float(response.headers.get("Retry-After", "5"))
                logger.warning(f"Scholar API rate limited, waiting {wait}s")
                time.sleep(wait)
                response = self._client.get(path, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Scholar API error: {e.response.status_code} — {e.response.text[:200]}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Scholar API request failed: {e}")
            raise

    def _parse_paper(self, data: dict, depth: int = 0) -> Paper:
        """Convert Scholar API response to Paper model."""
        authors = []
        for a in data.get("authors", []):
            name = a.get("name", "")
            if name:
                authors.append(name)

        external_ids = data.get("externalIds", {}) or {}
        doi = external_ids.get("DOI")

        open_access = data.get("openAccessPdf") or {}
        oa_url = open_access.get("url") if isinstance(open_access, dict) else None

        return Paper(
            paper_id=data.get("paperId", ""),
            title=data.get("title", "Untitled"),
            authors=authors,
            abstract=data.get("abstract", "") or "",
            year=data.get("year"),
            doi=doi,
            url=data.get("url", ""),
            venue=data.get("venue", "") or "",
            citation_count=data.get("citationCount", 0) or 0,
            reference_count=data.get("referenceCount", 0) or 0,
            source="scholar",
            open_access_pdf=oa_url,
            crawl_depth=depth,
        )

    def search(self, query: str, limit: int = 10) -> list[Paper]:
        """Search for papers by keyword query.

        Args:
            query: Search keywords (comma-separated or natural language)
            limit: Max results to return (max 100)

        Returns:
            List of matching Papers
        """
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": PAPER_FIELDS,
        }
        data = self._get("/paper/search", params=params)
        papers = []
        for item in data.get("data", []):
            try:
                papers.append(self._parse_paper(item))
            except Exception as e:
                logger.warning(f"Failed to parse paper: {e}")
        return papers

    def get_paper(self, paper_id: str) -> Paper:
        """Get detailed paper info by Semantic Scholar ID, DOI, or URL.

        Args:
            paper_id: Semantic Scholar ID, DOI (DOI:xxx), URL, or arXiv ID (ARXIV:xxx)

        Returns:
            Paper with metadata
        """
        params = {"fields": PAPER_FIELDS}
        data = self._get(f"/paper/{paper_id}", params=params)
        return self._parse_paper(data)

    def get_references(self, paper_id: str, limit: int = 100) -> list[Paper]:
        """Get papers referenced by the given paper.

        Args:
            paper_id: Semantic Scholar paper ID
            limit: Max references to return

        Returns:
            List of referenced Papers
        """
        params = {"fields": PAPER_FIELDS, "limit": min(limit, 500)}
        data = self._get(f"/paper/{paper_id}/references", params=params)
        papers = []
        for item in data.get("data", []):
            cited = item.get("citedPaper", {})
            if cited and cited.get("paperId"):
                try:
                    papers.append(self._parse_paper(cited))
                except Exception as e:
                    logger.warning(f"Failed to parse reference: {e}")
        return papers

    def get_citations(self, paper_id: str, limit: int = 100) -> list[Paper]:
        """Get papers that cite the given paper.

        Args:
            paper_id: Semantic Scholar paper ID
            limit: Max citations to return

        Returns:
            List of citing Papers
        """
        params = {"fields": PAPER_FIELDS, "limit": min(limit, 500)}
        data = self._get(f"/paper/{paper_id}/citations", params=params)
        papers = []
        for item in data.get("data", []):
            citing = item.get("citingPaper", {})
            if citing and citing.get("paperId"):
                try:
                    papers.append(self._parse_paper(citing))
                except Exception as e:
                    logger.warning(f"Failed to parse citation: {e}")
        return papers

    def resolve_paper_id(self, query: str) -> str | None:
        """Resolve a DOI, URL, or title to a Semantic Scholar paper ID.

        Args:
            query: DOI, arXiv URL, Semantic Scholar URL, or paper title

        Returns:
            Semantic Scholar paper ID or None
        """
        # If it looks like a DOI
        if query.startswith("10.") or query.startswith("doi:"):
            doi = query.replace("doi:", "").strip()
            try:
                paper = self.get_paper(f"DOI:{doi}")
                return paper.paper_id
            except Exception:
                pass

        # If it looks like a URL
        if query.startswith("http"):
            try:
                paper = self.get_paper(f"URL:{query}")
                return paper.paper_id
            except Exception:
                pass

        # If it looks like an arXiv ID
        if "arxiv" in query.lower():
            arxiv_id = query.split("/")[-1].split("v")[0]
            try:
                paper = self.get_paper(f"ARXIV:{arxiv_id}")
                return paper.paper_id
            except Exception:
                pass

        # Fall back to search
        results = self.search(query, limit=1)
        if results:
            return results[0].paper_id

        return None

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
