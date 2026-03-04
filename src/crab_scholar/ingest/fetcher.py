"""Paper content fetcher — downloads and extracts text from papers.

Tries open-access PDF first, falls back to abstract + metadata.
Uses Kreuzberg for PDF text extraction (same engine as sift-kg).
"""

import hashlib
import logging
import tempfile
from pathlib import Path

import httpx

from crab_scholar.models import Paper

logger = logging.getLogger(__name__)


def fetch_paper_text(paper: Paper, cache_dir: Path | None = None) -> str:
    """Fetch full text for a paper.

    Tries strategies in order:
    1. Paper already has full_text → return it
    2. Open-access PDF URL → download + extract via Kreuzberg
    3. Fall back to abstract + metadata

    Args:
        paper: Paper with metadata (needs open_access_pdf or abstract)
        cache_dir: Optional directory to cache downloaded PDFs

    Returns:
        Extracted text content
    """
    # Already have text
    if paper.full_text:
        return paper.full_text

    # Try open-access PDF
    if paper.open_access_pdf:
        try:
            text = _fetch_pdf_text(paper.open_access_pdf, cache_dir)
            if text and len(text.strip()) > 100:
                return text
        except Exception as e:
            logger.warning(f"PDF extraction failed for '{paper.title[:50]}': {e}")

    # Fall back to abstract + metadata
    return _build_metadata_text(paper)


def fetch_paper_text_from_path(path: Path) -> str:
    """Extract text from a local PDF file.

    Args:
        path: Path to the PDF file

    Returns:
        Extracted text content
    """
    return _extract_pdf(path)


def fetch_paper_text_from_url(url: str, cache_dir: Path | None = None) -> str:
    """Fetch and extract text from a URL (PDF or HTML).

    Args:
        url: URL to fetch
        cache_dir: Optional directory to cache downloads

    Returns:
        Extracted text content
    """
    if url.lower().endswith(".pdf") or "pdf" in url.lower():
        return _fetch_pdf_text(url, cache_dir)
    return _fetch_html_text(url)


def _fetch_pdf_text(url: str, cache_dir: Path | None = None) -> str:
    """Download a PDF and extract text."""
    # Check cache
    if cache_dir:
        cache_path = cache_dir / f"{_url_hash(url)}.pdf"
        if cache_path.exists():
            logger.debug(f"Using cached PDF: {cache_path}")
            return _extract_pdf(cache_path)

    logger.info(f"Downloading PDF: {url[:80]}...")
    response = httpx.get(
        url,
        follow_redirects=True,
        timeout=60.0,
        headers={"User-Agent": "CrabScholar/0.1 (academic research tool)"},
    )
    response.raise_for_status()

    # Save to cache or temp
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        save_path = cache_dir / f"{_url_hash(url)}.pdf"
        save_path.write_bytes(response.content)
        return _extract_pdf(save_path)
    else:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(response.content)
            return _extract_pdf(Path(f.name))


def _extract_pdf(path: Path) -> str:
    """Extract text from a PDF using Kreuzberg."""
    try:
        import asyncio

        from kreuzberg import extract_file

        result = asyncio.run(extract_file(path))
        return result.content
    except ImportError:
        logger.warning("Kreuzberg not available, trying pdfplumber")
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return "\n\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        except ImportError:
            logger.error("Neither kreuzberg nor pdfplumber available for PDF extraction")
            return ""


def _fetch_html_text(url: str) -> str:
    """Fetch a URL and extract text from HTML."""
    response = httpx.get(
        url,
        follow_redirects=True,
        timeout=30.0,
        headers={"User-Agent": "CrabScholar/0.1 (academic research tool)"},
    )
    response.raise_for_status()

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except ImportError:
        # Simple fallback: strip tags
        import re
        text = re.sub(r"<[^>]+>", " ", response.text)
        return re.sub(r"\s+", " ", text).strip()


def _build_metadata_text(paper: Paper) -> str:
    """Build text from paper metadata when full text is unavailable."""
    parts = [f"Title: {paper.title}"]
    if paper.authors:
        parts.append(f"Authors: {', '.join(paper.authors)}")
    if paper.year:
        parts.append(f"Year: {paper.year}")
    if paper.venue:
        parts.append(f"Venue: {paper.venue}")
    if paper.abstract:
        parts.append(f"\nAbstract:\n{paper.abstract}")
    return "\n".join(parts)


def _url_hash(url: str) -> str:
    """Generate a short hash for a URL (for caching)."""
    return hashlib.md5(url.encode()).hexdigest()[:12]
