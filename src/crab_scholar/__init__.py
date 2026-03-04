"""CrabScholar: Research paper analysis pipeline.

Ingest papers via text, keywords, or URLs. Crawl citations/references
to configurable depth. Analyze with pluggable LLM prompts. Build
knowledge graphs connecting papers, methods, datasets, and findings.
"""

from importlib.metadata import version as _get_version

try:
    __version__ = _get_version("research-claw")
except Exception:
    __version__ = "0.1.0"

__all__ = ["__version__"]
