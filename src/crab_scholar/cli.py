"""CrabScholar CLI.

Commands:
  rclaw analyze    — Analyze papers (by query, keywords, PDF, or text)
  rclaw build      — Build knowledge graph from analysis results
  rclaw view       — Interactive graph visualization in browser
  rclaw export     — Export graph to JSON/GraphML/GEXF/CSV
  rclaw search     — Search for papers via Semantic Scholar
  rclaw dimensions — List available analysis dimensions
  rclaw init       — Create a project config file
  rclaw info       — Show config and project info
"""

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from crab_scholar import __version__

console = Console()
app = typer.Typer(
    name="crab",
    help="🦀 CrabScholar — Research paper analysis pipeline with citation crawling and knowledge graphs",
    rich_markup_mode="rich",
    no_args_is_help=True,
)


class ExportFormat(str, Enum):
    json = "json"
    graphml = "graphml"
    gexf = "gexf"
    csv = "csv"


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
    )
    # Quiet noisy libraries
    for name in ("httpx", "httpcore", "litellm", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)


def version_callback(value: bool):
    if value:
        console.print(f"[bold cyan]CrabScholar[/] v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", help="Show version", callback=version_callback, is_eager=True,
    ),
):
    """🦀 CrabScholar — Research paper analysis pipeline."""
    pass


@app.command()
def analyze(
    query: Optional[str] = typer.Argument(None, help="Paper title, DOI, URL, or keywords"),
    keywords: Optional[str] = typer.Option(None, "--keywords", "-k", help="Comma-separated keywords for Scholar search"),
    pdf: Optional[Path] = typer.Option(None, "--pdf", "-p", help="Path to a local PDF file"),
    text: Optional[str] = typer.Option(None, "--text", "-t", help="Raw text to analyze"),
    depth: Optional[int] = typer.Option(None, "--depth", "-d", help="Citation crawl depth (default: 3)"),
    max_papers: Optional[int] = typer.Option(None, "--max-papers", "-m", help="Max papers to crawl"),
    dimensions: Optional[str] = typer.Option(None, "--dimensions", help="Comma-separated dimension names to use"),
    model: Optional[str] = typer.Option(None, "--model", help="Override default LLM model"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose logging"),
):
    """Analyze papers — search, crawl citations, extract insights, build graph.

    Examples:

      rclaw analyze "attention is all you need"

      rclaw analyze --keywords "LLM evaluation, benchmark contamination"

      rclaw analyze --pdf paper.pdf

      rclaw analyze --text "We propose a new method..."

      rclaw analyze "10.48550/arXiv.2303.08774" --depth 5
    """
    _setup_logging(verbose)

    from crab_scholar.config import CrabConfig
    from crab_scholar.pipeline import run_pipeline

    # Build config with overrides
    overrides = {}
    if depth:
        overrides["citation_depth"] = depth
    if max_papers:
        overrides["max_papers"] = max_papers
    if model:
        overrides["default_model"] = model
    if output:
        overrides["output_dir"] = output

    try:
        config = CrabConfig(**overrides)
    except Exception as e:
        console.print(f"[red]Config error:[/] {e}")
        raise typer.Exit(1)

    # Resolve input
    input_query = query or keywords
    pdf_path = pdf
    raw_text = text

    if not any([input_query, pdf_path, raw_text]):
        console.print("[red]Error:[/] Provide a query, --keywords, --pdf, or --text")
        raise typer.Exit(1)

    dim_list = [d.strip() for d in dimensions.split(",")] if dimensions else None

    console.print(Panel.fit(
        "[bold cyan]🦀 CrabScholar[/bold cyan]\n"
        f"Input: {input_query or pdf_path or 'raw text'}\n"
        f"Depth: {config.citation_depth} | Max papers: {config.max_papers}\n"
        f"Model: {config.default_model}\n"
        f"Output: {config.output_dir}",
        title="Analysis Pipeline",
    ))

    try:
        kg = run_pipeline(
            input_query=input_query,
            pdf_path=pdf_path,
            raw_text=raw_text,
            config=config,
        )
        console.print(f"\n[green]✓[/] Analysis complete!")
        console.print(f"  Entities: {kg.entity_count}")
        console.print(f"  Relations: {kg.relation_count}")
        console.print(f"  Output: {config.output_dir}")
    except Exception as e:
        console.print(f"\n[red]✗ Error:[/] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def build(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose logging"),
):
    """Build knowledge graph from analysis results."""
    _setup_logging(verbose)
    from crab_scholar.config import CrabConfig
    from crab_scholar.pipeline import run_build

    config = CrabConfig()
    out = output or config.output_dir

    try:
        kg = run_build(output_dir=out)
        console.print(f"[green]✓[/] Graph built: {kg.entity_count} entities, {kg.relation_count} relations")
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


@app.command()
def export(
    fmt: ExportFormat = typer.Argument(ExportFormat.json, help="Export format"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose logging"),
):
    """Export knowledge graph to JSON, GraphML, GEXF, or CSV."""
    _setup_logging(verbose)
    from crab_scholar.config import CrabConfig
    from crab_scholar.pipeline import run_export

    config = CrabConfig()
    out = output or config.output_dir

    try:
        path = run_export(fmt=fmt.value, output_dir=out)
        console.print(f"[green]✓[/] Exported ({fmt.value}) → {path}")
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search keywords"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose logging"),
):
    """Search for papers on Semantic Scholar."""
    _setup_logging(verbose)
    from crab_scholar.pipeline import run_search

    papers = run_search(query, limit=limit)

    if not papers:
        console.print("[yellow]No papers found.[/]")
        raise typer.Exit()

    table = Table(title=f"Search Results: '{query}'", show_lines=True)
    table.add_column("Title", style="bold", max_width=50)
    table.add_column("Authors", max_width=30)
    table.add_column("Year", justify="center", width=6)
    table.add_column("Citations", justify="right", width=10)
    table.add_column("ID", max_width=20)

    for p in papers:
        table.add_row(
            p.title[:50],
            p.short_authors,
            str(p.year or "—"),
            str(p.citation_count),
            p.paper_id[:20],
        )

    console.print(table)


@app.command()
def dimensions(
    prompts_dir: Optional[Path] = typer.Option(None, "--prompts-dir", help="Custom prompts directory"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose logging"),
):
    """List available analysis dimensions."""
    _setup_logging(verbose)
    from crab_scholar.analyze.dimensions import list_available_dimensions

    dims = list_available_dimensions(prompts_dir)

    table = Table(title="Analysis Dimensions", show_lines=True)
    table.add_column("Name", style="bold cyan")
    table.add_column("Display Name")
    table.add_column("Description", max_width=50)
    table.add_column("Source", justify="center")

    for d in dims:
        source_style = "green" if d["source"] == "bundled" else "yellow"
        table.add_row(
            d["name"],
            d["display_name"],
            d["description"][:50],
            f"[{source_style}]{d['source']}[/{source_style}]",
        )

    console.print(table)


@app.command()
def view(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory with graph.json"),
    top_n: Optional[int] = typer.Option(None, "--top", "-n", help="Show only top N nodes by degree"),
    min_confidence: Optional[float] = typer.Option(None, "--min-confidence", help="Minimum confidence threshold"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser automatically"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose logging"),
):
    """Open interactive knowledge graph viewer in browser.

    Requires a graph.json file (created by `rclaw build` or `rclaw analyze`).
    """
    _setup_logging(verbose)
    from crab_scholar.config import CrabConfig
    from crab_scholar.graph.knowledge_graph import KnowledgeGraph
    from crab_scholar.visualize import generate_view

    config = CrabConfig()
    out = output or config.output_dir
    graph_path = out / "graph.json"

    if not graph_path.exists():
        console.print(f"[red]Error:[/] No graph found at {graph_path}")
        console.print("Run [bold]rclaw analyze[/] or [bold]rclaw build[/] first.")
        raise typer.Exit(1)

    try:
        kg = KnowledgeGraph.load(graph_path)
        console.print(f"Loaded graph: {kg.entity_count} entities, {kg.relation_count} relations")

        html_path = out / "graph.html"
        generate_view(
            kg,
            output_path=html_path,
            open_browser=not no_browser,
            top_n=top_n,
            min_confidence=min_confidence,
        )
        console.print(f"[green]✓[/] Viewer generated → {html_path}")
    except ImportError:
        console.print("[red]Error:[/] pyvis is required. Install with: [bold]uv add pyvis[/]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def init(
    verbose: bool = typer.Option(False, "--verbose", help="Verbose logging"),
):
    """Initialize a CrabScholar project (creates rclaw.yaml and .env)."""
    _setup_logging(verbose)

    yaml_path = Path("rclaw.yaml")
    env_path = Path(".env")

    if yaml_path.exists():
        console.print(f"[yellow]rclaw.yaml already exists, skipping[/]")
    else:
        yaml_path.write_text(
            "# CrabScholar Configuration\n"
            "# See: https://github.com/imnotdev25/research-claw\n\n"
            "# LLM Settings\n"
            "# default_model: openai/gpt-4o-mini\n"
            "# fallback_models:\n"
            "#   - openai/gpt-3.5-turbo\n"
            "#   - anthropic/claude-3-haiku-20240307\n\n"
            "# base_url: http://localhost:8000/v1\n\n"
            "# Citation Crawling\n"
            "citation_depth: 3\n"
            "max_papers: 50\n\n"
            "# Output\n"
            "output: output\n"
            "concurrency: 4\n",
            encoding="utf-8",
        )
        console.print(f"[green]✓[/] Created {yaml_path}")

    if env_path.exists():
        console.print(f"[yellow].env already exists, skipping[/]")
    else:
        env_path.write_text(
            "# CrabScholar API Keys\n"
            "CRAB_API_KEY=sk-...\n"
            "# CRAB_SCHOLAR_API_KEY=\n",
            encoding="utf-8",
        )
        console.print(f"[green]✓[/] Created {env_path}")

    console.print("\n[bold]Next steps:[/]")
    console.print("  1. Edit .env and add your API key")
    console.print("  2. Run: rclaw analyze \"your paper title\"")


@app.command()
def info(
    verbose: bool = typer.Option(False, "--verbose", help="Verbose logging"),
):
    """Show current configuration and project info."""
    _setup_logging(verbose)
    from crab_scholar.config import CrabConfig

    try:
        config = CrabConfig()
    except Exception as e:
        console.print(f"[red]Config error:[/] {e}")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[bold cyan]CrabScholar[/] v{__version__}\n\n"
        f"Model: {config.default_model}\n"
        f"Fallbacks: {', '.join(config.fallback_models) or 'none'}\n"
        f"Base URL: {config.base_url or 'default'}\n"
        f"Depth: {config.citation_depth}\n"
        f"Max papers: {config.max_papers}\n"
        f"Output: {config.output_dir}\n"
        f"Concurrency: {config.concurrency}\n"
        f"Custom prompts: {config.prompts_dir or 'none'}",
        title="Configuration",
    ))


if __name__ == "__main__":
    app()
