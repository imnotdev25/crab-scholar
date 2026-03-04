# 🦀 ResearchClaw

Research paper analysis pipeline with citation crawling, pluggable LLM prompts, and knowledge graph building.

## Features

- **Multi-input**: Analyze papers by title, DOI, keywords, URL, local PDF, or raw text
- **Citation Crawling**: BFS traversal of references/citations via Semantic Scholar API (configurable depth, default 3)
- **5 Default Analysis Dimensions** (LLM Evaluation focus):
  1. **Paper Analysis** — overview, contributions, methodology
  2. **Dataset Crafting** — data creation, annotation, preprocessing
  3. **Evaluation Method** — benchmarks, baselines, evaluation setup
  4. **Metrics** — specific metrics, reported results
  5. **Statistical Tests** — significance tests, confidence intervals, rigor
- **Pluggable Prompts**: Add YAML files for custom dimensions, override defaults
- **Knowledge Graph**: NetworkX-based graph with paper/author/method/dataset/metric entities
- **Multi-Provider LLM**: Via LiteLLM — OpenAI, Anthropic, Ollama, vLLM, etc. with fallback chain
- **Export**: JSON, GraphML, GEXF, CSV

## Installation

```bash
uv sync
```

## Quick Start

```bash
# Initialize project config
uv run rclaw init

# Edit .env with your API key
nano .env

# Analyze a paper by title
uv run rclaw analyze "attention is all you need"

# Search by keywords
uv run rclaw analyze --keywords "LLM evaluation, benchmark contamination"

# Analyze a local PDF
uv run rclaw analyze --pdf paper.pdf

# Control crawl depth
uv run rclaw analyze "GPT-4 Technical Report" --depth 5

# Search without analyzing
uv run rclaw search "transformer evaluation"

# Build knowledge graph from results
uv run rclaw build

# Export graph
uv run rclaw export json
uv run rclaw export graphml
uv run rclaw export csv

# List analysis dimensions
uv run rclaw dimensions

# Show config
uv run rclaw info
```

## Configuration

Settings load from: CLI flags > env vars (`RCLAW_` prefix) > `.env` > `rclaw.yaml` > defaults.

```yaml
# rclaw.yaml
default_model: openai/gpt-4o-mini
fallback_models:
  - openai/gpt-3.5-turbo
  - anthropic/claude-3-haiku-20240307

citation_depth: 3
max_papers: 50
output: output
concurrency: 4
```

## Custom Prompts

Create YAML files in a custom directory:

```yaml
# my_prompts/bias_analysis.yaml
name: bias_analysis
display_name: "Bias Analysis"
description: "Analyze papers for bias in LLM evaluation"
system_message: "You are a bias analysis expert..."
extraction_prompt: |
  Analyze the paper for potential biases...
  Paper: {title}
  Text: {paper_text}
  ...
```

Then use: `uv run rclaw analyze "paper" --prompts-dir my_prompts/`

## Python API

```python
from research_claw.pipeline import run_pipeline
from research_claw.config import RClawConfig

config = RClawConfig(
    default_model="openai/gpt-4o-mini",
    citation_depth=3,
)

kg = run_pipeline(input_query="attention is all you need", config=config)
print(f"Entities: {kg.entity_count}, Relations: {kg.relation_count}")
```

## Architecture

```
Input (query/DOI/PDF/text)
    ↓
Scholar API → Resolve paper
    ↓
BFS Crawler → Expand citations/references (depth=N)
    ↓
Fetcher → Download PDFs, extract text
    ↓
Analyzer → Run pluggable dimensions (5 defaults)
    ↓
Graph Builder → Entities + Relations → NetworkX
    ↓
Export → JSON / GraphML / GEXF / CSV
```

## License

MIT
