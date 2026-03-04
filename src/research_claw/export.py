"""Export knowledge graph to various formats.

Supports JSON, GraphML, GEXF, and CSV export formats.
"""

import csv
import json
import logging
from pathlib import Path

import networkx as nx

from research_claw.graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


def export_graph(
    kg: KnowledgeGraph,
    fmt: str = "json",
    output_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Export the knowledge graph to a specified format.

    Args:
        kg: KnowledgeGraph to export
        fmt: Format — "json", "graphml", "gexf", or "csv"
        output_path: Explicit output path (overrides output_dir)
        output_dir: Directory for default-named output

    Returns:
        Path to the exported file/directory
    """
    fmt = fmt.lower().strip()
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        path = output_path or (output_dir / "graph.json" if output_dir else Path("graph.json"))
        _export_json(kg, path)
    elif fmt == "graphml":
        path = output_path or (output_dir / "graph.graphml" if output_dir else Path("graph.graphml"))
        _export_graphml(kg, path)
    elif fmt == "gexf":
        path = output_path or (output_dir / "graph.gexf" if output_dir else Path("graph.gexf"))
        _export_gexf(kg, path)
    elif fmt == "csv":
        path = output_path or (output_dir / "csv" if output_dir else Path("csv"))
        _export_csv(kg, path)
    else:
        raise ValueError(f"Unknown export format: {fmt!r}. Choose json, graphml, gexf, or csv.")

    logger.info(f"Exported graph ({fmt}) → {path}")
    return path


def _export_json(kg: KnowledgeGraph, path: Path) -> None:
    """Export as JSON."""
    kg.save(path)


def _export_graphml(kg: KnowledgeGraph, path: Path) -> None:
    """Export as GraphML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert complex attributes to strings for GraphML compatibility
    g = _flatten_graph(kg.graph)
    nx.write_graphml(g, str(path))


def _export_gexf(kg: KnowledgeGraph, path: Path) -> None:
    """Export as GEXF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    g = _flatten_graph(kg.graph)
    nx.write_gexf(g, str(path))


def _export_csv(kg: KnowledgeGraph, directory: Path) -> None:
    """Export as CSV (entities.csv + relations.csv)."""
    directory.mkdir(parents=True, exist_ok=True)

    # Entities
    entities_path = directory / "entities.csv"
    with open(entities_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "entity_type", "confidence", "source_papers"])
        for node_id, data in kg.graph.nodes(data=True):
            writer.writerow([
                node_id,
                data.get("name", ""),
                data.get("entity_type", ""),
                data.get("confidence", 0.5),
                "|".join(data.get("source_papers", [])),
            ])

    # Relations
    relations_path = directory / "relations.csv"
    with open(relations_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "relation_type", "confidence", "evidence", "source_paper"])
        for source, target, data in kg.graph.edges(data=True):
            writer.writerow([
                source,
                target,
                data.get("relation_type", ""),
                data.get("confidence", 0.5),
                data.get("evidence", ""),
                data.get("source_paper", ""),
            ])


def _flatten_graph(g: nx.MultiDiGraph) -> nx.DiGraph:
    """Convert MultiDiGraph to DiGraph with string attributes (for GraphML/GEXF)."""
    flat = nx.DiGraph()
    for node_id, data in g.nodes(data=True):
        attrs = {}
        for k, v in data.items():
            if isinstance(v, (list, dict)):
                attrs[k] = json.dumps(v, default=str)
            else:
                attrs[k] = str(v) if v is not None else ""
        flat.add_node(node_id, **attrs)

    for source, target, data in g.edges(data=True):
        attrs = {}
        for k, v in data.items():
            if isinstance(v, (list, dict)):
                attrs[k] = json.dumps(v, default=str)
            else:
                attrs[k] = str(v) if v is not None else ""
        if flat.has_edge(source, target):
            # Merge edge data for non-multi graph
            existing = flat.edges[source, target]
            for k, v in attrs.items():
                if k in existing and existing[k] != v:
                    existing[k] = existing[k] + " | " + v
                else:
                    existing[k] = v
        else:
            flat.add_edge(source, target, **attrs)

    return flat
