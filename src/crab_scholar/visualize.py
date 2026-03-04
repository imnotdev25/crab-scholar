"""Interactive knowledge graph visualization using pyvis.

Generates a standalone HTML file with a force-directed graph layout.
Entities colored by type, edges colored by relation type, with
interactive search, type toggles, color pickers, and detail sidebar.
"""

import json
import logging
import webbrowser
from pathlib import Path
from string import Template

from crab_scholar.graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# Node color palette — auto-assigned to entity types
NODE_PALETTE = [
    "#42A5F5",  # blue
    "#66BB6A",  # green
    "#FFA726",  # orange
    "#AB47BC",  # purple
    "#26C6DA",  # cyan
    "#EF5350",  # red
    "#FFEE58",  # yellow
    "#EC407A",  # pink
    "#9CCC65",  # lime
    "#FF7043",  # deep orange
    "#7E57C2",  # deep purple
    "#29B6F6",  # light blue
    "#8D6E63",  # brown
    "#78909C",  # gray
]

# Semantic entity colors — research-specific types
SEMANTIC_ENTITY_COLORS = {
    "PAPER":      "#42A5F5",  # blue — papers
    "AUTHOR":     "#66BB6A",  # green — people
    "METHOD":     "#FFA726",  # orange — techniques
    "DATASET":    "#AB47BC",  # purple — data
    "METRIC":     "#26C6DA",  # cyan — measurements
    "BENCHMARK":  "#EC407A",  # pink — benchmarks
    "FRAMEWORK":  "#9CCC65",  # lime — software
    "TASK":       "#FFD54F",  # amber — tasks
    "MODEL":      "#FF7043",  # deep orange — models
    "FINDING":    "#78909C",  # gray — results
}

# Edge color palette
EDGE_PALETTE = [
    "#4CAF50", "#FF7043", "#42A5F5", "#AB47BC",
    "#26A69A", "#EC407A", "#FFA726", "#66BB6A",
    "#7E57C2", "#29B6F6", "#EF5350", "#8D6E63",
]

# Semantic edge colors
SEMANTIC_EDGE_COLORS = {
    "CITES":           "#42A5F5",  # blue — citation
    "AUTHORED_BY":     "#66BB6A",  # green — authorship
    "USES_DATASET":    "#AB47BC",  # purple — data use
    "USES_METHOD":     "#FFA726",  # orange — methodology
    "EVALUATES_WITH":  "#26C6DA",  # cyan — metrics
    "PROPOSES":        "#EC407A",  # pink — proposal
    "EXTENDS":         "#7E57C2",  # deep purple — builds on
    "COMPARES_TO":     "#FF7043",  # deep orange — comparison
    "ACHIEVES":        "#9CCC65",  # lime — results
    "PART_OF":         "#78909C",  # gray — composition
}


def _color_for_entity(entity_type: str, color_map: dict[str, str]) -> str:
    if entity_type in color_map:
        return color_map[entity_type]
    if entity_type in SEMANTIC_ENTITY_COLORS:
        color_map[entity_type] = SEMANTIC_ENTITY_COLORS[entity_type]
    else:
        idx = len(color_map) % len(NODE_PALETTE)
        color_map[entity_type] = NODE_PALETTE[idx]
    return color_map[entity_type]


def _color_for_relation(rel_type: str, color_map: dict[str, str]) -> str:
    if rel_type in color_map:
        return color_map[rel_type]
    if rel_type in SEMANTIC_EDGE_COLORS:
        color_map[rel_type] = SEMANTIC_EDGE_COLORS[rel_type]
    else:
        idx = len(color_map) % len(EDGE_PALETTE)
        color_map[rel_type] = EDGE_PALETTE[idx]
    return color_map[rel_type]


def generate_view(
    kg: KnowledgeGraph,
    output_path: Path,
    open_browser: bool = True,
    top_n: int | None = None,
    min_confidence: float | None = None,
) -> Path:
    """Generate an interactive HTML visualization of the knowledge graph.

    Args:
        kg: Knowledge graph to visualize
        output_path: Path for output HTML file
        open_browser: Open in default browser
        top_n: Only show top N nodes by degree (+ their neighbors)
        min_confidence: Minimum confidence threshold

    Returns:
        Path to the generated HTML file
    """
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise ImportError(
            "pyvis is required for graph visualization.\n"
            "Install it with: uv add pyvis"
        ) from exc

    # Optional filtering
    g = kg.graph.copy()

    if min_confidence is not None:
        g.remove_nodes_from([
            n for n, data in g.nodes(data=True)
            if (data.get("confidence") or 0) < min_confidence
        ])

    if top_n is not None:
        ranked = sorted(g.degree(), key=lambda x: (-x[1], x[0]))
        hubs = {node for node, _ in ranked[:top_n]}
        keep = set(hubs)
        undirected = g.to_undirected()
        for hub in hubs:
            keep.update(undirected.neighbors(hub))
        g.remove_nodes_from([n for n in g.nodes() if n not in keep])

    # Create pyvis network
    net = Network(
        height="100%",
        width="100%",
        directed=True,
        bgcolor="#080c18",
        font_color="#e0e0e0",
        select_menu=False,
        filter_menu=False,
    )

    # Physics: force-directed layout
    net.set_options("""{
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -300,
                "centralGravity": 0.003,
                "springLength": 350,
                "springConstant": 0.03,
                "damping": 0.4,
                "avoidOverlap": 0.8
            },
            "solver": "forceAtlas2Based",
            "stabilization": { "enabled": true, "iterations": 250 },
            "enabled": true
        },
        "edges": {
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.9 } },
            "smooth": { "type": "curvedCW", "roundness": 0.12 },
            "color": { "opacity": 0.25 },
            "font": { "size": 0 },
            "hoverWidth": 2
        },
        "nodes": {
            "font": { "size": 0, "face": "Inter, sans-serif" },
            "borderWidth": 1.5,
            "borderWidthSelected": 3,
            "shadow": { "enabled": false }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 999999,
            "zoomView": true,
            "dragView": true,
            "hideEdgesOnDrag": true,
            "hideEdgesOnZoom": true
        }
    }""")

    degrees = dict(g.degree())
    entity_types_present: set[str] = set()
    entity_color_map: dict[str, str] = {}

    # Add nodes
    for node_id, data in g.nodes(data=True):
        entity_type = data.get("entity_type", "UNKNOWN")
        entity_types_present.add(entity_type)
        name = data.get("name", node_id)
        confidence = data.get("confidence", 0)
        color = _color_for_entity(entity_type, entity_color_map)
        degree = degrees.get(node_id, 0)

        tooltip_parts = [
            name,
            f"Type: {entity_type}",
            f"Connections: {degree}",
        ]
        if isinstance(confidence, (int, float)) and confidence > 0:
            tooltip_parts.append(f"Confidence: {confidence:.0%}")
        source_papers = data.get("source_papers", [])
        if source_papers:
            tooltip_parts.append(f"Sources: {len(source_papers)} papers")

        tooltip = "\n".join(tooltip_parts)
        size = max(8, min(55, 6 + degree * 3))

        net.add_node(
            node_id,
            label=name,
            title=tooltip,
            color={"background": color, "border": "rgba(255,255,255,0.08)",
                   "highlight": {"background": color, "border": "#6366f1"}},
            size=size,
            shape="dot",
            borderWidth=1.5,
            entity_type=entity_type,
            node_degree=degree,
            full_name=name,
        )

    # Add edges
    rel_color_map: dict[str, str] = {}
    seen_edges: set[tuple[str, str, str]] = set()

    for source, target, _key, data in g.edges(data=True, keys=True):
        relation_type = data.get("relation_type", "UNKNOWN")
        edge_key = (source, target, relation_type)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        edge_color = _color_for_relation(relation_type, rel_color_map)
        confidence = data.get("confidence", 0)
        evidence = data.get("evidence", "")

        source_name = g.nodes[source].get("name", source)
        target_name = g.nodes[target].get("name", target)

        tooltip = f"{source_name} → {relation_type} → {target_name}"
        if isinstance(confidence, (int, float)) and confidence > 0:
            tooltip += f"\nConfidence: {confidence:.0%}"

        net.add_edge(
            source,
            target,
            title=tooltip,
            color=edge_color,
            width=2,
            relation_type=relation_type,
            source_name=source_name,
            target_name=target_name,
            full_evidence=evidence or "",
            edge_confidence=float(confidence) if isinstance(confidence, (int, float)) else 0,
        )

    # Write HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output_path))

    # Fix heights for Firefox
    _fix_firefox_height(output_path)

    # Inject custom UI
    _inject_ui(output_path, kg, entity_types_present, entity_color_map, rel_color_map)

    logger.info(
        f"View generated: {g.number_of_nodes()} entities, "
        f"{g.number_of_edges()} relations → {output_path}"
    )

    if open_browser:
        webbrowser.open(f"file://{output_path.resolve()}")

    return output_path


def _fix_firefox_height(html_path: Path) -> None:
    """Fix graph container height for Firefox and inject Google Fonts."""
    html = html_path.read_text()
    fonts_link = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700'
        '&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">'
    )
    fix_css = (
        "<style>html, body { height: 100%; margin: 0; padding: 0; "
        "overflow: hidden; background: #080c18; } .card { height: 100%; }</style>"
    )
    html = html.replace("</head>", f"{fonts_link}\n{fix_css}\n</head>")
    html_path.write_text(html)


def _inject_ui(
    html_path: Path,
    kg: KnowledgeGraph,
    entity_types: set[str],
    entity_color_map: dict[str, str],
    rel_color_map: dict[str, str],
) -> None:
    """Inject sidebar with search, toggles, color pickers, and detail panel."""
    viewer_dir = Path(__file__).parent / "viewer"

    css = (viewer_dir / "styles.css").read_text()
    controls_template = Template((viewer_dir / "controls.html").read_text())
    app_js = (viewer_dir / "app.js").read_text()

    # Entity type controls
    entity_items = ""
    for et in sorted(entity_types):
        color = entity_color_map.get(et, NODE_PALETTE[0])
        entity_items += (
            f'<div class="type-row">'
            f'<input type="checkbox" checked data-etype="{et}" onchange="toggleEntityType(this)">'
            f'<input type="color" value="{color}" data-etype-color="{et}" onchange="changeEntityColor(this)">'
            f'<span class="type-label">{et}</span>'
            f"</div>"
        )

    # Relation type controls with counts
    rel_counts: dict[str, int] = {}
    for _, _, _, edata in kg.graph.edges(data=True, keys=True):
        rt = edata.get("relation_type", "UNKNOWN")
        rel_counts[rt] = rel_counts.get(rt, 0) + 1

    relation_items = ""
    for rt in sorted(rel_color_map.keys(), key=lambda r: rel_counts.get(r, 0), reverse=True):
        color = rel_color_map[rt]
        count = rel_counts.get(rt, 0)
        relation_items += (
            f'<div class="type-row">'
            f'<input type="checkbox" checked data-rtype="{rt}" onchange="toggleRelationType(this)">'
            f'<span class="type-label">{rt}</span>'
            f'<span class="type-count">{count}</span>'
            f"</div>"
        )

    controls_html = controls_template.substitute(
        entity_items=entity_items,
        relation_items=relation_items,
        default_degree=0,
        entity_count=kg.entity_count,
        relation_count=kg.relation_count,
    )

    config_json = json.dumps({
        "defaultDegree": 0,
    })

    injected = (
        f"<style>{css}</style>\n"
        f"{controls_html}\n"
        f"<script>var CRAB_CONFIG={config_json};</script>\n"
        f"<script>{app_js}</script>"
    )

    html = html_path.read_text()
    html = html.replace("</body>", f"{injected}</body>")
    html_path.write_text(html)
