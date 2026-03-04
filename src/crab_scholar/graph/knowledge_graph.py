"""Knowledge graph using NetworkX MultiDiGraph.

Wraps NetworkX with research-specific entity/relation management,
JSON serialization, and basic graph stats.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """NetworkX-based knowledge graph for research paper analysis."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.created_at = datetime.now(UTC).isoformat()
        self.updated_at = self.created_at

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeGraph":
        """Load graph from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        kg = cls()
        kg.created_at = data.get("metadata", {}).get("created_at", kg.created_at)
        kg.updated_at = data.get("metadata", {}).get("updated_at", kg.updated_at)

        for entity in data.get("entities", []):
            eid = entity.pop("id", "")
            if eid:
                kg.graph.add_node(eid, **entity)

        for rel in data.get("relations", []):
            source = rel.pop("source", "")
            target = rel.pop("target", "")
            if source and target:
                kg.graph.add_edge(source, target, **rel)

        return kg

    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        name: str,
        confidence: float = 0.5,
        source_papers: list[str] | None = None,
        **attrs: Any,
    ) -> None:
        """Add or update an entity node."""
        if self.graph.has_node(entity_id):
            # Merge: update attributes, combine source_papers
            existing = self.graph.nodes[entity_id]
            existing_papers = set(existing.get("source_papers", []))
            if source_papers:
                existing_papers.update(source_papers)
            existing["source_papers"] = sorted(existing_papers)
            # Keep higher confidence
            existing["confidence"] = max(existing.get("confidence", 0), confidence)
            existing.update(attrs)
        else:
            self.graph.add_node(
                entity_id,
                entity_type=entity_type,
                name=name,
                confidence=confidence,
                source_papers=source_papers or [],
                **attrs,
            )

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        confidence: float = 0.5,
        evidence: str = "",
        source_paper: str = "",
    ) -> bool:
        """Add a relation edge. Returns False if source/target missing."""
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            return False

        # Check for duplicate edges
        for _, _, data in self.graph.edges(source_id, data=True):
            if (data.get("relation_type") == relation_type and
                    _ == target_id):
                # Update confidence if higher
                data["confidence"] = max(data.get("confidence", 0), confidence)
                return True

        self.graph.add_edge(
            source_id,
            target_id,
            relation_type=relation_type,
            confidence=confidence,
            evidence=evidence,
            source_paper=source_paper,
        )
        return True

    def get_entity(self, entity_id: str) -> dict | None:
        """Get entity data by ID."""
        if self.graph.has_node(entity_id):
            return dict(self.graph.nodes[entity_id])
        return None

    def get_relations(self, entity_id: str, direction: str = "both") -> list[dict]:
        """Get relations for an entity."""
        relations = []
        if direction in ("out", "both"):
            for _, target, data in self.graph.edges(entity_id, data=True):
                relations.append({"source": entity_id, "target": target, **data})
        if direction in ("in", "both"):
            for source, _, data in self.graph.in_edges(entity_id, data=True):
                relations.append({"source": source, "target": entity_id, **data})
        return relations

    @property
    def entity_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def relation_count(self) -> int:
        return self.graph.number_of_edges()

    def export(self) -> dict:
        """Export graph as JSON-serializable dict."""
        entities = []
        for node_id, data in self.graph.nodes(data=True):
            entities.append({"id": node_id, **data})

        relations = []
        for source, target, data in self.graph.edges(data=True):
            relations.append({"source": source, "target": target, **data})

        return {
            "metadata": {
                "created_at": self.created_at,
                "updated_at": datetime.now(UTC).isoformat(),
                "entity_count": self.entity_count,
                "relation_count": self.relation_count,
            },
            "entities": entities,
            "relations": relations,
        }

    def save(self, path: str | Path) -> None:
        """Save graph to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.export()
        path.write_text(
            json.dumps(data, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Saved graph: {self.entity_count} entities, {self.relation_count} relations → {path}")
