"""Tests for the knowledge graph module."""

import json

from research_claw.graph.knowledge_graph import KnowledgeGraph
from research_claw.graph.builder import build_graph


class TestKnowledgeGraph:
    def test_add_entity(self):
        kg = KnowledgeGraph()
        kg.add_entity("paper:1", "PAPER", "Test Paper", confidence=0.9)
        assert kg.entity_count == 1
        entity = kg.get_entity("paper:1")
        assert entity["name"] == "Test Paper"
        assert entity["entity_type"] == "PAPER"

    def test_add_entity_merge(self):
        kg = KnowledgeGraph()
        kg.add_entity("paper:1", "PAPER", "Test", confidence=0.5, source_papers=["a"])
        kg.add_entity("paper:1", "PAPER", "Test", confidence=0.9, source_papers=["b"])
        assert kg.entity_count == 1
        entity = kg.get_entity("paper:1")
        assert entity["confidence"] == 0.9  # Takes max
        assert set(entity["source_papers"]) == {"a", "b"}

    def test_add_relation(self):
        kg = KnowledgeGraph()
        kg.add_entity("a", "X", "A")
        kg.add_entity("b", "Y", "B")
        result = kg.add_relation("a", "b", "RELATED_TO")
        assert result is True
        assert kg.relation_count == 1

    def test_add_relation_missing_node(self):
        kg = KnowledgeGraph()
        kg.add_entity("a", "X", "A")
        result = kg.add_relation("a", "nonexistent", "RELATED_TO")
        assert result is False

    def test_export(self):
        kg = KnowledgeGraph()
        kg.add_entity("a", "X", "A")
        kg.add_entity("b", "Y", "B")
        kg.add_relation("a", "b", "RELATED_TO")
        data = kg.export()
        assert data["metadata"]["entity_count"] == 2
        assert data["metadata"]["relation_count"] == 1
        assert len(data["entities"]) == 2
        assert len(data["relations"]) == 1

    def test_save_and_load(self, tmp_path):
        kg = KnowledgeGraph()
        kg.add_entity("a", "X", "A", confidence=0.8)
        kg.add_entity("b", "Y", "B")
        kg.add_relation("a", "b", "RELATED_TO", confidence=0.7)

        path = tmp_path / "test_graph.json"
        kg.save(path)

        loaded = KnowledgeGraph.load(path)
        assert loaded.entity_count == 2
        assert loaded.relation_count == 1


class TestBuildGraph:
    def test_build_from_analysis(self, sample_analysis):
        kg = build_graph([sample_analysis])
        # Should have paper + authors + extracted entities
        assert kg.entity_count > 0
        assert kg.relation_count > 0

        # Check paper node exists
        paper_entity = kg.get_entity(f"paper:{sample_analysis.paper.paper_id}")
        assert paper_entity is not None
        assert paper_entity["entity_type"] == "PAPER"

    def test_build_creates_author_edges(self, sample_analysis):
        kg = build_graph([sample_analysis])
        # Should have AUTHORED_BY relations
        relations = kg.get_relations(f"paper:{sample_analysis.paper.paper_id}", direction="out")
        rel_types = {r["relation_type"] for r in relations}
        assert "AUTHORED_BY" in rel_types
