#!/usr/bin/env python3
"""
Knowledge Graph Module for RAG System

Supports:
- Neo4j (production)
- NetworkX (local/testing)
- Entity extraction and relationship mapping
"""

import json
from typing import List, Dict, Optional, Tuple, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Entity:
    """Represents a knowledge graph entity."""
    id: str
    name: str
    type: str
    properties: Dict


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source_id: str
    target_id: str
    type: str
    properties: Dict


class BaseKnowledgeGraph(ABC):
    """Abstract base class for knowledge graphs."""

    @abstractmethod
    def add_entity(self, entity: Entity):
        pass

    @abstractmethod
    def add_relationship(self, relationship: Relationship):
        pass

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        pass

    @abstractmethod
    def get_neighbors(self, entity_id: str, relationship_type: str = None) -> List[Entity]:
        pass

    @abstractmethod
    def query(self, query: str) -> List[Dict]:
        pass


class NetworkXKnowledgeGraph(BaseKnowledgeGraph):
    """NetworkX-based knowledge graph for local use."""

    def __init__(self):
        try:
            import networkx as nx
            self.graph = nx.DiGraph()
            self.nx = nx
        except ImportError:
            print("NetworkX not installed. Using fallback.")
            self.graph = None
            self.entities = {}
            self.relationships = []
            self.adjacency = defaultdict(list)

    def add_entity(self, entity: Entity):
        if self.graph is not None:
            self.graph.add_node(
                entity.id,
                name=entity.name,
                type=entity.type,
                **entity.properties
            )
        else:
            self.entities[entity.id] = entity

    def add_relationship(self, relationship: Relationship):
        if self.graph is not None:
            self.graph.add_edge(
                relationship.source_id,
                relationship.target_id,
                type=relationship.type,
                **relationship.properties
            )
        else:
            self.relationships.append(relationship)
            self.adjacency[relationship.source_id].append(
                (relationship.target_id, relationship.type)
            )

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        if self.graph is not None:
            if entity_id in self.graph.nodes:
                node = self.graph.nodes[entity_id]
                return Entity(
                    id=entity_id,
                    name=node.get("name", ""),
                    type=node.get("type", ""),
                    properties={k: v for k, v in node.items() if k not in ["name", "type"]}
                )
        else:
            return self.entities.get(entity_id)
        return None

    def get_neighbors(self, entity_id: str, relationship_type: str = None) -> List[Entity]:
        neighbors = []
        if self.graph is not None:
            for neighbor_id in self.graph.neighbors(entity_id):
                edge_data = self.graph.get_edge_data(entity_id, neighbor_id)
                if relationship_type is None or edge_data.get("type") == relationship_type:
                    entity = self.get_entity(neighbor_id)
                    if entity:
                        neighbors.append(entity)
        else:
            for target_id, rel_type in self.adjacency.get(entity_id, []):
                if relationship_type is None or rel_type == relationship_type:
                    entity = self.entities.get(target_id)
                    if entity:
                        neighbors.append(entity)
        return neighbors

    def query(self, query: str) -> List[Dict]:
        """Simple keyword-based query."""
        results = []
        query_lower = query.lower()

        if self.graph is not None:
            for node_id, data in self.graph.nodes(data=True):
                if query_lower in data.get("name", "").lower():
                    results.append({"id": node_id, **data})
        else:
            for entity_id, entity in self.entities.items():
                if query_lower in entity.name.lower():
                    results.append({
                        "id": entity_id,
                        "name": entity.name,
                        "type": entity.type,
                        **entity.properties
                    })

        return results

    def get_subgraph(self, entity_id: str, depth: int = 2) -> Dict:
        """Get subgraph around an entity."""
        visited = set()
        edges = []

        def traverse(node_id, current_depth):
            if current_depth > depth or node_id in visited:
                return
            visited.add(node_id)

            for neighbor in self.get_neighbors(node_id):
                edges.append((node_id, neighbor.id))
                traverse(neighbor.id, current_depth + 1)

        traverse(entity_id, 0)

        nodes = [self.get_entity(nid) for nid in visited if self.get_entity(nid)]
        return {
            "nodes": [{"id": n.id, "name": n.name, "type": n.type} for n in nodes],
            "edges": [{"source": s, "target": t} for s, t in edges]
        }


class Neo4jKnowledgeGraph(BaseKnowledgeGraph):
    """Neo4j-based knowledge graph for production."""

    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.uri = uri
        self.user = user
        self.password = password

        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.neo4j = GraphDatabase
        except ImportError:
            print("Neo4j driver not installed. Using fallback.")
            self.driver = None
            self._fallback = NetworkXKnowledgeGraph()

    def _run_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        if self.driver is None:
            return []

        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]

    def add_entity(self, entity: Entity):
        if self.driver is not None:
            query = f"""
            MERGE (e:{entity.type} {{id: $id}})
            SET e.name = $name
            SET e += $properties
            """
            self._run_query(query, {
                "id": entity.id,
                "name": entity.name,
                "properties": entity.properties
            })
        else:
            self._fallback.add_entity(entity)

    def add_relationship(self, relationship: Relationship):
        if self.driver is not None:
            query = f"""
            MATCH (a {{id: $source_id}})
            MATCH (b {{id: $target_id}})
            MERGE (a)-[r:{relationship.type}]->(b)
            SET r += $properties
            """
            self._run_query(query, {
                "source_id": relationship.source_id,
                "target_id": relationship.target_id,
                "properties": relationship.properties
            })
        else:
            self._fallback.add_relationship(relationship)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        if self.driver is not None:
            query = "MATCH (e {id: $id}) RETURN e"
            results = self._run_query(query, {"id": entity_id})
            if results:
                node = results[0]["e"]
                return Entity(
                    id=node["id"],
                    name=node.get("name", ""),
                    type=list(node.labels)[0] if node.labels else "",
                    properties=dict(node)
                )
        else:
            return self._fallback.get_entity(entity_id)
        return None

    def get_neighbors(self, entity_id: str, relationship_type: str = None) -> List[Entity]:
        if self.driver is not None:
            rel_filter = f":{relationship_type}" if relationship_type else ""
            query = f"""
            MATCH (a {{id: $id}})-[{rel_filter}]->(b)
            RETURN b
            """
            results = self._run_query(query, {"id": entity_id})
            entities = []
            for record in results:
                node = record["b"]
                entities.append(Entity(
                    id=node["id"],
                    name=node.get("name", ""),
                    type=list(node.labels)[0] if node.labels else "",
                    properties=dict(node)
                ))
            return entities
        else:
            return self._fallback.get_neighbors(entity_id, relationship_type)

    def query(self, query: str) -> List[Dict]:
        if self.driver is not None:
            cypher = """
            MATCH (e)
            WHERE toLower(e.name) CONTAINS toLower($query)
            RETURN e
            LIMIT 10
            """
            return self._run_query(cypher, {"query": query})
        else:
            return self._fallback.query(query)


class EEGKnowledgeGraph:
    """Domain-specific knowledge graph for EEG stress detection."""

    def __init__(self, backend: str = "networkx"):
        if backend == "neo4j":
            self.graph = Neo4jKnowledgeGraph()
        else:
            self.graph = NetworkXKnowledgeGraph()

        self._initialize_domain_knowledge()

    def _initialize_domain_knowledge(self):
        """Initialize with EEG domain knowledge."""
        # EEG Bands
        bands = [
            ("delta", "Delta Band", "0.5-4 Hz, associated with deep sleep"),
            ("theta", "Theta Band", "4-8 Hz, associated with drowsiness, meditation"),
            ("alpha", "Alpha Band", "8-13 Hz, associated with relaxation, reduced during stress"),
            ("beta", "Beta Band", "13-30 Hz, associated with active thinking, increased during stress"),
            ("gamma", "Gamma Band", "30-100 Hz, associated with cognitive processing")
        ]

        for band_id, name, description in bands:
            self.graph.add_entity(Entity(
                id=band_id,
                name=name,
                type="EEGBand",
                properties={"description": description}
            ))

        # Biomarkers
        biomarkers = [
            ("alpha_suppression", "Alpha Suppression", "Reduction in alpha power during stress"),
            ("tbr", "Theta/Beta Ratio", "Ratio of theta to beta power, decreases with stress"),
            ("faa", "Frontal Alpha Asymmetry", "Difference between left and right frontal alpha")
        ]

        for bio_id, name, description in biomarkers:
            self.graph.add_entity(Entity(
                id=bio_id,
                name=name,
                type="Biomarker",
                properties={"description": description}
            ))

        # Relationships
        self.graph.add_relationship(Relationship(
            source_id="alpha",
            target_id="alpha_suppression",
            type="INDICATES",
            properties={"direction": "decrease"}
        ))

        self.graph.add_relationship(Relationship(
            source_id="theta",
            target_id="tbr",
            type="COMPONENT_OF",
            properties={"role": "numerator"}
        ))

        self.graph.add_relationship(Relationship(
            source_id="beta",
            target_id="tbr",
            type="COMPONENT_OF",
            properties={"role": "denominator"}
        ))

    def get_related_concepts(self, concept: str) -> List[Dict]:
        """Get concepts related to a query."""
        results = self.graph.query(concept)
        expanded = []

        for result in results:
            entity_id = result.get("id")
            if entity_id:
                neighbors = self.graph.get_neighbors(entity_id)
                expanded.append({
                    "concept": result,
                    "related": [{"id": n.id, "name": n.name, "type": n.type} for n in neighbors]
                })

        return expanded


if __name__ == "__main__":
    # Test knowledge graph
    kg = EEGKnowledgeGraph(backend="networkx")

    # Query
    results = kg.get_related_concepts("alpha")
    print("Related to 'alpha':")
    for r in results:
        print(f"  {r['concept']}")
        for rel in r['related']:
            print(f"    -> {rel['name']} ({rel['type']})")
