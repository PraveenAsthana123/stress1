#!/usr/bin/env python3
"""
Vector Database Module for RAG System

Supports multiple vector stores:
- FAISS (local, fast)
- ChromaDB (persistent, feature-rich)
- Custom in-memory store
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
import pickle


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(self, embeddings: np.ndarray, documents: List[Dict], ids: List[str] = None):
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float, Dict]]:
        pass

    @abstractmethod
    def delete(self, ids: List[str]):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store for fast similarity search."""

    def __init__(self, dimension: int = 384, index_type: str = "flat"):
        self.dimension = dimension
        self.index_type = index_type
        self.documents = {}
        self.id_map = {}
        self.reverse_id_map = {}
        self.current_idx = 0

        try:
            import faiss
            if index_type == "flat":
                self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine sim on normalized)
            elif index_type == "ivf":
                quantizer = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            else:
                self.index = faiss.IndexFlatIP(dimension)
            self.faiss = faiss
        except ImportError:
            print("FAISS not installed. Using fallback.")
            self.index = None
            self.embeddings = []

    def add(self, embeddings: np.ndarray, documents: List[Dict], ids: List[str] = None):
        if ids is None:
            ids = [f"doc_{self.current_idx + i}" for i in range(len(documents))]

        # Normalize embeddings for cosine similarity
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        embeddings = embeddings.astype(np.float32)

        if self.index is not None:
            self.index.add(embeddings)
        else:
            self.embeddings.extend(embeddings)

        for i, (doc_id, doc) in enumerate(zip(ids, documents)):
            idx = self.current_idx + i
            self.documents[doc_id] = doc
            self.id_map[doc_id] = idx
            self.reverse_id_map[idx] = doc_id

        self.current_idx += len(documents)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float, Dict]]:
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

        if self.index is not None:
            scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx in self.reverse_id_map:
                    doc_id = self.reverse_id_map[idx]
                    results.append((doc_id, float(score), self.documents[doc_id]))
            return results
        else:
            # Fallback: brute force
            if not self.embeddings:
                return []
            emb_array = np.array(self.embeddings)
            scores = np.dot(emb_array, query_embedding.T).flatten()
            top_k = np.argsort(scores)[::-1][:k]
            results = []
            for idx in top_k:
                if idx in self.reverse_id_map:
                    doc_id = self.reverse_id_map[idx]
                    results.append((doc_id, float(scores[idx]), self.documents[doc_id]))
            return results

    def delete(self, ids: List[str]):
        for doc_id in ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
                idx = self.id_map.pop(doc_id, None)
                if idx is not None:
                    self.reverse_id_map.pop(idx, None)

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "documents": self.documents,
            "id_map": self.id_map,
            "reverse_id_map": {str(k): v for k, v in self.reverse_id_map.items()},
            "current_idx": self.current_idx,
            "dimension": self.dimension
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        # Save index
        if self.index is not None:
            self.faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        else:
            with open(os.path.join(path, "embeddings.pkl"), "wb") as f:
                pickle.dump(self.embeddings, f)

    def load(self, path: str):
        # Load metadata
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        self.documents = metadata["documents"]
        self.id_map = metadata["id_map"]
        self.reverse_id_map = {int(k): v for k, v in metadata["reverse_id_map"].items()}
        self.current_idx = metadata["current_idx"]
        self.dimension = metadata["dimension"]

        # Load index
        index_path = os.path.join(path, "index.faiss")
        if os.path.exists(index_path) and self.index is not None:
            self.index = self.faiss.read_index(index_path)
        else:
            emb_path = os.path.join(path, "embeddings.pkl")
            if os.path.exists(emb_path):
                with open(emb_path, "rb") as f:
                    self.embeddings = pickle.load(f)


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-based vector store."""

    def __init__(self, collection_name: str = "eeg_stress_rag", persist_dir: str = None):
        self.collection_name = collection_name
        self.persist_dir = persist_dir

        try:
            import chromadb
            if persist_dir:
                self.client = chromadb.PersistentClient(path=persist_dir)
            else:
                self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.chromadb = chromadb
        except ImportError:
            print("ChromaDB not installed. Using fallback.")
            self.client = None
            self.collection = None
            self._fallback_store = FAISSVectorStore()

    def add(self, embeddings: np.ndarray, documents: List[Dict], ids: List[str] = None):
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        if self.collection is not None:
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=[json.dumps(doc) for doc in documents],
                ids=ids,
                metadatas=documents
            )
        else:
            self._fallback_store.add(embeddings, documents, ids)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float, Dict]]:
        if self.collection is not None:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k
            )
            output = []
            for i, (doc_id, distance) in enumerate(zip(results["ids"][0], results["distances"][0])):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                score = 1 - distance  # Convert distance to similarity
                output.append((doc_id, score, metadata))
            return output
        else:
            return self._fallback_store.search(query_embedding, k)

    def delete(self, ids: List[str]):
        if self.collection is not None:
            self.collection.delete(ids=ids)
        else:
            self._fallback_store.delete(ids)

    def save(self, path: str):
        if self.persist_dir:
            # ChromaDB auto-persists
            pass
        elif hasattr(self, '_fallback_store'):
            self._fallback_store.save(path)

    def load(self, path: str):
        if hasattr(self, '_fallback_store'):
            self._fallback_store.load(path)


class InMemoryVectorStore(BaseVectorStore):
    """Simple in-memory vector store."""

    def __init__(self, dimension: int = 384, **kwargs):
        self.dimension = dimension
        self.embeddings = []
        self.documents = []
        self.ids = []

    def add(self, embeddings: np.ndarray, documents: List[Dict], ids: List[str] = None):
        if ids is None:
            ids = [f"doc_{len(self.ids) + i}" for i in range(len(documents))]

        for emb, doc, doc_id in zip(embeddings, documents, ids):
            self.embeddings.append(emb / (np.linalg.norm(emb) + 1e-10))
            self.documents.append(doc)
            self.ids.append(doc_id)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float, Dict]]:
        if not self.embeddings:
            return []

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        emb_array = np.array(self.embeddings)
        scores = np.dot(emb_array, query_norm)

        top_k = np.argsort(scores)[::-1][:k]
        return [(self.ids[i], float(scores[i]), self.documents[i]) for i in top_k]

    def delete(self, ids: List[str]):
        indices_to_remove = [i for i, doc_id in enumerate(self.ids) if doc_id in ids]
        for i in sorted(indices_to_remove, reverse=True):
            del self.embeddings[i]
            del self.documents[i]
            del self.ids[i]

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        data = {
            "embeddings": [e.tolist() for e in self.embeddings],
            "documents": self.documents,
            "ids": self.ids
        }
        with open(os.path.join(path, "store.json"), "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        with open(os.path.join(path, "store.json"), "r") as f:
            data = json.load(f)
        self.embeddings = [np.array(e) for e in data["embeddings"]]
        self.documents = data["documents"]
        self.ids = data["ids"]


class VectorStoreFactory:
    """Factory for creating vector stores."""

    @staticmethod
    def create(store_type: str = "faiss", **kwargs) -> BaseVectorStore:
        stores = {
            "faiss": FAISSVectorStore,
            "chroma": ChromaVectorStore,
            "memory": InMemoryVectorStore
        }
        return stores.get(store_type, InMemoryVectorStore)(**kwargs)


if __name__ == "__main__":
    # Test vector store
    store = VectorStoreFactory.create("memory")

    # Add documents
    embeddings = np.random.randn(5, 384).astype(np.float32)
    documents = [
        {"text": "EEG stress detection", "source": "paper1"},
        {"text": "Alpha wave analysis", "source": "paper2"},
        {"text": "Deep learning for BCI", "source": "paper3"},
        {"text": "Cognitive load measurement", "source": "paper4"},
        {"text": "Neural signal processing", "source": "paper5"}
    ]

    store.add(embeddings, documents)

    # Search
    query = np.random.randn(384).astype(np.float32)
    results = store.search(query, k=3)

    print("Search Results:")
    for doc_id, score, doc in results:
        print(f"  {doc_id}: {score:.3f} - {doc['text']}")
