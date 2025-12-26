#!/usr/bin/env python3
"""
Embedding Module for RAG System

Supports multiple embedding models:
- Sentence Transformers (local)
- Ollama embeddings
- OpenAI embeddings
- Custom embeddings
"""

import numpy as np
from typing import List, Dict, Optional, Union
from abc import ABC, abstractmethod
import hashlib
import json


class BaseEmbedding(ABC):
    """Abstract base class for embeddings."""

    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass


class SentenceTransformerEmbedding(BaseEmbedding):
    """Sentence Transformer embeddings (local)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._dim = self.model.get_sentence_embedding_dimension()
        except ImportError:
            print("sentence-transformers not installed. Using fallback.")
            self.model = None
            self._dim = 384

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        if self.model:
            return self.model.encode(texts, convert_to_numpy=True)
        else:
            # Fallback: random embeddings for testing
            return np.random.randn(len(texts), self._dim).astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dim


class OllamaEmbedding(BaseEmbedding):
    """Ollama embeddings (local LLM)."""

    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self._dim = 768  # Default for nomic-embed-text

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        import requests

        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": text},
                    timeout=30
                )
                if response.status_code == 200:
                    emb = response.json().get("embedding", [])
                    embeddings.append(emb)
                    self._dim = len(emb)
                else:
                    embeddings.append(np.zeros(self._dim).tolist())
            except Exception as e:
                print(f"Ollama embedding error: {e}")
                embeddings.append(np.zeros(self._dim).tolist())

        return np.array(embeddings, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dim


class HashEmbedding(BaseEmbedding):
    """Fast hash-based embeddings for testing."""

    def __init__(self, dimension: int = 384):
        self._dim = dimension

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            # Create deterministic embedding from hash
            hash_bytes = hashlib.sha256(text.encode()).digest()
            np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
            emb = np.random.randn(self._dim).astype(np.float32)
            emb = emb / np.linalg.norm(emb)  # Normalize
            embeddings.append(emb)

        return np.array(embeddings)

    @property
    def dimension(self) -> int:
        return self._dim


class EmbeddingPipeline:
    """Unified embedding pipeline with caching."""

    def __init__(self, provider: str = "sentence_transformer", **kwargs):
        self.providers = {
            "sentence_transformer": SentenceTransformerEmbedding,
            "ollama": OllamaEmbedding,
            "hash": HashEmbedding
        }
        self.embedding = self.providers.get(provider, HashEmbedding)(**kwargs)
        self.cache = {}
        self.cache_enabled = True

    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def encode(self, texts: Union[str, List[str]], use_cache: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        if use_cache and self.cache_enabled:
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []

            for i, text in enumerate(texts):
                key = self._get_cache_key(text)
                if key in self.cache:
                    cached_embeddings.append((i, self.cache[key]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # Encode uncached texts
            if uncached_texts:
                new_embeddings = self.embedding.encode(uncached_texts)
                for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                    key = self._get_cache_key(text)
                    self.cache[key] = emb
                    cached_embeddings.append((idx, emb))

            # Sort by original index
            cached_embeddings.sort(key=lambda x: x[0])
            return np.array([emb for _, emb in cached_embeddings])

        return self.embedding.encode(texts)

    @property
    def dimension(self) -> int:
        return self.embedding.dimension

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10))

    def batch_similarity(self, query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
        """Compute similarities between query and multiple documents."""
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        doc_norms = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-10)
        return np.dot(doc_norms, query_norm)


if __name__ == "__main__":
    # Test embeddings
    pipeline = EmbeddingPipeline(provider="hash")

    texts = [
        "EEG stress detection using deep learning",
        "Alpha wave suppression during cognitive load",
        "Machine learning for brain-computer interfaces"
    ]

    embeddings = pipeline.encode(texts)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Dimension: {pipeline.dimension}")

    # Test similarity
    query = pipeline.encode("stress detection from brain signals")
    similarities = pipeline.batch_similarity(query[0], embeddings)
    print(f"Similarities: {similarities}")
