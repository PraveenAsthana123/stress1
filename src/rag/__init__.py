"""
EEG Stress RAG System

Comprehensive Retrieval-Augmented Generation system for EEG stress detection domain.

Components:
- core: Chunking, Embedding, RAG Pipeline
- vectordb: Vector stores (FAISS, ChromaDB, In-memory)
- graphdb: Knowledge graphs (Neo4j, NetworkX)
- cache: Caching layers (LRU, Redis, Disk)
- evaluation: Comprehensive evaluation metrics
- governance: AI governance (Explainability, Trust, Ethics, Compliance, Security)
"""

from .core.rag_pipeline import RAGPipeline, EEGStressRAG
from .core.embedding import EmbeddingPipeline
from .core.chunking import ChunkingPipeline

__version__ = "1.0.0"
__all__ = [
    "RAGPipeline",
    "EEGStressRAG",
    "EmbeddingPipeline",
    "ChunkingPipeline"
]
