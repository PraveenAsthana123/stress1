#!/usr/bin/env python3
"""
Document Chunking Strategies for RAG System

Implements multiple chunking strategies:
- Fixed-size chunking
- Semantic chunking
- Sentence-based chunking
- Sliding window chunking
- Recursive chunking
"""

import re
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class Chunk:
    """Represents a document chunk."""
    id: str
    text: str
    metadata: Dict
    start_idx: int
    end_idx: int
    token_count: int
    embedding: Optional[np.ndarray] = None


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: Dict = None) -> List[Chunk]:
        pass

    def _count_tokens(self, text: str) -> int:
        """Approximate token count (words * 1.3)."""
        return int(len(text.split()) * 1.3)


class FixedSizeChunker(BaseChunker):
    """Fixed-size chunking with overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: Dict = None) -> List[Chunk]:
        metadata = metadata or {}
        chunks = []
        words = text.split()

        start = 0
        chunk_id = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = ' '.join(words[start:end])

            chunks.append(Chunk(
                id=f"chunk_{chunk_id}",
                text=chunk_text,
                metadata={**metadata, "chunk_id": chunk_id, "strategy": "fixed_size"},
                start_idx=start,
                end_idx=end,
                token_count=self._count_tokens(chunk_text)
            ))

            start = end - self.overlap if end < len(words) else len(words)
            chunk_id += 1

        return chunks


class SentenceChunker(BaseChunker):
    """Sentence-based chunking."""

    def __init__(self, max_sentences: int = 5, min_chunk_size: int = 100):
        self.max_sentences = max_sentences
        self.min_chunk_size = min_chunk_size

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str, metadata: Dict = None) -> List[Chunk]:
        metadata = metadata or {}
        sentences = self._split_sentences(text)
        chunks = []

        current_chunk = []
        start_idx = 0
        chunk_id = 0

        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)

            if len(current_chunk) >= self.max_sentences:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        id=f"chunk_{chunk_id}",
                        text=chunk_text,
                        metadata={**metadata, "chunk_id": chunk_id, "strategy": "sentence"},
                        start_idx=start_idx,
                        end_idx=i,
                        token_count=self._count_tokens(chunk_text)
                    ))
                    chunk_id += 1
                    start_idx = i + 1
                    current_chunk = []

        # Handle remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                id=f"chunk_{chunk_id}",
                text=chunk_text,
                metadata={**metadata, "chunk_id": chunk_id, "strategy": "sentence"},
                start_idx=start_idx,
                end_idx=len(sentences),
                token_count=self._count_tokens(chunk_text)
            ))

        return chunks


class SemanticChunker(BaseChunker):
    """Semantic chunking based on topic boundaries."""

    def __init__(self, embedding_model=None, similarity_threshold: float = 0.7):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity."""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10)

    def chunk(self, text: str, metadata: Dict = None) -> List[Chunk]:
        metadata = metadata or {}
        sentences = re.split(r'(?<=[.!?])\s+', text)

        if not self.embedding_model or len(sentences) < 2:
            # Fallback to sentence chunker
            return SentenceChunker().chunk(text, metadata)

        chunks = []
        current_chunk = [sentences[0]]
        start_idx = 0
        chunk_id = 0

        for i in range(1, len(sentences)):
            # Compare semantic similarity
            prev_emb = self.embedding_model.encode(sentences[i-1])
            curr_emb = self.embedding_model.encode(sentences[i])
            similarity = self._compute_similarity(prev_emb, curr_emb)

            if similarity < self.similarity_threshold:
                # Start new chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(Chunk(
                    id=f"chunk_{chunk_id}",
                    text=chunk_text,
                    metadata={**metadata, "chunk_id": chunk_id, "strategy": "semantic"},
                    start_idx=start_idx,
                    end_idx=i,
                    token_count=self._count_tokens(chunk_text)
                ))
                chunk_id += 1
                start_idx = i
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])

        # Handle remaining
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                id=f"chunk_{chunk_id}",
                text=chunk_text,
                metadata={**metadata, "chunk_id": chunk_id, "strategy": "semantic"},
                start_idx=start_idx,
                end_idx=len(sentences),
                token_count=self._count_tokens(chunk_text)
            ))

        return chunks


class SlidingWindowChunker(BaseChunker):
    """Sliding window chunking for dense retrieval."""

    def __init__(self, window_size: int = 256, stride: int = 128):
        self.window_size = window_size
        self.stride = stride

    def chunk(self, text: str, metadata: Dict = None) -> List[Chunk]:
        metadata = metadata or {}
        words = text.split()
        chunks = []
        chunk_id = 0

        for start in range(0, len(words), self.stride):
            end = min(start + self.window_size, len(words))
            chunk_text = ' '.join(words[start:end])

            chunks.append(Chunk(
                id=f"chunk_{chunk_id}",
                text=chunk_text,
                metadata={**metadata, "chunk_id": chunk_id, "strategy": "sliding_window"},
                start_idx=start,
                end_idx=end,
                token_count=self._count_tokens(chunk_text)
            ))
            chunk_id += 1

            if end >= len(words):
                break

        return chunks


class RecursiveChunker(BaseChunker):
    """Recursive chunking with hierarchy."""

    def __init__(self, separators: List[str] = None, max_chunk_size: int = 512):
        self.separators = separators or ["\n\n", "\n", ". ", " "]
        self.max_chunk_size = max_chunk_size

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text."""
        if not separators:
            return [text]

        sep = separators[0]
        parts = text.split(sep)

        result = []
        for part in parts:
            if len(part.split()) > self.max_chunk_size:
                result.extend(self._split_text(part, separators[1:]))
            else:
                result.append(part)

        return result

    def chunk(self, text: str, metadata: Dict = None) -> List[Chunk]:
        metadata = metadata or {}
        parts = self._split_text(text, self.separators)

        chunks = []
        chunk_id = 0
        current_pos = 0

        for part in parts:
            if part.strip():
                chunks.append(Chunk(
                    id=f"chunk_{chunk_id}",
                    text=part.strip(),
                    metadata={**metadata, "chunk_id": chunk_id, "strategy": "recursive"},
                    start_idx=current_pos,
                    end_idx=current_pos + len(part),
                    token_count=self._count_tokens(part)
                ))
                chunk_id += 1
            current_pos += len(part) + 1

        return chunks


class ChunkingPipeline:
    """Unified chunking pipeline with multiple strategies."""

    def __init__(self, strategy: str = "fixed_size", **kwargs):
        self.strategies = {
            "fixed_size": FixedSizeChunker,
            "sentence": SentenceChunker,
            "semantic": SemanticChunker,
            "sliding_window": SlidingWindowChunker,
            "recursive": RecursiveChunker
        }
        self.chunker = self.strategies.get(strategy, FixedSizeChunker)(**kwargs)

    def process(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """Process text and return chunks."""
        return self.chunker.chunk(text, metadata)

    def process_documents(self, documents: List[Dict]) -> List[Chunk]:
        """Process multiple documents."""
        all_chunks = []
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            chunks = self.process(text, metadata)
            all_chunks.extend(chunks)
        return all_chunks


if __name__ == "__main__":
    # Test chunking
    sample_text = """
    EEG-based stress detection uses brain signals to identify mental stress states.
    The alpha band (8-13 Hz) typically shows suppression during stress.
    Theta/beta ratio is another important biomarker for cognitive load.
    Frontal alpha asymmetry indicates emotional valence and stress response.
    Machine learning models can classify stress vs baseline with high accuracy.
    Deep learning approaches like CNN-LSTM achieve state-of-the-art results.
    """

    for strategy in ["fixed_size", "sentence", "sliding_window", "recursive"]:
        pipeline = ChunkingPipeline(strategy=strategy, chunk_size=50)
        chunks = pipeline.process(sample_text, {"source": "test"})
        print(f"\n{strategy.upper()}: {len(chunks)} chunks")
        for chunk in chunks[:2]:
            print(f"  - {chunk.text[:50]}...")
