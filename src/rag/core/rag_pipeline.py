#!/usr/bin/env python3
"""
Main RAG Pipeline - Integrates all components

Features:
- Document ingestion and chunking
- Vector similarity search
- Knowledge graph augmentation
- Multi-layer caching
- Ollama LLM integration
- Streaming responses
"""

import json
import time
import hashlib
from typing import List, Dict, Optional, Generator, Any
from dataclasses import dataclass
import requests

# Import RAG components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag.core.chunking import ChunkingPipeline, Chunk
from rag.core.embedding import EmbeddingPipeline
from rag.vectordb.vector_store import VectorStoreFactory, BaseVectorStore
from rag.graphdb.knowledge_graph import EEGKnowledgeGraph
from rag.cache.cache_store import CacheManager, LRUCache, DiskCache


@dataclass
class RAGResponse:
    """RAG response with metadata."""
    answer: str
    sources: List[Dict]
    confidence: float
    latency_ms: float
    cached: bool
    graph_context: List[Dict]
    metadata: Dict


class OllamaLLM:
    """Ollama LLM integration for response generation."""

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.timeout = 120

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, system_prompt: str = None, stream: bool = False) -> str:
        """Generate response from Ollama."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": stream
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                if stream:
                    return self._stream_response(response)
                else:
                    result = response.json()
                    return result.get("message", {}).get("content", "")
            else:
                return f"Error: {response.status_code}"
        except requests.exceptions.ConnectionError:
            return "Error: Ollama not available. Please start Ollama service."
        except Exception as e:
            return f"Error: {str(e)}"

    def _stream_response(self, response) -> Generator[str, None, None]:
        """Stream response tokens."""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue


class RAGPipeline:
    """Main RAG Pipeline integrating all components."""

    def __init__(
        self,
        embedding_provider: str = "sentence_transformer",
        vector_store_type: str = "faiss",
        use_knowledge_graph: bool = True,
        use_cache: bool = True,
        ollama_model: str = "llama3.2",
        **kwargs
    ):
        # Initialize embedding pipeline
        self.embeddings = EmbeddingPipeline(provider=embedding_provider)

        # Initialize vector store
        self.vector_store = VectorStoreFactory.create(
            vector_store_type,
            dimension=self.embeddings.dimension
        )

        # Initialize knowledge graph
        self.knowledge_graph = EEGKnowledgeGraph() if use_knowledge_graph else None

        # Initialize cache
        if use_cache:
            self.cache = CacheManager(
                l1_cache=LRUCache(max_size=100),
                l2_cache=DiskCache(cache_dir=".cache/rag")
            )
        else:
            self.cache = None

        # Initialize LLM
        self.llm = OllamaLLM(model=ollama_model)

        # Chunking pipeline
        self.chunker = ChunkingPipeline(strategy="sentence")

        # Stats
        self.stats = {
            "queries": 0,
            "cache_hits": 0,
            "avg_latency_ms": 0,
            "documents_indexed": 0
        }

        # System prompt for EEG domain
        self.system_prompt = """You are an expert assistant specializing in EEG-based stress detection and brain-computer interfaces.

Your knowledge covers:
- EEG signal processing and analysis
- Stress biomarkers (alpha suppression, theta/beta ratio, frontal asymmetry)
- Machine learning for brain signal classification
- Neural feature extraction and preprocessing

When answering questions:
1. Use the provided context from retrieved documents
2. Cite specific sources when available
3. Be precise about technical details
4. Acknowledge uncertainty when information is incomplete

If the context doesn't contain relevant information, say so clearly."""

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def ingest_documents(self, documents: List[Dict], batch_size: int = 100):
        """Ingest documents into the RAG system."""
        all_chunks = []

        for doc in documents:
            text = doc.get("text", doc.get("content", ""))
            metadata = doc.get("metadata", {})

            # Add source info
            metadata["source"] = doc.get("source", "unknown")
            metadata["title"] = doc.get("title", "Untitled")

            chunks = self.chunker.process(text, metadata)
            all_chunks.extend(chunks)

        # Process in batches
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]

            # Generate embeddings
            texts = [chunk.text for chunk in batch]
            embeddings = self.embeddings.encode(texts)

            # Prepare documents for vector store
            docs = [
                {
                    "text": chunk.text,
                    "chunk_id": chunk.id,
                    **chunk.metadata
                }
                for chunk in batch
            ]
            ids = [f"doc_{i}_{j}" for j in range(len(batch))]

            # Add to vector store
            self.vector_store.add(embeddings, docs, ids)

        self.stats["documents_indexed"] += len(documents)
        return len(all_chunks)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant documents for query."""
        # Generate query embedding
        query_emb = self.embeddings.encode(query)[0]

        # Search vector store
        results = self.vector_store.search(query_emb, k=k)

        return [
            {
                "id": doc_id,
                "score": score,
                "text": doc.get("text", ""),
                "source": doc.get("source", "unknown"),
                "metadata": doc
            }
            for doc_id, score, doc in results
        ]

    def augment_with_knowledge_graph(self, query: str, context: List[Dict]) -> List[Dict]:
        """Augment context with knowledge graph information."""
        if not self.knowledge_graph:
            return []

        # Extract key terms from query
        key_terms = ["alpha", "beta", "theta", "delta", "gamma",
                     "stress", "eeg", "tbr", "faa", "suppression"]

        graph_context = []
        for term in key_terms:
            if term.lower() in query.lower():
                related = self.knowledge_graph.get_related_concepts(term)
                for item in related:
                    graph_context.append({
                        "concept": item.get("concept", {}),
                        "related": item.get("related", []),
                        "type": "knowledge_graph"
                    })

        return graph_context

    def generate_prompt(self, query: str, context: List[Dict], graph_context: List[Dict]) -> str:
        """Generate prompt for LLM."""
        prompt_parts = ["## Context from Retrieved Documents:\n"]

        for i, doc in enumerate(context, 1):
            prompt_parts.append(f"[{i}] Source: {doc.get('source', 'Unknown')}")
            prompt_parts.append(f"Content: {doc.get('text', '')[:500]}")
            prompt_parts.append("")

        if graph_context:
            prompt_parts.append("\n## Related Domain Knowledge:\n")
            for item in graph_context[:3]:
                concept = item.get("concept", {})
                prompt_parts.append(f"- {concept.get('name', 'Unknown')}: {concept.get('description', '')}")
                for rel in item.get("related", [])[:2]:
                    prompt_parts.append(f"  → {rel.get('name', '')} ({rel.get('type', '')})")

        prompt_parts.append(f"\n## Question:\n{query}")
        prompt_parts.append("\n## Instructions:\nProvide a comprehensive answer based on the context above. Cite sources using [1], [2], etc.")

        return "\n".join(prompt_parts)

    def query(self, query: str, k: int = 5, use_cache: bool = True) -> RAGResponse:
        """Execute RAG query and generate response."""
        start_time = time.time()
        cached = False

        # Check cache
        cache_key = self._get_cache_key(query)
        if use_cache and self.cache:
            cached_response = self.cache.get(cache_key)
            if cached_response:
                self.stats["cache_hits"] += 1
                cached_response["cached"] = True
                return RAGResponse(**cached_response)

        # Retrieve relevant documents
        context = self.retrieve(query, k=k)

        # Augment with knowledge graph
        graph_context = self.augment_with_knowledge_graph(query, context)

        # Generate prompt
        prompt = self.generate_prompt(query, context, graph_context)

        # Generate response
        if self.llm.is_available():
            answer = self.llm.generate(prompt, system_prompt=self.system_prompt)
        else:
            # Fallback response using context
            answer = self._generate_fallback_response(query, context, graph_context)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Calculate confidence based on retrieval scores
        avg_score = sum(doc.get("score", 0) for doc in context) / len(context) if context else 0
        confidence = min(avg_score, 1.0)

        # Prepare response
        response_data = {
            "answer": answer,
            "sources": [{"source": doc.get("source"), "score": doc.get("score")} for doc in context],
            "confidence": confidence,
            "latency_ms": latency_ms,
            "cached": cached,
            "graph_context": graph_context,
            "metadata": {
                "query": query,
                "num_sources": len(context),
                "llm_available": self.llm.is_available()
            }
        }

        # Cache response
        if use_cache and self.cache:
            self.cache.set(cache_key, response_data, ttl=3600)

        # Update stats
        self.stats["queries"] += 1
        self.stats["avg_latency_ms"] = (
            (self.stats["avg_latency_ms"] * (self.stats["queries"] - 1) + latency_ms)
            / self.stats["queries"]
        )

        return RAGResponse(**response_data)

    def _generate_fallback_response(self, query: str, context: List[Dict], graph_context: List[Dict]) -> str:
        """Generate response without LLM using retrieved context."""
        if not context:
            return "I couldn't find relevant information to answer your question. Please try rephrasing or ask about EEG stress detection topics."

        response_parts = ["Based on the available documents:\n"]

        for i, doc in enumerate(context[:3], 1):
            text = doc.get("text", "")[:300]
            source = doc.get("source", "Unknown")
            response_parts.append(f"[{i}] From {source}: {text}...")

        if graph_context:
            response_parts.append("\nRelated domain knowledge:")
            for item in graph_context[:2]:
                concept = item.get("concept", {})
                response_parts.append(f"- {concept.get('name', '')}: {concept.get('description', '')}")

        response_parts.append("\nNote: For more detailed analysis, please ensure Ollama is running with a suitable model.")

        return "\n".join(response_parts)

    def stream_query(self, query: str, k: int = 5) -> Generator[str, None, None]:
        """Stream RAG response."""
        context = self.retrieve(query, k=k)
        graph_context = self.augment_with_knowledge_graph(query, context)
        prompt = self.generate_prompt(query, context, graph_context)

        if self.llm.is_available():
            for token in self.llm.generate(prompt, system_prompt=self.system_prompt, stream=True):
                yield token
        else:
            yield self._generate_fallback_response(query, context, graph_context)

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        stats = self.stats.copy()

        if self.cache:
            stats["cache"] = self.cache.get_stats()

        stats["llm_available"] = self.llm.is_available()
        stats["llm_model"] = self.llm.model

        return stats

    def save(self, path: str):
        """Save pipeline state."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.vector_store.save(f"{path}/vector_store")

        with open(f"{path}/stats.json", "w") as f:
            json.dump(self.stats, f)

    def load(self, path: str):
        """Load pipeline state."""
        self.vector_store.load(f"{path}/vector_store")

        stats_path = f"{path}/stats.json"
        if Path(stats_path).exists():
            with open(stats_path, "r") as f:
                self.stats = json.load(f)


# EEG-specific RAG with domain knowledge
class EEGStressRAG(RAGPipeline):
    """Specialized RAG for EEG stress detection domain."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Add EEG-specific system prompt
        self.system_prompt = """You are an expert in EEG-based stress detection and analysis.

Your expertise includes:
- EEG frequency bands: Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-100 Hz)
- Stress biomarkers: Alpha suppression, Theta/Beta Ratio (TBR), Frontal Alpha Asymmetry (FAA)
- Signal processing: Filtering, artifact removal, feature extraction
- Machine learning: SVM, Random Forest, Deep Learning (CNN, LSTM, Transformers)
- Datasets: SAM-40, WESAD, DEAP, SEED

When answering:
1. Be precise about frequency ranges and biomarker interpretations
2. Cite relevant research when discussing findings
3. Explain the neurophysiological basis when relevant
4. Acknowledge limitations and individual variations"""

        # Initialize with domain-specific documents
        self._load_domain_knowledge()

    def _load_domain_knowledge(self):
        """Load EEG domain knowledge documents."""
        domain_docs = [
            {
                "text": """EEG-based stress detection relies on changes in brain wave patterns during stress.
                The alpha band (8-13 Hz) typically shows suppression during stress, while beta activity increases.
                Frontal alpha asymmetry (FAA) is calculated as the difference between left and right frontal electrodes,
                with rightward asymmetry associated with negative emotions and stress.""",
                "source": "EEG Stress Detection Fundamentals",
                "title": "Alpha Suppression and Stress"
            },
            {
                "text": """The Theta/Beta Ratio (TBR) is an important biomarker for attention and cognitive load.
                TBR = θ-power / β-power, typically measured at Fz, Cz, or Pz electrodes.
                Higher TBR is associated with reduced attention, while lower TBR indicates increased cognitive engagement.""",
                "source": "Cognitive Load Biomarkers",
                "title": "Theta Beta Ratio"
            },
            {
                "text": """SAM-40 dataset contains EEG recordings from 40 subjects during stress-inducing tasks.
                Tasks include mental arithmetic, Stroop test, and emotional stimuli.
                The dataset uses 64-channel EEG with 1000 Hz sampling rate.
                Ground truth labels are based on self-reported stress levels and galvanic skin response.""",
                "source": "SAM-40 Dataset Documentation",
                "title": "SAM-40 Dataset"
            },
            {
                "text": """WESAD (Wearable Stress and Affect Detection) is a multimodal dataset for stress detection.
                It includes chest-worn and wrist-worn physiological sensors.
                EEG is not included, but ECG, EMG, EDA, temperature, respiration, and accelerometer data are available.
                The dataset contains recordings from 15 subjects during baseline, stress, and amusement conditions.""",
                "source": "WESAD Dataset",
                "title": "WESAD Multimodal Dataset"
            },
            {
                "text": """Deep learning approaches for EEG stress classification include:
                - CNN for spatial feature extraction from electrode arrays
                - LSTM for temporal dynamics in EEG sequences
                - Attention mechanisms for highlighting relevant time windows
                - Hybrid CNN-LSTM-Attention models achieve state-of-the-art performance
                Features like band powers, spectral entropy, and connectivity measures improve accuracy.""",
                "source": "Deep Learning for EEG Analysis",
                "title": "Neural Network Architectures"
            }
        ]

        self.ingest_documents(domain_docs)


if __name__ == "__main__":
    # Test RAG pipeline
    print("Initializing EEG Stress RAG...")
    rag = EEGStressRAG(
        embedding_provider="hash",  # Use hash for testing without models
        vector_store_type="memory",
        ollama_model="llama3.2"
    )

    # Test queries
    test_queries = [
        "What is alpha suppression in stress detection?",
        "How does the Theta/Beta Ratio relate to cognitive load?",
        "What deep learning architectures work best for EEG classification?"
    ]

    print("\n" + "="*60)
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = rag.query(query, k=3)
        print(f"Answer: {response.answer[:300]}...")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Latency: {response.latency_ms:.1f}ms")
        print(f"Sources: {len(response.sources)}")

    # Print stats
    print("\n" + "="*60)
    print("Pipeline Stats:")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
