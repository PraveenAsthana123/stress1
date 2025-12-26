#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
RAG Pipeline for GenAI-RAG-EEG Architecture
================================================================================

Module: rag_pipeline.py
Project: GenAI-RAG-EEG for Stress Classification
Author: Research Team
License: MIT

================================================================================
OVERVIEW
================================================================================

This module implements the Retrieval-Augmented Generation (RAG) pipeline for
generating human-readable explanations of EEG-based stress classifications.
The explanations are grounded in scientific literature, providing evidence-based
reasoning for model predictions.

================================================================================
RAG PIPELINE ARCHITECTURE
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    RAG EXPLANATION PIPELINE                             │
    └─────────────────────────────────────────────────────────────────────────┘

              ┌─────────────────────────────────────────────────────────┐
              │                MODEL PREDICTION                         │
              │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
              │  │ Prediction  │  │ Confidence  │  │  EEG Features   │  │
              │  │  (0 or 1)   │  │   (0-1)     │  │ (α, β, θ, FAA)  │  │
              │  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │
              └─────────┼────────────────┼─────────────────┼───────────┘
                        └────────────────┼─────────────────┘
                                         │
                                         ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  STEP 1: QUERY CONSTRUCTION                                             │
    │  ═══════════════════════════════════════════════════════════════════    │
    │                                                                         │
    │  Build semantic query from prediction and EEG features:                │
    │                                                                         │
    │  Example Query:                                                         │
    │  "EEG high stress classification alpha band suppressed                  │
    │   beta band elevated frontal theta increased"                           │
    │                                                                         │
    │  Query Components:                                                      │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │ • Stress level from prediction                                  │   │
    │  │ • Alpha power state (suppressed/elevated)                       │   │
    │  │ • Beta power state (elevated/reduced)                          │   │
    │  │ • Theta power state (increased/normal)                         │   │
    │  │ • Frontal asymmetry direction                                  │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                                                                         │
    └────────────────────────────────────┬────────────────────────────────────┘
                                         │
                                         ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  STEP 2: DOCUMENT RETRIEVAL                                             │
    │  ═══════════════════════════════════════════════════════════════════    │
    │                                                                         │
    │  ┌─────────────────┐          ┌─────────────────────────────────────┐  │
    │  │   Query Text    │          │         KNOWLEDGE BASE              │  │
    │  │                 │          │  ┌─────────────────────────────┐    │  │
    │  └────────┬────────┘          │  │  Scientific Literature      │    │  │
    │           │                   │  │  • EEG stress biomarkers    │    │  │
    │           ▼                   │  │  • Frequency band studies   │    │  │
    │  ┌────────────────┐          │  │  • HPA axis research        │    │  │
    │  │ Sentence-BERT  │          │  │  • Cognitive paradigms      │    │  │
    │  │   Encoder      │          │  └───────────────┬─────────────┘    │  │
    │  │ (all-MiniLM)   │          │                  │                   │  │
    │  └────────┬───────┘          │                  ▼                   │  │
    │           │                   │  ┌─────────────────────────────┐    │  │
    │           │ 384-dim           │  │     FAISS Vector Index      │    │  │
    │           │ embedding         │  │   (Approximate NN Search)   │    │  │
    │           │                   │  └───────────────┬─────────────┘    │  │
    │           └─────────►         │                  │                   │  │
    │                   Similarity  │                  │                   │  │
    │                   Search      ◄──────────────────┘                   │  │
    │                               │                                      │  │
    │                               │  Top-K Results (ranked by score)    │  │
    │                               └─────────────────────────────────────┘  │
    │                                                                         │
    │  Similarity Metric: Cosine Similarity (Inner Product on L2-normalized) │
    │                                                                         │
    │            query · document                                             │
    │  sim = ─────────────────────────                                       │
    │         ||query|| × ||document||                                       │
    │                                                                         │
    └────────────────────────────────────┬────────────────────────────────────┘
                                         │
                                         ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  STEP 3: EXPLANATION GENERATION                                         │
    │  ═══════════════════════════════════════════════════════════════════    │
    │                                                                         │
    │  Combine prediction, features, and retrieved evidence:                 │
    │                                                                         │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │ EXPLANATION STRUCTURE                                           │   │
    │  ├─────────────────────────────────────────────────────────────────┤   │
    │  │ 1. Classification Statement                                     │   │
    │  │    "High stress detected with 89.2% confidence."               │   │
    │  │                                                                 │   │
    │  │ 2. Feature-Based Reasoning                                      │   │
    │  │    "Key EEG indicators: significant alpha band suppression     │   │
    │  │     indicating reduced relaxation; elevated beta band power     │   │
    │  │     suggesting heightened arousal."                             │   │
    │  │                                                                 │   │
    │  │ 3. Scientific Evidence                                          │   │
    │  │    "Supporting evidence from scientific literature:             │   │
    │  │     - Alpha Band Suppression in Stress: 'Alpha band power      │   │
    │  │       suppression is a well-established neural marker...'      │   │
    │  │       (Source: Klimesch, 1999; Ray & Cole, 1985)"              │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
KNOWLEDGE BASE STRUCTURE
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                       DOCUMENT SCHEMA                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Document:                                                               │
    │   id: str           # Unique identifier                                │
    │   title: str        # Document title                                   │
    │   content: str      # Full text content                                │
    │   source: str       # Citation/reference                               │
    │   embedding: ndarray  # 384-dim vector                                 │
    │   metadata: dict    # Additional info (year, authors, etc.)           │
    └─────────────────────────────────────────────────────────────────────────┘

    Default Knowledge Base Topics:
    ┌───────────────────────────────────────────────────────────────────────┐
    │ Topic                        │ Key Concepts                           │
    ├──────────────────────────────┼────────────────────────────────────────┤
    │ Alpha Band Stress           │ 8-13 Hz suppression, relaxation marker │
    │ Beta Band Arousal           │ 13-30 Hz elevation, vigilance          │
    │ Frontal Theta               │ 4-8 Hz, cognitive load, anxiety        │
    │ Frontal Alpha Asymmetry     │ Left/right dominance, emotion          │
    │ HPA Axis                    │ Cortisol, stress physiology            │
    │ Stress Paradigms            │ Stroop, TSST, mental arithmetic        │
    └───────────────────────────────────────────────────────────────────────┘

================================================================================
VECTOR STORE IMPLEMENTATION
================================================================================

    FAISS (Facebook AI Similarity Search):
    ══════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  Index Type: IndexFlatIP (Flat Index with Inner Product)              │
    │                                                                         │
    │  • Exact nearest neighbor search                                       │
    │  • L2 normalization converts inner product to cosine similarity       │
    │  • Memory: O(n × d) where n=docs, d=384                               │
    │  • Search time: O(n × d) - linear scan                                │
    │                                                                         │
    │  For larger knowledge bases, consider:                                 │
    │  • IndexIVFFlat: Inverted file index for faster search                │
    │  • IndexHNSW: Hierarchical navigable small world graphs              │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    Fallback (No FAISS):
    ════════════════════

    NumPy-based cosine similarity search:
    • Slower but always available
    • scores = embeddings @ query.T
    • indices = argsort(scores)[::-1][:top_k]

================================================================================
SENTENCE-BERT ENCODER
================================================================================

    Model: all-MiniLM-L6-v2 (via HuggingFace)
    ═════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  Properties:                                                            │
    │  • Embedding dimension: 384                                            │
    │  • Max sequence length: 512 tokens                                     │
    │  • Parameters: 22.7M                                                   │
    │  • Speed: ~14,200 sentences/sec on GPU                                │
    │                                                                         │
    │  Processing Pipeline:                                                   │
    │                                                                         │
    │  Text Input                                                             │
    │      │                                                                  │
    │      ▼                                                                  │
    │  ┌──────────┐                                                          │
    │  │ Tokenize │ → [CLS] tokens... [SEP]                                  │
    │  └────┬─────┘                                                          │
    │       ▼                                                                 │
    │  ┌──────────┐                                                          │
    │  │ MiniLM   │ → (batch, seq_len, 384)                                  │
    │  │ Encoder  │                                                          │
    │  └────┬─────┘                                                          │
    │       ▼                                                                 │
    │  ┌──────────┐                                                          │
    │  │ Mean     │ → (batch, 384)                                           │
    │  │ Pooling  │                                                          │
    │  └────┬─────┘                                                          │
    │       ▼                                                                 │
    │  ┌──────────┐                                                          │
    │  │   L2     │ → (batch, 384) normalized                                │
    │  │  Norm    │                                                          │
    │  └──────────┘                                                          │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
USAGE EXAMPLES
================================================================================

    Creating RAG Explainer:
    ```python
    from src.models.rag_pipeline import RAGExplainer, create_default_rag_explainer

    # With default knowledge base
    explainer = create_default_rag_explainer(device="cuda")

    # Or build custom knowledge base
    explainer = RAGExplainer()
    explainer.build_knowledge_base([
        {
            "id": "doc1",
            "title": "Custom Research Paper",
            "content": "Research findings...",
            "source": "Author et al., 2023"
        },
        # More documents...
    ])
    ```

    Generating Explanations:
    ```python
    explanation = explainer.generate_explanation(
        prediction=1,           # High stress
        confidence=0.89,
        eeg_features={
            "alpha_power": 0.32,
            "beta_power": 0.71,
            "theta_power": 0.58,
            "frontal_asymmetry": 0.15
        }
    )

    print(explanation["explanation"])
    # Output:
    # High stress detected with 89.0% confidence.
    # Key EEG indicators: significant alpha band suppression...
    # Supporting evidence from scientific literature:
    # - Alpha Band Suppression in Stress: "Alpha band power..."
    ```

    Saving/Loading Knowledge Base:
    ```python
    # Save
    explainer.vector_store.save("knowledge_base/")

    # Load
    explainer = RAGExplainer()
    explainer.load_knowledge_base("knowledge_base/")
    ```

================================================================================
DEPENDENCIES
================================================================================

    Required:
    - torch >= 2.0.0
    - transformers >= 4.30.0
    - numpy >= 1.21.0

    Optional:
    - faiss-cpu >= 1.7.0 (faster similarity search)
    - faiss-gpu (for GPU acceleration)

================================================================================
REFERENCES
================================================================================

    [1] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings
        using Siamese BERT-Networks. EMNLP 2019.

    [2] Johnson, J., et al. (2019). Billion-scale similarity search with GPUs.
        IEEE Transactions on Big Data.

    [3] Lewis, P., et al. (2020). Retrieval-Augmented Generation for
        Knowledge-Intensive NLP Tasks. NeurIPS 2020.

================================================================================
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not installed. Using simple cosine similarity search.")

from transformers import AutoTokenizer, AutoModel


@dataclass
class Document:
    """Represents a scientific document in the knowledge base."""
    id: str
    title: str
    content: str
    source: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


@dataclass
class RetrievalResult:
    """Result from document retrieval."""
    document: Document
    score: float
    rank: int


class DocumentEncoder:
    """
    Encodes documents using Sentence-BERT for semantic search.

    Uses all-MiniLM-L6-v2 for efficient 384-dimensional embeddings.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.embedding_dim = 384

    def mean_pooling(
        self,
        model_output: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply mean pooling to token embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding

        Returns:
            embeddings: (n_texts, 384) numpy array
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded)
                embeddings = self.mean_pooling(model_output, encoded["attention_mask"])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


class VectorStore:
    """
    Vector store for efficient similarity search.

    Uses FAISS if available, otherwise falls back to numpy cosine similarity.
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.documents: List[Document] = []
        self.index = None

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine sim

    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        self.documents.extend(documents)

        embeddings = np.array([doc.embedding for doc in documents])

        if FAISS_AVAILABLE and self.index is not None:
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype(np.float32))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector (384,)
            top_k: Number of results to return

        Returns:
            List of RetrievalResult objects
        """
        query = query_embedding.reshape(1, -1).astype(np.float32)

        if FAISS_AVAILABLE and self.index is not None:
            faiss.normalize_L2(query)
            scores, indices = self.index.search(query, min(top_k, len(self.documents)))
            scores = scores[0]
            indices = indices[0]
        else:
            # Fallback: cosine similarity
            embeddings = np.array([doc.embedding for doc in self.documents])
            scores = np.dot(embeddings, query.T).flatten()
            indices = np.argsort(scores)[::-1][:top_k]
            scores = scores[indices]

        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if idx < len(self.documents):
                results.append(RetrievalResult(
                    document=self.documents[idx],
                    score=float(score),
                    rank=rank
                ))

        return results

    def save(self, path: str):
        """Save vector store to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save documents
        docs_data = [{
            "id": doc.id,
            "title": doc.title,
            "content": doc.content,
            "source": doc.source,
            "metadata": doc.metadata
        } for doc in self.documents]

        with open(path / "documents.json", "w") as f:
            json.dump(docs_data, f)

        # Save embeddings
        embeddings = np.array([doc.embedding for doc in self.documents])
        np.save(path / "embeddings.npy", embeddings)

        # Save FAISS index
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, str(path / "index.faiss"))

    def load(self, path: str, encoder: Optional[DocumentEncoder] = None):
        """Load vector store from disk."""
        path = Path(path)

        # Load documents
        with open(path / "documents.json", "r") as f:
            docs_data = json.load(f)

        # Load embeddings
        embeddings = np.load(path / "embeddings.npy")

        # Reconstruct documents
        self.documents = []
        for i, doc_data in enumerate(docs_data):
            doc = Document(
                id=doc_data["id"],
                title=doc_data["title"],
                content=doc_data["content"],
                source=doc_data["source"],
                embedding=embeddings[i],
                metadata=doc_data.get("metadata")
            )
            self.documents.append(doc)

        # Load or rebuild FAISS index
        if FAISS_AVAILABLE:
            index_path = path / "index.faiss"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
            else:
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                faiss.normalize_L2(embeddings.astype(np.float32))
                self.index.add(embeddings.astype(np.float32))


class RAGExplainer:
    """
    RAG-based explanation generator for stress classification.

    Retrieves relevant scientific literature and generates
    human-readable explanations grounded in evidence.
    """

    def __init__(
        self,
        encoder: Optional[DocumentEncoder] = None,
        vector_store: Optional[VectorStore] = None,
        knowledge_base_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder or DocumentEncoder(device=self.device)
        self.vector_store = vector_store or VectorStore()

        if knowledge_base_path:
            self.load_knowledge_base(knowledge_base_path)

    def load_knowledge_base(self, path: str):
        """Load pre-built knowledge base."""
        self.vector_store.load(path, self.encoder)

    def build_knowledge_base(self, documents: List[Dict[str, str]]):
        """
        Build knowledge base from document list.

        Args:
            documents: List of dicts with keys: id, title, content, source
        """
        # Encode documents
        contents = [doc["content"] for doc in documents]
        embeddings = self.encoder.encode(contents)

        # Create Document objects
        doc_objects = []
        for i, doc in enumerate(documents):
            doc_objects.append(Document(
                id=doc.get("id", str(i)),
                title=doc.get("title", ""),
                content=doc["content"],
                source=doc.get("source", ""),
                embedding=embeddings[i],
                metadata=doc.get("metadata")
            ))

        self.vector_store.add_documents(doc_objects)

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of RetrievalResult objects
        """
        query_embedding = self.encoder.encode([query])[0]
        return self.vector_store.search(query_embedding, top_k)

    def generate_explanation(
        self,
        prediction: int,
        confidence: float,
        eeg_features: Dict[str, float],
        attention_weights: Optional[np.ndarray] = None,
        context: Optional[str] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Generate explanation for stress classification.

        Args:
            prediction: Predicted class (0=low stress, 1=high stress)
            confidence: Prediction confidence (0-1)
            eeg_features: Extracted EEG features (alpha, beta, theta power, etc.)
            attention_weights: Temporal attention weights from model
            context: Additional context string
            top_k: Number of documents to retrieve

        Returns:
            Dictionary with explanation and supporting evidence
        """
        # Build retrieval query from features
        stress_level = "high" if prediction == 1 else "low"
        query_parts = [f"EEG {stress_level} stress classification"]

        if "alpha_power" in eeg_features:
            alpha_state = "suppressed" if eeg_features["alpha_power"] < 0.5 else "elevated"
            query_parts.append(f"alpha band {alpha_state}")

        if "beta_power" in eeg_features:
            beta_state = "elevated" if eeg_features["beta_power"] > 0.5 else "reduced"
            query_parts.append(f"beta band {beta_state}")

        if "theta_power" in eeg_features:
            theta_state = "increased" if eeg_features["theta_power"] > 0.5 else "normal"
            query_parts.append(f"frontal theta {theta_state}")

        query = " ".join(query_parts)

        # Retrieve relevant documents
        results = self.retrieve(query, top_k)

        # Build explanation
        explanation_parts = []

        # Main classification statement
        if prediction == 1:
            explanation_parts.append(
                f"High stress detected with {confidence*100:.1f}% confidence."
            )
        else:
            explanation_parts.append(
                f"Low stress/baseline state detected with {confidence*100:.1f}% confidence."
            )

        # Feature-based reasoning
        feature_reasons = []
        if "alpha_power" in eeg_features:
            alpha = eeg_features["alpha_power"]
            if alpha < 0.4:
                feature_reasons.append("significant alpha band suppression indicating reduced relaxation")
            elif alpha > 0.6:
                feature_reasons.append("elevated alpha activity indicating relaxed state")

        if "beta_power" in eeg_features:
            beta = eeg_features["beta_power"]
            if beta > 0.6:
                feature_reasons.append("elevated beta band power suggesting heightened arousal")
            elif beta < 0.4:
                feature_reasons.append("reduced beta activity indicating calm state")

        if "frontal_asymmetry" in eeg_features:
            asymmetry = eeg_features["frontal_asymmetry"]
            if asymmetry > 0:
                feature_reasons.append("right frontal dominance associated with withdrawal motivation")
            else:
                feature_reasons.append("left frontal dominance associated with approach motivation")

        if feature_reasons:
            explanation_parts.append(
                "Key EEG indicators: " + "; ".join(feature_reasons) + "."
            )

        # Evidence from literature
        if results:
            explanation_parts.append("\nSupporting evidence from scientific literature:")
            for result in results[:3]:
                explanation_parts.append(
                    f"- {result.document.title}: \"{result.document.content[:200]}...\" "
                    f"(Source: {result.document.source})"
                )

        return {
            "prediction": prediction,
            "prediction_label": "High Stress" if prediction == 1 else "Low Stress",
            "confidence": confidence,
            "explanation": "\n".join(explanation_parts),
            "eeg_features": eeg_features,
            "retrieved_documents": [
                {
                    "title": r.document.title,
                    "content": r.document.content,
                    "source": r.document.source,
                    "relevance_score": r.score
                }
                for r in results
            ],
            "attention_weights": attention_weights.tolist() if attention_weights is not None else None
        }


# Default EEG stress knowledge base
DEFAULT_KNOWLEDGE_BASE = [
    {
        "id": "eeg_alpha_stress",
        "title": "Alpha Band Suppression in Stress",
        "content": "Alpha band (8-13 Hz) power suppression is a well-established neural marker of stress and cognitive load. During stress, alpha desynchronization occurs particularly in frontal and parietal regions, reflecting increased cortical activation and reduced relaxation states.",
        "source": "Klimesch, 1999; Ray & Cole, 1985"
    },
    {
        "id": "eeg_beta_arousal",
        "title": "Beta Band Elevation and Arousal",
        "content": "Elevated beta band (13-30 Hz) activity is associated with heightened arousal, vigilance, and active cognitive processing. Stress-induced increases in beta power reflect activation of the sympathetic nervous system and increased mental alertness.",
        "source": "Ray & Cole, 1985; Harmony, 2009"
    },
    {
        "id": "frontal_theta_stress",
        "title": "Frontal Theta and Cognitive Load",
        "content": "Increased frontal midline theta (4-8 Hz) activity is observed during high cognitive load and anxiety states. This pattern reflects working memory engagement and emotional processing in the anterior cingulate cortex.",
        "source": "Harmony et al., 2009; Sauseng et al., 2007"
    },
    {
        "id": "frontal_asymmetry",
        "title": "Frontal Alpha Asymmetry and Emotion",
        "content": "Frontal alpha asymmetry (FAA) reflects differential activation of left vs. right prefrontal cortex. Greater right frontal activity (lower right alpha) is associated with withdrawal motivation, negative affect, and stress, while left frontal dominance relates to approach motivation.",
        "source": "Davidson, 2004; Allen et al., 2004"
    },
    {
        "id": "hpa_axis_stress",
        "title": "HPA Axis and Stress Response",
        "content": "Acute stress activates the hypothalamic-pituitary-adrenal (HPA) axis, triggering cortisol release that modulates neural oscillatory patterns. EEG changes during stress reflect both immediate sympathetic activation and downstream hormonal effects on cortical excitability.",
        "source": "McEwen, 2007; Herman et al., 2016"
    },
    {
        "id": "cognitive_stress_paradigms",
        "title": "Cognitive Stress Induction Methods",
        "content": "Validated cognitive stress paradigms include the Stroop Color-Word Task, mental arithmetic with time pressure, and the Trier Social Stress Test. These tasks reliably induce measurable stress responses including EEG changes, elevated cortisol, and increased heart rate.",
        "source": "Kirschbaum et al., 1993; Dedovic et al., 2009"
    }
]


def create_default_rag_explainer(device: Optional[str] = None) -> RAGExplainer:
    """Create RAG explainer with default EEG stress knowledge base."""
    explainer = RAGExplainer(device=device)
    explainer.build_knowledge_base(DEFAULT_KNOWLEDGE_BASE)
    return explainer


if __name__ == "__main__":
    # Test the RAG pipeline
    print("Creating RAG Explainer with default knowledge base...")
    explainer = create_default_rag_explainer()

    # Test retrieval
    print("\nTesting retrieval:")
    results = explainer.retrieve("alpha suppression during stress", top_k=3)
    for r in results:
        print(f"  [{r.score:.3f}] {r.document.title}")

    # Test explanation generation
    print("\nTesting explanation generation:")
    explanation = explainer.generate_explanation(
        prediction=1,
        confidence=0.89,
        eeg_features={
            "alpha_power": 0.32,
            "beta_power": 0.71,
            "theta_power": 0.58,
            "frontal_asymmetry": 0.15
        }
    )

    print(f"\nPrediction: {explanation['prediction_label']}")
    print(f"Confidence: {explanation['confidence']:.1%}")
    print(f"\nExplanation:\n{explanation['explanation']}")
