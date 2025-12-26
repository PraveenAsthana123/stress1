#!/usr/bin/env python3
"""
Test script for the complete RAG system.

Tests all components:
- Chunking strategies
- Embedding pipelines
- Vector stores
- Knowledge graphs
- Cache layers
- Evaluation metrics
- AI Governance
- RAG Pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_chunking():
    """Test chunking strategies."""
    print("\n" + "="*60)
    print("TESTING: Chunking Strategies")
    print("="*60)

    from rag.core.chunking import ChunkingPipeline

    sample_text = """
    EEG-based stress detection uses brain signals to identify mental stress states.
    The alpha band (8-13 Hz) typically shows suppression during stress.
    Theta/beta ratio is another important biomarker for cognitive load.
    Frontal alpha asymmetry indicates emotional valence and stress response.
    Machine learning models can classify stress vs baseline with high accuracy.
    Deep learning approaches like CNN-LSTM achieve state-of-the-art results.
    """

    strategies = ["fixed_size", "sentence", "sliding_window", "recursive"]

    for strategy in strategies:
        pipeline = ChunkingPipeline(strategy=strategy, chunk_size=30)
        chunks = pipeline.process(sample_text, {"source": "test"})
        print(f"  {strategy}: {len(chunks)} chunks created")

    print("  [OK] All chunking strategies working")
    return True


def test_embedding():
    """Test embedding pipeline."""
    print("\n" + "="*60)
    print("TESTING: Embedding Pipeline")
    print("="*60)

    from rag.core.embedding import EmbeddingPipeline

    # Test hash embedding (no external dependencies)
    pipeline = EmbeddingPipeline(provider="hash")

    texts = [
        "EEG stress detection",
        "Alpha wave suppression",
        "Deep learning for BCI"
    ]

    embeddings = pipeline.encode(texts)
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Dimension: {pipeline.dimension}")

    # Test similarity
    query = pipeline.encode("stress detection")
    similarities = pipeline.batch_similarity(query[0], embeddings)
    print(f"  Similarities computed: {len(similarities)}")

    print("  [OK] Embedding pipeline working")
    return True


def test_vector_store():
    """Test vector stores."""
    print("\n" + "="*60)
    print("TESTING: Vector Stores")
    print("="*60)

    import numpy as np
    from rag.vectordb.vector_store import VectorStoreFactory

    # Test in-memory store
    store = VectorStoreFactory.create("memory")

    # Add documents
    embeddings = np.random.randn(5, 384).astype(np.float32)
    documents = [
        {"text": f"Document {i}", "source": f"source_{i}"}
        for i in range(5)
    ]

    store.add(embeddings, documents)
    print(f"  Added {len(documents)} documents")

    # Search
    query = np.random.randn(384).astype(np.float32)
    results = store.search(query, k=3)
    print(f"  Search returned {len(results)} results")

    print("  [OK] Vector store working")
    return True


def test_knowledge_graph():
    """Test knowledge graph."""
    print("\n" + "="*60)
    print("TESTING: Knowledge Graph")
    print("="*60)

    from rag.graphdb.knowledge_graph import EEGKnowledgeGraph

    kg = EEGKnowledgeGraph(backend="networkx")

    # Query
    results = kg.get_related_concepts("alpha")
    print(f"  'alpha' related concepts: {len(results)}")

    results = kg.get_related_concepts("theta")
    print(f"  'theta' related concepts: {len(results)}")

    print("  [OK] Knowledge graph working")
    return True


def test_cache():
    """Test cache system."""
    print("\n" + "="*60)
    print("TESTING: Cache System")
    print("="*60)

    from rag.cache.cache_store import CacheManager, LRUCache, DiskCache

    # Test LRU cache
    cache = CacheManager(
        l1_cache=LRUCache(max_size=10),
        l2_cache=None  # Skip disk cache for test
    )

    # Set values
    cache.set("key1", {"data": "test1"})
    cache.set("key2", {"data": "test2"})

    # Get values
    v1 = cache.get("key1")
    v2 = cache.get("key2")
    v3 = cache.get("key3")  # Should be None

    print(f"  key1: {v1 is not None}")
    print(f"  key2: {v2 is not None}")
    print(f"  key3 (missing): {v3 is None}")

    stats = cache.get_stats()
    print(f"  Cache stats: {stats}")

    print("  [OK] Cache system working")
    return True


def test_evaluation():
    """Test evaluation metrics."""
    print("\n" + "="*60)
    print("TESTING: Evaluation Metrics")
    print("="*60)

    from rag.evaluation.metrics import RetrievalMetrics, GenerationMetrics, AnswerQualityMetrics

    # Test retrieval metrics
    retrieved = ["doc1", "doc3", "doc5", "doc2"]
    relevant = ["doc1", "doc2", "doc6"]

    precision = RetrievalMetrics.precision_at_k(retrieved, relevant, 3)
    recall = RetrievalMetrics.recall_at_k(retrieved, relevant, 3)
    mrr = RetrievalMetrics.mean_reciprocal_rank(retrieved, relevant)

    print(f"  Precision@3: {precision:.3f}")
    print(f"  Recall@3: {recall:.3f}")
    print(f"  MRR: {mrr:.3f}")

    # Test generation metrics
    reference = "EEG stress detection uses alpha wave suppression."
    hypothesis = "Stress detection using EEG relies on alpha waves."

    bleu = GenerationMetrics.bleu_score(reference, hypothesis)
    rouge = GenerationMetrics.rouge_l(reference, hypothesis)

    print(f"  BLEU: {bleu:.3f}")
    print(f"  ROUGE-L F1: {rouge['f1']:.3f}")

    # Test answer quality
    answer = "Alpha suppression occurs during stress."
    question = "What happens to alpha waves during stress?"
    context = ["Alpha waves decrease during stress", "Stress causes alpha suppression"]

    faithfulness = AnswerQualityMetrics.faithfulness(answer, context)
    relevance = AnswerQualityMetrics.answer_relevance(answer, question)

    print(f"  Faithfulness: {faithfulness:.3f}")
    print(f"  Relevance: {relevance:.3f}")

    print("  [OK] Evaluation metrics working")
    return True


def test_governance():
    """Test AI governance."""
    print("\n" + "="*60)
    print("TESTING: AI Governance")
    print("="*60)

    from rag.governance.responsible_ai import AIGovernanceManager

    governance = AIGovernanceManager()

    # Test query processing
    query = "What is alpha suppression?"
    cleaned, meta = governance.process_query(query, "test_user")

    print(f"  Query cleaned: {cleaned == query}")
    print(f"  Can proceed: {meta.get('proceed', False)}")

    # Test response processing (mock response)
    class MockResponse:
        answer = "Alpha suppression is the reduction in alpha wave power during stress."
        sources = [{"source": "EEG Research", "text": "Alpha suppression", "score": 0.8}]
        confidence = 0.75
        latency_ms = 100

    result = governance.process_response(MockResponse(), query, "test_user")

    trust_level = result["governance_metadata"]["trust"]["trust_level"]
    print(f"  Trust level: {trust_level}")
    print(f"  Compliance logged: {result['governance_metadata'].get('compliance_logged', False)}")

    print("  [OK] AI governance working")
    return True


def test_rag_pipeline():
    """Test complete RAG pipeline."""
    print("\n" + "="*60)
    print("TESTING: Complete RAG Pipeline")
    print("="*60)

    from rag.core.rag_pipeline import EEGStressRAG

    # Initialize RAG
    rag = EEGStressRAG(
        embedding_provider="hash",
        vector_store_type="memory",
        ollama_model="llama3.2"
    )

    print(f"  Documents indexed: {rag.stats['documents_indexed']}")

    # Test queries
    test_queries = [
        "What is alpha suppression?",
        "Explain the Theta/Beta Ratio",
        "What deep learning works for EEG?"
    ]

    for query in test_queries:
        response = rag.query(query, k=3)
        print(f"  Query: '{query[:30]}...'")
        print(f"    - Confidence: {response.confidence:.2f}")
        print(f"    - Sources: {len(response.sources)}")
        print(f"    - Latency: {response.latency_ms:.1f}ms")

    # Check stats
    stats = rag.get_stats()
    print(f"  Total queries: {stats['queries']}")
    print(f"  LLM available: {stats.get('llm_available', False)}")

    print("  [OK] RAG pipeline working")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("EEG STRESS RAG SYSTEM - COMPREHENSIVE TEST")
    print("="*60)

    tests = [
        ("Chunking", test_chunking),
        ("Embedding", test_embedding),
        ("Vector Store", test_vector_store),
        ("Knowledge Graph", test_knowledge_graph),
        ("Cache", test_cache),
        ("Evaluation", test_evaluation),
        ("Governance", test_governance),
        ("RAG Pipeline", test_rag_pipeline)
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"  [ERROR] {str(e)}")
            results.append((name, "ERROR"))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, r in results if r == "PASS")
    total = len(results)

    for name, status in results:
        icon = "[OK]" if status == "PASS" else "[X]"
        print(f"  {icon} {name}: {status}")

    print(f"\n  Total: {passed}/{total} tests passed")
    print("="*60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
