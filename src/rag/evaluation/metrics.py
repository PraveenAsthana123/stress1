#!/usr/bin/env python3
"""
RAG Evaluation Metrics Module

Comprehensive metrics for evaluating RAG system performance:
- Retrieval metrics (Precision, Recall, MRR, NDCG)
- Generation metrics (BLEU, ROUGE, BERTScore)
- Answer quality metrics (Faithfulness, Relevance, Coherence)
- System metrics (Latency, Throughput, Cache hit rate)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import json
import time
import re


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metric_name: str
    score: float
    details: Dict
    timestamp: float


class RetrievalMetrics:
    """Metrics for evaluating retrieval quality."""

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        """Calculate Precision@K."""
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)
        relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant_set)
        return relevant_retrieved / k if k > 0 else 0.0

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        """Calculate Recall@K."""
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)
        relevant_retrieved = len(retrieved_k.intersection(relevant_set))
        return relevant_retrieved / len(relevant_set) if relevant_set else 0.0

    @staticmethod
    def f1_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        """Calculate F1@K."""
        precision = RetrievalMetrics.precision_at_k(retrieved, relevant, k)
        recall = RetrievalMetrics.recall_at_k(retrieved, relevant, k)
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    @staticmethod
    def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
        """Calculate Mean Reciprocal Rank (MRR)."""
        relevant_set = set(relevant)
        for i, doc in enumerate(retrieved, 1):
            if doc in relevant_set:
                return 1.0 / i
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: List[str], relevance_scores: Dict[str, float] = None, k: int = 5) -> float:
        """Calculate Normalized Discounted Cumulative Gain (NDCG@K)."""
        if relevance_scores is None:
            relevance_scores = {doc: 1.0 for doc in relevant}

        dcg = 0.0
        for i, doc in enumerate(retrieved[:k]):
            rel = relevance_scores.get(doc, 0.0)
            dcg += rel / np.log2(i + 2)

        # Ideal DCG
        ideal_rels = sorted([relevance_scores.get(doc, 0.0) for doc in relevant], reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def hit_rate(retrieved: List[str], relevant: List[str]) -> float:
        """Calculate Hit Rate (whether any relevant document was retrieved)."""
        relevant_set = set(relevant)
        return 1.0 if any(doc in relevant_set for doc in retrieved) else 0.0

    @staticmethod
    def average_precision(retrieved: List[str], relevant: List[str]) -> float:
        """Calculate Average Precision."""
        relevant_set = set(relevant)
        if not relevant_set:
            return 0.0

        precision_sum = 0.0
        relevant_count = 0

        for i, doc in enumerate(retrieved):
            if doc in relevant_set:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)

        return precision_sum / len(relevant_set)


class GenerationMetrics:
    """Metrics for evaluating generated text quality."""

    @staticmethod
    def bleu_score(reference: str, hypothesis: str, n: int = 4) -> float:
        """Calculate BLEU score (simplified implementation)."""
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

        if len(hyp_tokens) == 0:
            return 0.0

        scores = []
        for i in range(1, min(n + 1, len(hyp_tokens) + 1)):
            ref_ngrams = defaultdict(int)
            hyp_ngrams = defaultdict(int)

            for j in range(len(ref_tokens) - i + 1):
                ngram = tuple(ref_tokens[j:j + i])
                ref_ngrams[ngram] += 1

            for j in range(len(hyp_tokens) - i + 1):
                ngram = tuple(hyp_tokens[j:j + i])
                hyp_ngrams[ngram] += 1

            matches = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
            total = sum(hyp_ngrams.values())
            scores.append(matches / total if total > 0 else 0.0)

        # Brevity penalty
        bp = min(1.0, np.exp(1 - len(ref_tokens) / len(hyp_tokens))) if len(hyp_tokens) > 0 else 0.0

        # Geometric mean
        if all(s > 0 for s in scores):
            return bp * np.exp(np.mean(np.log(scores)))
        return 0.0

    @staticmethod
    def rouge_l(reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate ROUGE-L score."""
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

        if not ref_tokens or not hyp_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Longest Common Subsequence
        m, n = len(ref_tokens), len(hyp_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_length = dp[m][n]
        precision = lcs_length / n if n > 0 else 0.0
        recall = lcs_length / m if m > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    @staticmethod
    def semantic_similarity(text1: str, text2: str, embedding_model=None) -> float:
        """Calculate semantic similarity using embeddings."""
        if embedding_model is None:
            # Fallback: word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0.0

        emb1 = embedding_model.encode(text1)
        emb2 = embedding_model.encode(text2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10))


class AnswerQualityMetrics:
    """Metrics for evaluating answer quality in RAG systems."""

    @staticmethod
    def faithfulness(answer: str, context: List[str]) -> float:
        """
        Measure how faithful the answer is to the provided context.
        Higher score = answer is well-grounded in the context.
        """
        answer_sentences = [s.strip() for s in re.split(r'[.!?]', answer) if s.strip()]
        context_text = ' '.join(context).lower()

        if not answer_sentences:
            return 0.0

        supported_count = 0
        for sentence in answer_sentences:
            # Check if key phrases from sentence appear in context
            words = sentence.lower().split()
            if len(words) >= 3:
                # Check for phrase overlap
                matched = 0
                for i in range(len(words) - 2):
                    phrase = ' '.join(words[i:i + 3])
                    if phrase in context_text:
                        matched += 1
                if matched > 0:
                    supported_count += 1

        return supported_count / len(answer_sentences) if answer_sentences else 0.0

    @staticmethod
    def answer_relevance(answer: str, question: str) -> float:
        """
        Measure how relevant the answer is to the question.
        """
        question_words = set(question.lower().split()) - {'what', 'how', 'why', 'when', 'where', 'is', 'are', 'the', 'a', 'an'}
        answer_words = set(answer.lower().split())

        if not question_words:
            return 1.0

        overlap = question_words.intersection(answer_words)
        return len(overlap) / len(question_words)

    @staticmethod
    def coherence(text: str) -> float:
        """
        Measure coherence of the text.
        Based on sentence connectivity and logical flow.
        """
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]

        if len(sentences) <= 1:
            return 1.0

        # Check for discourse markers and connectives
        connectives = {'however', 'therefore', 'furthermore', 'moreover', 'additionally',
                       'consequently', 'thus', 'hence', 'also', 'first', 'second', 'finally',
                       'in addition', 'as a result', 'for example', 'in conclusion'}

        connective_count = sum(1 for s in sentences if any(c in s.lower() for c in connectives))

        # Check for pronoun references (indicates coherent flow)
        pronouns = {'it', 'this', 'these', 'they', 'them', 'their'}
        pronoun_refs = sum(1 for s in sentences[1:] if any(p in s.lower().split()[:3] for p in pronouns))

        # Combined score
        connective_score = min(connective_count / (len(sentences) - 1), 1.0) if len(sentences) > 1 else 0
        pronoun_score = min(pronoun_refs / (len(sentences) - 1), 1.0) if len(sentences) > 1 else 0

        return 0.5 * connective_score + 0.5 * pronoun_score

    @staticmethod
    def completeness(answer: str, expected_topics: List[str]) -> float:
        """
        Measure if the answer covers all expected topics.
        """
        answer_lower = answer.lower()
        covered = sum(1 for topic in expected_topics if topic.lower() in answer_lower)
        return covered / len(expected_topics) if expected_topics else 1.0


class SystemMetrics:
    """Metrics for system performance."""

    def __init__(self):
        self.latencies = []
        self.query_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.error_count = 0

    def record_query(self, latency_ms: float, cached: bool = False, error: bool = False):
        """Record a query for metrics calculation."""
        self.latencies.append(latency_ms)
        self.query_times.append(time.time())

        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        if error:
            self.error_count += 1

    def get_metrics(self) -> Dict:
        """Get current system metrics."""
        if not self.latencies:
            return {
                "total_queries": 0,
                "avg_latency_ms": 0,
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "cache_hit_rate": 0,
                "error_rate": 0,
                "qps": 0
            }

        latencies = np.array(self.latencies)
        total_queries = len(self.latencies)

        # Calculate QPS (queries per second) over last minute
        recent_queries = [t for t in self.query_times if time.time() - t < 60]
        qps = len(recent_queries) / 60.0 if recent_queries else 0

        return {
            "total_queries": total_queries,
            "avg_latency_ms": float(np.mean(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "cache_hit_rate": self.cache_hits / total_queries if total_queries > 0 else 0,
            "error_rate": self.error_count / total_queries if total_queries > 0 else 0,
            "qps": qps
        }


class RAGEvaluator:
    """Comprehensive RAG evaluation framework."""

    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()
        self.answer_quality = AnswerQualityMetrics()
        self.system_metrics = SystemMetrics()
        self.evaluation_history = []

    def evaluate_retrieval(self, retrieved_docs: List[str], relevant_docs: List[str], k: int = 5) -> Dict:
        """Evaluate retrieval performance."""
        return {
            "precision@k": RetrievalMetrics.precision_at_k(retrieved_docs, relevant_docs, k),
            "recall@k": RetrievalMetrics.recall_at_k(retrieved_docs, relevant_docs, k),
            "f1@k": RetrievalMetrics.f1_at_k(retrieved_docs, relevant_docs, k),
            "mrr": RetrievalMetrics.mean_reciprocal_rank(retrieved_docs, relevant_docs),
            "ndcg@k": RetrievalMetrics.ndcg_at_k(retrieved_docs, relevant_docs, k=k),
            "hit_rate": RetrievalMetrics.hit_rate(retrieved_docs, relevant_docs),
            "map": RetrievalMetrics.average_precision(retrieved_docs, relevant_docs)
        }

    def evaluate_generation(self, reference: str, generated: str) -> Dict:
        """Evaluate generation quality."""
        rouge = GenerationMetrics.rouge_l(reference, generated)
        return {
            "bleu": GenerationMetrics.bleu_score(reference, generated),
            "rouge_l_precision": rouge["precision"],
            "rouge_l_recall": rouge["recall"],
            "rouge_l_f1": rouge["f1"],
            "semantic_similarity": GenerationMetrics.semantic_similarity(reference, generated, self.embedding_model)
        }

    def evaluate_answer_quality(self, answer: str, question: str, context: List[str], expected_topics: List[str] = None) -> Dict:
        """Evaluate answer quality."""
        return {
            "faithfulness": AnswerQualityMetrics.faithfulness(answer, context),
            "relevance": AnswerQualityMetrics.answer_relevance(answer, question),
            "coherence": AnswerQualityMetrics.coherence(answer),
            "completeness": AnswerQualityMetrics.completeness(answer, expected_topics or [])
        }

    def evaluate_response(self, response: Any, ground_truth: Dict = None) -> Dict:
        """Comprehensive evaluation of a RAG response."""
        results = {
            "timestamp": time.time(),
            "metrics": {}
        }

        # Record system metrics
        if hasattr(response, 'latency_ms'):
            self.system_metrics.record_query(
                response.latency_ms,
                cached=getattr(response, 'cached', False)
            )
            results["metrics"]["latency_ms"] = response.latency_ms

        # Evaluate answer quality
        if hasattr(response, 'answer') and ground_truth:
            answer = response.answer
            question = ground_truth.get('question', '')
            context = [s.get('text', '') for s in getattr(response, 'sources', [])]

            results["metrics"]["answer_quality"] = self.evaluate_answer_quality(
                answer, question, context,
                ground_truth.get('expected_topics', [])
            )

            # If reference answer provided
            if 'reference_answer' in ground_truth:
                results["metrics"]["generation"] = self.evaluate_generation(
                    ground_truth['reference_answer'], answer
                )

        # Calculate overall score
        if results["metrics"]:
            scores = []
            if "answer_quality" in results["metrics"]:
                aq = results["metrics"]["answer_quality"]
                scores.extend([aq.get("faithfulness", 0), aq.get("relevance", 0), aq.get("coherence", 0)])

            results["overall_score"] = np.mean(scores) if scores else 0.0

        self.evaluation_history.append(results)
        return results

    def get_aggregate_metrics(self) -> Dict:
        """Get aggregate metrics across all evaluations."""
        if not self.evaluation_history:
            return {}

        all_scores = [e.get("overall_score", 0) for e in self.evaluation_history]

        return {
            "total_evaluations": len(self.evaluation_history),
            "avg_overall_score": np.mean(all_scores),
            "score_std": np.std(all_scores),
            "system_metrics": self.system_metrics.get_metrics()
        }

    def export_report(self, filepath: str):
        """Export evaluation report to JSON."""
        report = {
            "summary": self.get_aggregate_metrics(),
            "history": self.evaluation_history
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
    # Test evaluation metrics
    print("Testing RAG Evaluation Metrics\n" + "="*50)

    # Test retrieval metrics
    retrieved = ["doc1", "doc3", "doc5", "doc2", "doc4"]
    relevant = ["doc1", "doc2", "doc6"]

    print("\nRetrieval Metrics:")
    print(f"  Precision@3: {RetrievalMetrics.precision_at_k(retrieved, relevant, 3):.3f}")
    print(f"  Recall@3: {RetrievalMetrics.recall_at_k(retrieved, relevant, 3):.3f}")
    print(f"  F1@3: {RetrievalMetrics.f1_at_k(retrieved, relevant, 3):.3f}")
    print(f"  MRR: {RetrievalMetrics.mean_reciprocal_rank(retrieved, relevant):.3f}")
    print(f"  NDCG@5: {RetrievalMetrics.ndcg_at_k(retrieved, relevant, k=5):.3f}")

    # Test generation metrics
    reference = "EEG-based stress detection uses alpha wave suppression as a key biomarker."
    hypothesis = "Stress detection using EEG relies on alpha wave patterns and suppression markers."

    print("\nGeneration Metrics:")
    print(f"  BLEU: {GenerationMetrics.bleu_score(reference, hypothesis):.3f}")
    rouge = GenerationMetrics.rouge_l(reference, hypothesis)
    print(f"  ROUGE-L F1: {rouge['f1']:.3f}")

    # Test answer quality
    answer = "Alpha suppression occurs during stress. This is because the brain's relaxation patterns are disrupted. Therefore, monitoring alpha waves is important for stress detection."
    question = "What happens to alpha waves during stress?"
    context = ["Alpha waves decrease during stress", "Stress causes alpha suppression", "EEG measures brain activity"]

    print("\nAnswer Quality Metrics:")
    print(f"  Faithfulness: {AnswerQualityMetrics.faithfulness(answer, context):.3f}")
    print(f"  Relevance: {AnswerQualityMetrics.answer_relevance(answer, question):.3f}")
    print(f"  Coherence: {AnswerQualityMetrics.coherence(answer):.3f}")

    print("\n" + "="*50 + "\nAll tests completed!")
