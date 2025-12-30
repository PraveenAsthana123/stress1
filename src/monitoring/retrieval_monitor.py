"""
Phase 2: Representation & Retrieval Analysis Monitoring

Monitors chunking quality, embedding drift, and retrieval performance
for RAG systems.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import hashlib
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================

class ChunkQuality(Enum):
    """Chunk quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


class DriftSeverity(Enum):
    """Embedding drift severity."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RetrievalQuality(Enum):
    """Retrieval quality levels."""
    EXCELLENT = "excellent"  # Precision@K > 0.9
    GOOD = "good"           # Precision@K > 0.7
    ACCEPTABLE = "acceptable"  # Precision@K > 0.5
    POOR = "poor"           # Precision@K <= 0.5


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ChunkStats:
    """Statistics for a chunk."""
    chunk_id: str
    token_count: int
    char_count: int
    sentence_count: int
    has_complete_sentences: bool
    semantic_coherence: float  # 0-1
    quality: ChunkQuality


@dataclass
class ChunkingValidationResult:
    """Result of chunking validation."""
    total_chunks: int
    avg_token_count: float
    avg_coherence: float
    quality_distribution: Dict[ChunkQuality, int]
    issues: List[Dict[str, Any]]
    passed: bool
    recommendations: List[str]


@dataclass
class EmbeddingSnapshot:
    """Snapshot of embedding distribution."""
    snapshot_id: str
    timestamp: datetime
    mean_vector: np.ndarray
    std_vector: np.ndarray
    sample_size: int
    checksum: str


@dataclass
class DriftReport:
    """Report of embedding drift."""
    baseline_snapshot: str
    current_snapshot: str
    cosine_drift: float
    euclidean_drift: float
    dimension_drifts: List[float]
    severity: DriftSeverity
    requires_reindex: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RetrievalMetrics:
    """Retrieval performance metrics."""
    query_id: str
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    mrr: float
    latency_ms: float
    k: int


@dataclass
class RetrievalQualityReport:
    """Overall retrieval quality report."""
    total_queries: int
    avg_precision: float
    avg_recall: float
    avg_ndcg: float
    avg_mrr: float
    avg_latency_ms: float
    quality_level: RetrievalQuality
    passed: bool
    issues: List[str]


# =============================================================================
# Phase 2.1: Chunking Validator
# =============================================================================

class ChunkingValidator:
    """
    Validates chunking quality and consistency.

    Purpose: Ensure chunks are semantically coherent and properly sized
    Measurement: Token count distribution, coherence scores, boundary quality
    Pass/Fail: >90% chunks meet quality criteria
    """

    def __init__(
        self,
        min_tokens: int = 50,
        max_tokens: int = 512,
        target_tokens: int = 256,
        min_coherence: float = 0.6
    ):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.target_tokens = target_tokens
        self.min_coherence = min_coherence

    def validate_chunk(
        self,
        chunk_id: str,
        text: str,
        embedding: Optional[np.ndarray] = None
    ) -> ChunkStats:
        """Validate a single chunk."""
        # Count tokens (approximate)
        token_count = len(text.split())
        char_count = len(text)

        # Count sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        sentence_count = len(sentences)

        # Check for complete sentences
        has_complete = text.strip().endswith(('.', '!', '?'))

        # Calculate coherence (simplified - would use embeddings in production)
        coherence = self._estimate_coherence(text, embedding)

        # Determine quality
        quality = self._assess_quality(
            token_count, has_complete, coherence
        )

        return ChunkStats(
            chunk_id=chunk_id,
            token_count=token_count,
            char_count=char_count,
            sentence_count=sentence_count,
            has_complete_sentences=has_complete,
            semantic_coherence=coherence,
            quality=quality
        )

    def _estimate_coherence(
        self,
        text: str,
        embedding: Optional[np.ndarray]
    ) -> float:
        """Estimate semantic coherence of chunk."""
        # Simplified coherence estimation
        # In production, use sentence embeddings similarity

        # Check for topic drift indicators
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) <= 1:
            return 0.9  # Single sentence = coherent

        # Check for coherence indicators
        coherence = 0.8  # Base coherence

        # Penalize very long chunks
        if len(text.split()) > self.max_tokens:
            coherence -= 0.2

        # Penalize chunks with many topics
        unique_words = len(set(text.lower().split()))
        total_words = len(text.split())
        if total_words > 0:
            diversity = unique_words / total_words
            if diversity > 0.9:  # Very diverse vocabulary
                coherence -= 0.1

        return max(0.0, min(1.0, coherence))

    def _assess_quality(
        self,
        token_count: int,
        has_complete: bool,
        coherence: float
    ) -> ChunkQuality:
        """Assess chunk quality."""
        score = 0

        # Token count scoring
        if self.min_tokens <= token_count <= self.max_tokens:
            if abs(token_count - self.target_tokens) < 50:
                score += 2
            else:
                score += 1

        # Completeness scoring
        if has_complete:
            score += 1

        # Coherence scoring
        if coherence >= 0.8:
            score += 2
        elif coherence >= self.min_coherence:
            score += 1

        # Map score to quality
        if score >= 5:
            return ChunkQuality.EXCELLENT
        elif score >= 4:
            return ChunkQuality.GOOD
        elif score >= 2:
            return ChunkQuality.ACCEPTABLE
        else:
            return ChunkQuality.POOR

    def validate_all(
        self,
        chunks: List[Tuple[str, str, Optional[np.ndarray]]]
    ) -> ChunkingValidationResult:
        """Validate all chunks."""
        stats = [
            self.validate_chunk(chunk_id, text, emb)
            for chunk_id, text, emb in chunks
        ]

        # Calculate aggregates
        total = len(stats)
        avg_tokens = np.mean([s.token_count for s in stats]) if stats else 0
        avg_coherence = np.mean([s.semantic_coherence for s in stats]) if stats else 0

        # Quality distribution
        quality_dist = {q: 0 for q in ChunkQuality}
        for s in stats:
            quality_dist[s.quality] += 1

        # Identify issues
        issues = []
        for s in stats:
            if s.quality == ChunkQuality.POOR:
                issues.append({
                    "chunk_id": s.chunk_id,
                    "issue": "poor_quality",
                    "details": {
                        "tokens": s.token_count,
                        "coherence": s.semantic_coherence,
                        "complete": s.has_complete_sentences
                    }
                })

        # Calculate pass rate
        good_chunks = quality_dist[ChunkQuality.EXCELLENT] + \
                     quality_dist[ChunkQuality.GOOD] + \
                     quality_dist[ChunkQuality.ACCEPTABLE]
        pass_rate = good_chunks / total if total else 0
        passed = pass_rate >= 0.9

        # Generate recommendations
        recommendations = []
        if avg_tokens < self.min_tokens:
            recommendations.append("Increase chunk size")
        elif avg_tokens > self.max_tokens:
            recommendations.append("Decrease chunk size")

        if avg_coherence < self.min_coherence:
            recommendations.append("Improve chunking boundaries")

        if quality_dist[ChunkQuality.POOR] > total * 0.1:
            recommendations.append("Review poor quality chunks")

        return ChunkingValidationResult(
            total_chunks=total,
            avg_token_count=float(avg_tokens),
            avg_coherence=float(avg_coherence),
            quality_distribution=quality_dist,
            issues=issues,
            passed=passed,
            recommendations=recommendations
        )


# =============================================================================
# Phase 2.2: Embedding Drift Detector
# =============================================================================

class EmbeddingDriftDetector:
    """
    Detects drift in embedding distributions over time.

    Purpose: Monitor embedding quality and distribution changes
    Measurement: Cosine drift, Euclidean drift, dimension-wise changes
    Pass/Fail: Drift < threshold (typically 0.1 for cosine)
    """

    def __init__(
        self,
        cosine_threshold: float = 0.1,
        euclidean_threshold: float = 0.5,
        dimension_threshold: float = 0.2
    ):
        self.cosine_threshold = cosine_threshold
        self.euclidean_threshold = euclidean_threshold
        self.dimension_threshold = dimension_threshold
        self.snapshots: Dict[str, EmbeddingSnapshot] = {}
        self.baseline_id: Optional[str] = None

    def create_snapshot(
        self,
        embeddings: np.ndarray,
        snapshot_id: Optional[str] = None
    ) -> EmbeddingSnapshot:
        """Create a snapshot of embedding distribution."""
        if snapshot_id is None:
            snapshot_id = hashlib.md5(
                f"{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]

        mean_vec = np.mean(embeddings, axis=0)
        std_vec = np.std(embeddings, axis=0)

        # Create checksum for reproducibility
        checksum = hashlib.md5(
            embeddings.tobytes()
        ).hexdigest()

        snapshot = EmbeddingSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now(),
            mean_vector=mean_vec,
            std_vector=std_vec,
            sample_size=len(embeddings),
            checksum=checksum
        )

        self.snapshots[snapshot_id] = snapshot
        return snapshot

    def set_baseline(self, snapshot_id: str) -> bool:
        """Set baseline snapshot for drift comparison."""
        if snapshot_id not in self.snapshots:
            return False
        self.baseline_id = snapshot_id
        return True

    def detect_drift(
        self,
        current_embeddings: np.ndarray,
        baseline_id: Optional[str] = None
    ) -> DriftReport:
        """Detect drift between current embeddings and baseline."""
        baseline_id = baseline_id or self.baseline_id
        if not baseline_id or baseline_id not in self.snapshots:
            raise ValueError("No baseline snapshot available")

        baseline = self.snapshots[baseline_id]
        current = self.create_snapshot(current_embeddings)

        # Calculate drifts
        cosine_drift = self._cosine_distance(
            baseline.mean_vector, current.mean_vector
        )
        euclidean_drift = np.linalg.norm(
            baseline.mean_vector - current.mean_vector
        )

        # Per-dimension drift
        dimension_drifts = np.abs(
            baseline.mean_vector - current.mean_vector
        ).tolist()

        # Determine severity
        severity = self._assess_severity(cosine_drift, euclidean_drift)

        # Determine if reindexing needed
        requires_reindex = severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]

        return DriftReport(
            baseline_snapshot=baseline_id,
            current_snapshot=current.snapshot_id,
            cosine_drift=float(cosine_drift),
            euclidean_drift=float(euclidean_drift),
            dimension_drifts=dimension_drifts,
            severity=severity,
            requires_reindex=requires_reindex
        )

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine distance between vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        similarity = dot / (norm_a * norm_b)
        return float(1 - similarity)

    def _assess_severity(
        self,
        cosine_drift: float,
        euclidean_drift: float
    ) -> DriftSeverity:
        """Assess drift severity."""
        if cosine_drift < self.cosine_threshold * 0.5:
            return DriftSeverity.NONE
        elif cosine_drift < self.cosine_threshold:
            return DriftSeverity.LOW
        elif cosine_drift < self.cosine_threshold * 2:
            return DriftSeverity.MEDIUM
        elif cosine_drift < self.cosine_threshold * 4:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL

    def get_drift_history(self) -> List[DriftReport]:
        """Get history of drift measurements."""
        # Would store drift reports in production
        return []


# =============================================================================
# Phase 2.3: Retrieval Quality Analyzer
# =============================================================================

class RetrievalQualityAnalyzer:
    """
    Analyzes retrieval quality and performance.

    Purpose: Measure retrieval accuracy and relevance
    Measurement: Precision@K, Recall@K, NDCG, MRR, latency
    Pass/Fail: Precision@5 > 0.7, Latency < 200ms
    """

    def __init__(
        self,
        precision_threshold: float = 0.7,
        latency_threshold_ms: float = 200.0,
        default_k: int = 5
    ):
        self.precision_threshold = precision_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.default_k = default_k
        self.metrics_history: List[RetrievalMetrics] = []

    def evaluate_retrieval(
        self,
        query_id: str,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        latency_ms: float,
        k: Optional[int] = None
    ) -> RetrievalMetrics:
        """Evaluate a single retrieval."""
        k = k or self.default_k
        retrieved_at_k = retrieved_ids[:k]

        # Calculate metrics
        precision = self._precision_at_k(retrieved_at_k, relevant_ids)
        recall = self._recall_at_k(retrieved_at_k, relevant_ids)
        ndcg = self._ndcg_at_k(retrieved_ids, relevant_ids, k)
        mrr = self._mrr(retrieved_ids, relevant_ids)

        metrics = RetrievalMetrics(
            query_id=query_id,
            precision_at_k=precision,
            recall_at_k=recall,
            ndcg_at_k=ndcg,
            mrr=mrr,
            latency_ms=latency_ms,
            k=k
        )

        self.metrics_history.append(metrics)
        return metrics

    def _precision_at_k(
        self,
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """Calculate Precision@K."""
        if not retrieved:
            return 0.0
        relevant_set = set(relevant)
        hits = sum(1 for r in retrieved if r in relevant_set)
        return hits / len(retrieved)

    def _recall_at_k(
        self,
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """Calculate Recall@K."""
        if not relevant:
            return 1.0
        relevant_set = set(relevant)
        hits = sum(1 for r in retrieved if r in relevant_set)
        return hits / len(relevant)

    def _ndcg_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """Calculate NDCG@K."""
        relevant_set = set(relevant)

        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant_set:
                dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0

        # Ideal DCG
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

        return dcg / idcg if idcg > 0 else 0.0

    def _mrr(
        self,
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """Calculate Mean Reciprocal Rank."""
        relevant_set = set(relevant)
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    def get_quality_report(self) -> RetrievalQualityReport:
        """Generate quality report from metrics history."""
        if not self.metrics_history:
            return RetrievalQualityReport(
                total_queries=0,
                avg_precision=0.0,
                avg_recall=0.0,
                avg_ndcg=0.0,
                avg_mrr=0.0,
                avg_latency_ms=0.0,
                quality_level=RetrievalQuality.POOR,
                passed=False,
                issues=["No metrics collected"]
            )

        # Calculate averages
        avg_precision = np.mean([m.precision_at_k for m in self.metrics_history])
        avg_recall = np.mean([m.recall_at_k for m in self.metrics_history])
        avg_ndcg = np.mean([m.ndcg_at_k for m in self.metrics_history])
        avg_mrr = np.mean([m.mrr for m in self.metrics_history])
        avg_latency = np.mean([m.latency_ms for m in self.metrics_history])

        # Determine quality level
        if avg_precision >= 0.9:
            quality = RetrievalQuality.EXCELLENT
        elif avg_precision >= 0.7:
            quality = RetrievalQuality.GOOD
        elif avg_precision >= 0.5:
            quality = RetrievalQuality.ACCEPTABLE
        else:
            quality = RetrievalQuality.POOR

        # Check pass criteria
        precision_passed = avg_precision >= self.precision_threshold
        latency_passed = avg_latency <= self.latency_threshold_ms
        passed = precision_passed and latency_passed

        # Identify issues
        issues = []
        if not precision_passed:
            issues.append(
                f"Precision {avg_precision:.2%} below threshold {self.precision_threshold:.2%}"
            )
        if not latency_passed:
            issues.append(
                f"Latency {avg_latency:.0f}ms above threshold {self.latency_threshold_ms:.0f}ms"
            )

        return RetrievalQualityReport(
            total_queries=len(self.metrics_history),
            avg_precision=float(avg_precision),
            avg_recall=float(avg_recall),
            avg_ndcg=float(avg_ndcg),
            avg_mrr=float(avg_mrr),
            avg_latency_ms=float(avg_latency),
            quality_level=quality,
            passed=passed,
            issues=issues
        )

    def reset_metrics(self):
        """Reset metrics history."""
        self.metrics_history = []


# =============================================================================
# Phase 2: Unified Retrieval Monitor
# =============================================================================

class RetrievalPhaseMonitor:
    """
    Unified monitor for Phase 2: Representation & Retrieval Analysis.

    Combines chunking, embedding, and retrieval monitoring.
    """

    def __init__(self):
        self.chunking_validator = ChunkingValidator()
        self.drift_detector = EmbeddingDriftDetector()
        self.quality_analyzer = RetrievalQualityAnalyzer()

    def run_full_analysis(
        self,
        chunks: List[Tuple[str, str, Optional[np.ndarray]]],
        embeddings: Optional[np.ndarray] = None,
        retrieval_results: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Run full Phase 2 analysis."""
        results = {}

        # Chunking validation
        chunking_result = self.chunking_validator.validate_all(chunks)
        results["chunking"] = {
            "total_chunks": chunking_result.total_chunks,
            "avg_tokens": chunking_result.avg_token_count,
            "avg_coherence": chunking_result.avg_coherence,
            "quality_distribution": {
                q.value: c for q, c in chunking_result.quality_distribution.items()
            },
            "passed": chunking_result.passed,
            "issues_count": len(chunking_result.issues),
            "recommendations": chunking_result.recommendations
        }

        # Embedding drift (if embeddings provided)
        if embeddings is not None:
            if self.drift_detector.baseline_id is None:
                # Create baseline
                snapshot = self.drift_detector.create_snapshot(embeddings, "baseline")
                self.drift_detector.set_baseline("baseline")
                results["embedding_drift"] = {
                    "baseline_created": True,
                    "snapshot_id": snapshot.snapshot_id,
                    "sample_size": snapshot.sample_size
                }
            else:
                # Detect drift
                drift_report = self.drift_detector.detect_drift(embeddings)
                results["embedding_drift"] = {
                    "cosine_drift": drift_report.cosine_drift,
                    "euclidean_drift": drift_report.euclidean_drift,
                    "severity": drift_report.severity.value,
                    "requires_reindex": drift_report.requires_reindex,
                    "passed": drift_report.severity in [
                        DriftSeverity.NONE, DriftSeverity.LOW
                    ]
                }

        # Retrieval quality (if results provided)
        if retrieval_results:
            for result in retrieval_results:
                self.quality_analyzer.evaluate_retrieval(
                    query_id=result.get("query_id", "unknown"),
                    retrieved_ids=result.get("retrieved_ids", []),
                    relevant_ids=result.get("relevant_ids", []),
                    latency_ms=result.get("latency_ms", 0.0)
                )

            quality_report = self.quality_analyzer.get_quality_report()
            results["retrieval_quality"] = {
                "total_queries": quality_report.total_queries,
                "avg_precision": quality_report.avg_precision,
                "avg_recall": quality_report.avg_recall,
                "avg_ndcg": quality_report.avg_ndcg,
                "avg_mrr": quality_report.avg_mrr,
                "avg_latency_ms": quality_report.avg_latency_ms,
                "quality_level": quality_report.quality_level.value,
                "passed": quality_report.passed,
                "issues": quality_report.issues
            }

        # Overall pass status
        results["overall_passed"] = all([
            results.get("chunking", {}).get("passed", True),
            results.get("embedding_drift", {}).get("passed", True),
            results.get("retrieval_quality", {}).get("passed", True)
        ])

        return results


__all__ = [
    # Enums
    "ChunkQuality",
    "DriftSeverity",
    "RetrievalQuality",
    # Data classes
    "ChunkStats",
    "ChunkingValidationResult",
    "EmbeddingSnapshot",
    "DriftReport",
    "RetrievalMetrics",
    "RetrievalQualityReport",
    # Monitors
    "ChunkingValidator",
    "EmbeddingDriftDetector",
    "RetrievalQualityAnalyzer",
    "RetrievalPhaseMonitor",
]
