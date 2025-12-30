"""
A12) Cost Governance + Degrade Modes for EEG-RAG

Comprehensive module for:
- Unit economics targets ($/query, tokens, latency)
- Cost driver breakdown
- Performance SLOs
- Cost tiers by intent
- Quality floors
- Token budgeting
- Cache strategies
- Graceful degradation ladder
- Cost monitoring and alerting

This keeps latency + cost per query predictable.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)


class CostTier(Enum):
    """Cost tiers by query complexity."""
    T0_SIMPLE = "simple"  # Fast lookup, minimal processing
    T1_STANDARD = "standard"  # Normal RAG flow
    T2_COMPLEX = "complex"  # Multi-hop, comparison
    T3_VERIFIED = "verified"  # Full verification pipeline


class DegradationLevel(Enum):
    """Degradation levels in order of severity."""
    FULL = "full"  # Full pipeline
    REDUCED_K = "reduced_k"  # Fewer retrieval results
    NO_HYDE = "no_hyde"  # Skip hypothetical document expansion
    NO_RERANK = "no_rerank"  # Skip reranking
    SMALLER_CONTEXT = "smaller_context"  # Reduce context window
    SUMMARY_ONLY = "summary_only"  # Return cached/summarized
    ABSTAIN = "abstain"  # Cannot answer


@dataclass
class CostBudget:
    """Cost budget per query."""
    max_tokens: int
    max_latency_ms: float
    max_rerank_calls: int
    max_cost_usd: float
    quality_floor_groundedness: float
    quality_floor_ors: float


@dataclass
class CostMetrics:
    """Observed cost metrics for a query."""
    query_id: str
    tokens_embedding: int
    tokens_generation: int
    tokens_total: int
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float
    total_latency_ms: float
    rerank_calls: int
    estimated_cost_usd: float
    cache_hit: bool
    degradation_level: DegradationLevel


@dataclass
class QualityFloor:
    """Minimum acceptable quality."""
    min_groundedness: float
    min_ors_rag: float
    max_unsupported_claims: float


class CostModel:
    """Model for estimating query costs."""

    # Pricing per 1K tokens (example rates)
    PRICING = {
        'embedding': 0.0001,  # $/1K tokens
        'generation_input': 0.001,
        'generation_output': 0.002,
        'rerank': 0.0005,
    }

    def estimate_cost(self, metrics: CostMetrics) -> float:
        """Estimate cost in USD."""
        cost = 0.0
        cost += (metrics.tokens_embedding / 1000) * self.PRICING['embedding']
        cost += (metrics.tokens_generation / 1000) * self.PRICING['generation_output']
        cost += metrics.rerank_calls * 0.001  # Per call cost

        return cost

    def breakdown(self, metrics: CostMetrics) -> Dict[str, float]:
        """Break down cost by component."""
        return {
            'embedding': (metrics.tokens_embedding / 1000) * self.PRICING['embedding'],
            'generation': (metrics.tokens_generation / 1000) * self.PRICING['generation_output'],
            'rerank': metrics.rerank_calls * 0.001,
            'total': self.estimate_cost(metrics),
        }


class PerformanceSLO:
    """Performance SLOs."""

    def __init__(self):
        self.slos = {
            'latency_p50_ms': 500,
            'latency_p95_ms': 2000,
            'latency_p99_ms': 5000,
            'throughput_qps': 10,
            'error_rate': 0.01,
            'cost_per_query': 0.05,
        }

    def check_compliance(self, metrics: List[CostMetrics]) -> Dict[str, Any]:
        """Check SLO compliance."""
        if not metrics:
            return {'compliant': True, 'violations': []}

        latencies = [m.total_latency_ms for m in metrics]
        costs = [m.estimated_cost_usd for m in metrics]

        violations = []

        p50 = np.percentile(latencies, 50)
        if p50 > self.slos['latency_p50_ms']:
            violations.append(f"P50 latency: {p50:.0f}ms > {self.slos['latency_p50_ms']}ms")

        p95 = np.percentile(latencies, 95)
        if p95 > self.slos['latency_p95_ms']:
            violations.append(f"P95 latency: {p95:.0f}ms > {self.slos['latency_p95_ms']}ms")

        avg_cost = np.mean(costs)
        if avg_cost > self.slos['cost_per_query']:
            violations.append(f"Avg cost: ${avg_cost:.4f} > ${self.slos['cost_per_query']}")

        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'metrics': {
                'latency_p50': p50,
                'latency_p95': p95,
                'avg_cost': avg_cost,
            },
        }


class IntentCostMapper:
    """Map query intent to cost tier."""

    INTENT_MAP = {
        # Simple lookups
        'definition': CostTier.T0_SIMPLE,
        'glossary': CostTier.T0_SIMPLE,

        # Standard queries
        'procedure': CostTier.T1_STANDARD,
        'explanation': CostTier.T1_STANDARD,
        'troubleshooting': CostTier.T1_STANDARD,

        # Complex queries
        'comparison': CostTier.T2_COMPLEX,
        'multi_step': CostTier.T2_COMPLEX,

        # High-risk requiring verification
        'parameter': CostTier.T3_VERIFIED,
        'clinical_adjacent': CostTier.T3_VERIFIED,
    }

    def get_tier(self, intent: str) -> CostTier:
        """Get cost tier for intent."""
        return self.INTENT_MAP.get(intent, CostTier.T1_STANDARD)

    def get_budget(self, tier: CostTier) -> CostBudget:
        """Get budget for tier."""
        budgets = {
            CostTier.T0_SIMPLE: CostBudget(
                max_tokens=1000,
                max_latency_ms=500,
                max_rerank_calls=0,
                max_cost_usd=0.01,
                quality_floor_groundedness=0.7,
                quality_floor_ors=0.7,
            ),
            CostTier.T1_STANDARD: CostBudget(
                max_tokens=2000,
                max_latency_ms=1500,
                max_rerank_calls=1,
                max_cost_usd=0.03,
                quality_floor_groundedness=0.8,
                quality_floor_ors=0.8,
            ),
            CostTier.T2_COMPLEX: CostBudget(
                max_tokens=4000,
                max_latency_ms=3000,
                max_rerank_calls=2,
                max_cost_usd=0.08,
                quality_floor_groundedness=0.85,
                quality_floor_ors=0.85,
            ),
            CostTier.T3_VERIFIED: CostBudget(
                max_tokens=6000,
                max_latency_ms=5000,
                max_rerank_calls=3,
                max_cost_usd=0.15,
                quality_floor_groundedness=0.95,
                quality_floor_ors=0.9,
            ),
        }
        return budgets.get(tier, budgets[CostTier.T1_STANDARD])


class TokenBudgetManager:
    """Manage token budgets for context packing."""

    def __init__(self, max_context_tokens: int = 4000):
        self.max_context = max_context_tokens
        self.allocations = {
            'system': 200,
            'question': 100,
            'evidence': 3000,
            'output_reserve': 500,
        }

    def allocate(self, budget: CostBudget) -> Dict[str, int]:
        """Allocate tokens based on budget."""
        # Scale allocations based on budget
        scale = min(1.0, budget.max_tokens / 4000)

        return {
            'system': int(self.allocations['system']),
            'question': int(self.allocations['question']),
            'evidence': int(self.allocations['evidence'] * scale),
            'output_reserve': int(self.allocations['output_reserve'] * scale),
        }

    def pack_evidence(
        self,
        chunks: List[Dict[str, Any]],
        max_tokens: int
    ) -> List[Dict[str, Any]]:
        """Pack evidence chunks within token budget."""
        packed = []
        total_tokens = 0

        # Sort by relevance score
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.get('score', 0),
            reverse=True
        )

        for chunk in sorted_chunks:
            chunk_tokens = len(chunk.get('text', '').split()) * 1.3  # Rough estimate
            if total_tokens + chunk_tokens <= max_tokens:
                packed.append(chunk)
                total_tokens += chunk_tokens
            else:
                break

        return packed


class CacheManager:
    """Cost-aware caching."""

    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl_seconds
        self.stats = {
            'hits': 0,
            'misses': 0,
        }

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get from cache."""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                self.stats['hits'] += 1
                return entry['value']
            else:
                del self.cache[key]

        self.stats['misses'] += 1
        return None

    def set(self, key: str, value: Dict[str, Any]):
        """Set in cache."""
        self.cache[key] = {
            'value': value,
            'timestamp': time.time(),
        }

    def invalidate_by_version(self, version: str):
        """Invalidate entries by version."""
        keys_to_delete = [
            k for k, v in self.cache.items()
            if v.get('value', {}).get('version') != version
        ]
        for key in keys_to_delete:
            del self.cache[key]

    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.stats['hits'] + self.stats['misses']
        return self.stats['hits'] / max(1, total)


class DegradationLadder:
    """Define graceful degradation steps."""

    LADDER = [
        DegradationLevel.FULL,
        DegradationLevel.REDUCED_K,
        DegradationLevel.NO_HYDE,
        DegradationLevel.NO_RERANK,
        DegradationLevel.SMALLER_CONTEXT,
        DegradationLevel.SUMMARY_ONLY,
        DegradationLevel.ABSTAIN,
    ]

    def __init__(self):
        self.current_level = DegradationLevel.FULL

    def get_config(self, level: DegradationLevel) -> Dict[str, Any]:
        """Get configuration for degradation level."""
        configs = {
            DegradationLevel.FULL: {
                'top_k': 10,
                'use_hyde': True,
                'use_rerank': True,
                'max_context_tokens': 4000,
                'use_cache': True,
            },
            DegradationLevel.REDUCED_K: {
                'top_k': 5,
                'use_hyde': True,
                'use_rerank': True,
                'max_context_tokens': 4000,
                'use_cache': True,
            },
            DegradationLevel.NO_HYDE: {
                'top_k': 5,
                'use_hyde': False,
                'use_rerank': True,
                'max_context_tokens': 3000,
                'use_cache': True,
            },
            DegradationLevel.NO_RERANK: {
                'top_k': 5,
                'use_hyde': False,
                'use_rerank': False,
                'max_context_tokens': 2500,
                'use_cache': True,
            },
            DegradationLevel.SMALLER_CONTEXT: {
                'top_k': 3,
                'use_hyde': False,
                'use_rerank': False,
                'max_context_tokens': 1500,
                'use_cache': True,
            },
            DegradationLevel.SUMMARY_ONLY: {
                'top_k': 0,
                'use_hyde': False,
                'use_rerank': False,
                'max_context_tokens': 500,
                'use_cache': True,
                'summary_mode': True,
            },
            DegradationLevel.ABSTAIN: {
                'abstain': True,
            },
        }
        return configs.get(level, configs[DegradationLevel.FULL])

    def step_down(self) -> DegradationLevel:
        """Step down one level."""
        current_idx = self.LADDER.index(self.current_level)
        if current_idx < len(self.LADDER) - 1:
            self.current_level = self.LADDER[current_idx + 1]
        return self.current_level

    def reset(self):
        """Reset to full level."""
        self.current_level = DegradationLevel.FULL


class DegradationTrigger:
    """Trigger conditions for degradation."""

    def __init__(
        self,
        latency_threshold_ms: float = 2000,
        cost_threshold_usd: float = 0.1,
        gpu_util_threshold: float = 0.9,
        queue_depth_threshold: int = 50
    ):
        self.latency_threshold = latency_threshold_ms
        self.cost_threshold = cost_threshold_usd
        self.gpu_threshold = gpu_util_threshold
        self.queue_threshold = queue_depth_threshold

    def should_degrade(
        self,
        current_latency_ms: float,
        current_cost_usd: float,
        gpu_utilization: float = 0.0,
        queue_depth: int = 0
    ) -> Tuple[bool, List[str]]:
        """Check if degradation should be triggered."""
        triggers = []

        if current_latency_ms > self.latency_threshold:
            triggers.append(f"latency: {current_latency_ms:.0f}ms > {self.latency_threshold}ms")

        if current_cost_usd > self.cost_threshold:
            triggers.append(f"cost: ${current_cost_usd:.4f} > ${self.cost_threshold}")

        if gpu_utilization > self.gpu_threshold:
            triggers.append(f"GPU: {gpu_utilization:.1%} > {self.gpu_threshold:.1%}")

        if queue_depth > self.queue_threshold:
            triggers.append(f"queue: {queue_depth} > {self.queue_threshold}")

        return len(triggers) > 0, triggers


class QualityGate:
    """Ensure degradation doesn't violate quality floors."""

    def __init__(self, floor: QualityFloor):
        self.floor = floor

    def check_post_degrade(
        self,
        groundedness: float,
        ors_rag: float,
        unsupported_claims: float
    ) -> Tuple[bool, List[str]]:
        """Check if quality meets floors after degradation."""
        violations = []

        if groundedness < self.floor.min_groundedness:
            violations.append(f"groundedness: {groundedness:.2f} < {self.floor.min_groundedness}")

        if ors_rag < self.floor.min_ors_rag:
            violations.append(f"ORS_RAG: {ors_rag:.2f} < {self.floor.min_ors_rag}")

        if unsupported_claims > self.floor.max_unsupported_claims:
            violations.append(f"unsupported: {unsupported_claims:.2f} > {self.floor.max_unsupported_claims}")

        return len(violations) == 0, violations


class CostMonitor:
    """Monitor cost metrics and detect anomalies."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history: List[CostMetrics] = []
        self.baseline: Optional[Dict[str, float]] = None

    def log_query(self, metrics: CostMetrics):
        """Log query metrics."""
        self.history.append(metrics)
        if len(self.history) > self.window_size * 2:
            self.history = self.history[-self.window_size:]

    def set_baseline(self):
        """Set baseline from current history."""
        if len(self.history) >= self.window_size:
            recent = self.history[-self.window_size:]
            self.baseline = {
                'avg_tokens': np.mean([m.tokens_total for m in recent]),
                'avg_latency': np.mean([m.total_latency_ms for m in recent]),
                'avg_cost': np.mean([m.estimated_cost_usd for m in recent]),
            }

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect cost anomalies."""
        if not self.baseline or len(self.history) < 10:
            return []

        recent = self.history[-10:]
        anomalies = []

        # Token spike
        avg_tokens = np.mean([m.tokens_total for m in recent])
        if avg_tokens > self.baseline['avg_tokens'] * 1.5:
            anomalies.append({
                'type': 'token_spike',
                'current': avg_tokens,
                'baseline': self.baseline['avg_tokens'],
            })

        # Cost creep
        avg_cost = np.mean([m.estimated_cost_usd for m in recent])
        if avg_cost > self.baseline['avg_cost'] * 1.3:
            anomalies.append({
                'type': 'cost_creep',
                'current': avg_cost,
                'baseline': self.baseline['avg_cost'],
            })

        return anomalies

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for dashboard."""
        if not self.history:
            return {}

        recent = self.history[-self.window_size:]

        return {
            'queries_logged': len(self.history),
            'avg_tokens': np.mean([m.tokens_total for m in recent]),
            'avg_latency_ms': np.mean([m.total_latency_ms for m in recent]),
            'avg_cost_usd': np.mean([m.estimated_cost_usd for m in recent]),
            'p95_latency_ms': np.percentile([m.total_latency_ms for m in recent], 95),
            'cache_hit_rate': sum(1 for m in recent if m.cache_hit) / len(recent),
            'degradation_rate': sum(
                1 for m in recent
                if m.degradation_level != DegradationLevel.FULL
            ) / len(recent),
        }


class CostGovernanceEngine:
    """Main cost governance engine."""

    def __init__(self):
        self.intent_mapper = IntentCostMapper()
        self.token_manager = TokenBudgetManager()
        self.cache = CacheManager()
        self.degradation = DegradationLadder()
        self.trigger = DegradationTrigger()
        self.monitor = CostMonitor()
        self.slo = PerformanceSLO()
        self.cost_model = CostModel()

    def get_query_config(
        self,
        intent: str,
        current_load: float = 0.5
    ) -> Dict[str, Any]:
        """Get configuration for query based on intent and load."""
        tier = self.intent_mapper.get_tier(intent)
        budget = self.intent_mapper.get_budget(tier)

        # Check if degradation needed
        if current_load > 0.8:
            self.degradation.step_down()

        config = self.degradation.get_config(self.degradation.current_level)
        config['budget'] = budget
        config['tier'] = tier

        return config

    def log_and_check(self, metrics: CostMetrics) -> Dict[str, Any]:
        """Log metrics and check for issues."""
        self.monitor.log_query(metrics)

        # Check for degradation triggers
        should_degrade, triggers = self.trigger.should_degrade(
            metrics.total_latency_ms,
            metrics.estimated_cost_usd,
        )

        # Check for anomalies
        anomalies = self.monitor.detect_anomalies()

        return {
            'degradation_triggered': should_degrade,
            'triggers': triggers,
            'anomalies': anomalies,
            'slo_compliance': self.slo.check_compliance([metrics]),
        }


if __name__ == '__main__':
    # Demo usage
    engine = CostGovernanceEngine()

    # Get config for query
    config = engine.get_query_config('definition')
    print(f"Config for 'definition': {config}")

    # Simulate query metrics
    metrics = CostMetrics(
        query_id='q_001',
        tokens_embedding=100,
        tokens_generation=500,
        tokens_total=600,
        retrieval_ms=200,
        rerank_ms=300,
        generation_ms=800,
        total_latency_ms=1300,
        rerank_calls=1,
        estimated_cost_usd=0.02,
        cache_hit=False,
        degradation_level=DegradationLevel.FULL,
    )

    result = engine.log_and_check(metrics)
    print(f"Check result: {result}")

    # Get dashboard metrics
    dashboard = engine.monitor.get_dashboard_metrics()
    print(f"Dashboard: {dashboard}")
