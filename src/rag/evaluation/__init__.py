"""Evaluation Module - Comprehensive RAG metrics and regression testing."""
from .metrics import (
    RetrievalMetrics,
    GenerationMetrics,
    AnswerQualityMetrics,
    SystemMetrics,
    RAGEvaluator,
    EvaluationResult
)

# A1: RAG Evaluation & Regression Harness
from .regression_harness import (
    QueryType,
    GoldQuery,
    EvalResult,
    RegressionGate,
    RetrievalMetrics as HarnessRetrievalMetrics,
    ContextQualityMetrics,
    GroundednessMetrics,
    RelevancyMetrics,
    RedTeamTestPack,
    EvalRunner,
    RegressionTestSuite,
    GoldSetManager,
    create_eeg_gold_set,
)

__all__ = [
    # Original metrics
    "RetrievalMetrics",
    "GenerationMetrics",
    "AnswerQualityMetrics",
    "SystemMetrics",
    "RAGEvaluator",
    "EvaluationResult",
    # A1: Regression Harness
    "QueryType",
    "GoldQuery",
    "EvalResult",
    "RegressionGate",
    "HarnessRetrievalMetrics",
    "ContextQualityMetrics",
    "GroundednessMetrics",
    "RelevancyMetrics",
    "RedTeamTestPack",
    "EvalRunner",
    "RegressionTestSuite",
    "GoldSetManager",
    "create_eeg_gold_set",
]
