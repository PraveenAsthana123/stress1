"""
Production Monitoring Framework for RAG Systems

Comprehensive monitoring covering:
- Phase 1: Knowledge & Data Analysis
- Phase 2: Representation & Retrieval Analysis
- Phase 3: Generation & Reasoning Analysis
- Phase 4: Decision Policy Analysis
- Phase 8-11: Explainability, Robustness, Statistics, Benchmarking
- Phase 12-15: Scalability, Governance, Production, ROI

Note: Phases 5-7 (Agent, A2A, mCP) are not included as this paper
has no agent architecture.

Each phase provides monitoring modules with:
- Purpose and measurement targets
- Practical measurement approaches
- Pass/Fail criteria
- Edge case handling
"""

# Phase 1: Knowledge & Data Analysis
from .knowledge_monitor import (
    # Enums
    SourceType,
    DataQuality,
    CoverageLevel,
    FreshnessStatus,
    ConflictSeverity,
    # Data classes
    KnowledgeSource,
    SourceInventoryResult,
    AuthorityValidation,
    CoverageAnalysis,
    FreshnessCheck,
    ConflictReport,
    # Monitors
    KnowledgeSourceInventory,
    SourceAuthorityValidator,
    KnowledgeCoverageAnalyzer,
    DocumentFreshnessChecker,
    KnowledgeConflictScanner,
    KnowledgePhaseMonitor,
)

# Phase 2: Representation & Retrieval Analysis
from .retrieval_monitor import (
    # Enums
    ChunkQuality,
    DriftSeverity,
    RetrievalQuality,
    # Data classes
    ChunkStats,
    ChunkingValidationResult,
    EmbeddingSnapshot,
    DriftReport,
    RetrievalMetrics,
    RetrievalQualityReport,
    # Monitors
    ChunkingValidator,
    EmbeddingDriftDetector,
    RetrievalQualityAnalyzer,
    RetrievalPhaseMonitor,
)

# Phase 3: Generation & Reasoning Analysis
from .generation_monitor import (
    # Enums
    PromptRisk,
    HallucinationType,
    GenerationQuality,
    GroundingLevel,
    # Data classes
    PromptCheck,
    HallucinationDetection,
    GroundingAnalysis,
    GenerationMetrics,
    GenerationReport,
    # Monitors
    PromptIntegrityChecker,
    HallucinationDetector,
    GenerationQualityAnalyzer as GenerationAnalyzer,
    GenerationPhaseMonitor,
)

# Phase 4: Decision Policy Analysis
from .decision_monitor import (
    # Enums
    DecisionType,
    PolicyCompliance,
    ConfidenceCalibration,
    RiskCategory,
    # Data classes
    Decision,
    PolicyRule,
    PolicyViolation,
    CalibrationMetrics,
    DecisionAnalysisReport,
    # Monitors
    DecisionPolicyAnalyzer,
    ConfidenceCalibrationAnalyzer,
    DecisionQualityScorer,
    DecisionPhaseMonitor,
)

# Phase 8-11: Explainability, Robustness, Statistics, Benchmarking
from .agent_monitor import (
    # Enums
    ExplainabilityLevel,
    RobustnessLevel,
    StatisticalSignificance,
    BenchmarkRank,
    # Data classes
    ExplanationQuality,
    RobustnessTest,
    StatisticalTest,
    BenchmarkResult,
    AnalysisReport,
    # Analyzers
    ExplainabilityAnalyzer,
    RobustnessAnalyzer,
    StatisticalValidator,
    BenchmarkAnalyzer,
    AgentBehaviorAnalyzer,
    # Placeholders (for compatibility)
    A2AInteractionMonitor,
    MCPEnforcementMonitor,
)

# Phase 12 & 14: Scalability, Deployment & Production Monitoring
from .production_monitor import (
    # Enums
    ScalabilityLevel,
    DeploymentStatus,
    DriftType,
    AlertSeverity,
    # Data classes
    LatencyMetrics,
    ThroughputMetrics,
    ScalabilityTest,
    DriftDetection,
    ProductionAlert,
    ProductionHealthReport,
    # Monitors
    ScalabilityMonitor,
    ProductionDriftMonitor,
    ProductionHealthMonitor,
)

# Phase 13: Governance, Security & Compliance
from .governance_monitor import (
    # Enums
    ComplianceStatus,
    SecurityLevel,
    AuditEventType,
    RegulatoryFramework,
    # Data classes
    AuditEvent,
    PolicyViolation as GovernancePolicyViolation,
    ComplianceCheck,
    SecurityAssessment,
    GovernanceReport,
    # Monitors
    AuditLogger,
    PolicyEnforcer,
    ComplianceChecker,
    SecurityAssessor,
    GovernanceMonitor,
)

# Phase 15: Value, ROI & Executive Impact
from .roi_monitor import (
    # Enums
    ValueCategory,
    ROIStatus,
    ImpactLevel,
    StakeholderType,
    # Data classes
    CostMetric,
    BenefitMetric,
    ROICalculation,
    UsageMetrics,
    QualityImpact,
    ExecutiveSummary,
    ROIReport,
    # Analyzers
    CostTracker,
    BenefitTracker,
    ROICalculator,
    UsageAnalyzer,
    QualityImpactAnalyzer,
    ROIAnalyzer,
)


__all__ = [
    # Phase 1: Knowledge
    "SourceType",
    "DataQuality",
    "CoverageLevel",
    "FreshnessStatus",
    "ConflictSeverity",
    "KnowledgeSource",
    "SourceInventoryResult",
    "AuthorityValidation",
    "CoverageAnalysis",
    "FreshnessCheck",
    "ConflictReport",
    "KnowledgeSourceInventory",
    "SourceAuthorityValidator",
    "KnowledgeCoverageAnalyzer",
    "DocumentFreshnessChecker",
    "KnowledgeConflictScanner",
    "KnowledgePhaseMonitor",

    # Phase 2: Retrieval
    "ChunkQuality",
    "DriftSeverity",
    "RetrievalQuality",
    "ChunkStats",
    "ChunkingValidationResult",
    "EmbeddingSnapshot",
    "DriftReport",
    "RetrievalMetrics",
    "RetrievalQualityReport",
    "ChunkingValidator",
    "EmbeddingDriftDetector",
    "RetrievalQualityAnalyzer",
    "RetrievalPhaseMonitor",

    # Phase 3: Generation
    "PromptRisk",
    "HallucinationType",
    "GenerationQuality",
    "GroundingLevel",
    "PromptCheck",
    "HallucinationDetection",
    "GroundingAnalysis",
    "GenerationMetrics",
    "GenerationReport",
    "PromptIntegrityChecker",
    "HallucinationDetector",
    "GenerationAnalyzer",
    "GenerationPhaseMonitor",

    # Phase 4: Decision
    "DecisionType",
    "PolicyCompliance",
    "ConfidenceCalibration",
    "RiskCategory",
    "Decision",
    "PolicyRule",
    "PolicyViolation",
    "CalibrationMetrics",
    "DecisionAnalysisReport",
    "DecisionPolicyAnalyzer",
    "ConfidenceCalibrationAnalyzer",
    "DecisionQualityScorer",
    "DecisionPhaseMonitor",

    # Phase 8-11: Analysis
    "ExplainabilityLevel",
    "RobustnessLevel",
    "StatisticalSignificance",
    "BenchmarkRank",
    "ExplanationQuality",
    "RobustnessTest",
    "StatisticalTest",
    "BenchmarkResult",
    "AnalysisReport",
    "ExplainabilityAnalyzer",
    "RobustnessAnalyzer",
    "StatisticalValidator",
    "BenchmarkAnalyzer",
    "AgentBehaviorAnalyzer",
    "A2AInteractionMonitor",
    "MCPEnforcementMonitor",

    # Phase 12 & 14: Production
    "ScalabilityLevel",
    "DeploymentStatus",
    "DriftType",
    "AlertSeverity",
    "LatencyMetrics",
    "ThroughputMetrics",
    "ScalabilityTest",
    "DriftDetection",
    "ProductionAlert",
    "ProductionHealthReport",
    "ScalabilityMonitor",
    "ProductionDriftMonitor",
    "ProductionHealthMonitor",

    # Phase 13: Governance
    "ComplianceStatus",
    "SecurityLevel",
    "AuditEventType",
    "RegulatoryFramework",
    "AuditEvent",
    "GovernancePolicyViolation",
    "ComplianceCheck",
    "SecurityAssessment",
    "GovernanceReport",
    "AuditLogger",
    "PolicyEnforcer",
    "ComplianceChecker",
    "SecurityAssessor",
    "GovernanceMonitor",

    # Phase 15: ROI
    "ValueCategory",
    "ROIStatus",
    "ImpactLevel",
    "StakeholderType",
    "CostMetric",
    "BenefitMetric",
    "ROICalculation",
    "UsageMetrics",
    "QualityImpact",
    "ExecutiveSummary",
    "ROIReport",
    "CostTracker",
    "BenefitTracker",
    "ROICalculator",
    "UsageAnalyzer",
    "QualityImpactAnalyzer",
    "ROIAnalyzer",
]
