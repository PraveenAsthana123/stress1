"""AI Governance Module - Responsible, Explainable, Trustworthy AI."""
from .responsible_ai import (
    ExplainabilityModule,
    TrustModule,
    EthicalAIModule,
    ComplianceModule,
    SecurityModule,
    AIGovernanceManager,
    AuditEntry,
    ExplanationResult
)

# A3: PHI/PII Detection & Redaction
from .phi_pii_detection import (
    RiskLevel,
    EntityType,
    RedactionAction,
    DetectedEntity,
    RedactionResult,
    IngestionGateResult,
    SensitiveFieldPolicy,
    RuleBasedDetector,
    NERDetector,
    HybridDetector,
    Redactor,
    TableRedactor,
    IngestionGate,
    BackscanManager,
    PHIMonitor,
)

# A4: Role-Based Access Control
from .rbac import (
    Role,
    AccessLevel,
    ResourceType,
    ACLMetadata,
    UserContext,
    AccessDecision,
    RoleAccessMatrix,
    PolicyEvaluator,
    RetrievalFilter,
    CacheKeyBuilder,
    BreakGlassManager,
    RBACauditLog,
    RBACTestSuite,
    RBACManager,
)

# A5: Document Trust & Authority Scoring
from .document_authority import (
    AuthorityLevel,
    ApprovalStatus,
    AuthorityMetadata,
    TrustScore,
    AuthorityRubric,
    FreshnessCalculator,
    TrustScoreCalculator,
    RetrievalRanker,
    SourceDiversityEnforcer,
    ConflictDetector,
    ConflictResolver,
    DeprecationRegistry,
    AuthorityMonitor,
    AuthorityManager,
)

# A7: Claim-to-Evidence Verification
from .claim_verification import (
    VerificationStatus,
    FallbackAction,
    Claim,
    Citation,
    VerificationResult,
    VerificationReport,
    ClaimExtractor,
    EvidenceRequirements,
    LexicalSupportChecker,
    NumericVerifier,
    SemanticEntailmentChecker,
    TableEvidenceLinker,
    ClaimVerifier,
    VerificationPipeline,
    VerificationMonitor,
)

# A11: Lineage Versioning
from .lineage_versioning import (
    DocumentID,
    ChunkID,
    PipelineManifest,
    CorpusSnapshot,
    RunManifest,
    LineageScopeSpec,
    DocumentRegistry,
    CorpusSnapshotManager,
    RunManifestStore,
    AnswerReplay,
    RunDiff,
    ReleaseGate,
    LineageMonitor,
    LineageManager,
)

# A12: Cost Governance
from .cost_governance import (
    CostTier,
    DegradationLevel,
    CostBudget,
    CostMetrics,
    QualityFloor,
    CostModel,
    PerformanceSLO,
    IntentCostMapper,
    TokenBudgetManager,
    CacheManager,
    DegradationLadder,
    DegradationTrigger,
    QualityGate,
    CostMonitor,
    CostGovernanceEngine,
)

# A15: Decision Policy Layer
from .decision_policy import (
    DecisionOutcome,
    RiskTier,
    EvidenceLevel,
    QueryContext,
    DecisionResult,
    ThresholdConfig,
    RiskTierClassifier,
    EvidenceSufficiencyChecker,
    ThresholdManager,
    DecisionMatrix,
    PreAnswerChecker,
    PostAnswerChecker,
    EscalationRouter,
    ExplanationGenerator,
    DecisionLogger,
    DecisionPolicyEngine,
    DecisionMonitor,
)

__all__ = [
    # Original
    "ExplainabilityModule",
    "TrustModule",
    "EthicalAIModule",
    "ComplianceModule",
    "SecurityModule",
    "AIGovernanceManager",
    "AuditEntry",
    "ExplanationResult",
    # A3: PHI/PII
    "RiskLevel",
    "EntityType",
    "RedactionAction",
    "IngestionGate",
    "HybridDetector",
    "Redactor",
    "PHIMonitor",
    # A4: RBAC
    "Role",
    "AccessLevel",
    "RBACManager",
    "UserContext",
    "PolicyEvaluator",
    # A5: Authority
    "AuthorityLevel",
    "AuthorityManager",
    "TrustScoreCalculator",
    "ConflictDetector",
    # A7: Verification
    "ClaimVerifier",
    "VerificationPipeline",
    "VerificationStatus",
    # A11: Lineage
    "LineageManager",
    "RunManifest",
    "CorpusSnapshot",
    # A12: Cost
    "CostGovernanceEngine",
    "CostTier",
    "DegradationLevel",
    # A15: Decision
    "DecisionPolicyEngine",
    "DecisionOutcome",
    "RiskTier",
]
