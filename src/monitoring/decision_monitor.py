"""
Phase 4: Decision Policy Analysis Monitoring

Monitors decision-making quality, policy compliance, and confidence calibration
for RAG systems.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================

class DecisionType(Enum):
    """Types of decisions."""
    ANSWER = "answer"           # Provide full answer
    PARTIAL = "partial"         # Partial answer with caveats
    ABSTAIN = "abstain"         # Refuse to answer
    ESCALATE = "escalate"       # Escalate to human
    CLARIFY = "clarify"         # Request clarification


class PolicyCompliance(Enum):
    """Policy compliance status."""
    COMPLIANT = "compliant"
    MINOR_VIOLATION = "minor_violation"
    MAJOR_VIOLATION = "major_violation"
    CRITICAL_VIOLATION = "critical_violation"


class ConfidenceCalibration(Enum):
    """Confidence calibration status."""
    WELL_CALIBRATED = "well_calibrated"
    OVERCONFIDENT = "overconfident"
    UNDERCONFIDENT = "underconfident"
    UNCALIBRATED = "uncalibrated"


class RiskCategory(Enum):
    """Risk categories for decisions."""
    SAFETY = "safety"
    PRIVACY = "privacy"
    ACCURACY = "accuracy"
    COMPLIANCE = "compliance"
    REPUTATION = "reputation"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Decision:
    """A single decision record."""
    decision_id: str
    query: str
    decision_type: DecisionType
    confidence: float
    evidence_strength: float
    risk_factors: List[RiskCategory]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyRule:
    """A decision policy rule."""
    rule_id: str
    name: str
    condition: str
    required_action: DecisionType
    risk_category: RiskCategory
    severity: str  # low, medium, high, critical


@dataclass
class PolicyViolation:
    """A policy violation record."""
    decision_id: str
    rule_id: str
    rule_name: str
    violation_details: str
    severity: str
    recommended_action: str


@dataclass
class CalibrationMetrics:
    """Calibration metrics for confidence scores."""
    expected_calibration_error: float  # ECE
    maximum_calibration_error: float   # MCE
    brier_score: float
    calibration_status: ConfidenceCalibration
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]


@dataclass
class DecisionAnalysisReport:
    """Comprehensive decision analysis report."""
    total_decisions: int
    decision_distribution: Dict[DecisionType, int]
    avg_confidence: float
    avg_evidence_strength: float
    policy_compliance_rate: float
    calibration_metrics: CalibrationMetrics
    violations: List[PolicyViolation]
    passed: bool
    issues: List[str]
    recommendations: List[str]


# =============================================================================
# Phase 4.1: Decision Policy Analyzer
# =============================================================================

class DecisionPolicyAnalyzer:
    """
    Analyzes decision quality and policy compliance.

    Purpose: Ensure decisions follow established policies
    Measurement: Policy compliance rate, decision distribution
    Pass/Fail: >95% policy compliance
    """

    def __init__(self):
        self.policies: Dict[str, PolicyRule] = {}
        self.decisions: List[Decision] = []
        self.violations: List[PolicyViolation] = []

        # Default policies
        self._init_default_policies()

    def _init_default_policies(self):
        """Initialize default decision policies."""
        default_policies = [
            PolicyRule(
                rule_id="P001",
                name="low_confidence_abstain",
                condition="confidence < 0.3",
                required_action=DecisionType.ABSTAIN,
                risk_category=RiskCategory.ACCURACY,
                severity="high"
            ),
            PolicyRule(
                rule_id="P002",
                name="low_evidence_partial",
                condition="evidence_strength < 0.5 and confidence >= 0.3",
                required_action=DecisionType.PARTIAL,
                risk_category=RiskCategory.ACCURACY,
                severity="medium"
            ),
            PolicyRule(
                rule_id="P003",
                name="safety_risk_escalate",
                condition="SAFETY in risk_factors",
                required_action=DecisionType.ESCALATE,
                risk_category=RiskCategory.SAFETY,
                severity="critical"
            ),
            PolicyRule(
                rule_id="P004",
                name="privacy_risk_abstain",
                condition="PRIVACY in risk_factors and confidence < 0.9",
                required_action=DecisionType.ABSTAIN,
                risk_category=RiskCategory.PRIVACY,
                severity="high"
            ),
        ]

        for policy in default_policies:
            self.policies[policy.rule_id] = policy

    def add_policy(self, policy: PolicyRule):
        """Add a decision policy."""
        self.policies[policy.rule_id] = policy

    def record_decision(self, decision: Decision) -> List[PolicyViolation]:
        """Record a decision and check for violations."""
        self.decisions.append(decision)

        # Check all policies
        violations = []
        for policy in self.policies.values():
            if self._check_policy_violation(decision, policy):
                violation = PolicyViolation(
                    decision_id=decision.decision_id,
                    rule_id=policy.rule_id,
                    rule_name=policy.name,
                    violation_details=f"Decision {decision.decision_type.value} "
                                     f"violates policy {policy.name}",
                    severity=policy.severity,
                    recommended_action=policy.required_action.value
                )
                violations.append(violation)
                self.violations.append(violation)

        return violations

    def _check_policy_violation(
        self,
        decision: Decision,
        policy: PolicyRule
    ) -> bool:
        """Check if decision violates a policy."""
        # Evaluate condition
        condition_met = False

        if "confidence <" in policy.condition:
            threshold = float(policy.condition.split("<")[1].strip())
            condition_met = decision.confidence < threshold
        elif "confidence >=" in policy.condition:
            threshold = float(policy.condition.split(">=")[1].strip().split()[0])
            condition_met = decision.confidence >= threshold
        elif "evidence_strength <" in policy.condition:
            threshold = float(policy.condition.split("<")[1].strip().split()[0])
            condition_met = decision.evidence_strength < threshold

        # Check risk factors
        for risk in RiskCategory:
            if f"{risk.name} in risk_factors" in policy.condition:
                if risk in decision.risk_factors:
                    condition_met = True

        # If condition met, check if action matches required action
        if condition_met:
            return decision.decision_type != policy.required_action

        return False

    def get_compliance_rate(self) -> float:
        """Get policy compliance rate."""
        if not self.decisions:
            return 1.0

        decisions_with_violations = set(v.decision_id for v in self.violations)
        compliant = len(self.decisions) - len(decisions_with_violations)
        return compliant / len(self.decisions)

    def get_decision_distribution(self) -> Dict[DecisionType, int]:
        """Get distribution of decision types."""
        dist = {dt: 0 for dt in DecisionType}
        for d in self.decisions:
            dist[d.decision_type] += 1
        return dist

    def get_risk_distribution(self) -> Dict[RiskCategory, int]:
        """Get distribution of risk categories."""
        dist = {rc: 0 for rc in RiskCategory}
        for d in self.decisions:
            for risk in d.risk_factors:
                dist[risk] += 1
        return dist


# =============================================================================
# Phase 4.2: Confidence Calibration Analyzer
# =============================================================================

class ConfidenceCalibrationAnalyzer:
    """
    Analyzes confidence calibration of decisions.

    Purpose: Ensure confidence scores are well-calibrated
    Measurement: ECE, MCE, Brier score
    Pass/Fail: ECE < 0.1
    """

    def __init__(self, n_bins: int = 10, ece_threshold: float = 0.1):
        self.n_bins = n_bins
        self.ece_threshold = ece_threshold
        self.predictions: List[Tuple[float, bool]] = []

    def record_prediction(self, confidence: float, correct: bool):
        """Record a prediction with its outcome."""
        self.predictions.append((confidence, correct))

    def analyze_calibration(self) -> CalibrationMetrics:
        """Analyze confidence calibration."""
        if not self.predictions:
            return CalibrationMetrics(
                expected_calibration_error=0.0,
                maximum_calibration_error=0.0,
                brier_score=0.0,
                calibration_status=ConfidenceCalibration.UNCALIBRATED,
                bin_accuracies=[],
                bin_confidences=[],
                bin_counts=[]
            )

        confidences = np.array([p[0] for p in self.predictions])
        correctness = np.array([p[1] for p in self.predictions])

        # Bin predictions
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i in range(self.n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]

            mask = (confidences >= lower) & (confidences < upper)
            if i == self.n_bins - 1:
                mask = (confidences >= lower) & (confidences <= upper)

            bin_count = mask.sum()
            bin_counts.append(int(bin_count))

            if bin_count > 0:
                bin_accuracy = correctness[mask].mean()
                bin_confidence = confidences[mask].mean()
            else:
                bin_accuracy = 0.0
                bin_confidence = (lower + upper) / 2

            bin_accuracies.append(float(bin_accuracy))
            bin_confidences.append(float(bin_confidence))

        # Calculate ECE
        total_samples = len(self.predictions)
        ece = sum(
            (count / total_samples) * abs(acc - conf)
            for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts)
            if count > 0
        )

        # Calculate MCE
        mce = max(
            abs(acc - conf)
            for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts)
            if count > 0
        ) if any(c > 0 for c in bin_counts) else 0.0

        # Calculate Brier score
        brier = np.mean((confidences - correctness) ** 2)

        # Determine calibration status
        avg_confidence = confidences.mean()
        avg_accuracy = correctness.mean()

        if ece < self.ece_threshold:
            status = ConfidenceCalibration.WELL_CALIBRATED
        elif avg_confidence > avg_accuracy + 0.1:
            status = ConfidenceCalibration.OVERCONFIDENT
        elif avg_confidence < avg_accuracy - 0.1:
            status = ConfidenceCalibration.UNDERCONFIDENT
        else:
            status = ConfidenceCalibration.UNCALIBRATED

        return CalibrationMetrics(
            expected_calibration_error=float(ece),
            maximum_calibration_error=float(mce),
            brier_score=float(brier),
            calibration_status=status,
            bin_accuracies=bin_accuracies,
            bin_confidences=bin_confidences,
            bin_counts=bin_counts
        )

    def reset(self):
        """Reset prediction history."""
        self.predictions = []


# =============================================================================
# Phase 4.3: Decision Quality Scorer
# =============================================================================

class DecisionQualityScorer:
    """
    Scores overall decision quality.

    Purpose: Provide quality scores for decisions
    Measurement: Composite quality score
    Pass/Fail: Average score > 0.7
    """

    def __init__(self):
        self.weights = {
            "confidence_accuracy": 0.3,
            "evidence_quality": 0.3,
            "policy_compliance": 0.2,
            "risk_management": 0.2
        }

    def score_decision(
        self,
        decision: Decision,
        actual_correct: Optional[bool] = None,
        policy_compliant: bool = True
    ) -> Dict[str, float]:
        """Score a single decision."""
        scores = {}

        # Confidence accuracy score
        if actual_correct is not None:
            if actual_correct:
                # Reward high confidence when correct
                scores["confidence_accuracy"] = decision.confidence
            else:
                # Penalize high confidence when wrong
                scores["confidence_accuracy"] = 1 - decision.confidence
        else:
            # Without ground truth, use evidence alignment
            scores["confidence_accuracy"] = min(
                decision.confidence,
                decision.evidence_strength
            )

        # Evidence quality score
        scores["evidence_quality"] = decision.evidence_strength

        # Policy compliance score
        scores["policy_compliance"] = 1.0 if policy_compliant else 0.0

        # Risk management score
        risk_score = 1.0
        if decision.risk_factors:
            if RiskCategory.SAFETY in decision.risk_factors:
                if decision.decision_type not in [DecisionType.ABSTAIN, DecisionType.ESCALATE]:
                    risk_score -= 0.4
            if RiskCategory.PRIVACY in decision.risk_factors:
                if decision.decision_type == DecisionType.ANSWER:
                    risk_score -= 0.3
            if RiskCategory.COMPLIANCE in decision.risk_factors:
                if decision.decision_type == DecisionType.ANSWER:
                    risk_score -= 0.2
        scores["risk_management"] = max(0.0, risk_score)

        # Calculate composite score
        composite = sum(
            scores[k] * self.weights[k]
            for k in self.weights
        )
        scores["composite"] = composite

        return scores


# =============================================================================
# Phase 4: Unified Decision Monitor
# =============================================================================

class DecisionPhaseMonitor:
    """
    Unified monitor for Phase 4: Decision Policy Analysis.

    Combines policy analysis, calibration, and quality scoring.
    """

    def __init__(self):
        self.policy_analyzer = DecisionPolicyAnalyzer()
        self.calibration_analyzer = ConfidenceCalibrationAnalyzer()
        self.quality_scorer = DecisionQualityScorer()

    def record_decision(
        self,
        decision: Decision,
        actual_correct: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Record a decision and get analysis."""
        # Check policy compliance
        violations = self.policy_analyzer.record_decision(decision)

        # Record for calibration
        if actual_correct is not None:
            self.calibration_analyzer.record_prediction(
                decision.confidence,
                actual_correct
            )

        # Score decision
        quality_scores = self.quality_scorer.score_decision(
            decision,
            actual_correct,
            policy_compliant=len(violations) == 0
        )

        return {
            "decision_id": decision.decision_id,
            "decision_type": decision.decision_type.value,
            "violations": [
                {"rule": v.rule_name, "severity": v.severity}
                for v in violations
            ],
            "quality_scores": quality_scores,
            "passed": len(violations) == 0 and quality_scores["composite"] >= 0.7
        }

    def run_full_analysis(
        self,
        decisions: List[Dict[str, Any]]
    ) -> DecisionAnalysisReport:
        """
        Run full Phase 4 analysis.

        Each decision dict should contain:
        - decision_id: Unique identifier
        - query: The input query
        - decision_type: Type of decision (answer, partial, abstain, etc.)
        - confidence: Confidence score (0-1)
        - evidence_strength: Evidence strength score (0-1)
        - risk_factors: List of risk categories
        - actual_correct: Optional ground truth
        """
        all_scores = []

        for d in decisions:
            decision = Decision(
                decision_id=d.get("decision_id", "unknown"),
                query=d.get("query", ""),
                decision_type=DecisionType(d.get("decision_type", "answer")),
                confidence=d.get("confidence", 0.5),
                evidence_strength=d.get("evidence_strength", 0.5),
                risk_factors=[
                    RiskCategory(r) for r in d.get("risk_factors", [])
                ]
            )

            result = self.record_decision(
                decision,
                d.get("actual_correct")
            )
            all_scores.append(result["quality_scores"]["composite"])

        # Get calibration metrics
        calibration = self.calibration_analyzer.analyze_calibration()

        # Get compliance rate
        compliance_rate = self.policy_analyzer.get_compliance_rate()

        # Get distributions
        decision_dist = self.policy_analyzer.get_decision_distribution()

        # Calculate averages
        avg_confidence = np.mean([d.get("confidence", 0.5) for d in decisions]) \
                        if decisions else 0.0
        avg_evidence = np.mean([d.get("evidence_strength", 0.5) for d in decisions]) \
                      if decisions else 0.0

        # Check pass criteria
        calibration_passed = calibration.expected_calibration_error < 0.1
        compliance_passed = compliance_rate >= 0.95
        quality_passed = np.mean(all_scores) >= 0.7 if all_scores else False

        passed = calibration_passed and compliance_passed and quality_passed

        # Identify issues
        issues = []
        if not calibration_passed:
            issues.append(
                f"Calibration error {calibration.expected_calibration_error:.2%} "
                f"above threshold 10%"
            )
        if not compliance_passed:
            issues.append(
                f"Policy compliance {compliance_rate:.2%} below threshold 95%"
            )
        if not quality_passed:
            avg_score = np.mean(all_scores) if all_scores else 0
            issues.append(f"Average quality score {avg_score:.2f} below threshold 0.7")

        # Recommendations
        recommendations = []
        if calibration.calibration_status == ConfidenceCalibration.OVERCONFIDENT:
            recommendations.append("Apply temperature scaling to reduce overconfidence")
        elif calibration.calibration_status == ConfidenceCalibration.UNDERCONFIDENT:
            recommendations.append("Review confidence estimation approach")

        if not compliance_passed:
            recommendations.append("Review decision policies and training data")

        return DecisionAnalysisReport(
            total_decisions=len(decisions),
            decision_distribution={k.value: v for k, v in decision_dist.items()},
            avg_confidence=float(avg_confidence),
            avg_evidence_strength=float(avg_evidence),
            policy_compliance_rate=compliance_rate,
            calibration_metrics=calibration,
            violations=self.policy_analyzer.violations,
            passed=passed,
            issues=issues,
            recommendations=recommendations
        )


__all__ = [
    # Enums
    "DecisionType",
    "PolicyCompliance",
    "ConfidenceCalibration",
    "RiskCategory",
    # Data classes
    "Decision",
    "PolicyRule",
    "PolicyViolation",
    "CalibrationMetrics",
    "DecisionAnalysisReport",
    # Monitors
    "DecisionPolicyAnalyzer",
    "ConfidenceCalibrationAnalyzer",
    "DecisionQualityScorer",
    "DecisionPhaseMonitor",
]
