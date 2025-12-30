"""
A15) Decision Policy Layer — Answer vs Abstain vs Escalate for EEG-RAG

Comprehensive module for:
- Decision outcomes (Answer, Partial, Abstain, Escalate)
- Query risk tier classification
- Evidence sufficiency rules
- Confidence thresholds by tier
- Decision matrix mapping
- Pre/Post answer checks
- Escalation pathways
- User-facing explanations
- Decision logging and monitoring

This converts RAG from "always answers" into a decision system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DecisionOutcome(Enum):
    """Allowed system decision outcomes."""
    ANSWER = "answer"  # Full confident answer
    PARTIAL_ANSWER = "partial_answer"  # Answer with caveats
    ABSTAIN = "abstain"  # Don't know
    ESCALATE_HUMAN = "escalate_human"  # Human review needed
    ESCALATE_TOOL = "escalate_tool"  # External tool needed
    REFUSE = "refuse"  # Policy violation


class RiskTier(Enum):
    """Query risk tier classification."""
    T0_DEFINITION = "t0_definition"  # Low risk: definitions, concepts
    T1_PROCEDURE = "t1_procedure"  # Medium risk: procedures, SOPs
    T2_PARAMETER = "t2_parameter"  # Higher risk: parameter guidance
    T3_CLINICAL = "t3_clinical"  # Highest risk: clinical-adjacent, safety


class EvidenceLevel(Enum):
    """Evidence sufficiency levels."""
    STRONG = "strong"  # Multiple high-authority sources
    MODERATE = "moderate"  # At least one good source
    WEAK = "weak"  # Low confidence sources
    NONE = "none"  # No relevant evidence


@dataclass
class QueryContext:
    """Context for decision making."""
    query: str
    intent: str
    risk_tier: RiskTier
    user_role: str
    evidence_level: EvidenceLevel
    confidence_score: float
    groundedness_score: float
    entropy: float  # Uncertainty measure
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionResult:
    """Result of decision policy evaluation."""
    outcome: DecisionOutcome
    confidence: float
    explanation: str
    user_message: str
    internal_reason: str
    escalation_target: Optional[str] = None
    partial_content: Optional[str] = None
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)


@dataclass
class ThresholdConfig:
    """Threshold configuration per risk tier."""
    min_confidence: float
    min_groundedness: float
    max_entropy: float
    min_sources: int
    min_authority_level: str


class RiskTierClassifier:
    """Classify queries into risk tiers."""

    # Intent patterns for each tier
    TIER_PATTERNS = {
        RiskTier.T0_DEFINITION: [
            r'what is', r'define', r'explain', r'meaning of',
            r'difference between', r'overview', r'introduction',
        ],
        RiskTier.T1_PROCEDURE: [
            r'how to', r'steps for', r'procedure', r'process',
            r'setup', r'configure', r'install', r'prepare',
        ],
        RiskTier.T2_PARAMETER: [
            r'threshold', r'value', r'setting', r'parameter',
            r'range', r'limit', r'specification', r'recommended',
        ],
        RiskTier.T3_CLINICAL: [
            r'diagnosis', r'treatment', r'patient', r'clinical',
            r'seizure', r'abnormal', r'interpret', r'emergency',
        ],
    }

    def classify(self, query: str, intent: Optional[str] = None) -> RiskTier:
        """Classify query into risk tier."""
        query_lower = query.lower()

        # Check patterns in reverse order (highest risk first)
        for tier in [RiskTier.T3_CLINICAL, RiskTier.T2_PARAMETER,
                     RiskTier.T1_PROCEDURE, RiskTier.T0_DEFINITION]:
            patterns = self.TIER_PATTERNS.get(tier, [])
            for pattern in patterns:
                if pattern in query_lower:
                    return tier

        # Default to procedure (medium risk)
        return RiskTier.T1_PROCEDURE


class EvidenceSufficiencyChecker:
    """Check if evidence meets sufficiency requirements."""

    def __init__(self, tier_requirements: Optional[Dict[RiskTier, Dict[str, Any]]] = None):
        self.requirements = tier_requirements or self._default_requirements()

    def _default_requirements(self) -> Dict[RiskTier, Dict[str, Any]]:
        """Default evidence requirements by tier."""
        return {
            RiskTier.T0_DEFINITION: {
                'min_sources': 1,
                'min_authority': 'internal_approved',
                'freshness_required': False,
            },
            RiskTier.T1_PROCEDURE: {
                'min_sources': 1,
                'min_authority': 'internal_approved',
                'freshness_required': True,
            },
            RiskTier.T2_PARAMETER: {
                'min_sources': 2,
                'min_authority': 'vendor_manual',
                'freshness_required': True,
            },
            RiskTier.T3_CLINICAL: {
                'min_sources': 2,
                'min_authority': 'peer_reviewed',
                'freshness_required': True,
                'verifier_required': True,
            },
        }

    def check(
        self,
        sources: List[Dict[str, Any]],
        risk_tier: RiskTier
    ) -> Tuple[EvidenceLevel, Dict[str, Any]]:
        """Check evidence sufficiency."""
        req = self.requirements.get(risk_tier, {})
        details = {}

        # Check source count
        min_sources = req.get('min_sources', 1)
        has_enough_sources = len(sources) >= min_sources
        details['source_count'] = len(sources)
        details['min_required'] = min_sources

        # Check authority levels
        authority_order = ['notes', 'internal_draft', 'internal_approved',
                          'vendor_manual', 'peer_reviewed', 'standard']
        min_auth = req.get('min_authority', 'notes')
        min_auth_idx = authority_order.index(min_auth) if min_auth in authority_order else 0

        source_authorities = [s.get('authority_level', 'notes') for s in sources]
        meets_authority = any(
            authority_order.index(a) >= min_auth_idx
            for a in source_authorities
            if a in authority_order
        )
        details['meets_authority'] = meets_authority

        # Determine level
        if has_enough_sources and meets_authority:
            if len(sources) >= min_sources * 2:
                level = EvidenceLevel.STRONG
            else:
                level = EvidenceLevel.MODERATE
        elif has_enough_sources or meets_authority:
            level = EvidenceLevel.WEAK
        else:
            level = EvidenceLevel.NONE

        details['level'] = level.value
        return level, details


class ThresholdManager:
    """Manage confidence thresholds by tier."""

    def __init__(self):
        self.thresholds = self._default_thresholds()

    def _default_thresholds(self) -> Dict[RiskTier, ThresholdConfig]:
        """Default thresholds by risk tier."""
        return {
            RiskTier.T0_DEFINITION: ThresholdConfig(
                min_confidence=0.7,
                min_groundedness=0.8,
                max_entropy=0.5,
                min_sources=1,
                min_authority_level='internal_approved',
            ),
            RiskTier.T1_PROCEDURE: ThresholdConfig(
                min_confidence=0.8,
                min_groundedness=0.85,
                max_entropy=0.4,
                min_sources=1,
                min_authority_level='internal_approved',
            ),
            RiskTier.T2_PARAMETER: ThresholdConfig(
                min_confidence=0.85,
                min_groundedness=0.9,
                max_entropy=0.3,
                min_sources=2,
                min_authority_level='vendor_manual',
            ),
            RiskTier.T3_CLINICAL: ThresholdConfig(
                min_confidence=0.95,
                min_groundedness=0.95,
                max_entropy=0.2,
                min_sources=2,
                min_authority_level='peer_reviewed',
            ),
        }

    def get_thresholds(self, tier: RiskTier) -> ThresholdConfig:
        """Get thresholds for tier."""
        return self.thresholds.get(tier, self.thresholds[RiskTier.T1_PROCEDURE])

    def meets_thresholds(
        self,
        context: QueryContext
    ) -> Tuple[bool, List[str], List[str]]:
        """Check if context meets thresholds."""
        thresholds = self.get_thresholds(context.risk_tier)
        passed = []
        failed = []

        if context.confidence_score >= thresholds.min_confidence:
            passed.append(f"confidence: {context.confidence_score:.2f} >= {thresholds.min_confidence}")
        else:
            failed.append(f"confidence: {context.confidence_score:.2f} < {thresholds.min_confidence}")

        if context.groundedness_score >= thresholds.min_groundedness:
            passed.append(f"groundedness: {context.groundedness_score:.2f} >= {thresholds.min_groundedness}")
        else:
            failed.append(f"groundedness: {context.groundedness_score:.2f} < {thresholds.min_groundedness}")

        if context.entropy <= thresholds.max_entropy:
            passed.append(f"entropy: {context.entropy:.2f} <= {thresholds.max_entropy}")
        else:
            failed.append(f"entropy: {context.entropy:.2f} > {thresholds.max_entropy}")

        if len(context.sources) >= thresholds.min_sources:
            passed.append(f"sources: {len(context.sources)} >= {thresholds.min_sources}")
        else:
            failed.append(f"sources: {len(context.sources)} < {thresholds.min_sources}")

        return len(failed) == 0, passed, failed


class DecisionMatrix:
    """Map signals to decision outcomes."""

    def __init__(self):
        self.matrix = self._build_matrix()

    def _build_matrix(self) -> Dict[Tuple, DecisionOutcome]:
        """Build decision matrix: (risk, evidence, confidence_ok) -> outcome."""
        matrix = {}

        # T0: Definitions - more lenient
        matrix[(RiskTier.T0_DEFINITION, EvidenceLevel.STRONG, True)] = DecisionOutcome.ANSWER
        matrix[(RiskTier.T0_DEFINITION, EvidenceLevel.MODERATE, True)] = DecisionOutcome.ANSWER
        matrix[(RiskTier.T0_DEFINITION, EvidenceLevel.WEAK, True)] = DecisionOutcome.PARTIAL_ANSWER
        matrix[(RiskTier.T0_DEFINITION, EvidenceLevel.NONE, True)] = DecisionOutcome.ABSTAIN
        matrix[(RiskTier.T0_DEFINITION, EvidenceLevel.STRONG, False)] = DecisionOutcome.PARTIAL_ANSWER
        matrix[(RiskTier.T0_DEFINITION, EvidenceLevel.MODERATE, False)] = DecisionOutcome.PARTIAL_ANSWER
        matrix[(RiskTier.T0_DEFINITION, EvidenceLevel.WEAK, False)] = DecisionOutcome.ABSTAIN
        matrix[(RiskTier.T0_DEFINITION, EvidenceLevel.NONE, False)] = DecisionOutcome.ABSTAIN

        # T1: Procedures - moderate
        matrix[(RiskTier.T1_PROCEDURE, EvidenceLevel.STRONG, True)] = DecisionOutcome.ANSWER
        matrix[(RiskTier.T1_PROCEDURE, EvidenceLevel.MODERATE, True)] = DecisionOutcome.ANSWER
        matrix[(RiskTier.T1_PROCEDURE, EvidenceLevel.WEAK, True)] = DecisionOutcome.PARTIAL_ANSWER
        matrix[(RiskTier.T1_PROCEDURE, EvidenceLevel.NONE, True)] = DecisionOutcome.ABSTAIN
        matrix[(RiskTier.T1_PROCEDURE, EvidenceLevel.STRONG, False)] = DecisionOutcome.PARTIAL_ANSWER
        matrix[(RiskTier.T1_PROCEDURE, EvidenceLevel.MODERATE, False)] = DecisionOutcome.ABSTAIN
        matrix[(RiskTier.T1_PROCEDURE, EvidenceLevel.WEAK, False)] = DecisionOutcome.ABSTAIN
        matrix[(RiskTier.T1_PROCEDURE, EvidenceLevel.NONE, False)] = DecisionOutcome.ESCALATE_HUMAN

        # T2: Parameters - stricter
        matrix[(RiskTier.T2_PARAMETER, EvidenceLevel.STRONG, True)] = DecisionOutcome.ANSWER
        matrix[(RiskTier.T2_PARAMETER, EvidenceLevel.MODERATE, True)] = DecisionOutcome.PARTIAL_ANSWER
        matrix[(RiskTier.T2_PARAMETER, EvidenceLevel.WEAK, True)] = DecisionOutcome.ABSTAIN
        matrix[(RiskTier.T2_PARAMETER, EvidenceLevel.NONE, True)] = DecisionOutcome.ESCALATE_HUMAN
        matrix[(RiskTier.T2_PARAMETER, EvidenceLevel.STRONG, False)] = DecisionOutcome.PARTIAL_ANSWER
        matrix[(RiskTier.T2_PARAMETER, EvidenceLevel.MODERATE, False)] = DecisionOutcome.ABSTAIN
        matrix[(RiskTier.T2_PARAMETER, EvidenceLevel.WEAK, False)] = DecisionOutcome.ESCALATE_HUMAN
        matrix[(RiskTier.T2_PARAMETER, EvidenceLevel.NONE, False)] = DecisionOutcome.ESCALATE_HUMAN

        # T3: Clinical - strictest
        matrix[(RiskTier.T3_CLINICAL, EvidenceLevel.STRONG, True)] = DecisionOutcome.PARTIAL_ANSWER
        matrix[(RiskTier.T3_CLINICAL, EvidenceLevel.MODERATE, True)] = DecisionOutcome.ESCALATE_HUMAN
        matrix[(RiskTier.T3_CLINICAL, EvidenceLevel.WEAK, True)] = DecisionOutcome.ESCALATE_HUMAN
        matrix[(RiskTier.T3_CLINICAL, EvidenceLevel.NONE, True)] = DecisionOutcome.REFUSE
        matrix[(RiskTier.T3_CLINICAL, EvidenceLevel.STRONG, False)] = DecisionOutcome.ESCALATE_HUMAN
        matrix[(RiskTier.T3_CLINICAL, EvidenceLevel.MODERATE, False)] = DecisionOutcome.ESCALATE_HUMAN
        matrix[(RiskTier.T3_CLINICAL, EvidenceLevel.WEAK, False)] = DecisionOutcome.REFUSE
        matrix[(RiskTier.T3_CLINICAL, EvidenceLevel.NONE, False)] = DecisionOutcome.REFUSE

        return matrix

    def get_outcome(
        self,
        risk_tier: RiskTier,
        evidence_level: EvidenceLevel,
        confidence_ok: bool
    ) -> DecisionOutcome:
        """Get outcome from matrix."""
        key = (risk_tier, evidence_level, confidence_ok)
        return self.matrix.get(key, DecisionOutcome.ABSTAIN)


class PreAnswerChecker:
    """Pre-generation checks."""

    def __init__(self):
        self.forbidden_intents = [
            'diagnosis', 'treatment_recommendation', 'medication',
            'patient_identification', 'data_exfiltration',
        ]

    def check(
        self,
        context: QueryContext
    ) -> Tuple[bool, List[str], List[str]]:
        """Run pre-answer checks."""
        passed = []
        failed = []

        # Intent check
        if context.intent not in self.forbidden_intents:
            passed.append("intent_allowed")
        else:
            failed.append(f"forbidden_intent: {context.intent}")

        # Role-based check
        if context.risk_tier == RiskTier.T3_CLINICAL:
            if context.user_role in ['clinical', 'admin']:
                passed.append("role_authorized")
            else:
                failed.append(f"role_unauthorized: {context.user_role} for clinical")

        # PII safety check
        if 'pii_detected' not in context.metadata:
            passed.append("pii_safe")
        else:
            failed.append("pii_detected_in_query")

        return len(failed) == 0, passed, failed


class PostAnswerChecker:
    """Post-generation checks."""

    def __init__(self, min_groundedness: float = 0.85):
        self.min_groundedness = min_groundedness

    def check(
        self,
        answer: str,
        context: QueryContext,
        verification_result: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str], List[str]]:
        """Run post-answer checks."""
        passed = []
        failed = []

        # Groundedness check
        if context.groundedness_score >= self.min_groundedness:
            passed.append(f"groundedness: {context.groundedness_score:.2f}")
        else:
            failed.append(f"low_groundedness: {context.groundedness_score:.2f}")

        # Evidence verification
        if verification_result:
            if verification_result.get('status') == 'supported':
                passed.append("evidence_verified")
            elif verification_result.get('status') == 'contradicted':
                failed.append("evidence_contradicted")

        # Safety check (simple pattern matching)
        unsafe_patterns = ['i am a doctor', 'medical advice', 'take this medication']
        answer_lower = answer.lower()
        if not any(p in answer_lower for p in unsafe_patterns):
            passed.append("safety_check")
        else:
            failed.append("safety_violation")

        return len(failed) == 0, passed, failed


class EscalationRouter:
    """Route escalations to appropriate targets."""

    def __init__(self):
        self.routes = {
            'human_review': {
                'target': 'review_queue',
                'priority': 'normal',
                'sla_hours': 24,
            },
            'expert_consult': {
                'target': 'expert_pool',
                'priority': 'high',
                'sla_hours': 4,
            },
            'external_tool': {
                'target': 'tool_api',
                'priority': 'immediate',
                'sla_hours': 0,
            },
        }

    def route(
        self,
        context: QueryContext,
        outcome: DecisionOutcome
    ) -> Dict[str, Any]:
        """Route escalation to target."""
        if outcome == DecisionOutcome.ESCALATE_HUMAN:
            if context.risk_tier == RiskTier.T3_CLINICAL:
                route = self.routes['expert_consult']
            else:
                route = self.routes['human_review']
        elif outcome == DecisionOutcome.ESCALATE_TOOL:
            route = self.routes['external_tool']
        else:
            route = None

        return {
            'route': route,
            'context_summary': {
                'query': context.query[:100],
                'risk_tier': context.risk_tier.value,
                'confidence': context.confidence_score,
            },
            'timestamp': datetime.now().isoformat(),
        }


class ExplanationGenerator:
    """Generate user-facing explanations."""

    TEMPLATES = {
        DecisionOutcome.ANSWER: "Based on {source_count} sources, here's what I found:",
        DecisionOutcome.PARTIAL_ANSWER: "I can provide partial information. Note that {caveat}:",
        DecisionOutcome.ABSTAIN: "I don't have enough information to answer this confidently. {reason}",
        DecisionOutcome.ESCALATE_HUMAN: "This question requires expert review. I've forwarded it to {target}.",
        DecisionOutcome.REFUSE: "I cannot answer this type of question. {reason}",
    }

    def generate(
        self,
        outcome: DecisionOutcome,
        context: QueryContext,
        details: Dict[str, Any]
    ) -> str:
        """Generate user-facing explanation."""
        template = self.TEMPLATES.get(outcome, "")

        if outcome == DecisionOutcome.ANSWER:
            return template.format(source_count=len(context.sources))

        elif outcome == DecisionOutcome.PARTIAL_ANSWER:
            caveat = details.get('caveat', 'some information may be incomplete')
            return template.format(caveat=caveat)

        elif outcome == DecisionOutcome.ABSTAIN:
            reason = details.get('reason', "I couldn't find relevant sources.")
            return template.format(reason=reason)

        elif outcome == DecisionOutcome.ESCALATE_HUMAN:
            target = details.get('target', 'our team')
            return template.format(target=target)

        elif outcome == DecisionOutcome.REFUSE:
            reason = details.get('reason', "This falls outside my scope.")
            return template.format(reason=reason)

        return ""


class DecisionLogger:
    """Log all decisions for audit and learning."""

    def __init__(self, log_path: str = "data/audit/decisions"):
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.entries: List[Dict[str, Any]] = []

    def log(
        self,
        context: QueryContext,
        result: DecisionResult
    ):
        """Log decision."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'query_hash': hash(context.query) % 10000000,
            'risk_tier': context.risk_tier.value,
            'evidence_level': context.evidence_level.value,
            'confidence': context.confidence_score,
            'groundedness': context.groundedness_score,
            'outcome': result.outcome.value,
            'checks_passed': result.checks_passed,
            'checks_failed': result.checks_failed,
            'escalation_target': result.escalation_target,
        }
        self.entries.append(entry)

        # Flush periodically
        if len(self.entries) >= 100:
            self._flush()

    def _flush(self):
        """Flush entries to disk."""
        if not self.entries:
            return

        log_file = self.log_path / f"decisions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(log_file, 'a') as f:
            for entry in self.entries:
                f.write(json.dumps(entry) + '\n')
        self.entries = []


class DecisionPolicyEngine:
    """Main decision policy engine."""

    def __init__(self):
        self.tier_classifier = RiskTierClassifier()
        self.evidence_checker = EvidenceSufficiencyChecker()
        self.threshold_manager = ThresholdManager()
        self.decision_matrix = DecisionMatrix()
        self.pre_checker = PreAnswerChecker()
        self.post_checker = PostAnswerChecker()
        self.escalation_router = EscalationRouter()
        self.explanation_gen = ExplanationGenerator()
        self.logger = DecisionLogger()

    def decide(
        self,
        context: QueryContext,
        answer: Optional[str] = None
    ) -> DecisionResult:
        """Make decision on how to respond."""
        all_passed = []
        all_failed = []

        # Pre-answer checks
        pre_ok, pre_passed, pre_failed = self.pre_checker.check(context)
        all_passed.extend(pre_passed)
        all_failed.extend(pre_failed)

        if not pre_ok:
            result = DecisionResult(
                outcome=DecisionOutcome.REFUSE,
                confidence=1.0,
                explanation="Query blocked by safety policy.",
                user_message="I cannot process this request.",
                internal_reason=f"Pre-check failures: {pre_failed}",
                checks_passed=all_passed,
                checks_failed=all_failed,
            )
            self.logger.log(context, result)
            return result

        # Check evidence sufficiency
        evidence_level, evidence_details = self.evidence_checker.check(
            context.sources, context.risk_tier
        )
        context.evidence_level = evidence_level

        # Check thresholds
        thresholds_ok, thresh_passed, thresh_failed = self.threshold_manager.meets_thresholds(context)
        all_passed.extend(thresh_passed)
        all_failed.extend(thresh_failed)

        # Get outcome from matrix
        outcome = self.decision_matrix.get_outcome(
            context.risk_tier,
            evidence_level,
            thresholds_ok
        )

        # Post-answer checks if we have an answer
        if answer and outcome in [DecisionOutcome.ANSWER, DecisionOutcome.PARTIAL_ANSWER]:
            post_ok, post_passed, post_failed = self.post_checker.check(answer, context)
            all_passed.extend(post_passed)
            all_failed.extend(post_failed)

            if not post_ok:
                # Downgrade outcome
                if outcome == DecisionOutcome.ANSWER:
                    outcome = DecisionOutcome.PARTIAL_ANSWER
                else:
                    outcome = DecisionOutcome.ABSTAIN

        # Handle escalation
        escalation_target = None
        if outcome in [DecisionOutcome.ESCALATE_HUMAN, DecisionOutcome.ESCALATE_TOOL]:
            route_info = self.escalation_router.route(context, outcome)
            escalation_target = route_info.get('route', {}).get('target')

        # Generate explanation
        details = {
            'caveat': 'verify with authoritative source' if evidence_level == EvidenceLevel.WEAK else '',
            'reason': '; '.join(all_failed) if all_failed else 'insufficient evidence',
            'target': escalation_target or 'specialist',
        }
        user_message = self.explanation_gen.generate(outcome, context, details)

        result = DecisionResult(
            outcome=outcome,
            confidence=context.confidence_score,
            explanation=user_message,
            user_message=user_message,
            internal_reason=f"Matrix: {context.risk_tier.value}/{evidence_level.value}/{thresholds_ok}",
            escalation_target=escalation_target,
            checks_passed=all_passed,
            checks_failed=all_failed,
        )

        self.logger.log(context, result)
        return result


class DecisionMonitor:
    """Monitor decision quality."""

    def __init__(self):
        self.metrics = {
            'total_decisions': 0,
            'by_outcome': {},
            'by_tier': {},
            'false_answer_rate': 0.0,
            'unnecessary_abstain_rate': 0.0,
        }

    def log_decision(self, context: QueryContext, result: DecisionResult):
        """Log decision for monitoring."""
        self.metrics['total_decisions'] += 1

        outcome = result.outcome.value
        if outcome not in self.metrics['by_outcome']:
            self.metrics['by_outcome'][outcome] = 0
        self.metrics['by_outcome'][outcome] += 1

        tier = context.risk_tier.value
        if tier not in self.metrics['by_tier']:
            self.metrics['by_tier'][tier] = {'total': 0, 'outcomes': {}}
        self.metrics['by_tier'][tier]['total'] += 1

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for dashboard."""
        total = self.metrics['total_decisions']
        if total > 0:
            self.metrics['answer_rate'] = self.metrics['by_outcome'].get('answer', 0) / total
            self.metrics['abstain_rate'] = self.metrics['by_outcome'].get('abstain', 0) / total
            self.metrics['escalation_rate'] = (
                self.metrics['by_outcome'].get('escalate_human', 0) +
                self.metrics['by_outcome'].get('escalate_tool', 0)
            ) / total

        return self.metrics


if __name__ == '__main__':
    # Demo usage
    engine = DecisionPolicyEngine()

    # Test query
    context = QueryContext(
        query="What is the alpha wave frequency range?",
        intent="definition",
        risk_tier=RiskTier.T0_DEFINITION,
        user_role="researcher",
        evidence_level=EvidenceLevel.MODERATE,
        confidence_score=0.85,
        groundedness_score=0.9,
        entropy=0.3,
        sources=[
            {'doc_id': 'eeg_basics', 'authority_level': 'peer_reviewed'},
        ],
    )

    result = engine.decide(context)

    print(f"Outcome: {result.outcome.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"User message: {result.user_message}")
    print(f"Checks passed: {result.checks_passed}")
    print(f"Checks failed: {result.checks_failed}")
