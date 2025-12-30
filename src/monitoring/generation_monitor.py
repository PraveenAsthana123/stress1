"""
Phase 3: Generation & Reasoning Analysis Monitoring

Monitors prompt integrity, hallucination detection, and generation quality
for RAG systems.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
import re
import hashlib
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================

class PromptRisk(Enum):
    """Prompt risk levels."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HallucinationType(Enum):
    """Types of hallucination."""
    FACTUAL = "factual"          # Made up facts
    NUMERIC = "numeric"          # Incorrect numbers
    CITATION = "citation"        # Fake citations
    ENTITY = "entity"            # Made up entities
    TEMPORAL = "temporal"        # Wrong dates/times
    LOGICAL = "logical"          # Logical inconsistencies


class GenerationQuality(Enum):
    """Generation quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


class GroundingLevel(Enum):
    """Grounding in retrieved context."""
    FULLY_GROUNDED = "fully_grounded"
    MOSTLY_GROUNDED = "mostly_grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    UNGROUNDED = "ungrounded"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PromptCheck:
    """Result of prompt integrity check."""
    prompt_hash: str
    risk_level: PromptRisk
    issues: List[Dict[str, Any]]
    sanitized_prompt: str
    modifications: List[str]
    passed: bool


@dataclass
class HallucinationDetection:
    """Result of hallucination detection."""
    claim_id: str
    claim_text: str
    hallucination_type: Optional[HallucinationType]
    confidence: float
    evidence_found: bool
    supporting_chunks: List[str]
    is_hallucination: bool


@dataclass
class GroundingAnalysis:
    """Analysis of generation grounding."""
    response_id: str
    grounding_level: GroundingLevel
    grounded_claims: int
    ungrounded_claims: int
    grounding_score: float  # 0-1
    ungrounded_details: List[Dict[str, Any]]


@dataclass
class GenerationMetrics:
    """Metrics for a single generation."""
    response_id: str
    token_count: int
    latency_ms: float
    grounding_score: float
    hallucination_count: int
    quality: GenerationQuality
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GenerationReport:
    """Overall generation quality report."""
    total_generations: int
    avg_grounding_score: float
    hallucination_rate: float
    avg_latency_ms: float
    quality_distribution: Dict[GenerationQuality, int]
    passed: bool
    issues: List[str]
    recommendations: List[str]


# =============================================================================
# Phase 3.1: Prompt Integrity Checker
# =============================================================================

class PromptIntegrityChecker:
    """
    Validates prompt integrity and safety.

    Purpose: Ensure prompts are safe and well-formed
    Measurement: Injection detection rate, sanitization effectiveness
    Pass/Fail: No high-risk prompts pass through
    """

    def __init__(self):
        # Injection patterns
        self.injection_patterns = [
            r"ignore\s+(previous|above|all)\s+instructions?",
            r"disregard\s+(previous|above|all)",
            r"you\s+are\s+now\s+",
            r"pretend\s+you\s+are",
            r"act\s+as\s+if",
            r"system\s*:\s*",
            r"\[INST\]",
            r"<\|im_start\|>",
            r"<\|system\|>",
        ]

        # Sensitive patterns
        self.sensitive_patterns = [
            r"password",
            r"api[_\s]?key",
            r"secret[_\s]?key",
            r"access[_\s]?token",
            r"private[_\s]?key",
        ]

        # Role-specific blocklists
        self.blocklist = [
            "jailbreak",
            "bypass",
            "override",
            "sudo",
            "admin mode",
        ]

    def check_prompt(self, prompt: str) -> PromptCheck:
        """Check prompt for integrity issues."""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        issues = []
        modifications = []
        sanitized = prompt

        # Check injection patterns
        for pattern in self.injection_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                issues.append({
                    "type": "injection",
                    "pattern": pattern,
                    "severity": "high"
                })
                # Remove injection attempt
                sanitized = re.sub(pattern, "[REMOVED]", sanitized, flags=re.IGNORECASE)
                modifications.append(f"Removed injection pattern: {pattern}")

        # Check sensitive patterns
        for pattern in self.sensitive_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                issues.append({
                    "type": "sensitive",
                    "pattern": pattern,
                    "severity": "medium"
                })

        # Check blocklist
        prompt_lower = prompt.lower()
        for blocked in self.blocklist:
            if blocked in prompt_lower:
                issues.append({
                    "type": "blocklist",
                    "term": blocked,
                    "severity": "high"
                })
                sanitized = sanitized.replace(blocked, "[BLOCKED]")
                modifications.append(f"Blocked term: {blocked}")

        # Determine risk level
        risk_level = self._assess_risk(issues)
        passed = risk_level in [PromptRisk.SAFE, PromptRisk.LOW]

        return PromptCheck(
            prompt_hash=prompt_hash,
            risk_level=risk_level,
            issues=issues,
            sanitized_prompt=sanitized,
            modifications=modifications,
            passed=passed
        )

    def _assess_risk(self, issues: List[Dict[str, Any]]) -> PromptRisk:
        """Assess overall risk level."""
        if not issues:
            return PromptRisk.SAFE

        severities = [i.get("severity", "low") for i in issues]

        if "critical" in severities:
            return PromptRisk.CRITICAL
        if severities.count("high") >= 2:
            return PromptRisk.CRITICAL
        if "high" in severities:
            return PromptRisk.HIGH
        if "medium" in severities:
            return PromptRisk.MEDIUM
        return PromptRisk.LOW

    def validate_template(
        self,
        template: str,
        required_placeholders: List[str]
    ) -> Tuple[bool, List[str]]:
        """Validate prompt template."""
        issues = []

        # Check for required placeholders
        for placeholder in required_placeholders:
            if f"{{{placeholder}}}" not in template:
                issues.append(f"Missing required placeholder: {placeholder}")

        # Check for balanced braces
        open_braces = template.count("{")
        close_braces = template.count("}")
        if open_braces != close_braces:
            issues.append("Unbalanced braces in template")

        return len(issues) == 0, issues


# =============================================================================
# Phase 3.2: Hallucination Detector
# =============================================================================

class HallucinationDetector:
    """
    Detects hallucinations in generated content.

    Purpose: Identify claims not supported by retrieved context
    Measurement: Hallucination rate, false positive rate
    Pass/Fail: Hallucination rate < 5%
    """

    def __init__(
        self,
        hallucination_threshold: float = 0.05,
        confidence_threshold: float = 0.7
    ):
        self.hallucination_threshold = hallucination_threshold
        self.confidence_threshold = confidence_threshold
        self.detection_history: List[HallucinationDetection] = []

    def detect_hallucination(
        self,
        claim_id: str,
        claim_text: str,
        context_chunks: List[str],
        claim_type: Optional[str] = None
    ) -> HallucinationDetection:
        """Detect if a claim is hallucinated."""
        # Search for evidence in context
        evidence_found, supporting_chunks, confidence = \
            self._search_evidence(claim_text, context_chunks)

        # Determine hallucination type if applicable
        hallucination_type = None
        is_hallucination = not evidence_found and confidence >= self.confidence_threshold

        if is_hallucination:
            hallucination_type = self._classify_hallucination(claim_text, claim_type)

        detection = HallucinationDetection(
            claim_id=claim_id,
            claim_text=claim_text,
            hallucination_type=hallucination_type,
            confidence=confidence,
            evidence_found=evidence_found,
            supporting_chunks=supporting_chunks,
            is_hallucination=is_hallucination
        )

        self.detection_history.append(detection)
        return detection

    def _search_evidence(
        self,
        claim: str,
        context_chunks: List[str]
    ) -> Tuple[bool, List[str], float]:
        """Search for evidence supporting the claim."""
        supporting = []
        claim_tokens = set(claim.lower().split())

        for chunk in context_chunks:
            chunk_tokens = set(chunk.lower().split())
            overlap = len(claim_tokens & chunk_tokens)
            overlap_ratio = overlap / len(claim_tokens) if claim_tokens else 0

            if overlap_ratio > 0.5:
                supporting.append(chunk)

        evidence_found = len(supporting) > 0
        confidence = min(len(supporting) / 3, 1.0) if supporting else 0.8

        return evidence_found, supporting, confidence

    def _classify_hallucination(
        self,
        claim: str,
        claim_type: Optional[str]
    ) -> HallucinationType:
        """Classify the type of hallucination."""
        # Check for numeric hallucination
        if re.search(r'\d+\.?\d*%?', claim):
            return HallucinationType.NUMERIC

        # Check for citation hallucination
        if re.search(r'\(\d{4}\)|\[\d+\]|et al\.', claim):
            return HallucinationType.CITATION

        # Check for temporal hallucination
        if re.search(r'\b(19|20)\d{2}\b|january|february|march|april|may|june|'
                    r'july|august|september|october|november|december', claim, re.I):
            return HallucinationType.TEMPORAL

        # Default to factual
        return HallucinationType.FACTUAL

    def get_hallucination_rate(self) -> float:
        """Get hallucination rate from detection history."""
        if not self.detection_history:
            return 0.0
        hallucinated = sum(1 for d in self.detection_history if d.is_hallucination)
        return hallucinated / len(self.detection_history)

    def check_threshold(self) -> Tuple[bool, float]:
        """Check if hallucination rate is within threshold."""
        rate = self.get_hallucination_rate()
        return rate <= self.hallucination_threshold, rate

    def get_statistics(self) -> Dict[str, Any]:
        """Get hallucination statistics."""
        if not self.detection_history:
            return {"total": 0, "hallucinations": 0, "rate": 0.0}

        hallucinations = [d for d in self.detection_history if d.is_hallucination]

        type_counts = {}
        for h in hallucinations:
            if h.hallucination_type:
                t = h.hallucination_type.value
                type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_claims": len(self.detection_history),
            "hallucinations": len(hallucinations),
            "rate": len(hallucinations) / len(self.detection_history),
            "by_type": type_counts,
            "passed": self.get_hallucination_rate() <= self.hallucination_threshold
        }

    def reset(self):
        """Reset detection history."""
        self.detection_history = []


# =============================================================================
# Phase 3.3: Generation Quality Analyzer
# =============================================================================

class GenerationQualityAnalyzer:
    """
    Analyzes overall generation quality.

    Purpose: Measure generation accuracy, relevance, and groundedness
    Measurement: Grounding score, quality metrics, latency
    Pass/Fail: Grounding > 0.8, Quality >= ACCEPTABLE
    """

    def __init__(
        self,
        grounding_threshold: float = 0.8,
        latency_threshold_ms: float = 5000.0
    ):
        self.grounding_threshold = grounding_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.metrics_history: List[GenerationMetrics] = []

    def analyze_grounding(
        self,
        response_id: str,
        claims: List[str],
        context_chunks: List[str]
    ) -> GroundingAnalysis:
        """Analyze grounding of response in context."""
        grounded = 0
        ungrounded = 0
        ungrounded_details = []

        for i, claim in enumerate(claims):
            is_grounded = self._check_claim_grounding(claim, context_chunks)
            if is_grounded:
                grounded += 1
            else:
                ungrounded += 1
                ungrounded_details.append({
                    "claim_index": i,
                    "claim": claim[:100],
                    "reason": "No supporting evidence in context"
                })

        total = grounded + ungrounded
        grounding_score = grounded / total if total > 0 else 1.0

        # Determine grounding level
        if grounding_score >= 0.95:
            level = GroundingLevel.FULLY_GROUNDED
        elif grounding_score >= 0.8:
            level = GroundingLevel.MOSTLY_GROUNDED
        elif grounding_score >= 0.5:
            level = GroundingLevel.PARTIALLY_GROUNDED
        else:
            level = GroundingLevel.UNGROUNDED

        return GroundingAnalysis(
            response_id=response_id,
            grounding_level=level,
            grounded_claims=grounded,
            ungrounded_claims=ungrounded,
            grounding_score=grounding_score,
            ungrounded_details=ungrounded_details
        )

    def _check_claim_grounding(
        self,
        claim: str,
        context_chunks: List[str]
    ) -> bool:
        """Check if claim is grounded in context."""
        claim_tokens = set(claim.lower().split())

        for chunk in context_chunks:
            chunk_tokens = set(chunk.lower().split())
            overlap = len(claim_tokens & chunk_tokens)

            if len(claim_tokens) > 0:
                overlap_ratio = overlap / len(claim_tokens)
                if overlap_ratio > 0.4:
                    return True

        return False

    def record_generation(
        self,
        response_id: str,
        response_text: str,
        latency_ms: float,
        grounding_analysis: GroundingAnalysis,
        hallucination_count: int = 0
    ) -> GenerationMetrics:
        """Record metrics for a generation."""
        # Count tokens (approximate)
        token_count = len(response_text.split())

        # Determine quality
        quality = self._assess_quality(
            grounding_analysis.grounding_score,
            hallucination_count,
            latency_ms
        )

        metrics = GenerationMetrics(
            response_id=response_id,
            token_count=token_count,
            latency_ms=latency_ms,
            grounding_score=grounding_analysis.grounding_score,
            hallucination_count=hallucination_count,
            quality=quality
        )

        self.metrics_history.append(metrics)
        return metrics

    def _assess_quality(
        self,
        grounding_score: float,
        hallucination_count: int,
        latency_ms: float
    ) -> GenerationQuality:
        """Assess generation quality."""
        score = 0

        # Grounding contribution
        if grounding_score >= 0.95:
            score += 3
        elif grounding_score >= 0.8:
            score += 2
        elif grounding_score >= 0.6:
            score += 1

        # Hallucination penalty
        if hallucination_count == 0:
            score += 2
        elif hallucination_count <= 1:
            score += 1
        elif hallucination_count > 3:
            score -= 1

        # Latency contribution
        if latency_ms <= self.latency_threshold_ms / 2:
            score += 1

        # Map to quality
        if score >= 5:
            return GenerationQuality.EXCELLENT
        elif score >= 4:
            return GenerationQuality.GOOD
        elif score >= 2:
            return GenerationQuality.ACCEPTABLE
        elif score >= 0:
            return GenerationQuality.POOR
        else:
            return GenerationQuality.FAILED

    def get_report(self) -> GenerationReport:
        """Generate quality report."""
        if not self.metrics_history:
            return GenerationReport(
                total_generations=0,
                avg_grounding_score=0.0,
                hallucination_rate=0.0,
                avg_latency_ms=0.0,
                quality_distribution={q: 0 for q in GenerationQuality},
                passed=False,
                issues=["No generations recorded"],
                recommendations=[]
            )

        # Calculate averages
        avg_grounding = sum(m.grounding_score for m in self.metrics_history) / \
                       len(self.metrics_history)
        total_hallucinations = sum(m.hallucination_count for m in self.metrics_history)
        hallucination_rate = total_hallucinations / len(self.metrics_history)
        avg_latency = sum(m.latency_ms for m in self.metrics_history) / \
                     len(self.metrics_history)

        # Quality distribution
        quality_dist = {q: 0 for q in GenerationQuality}
        for m in self.metrics_history:
            quality_dist[m.quality] += 1

        # Check pass criteria
        grounding_passed = avg_grounding >= self.grounding_threshold
        latency_passed = avg_latency <= self.latency_threshold_ms
        passed = grounding_passed and latency_passed

        # Identify issues
        issues = []
        if not grounding_passed:
            issues.append(
                f"Grounding {avg_grounding:.2%} below threshold {self.grounding_threshold:.2%}"
            )
        if not latency_passed:
            issues.append(
                f"Latency {avg_latency:.0f}ms above threshold {self.latency_threshold_ms:.0f}ms"
            )
        if hallucination_rate > 0.05:
            issues.append(f"High hallucination rate: {hallucination_rate:.2%}")

        # Recommendations
        recommendations = []
        if not grounding_passed:
            recommendations.append("Improve context retrieval or prompt engineering")
        if hallucination_rate > 0.05:
            recommendations.append("Add fact-checking or claim verification")
        if not latency_passed:
            recommendations.append("Optimize generation or use faster model")

        return GenerationReport(
            total_generations=len(self.metrics_history),
            avg_grounding_score=avg_grounding,
            hallucination_rate=hallucination_rate,
            avg_latency_ms=avg_latency,
            quality_distribution=quality_dist,
            passed=passed,
            issues=issues,
            recommendations=recommendations
        )

    def reset(self):
        """Reset metrics history."""
        self.metrics_history = []


# =============================================================================
# Phase 3: Unified Generation Monitor
# =============================================================================

class GenerationPhaseMonitor:
    """
    Unified monitor for Phase 3: Generation & Reasoning Analysis.

    Combines prompt checking, hallucination detection, and quality analysis.
    """

    def __init__(self):
        self.prompt_checker = PromptIntegrityChecker()
        self.hallucination_detector = HallucinationDetector()
        self.quality_analyzer = GenerationQualityAnalyzer()

    def run_full_analysis(
        self,
        generations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run full Phase 3 analysis.

        Each generation should contain:
        - prompt: The input prompt
        - response: The generated response
        - claims: List of claims extracted from response
        - context_chunks: Retrieved context used for generation
        - latency_ms: Generation latency
        """
        results = {
            "prompt_checks": [],
            "hallucinations": [],
            "generations": []
        }

        for gen in generations:
            response_id = gen.get("response_id", hashlib.md5(
                gen.get("response", "").encode()
            ).hexdigest()[:12])

            # Check prompt integrity
            prompt = gen.get("prompt", "")
            prompt_check = self.prompt_checker.check_prompt(prompt)
            results["prompt_checks"].append({
                "response_id": response_id,
                "risk_level": prompt_check.risk_level.value,
                "passed": prompt_check.passed,
                "issues_count": len(prompt_check.issues)
            })

            # Detect hallucinations
            claims = gen.get("claims", [])
            context = gen.get("context_chunks", [])

            for i, claim in enumerate(claims):
                detection = self.hallucination_detector.detect_hallucination(
                    claim_id=f"{response_id}_{i}",
                    claim_text=claim,
                    context_chunks=context
                )
                if detection.is_hallucination:
                    results["hallucinations"].append({
                        "response_id": response_id,
                        "claim": claim[:100],
                        "type": detection.hallucination_type.value if detection.hallucination_type else None,
                        "confidence": detection.confidence
                    })

            # Analyze grounding
            grounding = self.quality_analyzer.analyze_grounding(
                response_id=response_id,
                claims=claims,
                context_chunks=context
            )

            # Record metrics
            metrics = self.quality_analyzer.record_generation(
                response_id=response_id,
                response_text=gen.get("response", ""),
                latency_ms=gen.get("latency_ms", 0.0),
                grounding_analysis=grounding,
                hallucination_count=len([h for h in results["hallucinations"]
                                        if h["response_id"] == response_id])
            )

            results["generations"].append({
                "response_id": response_id,
                "grounding_level": grounding.grounding_level.value,
                "grounding_score": grounding.grounding_score,
                "quality": metrics.quality.value
            })

        # Get overall statistics
        hallucination_stats = self.hallucination_detector.get_statistics()
        generation_report = self.quality_analyzer.get_report()

        # Compile summary
        prompt_passed = all(p["passed"] for p in results["prompt_checks"])

        results["summary"] = {
            "total_generations": len(generations),
            "prompt_check_passed": prompt_passed,
            "hallucination_rate": hallucination_stats["rate"],
            "hallucination_passed": hallucination_stats["passed"],
            "avg_grounding_score": generation_report.avg_grounding_score,
            "avg_latency_ms": generation_report.avg_latency_ms,
            "quality_distribution": {
                q.value: c for q, c in generation_report.quality_distribution.items()
            },
            "overall_passed": (
                prompt_passed and
                hallucination_stats["passed"] and
                generation_report.passed
            ),
            "issues": generation_report.issues,
            "recommendations": generation_report.recommendations
        }

        return results


__all__ = [
    # Enums
    "PromptRisk",
    "HallucinationType",
    "GenerationQuality",
    "GroundingLevel",
    # Data classes
    "PromptCheck",
    "HallucinationDetection",
    "GroundingAnalysis",
    "GenerationMetrics",
    "GenerationReport",
    # Monitors
    "PromptIntegrityChecker",
    "HallucinationDetector",
    "GenerationQualityAnalyzer",
    "GenerationPhaseMonitor",
]
