"""
Phase 5-7: Analysis Monitoring (Non-Agent Version)

Since this paper has no agent, these phases cover:
- Phase 8: Explainability, Interpretability & Trust
- Phase 9: Robustness & Sensitivity
- Phase 10: Statistical Validation
- Phase 11: Benchmarking & Comparative Analysis
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================

class ExplainabilityLevel(Enum):
    """Explainability levels."""
    FULL = "full"           # Complete explanation with all details
    PARTIAL = "partial"     # Key factors explained
    MINIMAL = "minimal"     # Basic explanation only
    NONE = "none"          # No explanation available


class RobustnessLevel(Enum):
    """Robustness assessment levels."""
    ROBUST = "robust"           # Stable under perturbations
    MODERATE = "moderate"       # Some sensitivity
    FRAGILE = "fragile"         # High sensitivity
    UNTESTED = "untested"


class StatisticalSignificance(Enum):
    """Statistical significance levels."""
    HIGHLY_SIGNIFICANT = "p<0.001"
    SIGNIFICANT = "p<0.01"
    MARGINALLY_SIGNIFICANT = "p<0.05"
    NOT_SIGNIFICANT = "p>=0.05"


class BenchmarkRank(Enum):
    """Benchmark ranking."""
    SOTA = "state_of_the_art"
    COMPETITIVE = "competitive"
    BASELINE = "baseline"
    BELOW_BASELINE = "below_baseline"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExplanationQuality:
    """Quality metrics for an explanation."""
    explanation_id: str
    level: ExplainabilityLevel
    completeness: float      # 0-1
    faithfulness: float      # 0-1
    consistency: float       # 0-1
    human_readable: bool
    includes_evidence: bool
    includes_confidence: bool


@dataclass
class RobustnessTest:
    """Result of a robustness test."""
    test_id: str
    perturbation_type: str
    perturbation_magnitude: float
    original_output: Any
    perturbed_output: Any
    output_changed: bool
    change_magnitude: float
    passed: bool


@dataclass
class StatisticalTest:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float]
    significance: StatisticalSignificance
    confidence_interval: Tuple[float, float]
    sample_size: int


@dataclass
class BenchmarkResult:
    """Result of benchmark comparison."""
    model_name: str
    dataset: str
    metric_name: str
    score: float
    baseline_score: float
    sota_score: float
    rank: BenchmarkRank
    percentile: float


@dataclass
class AnalysisReport:
    """Comprehensive analysis report."""
    explainability_score: float
    robustness_score: float
    statistical_validity: bool
    benchmark_rank: BenchmarkRank
    detailed_metrics: Dict[str, Any]
    passed: bool
    issues: List[str]
    recommendations: List[str]


# =============================================================================
# Phase 8: Explainability Analyzer
# =============================================================================

class ExplainabilityAnalyzer:
    """
    Analyzes explainability and interpretability.

    Purpose: Ensure explanations are complete, faithful, and understandable
    Measurement: Completeness, faithfulness, consistency scores
    Pass/Fail: Average explainability score > 0.7
    """

    def __init__(self, min_score: float = 0.7):
        self.min_score = min_score
        self.explanations: List[ExplanationQuality] = []

    def analyze_explanation(
        self,
        explanation_id: str,
        explanation_text: str,
        prediction: Any,
        evidence_chunks: List[str],
        confidence: Optional[float] = None
    ) -> ExplanationQuality:
        """Analyze quality of an explanation."""
        # Check completeness
        completeness = self._assess_completeness(
            explanation_text, evidence_chunks
        )

        # Check faithfulness (does explanation reflect actual reasoning)
        faithfulness = self._assess_faithfulness(
            explanation_text, evidence_chunks, prediction
        )

        # Check consistency
        consistency = self._assess_consistency(explanation_text)

        # Check human readability
        human_readable = self._is_human_readable(explanation_text)

        # Check for evidence references
        includes_evidence = any(
            chunk[:30].lower() in explanation_text.lower()
            for chunk in evidence_chunks
        ) if evidence_chunks else True

        # Check for confidence
        includes_confidence = (
            confidence is not None and
            (str(confidence) in explanation_text or
             f"{confidence:.0%}" in explanation_text or
             f"{confidence:.1%}" in explanation_text)
        )

        # Determine level
        avg_score = (completeness + faithfulness + consistency) / 3
        if avg_score >= 0.8 and human_readable:
            level = ExplainabilityLevel.FULL
        elif avg_score >= 0.6:
            level = ExplainabilityLevel.PARTIAL
        elif avg_score >= 0.3:
            level = ExplainabilityLevel.MINIMAL
        else:
            level = ExplainabilityLevel.NONE

        quality = ExplanationQuality(
            explanation_id=explanation_id,
            level=level,
            completeness=completeness,
            faithfulness=faithfulness,
            consistency=consistency,
            human_readable=human_readable,
            includes_evidence=includes_evidence,
            includes_confidence=includes_confidence
        )

        self.explanations.append(quality)
        return quality

    def _assess_completeness(
        self,
        explanation: str,
        evidence: List[str]
    ) -> float:
        """Assess completeness of explanation."""
        if not explanation:
            return 0.0

        score = 0.5  # Base score for having an explanation

        # Check length (longer is generally more complete)
        word_count = len(explanation.split())
        if word_count >= 50:
            score += 0.2
        elif word_count >= 20:
            score += 0.1

        # Check for key components
        components = ["because", "therefore", "since", "as a result", "evidence"]
        for component in components:
            if component in explanation.lower():
                score += 0.06

        return min(1.0, score)

    def _assess_faithfulness(
        self,
        explanation: str,
        evidence: List[str],
        prediction: Any
    ) -> float:
        """Assess faithfulness of explanation to actual reasoning."""
        if not explanation or not evidence:
            return 0.5

        # Check evidence overlap
        explanation_tokens = set(explanation.lower().split())
        evidence_tokens = set()
        for chunk in evidence:
            evidence_tokens.update(chunk.lower().split())

        overlap = len(explanation_tokens & evidence_tokens)
        overlap_ratio = overlap / len(explanation_tokens) if explanation_tokens else 0

        return min(1.0, overlap_ratio + 0.3)

    def _assess_consistency(self, explanation: str) -> float:
        """Assess internal consistency of explanation."""
        if not explanation:
            return 0.0

        # Check for contradictory phrases
        contradictions = [
            ("is", "is not"),
            ("can", "cannot"),
            ("will", "will not"),
            ("high", "low"),
            ("increase", "decrease"),
        ]

        for pos, neg in contradictions:
            if pos in explanation.lower() and neg in explanation.lower():
                return 0.6  # Some inconsistency detected

        return 0.9  # Default to high consistency

    def _is_human_readable(self, explanation: str) -> bool:
        """Check if explanation is human readable."""
        if not explanation:
            return False

        # Check for readable structure
        word_count = len(explanation.split())
        if word_count < 5:
            return False

        # Check average word length (very long words indicate jargon)
        words = explanation.split()
        avg_word_length = sum(len(w) for w in words) / len(words)
        if avg_word_length > 12:
            return False

        return True

    def get_average_score(self) -> float:
        """Get average explainability score."""
        if not self.explanations:
            return 0.0

        scores = [
            (e.completeness + e.faithfulness + e.consistency) / 3
            for e in self.explanations
        ]
        return np.mean(scores)

    def check_threshold(self) -> Tuple[bool, float]:
        """Check if average score meets threshold."""
        avg = self.get_average_score()
        return avg >= self.min_score, avg


# =============================================================================
# Phase 9: Robustness Analyzer
# =============================================================================

class RobustnessAnalyzer:
    """
    Analyzes system robustness to perturbations.

    Purpose: Test stability under various perturbations
    Measurement: Perturbation tolerance, output stability
    Pass/Fail: >90% tests pass robustness criteria
    """

    def __init__(self, stability_threshold: float = 0.1):
        self.stability_threshold = stability_threshold
        self.tests: List[RobustnessTest] = []

    def test_perturbation(
        self,
        test_id: str,
        perturbation_type: str,
        perturbation_magnitude: float,
        original_output: Any,
        perturbed_output: Any,
        output_comparison_fn: Optional[callable] = None
    ) -> RobustnessTest:
        """Test robustness to a specific perturbation."""
        # Compare outputs
        if output_comparison_fn:
            change_magnitude = output_comparison_fn(original_output, perturbed_output)
        else:
            change_magnitude = self._default_comparison(
                original_output, perturbed_output
            )

        output_changed = change_magnitude > 0.0
        passed = change_magnitude <= self.stability_threshold

        test = RobustnessTest(
            test_id=test_id,
            perturbation_type=perturbation_type,
            perturbation_magnitude=perturbation_magnitude,
            original_output=original_output,
            perturbed_output=perturbed_output,
            output_changed=output_changed,
            change_magnitude=change_magnitude,
            passed=passed
        )

        self.tests.append(test)
        return test

    def _default_comparison(self, original: Any, perturbed: Any) -> float:
        """Default output comparison."""
        if isinstance(original, (int, float)) and isinstance(perturbed, (int, float)):
            if original == 0:
                return abs(perturbed)
            return abs(original - perturbed) / abs(original)

        if isinstance(original, str) and isinstance(perturbed, str):
            # Jaccard distance for strings
            orig_tokens = set(original.lower().split())
            pert_tokens = set(perturbed.lower().split())
            if not orig_tokens and not pert_tokens:
                return 0.0
            intersection = len(orig_tokens & pert_tokens)
            union = len(orig_tokens | pert_tokens)
            return 1 - (intersection / union) if union > 0 else 0.0

        if isinstance(original, np.ndarray) and isinstance(perturbed, np.ndarray):
            return float(np.linalg.norm(original - perturbed) / (np.linalg.norm(original) + 1e-10))

        # Default: binary comparison
        return 0.0 if original == perturbed else 1.0

    def get_robustness_score(self) -> float:
        """Get overall robustness score (pass rate)."""
        if not self.tests:
            return 1.0
        passed = sum(1 for t in self.tests if t.passed)
        return passed / len(self.tests)

    def get_robustness_level(self) -> RobustnessLevel:
        """Get robustness level based on pass rate."""
        score = self.get_robustness_score()
        if score >= 0.95:
            return RobustnessLevel.ROBUST
        elif score >= 0.8:
            return RobustnessLevel.MODERATE
        elif score >= 0.5:
            return RobustnessLevel.FRAGILE
        else:
            return RobustnessLevel.FRAGILE

    def get_failure_analysis(self) -> Dict[str, Any]:
        """Analyze failed robustness tests."""
        failed = [t for t in self.tests if not t.passed]

        if not failed:
            return {"failures": 0, "analysis": "All tests passed"}

        by_type = {}
        for t in failed:
            if t.perturbation_type not in by_type:
                by_type[t.perturbation_type] = []
            by_type[t.perturbation_type].append(t.change_magnitude)

        return {
            "failures": len(failed),
            "failure_rate": len(failed) / len(self.tests),
            "by_perturbation_type": {
                k: {"count": len(v), "avg_change": np.mean(v)}
                for k, v in by_type.items()
            },
            "worst_case": max(t.change_magnitude for t in failed)
        }


# =============================================================================
# Phase 10: Statistical Validator
# =============================================================================

class StatisticalValidator:
    """
    Validates results with statistical rigor.

    Purpose: Ensure statistical validity of claims
    Measurement: p-values, effect sizes, confidence intervals
    Pass/Fail: Claims supported by p < 0.05 with adequate effect size
    """

    def __init__(self, alpha: float = 0.05, min_effect_size: float = 0.2):
        self.alpha = alpha
        self.min_effect_size = min_effect_size
        self.tests: List[StatisticalTest] = []

    def t_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        test_name: str = "t_test",
        paired: bool = False
    ) -> StatisticalTest:
        """Perform t-test between two groups."""
        if paired:
            statistic, p_value = stats.ttest_rel(group1, group2)
        else:
            statistic, p_value = stats.ttest_ind(group1, group2)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(group1) - 1) * np.var(group1) + (len(group2) - 1) * np.var(group2)) /
            (len(group1) + len(group2) - 2)
        )
        effect_size = abs(np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

        # Confidence interval for difference
        diff = np.mean(group1) - np.mean(group2)
        se = pooled_std * np.sqrt(1/len(group1) + 1/len(group2))
        ci = (diff - 1.96 * se, diff + 1.96 * se)

        # Determine significance
        significance = self._get_significance(p_value)

        test = StatisticalTest(
            test_name=test_name,
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            significance=significance,
            confidence_interval=ci,
            sample_size=len(group1) + len(group2)
        )

        self.tests.append(test)
        return test

    def wilcoxon_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        test_name: str = "wilcoxon"
    ) -> StatisticalTest:
        """Perform Wilcoxon signed-rank test."""
        statistic, p_value = stats.wilcoxon(group1, group2)

        # Effect size (rank-biserial correlation)
        n = len(group1)
        effect_size = 1 - (2 * statistic) / (n * (n + 1))

        significance = self._get_significance(p_value)

        test = StatisticalTest(
            test_name=test_name,
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            significance=significance,
            confidence_interval=(float('nan'), float('nan')),
            sample_size=len(group1) + len(group2)
        )

        self.tests.append(test)
        return test

    def bootstrap_ci(
        self,
        data: np.ndarray,
        statistic_fn: callable,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, Tuple[float, float]]:
        """Calculate bootstrap confidence interval."""
        n = len(data)
        statistics = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            stat = statistic_fn(sample)
            statistics.append(stat)

        statistics = np.array(statistics)
        alpha = (1 - confidence) / 2
        ci_lower = np.percentile(statistics, alpha * 100)
        ci_upper = np.percentile(statistics, (1 - alpha) * 100)

        return float(statistic_fn(data)), (float(ci_lower), float(ci_upper))

    def _get_significance(self, p_value: float) -> StatisticalSignificance:
        """Determine significance level from p-value."""
        if p_value < 0.001:
            return StatisticalSignificance.HIGHLY_SIGNIFICANT
        elif p_value < 0.01:
            return StatisticalSignificance.SIGNIFICANT
        elif p_value < 0.05:
            return StatisticalSignificance.MARGINALLY_SIGNIFICANT
        else:
            return StatisticalSignificance.NOT_SIGNIFICANT

    def validate_claim(
        self,
        test: StatisticalTest
    ) -> Tuple[bool, str]:
        """Validate if a claim is statistically supported."""
        # Check significance
        if test.p_value >= self.alpha:
            return False, f"Not significant (p={test.p_value:.4f})"

        # Check effect size
        if test.effect_size is not None and test.effect_size < self.min_effect_size:
            return False, f"Effect size too small (d={test.effect_size:.3f})"

        return True, "Claim statistically supported"

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all statistical tests."""
        if not self.tests:
            return {"total_tests": 0}

        significant = sum(
            1 for t in self.tests
            if t.significance != StatisticalSignificance.NOT_SIGNIFICANT
        )

        return {
            "total_tests": len(self.tests),
            "significant": significant,
            "significance_rate": significant / len(self.tests),
            "by_significance": {
                s.value: sum(1 for t in self.tests if t.significance == s)
                for s in StatisticalSignificance
            }
        }


# =============================================================================
# Phase 11: Benchmark Analyzer
# =============================================================================

class BenchmarkAnalyzer:
    """
    Analyzes performance against benchmarks.

    Purpose: Compare against baselines and state-of-the-art
    Measurement: Relative performance, ranking
    Pass/Fail: Performance >= baseline
    """

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.sota: Dict[str, Dict[str, float]] = {}

    def register_baseline(
        self,
        dataset: str,
        metric_name: str,
        score: float
    ):
        """Register a baseline score."""
        if dataset not in self.baselines:
            self.baselines[dataset] = {}
        self.baselines[dataset][metric_name] = score

    def register_sota(
        self,
        dataset: str,
        metric_name: str,
        score: float
    ):
        """Register state-of-the-art score."""
        if dataset not in self.sota:
            self.sota[dataset] = {}
        self.sota[dataset][metric_name] = score

    def evaluate(
        self,
        model_name: str,
        dataset: str,
        metric_name: str,
        score: float
    ) -> BenchmarkResult:
        """Evaluate model against benchmarks."""
        baseline = self.baselines.get(dataset, {}).get(metric_name, 0.0)
        sota = self.sota.get(dataset, {}).get(metric_name, 1.0)

        # Determine rank
        if sota > 0 and score >= sota * 0.99:
            rank = BenchmarkRank.SOTA
        elif baseline > 0 and score >= baseline * 1.1:
            rank = BenchmarkRank.COMPETITIVE
        elif baseline > 0 and score >= baseline:
            rank = BenchmarkRank.BASELINE
        else:
            rank = BenchmarkRank.BELOW_BASELINE

        # Calculate percentile (between baseline and SOTA)
        if sota > baseline:
            percentile = (score - baseline) / (sota - baseline) * 100
            percentile = max(0, min(100, percentile))
        else:
            percentile = 50.0 if score >= baseline else 0.0

        result = BenchmarkResult(
            model_name=model_name,
            dataset=dataset,
            metric_name=metric_name,
            score=score,
            baseline_score=baseline,
            sota_score=sota,
            rank=rank,
            percentile=percentile
        )

        self.results.append(result)
        return result

    def get_comparison_table(self) -> Dict[str, Dict[str, Any]]:
        """Get comparison table of all results."""
        table = {}
        for r in self.results:
            key = f"{r.dataset}_{r.metric_name}"
            if key not in table:
                table[key] = {
                    "dataset": r.dataset,
                    "metric": r.metric_name,
                    "baseline": r.baseline_score,
                    "sota": r.sota_score,
                    "models": {}
                }
            table[key]["models"][r.model_name] = {
                "score": r.score,
                "rank": r.rank.value,
                "percentile": r.percentile
            }
        return table

    def get_best_rank(self) -> BenchmarkRank:
        """Get best achieved rank."""
        if not self.results:
            return BenchmarkRank.BELOW_BASELINE

        rank_order = [
            BenchmarkRank.SOTA,
            BenchmarkRank.COMPETITIVE,
            BenchmarkRank.BASELINE,
            BenchmarkRank.BELOW_BASELINE
        ]

        for rank in rank_order:
            if any(r.rank == rank for r in self.results):
                return rank

        return BenchmarkRank.BELOW_BASELINE


# =============================================================================
# Combined Analysis Monitor
# =============================================================================

class A2AInteractionMonitor:
    """Placeholder for A2A monitoring (not used in non-agent systems)."""
    pass


class MCPEnforcementMonitor:
    """Placeholder for mCP monitoring (not used in non-agent systems)."""
    pass


class AgentBehaviorAnalyzer:
    """
    Combined analyzer for non-agent analysis phases.

    Covers Phases 8-11: Explainability, Robustness, Statistics, Benchmarking
    """

    def __init__(self):
        self.explainability = ExplainabilityAnalyzer()
        self.robustness = RobustnessAnalyzer()
        self.statistics = StatisticalValidator()
        self.benchmarks = BenchmarkAnalyzer()

    def run_full_analysis(
        self,
        explanations: Optional[List[Dict[str, Any]]] = None,
        robustness_tests: Optional[List[Dict[str, Any]]] = None,
        statistical_data: Optional[Dict[str, np.ndarray]] = None,
        benchmark_scores: Optional[List[Dict[str, Any]]] = None
    ) -> AnalysisReport:
        """Run full analysis across all phases."""
        issues = []
        recommendations = []
        detailed = {}

        # Phase 8: Explainability
        if explanations:
            for exp in explanations:
                self.explainability.analyze_explanation(
                    explanation_id=exp.get("id", "unknown"),
                    explanation_text=exp.get("text", ""),
                    prediction=exp.get("prediction"),
                    evidence_chunks=exp.get("evidence", []),
                    confidence=exp.get("confidence")
                )

            exp_passed, exp_score = self.explainability.check_threshold()
            detailed["explainability"] = {
                "score": exp_score,
                "passed": exp_passed,
                "count": len(self.explainability.explanations)
            }

            if not exp_passed:
                issues.append(f"Explainability score {exp_score:.2f} below threshold")
                recommendations.append("Improve explanation completeness and faithfulness")
        else:
            exp_score = 0.0
            detailed["explainability"] = {"score": 0, "passed": True, "count": 0}

        # Phase 9: Robustness
        if robustness_tests:
            for test in robustness_tests:
                self.robustness.test_perturbation(
                    test_id=test.get("id", "unknown"),
                    perturbation_type=test.get("type", "unknown"),
                    perturbation_magnitude=test.get("magnitude", 0.0),
                    original_output=test.get("original"),
                    perturbed_output=test.get("perturbed")
                )

            rob_score = self.robustness.get_robustness_score()
            rob_level = self.robustness.get_robustness_level()
            detailed["robustness"] = {
                "score": rob_score,
                "level": rob_level.value,
                "passed": rob_level in [RobustnessLevel.ROBUST, RobustnessLevel.MODERATE],
                "failures": self.robustness.get_failure_analysis()
            }

            if rob_level == RobustnessLevel.FRAGILE:
                issues.append("System is fragile to perturbations")
                recommendations.append("Add robustness training or input validation")
        else:
            rob_score = 0.0
            detailed["robustness"] = {"score": 0, "passed": True, "level": "untested"}

        # Phase 10: Statistics
        stat_valid = True
        if statistical_data:
            groups = list(statistical_data.keys())
            if len(groups) >= 2:
                test = self.statistics.t_test(
                    statistical_data[groups[0]],
                    statistical_data[groups[1]],
                    f"{groups[0]}_vs_{groups[1]}"
                )
                stat_valid, msg = self.statistics.validate_claim(test)
                detailed["statistics"] = {
                    "test": test.test_name,
                    "p_value": test.p_value,
                    "effect_size": test.effect_size,
                    "significance": test.significance.value,
                    "valid": stat_valid
                }

                if not stat_valid:
                    issues.append(f"Statistical claim not supported: {msg}")
        else:
            detailed["statistics"] = {"valid": True, "note": "No data provided"}

        # Phase 11: Benchmarks
        best_rank = BenchmarkRank.BELOW_BASELINE
        if benchmark_scores:
            for score in benchmark_scores:
                self.benchmarks.register_baseline(
                    score.get("dataset", "default"),
                    score.get("metric", "accuracy"),
                    score.get("baseline", 0.0)
                )
                self.benchmarks.register_sota(
                    score.get("dataset", "default"),
                    score.get("metric", "accuracy"),
                    score.get("sota", 1.0)
                )
                self.benchmarks.evaluate(
                    score.get("model", "model"),
                    score.get("dataset", "default"),
                    score.get("metric", "accuracy"),
                    score.get("score", 0.0)
                )

            best_rank = self.benchmarks.get_best_rank()
            detailed["benchmarks"] = {
                "best_rank": best_rank.value,
                "comparison": self.benchmarks.get_comparison_table()
            }

            if best_rank == BenchmarkRank.BELOW_BASELINE:
                issues.append("Performance below baseline")
                recommendations.append("Review model architecture or training")
        else:
            detailed["benchmarks"] = {"best_rank": "not_tested"}

        # Compile report
        overall_passed = (
            detailed.get("explainability", {}).get("passed", True) and
            detailed.get("robustness", {}).get("passed", True) and
            stat_valid and
            best_rank != BenchmarkRank.BELOW_BASELINE
        )

        return AnalysisReport(
            explainability_score=exp_score,
            robustness_score=rob_score,
            statistical_validity=stat_valid,
            benchmark_rank=best_rank,
            detailed_metrics=detailed,
            passed=overall_passed,
            issues=issues,
            recommendations=recommendations
        )


__all__ = [
    # Enums
    "ExplainabilityLevel",
    "RobustnessLevel",
    "StatisticalSignificance",
    "BenchmarkRank",
    # Data classes
    "ExplanationQuality",
    "RobustnessTest",
    "StatisticalTest",
    "BenchmarkResult",
    "AnalysisReport",
    # Analyzers
    "ExplainabilityAnalyzer",
    "RobustnessAnalyzer",
    "StatisticalValidator",
    "BenchmarkAnalyzer",
    "AgentBehaviorAnalyzer",
    # Placeholders
    "A2AInteractionMonitor",
    "MCPEnforcementMonitor",
]
