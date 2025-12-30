"""
Phase 15: Value, ROI & Executive Impact Monitoring

Monitors business value, return on investment, and executive-level metrics
for RAG systems.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================

class ValueCategory(Enum):
    """Categories of business value."""
    EFFICIENCY = "efficiency"           # Time/cost savings
    QUALITY = "quality"                 # Improved accuracy/outcomes
    REVENUE = "revenue"                 # Direct revenue impact
    RISK_REDUCTION = "risk_reduction"   # Reduced errors/compliance risk
    INNOVATION = "innovation"           # New capabilities enabled


class ROIStatus(Enum):
    """ROI assessment status."""
    POSITIVE = "positive"               # ROI > 0
    BREAK_EVEN = "break_even"           # ROI ≈ 0
    NEGATIVE = "negative"               # ROI < 0
    PROJECTED = "projected"             # Future projection


class ImpactLevel(Enum):
    """Business impact level."""
    TRANSFORMATIONAL = "transformational"   # Major business change
    SIGNIFICANT = "significant"             # Notable improvement
    MODERATE = "moderate"                   # Measurable benefit
    MINIMAL = "minimal"                     # Small benefit
    NONE = "none"                           # No measurable impact


class StakeholderType(Enum):
    """Types of stakeholders."""
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    OPERATIONAL = "operational"
    END_USER = "end_user"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CostMetric:
    """Cost-related metric."""
    category: str
    amount: float
    currency: str = "USD"
    period: str = "monthly"
    description: str = ""


@dataclass
class BenefitMetric:
    """Benefit-related metric."""
    category: ValueCategory
    amount: float
    currency: str = "USD"
    period: str = "monthly"
    quantifiable: bool = True
    description: str = ""


@dataclass
class ROICalculation:
    """ROI calculation result."""
    total_costs: float
    total_benefits: float
    net_value: float
    roi_percentage: float
    payback_months: float
    status: ROIStatus
    calculated_at: datetime = field(default_factory=datetime.now)


@dataclass
class UsageMetrics:
    """System usage metrics."""
    total_queries: int
    unique_users: int
    queries_per_user: float
    peak_usage_hour: int
    adoption_rate: float  # % of potential users
    retention_rate: float  # % returning users


@dataclass
class QualityImpact:
    """Quality improvement impact."""
    metric_name: str
    baseline_value: float
    current_value: float
    improvement_percentage: float
    statistical_significance: bool


@dataclass
class ExecutiveSummary:
    """Executive-level summary."""
    roi_status: ROIStatus
    roi_percentage: float
    key_achievements: List[str]
    key_risks: List[str]
    usage_trend: str  # "increasing", "stable", "decreasing"
    quality_trend: str
    recommendations: List[str]
    confidence_level: str  # "high", "medium", "low"


@dataclass
class ROIReport:
    """Comprehensive ROI report."""
    roi_calculation: ROICalculation
    usage_metrics: UsageMetrics
    quality_impacts: List[QualityImpact]
    executive_summary: ExecutiveSummary
    cost_breakdown: Dict[str, float]
    benefit_breakdown: Dict[str, float]
    passed: bool
    issues: List[str]


# =============================================================================
# Phase 15.1: Cost Tracker
# =============================================================================

class CostTracker:
    """
    Tracks all system costs.

    Purpose: Monitor and categorize system costs
    Measurement: Total cost, cost per query, cost trends
    Pass/Fail: Costs within budget, trending stable/down
    """

    def __init__(self, budget_monthly: float = 10000.0):
        self.budget_monthly = budget_monthly
        self.costs: List[CostMetric] = []

    def record_cost(
        self,
        category: str,
        amount: float,
        currency: str = "USD",
        period: str = "monthly",
        description: str = ""
    ) -> CostMetric:
        """Record a cost."""
        metric = CostMetric(
            category=category,
            amount=amount,
            currency=currency,
            period=period,
            description=description
        )
        self.costs.append(metric)
        return metric

    def get_total_cost(self, period: str = "monthly") -> float:
        """Get total cost for period."""
        return sum(c.amount for c in self.costs if c.period == period)

    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown by category."""
        breakdown = {}
        for cost in self.costs:
            if cost.category not in breakdown:
                breakdown[cost.category] = 0
            breakdown[cost.category] += cost.amount
        return breakdown

    def check_budget(self) -> Tuple[bool, float]:
        """Check if within budget."""
        total = self.get_total_cost()
        utilization = total / self.budget_monthly if self.budget_monthly > 0 else 0
        return total <= self.budget_monthly, utilization

    def calculate_cost_per_query(self, query_count: int) -> float:
        """Calculate cost per query."""
        if query_count == 0:
            return 0.0
        return self.get_total_cost() / query_count


# =============================================================================
# Phase 15.2: Benefit Tracker
# =============================================================================

class BenefitTracker:
    """
    Tracks all system benefits.

    Purpose: Quantify and categorize benefits
    Measurement: Total benefit value, benefit by category
    Pass/Fail: Benefits exceed costs
    """

    def __init__(self):
        self.benefits: List[BenefitMetric] = []

    def record_benefit(
        self,
        category: ValueCategory,
        amount: float,
        currency: str = "USD",
        period: str = "monthly",
        quantifiable: bool = True,
        description: str = ""
    ) -> BenefitMetric:
        """Record a benefit."""
        metric = BenefitMetric(
            category=category,
            amount=amount,
            currency=currency,
            period=period,
            quantifiable=quantifiable,
            description=description
        )
        self.benefits.append(metric)
        return metric

    def get_total_benefit(self, period: str = "monthly") -> float:
        """Get total quantifiable benefit for period."""
        return sum(
            b.amount for b in self.benefits
            if b.period == period and b.quantifiable
        )

    def get_benefit_breakdown(self) -> Dict[str, float]:
        """Get benefit breakdown by category."""
        breakdown = {}
        for benefit in self.benefits:
            cat = benefit.category.value
            if cat not in breakdown:
                breakdown[cat] = 0
            breakdown[cat] += benefit.amount
        return breakdown


# =============================================================================
# Phase 15.3: ROI Calculator
# =============================================================================

class ROICalculator:
    """
    Calculates return on investment.

    Purpose: Calculate and track ROI metrics
    Measurement: ROI percentage, payback period
    Pass/Fail: ROI > 0%, payback < 12 months
    """

    def __init__(self):
        self.calculations: List[ROICalculation] = []

    def calculate_roi(
        self,
        costs: float,
        benefits: float,
        investment_period_months: int = 12
    ) -> ROICalculation:
        """Calculate ROI."""
        net_value = benefits - costs
        roi_percentage = (net_value / costs * 100) if costs > 0 else 0

        # Calculate payback period
        monthly_benefit = benefits / investment_period_months if investment_period_months > 0 else 0
        payback_months = costs / monthly_benefit if monthly_benefit > 0 else float('inf')

        # Determine status
        if roi_percentage > 10:
            status = ROIStatus.POSITIVE
        elif roi_percentage > -10:
            status = ROIStatus.BREAK_EVEN
        else:
            status = ROIStatus.NEGATIVE

        calc = ROICalculation(
            total_costs=costs,
            total_benefits=benefits,
            net_value=net_value,
            roi_percentage=roi_percentage,
            payback_months=payback_months,
            status=status
        )

        self.calculations.append(calc)
        return calc

    def get_latest_roi(self) -> Optional[ROICalculation]:
        """Get most recent ROI calculation."""
        return self.calculations[-1] if self.calculations else None

    def get_roi_trend(self) -> str:
        """Get ROI trend over time."""
        if len(self.calculations) < 2:
            return "insufficient_data"

        recent = self.calculations[-3:]
        if len(recent) < 2:
            return "insufficient_data"

        if recent[-1].roi_percentage > recent[0].roi_percentage:
            return "improving"
        elif recent[-1].roi_percentage < recent[0].roi_percentage:
            return "declining"
        else:
            return "stable"


# =============================================================================
# Phase 15.4: Usage Analyzer
# =============================================================================

class UsageAnalyzer:
    """
    Analyzes system usage patterns.

    Purpose: Track adoption and usage metrics
    Measurement: User count, query count, adoption rate
    Pass/Fail: Adoption > 50%, retention > 70%
    """

    def __init__(self, potential_users: int = 100):
        self.potential_users = potential_users
        self.usage_records: List[Dict[str, Any]] = []
        self.user_sessions: Dict[str, List[datetime]] = {}

    def record_usage(
        self,
        user_id: str,
        query_count: int = 1,
        timestamp: Optional[datetime] = None
    ):
        """Record a usage event."""
        ts = timestamp or datetime.now()

        self.usage_records.append({
            "user_id": user_id,
            "query_count": query_count,
            "timestamp": ts
        })

        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(ts)

    def get_usage_metrics(
        self,
        period_days: int = 30
    ) -> UsageMetrics:
        """Get usage metrics for period."""
        cutoff = datetime.now() - timedelta(days=period_days)

        # Filter recent records
        recent = [r for r in self.usage_records if r["timestamp"] >= cutoff]

        total_queries = sum(r["query_count"] for r in recent)
        unique_users = len(set(r["user_id"] for r in recent))

        queries_per_user = total_queries / unique_users if unique_users > 0 else 0

        # Peak usage hour
        hours = [r["timestamp"].hour for r in recent]
        peak_hour = max(set(hours), key=hours.count) if hours else 12

        # Adoption rate
        adoption_rate = unique_users / self.potential_users if self.potential_users > 0 else 0

        # Retention rate (users who came back)
        returning_users = sum(
            1 for sessions in self.user_sessions.values()
            if len([s for s in sessions if s >= cutoff]) > 1
        )
        retention_rate = returning_users / unique_users if unique_users > 0 else 0

        return UsageMetrics(
            total_queries=total_queries,
            unique_users=unique_users,
            queries_per_user=queries_per_user,
            peak_usage_hour=peak_hour,
            adoption_rate=adoption_rate,
            retention_rate=retention_rate
        )


# =============================================================================
# Phase 15.5: Quality Impact Analyzer
# =============================================================================

class QualityImpactAnalyzer:
    """
    Analyzes quality improvements.

    Purpose: Measure quality impact vs baseline
    Measurement: Improvement percentages, statistical significance
    Pass/Fail: Statistically significant improvement
    """

    def __init__(self):
        self.baselines: Dict[str, float] = {}
        self.current_values: Dict[str, List[float]] = {}
        self.impacts: List[QualityImpact] = []

    def set_baseline(self, metric_name: str, value: float):
        """Set baseline value for a metric."""
        self.baselines[metric_name] = value

    def record_value(self, metric_name: str, value: float):
        """Record a current metric value."""
        if metric_name not in self.current_values:
            self.current_values[metric_name] = []
        self.current_values[metric_name].append(value)

    def calculate_impact(self, metric_name: str) -> Optional[QualityImpact]:
        """Calculate quality impact for a metric."""
        if metric_name not in self.baselines:
            return None

        if metric_name not in self.current_values:
            return None

        values = self.current_values[metric_name]
        if not values:
            return None

        baseline = self.baselines[metric_name]
        current = np.mean(values)

        improvement = ((current - baseline) / baseline * 100) if baseline != 0 else 0

        # Simple significance test (would use proper stats in production)
        std = np.std(values) if len(values) > 1 else 0
        significant = abs(current - baseline) > 2 * std if std > 0 else abs(improvement) > 5

        impact = QualityImpact(
            metric_name=metric_name,
            baseline_value=baseline,
            current_value=current,
            improvement_percentage=improvement,
            statistical_significance=significant
        )

        self.impacts.append(impact)
        return impact

    def get_all_impacts(self) -> List[QualityImpact]:
        """Calculate impacts for all metrics."""
        impacts = []
        for metric_name in self.baselines:
            impact = self.calculate_impact(metric_name)
            if impact:
                impacts.append(impact)
        return impacts


# =============================================================================
# Phase 15: Unified ROI Analyzer
# =============================================================================

class ROIAnalyzer:
    """
    Unified analyzer for Phase 15: Value, ROI & Executive Impact.

    Combines cost tracking, benefit tracking, ROI calculation,
    usage analysis, and quality impact assessment.
    """

    def __init__(self, budget_monthly: float = 10000.0, potential_users: int = 100):
        self.cost_tracker = CostTracker(budget_monthly)
        self.benefit_tracker = BenefitTracker()
        self.roi_calculator = ROICalculator()
        self.usage_analyzer = UsageAnalyzer(potential_users)
        self.quality_analyzer = QualityImpactAnalyzer()

    def generate_report(self) -> ROIReport:
        """Generate comprehensive ROI report."""
        # Get costs and benefits
        total_costs = self.cost_tracker.get_total_cost()
        total_benefits = self.benefit_tracker.get_total_benefit()

        # Calculate ROI
        roi_calc = self.roi_calculator.calculate_roi(total_costs, total_benefits)

        # Get usage metrics
        usage = self.usage_analyzer.get_usage_metrics()

        # Get quality impacts
        quality_impacts = self.quality_analyzer.get_all_impacts()

        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            roi_calc, usage, quality_impacts
        )

        # Get breakdowns
        cost_breakdown = self.cost_tracker.get_cost_breakdown()
        benefit_breakdown = self.benefit_tracker.get_benefit_breakdown()

        # Determine pass status
        passed = (
            roi_calc.status == ROIStatus.POSITIVE and
            usage.adoption_rate >= 0.5 and
            usage.retention_rate >= 0.7
        )

        # Identify issues
        issues = []
        if roi_calc.status == ROIStatus.NEGATIVE:
            issues.append(f"Negative ROI: {roi_calc.roi_percentage:.1f}%")
        if usage.adoption_rate < 0.5:
            issues.append(f"Low adoption: {usage.adoption_rate:.1%}")
        if usage.retention_rate < 0.7:
            issues.append(f"Low retention: {usage.retention_rate:.1%}")

        in_budget, budget_util = self.cost_tracker.check_budget()
        if not in_budget:
            issues.append(f"Over budget: {budget_util:.1%} utilization")

        return ROIReport(
            roi_calculation=roi_calc,
            usage_metrics=usage,
            quality_impacts=quality_impacts,
            executive_summary=executive_summary,
            cost_breakdown=cost_breakdown,
            benefit_breakdown=benefit_breakdown,
            passed=passed,
            issues=issues
        )

    def _generate_executive_summary(
        self,
        roi: ROICalculation,
        usage: UsageMetrics,
        quality: List[QualityImpact]
    ) -> ExecutiveSummary:
        """Generate executive-level summary."""
        # Determine usage trend
        usage_trend = "stable"
        if usage.adoption_rate > 0.7:
            usage_trend = "increasing"
        elif usage.adoption_rate < 0.3:
            usage_trend = "decreasing"

        # Determine quality trend
        significant_improvements = [q for q in quality if q.statistical_significance and q.improvement_percentage > 0]
        if len(significant_improvements) >= len(quality) / 2:
            quality_trend = "improving"
        elif any(q.improvement_percentage < -10 for q in quality):
            quality_trend = "declining"
        else:
            quality_trend = "stable"

        # Key achievements
        achievements = []
        if roi.status == ROIStatus.POSITIVE:
            achievements.append(f"Positive ROI of {roi.roi_percentage:.1f}%")
        if usage.adoption_rate >= 0.5:
            achievements.append(f"{usage.adoption_rate:.0%} user adoption")
        for q in significant_improvements[:3]:
            achievements.append(f"{q.improvement_percentage:.1f}% improvement in {q.metric_name}")

        # Key risks
        risks = []
        if roi.status == ROIStatus.NEGATIVE:
            risks.append("Negative return on investment")
        if usage.retention_rate < 0.5:
            risks.append("Low user retention")
        if roi.payback_months > 24:
            risks.append(f"Long payback period ({roi.payback_months:.0f} months)")

        # Recommendations
        recommendations = []
        if usage.adoption_rate < 0.5:
            recommendations.append("Focus on user onboarding and training")
        if usage.retention_rate < 0.7:
            recommendations.append("Investigate user churn causes")
        if roi.status != ROIStatus.POSITIVE:
            recommendations.append("Review cost structure and value drivers")

        # Confidence level
        if usage.total_queries > 1000 and len(quality) >= 3:
            confidence = "high"
        elif usage.total_queries > 100:
            confidence = "medium"
        else:
            confidence = "low"

        return ExecutiveSummary(
            roi_status=roi.status,
            roi_percentage=roi.roi_percentage,
            key_achievements=achievements,
            key_risks=risks,
            usage_trend=usage_trend,
            quality_trend=quality_trend,
            recommendations=recommendations,
            confidence_level=confidence
        )


__all__ = [
    # Enums
    "ValueCategory",
    "ROIStatus",
    "ImpactLevel",
    "StakeholderType",
    # Data classes
    "CostMetric",
    "BenefitMetric",
    "ROICalculation",
    "UsageMetrics",
    "QualityImpact",
    "ExecutiveSummary",
    "ROIReport",
    # Analyzers
    "CostTracker",
    "BenefitTracker",
    "ROICalculator",
    "UsageAnalyzer",
    "QualityImpactAnalyzer",
    "ROIAnalyzer",
]
