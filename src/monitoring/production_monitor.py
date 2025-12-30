"""
Phase 12 & 14: Scalability, Deployment & Production Monitoring

Monitors system scalability, deployment health, and production drift
for RAG systems.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================

class ScalabilityLevel(Enum):
    """Scalability assessment."""
    EXCELLENT = "excellent"     # Linear or better scaling
    GOOD = "good"              # Sub-linear but acceptable
    MODERATE = "moderate"       # Some bottlenecks
    POOR = "poor"              # Significant scaling issues


class DeploymentStatus(Enum):
    """Deployment health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class DriftType(Enum):
    """Types of production drift."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    MODEL_DRIFT = "model_drift"
    PERFORMANCE_DRIFT = "performance_drift"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LatencyMetrics:
    """Latency performance metrics."""
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    sample_count: int


@dataclass
class ThroughputMetrics:
    """Throughput performance metrics."""
    requests_per_second: float
    tokens_per_second: float
    successful_requests: int
    failed_requests: int
    success_rate: float


@dataclass
class ScalabilityTest:
    """Result of scalability test."""
    test_id: str
    load_level: int  # e.g., concurrent users
    latency: LatencyMetrics
    throughput: ThroughputMetrics
    resource_usage: Dict[str, float]
    passed: bool


@dataclass
class DriftDetection:
    """Result of drift detection."""
    drift_id: str
    drift_type: DriftType
    metric_name: str
    baseline_value: float
    current_value: float
    drift_magnitude: float
    detected_at: datetime
    is_significant: bool


@dataclass
class ProductionAlert:
    """Production system alert."""
    alert_id: str
    severity: AlertSeverity
    category: str
    message: str
    metric_value: Optional[float]
    threshold: Optional[float]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ProductionHealthReport:
    """Comprehensive production health report."""
    status: DeploymentStatus
    uptime_percentage: float
    latency_metrics: LatencyMetrics
    throughput_metrics: ThroughputMetrics
    drift_detected: List[DriftDetection]
    active_alerts: List[ProductionAlert]
    scalability_level: ScalabilityLevel
    passed: bool
    issues: List[str]
    recommendations: List[str]


# =============================================================================
# Phase 12: Scalability Monitor
# =============================================================================

class ScalabilityMonitor:
    """
    Monitors system scalability and performance.

    Purpose: Ensure system scales appropriately under load
    Measurement: Latency percentiles, throughput, resource usage
    Pass/Fail: P99 latency < SLA, linear scaling maintained
    """

    def __init__(
        self,
        p99_threshold_ms: float = 500.0,
        success_rate_threshold: float = 0.99
    ):
        self.p99_threshold_ms = p99_threshold_ms
        self.success_rate_threshold = success_rate_threshold
        self.tests: List[ScalabilityTest] = []
        self.latencies: List[float] = []

    def record_request(
        self,
        latency_ms: float,
        success: bool = True,
        tokens: int = 0
    ):
        """Record a single request."""
        self.latencies.append(latency_ms)

    def run_load_test(
        self,
        test_id: str,
        load_level: int,
        latencies: List[float],
        successful: int,
        failed: int,
        tokens_processed: int,
        duration_seconds: float,
        resource_usage: Optional[Dict[str, float]] = None
    ) -> ScalabilityTest:
        """Run and record a load test."""
        # Calculate latency metrics
        latency_array = np.array(latencies)
        latency_metrics = LatencyMetrics(
            p50_ms=float(np.percentile(latency_array, 50)),
            p90_ms=float(np.percentile(latency_array, 90)),
            p95_ms=float(np.percentile(latency_array, 95)),
            p99_ms=float(np.percentile(latency_array, 99)),
            mean_ms=float(np.mean(latency_array)),
            std_ms=float(np.std(latency_array)),
            sample_count=len(latencies)
        )

        # Calculate throughput
        total_requests = successful + failed
        throughput = ThroughputMetrics(
            requests_per_second=total_requests / duration_seconds if duration_seconds > 0 else 0,
            tokens_per_second=tokens_processed / duration_seconds if duration_seconds > 0 else 0,
            successful_requests=successful,
            failed_requests=failed,
            success_rate=successful / total_requests if total_requests > 0 else 0
        )

        # Check pass criteria
        passed = (
            latency_metrics.p99_ms <= self.p99_threshold_ms and
            throughput.success_rate >= self.success_rate_threshold
        )

        test = ScalabilityTest(
            test_id=test_id,
            load_level=load_level,
            latency=latency_metrics,
            throughput=throughput,
            resource_usage=resource_usage or {},
            passed=passed
        )

        self.tests.append(test)
        return test

    def get_current_latency_metrics(self) -> LatencyMetrics:
        """Get current latency metrics from recorded requests."""
        if not self.latencies:
            return LatencyMetrics(0, 0, 0, 0, 0, 0, 0)

        arr = np.array(self.latencies)
        return LatencyMetrics(
            p50_ms=float(np.percentile(arr, 50)),
            p90_ms=float(np.percentile(arr, 90)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr)),
            sample_count=len(self.latencies)
        )

    def assess_scalability(self) -> ScalabilityLevel:
        """Assess overall scalability from tests."""
        if len(self.tests) < 2:
            return ScalabilityLevel.MODERATE

        # Sort by load level
        sorted_tests = sorted(self.tests, key=lambda t: t.load_level)

        # Check if latency scales linearly or better
        load_levels = [t.load_level for t in sorted_tests]
        latencies = [t.latency.p99_ms for t in sorted_tests]

        if len(load_levels) >= 2:
            # Calculate scaling factor
            load_ratio = load_levels[-1] / load_levels[0]
            latency_ratio = latencies[-1] / latencies[0] if latencies[0] > 0 else float('inf')

            if latency_ratio <= load_ratio * 0.5:
                return ScalabilityLevel.EXCELLENT
            elif latency_ratio <= load_ratio:
                return ScalabilityLevel.GOOD
            elif latency_ratio <= load_ratio * 2:
                return ScalabilityLevel.MODERATE
            else:
                return ScalabilityLevel.POOR

        return ScalabilityLevel.MODERATE


# =============================================================================
# Phase 14: Production Drift Monitor
# =============================================================================

class ProductionDriftMonitor:
    """
    Monitors production drift and system health.

    Purpose: Detect data/model/performance drift in production
    Measurement: Statistical drift tests, performance degradation
    Pass/Fail: No significant drift detected, SLAs maintained
    """

    def __init__(
        self,
        drift_threshold: float = 0.1,
        window_size: int = 1000
    ):
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.baseline_metrics: Dict[str, float] = {}
        self.current_metrics: Dict[str, List[float]] = {}
        self.drift_detections: List[DriftDetection] = []
        self.alerts: List[ProductionAlert] = []

    def set_baseline(self, metric_name: str, value: float):
        """Set baseline value for a metric."""
        self.baseline_metrics[metric_name] = value

    def record_metric(self, metric_name: str, value: float):
        """Record a metric observation."""
        if metric_name not in self.current_metrics:
            self.current_metrics[metric_name] = []

        self.current_metrics[metric_name].append(value)

        # Keep only recent values
        if len(self.current_metrics[metric_name]) > self.window_size:
            self.current_metrics[metric_name] = \
                self.current_metrics[metric_name][-self.window_size:]

    def check_drift(self, metric_name: str) -> Optional[DriftDetection]:
        """Check for drift in a specific metric."""
        if metric_name not in self.baseline_metrics:
            return None

        if metric_name not in self.current_metrics:
            return None

        values = self.current_metrics[metric_name]
        if len(values) < 10:
            return None

        baseline = self.baseline_metrics[metric_name]
        current = np.mean(values[-100:]) if len(values) >= 100 else np.mean(values)

        # Calculate drift magnitude
        drift_magnitude = abs(current - baseline) / abs(baseline) if baseline != 0 else abs(current)

        is_significant = drift_magnitude > self.drift_threshold

        if is_significant:
            detection = DriftDetection(
                drift_id=f"drift_{metric_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                drift_type=DriftType.PERFORMANCE_DRIFT,
                metric_name=metric_name,
                baseline_value=baseline,
                current_value=current,
                drift_magnitude=drift_magnitude,
                detected_at=datetime.now(),
                is_significant=True
            )
            self.drift_detections.append(detection)
            return detection

        return None

    def check_all_drift(self) -> List[DriftDetection]:
        """Check drift for all metrics."""
        detections = []
        for metric_name in self.baseline_metrics:
            detection = self.check_drift(metric_name)
            if detection:
                detections.append(detection)
        return detections

    def create_alert(
        self,
        severity: AlertSeverity,
        category: str,
        message: str,
        metric_value: Optional[float] = None,
        threshold: Optional[float] = None
    ) -> ProductionAlert:
        """Create a production alert."""
        alert = ProductionAlert(
            alert_id=f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            severity=severity,
            category=category,
            message=message,
            metric_value=metric_value,
            threshold=threshold
        )
        self.alerts.append(alert)
        return alert

    def get_active_alerts(
        self,
        max_age_hours: int = 24
    ) -> List[ProductionAlert]:
        """Get active alerts within time window."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        return [a for a in self.alerts if a.timestamp >= cutoff]

    def get_deployment_status(self) -> DeploymentStatus:
        """Determine current deployment status."""
        active = self.get_active_alerts()

        critical = sum(1 for a in active if a.severity == AlertSeverity.CRITICAL)
        errors = sum(1 for a in active if a.severity == AlertSeverity.ERROR)

        if critical > 0:
            return DeploymentStatus.CRITICAL
        elif errors > 2 or len(self.drift_detections) > 3:
            return DeploymentStatus.DEGRADED
        else:
            return DeploymentStatus.HEALTHY


# =============================================================================
# Combined Production Health Monitor
# =============================================================================

class ProductionHealthMonitor:
    """
    Combined monitor for production health.

    Covers Phase 12 (Scalability) and Phase 14 (Production Monitoring).
    """

    def __init__(self):
        self.scalability = ScalabilityMonitor()
        self.drift = ProductionDriftMonitor()
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0

    def record_request(
        self,
        latency_ms: float,
        success: bool = True,
        tokens: int = 0,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Record a production request."""
        self.request_count += 1
        if not success:
            self.error_count += 1

        self.scalability.record_request(latency_ms, success, tokens)

        # Record metrics for drift detection
        if metrics:
            for name, value in metrics.items():
                self.drift.record_metric(name, value)

        self.drift.record_metric("latency_ms", latency_ms)
        self.drift.record_metric("success", 1.0 if success else 0.0)

    def get_uptime_percentage(self) -> float:
        """Calculate uptime percentage."""
        if self.request_count == 0:
            return 100.0
        return (self.request_count - self.error_count) / self.request_count * 100

    def run_health_check(self) -> ProductionHealthReport:
        """Run comprehensive health check."""
        # Get metrics
        latency = self.scalability.get_current_latency_metrics()

        # Estimate throughput
        runtime = (datetime.now() - self.start_time).total_seconds()
        throughput = ThroughputMetrics(
            requests_per_second=self.request_count / runtime if runtime > 0 else 0,
            tokens_per_second=0,  # Would need to track
            successful_requests=self.request_count - self.error_count,
            failed_requests=self.error_count,
            success_rate=self.get_uptime_percentage() / 100
        )

        # Check drift
        drift_detected = self.drift.check_all_drift()

        # Get alerts
        active_alerts = self.drift.get_active_alerts()

        # Assess scalability
        scalability_level = self.scalability.assess_scalability()

        # Determine status
        status = self.drift.get_deployment_status()

        # Check pass criteria
        passed = (
            status == DeploymentStatus.HEALTHY and
            len(drift_detected) == 0 and
            throughput.success_rate >= 0.99
        )

        # Identify issues
        issues = []
        if status != DeploymentStatus.HEALTHY:
            issues.append(f"Deployment status: {status.value}")
        if drift_detected:
            issues.append(f"Drift detected in {len(drift_detected)} metrics")
        if throughput.success_rate < 0.99:
            issues.append(f"Success rate {throughput.success_rate:.2%} below 99%")

        # Recommendations
        recommendations = []
        if scalability_level in [ScalabilityLevel.MODERATE, ScalabilityLevel.POOR]:
            recommendations.append("Review system capacity and scaling policies")
        if drift_detected:
            recommendations.append("Investigate drift causes and consider retraining")
        if active_alerts:
            recommendations.append("Address active alerts")

        return ProductionHealthReport(
            status=status,
            uptime_percentage=self.get_uptime_percentage(),
            latency_metrics=latency,
            throughput_metrics=throughput,
            drift_detected=drift_detected,
            active_alerts=active_alerts,
            scalability_level=scalability_level,
            passed=passed,
            issues=issues,
            recommendations=recommendations
        )


__all__ = [
    # Enums
    "ScalabilityLevel",
    "DeploymentStatus",
    "DriftType",
    "AlertSeverity",
    # Data classes
    "LatencyMetrics",
    "ThroughputMetrics",
    "ScalabilityTest",
    "DriftDetection",
    "ProductionAlert",
    "ProductionHealthReport",
    # Monitors
    "ScalabilityMonitor",
    "ProductionDriftMonitor",
    "ProductionHealthMonitor",
]
