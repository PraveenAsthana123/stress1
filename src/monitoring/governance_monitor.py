"""
Phase 13: Governance, Security & Compliance Monitoring

Monitors governance policies, security posture, and regulatory compliance
for RAG systems.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
import hashlib
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================

class ComplianceStatus(Enum):
    """Compliance status."""
    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    UNKNOWN = "unknown"


class SecurityLevel(Enum):
    """Security posture level."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CRITICAL = "critical"


class AuditEventType(Enum):
    """Types of audit events."""
    ACCESS = "access"
    MODIFICATION = "modification"
    DELETION = "deletion"
    EXPORT = "export"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    POLICY_CHANGE = "policy_change"


class RegulatoryFramework(Enum):
    """Regulatory frameworks."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    FERPA = "ferpa"
    CCPA = "ccpa"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AuditEvent:
    """A single audit event."""
    event_id: str
    event_type: AuditEventType
    actor: str
    resource: str
    action: str
    timestamp: datetime
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyViolation:
    """A policy violation record."""
    violation_id: str
    policy_name: str
    severity: str
    description: str
    resource: str
    detected_at: datetime
    resolved: bool = False
    resolution: Optional[str] = None


@dataclass
class ComplianceCheck:
    """Result of a compliance check."""
    framework: RegulatoryFramework
    check_name: str
    status: ComplianceStatus
    findings: List[str]
    evidence: List[str]
    checked_at: datetime


@dataclass
class SecurityAssessment:
    """Result of security assessment."""
    assessment_id: str
    level: SecurityLevel
    vulnerabilities: List[Dict[str, Any]]
    controls_passed: int
    controls_failed: int
    risk_score: float  # 0-10
    assessed_at: datetime


@dataclass
class GovernanceReport:
    """Comprehensive governance report."""
    compliance_status: Dict[str, ComplianceStatus]
    security_level: SecurityLevel
    policy_violations: List[PolicyViolation]
    audit_summary: Dict[str, int]
    risk_score: float
    passed: bool
    issues: List[str]
    recommendations: List[str]


# =============================================================================
# Phase 13.1: Audit Logger
# =============================================================================

class AuditLogger:
    """
    Logs and tracks audit events.

    Purpose: Maintain comprehensive audit trail
    Measurement: Event coverage, completeness
    Pass/Fail: All critical events logged, no gaps
    """

    def __init__(self, retention_days: int = 365):
        self.retention_days = retention_days
        self.events: List[AuditEvent] = []

    def log_event(
        self,
        event_type: AuditEventType,
        actor: str,
        resource: str,
        action: str,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log an audit event."""
        event = AuditEvent(
            event_id=hashlib.md5(
                f"{datetime.now().isoformat()}{actor}{action}".encode()
            ).hexdigest()[:12],
            event_type=event_type,
            actor=actor,
            resource=resource,
            action=action,
            timestamp=datetime.now(),
            success=success,
            details=details or {}
        )

        self.events.append(event)
        logger.info(f"Audit: {event_type.value} by {actor} on {resource}")

        return event

    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        actor: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AuditEvent]:
        """Query audit events with filters."""
        results = self.events

        if event_type:
            results = [e for e in results if e.event_type == event_type]

        if actor:
            results = [e for e in results if e.actor == actor]

        if start_time:
            results = [e for e in results if e.timestamp >= start_time]

        if end_time:
            results = [e for e in results if e.timestamp <= end_time]

        return results

    def get_summary(
        self,
        period_hours: int = 24
    ) -> Dict[str, int]:
        """Get summary of recent events."""
        cutoff = datetime.now() - timedelta(hours=period_hours)
        recent = [e for e in self.events if e.timestamp >= cutoff]

        summary = {"total": len(recent), "success": 0, "failure": 0}
        for event_type in AuditEventType:
            summary[event_type.value] = sum(
                1 for e in recent if e.event_type == event_type
            )
        summary["success"] = sum(1 for e in recent if e.success)
        summary["failure"] = sum(1 for e in recent if not e.success)

        return summary

    def purge_old_events(self):
        """Purge events older than retention period."""
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        old_count = len(self.events)
        self.events = [e for e in self.events if e.timestamp >= cutoff]
        purged = old_count - len(self.events)
        logger.info(f"Purged {purged} old audit events")
        return purged


# =============================================================================
# Phase 13.2: Policy Enforcer
# =============================================================================

class PolicyEnforcer:
    """
    Enforces governance policies.

    Purpose: Ensure all operations comply with policies
    Measurement: Violation rate, enforcement coverage
    Pass/Fail: <1% policy violations
    """

    def __init__(self):
        self.policies: Dict[str, Dict[str, Any]] = {}
        self.violations: List[PolicyViolation] = []

    def register_policy(
        self,
        policy_name: str,
        policy_type: str,
        rules: Dict[str, Any],
        severity: str = "medium"
    ):
        """Register a governance policy."""
        self.policies[policy_name] = {
            "type": policy_type,
            "rules": rules,
            "severity": severity,
            "enabled": True
        }

    def check_policy(
        self,
        policy_name: str,
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[PolicyViolation]]:
        """Check if an action complies with a policy."""
        if policy_name not in self.policies:
            return True, None

        policy = self.policies[policy_name]
        if not policy["enabled"]:
            return True, None

        # Check rules
        rules = policy["rules"]
        violations = []

        for rule_name, rule_value in rules.items():
            if rule_name in context:
                if not self._check_rule(rule_value, context[rule_name]):
                    violations.append(f"Rule '{rule_name}' violated")

        if violations:
            violation = PolicyViolation(
                violation_id=hashlib.md5(
                    f"{policy_name}{datetime.now().isoformat()}".encode()
                ).hexdigest()[:12],
                policy_name=policy_name,
                severity=policy["severity"],
                description="; ".join(violations),
                resource=context.get("resource", "unknown"),
                detected_at=datetime.now()
            )
            self.violations.append(violation)
            return False, violation

        return True, None

    def _check_rule(self, rule: Any, value: Any) -> bool:
        """Check a single rule."""
        if isinstance(rule, dict):
            if "max" in rule and value > rule["max"]:
                return False
            if "min" in rule and value < rule["min"]:
                return False
            if "allowed" in rule and value not in rule["allowed"]:
                return False
            if "blocked" in rule and value in rule["blocked"]:
                return False
        elif isinstance(rule, (list, set)):
            return value in rule
        else:
            return value == rule

        return True

    def resolve_violation(
        self,
        violation_id: str,
        resolution: str
    ) -> bool:
        """Mark a violation as resolved."""
        for v in self.violations:
            if v.violation_id == violation_id:
                v.resolved = True
                v.resolution = resolution
                return True
        return False

    def get_violation_rate(self, period_hours: int = 24) -> float:
        """Get violation rate for period."""
        cutoff = datetime.now() - timedelta(hours=period_hours)
        recent = [v for v in self.violations if v.detected_at >= cutoff]
        # Would need total operations count for accurate rate
        return len(recent)  # Return count for now

    def get_unresolved_violations(self) -> List[PolicyViolation]:
        """Get list of unresolved violations."""
        return [v for v in self.violations if not v.resolved]


# =============================================================================
# Phase 13.3: Compliance Checker
# =============================================================================

class ComplianceChecker:
    """
    Checks compliance with regulatory frameworks.

    Purpose: Ensure regulatory compliance
    Measurement: Compliance check pass rate
    Pass/Fail: 100% compliance on critical controls
    """

    def __init__(self):
        self.checks: List[ComplianceCheck] = []
        self.control_definitions: Dict[RegulatoryFramework, Dict[str, Any]] = {}
        self._init_default_controls()

    def _init_default_controls(self):
        """Initialize default compliance controls."""
        self.control_definitions = {
            RegulatoryFramework.GDPR: {
                "data_minimization": "Collect only necessary data",
                "consent_tracking": "Track user consent for data processing",
                "right_to_erasure": "Support data deletion requests",
                "data_portability": "Support data export",
                "breach_notification": "Notify within 72 hours of breach",
            },
            RegulatoryFramework.HIPAA: {
                "access_control": "Implement access controls for PHI",
                "audit_logging": "Log all PHI access",
                "encryption": "Encrypt PHI at rest and in transit",
                "minimum_necessary": "Limit PHI access to minimum necessary",
                "baa_in_place": "Business Associate Agreements with vendors",
            },
            RegulatoryFramework.SOC2: {
                "security": "Security controls implemented",
                "availability": "System availability monitoring",
                "processing_integrity": "Data processing integrity checks",
                "confidentiality": "Data confidentiality controls",
                "privacy": "Privacy controls implemented",
            },
        }

    def run_compliance_check(
        self,
        framework: RegulatoryFramework,
        evidence: Dict[str, bool]
    ) -> List[ComplianceCheck]:
        """Run compliance check for a framework."""
        checks = []
        controls = self.control_definitions.get(framework, {})

        for control_name, description in controls.items():
            has_evidence = evidence.get(control_name, False)

            status = ComplianceStatus.COMPLIANT if has_evidence else ComplianceStatus.NON_COMPLIANT
            findings = [] if has_evidence else [f"Missing: {description}"]
            evidence_list = [control_name] if has_evidence else []

            check = ComplianceCheck(
                framework=framework,
                check_name=control_name,
                status=status,
                findings=findings,
                evidence=evidence_list,
                checked_at=datetime.now()
            )
            checks.append(check)
            self.checks.append(check)

        return checks

    def get_compliance_status(
        self,
        framework: RegulatoryFramework
    ) -> ComplianceStatus:
        """Get overall compliance status for a framework."""
        relevant = [c for c in self.checks if c.framework == framework]
        if not relevant:
            return ComplianceStatus.UNKNOWN

        non_compliant = sum(
            1 for c in relevant if c.status == ComplianceStatus.NON_COMPLIANT
        )

        if non_compliant == 0:
            return ComplianceStatus.COMPLIANT
        elif non_compliant < len(relevant) / 2:
            return ComplianceStatus.PARTIAL
        else:
            return ComplianceStatus.NON_COMPLIANT

    def get_compliance_summary(self) -> Dict[str, ComplianceStatus]:
        """Get compliance summary for all frameworks."""
        frameworks = set(c.framework for c in self.checks)
        return {
            f.value: self.get_compliance_status(f)
            for f in frameworks
        }


# =============================================================================
# Phase 13.4: Security Assessor
# =============================================================================

class SecurityAssessor:
    """
    Assesses security posture.

    Purpose: Evaluate security controls and vulnerabilities
    Measurement: Risk score, vulnerability count
    Pass/Fail: Risk score < 5, no critical vulnerabilities
    """

    def __init__(self):
        self.assessments: List[SecurityAssessment] = []
        self.security_controls = [
            "encryption_at_rest",
            "encryption_in_transit",
            "access_control",
            "authentication",
            "authorization",
            "audit_logging",
            "vulnerability_scanning",
            "incident_response",
            "backup_recovery",
            "network_security",
        ]

    def run_assessment(
        self,
        controls_status: Dict[str, bool],
        vulnerabilities: Optional[List[Dict[str, Any]]] = None
    ) -> SecurityAssessment:
        """Run security assessment."""
        vulnerabilities = vulnerabilities or []

        passed = sum(1 for c in self.security_controls if controls_status.get(c, False))
        failed = len(self.security_controls) - passed

        # Calculate risk score (0-10)
        control_risk = (failed / len(self.security_controls)) * 5

        vuln_risk = 0
        for v in vulnerabilities:
            severity = v.get("severity", "low")
            if severity == "critical":
                vuln_risk += 2.5
            elif severity == "high":
                vuln_risk += 1.5
            elif severity == "medium":
                vuln_risk += 0.5
            elif severity == "low":
                vuln_risk += 0.1

        risk_score = min(10, control_risk + vuln_risk)

        # Determine level
        if risk_score <= 2:
            level = SecurityLevel.HIGH
        elif risk_score <= 5:
            level = SecurityLevel.MEDIUM
        elif risk_score <= 8:
            level = SecurityLevel.LOW
        else:
            level = SecurityLevel.CRITICAL

        assessment = SecurityAssessment(
            assessment_id=hashlib.md5(
                datetime.now().isoformat().encode()
            ).hexdigest()[:12],
            level=level,
            vulnerabilities=vulnerabilities,
            controls_passed=passed,
            controls_failed=failed,
            risk_score=risk_score,
            assessed_at=datetime.now()
        )

        self.assessments.append(assessment)
        return assessment

    def get_latest_assessment(self) -> Optional[SecurityAssessment]:
        """Get most recent assessment."""
        return self.assessments[-1] if self.assessments else None


# =============================================================================
# Phase 13: Unified Governance Monitor
# =============================================================================

class GovernanceMonitor:
    """
    Unified monitor for Phase 13: Governance, Security & Compliance.

    Combines audit logging, policy enforcement, compliance, and security.
    """

    def __init__(self):
        self.audit = AuditLogger()
        self.policy = PolicyEnforcer()
        self.compliance = ComplianceChecker()
        self.security = SecurityAssessor()

    def run_governance_check(
        self,
        compliance_evidence: Optional[Dict[RegulatoryFramework, Dict[str, bool]]] = None,
        security_controls: Optional[Dict[str, bool]] = None,
        vulnerabilities: Optional[List[Dict[str, Any]]] = None
    ) -> GovernanceReport:
        """Run comprehensive governance check."""
        issues = []
        recommendations = []

        # Run compliance checks
        compliance_status = {}
        if compliance_evidence:
            for framework, evidence in compliance_evidence.items():
                self.compliance.run_compliance_check(framework, evidence)

            compliance_status = self.compliance.get_compliance_summary()

            for framework, status in compliance_status.items():
                if status != ComplianceStatus.COMPLIANT:
                    issues.append(f"{framework}: {status.value}")
                    recommendations.append(f"Address {framework} compliance gaps")

        # Run security assessment
        security_level = SecurityLevel.MEDIUM
        risk_score = 5.0
        if security_controls:
            assessment = self.security.run_assessment(
                security_controls,
                vulnerabilities
            )
            security_level = assessment.level
            risk_score = assessment.risk_score

            if security_level in [SecurityLevel.LOW, SecurityLevel.CRITICAL]:
                issues.append(f"Security level: {security_level.value}")
                recommendations.append("Address security vulnerabilities")

        # Get policy violations
        violations = self.policy.get_unresolved_violations()
        if violations:
            issues.append(f"{len(violations)} unresolved policy violations")
            recommendations.append("Resolve policy violations")

        # Get audit summary
        audit_summary = self.audit.get_summary()

        # Determine pass status
        passed = (
            all(s == ComplianceStatus.COMPLIANT for s in compliance_status.values()) and
            security_level in [SecurityLevel.HIGH, SecurityLevel.MEDIUM] and
            len(violations) == 0
        )

        return GovernanceReport(
            compliance_status={k: v.value for k, v in compliance_status.items()}
                             if compliance_status else {},
            security_level=security_level,
            policy_violations=violations,
            audit_summary=audit_summary,
            risk_score=risk_score,
            passed=passed,
            issues=issues,
            recommendations=recommendations
        )


__all__ = [
    # Enums
    "ComplianceStatus",
    "SecurityLevel",
    "AuditEventType",
    "RegulatoryFramework",
    # Data classes
    "AuditEvent",
    "PolicyViolation",
    "ComplianceCheck",
    "SecurityAssessment",
    "GovernanceReport",
    # Monitors
    "AuditLogger",
    "PolicyEnforcer",
    "ComplianceChecker",
    "SecurityAssessor",
    "GovernanceMonitor",
]
