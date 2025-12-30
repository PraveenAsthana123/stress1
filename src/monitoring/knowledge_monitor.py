"""
Phase 1: Knowledge & Data Analysis Monitoring

Monitors knowledge sources, data quality, coverage, freshness, and conflicts
for RAG/Agentic systems.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================

class SourceType(Enum):
    """Types of knowledge sources."""
    PEER_REVIEWED = "peer_reviewed"
    VENDOR_MANUAL = "vendor_manual"
    INTERNAL_DOC = "internal_doc"
    DOMAIN_EXPERT = "domain_expert"
    USER_GENERATED = "user_generated"
    EXTERNAL_API = "external_api"


class DataQuality(Enum):
    """Data quality levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class CoverageLevel(Enum):
    """Coverage assessment levels."""
    COMPLETE = "complete"
    PARTIAL = "partial"
    MINIMAL = "minimal"
    MISSING = "missing"


class FreshnessStatus(Enum):
    """Document freshness status."""
    CURRENT = "current"
    STALE = "stale"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


class ConflictSeverity(Enum):
    """Conflict severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class KnowledgeSource:
    """Represents a knowledge source."""
    source_id: str
    name: str
    source_type: SourceType
    authority_score: float  # 0-1
    last_updated: datetime
    document_count: int
    chunk_count: int
    topics: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceInventoryResult:
    """Result of source inventory analysis."""
    total_sources: int
    sources_by_type: Dict[SourceType, int]
    authority_distribution: Dict[str, int]  # high/medium/low counts
    coverage_gaps: List[str]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AuthorityValidation:
    """Result of authority validation."""
    source_id: str
    is_valid: bool
    authority_level: str
    validation_checks: Dict[str, bool]
    issues: List[str]
    score: float


@dataclass
class CoverageAnalysis:
    """Result of coverage analysis."""
    domain: str
    coverage_level: CoverageLevel
    covered_topics: List[str]
    missing_topics: List[str]
    coverage_percentage: float
    recommendations: List[str]


@dataclass
class FreshnessCheck:
    """Result of freshness check."""
    document_id: str
    status: FreshnessStatus
    last_updated: datetime
    age_days: int
    expiry_date: Optional[datetime]
    needs_refresh: bool
    priority: str


@dataclass
class ConflictReport:
    """Report of detected conflicts."""
    conflict_id: str
    severity: ConflictSeverity
    sources_involved: List[str]
    conflicting_claims: List[Dict[str, str]]
    resolution_strategy: str
    resolved: bool = False


# =============================================================================
# Phase 1.1: Knowledge Source Inventory
# =============================================================================

class KnowledgeSourceInventory:
    """
    Monitors and inventories all knowledge sources.

    Purpose: Catalog all knowledge sources with authority levels
    Measurement: Source count, type distribution, authority scores
    Pass/Fail: All sources cataloged with valid metadata
    """

    def __init__(self):
        self.sources: Dict[str, KnowledgeSource] = {}
        self.authority_thresholds = {
            "high": 0.8,
            "medium": 0.5,
            "low": 0.0
        }

    def register_source(self, source: KnowledgeSource) -> bool:
        """Register a knowledge source."""
        if not self._validate_source(source):
            logger.warning(f"Invalid source: {source.source_id}")
            return False

        self.sources[source.source_id] = source
        logger.info(f"Registered source: {source.name} ({source.source_type.value})")
        return True

    def _validate_source(self, source: KnowledgeSource) -> bool:
        """Validate source metadata."""
        if not source.source_id or not source.name:
            return False
        if not 0 <= source.authority_score <= 1:
            return False
        if source.document_count < 0:
            return False
        return True

    def get_inventory(self) -> SourceInventoryResult:
        """Get complete source inventory."""
        sources_by_type: Dict[SourceType, int] = {}
        authority_dist = {"high": 0, "medium": 0, "low": 0}

        for source in self.sources.values():
            # Count by type
            sources_by_type[source.source_type] = \
                sources_by_type.get(source.source_type, 0) + 1

            # Categorize by authority
            if source.authority_score >= self.authority_thresholds["high"]:
                authority_dist["high"] += 1
            elif source.authority_score >= self.authority_thresholds["medium"]:
                authority_dist["medium"] += 1
            else:
                authority_dist["low"] += 1

        # Identify coverage gaps
        coverage_gaps = self._identify_coverage_gaps()
        recommendations = self._generate_recommendations(authority_dist, coverage_gaps)

        return SourceInventoryResult(
            total_sources=len(self.sources),
            sources_by_type=sources_by_type,
            authority_distribution=authority_dist,
            coverage_gaps=coverage_gaps,
            recommendations=recommendations
        )

    def _identify_coverage_gaps(self) -> List[str]:
        """Identify gaps in source coverage."""
        gaps = []

        # Check for missing source types
        present_types = {s.source_type for s in self.sources.values()}
        for source_type in SourceType:
            if source_type not in present_types:
                gaps.append(f"No {source_type.value} sources")

        # Check authority coverage
        high_authority = [s for s in self.sources.values()
                        if s.authority_score >= self.authority_thresholds["high"]]
        if not high_authority:
            gaps.append("No high-authority sources")

        return gaps

    def _generate_recommendations(
        self,
        authority_dist: Dict[str, int],
        coverage_gaps: List[str]
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        if authority_dist["high"] == 0:
            recommendations.append("Add peer-reviewed or expert-validated sources")

        if authority_dist["low"] > authority_dist["high"]:
            recommendations.append("Increase proportion of high-authority sources")

        for gap in coverage_gaps:
            recommendations.append(f"Address gap: {gap}")

        return recommendations


# =============================================================================
# Phase 1.2: Source Authority Validation
# =============================================================================

class SourceAuthorityValidator:
    """
    Validates authority and trustworthiness of knowledge sources.

    Purpose: Verify source credibility and authority
    Measurement: Validation pass rate, authority score accuracy
    Pass/Fail: >90% sources pass authority validation
    """

    def __init__(self):
        self.validation_rules = {
            SourceType.PEER_REVIEWED: self._validate_peer_reviewed,
            SourceType.VENDOR_MANUAL: self._validate_vendor,
            SourceType.INTERNAL_DOC: self._validate_internal,
            SourceType.DOMAIN_EXPERT: self._validate_expert,
            SourceType.USER_GENERATED: self._validate_user_generated,
            SourceType.EXTERNAL_API: self._validate_external,
        }

    def validate_source(self, source: KnowledgeSource) -> AuthorityValidation:
        """Validate a single source."""
        validator = self.validation_rules.get(source.source_type)
        if not validator:
            return AuthorityValidation(
                source_id=source.source_id,
                is_valid=False,
                authority_level="unknown",
                validation_checks={},
                issues=["Unknown source type"],
                score=0.0
            )

        return validator(source)

    def validate_all(
        self,
        sources: List[KnowledgeSource]
    ) -> Tuple[List[AuthorityValidation], float]:
        """Validate all sources and return pass rate."""
        results = [self.validate_source(s) for s in sources]
        pass_count = sum(1 for r in results if r.is_valid)
        pass_rate = pass_count / len(results) if results else 0.0
        return results, pass_rate

    def _validate_peer_reviewed(self, source: KnowledgeSource) -> AuthorityValidation:
        """Validate peer-reviewed source."""
        checks = {
            "has_doi": "doi" in source.metadata,
            "has_publication_date": "publication_date" in source.metadata,
            "has_authors": "authors" in source.metadata,
            "has_journal": "journal" in source.metadata,
            "recent_enough": self._check_recency(source, max_age_years=10),
        }

        issues = [k for k, v in checks.items() if not v]
        is_valid = len(issues) <= 1  # Allow one missing field

        score = sum(checks.values()) / len(checks)
        authority_level = "high" if score >= 0.8 else "medium" if score >= 0.5 else "low"

        return AuthorityValidation(
            source_id=source.source_id,
            is_valid=is_valid,
            authority_level=authority_level,
            validation_checks=checks,
            issues=issues,
            score=score
        )

    def _validate_vendor(self, source: KnowledgeSource) -> AuthorityValidation:
        """Validate vendor manual source."""
        checks = {
            "has_version": "version" in source.metadata,
            "has_vendor": "vendor" in source.metadata,
            "has_product": "product" in source.metadata,
            "recent_update": self._check_recency(source, max_age_years=3),
        }

        issues = [k for k, v in checks.items() if not v]
        is_valid = checks["has_vendor"] and checks["has_product"]

        score = sum(checks.values()) / len(checks)
        authority_level = "high" if is_valid and score >= 0.75 else "medium"

        return AuthorityValidation(
            source_id=source.source_id,
            is_valid=is_valid,
            authority_level=authority_level,
            validation_checks=checks,
            issues=issues,
            score=score
        )

    def _validate_internal(self, source: KnowledgeSource) -> AuthorityValidation:
        """Validate internal document source."""
        checks = {
            "has_owner": "owner" in source.metadata,
            "has_department": "department" in source.metadata,
            "is_approved": source.metadata.get("approved", False),
            "recent_review": self._check_recency(source, max_age_years=2),
        }

        issues = [k for k, v in checks.items() if not v]
        is_valid = checks["is_approved"]

        score = sum(checks.values()) / len(checks)
        authority_level = "high" if is_valid else "low"

        return AuthorityValidation(
            source_id=source.source_id,
            is_valid=is_valid,
            authority_level=authority_level,
            validation_checks=checks,
            issues=issues,
            score=score
        )

    def _validate_expert(self, source: KnowledgeSource) -> AuthorityValidation:
        """Validate domain expert source."""
        checks = {
            "has_expert_name": "expert_name" in source.metadata,
            "has_credentials": "credentials" in source.metadata,
            "has_domain": "domain" in source.metadata,
            "is_verified": source.metadata.get("verified", False),
        }

        issues = [k for k, v in checks.items() if not v]
        is_valid = checks["is_verified"] and checks["has_credentials"]

        score = sum(checks.values()) / len(checks)
        authority_level = "high" if is_valid else "medium" if score >= 0.5 else "low"

        return AuthorityValidation(
            source_id=source.source_id,
            is_valid=is_valid,
            authority_level=authority_level,
            validation_checks=checks,
            issues=issues,
            score=score
        )

    def _validate_user_generated(self, source: KnowledgeSource) -> AuthorityValidation:
        """Validate user-generated source."""
        checks = {
            "has_author": "author" in source.metadata,
            "is_moderated": source.metadata.get("moderated", False),
            "has_votes": source.metadata.get("votes", 0) > 0,
        }

        issues = [k for k, v in checks.items() if not v]
        is_valid = checks["is_moderated"]

        score = sum(checks.values()) / len(checks)
        authority_level = "low"  # User-generated always low authority

        return AuthorityValidation(
            source_id=source.source_id,
            is_valid=is_valid,
            authority_level=authority_level,
            validation_checks=checks,
            issues=issues,
            score=score
        )

    def _validate_external(self, source: KnowledgeSource) -> AuthorityValidation:
        """Validate external API source."""
        checks = {
            "has_api_name": "api_name" in source.metadata,
            "has_provider": "provider" in source.metadata,
            "is_authenticated": source.metadata.get("authenticated", False),
            "has_sla": "sla" in source.metadata,
        }

        issues = [k for k, v in checks.items() if not v]
        is_valid = checks["is_authenticated"]

        score = sum(checks.values()) / len(checks)
        authority_level = "medium" if is_valid else "low"

        return AuthorityValidation(
            source_id=source.source_id,
            is_valid=is_valid,
            authority_level=authority_level,
            validation_checks=checks,
            issues=issues,
            score=score
        )

    def _check_recency(self, source: KnowledgeSource, max_age_years: int) -> bool:
        """Check if source is recent enough."""
        age = datetime.now() - source.last_updated
        return age.days < (max_age_years * 365)


# =============================================================================
# Phase 1.3: Knowledge Coverage Analysis
# =============================================================================

class KnowledgeCoverageAnalyzer:
    """
    Analyzes knowledge coverage across domains.

    Purpose: Identify coverage gaps and blind spots
    Measurement: Coverage percentage per domain/topic
    Pass/Fail: >80% coverage in critical domains
    """

    def __init__(self):
        self.domain_taxonomy: Dict[str, List[str]] = {}
        self.coverage_requirements: Dict[str, float] = {}

    def define_domain(
        self,
        domain: str,
        topics: List[str],
        required_coverage: float = 0.8
    ):
        """Define a domain with required topics."""
        self.domain_taxonomy[domain] = topics
        self.coverage_requirements[domain] = required_coverage

    def analyze_coverage(
        self,
        domain: str,
        sources: List[KnowledgeSource]
    ) -> CoverageAnalysis:
        """Analyze coverage for a domain."""
        if domain not in self.domain_taxonomy:
            return CoverageAnalysis(
                domain=domain,
                coverage_level=CoverageLevel.UNKNOWN,
                covered_topics=[],
                missing_topics=[],
                coverage_percentage=0.0,
                recommendations=["Domain not defined in taxonomy"]
            )

        required_topics = set(self.domain_taxonomy[domain])
        covered_topics = set()

        for source in sources:
            covered_topics.update(
                t for t in source.topics if t in required_topics
            )

        missing_topics = required_topics - covered_topics
        coverage_pct = len(covered_topics) / len(required_topics) if required_topics else 0.0

        # Determine coverage level
        if coverage_pct >= 0.9:
            level = CoverageLevel.COMPLETE
        elif coverage_pct >= 0.7:
            level = CoverageLevel.PARTIAL
        elif coverage_pct >= 0.3:
            level = CoverageLevel.MINIMAL
        else:
            level = CoverageLevel.MISSING

        recommendations = []
        if missing_topics:
            recommendations.append(f"Add sources covering: {', '.join(list(missing_topics)[:5])}")

        required = self.coverage_requirements.get(domain, 0.8)
        if coverage_pct < required:
            recommendations.append(
                f"Coverage {coverage_pct:.1%} below required {required:.1%}"
            )

        return CoverageAnalysis(
            domain=domain,
            coverage_level=level,
            covered_topics=list(covered_topics),
            missing_topics=list(missing_topics),
            coverage_percentage=coverage_pct,
            recommendations=recommendations
        )

    def get_overall_coverage(
        self,
        sources: List[KnowledgeSource]
    ) -> Dict[str, CoverageAnalysis]:
        """Get coverage analysis for all domains."""
        return {
            domain: self.analyze_coverage(domain, sources)
            for domain in self.domain_taxonomy
        }


# =============================================================================
# Phase 1.4: Document Freshness Checker
# =============================================================================

class DocumentFreshnessChecker:
    """
    Monitors document freshness and staleness.

    Purpose: Ensure knowledge is current and not outdated
    Measurement: Age distribution, stale document count
    Pass/Fail: <10% documents past refresh date
    """

    def __init__(self):
        self.refresh_policies: Dict[SourceType, timedelta] = {
            SourceType.PEER_REVIEWED: timedelta(days=365 * 5),  # 5 years
            SourceType.VENDOR_MANUAL: timedelta(days=365),      # 1 year
            SourceType.INTERNAL_DOC: timedelta(days=180),       # 6 months
            SourceType.DOMAIN_EXPERT: timedelta(days=365 * 2),  # 2 years
            SourceType.USER_GENERATED: timedelta(days=90),      # 3 months
            SourceType.EXTERNAL_API: timedelta(days=30),        # 1 month
        }

    def set_refresh_policy(self, source_type: SourceType, max_age: timedelta):
        """Set refresh policy for a source type."""
        self.refresh_policies[source_type] = max_age

    def check_freshness(
        self,
        document_id: str,
        last_updated: datetime,
        source_type: SourceType
    ) -> FreshnessCheck:
        """Check freshness of a single document."""
        age = datetime.now() - last_updated
        max_age = self.refresh_policies.get(source_type, timedelta(days=365))
        expiry_date = last_updated + max_age

        # Determine status
        if age < max_age * 0.5:
            status = FreshnessStatus.CURRENT
            priority = "low"
        elif age < max_age:
            status = FreshnessStatus.STALE
            priority = "medium"
        else:
            status = FreshnessStatus.EXPIRED
            priority = "high"

        needs_refresh = status in [FreshnessStatus.STALE, FreshnessStatus.EXPIRED]

        return FreshnessCheck(
            document_id=document_id,
            status=status,
            last_updated=last_updated,
            age_days=age.days,
            expiry_date=expiry_date,
            needs_refresh=needs_refresh,
            priority=priority
        )

    def check_all(
        self,
        documents: List[Tuple[str, datetime, SourceType]]
    ) -> Tuple[List[FreshnessCheck], float]:
        """Check freshness of all documents."""
        results = [
            self.check_freshness(doc_id, updated, source_type)
            for doc_id, updated, source_type in documents
        ]

        stale_count = sum(1 for r in results if r.needs_refresh)
        stale_rate = stale_count / len(results) if results else 0.0

        return results, stale_rate

    def get_refresh_queue(
        self,
        freshness_checks: List[FreshnessCheck],
        max_items: int = 10
    ) -> List[FreshnessCheck]:
        """Get prioritized refresh queue."""
        needs_refresh = [c for c in freshness_checks if c.needs_refresh]

        # Sort by priority (high first) then by age (oldest first)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        needs_refresh.sort(
            key=lambda x: (priority_order.get(x.priority, 3), -x.age_days)
        )

        return needs_refresh[:max_items]


# =============================================================================
# Phase 1.5: Knowledge Conflict Scanner
# =============================================================================

class KnowledgeConflictScanner:
    """
    Detects and manages conflicts in knowledge base.

    Purpose: Identify contradictory information across sources
    Measurement: Conflict count, resolution rate
    Pass/Fail: All critical conflicts resolved
    """

    def __init__(self):
        self.conflicts: Dict[str, ConflictReport] = {}
        self.resolution_strategies = {
            ConflictSeverity.CRITICAL: "manual_review",
            ConflictSeverity.HIGH: "authority_based",
            ConflictSeverity.MEDIUM: "recency_based",
            ConflictSeverity.LOW: "ignore",
        }

    def scan_for_conflicts(
        self,
        claims: List[Dict[str, Any]]
    ) -> List[ConflictReport]:
        """Scan claims for conflicts."""
        conflicts = []

        # Group claims by topic
        claims_by_topic: Dict[str, List[Dict]] = {}
        for claim in claims:
            topic = claim.get("topic", "unknown")
            if topic not in claims_by_topic:
                claims_by_topic[topic] = []
            claims_by_topic[topic].append(claim)

        # Check each topic for conflicts
        for topic, topic_claims in claims_by_topic.items():
            if len(topic_claims) < 2:
                continue

            conflict = self._detect_conflict(topic, topic_claims)
            if conflict:
                conflicts.append(conflict)
                self.conflicts[conflict.conflict_id] = conflict

        return conflicts

    def _detect_conflict(
        self,
        topic: str,
        claims: List[Dict[str, Any]]
    ) -> Optional[ConflictReport]:
        """Detect conflict in claims about the same topic."""
        # Simple conflict detection based on contradictory values
        values = [c.get("value") for c in claims if "value" in c]

        if len(set(values)) <= 1:
            return None  # No conflict

        # Determine severity based on difference
        severity = self._assess_severity(claims)

        conflict_id = hashlib.md5(
            f"{topic}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        return ConflictReport(
            conflict_id=conflict_id,
            severity=severity,
            sources_involved=[c.get("source_id", "unknown") for c in claims],
            conflicting_claims=[
                {"source": c.get("source_id"), "value": c.get("value")}
                for c in claims
            ],
            resolution_strategy=self.resolution_strategies[severity]
        )

    def _assess_severity(self, claims: List[Dict[str, Any]]) -> ConflictSeverity:
        """Assess conflict severity."""
        # Check if any claim is from high-authority source
        authorities = [c.get("authority", 0) for c in claims]

        if max(authorities) >= 0.9:
            return ConflictSeverity.HIGH
        elif max(authorities) >= 0.7:
            return ConflictSeverity.MEDIUM
        else:
            return ConflictSeverity.LOW

    def resolve_conflict(
        self,
        conflict_id: str,
        resolution: str = "auto"
    ) -> bool:
        """Resolve a conflict."""
        if conflict_id not in self.conflicts:
            return False

        conflict = self.conflicts[conflict_id]
        conflict.resolved = True

        logger.info(f"Resolved conflict {conflict_id} via {resolution}")
        return True

    def get_unresolved_conflicts(
        self,
        severity_filter: Optional[ConflictSeverity] = None
    ) -> List[ConflictReport]:
        """Get list of unresolved conflicts."""
        unresolved = [c for c in self.conflicts.values() if not c.resolved]

        if severity_filter:
            unresolved = [c for c in unresolved if c.severity == severity_filter]

        return unresolved

    def get_conflict_stats(self) -> Dict[str, Any]:
        """Get conflict statistics."""
        total = len(self.conflicts)
        resolved = sum(1 for c in self.conflicts.values() if c.resolved)

        by_severity = {}
        for severity in ConflictSeverity:
            by_severity[severity.value] = sum(
                1 for c in self.conflicts.values() if c.severity == severity
            )

        return {
            "total_conflicts": total,
            "resolved": resolved,
            "unresolved": total - resolved,
            "resolution_rate": resolved / total if total else 1.0,
            "by_severity": by_severity
        }


# =============================================================================
# Phase 1: Unified Knowledge Monitor
# =============================================================================

class KnowledgePhaseMonitor:
    """
    Unified monitor for Phase 1: Knowledge & Data Analysis.

    Combines all knowledge monitoring capabilities.
    """

    def __init__(self):
        self.inventory = KnowledgeSourceInventory()
        self.authority_validator = SourceAuthorityValidator()
        self.coverage_analyzer = KnowledgeCoverageAnalyzer()
        self.freshness_checker = DocumentFreshnessChecker()
        self.conflict_scanner = KnowledgeConflictScanner()

    def run_full_analysis(
        self,
        sources: List[KnowledgeSource],
        claims: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Run full Phase 1 analysis."""
        # Register all sources
        for source in sources:
            self.inventory.register_source(source)

        # Get inventory
        inventory_result = self.inventory.get_inventory()

        # Validate authority
        authority_results, authority_pass_rate = \
            self.authority_validator.validate_all(sources)

        # Check freshness
        doc_list = [
            (s.source_id, s.last_updated, s.source_type)
            for s in sources
        ]
        freshness_results, stale_rate = self.freshness_checker.check_all(doc_list)

        # Scan for conflicts
        conflict_reports = []
        if claims:
            conflict_reports = self.conflict_scanner.scan_for_conflicts(claims)

        # Compile results
        return {
            "inventory": {
                "total_sources": inventory_result.total_sources,
                "by_type": {k.value: v for k, v in inventory_result.sources_by_type.items()},
                "authority_distribution": inventory_result.authority_distribution,
                "coverage_gaps": inventory_result.coverage_gaps,
            },
            "authority": {
                "pass_rate": authority_pass_rate,
                "passed": authority_pass_rate >= 0.9,
                "issues": [
                    {"source": r.source_id, "issues": r.issues}
                    for r in authority_results if not r.is_valid
                ]
            },
            "freshness": {
                "stale_rate": stale_rate,
                "passed": stale_rate < 0.1,
                "refresh_queue": [
                    {"doc_id": f.document_id, "age_days": f.age_days, "priority": f.priority}
                    for f in self.freshness_checker.get_refresh_queue(freshness_results)
                ]
            },
            "conflicts": {
                "total": len(conflict_reports),
                "critical": sum(1 for c in conflict_reports
                               if c.severity == ConflictSeverity.CRITICAL),
                "passed": all(c.severity != ConflictSeverity.CRITICAL
                             for c in conflict_reports)
            },
            "overall_passed": (
                authority_pass_rate >= 0.9 and
                stale_rate < 0.1 and
                all(c.severity != ConflictSeverity.CRITICAL for c in conflict_reports)
            )
        }


__all__ = [
    # Enums
    "SourceType",
    "DataQuality",
    "CoverageLevel",
    "FreshnessStatus",
    "ConflictSeverity",
    # Data classes
    "KnowledgeSource",
    "SourceInventoryResult",
    "AuthorityValidation",
    "CoverageAnalysis",
    "FreshnessCheck",
    "ConflictReport",
    # Monitors
    "KnowledgeSourceInventory",
    "SourceAuthorityValidator",
    "KnowledgeCoverageAnalyzer",
    "DocumentFreshnessChecker",
    "KnowledgeConflictScanner",
    "KnowledgePhaseMonitor",
]
