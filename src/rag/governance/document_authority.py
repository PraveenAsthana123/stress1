"""
A5) Document Trust & Authority Scoring for EEG-RAG

Comprehensive module for:
- Authority level taxonomy
- Trust scoring (base, freshness, approval)
- Retrieval ranking integration
- Source diversity enforcement
- Conflict detection and resolution
- Deprecation handling

This prevents outdated/low-quality documents from dominating retrieval.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AuthorityLevel(Enum):
    """Document authority levels for EEG domain."""
    A_STANDARD = "standard"  # IEC/IEEE standards
    B_PEER_REVIEWED = "peer_reviewed"  # Published papers
    C_VENDOR_MANUAL = "vendor_manual"  # Device documentation
    D_INTERNAL_APPROVED = "internal_approved"  # Approved SOPs
    E_INTERNAL_DRAFT = "internal_draft"  # Draft documents
    F_NOTES = "notes"  # Lab notes, unknown sources


class ApprovalStatus(Enum):
    """Document approval status."""
    APPROVED = "approved"
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    DEPRECATED = "deprecated"
    UNKNOWN = "unknown"


@dataclass
class AuthorityMetadata:
    """Authority metadata for documents."""
    doc_id: str
    authority_level: AuthorityLevel
    owner: str
    approval_status: ApprovalStatus
    effective_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    version: str = "1.0"
    supersedes: Optional[str] = None  # doc_id of superseded doc
    superseded_by: Optional[str] = None
    jurisdiction: Optional[str] = None  # Site/region applicability
    tags: Set[str] = field(default_factory=set)


@dataclass
class TrustScore:
    """Computed trust score for a document."""
    doc_id: str
    base_score: float
    freshness_score: float
    approval_score: float
    site_fit_score: float
    composite_score: float
    components: Dict[str, float]


class AuthorityRubric:
    """Authority scoring rubric for EEG domain."""

    # Base trust scores by authority level
    BASE_SCORES = {
        AuthorityLevel.A_STANDARD: 1.0,
        AuthorityLevel.B_PEER_REVIEWED: 0.9,
        AuthorityLevel.C_VENDOR_MANUAL: 0.8,
        AuthorityLevel.D_INTERNAL_APPROVED: 0.75,
        AuthorityLevel.E_INTERNAL_DRAFT: 0.5,
        AuthorityLevel.F_NOTES: 0.3,
    }

    # Approval status multipliers
    APPROVAL_MULTIPLIERS = {
        ApprovalStatus.APPROVED: 1.0,
        ApprovalStatus.UNDER_REVIEW: 0.8,
        ApprovalStatus.DRAFT: 0.6,
        ApprovalStatus.DEPRECATED: 0.3,
        ApprovalStatus.UNKNOWN: 0.5,
    }

    # Evergreen documents (don't decay with age)
    EVERGREEN_TYPES = {
        AuthorityLevel.A_STANDARD,
    }

    @classmethod
    def get_base_score(cls, level: AuthorityLevel) -> float:
        """Get base trust score for authority level."""
        return cls.BASE_SCORES.get(level, 0.3)

    @classmethod
    def get_approval_multiplier(cls, status: ApprovalStatus) -> float:
        """Get approval status multiplier."""
        return cls.APPROVAL_MULTIPLIERS.get(status, 0.5)


class FreshnessCalculator:
    """Calculate document freshness scores."""

    def __init__(
        self,
        half_life_days: int = 365,
        min_freshness: float = 0.3
    ):
        self.half_life_days = half_life_days
        self.min_freshness = min_freshness

    def calculate(
        self,
        metadata: AuthorityMetadata,
        reference_date: Optional[datetime] = None
    ) -> float:
        """Calculate freshness score with exponential decay."""
        if reference_date is None:
            reference_date = datetime.now()

        # Evergreen documents don't decay
        if metadata.authority_level in AuthorityRubric.EVERGREEN_TYPES:
            return 1.0

        # Check expiry
        if metadata.expiry_date and reference_date > metadata.expiry_date:
            return 0.0  # Expired

        # Check effective date
        if metadata.effective_date:
            age_days = (reference_date - metadata.effective_date).days
            if age_days < 0:
                return 0.0  # Not yet effective

            # Exponential decay
            decay = np.exp(-np.log(2) * age_days / self.half_life_days)
            return max(self.min_freshness, decay)

        # No date info - assume moderate freshness
        return 0.7


class TrustScoreCalculator:
    """Calculate composite trust scores for documents."""

    def __init__(
        self,
        alpha: float = 0.2,  # Authority boost factor in ranking
        freshness_calculator: Optional[FreshnessCalculator] = None
    ):
        self.alpha = alpha
        self.freshness_calc = freshness_calculator or FreshnessCalculator()

    def calculate(
        self,
        metadata: AuthorityMetadata,
        user_site: Optional[str] = None
    ) -> TrustScore:
        """Calculate composite trust score."""
        # Base score from authority level
        base = AuthorityRubric.get_base_score(metadata.authority_level)

        # Freshness score
        freshness = self.freshness_calc.calculate(metadata)

        # Approval multiplier
        approval = AuthorityRubric.get_approval_multiplier(metadata.approval_status)

        # Site fit score
        site_fit = self._calculate_site_fit(metadata, user_site)

        # Composite score
        composite = base * freshness * approval * site_fit

        return TrustScore(
            doc_id=metadata.doc_id,
            base_score=base,
            freshness_score=freshness,
            approval_score=approval,
            site_fit_score=site_fit,
            composite_score=composite,
            components={
                'base': base,
                'freshness': freshness,
                'approval': approval,
                'site_fit': site_fit,
            }
        )

    def _calculate_site_fit(
        self,
        metadata: AuthorityMetadata,
        user_site: Optional[str]
    ) -> float:
        """Calculate site/jurisdiction fit score."""
        if not metadata.jurisdiction:
            return 1.0  # Universal applicability

        if not user_site:
            return 0.9  # No user site - slight penalty

        if metadata.jurisdiction == user_site:
            return 1.0
        elif metadata.jurisdiction == '*':
            return 1.0  # Applies everywhere

        return 0.7  # Different site


class RetrievalRanker:
    """Integrate trust scores into retrieval ranking."""

    def __init__(self, trust_calculator: TrustScoreCalculator):
        self.trust_calc = trust_calculator

    def rerank(
        self,
        results: List[Dict[str, Any]],
        metadata_map: Dict[str, AuthorityMetadata],
        user_site: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Rerank results incorporating trust scores."""
        scored_results = []

        for result in results:
            doc_id = result.get('doc_id')
            semantic_score = result.get('score', 0.0)

            if doc_id in metadata_map:
                trust = self.trust_calc.calculate(metadata_map[doc_id], user_site)
                # Combined score: semantic + authority boost
                final_score = semantic_score * (1 + self.trust_calc.alpha * trust.composite_score)
                result['trust_score'] = trust.composite_score
                result['trust_components'] = trust.components
            else:
                # Unknown doc - penalize
                final_score = semantic_score * 0.5
                result['trust_score'] = 0.3

            result['final_score'] = final_score
            scored_results.append(result)

        # Sort by final score
        scored_results.sort(key=lambda x: x['final_score'], reverse=True)
        return scored_results


class SourceDiversityEnforcer:
    """Enforce source diversity in retrieval."""

    def __init__(self, min_sources: int = 2):
        self.min_sources = min_sources

    def enforce(
        self,
        results: List[Dict[str, Any]],
        max_per_source: int = 3
    ) -> List[Dict[str, Any]]:
        """Enforce diversity by limiting chunks per source."""
        by_source: Dict[str, List[Dict[str, Any]]] = {}

        for result in results:
            doc_id = result.get('doc_id', 'unknown')
            if doc_id not in by_source:
                by_source[doc_id] = []
            by_source[doc_id].append(result)

        # Interleave results from different sources
        diversified = []
        source_idx = {s: 0 for s in by_source}

        while len(diversified) < len(results):
            added_any = False
            for source, chunks in by_source.items():
                idx = source_idx[source]
                if idx < min(len(chunks), max_per_source):
                    diversified.append(chunks[idx])
                    source_idx[source] += 1
                    added_any = True

            if not added_any:
                break

        return diversified

    def check_diversity(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check if results meet diversity requirements."""
        unique_sources = set(r.get('doc_id') for r in results)
        meets_requirement = len(unique_sources) >= self.min_sources

        return {
            'unique_sources': len(unique_sources),
            'required_sources': self.min_sources,
            'meets_requirement': meets_requirement,
            'sources': list(unique_sources),
        }


class ConflictDetector:
    """Detect conflicts between document sources."""

    def __init__(self):
        self.conflict_patterns = self._load_conflict_patterns()

    def _load_conflict_patterns(self) -> List[Dict[str, Any]]:
        """Load patterns that indicate potential conflicts."""
        return [
            {
                'type': 'numeric_mismatch',
                'pattern': r'(\d+(?:\.\d+)?)\s*(?:Hz|µV|ms)',
            },
            {
                'type': 'range_conflict',
                'pattern': r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*Hz',
            },
            {
                'type': 'procedure_conflict',
                'keywords': ['must', 'should', 'should not', 'never', 'always'],
            },
        ]

    def detect_conflicts(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect potential conflicts between chunks."""
        conflicts = []

        # Compare pairs of chunks
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                conflict = self._compare_chunks(chunks[i], chunks[j])
                if conflict:
                    conflicts.append(conflict)

        return conflicts

    def _compare_chunks(
        self,
        chunk1: Dict[str, Any],
        chunk2: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Compare two chunks for conflicts."""
        import re

        text1 = chunk1.get('text', '').lower()
        text2 = chunk2.get('text', '').lower()

        # Check for numeric mismatches
        hz_pattern = r'(\d+(?:\.\d+)?)\s*hz'
        hz1 = set(re.findall(hz_pattern, text1))
        hz2 = set(re.findall(hz_pattern, text2))

        if hz1 and hz2:
            # Same topic (likely) but different values
            if hz1 != hz2:
                return {
                    'type': 'numeric_conflict',
                    'chunk1_id': chunk1.get('chunk_id'),
                    'chunk2_id': chunk2.get('chunk_id'),
                    'values1': list(hz1),
                    'values2': list(hz2),
                    'severity': 'medium',
                }

        # Check for contradictory qualifiers
        if ('must' in text1 and 'should not' in text2) or \
           ('always' in text1 and 'never' in text2):
            return {
                'type': 'qualifier_conflict',
                'chunk1_id': chunk1.get('chunk_id'),
                'chunk2_id': chunk2.get('chunk_id'),
                'severity': 'high',
            }

        return None


class ConflictResolver:
    """Resolve conflicts between document sources."""

    # Resolution priority order
    PRIORITY_ORDER = [
        ApprovalStatus.APPROVED,
        ApprovalStatus.UNDER_REVIEW,
        ApprovalStatus.DRAFT,
    ]

    def resolve(
        self,
        conflict: Dict[str, Any],
        metadata_map: Dict[str, AuthorityMetadata]
    ) -> Dict[str, Any]:
        """Resolve conflict based on authority and freshness."""
        chunk1_doc = conflict.get('chunk1_id', '').split('_')[0]
        chunk2_doc = conflict.get('chunk2_id', '').split('_')[0]

        meta1 = metadata_map.get(chunk1_doc)
        meta2 = metadata_map.get(chunk2_doc)

        if not meta1 or not meta2:
            return {
                'resolution': 'surface_both',
                'reason': 'Missing metadata',
                'conflict': conflict,
            }

        # Compare authority levels
        if meta1.authority_level.value < meta2.authority_level.value:
            winner = chunk1_doc
            reason = f"Higher authority: {meta1.authority_level.value}"
        elif meta2.authority_level.value < meta1.authority_level.value:
            winner = chunk2_doc
            reason = f"Higher authority: {meta2.authority_level.value}"
        else:
            # Same authority - check supersession
            if meta1.supersedes == meta2.doc_id:
                winner = chunk1_doc
                reason = "Supersedes older version"
            elif meta2.supersedes == meta1.doc_id:
                winner = chunk2_doc
                reason = "Supersedes older version"
            else:
                # Check freshness
                if meta1.effective_date and meta2.effective_date:
                    if meta1.effective_date > meta2.effective_date:
                        winner = chunk1_doc
                        reason = "More recent"
                    else:
                        winner = chunk2_doc
                        reason = "More recent"
                else:
                    return {
                        'resolution': 'surface_both',
                        'reason': 'Cannot determine priority',
                        'conflict': conflict,
                    }

        return {
            'resolution': 'prefer',
            'preferred': winner,
            'reason': reason,
            'conflict': conflict,
        }


class DeprecationRegistry:
    """Track deprecated documents."""

    def __init__(self, registry_path: str = "data/authority/deprecated"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.deprecated: Dict[str, Dict[str, Any]] = {}

    def deprecate(
        self,
        doc_id: str,
        superseded_by: Optional[str] = None,
        reason: str = ""
    ):
        """Mark document as deprecated."""
        self.deprecated[doc_id] = {
            'deprecated_at': datetime.now().isoformat(),
            'superseded_by': superseded_by,
            'reason': reason,
        }
        self._save()

    def is_deprecated(self, doc_id: str) -> bool:
        """Check if document is deprecated."""
        return doc_id in self.deprecated

    def get_replacement(self, doc_id: str) -> Optional[str]:
        """Get replacement document ID."""
        if doc_id in self.deprecated:
            return self.deprecated[doc_id].get('superseded_by')
        return None

    def _save(self):
        """Save registry to disk."""
        with open(self.registry_path / "registry.json", 'w') as f:
            json.dump(self.deprecated, f, indent=2)

    def _load(self):
        """Load registry from disk."""
        path = self.registry_path / "registry.json"
        if path.exists():
            with open(path, 'r') as f:
                self.deprecated = json.load(f)


class AuthorityMonitor:
    """Monitor authority-related metrics."""

    def __init__(self):
        self.metrics = {
            'retrievals_by_authority': {},
            'low_authority_rate': 0.0,
            'expired_retrieval_attempts': 0,
            'conflict_frequency': 0,
        }

    def log_retrieval(
        self,
        results: List[Dict[str, Any]],
        metadata_map: Dict[str, AuthorityMetadata]
    ):
        """Log retrieval for monitoring."""
        for result in results:
            doc_id = result.get('doc_id')
            if doc_id in metadata_map:
                level = metadata_map[doc_id].authority_level.value
                if level not in self.metrics['retrievals_by_authority']:
                    self.metrics['retrievals_by_authority'][level] = 0
                self.metrics['retrievals_by_authority'][level] += 1

        # Calculate low authority rate
        total = sum(self.metrics['retrievals_by_authority'].values())
        low = self.metrics['retrievals_by_authority'].get('notes', 0) + \
              self.metrics['retrievals_by_authority'].get('internal_draft', 0)

        if total > 0:
            self.metrics['low_authority_rate'] = low / total

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for dashboard."""
        return self.metrics


class AuthorityManager:
    """Main authority management coordinator."""

    def __init__(
        self,
        alpha: float = 0.2,
        min_sources: int = 2
    ):
        self.trust_calc = TrustScoreCalculator(alpha=alpha)
        self.ranker = RetrievalRanker(self.trust_calc)
        self.diversity = SourceDiversityEnforcer(min_sources=min_sources)
        self.conflict_detector = ConflictDetector()
        self.conflict_resolver = ConflictResolver()
        self.deprecation = DeprecationRegistry()
        self.monitor = AuthorityMonitor()

    def process_retrieval(
        self,
        results: List[Dict[str, Any]],
        metadata_map: Dict[str, AuthorityMetadata],
        user_site: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process retrieval with authority scoring."""
        # Filter deprecated
        filtered = [
            r for r in results
            if not self.deprecation.is_deprecated(r.get('doc_id', ''))
        ]

        # Rerank by authority
        reranked = self.ranker.rerank(filtered, metadata_map, user_site)

        # Enforce diversity
        diversified = self.diversity.enforce(reranked)

        # Detect conflicts
        conflicts = self.conflict_detector.detect_conflicts(diversified)

        # Resolve conflicts
        resolutions = [
            self.conflict_resolver.resolve(c, metadata_map)
            for c in conflicts
        ]

        # Log for monitoring
        self.monitor.log_retrieval(diversified, metadata_map)

        return {
            'results': diversified,
            'diversity_check': self.diversity.check_diversity(diversified),
            'conflicts': conflicts,
            'resolutions': resolutions,
        }


if __name__ == '__main__':
    # Demo usage
    manager = AuthorityManager()

    # Create test metadata
    metadata_map = {
        'ieee_standard_001': AuthorityMetadata(
            doc_id='ieee_standard_001',
            authority_level=AuthorityLevel.A_STANDARD,
            owner='IEEE',
            approval_status=ApprovalStatus.APPROVED,
            effective_date=datetime(2020, 1, 1),
        ),
        'internal_sop_001': AuthorityMetadata(
            doc_id='internal_sop_001',
            authority_level=AuthorityLevel.D_INTERNAL_APPROVED,
            owner='Lab Admin',
            approval_status=ApprovalStatus.APPROVED,
            effective_date=datetime(2024, 1, 1),
        ),
        'draft_notes_001': AuthorityMetadata(
            doc_id='draft_notes_001',
            authority_level=AuthorityLevel.F_NOTES,
            owner='Researcher',
            approval_status=ApprovalStatus.DRAFT,
        ),
    }

    # Test trust scoring
    for doc_id, meta in metadata_map.items():
        score = manager.trust_calc.calculate(meta)
        print(f"{doc_id}: {score.composite_score:.2f}")

    # Test retrieval processing
    results = [
        {'doc_id': 'ieee_standard_001', 'chunk_id': 'c1', 'score': 0.9, 'text': 'Alpha: 8-13 Hz'},
        {'doc_id': 'internal_sop_001', 'chunk_id': 'c2', 'score': 0.85, 'text': 'Alpha: 8-12 Hz'},
        {'doc_id': 'draft_notes_001', 'chunk_id': 'c3', 'score': 0.95, 'text': 'Random notes'},
    ]

    output = manager.process_retrieval(results, metadata_map)
    print(f"\nDiversity check: {output['diversity_check']}")
    print(f"Conflicts found: {len(output['conflicts'])}")
