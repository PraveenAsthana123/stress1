"""
A4) Role-Based Access Control (RBAC) for Retrieval in EEG-RAG

Comprehensive module for:
- Role and access level definitions
- Resource type management
- Policy model (RBAC + ABAC)
- Document ACL tagging
- Retrieval filtering
- Cache isolation
- Break-glass emergency access
- Audit logging

This ensures retrieval never returns content users aren't authorized to see.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Callable
from enum import Enum
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class Role(Enum):
    """User roles in EEG-RAG system."""
    STUDENT = "student"
    RESEARCHER = "researcher"
    ENGINEER = "engineer"
    CLINICAL = "clinical"
    ADMIN = "admin"
    AUDITOR = "auditor"


class AccessLevel(Enum):
    """Document access levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"


class ResourceType(Enum):
    """RAG resource types subject to RBAC."""
    DOCUMENT = "document"
    CHUNK = "chunk"
    TABLE = "table"
    KG_NODE = "kg_node"
    KG_EDGE = "kg_edge"
    LOG = "log"
    PROMPT = "prompt"
    CONFIG = "config"


@dataclass
class ACLMetadata:
    """Access control metadata for documents."""
    access_level: AccessLevel
    owner: str
    tenant_id: str
    project_id: Optional[str] = None
    site_id: Optional[str] = None
    approved: bool = True
    effective_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    allowed_roles: Set[Role] = field(default_factory=set)
    denied_roles: Set[Role] = field(default_factory=set)


@dataclass
class UserContext:
    """User context for access control."""
    user_id: str
    roles: Set[Role]
    tenant_id: str
    project_ids: Set[str] = field(default_factory=set)
    site_ids: Set[str] = field(default_factory=set)
    is_break_glass: bool = False
    break_glass_expiry: Optional[datetime] = None


@dataclass
class AccessDecision:
    """Result of access control decision."""
    allowed: bool
    reason: str
    checked_policies: List[str]
    audit_required: bool = False


class RoleAccessMatrix:
    """Role to access level mapping."""

    # Default role permissions
    ROLE_PERMISSIONS = {
        Role.STUDENT: {AccessLevel.PUBLIC},
        Role.RESEARCHER: {AccessLevel.PUBLIC, AccessLevel.INTERNAL},
        Role.ENGINEER: {AccessLevel.PUBLIC, AccessLevel.INTERNAL},
        Role.CLINICAL: {AccessLevel.PUBLIC, AccessLevel.INTERNAL, AccessLevel.RESTRICTED},
        Role.ADMIN: {AccessLevel.PUBLIC, AccessLevel.INTERNAL, AccessLevel.RESTRICTED, AccessLevel.CONFIDENTIAL},
        Role.AUDITOR: {AccessLevel.PUBLIC, AccessLevel.INTERNAL},  # Plus audit logs
    }

    @classmethod
    def can_access(cls, role: Role, access_level: AccessLevel) -> bool:
        """Check if role can access given level."""
        allowed_levels = cls.ROLE_PERMISSIONS.get(role, set())
        return access_level in allowed_levels


class PolicyEvaluator:
    """Evaluate access policies for resources."""

    def __init__(self):
        self.policies: List[Callable[[UserContext, ACLMetadata], Optional[AccessDecision]]] = []
        self._register_default_policies()

    def _register_default_policies(self):
        """Register default access policies."""

        # Policy 1: Role-based access
        def role_policy(user: UserContext, acl: ACLMetadata) -> Optional[AccessDecision]:
            for role in user.roles:
                if RoleAccessMatrix.can_access(role, acl.access_level):
                    return None  # Continue checking
            return AccessDecision(
                allowed=False,
                reason=f"No role has access to {acl.access_level.value}",
                checked_policies=['role_policy'],
            )

        # Policy 2: Tenant isolation
        def tenant_policy(user: UserContext, acl: ACLMetadata) -> Optional[AccessDecision]:
            if acl.tenant_id != user.tenant_id:
                return AccessDecision(
                    allowed=False,
                    reason=f"Tenant mismatch: {user.tenant_id} != {acl.tenant_id}",
                    checked_policies=['tenant_policy'],
                )
            return None

        # Policy 3: Project scope
        def project_policy(user: UserContext, acl: ACLMetadata) -> Optional[AccessDecision]:
            if acl.project_id and acl.project_id not in user.project_ids:
                return AccessDecision(
                    allowed=False,
                    reason=f"User not in project {acl.project_id}",
                    checked_policies=['project_policy'],
                )
            return None

        # Policy 4: Expiry check
        def expiry_policy(user: UserContext, acl: ACLMetadata) -> Optional[AccessDecision]:
            now = datetime.now()
            if acl.expiry_date and now > acl.expiry_date:
                return AccessDecision(
                    allowed=False,
                    reason=f"Resource expired on {acl.expiry_date}",
                    checked_policies=['expiry_policy'],
                )
            if acl.effective_date and now < acl.effective_date:
                return AccessDecision(
                    allowed=False,
                    reason=f"Resource not effective until {acl.effective_date}",
                    checked_policies=['expiry_policy'],
                )
            return None

        # Policy 5: Approval check
        def approval_policy(user: UserContext, acl: ACLMetadata) -> Optional[AccessDecision]:
            if not acl.approved:
                if Role.ADMIN not in user.roles:
                    return AccessDecision(
                        allowed=False,
                        reason="Resource not approved",
                        checked_policies=['approval_policy'],
                    )
            return None

        # Policy 6: Explicit deny
        def explicit_deny_policy(user: UserContext, acl: ACLMetadata) -> Optional[AccessDecision]:
            if user.roles & acl.denied_roles:
                return AccessDecision(
                    allowed=False,
                    reason="Role explicitly denied",
                    checked_policies=['explicit_deny_policy'],
                )
            return None

        # Policy 7: Break-glass override
        def break_glass_policy(user: UserContext, acl: ACLMetadata) -> Optional[AccessDecision]:
            if user.is_break_glass:
                if user.break_glass_expiry and datetime.now() < user.break_glass_expiry:
                    return AccessDecision(
                        allowed=True,
                        reason="Break-glass access granted",
                        checked_policies=['break_glass_policy'],
                        audit_required=True,
                    )
            return None

        self.policies = [
            break_glass_policy,  # Check first - can override all
            explicit_deny_policy,
            tenant_policy,
            project_policy,
            role_policy,
            expiry_policy,
            approval_policy,
        ]

    def evaluate(self, user: UserContext, acl: ACLMetadata) -> AccessDecision:
        """Evaluate all policies and return decision."""
        checked = []

        for policy in self.policies:
            decision = policy(user, acl)
            checked.append(policy.__name__)

            if decision is not None:
                decision.checked_policies = checked
                return decision

        # All policies passed
        return AccessDecision(
            allowed=True,
            reason="All policies passed",
            checked_policies=checked,
        )


class RetrievalFilter:
    """Filter retrieval results based on RBAC."""

    def __init__(self, evaluator: PolicyEvaluator):
        self.evaluator = evaluator

    def filter_chunks(
        self,
        chunks: List[Dict[str, Any]],
        user: UserContext
    ) -> List[Dict[str, Any]]:
        """Filter chunks based on user access."""
        allowed_chunks = []

        for chunk in chunks:
            acl = self._extract_acl(chunk)
            decision = self.evaluator.evaluate(user, acl)

            if decision.allowed:
                allowed_chunks.append(chunk)
            else:
                logger.debug(f"Chunk {chunk.get('chunk_id')} denied: {decision.reason}")

        return allowed_chunks

    def _extract_acl(self, chunk: Dict[str, Any]) -> ACLMetadata:
        """Extract ACL metadata from chunk."""
        metadata = chunk.get('metadata', {})

        return ACLMetadata(
            access_level=AccessLevel(metadata.get('access_level', 'public')),
            owner=metadata.get('owner', 'unknown'),
            tenant_id=metadata.get('tenant_id', 'default'),
            project_id=metadata.get('project_id'),
            site_id=metadata.get('site_id'),
            approved=metadata.get('approved', True),
            effective_date=self._parse_date(metadata.get('effective_date')),
            expiry_date=self._parse_date(metadata.get('expiry_date')),
        )

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime."""
        if date_str:
            try:
                return datetime.fromisoformat(date_str)
            except (ValueError, TypeError):
                pass
        return None

    def build_filter_query(self, user: UserContext) -> Dict[str, Any]:
        """Build filter query for vector DB."""
        # Build metadata filter for pre-filtering
        allowed_levels = set()
        for role in user.roles:
            allowed_levels.update(RoleAccessMatrix.ROLE_PERMISSIONS.get(role, set()))

        return {
            'tenant_id': {'$eq': user.tenant_id},
            'access_level': {'$in': [l.value for l in allowed_levels]},
            'approved': {'$eq': True},
        }


class CacheKeyBuilder:
    """Build RBAC-aware cache keys."""

    @staticmethod
    def build_key(
        query: str,
        user: UserContext,
        config_version: str = "v1"
    ) -> str:
        """Build cache key including role/tenant scope."""
        components = [
            query,
            user.tenant_id,
            ','.join(sorted(r.value for r in user.roles)),
            ','.join(sorted(user.project_ids)),
            config_version,
        ]
        key_string = '|'.join(components)
        return hashlib.sha256(key_string.encode()).hexdigest()


class BreakGlassManager:
    """Manage break-glass emergency access."""

    def __init__(self, log_path: str = "data/audit/break_glass"):
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    def grant_access(
        self,
        user_id: str,
        approver_id: str,
        justification: str,
        duration_minutes: int = 60
    ) -> UserContext:
        """Grant break-glass access."""
        expiry = datetime.now() + timedelta(minutes=duration_minutes)

        session = {
            'user_id': user_id,
            'approver_id': approver_id,
            'justification': justification,
            'granted_at': datetime.now().isoformat(),
            'expires_at': expiry.isoformat(),
            'duration_minutes': duration_minutes,
        }

        self.active_sessions[user_id] = session
        self._log_grant(session)

        logger.warning(f"Break-glass access granted to {user_id} by {approver_id}")

        return UserContext(
            user_id=user_id,
            roles={Role.ADMIN},
            tenant_id='*',
            is_break_glass=True,
            break_glass_expiry=expiry,
        )

    def revoke_access(self, user_id: str):
        """Revoke break-glass access."""
        if user_id in self.active_sessions:
            session = self.active_sessions.pop(user_id)
            self._log_revoke(user_id, session)
            logger.info(f"Break-glass access revoked for {user_id}")

    def _log_grant(self, session: Dict[str, Any]):
        """Log break-glass grant."""
        log_file = self.log_path / f"grant_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump(session, f, indent=2)

    def _log_revoke(self, user_id: str, session: Dict[str, Any]):
        """Log break-glass revoke."""
        log_file = self.log_path / f"revoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        data = {
            **session,
            'revoked_at': datetime.now().isoformat(),
        }
        with open(log_file, 'w') as f:
            json.dump(data, f, indent=2)


class RBACauditLog:
    """Audit logging for RBAC decisions."""

    def __init__(self, log_path: str = "data/audit/rbac"):
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.entries: List[Dict[str, Any]] = []

    def log_decision(
        self,
        user: UserContext,
        resource_id: str,
        resource_type: ResourceType,
        decision: AccessDecision
    ):
        """Log access control decision."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user.user_id,
            'roles': [r.value for r in user.roles],
            'tenant_id': user.tenant_id,
            'resource_id': resource_id,
            'resource_type': resource_type.value,
            'allowed': decision.allowed,
            'reason': decision.reason,
            'policies_checked': decision.checked_policies,
            'is_break_glass': user.is_break_glass,
        }

        self.entries.append(entry)

        # Write to file periodically
        if len(self.entries) >= 100:
            self._flush()

    def _flush(self):
        """Flush entries to disk."""
        if not self.entries:
            return

        log_file = self.log_path / f"rbac_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(log_file, 'a') as f:
            for entry in self.entries:
                f.write(json.dumps(entry) + '\n')

        self.entries = []

    def get_user_access_report(self, user_id: str) -> Dict[str, Any]:
        """Generate access report for user."""
        user_entries = [e for e in self.entries if e['user_id'] == user_id]
        return {
            'user_id': user_id,
            'total_requests': len(user_entries),
            'allowed': sum(1 for e in user_entries if e['allowed']),
            'denied': sum(1 for e in user_entries if not e['allowed']),
            'break_glass_uses': sum(1 for e in user_entries if e['is_break_glass']),
        }


class RBACTestSuite:
    """Security regression tests for RBAC."""

    def __init__(self, evaluator: PolicyEvaluator):
        self.evaluator = evaluator
        self.test_results: List[Dict[str, Any]] = []

    def test_role_escalation(self) -> bool:
        """Test that lower roles can't access higher content."""
        student = UserContext(
            user_id="test_student",
            roles={Role.STUDENT},
            tenant_id="test_tenant",
        )

        restricted_acl = ACLMetadata(
            access_level=AccessLevel.RESTRICTED,
            owner="admin",
            tenant_id="test_tenant",
        )

        decision = self.evaluator.evaluate(student, restricted_acl)

        passed = not decision.allowed
        self.test_results.append({
            'test': 'role_escalation',
            'passed': passed,
            'details': decision.reason,
        })
        return passed

    def test_tenant_isolation(self) -> bool:
        """Test tenant isolation."""
        user = UserContext(
            user_id="test_user",
            roles={Role.RESEARCHER},
            tenant_id="tenant_a",
        )

        other_tenant_acl = ACLMetadata(
            access_level=AccessLevel.INTERNAL,
            owner="owner",
            tenant_id="tenant_b",
        )

        decision = self.evaluator.evaluate(user, other_tenant_acl)

        passed = not decision.allowed
        self.test_results.append({
            'test': 'tenant_isolation',
            'passed': passed,
            'details': decision.reason,
        })
        return passed

    def test_expiry_enforcement(self) -> bool:
        """Test document expiry is enforced."""
        user = UserContext(
            user_id="test_user",
            roles={Role.ADMIN},
            tenant_id="test_tenant",
        )

        expired_acl = ACLMetadata(
            access_level=AccessLevel.INTERNAL,
            owner="owner",
            tenant_id="test_tenant",
            expiry_date=datetime.now() - timedelta(days=1),
        )

        decision = self.evaluator.evaluate(user, expired_acl)

        passed = not decision.allowed
        self.test_results.append({
            'test': 'expiry_enforcement',
            'passed': passed,
            'details': decision.reason,
        })
        return passed

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all RBAC tests."""
        tests = [
            self.test_role_escalation,
            self.test_tenant_isolation,
            self.test_expiry_enforcement,
        ]

        all_passed = all(test() for test in tests)

        return {
            'all_passed': all_passed,
            'results': self.test_results,
            'timestamp': datetime.now().isoformat(),
        }


class RBACManager:
    """Main RBAC manager coordinating all components."""

    def __init__(
        self,
        audit_path: str = "data/audit/rbac",
        break_glass_path: str = "data/audit/break_glass"
    ):
        self.evaluator = PolicyEvaluator()
        self.filter = RetrievalFilter(self.evaluator)
        self.audit = RBACauditLog(audit_path)
        self.break_glass = BreakGlassManager(break_glass_path)

    def check_access(
        self,
        user: UserContext,
        resource_id: str,
        resource_type: ResourceType,
        acl: ACLMetadata
    ) -> AccessDecision:
        """Check access and log decision."""
        decision = self.evaluator.evaluate(user, acl)
        self.audit.log_decision(user, resource_id, resource_type, decision)
        return decision

    def filter_retrieval(
        self,
        chunks: List[Dict[str, Any]],
        user: UserContext
    ) -> List[Dict[str, Any]]:
        """Filter retrieval results."""
        return self.filter.filter_chunks(chunks, user)

    def get_cache_key(
        self,
        query: str,
        user: UserContext
    ) -> str:
        """Get RBAC-aware cache key."""
        return CacheKeyBuilder.build_key(query, user)


if __name__ == '__main__':
    # Demo usage
    manager = RBACManager()

    # Create test user
    user = UserContext(
        user_id="researcher_001",
        roles={Role.RESEARCHER},
        tenant_id="university_a",
        project_ids={"eeg_study_1"},
    )

    # Create test chunks
    chunks = [
        {
            'chunk_id': 'chunk_001',
            'text': 'Public EEG information',
            'metadata': {
                'access_level': 'public',
                'owner': 'system',
                'tenant_id': 'university_a',
            }
        },
        {
            'chunk_id': 'chunk_002',
            'text': 'Restricted clinical data',
            'metadata': {
                'access_level': 'restricted',
                'owner': 'clinical',
                'tenant_id': 'university_a',
            }
        },
    ]

    # Filter chunks
    allowed = manager.filter_retrieval(chunks, user)
    print(f"User {user.user_id} can access {len(allowed)}/{len(chunks)} chunks")

    # Run tests
    tests = RBACTestSuite(manager.evaluator)
    results = tests.run_all_tests()
    print(f"RBAC tests passed: {results['all_passed']}")
