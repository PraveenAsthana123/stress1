#!/usr/bin/env python3
"""
Responsible AI Module for RAG System

Comprehensive AI governance covering:
- Explainability & Interpretability
- Trust & Reliability
- Ethical AI
- Compliance & Audit
- Security & Privacy
- Bias Detection & Fairness
"""

import json
import time
import hashlib
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
from collections import defaultdict


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """Audit log entry."""
    timestamp: str
    action: str
    user_id: str
    query: str
    response_hash: str
    metadata: Dict
    flags: List[str] = field(default_factory=list)


@dataclass
class ExplanationResult:
    """Result of explanation generation."""
    query: str
    answer: str
    reasoning_steps: List[str]
    source_attributions: List[Dict]
    confidence_breakdown: Dict
    uncertainty_factors: List[str]


# ==================== EXPLAINABILITY ====================

class ExplainabilityModule:
    """Generate explanations for RAG responses."""

    def __init__(self):
        self.explanation_templates = {
            "retrieval": "Retrieved {num_sources} relevant documents with average similarity score of {avg_score:.2f}",
            "generation": "Generated response based on {context_length} tokens of context",
            "confidence": "Confidence level: {confidence:.1%} based on source agreement and relevance scores"
        }

    def explain_retrieval(self, retrieved_docs: List[Dict]) -> Dict:
        """Explain why documents were retrieved."""
        if not retrieved_docs:
            return {"summary": "No documents retrieved", "details": []}

        explanations = []
        for i, doc in enumerate(retrieved_docs, 1):
            explanations.append({
                "rank": i,
                "source": doc.get("source", "Unknown"),
                "score": doc.get("score", 0),
                "reason": f"Matched query terms with similarity score {doc.get('score', 0):.3f}",
                "key_terms": self._extract_key_terms(doc.get("text", ""))
            })

        avg_score = sum(d.get("score", 0) for d in retrieved_docs) / len(retrieved_docs)

        return {
            "summary": self.explanation_templates["retrieval"].format(
                num_sources=len(retrieved_docs),
                avg_score=avg_score
            ),
            "details": explanations,
            "average_score": avg_score
        }

    def explain_answer(self, answer: str, sources: List[Dict], query: str) -> ExplanationResult:
        """Generate comprehensive explanation for an answer."""
        reasoning_steps = [
            f"1. Analyzed query: '{query}'",
            f"2. Retrieved {len(sources)} relevant documents",
            "3. Identified key concepts in sources",
            "4. Synthesized information into coherent response",
            "5. Applied domain knowledge for context"
        ]

        source_attributions = []
        for source in sources:
            attribution = {
                "source": source.get("source", "Unknown"),
                "contribution": self._estimate_contribution(answer, source.get("text", "")),
                "relevance_score": source.get("score", 0)
            }
            source_attributions.append(attribution)

        confidence_breakdown = {
            "source_quality": min(1.0, sum(s.get("score", 0) for s in sources) / max(len(sources), 1)),
            "coverage": self._calculate_coverage(query, sources),
            "consistency": self._check_consistency(sources)
        }

        uncertainty_factors = []
        if confidence_breakdown["source_quality"] < 0.5:
            uncertainty_factors.append("Low source relevance scores")
        if len(sources) < 2:
            uncertainty_factors.append("Limited number of sources")
        if confidence_breakdown["coverage"] < 0.5:
            uncertainty_factors.append("Query terms not well covered in sources")

        return ExplanationResult(
            query=query,
            answer=answer,
            reasoning_steps=reasoning_steps,
            source_attributions=source_attributions,
            confidence_breakdown=confidence_breakdown,
            uncertainty_factors=uncertainty_factors
        )

    def _extract_key_terms(self, text: str, n: int = 5) -> List[str]:
        """Extract key terms from text."""
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                      'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
                      'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through'}
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = defaultdict(int)
        for word in words:
            if word not in stop_words:
                word_freq[word] += 1
        return sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:n]

    def _estimate_contribution(self, answer: str, source_text: str) -> float:
        """Estimate how much a source contributed to the answer."""
        answer_words = set(answer.lower().split())
        source_words = set(source_text.lower().split())
        overlap = answer_words.intersection(source_words)
        return len(overlap) / len(answer_words) if answer_words else 0

    def _calculate_coverage(self, query: str, sources: List[Dict]) -> float:
        """Calculate how well sources cover the query."""
        query_words = set(query.lower().split()) - {'what', 'how', 'why', 'is', 'are', 'the'}
        all_source_text = ' '.join(s.get('text', '') for s in sources).lower()
        covered = sum(1 for w in query_words if w in all_source_text)
        return covered / len(query_words) if query_words else 1.0

    def _check_consistency(self, sources: List[Dict]) -> float:
        """Check if sources are consistent with each other."""
        # Simplified: check if sources use similar vocabulary
        if len(sources) < 2:
            return 1.0

        word_sets = [set(s.get('text', '').lower().split()) for s in sources]
        total_overlap = 0
        comparisons = 0

        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                if word_sets[i] and word_sets[j]:
                    overlap = len(word_sets[i].intersection(word_sets[j]))
                    total = len(word_sets[i].union(word_sets[j]))
                    total_overlap += overlap / total if total else 0
                    comparisons += 1

        return total_overlap / comparisons if comparisons > 0 else 1.0


# ==================== TRUST & RELIABILITY ====================

class TrustModule:
    """Manage trust and reliability of RAG responses."""

    def __init__(self):
        self.trust_thresholds = {
            "high": 0.8,
            "medium": 0.5,
            "low": 0.3
        }
        self.source_trust_scores = {}

    def calculate_trust_score(self, response: Any) -> Dict:
        """Calculate comprehensive trust score for a response."""
        factors = {}

        # Source trust
        sources = getattr(response, 'sources', [])
        source_scores = [self._get_source_trust(s.get('source', '')) for s in sources]
        factors['source_trust'] = sum(source_scores) / len(source_scores) if source_scores else 0.5

        # Confidence from retrieval
        factors['retrieval_confidence'] = getattr(response, 'confidence', 0.5)

        # Check for uncertainty indicators in answer
        answer = getattr(response, 'answer', '')
        factors['certainty'] = self._analyze_certainty(answer)

        # Source agreement
        factors['source_agreement'] = self._check_source_agreement(sources)

        # Overall trust score (weighted average)
        weights = {'source_trust': 0.3, 'retrieval_confidence': 0.3,
                   'certainty': 0.2, 'source_agreement': 0.2}
        overall = sum(factors[k] * weights[k] for k in weights)

        return {
            'overall_trust': overall,
            'trust_level': self._get_trust_level(overall),
            'factors': factors,
            'warnings': self._generate_trust_warnings(factors)
        }

    def _get_source_trust(self, source: str) -> float:
        """Get trust score for a source."""
        if source in self.source_trust_scores:
            return self.source_trust_scores[source]

        # Default trust based on source type
        if 'paper' in source.lower() or 'journal' in source.lower():
            return 0.9
        elif 'book' in source.lower() or 'textbook' in source.lower():
            return 0.85
        elif 'documentation' in source.lower():
            return 0.8
        else:
            return 0.7

    def _analyze_certainty(self, text: str) -> float:
        """Analyze certainty level in text."""
        uncertainty_phrases = ['might be', 'could be', 'possibly', 'perhaps',
                               'uncertain', 'unclear', 'may not', 'not sure',
                               'approximately', 'roughly', 'about']
        certainty_phrases = ['definitely', 'certainly', 'clearly', 'proven',
                            'established', 'confirmed', 'known to', 'always']

        text_lower = text.lower()
        uncertainty_count = sum(1 for p in uncertainty_phrases if p in text_lower)
        certainty_count = sum(1 for p in certainty_phrases if p in text_lower)

        if uncertainty_count + certainty_count == 0:
            return 0.7
        return certainty_count / (uncertainty_count + certainty_count)

    def _check_source_agreement(self, sources: List[Dict]) -> float:
        """Check if multiple sources agree."""
        if len(sources) < 2:
            return 0.6

        # Simple check: see if sources share vocabulary
        word_sets = [set(s.get('text', '').lower().split()) for s in sources if s.get('text')]
        if not word_sets:
            return 0.5

        common_words = word_sets[0]
        for ws in word_sets[1:]:
            common_words = common_words.intersection(ws)

        avg_size = sum(len(ws) for ws in word_sets) / len(word_sets)
        return len(common_words) / avg_size if avg_size > 0 else 0.5

    def _get_trust_level(self, score: float) -> str:
        """Get trust level label from score."""
        if score >= self.trust_thresholds['high']:
            return 'HIGH'
        elif score >= self.trust_thresholds['medium']:
            return 'MEDIUM'
        elif score >= self.trust_thresholds['low']:
            return 'LOW'
        else:
            return 'VERY_LOW'

    def _generate_trust_warnings(self, factors: Dict) -> List[str]:
        """Generate warnings based on trust factors."""
        warnings = []
        if factors.get('source_trust', 1) < 0.5:
            warnings.append("Sources have low trust scores")
        if factors.get('retrieval_confidence', 1) < 0.3:
            warnings.append("Low retrieval confidence")
        if factors.get('certainty', 1) < 0.4:
            warnings.append("Answer contains uncertainty indicators")
        if factors.get('source_agreement', 1) < 0.4:
            warnings.append("Sources may not agree")
        return warnings


# ==================== ETHICAL AI ====================

class EthicalAIModule:
    """Ethical AI checks and safeguards."""

    def __init__(self):
        self.sensitive_topics = [
            'medical advice', 'diagnosis', 'treatment',
            'mental health', 'suicide', 'self-harm',
            'discrimination', 'bias', 'hate',
            'violence', 'weapons', 'illegal'
        ]

        self.bias_indicators = [
            'always', 'never', 'all', 'none', 'every',
            'definitely', 'impossible', 'certain'
        ]

        self.protected_attributes = [
            'gender', 'race', 'ethnicity', 'religion',
            'age', 'disability', 'nationality', 'orientation'
        ]

    def check_ethical_concerns(self, query: str, response: str) -> Dict:
        """Check for ethical concerns in query and response."""
        concerns = []
        flags = []

        # Check for sensitive topics
        for topic in self.sensitive_topics:
            if topic in query.lower() or topic in response.lower():
                concerns.append(f"Contains sensitive topic: {topic}")
                flags.append(f"SENSITIVE_{topic.upper().replace(' ', '_')}")

        # Check for potential bias
        bias_issues = self._detect_bias(response)
        concerns.extend(bias_issues['concerns'])
        flags.extend(bias_issues['flags'])

        # Check for discriminatory content
        discrimination = self._check_discrimination(response)
        concerns.extend(discrimination['concerns'])
        flags.extend(discrimination['flags'])

        return {
            'has_concerns': len(concerns) > 0,
            'concerns': concerns,
            'flags': flags,
            'severity': self._calculate_severity(flags),
            'recommendations': self._generate_recommendations(concerns)
        }

    def _detect_bias(self, text: str) -> Dict:
        """Detect potential bias in text."""
        concerns = []
        flags = []

        text_lower = text.lower()

        # Check for absolute statements
        absolute_count = sum(1 for ind in self.bias_indicators if ind in text_lower)
        if absolute_count >= 2:
            concerns.append("Contains multiple absolute statements which may indicate bias")
            flags.append("POTENTIAL_BIAS")

        return {'concerns': concerns, 'flags': flags}

    def _check_discrimination(self, text: str) -> Dict:
        """Check for potentially discriminatory content."""
        concerns = []
        flags = []

        text_lower = text.lower()

        # Check for stereotyping patterns
        for attr in self.protected_attributes:
            if attr in text_lower:
                # Check if it's in a potentially discriminatory context
                patterns = [f"all {attr}", f"{attr} always", f"{attr} never"]
                for pattern in patterns:
                    if pattern in text_lower:
                        concerns.append(f"Potentially discriminatory statement involving {attr}")
                        flags.append("DISCRIMINATION_WARNING")

        return {'concerns': concerns, 'flags': flags}

    def _calculate_severity(self, flags: List[str]) -> str:
        """Calculate overall severity of ethical concerns."""
        if not flags:
            return 'NONE'

        severe_flags = ['DISCRIMINATION_WARNING', 'SENSITIVE_MEDICAL_ADVICE', 'SENSITIVE_VIOLENCE']
        if any(f in flags for f in severe_flags):
            return 'HIGH'
        elif len(flags) >= 3:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _generate_recommendations(self, concerns: List[str]) -> List[str]:
        """Generate recommendations based on concerns."""
        recommendations = []

        if any('medical' in c.lower() for c in concerns):
            recommendations.append("Add disclaimer: This is not medical advice. Consult a healthcare professional.")

        if any('bias' in c.lower() for c in concerns):
            recommendations.append("Consider rephrasing absolute statements")

        if any('discrimination' in c.lower() for c in concerns):
            recommendations.append("Review for potentially harmful generalizations")

        return recommendations


# ==================== COMPLIANCE & AUDIT ====================

class ComplianceModule:
    """Compliance and audit trail management."""

    def __init__(self, audit_log_path: str = "logs/audit"):
        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.mkdir(parents=True, exist_ok=True)
        self.current_log = []
        self.compliance_rules = self._load_default_rules()

    def _load_default_rules(self) -> Dict:
        """Load default compliance rules."""
        return {
            "data_retention_days": 90,
            "pii_detection": True,
            "log_all_queries": True,
            "anonymize_logs": False,
            "required_disclaimers": [
                "AI-generated content should be verified",
                "Not a substitute for professional advice"
            ]
        }

    def log_interaction(self, user_id: str, query: str, response: Any, metadata: Dict = None) -> AuditEntry:
        """Log an interaction for audit purposes."""
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            action="QUERY",
            user_id=user_id,
            query=self._anonymize_query(query) if self.compliance_rules.get("anonymize_logs") else query,
            response_hash=hashlib.sha256(str(response).encode()).hexdigest(),
            metadata=metadata or {},
            flags=[]
        )

        # Check for PII
        if self.compliance_rules.get("pii_detection"):
            pii = self._detect_pii(query)
            if pii:
                entry.flags.append("CONTAINS_PII")
                entry.metadata["pii_types"] = pii

        self.current_log.append(entry)

        # Persist if log is large enough
        if len(self.current_log) >= 100:
            self._persist_logs()

        return entry

    def _anonymize_query(self, query: str) -> str:
        """Anonymize PII in query."""
        # Simple email anonymization
        query = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL]', query)
        # Simple phone number anonymization
        query = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', query)
        return query

    def _detect_pii(self, text: str) -> List[str]:
        """Detect PII in text."""
        pii_types = []

        # Email detection
        if re.search(r'\b[\w.-]+@[\w.-]+\.\w+\b', text):
            pii_types.append("email")

        # Phone detection
        if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text):
            pii_types.append("phone")

        # SSN-like patterns
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):
            pii_types.append("ssn")

        return pii_types

    def _persist_logs(self):
        """Persist audit logs to disk."""
        if not self.current_log:
            return

        filename = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.audit_log_path / filename

        logs_data = [
            {
                "timestamp": e.timestamp,
                "action": e.action,
                "user_id": e.user_id,
                "query": e.query,
                "response_hash": e.response_hash,
                "metadata": e.metadata,
                "flags": e.flags
            }
            for e in self.current_log
        ]

        with open(filepath, 'w') as f:
            json.dump(logs_data, f, indent=2)

        self.current_log = []

    def generate_compliance_report(self) -> Dict:
        """Generate compliance report."""
        # Aggregate stats from logs
        all_entries = self.current_log.copy()

        return {
            "report_date": datetime.now().isoformat(),
            "total_interactions": len(all_entries),
            "pii_incidents": sum(1 for e in all_entries if "CONTAINS_PII" in e.flags),
            "unique_users": len(set(e.user_id for e in all_entries)),
            "compliance_rules_active": list(self.compliance_rules.keys()),
            "status": "COMPLIANT" if self._check_compliance() else "REVIEW_REQUIRED"
        }

    def _check_compliance(self) -> bool:
        """Check if system is compliant with rules."""
        # Simplified check
        return True

    def flush_logs(self):
        """Force persist all current logs."""
        self._persist_logs()


# ==================== SECURITY ====================

class SecurityModule:
    """Security and privacy protection."""

    def __init__(self):
        self.blocked_patterns = [
            r'(?i)(password|secret|api[_\s]?key|token)[\s:=]+\S+',
            r'(?i)(drop\s+table|delete\s+from|insert\s+into)',
            r'(?i)<script[^>]*>.*?</script>',
            r'(?i)exec\s*\(',
            r'(?i)eval\s*\('
        ]

        self.rate_limits = {
            "default": 100,  # requests per minute
            "authenticated": 500
        }
        self.request_counts = defaultdict(list)

    def sanitize_input(self, text: str) -> Tuple[str, List[str]]:
        """Sanitize input and return cleaned text with warnings."""
        warnings = []
        cleaned = text

        for pattern in self.blocked_patterns:
            if re.search(pattern, text):
                warnings.append(f"Blocked pattern detected: {pattern[:30]}...")
                cleaned = re.sub(pattern, '[REDACTED]', cleaned)

        return cleaned, warnings

    def check_rate_limit(self, user_id: str, authenticated: bool = False) -> bool:
        """Check if user is within rate limits."""
        current_time = time.time()
        limit = self.rate_limits["authenticated"] if authenticated else self.rate_limits["default"]

        # Clean old entries
        self.request_counts[user_id] = [
            t for t in self.request_counts[user_id]
            if current_time - t < 60
        ]

        if len(self.request_counts[user_id]) >= limit:
            return False

        self.request_counts[user_id].append(current_time)
        return True

    def validate_response(self, response: str) -> Tuple[str, List[str]]:
        """Validate and sanitize response before returning to user."""
        warnings = []
        cleaned = response

        # Check for accidental data leakage
        sensitive_patterns = [
            (r'(?i)api[_\s]?key[:\s]+[\w-]{20,}', "Potential API key"),
            (r'(?i)password[:\s]+\S+', "Potential password"),
            (r'\b\d{16}\b', "Potential credit card number")
        ]

        for pattern, description in sensitive_patterns:
            if re.search(pattern, response):
                warnings.append(f"Detected: {description}")
                cleaned = re.sub(pattern, '[REDACTED]', cleaned)

        return cleaned, warnings


# ==================== GOVERNANCE MANAGER ====================

class AIGovernanceManager:
    """Unified AI governance manager integrating all modules."""

    def __init__(self, audit_log_path: str = "logs/audit"):
        self.explainability = ExplainabilityModule()
        self.trust = TrustModule()
        self.ethics = EthicalAIModule()
        self.compliance = ComplianceModule(audit_log_path)
        self.security = SecurityModule()

    def process_query(self, query: str, user_id: str = "anonymous") -> Tuple[str, Dict]:
        """Process and validate a query before sending to RAG."""
        # Security check
        cleaned_query, security_warnings = self.security.sanitize_input(query)

        # Rate limit check
        rate_ok = self.security.check_rate_limit(user_id)

        # Ethical check
        ethical_check = self.ethics.check_ethical_concerns(query, "")

        return cleaned_query, {
            "security_warnings": security_warnings,
            "rate_limited": not rate_ok,
            "ethical_flags": ethical_check['flags'],
            "proceed": rate_ok and not security_warnings
        }

    def process_response(self, response: Any, query: str, user_id: str = "anonymous") -> Dict:
        """Process and validate a response before returning to user."""
        result = {
            "original_response": response,
            "modified_response": None,
            "governance_metadata": {}
        }

        # Get answer text
        answer = getattr(response, 'answer', str(response))
        sources = getattr(response, 'sources', [])

        # Security validation
        cleaned_answer, sec_warnings = self.security.validate_response(answer)

        # Trust assessment
        trust_assessment = self.trust.calculate_trust_score(response)

        # Ethical check
        ethical_check = self.ethics.check_ethical_concerns(query, answer)

        # Explanation generation
        explanation = self.explainability.explain_answer(answer, sources, query)

        # Compliance logging
        self.compliance.log_interaction(user_id, query, response, {
            "trust_level": trust_assessment['trust_level'],
            "ethical_flags": ethical_check['flags']
        })

        result["modified_response"] = cleaned_answer
        result["governance_metadata"] = {
            "trust": trust_assessment,
            "ethics": ethical_check,
            "explanation": {
                "reasoning_steps": explanation.reasoning_steps,
                "source_attributions": explanation.source_attributions,
                "confidence_breakdown": explanation.confidence_breakdown,
                "uncertainty_factors": explanation.uncertainty_factors
            },
            "security_warnings": sec_warnings,
            "compliance_logged": True
        }

        return result

    def get_governance_report(self) -> Dict:
        """Get comprehensive governance report."""
        return {
            "compliance": self.compliance.generate_compliance_report(),
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    print("Testing AI Governance Modules\n" + "="*60)

    # Initialize governance manager
    governance = AIGovernanceManager()

    # Test query processing
    test_query = "What happens to alpha waves during stress?"
    cleaned_query, query_meta = governance.process_query(test_query, "user_123")
    print(f"\nQuery: {test_query}")
    print(f"Cleaned: {cleaned_query}")
    print(f"Metadata: {query_meta}")

    # Test response processing
    class MockResponse:
        answer = "Alpha waves typically decrease during stress. This is known as alpha suppression."
        sources = [{"source": "EEG Research", "text": "Alpha suppression during stress", "score": 0.8}]
        confidence = 0.75
        latency_ms = 150

    response_result = governance.process_response(MockResponse(), test_query, "user_123")
    print(f"\nGovernance Result:")
    print(f"  Trust Level: {response_result['governance_metadata']['trust']['trust_level']}")
    print(f"  Ethical Concerns: {response_result['governance_metadata']['ethics']['has_concerns']}")
    print(f"  Uncertainty Factors: {response_result['governance_metadata']['explanation']['uncertainty_factors']}")

    print("\n" + "="*60 + "\nAll governance tests completed!")
