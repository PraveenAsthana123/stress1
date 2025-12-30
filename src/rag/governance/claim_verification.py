"""
A7) Claim-to-Evidence Verification for EEG-RAG

Comprehensive module for:
- Claim segmentation and extraction
- Evidence requirement definitions
- Structured citation format
- Support checking (lexical + semantic)
- Numeric and unit verification
- Table-aware evidence linking
- Contradiction detection
- Fallback behaviors

This blocks "fluent answers with citations" that are actually unsupported.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import re
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Claim verification status."""
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


class FallbackAction(Enum):
    """Actions when verification fails."""
    REGENERATE = "regenerate"
    RE_RETRIEVE = "re_retrieve"
    ABSTAIN = "abstain"
    ASK_CLARIFICATION = "ask_clarification"
    PROCEED_WITH_CAVEAT = "proceed_with_caveat"


@dataclass
class Claim:
    """A single verifiable claim."""
    claim_id: str
    text: str
    source_sentence: str
    claim_type: str  # 'factual', 'numeric', 'procedural', 'definitional'
    entities: List[str] = field(default_factory=list)
    numbers: List[Tuple[float, str]] = field(default_factory=list)  # (value, unit)


@dataclass
class Citation:
    """Structured citation linking claim to evidence."""
    claim_id: str
    chunk_id: str
    span_start: int
    span_end: int
    quote: str
    confidence: float


@dataclass
class VerificationResult:
    """Result of claim verification."""
    claim: Claim
    status: VerificationStatus
    supporting_citations: List[Citation]
    confidence: float
    verification_details: Dict[str, Any]
    failure_reason: Optional[str] = None


@dataclass
class VerificationReport:
    """Complete verification report for an answer."""
    answer_id: str
    claims: List[Claim]
    verifications: List[VerificationResult]
    overall_status: VerificationStatus
    supported_ratio: float
    recommended_action: FallbackAction
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ClaimExtractor:
    """Extract verifiable claims from answer text."""

    # Patterns for claim extraction
    CLAIM_PATTERNS = [
        (r'([A-Z][^.!?]*(?:is|are|was|were|has|have|can|should|must)[^.!?]*[.!?])', 'factual'),
        (r'([A-Z][^.!?]*\d+(?:\.\d+)?\s*(?:Hz|µV|ms|kHz|mV)[^.!?]*[.!?])', 'numeric'),
        (r'([A-Z][^.!?]*(?:first|second|then|finally|step)[^.!?]*[.!?])', 'procedural'),
        (r'([A-Z][^.!?]*(?:defined as|refers to|means|is called)[^.!?]*[.!?])', 'definitional'),
    ]

    # EEG domain entities for extraction
    EEG_ENTITIES = {
        'alpha', 'beta', 'theta', 'delta', 'gamma',
        'electrode', 'montage', 'artifact', 'filter',
        'bandpass', 'notch', 'reference', 'impedance',
        'eeg', 'eog', 'emg', 'ecg',
    }

    def extract_claims(self, text: str) -> List[Claim]:
        """Extract claims from text."""
        claims = []
        claim_id = 0

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            claim_type = self._classify_claim(sentence)
            if claim_type:
                claim = Claim(
                    claim_id=f"claim_{claim_id}",
                    text=sentence.strip(),
                    source_sentence=sentence,
                    claim_type=claim_type,
                    entities=self._extract_entities(sentence),
                    numbers=self._extract_numbers(sentence),
                )
                claims.append(claim)
                claim_id += 1

        return claims

    def _classify_claim(self, sentence: str) -> Optional[str]:
        """Classify claim type."""
        sentence_lower = sentence.lower()

        # Check for numeric content
        if re.search(r'\d+(?:\.\d+)?\s*(?:Hz|µV|ms|kHz|mV)', sentence):
            return 'numeric'

        # Check for definitional
        if any(p in sentence_lower for p in ['is defined as', 'refers to', 'means']):
            return 'definitional'

        # Check for procedural
        if any(p in sentence_lower for p in ['first', 'then', 'step', 'finally']):
            return 'procedural'

        # Default to factual if contains verb
        if re.search(r'\b(?:is|are|was|were|has|have)\b', sentence_lower):
            return 'factual'

        return None

    def _extract_entities(self, text: str) -> List[str]:
        """Extract EEG domain entities."""
        text_lower = text.lower()
        found = []
        for entity in self.EEG_ENTITIES:
            if entity in text_lower:
                found.append(entity)
        return found

    def _extract_numbers(self, text: str) -> List[Tuple[float, str]]:
        """Extract numbers with units."""
        pattern = r'(\d+(?:\.\d+)?)\s*(Hz|µV|ms|kHz|mV|%)'
        matches = re.findall(pattern, text)
        return [(float(m[0]), m[1]) for m in matches]


class EvidenceRequirements:
    """Define what counts as sufficient evidence."""

    def __init__(
        self,
        min_sources: int = 1,
        require_span_match: bool = True,
        numeric_tolerance: float = 0.05
    ):
        self.min_sources = min_sources
        self.require_span_match = require_span_match
        self.numeric_tolerance = numeric_tolerance

    def is_sufficient(
        self,
        claim: Claim,
        citations: List[Citation]
    ) -> Tuple[bool, str]:
        """Check if evidence is sufficient for claim."""
        if len(citations) < self.min_sources:
            return False, f"Insufficient sources: {len(citations)} < {self.min_sources}"

        if self.require_span_match:
            has_span = any(c.span_start >= 0 for c in citations)
            if not has_span:
                return False, "No span-level evidence"

        # Numeric claims require exact match
        if claim.claim_type == 'numeric' and claim.numbers:
            has_numeric_match = any(c.confidence >= 0.9 for c in citations)
            if not has_numeric_match:
                return False, "Numeric claim lacks high-confidence match"

        return True, "Evidence sufficient"


class LexicalSupportChecker:
    """Fast lexical support checking."""

    def __init__(self, min_overlap: float = 0.3):
        self.min_overlap = min_overlap

    def check_support(
        self,
        claim: Claim,
        chunk_text: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Check lexical support for claim."""
        claim_words = set(claim.text.lower().split())
        chunk_words = set(chunk_text.lower().split())

        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'to', 'of', 'and', 'or', 'in', 'on', 'at', 'for', 'with'}
        claim_words -= stopwords
        chunk_words -= stopwords

        if not claim_words:
            return 0.0, {'reason': 'No content words in claim'}

        overlap = len(claim_words & chunk_words)
        overlap_ratio = overlap / len(claim_words)

        # Entity matching boost
        entity_matches = sum(1 for e in claim.entities if e in chunk_text.lower())
        entity_boost = 0.1 * entity_matches

        score = min(1.0, overlap_ratio + entity_boost)

        return score, {
            'overlap_ratio': overlap_ratio,
            'matched_words': list(claim_words & chunk_words),
            'entity_matches': entity_matches,
        }


class NumericVerifier:
    """Verify numeric claims against evidence."""

    # EEG domain numeric patterns
    PATTERNS = {
        'frequency': r'(\d+(?:\.\d+)?)\s*(?:Hz|kHz)',
        'amplitude': r'(\d+(?:\.\d+)?)\s*(?:µV|mV)',
        'time': r'(\d+(?:\.\d+)?)\s*(?:ms|s)',
        'percentage': r'(\d+(?:\.\d+)?)\s*%',
    }

    def __init__(self, tolerance: float = 0.05):
        self.tolerance = tolerance

    def verify(
        self,
        claim: Claim,
        chunk_text: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verify numeric values in claim against chunk."""
        if not claim.numbers:
            return True, {'reason': 'No numbers to verify'}

        chunk_numbers = self._extract_numbers(chunk_text)

        matches = []
        mismatches = []

        for claim_val, claim_unit in claim.numbers:
            matched = False
            for chunk_val, chunk_unit in chunk_numbers:
                if claim_unit == chunk_unit:
                    if abs(claim_val - chunk_val) <= claim_val * self.tolerance:
                        matches.append((claim_val, chunk_val, claim_unit))
                        matched = True
                        break
            if not matched:
                mismatches.append((claim_val, claim_unit))

        if mismatches:
            return False, {
                'matches': matches,
                'mismatches': mismatches,
                'reason': f"Numeric mismatch: {mismatches}",
            }

        return True, {
            'matches': matches,
            'reason': 'All numbers verified',
        }

    def _extract_numbers(self, text: str) -> List[Tuple[float, str]]:
        """Extract numbers with units from text."""
        results = []
        for pattern_name, pattern in self.PATTERNS.items():
            for match in re.finditer(pattern, text):
                value = float(match.group(1))
                unit = match.group(0).replace(match.group(1), '').strip()
                results.append((value, unit))
        return results


class SemanticEntailmentChecker:
    """Semantic entailment checking for claim support."""

    def __init__(self, model_name: str = "default"):
        self.model_name = model_name
        # Placeholder for NLI model

    def check_entailment(
        self,
        claim: str,
        evidence: str
    ) -> Tuple[str, float]:
        """
        Check if evidence entails claim.

        Returns: (label, confidence)
        - label: 'entailment', 'neutral', 'contradiction'
        """
        # Simplified heuristic (would use NLI model in practice)
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()

        # Check for contradiction signals
        if ('not' in claim_lower) != ('not' in evidence_lower):
            # Potential negation mismatch
            return 'contradiction', 0.7

        # Check for keyword overlap
        claim_words = set(claim_lower.split())
        evidence_words = set(evidence_lower.split())
        overlap = len(claim_words & evidence_words) / max(1, len(claim_words))

        if overlap > 0.5:
            return 'entailment', overlap
        elif overlap > 0.2:
            return 'neutral', overlap
        else:
            return 'neutral', 0.3


class TableEvidenceLinker:
    """Link claims to table-based evidence."""

    def find_table_evidence(
        self,
        claim: Claim,
        tables: List[Dict[str, Any]]
    ) -> List[Citation]:
        """Find evidence for claim in tables."""
        citations = []

        for table in tables:
            table_id = table.get('table_id', '')
            rows = table.get('rows', [])
            headers = table.get('headers', [])

            for row_idx, row in enumerate(rows):
                # Check if row contains claim entities or numbers
                row_text = ' '.join(str(cell) for cell in row)

                # Entity match
                entity_match = any(e in row_text.lower() for e in claim.entities)

                # Number match
                number_match = False
                for claim_val, claim_unit in claim.numbers:
                    if str(claim_val) in row_text and claim_unit in row_text:
                        number_match = True
                        break

                if entity_match or number_match:
                    citations.append(Citation(
                        claim_id=claim.claim_id,
                        chunk_id=f"{table_id}_row_{row_idx}",
                        span_start=0,
                        span_end=len(row_text),
                        quote=row_text[:200],
                        confidence=0.8 if number_match else 0.6,
                    ))

        return citations


class ClaimVerifier:
    """Main claim verification engine."""

    def __init__(
        self,
        min_sources: int = 1,
        numeric_tolerance: float = 0.05,
        min_support_score: float = 0.5
    ):
        self.requirements = EvidenceRequirements(
            min_sources=min_sources,
            numeric_tolerance=numeric_tolerance,
        )
        self.lexical_checker = LexicalSupportChecker()
        self.numeric_verifier = NumericVerifier(tolerance=numeric_tolerance)
        self.semantic_checker = SemanticEntailmentChecker()
        self.table_linker = TableEvidenceLinker()
        self.min_support_score = min_support_score

    def verify_claim(
        self,
        claim: Claim,
        chunks: List[Dict[str, Any]],
        tables: Optional[List[Dict[str, Any]]] = None
    ) -> VerificationResult:
        """Verify a single claim against evidence."""
        supporting_citations = []
        verification_details = {}

        for chunk in chunks:
            chunk_id = chunk.get('chunk_id', '')
            chunk_text = chunk.get('text', '')

            # Lexical check
            lex_score, lex_details = self.lexical_checker.check_support(claim, chunk_text)

            if lex_score < self.min_support_score:
                continue

            # Numeric verification (if applicable)
            numeric_ok, numeric_details = True, {}
            if claim.claim_type == 'numeric':
                numeric_ok, numeric_details = self.numeric_verifier.verify(claim, chunk_text)

            # Semantic check
            sem_label, sem_conf = self.semantic_checker.check_entailment(claim.text, chunk_text)

            if sem_label == 'contradiction':
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.CONTRADICTED,
                    supporting_citations=[],
                    confidence=sem_conf,
                    verification_details={
                        'contradiction_source': chunk_id,
                        'semantic': {'label': sem_label, 'confidence': sem_conf},
                    },
                    failure_reason=f"Contradicted by {chunk_id}",
                )

            if lex_score >= self.min_support_score and numeric_ok:
                # Find best matching span
                span_start, span_end, quote = self._find_best_span(claim.text, chunk_text)

                citations = Citation(
                    claim_id=claim.claim_id,
                    chunk_id=chunk_id,
                    span_start=span_start,
                    span_end=span_end,
                    quote=quote,
                    confidence=lex_score * (0.9 if numeric_ok else 0.7),
                )
                supporting_citations.append(citations)

                verification_details[chunk_id] = {
                    'lexical': lex_details,
                    'numeric': numeric_details,
                    'semantic': {'label': sem_label, 'confidence': sem_conf},
                }

        # Check table evidence
        if tables:
            table_citations = self.table_linker.find_table_evidence(claim, tables)
            supporting_citations.extend(table_citations)

        # Determine status
        is_sufficient, reason = self.requirements.is_sufficient(claim, supporting_citations)

        if is_sufficient:
            status = VerificationStatus.SUPPORTED
            confidence = np.mean([c.confidence for c in supporting_citations])
        elif supporting_citations:
            status = VerificationStatus.PARTIALLY_SUPPORTED
            confidence = np.mean([c.confidence for c in supporting_citations]) * 0.7
        else:
            status = VerificationStatus.UNSUPPORTED
            confidence = 0.0

        return VerificationResult(
            claim=claim,
            status=status,
            supporting_citations=supporting_citations,
            confidence=confidence,
            verification_details=verification_details,
            failure_reason=None if is_sufficient else reason,
        )

    def _find_best_span(
        self,
        claim: str,
        chunk: str,
        context_window: int = 50
    ) -> Tuple[int, int, str]:
        """Find best matching span in chunk for claim."""
        claim_words = claim.lower().split()[:5]  # First 5 words
        chunk_lower = chunk.lower()

        best_pos = -1
        for word in claim_words:
            pos = chunk_lower.find(word)
            if pos >= 0:
                if best_pos < 0 or pos < best_pos:
                    best_pos = pos

        if best_pos >= 0:
            start = max(0, best_pos - context_window)
            end = min(len(chunk), best_pos + context_window)
            return start, end, chunk[start:end]

        return -1, -1, ""


class VerificationPipeline:
    """Complete verification pipeline for RAG answers."""

    def __init__(
        self,
        verifier: Optional[ClaimVerifier] = None,
        fallback_threshold: float = 0.8
    ):
        self.claim_extractor = ClaimExtractor()
        self.verifier = verifier or ClaimVerifier()
        self.fallback_threshold = fallback_threshold

    def verify_answer(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
        tables: Optional[List[Dict[str, Any]]] = None,
        answer_id: str = "ans_001"
    ) -> VerificationReport:
        """Verify complete answer."""
        # Extract claims
        claims = self.claim_extractor.extract_claims(answer)

        if not claims:
            return VerificationReport(
                answer_id=answer_id,
                claims=[],
                verifications=[],
                overall_status=VerificationStatus.UNVERIFIABLE,
                supported_ratio=1.0,
                recommended_action=FallbackAction.PROCEED_WITH_CAVEAT,
            )

        # Verify each claim
        verifications = []
        for claim in claims:
            result = self.verifier.verify_claim(claim, chunks, tables)
            verifications.append(result)

        # Calculate overall status
        supported = sum(1 for v in verifications if v.status == VerificationStatus.SUPPORTED)
        partial = sum(1 for v in verifications if v.status == VerificationStatus.PARTIALLY_SUPPORTED)
        contradicted = sum(1 for v in verifications if v.status == VerificationStatus.CONTRADICTED)

        supported_ratio = (supported + 0.5 * partial) / len(verifications)

        if contradicted > 0:
            overall_status = VerificationStatus.CONTRADICTED
            recommended_action = FallbackAction.REGENERATE
        elif supported_ratio >= self.fallback_threshold:
            overall_status = VerificationStatus.SUPPORTED
            recommended_action = FallbackAction.PROCEED_WITH_CAVEAT
        elif supported_ratio >= 0.5:
            overall_status = VerificationStatus.PARTIALLY_SUPPORTED
            recommended_action = FallbackAction.PROCEED_WITH_CAVEAT
        else:
            overall_status = VerificationStatus.UNSUPPORTED
            recommended_action = FallbackAction.RE_RETRIEVE

        return VerificationReport(
            answer_id=answer_id,
            claims=claims,
            verifications=verifications,
            overall_status=overall_status,
            supported_ratio=supported_ratio,
            recommended_action=recommended_action,
        )


class VerificationMonitor:
    """Monitor verification metrics."""

    def __init__(self):
        self.metrics = {
            'total_claims': 0,
            'supported': 0,
            'unsupported': 0,
            'contradicted': 0,
            'regeneration_rate': 0.0,
        }
        self.history: List[Dict[str, Any]] = []

    def log_report(self, report: VerificationReport):
        """Log verification report."""
        self.metrics['total_claims'] += len(report.claims)

        for v in report.verifications:
            if v.status == VerificationStatus.SUPPORTED:
                self.metrics['supported'] += 1
            elif v.status == VerificationStatus.UNSUPPORTED:
                self.metrics['unsupported'] += 1
            elif v.status == VerificationStatus.CONTRADICTED:
                self.metrics['contradicted'] += 1

        self.history.append({
            'answer_id': report.answer_id,
            'timestamp': report.timestamp,
            'supported_ratio': report.supported_ratio,
            'action': report.recommended_action.value,
        })

    def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics."""
        total = self.metrics['total_claims']
        if total > 0:
            self.metrics['support_rate'] = self.metrics['supported'] / total
            self.metrics['unsupported_rate'] = self.metrics['unsupported'] / total

        regens = sum(1 for h in self.history if h['action'] == 'regenerate')
        if self.history:
            self.metrics['regeneration_rate'] = regens / len(self.history)

        return self.metrics


if __name__ == '__main__':
    # Demo usage
    pipeline = VerificationPipeline()

    answer = """
    Alpha waves are rhythmic brain oscillations in the frequency range of 8-13 Hz.
    They are typically observed when a person is relaxed with eyes closed.
    The electrode placement should follow the 10-20 system for proper recording.
    A bandpass filter of 0.5-45 Hz is recommended for preprocessing.
    """

    chunks = [
        {
            'chunk_id': 'chunk_001',
            'text': 'Alpha rhythm refers to brain oscillations between 8 and 13 Hz, commonly seen in relaxed, awake individuals with closed eyes.',
        },
        {
            'chunk_id': 'chunk_002',
            'text': 'The international 10-20 electrode placement system is the standard for EEG recording.',
        },
        {
            'chunk_id': 'chunk_003',
            'text': 'Recommended preprocessing includes bandpass filtering from 0.5 to 45 Hz.',
        },
    ]

    report = pipeline.verify_answer(answer, chunks)

    print(f"Overall status: {report.overall_status.value}")
    print(f"Supported ratio: {report.supported_ratio:.2%}")
    print(f"Recommended action: {report.recommended_action.value}")
    print(f"\nClaims verified: {len(report.claims)}")

    for v in report.verifications:
        print(f"  - {v.claim.text[:50]}... : {v.status.value} ({v.confidence:.2f})")
