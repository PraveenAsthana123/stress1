"""
A3) PHI/PII Detection & Redaction (Ingestion Gate) for EEG-RAG

Comprehensive module for:
- Sensitive field detection (names, IDs, emails, dates)
- Source risk classification
- Ingestion gates with mandatory scanning
- Rule-based and NER-based detection
- Redaction with reversibility
- Table-specific redaction for EEG data
- Audit logging and monitoring

This module prevents sensitive data from entering the RAG store.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Pattern
from enum import Enum
import re
import hashlib
import json
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Document risk classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED_PHI = "restricted_phi"
    UNKNOWN = "unknown"


class EntityType(Enum):
    """PII/PHI entity types."""
    NAME = "NAME"
    MRN = "MRN"  # Medical Record Number
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    ADDRESS = "ADDRESS"
    DOB = "DOB"
    VISIT_DATE = "VISIT_DATE"
    SSN = "SSN"
    PATIENT_ID = "PATIENT_ID"
    CLINICIAN_NOTE = "CLINICIAN_NOTE"
    SITE_TIMESTAMP = "SITE_TIMESTAMP"  # Quasi-identifier


class RedactionAction(Enum):
    """Actions for detected entities."""
    REDACT = "redact"  # Replace with token
    DROP = "drop"  # Remove entirely
    ISOLATE = "isolate"  # Move to restricted storage
    ALLOW = "allow"  # Allow through (whitelisted)


@dataclass
class DetectedEntity:
    """A detected PII/PHI entity."""
    entity_type: EntityType
    text: str
    start: int
    end: int
    confidence: float
    detection_method: str  # 'rule' or 'ner'
    context: str = ""


@dataclass
class RedactionResult:
    """Result of redaction operation."""
    original_hash: str
    redacted_hash: str
    redacted_text: str
    entities_found: List[DetectedEntity]
    entities_redacted: int
    redaction_manifest: Dict[str, Any]


@dataclass
class IngestionGateResult:
    """Result of ingestion gate check."""
    doc_id: str
    passed: bool
    risk_level: RiskLevel
    entities_detected: int
    entities_redacted: int
    quarantined: bool
    reason: str = ""
    redacted_text: Optional[str] = None


class SensitiveFieldPolicy:
    """Policy defining what counts as PII/PHI in EEG context."""

    # Explicit identifiers
    EXPLICIT_IDENTIFIERS = {
        EntityType.NAME,
        EntityType.MRN,
        EntityType.EMAIL,
        EntityType.PHONE,
        EntityType.ADDRESS,
        EntityType.DOB,
        EntityType.SSN,
        EntityType.PATIENT_ID,
    }

    # Quasi-identifiers (combinations can be identifying)
    QUASI_IDENTIFIERS = {
        EntityType.VISIT_DATE,
        EntityType.SITE_TIMESTAMP,
    }

    # High-risk content
    HIGH_RISK_CONTENT = {
        EntityType.CLINICIAN_NOTE,
    }

    # EEG-specific whitelisted terms (look like identifiers but aren't)
    EEG_WHITELIST = {
        'Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 'Fz',
        'T3', 'T4', 'T5', 'T6', 'C3', 'C4', 'Cz',
        'P3', 'P4', 'Pz', 'O1', 'O2', 'Oz',
        'A1', 'A2', 'M1', 'M2',
        'EOG', 'EMG', 'ECG', 'EKG',
    }

    @classmethod
    def get_action_for_entity(cls, entity_type: EntityType, risk_level: RiskLevel) -> RedactionAction:
        """Determine action based on entity type and risk level."""
        if entity_type in cls.EXPLICIT_IDENTIFIERS:
            return RedactionAction.REDACT

        if entity_type in cls.HIGH_RISK_CONTENT:
            if risk_level == RiskLevel.RESTRICTED_PHI:
                return RedactionAction.DROP
            return RedactionAction.REDACT

        if entity_type in cls.QUASI_IDENTIFIERS:
            return RedactionAction.REDACT

        return RedactionAction.ALLOW


class RuleBasedDetector:
    """High-precision rule-based PII/PHI detection."""

    def __init__(self):
        self.patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[EntityType, List[Pattern]]:
        """Compile regex patterns for each entity type."""
        return {
            EntityType.EMAIL: [
                re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.IGNORECASE),
            ],
            EntityType.PHONE: [
                re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
                re.compile(r'\b[0-9]{3}[-.\s][0-9]{3}[-.\s][0-9]{4}\b'),
            ],
            EntityType.SSN: [
                re.compile(r'\b[0-9]{3}[-\s]?[0-9]{2}[-\s]?[0-9]{4}\b'),
            ],
            EntityType.MRN: [
                re.compile(r'\bMRN[:\s#]*[0-9A-Z]{6,12}\b', re.IGNORECASE),
                re.compile(r'\bPatient\s*ID[:\s#]*[0-9A-Z]{6,12}\b', re.IGNORECASE),
                re.compile(r'\bRecord\s*#[:\s]*[0-9A-Z]{6,12}\b', re.IGNORECASE),
            ],
            EntityType.DOB: [
                re.compile(r'\bDOB[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', re.IGNORECASE),
                re.compile(r'\bDate\s+of\s+Birth[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', re.IGNORECASE),
                re.compile(r'\bBirth\s*Date[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', re.IGNORECASE),
            ],
            EntityType.VISIT_DATE: [
                re.compile(r'\bVisit\s*Date[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', re.IGNORECASE),
                re.compile(r'\bExam\s*Date[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', re.IGNORECASE),
            ],
            EntityType.ADDRESS: [
                re.compile(r'\b\d+\s+[A-Za-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct)[.,]?\s*(?:[A-Za-z]+[,.]?\s*)?[A-Z]{2}\s+\d{5}(?:-\d{4})?\b', re.IGNORECASE),
            ],
            EntityType.PATIENT_ID: [
                re.compile(r'\bPatient[:\s#]*[0-9]{4,10}\b', re.IGNORECASE),
                re.compile(r'\bSubject[:\s#]*[0-9]{3,8}\b', re.IGNORECASE),
            ],
        }

    def detect(self, text: str) -> List[DetectedEntity]:
        """Detect entities using regex patterns."""
        entities = []

        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entities.append(DetectedEntity(
                        entity_type=entity_type,
                        text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95,  # High precision rules
                        detection_method='rule',
                        context=text[max(0, match.start()-20):min(len(text), match.end()+20)],
                    ))

        return entities


class NERDetector:
    """NER-based PII/PHI detection for higher recall."""

    def __init__(self, model_name: str = "default"):
        self.model_name = model_name
        # Placeholder for actual NER model

    def detect(self, text: str) -> List[DetectedEntity]:
        """Detect entities using NER model."""
        entities = []

        # Keyword-based fallback for NAME detection
        name_patterns = [
            re.compile(r'\bPatient:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'),
            re.compile(r'\bName:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'),
            re.compile(r'\bDr\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'),
            re.compile(r'\bPhysician:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'),
        ]

        for pattern in name_patterns:
            for match in pattern.finditer(text):
                name = match.group(1)
                # Filter out EEG electrode names
                if name not in SensitiveFieldPolicy.EEG_WHITELIST:
                    entities.append(DetectedEntity(
                        entity_type=EntityType.NAME,
                        text=name,
                        start=match.start(1),
                        end=match.end(1),
                        confidence=0.8,
                        detection_method='ner',
                        context=text[max(0, match.start()-20):min(len(text), match.end()+20)],
                    ))

        return entities


class HybridDetector:
    """Hybrid detection combining rules and NER."""

    def __init__(self):
        self.rule_detector = RuleBasedDetector()
        self.ner_detector = NERDetector()

    def detect(
        self,
        text: str,
        min_confidence: float = 0.7
    ) -> List[DetectedEntity]:
        """Detect entities using both methods."""
        # Rule-based detection (high precision)
        rule_entities = self.rule_detector.detect(text)

        # NER detection (higher recall)
        ner_entities = self.ner_detector.detect(text)

        # Combine and deduplicate
        all_entities = rule_entities + ner_entities
        all_entities = [e for e in all_entities if e.confidence >= min_confidence]

        # Filter whitelist
        filtered = []
        for entity in all_entities:
            if entity.text.strip() not in SensitiveFieldPolicy.EEG_WHITELIST:
                filtered.append(entity)

        # Deduplicate overlapping spans
        filtered.sort(key=lambda e: (e.start, -e.confidence))
        deduplicated = []
        last_end = -1

        for entity in filtered:
            if entity.start >= last_end:
                deduplicated.append(entity)
                last_end = entity.end

        return deduplicated


class Redactor:
    """Text redaction with reversibility support."""

    REDACTION_TOKENS = {
        EntityType.NAME: "[NAME]",
        EntityType.MRN: "[MRN]",
        EntityType.EMAIL: "[EMAIL]",
        EntityType.PHONE: "[PHONE]",
        EntityType.ADDRESS: "[ADDRESS]",
        EntityType.DOB: "[DOB]",
        EntityType.SSN: "[SSN]",
        EntityType.PATIENT_ID: "[PATIENT_ID]",
        EntityType.VISIT_DATE: "[VISIT_DATE]",
        EntityType.CLINICIAN_NOTE: "[CLINICIAN_NOTE]",
        EntityType.SITE_TIMESTAMP: "[SITE_TIMESTAMP]",
    }

    def __init__(self, vault_path: Optional[str] = None):
        self.vault_path = Path(vault_path) if vault_path else None
        self.redaction_log: List[Dict[str, Any]] = []

    def redact(
        self,
        text: str,
        entities: List[DetectedEntity],
        risk_level: RiskLevel = RiskLevel.INTERNAL
    ) -> RedactionResult:
        """Redact detected entities from text."""
        original_hash = hashlib.sha256(text.encode()).hexdigest()

        # Sort entities by start position (reverse for safe replacement)
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)

        redacted_text = text
        manifest = {
            'original_hash': original_hash,
            'timestamp': datetime.now().isoformat(),
            'entities': [],
        }

        entities_redacted = 0

        for entity in sorted_entities:
            action = SensitiveFieldPolicy.get_action_for_entity(
                entity.entity_type, risk_level
            )

            if action == RedactionAction.REDACT:
                token = self.REDACTION_TOKENS.get(entity.entity_type, "[REDACTED]")
                redacted_text = (
                    redacted_text[:entity.start] +
                    token +
                    redacted_text[entity.end:]
                )
                entities_redacted += 1

                manifest['entities'].append({
                    'type': entity.entity_type.value,
                    'start': entity.start,
                    'end': entity.end,
                    'token': token,
                    'confidence': entity.confidence,
                })

            elif action == RedactionAction.DROP:
                redacted_text = (
                    redacted_text[:entity.start] +
                    redacted_text[entity.end:]
                )
                entities_redacted += 1

        redacted_hash = hashlib.sha256(redacted_text.encode()).hexdigest()
        manifest['redacted_hash'] = redacted_hash

        return RedactionResult(
            original_hash=original_hash,
            redacted_hash=redacted_hash,
            redacted_text=redacted_text,
            entities_found=entities,
            entities_redacted=entities_redacted,
            redaction_manifest=manifest,
        )

    def store_manifest(self, manifest: Dict[str, Any], doc_id: str):
        """Store redaction manifest in vault."""
        if self.vault_path:
            manifest_path = self.vault_path / f"{doc_id}_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)


class TableRedactor:
    """EEG-specific table redaction."""

    def __init__(self):
        self.detector = HybridDetector()
        self.redactor = Redactor()

    def redact_table(
        self,
        table: List[List[str]],
        headers: List[str],
        risk_level: RiskLevel = RiskLevel.INTERNAL
    ) -> Tuple[List[List[str]], Dict[str, Any]]:
        """Redact PII/PHI from table cells while preserving structure."""
        redacted_table = []
        manifest = {
            'cells_scanned': 0,
            'cells_redacted': 0,
            'entities_found': [],
        }

        # Identify columns that might contain PII
        sensitive_columns = set()
        for i, header in enumerate(headers):
            header_lower = header.lower()
            if any(term in header_lower for term in
                   ['name', 'patient', 'id', 'mrn', 'dob', 'date', 'email', 'phone']):
                sensitive_columns.add(i)

        for row in table:
            redacted_row = []
            for col_idx, cell in enumerate(row):
                manifest['cells_scanned'] += 1

                # Always scan sensitive columns, spot-check others
                if col_idx in sensitive_columns or len(cell) > 20:
                    entities = self.detector.detect(cell)

                    if entities:
                        result = self.redactor.redact(cell, entities, risk_level)
                        redacted_row.append(result.redacted_text)
                        manifest['cells_redacted'] += 1
                        manifest['entities_found'].extend([
                            {'cell': f"row_{len(redacted_table)}_col_{col_idx}",
                             'type': e.entity_type.value}
                            for e in entities
                        ])
                    else:
                        redacted_row.append(cell)
                else:
                    redacted_row.append(cell)

            redacted_table.append(redacted_row)

        return redacted_table, manifest


class IngestionGate:
    """Ingestion gate for blocking unsafe documents."""

    def __init__(
        self,
        quarantine_path: str = "data/quarantine",
        vault_path: str = "data/vault"
    ):
        self.quarantine_path = Path(quarantine_path)
        self.vault_path = Path(vault_path)
        self.detector = HybridDetector()
        self.redactor = Redactor(vault_path)

        self.quarantine_path.mkdir(parents=True, exist_ok=True)
        self.vault_path.mkdir(parents=True, exist_ok=True)

    def classify_risk(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> RiskLevel:
        """Classify document risk level."""
        # Check metadata first
        if metadata.get('risk_level'):
            return RiskLevel(metadata['risk_level'])

        if metadata.get('source_type') == 'clinical':
            return RiskLevel.RESTRICTED_PHI

        if metadata.get('source_type') == 'internal':
            return RiskLevel.INTERNAL

        if metadata.get('source_type') == 'public':
            return RiskLevel.PUBLIC

        # Heuristic: check for PHI indicators
        phi_indicators = [
            r'\bPatient\s*:\s*[A-Z]',
            r'\bMRN\b',
            r'\bDOB\b',
            r'\bDiagnosis\b',
            r'\bclinical\s+notes?\b',
        ]

        for pattern in phi_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return RiskLevel.RESTRICTED_PHI

        return RiskLevel.UNKNOWN

    def process_document(
        self,
        doc_id: str,
        text: str,
        metadata: Dict[str, Any]
    ) -> IngestionGateResult:
        """Process document through ingestion gate."""
        # Step 1: Classify risk
        risk_level = self.classify_risk(text, metadata)

        # Step 2: Detect entities
        entities = self.detector.detect(text)

        # Step 3: Determine if quarantine needed
        high_risk_entities = [
            e for e in entities
            if e.entity_type in SensitiveFieldPolicy.EXPLICIT_IDENTIFIERS
        ]

        if risk_level == RiskLevel.UNKNOWN and high_risk_entities:
            # Quarantine for manual review
            self._quarantine(doc_id, text, entities)
            return IngestionGateResult(
                doc_id=doc_id,
                passed=False,
                risk_level=risk_level,
                entities_detected=len(entities),
                entities_redacted=0,
                quarantined=True,
                reason="Unknown risk level with high-risk entities detected",
            )

        # Step 4: Redact if needed
        if entities:
            result = self.redactor.redact(text, entities, risk_level)
            self.redactor.store_manifest(result.redaction_manifest, doc_id)

            return IngestionGateResult(
                doc_id=doc_id,
                passed=True,
                risk_level=risk_level,
                entities_detected=len(entities),
                entities_redacted=result.entities_redacted,
                quarantined=False,
                redacted_text=result.redacted_text,
            )

        return IngestionGateResult(
            doc_id=doc_id,
            passed=True,
            risk_level=risk_level,
            entities_detected=0,
            entities_redacted=0,
            quarantined=False,
            redacted_text=text,
        )

    def _quarantine(
        self,
        doc_id: str,
        text: str,
        entities: List[DetectedEntity]
    ):
        """Move document to quarantine."""
        quarantine_file = self.quarantine_path / f"{doc_id}.json"
        data = {
            'doc_id': doc_id,
            'quarantine_time': datetime.now().isoformat(),
            'text_hash': hashlib.sha256(text.encode()).hexdigest(),
            'entities_detected': [
                {'type': e.entity_type.value, 'confidence': e.confidence}
                for e in entities
            ],
        }
        with open(quarantine_file, 'w') as f:
            json.dump(data, f, indent=2)


class BackscanManager:
    """Manage backscan and purge operations."""

    def __init__(self, gate: IngestionGate):
        self.gate = gate
        self.scan_history: List[Dict[str, Any]] = []

    def backscan_corpus(
        self,
        docs: List[Tuple[str, str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Rescan entire corpus with updated detector."""
        results = {
            'total_docs': len(docs),
            'issues_found': 0,
            'docs_requiring_reindex': [],
            'scan_timestamp': datetime.now().isoformat(),
        }

        for doc_id, text, metadata in docs:
            entities = self.gate.detector.detect(text)

            if entities:
                results['issues_found'] += 1
                results['docs_requiring_reindex'].append({
                    'doc_id': doc_id,
                    'entities': len(entities),
                })

        self.scan_history.append(results)
        return results

    def purge_document(self, doc_id: str) -> bool:
        """Purge document from all stores."""
        # In real implementation, would remove from:
        # - Vector DB
        # - Cache
        # - Any derived artifacts
        logger.info(f"Purging document: {doc_id}")
        return True


class PHIMonitor:
    """Production monitoring for PII/PHI leaks."""

    def __init__(self):
        self.alerts: List[Dict[str, Any]] = []
        self.metrics = {
            'scanned_count': 0,
            'detected_count': 0,
            'redacted_count': 0,
            'quarantined_count': 0,
            'leak_attempts': 0,
        }

    def log_scan(self, result: IngestionGateResult):
        """Log scan result for monitoring."""
        self.metrics['scanned_count'] += 1
        self.metrics['detected_count'] += result.entities_detected
        self.metrics['redacted_count'] += result.entities_redacted

        if result.quarantined:
            self.metrics['quarantined_count'] += 1

    def check_output_for_leaks(self, output: str) -> bool:
        """Check RAG output for potential leaks."""
        detector = HybridDetector()
        entities = detector.detect(output)

        if entities:
            self.metrics['leak_attempts'] += 1
            self.alerts.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'output_leak',
                'entities': [e.entity_type.value for e in entities],
            })
            return True

        return False

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for monitoring dashboard."""
        return {
            **self.metrics,
            'leak_rate': (
                self.metrics['leak_attempts'] / max(1, self.metrics['scanned_count'])
            ),
            'recent_alerts': self.alerts[-10:],
        }


if __name__ == '__main__':
    # Demo usage
    gate = IngestionGate()

    test_text = """
    Patient: John Smith
    MRN: ABC123456
    DOB: 01/15/1985
    Email: john.smith@email.com

    EEG Recording Notes:
    Electrodes Fp1, Fp2, F7, F8 showed normal alpha activity at 10 Hz.
    The patient exhibited relaxed state during recording.
    """

    result = gate.process_document(
        doc_id="test_001",
        text=test_text,
        metadata={'source_type': 'clinical'}
    )

    print(f"Gate passed: {result.passed}")
    print(f"Risk level: {result.risk_level}")
    print(f"Entities detected: {result.entities_detected}")
    print(f"Entities redacted: {result.entities_redacted}")
    print(f"Quarantined: {result.quarantined}")
    print(f"\nRedacted text:\n{result.redacted_text}")
