"""
A9) Table-RAG Strategy (Row-Level Retrieval) for EEG-RAG

Comprehensive module for:
- Table identification and extraction
- Consistent table schema
- Row-level text generation for embeddings
- Hybrid indexing (vector + keyword)
- Key-field filtering
- Table-to-answer generation
- Row-level verification
- Multi-row aggregation
- Conflict handling in tables

This makes tables first-class retrievable objects for EEG knowledge.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import re
import hashlib
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TableType(Enum):
    """Types of EEG tables."""
    BAND_DEFINITIONS = "band_definitions"
    ARTIFACT_RULES = "artifact_rules"
    DEVICE_CONFIG = "device_config"
    ELECTRODE_MAPPING = "electrode_mapping"
    SOP_PARAMETERS = "sop_parameters"
    FILTER_SETTINGS = "filter_settings"
    MONTAGE_DEFINITIONS = "montage_definitions"
    THRESHOLD_TABLE = "threshold_table"


@dataclass
class TableSchema:
    """Consistent table schema for EEG tables."""
    table_id: str
    doc_guid: str
    doc_version: str
    title: str
    table_type: TableType
    columns: List[str]
    rows: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    page_ref: Optional[int] = None


@dataclass
class TableRow:
    """Single table row for retrieval."""
    row_id: str
    table_id: str
    doc_guid: str
    row_index: int
    key_fields: Dict[str, str]
    values: Dict[str, Any]
    row_text: str  # Text for embedding
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TableQueryResult:
    """Result of table query."""
    rows: List[TableRow]
    table_metadata: Dict[str, Any]
    confidence: float
    aggregated: bool = False


class TableExtractor:
    """Extract tables from documents."""

    # Patterns indicating table presence
    TABLE_INDICATORS = [
        r'\|\s*\w+\s*\|',  # Markdown table
        r'<table>',  # HTML table
        r'Band\s+Frequency',  # EEG-specific
        r'Parameter\s+Value',
        r'Electrode\s+Position',
    ]

    def extract_tables(
        self,
        text: str,
        doc_guid: str,
        doc_version: str
    ) -> List[TableSchema]:
        """Extract tables from document text."""
        tables = []

        # Try markdown table extraction
        md_tables = self._extract_markdown_tables(text, doc_guid, doc_version)
        tables.extend(md_tables)

        # Try structured data patterns
        pattern_tables = self._extract_pattern_tables(text, doc_guid, doc_version)
        tables.extend(pattern_tables)

        return tables

    def _extract_markdown_tables(
        self,
        text: str,
        doc_guid: str,
        doc_version: str
    ) -> List[TableSchema]:
        """Extract markdown-style tables."""
        tables = []

        # Find table blocks
        table_pattern = r'(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n?)+)'
        matches = re.finditer(table_pattern, text)

        for i, match in enumerate(matches):
            table_text = match.group(1)
            lines = table_text.strip().split('\n')

            if len(lines) < 3:
                continue

            # Parse header
            headers = [h.strip() for h in lines[0].split('|') if h.strip()]

            # Parse rows
            rows = []
            for line in lines[2:]:  # Skip header and separator
                cells = [c.strip() for c in line.split('|') if c.strip()]
                if len(cells) == len(headers):
                    row = {headers[j]: cells[j] for j in range(len(headers))}
                    rows.append(row)

            if rows:
                table_id = hashlib.sha256(
                    f"{doc_guid}_{doc_version}_table_{i}".encode()
                ).hexdigest()[:12]

                tables.append(TableSchema(
                    table_id=table_id,
                    doc_guid=doc_guid,
                    doc_version=doc_version,
                    title=self._infer_title(text, match.start()),
                    table_type=self._infer_table_type(headers, rows),
                    columns=headers,
                    rows=rows,
                ))

        return tables

    def _extract_pattern_tables(
        self,
        text: str,
        doc_guid: str,
        doc_version: str
    ) -> List[TableSchema]:
        """Extract tables from structured patterns."""
        tables = []

        # EEG frequency band pattern
        band_pattern = r'(Alpha|Beta|Theta|Delta|Gamma)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*Hz'
        band_matches = re.findall(band_pattern, text, re.IGNORECASE)

        if len(band_matches) >= 3:
            rows = [
                {'Band': m[0], 'Low_Hz': m[1], 'High_Hz': m[2]}
                for m in band_matches
            ]

            table_id = hashlib.sha256(
                f"{doc_guid}_{doc_version}_bands".encode()
            ).hexdigest()[:12]

            tables.append(TableSchema(
                table_id=table_id,
                doc_guid=doc_guid,
                doc_version=doc_version,
                title="EEG Frequency Bands",
                table_type=TableType.BAND_DEFINITIONS,
                columns=['Band', 'Low_Hz', 'High_Hz'],
                rows=rows,
            ))

        return tables

    def _infer_title(self, text: str, position: int) -> str:
        """Infer table title from context."""
        # Look for heading before table
        before = text[max(0, position - 200):position]
        heading_match = re.search(r'#+\s*(.+?)\n', before)
        if heading_match:
            return heading_match.group(1).strip()
        return "Untitled Table"

    def _infer_table_type(
        self,
        headers: List[str],
        rows: List[Dict[str, Any]]
    ) -> TableType:
        """Infer table type from content."""
        headers_lower = [h.lower() for h in headers]

        if any('band' in h or 'frequency' in h for h in headers_lower):
            return TableType.BAND_DEFINITIONS
        if any('electrode' in h for h in headers_lower):
            return TableType.ELECTRODE_MAPPING
        if any('filter' in h for h in headers_lower):
            return TableType.FILTER_SETTINGS
        if any('threshold' in h for h in headers_lower):
            return TableType.THRESHOLD_TABLE
        if any('device' in h or 'config' in h for h in headers_lower):
            return TableType.DEVICE_CONFIG

        return TableType.SOP_PARAMETERS


class TableNormalizer:
    """Normalize table values for consistent retrieval."""

    # Unit conversions
    UNIT_MAP = {
        'hz': 'Hz',
        'khz': 'kHz',
        'mv': 'mV',
        'uv': 'µV',
        'µv': 'µV',
        'ms': 'ms',
        's': 's',
    }

    # Band name normalization
    BAND_MAP = {
        'alpha': 'Alpha',
        'beta': 'Beta',
        'theta': 'Theta',
        'delta': 'Delta',
        'gamma': 'Gamma',
    }

    def normalize_table(self, table: TableSchema) -> TableSchema:
        """Normalize table values."""
        normalized_rows = []

        for row in table.rows:
            normalized = {}
            for key, value in row.items():
                normalized[key] = self._normalize_value(str(value), key)
            normalized_rows.append(normalized)

        return TableSchema(
            table_id=table.table_id,
            doc_guid=table.doc_guid,
            doc_version=table.doc_version,
            title=table.title,
            table_type=table.table_type,
            columns=table.columns,
            rows=normalized_rows,
            metadata=table.metadata,
            page_ref=table.page_ref,
        )

    def _normalize_value(self, value: str, column: str) -> str:
        """Normalize a single value."""
        value = value.strip()

        # Band name normalization
        value_lower = value.lower()
        if value_lower in self.BAND_MAP:
            return self.BAND_MAP[value_lower]

        # Unit normalization
        for old, new in self.UNIT_MAP.items():
            value = re.sub(rf'\b{old}\b', new, value, flags=re.IGNORECASE)

        return value


class RowTextGenerator:
    """Generate row-level text for embeddings."""

    def generate(self, table: TableSchema) -> List[TableRow]:
        """Generate TableRow objects with embedding text."""
        rows = []

        for idx, row_data in enumerate(table.rows):
            # Build row text: title + key columns + values
            row_text = self._build_row_text(table, row_data)

            # Identify key fields
            key_fields = self._extract_key_fields(table, row_data)

            row_id = f"{table.table_id}_row_{idx}"

            rows.append(TableRow(
                row_id=row_id,
                table_id=table.table_id,
                doc_guid=table.doc_guid,
                row_index=idx,
                key_fields=key_fields,
                values=row_data,
                row_text=row_text,
                metadata={
                    'table_title': table.title,
                    'table_type': table.table_type.value,
                },
            ))

        return rows

    def _build_row_text(
        self,
        table: TableSchema,
        row_data: Dict[str, Any]
    ) -> str:
        """Build text for row embedding."""
        parts = [table.title]

        for col, value in row_data.items():
            parts.append(f"{col}: {value}")

        return '. '.join(parts)

    def _extract_key_fields(
        self,
        table: TableSchema,
        row_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Extract key fields for filtering."""
        key_fields = {}

        # First column is usually the key
        if table.columns:
            first_col = table.columns[0]
            if first_col in row_data:
                key_fields['primary_key'] = str(row_data[first_col])

        # Look for known key fields
        for col in ['Band', 'Electrode', 'Device', 'Channel', 'Parameter']:
            if col in row_data:
                key_fields[col.lower()] = str(row_data[col])

        return key_fields


class TableIndex:
    """Index for table row retrieval."""

    def __init__(self):
        self.rows: Dict[str, TableRow] = {}
        self.by_table: Dict[str, List[str]] = {}
        self.by_key: Dict[str, List[str]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}

    def add_row(self, row: TableRow, embedding: Optional[np.ndarray] = None):
        """Add row to index."""
        self.rows[row.row_id] = row

        # Index by table
        if row.table_id not in self.by_table:
            self.by_table[row.table_id] = []
        self.by_table[row.table_id].append(row.row_id)

        # Index by key fields
        for key, value in row.key_fields.items():
            index_key = f"{key}:{value}".lower()
            if index_key not in self.by_key:
                self.by_key[index_key] = []
            self.by_key[index_key].append(row.row_id)

        if embedding is not None:
            self.embeddings[row.row_id] = embedding

    def search_by_key(
        self,
        key: str,
        value: str
    ) -> List[TableRow]:
        """Search rows by key field."""
        index_key = f"{key}:{value}".lower()
        row_ids = self.by_key.get(index_key, [])
        return [self.rows[rid] for rid in row_ids if rid in self.rows]

    def search_by_table(self, table_id: str) -> List[TableRow]:
        """Get all rows from a table."""
        row_ids = self.by_table.get(table_id, [])
        return [self.rows[rid] for rid in row_ids if rid in self.rows]

    def semantic_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[TableRow, float]]:
        """Semantic search over row embeddings."""
        if not self.embeddings:
            return []

        scores = []
        for row_id, emb in self.embeddings.items():
            similarity = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            scores.append((row_id, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for row_id, score in scores[:top_k]:
            if row_id in self.rows:
                results.append((self.rows[row_id], score))

        return results


class TableQueryRouter:
    """Route queries to table mode when appropriate."""

    TABLE_INDICATORS = [
        'threshold', 'range', 'parameter', 'value', 'setting',
        'compare', 'difference', 'which band', 'frequency of',
        'what is the', 'list of', 'table',
    ]

    def should_use_table_mode(self, query: str) -> bool:
        """Determine if query should use table retrieval."""
        query_lower = query.lower()
        return any(ind in query_lower for ind in self.TABLE_INDICATORS)

    def extract_filters(self, query: str) -> Dict[str, str]:
        """Extract filter criteria from query."""
        filters = {}

        # Band filter
        bands = ['alpha', 'beta', 'theta', 'delta', 'gamma']
        for band in bands:
            if band in query.lower():
                filters['band'] = band.capitalize()

        # Device filter (example)
        device_match = re.search(r'(?:device|system)\s+(\w+)', query, re.IGNORECASE)
        if device_match:
            filters['device'] = device_match.group(1)

        return filters


class TableAnswerGenerator:
    """Generate answers from table data."""

    def generate(
        self,
        rows: List[TableRow],
        query: str
    ) -> str:
        """Generate structured answer from rows."""
        if not rows:
            return "No relevant table data found."

        # Single row
        if len(rows) == 1:
            return self._format_single_row(rows[0])

        # Multiple rows - create mini-table
        return self._format_multi_row(rows)

    def _format_single_row(self, row: TableRow) -> str:
        """Format single row answer."""
        parts = []
        for key, value in row.values.items():
            parts.append(f"{key}: {value}")

        answer = "; ".join(parts)
        citation = f"[Source: {row.metadata.get('table_title', 'Table')}, Doc: {row.doc_guid}]"

        return f"{answer}\n\n{citation}"

    def _format_multi_row(self, rows: List[TableRow]) -> str:
        """Format multi-row answer as structured table."""
        if not rows:
            return ""

        # Get all columns
        columns = set()
        for row in rows:
            columns.update(row.values.keys())
        columns = sorted(columns)

        # Build table
        lines = []
        header = " | ".join(columns)
        separator = " | ".join(["-" * len(c) for c in columns])

        lines.append(header)
        lines.append(separator)

        for row in rows:
            cells = [str(row.values.get(c, "-")) for c in columns]
            lines.append(" | ".join(cells))

        return "\n".join(lines)


class TableVerifier:
    """Verify answers against table data."""

    def verify_numeric(
        self,
        claim_value: float,
        claim_unit: str,
        table_row: TableRow
    ) -> Tuple[bool, str]:
        """Verify numeric claim against row."""
        for key, value in table_row.values.items():
            # Extract numeric values from cell
            numbers = re.findall(r'(\d+(?:\.\d+)?)', str(value))
            for num in numbers:
                if abs(float(num) - claim_value) < 0.01:  # Tolerance
                    return True, f"Matched: {key}={value}"

        return False, "No numeric match"

    def verify_claim(
        self,
        claim: str,
        rows: List[TableRow]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verify claim against table rows."""
        claim_lower = claim.lower()

        for row in rows:
            row_text_lower = row.row_text.lower()

            # Check for key term overlap
            claim_words = set(claim_lower.split())
            row_words = set(row_text_lower.split())
            overlap = len(claim_words & row_words) / max(1, len(claim_words))

            if overlap > 0.5:
                return True, {
                    'matched_row': row.row_id,
                    'overlap': overlap,
                }

        return False, {'reason': 'No matching row'}


class TableAggregator:
    """Aggregate multiple rows for complex queries."""

    def aggregate_by_key(
        self,
        rows: List[TableRow],
        key: str
    ) -> Dict[str, List[TableRow]]:
        """Group rows by key field."""
        groups = {}
        for row in rows:
            key_value = row.key_fields.get(key, 'unknown')
            if key_value not in groups:
                groups[key_value] = []
            groups[key_value].append(row)
        return groups

    def summarize_group(
        self,
        rows: List[TableRow],
        numeric_column: str
    ) -> Dict[str, Any]:
        """Summarize numeric column in group."""
        values = []
        for row in rows:
            val = row.values.get(numeric_column)
            if val:
                numbers = re.findall(r'(\d+(?:\.\d+)?)', str(val))
                values.extend([float(n) for n in numbers])

        if values:
            return {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'count': len(values),
            }
        return {}


class TableRAGPipeline:
    """Complete Table-RAG pipeline."""

    def __init__(self):
        self.extractor = TableExtractor()
        self.normalizer = TableNormalizer()
        self.row_generator = RowTextGenerator()
        self.index = TableIndex()
        self.router = TableQueryRouter()
        self.answer_gen = TableAnswerGenerator()
        self.verifier = TableVerifier()
        self.aggregator = TableAggregator()

    def ingest_document(
        self,
        text: str,
        doc_guid: str,
        doc_version: str
    ) -> List[TableRow]:
        """Ingest tables from document."""
        # Extract tables
        tables = self.extractor.extract_tables(text, doc_guid, doc_version)

        all_rows = []
        for table in tables:
            # Normalize
            normalized = self.normalizer.normalize_table(table)

            # Generate rows
            rows = self.row_generator.generate(normalized)

            # Add to index
            for row in rows:
                self.index.add_row(row)
                all_rows.append(row)

        return all_rows

    def query(
        self,
        query: str,
        use_table_mode: Optional[bool] = None
    ) -> TableQueryResult:
        """Query table data."""
        # Determine mode
        if use_table_mode is None:
            use_table_mode = self.router.should_use_table_mode(query)

        if not use_table_mode:
            return TableQueryResult(rows=[], table_metadata={}, confidence=0.0)

        # Extract filters
        filters = self.router.extract_filters(query)

        # Search by filters
        rows = []
        for key, value in filters.items():
            matched = self.index.search_by_key(key, value)
            rows.extend(matched)

        # Deduplicate
        seen = set()
        unique_rows = []
        for row in rows:
            if row.row_id not in seen:
                seen.add(row.row_id)
                unique_rows.append(row)

        return TableQueryResult(
            rows=unique_rows,
            table_metadata={'filters': filters},
            confidence=0.8 if unique_rows else 0.3,
        )

    def generate_answer(
        self,
        query: str,
        result: TableQueryResult
    ) -> str:
        """Generate answer from table query result."""
        return self.answer_gen.generate(result.rows, query)


if __name__ == '__main__':
    # Demo usage
    pipeline = TableRAGPipeline()

    # Sample document with table
    doc_text = """
    # EEG Frequency Bands

    | Band | Frequency (Hz) | State |
    |------|----------------|-------|
    | Delta | 0.5-4 Hz | Deep sleep |
    | Theta | 4-8 Hz | Drowsiness |
    | Alpha | 8-13 Hz | Relaxed |
    | Beta | 13-30 Hz | Active |
    | Gamma | 30-100 Hz | Cognitive |
    """

    # Ingest
    rows = pipeline.ingest_document(doc_text, "doc_001", "v1")
    print(f"Ingested {len(rows)} rows")

    # Query
    result = pipeline.query("What is the frequency range of alpha waves?")
    print(f"Found {len(result.rows)} matching rows")

    # Generate answer
    answer = pipeline.generate_answer("alpha frequency", result)
    print(f"Answer:\n{answer}")
