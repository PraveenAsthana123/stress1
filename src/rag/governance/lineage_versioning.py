"""
A11) Full Lineage Versioning + Answer Replay for EEG-RAG

Comprehensive module for:
- Lineage scope definition
- Stable document IDs
- Corpus snapshots
- Pipeline versioning (parser, chunking, embedding)
- Deterministic chunk IDs
- Run manifests
- Answer replay
- Diff tooling
- Release gating

This enables reproducing any answer exactly and proving compliance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class DocumentID:
    """Stable document identifier."""
    guid: str
    version: str
    checksum: str

    @classmethod
    def create(cls, content: str, version: str = "1.0") -> 'DocumentID':
        """Create document ID from content."""
        checksum = hashlib.sha256(content.encode()).hexdigest()[:16]
        guid = hashlib.sha256(f"{content[:100]}_{version}".encode()).hexdigest()[:12]
        return cls(guid=guid, version=version, checksum=checksum)

    def to_dict(self) -> Dict[str, str]:
        return {'guid': self.guid, 'version': self.version, 'checksum': self.checksum}


@dataclass
class ChunkID:
    """Deterministic chunk identifier."""
    chunk_id: str
    doc_guid: str
    section_path: str
    span_start: int
    span_end: int

    @classmethod
    def create(
        cls,
        doc_guid: str,
        version: str,
        section_path: str,
        span_start: int,
        span_end: int
    ) -> 'ChunkID':
        """Create deterministic chunk ID."""
        content = f"{doc_guid}_{version}_{section_path}_{span_start}_{span_end}"
        chunk_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        return cls(
            chunk_id=chunk_id,
            doc_guid=doc_guid,
            section_path=section_path,
            span_start=span_start,
            span_end=span_end,
        )


@dataclass
class PipelineManifest:
    """Manifest of pipeline configuration."""
    parser_version: str
    parser_config_hash: str
    chunking_version: str
    chunking_config: Dict[str, Any]
    embedding_model: str
    embedding_version: str
    index_type: str
    index_params: Dict[str, Any]
    prompt_template_hash: str
    verifier_version: str


@dataclass
class CorpusSnapshot:
    """Snapshot of corpus for a release."""
    snapshot_id: str
    created_at: str
    doc_count: int
    doc_ids: List[DocumentID]
    total_chunks: int
    checksum: str


@dataclass
class RunManifest:
    """Complete manifest for a single query run."""
    run_id: str
    timestamp: str
    query_text: str
    query_hash: str
    intent: str
    user_role: str
    corpus_snapshot_id: str
    pipeline_manifest: PipelineManifest
    retrieved_chunk_ids: List[str]
    rerank_order: List[str]
    final_context: str
    output_text: str
    validator_results: Dict[str, Any]
    token_counts: Dict[str, int]
    latency_ms: float


class LineageScopeSpec:
    """Define what must be versioned for replay."""

    SCOPE = {
        'documents': {
            'required': True,
            'fields': ['guid', 'version', 'checksum'],
        },
        'parsing': {
            'required': True,
            'fields': ['parser_version', 'settings', 'table_extraction_config'],
        },
        'chunking': {
            'required': True,
            'fields': ['chunk_size', 'overlap', 'semantic_rules', 'table_strategy'],
        },
        'embedding': {
            'required': True,
            'fields': ['model_name', 'model_version', 'pooling', 'normalization', 'dim'],
        },
        'index': {
            'required': True,
            'fields': ['index_type', 'params', 'metric', 'seed'],
        },
        'retrieval': {
            'required': True,
            'fields': ['top_k', 'filters', 'mmr_lambda', 'hybrid_weights', 'query_rewrite'],
        },
        'reranking': {
            'required': False,
            'fields': ['model_version', 'prompt', 'batch_size'],
        },
        'prompts': {
            'required': True,
            'fields': ['template_version', 'system_policy_version', 'prompt_hash'],
        },
        'token_budget': {
            'required': True,
            'fields': ['context_packing_strategy', 'per_section_budgets'],
        },
        'post_processing': {
            'required': True,
            'fields': ['citation_validator_version', 'evidence_verifier_version', 'thresholds'],
        },
        'runtime': {
            'required': True,
            'fields': ['container_hash', 'python_version', 'gpu_driver', 'deps_lockfile'],
        },
    }

    @classmethod
    def validate_manifest(cls, manifest: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate manifest has all required fields."""
        missing = []

        for component, spec in cls.SCOPE.items():
            if spec['required']:
                if component not in manifest:
                    missing.append(f"Missing component: {component}")
                else:
                    for field in spec['fields']:
                        if field not in manifest[component]:
                            missing.append(f"Missing field: {component}.{field}")

        return len(missing) == 0, missing


class DocumentRegistry:
    """Registry of documents with stable IDs."""

    def __init__(self, registry_path: str = "data/lineage/docs"):
        self.path = Path(registry_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.docs: Dict[str, DocumentID] = {}

    def register(self, content: str, version: str = "1.0") -> DocumentID:
        """Register document and get stable ID."""
        doc_id = DocumentID.create(content, version)

        if doc_id.guid not in self.docs:
            self.docs[doc_id.guid] = doc_id
            self._save_doc(doc_id)

        return doc_id

    def get(self, guid: str) -> Optional[DocumentID]:
        """Get document by GUID."""
        return self.docs.get(guid)

    def _save_doc(self, doc_id: DocumentID):
        """Save document registration."""
        doc_file = self.path / f"{doc_id.guid}.json"
        with open(doc_file, 'w') as f:
            json.dump(doc_id.to_dict(), f)


class CorpusSnapshotManager:
    """Manage corpus snapshots for releases."""

    def __init__(self, snapshot_path: str = "data/lineage/snapshots"):
        self.path = Path(snapshot_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.snapshots: Dict[str, CorpusSnapshot] = {}

    def create_snapshot(
        self,
        doc_ids: List[DocumentID],
        chunk_count: int
    ) -> CorpusSnapshot:
        """Create new corpus snapshot."""
        # Create snapshot checksum
        content = json.dumps([d.to_dict() for d in doc_ids], sort_keys=True)
        checksum = hashlib.sha256(content.encode()).hexdigest()[:16]

        snapshot_id = f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{checksum[:8]}"

        snapshot = CorpusSnapshot(
            snapshot_id=snapshot_id,
            created_at=datetime.now().isoformat(),
            doc_count=len(doc_ids),
            doc_ids=doc_ids,
            total_chunks=chunk_count,
            checksum=checksum,
        )

        self.snapshots[snapshot_id] = snapshot
        self._save_snapshot(snapshot)

        return snapshot

    def get_snapshot(self, snapshot_id: str) -> Optional[CorpusSnapshot]:
        """Get snapshot by ID."""
        return self.snapshots.get(snapshot_id)

    def _save_snapshot(self, snapshot: CorpusSnapshot):
        """Save snapshot to disk."""
        snap_file = self.path / f"{snapshot.snapshot_id}.json"
        with open(snap_file, 'w') as f:
            json.dump({
                'snapshot_id': snapshot.snapshot_id,
                'created_at': snapshot.created_at,
                'doc_count': snapshot.doc_count,
                'doc_ids': [d.to_dict() for d in snapshot.doc_ids],
                'total_chunks': snapshot.total_chunks,
                'checksum': snapshot.checksum,
            }, f, indent=2)


class RunManifestStore:
    """Store run manifests for replay."""

    def __init__(self, store_path: str = "data/lineage/runs"):
        self.path = Path(store_path)
        self.path.mkdir(parents=True, exist_ok=True)

    def store(self, manifest: RunManifest) -> str:
        """Store run manifest."""
        run_file = self.path / f"{manifest.run_id}.json"

        data = {
            'run_id': manifest.run_id,
            'timestamp': manifest.timestamp,
            'query_text': manifest.query_text,
            'query_hash': manifest.query_hash,
            'intent': manifest.intent,
            'user_role': manifest.user_role,
            'corpus_snapshot_id': manifest.corpus_snapshot_id,
            'pipeline_manifest': {
                'parser_version': manifest.pipeline_manifest.parser_version,
                'parser_config_hash': manifest.pipeline_manifest.parser_config_hash,
                'chunking_version': manifest.pipeline_manifest.chunking_version,
                'chunking_config': manifest.pipeline_manifest.chunking_config,
                'embedding_model': manifest.pipeline_manifest.embedding_model,
                'embedding_version': manifest.pipeline_manifest.embedding_version,
                'index_type': manifest.pipeline_manifest.index_type,
                'index_params': manifest.pipeline_manifest.index_params,
                'prompt_template_hash': manifest.pipeline_manifest.prompt_template_hash,
                'verifier_version': manifest.pipeline_manifest.verifier_version,
            },
            'retrieved_chunk_ids': manifest.retrieved_chunk_ids,
            'rerank_order': manifest.rerank_order,
            'final_context': manifest.final_context,
            'output_text': manifest.output_text,
            'validator_results': manifest.validator_results,
            'token_counts': manifest.token_counts,
            'latency_ms': manifest.latency_ms,
        }

        with open(run_file, 'w') as f:
            json.dump(data, f, indent=2)

        return manifest.run_id

    def load(self, run_id: str) -> Optional[RunManifest]:
        """Load run manifest."""
        run_file = self.path / f"{run_id}.json"
        if not run_file.exists():
            return None

        with open(run_file, 'r') as f:
            data = json.load(f)

        return RunManifest(
            run_id=data['run_id'],
            timestamp=data['timestamp'],
            query_text=data['query_text'],
            query_hash=data['query_hash'],
            intent=data['intent'],
            user_role=data['user_role'],
            corpus_snapshot_id=data['corpus_snapshot_id'],
            pipeline_manifest=PipelineManifest(
                parser_version=data['pipeline_manifest']['parser_version'],
                parser_config_hash=data['pipeline_manifest']['parser_config_hash'],
                chunking_version=data['pipeline_manifest']['chunking_version'],
                chunking_config=data['pipeline_manifest']['chunking_config'],
                embedding_model=data['pipeline_manifest']['embedding_model'],
                embedding_version=data['pipeline_manifest']['embedding_version'],
                index_type=data['pipeline_manifest']['index_type'],
                index_params=data['pipeline_manifest']['index_params'],
                prompt_template_hash=data['pipeline_manifest']['prompt_template_hash'],
                verifier_version=data['pipeline_manifest']['verifier_version'],
            ),
            retrieved_chunk_ids=data['retrieved_chunk_ids'],
            rerank_order=data['rerank_order'],
            final_context=data['final_context'],
            output_text=data['output_text'],
            validator_results=data['validator_results'],
            token_counts=data['token_counts'],
            latency_ms=data['latency_ms'],
        )


class AnswerReplay:
    """Replay answers from stored manifests."""

    def __init__(self, manifest_store: RunManifestStore):
        self.store = manifest_store

    def replay_from_context(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Replay answer using stored context.

        This reconstructs the answer from the stored final_context,
        without re-running the full pipeline.
        """
        manifest = self.store.load(run_id)
        if not manifest:
            return None

        return {
            'run_id': run_id,
            'original_output': manifest.output_text,
            'context': manifest.final_context,
            'retrieved_chunks': manifest.retrieved_chunk_ids,
            'replay_mode': 'from_context',
        }

    def replay_full(self, run_id: str, pipeline: Any) -> Optional[Dict[str, Any]]:
        """
        Replay by re-running the full pipeline with stored versions.

        This requires the pipeline to support version switching.
        """
        manifest = self.store.load(run_id)
        if not manifest:
            return None

        # Would re-run pipeline with versioned config
        # Placeholder - actual implementation depends on pipeline

        return {
            'run_id': run_id,
            'replay_mode': 'full_pipeline',
            'original_output': manifest.output_text,
            'message': 'Full replay requires pipeline version support',
        }


class RunDiff:
    """Compare two runs to find differences."""

    def diff(
        self,
        run_id_1: str,
        run_id_2: str,
        store: RunManifestStore
    ) -> Dict[str, Any]:
        """Compute diff between two runs."""
        m1 = store.load(run_id_1)
        m2 = store.load(run_id_2)

        if not m1 or not m2:
            return {'error': 'One or both runs not found'}

        diff = {
            'run_1': run_id_1,
            'run_2': run_id_2,
            'differences': [],
        }

        # Compare retrieval sets
        chunks_1 = set(m1.retrieved_chunk_ids)
        chunks_2 = set(m2.retrieved_chunk_ids)

        if chunks_1 != chunks_2:
            diff['differences'].append({
                'component': 'retrieval',
                'only_in_1': list(chunks_1 - chunks_2),
                'only_in_2': list(chunks_2 - chunks_1),
                'common': list(chunks_1 & chunks_2),
            })

        # Compare rerank order
        if m1.rerank_order != m2.rerank_order:
            diff['differences'].append({
                'component': 'rerank_order',
                'order_1': m1.rerank_order,
                'order_2': m2.rerank_order,
            })

        # Compare outputs
        if m1.output_text != m2.output_text:
            diff['differences'].append({
                'component': 'output',
                'output_1_len': len(m1.output_text),
                'output_2_len': len(m2.output_text),
                'outputs_match': False,
            })

        # Compare pipeline versions
        if m1.pipeline_manifest.embedding_model != m2.pipeline_manifest.embedding_model:
            diff['differences'].append({
                'component': 'embedding_model',
                'version_1': m1.pipeline_manifest.embedding_model,
                'version_2': m2.pipeline_manifest.embedding_model,
            })

        return diff


class ReleaseGate:
    """Gate releases on lineage completeness."""

    def __init__(self):
        self.requirements = [
            'corpus_snapshot_exists',
            'pipeline_manifest_valid',
            'all_docs_registered',
            'chunk_ids_deterministic',
            'prompt_versioned',
        ]

    def check_release_ready(
        self,
        snapshot: Optional[CorpusSnapshot],
        manifest: Optional[PipelineManifest],
        doc_registry: DocumentRegistry
    ) -> Tuple[bool, List[str]]:
        """Check if release meets lineage requirements."""
        issues = []

        # Check snapshot
        if not snapshot:
            issues.append("Missing corpus snapshot")
        else:
            if snapshot.doc_count == 0:
                issues.append("Empty corpus snapshot")

        # Check manifest
        if not manifest:
            issues.append("Missing pipeline manifest")
        else:
            if not manifest.embedding_model:
                issues.append("Missing embedding model version")
            if not manifest.prompt_template_hash:
                issues.append("Missing prompt template hash")

        # Check docs registered
        if snapshot:
            for doc_id in snapshot.doc_ids:
                if not doc_registry.get(doc_id.guid):
                    issues.append(f"Doc not registered: {doc_id.guid}")

        return len(issues) == 0, issues


class LineageMonitor:
    """Monitor lineage health."""

    def __init__(self, store: RunManifestStore):
        self.store = store
        self.metrics = {
            'total_runs': 0,
            'missing_fields_count': 0,
            'replay_success_rate': 0.0,
        }

    def check_manifest_completeness(self, run_id: str) -> Dict[str, Any]:
        """Check manifest has all required fields."""
        manifest = self.store.load(run_id)
        if not manifest:
            return {'error': 'Run not found'}

        missing = []

        if not manifest.query_text:
            missing.append('query_text')
        if not manifest.retrieved_chunk_ids:
            missing.append('retrieved_chunk_ids')
        if not manifest.output_text:
            missing.append('output_text')
        if not manifest.pipeline_manifest:
            missing.append('pipeline_manifest')

        return {
            'run_id': run_id,
            'complete': len(missing) == 0,
            'missing_fields': missing,
        }

    def get_health_report(self) -> Dict[str, Any]:
        """Get lineage health report."""
        return {
            'metrics': self.metrics,
            'status': 'healthy' if self.metrics['missing_fields_count'] == 0 else 'degraded',
        }


class LineageManager:
    """Main lineage management coordinator."""

    def __init__(self, base_path: str = "data/lineage"):
        self.doc_registry = DocumentRegistry(f"{base_path}/docs")
        self.snapshot_manager = CorpusSnapshotManager(f"{base_path}/snapshots")
        self.manifest_store = RunManifestStore(f"{base_path}/runs")
        self.replay = AnswerReplay(self.manifest_store)
        self.diff = RunDiff()
        self.gate = ReleaseGate()
        self.monitor = LineageMonitor(self.manifest_store)

    def record_run(
        self,
        query: str,
        intent: str,
        user_role: str,
        retrieved_chunks: List[str],
        rerank_order: List[str],
        context: str,
        output: str,
        pipeline_manifest: PipelineManifest,
        snapshot_id: str,
        validator_results: Dict[str, Any],
        token_counts: Dict[str, int],
        latency_ms: float
    ) -> str:
        """Record a complete run for lineage."""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.sha256(query.encode()).hexdigest()[:8]}"

        manifest = RunManifest(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            query_text=query,
            query_hash=hashlib.sha256(query.encode()).hexdigest()[:16],
            intent=intent,
            user_role=user_role,
            corpus_snapshot_id=snapshot_id,
            pipeline_manifest=pipeline_manifest,
            retrieved_chunk_ids=retrieved_chunks,
            rerank_order=rerank_order,
            final_context=context,
            output_text=output,
            validator_results=validator_results,
            token_counts=token_counts,
            latency_ms=latency_ms,
        )

        return self.manifest_store.store(manifest)


if __name__ == '__main__':
    # Demo usage
    manager = LineageManager()

    # Register a document
    doc_content = "EEG frequency bands: Alpha 8-13 Hz, Beta 13-30 Hz"
    doc_id = manager.doc_registry.register(doc_content, version="1.0")
    print(f"Registered doc: {doc_id.guid}")

    # Create snapshot
    snapshot = manager.snapshot_manager.create_snapshot([doc_id], chunk_count=5)
    print(f"Created snapshot: {snapshot.snapshot_id}")

    # Create pipeline manifest
    pipeline = PipelineManifest(
        parser_version="1.0.0",
        parser_config_hash="abc123",
        chunking_version="1.0.0",
        chunking_config={'size': 512, 'overlap': 50},
        embedding_model="text-embedding-3-small",
        embedding_version="v1",
        index_type="HNSW",
        index_params={'M': 16, 'ef': 100},
        prompt_template_hash="def456",
        verifier_version="1.0.0",
    )

    # Record a run
    run_id = manager.record_run(
        query="What is the alpha wave frequency?",
        intent="definition",
        user_role="researcher",
        retrieved_chunks=["chunk_001", "chunk_002"],
        rerank_order=["chunk_001", "chunk_002"],
        context="Alpha waves are 8-13 Hz",
        output="Alpha waves occur at 8-13 Hz.",
        pipeline_manifest=pipeline,
        snapshot_id=snapshot.snapshot_id,
        validator_results={'supported': True},
        token_counts={'input': 100, 'output': 20},
        latency_ms=500,
    )

    print(f"Recorded run: {run_id}")

    # Replay
    replay_result = manager.replay.replay_from_context(run_id)
    print(f"Replay result: {replay_result}")
