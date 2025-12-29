#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Run Manager - Experiment Tracking & Reproducibility
================================================================================

Implements:
- Run identity (unique run_id per execution)
- Config trace (save resolved config)
- Data fingerprint (log dataset stats)
- Timing instrumentation (track stage durations)
- Metrics persistence (save to JSON/CSV)

Cross-platform compatible (Windows/Linux/macOS).

Usage:
    from src.utils.run_manager import RunManager

    with RunManager(experiment_name="stress_classification") as run:
        run.log_config(config)
        run.log_data_fingerprint(X, y, "SAM-40")

        with run.timer("training"):
            # training code
            pass

        run.log_metrics({"accuracy": 0.95, "f1": 0.94})

Author: GenAI-RAG-EEG Team
Version: 3.0.0
================================================================================
"""

import os
import sys
import json
import time
import hashlib
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import logging

import numpy as np


# =============================================================================
# RUN IDENTITY
# =============================================================================

def get_git_hash(short: bool = True) -> str:
    """Get current git commit hash."""
    try:
        cmd = ["git", "rev-parse", "--short" if short else "", "HEAD"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        return result.stdout.strip() if result.returncode == 0 else "nogit"
    except Exception:
        return "nogit"


def generate_run_id() -> str:
    """
    Generate unique run ID: timestamp + git hash.

    Format: YYYYMMDD_HHMMSS_<git_hash>
    Example: 20251229_114500_abc1234
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_hash = get_git_hash(short=True)
    return f"{timestamp}_{git_hash}"


# =============================================================================
# DATA FINGERPRINT
# =============================================================================

@dataclass
class DataFingerprint:
    """Data fingerprint for reproducibility tracking."""
    dataset_name: str
    n_samples: int
    n_channels: int
    n_timepoints: int
    dtype: str
    shape: tuple

    # Statistics
    mean: float
    std: float
    min_val: float
    max_val: float

    # Labels
    n_classes: int
    class_distribution: Dict[int, int]

    # Hash for drift detection
    schema_hash: str
    sample_hash: str  # Hash of first 10 samples

    # Metadata
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_data(cls, X: np.ndarray, y: np.ndarray, name: str = "unknown") -> 'DataFingerprint':
        """Create fingerprint from data arrays."""
        # Compute hashes
        schema_str = f"{X.shape}_{X.dtype}_{y.shape}_{y.dtype}"
        schema_hash = hashlib.md5(schema_str.encode()).hexdigest()[:8]

        # Sample hash (first 10 samples)
        sample_data = X[:10].tobytes() if len(X) >= 10 else X.tobytes()
        sample_hash = hashlib.md5(sample_data).hexdigest()[:8]

        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_dist = {int(k): int(v) for k, v in zip(unique, counts)}

        return cls(
            dataset_name=name,
            n_samples=len(X),
            n_channels=X.shape[1] if len(X.shape) > 1 else 1,
            n_timepoints=X.shape[2] if len(X.shape) > 2 else X.shape[1] if len(X.shape) > 1 else len(X),
            dtype=str(X.dtype),
            shape=X.shape,
            mean=float(X.mean()),
            std=float(X.std()),
            min_val=float(X.min()),
            max_val=float(X.max()),
            n_classes=len(unique),
            class_distribution=class_dist,
            schema_hash=schema_hash,
            sample_hash=sample_hash,
            timestamp=datetime.now().isoformat()
        )


# =============================================================================
# TIMING INSTRUMENTATION
# =============================================================================

class TimingTracker:
    """Track timing for different stages."""

    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}

    def start(self, stage: str):
        """Start timing a stage."""
        self.start_times[stage] = time.time()

    def stop(self, stage: str) -> float:
        """Stop timing and return duration."""
        if stage not in self.start_times:
            return 0.0
        duration = time.time() - self.start_times[stage]
        self.timings[stage] = duration
        del self.start_times[stage]
        return duration

    @contextmanager
    def timer(self, stage: str):
        """Context manager for timing."""
        self.start(stage)
        try:
            yield
        finally:
            self.stop(stage)

    def get_timings(self) -> Dict[str, float]:
        """Get all recorded timings."""
        return self.timings.copy()

    def total_time(self) -> float:
        """Get total time across all stages."""
        return sum(self.timings.values())


# =============================================================================
# RUN MANAGER
# =============================================================================

class RunManager:
    """
    Comprehensive run management for ML experiments.

    Features:
    - Unique run ID per execution
    - Automatic output directory creation
    - Config saving
    - Data fingerprinting
    - Timing instrumentation
    - Metrics persistence
    - Environment logging

    Usage:
        with RunManager("experiment") as run:
            run.log_config(config)
            run.log_metrics({"accuracy": 0.95})
    """

    def __init__(
        self,
        experiment_name: str = "run",
        output_base: Optional[Path] = None,
        run_id: Optional[str] = None,
        log_to_console: bool = True,
        log_to_file: bool = True
    ):
        """
        Initialize run manager.

        Args:
            experiment_name: Name of experiment
            output_base: Base directory for outputs (default: PROJECT_ROOT/outputs)
            run_id: Custom run ID (default: auto-generated)
            log_to_console: Log to stdout
            log_to_file: Log to file
        """
        self.experiment_name = experiment_name
        self.run_id = run_id or generate_run_id()

        # Set output directory
        if output_base is None:
            try:
                from ..config import PROJECT_ROOT
                output_base = PROJECT_ROOT / "outputs"
            except ImportError:
                output_base = Path(__file__).parent.parent.parent / "outputs"

        self.output_dir = Path(output_base) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.timing = TimingTracker()
        self.metrics: Dict[str, Any] = {}
        self.fingerprints: List[DataFingerprint] = []

        # Setup logging
        self._setup_logging(log_to_console, log_to_file)

        # Track start time
        self._start_time = time.time()

    def _setup_logging(self, console: bool, file: bool):
        """Setup logging to console and file."""
        self.logger = logging.getLogger(f"run_{self.run_id}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers

        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        if file:
            log_file = self.output_dir / "run.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def __enter__(self):
        """Start the run."""
        self._log_startup_banner()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the run and save all outputs."""
        total_time = time.time() - self._start_time

        # Save all outputs
        self._save_timings()
        self._save_metrics()
        self._save_fingerprints()
        self._save_environment()

        # Log summary
        self.logger.info("=" * 60)
        self.logger.info("RUN COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"  Run ID: {self.run_id}")
        self.logger.info(f"  Total Time: {total_time:.2f}s ({total_time/60:.1f} min)")
        self.logger.info(f"  Output Dir: {self.output_dir}")

        if exc_type is not None:
            self.logger.error(f"  Status: FAILED - {exc_type.__name__}: {exc_val}")
        else:
            self.logger.info("  Status: SUCCESS")

        self.logger.info("=" * 60)

        return False  # Don't suppress exceptions

    def _log_startup_banner(self):
        """Log startup banner with environment info."""
        self.logger.info("=" * 60)
        self.logger.info("  GenAI-RAG-EEG RUN STARTED")
        self.logger.info("=" * 60)
        self.logger.info(f"  Run ID: {self.run_id}")
        self.logger.info(f"  Experiment: {self.experiment_name}")
        self.logger.info(f"  Timestamp: {datetime.now().isoformat()}")
        self.logger.info("")
        self.logger.info("  Environment:")
        self.logger.info(f"    OS: {platform.system()} {platform.release()}")
        self.logger.info(f"    Python: {platform.python_version()}")

        try:
            import torch
            self.logger.info(f"    PyTorch: {torch.__version__}")
            if torch.cuda.is_available():
                self.logger.info(f"    GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.logger.info("    GPU: Not available")
        except ImportError:
            self.logger.info("    PyTorch: Not installed")

        self.logger.info(f"    Git Hash: {get_git_hash()}")
        self.logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # CONFIG TRACE
    # -------------------------------------------------------------------------

    def log_config(self, config: Any, name: str = "config"):
        """
        Save resolved configuration.

        Args:
            config: Config object (dataclass) or dict
            name: Config file name
        """
        config_path = self.output_dir / f"{name}_resolved.json"

        # Convert to dict
        if hasattr(config, '__dict__'):
            config_dict = self._config_to_dict(config)
        else:
            config_dict = dict(config)

        # Add metadata
        config_dict['_run_id'] = self.run_id
        config_dict['_timestamp'] = datetime.now().isoformat()

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

        self.logger.info(f"Config saved: {config_path}")

    def _config_to_dict(self, obj: Any) -> dict:
        """Recursively convert config to dict."""
        if hasattr(obj, '__dataclass_fields__'):
            return {k: self._config_to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, dict):
            return {k: self._config_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (list, tuple)):
            return [self._config_to_dict(v) for v in obj]
        else:
            return obj

    # -------------------------------------------------------------------------
    # DATA FINGERPRINT
    # -------------------------------------------------------------------------

    def log_data_fingerprint(
        self,
        X: np.ndarray,
        y: np.ndarray,
        name: str = "dataset"
    ) -> DataFingerprint:
        """
        Log data fingerprint for reproducibility tracking.

        Args:
            X: Data array
            y: Labels array
            name: Dataset name

        Returns:
            DataFingerprint object
        """
        fingerprint = DataFingerprint.from_data(X, y, name)
        self.fingerprints.append(fingerprint)

        self.logger.info(f"Data Fingerprint [{name}]:")
        self.logger.info(f"  Shape: {fingerprint.shape}")
        self.logger.info(f"  Samples: {fingerprint.n_samples}")
        self.logger.info(f"  Classes: {fingerprint.class_distribution}")
        self.logger.info(f"  Schema Hash: {fingerprint.schema_hash}")

        return fingerprint

    def _save_fingerprints(self):
        """Save all data fingerprints."""
        if not self.fingerprints:
            return

        fingerprint_path = self.output_dir / "data_fingerprints.json"
        fingerprints_dict = [fp.to_dict() for fp in self.fingerprints]

        with open(fingerprint_path, 'w') as f:
            json.dump(fingerprints_dict, f, indent=2, default=str)

    # -------------------------------------------------------------------------
    # TIMING
    # -------------------------------------------------------------------------

    @contextmanager
    def timer(self, stage: str):
        """
        Time a stage of execution.

        Usage:
            with run.timer("training"):
                train_model()
        """
        self.logger.info(f"Starting: {stage}")
        start = time.time()

        with self.timing.timer(stage):
            yield

        duration = time.time() - start
        self.logger.info(f"Completed: {stage} ({duration:.2f}s)")

    def _save_timings(self):
        """Save timing information."""
        timings_path = self.output_dir / "timings.json"

        timings_data = {
            'stages': self.timing.get_timings(),
            'total': self.timing.total_time(),
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat()
        }

        with open(timings_path, 'w') as f:
            json.dump(timings_data, f, indent=2)

    # -------------------------------------------------------------------------
    # METRICS
    # -------------------------------------------------------------------------

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics.

        Args:
            metrics: Dict of metric name -> value
            step: Optional step/epoch number
        """
        if step is not None:
            metrics['step'] = step

        metrics['timestamp'] = datetime.now().isoformat()

        # Store
        for k, v in metrics.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append(v)

        # Log
        metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in metrics.items() if k != 'timestamp')
        self.logger.info(f"Metrics: {metrics_str}")

    def _save_metrics(self):
        """Save metrics to JSON and CSV."""
        if not self.metrics:
            return

        # Save JSON
        json_path = self.output_dir / "metrics.json"
        with open(json_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        # Save CSV (if metrics have same length)
        try:
            import csv
            csv_path = self.output_dir / "metrics.csv"

            # Get keys and ensure all have same length
            keys = list(self.metrics.keys())
            lengths = [len(v) for v in self.metrics.values()]

            if len(set(lengths)) == 1:
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(keys)
                    for i in range(lengths[0]):
                        row = [self.metrics[k][i] for k in keys]
                        writer.writerow(row)
        except Exception:
            pass  # CSV is optional

    # -------------------------------------------------------------------------
    # ENVIRONMENT
    # -------------------------------------------------------------------------

    def _save_environment(self):
        """Save environment information."""
        env_path = self.output_dir / "environment.json"

        env_info = {
            'os': platform.system(),
            'os_release': platform.release(),
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'machine': platform.machine(),
            'git_hash': get_git_hash(short=False),
            'timestamp': datetime.now().isoformat(),
            'run_id': self.run_id
        }

        # Add package versions
        try:
            import torch
            env_info['torch_version'] = torch.__version__
            env_info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                env_info['cuda_version'] = torch.version.cuda
                env_info['gpu_name'] = torch.cuda.get_device_name(0)
        except ImportError:
            pass

        try:
            import numpy
            env_info['numpy_version'] = numpy.__version__
        except ImportError:
            pass

        with open(env_path, 'w') as f:
            json.dump(env_info, f, indent=2)

    # -------------------------------------------------------------------------
    # SHAPE TRACE
    # -------------------------------------------------------------------------

    def log_shape(self, name: str, data: Any):
        """Log data/tensor shape for debugging."""
        if hasattr(data, 'shape'):
            shape = data.shape
            dtype = data.dtype if hasattr(data, 'dtype') else type(data).__name__
            self.logger.debug(f"Shape [{name}]: {shape} ({dtype})")
        else:
            self.logger.debug(f"Shape [{name}]: {type(data).__name__}")

    # -------------------------------------------------------------------------
    # CONVENIENCE
    # -------------------------------------------------------------------------

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_run(experiment_name: str = "run", **kwargs) -> RunManager:
    """Create a new run manager."""
    return RunManager(experiment_name=experiment_name, **kwargs)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    # Test run manager
    print("Testing Run Manager...")

    with RunManager("test_experiment") as run:
        # Log config
        config = {"learning_rate": 0.001, "batch_size": 64}
        run.log_config(config)

        # Create sample data
        X = np.random.randn(100, 32, 512).astype(np.float32)
        y = np.random.randint(0, 2, size=100)

        # Log data fingerprint
        run.log_data_fingerprint(X, y, "test_data")

        # Time some operations
        with run.timer("data_loading"):
            time.sleep(0.1)

        with run.timer("training"):
            time.sleep(0.2)

        with run.timer("evaluation"):
            time.sleep(0.1)

        # Log metrics
        run.log_metrics({"accuracy": 0.95, "f1": 0.94, "auc": 0.98})

    print(f"\nOutputs saved to: outputs/")
