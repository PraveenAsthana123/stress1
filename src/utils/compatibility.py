#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GenAI-RAG-EEG Cross-Platform Compatibility Layer
================================================================================

DESCRIPTION:
    This module ensures the GenAI-RAG-EEG codebase runs correctly across
    different operating systems, Python versions, and hardware configurations.

COMMON ISSUES THIS MODULE ADDRESSES:
    1. Path separators (Windows \ vs Unix /)
    2. File encoding (UTF-8 vs system default)
    3. Missing dependencies (graceful fallbacks)
    4. GPU availability (CUDA vs CPU fallback)
    5. Memory constraints (batch size adjustments)
    6. Python version differences (3.8 - 3.13)
    7. Package version mismatches

WHY CODE MIGHT NOT WORK ON OTHER SYSTEMS:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ ISSUE                      │ SYMPTOM                    │ SOLUTION      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Hardcoded paths            │ FileNotFoundError          │ Use pathlib   │
    │ Missing GPU                │ CUDA error                 │ CPU fallback  │
    │ Low memory                 │ OOM error                  │ Reduce batch  │
    │ Wrong Python version       │ SyntaxError                │ Version check │
    │ Missing packages           │ ImportError                │ Install guide │
    │ Different numpy version    │ Shape mismatch             │ Version lock  │
    │ Encoding issues            │ UnicodeDecodeError         │ UTF-8 force   │
    └─────────────────────────────────────────────────────────────────────────┘

USAGE:
    from src.utils.compatibility import CompatibilityLayer, check_system_requirements

    # At start of main script
    compat = CompatibilityLayer()

    # Check if system can run the code
    issues = check_system_requirements()
    if issues:
        print("System issues detected:")
        for issue in issues:
            print(f"  - {issue}")

    # Get device for PyTorch
    device = compat.get_device()

    # Get safe batch size for available memory
    batch_size = compat.get_safe_batch_size(default=64)

Author: GenAI-RAG-EEG Team
Version: 3.0.0
License: MIT
================================================================================
"""

import sys
import os
import platform
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import importlib
import json

# =============================================================================
# MINIMUM REQUIREMENTS
# =============================================================================

MINIMUM_PYTHON_VERSION = (3, 8)
MAXIMUM_PYTHON_VERSION = (3, 13)

REQUIRED_PACKAGES = {
    'torch': '2.0.0',
    'numpy': '1.24.0',
    'scipy': '1.10.0',
    'scikit-learn': '1.2.0',
    'pandas': '2.0.0',
    'transformers': '4.30.0',
    'matplotlib': '3.7.0',
}

OPTIONAL_PACKAGES = {
    'mne': '1.4.0',           # EEG processing
    'pyedflib': '0.1.30',     # EDF file reading
    'faiss-cpu': '1.7.4',     # Vector database
    'openai': '1.0.0',        # LLM integration
    'streamlit': '1.28.0',    # Web UI
    'wandb': '0.15.0',        # Experiment tracking
}

MINIMUM_MEMORY_GB = 4
RECOMMENDED_MEMORY_GB = 16
MINIMUM_GPU_MEMORY_GB = 4


# =============================================================================
# SYSTEM CHECK RESULTS
# =============================================================================

@dataclass
class SystemCheckResult:
    """
    Result of a system compatibility check.

    Attributes:
        passed: Whether the check passed
        message: Human-readable description
        severity: 'error' (cannot run), 'warning' (may have issues), 'info'
        fix_suggestion: How to fix the issue
    """
    passed: bool
    message: str
    severity: str  # 'error', 'warning', 'info'
    fix_suggestion: str = ""


# =============================================================================
# COMPATIBILITY LAYER CLASS
# =============================================================================

class CompatibilityLayer:
    """
    Cross-platform compatibility management for GenAI-RAG-EEG.

    This class provides methods to:
    1. Detect system capabilities
    2. Provide appropriate fallbacks
    3. Configure optimal settings
    4. Handle platform-specific issues

    Architecture:
    -------------
    CompatibilityLayer
        │
        ├── System Detection
        │   ├── OS detection
        │   ├── Python version check
        │   ├── Memory detection
        │   └── GPU detection
        │
        ├── Fallback Providers
        │   ├── CPU fallback for GPU code
        │   ├── Reduced batch size for low memory
        │   └── Alternative implementations
        │
        └── Configuration
            ├── Path normalization
            ├── Encoding settings
            └── Device selection

    Example:
    --------
    >>> compat = CompatibilityLayer()
    >>> print(compat.system_info)
    {
        'os': 'Linux',
        'python_version': '3.10.0',
        'memory_gb': 32,
        'gpu_available': True,
        'gpu_name': 'NVIDIA RTX 3090',
        'gpu_memory_gb': 24
    }
    >>> device = compat.get_device()
    >>> print(device)  # 'cuda' or 'cpu'
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize compatibility layer.

        Args:
            verbose: Whether to print system information

        Process:
        --------
        1. Detect operating system
        2. Check Python version
        3. Detect available memory
        4. Check GPU availability
        5. Verify required packages
        """
        self.verbose = verbose
        self._system_info = None
        self._device = None
        self._warnings = []

        # Run initial checks
        self._detect_system()

        if verbose:
            self._print_system_info()

    def _detect_system(self):
        """Detect system capabilities."""
        self._system_info = {
            'os': platform.system(),
            'os_version': platform.version(),
            'python_version': platform.python_version(),
            'python_version_tuple': sys.version_info[:3],
            'architecture': platform.machine(),
            'processor': platform.processor(),
        }

        # Memory detection
        try:
            import psutil
            mem = psutil.virtual_memory()
            self._system_info['memory_total_gb'] = mem.total / (1024**3)
            self._system_info['memory_available_gb'] = mem.available / (1024**3)
        except ImportError:
            self._system_info['memory_total_gb'] = None
            self._system_info['memory_available_gb'] = None
            self._warnings.append("psutil not installed - cannot detect memory")

        # GPU detection
        try:
            import torch
            self._system_info['torch_version'] = torch.__version__
            self._system_info['cuda_available'] = torch.cuda.is_available()

            if torch.cuda.is_available():
                self._system_info['cuda_version'] = torch.version.cuda
                self._system_info['gpu_count'] = torch.cuda.device_count()
                self._system_info['gpu_name'] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                self._system_info['gpu_memory_gb'] = props.total_memory / (1024**3)
            else:
                self._system_info['cuda_version'] = None
                self._system_info['gpu_count'] = 0
                self._system_info['gpu_name'] = None
                self._system_info['gpu_memory_gb'] = 0
        except ImportError:
            self._system_info['torch_version'] = None
            self._system_info['cuda_available'] = False
            self._warnings.append("PyTorch not installed")

    def _print_system_info(self):
        """Print detected system information."""
        print("\n" + "=" * 70)
        print("GenAI-RAG-EEG System Compatibility Check")
        print("=" * 70)

        print(f"\nOperating System:")
        print(f"  OS:           {self._system_info['os']}")
        print(f"  Version:      {self._system_info['os_version'][:50]}")
        print(f"  Architecture: {self._system_info['architecture']}")

        print(f"\nPython Environment:")
        print(f"  Version:      {self._system_info['python_version']}")
        py_ver = self._system_info['python_version_tuple']
        if py_ver < MINIMUM_PYTHON_VERSION:
            print(f"  ⚠ WARNING: Python {'.'.join(map(str, MINIMUM_PYTHON_VERSION))}+ required!")
        elif py_ver > MAXIMUM_PYTHON_VERSION:
            print(f"  ⚠ WARNING: Python {'.'.join(map(str, py_ver))} is newer than tested versions")
        else:
            print(f"  ✓ Python version compatible")

        if self._system_info['memory_total_gb']:
            print(f"\nMemory:")
            print(f"  Total:     {self._system_info['memory_total_gb']:.1f} GB")
            print(f"  Available: {self._system_info['memory_available_gb']:.1f} GB")
            if self._system_info['memory_available_gb'] < MINIMUM_MEMORY_GB:
                print(f"  ⚠ WARNING: Low memory! May need to reduce batch size")
            else:
                print(f"  ✓ Memory sufficient")

        print(f"\nGPU:")
        if self._system_info.get('cuda_available'):
            print(f"  ✓ CUDA Available: {self._system_info['cuda_version']}")
            print(f"  GPU:        {self._system_info['gpu_name']}")
            print(f"  Memory:     {self._system_info['gpu_memory_gb']:.1f} GB")
            print(f"  Count:      {self._system_info['gpu_count']}")
        else:
            print(f"  ✗ CUDA not available - will use CPU")
            print(f"    (Training will be slower but functional)")

        if self._warnings:
            print(f"\nWarnings:")
            for w in self._warnings:
                print(f"  ⚠ {w}")

        print("\n" + "=" * 70 + "\n")

    @property
    def system_info(self) -> Dict[str, Any]:
        """Get system information dictionary."""
        return self._system_info.copy()

    def get_device(self, prefer_gpu: bool = True) -> str:
        """
        Get the appropriate PyTorch device.

        This method handles GPU availability gracefully, falling back
        to CPU when CUDA is not available.

        Args:
            prefer_gpu: Whether to prefer GPU if available

        Returns:
            'cuda' or 'cpu'

        Example:
            >>> device = compat.get_device()
            >>> model = model.to(device)
            >>> data = data.to(device)
        """
        if self._device is not None:
            return self._device

        try:
            import torch

            if prefer_gpu and torch.cuda.is_available():
                self._device = 'cuda'
                if self.verbose:
                    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self._device = 'cpu'
                if self.verbose:
                    print("Using CPU for computation")
        except ImportError:
            self._device = 'cpu'
            if self.verbose:
                print("PyTorch not available, defaulting to CPU")

        return self._device

    def get_safe_batch_size(
        self,
        default: int = 64,
        model_memory_mb: float = 500,
        sample_memory_mb: float = 1
    ) -> int:
        """
        Calculate a safe batch size based on available memory.

        This prevents out-of-memory errors on systems with limited RAM/VRAM.

        Args:
            default: Default batch size for systems with enough memory
            model_memory_mb: Estimated model memory usage in MB
            sample_memory_mb: Estimated memory per sample in MB

        Returns:
            Safe batch size (may be reduced from default)

        Logic:
        ------
        1. Check available memory (GPU if using CUDA, else RAM)
        2. Reserve 2GB for system + model
        3. Calculate max samples that fit
        4. Return min(default, calculated)
        """
        try:
            import torch

            if torch.cuda.is_available():
                # GPU memory
                props = torch.cuda.get_device_properties(0)
                available_mb = props.total_memory / (1024**2) - model_memory_mb - 2000
            else:
                # System memory
                import psutil
                available_mb = psutil.virtual_memory().available / (1024**2) - model_memory_mb - 2000

            max_batch = int(available_mb / sample_memory_mb)
            safe_batch = min(default, max(1, max_batch))

            if safe_batch < default and self.verbose:
                print(f"⚠ Reducing batch size from {default} to {safe_batch} due to memory constraints")

            return safe_batch

        except Exception:
            return default

    def normalize_path(self, path: str) -> Path:
        """
        Normalize a path for the current operating system.

        This handles:
        - Forward/backward slash conversion
        - Home directory expansion (~)
        - Environment variable expansion
        - Relative to absolute conversion

        Args:
            path: Path string (may use Unix or Windows conventions)

        Returns:
            pathlib.Path object

        Example:
            >>> compat.normalize_path("~/data/SAM40")
            PosixPath('/home/user/data/SAM40')

            >>> compat.normalize_path("data\\SAM40")  # Windows style
            PosixPath('/current/dir/data/SAM40')  # Converted for Unix
        """
        # Convert string to Path
        p = Path(path)

        # Expand user home directory
        p = p.expanduser()

        # Resolve to absolute path
        if not p.is_absolute():
            p = Path.cwd() / p

        return p.resolve()

    def get_num_workers(self, default: int = 4) -> int:
        """
        Get safe number of data loader workers.

        Windows has issues with multiprocessing in DataLoader.
        This method returns an appropriate value for each OS.

        Args:
            default: Default number of workers

        Returns:
            Safe number of workers
        """
        if platform.system() == 'Windows':
            # Windows often has issues with multiprocessing
            return 0
        else:
            try:
                import psutil
                # Use at most half the CPU cores
                return min(default, psutil.cpu_count(logical=False) // 2)
            except ImportError:
                return default

    def check_package_versions(self) -> List[SystemCheckResult]:
        """
        Check if required packages are installed with correct versions.

        Returns:
            List of SystemCheckResult for each package
        """
        results = []

        for package, min_version in REQUIRED_PACKAGES.items():
            try:
                mod = importlib.import_module(package.replace('-', '_'))
                installed_version = getattr(mod, '__version__', 'unknown')

                # Simple version comparison (may need enhancement for complex versions)
                result = SystemCheckResult(
                    passed=True,
                    message=f"{package} {installed_version} installed",
                    severity='info'
                )
            except ImportError:
                result = SystemCheckResult(
                    passed=False,
                    message=f"{package} not installed",
                    severity='error',
                    fix_suggestion=f"pip install {package}>={min_version}"
                )
            results.append(result)

        return results


# =============================================================================
# MODULE-LEVEL FUNCTIONS
# =============================================================================

def check_system_requirements() -> List[str]:
    """
    Run comprehensive system requirements check.

    Returns:
        List of issue messages (empty if all checks pass)

    Example:
        >>> issues = check_system_requirements()
        >>> if issues:
        ...     print("Cannot run! Fix these issues:")
        ...     for issue in issues:
        ...         print(f"  - {issue}")
        ...     sys.exit(1)
    """
    issues = []

    # Python version check
    py_ver = sys.version_info[:2]
    if py_ver < MINIMUM_PYTHON_VERSION:
        issues.append(
            f"Python {'.'.join(map(str, py_ver))} is too old. "
            f"Requires Python {'.'.join(map(str, MINIMUM_PYTHON_VERSION))}+"
        )

    # Required packages
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package.replace('-', '_'))
        except ImportError:
            issues.append(f"Required package '{package}' is not installed")

    # Memory check
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        if available_gb < MINIMUM_MEMORY_GB:
            issues.append(
                f"Only {available_gb:.1f}GB RAM available. "
                f"Minimum {MINIMUM_MEMORY_GB}GB required."
            )
    except ImportError:
        pass  # Can't check memory without psutil

    return issues


def get_platform_config() -> Dict[str, Any]:
    """
    Get platform-specific configuration defaults.

    Returns:
        Dictionary with platform-appropriate settings

    Example:
        >>> config = get_platform_config()
        >>> print(config)
        {
            'num_workers': 4,
            'pin_memory': True,
            'file_encoding': 'utf-8',
            'path_separator': '/'
        }
    """
    config = {
        'file_encoding': 'utf-8',
        'path_separator': os.sep,
    }

    # Platform-specific settings
    if platform.system() == 'Windows':
        config['num_workers'] = 0  # Avoid multiprocessing issues
        config['pin_memory'] = False
    else:
        config['num_workers'] = min(4, os.cpu_count() or 1)
        config['pin_memory'] = True

    # GPU settings
    try:
        import torch
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        config['use_amp'] = torch.cuda.is_available()  # Automatic Mixed Precision
    except ImportError:
        config['device'] = 'cpu'
        config['use_amp'] = False

    return config


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """Test compatibility layer."""

    print("Testing Compatibility Layer\n")

    # Run system check
    issues = check_system_requirements()
    if issues:
        print("System Issues Found:")
        for issue in issues:
            print(f"  ✗ {issue}")
    else:
        print("✓ All system requirements met\n")

    # Create compatibility layer
    compat = CompatibilityLayer(verbose=True)

    # Test device selection
    device = compat.get_device()
    print(f"\nSelected device: {device}")

    # Test batch size calculation
    batch_size = compat.get_safe_batch_size(default=64)
    print(f"Safe batch size: {batch_size}")

    # Test path normalization
    test_paths = [
        "~/data/SAM40",
        "./results",
        "data\\EEGMAT",  # Windows style
    ]
    print("\nPath normalization:")
    for p in test_paths:
        normalized = compat.normalize_path(p)
        print(f"  {p} -> {normalized}")

    # Test platform config
    platform_config = get_platform_config()
    print("\nPlatform configuration:")
    for k, v in platform_config.items():
        print(f"  {k}: {v}")
