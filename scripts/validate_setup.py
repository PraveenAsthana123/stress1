#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GenAI-RAG-EEG Setup Validation Script
================================================================================

DESCRIPTION:
    Comprehensive validation script that checks if everything is properly
    set up to run the GenAI-RAG-EEG system. Run this script first to identify
    and fix any issues before running the main pipeline.

WHAT THIS SCRIPT CHECKS:
    1. Python version compatibility
    2. Required package installation
    3. Optional package availability
    4. Data directory structure
    5. Sample data presence
    6. GPU/CUDA availability
    7. Memory requirements
    8. Model loading test
    9. Configuration validity

COMMON ISSUES AND SOLUTIONS:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Issue                        │ Solution                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Python version too old       │ Install Python 3.8 or newer             │
    │ Missing torch                │ pip install torch                       │
    │ Missing data files           │ Run: python scripts/generate_sample_data.py │
    │ CUDA not available           │ Install CUDA toolkit or use CPU mode    │
    │ Out of memory                │ Reduce batch_size in config             │
    │ Permission denied            │ Check file/directory permissions        │
    └─────────────────────────────────────────────────────────────────────────┘

USAGE:
    python scripts/validate_setup.py

    # Verbose mode with detailed output
    python scripts/validate_setup.py --verbose

    # Fix issues automatically where possible
    python scripts/validate_setup.py --auto-fix

EXIT CODES:
    0: All checks passed
    1: Critical issues found (cannot run)
    2: Warnings found (may have issues)

Author: GenAI-RAG-EEG Team
Version: 3.0.0
License: MIT
================================================================================
"""

import sys
import os
import platform
import importlib
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# VALIDATION RESULT TYPES
# =============================================================================

class CheckStatus(Enum):
    """Status of a validation check."""
    PASS = "✓"
    WARN = "⚠"
    FAIL = "✗"
    SKIP = "○"


@dataclass
class CheckResult:
    """Result of a single validation check."""
    name: str
    status: CheckStatus
    message: str
    fix_command: Optional[str] = None
    details: Optional[str] = None


# =============================================================================
# REQUIREMENTS
# =============================================================================

REQUIRED_PYTHON_MIN = (3, 8)
REQUIRED_PYTHON_MAX = (3, 13)

REQUIRED_PACKAGES = [
    ('torch', '2.0.0', 'Deep learning framework'),
    ('numpy', '1.24.0', 'Numerical computing'),
    ('scipy', '1.10.0', 'Scientific computing'),
    ('pandas', '2.0.0', 'Data manipulation'),
    ('sklearn', '1.2.0', 'Machine learning utilities'),
    ('matplotlib', '3.7.0', 'Visualization'),
    ('transformers', '4.30.0', 'Text encoding models'),
]

OPTIONAL_PACKAGES = [
    ('mne', '1.4.0', 'EEG signal processing'),
    ('pyedflib', '0.1.30', 'EDF file reading'),
    ('faiss', '1.7.4', 'Vector database'),
    ('openai', '1.0.0', 'LLM integration'),
    ('streamlit', '1.28.0', 'Web UI'),
    ('wandb', '0.15.0', 'Experiment tracking'),
    ('rich', '13.0.0', 'Beautiful terminal output'),
    ('psutil', '5.9.0', 'System monitoring'),
]

REQUIRED_DIRECTORIES = [
    'data',
    'src',
    'src/models',
    'src/data',
    'src/utils',
]

REQUIRED_FILES = [
    'src/config.py',
    'src/models/eeg_encoder.py',
    'requirements.txt',
]


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def check_python_version() -> CheckResult:
    """Check Python version compatibility."""
    version = sys.version_info[:2]
    version_str = f"{version[0]}.{version[1]}"

    if version < REQUIRED_PYTHON_MIN:
        return CheckResult(
            name="Python Version",
            status=CheckStatus.FAIL,
            message=f"Python {version_str} is too old",
            fix_command=f"Install Python {REQUIRED_PYTHON_MIN[0]}.{REQUIRED_PYTHON_MIN[1]} or newer"
        )
    elif version > REQUIRED_PYTHON_MAX:
        return CheckResult(
            name="Python Version",
            status=CheckStatus.WARN,
            message=f"Python {version_str} is newer than tested versions",
            details="Code should work but may have minor issues"
        )
    else:
        return CheckResult(
            name="Python Version",
            status=CheckStatus.PASS,
            message=f"Python {version_str}"
        )


def check_package(name: str, min_version: str, description: str) -> CheckResult:
    """Check if a package is installed."""
    # Handle package name variations
    import_name = name.replace('-', '_')
    if name == 'sklearn':
        import_name = 'sklearn'
        pip_name = 'scikit-learn'
    elif name == 'faiss':
        import_name = 'faiss'
        pip_name = 'faiss-cpu'
    else:
        pip_name = name

    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return CheckResult(
            name=f"{name}",
            status=CheckStatus.PASS,
            message=f"v{version} ({description})"
        )
    except ImportError:
        return CheckResult(
            name=f"{name}",
            status=CheckStatus.FAIL,
            message=f"Not installed ({description})",
            fix_command=f"pip install {pip_name}>={min_version}"
        )


def check_directory(path: str) -> CheckResult:
    """Check if a directory exists."""
    full_path = PROJECT_ROOT / path
    if full_path.exists():
        return CheckResult(
            name=f"Directory: {path}",
            status=CheckStatus.PASS,
            message="Exists"
        )
    else:
        return CheckResult(
            name=f"Directory: {path}",
            status=CheckStatus.FAIL,
            message="Missing",
            fix_command=f"mkdir -p {path}"
        )


def check_file(path: str) -> CheckResult:
    """Check if a file exists."""
    full_path = PROJECT_ROOT / path
    if full_path.exists():
        size = full_path.stat().st_size
        return CheckResult(
            name=f"File: {path}",
            status=CheckStatus.PASS,
            message=f"Exists ({size:,} bytes)"
        )
    else:
        return CheckResult(
            name=f"File: {path}",
            status=CheckStatus.FAIL,
            message="Missing"
        )


def check_sample_data() -> CheckResult:
    """Check if sample data is available."""
    sample_dir = PROJECT_ROOT / 'data' / 'sample_validation'

    if not sample_dir.exists():
        return CheckResult(
            name="Sample Data",
            status=CheckStatus.WARN,
            message="Not generated",
            fix_command="python scripts/generate_sample_data.py"
        )

    # Check for sample files
    expected_files = ['sam40_sample.npz', 'wesad_sample.npz', 'eegmat_sample.npz']
    missing = [f for f in expected_files if not (sample_dir / f).exists()]

    if missing:
        return CheckResult(
            name="Sample Data",
            status=CheckStatus.WARN,
            message=f"Missing: {', '.join(missing)}",
            fix_command="python scripts/generate_sample_data.py"
        )

    return CheckResult(
        name="Sample Data",
        status=CheckStatus.PASS,
        message="All sample datasets available"
    )


def check_gpu() -> CheckResult:
    """Check GPU/CUDA availability."""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return CheckResult(
                name="GPU",
                status=CheckStatus.PASS,
                message=f"{gpu_name} ({gpu_mem:.1f} GB)",
                details=f"CUDA {torch.version.cuda}"
            )
        else:
            return CheckResult(
                name="GPU",
                status=CheckStatus.WARN,
                message="CUDA not available, will use CPU",
                details="Training will be slower but functional"
            )
    except ImportError:
        return CheckResult(
            name="GPU",
            status=CheckStatus.SKIP,
            message="Cannot check (PyTorch not installed)"
        )


def check_memory() -> CheckResult:
    """Check available system memory."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        avail_gb = mem.available / (1024**3)

        if avail_gb < 4:
            return CheckResult(
                name="Memory",
                status=CheckStatus.WARN,
                message=f"{avail_gb:.1f} GB available (low)",
                details=f"Total: {total_gb:.1f} GB. Consider closing other apps."
            )
        else:
            return CheckResult(
                name="Memory",
                status=CheckStatus.PASS,
                message=f"{avail_gb:.1f} GB available / {total_gb:.1f} GB total"
            )
    except ImportError:
        return CheckResult(
            name="Memory",
            status=CheckStatus.SKIP,
            message="Cannot check (psutil not installed)"
        )


def check_model_loading() -> CheckResult:
    """Test if the model can be loaded."""
    try:
        import torch
        from src.models.eeg_encoder import EEGEncoder

        model = EEGEncoder()
        n_params = sum(p.numel() for p in model.parameters())

        # Test forward pass
        x = torch.randn(2, 32, 512)
        with torch.no_grad():
            output, _ = model(x)

        return CheckResult(
            name="Model Loading",
            status=CheckStatus.PASS,
            message=f"EEGEncoder loaded ({n_params:,} params)",
            details=f"Output shape: {tuple(output.shape)}"
        )
    except Exception as e:
        return CheckResult(
            name="Model Loading",
            status=CheckStatus.FAIL,
            message=f"Failed: {str(e)[:50]}"
        )


def check_config() -> CheckResult:
    """Validate configuration module."""
    try:
        from src.config import Config, get_config

        config = get_config()

        # Check critical values
        issues = []
        if config.training.batch_size > 256:
            issues.append("Very large batch_size may cause OOM")
        if config.training.epochs > 500:
            issues.append("Very high epochs may cause overfitting")

        if issues:
            return CheckResult(
                name="Configuration",
                status=CheckStatus.WARN,
                message="Loaded with warnings",
                details="; ".join(issues)
            )
        else:
            return CheckResult(
                name="Configuration",
                status=CheckStatus.PASS,
                message="Valid configuration loaded"
            )
    except Exception as e:
        return CheckResult(
            name="Configuration",
            status=CheckStatus.FAIL,
            message=f"Error: {str(e)[:50]}"
        )


# =============================================================================
# MAIN VALIDATION
# =============================================================================

def run_validation(verbose: bool = False) -> Tuple[List[CheckResult], int]:
    """
    Run all validation checks.

    Args:
        verbose: Whether to print detailed output

    Returns:
        Tuple of (results list, exit code)
    """
    results = []

    # Section 1: Python Environment
    print("\n" + "=" * 60)
    print("1. Python Environment")
    print("=" * 60)

    results.append(check_python_version())
    print_result(results[-1], verbose)

    # Section 2: Required Packages
    print("\n" + "=" * 60)
    print("2. Required Packages")
    print("=" * 60)

    for name, version, desc in REQUIRED_PACKAGES:
        result = check_package(name, version, desc)
        results.append(result)
        print_result(result, verbose)

    # Section 3: Optional Packages
    print("\n" + "=" * 60)
    print("3. Optional Packages")
    print("=" * 60)

    for name, version, desc in OPTIONAL_PACKAGES:
        result = check_package(name, version, desc)
        # Don't fail for optional packages
        if result.status == CheckStatus.FAIL:
            result.status = CheckStatus.SKIP
        results.append(result)
        print_result(result, verbose)

    # Section 4: Project Structure
    print("\n" + "=" * 60)
    print("4. Project Structure")
    print("=" * 60)

    for directory in REQUIRED_DIRECTORIES:
        result = check_directory(directory)
        results.append(result)
        print_result(result, verbose)

    for file in REQUIRED_FILES:
        result = check_file(file)
        results.append(result)
        print_result(result, verbose)

    # Section 5: Data
    print("\n" + "=" * 60)
    print("5. Data")
    print("=" * 60)

    results.append(check_sample_data())
    print_result(results[-1], verbose)

    # Section 6: Hardware
    print("\n" + "=" * 60)
    print("6. Hardware")
    print("=" * 60)

    results.append(check_gpu())
    print_result(results[-1], verbose)

    results.append(check_memory())
    print_result(results[-1], verbose)

    # Section 7: Functionality
    print("\n" + "=" * 60)
    print("7. Functionality Tests")
    print("=" * 60)

    results.append(check_model_loading())
    print_result(results[-1], verbose)

    results.append(check_config())
    print_result(results[-1], verbose)

    # Calculate exit code
    n_pass = sum(1 for r in results if r.status == CheckStatus.PASS)
    n_warn = sum(1 for r in results if r.status == CheckStatus.WARN)
    n_fail = sum(1 for r in results if r.status == CheckStatus.FAIL)
    n_skip = sum(1 for r in results if r.status == CheckStatus.SKIP)

    if n_fail > 0:
        exit_code = 1
    elif n_warn > 0:
        exit_code = 2
    else:
        exit_code = 0

    return results, exit_code


def print_result(result: CheckResult, verbose: bool):
    """Print a single check result."""
    status_symbol = result.status.value
    print(f"  {status_symbol} {result.name}: {result.message}")

    if verbose and result.details:
        print(f"      Details: {result.details}")

    if verbose and result.fix_command:
        print(f"      Fix: {result.fix_command}")


def print_summary(results: List[CheckResult], exit_code: int):
    """Print validation summary."""
    n_pass = sum(1 for r in results if r.status == CheckStatus.PASS)
    n_warn = sum(1 for r in results if r.status == CheckStatus.WARN)
    n_fail = sum(1 for r in results if r.status == CheckStatus.FAIL)
    n_skip = sum(1 for r in results if r.status == CheckStatus.SKIP)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"\n  ✓ Passed:  {n_pass}")
    print(f"  ⚠ Warning: {n_warn}")
    print(f"  ✗ Failed:  {n_fail}")
    print(f"  ○ Skipped: {n_skip}")

    if exit_code == 0:
        print("\n  ✓ All checks passed! Ready to run.")
    elif exit_code == 1:
        print("\n  ✗ Critical issues found. Fix required issues before running.")
        print("\n  Failed checks:")
        for r in results:
            if r.status == CheckStatus.FAIL:
                print(f"    - {r.name}: {r.message}")
                if r.fix_command:
                    print(f"      Fix: {r.fix_command}")
    else:
        print("\n  ⚠ Warnings found. Code may run but could have issues.")

    print("\n" + "=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate GenAI-RAG-EEG setup',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit codes:
  0: All checks passed
  1: Critical issues (cannot run)
  2: Warnings (may have issues)

Examples:
  python validate_setup.py
  python validate_setup.py --verbose
  python validate_setup.py --auto-fix
        """
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show detailed output for each check'
    )
    parser.add_argument(
        '--auto-fix', action='store_true',
        help='Attempt to fix issues automatically'
    )
    parser.add_argument(
        '--json', action='store_true',
        help='Output results as JSON'
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("GenAI-RAG-EEG Setup Validation")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Run validation
    results, exit_code = run_validation(verbose=args.verbose)

    # Print summary
    print_summary(results, exit_code)

    # JSON output if requested
    if args.json:
        json_output = {
            'timestamp': datetime.now().isoformat(),
            'exit_code': exit_code,
            'results': [
                {
                    'name': r.name,
                    'status': r.status.name,
                    'message': r.message,
                    'fix_command': r.fix_command,
                    'details': r.details
                }
                for r in results
            ]
        }
        print("\nJSON Output:")
        print(json.dumps(json_output, indent=2))

    # Auto-fix if requested
    if args.auto_fix:
        failed = [r for r in results if r.status == CheckStatus.FAIL and r.fix_command]
        if failed:
            print("\nAttempting auto-fix...")
            for r in failed:
                if r.fix_command.startswith('pip install'):
                    print(f"  Running: {r.fix_command}")
                    os.system(r.fix_command)
                elif r.fix_command.startswith('mkdir'):
                    print(f"  Running: {r.fix_command}")
                    os.system(r.fix_command)
                elif r.fix_command.startswith('python'):
                    print(f"  Running: {r.fix_command}")
                    os.system(r.fix_command)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
