#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GenAI-RAG-EEG Utilities Package
================================================================================

This package provides utility modules for the GenAI-RAG-EEG system:
- logger: Detailed command-line logging with progress tracking
- compatibility: Backward/forward compatibility layer
- run_manager: Experiment tracking & reproducibility

Author: GenAI-RAG-EEG Team
Version: 3.0.0
License: MIT
================================================================================
"""

from .logger import setup_logger, get_logger, LogLevel
from .compatibility import CompatibilityLayer, check_system_requirements
from .run_manager import RunManager, DataFingerprint, create_run, generate_run_id

__all__ = [
    # Logger
    'setup_logger',
    'get_logger',
    'LogLevel',
    # Compatibility
    'CompatibilityLayer',
    'check_system_requirements',
    # Run Manager
    'RunManager',
    'DataFingerprint',
    'create_run',
    'generate_run_id',
]
