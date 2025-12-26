#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Pytest Configuration and Shared Fixtures
================================================================================

This file contains fixtures and configuration shared across all tests.

================================================================================
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def device():
    """Get available compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_seed():
    """Set random seeds for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def sample_eeg_batch():
    """Generate batch of sample EEG data."""
    return torch.randn(8, 32, 512)


@pytest.fixture
def sample_eeg_single():
    """Generate single sample EEG data."""
    return torch.randn(1, 32, 512)


@pytest.fixture
def sample_labels():
    """Generate sample binary labels."""
    return torch.randint(0, 2, (8,))


# =============================================================================
# SKIP CONDITIONS
# =============================================================================

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Test requires GPU"
)

requires_transformers = pytest.mark.skipif(
    True,  # Will be updated based on import check
    reason="Test requires transformers library"
)

try:
    import transformers
    requires_transformers = pytest.mark.skipif(False, reason="")
except ImportError:
    pass
