#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Unit Tests for Data Processing
================================================================================

Tests for:
- Sample data generation
- Data preprocessing
- Dataset classes

Run:
    pytest tests/test_data.py -v

================================================================================
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# SAMPLE DATA TESTS
# =============================================================================

class TestSampleDataGeneration:
    """Test sample EEG data generation."""

    def test_sample_data_import(self):
        """Test sample data module can be imported."""
        try:
            from data.sample.sample_eeg_data import generate_sample_data
            assert callable(generate_sample_data)
        except ImportError:
            pytest.skip("Sample data module not available")

    def test_generate_sample_data_shapes(self):
        """Test generated data has correct shapes."""
        try:
            from data.sample.sample_eeg_data import generate_sample_data

            X_train, y_train, X_test, y_test = generate_sample_data(n_samples=100)

            # Check shapes
            assert X_train.ndim == 3
            assert X_train.shape[1] == 32  # channels
            assert X_train.shape[2] == 512  # time samples

            # Check label shapes
            assert len(y_train) == X_train.shape[0]
            assert len(y_test) == X_test.shape[0]

        except ImportError:
            pytest.skip("Sample data module not available")

    def test_generate_sample_data_labels(self):
        """Test labels are binary."""
        try:
            from data.sample.sample_eeg_data import generate_sample_data

            _, y_train, _, y_test = generate_sample_data(n_samples=100)

            # Check binary labels
            assert set(np.unique(y_train)).issubset({0, 1})
            assert set(np.unique(y_test)).issubset({0, 1})

        except ImportError:
            pytest.skip("Sample data module not available")

    def test_generate_sample_data_balanced(self):
        """Test data is approximately balanced."""
        try:
            from data.sample.sample_eeg_data import generate_sample_data

            _, y_train, _, y_test = generate_sample_data(n_samples=100)

            # Check roughly balanced
            train_ratio = y_train.sum() / len(y_train)
            assert 0.3 < train_ratio < 0.7

        except ImportError:
            pytest.skip("Sample data module not available")

    def test_generate_single_channel_eeg(self):
        """Test single channel EEG generation."""
        try:
            from data.sample.sample_eeg_data import generate_single_channel_eeg

            signal = generate_single_channel_eeg(n_samples=512, is_stress=True)

            assert signal.shape == (512,)
            assert not np.isnan(signal).any()
            assert not np.isinf(signal).any()

        except ImportError:
            pytest.skip("Sample data module not available")

    def test_generate_multichannel_eeg(self):
        """Test multi-channel EEG generation."""
        try:
            from data.sample.sample_eeg_data import generate_multichannel_eeg

            eeg = generate_multichannel_eeg(n_samples=512, n_channels=32, is_stress=False)

            assert eeg.shape == (32, 512)
            assert not np.isnan(eeg).any()
            assert not np.isinf(eeg).any()

        except ImportError:
            pytest.skip("Sample data module not available")


# =============================================================================
# PREPROCESSING TESTS
# =============================================================================

class TestPreprocessing:
    """Test data preprocessing functions."""

    def test_zscore_normalization(self):
        """Test z-score normalization."""
        # Create sample data
        data = np.random.randn(32, 512) * 10 + 5  # Non-zero mean/std

        # Apply z-score
        normalized = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)

        # Check properties
        assert np.allclose(normalized.mean(axis=1), 0, atol=1e-10)
        assert np.allclose(normalized.std(axis=1), 1, atol=1e-10)

    def test_bandpass_filter_parameters(self):
        """Test bandpass filter produces reasonable output."""
        from scipy.signal import butter, filtfilt

        # Create sample signal with known frequencies
        fs = 256
        t = np.arange(0, 2, 1/fs)

        # 10 Hz (alpha) + 50 Hz (noise)
        signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t)

        # Apply bandpass filter (0.5-45 Hz)
        b, a = butter(4, [0.5, 45], btype='band', fs=fs)
        filtered = filtfilt(b, a, signal)

        # 50 Hz should be attenuated
        # Check filtered signal has lower high-frequency content
        fft_original = np.abs(np.fft.rfft(signal))
        fft_filtered = np.abs(np.fft.rfft(filtered))

        # High frequency content should be reduced
        high_freq_idx = int(50 * len(t) / fs)
        assert fft_filtered[high_freq_idx] < fft_original[high_freq_idx]


# =============================================================================
# CONTEXT GENERATION TESTS
# =============================================================================

class TestContextGeneration:
    """Test context string generation."""

    def test_generate_sample_contexts(self):
        """Test sample context generation."""
        try:
            from data.sample.sample_eeg_data import generate_sample_contexts

            contexts = generate_sample_contexts(n_samples=10)

            assert len(contexts) == 10
            for ctx in contexts:
                assert "Task:" in ctx
                assert isinstance(ctx, str)

        except ImportError:
            pytest.skip("Sample data module not available")


# =============================================================================
# DATA VALIDATION TESTS
# =============================================================================

class TestDataValidation:
    """Test data validation and sanity checks."""

    def test_eeg_value_range(self):
        """Test EEG values are in reasonable range after normalization."""
        try:
            from data.sample.sample_eeg_data import generate_sample_data

            X_train, _, _, _ = generate_sample_data(n_samples=20)

            # Z-scored data should mostly be within [-5, 5]
            assert X_train.min() > -100
            assert X_train.max() < 100

        except ImportError:
            pytest.skip("Sample data module not available")

    def test_no_constant_channels(self):
        """Test no channels are constant (all same value)."""
        try:
            from data.sample.sample_eeg_data import generate_multichannel_eeg

            eeg = generate_multichannel_eeg()

            # Check each channel has variance
            for ch in range(eeg.shape[0]):
                assert eeg[ch].std() > 0, f"Channel {ch} is constant"

        except ImportError:
            pytest.skip("Sample data module not available")


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
