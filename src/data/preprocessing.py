#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
EEG Data Preprocessing Pipeline for GenAI-RAG-EEG
================================================================================

Module: preprocessing.py
Project: GenAI-RAG-EEG for Stress Classification
Author: Research Team
License: MIT

================================================================================
OVERVIEW
================================================================================

This module implements a complete EEG data preprocessing pipeline designed for
stress classification tasks. The pipeline follows best practices from
neurophysiology literature and is optimized for deep learning applications.

================================================================================
PREPROCESSING PIPELINE ARCHITECTURE
================================================================================

                    ┌─────────────────────────────────────────────────────┐
                    │           RAW EEG INPUT                             │
                    │       (channels × time_samples)                     │
                    │       e.g., (32 × 15360) for 60s @ 256Hz           │
                    └─────────────────────┬───────────────────────────────┘
                                          │
                                          ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  STEP 1: BANDPASS FILTER                                                │
    │  ════════════════════════════════════════════════════════════════════   │
    │                                                                         │
    │  Purpose: Remove DC offset and high-frequency noise                     │
    │                                                                         │
    │  Implementation:                                                        │
    │  • Butterworth IIR filter (4th order)                                  │
    │  • Zero-phase filtering (scipy.signal.filtfilt)                        │
    │  • Passband: 0.5 - 45 Hz                                               │
    │                                                                         │
    │  Transfer Function:                                                     │
    │                           1                                             │
    │  H(s) = ─────────────────────────────────────                          │
    │         (s² + √2·ωc·s + ωc²)²                                          │
    │                                                                         │
    │  Frequency Response:                                                    │
    │                                                                         │
    │  Gain │                    ┌───────────────┐                           │
    │  (dB) │                   ╱                 ╲                          │
    │    0 ─┼─────────────────┬┘                   └┬───────────────         │
    │  -20 ─┤                ╱                       ╲                       │
    │  -40 ─┤              ╱                           ╲                     │
    │       └──────┬───────┬─────────────────────┬─────┬─────► Freq (Hz)    │
    │             0.5      4                    45     100                   │
    │                                                                         │
    │  Preserves: Delta (0.5-4), Theta (4-8), Alpha (8-13),                  │
    │             Beta (13-30), Low Gamma (30-45)                            │
    └─────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  STEP 2: NOTCH FILTER                                                   │
    │  ════════════════════════════════════════════════════════════════════   │
    │                                                                         │
    │  Purpose: Remove power line interference                                │
    │                                                                         │
    │  Configuration:                                                         │
    │  • Europe/Asia: 50 Hz                                                  │
    │  • Americas: 60 Hz                                                     │
    │  • Quality factor: Q = 30                                              │
    │                                                                         │
    │  Notch Response:                                                        │
    │                                                                         │
    │  Gain │       ╲        ╱                                               │
    │  (dB) │        ╲      ╱                                                │
    │    0 ─┼─────────╲────╱─────────────────────────────                    │
    │  -20 ─┤          ╲  ╱                                                  │
    │  -40 ─┤           ╲╱                                                   │
    │       └────────────┬────────────────────────────────► Freq (Hz)        │
    │                   50 Hz                                                 │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  STEP 3: EPOCH SEGMENTATION                                             │
    │  ════════════════════════════════════════════════════════════════════   │
    │                                                                         │
    │  Purpose: Divide continuous EEG into fixed-length analysis windows     │
    │                                                                         │
    │  Parameters:                                                            │
    │  • Window size: 4 seconds (1024 samples @ 256 Hz)                      │
    │  • Overlap: 50% (512 samples step)                                     │
    │                                                                         │
    │  Sliding Window Visualization:                                          │
    │                                                                         │
    │  ├────────────────────────────────────────────────────────► Time       │
    │                                                                         │
    │  │◄──── Window 1 ────►│                                                │
    │  │        │◄──── Window 2 ────►│                                       │
    │  │        │        │◄──── Window 3 ────►│                              │
    │  │        │        │        │◄──── Window 4 ────►│                     │
    │  │        │        │        │        │                                 │
    │  └────────┴────────┴────────┴────────┴────────────                     │
    │           ↑        ↑        ↑        ↑                                 │
    │         50%      50%      50%      50%                                 │
    │       overlap   overlap  overlap  overlap                              │
    │                                                                         │
    │  Output: (n_epochs × channels × window_samples)                        │
    │          e.g., (28 × 32 × 1024) for 60s recording                      │
    │                                                                         │
    │  Number of epochs = ⌊(T - W) / S⌋ + 1                                  │
    │  where T = total samples, W = window size, S = step size               │
    └─────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  STEP 4: ARTIFACT REJECTION                                             │
    │  ════════════════════════════════════════════════════════════════════   │
    │                                                                         │
    │  Purpose: Remove epochs contaminated by artifacts                       │
    │                                                                         │
    │  Threshold-based Rejection:                                             │
    │  • Default threshold: ±100 μV (typical for research-grade EEG)         │
    │  • EEGMAT threshold: ±150 μV (consumer-grade devices)                  │
    │                                                                         │
    │  Artifact Types Detected:                                               │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │ Artifact Type   │ Characteristics           │ Typical Range    │   │
    │  ├─────────────────┼───────────────────────────┼──────────────────┤   │
    │  │ Eye blink       │ Sharp positive spike      │ 50-200 μV        │   │
    │  │ Eye movement    │ Slow drift                │ 20-100 μV        │   │
    │  │ Muscle (EMG)    │ High frequency bursts     │ 10-100 μV        │   │
    │  │ Movement        │ Large slow waves          │ >100 μV          │   │
    │  │ Electrode pop   │ Very sharp transient      │ >200 μV          │   │
    │  └─────────────────┴───────────────────────────┴──────────────────┘   │
    │                                                                         │
    │  Quality Control:                                                       │
    │  • Warning if rejection rate > 20%                                     │
    │  • Returns rejection statistics for reporting                          │
    └─────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  STEP 5: NORMALIZATION                                                  │
    │  ════════════════════════════════════════════════════════════════════   │
    │                                                                         │
    │  Purpose: Standardize data for neural network input                    │
    │                                                                         │
    │  Z-Score Normalization (Default):                                       │
    │                                                                         │
    │           x - μ                                                         │
    │  z = ───────────     where μ = mean, σ = std dev                       │
    │           σ                                                             │
    │                                                                         │
    │  Applied per-channel to preserve relative amplitude differences        │
    │                                                                         │
    │  Before:                    After:                                      │
    │    Ch1: μ=50, σ=20           Ch1: μ≈0, σ≈1                             │
    │    Ch2: μ=-10, σ=15          Ch2: μ≈0, σ≈1                             │
    │    Ch3: μ=30, σ=25           Ch3: μ≈0, σ≈1                             │
    │                                                                         │
    │  Alternative: Min-Max Normalization                                     │
    │                                                                         │
    │         x - min                                                         │
    │  x' = ───────────    → Range [0, 1]                                    │
    │        max - min                                                        │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────────────────┐
                    │           PREPROCESSED OUTPUT                       │
                    │       (n_epochs × channels × samples)               │
                    │       Ready for deep learning model                 │
                    └─────────────────────────────────────────────────────┘

================================================================================
FREQUENCY BANDS REFERENCE
================================================================================

    ┌───────────────────────────────────────────────────────────────────────┐
    │                    EEG FREQUENCY BANDS                                │
    ├───────────┬───────────────┬────────────────────────────────────────────┤
    │ Band      │ Frequency     │ Associated Mental States                  │
    ├───────────┼───────────────┼────────────────────────────────────────────┤
    │ Delta     │ 0.5 - 4 Hz    │ Deep sleep, unconsciousness               │
    │ Theta     │ 4 - 8 Hz      │ Drowsiness, light sleep, meditation       │
    │ Alpha     │ 8 - 13 Hz     │ Relaxed wakefulness, eyes closed          │
    │ Beta      │ 13 - 30 Hz    │ Active thinking, focus, anxiety, stress   │
    │ Gamma     │ 30 - 45 Hz    │ Higher cognitive functions, perception    │
    └───────────┴───────────────┴────────────────────────────────────────────┘

    Stress Indicators:
    • Decreased Alpha power (8-13 Hz) in frontal regions
    • Increased Beta power (13-30 Hz) indicating heightened arousal
    • Frontal Alpha Asymmetry (FAA) changes
    • Increased Theta/Beta ratio variations

================================================================================
DATASET-SPECIFIC CONFIGURATIONS
================================================================================

    ┌────────────┬───────────┬──────────┬───────────┬─────────────────────────┐
    │ Dataset    │ Channels  │ Fs (Hz)  │ Threshold │ Notes                   │
    ├────────────┼───────────┼──────────┼───────────┼─────────────────────────┤
    │ DEAP       │ 32        │ 128      │ 100 μV    │ Downsampled from 512 Hz │
    │ SAM-40     │ 32        │ 256      │ 100 μV    │ Research-grade EEG      │
    │ EEGMAT     │ 14        │ 128      │ 150 μV    │ Consumer-grade device   │
    └────────────┴───────────┴──────────┴───────────┴─────────────────────────┘

================================================================================
USAGE EXAMPLES
================================================================================

    Basic Usage:
    ```python
    from src.data.preprocessing import EEGPreprocessor, PreprocessingConfig

    # Create preprocessor with default settings
    preprocessor = EEGPreprocessor(fs=256.0)

    # Process raw EEG data
    result = preprocessor.process(raw_eeg, labels, return_stats=True)

    # Access results
    epochs = result['epochs']      # (n_epochs, channels, time)
    labels = result['labels']      # (n_epochs,)
    stats = result['stats']        # Processing statistics
    ```

    Custom Configuration:
    ```python
    config = PreprocessingConfig(
        lowcut=1.0,              # Higher lowcut for less drift
        highcut=40.0,            # Lower highcut for less noise
        window_size=2.0,         # Shorter epochs
        overlap=0.75,            # More overlap for data augmentation
        artifact_threshold=80.0  # Stricter artifact rejection
    )
    preprocessor = EEGPreprocessor(config=config, fs=256.0)
    ```

    Dataset-Specific:
    ```python
    from src.data.preprocessing import get_dataset_config

    config, fs = get_dataset_config("deap")
    preprocessor = EEGPreprocessor(config=config, fs=fs)
    ```

================================================================================
DEPENDENCIES
================================================================================

    Required:
    - numpy >= 1.21.0

    Optional (for full functionality):
    - scipy >= 1.7.0 (filtering, PSD computation)
    - mne >= 1.0.0 (ICA artifact removal)

================================================================================
REFERENCES
================================================================================

    [1] Delorme, A., & Makeig, S. (2004). EEGLAB: an open source toolbox for
        analysis of single-trial EEG dynamics. Journal of neuroscience methods.

    [2] Koelstra, S., et al. (2012). DEAP: A Database for Emotion Analysis
        using Physiological Signals. IEEE Transactions on Affective Computing.

    [3] Welch, P. (1967). The use of fast Fourier transform for the estimation
        of power spectra. IEEE Transactions on audio and electroacoustics.

================================================================================
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Union
from dataclasses import dataclass
from pathlib import Path
import warnings

try:
    from scipy import signal
    from scipy.stats import zscore
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Some preprocessing functions will be limited.")

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    warnings.warn("MNE-Python not available. ICA artifact removal will be disabled.")


@dataclass
class PreprocessingConfig:
    """Configuration for EEG preprocessing."""
    # Filtering
    lowcut: float = 0.5
    highcut: float = 45.0
    filter_order: int = 4
    notch_freq: float = 50.0  # Power line frequency (50Hz Europe, 60Hz Americas)

    # Segmentation
    window_size: float = 4.0  # seconds
    overlap: float = 0.5  # 50% overlap

    # Normalization
    normalize: bool = True
    normalize_method: str = "zscore"  # "zscore" or "minmax"

    # Re-referencing
    use_car: bool = True  # Common Average Reference

    # Artifact rejection
    artifact_threshold: float = 100.0  # μV threshold for epoch rejection
    use_ica: bool = True
    n_ica_components: int = 15


class CommonAverageReference:
    """
    Apply Common Average Reference (CAR) re-referencing.

    CAR subtracts the mean of all channels from each channel,
    reducing common noise and improving spatial resolution.
    """

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply CAR.

        Args:
            data: EEG data (channels, time) or (batch, channels, time)

        Returns:
            Re-referenced data
        """
        if data.ndim == 2:
            avg = data.mean(axis=0, keepdims=True)
            return data - avg
        elif data.ndim == 3:
            avg = data.mean(axis=1, keepdims=True)
            return data - avg
        else:
            return data


class BandpassFilter:
    """
    Butterworth bandpass filter for EEG signals.

    Default: 0.5-45 Hz, 4th order
    """

    def __init__(
        self,
        lowcut: float = 0.5,
        highcut: float = 45.0,
        fs: float = 256.0,
        order: int = 4
    ):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order

        if SCIPY_AVAILABLE:
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            self.b, self.a = signal.butter(order, [low, high], btype='band')

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter.

        Args:
            data: EEG data (channels, time) or (batch, channels, time)

        Returns:
            Filtered data with same shape
        """
        if not SCIPY_AVAILABLE:
            warnings.warn("scipy not available, returning unfiltered data")
            return data

        if data.ndim == 2:
            return signal.filtfilt(self.b, self.a, data, axis=1)
        elif data.ndim == 3:
            return np.array([
                signal.filtfilt(self.b, self.a, x, axis=1)
                for x in data
            ])
        else:
            raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")


class NotchFilter:
    """
    Notch filter for removing power line interference.

    Default: 50 Hz (or 60 Hz for Americas)
    """

    def __init__(
        self,
        freq: float = 50.0,
        fs: float = 256.0,
        quality: float = 30.0
    ):
        self.freq = freq
        self.fs = fs

        if SCIPY_AVAILABLE:
            self.b, self.a = signal.iirnotch(freq, quality, fs)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply notch filter."""
        if not SCIPY_AVAILABLE:
            return data

        if data.ndim == 2:
            return signal.filtfilt(self.b, self.a, data, axis=1)
        elif data.ndim == 3:
            return np.array([
                signal.filtfilt(self.b, self.a, x, axis=1)
                for x in data
            ])
        else:
            return data


class EpochSegmenter:
    """
    Segment continuous EEG into fixed-length epochs.

    Default: 4-second windows with 50% overlap
    """

    def __init__(
        self,
        window_size: float = 4.0,
        overlap: float = 0.5,
        fs: float = 256.0
    ):
        self.window_size = window_size
        self.overlap = overlap
        self.fs = fs

        self.window_samples = int(window_size * fs)
        self.step_samples = int(self.window_samples * (1 - overlap))

    def __call__(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Segment data into epochs.

        Args:
            data: Continuous EEG (channels, time)
            labels: Optional labels per time point

        Returns:
            epochs: (n_epochs, channels, window_samples)
            epoch_labels: (n_epochs,) if labels provided
        """
        n_channels, n_samples = data.shape

        # Calculate number of epochs
        n_epochs = (n_samples - self.window_samples) // self.step_samples + 1

        epochs = np.zeros((n_epochs, n_channels, self.window_samples))
        epoch_labels = np.zeros(n_epochs) if labels is not None else None

        for i in range(n_epochs):
            start = i * self.step_samples
            end = start + self.window_samples
            epochs[i] = data[:, start:end]

            if labels is not None:
                # Use majority label in window
                window_labels = labels[start:end]
                epoch_labels[i] = np.argmax(np.bincount(window_labels.astype(int)))

        return epochs, epoch_labels


class Normalizer:
    """
    Normalize EEG data.

    Supports z-score and min-max normalization.
    """

    def __init__(self, method: str = "zscore", per_channel: bool = True):
        self.method = method
        self.per_channel = per_channel

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data.

        Args:
            data: EEG data (channels, time) or (batch, channels, time)

        Returns:
            Normalized data
        """
        if self.method == "zscore":
            if self.per_channel:
                if data.ndim == 2:
                    return zscore(data, axis=1) if SCIPY_AVAILABLE else (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
                elif data.ndim == 3:
                    # Normalize each epoch per channel
                    return np.array([
                        (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
                        for x in data
                    ])
            else:
                mean = data.mean()
                std = data.std() + 1e-8
                return (data - mean) / std

        elif self.method == "minmax":
            if self.per_channel:
                if data.ndim == 2:
                    min_vals = data.min(axis=1, keepdims=True)
                    max_vals = data.max(axis=1, keepdims=True)
                    return (data - min_vals) / (max_vals - min_vals + 1e-8)
                elif data.ndim == 3:
                    return np.array([
                        (x - x.min(axis=1, keepdims=True)) / (x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True) + 1e-8)
                        for x in data
                    ])
            else:
                return (data - data.min()) / (data.max() - data.min() + 1e-8)

        else:
            raise ValueError(f"Unknown normalization method: {self.method}")


class ArtifactRejector:
    """
    Reject epochs with artifacts based on amplitude threshold.

    Default threshold: ±100 μV
    """

    def __init__(self, threshold: float = 100.0):
        self.threshold = threshold

    def __call__(
        self,
        epochs: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Reject epochs exceeding threshold.

        Args:
            epochs: (n_epochs, channels, time)
            labels: Optional epoch labels

        Returns:
            clean_epochs: Epochs within threshold
            clean_labels: Corresponding labels
            rejection_mask: Boolean mask of rejected epochs
        """
        # Check if any sample exceeds threshold
        max_amp = np.abs(epochs).max(axis=(1, 2))
        keep_mask = max_amp < self.threshold

        clean_epochs = epochs[keep_mask]
        clean_labels = labels[keep_mask] if labels is not None else None

        rejection_rate = 1 - keep_mask.mean()
        if rejection_rate > 0.2:
            warnings.warn(
                f"High rejection rate: {rejection_rate:.1%} of epochs rejected. "
                "Consider adjusting threshold or checking data quality."
            )

        return clean_epochs, clean_labels, ~keep_mask


class EEGPreprocessor:
    """
    Complete EEG preprocessing pipeline.

    Pipeline steps:
    1. Band-pass filtering (0.5-45 Hz)
    2. Notch filtering (50/60 Hz)
    3. Epoch segmentation (4s, 50% overlap)
    4. Artifact rejection (±100 μV)
    5. Z-score normalization
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None, fs: float = 256.0):
        self.config = config or PreprocessingConfig()
        self.fs = fs

        # Initialize components
        self.bandpass = BandpassFilter(
            lowcut=self.config.lowcut,
            highcut=self.config.highcut,
            fs=fs,
            order=self.config.filter_order
        )

        self.notch = NotchFilter(
            freq=self.config.notch_freq,
            fs=fs
        )

        self.car = CommonAverageReference() if self.config.use_car else None

        self.segmenter = EpochSegmenter(
            window_size=self.config.window_size,
            overlap=self.config.overlap,
            fs=fs
        )

        self.rejector = ArtifactRejector(
            threshold=self.config.artifact_threshold
        )

        self.normalizer = Normalizer(
            method=self.config.normalize_method
        )

    def process(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        return_stats: bool = False
    ) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Apply complete preprocessing pipeline.

        Args:
            data: Raw EEG (channels, time)
            labels: Optional labels per time point
            return_stats: Whether to return processing statistics

        Returns:
            Dictionary with:
            - epochs: Preprocessed epochs (n_epochs, channels, time)
            - labels: Epoch labels (if provided)
            - stats: Processing statistics (if requested)
        """
        stats = {}

        # 1. Common Average Reference (if enabled)
        if self.car is not None:
            data = self.car(data)

        # 2. Band-pass filtering
        filtered = self.bandpass(data)

        # 3. Notch filtering
        filtered = self.notch(filtered)

        # 4. Epoch segmentation
        epochs, epoch_labels = self.segmenter(filtered, labels)
        stats["n_epochs_total"] = len(epochs)

        # 5. Artifact rejection
        epochs, epoch_labels, rejected = self.rejector(epochs, epoch_labels)
        stats["n_epochs_rejected"] = rejected.sum()
        stats["n_epochs_retained"] = len(epochs)
        stats["rejection_rate"] = rejected.mean()

        # 6. Normalization
        if self.config.normalize:
            epochs = self.normalizer(epochs)

        result = {
            "epochs": epochs,
            "labels": epoch_labels
        }

        if return_stats:
            result["stats"] = stats

        return result


def get_dataset_config(dataset_name: str) -> Tuple[PreprocessingConfig, float]:
    """
    Get preprocessing configuration for specific dataset.

    Args:
        dataset_name: One of "deap", "sam40", "eegmat"

    Returns:
        config: PreprocessingConfig for the dataset
        sampling_rate: Dataset sampling rate
    """
    if dataset_name.lower() == "deap":
        config = PreprocessingConfig(
            lowcut=0.5,
            highcut=45.0,
            notch_freq=50.0,
            window_size=4.0,
            overlap=0.5,
            artifact_threshold=100.0,
            n_ica_components=15
        )
        fs = 128.0  # DEAP is downsampled to 128 Hz

    elif dataset_name.lower() == "sam40":
        config = PreprocessingConfig(
            lowcut=0.5,
            highcut=45.0,
            notch_freq=50.0,
            window_size=4.0,
            overlap=0.5,
            artifact_threshold=100.0,
            n_ica_components=15
        )
        fs = 256.0

    elif dataset_name.lower() == "eegmat":
        config = PreprocessingConfig(
            lowcut=0.5,
            highcut=45.0,
            notch_freq=50.0,
            window_size=4.0,
            overlap=0.5,
            artifact_threshold=150.0,  # Higher for consumer-grade device
            n_ica_components=10
        )
        fs = 128.0

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return config, fs


def compute_psd_features(
    epochs: np.ndarray,
    fs: float = 256.0,
    bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute power spectral density features for epochs.

    Args:
        epochs: (n_epochs, channels, time)
        fs: Sampling frequency
        bands: Frequency band definitions

    Returns:
        Dictionary of band power features
    """
    if not SCIPY_AVAILABLE:
        warnings.warn("scipy not available, cannot compute PSD features")
        return {}

    if bands is None:
        bands = {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 45.0)
        }

    n_epochs, n_channels, n_samples = epochs.shape

    features = {band: np.zeros((n_epochs, n_channels)) for band in bands}

    for i in range(n_epochs):
        for j in range(n_channels):
            # Compute PSD using Welch's method
            freqs, psd = signal.welch(epochs[i, j], fs=fs, nperseg=min(256, n_samples))

            # Compute band powers
            for band_name, (low, high) in bands.items():
                idx = np.where((freqs >= low) & (freqs <= high))[0]
                features[band_name][i, j] = np.trapz(psd[idx], freqs[idx])

    return features


if __name__ == "__main__":
    print("Testing EEG Preprocessing Pipeline")
    print("=" * 50)

    # Create synthetic EEG data
    np.random.seed(42)
    fs = 256  # Hz
    duration = 60  # seconds
    n_channels = 32
    n_samples = int(fs * duration)

    # Simulate EEG with some noise
    t = np.arange(n_samples) / fs
    data = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        # Alpha rhythm (10 Hz)
        data[ch] += 20 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
        # Beta rhythm (20 Hz)
        data[ch] += 10 * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)
        # Noise
        data[ch] += 5 * np.random.randn(n_samples)

    # Add some artifacts
    data[0, 5000:5100] = 150  # Artifact in channel 0

    # Create labels
    labels = np.zeros(n_samples)
    labels[n_samples // 2:] = 1  # Second half is "stress"

    print(f"Raw data shape: {data.shape}")
    print(f"Sampling rate: {fs} Hz")
    print(f"Duration: {duration} seconds")

    # Process
    preprocessor = EEGPreprocessor(fs=fs)
    result = preprocessor.process(data, labels, return_stats=True)

    print(f"\nProcessed epochs shape: {result['epochs'].shape}")
    print(f"Labels shape: {result['labels'].shape}")
    print(f"\nProcessing statistics:")
    for key, value in result["stats"].items():
        print(f"  {key}: {value}")

    # Compute features
    features = compute_psd_features(result["epochs"], fs=fs)
    print(f"\nPSD features computed:")
    for band, power in features.items():
        print(f"  {band}: shape {power.shape}, mean={power.mean():.4f}")
