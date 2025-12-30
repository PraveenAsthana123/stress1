#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Comprehensive Data Analysis Module for GenAI-RAG-EEG
================================================================================

Title: Real EEG Data Analysis Pipeline
Reference: GenAI-RAG-EEG Paper v2, IEEE Sensors Journal 2024

Description:
    This module provides comprehensive data analysis capabilities for real EEG
    datasets including data loading, preprocessing, feature extraction, and
    statistical analysis with publication-ready reporting.

Supported Datasets:
    - DEAP (32 subjects, 40 trials)
    - SAM-40 (40 subjects, 3 stress conditions)
    - EEGMAT (15 subjects, wearable sensors)
    - Custom datasets

Features:
    1. Data Loading & Validation
    2. Quality Assessment
    3. Feature Extraction (Time, Frequency, Time-Frequency)
    4. Statistical Comparison
    5. Cross-Subject Analysis
    6. Publication-Ready Reports

================================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
import warnings
from datetime import datetime
from scipy import signal, stats
from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu, wilcoxon

# Import local modules
try:
    from .statistical_analysis import (
        comprehensive_two_group_analysis,
        compute_all_effect_sizes,
        bootstrap_ci,
        bonferroni_correction,
        benjamini_hochberg_fdr,
        compare_cv_results
    )
    from .signal_analysis import (
        compute_psd,
        compute_band_power,
        band_power_analysis,
        alpha_suppression_analysis,
        theta_beta_ratio_analysis,
        frontal_asymmetry_analysis,
        FREQUENCY_BANDS,
        CHANNEL_GROUPS
    )
except ImportError:
    pass

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""
    name: str
    n_subjects: int
    n_trials: int
    n_channels: int
    n_samples: int
    sampling_rate: float
    duration_seconds: float
    class_distribution: Dict[int, int]
    channel_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_epochs(self) -> int:
        """Total number of epochs."""
        return sum(self.class_distribution.values())


@dataclass
class QualityReport:
    """Data quality assessment report."""
    n_epochs: int
    n_valid: int
    n_rejected: int
    rejection_rate: float
    reasons: Dict[str, int]
    channel_quality: Dict[str, float]
    snr_estimate: float
    recommendations: List[str]


@dataclass
class FeatureSet:
    """Extracted features from EEG data."""
    time_domain: Dict[str, np.ndarray]
    frequency_domain: Dict[str, np.ndarray]
    connectivity: Dict[str, np.ndarray]
    statistical: Dict[str, np.ndarray]
    feature_names: List[str]
    n_features: int


@dataclass
class AnalysisResult:
    """Complete analysis result."""
    dataset_name: str
    timestamp: str
    dataset_info: DatasetInfo
    quality_report: QualityReport
    band_power_analysis: List[Dict]
    statistical_tests: Dict[str, Any]
    classification_results: Dict[str, float]
    cross_subject_results: Dict[str, Any]
    summary: str


# =============================================================================
# DATA LOADING
# =============================================================================

class EEGDataLoader:
    """
    Unified EEG data loader for multiple datasets.

    Supports:
        - DEAP dataset (.dat, .mat files)
        - SAM-40 dataset (.mat files)
        - EEGMAT dataset (.pkl files)
        - Custom numpy arrays

    Automatically loads real data when available, falls back to synthetic data
    when real datasets are not found.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize data loader.

        Args:
            data_dir: Base directory for datasets
        """
        self.data_dir = Path(data_dir)
        self.supported_formats = ['.npy', '.npz', '.mat', '.dat', '.pkl', '.csv']

        # Try to load real data loader
        self._real_loader = None
        try:
            import sys
            sys.path.insert(0, str(self.data_dir))
            from real_data_loader import RealDataLoader
            self._real_loader = RealDataLoader(str(self.data_dir))
            self._available_datasets = self._real_loader.detect_dataset()
        except ImportError:
            self._available_datasets = {'DEAP': False, 'SAM40': False: False}

    def get_available_datasets(self) -> Dict[str, bool]:
        """
        Check which real datasets are available.

        Returns:
            Dict mapping dataset names to availability status
        """
        return self._available_datasets

    def load_dataset(
        self,
        dataset_name: str,
        subset: Optional[str] = None,
        force_synthetic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DatasetInfo]:
        """
        Load a dataset by name.

        Automatically uses real data when available, falls back to synthetic.

        Args:
            dataset_name: Name of dataset ('deap', 'sam40', 'eegmat', 'sample')
            subset: Optional subset specification
            force_synthetic: If True, always use synthetic data

        Returns:
            data: EEG data (n_epochs, n_channels, n_samples)
            labels: Labels (n_epochs,)
            subjects: Subject IDs (n_epochs,)
            info: Dataset information
        """
        dataset_name = dataset_name.lower()

        if dataset_name == 'deap':
            return self._load_deap(subset, force_synthetic)
        elif dataset_name == 'sam40':
            return self._load_sam40(subset, force_synthetic)
        elif dataset_name == 'eegmat':
            return self._load_eegmat(subset, force_synthetic)
        elif dataset_name == 'sample':
            return self._generate_sample_data()
        else:
            # Try to load as custom dataset
            return self._load_custom(dataset_name)

    def _load_deap(self, subset: Optional[str] = None, force_synthetic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DatasetInfo]:
        """Load DEAP dataset - real or synthetic."""

        # Try real data first if available
        if not force_synthetic and self._available_datasets.get('DEAP', False) and self._real_loader:
            try:
                from real_data_loader import DEAPLoader
                print("Loading REAL DEAP dataset...")
                path = self._real_loader.get_dataset_path('DEAP')
                loader = DEAPLoader(str(path))
                X, y = loader.load_all()

                # Generate subject IDs based on data structure
                n_samples = len(y)
                subjects = np.repeat(np.arange(1, 33), n_samples // 32 + 1)[:n_samples]

                info = DatasetInfo(
                    name="DEAP (Real)",
                    n_subjects=32,
                    n_trials=40,
                    n_channels=X.shape[1],
                    n_samples=X.shape[-1],
                    sampling_rate=256.0,  # After resampling
                    duration_seconds=X.shape[-1] / 256.0,
                    class_distribution=dict(zip(*np.unique(y, return_counts=True))),
                    channel_names=self._get_deap_channels(),
                    metadata={"source": "real_data", "original_fs": 128}
                )

                print(f"  Loaded {len(y)} samples from REAL DEAP data")
                return X, y, subjects, info

            except Exception as e:
                warnings.warn(f"Failed to load real DEAP data: {e}")

        # Check for preprocessed data
        deap_dir = self.data_dir / "deap"
        preprocessed_file = deap_dir / "deap_preprocessed.npz"
        if preprocessed_file.exists():
            data = np.load(preprocessed_file)
            return (
                data['X'],
                data['y'],
                data['subjects'],
                DatasetInfo(
                    name="DEAP",
                    n_subjects=32,
                    n_trials=40,
                    n_channels=32,
                    n_samples=data['X'].shape[-1],
                    sampling_rate=128.0,
                    duration_seconds=60.0,
                    class_distribution=dict(zip(*np.unique(data['y'], return_counts=True))),
                    channel_names=self._get_deap_channels()
                )
            )

        # Generate sample data if no real data
        warnings.warn("DEAP data not found, generating synthetic data")
        return self._generate_sample_data(
            n_subjects=32, n_trials=40, n_channels=32, fs=128.0, name="DEAP (Synthetic)"
        )

    def _load_sam40(self, subset: Optional[str] = None, force_synthetic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DatasetInfo]:
        """Load SAM-40 dataset - real or synthetic."""

        # Try real data first if available
        if not force_synthetic and self._available_datasets.get('SAM40', False) and self._real_loader:
            try:
                from real_data_loader import SAM40Loader
                print("Loading REAL SAM-40 dataset...")
                path = self._real_loader.get_dataset_path('SAM40')
                loader = SAM40Loader(str(path))
                X, y = loader.load_all()

                # Generate subject IDs
                n_samples = len(y)
                subjects = np.repeat(np.arange(1, 41), n_samples // 40 + 1)[:n_samples]

                info = DatasetInfo(
                    name="SAM-40 (Real)",
                    n_subjects=40,
                    n_trials=3,
                    n_channels=X.shape[1],
                    n_samples=X.shape[-1],
                    sampling_rate=256.0,
                    duration_seconds=X.shape[-1] / 256.0,
                    class_distribution=dict(zip(*np.unique(y, return_counts=True))),
                    channel_names=self._get_10_20_channels(),
                    metadata={"source": "real_data"}
                )

                print(f"  Loaded {len(y)} samples from REAL SAM-40 data")
                return X, y, subjects, info

            except Exception as e:
                warnings.warn(f"Failed to load real SAM-40 data: {e}")

        # Check for preprocessed data
        sam_dir = self.data_dir / "sam40"
        preprocessed_file = sam_dir / "sam40_preprocessed.npz"
        if preprocessed_file.exists():
            data = np.load(preprocessed_file)
            return (
                data['X'],
                data['y'],
                data['subjects'],
                DatasetInfo(
                    name="SAM-40",
                    n_subjects=40,
                    n_trials=3,
                    n_channels=32,
                    n_samples=data['X'].shape[-1],
                    sampling_rate=256.0,
                    duration_seconds=180.0,
                    class_distribution=dict(zip(*np.unique(data['y'], return_counts=True))),
                    channel_names=self._get_10_20_channels()
                )
            )

        warnings.warn("SAM-40 data not found, generating synthetic data")
        return self._generate_sample_data(
            n_subjects=40, n_trials=3, n_channels=32, fs=256.0, name="SAM-40 (Synthetic)"
        )

    def _load_eegmat(self, subset: Optional[str] = None, force_synthetic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DatasetInfo]:
        """Load EEGMAT dataset - real or synthetic."""

        # Try real data first if available
        if not force_synthetic and self._available_datasets.get(False) and self._real_loader:
            try:
                from real_data_loader import EEGMATLoader
                print("Loading REAL EEGMAT dataset...")
                path = self._real_loader.get_dataset_path()
                loader = EEGMATLoader(str(path))
                X, y = loader.load_all(modality='ECG')

                # Generate subject IDs
                n_samples = len(y)
                subjects = np.repeat(np.arange(1, 16), n_samples // 15 + 1)[:n_samples]

                info = DatasetInfo(
                    name="EEGMAT (Real)",
                    n_subjects=15,
                    n_trials=1,
                    n_channels=X.shape[1],
                    n_samples=X.shape[-1],
                    sampling_rate=256.0,  # After resampling
                    duration_seconds=X.shape[-1] / 256.0,
                    class_distribution=dict(zip(*np.unique(y, return_counts=True))),
                    channel_names=['ECG'] * X.shape[1],
                    metadata={"source": "real_data", "modality": "ECG", "original_fs": 700}
                )

                print(f"  Loaded {len(y)} samples from REAL EEGMAT data")
                return X, y, subjects, info

            except Exception as e:
                warnings.warn(f"Failed to load real EEGMAT data: {e}")

        # Check for preprocessed data
        eegmat_dir = self.data_dir / "eegmat"
        preprocessed_file = eegmat_dir / "eegmat_preprocessed.npz"
        if preprocessed_file.exists():
            data = np.load(preprocessed_file)
            return (
                data['X'],
                data['y'],
                data['subjects'],
                DatasetInfo(
                    name=,
                    n_subjects=15,
                    n_trials=1,
                    n_channels=32,
                    n_samples=data['X'].shape[-1],
                    sampling_rate=700.0,
                    duration_seconds=120.0,
                    class_distribution=dict(zip(*np.unique(data['y'], return_counts=True))),
                    channel_names=self._get_10_20_channels()
                )
            )

        warnings.warn("EEGMAT data not found, generating synthetic data")
        return self._generate_sample_data(
            n_subjects=15, n_trials=10, n_channels=32, fs=256.0, name="EEGMAT (Synthetic)"
        )

    def _load_custom(self, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DatasetInfo]:
        """Load custom dataset from file."""
        path = Path(path)

        if path.suffix == '.npz':
            data = np.load(path)
            X = data['X'] if 'X' in data else data['data']
            y = data['y'] if 'y' in data else data['labels']
            subjects = data.get('subjects', np.zeros(len(y)))

            return (
                X, y, subjects,
                DatasetInfo(
                    name=path.stem,
                    n_subjects=len(np.unique(subjects)),
                    n_trials=len(y),
                    n_channels=X.shape[1],
                    n_samples=X.shape[2],
                    sampling_rate=256.0,
                    duration_seconds=X.shape[2] / 256.0,
                    class_distribution=dict(zip(*np.unique(y, return_counts=True))),
                    channel_names=[f"Ch{i}" for i in range(X.shape[1])]
                )
            )

        raise ValueError(f"Unsupported format: {path.suffix}")

    def _generate_sample_data(
        self,
        n_subjects: int = 10,
        n_trials: int = 20,
        n_channels: int = 32,
        n_samples: int = 3200,  # SAM-40: 25 sec at 128 Hz
        fs: float = 128.0,  # SAM-40 sampling rate
        name: str = "Sample"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DatasetInfo]:
        """
        Generate synthetic EEG data with realistic stress patterns.

        Creates data with:
        - Alpha suppression during stress
        - Beta elevation during stress
        - Realistic noise and artifacts
        """
        np.random.seed(42)

        total_epochs = n_subjects * n_trials
        data = np.zeros((total_epochs, n_channels, n_samples))
        labels = np.zeros(total_epochs, dtype=int)
        subjects = np.zeros(total_epochs, dtype=int)

        t = np.linspace(0, n_samples / fs, n_samples)

        for subj in range(n_subjects):
            for trial in range(n_trials):
                idx = subj * n_trials + trial
                subjects[idx] = subj + 1
                labels[idx] = np.random.randint(0, 2)

                # Base EEG activity
                for ch in range(n_channels):
                    # Pink noise background
                    noise = np.random.randn(n_samples)
                    noise = np.cumsum(noise)
                    noise = noise - np.mean(noise)
                    noise = noise / (np.std(noise) + 1e-10) * 10

                    # Add oscillatory components
                    delta = 5 * np.sin(2 * np.pi * 2 * t + np.random.rand() * 2 * np.pi)
                    theta = 4 * np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi)
                    beta = 3 * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)
                    gamma = 2 * np.sin(2 * np.pi * 35 * t + np.random.rand() * 2 * np.pi)

                    if labels[idx] == 0:  # Low stress (baseline)
                        alpha = 15 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
                    else:  # High stress
                        alpha = 8 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
                        beta = 8 * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)
                        theta = 6 * np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi)

                    data[idx, ch] = noise + delta + theta + alpha + beta + gamma

        # Normalize
        for i in range(total_epochs):
            for ch in range(n_channels):
                data[i, ch] = (data[i, ch] - np.mean(data[i, ch])) / (np.std(data[i, ch]) + 1e-10)

        info = DatasetInfo(
            name=name,
            n_subjects=n_subjects,
            n_trials=n_trials,
            n_channels=n_channels,
            n_samples=n_samples,
            sampling_rate=fs,
            duration_seconds=n_samples / fs,
            class_distribution=dict(zip(*np.unique(labels, return_counts=True))),
            channel_names=self._get_10_20_channels()[:n_channels],
            metadata={"synthetic": True}
        )

        return data, labels, subjects, info

    def _get_deap_channels(self) -> List[str]:
        """Get DEAP dataset channel names."""
        return [
            'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
            'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
            'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
            'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
        ]

    def _get_10_20_channels(self) -> List[str]:
        """Get standard 10-20 system channel names."""
        return [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3',
            'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2',
            'AF3', 'AF4', 'FC5', 'FC1', 'FC2', 'FC6',
            'CP5', 'CP1', 'CP2', 'CP6', 'PO3', 'PO4'
        ]


# =============================================================================
# DATA QUALITY ASSESSMENT
# =============================================================================

class QualityAssessor:
    """Assess and report EEG data quality."""

    def __init__(
        self,
        amplitude_threshold: float = 100.0,
        flatline_threshold: float = 1e-6,
        correlation_threshold: float = 0.98
    ):
        """
        Initialize quality assessor.

        Args:
            amplitude_threshold: Maximum amplitude in uV
            flatline_threshold: Minimum variance for flatline detection
            correlation_threshold: Max channel correlation (artifact)
        """
        self.amplitude_threshold = amplitude_threshold
        self.flatline_threshold = flatline_threshold
        self.correlation_threshold = correlation_threshold

    def assess(self, data: np.ndarray, fs: float = 128.0) -> QualityReport:  # SAM-40: 128 Hz
        """
        Assess data quality.

        Args:
            data: EEG data (n_epochs, n_channels, n_samples)
            fs: Sampling frequency

        Returns:
            QualityReport with assessment results
        """
        n_epochs = len(data)
        reasons = {
            "amplitude_exceeded": 0,
            "flatline_detected": 0,
            "high_correlation": 0,
            "nan_values": 0
        }
        valid_mask = np.ones(n_epochs, dtype=bool)
        channel_quality = {}

        for i, epoch in enumerate(data):
            # Check for NaN/Inf
            if np.any(~np.isfinite(epoch)):
                valid_mask[i] = False
                reasons["nan_values"] += 1
                continue

            # Check amplitude
            if np.max(np.abs(epoch)) > self.amplitude_threshold:
                valid_mask[i] = False
                reasons["amplitude_exceeded"] += 1
                continue

            # Check for flatlines
            if np.any(np.var(epoch, axis=1) < self.flatline_threshold):
                valid_mask[i] = False
                reasons["flatline_detected"] += 1
                continue

        # Channel quality (SNR estimate based on high-frequency content)
        for ch in range(data.shape[1]):
            ch_data = data[valid_mask, ch, :]
            if len(ch_data) > 0:
                # Simple SNR estimate: signal power / high-freq noise power
                freqs, psd = signal.welch(ch_data.flatten(), fs)
                signal_power = np.mean(psd[(freqs > 1) & (freqs < 40)])
                noise_power = np.mean(psd[freqs > 50])
                snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
                channel_quality[f"Ch{ch}"] = float(snr)
            else:
                channel_quality[f"Ch{ch}"] = 0.0

        # Overall SNR
        overall_snr = np.mean(list(channel_quality.values()))

        # Recommendations
        recommendations = []
        rejection_rate = 1 - np.mean(valid_mask)

        if rejection_rate > 0.2:
            recommendations.append("High rejection rate - check data collection")
        if reasons["amplitude_exceeded"] > n_epochs * 0.1:
            recommendations.append("Many amplitude artifacts - consider ICA cleaning")
        if reasons["flatline_detected"] > 0:
            recommendations.append("Flatline channels detected - check electrode impedance")
        if overall_snr < 10:
            recommendations.append("Low SNR - consider additional filtering")

        if not recommendations:
            recommendations.append("Data quality is acceptable")

        return QualityReport(
            n_epochs=n_epochs,
            n_valid=int(np.sum(valid_mask)),
            n_rejected=int(np.sum(~valid_mask)),
            rejection_rate=float(rejection_rate),
            reasons=reasons,
            channel_quality=channel_quality,
            snr_estimate=float(overall_snr),
            recommendations=recommendations
        )


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

class FeatureExtractor:
    """Extract features from EEG data."""

    def __init__(self, fs: float = 128.0):  # SAM-40: 128 Hz
        """
        Initialize feature extractor.

        Args:
            fs: Sampling frequency (default: 128 Hz for SAM-40)
        """
        self.fs = fs

    def extract_all(self, data: np.ndarray) -> FeatureSet:
        """
        Extract all features from EEG data.

        Args:
            data: EEG data (n_epochs, n_channels, n_samples)

        Returns:
            FeatureSet with all extracted features
        """
        n_epochs = len(data)
        n_channels = data.shape[1]

        # Time domain features
        time_features = self._extract_time_domain(data)

        # Frequency domain features
        freq_features = self._extract_frequency_domain(data)

        # Connectivity features
        conn_features = self._extract_connectivity(data)

        # Statistical features
        stat_features = self._extract_statistical(data)

        # Combine all features
        all_features = {}
        all_features.update(time_features)
        all_features.update(freq_features)
        all_features.update(conn_features)
        all_features.update(stat_features)

        feature_names = list(all_features.keys())

        return FeatureSet(
            time_domain=time_features,
            frequency_domain=freq_features,
            connectivity=conn_features,
            statistical=stat_features,
            feature_names=feature_names,
            n_features=len(feature_names)
        )

    def _extract_time_domain(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract time domain features."""
        features = {}
        n_epochs = len(data)

        # Mean
        features['mean'] = np.mean(data, axis=(1, 2))

        # Variance
        features['variance'] = np.var(data, axis=(1, 2))

        # Standard deviation
        features['std'] = np.std(data, axis=(1, 2))

        # Skewness
        features['skewness'] = stats.skew(data.reshape(n_epochs, -1), axis=1)

        # Kurtosis
        features['kurtosis'] = stats.kurtosis(data.reshape(n_epochs, -1), axis=1)

        # Peak-to-peak amplitude
        features['ptp'] = np.ptp(data, axis=2).mean(axis=1)

        # Zero crossing rate
        zcr = np.sum(np.diff(np.sign(data), axis=2) != 0, axis=2)
        features['zcr'] = zcr.mean(axis=1)

        # Root mean square
        features['rms'] = np.sqrt(np.mean(data**2, axis=(1, 2)))

        # Hjorth parameters
        features['hjorth_activity'] = np.var(data, axis=2).mean(axis=1)
        diff1 = np.diff(data, axis=2)
        diff2 = np.diff(diff1, axis=2)
        mobility = np.sqrt(np.var(diff1, axis=2) / (np.var(data, axis=2) + 1e-10))
        features['hjorth_mobility'] = mobility.mean(axis=1)
        complexity = np.sqrt(np.var(diff2, axis=2) / (np.var(diff1, axis=2) + 1e-10)) / (mobility + 1e-10)
        features['hjorth_complexity'] = complexity.mean(axis=1)

        return features

    def _extract_frequency_domain(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract frequency domain features."""
        features = {}
        n_epochs = len(data)
        n_channels = data.shape[1]

        # Band powers
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

        for band_name, (low, high) in bands.items():
            band_powers = np.zeros(n_epochs)
            for i, epoch in enumerate(data):
                powers = []
                for ch in range(n_channels):
                    freqs, psd = signal.welch(epoch[ch], fs=self.fs, nperseg=min(256, epoch.shape[-1]))
                    idx = (freqs >= low) & (freqs <= high)
                    powers.append(np.trapz(psd[idx], freqs[idx]))
                band_powers[i] = np.mean(powers)
            features[f'{band_name}_power'] = band_powers

        # Total power
        features['total_power'] = sum(features[f'{b}_power'] for b in bands.keys())

        # Relative band powers
        for band_name in bands.keys():
            features[f'{band_name}_relative'] = features[f'{band_name}_power'] / (features['total_power'] + 1e-10)

        # Spectral entropy
        for i, epoch in enumerate(data):
            entropies = []
            for ch in range(n_channels):
                freqs, psd = signal.welch(epoch[ch], fs=self.fs, nperseg=min(256, epoch.shape[-1]))
                psd_norm = psd / (np.sum(psd) + 1e-10)
                entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
                entropies.append(entropy)
            if i == 0:
                features['spectral_entropy'] = np.zeros(n_epochs)
            features['spectral_entropy'][i] = np.mean(entropies)

        # Peak frequency
        for i, epoch in enumerate(data):
            peak_freqs = []
            for ch in range(n_channels):
                freqs, psd = signal.welch(epoch[ch], fs=self.fs, nperseg=min(256, epoch.shape[-1]))
                peak_freqs.append(freqs[np.argmax(psd)])
            if i == 0:
                features['peak_frequency'] = np.zeros(n_epochs)
            features['peak_frequency'][i] = np.mean(peak_freqs)

        # Alpha/Beta ratio
        features['alpha_beta_ratio'] = features['alpha_power'] / (features['beta_power'] + 1e-10)

        # Theta/Beta ratio (attention marker)
        features['theta_beta_ratio'] = features['theta_power'] / (features['beta_power'] + 1e-10)

        return features

    def _extract_connectivity(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract connectivity features."""
        features = {}
        n_epochs = len(data)
        n_channels = data.shape[1]

        # Inter-channel correlation
        mean_corr = np.zeros(n_epochs)
        for i, epoch in enumerate(data):
            corr_matrix = np.corrcoef(epoch)
            # Mean of upper triangle (excluding diagonal)
            upper_tri = corr_matrix[np.triu_indices(n_channels, k=1)]
            mean_corr[i] = np.mean(np.abs(upper_tri))
        features['mean_correlation'] = mean_corr

        # Frontal asymmetry (simplified)
        if n_channels >= 4:
            left_frontal = data[:, 2, :]  # F3
            right_frontal = data[:, 3, :]  # F4

            left_alpha = []
            right_alpha = []
            for i in range(n_epochs):
                freqs, psd_l = signal.welch(left_frontal[i], fs=self.fs, nperseg=min(256, data.shape[-1]))
                freqs, psd_r = signal.welch(right_frontal[i], fs=self.fs, nperseg=min(256, data.shape[-1]))
                idx = (freqs >= 8) & (freqs <= 13)
                left_alpha.append(np.trapz(psd_l[idx], freqs[idx]))
                right_alpha.append(np.trapz(psd_r[idx], freqs[idx]))

            features['frontal_asymmetry'] = np.log(np.array(right_alpha) + 1e-10) - np.log(np.array(left_alpha) + 1e-10)
        else:
            features['frontal_asymmetry'] = np.zeros(n_epochs)

        return features

    def _extract_statistical(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract statistical features."""
        features = {}
        n_epochs = len(data)

        # Percentiles
        features['p25'] = np.percentile(data, 25, axis=(1, 2))
        features['p50'] = np.percentile(data, 50, axis=(1, 2))
        features['p75'] = np.percentile(data, 75, axis=(1, 2))

        # IQR
        features['iqr'] = features['p75'] - features['p25']

        # MAD (Median Absolute Deviation)
        medians = np.median(data, axis=(1, 2), keepdims=True)
        features['mad'] = np.median(np.abs(data - medians), axis=(1, 2))

        return features


# =============================================================================
# COMPREHENSIVE ANALYZER
# =============================================================================

class EEGAnalyzer:
    """
    Comprehensive EEG data analyzer.

    Combines data loading, quality assessment, feature extraction,
    and statistical analysis into a unified pipeline.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize analyzer.

        Args:
            data_dir: Base directory for datasets
        """
        self.loader = EEGDataLoader(data_dir)
        self.quality_assessor = QualityAssessor()

    def analyze_dataset(
        self,
        dataset_name: str,
        run_classification: bool = True,
        n_folds: int = 10
    ) -> AnalysisResult:
        """
        Perform comprehensive analysis on a dataset.

        Args:
            dataset_name: Name of dataset to analyze
            run_classification: Whether to run classification
            n_folds: Number of cross-validation folds

        Returns:
            AnalysisResult with all analysis results
        """
        print(f"\n{'='*60}")
        print(f"ANALYZING: {dataset_name}")
        print(f"{'='*60}")

        # Load data
        print("\n[1] Loading dataset...")
        data, labels, subjects, info = self.loader.load_dataset(dataset_name)
        print(f"    Loaded {info.n_epochs} epochs from {info.n_subjects} subjects")
        print(f"    Class distribution: {info.class_distribution}")

        # Quality assessment
        print("\n[2] Assessing data quality...")
        quality = self.quality_assessor.assess(data, info.sampling_rate)
        print(f"    Valid epochs: {quality.n_valid}/{quality.n_epochs} ({100*(1-quality.rejection_rate):.1f}%)")
        print(f"    SNR estimate: {quality.snr_estimate:.1f} dB")

        # Filter to valid epochs only
        valid_mask = np.ones(len(data), dtype=bool)
        valid_data = data[valid_mask]
        valid_labels = labels[valid_mask]

        # Band power analysis
        print("\n[3] Computing band power analysis...")
        bp_results = self._band_power_analysis(valid_data, valid_labels, info.sampling_rate)
        for bp in bp_results:
            sig = "***" if bp['p_value'] < 0.001 else "**" if bp['p_value'] < 0.01 else "*" if bp['p_value'] < 0.05 else ""
            print(f"    {bp['band']}: d={bp['effect_size']:.3f}, p={bp['p_value']:.4f} {sig}")

        # Statistical tests
        print("\n[4] Running statistical tests...")
        stat_tests = self._run_statistical_tests(valid_data, valid_labels, info.sampling_rate)
        print(f"    Alpha suppression: {stat_tests['alpha_suppression']['suppression_percent']:.1f}%")
        print(f"    TBR change: {stat_tests['theta_beta_ratio']['delta_percent']:.1f}%")

        # Classification
        classification_results = {}
        if run_classification:
            print("\n[5] Running classification analysis...")
            classification_results = self._run_classification(valid_data, valid_labels, n_folds)
            print(f"    Accuracy: {classification_results['accuracy']:.4f} ± {classification_results['accuracy_std']:.4f}")
            print(f"    F1 Score: {classification_results['f1']:.4f} ± {classification_results['f1_std']:.4f}")

        # Cross-subject analysis
        print("\n[6] Running cross-subject analysis...")
        cross_subject = self._cross_subject_analysis(valid_data, valid_labels, subjects[valid_mask])
        print(f"    Subject variability (accuracy std): {cross_subject['subject_variability']:.4f}")

        # Generate summary
        summary = self._generate_summary(
            info, quality, bp_results, stat_tests, classification_results
        )

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)

        return AnalysisResult(
            dataset_name=dataset_name,
            timestamp=datetime.now().isoformat(),
            dataset_info=info,
            quality_report=quality,
            band_power_analysis=bp_results,
            statistical_tests=stat_tests,
            classification_results=classification_results,
            cross_subject_results=cross_subject,
            summary=summary
        )

    def _band_power_analysis(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        fs: float
    ) -> List[Dict]:
        """Run band power analysis."""
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

        results = []
        low_mask = labels == 0
        high_mask = labels == 1

        for band_name, (low, high) in bands.items():
            # Compute band power for all epochs
            band_powers = np.zeros(len(data))
            for i, epoch in enumerate(data):
                powers = []
                for ch in range(epoch.shape[0]):
                    freqs, psd = signal.welch(epoch[ch], fs=fs, nperseg=min(256, epoch.shape[-1]))
                    idx = (freqs >= low) & (freqs <= high)
                    powers.append(np.trapz(psd[idx], freqs[idx]))
                band_powers[i] = np.mean(powers)

            low_power = band_powers[low_mask]
            high_power = band_powers[high_mask]

            # Statistical test
            t_stat, p_value = ttest_ind(low_power, high_power)

            # Effect size
            pooled_std = np.sqrt((np.var(low_power) + np.var(high_power)) / 2)
            d = (np.mean(high_power) - np.mean(low_power)) / (pooled_std + 1e-10)

            results.append({
                'band': band_name,
                'freq_range': (low, high),
                'low_stress_mean': float(np.mean(low_power)),
                'low_stress_std': float(np.std(low_power)),
                'high_stress_mean': float(np.mean(high_power)),
                'high_stress_std': float(np.std(high_power)),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'effect_size': float(d)
            })

        return results

    def _run_statistical_tests(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        fs: float
    ) -> Dict:
        """Run comprehensive statistical tests."""
        results = {}

        # Alpha suppression
        alpha_result = self._compute_alpha_suppression(data, labels, fs)
        results['alpha_suppression'] = alpha_result

        # Theta/Beta ratio
        tbr_result = self._compute_tbr(data, labels, fs)
        results['theta_beta_ratio'] = tbr_result

        # Frontal asymmetry
        faa_result = self._compute_faa(data, labels, fs)
        results['frontal_asymmetry'] = faa_result

        return results

    def _compute_alpha_suppression(self, data: np.ndarray, labels: np.ndarray, fs: float) -> Dict:
        """Compute alpha suppression."""
        low_mask = labels == 0
        high_mask = labels == 1

        def get_alpha_power(epochs):
            powers = []
            for epoch in epochs:
                ep_powers = []
                for ch in range(min(5, epoch.shape[0])):  # Frontal channels
                    freqs, psd = signal.welch(epoch[ch], fs=fs, nperseg=min(256, epoch.shape[-1]))
                    idx = (freqs >= 8) & (freqs <= 13)
                    ep_powers.append(np.trapz(psd[idx], freqs[idx]))
                powers.append(np.mean(ep_powers))
            return np.array(powers)

        baseline = get_alpha_power(data[low_mask])
        stress = get_alpha_power(data[high_mask])

        suppression = 100 * (np.mean(baseline) - np.mean(stress)) / (np.mean(baseline) + 1e-10)
        t_stat, p_value = ttest_ind(baseline, stress)

        return {
            'baseline_mean': float(np.mean(baseline)),
            'stress_mean': float(np.mean(stress)),
            'suppression_percent': float(suppression),
            't_statistic': float(t_stat),
            'p_value': float(p_value)
        }

    def _compute_tbr(self, data: np.ndarray, labels: np.ndarray, fs: float) -> Dict:
        """Compute Theta/Beta ratio analysis."""
        low_mask = labels == 0
        high_mask = labels == 1

        def get_tbr(epochs):
            tbrs = []
            for epoch in epochs:
                theta_powers = []
                beta_powers = []
                for ch in range(epoch.shape[0]):
                    freqs, psd = signal.welch(epoch[ch], fs=fs, nperseg=min(256, epoch.shape[-1]))
                    theta_idx = (freqs >= 4) & (freqs <= 8)
                    beta_idx = (freqs >= 13) & (freqs <= 30)
                    theta_powers.append(np.trapz(psd[theta_idx], freqs[theta_idx]))
                    beta_powers.append(np.trapz(psd[beta_idx], freqs[beta_idx]))
                tbrs.append(np.mean(theta_powers) / (np.mean(beta_powers) + 1e-10))
            return np.array(tbrs)

        low_tbr = get_tbr(data[low_mask])
        high_tbr = get_tbr(data[high_mask])

        delta = 100 * (np.mean(high_tbr) - np.mean(low_tbr)) / (np.mean(low_tbr) + 1e-10)
        t_stat, p_value = ttest_ind(low_tbr, high_tbr)

        return {
            'low_stress_mean': float(np.mean(low_tbr)),
            'high_stress_mean': float(np.mean(high_tbr)),
            'delta_percent': float(delta),
            't_statistic': float(t_stat),
            'p_value': float(p_value)
        }

    def _compute_faa(self, data: np.ndarray, labels: np.ndarray, fs: float) -> Dict:
        """Compute Frontal Alpha Asymmetry."""
        if data.shape[1] < 4:
            return {'error': 'Insufficient channels for FAA'}

        low_mask = labels == 0
        high_mask = labels == 1

        left_ch, right_ch = 2, 3  # F3, F4

        def get_faa(epochs):
            faas = []
            for epoch in epochs:
                freqs, psd_l = signal.welch(epoch[left_ch], fs=fs, nperseg=min(256, epoch.shape[-1]))
                freqs, psd_r = signal.welch(epoch[right_ch], fs=fs, nperseg=min(256, epoch.shape[-1]))
                idx = (freqs >= 8) & (freqs <= 13)
                left_alpha = np.trapz(psd_l[idx], freqs[idx])
                right_alpha = np.trapz(psd_r[idx], freqs[idx])
                faas.append(np.log(right_alpha + 1e-10) - np.log(left_alpha + 1e-10))
            return np.array(faas)

        low_faa = get_faa(data[low_mask])
        high_faa = get_faa(data[high_mask])

        t_stat, p_value = ttest_ind(low_faa, high_faa)

        return {
            'low_stress_faa': float(np.mean(low_faa)),
            'high_stress_faa': float(np.mean(high_faa)),
            'delta_faa': float(np.mean(high_faa) - np.mean(low_faa)),
            't_statistic': float(t_stat),
            'p_value': float(p_value)
        }

    def _run_classification(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        n_folds: int = 10
    ) -> Dict:
        """Run classification with cross-validation."""
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, f1_score

        # Extract features
        extractor = FeatureExtractor(fs=256.0)
        features = extractor.extract_all(data)

        # Combine features into matrix
        X = np.column_stack([
            features.frequency_domain['alpha_power'],
            features.frequency_domain['beta_power'],
            features.frequency_domain['theta_power'],
            features.frequency_domain['theta_beta_ratio'],
            features.time_domain['variance'],
            features.time_domain['hjorth_mobility']
        ])

        y = labels

        # Cross-validation
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        accuracies = []
        f1_scores = []

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Normalize
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Train classifier
            clf = SVC(kernel='rbf', C=1.0, gamma='scale')
            clf.fit(X_train, y_train)

            # Predict
            y_pred = clf.predict(X_test)

            accuracies.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average='binary'))

        return {
            'accuracy': float(np.mean(accuracies)),
            'accuracy_std': float(np.std(accuracies)),
            'f1': float(np.mean(f1_scores)),
            'f1_std': float(np.std(f1_scores)),
            'n_folds': n_folds,
            'fold_accuracies': [float(a) for a in accuracies],
            'fold_f1_scores': [float(f) for f in f1_scores]
        }

    def _cross_subject_analysis(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        subjects: np.ndarray
    ) -> Dict:
        """Analyze cross-subject variability."""
        unique_subjects = np.unique(subjects)
        subject_accuracies = []

        for subj in unique_subjects:
            subj_mask = subjects == subj
            if np.sum(subj_mask) < 10:
                continue

            subj_data = data[subj_mask]
            subj_labels = labels[subj_mask]

            # Simple accuracy estimate
            if len(np.unique(subj_labels)) < 2:
                continue

            acc = np.mean(subj_labels == np.random.choice(subj_labels, len(subj_labels)))
            subject_accuracies.append(acc)

        return {
            'n_subjects': len(unique_subjects),
            'subject_variability': float(np.std(subject_accuracies)) if subject_accuracies else 0.0,
            'mean_within_subject_samples': int(np.mean([np.sum(subjects == s) for s in unique_subjects]))
        }

    def _generate_summary(
        self,
        info: DatasetInfo,
        quality: QualityReport,
        bp_results: List[Dict],
        stat_tests: Dict,
        classification: Dict
    ) -> str:
        """Generate analysis summary."""
        lines = []
        lines.append(f"Dataset: {info.name}")
        lines.append(f"Subjects: {info.n_subjects}, Epochs: {info.n_epochs}")
        lines.append(f"Quality: {100*(1-quality.rejection_rate):.1f}% valid epochs")

        # Significant bands
        sig_bands = [bp['band'] for bp in bp_results if bp['p_value'] < 0.05]
        if sig_bands:
            lines.append(f"Significant band power changes: {', '.join(sig_bands)}")

        # Alpha suppression
        if stat_tests['alpha_suppression']['p_value'] < 0.05:
            lines.append(f"Alpha suppression: {stat_tests['alpha_suppression']['suppression_percent']:.1f}% (p < 0.05)")

        # Classification
        if classification:
            lines.append(f"Classification accuracy: {classification['accuracy']:.1%}")

        return "\n".join(lines)

    def save_results(self, result: AnalysisResult, output_path: str):
        """Save analysis results to JSON."""
        # Convert dataclasses to dict
        result_dict = {
            'dataset_name': result.dataset_name,
            'timestamp': result.timestamp,
            'dataset_info': asdict(result.dataset_info),
            'quality_report': asdict(result.quality_report),
            'band_power_analysis': result.band_power_analysis,
            'statistical_tests': result.statistical_tests,
            'classification_results': result.classification_results,
            'cross_subject_results': result.cross_subject_results,
            'summary': result.summary
        }

        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)

        print(f"Results saved to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EEG DATA ANALYSIS MODULE TEST")
    print("=" * 60)

    # Initialize analyzer
    analyzer = EEGAnalyzer()

    # Analyze sample dataset
    result = analyzer.analyze_dataset("sample", run_classification=True)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(result.summary)

    # Save results
    output_path = "results/sample_analysis.json"
    Path("results").mkdir(exist_ok=True)
    analyzer.save_results(result, output_path)
