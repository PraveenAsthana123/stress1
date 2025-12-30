#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Exploratory Data Analysis (EDA) Module for EEG Data
================================================================================

Comprehensive EDA for EEG stress classification:
- Signal quality analysis
- Time-domain analysis
- Frequency-domain analysis
- Spatial/channel analysis
- Class separability analysis
- Redundancy analysis
- Leakage detection

Author: GenAI-RAG-EEG Team
Version: 3.0.0
License: MIT
================================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
from pathlib import Path

from scipy import stats, signal
from scipy.stats import pearsonr, spearmanr, kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


@dataclass
class EDAConfig:
    """Configuration for EDA."""
    sampling_rate: float = 128.0  # SAM-40: 128 Hz
    frequency_bands: Dict[str, Tuple[float, float]] = None

    def __post_init__(self):
        if self.frequency_bands is None:
            self.frequency_bands = {
                'delta': (0.5, 4.0),
                'theta': (4.0, 8.0),
                'alpha': (8.0, 13.0),
                'beta': (13.0, 30.0),
                'gamma': (30.0, 45.0)
            }


class SignalQualityAnalyzer:
    """Analyze signal quality of EEG data."""

    def __init__(self, config: Optional[EDAConfig] = None):
        self.config = config or EDAConfig()

    def compute_sqi(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Compute Signal Quality Index (SQI) metrics.

        Args:
            data: EEG data (n_samples, n_channels, n_time) or (n_channels, n_time)

        Returns:
            Dictionary of SQI metrics
        """
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        n_samples, n_channels, n_time = data.shape

        # Per-channel metrics
        rms = np.sqrt(np.mean(data ** 2, axis=2))  # (n_samples, n_channels)
        peak_to_peak = data.max(axis=2) - data.min(axis=2)
        variance = data.var(axis=2)

        # Flatline detection
        flatline_ratio = np.mean(np.abs(np.diff(data, axis=2)) < 1e-6, axis=2)

        # Clipping detection
        clip_threshold = np.percentile(np.abs(data), 99.9)
        clip_ratio = np.mean(np.abs(data) > clip_threshold * 0.95, axis=2)

        # Overall SQI score (heuristic)
        sqi_score = 1.0 - (flatline_ratio.mean() + clip_ratio.mean()) / 2

        return {
            'rms_mean': float(rms.mean()),
            'rms_std': float(rms.std()),
            'rms_per_channel': rms.mean(axis=0).tolist(),
            'peak_to_peak_mean': float(peak_to_peak.mean()),
            'variance_mean': float(variance.mean()),
            'flatline_ratio': float(flatline_ratio.mean()),
            'clip_ratio': float(clip_ratio.mean()),
            'sqi_score': float(sqi_score),
            'n_good_samples': int((sqi_score > 0.8).sum()) if isinstance(sqi_score, np.ndarray) else 1,
            'quality_assessment': 'good' if sqi_score > 0.8 else 'moderate' if sqi_score > 0.5 else 'poor'
        }

    def detect_bad_channels(
        self,
        data: np.ndarray,
        threshold: float = 3.0
    ) -> Dict[str, Any]:
        """
        Detect bad channels based on statistical outliers.
        """
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        n_samples, n_channels, n_time = data.shape

        # Compute channel statistics
        channel_var = data.var(axis=(0, 2))
        channel_kurt = kurtosis(data.reshape(-1, n_channels), axis=0)

        # Z-score based detection
        var_zscore = np.abs(stats.zscore(channel_var))
        kurt_zscore = np.abs(stats.zscore(channel_kurt))

        bad_var = var_zscore > threshold
        bad_kurt = kurt_zscore > threshold
        bad_channels = bad_var | bad_kurt

        return {
            'bad_channel_indices': np.where(bad_channels)[0].tolist(),
            'n_bad_channels': int(bad_channels.sum()),
            'bad_ratio': float(bad_channels.mean()),
            'channel_variance': channel_var.tolist(),
            'channel_kurtosis': channel_kurt.tolist(),
            'variance_outliers': np.where(bad_var)[0].tolist(),
            'kurtosis_outliers': np.where(bad_kurt)[0].tolist()
        }


class TimeDomainAnalyzer:
    """Time-domain analysis of EEG signals."""

    def __init__(self, config: Optional[EDAConfig] = None):
        self.config = config or EDAConfig()

    def compute_statistics(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Compute time-domain statistics.
        """
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        return {
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'skewness': float(skew(data.flatten())),
            'kurtosis': float(kurtosis(data.flatten())),
            'rms': float(np.sqrt(np.mean(data ** 2))),
            'peak_to_peak': float(data.max() - data.min())
        }

    def compute_hjorth_parameters(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute Hjorth parameters (Activity, Mobility, Complexity).
        """
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        n_samples, n_channels, n_time = data.shape

        # First derivative
        diff1 = np.diff(data, axis=2)
        # Second derivative
        diff2 = np.diff(diff1, axis=2)

        # Activity = variance
        activity = data.var(axis=2)

        # Mobility = std(diff1) / std(data)
        mobility = diff1.std(axis=2) / (data.std(axis=2) + 1e-8)

        # Complexity = mobility(diff1) / mobility
        mobility_diff1 = diff2.std(axis=2) / (diff1.std(axis=2) + 1e-8)
        complexity = mobility_diff1 / (mobility + 1e-8)

        return {
            'activity_mean': float(activity.mean()),
            'activity_per_channel': activity.mean(axis=0).tolist(),
            'mobility_mean': float(mobility.mean()),
            'mobility_per_channel': mobility.mean(axis=0).tolist(),
            'complexity_mean': float(complexity.mean()),
            'complexity_per_channel': complexity.mean(axis=0).tolist()
        }

    def compute_class_statistics(
        self,
        data: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute statistics per class.
        """
        classes = np.unique(labels)
        class_stats = {}

        for c in classes:
            mask = labels == c
            class_data = data[mask]

            class_stats[f'class_{c}'] = {
                'n_samples': int(mask.sum()),
                'mean': float(class_data.mean()),
                'std': float(class_data.std()),
                'hjorth': self.compute_hjorth_parameters(class_data)
            }

        return class_stats


class FrequencyDomainAnalyzer:
    """Frequency-domain analysis of EEG signals."""

    def __init__(self, config: Optional[EDAConfig] = None):
        self.config = config or EDAConfig()

    def compute_band_powers(
        self,
        data: np.ndarray,
        normalize: bool = True
    ) -> Dict[str, Any]:
        """
        Compute power in standard frequency bands.
        """
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        n_samples, n_channels, n_time = data.shape
        fs = self.config.sampling_rate

        band_powers = {band: [] for band in self.config.frequency_bands}

        for i in range(n_samples):
            for j in range(n_channels):
                freqs, psd = signal.welch(data[i, j], fs=fs, nperseg=min(256, n_time))

                for band_name, (low, high) in self.config.frequency_bands.items():
                    idx = (freqs >= low) & (freqs <= high)
                    power = np.trapz(psd[idx], freqs[idx])
                    band_powers[band_name].append(power)

        # Convert to arrays
        for band in band_powers:
            band_powers[band] = np.array(band_powers[band]).reshape(n_samples, n_channels)

        # Normalize (relative power)
        if normalize:
            total_power = sum(band_powers[b].sum(axis=1, keepdims=True) for b in band_powers)
            for band in band_powers:
                band_powers[band] = band_powers[band] / (total_power + 1e-8)

        # Summary statistics
        result = {}
        for band in band_powers:
            result[f'{band}_mean'] = float(band_powers[band].mean())
            result[f'{band}_std'] = float(band_powers[band].std())
            result[f'{band}_per_channel'] = band_powers[band].mean(axis=0).tolist()

        result['raw_band_powers'] = {k: v.mean(axis=0).tolist() for k, v in band_powers.items()}

        return result

    def compute_spectral_entropy(self, data: np.ndarray) -> Dict[str, float]:
        """
        Compute spectral entropy.
        """
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        n_samples, n_channels, n_time = data.shape
        fs = self.config.sampling_rate

        entropies = []

        for i in range(n_samples):
            for j in range(n_channels):
                freqs, psd = signal.welch(data[i, j], fs=fs, nperseg=min(256, n_time))
                # Normalize PSD
                psd_norm = psd / (psd.sum() + 1e-8)
                # Shannon entropy
                entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-8))
                entropies.append(entropy)

        entropies = np.array(entropies).reshape(n_samples, n_channels)

        return {
            'spectral_entropy_mean': float(entropies.mean()),
            'spectral_entropy_std': float(entropies.std()),
            'spectral_entropy_per_channel': entropies.mean(axis=0).tolist()
        }


class SpatialAnalyzer:
    """Spatial/channel analysis of EEG data."""

    def __init__(self, config: Optional[EDAConfig] = None):
        self.config = config or EDAConfig()

    def compute_channel_correlations(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Compute correlation matrix between channels.
        """
        if data.ndim == 3:
            # Flatten to (n_channels, n_total_time)
            data = data.reshape(data.shape[0] * data.shape[2], data.shape[1]).T

        n_channels = data.shape[0]
        corr_matrix = np.corrcoef(data)

        # Get upper triangle (excluding diagonal)
        upper_tri = corr_matrix[np.triu_indices(n_channels, k=1)]

        return {
            'mean_correlation': float(upper_tri.mean()),
            'std_correlation': float(upper_tri.std()),
            'max_correlation': float(upper_tri.max()),
            'min_correlation': float(upper_tri.min()),
            'n_high_correlations': int((np.abs(upper_tri) > 0.8).sum()),
            'correlation_matrix': corr_matrix.tolist()
        }

    def compute_channel_importance(
        self,
        data: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute channel importance using univariate AUC.
        """
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        n_samples, n_channels, n_time = data.shape

        # Use mean power per channel as feature
        channel_features = data.mean(axis=2)  # (n_samples, n_channels)

        # Compute AUC for each channel
        channel_aucs = []
        for j in range(n_channels):
            try:
                auc = roc_auc_score(labels, channel_features[:, j])
                # Make AUC symmetric around 0.5
                auc = max(auc, 1 - auc)
            except:
                auc = 0.5
            channel_aucs.append(auc)

        channel_aucs = np.array(channel_aucs)

        # Rank channels
        ranking = np.argsort(channel_aucs)[::-1]

        return {
            'channel_aucs': channel_aucs.tolist(),
            'best_channels': ranking[:5].tolist(),
            'worst_channels': ranking[-5:].tolist(),
            'mean_auc': float(channel_aucs.mean()),
            'max_auc': float(channel_aucs.max())
        }


class ClassSeparabilityAnalyzer:
    """Analyze class separability."""

    def __init__(self, config: Optional[EDAConfig] = None):
        self.config = config or EDAConfig()

    def compute_effect_sizes(
        self,
        data: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute Cohen's d effect size for class differences.
        """
        if data.ndim == 3:
            # Use mean as summary
            features = data.mean(axis=2)
        else:
            features = data

        class_0 = features[labels == 0]
        class_1 = features[labels == 1]

        # Per-feature Cohen's d
        n0, n1 = len(class_0), len(class_1)
        pooled_std = np.sqrt(
            ((n0 - 1) * class_0.std(axis=0) ** 2 + (n1 - 1) * class_1.std(axis=0) ** 2)
            / (n0 + n1 - 2)
        )

        cohens_d = (class_1.mean(axis=0) - class_0.mean(axis=0)) / (pooled_std + 1e-8)

        return {
            'cohens_d_mean': float(np.mean(np.abs(cohens_d))),
            'cohens_d_max': float(np.max(np.abs(cohens_d))),
            'cohens_d_per_feature': cohens_d.tolist(),
            'n_large_effects': int((np.abs(cohens_d) > 0.8).sum()),
            'n_medium_effects': int((np.abs(cohens_d) > 0.5).sum()),
            'n_small_effects': int((np.abs(cohens_d) > 0.2).sum())
        }

    def compute_lda_projection(
        self,
        data: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Project data using LDA and assess separability.
        """
        if data.ndim == 3:
            features = data.reshape(data.shape[0], -1)
        else:
            features = data

        # Limit features if too many
        if features.shape[1] > 1000:
            pca = PCA(n_components=100)
            features = pca.fit_transform(features)

        lda = LinearDiscriminantAnalysis(n_components=1)
        lda_features = lda.fit_transform(features, labels)

        # Separability score
        class_0 = lda_features[labels == 0]
        class_1 = lda_features[labels == 1]
        overlap = self._compute_overlap(class_0, class_1)

        return {
            'lda_explained_variance': float(lda.explained_variance_ratio_[0]) if hasattr(lda, 'explained_variance_ratio_') else None,
            'lda_class_0_mean': float(class_0.mean()),
            'lda_class_1_mean': float(class_1.mean()),
            'lda_separation': float(abs(class_0.mean() - class_1.mean())),
            'class_overlap': float(overlap)
        }

    def _compute_overlap(self, dist1: np.ndarray, dist2: np.ndarray) -> float:
        """Compute overlap between two distributions."""
        min_val = min(dist1.min(), dist2.min())
        max_val = max(dist1.max(), dist2.max())
        bins = np.linspace(min_val, max_val, 50)

        hist1, _ = np.histogram(dist1, bins=bins, density=True)
        hist2, _ = np.histogram(dist2, bins=bins, density=True)

        overlap = np.minimum(hist1, hist2).sum() * (bins[1] - bins[0])
        return float(overlap)


class RedundancyAnalyzer:
    """Analyze feature redundancy."""

    def compute_redundancy(
        self,
        data: np.ndarray,
        threshold: float = 0.9
    ) -> Dict[str, Any]:
        """
        Detect redundant features based on correlation.
        """
        if data.ndim == 3:
            features = data.reshape(data.shape[0], -1)
        else:
            features = data

        # Compute correlation matrix
        corr_matrix = np.corrcoef(features.T)

        # Find highly correlated pairs
        n_features = corr_matrix.shape[0]
        redundant_pairs = []

        for i in range(n_features):
            for j in range(i + 1, n_features):
                if abs(corr_matrix[i, j]) > threshold:
                    redundant_pairs.append((i, j, float(corr_matrix[i, j])))

        # Identify features to potentially remove
        features_to_remove = set()
        for i, j, _ in redundant_pairs:
            features_to_remove.add(j)

        return {
            'n_redundant_pairs': len(redundant_pairs),
            'redundant_pairs': redundant_pairs[:20],  # Top 20
            'features_to_remove': list(features_to_remove),
            'n_features_to_remove': len(features_to_remove),
            'remaining_features': n_features - len(features_to_remove)
        }


class LeakageDetector:
    """Detect potential data leakage."""

    def check_leakage(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        subject_ids: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Check for potential data leakage.
        """
        results = {'leakage_detected': False, 'warnings': []}

        if data.ndim == 3:
            features = data.mean(axis=2)
        else:
            features = data

        # Test 1: Can we predict labels from simple features?
        try:
            lr = LogisticRegression(max_iter=1000, random_state=42)
            cv_scores = cross_val_score(lr, features, labels, cv=5)
            if cv_scores.mean() > 0.95:
                results['warnings'].append(
                    f"High CV accuracy ({cv_scores.mean():.2f}) may indicate leakage"
                )
                results['leakage_detected'] = True
        except:
            pass

        # Test 2: Can we predict subject from features? (if subject IDs provided)
        if subject_ids is not None and len(np.unique(subject_ids)) > 1:
            try:
                lr = LogisticRegression(max_iter=1000, random_state=42)
                cv_scores = cross_val_score(
                    lr, features, subject_ids, cv=min(5, len(np.unique(subject_ids)))
                )
                if cv_scores.mean() > 0.9:
                    results['warnings'].append(
                        f"Can predict subject ID ({cv_scores.mean():.2f}) - possible subject leakage"
                    )
            except:
                pass

        # Test 3: Check for suspiciously perfect features
        for i in range(min(features.shape[1], 10)):
            auc = roc_auc_score(labels, features[:, i])
            if auc > 0.99 or auc < 0.01:
                results['warnings'].append(f"Feature {i} has near-perfect AUC: {auc:.4f}")
                results['leakage_detected'] = True

        return results


class ComprehensiveEDA:
    """
    Complete EDA pipeline for EEG data.
    """

    def __init__(self, config: Optional[EDAConfig] = None):
        self.config = config or EDAConfig()
        self.signal_quality = SignalQualityAnalyzer(self.config)
        self.time_domain = TimeDomainAnalyzer(self.config)
        self.freq_domain = FrequencyDomainAnalyzer(self.config)
        self.spatial = SpatialAnalyzer(self.config)
        self.separability = ClassSeparabilityAnalyzer(self.config)
        self.redundancy = RedundancyAnalyzer()
        self.leakage = LeakageDetector()

    def run_full_eda(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        subject_ids: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Run complete EDA pipeline.

        Args:
            data: EEG data (n_samples, n_channels, n_time)
            labels: Class labels
            subject_ids: Optional subject identifiers

        Returns:
            Comprehensive EDA report
        """
        report = {
            'data_shape': data.shape,
            'n_samples': data.shape[0],
            'n_channels': data.shape[1],
            'n_timepoints': data.shape[2],
            'class_distribution': {
                int(c): int((labels == c).sum()) for c in np.unique(labels)
            }
        }

        print("Running Signal Quality Analysis...")
        report['signal_quality'] = self.signal_quality.compute_sqi(data)
        report['bad_channels'] = self.signal_quality.detect_bad_channels(data)

        print("Running Time-Domain Analysis...")
        report['time_domain'] = self.time_domain.compute_statistics(data)
        report['hjorth'] = self.time_domain.compute_hjorth_parameters(data)
        report['class_statistics'] = self.time_domain.compute_class_statistics(data, labels)

        print("Running Frequency-Domain Analysis...")
        report['band_powers'] = self.freq_domain.compute_band_powers(data)
        report['spectral_entropy'] = self.freq_domain.compute_spectral_entropy(data)

        print("Running Spatial Analysis...")
        report['channel_correlations'] = self.spatial.compute_channel_correlations(data)
        report['channel_importance'] = self.spatial.compute_channel_importance(data, labels)

        print("Running Class Separability Analysis...")
        report['effect_sizes'] = self.separability.compute_effect_sizes(data, labels)
        report['lda_projection'] = self.separability.compute_lda_projection(data, labels)

        print("Running Leakage Detection...")
        report['leakage_check'] = self.leakage.check_leakage(data, labels, subject_ids)

        return report


if __name__ == "__main__":
    print("Testing EDA Module")
    print("=" * 50)

    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    n_channels = 32
    n_time = 3200  # SAM-40: 25 sec at 128 Hz

    # Simulate EEG with class differences
    X = np.random.randn(n_samples, n_channels, n_time).astype(np.float32)
    y = np.random.randint(0, 2, n_samples)

    # Add class-specific patterns
    X[y == 1, :, :] += 0.3  # Class 1 has higher mean

    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.unique(y, return_counts=True)}")

    # Run EDA
    eda = ComprehensiveEDA()
    report = eda.run_full_eda(X, y)

    print("\n" + "=" * 50)
    print("EDA Summary:")
    print(f"  Signal Quality: {report['signal_quality']['quality_assessment']}")
    print(f"  Bad Channels: {report['bad_channels']['n_bad_channels']}")
    print(f"  Alpha Power: {report['band_powers']['alpha_mean']:.4f}")
    print(f"  Effect Size (d): {report['effect_sizes']['cohens_d_mean']:.3f}")
    print(f"  Leakage: {'DETECTED' if report['leakage_check']['leakage_detected'] else 'None'}")

    print("\n✓ EDA module works!")
