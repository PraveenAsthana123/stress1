#!/usr/bin/env python3
"""
================================================================================
GenAI-RAG-EEG: Comprehensive Pipeline Runner
================================================================================

This script runs the complete ML pipeline with detailed CLI output:
1. Data Generation & Preprocessing
2. 1D/2D Conversion & Standardization/Normalization
3. Exploratory Data Analysis (EDA)
4. Model Training
5. Model Validation
6. Model Testing
7. Benchmarking
8. Report Generation

CLI OUTPUT:
- Step-by-step progress with timing
- Color-coded status messages
- Detailed metrics at each stage
- Final summary report

Usage: python run_pipeline.py
       python run_pipeline.py --verbose
       python run_pipeline.py --quiet

Author: GenAI-RAG-EEG Team
Version: 3.0.0
================================================================================
"""

import os
import sys
import time
import json
import warnings
import platform
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import custom logger for detailed CLI output
try:
    from src.utils.logger import setup_logger, GenAILogger
    from src.utils.compatibility import CompatibilityLayer
    CUSTOM_LOGGER_AVAILABLE = True
except ImportError:
    CUSTOM_LOGGER_AVAILABLE = False


def print_pipeline_banner():
    """Print detailed pipeline banner with system info."""
    print("\n" + "=" * 70)
    print("  ╔═══════════════════════════════════════════════════════════════╗")
    print("  ║       GenAI-RAG-EEG: Comprehensive Pipeline Runner            ║")
    print("  ║         Complete ML Pipeline with Detailed Logging            ║")
    print("  ╚═══════════════════════════════════════════════════════════════╝")
    print("=" * 70)

    print(f"\n  Pipeline Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Python: {platform.python_version()}")

    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  GPU: {gpu_name}")
        else:
            print(f"  GPU: Not available (using CPU)")
    except ImportError:
        print(f"  GPU: PyTorch not available")

    print("\n  Pipeline Steps:")
    print("    [1] Environment Setup")
    print("    [2] Data Generation & Preprocessing")
    print("    [3] 1D/2D Conversion & Normalization")
    print("    [4] Exploratory Data Analysis (EDA)")
    print("    [5] Model Training")
    print("    [6] Model Validation")
    print("    [7] Model Testing")
    print("    [8] Report Generation")
    print("=" * 70)
    print()


def print_step_header(step_num: int, total: int, title: str):
    """Print formatted step header."""
    print(f"\n{'─' * 70}")
    print(f"  [{step_num}/{total}] {title}")
    print(f"{'─' * 70}")


def print_step_complete(step_num: int, title: str, elapsed: float, status: str = "DONE"):
    """Print step completion message."""
    print(f"  ✓ {title} completed in {elapsed:.2f}s [{status}]")


# Print startup banner
print_pipeline_banner()

# ============================================================================
# SECTION 1: IMPORTS AND SETUP
# ============================================================================

print("[1/8] Setting up environment...")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    print(f"  ✓ PyTorch {torch.__version__}")
except ImportError:
    TORCH_AVAILABLE = False
    print("  ✗ PyTorch not available")

try:
    from scipy import signal, stats
    from scipy.stats import zscore, pearsonr, spearmanr, ttest_ind
    SCIPY_AVAILABLE = True
    print(f"  ✓ SciPy available")
except ImportError:
    SCIPY_AVAILABLE = False
    print("  ✗ SciPy not available")

try:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, matthews_corrcoef, balanced_accuracy_score,
        confusion_matrix, classification_report
    )
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, StratifiedKFold
    SKLEARN_AVAILABLE = True
    print(f"  ✓ scikit-learn available")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("  ✗ scikit-learn not available")

# Set random seeds
SEED = 42
np.random.seed(SEED)
if TORCH_AVAILABLE:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
print(f"  ✓ Device: {DEVICE}")
print()

# ============================================================================
# SECTION 2: DATA GENERATION
# ============================================================================

print("[2/8] Generating Synthetic EEG Data...")

@dataclass
class DataConfig:
    n_subjects: int = 10
    n_trials_per_subject: int = 20
    n_channels: int = 32
    n_time_samples: int = 512
    sampling_rate: float = 128.0
    class_balance: float = 0.5

def generate_eeg_data(config: DataConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic EEG data with stress-related features.

    Low stress: Higher alpha power (10 Hz)
    High stress: Lower alpha, higher beta power (20 Hz)
    """
    n_total = config.n_subjects * config.n_trials_per_subject
    data = np.zeros((n_total, config.n_channels, config.n_time_samples))
    labels = np.zeros(n_total, dtype=np.int64)
    subject_ids = np.zeros(n_total, dtype=np.int64)

    t = np.linspace(0, config.n_time_samples / config.sampling_rate, config.n_time_samples)

    for s in range(config.n_subjects):
        subject_offset = s * config.n_trials_per_subject

        # Subject-specific variability
        subject_alpha_var = 0.8 + 0.4 * np.random.rand()
        subject_beta_var = 0.8 + 0.4 * np.random.rand()

        for trial in range(config.n_trials_per_subject):
            idx = subject_offset + trial

            is_stress = np.random.rand() < config.class_balance
            labels[idx] = int(is_stress)
            subject_ids[idx] = s + 1

            for ch in range(config.n_channels):
                # Base noise
                noise = np.random.randn(config.n_time_samples) * 5

                # Channel-specific phase
                phase = np.random.rand() * 2 * np.pi

                if is_stress:
                    # High stress: suppressed alpha, elevated beta
                    alpha = 5 * subject_alpha_var * np.sin(2 * np.pi * 10 * t + phase)
                    beta = 15 * subject_beta_var * np.sin(2 * np.pi * 20 * t + phase)
                    theta = 8 * np.sin(2 * np.pi * 6 * t + phase)  # Frontal theta
                else:
                    # Low stress: elevated alpha, reduced beta
                    alpha = 20 * subject_alpha_var * np.sin(2 * np.pi * 10 * t + phase)
                    beta = 5 * subject_beta_var * np.sin(2 * np.pi * 20 * t + phase)
                    theta = 3 * np.sin(2 * np.pi * 6 * t + phase)

                data[idx, ch] = noise + alpha + beta + theta

    return data, labels, subject_ids

# Generate data for each dataset
datasets = {}

# SAM-40 style (primary - cognitive stress)
print("  Generating SAM-40 style data (40 subjects, 256 Hz)...")
sam40_config = DataConfig(n_subjects=40, n_trials_per_subject=12, sampling_rate=256.0)
sam40_data, sam40_labels, sam40_subjects = generate_eeg_data(sam40_config)
datasets['sam40'] = {'data': sam40_data, 'labels': sam40_labels, 'subjects': sam40_subjects, 'config': sam40_config}
print(f"    Shape: {sam40_data.shape}, Labels: {np.bincount(sam40_labels)}")

# DEAP style (benchmark - arousal proxy)
print("  Generating DEAP style data (32 subjects, 128 Hz)...")
deap_config = DataConfig(n_subjects=32, n_trials_per_subject=40, sampling_rate=128.0)
deap_data, deap_labels, deap_subjects = generate_eeg_data(deap_config)
datasets['deap'] = {'data': deap_data, 'labels': deap_labels, 'subjects': deap_subjects, 'config': deap_config}
print(f"    Shape: {deap_data.shape}, Labels: {np.bincount(deap_labels)}")

# EEGMAT style (supplementary - workload)
print("  Generating EEGMAT style data (25 subjects, 14 channels)...")
eegmat_config = DataConfig(n_subjects=25, n_trials_per_subject=20, n_channels=14, sampling_rate=128.0)
eegmat_data, eegmat_labels, eegmat_subjects = generate_eeg_data(eegmat_config)
datasets['eegmat'] = {'data': eegmat_data, 'labels': eegmat_labels, 'subjects': eegmat_subjects, 'config': eegmat_config}
print(f"    Shape: {eegmat_data.shape}, Labels: {np.bincount(eegmat_labels)}")

print(f"  ✓ Generated {sum(d['data'].shape[0] for d in datasets.values())} total samples")
print()

# ============================================================================
# SECTION 3: DATA PREPROCESSING
# ============================================================================

print("[3/8] Data Preprocessing...")

def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    """Apply Butterworth bandpass filter."""
    if not SCIPY_AVAILABLE:
        return data
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')

    if data.ndim == 2:
        return signal.filtfilt(b, a, data, axis=1)
    elif data.ndim == 3:
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = signal.filtfilt(b, a, data[i], axis=1)
        return filtered
    return data

def notch_filter(data: np.ndarray, freq: float, fs: float, quality: float = 30.0) -> np.ndarray:
    """Apply notch filter for power line noise."""
    if not SCIPY_AVAILABLE:
        return data
    b, a = signal.iirnotch(freq, quality, fs)

    if data.ndim == 2:
        return signal.filtfilt(b, a, data, axis=1)
    elif data.ndim == 3:
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = signal.filtfilt(b, a, data[i], axis=1)
        return filtered
    return data

preprocessing_results = {}

for name, dataset in datasets.items():
    print(f"  Processing {name.upper()}...")
    data = dataset['data'].copy()
    fs = dataset['config'].sampling_rate

    original_stats = {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data))
    }

    # Step 1: Bandpass filter (0.5-45 Hz)
    data = bandpass_filter(data, 0.5, 45.0, fs)
    print(f"    ✓ Bandpass filter (0.5-45 Hz)")

    # Step 2: Notch filter (50 Hz)
    data = notch_filter(data, 50.0, fs)
    print(f"    ✓ Notch filter (50 Hz)")

    filtered_stats = {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data))
    }

    # Step 3: Artifact rejection (±100 μV threshold)
    threshold = 100.0
    max_amp = np.abs(data).max(axis=(1, 2))
    keep_mask = max_amp < threshold
    rejected_count = (~keep_mask).sum()
    data = data[keep_mask]
    labels = dataset['labels'][keep_mask]
    subjects = dataset['subjects'][keep_mask]

    print(f"    ✓ Artifact rejection: {rejected_count} epochs removed ({100*rejected_count/len(keep_mask):.1f}%)")

    preprocessing_results[name] = {
        'original_stats': original_stats,
        'filtered_stats': filtered_stats,
        'original_samples': len(keep_mask),
        'retained_samples': len(data),
        'rejection_rate': float(rejected_count / len(keep_mask))
    }

    # Update dataset
    dataset['data_processed'] = data
    dataset['labels_processed'] = labels
    dataset['subjects_processed'] = subjects

print("  ✓ Preprocessing complete")
print()

# ============================================================================
# SECTION 4: 1D/2D CONVERSION, STANDARDIZATION, NORMALIZATION
# ============================================================================

print("[4/8] Data Conversion & Normalization...")

normalization_results = {}

for name, dataset in datasets.items():
    print(f"  Processing {name.upper()}...")
    data = dataset['data_processed']
    n_samples, n_channels, n_time = data.shape

    # 1D representation (flattened)
    data_1d = data.reshape(n_samples, -1)
    print(f"    ✓ 1D representation: {data_1d.shape}")

    # 2D representation (channels x time as image)
    data_2d = data  # Already in (samples, channels, time) format
    print(f"    ✓ 2D representation: {data_2d.shape}")

    # Z-score normalization (per channel)
    data_zscore = np.zeros_like(data)
    for i in range(n_samples):
        for ch in range(n_channels):
            channel_data = data[i, ch]
            data_zscore[i, ch] = (channel_data - channel_data.mean()) / (channel_data.std() + 1e-8)

    zscore_stats = {
        'mean': float(np.mean(data_zscore)),
        'std': float(np.std(data_zscore)),
        'min': float(np.min(data_zscore)),
        'max': float(np.max(data_zscore))
    }
    print(f"    ✓ Z-score normalization: mean={zscore_stats['mean']:.4f}, std={zscore_stats['std']:.4f}")

    # Min-Max normalization
    data_minmax = np.zeros_like(data)
    for i in range(n_samples):
        for ch in range(n_channels):
            channel_data = data[i, ch]
            min_val, max_val = channel_data.min(), channel_data.max()
            data_minmax[i, ch] = (channel_data - min_val) / (max_val - min_val + 1e-8)

    minmax_stats = {
        'mean': float(np.mean(data_minmax)),
        'std': float(np.std(data_minmax)),
        'min': float(np.min(data_minmax)),
        'max': float(np.max(data_minmax))
    }
    print(f"    ✓ Min-Max normalization: range=[{minmax_stats['min']:.4f}, {minmax_stats['max']:.4f}]")

    # Standardization (global)
    if SKLEARN_AVAILABLE:
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(data_1d)
        std_stats = {
            'mean': float(np.mean(data_standardized)),
            'std': float(np.std(data_standardized))
        }
        print(f"    ✓ Global standardization: mean={std_stats['mean']:.6f}, std={std_stats['std']:.4f}")
    else:
        data_standardized = (data_1d - data_1d.mean()) / data_1d.std()
        std_stats = {'mean': float(np.mean(data_standardized)), 'std': float(np.std(data_standardized))}

    normalization_results[name] = {
        'zscore': zscore_stats,
        'minmax': minmax_stats,
        'standardization': std_stats
    }

    # Store normalized versions
    dataset['data_zscore'] = data_zscore
    dataset['data_minmax'] = data_minmax
    dataset['data_1d'] = data_1d

print("  ✓ Conversion & normalization complete")
print()

# ============================================================================
# SECTION 5: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("[5/8] Exploratory Data Analysis (EDA)...")

eda_results = {}

for name, dataset in datasets.items():
    print(f"  Analyzing {name.upper()}...")
    data = dataset['data_zscore']
    labels = dataset['labels_processed']

    n_samples = len(labels)
    n_class_0 = (labels == 0).sum()
    n_class_1 = (labels == 1).sum()

    # Class distribution
    class_dist = {
        'total_samples': int(n_samples),
        'class_0_count': int(n_class_0),
        'class_1_count': int(n_class_1),
        'class_0_pct': float(100 * n_class_0 / n_samples),
        'class_1_pct': float(100 * n_class_1 / n_samples),
        'balance_ratio': float(min(n_class_0, n_class_1) / max(n_class_0, n_class_1))
    }
    print(f"    Class distribution: {n_class_0} low stress, {n_class_1} high stress ({class_dist['balance_ratio']:.2f} ratio)")

    # Feature statistics by class
    class_0_data = data[labels == 0]
    class_1_data = data[labels == 1]

    # Compute power in different frequency bands (simplified)
    feature_stats = {
        'class_0_mean': float(np.mean(class_0_data)),
        'class_0_std': float(np.std(class_0_data)),
        'class_1_mean': float(np.mean(class_1_data)),
        'class_1_std': float(np.std(class_1_data))
    }

    # Statistical test between classes
    if SCIPY_AVAILABLE:
        # Flatten and sample for t-test
        sample_size = min(1000, len(class_0_data), len(class_1_data))
        class_0_sample = class_0_data.reshape(len(class_0_data), -1)[:sample_size].mean(axis=1)
        class_1_sample = class_1_data.reshape(len(class_1_data), -1)[:sample_size].mean(axis=1)

        t_stat, p_value = ttest_ind(class_0_sample, class_1_sample)
        statistical_test = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
        print(f"    T-test: t={t_stat:.4f}, p={p_value:.6f} ({'significant' if p_value < 0.05 else 'not significant'})")
    else:
        statistical_test = {'t_statistic': None, 'p_value': None, 'significant': None}

    # Channel correlations (average)
    n_channels = data.shape[1]
    channel_corrs = []
    for i in range(min(5, n_channels)):
        for j in range(i+1, min(5, n_channels)):
            if SCIPY_AVAILABLE:
                r, _ = pearsonr(data[:, i].flatten()[:1000], data[:, j].flatten()[:1000])
                channel_corrs.append(r)

    avg_channel_corr = float(np.mean(channel_corrs)) if channel_corrs else 0.0
    print(f"    Average channel correlation: {avg_channel_corr:.4f}")

    eda_results[name] = {
        'class_distribution': class_dist,
        'feature_statistics': feature_stats,
        'statistical_test': statistical_test,
        'avg_channel_correlation': avg_channel_corr
    }

print("  ✓ EDA complete")
print()

# ============================================================================
# SECTION 6: MODEL DEFINITION
# ============================================================================

print("[6/8] Building Models...")

if TORCH_AVAILABLE:
    class SelfAttention(nn.Module):
        """Self-attention mechanism."""
        def __init__(self, hidden_size: int = 128, attention_dim: int = 64):
            super().__init__()
            self.W_a = nn.Linear(hidden_size, attention_dim)
            self.w_a = nn.Linear(attention_dim, 1, bias=False)

        def forward(self, x):
            energy = torch.tanh(self.W_a(x))
            scores = self.w_a(energy).squeeze(-1)
            weights = F.softmax(scores, dim=-1)
            context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
            return context, weights

    class EEGEncoder(nn.Module):
        """1D-CNN + Bi-LSTM + Attention encoder."""
        def __init__(self, n_channels: int = 32, n_time: int = 512, dropout: float = 0.3):
            super().__init__()

            # Conv blocks
            self.conv1 = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            )
            self.conv3 = nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            )

            # Bi-LSTM
            self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)

            # Attention
            self.attention = SelfAttention(128, 64)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = x.permute(0, 2, 1)
            x, _ = self.lstm(x)
            context, attn_weights = self.attention(x)
            return context, attn_weights

    class GenAIRAGEEG(nn.Module):
        """Complete GenAI-RAG-EEG model."""
        def __init__(self, n_channels: int = 32, n_time: int = 512, n_classes: int = 2, dropout: float = 0.3):
            super().__init__()
            self.encoder = EEGEncoder(n_channels, n_time, dropout)
            self.classifier = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, n_classes)
            )

        def forward(self, x):
            features, attn = self.encoder(x)
            logits = self.classifier(features)
            return {'logits': logits, 'features': features, 'attention': attn}

        def predict(self, x):
            output = self.forward(x)
            return torch.argmax(output['logits'], dim=1)

    # Baseline models for comparison
    class SimpleCNN(nn.Module):
        """Simple CNN baseline."""
        def __init__(self, n_channels: int = 32, n_time: int = 512, n_classes: int = 2):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(n_channels, 32, 7, padding=3),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(32, 64, 5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            self.classifier = nn.Linear(64, n_classes)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return {'logits': self.classifier(x)}

    class SimpleLSTM(nn.Module):
        """Simple LSTM baseline."""
        def __init__(self, n_channels: int = 32, n_time: int = 512, n_classes: int = 2):
            super().__init__()
            self.lstm = nn.LSTM(n_channels, 64, batch_first=True, bidirectional=True)
            self.classifier = nn.Linear(128, n_classes)

        def forward(self, x):
            x = x.permute(0, 2, 1)  # (batch, time, channels)
            _, (h, _) = self.lstm(x)
            h = torch.cat([h[-2], h[-1]], dim=1)
            return {'logits': self.classifier(h)}

    print("  ✓ GenAI-RAG-EEG model defined")
    print("  ✓ SimpleCNN baseline defined")
    print("  ✓ SimpleLSTM baseline defined")

    # Count parameters
    test_model = GenAIRAGEEG(32, 512, 2)
    total_params = sum(p.numel() for p in test_model.parameters())
    trainable_params = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
    print(f"  ✓ GenAI-RAG-EEG parameters: {trainable_params:,} trainable / {total_params:,} total")

else:
    print("  ✗ PyTorch not available, skipping model definition")

print()

# ============================================================================
# SECTION 7: TRAINING, VALIDATION, AND TESTING
# ============================================================================

print("[7/8] Model Training, Validation & Testing...")

training_results = {}

if TORCH_AVAILABLE:
    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output['logits'], batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(output['logits'], dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        return total_loss / len(loader), correct / total

    def evaluate(model, loader, criterion, device):
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                output = model(batch_x)
                loss = criterion(output['logits'], batch_y)

                total_loss += loss.item()
                probs = F.softmax(output['logits'], dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        metrics = {
            'loss': total_loss / len(loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='binary'),
            'precision': precision_score(all_labels, all_preds, average='binary'),
            'recall': recall_score(all_labels, all_preds, average='binary'),
            'mcc': matthews_corrcoef(all_labels, all_preds)
        }

        try:
            metrics['auc'] = roc_auc_score(all_labels, all_probs)
        except:
            metrics['auc'] = 0.5

        return metrics, all_preds, all_labels

    # Train on each dataset
    for dataset_name in ['sam40', 'deap', 'eegmat']:
        print(f"\n  Training on {dataset_name.upper()}...")

        dataset = datasets[dataset_name]
        data = dataset['data_zscore']
        labels = dataset['labels_processed']
        n_channels = data.shape[1]
        n_time = data.shape[2]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, random_state=SEED, stratify=labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train
        )

        print(f"    Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Initialize models
        models = {
            'GenAI-RAG-EEG': GenAIRAGEEG(n_channels, n_time, 2, dropout=0.3),
            'SimpleCNN': SimpleCNN(n_channels, n_time, 2),
            'SimpleLSTM': SimpleLSTM(n_channels, n_time, 2)
        }

        dataset_results = {}

        for model_name, model in models.items():
            model = model.to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

            # Training loop
            n_epochs = 20
            best_val_acc = 0
            best_model_state = None
            patience_counter = 0

            history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

            start_time = time.time()

            for epoch in range(n_epochs):
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
                val_metrics, _, _ = evaluate(model, val_loader, criterion, DEVICE)

                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_metrics['loss'])
                history['val_acc'].append(val_metrics['accuracy'])

                scheduler.step(val_metrics['loss'])

                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= 5:
                    break

            training_time = time.time() - start_time

            # Restore best model
            if best_model_state:
                model.load_state_dict(best_model_state)

            # Final evaluation on test set
            test_metrics, test_preds, test_labels = evaluate(model, test_loader, criterion, DEVICE)

            dataset_results[model_name] = {
                'training_time': training_time,
                'epochs_trained': len(history['train_loss']),
                'best_val_accuracy': best_val_acc,
                'test_metrics': {k: float(v) for k, v in test_metrics.items()},
                'history': {k: [float(x) for x in v] for k, v in history.items()}
            }

            print(f"    {model_name}: Test Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")

        training_results[dataset_name] = dataset_results

else:
    print("  ✗ PyTorch not available, skipping training")

print("\n  ✓ Training complete")
print()

# ============================================================================
# SECTION 8: BENCHMARKING & REPORT
# ============================================================================

print("[8/8] Benchmarking & Report Generation...")

# Compile benchmark results
benchmark_results = {}

for dataset_name, results in training_results.items():
    print(f"\n  {dataset_name.upper()} Results:")
    print("  " + "-" * 60)
    print(f"  {'Model':<20} {'Accuracy':>10} {'F1':>10} {'AUC':>10} {'MCC':>10}")
    print("  " + "-" * 60)

    benchmark_results[dataset_name] = {}

    for model_name, model_results in results.items():
        metrics = model_results['test_metrics']
        print(f"  {model_name:<20} {metrics['accuracy']:>10.4f} {metrics['f1']:>10.4f} {metrics['auc']:>10.4f} {metrics['mcc']:>10.4f}")

        benchmark_results[dataset_name][model_name] = {
            'accuracy': metrics['accuracy'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'auc': metrics['auc'],
            'mcc': metrics['mcc']
        }

# Calculate improvements
print("\n  Performance Improvement (GenAI-RAG-EEG vs baselines):")
print("  " + "-" * 60)

for dataset_name in benchmark_results:
    if 'GenAI-RAG-EEG' in benchmark_results[dataset_name]:
        ours = benchmark_results[dataset_name]['GenAI-RAG-EEG']['accuracy']
        for baseline in ['SimpleCNN', 'SimpleLSTM']:
            if baseline in benchmark_results[dataset_name]:
                baseline_acc = benchmark_results[dataset_name][baseline]['accuracy']
                improvement = (ours - baseline_acc) * 100
                print(f"  {dataset_name.upper()} vs {baseline}: +{improvement:.2f}%")

# Create comprehensive report
report = {
    'timestamp': datetime.now().isoformat(),
    'configuration': {
        'seed': SEED,
        'device': DEVICE,
        'torch_available': TORCH_AVAILABLE,
        'sklearn_available': SKLEARN_AVAILABLE,
        'scipy_available': SCIPY_AVAILABLE
    },
    'datasets': {
        name: {
            'config': asdict(ds['config']),
            'samples': ds['data'].shape[0],
            'channels': ds['data'].shape[1],
            'time_samples': ds['data'].shape[2]
        }
        for name, ds in datasets.items()
    },
    'preprocessing': preprocessing_results,
    'normalization': normalization_results,
    'eda': eda_results,
    'training': training_results,
    'benchmarks': benchmark_results
}

# Save report
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

report_path = results_dir / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f"\n  ✓ Report saved to {report_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("PIPELINE EXECUTION SUMMARY")
print("=" * 70)

print("\n📊 DATA SUMMARY:")
for name, ds in datasets.items():
    info = report['datasets'][name]
    print(f"  {name.upper()}: {info['samples']} samples, {info['channels']} channels, {info['time_samples']} time points")

print("\n🔧 PREPROCESSING:")
for name, prep in preprocessing_results.items():
    print(f"  {name.upper()}: {prep['retained_samples']}/{prep['original_samples']} samples retained ({100*(1-prep['rejection_rate']):.1f}%)")

print("\n📈 EDA HIGHLIGHTS:")
for name, eda in eda_results.items():
    cd = eda['class_distribution']
    st = eda['statistical_test']
    sig = "✓ significant" if st.get('significant') else "✗ not significant"
    print(f"  {name.upper()}: {cd['class_0_count']} vs {cd['class_1_count']} (ratio: {cd['balance_ratio']:.2f}), t-test: {sig}")

print("\n🏆 MODEL PERFORMANCE:")
for name, results in benchmark_results.items():
    if 'GenAI-RAG-EEG' in results:
        m = results['GenAI-RAG-EEG']
        print(f"  {name.upper()} (GenAI-RAG-EEG): Acc={m['accuracy']:.4f}, F1={m['f1']:.4f}, AUC={m['auc']:.4f}")

print("\n" + "=" * 70)
print(f"Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
