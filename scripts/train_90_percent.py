#!/usr/bin/env python3
"""
================================================================================
Optimized Training for 90% Accuracy on EEG Stress Classification
================================================================================

Strategy: Feature-based approach with ensemble learning
- Extract robust EEG features (band power, statistical, connectivity)
- Use ensemble of classifiers (RF, XGBoost, SVM, MLP)
- Proper cross-validation
- Feature selection

This approach works better on small datasets than deep learning.
================================================================================
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix, cohen_kappa_score, classification_report
)
from sklearn.pipeline import Pipeline

# Imbalanced learning
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

# Signal processing
from scipy import signal
from scipy.stats import skew, kurtosis, entropy

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "paper"


# ==============================================================================
# FEATURE EXTRACTION
# ==============================================================================

def extract_band_power(data, fs=256):
    """Extract band power features."""
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }

    n_samples, n_channels, n_timepoints = data.shape
    features = []

    for i in range(n_samples):
        sample_features = []
        for ch in range(n_channels):
            sig = data[i, ch]

            # Compute PSD
            freqs, psd = signal.welch(sig, fs=fs, nperseg=min(256, n_timepoints))

            # Band powers
            for band_name, (low, high) in bands.items():
                idx = np.logical_and(freqs >= low, freqs <= high)
                bp = np.mean(psd[idx]) if np.any(idx) else 0
                sample_features.append(bp)

            # Relative band powers
            total_power = np.sum(psd)
            for band_name, (low, high) in bands.items():
                idx = np.logical_and(freqs >= low, freqs <= high)
                rbp = np.sum(psd[idx]) / (total_power + 1e-10)
                sample_features.append(rbp)

        features.append(sample_features)

    return np.array(features)


def extract_statistical_features(data):
    """Extract statistical features from EEG."""
    n_samples, n_channels, n_timepoints = data.shape
    features = []

    for i in range(n_samples):
        sample_features = []
        for ch in range(n_channels):
            sig = data[i, ch]

            # Time domain features
            sample_features.extend([
                np.mean(sig),
                np.std(sig),
                np.var(sig),
                skew(sig),
                kurtosis(sig),
                np.max(sig) - np.min(sig),  # Peak-to-peak
                np.sqrt(np.mean(sig**2)),   # RMS
                np.sum(np.abs(np.diff(sig))),  # Line length
                np.sum(sig**2),  # Energy
                np.mean(np.abs(sig)),  # Mean absolute value
            ])

            # Zero crossings
            zc = np.sum(np.diff(np.signbit(sig).astype(int)))
            sample_features.append(zc)

            # Hjorth parameters
            diff1 = np.diff(sig)
            diff2 = np.diff(diff1)

            var0 = np.var(sig)
            var1 = np.var(diff1)
            var2 = np.var(diff2)

            mobility = np.sqrt(var1 / (var0 + 1e-10))
            complexity = np.sqrt(var2 / (var1 + 1e-10)) / (mobility + 1e-10)

            sample_features.extend([mobility, complexity])

        features.append(sample_features)

    return np.array(features)


def extract_connectivity_features(data):
    """Extract connectivity features (correlation between channels)."""
    n_samples, n_channels, _ = data.shape
    features = []

    for i in range(n_samples):
        # Correlation matrix
        corr = np.corrcoef(data[i])

        # Upper triangle (excluding diagonal)
        idx = np.triu_indices(n_channels, k=1)
        corr_features = corr[idx]

        # Statistics of correlations
        feat = [
            np.mean(corr_features),
            np.std(corr_features),
            np.max(corr_features),
            np.min(corr_features),
            np.median(corr_features)
        ]

        features.append(feat)

    return np.array(features)


def extract_ratio_features(band_powers, n_channels):
    """Extract band power ratio features (important for stress detection)."""
    n_samples = band_powers.shape[0]
    n_bands = 5  # delta, theta, alpha, beta, gamma

    features = []

    for i in range(n_samples):
        sample_features = []

        for ch in range(n_channels):
            start_idx = ch * n_bands * 2  # *2 because we have absolute and relative

            delta = band_powers[i, start_idx + 0]
            theta = band_powers[i, start_idx + 1]
            alpha = band_powers[i, start_idx + 2]
            beta = band_powers[i, start_idx + 3]
            gamma = band_powers[i, start_idx + 4]

            # Important ratios for stress detection
            sample_features.extend([
                beta / (alpha + 1e-10),      # Beta/Alpha (stress indicator)
                theta / (alpha + 1e-10),     # Theta/Alpha
                (theta + alpha) / (beta + 1e-10),  # (Theta+Alpha)/Beta
                beta / (theta + 1e-10),      # Beta/Theta
                gamma / (beta + 1e-10),      # Gamma/Beta
                (alpha + beta) / (delta + theta + 1e-10),  # Engagement index
            ])

        features.append(sample_features)

    return np.array(features)


def extract_all_features(data, fs=256):
    """Extract all features from EEG data."""
    print("  Extracting band power features...")
    band_features = extract_band_power(data, fs)

    print("  Extracting statistical features...")
    stat_features = extract_statistical_features(data)

    print("  Extracting connectivity features...")
    conn_features = extract_connectivity_features(data)

    print("  Extracting ratio features...")
    ratio_features = extract_ratio_features(band_features, data.shape[1])

    # Combine all features
    all_features = np.hstack([
        band_features,
        stat_features,
        conn_features,
        ratio_features
    ])

    print(f"  Total features: {all_features.shape[1]}")

    return all_features


# ==============================================================================
# MODEL TRAINING
# ==============================================================================

def create_ensemble():
    """Create ensemble of classifiers."""

    # Base classifiers
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    svm = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        max_iter=500,
        early_stopping=True,
        random_state=42
    )

    knn = KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',
        n_jobs=-1
    )

    # Voting ensemble
    voting = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('svm', svm),
            ('mlp', mlp),
            ('knn', knn)
        ],
        voting='soft',
        n_jobs=-1
    )

    return voting


def train_with_features(X, y, dataset_name):
    """Train using feature-based approach with ensemble."""

    print(f"\n{'='*60}")
    print(f"TRAINING: {dataset_name}")
    print(f"{'='*60}")
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"Class distribution: {np.bincount(y)}")

    # Handle NaN and Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Feature scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle any remaining NaN after scaling
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Feature selection - keep top features
    n_features = min(300, X.shape[1])
    selector = SelectKBest(f_classif, k=n_features)
    X_selected = selector.fit_transform(X_scaled, y)
    print(f"Selected top {n_features} features")

    # Apply SMOTE for class balancing
    if IMBLEARN_AVAILABLE and np.bincount(y)[0] != np.bincount(y)[1]:
        print("Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_selected, y)
        print(f"After SMOTE: {np.bincount(y_resampled)}")
    else:
        X_resampled, y_resampled = X_selected, y

    # Cross-validation with custom training
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_preds = []
    all_proba = []
    all_true = []

    print("\nTraining ensemble with cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_selected, y)):
        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Apply SMOTE only on training data
        if IMBLEARN_AVAILABLE:
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train

        # Train ensemble
        ensemble = create_ensemble()
        ensemble.fit(X_train_res, y_train_res)

        # Predict
        preds = ensemble.predict(X_val)
        proba = ensemble.predict_proba(X_val)[:, 1]

        all_preds.extend(preds)
        all_proba.extend(proba)
        all_true.extend(y_val)

        fold_acc = accuracy_score(y_val, preds)
        print(f"  Fold {fold+1}: Acc={fold_acc*100:.1f}%")

    y_pred = np.array(all_preds)
    y_proba = np.array(all_proba)
    y = np.array(all_true)

    # Compute metrics
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results = {
        "accuracy": round(accuracy_score(y, y_pred) * 100, 2),
        "precision": round(precision_score(y, y_pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y, y_pred, zero_division=0) * 100, 2),
        "f1_score": round(f1_score(y, y_pred, zero_division=0) * 100, 2),
        "auc_roc": round(roc_auc_score(y, y_proba) * 100, 2),
        "specificity": round(tn / (tn + fp) * 100, 2) if (tn + fp) > 0 else 0,
        "cohens_kappa": round(cohen_kappa_score(y, y_pred), 4),
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "raw": cm.tolist()
        }
    }

    print(f"\n{dataset_name} RESULTS:")
    print(f"  Accuracy:    {results['accuracy']}%")
    print(f"  F1 Score:    {results['f1_score']}%")
    print(f"  AUC-ROC:     {results['auc_roc']}%")
    print(f"  Precision:   {results['precision']}%")
    print(f"  Recall:      {results['recall']}%")
    print(f"  Specificity: {results['specificity']}%")
    print(f"  Kappa:       {results['cohens_kappa']}")
    print(f"  CM: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return results


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_sam40():
    """Load SAM-40 dataset."""
    sam40_path = DATA_DIR / "SAM40" / "filtered_data"
    if not sam40_path.exists():
        return None, None

    try:
        import scipy.io as sio
    except:
        return None, None

    print(f"Loading SAM-40 from {sam40_path}")

    X_list, y_list = [], []

    for mat_file in sorted(sam40_path.glob("*.mat")):
        try:
            data = sio.loadmat(mat_file, squeeze_me=True)
            filename = mat_file.stem
            label = 0 if filename.startswith('Relax') else 1

            for key in data.keys():
                if not key.startswith('__'):
                    val = data[key]
                    if isinstance(val, np.ndarray) and len(val.shape) >= 2:
                        eeg = val.T if val.shape[0] > val.shape[1] else val

                        # Standardize size
                        n_ch, n_tp = eeg.shape
                        eeg_std = np.zeros((32, 512))
                        eeg_std[:min(n_ch, 32), :min(n_tp, 512)] = eeg[:min(n_ch, 32), :min(n_tp, 512)]

                        X_list.append(eeg_std)
                        y_list.append(label)
                        break
        except:
            continue

    if X_list:
        X = np.array(X_list)
        y = np.array(y_list)
        print(f"  Loaded: {len(X)} samples, {sum(y==0)} baseline, {sum(y==1)} stress")
        return X, y

    return None, None


def load_eegmat():
    """Load EEGMAT dataset."""
    sample_path = DATA_DIR / "EEGMAT" / "sample_100"
    if not sample_path.exists():
        return None, None

    print(f"Loading EEGMAT from {sample_path}")

    try:
        X_file = sample_path / "X_eegmat_100.npy"
        y_file = sample_path / "y_eegmat_100.npy"
        if not X_file.exists():
            X_file = sample_path / "X.npy"
            y_file = sample_path / "y.npy"

        X = np.load(X_file)
        y = np.load(y_file)

        # Reshape
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 32, -1)

        # Standardize
        X_std = np.zeros((X.shape[0], 32, 512))
        for i in range(X.shape[0]):
            ch = min(X.shape[1], 32)
            tp = min(X.shape[2], 512)
            X_std[i, :ch, :tp] = X[i, :ch, :tp]

        print(f"  Loaded: {len(X_std)} samples, {sum(y==0)} baseline, {sum(y==1)} stress")
        return X_std, y

    except Exception as e:
        print(f"  Error: {e}")
        return None, None


def plot_confusion_matrices(results_dict, save_path):
    """Plot confusion matrices."""
    n = len(results_dict)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]

    for idx, (name, data) in enumerate(results_dict.items()):
        ax = axes[idx]
        cm = np.array(data["confusion_matrix"]["raw"])
        cm_pct = cm / cm.sum() * 100

        sns.heatmap(cm_pct, annot=False, cmap='Blues', ax=ax,
                   xticklabels=['Baseline', 'Stress'],
                   yticklabels=['Baseline', 'Stress'], cbar=False, vmin=0, vmax=60)

        for i in range(2):
            for j in range(2):
                color = 'white' if cm_pct[i,j] > 30 else 'black'
                ax.text(j+0.5, i+0.35, f'{cm_pct[i,j]:.1f}%', ha='center', va='center',
                       fontsize=14, fontweight='bold', color=color)
                ax.text(j+0.5, i+0.65, f'(n={cm[i,j]})', ha='center', va='center',
                       fontsize=10, color=color)

        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title(f'{name}\n(Acc: {data["accuracy"]:.1f}%, F1: {data["f1_score"]:.1f}%)',
                    fontsize=13, fontweight='bold')

    plt.suptitle('Confusion Matrices - Feature-Based Ensemble Model\n(5-Fold Stratified CV)',
                fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("="*60)
    print("FEATURE-BASED ENSEMBLE TRAINING FOR 90% ACCURACY")
    print("="*60)
    print(f"Started: {datetime.now()}")

    results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": "Feature-Based Ensemble (RF + GB + SVM + MLP + KNN)",
            "features": [
                "Band power (delta, theta, alpha, beta, gamma)",
                "Relative band power",
                "Statistical (mean, std, skew, kurtosis, etc.)",
                "Hjorth parameters (mobility, complexity)",
                "Connectivity (correlation)",
                "Band power ratios (beta/alpha, theta/alpha, etc.)"
            ],
            "validation": "5-Fold Stratified CV",
            "is_real_data": True
        },
        "datasets": {}
    }

    # Train SAM-40
    X_sam40, y_sam40 = load_sam40()
    if X_sam40 is not None:
        print("\nExtracting features for SAM-40...")
        X_features = extract_all_features(X_sam40)
        results["datasets"]["SAM-40"] = train_with_features(X_features, y_sam40, "SAM-40")

    # Train EEGMAT
    X_eegmat, y_eegmat = load_eegmat()
    if X_eegmat is not None:
        print("\nExtracting features for EEGMAT...")
        X_features = extract_all_features(X_eegmat)
        results["datasets"]["EEGMAT"] = train_with_features(X_features, y_eegmat, "EEGMAT")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "feature_ensemble_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_DIR / 'feature_ensemble_results.json'}")

    # Plot confusion matrices
    if results["datasets"]:
        OUTPUT_DIR.mkdir(exist_ok=True)
        plot_confusion_matrices(results["datasets"], OUTPUT_DIR / "fig11_confusion_matrices_90pct.png")

    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

    for name, data in results["datasets"].items():
        print(f"\n{name}:")
        print(f"  Accuracy: {data['accuracy']}%")
        print(f"  F1 Score: {data['f1_score']}%")
        print(f"  AUC-ROC:  {data['auc_roc']}%")
        print(f"  Kappa:    {data['cohens_kappa']}")

    # Check if we hit 90%
    accuracies = [d['accuracy'] for d in results["datasets"].values()]
    if all(a >= 90 for a in accuracies):
        print("\n*** TARGET 90% ACHIEVED! ***")
    else:
        print(f"\n*** Current best: {max(accuracies):.1f}% ***")

    return results


if __name__ == "__main__":
    main()
