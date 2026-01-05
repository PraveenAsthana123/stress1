#!/usr/bin/env python3
"""
SAM-40 Balanced Training with Subject-wise Normalization
Target: 90%+ accuracy
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from scipy import signal
from scipy.stats import skew, kurtosis
import scipy.io as sio

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

PROJECT_ROOT = Path("/media/praveen/Asthana3/rajveer/eeg-stress-rag")
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def hjorth_parameters(sig):
    """Compute Hjorth parameters: Activity, Mobility, Complexity."""
    activity = np.var(sig)
    diff1 = np.diff(sig)
    diff2 = np.diff(diff1)
    mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
    complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)
    return activity, mobility, complexity


def spectral_entropy(sig, fs=128):
    """Compute spectral entropy."""
    freqs, psd = signal.welch(sig, fs=fs, nperseg=min(64, len(sig)))
    psd_norm = psd / (np.sum(psd) + 1e-10)
    return -np.sum(psd_norm * np.log2(psd_norm + 1e-10))


def extract_features(eeg, fs=128):
    """Extract comprehensive EEG features."""
    features = []
    n_channels = min(eeg.shape[0], 32)

    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}

    all_band_powers = {b: [] for b in bands}

    for ch in range(n_channels):
        sig = eeg[ch]

        # Band powers
        freqs, psd = signal.welch(sig, fs=fs, nperseg=min(64, len(sig)))
        total_power = np.sum(psd) + 1e-10

        for band_name, (low, high) in bands.items():
            idx = (freqs >= low) & (freqs <= high)
            bp = np.sum(psd[idx])
            all_band_powers[band_name].append(bp)
            features.extend([np.log1p(bp), bp / total_power])

        # Hjorth parameters
        activity, mobility, complexity = hjorth_parameters(sig)
        features.extend([activity, mobility, complexity])

        # Statistical
        features.extend([
            np.mean(sig), np.std(sig), skew(sig), kurtosis(sig),
            np.sqrt(np.mean(sig**2)), np.max(sig) - np.min(sig)
        ])

        # Spectral entropy
        features.append(spectral_entropy(sig, fs))

    # Global ratios (stress biomarkers)
    alpha = np.mean(all_band_powers['alpha']) + 1e-10
    beta = np.mean(all_band_powers['beta']) + 1e-10
    theta = np.mean(all_band_powers['theta']) + 1e-10

    features.extend([
        beta / alpha,  # Stress indicator
        theta / beta,
        (alpha + theta) / beta,
        np.std(all_band_powers['alpha']) / alpha,  # Alpha variability
        np.std(all_band_powers['beta']) / beta,    # Beta variability
    ])

    # Frontal asymmetry (if enough channels)
    if n_channels >= 4:
        left_alpha = np.mean([all_band_powers['alpha'][i] for i in [0, 2] if i < n_channels])
        right_alpha = np.mean([all_band_powers['alpha'][i] for i in [1, 3] if i < n_channels])
        features.append(np.log(right_alpha + 1e-10) - np.log(left_alpha + 1e-10))

    return np.array(features)


def load_sam40():
    """Load SAM-40 with subject info for per-subject normalization."""
    sam40_path = DATA_DIR / "SAM40" / "filtered_data"

    if not sam40_path.exists():
        return None, None, None

    print(f"Loading SAM-40 from {sam40_path}")

    X_list, y_list, subjects = [], [], []

    for mat_file in sorted(sam40_path.glob("*.mat")):
        try:
            data = sio.loadmat(mat_file, squeeze_me=True)
            filename = mat_file.stem

            if filename.startswith('Relax'):
                label = 0
            elif any(filename.startswith(x) for x in ['Arithmetic', 'Mirror', 'Stroop']):
                label = 1
            else:
                continue

            parts = filename.split('_')
            subj_idx = parts.index('sub') + 1 if 'sub' in parts else 0
            subj_id = int(parts[subj_idx]) if subj_idx > 0 else 0

            for key in data.keys():
                if not key.startswith('__'):
                    val = data[key]
                    if isinstance(val, np.ndarray) and len(val.shape) >= 2:
                        eeg = val.T if val.shape[0] > val.shape[1] else val
                        X_list.append(eeg)
                        y_list.append(label)
                        subjects.append(subj_id)
                        break
        except:
            continue

    print(f"  Loaded {len(X_list)} samples from {len(set(subjects))} subjects")
    print(f"  Classes: {sum(1 for y in y_list if y==0)} Relax, {sum(1 for y in y_list if y==1)} Stress")

    return X_list, np.array(y_list), np.array(subjects)


def per_subject_normalize(X_list, subjects):
    """Normalize EEG data per subject."""
    normalized = []
    unique_subjects = np.unique(subjects)

    for subj in unique_subjects:
        subj_idx = np.where(subjects == subj)[0]

        # Get all data for this subject
        subj_data = [X_list[i] for i in subj_idx]

        # Compute subject-level mean and std
        all_data = np.concatenate([d.flatten() for d in subj_data])
        mean, std = np.mean(all_data), np.std(all_data) + 1e-10

        # Normalize each sample
        for i in subj_idx:
            normalized.append((X_list[i] - mean) / std)

    return normalized


def main():
    print("="*60)
    print("SAM-40 Balanced Training")
    print("Method: SMOTE + Ensemble + Per-Subject Normalization")
    print("="*60)

    X_list, y, subjects = load_sam40()
    if X_list is None:
        return

    # Per-subject normalization
    print("\nApplying per-subject normalization...")
    X_norm = per_subject_normalize(X_list, subjects)

    # Extract features
    print("Extracting features...")
    X_features = []
    for i, eeg in enumerate(X_norm):
        features = extract_features(eeg)
        X_features.append(features)

    X = np.array(X_features)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    print(f"Feature shape: {X.shape}")

    # Train with SMOTE balancing
    print(f"\n{'='*60}")
    print("Training with SMOTE Balancing")
    print(f"{'='*60}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds, all_proba, all_true = [], [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # SMOTE for balancing
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

        # Train ensemble
        rf = RandomForestClassifier(n_estimators=500, max_depth=15, class_weight='balanced',
                                    random_state=42, n_jobs=-1)
        rf.fit(X_train_bal, y_train_bal)

        preds = rf.predict(X_val_scaled)
        proba = rf.predict_proba(X_val_scaled)[:, 1]

        acc = accuracy_score(y_val, preds)
        print(f"Fold {fold+1}: {acc*100:.2f}%")

        all_preds.extend(preds)
        all_proba.extend(proba)
        all_true.extend(y_val)

    y_pred = np.array(all_preds)
    y_proba = np.array(all_proba)
    y_true = np.array(all_true)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"F1 Score: {f1*100:.2f}%")
    print(f"AUC-ROC:  {auc*100:.2f}%")
    print(f"Kappa:    {kappa:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # Save
    results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "method": "SMOTE + RandomForest + Per-Subject Normalization",
            "task": "Binary (Stress vs Relax)"
        },
        "results": {
            "accuracy": round(acc * 100, 2),
            "f1_macro": round(f1 * 100, 2),
            "auc_roc": round(auc * 100, 2),
            "kappa": round(kappa, 4),
            "confusion_matrix": cm.tolist()
        }
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "sam40_balanced_v2_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to: {RESULTS_DIR / 'sam40_balanced_v2_results.json'}")


if __name__ == "__main__":
    main()
