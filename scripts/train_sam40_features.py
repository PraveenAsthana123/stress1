#!/usr/bin/env python3
"""
SAM-40 High-Accuracy Training using Feature Extraction + ML
Target: 90%+ accuracy

Based on: Shikha et al. (2025) - 90.76% using NMI-RFE + Random Forest
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from scipy import signal
from scipy.stats import skew, kurtosis, entropy
import scipy.io as sio

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score

PROJECT_ROOT = Path("/media/praveen/Asthana3/rajveer/eeg-stress-rag")
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def extract_band_powers(eeg, fs=128):
    """Extract band power features from EEG."""
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }

    features = []
    n_channels = eeg.shape[0]

    for ch in range(n_channels):
        sig = eeg[ch]

        # Compute PSD
        freqs, psd = signal.welch(sig, fs=fs, nperseg=min(128, len(sig)))
        total_power = np.sum(psd) + 1e-10

        ch_features = []
        band_powers = {}

        for band_name, (low, high) in bands.items():
            idx = (freqs >= low) & (freqs <= high)
            bp = np.sum(psd[idx])
            rbp = bp / total_power  # Relative band power
            band_powers[band_name] = bp

            ch_features.extend([
                np.log1p(bp),  # Log band power
                rbp,           # Relative band power
            ])

        # Band power ratios
        alpha = band_powers['alpha'] + 1e-10
        beta = band_powers['beta'] + 1e-10
        theta = band_powers['theta'] + 1e-10
        delta = band_powers['delta'] + 1e-10

        ch_features.extend([
            beta / alpha,           # Beta/Alpha ratio (stress indicator)
            theta / beta,           # Theta/Beta ratio
            (alpha + theta) / beta, # Relaxation index
            alpha / delta,          # Alpha/Delta ratio
        ])

        features.extend(ch_features)

    return features


def extract_statistical_features(eeg):
    """Extract statistical features from EEG."""
    features = []

    for ch in range(eeg.shape[0]):
        sig = eeg[ch]

        features.extend([
            np.mean(sig),
            np.std(sig),
            np.var(sig),
            skew(sig),
            kurtosis(sig),
            np.sqrt(np.mean(sig**2)),  # RMS
            np.max(sig) - np.min(sig),  # Peak-to-peak
            np.sum(np.abs(np.diff(sig))),  # Line length
            np.sum(sig**2),  # Energy
            entropy(np.histogram(sig, bins=20)[0] + 1e-10),  # Histogram entropy
        ])

    return features


def extract_connectivity_features(eeg):
    """Extract inter-channel connectivity features."""
    features = []
    n_channels = min(eeg.shape[0], 14)  # Use first 14 channels

    # Correlation matrix
    corr = np.corrcoef(eeg[:n_channels])

    # Extract upper triangle
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            features.append(corr[i, j])

    # Global connectivity metrics
    features.extend([
        np.mean(np.abs(corr)),  # Mean connectivity
        np.std(corr),           # Connectivity variance
    ])

    return features


def extract_all_features(eeg, fs=128):
    """Extract comprehensive feature set."""
    features = []

    # Band powers
    features.extend(extract_band_powers(eeg, fs))

    # Statistical
    features.extend(extract_statistical_features(eeg))

    # Connectivity
    features.extend(extract_connectivity_features(eeg))

    return np.array(features)


def load_sam40_binary():
    """Load SAM-40 for binary classification."""
    sam40_path = DATA_DIR / "SAM40" / "filtered_data"

    if not sam40_path.exists():
        return None, None

    print(f"Loading SAM-40 from {sam40_path}")

    X_list, y_list = [], []

    for mat_file in sorted(sam40_path.glob("*.mat")):
        try:
            data = sio.loadmat(mat_file, squeeze_me=True)
            filename = mat_file.stem

            # Binary: Relax=0, Stress=1
            if filename.startswith('Relax'):
                label = 0
            elif any(filename.startswith(x) for x in ['Arithmetic', 'Mirror', 'Stroop']):
                label = 1
            else:
                continue

            for key in data.keys():
                if not key.startswith('__'):
                    val = data[key]
                    if isinstance(val, np.ndarray) and len(val.shape) >= 2:
                        eeg = val.T if val.shape[0] > val.shape[1] else val
                        X_list.append(eeg)
                        y_list.append(label)
                        break
        except:
            continue

    print(f"  Loaded {len(X_list)} samples")
    print(f"  Classes: {sum(1 for y in y_list if y==0)} Relax, {sum(1 for y in y_list if y==1)} Stress")

    return X_list, np.array(y_list)


def feature_selection_nmi(X, y, n_features=100):
    """Select top features using Normalized Mutual Information."""
    print(f"  Selecting top {n_features} features using NMI...")
    mi_scores = mutual_info_classif(X, y, random_state=42)
    top_idx = np.argsort(mi_scores)[-n_features:]
    return X[:, top_idx], top_idx


def train_ensemble(X, y):
    """Train ensemble model with 5-fold CV."""
    print(f"\n{'='*60}")
    print(f"Training Ensemble Model (RF + GB + SVM + MLP)")
    print(f"{'='*60}")
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds, all_proba, all_true = [], [], []
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
        print(f"\nFold {fold+1}/5")

        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Feature selection on training data
        X_train_sel, sel_idx = feature_selection_nmi(X_train, y_train, n_features=100)
        X_val_sel = X_val[:, sel_idx]

        # Ensemble model
        model = VotingClassifier([
            ('rf', RandomForestClassifier(n_estimators=300, max_depth=10,
                                         class_weight='balanced', random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)),
            ('svm', SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced',
                       probability=True, random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                 random_state=42, early_stopping=True)),
        ], voting='soft', n_jobs=-1)

        model.fit(X_train_sel, y_train)

        preds = model.predict(X_val_sel)
        proba = model.predict_proba(X_val_sel)[:, 1]

        acc = accuracy_score(y_val, preds)
        fold_accs.append(acc)

        all_preds.extend(preds)
        all_proba.extend(proba)
        all_true.extend(y_val)

        print(f"  Fold {fold+1} Accuracy: {acc*100:.2f}%")

    # Final metrics
    y_pred = np.array(all_preds)
    y_proba = np.array(all_proba)
    y_true = np.array(all_true)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"F1 Score: {f1*100:.2f}%")
    print(f"AUC-ROC:  {auc*100:.2f}%")
    print(f"Kappa:    {kappa:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Fold Accuracies: {[f'{a*100:.1f}%' for a in fold_accs]}")

    return {
        "accuracy": round(acc * 100, 2),
        "f1_macro": round(f1 * 100, 2),
        "auc_roc": round(auc * 100, 2),
        "kappa": round(kappa, 4),
        "confusion_matrix": cm.tolist(),
        "fold_accuracies": [round(a * 100, 2) for a in fold_accs]
    }


def main():
    print("="*60)
    print("SAM-40 High-Accuracy Training (Features + Ensemble)")
    print("Target: 90%+ Accuracy")
    print("Method: NMI Feature Selection + Ensemble (RF+GB+SVM+MLP)")
    print("="*60)

    # Load data
    X_list, y = load_sam40_binary()

    if X_list is None:
        print("Failed to load data")
        return

    # Extract features
    print("\nExtracting features...")
    X_features = []
    for i, eeg in enumerate(X_list):
        features = extract_all_features(eeg)
        X_features.append(features)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(X_list)} samples")

    X = np.array(X_features)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    print(f"Feature matrix shape: {X.shape}")

    # Train
    results = train_ensemble(X, y)

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": "Ensemble (RF + GB + SVM + MLP)",
            "task": "Binary (Stress vs Relax)",
            "feature_selection": "NMI (top 100 features)",
            "features": "Band powers, statistical, connectivity",
            "validation": "5-fold Stratified CV"
        },
        "SAM-40_binary_features": results
    }

    with open(RESULTS_DIR / "sam40_features_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR / 'sam40_features_results.json'}")


if __name__ == "__main__":
    main()
