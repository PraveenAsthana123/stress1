#!/usr/bin/env python3
"""
SAM-40 with Data Augmentation and Window Sliding
Target: 90%+ accuracy
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score, roc_auc_score
from scipy import signal
from scipy.stats import skew, kurtosis
import scipy.io as sio

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    IMBLEARN_AVAILABLE = True
except:
    IMBLEARN_AVAILABLE = False

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def extract_features(data, fs=128):
    """Extract EEG features."""
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
    n_samples, n_channels, n_timepoints = data.shape
    all_features = []

    for i in range(n_samples):
        feat = []
        channel_powers = {b: [] for b in bands}

        for ch in range(n_channels):
            sig = data[i, ch]
            nperseg = min(256, len(sig))
            freqs, psd = signal.welch(sig, fs=fs, nperseg=nperseg)
            total = np.sum(psd) + 1e-10

            for name, (low, high) in bands.items():
                idx = np.logical_and(freqs >= low, freqs <= high)
                bp = np.sum(psd[idx])
                channel_powers[name].append(bp)
                feat.extend([bp, bp/total])

            # Ratios
            alpha = np.sum(psd[np.logical_and(freqs >= 8, freqs <= 13)]) + 1e-10
            beta = np.sum(psd[np.logical_and(freqs >= 13, freqs <= 30)]) + 1e-10
            theta = np.sum(psd[np.logical_and(freqs >= 4, freqs <= 8)]) + 1e-10
            feat.extend([beta/alpha, theta/beta, theta/alpha])

            # Stats
            feat.extend([np.mean(sig), np.std(sig), skew(sig), kurtosis(sig)])

        # Cross-channel features
        for name in bands:
            powers = channel_powers[name]
            feat.extend([np.mean(powers), np.std(powers), np.max(powers), np.min(powers)])

        all_features.append(feat)

    return np.nan_to_num(np.array(all_features))


def sliding_window_augment(X, y, window_size=1600, step=800):
    """Augment data with sliding windows."""
    X_aug, y_aug = [], []

    for i in range(len(X)):
        sample = X[i]
        n_ch, n_tp = sample.shape

        for start in range(0, n_tp - window_size + 1, step):
            window = sample[:, start:start+window_size]
            X_aug.append(window)
            y_aug.append(y[i])

    return np.array(X_aug), np.array(y_aug)


def noise_augment(X, y, n_copies=2, noise_level=0.1):
    """Add Gaussian noise augmentation."""
    X_aug, y_aug = [X], [y]

    for _ in range(n_copies):
        noise = np.random.normal(0, noise_level * np.std(X), X.shape)
        X_aug.append(X + noise)
        y_aug.append(y)

    return np.vstack(X_aug), np.concatenate(y_aug)


def load_sam40():
    """Load SAM-40 binary."""
    sam40_path = DATA_DIR / "SAM40" / "filtered_data"
    if not sam40_path.exists():
        return None, None, None

    print(f"Loading SAM-40...")
    X_list, y_list, subj_list = [], [], []

    for mat_file in sorted(sam40_path.glob("*.mat")):
        try:
            data = sio.loadmat(mat_file, squeeze_me=True)
            filename = mat_file.stem
            label = 0 if filename.startswith('Relax') else 1

            # Get subject
            parts = filename.split('_')
            if 'sub' in parts:
                subj = int(parts[parts.index('sub') + 1])
            else:
                subj = 0

            for key in data.keys():
                if not key.startswith('__'):
                    val = data[key]
                    if isinstance(val, np.ndarray) and len(val.shape) >= 2:
                        eeg = val.T if val.shape[0] > val.shape[1] else val
                        n_ch, n_tp = eeg.shape
                        eeg_std = np.zeros((32, 3200))
                        eeg_std[:min(n_ch,32), :min(n_tp,3200)] = eeg[:min(n_ch,32), :min(n_tp,3200)]
                        X_list.append(eeg_std)
                        y_list.append(label)
                        subj_list.append(subj)
                        break
        except:
            continue

    if X_list:
        return np.array(X_list), np.array(y_list), np.array(subj_list)
    return None, None, None


def train_with_augmentation():
    """Train with multiple augmentation strategies."""
    X, y, subjects = load_sam40()
    if X is None:
        return None

    print(f"  Original: {len(X)} samples, Relax={sum(y==0)}, Stress={sum(y==1)}")

    # Sliding window augmentation
    print("Applying sliding window augmentation...")
    X_aug, y_aug = sliding_window_augment(X, y, window_size=1600, step=400)
    print(f"  After sliding window: {len(X_aug)} samples")

    # Extract features
    print("Extracting features...")
    X_feat = extract_features(X_aug, fs=128)
    print(f"  Features: {X_feat.shape[1]}")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)

    # Feature selection
    n_feat = min(400, X_scaled.shape[1])
    selector = SelectKBest(mutual_info_classif, k=n_feat)
    X_sel = selector.fit_transform(X_scaled, y_aug)

    print(f"\nTraining with 5-fold CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_true, y_pred, y_proba = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_sel, y_aug)):
        X_tr, X_val = X_sel[tr_idx], X_sel[val_idx]
        y_tr, y_val = y_aug[tr_idx], y_aug[val_idx]

        # Balance training data
        if IMBLEARN_AVAILABLE:
            try:
                smote = SMOTE(random_state=42)
                X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
            except:
                pass

        # Strong ensemble
        rf = RandomForestClassifier(n_estimators=500, max_depth=25, min_samples_split=2, class_weight='balanced', random_state=42, n_jobs=-1)
        et = ExtraTreesClassifier(n_estimators=500, max_depth=25, min_samples_split=2, class_weight='balanced', random_state=42, n_jobs=-1)
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42)

        # Train all and average predictions
        rf.fit(X_tr, y_tr)
        et.fit(X_tr, y_tr)
        gb.fit(X_tr, y_tr)

        # Average probabilities
        p1 = rf.predict_proba(X_val)[:, 1]
        p2 = et.predict_proba(X_val)[:, 1]
        p3 = gb.predict_proba(X_val)[:, 1]
        prob = (p1 + p2 + p3) / 3

        pred = (prob >= 0.5).astype(int)

        y_true.extend(y_val)
        y_pred.extend(pred)
        y_proba.extend(prob)

        fold_acc = accuracy_score(y_val, pred)
        print(f"  Fold {fold+1}: {fold_acc*100:.1f}%")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results = {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "f1_score": round(f1_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred) * 100, 2),
        "recall": round(recall_score(y_true, y_pred) * 100, 2),
        "specificity": round(tn / (tn + fp) * 100, 2),
        "auc_roc": round(roc_auc_score(y_true, y_proba) * 100, 2),
        "cohens_kappa": round(cohen_kappa_score(y_true, y_pred), 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "n_samples_augmented": len(X_aug)
    }

    print(f"\n{'='*50}")
    print("SAM-40 AUGMENTED RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:    {results['accuracy']}%")
    print(f"F1 Score:    {results['f1_score']}%")
    print(f"Precision:   {results['precision']}%")
    print(f"Recall:      {results['recall']}%")
    print(f"Specificity: {results['specificity']}%")
    print(f"AUC-ROC:     {results['auc_roc']}%")
    print(f"Kappa:       {results['cohens_kappa']}")

    return results


def main():
    print("="*50)
    print("SAM-40 WITH DATA AUGMENTATION")
    print("="*50)
    print(f"Started: {datetime.now()}")

    results = train_with_augmentation()

    if results:
        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "model": "Ensemble (RF+ET+GB) with averaging",
                "augmentation": "Sliding window (1600 samples, step 400)",
                "validation": "5-fold Stratified CV",
                "is_real_data": True
            },
            "SAM-40": results
        }

        RESULTS_DIR.mkdir(exist_ok=True)
        with open(RESULTS_DIR / "sam40_augmented_results.json", 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved: {RESULTS_DIR / 'sam40_augmented_results.json'}")

        if results['accuracy'] >= 90:
            print("\n*** TARGET 90% ACHIEVED! ***")

    return results


if __name__ == "__main__":
    main()
