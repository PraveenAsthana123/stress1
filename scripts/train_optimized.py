#!/usr/bin/env python3
"""
Optimized training with hyperparameter tuning for maximum accuracy.
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except:
    SMOTE_AVAILABLE = False

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "paper"


def extract_features(data, fs=256):
    """Extract comprehensive EEG features."""
    n_samples, n_channels, n_timepoints = data.shape

    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
             'beta': (13, 30), 'gamma': (30, 45)}

    all_features = []

    for i in range(n_samples):
        features = []

        for ch in range(n_channels):
            sig = data[i, ch]

            # Band powers
            freqs, psd = signal.welch(sig, fs=fs, nperseg=min(256, n_timepoints))
            total_power = np.sum(psd) + 1e-10

            for band, (low, high) in bands.items():
                idx = (freqs >= low) & (freqs <= high)
                bp = np.mean(psd[idx]) if np.any(idx) else 0
                rbp = np.sum(psd[idx]) / total_power
                features.extend([bp, rbp])

            # Key ratios
            alpha_idx = (freqs >= 8) & (freqs <= 13)
            beta_idx = (freqs >= 13) & (freqs <= 30)
            theta_idx = (freqs >= 4) & (freqs <= 8)

            alpha_power = np.sum(psd[alpha_idx]) + 1e-10
            beta_power = np.sum(psd[beta_idx]) + 1e-10
            theta_power = np.sum(psd[theta_idx]) + 1e-10

            features.extend([
                beta_power / alpha_power,  # Stress indicator
                theta_power / alpha_power,
                (alpha_power + theta_power) / beta_power
            ])

            # Statistical
            features.extend([
                np.mean(sig), np.std(sig), skew(sig), kurtosis(sig),
                np.sqrt(np.mean(sig**2)),  # RMS
                np.sum(np.abs(np.diff(sig))),  # Line length
            ])

        # Asymmetry features (frontal)
        if n_channels >= 4:
            for b_idx in range(5):
                left = features[b_idx * 2]
                right = features[(n_channels//2) * 10 + b_idx * 2]
                features.append((right - left) / (right + left + 1e-10))

        all_features.append(features)

    return np.nan_to_num(np.array(all_features), nan=0, posinf=0, neginf=0)


def train_optimized(X, y, dataset_name):
    """Train with optimized models."""
    print(f"\n{'='*60}")
    print(f"OPTIMIZED TRAINING: {dataset_name}")
    print(f"{'='*60}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Best model with tuned hyperparameters
    if XGB_AVAILABLE:
        model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            scale_pos_weight=len(y[y==0]) / (len(y[y==1]) + 1),
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )

    all_preds, all_proba, all_true = [], [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # SMOTE on training only
        if SMOTE_AVAILABLE and len(np.unique(y_train)) > 1:
            try:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            except:
                pass

        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        proba = model.predict_proba(X_val)[:, 1]

        all_preds.extend(preds)
        all_proba.extend(proba)
        all_true.extend(y_val)

        acc = accuracy_score(y_val, preds)
        print(f"  Fold {fold+1}: {acc*100:.1f}%")

    y_pred = np.array(all_preds)
    y_proba = np.array(all_proba)
    y_true = np.array(all_true)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results = {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "f1_score": round(f1_score(y_true, y_pred) * 100, 2),
        "auc_roc": round(roc_auc_score(y_true, y_proba) * 100, 2),
        "precision": round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0,
        "recall": round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0,
        "specificity": round(tn / (tn + fp) * 100, 2) if (tn + fp) > 0 else 0,
        "cohens_kappa": round(cohen_kappa_score(y_true, y_pred), 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "raw": cm.tolist()}
    }

    print(f"\nRESULTS: Acc={results['accuracy']}%, F1={results['f1_score']}%, AUC={results['auc_roc']}%")
    print(f"CM: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return results


def load_data():
    """Load both datasets."""
    datasets = {}

    # SAM-40
    sam40_path = DATA_DIR / "SAM40" / "filtered_data"
    if sam40_path.exists():
        import scipy.io as sio
        X_list, y_list = [], []
        for f in sorted(sam40_path.glob("*.mat")):
            try:
                data = sio.loadmat(f, squeeze_me=True)
                label = 0 if f.stem.startswith('Relax') else 1
                for k in data:
                    if not k.startswith('__'):
                        v = data[k]
                        if isinstance(v, np.ndarray) and len(v.shape) >= 2:
                            eeg = v.T if v.shape[0] > v.shape[1] else v
                            eeg_std = np.zeros((32, 512))
                            eeg_std[:min(eeg.shape[0], 32), :min(eeg.shape[1], 512)] = eeg[:min(eeg.shape[0], 32), :min(eeg.shape[1], 512)]
                            X_list.append(eeg_std)
                            y_list.append(label)
                            break
            except:
                pass
        if X_list:
            datasets['SAM-40'] = (np.array(X_list), np.array(y_list))
            print(f"SAM-40: {len(X_list)} samples")

    # EEGMAT
    eegmat_path = DATA_DIR / "EEGMAT" / "sample_100"
    if eegmat_path.exists():
        try:
            X_file = eegmat_path / "X_eegmat_100.npy"
            y_file = eegmat_path / "y_eegmat_100.npy"
            if not X_file.exists():
                X_file = eegmat_path / "X.npy"
                y_file = eegmat_path / "y.npy"
            X = np.load(X_file)
            y = np.load(y_file)
            if len(X.shape) == 2:
                X = X.reshape(X.shape[0], 32, -1)
            X_std = np.zeros((X.shape[0], 32, 512))
            for i in range(X.shape[0]):
                X_std[i, :min(X.shape[1], 32), :min(X.shape[2], 512)] = X[i, :min(X.shape[1], 32), :min(X.shape[2], 512)]
            datasets['EEGMAT'] = (X_std, y)
            print(f"EEGMAT: {len(X_std)} samples")
        except Exception as e:
            print(f"EEGMAT error: {e}")

    return datasets


def plot_results(results, save_path):
    """Plot confusion matrices."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]

    for idx, (name, data) in enumerate(results.items()):
        cm = np.array(data["confusion_matrix"]["raw"])
        cm_pct = cm / cm.sum() * 100

        ax = axes[idx]
        sns.heatmap(cm_pct, annot=False, cmap='Blues', ax=ax,
                   xticklabels=['Baseline', 'Stress'], yticklabels=['Baseline', 'Stress'], cbar=False)

        for i in range(2):
            for j in range(2):
                color = 'white' if cm_pct[i,j] > 30 else 'black'
                ax.text(j+0.5, i+0.35, f'{cm_pct[i,j]:.1f}%', ha='center', va='center', fontsize=14, fontweight='bold', color=color)
                ax.text(j+0.5, i+0.65, f'(n={cm[i,j]})', ha='center', va='center', fontsize=10, color=color)

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{name}\n(Acc: {data["accuracy"]}%, F1: {data["f1_score"]}%)', fontweight='bold')

    plt.suptitle('Real Confusion Matrices - XGBoost Classifier', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("="*60)
    print("OPTIMIZED TRAINING FOR MAXIMUM ACCURACY")
    print("="*60)

    datasets = load_data()
    results = {"metadata": {"generated_at": datetime.now().isoformat(), "is_real_data": True}, "datasets": {}}

    for name, (X, y) in datasets.items():
        print(f"\nExtracting features for {name}...")
        X_features = extract_features(X)
        print(f"Features: {X_features.shape}")
        results["datasets"][name] = train_optimized(X_features, y, name)

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(RESULTS_DIR / "optimized_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    plot_results(results["datasets"], OUTPUT_DIR / "fig11_confusion_matrices_OPTIMIZED.png")

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for name, data in results["datasets"].items():
        print(f"{name}: Acc={data['accuracy']}%, F1={data['f1_score']}%, AUC={data['auc_roc']}%")

    return results


if __name__ == "__main__":
    main()
