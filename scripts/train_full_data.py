#!/usr/bin/env python3
"""
Train with FULL EEGMAT dataset (72 EDF files) + SAM-40 for maximum accuracy.
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("MNE not available. Install with: pip install mne")

try:
    from pyedflib import EdfReader
    PYEDF_AVAILABLE = True
except ImportError:
    PYEDF_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except:
    SMOTE_AVAILABLE = False

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "paper"


def load_edf_file(filepath):
    """Load EDF file and return data."""
    if MNE_AVAILABLE:
        try:
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            data = raw.get_data()
            fs = raw.info['sfreq']
            return data, fs
        except Exception as e:
            pass

    # Fallback: try pyedflib
    if PYEDF_AVAILABLE:
        try:
            f = EdfReader(str(filepath))
            n_channels = f.signals_in_file
            data = []
            for i in range(n_channels):
                data.append(f.readSignal(i))
            f.close()
            return np.array(data), 500  # Assume 500 Hz
        except:
            pass

    return None, None


def segment_data(data, fs, segment_length=4, overlap=0.5):
    """Segment continuous EEG into fixed-length windows."""
    n_channels, n_samples = data.shape
    samples_per_segment = int(segment_length * fs)
    step = int(samples_per_segment * (1 - overlap))

    segments = []
    for start in range(0, n_samples - samples_per_segment, step):
        segment = data[:, start:start + samples_per_segment]
        segments.append(segment)

    return segments


def load_full_eegmat():
    """Load ALL EEGMAT EDF files."""
    edf_dir = DATA_DIR / "EEGMAT" / "eeg-during-mental-arithmetic-tasks-1.0.0"

    if not edf_dir.exists():
        print(f"EDF directory not found: {edf_dir}")
        return None, None, None

    print(f"Loading full EEGMAT from {edf_dir}")

    X_list = []
    y_list = []
    subjects = []

    # Subject00_1.edf = baseline (relaxed), Subject00_2.edf = mental arithmetic (stress)
    for i in range(36):  # 36 subjects
        subj_id = f"Subject{i:02d}"

        for condition in [1, 2]:
            edf_file = edf_dir / f"{subj_id}_{condition}.edf"

            if not edf_file.exists():
                continue

            data, fs = load_edf_file(edf_file)

            if data is None:
                continue

            # Label: 1=baseline (condition 1), 2=mental arithmetic/stress (condition 2)
            label = 0 if condition == 1 else 1

            # Segment into 4-second windows
            segments = segment_data(data, fs, segment_length=4, overlap=0.5)

            for seg in segments:
                # Standardize to 32 channels x 512 samples
                n_ch, n_tp = seg.shape
                seg_std = np.zeros((32, 512))

                # Resample if needed
                if n_tp != 512:
                    for ch in range(min(n_ch, 32)):
                        seg_std[ch] = signal.resample(seg[ch], 512)
                else:
                    seg_std[:min(n_ch, 32), :] = seg[:min(n_ch, 32), :]

                X_list.append(seg_std)
                y_list.append(label)
                subjects.append(i)

    if X_list:
        X = np.array(X_list)
        y = np.array(y_list)
        subjects = np.array(subjects)
        print(f"  Loaded {len(X)} segments from {len(np.unique(subjects))} subjects")
        print(f"  Labels: {sum(y==0)} baseline, {sum(y==1)} stress")
        return X, y, subjects

    return None, None, None


def load_sam40():
    """Load SAM-40 dataset."""
    sam40_path = DATA_DIR / "SAM40" / "filtered_data"
    if not sam40_path.exists():
        return None, None, None

    try:
        import scipy.io as sio
    except:
        return None, None, None

    print(f"Loading SAM-40 from {sam40_path}")

    X_list, y_list, subjects = [], [], []

    for f in sorted(sam40_path.glob("*.mat")):
        try:
            data = sio.loadmat(f, squeeze_me=True)
            label = 0 if f.stem.startswith('Relax') else 1

            # Extract subject ID
            parts = f.stem.split('_')
            subj_idx = parts.index('sub') + 1 if 'sub' in parts else 0
            subj_id = int(parts[subj_idx]) if subj_idx > 0 else 0

            for k in data:
                if not k.startswith('__'):
                    v = data[k]
                    if isinstance(v, np.ndarray) and len(v.shape) >= 2:
                        eeg = v.T if v.shape[0] > v.shape[1] else v
                        eeg_std = np.zeros((32, 512))
                        ch, tp = min(eeg.shape[0], 32), min(eeg.shape[1], 512)
                        eeg_std[:ch, :tp] = eeg[:ch, :tp]
                        X_list.append(eeg_std)
                        y_list.append(label)
                        subjects.append(subj_id + 100)  # Offset to avoid collision
                        break
        except:
            pass

    if X_list:
        X = np.array(X_list)
        y = np.array(y_list)
        subjects = np.array(subjects)
        print(f"  Loaded {len(X)} samples from {len(np.unique(subjects))} subjects")
        print(f"  Labels: {sum(y==0)} baseline, {sum(y==1)} stress")
        return X, y, subjects

    return None, None, None


def extract_features(data, fs=256):
    """Extract features for classification."""
    n_samples, n_channels, n_timepoints = data.shape

    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
             'beta': (13, 30), 'gamma': (30, 45)}

    all_features = []

    for i in range(n_samples):
        features = []
        band_powers = {b: [] for b in bands}

        for ch in range(n_channels):
            sig = data[i, ch]

            # PSD
            freqs, psd = signal.welch(sig, fs=fs, nperseg=min(256, n_timepoints))
            total = np.sum(psd) + 1e-10

            for band, (low, high) in bands.items():
                idx = (freqs >= low) & (freqs <= high)
                bp = np.mean(psd[idx]) if np.any(idx) else 0
                rbp = np.sum(psd[idx]) / total
                features.extend([np.log1p(bp), rbp])
                band_powers[band].append(np.sum(psd[idx]))

            # Stats
            features.extend([
                np.mean(sig), np.std(sig), skew(sig), kurtosis(sig),
                np.sqrt(np.mean(sig**2)), np.sum(np.abs(np.diff(sig)))
            ])

        # Global ratios
        alpha = np.mean(band_powers['alpha']) + 1e-10
        beta = np.mean(band_powers['beta']) + 1e-10
        theta = np.mean(band_powers['theta']) + 1e-10

        features.extend([beta/alpha, theta/alpha, (alpha+theta)/beta])

        all_features.append(features)

    return np.nan_to_num(np.array(all_features), nan=0, posinf=0, neginf=0)


def train_model(X, y, dataset_name):
    """Train and evaluate model."""
    print(f"\n{'='*60}")
    print(f"TRAINING: {dataset_name}")
    print(f"{'='*60}")
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"Classes: {np.bincount(y)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Strong ensemble
    model = VotingClassifier([
        ('rf', RandomForestClassifier(n_estimators=500, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=300, max_depth=5, random_state=42)),
        ('svm', SVC(kernel='rbf', C=10, class_weight='balanced', probability=True, random_state=42)),
    ], voting='soft', n_jobs=-1)

    all_preds, all_proba, all_true = [], [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if SMOTE_AVAILABLE:
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

    return results


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
                   xticklabels=['Baseline', 'Stress'], yticklabels=['Baseline', 'Stress'],
                   cbar=False, vmin=0, vmax=60)

        for i in range(2):
            for j in range(2):
                color = 'white' if cm_pct[i,j] > 30 else 'black'
                ax.text(j+0.5, i+0.35, f'{cm_pct[i,j]:.1f}%', ha='center', va='center',
                       fontsize=14, fontweight='bold', color=color)
                ax.text(j+0.5, i+0.65, f'(n={cm[i,j]})', ha='center', va='center',
                       fontsize=10, color=color)

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{name}\n(Acc: {data["accuracy"]}%)', fontweight='bold')

    plt.suptitle('Confusion Matrices - Full Dataset Training', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("="*60)
    print("TRAINING WITH FULL DATASET")
    print("="*60)

    results = {"metadata": {"generated_at": datetime.now().isoformat(), "is_real_data": True}, "datasets": {}}

    # Load EEGMAT full
    X_eegmat, y_eegmat, _ = load_full_eegmat()
    if X_eegmat is not None:
        print(f"\nExtracting features for EEGMAT...")
        X_feat = extract_features(X_eegmat)
        results["datasets"]["EEGMAT-Full"] = train_model(X_feat, y_eegmat, "EEGMAT-Full")

    # Load SAM-40
    X_sam40, y_sam40, _ = load_sam40()
    if X_sam40 is not None:
        print(f"\nExtracting features for SAM-40...")
        X_feat = extract_features(X_sam40)
        results["datasets"]["SAM-40"] = train_model(X_feat, y_sam40, "SAM-40")

    # Combined
    if X_eegmat is not None and X_sam40 is not None:
        print("\nCombining datasets...")
        X_combined = np.vstack([X_eegmat, X_sam40])
        y_combined = np.concatenate([y_eegmat, y_sam40])
        print(f"Combined: {len(X_combined)} samples")
        X_feat = extract_features(X_combined)
        results["datasets"]["Combined"] = train_model(X_feat, y_combined, "Combined")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(RESULTS_DIR / "full_data_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    if results["datasets"]:
        plot_results(results["datasets"], OUTPUT_DIR / "fig11_confusion_matrices_FULL.png")

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for name, data in results["datasets"].items():
        print(f"{name}: Acc={data['accuracy']}%, F1={data['f1_score']}%")

    max_acc = max(d['accuracy'] for d in results["datasets"].values())
    print(f"\n*** Best Accuracy: {max_acc}% ***")

    return results


if __name__ == "__main__":
    main()
