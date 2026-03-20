#!/usr/bin/env python3
"""
Full Dataset Training: EEGMAT + SAM-40 Ensemble Classifier
===========================================================
Binary stress classification using VotingClassifier (RF + GB + SVM)
EEGMAT: 99.31% | SAM-40: 72.92% | Combined: 95.83%

Authors: Praveen Asthana, Rajveer Singh Lalawat, Sarita Singh Gond
Version: 6.0.0
Python: 3.7+ (Windows/Linux/Mac compatible)

Usage:
    python scripts/train_full_data.py

Configuration:
    Edit config.py to change data paths, hyperparameters, etc.
"""

import sys
import os
import logging
import traceback
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Add project root to path so config.py can be imported from any directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Import project configuration (all paths and hyperparameters)
# Users: Edit config.py to change data paths for YOUR system
# ---------------------------------------------------------------------------
try:
    from config import (
        SAM40_DIR, EEGMAT_DIR, RESULTS_DIR, PAPER_DIR,
        RF_N_ESTIMATORS, RF_MAX_DEPTH,
        GB_N_ESTIMATORS, GB_MAX_DEPTH,
        SVM_KERNEL, SVM_C,
        N_FOLDS, RANDOM_SEED,
        SEGMENT_WINDOW, SEGMENT_OVERLAP,
        setup_directories
    )
except ImportError:
    # Fallback if config.py not found
    SAM40_DIR = PROJECT_ROOT / "data" / "SAM40" / "filtered_data"
    EEGMAT_DIR = PROJECT_ROOT / "data" / "EEGMAT" / "eeg-during-mental-arithmetic-tasks-1.0.0"
    RESULTS_DIR = PROJECT_ROOT / "results"
    PAPER_DIR = PROJECT_ROOT / "paper"
    RF_N_ESTIMATORS = 500
    RF_MAX_DEPTH = 15
    GB_N_ESTIMATORS = 300
    GB_MAX_DEPTH = 5
    SVM_KERNEL = "rbf"
    SVM_C = 10
    N_FOLDS = 5
    RANDOM_SEED = 42
    SEGMENT_WINDOW = 4
    SEGMENT_OVERLAP = 0.5
    def setup_directories():
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        PAPER_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ML imports
# ---------------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, cohen_kappa_score
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (for servers/CI)
import matplotlib.pyplot as plt
import seaborn as sns

# Optional imports with graceful fallback
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

try:
    from pyedflib import EdfReader
    PYEDF_AVAILABLE = True
except ImportError:
    PYEDF_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ===========================================================================
# DATA LOADING FUNCTIONS
# ===========================================================================

def load_edf_file(filepath):
    """
    Load an EDF (European Data Format) file containing EEG recordings.

    Attempts MNE-Python first (recommended), falls back to pyedflib.
    EDF is the standard format for the EEGMAT dataset from PhysioNet.

    Args:
        filepath (str or Path): Path to .edf file

    Returns:
        tuple: (data, sampling_rate) where data is (n_channels, n_samples)
               Returns (None, None) on failure
    """
    # Method 1: MNE-Python (preferred - handles annotations, events)
    if MNE_AVAILABLE:
        try:
            raw = mne.io.read_raw_edf(str(filepath), preload=True, verbose=False)
            data = raw.get_data()  # (n_channels, n_samples) in volts
            fs = raw.info['sfreq']  # Sampling frequency
            logger.debug(f"  MNE loaded: {filepath.name} ({data.shape}, {fs} Hz)")
            return data, fs
        except Exception as e:
            logger.debug(f"  MNE failed for {filepath.name}: {e}")

    # Method 2: pyedflib (fallback - lower-level, no preprocessing)
    if PYEDF_AVAILABLE:
        try:
            f = EdfReader(str(filepath))
            n_channels = f.signals_in_file
            data = []
            for i in range(n_channels):
                data.append(f.readSignal(i))
            f.close()
            logger.debug(f"  pyedflib loaded: {filepath.name} ({n_channels}ch)")
            return np.array(data), 500  # EEGMAT default: 500 Hz
        except Exception as e:
            logger.warning(f"  pyedflib failed for {filepath.name}: {e}")

    logger.error(f"Cannot load {filepath.name}: install mne or pyedflib")
    return None, None


def segment_data(data, fs, segment_length=4, overlap=0.5):
    """
    Segment continuous EEG into fixed-length overlapping windows.

    Windowing is essential because:
      - Raw EEG recordings are 60+ seconds long
      - ML models need fixed-size input
      - Overlapping windows increase training data
      - Short windows capture transient stress patterns

    Args:
        data (np.ndarray): EEG data (n_channels, n_samples)
        fs (float): Sampling frequency in Hz
        segment_length (int): Window length in seconds (default: 4)
        overlap (float): Overlap fraction 0-1 (default: 0.5 = 50%)

    Returns:
        list: List of np.ndarray segments, each (n_channels, samples_per_segment)
    """
    n_channels, n_samples = data.shape
    samples_per_segment = int(segment_length * fs)
    step = int(samples_per_segment * (1 - overlap))

    segments = []
    for start in range(0, n_samples - samples_per_segment, step):
        segment = data[:, start:start + samples_per_segment]
        segments.append(segment)

    logger.debug(f"  Segmented: {n_samples} samples -> {len(segments)} segments "
                 f"({segment_length}s, {overlap*100:.0f}% overlap)")
    return segments


def load_full_eegmat():
    """
    Load ALL EEGMAT EDF files (72 files: 36 subjects x 2 conditions).

    EEGMAT Dataset:
      - 36 subjects performing mental arithmetic tasks
      - 21 EEG channels, 500 Hz sampling rate
      - Condition 1: Baseline/relaxed (non-stress)
      - Condition 2: Mental arithmetic (stress)
      - File format: .edf (European Data Format)

    Reference: Zyma et al. (2019). PhysioNet: EEG During Mental Arithmetic Tasks.
               DOI: 10.13026/C2JQ1P

    Returns:
        tuple: (X, y, subjects) where X is (n_segments, 32, 512)
    """
    if not EEGMAT_DIR.exists():
        logger.error(f"EEGMAT directory not found: {EEGMAT_DIR}")
        logger.error("Download from: https://physionet.org/content/eegmat/1.0.0/")
        logger.error("Then update EEGMAT_DIR in config.py")
        return None, None, None

    logger.info(f"Loading full EEGMAT from {EEGMAT_DIR}")

    X_list, y_list, subjects = [], [], []
    files_loaded = 0

    # Subject00_1.edf = baseline, Subject00_2.edf = mental arithmetic
    for i in range(36):
        subj_id = f"Subject{i:02d}"

        for condition in [1, 2]:
            edf_file = EEGMAT_DIR / f"{subj_id}_{condition}.edf"

            if not edf_file.exists():
                logger.debug(f"  Missing: {edf_file.name}")
                continue

            data, fs = load_edf_file(edf_file)
            if data is None:
                continue

            files_loaded += 1

            # Label: condition 1 = baseline (0), condition 2 = stress (1)
            label = 0 if condition == 1 else 1

            # Segment into 4-second windows with 50% overlap
            segments = segment_data(data, fs,
                                    segment_length=SEGMENT_WINDOW,
                                    overlap=SEGMENT_OVERLAP)

            for seg in segments:
                # Standardize to 32 channels x 512 samples for model input
                n_ch, n_tp = seg.shape
                seg_std = np.zeros((32, 512))

                # Resample time dimension if needed (500 Hz -> 128 Hz equivalent)
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
        logger.info(f"  Loaded {files_loaded} EDF files -> {len(X)} segments")
        logger.info(f"  From {len(np.unique(subjects))} subjects")
        logger.info(f"  Labels: {sum(y == 0)} baseline, {sum(y == 1)} stress")
        return X, y, subjects

    logger.error("No EEGMAT data could be loaded.")
    return None, None, None


def load_sam40():
    """
    Load SAM-40 dataset for combined training.

    SAM-40 files are .mat format with 'Clean_data' key containing
    preprocessed EEG data (32 channels x 3200 samples at 128 Hz).

    Returns:
        tuple: (X, y, subjects) where X is (n_samples, 32, 512)
    """
    if not SAM40_DIR.exists():
        logger.warning(f"SAM-40 directory not found: {SAM40_DIR}")
        return None, None, None

    try:
        import scipy.io as sio
    except ImportError:
        logger.error("scipy not installed. Required for .mat file loading.")
        return None, None, None

    logger.info(f"Loading SAM-40 from {SAM40_DIR}")

    X_list, y_list, subjects = [], [], []

    for f in sorted(SAM40_DIR.glob("*.mat")):
        try:
            data = sio.loadmat(str(f), squeeze_me=True)

            # Binary label from filename
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

                        # Standardize to 32ch x 512 samples
                        eeg_std = np.zeros((32, 512))
                        ch = min(eeg.shape[0], 32)
                        tp = min(eeg.shape[1], 512)
                        eeg_std[:ch, :tp] = eeg[:ch, :tp]

                        X_list.append(eeg_std)
                        y_list.append(label)
                        subjects.append(subj_id + 100)  # Offset to avoid EEGMAT collision
                        break

        except Exception as e:
            logger.warning(f"Error loading {f.name}: {e}")
            continue

    if X_list:
        X = np.array(X_list)
        y = np.array(y_list)
        subjects = np.array(subjects)
        logger.info(f"  Loaded {len(X)} samples from {len(np.unique(subjects))} subjects")
        logger.info(f"  Labels: {sum(y == 0)} baseline, {sum(y == 1)} stress")
        return X, y, subjects

    return None, None, None


# ===========================================================================
# FEATURE EXTRACTION
# ===========================================================================

def extract_features(data, fs=256):
    """
    Extract features from segmented EEG data for ensemble classification.

    Features per channel (16 per channel):
      - Band powers (5 bands x 2): log power + relative power
      - Statistical (6): mean, std, skewness, kurtosis, RMS, line length

    Global features (3):
      - Beta/Alpha ratio (stress indicator)
      - Theta/Alpha ratio
      - Relaxation index: (alpha + theta) / beta

    Args:
        data (np.ndarray): EEG segments (n_samples, n_channels, n_timepoints)
        fs (int): Effective sampling frequency after resampling

    Returns:
        np.ndarray: Feature matrix (n_samples, n_features)
    """
    n_samples, n_channels, n_timepoints = data.shape

    # Standard EEG frequency bands
    bands = {
        'delta': (0.5, 4),    # Deep sleep
        'theta': (4, 8),      # Drowsiness, meditation
        'alpha': (8, 13),     # Relaxation (SUPPRESSED in stress)
        'beta':  (13, 30),    # Active thinking (ENHANCED in stress)
        'gamma': (30, 45)     # High cognition
    }

    all_features = []

    for i in range(n_samples):
        features = []
        band_powers = {b: [] for b in bands}

        for ch in range(n_channels):
            sig = data[i, ch]

            # PSD via Welch's method
            freqs, psd = signal.welch(sig, fs=fs, nperseg=min(256, n_timepoints))
            total = np.sum(psd) + 1e-10

            # Band power features
            for band, (low, high) in bands.items():
                idx = (freqs >= low) & (freqs <= high)
                bp = np.mean(psd[idx]) if np.any(idx) else 0
                rbp = np.sum(psd[idx]) / total
                features.extend([np.log1p(bp), rbp])
                band_powers[band].append(np.sum(psd[idx]))

            # Statistical features
            features.extend([
                np.mean(sig),                        # Mean amplitude
                np.std(sig),                         # Standard deviation
                skew(sig),                           # Skewness
                kurtosis(sig),                       # Kurtosis
                np.sqrt(np.mean(sig**2)),            # RMS
                np.sum(np.abs(np.diff(sig)))         # Line length
            ])

        # Global stress biomarker ratios
        alpha = np.mean(band_powers['alpha']) + 1e-10
        beta = np.mean(band_powers['beta']) + 1e-10
        theta = np.mean(band_powers['theta']) + 1e-10

        features.extend([
            beta / alpha,                   # Beta/Alpha ratio (stress indicator)
            theta / alpha,                  # Theta/Alpha ratio
            (alpha + theta) / beta          # Relaxation index
        ])

        all_features.append(features)

        if (i + 1) % 500 == 0:
            logger.info(f"  Extracted features: {i + 1}/{n_samples}")

    return np.nan_to_num(np.array(all_features), nan=0, posinf=0, neginf=0)


# ===========================================================================
# MODEL TRAINING
# ===========================================================================

def train_model(X, y, dataset_name):
    """
    Train ensemble classifier with 5-fold stratified cross-validation.

    Ensemble: VotingClassifier (soft voting)
      - RandomForest (500 trees, depth 15)
      - GradientBoosting (300 trees, depth 5)
      - SVM (RBF kernel, C=10)

    With SMOTE oversampling per fold to handle class imbalance.

    Args:
        X (np.ndarray): Feature matrix (n_samples, n_features)
        y (np.ndarray): Labels (0=baseline, 1=stress)
        dataset_name (str): Name for logging

    Returns:
        dict: Results with accuracy, F1, AUC-ROC, confusion matrix, etc.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"TRAINING: {dataset_name}")
    logger.info(f"{'=' * 60}")
    logger.info(f"Samples: {len(X)}, Features: {X.shape[1]}")
    logger.info(f"Class distribution: {np.bincount(y)}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # Ensemble model: RF + GB + SVM with soft voting
    model = VotingClassifier([
        ('rf', RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=GB_N_ESTIMATORS, max_depth=GB_MAX_DEPTH,
            random_state=RANDOM_SEED
        )),
        ('svm', SVC(
            kernel=SVM_KERNEL, C=SVM_C,
            class_weight='balanced', probability=True, random_state=RANDOM_SEED
        )),
    ], voting='soft', n_jobs=-1)

    all_preds, all_proba, all_true = [], [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # SMOTE oversampling (per fold to prevent data leakage)
        if SMOTE_AVAILABLE:
            try:
                smote = SMOTE(random_state=RANDOM_SEED)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            except Exception as e:
                logger.debug(f"  SMOTE skipped for fold {fold + 1}: {e}")

        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        proba = model.predict_proba(X_val)[:, 1]

        all_preds.extend(preds)
        all_proba.extend(proba)
        all_true.extend(y_val)

        acc = accuracy_score(y_val, preds)
        logger.info(f"  Fold {fold + 1}: {acc * 100:.1f}%")

    # Aggregate results
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
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp),
            "raw": cm.tolist()
        }
    }

    logger.info(f"\nRESULTS: Acc={results['accuracy']}%, "
                f"F1={results['f1_score']}%, AUC={results['auc_roc']}%")

    return results


# ===========================================================================
# VISUALIZATION
# ===========================================================================

def plot_results(results, save_path):
    """Plot confusion matrices for all datasets."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for idx, (name, data) in enumerate(results.items()):
        cm = np.array(data["confusion_matrix"]["raw"])
        cm_pct = cm / cm.sum() * 100

        ax = axes[idx]
        sns.heatmap(cm_pct, annot=False, cmap='Blues', ax=ax,
                    xticklabels=['Baseline', 'Stress'],
                    yticklabels=['Baseline', 'Stress'],
                    cbar=False, vmin=0, vmax=60)

        for i in range(2):
            for j in range(2):
                color = 'white' if cm_pct[i, j] > 30 else 'black'
                ax.text(j + 0.5, i + 0.35, f'{cm_pct[i, j]:.1f}%',
                        ha='center', va='center', fontsize=14,
                        fontweight='bold', color=color)
                ax.text(j + 0.5, i + 0.65, f'(n={cm[i, j]})',
                        ha='center', va='center', fontsize=10, color=color)

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{name}\n(Acc: {data["accuracy"]}%)', fontweight='bold')

    plt.suptitle('Confusion Matrices - Full Dataset Training',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def main():
    """
    Main training pipeline for full dataset evaluation.

    Pipeline:
        1. Load EEGMAT (72 EDF files -> segments)
        2. Load SAM-40 (480 .mat files -> segments)
        3. Extract features for each dataset
        4. Train ensemble with 5-fold CV + SMOTE
        5. Optional: combine datasets and retrain
        6. Save results and confusion matrix plots
    """
    start_time = datetime.now()

    logger.info("=" * 60)
    logger.info("TRAINING WITH FULL DATASET")
    logger.info(f"Ensemble: RF({RF_N_ESTIMATORS}) + GB({GB_N_ESTIMATORS}) + SVM({SVM_KERNEL}, C={SVM_C})")
    logger.info(f"CV: {N_FOLDS}-fold stratified, seed={RANDOM_SEED}")
    logger.info("=" * 60)

    setup_directories()

    results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "is_real_data": True,
            "python_version": sys.version,
            "config": {
                "rf": f"n={RF_N_ESTIMATORS}, depth={RF_MAX_DEPTH}",
                "gb": f"n={GB_N_ESTIMATORS}, depth={GB_MAX_DEPTH}",
                "svm": f"kernel={SVM_KERNEL}, C={SVM_C}",
                "cv_folds": N_FOLDS,
                "segment": f"{SEGMENT_WINDOW}s, {SEGMENT_OVERLAP*100:.0f}% overlap"
            }
        },
        "datasets": {}
    }

    # --- EEGMAT ---
    X_eegmat, y_eegmat, _ = load_full_eegmat()
    if X_eegmat is not None:
        logger.info(f"\nExtracting features for EEGMAT ({len(X_eegmat)} segments)...")
        X_feat = extract_features(X_eegmat)
        results["datasets"]["EEGMAT-Full"] = train_model(X_feat, y_eegmat, "EEGMAT-Full")

    # --- SAM-40 ---
    X_sam40, y_sam40, _ = load_sam40()
    if X_sam40 is not None:
        logger.info(f"\nExtracting features for SAM-40 ({len(X_sam40)} samples)...")
        X_feat = extract_features(X_sam40)
        results["datasets"]["SAM-40"] = train_model(X_feat, y_sam40, "SAM-40")

    # --- Combined ---
    if X_eegmat is not None and X_sam40 is not None:
        logger.info("\nCombining datasets...")
        X_combined = np.vstack([X_eegmat, X_sam40])
        y_combined = np.concatenate([y_eegmat, y_sam40])
        logger.info(f"Combined: {len(X_combined)} samples")
        X_feat = extract_features(X_combined)
        results["datasets"]["Combined"] = train_model(X_feat, y_combined, "Combined")

    # Save results
    with open(RESULTS_DIR / "full_data_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Plot confusion matrices
    if results["datasets"]:
        plot_results(results["datasets"],
                     PAPER_DIR / "fig11_confusion_matrices_FULL.png")

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"\n{'=' * 60}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'=' * 60}")
    for name, data in results["datasets"].items():
        logger.info(f"  {name}: Acc={data['accuracy']}%, F1={data['f1_score']}%")

    if results["datasets"]:
        max_acc = max(d['accuracy'] for d in results["datasets"].values())
        logger.info(f"\n  Best Accuracy: {max_acc}%")

    logger.info(f"  Total time: {elapsed:.1f} seconds")
    logger.info(f"  Results saved: {RESULTS_DIR / 'full_data_results.json'}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
