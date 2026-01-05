#!/usr/bin/env python3
"""
SAM-40 Binary Classification: Relax vs Stress (Arithmetic+Mirror+Stroop)
Target: 90%+ accuracy with proper class balancing
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
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
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except:
    IMBLEARN_AVAILABLE = False

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def extract_features(data, fs=128):
    """Extract comprehensive EEG features."""
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
    n_samples, n_channels, n_timepoints = data.shape
    all_features = []

    for i in range(n_samples):
        feat = []
        for ch in range(n_channels):
            sig = data[i, ch]

            # Band powers
            nperseg = min(256, len(sig))
            freqs, psd = signal.welch(sig, fs=fs, nperseg=nperseg)
            total_power = np.sum(psd) + 1e-10

            bp = {}
            for name, (low, high) in bands.items():
                idx = np.logical_and(freqs >= low, freqs <= high)
                bp[name] = np.sum(psd[idx])
                feat.append(bp[name])
                feat.append(bp[name] / total_power)

            # Ratios
            feat.extend([
                bp['beta'] / (bp['alpha'] + 1e-10),
                bp['theta'] / (bp['beta'] + 1e-10),
                bp['gamma'] / (bp['beta'] + 1e-10),
                (bp['theta'] + bp['alpha']) / (bp['beta'] + bp['gamma'] + 1e-10),
            ])

            # Statistics
            feat.extend([
                np.mean(sig), np.std(sig), skew(sig), kurtosis(sig),
                np.max(sig) - np.min(sig), np.sqrt(np.mean(sig**2)),
            ])

            # Hjorth
            d1 = np.diff(sig)
            d2 = np.diff(d1)
            v0, v1, v2 = np.var(sig)+1e-10, np.var(d1)+1e-10, np.var(d2)+1e-10
            mob = np.sqrt(v1/v0)
            comp = np.sqrt(v2/v1) / (mob + 1e-10)
            feat.extend([mob, comp])

        all_features.append(feat)

    return np.nan_to_num(np.array(all_features), nan=0.0, posinf=0.0, neginf=0.0)


def load_sam40_binary():
    """Load SAM-40 as binary: Relax=0, Stress=1."""
    sam40_path = DATA_DIR / "SAM40" / "filtered_data"
    if not sam40_path.exists():
        return None, None

    print(f"Loading SAM-40 from {sam40_path}")
    X_list, y_list = [], []

    for mat_file in sorted(sam40_path.glob("*.mat")):
        try:
            data = sio.loadmat(mat_file, squeeze_me=True)
            filename = mat_file.stem

            # Binary: Relax=0, else=1
            label = 0 if filename.startswith('Relax') else 1

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
                        break
        except:
            continue

    if X_list:
        X, y = np.array(X_list), np.array(y_list)
        print(f"  Loaded: {len(X)} samples, Relax={sum(y==0)}, Stress={sum(y==1)}")
        return X, y
    return None, None


def train_binary():
    """Train with aggressive undersampling for balance."""
    X, y = load_sam40_binary()
    if X is None:
        return None

    print("\nExtracting features...")
    X_feat = extract_features(X)
    print(f"  Features: {X_feat.shape[1]}")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)

    # Feature selection
    n_feat = min(300, X_scaled.shape[1])
    selector = SelectKBest(mutual_info_classif, k=n_feat)
    X_sel = selector.fit_transform(X_scaled, y)

    print(f"\n5-Fold CV Training...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_true, y_pred, y_proba = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_sel, y)):
        X_tr, X_val = X_sel[tr_idx], X_sel[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # Balance: undersample majority class
        idx_0 = np.where(y_tr == 0)[0]
        idx_1 = np.where(y_tr == 1)[0]

        # Undersample stress to match relax
        n_samples = min(len(idx_0), len(idx_1))
        np.random.seed(42 + fold)
        idx_1_down = np.random.choice(idx_1, n_samples, replace=False)
        idx_balanced = np.concatenate([idx_0, idx_1_down])
        np.random.shuffle(idx_balanced)

        X_tr_bal = X_tr[idx_balanced]
        y_tr_bal = y_tr[idx_balanced]

        # Also try SMOTE on balanced data for more diversity
        if IMBLEARN_AVAILABLE:
            try:
                smote = SMOTE(random_state=42, k_neighbors=3)
                X_tr_bal, y_tr_bal = smote.fit_resample(X_tr_bal, y_tr_bal)
            except:
                pass

        # Ensemble
        rf = RandomForestClassifier(n_estimators=300, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
        et = ExtraTreesClassifier(n_estimators=300, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)

        if XGB_AVAILABLE:
            xgb = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
            model = VotingClassifier([('rf', rf), ('et', et), ('xgb', xgb)], voting='soft', n_jobs=-1)
        else:
            model = VotingClassifier([('rf', rf), ('et', et)], voting='soft', n_jobs=-1)

        model.fit(X_tr_bal, y_tr_bal)

        pred = model.predict(X_val)
        prob = model.predict_proba(X_val)[:, 1]

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
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }

    print(f"\n{'='*50}")
    print("SAM-40 BINARY RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:    {results['accuracy']}%")
    print(f"F1 Score:    {results['f1_score']}%")
    print(f"Precision:   {results['precision']}%")
    print(f"Recall:      {results['recall']}%")
    print(f"Specificity: {results['specificity']}%")
    print(f"AUC-ROC:     {results['auc_roc']}%")
    print(f"Kappa:       {results['cohens_kappa']}")
    print(f"CM: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return results


def main():
    print("="*50)
    print("SAM-40 BINARY CLASSIFICATION")
    print("="*50)
    print(f"Started: {datetime.now()}")

    results = train_binary()

    if results:
        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "model": "Voting Ensemble (RF+ET+XGB)",
                "task": "Binary (Relax vs Stress)",
                "validation": "5-fold Stratified CV",
                "is_real_data": True
            },
            "SAM-40": results
        }

        RESULTS_DIR.mkdir(exist_ok=True)
        with open(RESULTS_DIR / "sam40_binary_results.json", 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved: {RESULTS_DIR / 'sam40_binary_results.json'}")

        if results['accuracy'] >= 90:
            print("\n*** TARGET 90% ACHIEVED! ***")

    return results


if __name__ == "__main__":
    main()
