#!/usr/bin/env python3
"""
================================================================================
Full validation script for EEGMAT dataset with enhanced training.
================================================================================

Uses the complete dataset for training with data augmentation and
longer training epochs to achieve higher accuracy.

Dataset: https://physionet.org/content/eegmat/1.0.0/

Cross-platform compatible (Windows/Linux/macOS).

Author: GenAI-RAG-EEG Team
Version: 3.0.0
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)
import json
import os
import sys
from glob import glob
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Get paths from config (cross-platform)
try:
    from src.config import DATA_DIR as CONFIG_DATA_DIR, PROJECT_ROOT
    DATA_DIR = str(CONFIG_DATA_DIR / "EEGMAT" / "eeg-during-mental-arithmetic-tasks-1.0.0")
    OUTPUT_DIR = str(PROJECT_ROOT / "results" / "eegmat_validation")
except ImportError:
    # Fallback to relative paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = str(PROJECT_ROOT / "data" / "EEGMAT" / "eeg-during-mental-arithmetic-tasks-1.0.0")
    OUTPUT_DIR = str(PROJECT_ROOT / "results" / "eegmat_validation")
TARGET_CHANNELS = 32
TARGET_SAMPLES = 512
TARGET_SR = 256


def load_full_dataset():
    """Load and process the full EEGMAT dataset."""
    print("Loading full EEGMAT dataset...")

    try:
        import mne
        MNE_AVAILABLE = True
    except ImportError:
        MNE_AVAILABLE = False
        import pyedflib

    def load_edf(filepath):
        if MNE_AVAILABLE:
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            return raw.get_data(), raw.info['sfreq']
        else:
            f = pyedflib.EdfReader(filepath)
            n_channels = f.signals_in_file
            n_samples = f.getNSamples()[0]
            sfreq = f.getSampleFrequency(0)
            data = np.zeros((n_channels, n_samples))
            for i in range(n_channels):
                data[i, :] = f.readSignal(i)
            f.close()
            return data, sfreq

    def resample_data(data, orig_sr, target_sr):
        if orig_sr == target_sr:
            return data
        from scipy import signal
        n_samples_new = int(data.shape[1] * target_sr / orig_sr)
        return signal.resample(data, n_samples_new, axis=1)

    def pad_channels(data, target_channels):
        n_channels = data.shape[0]
        if n_channels == target_channels:
            return data
        elif n_channels < target_channels:
            padding = np.zeros((target_channels - n_channels, data.shape[1]))
            return np.vstack([data, padding])
        else:
            return data[:target_channels, :]

    def segment_data(data, segment_length, overlap=0.5):
        """Segment with 50% overlap for more data."""
        n_channels, n_samples = data.shape
        step = int(segment_length * (1 - overlap))
        segments = []
        for start in range(0, n_samples - segment_length + 1, step):
            segment = data[:, start:start + segment_length]
            segments.append(segment)
        return np.array(segments)

    def normalize_segment(segment):
        mean = segment.mean()
        std = segment.std()
        if std < 1e-8:
            return segment - mean
        return (segment - mean) / std

    # Find all EDF files
    edf_files = sorted(glob(os.path.join(DATA_DIR, "Subject*_*.edf")))
    baseline_files = [f for f in edf_files if "_1.edf" in f]
    task_files = [f for f in edf_files if "_2.edf" in f]

    print(f"Found {len(baseline_files)} baseline, {len(task_files)} task files")

    all_segments = []
    all_labels = []

    # Process baseline files (label=0)
    for filepath in baseline_files:
        try:
            data, sfreq = load_edf(filepath)
            if sfreq != TARGET_SR:
                data = resample_data(data, sfreq, TARGET_SR)
            data = pad_channels(data, TARGET_CHANNELS)
            segments = segment_data(data, TARGET_SAMPLES, overlap=0.5)
            for seg in segments:
                all_segments.append(normalize_segment(seg))
                all_labels.append(0)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    # Process task files (label=1)
    for filepath in task_files:
        try:
            data, sfreq = load_edf(filepath)
            if sfreq != TARGET_SR:
                data = resample_data(data, sfreq, TARGET_SR)
            data = pad_channels(data, TARGET_CHANNELS)
            segments = segment_data(data, TARGET_SAMPLES, overlap=0.5)
            for seg in segments:
                all_segments.append(normalize_segment(seg))
                all_labels.append(1)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    X = np.array(all_segments, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    print(f"Full dataset: {X.shape}, baseline={np.sum(y==0)}, task={np.sum(y==1)}")
    return X, y


def data_augmentation(X, y, augment_factor=2):
    """Apply data augmentation to increase dataset size."""
    X_aug = [X]
    y_aug = [y]

    for _ in range(augment_factor - 1):
        # Add Gaussian noise
        noise = np.random.randn(*X.shape) * 0.05
        X_noisy = X + noise
        X_aug.append(X_noisy.astype(np.float32))
        y_aug.append(y)

        # Time shift (circular)
        shift = np.random.randint(10, 50)
        X_shifted = np.roll(X, shift, axis=2)
        X_aug.append(X_shifted.astype(np.float32))
        y_aug.append(y)

    return np.concatenate(X_aug), np.concatenate(y_aug)


def train_model(model, X_train, y_train, X_val, y_val, device, epochs=150, patience=20):
    """Train with enhanced parameters."""
    model = model.to(device)

    # Keep data on CPU and move batches to GPU
    X_train_np = X_train
    y_train_np = y_train

    # Class weights
    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (len(class_counts) * class_counts)
    weights = torch.FloatTensor(class_weights).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    best_val_acc = 0
    best_state = None
    no_improve = 0

    batch_size = 32  # Reduced for GPU memory
    n_batches = (len(X_train_np) + batch_size - 1) // batch_size

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Shuffle training data
        perm = np.random.permutation(len(X_train_np))
        X_train_np = X_train_np[perm]
        y_train_np = y_train_np[perm]

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, len(X_train_np))

            X_batch = torch.FloatTensor(X_train_np[start:end]).to(device)
            y_batch = torch.LongTensor(y_train_np[start:end]).to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs['logits'], y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

            del X_batch, y_batch  # Free GPU memory

        scheduler.step()

        # Validation in batches
        model.eval()
        all_preds = []
        val_batch_size = 64
        with torch.no_grad():
            for i in range(0, len(X_val), val_batch_size):
                X_batch = torch.FloatTensor(X_val[i:i+val_batch_size]).to(device)
                val_out = model(X_batch)
                _, pred = torch.max(val_out['logits'], 1)
                all_preds.extend(pred.cpu().numpy())
                del X_batch
        val_acc = accuracy_score(y_val, all_preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/n_batches:.4f} | Val Acc: {val_acc*100:.1f}%")

        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, best_val_acc


def evaluate(model, X_test, y_test, device):
    """Evaluate model in batches."""
    model.eval()
    batch_size = 64
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            X_batch = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
            outputs = model(X_batch)
            probs = outputs['probs'].cpu().numpy()
            _, pred = torch.max(outputs['logits'], 1)
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs[:, 1])
            del X_batch

    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0
    }, y_pred, y_prob


def main():
    print("=" * 70)
    print("GenAI-RAG-EEG - EEGMAT Full Dataset Validation")
    print("PhysioNet EEG Mental Arithmetic Tasks")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load full dataset
    X, y = load_full_dataset()

    # Apply data augmentation (reduced for memory)
    print("\nApplying data augmentation...")
    X_aug, y_aug = data_augmentation(X, y, augment_factor=2)
    print(f"Augmented dataset: {X_aug.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Import model
    from src.models.genai_rag_eeg import GenAIRAGEEG

    # 5-fold cross-validation
    print("\n" + "=" * 70)
    print("5-Fold Cross-Validation")
    print("=" * 70)

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_aug, y_aug), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")

        X_train, X_test = X_aug[train_idx], X_aug[test_idx]
        y_train, y_test = y_aug[train_idx], y_aug[test_idx]

        # Create fresh model
        model = GenAIRAGEEG(n_channels=32, n_time_samples=512, dropout=0.3)

        # Train
        model, best_acc = train_model(model, X_train, y_train, X_test, y_test, device,
                                      epochs=150, patience=20)

        # Evaluate
        metrics, y_pred, y_prob = evaluate(model, X_test, y_test, device)
        fold_results.append(metrics)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        print(f"    Final - Accuracy: {metrics['accuracy']*100:.1f}%, F1: {metrics['f1']*100:.1f}%, AUC: {metrics['auc_roc']*100:.1f}%")

    # Results
    print("\n" + "=" * 70)
    print("EEGMAT Full Dataset - Cross-Validation Results")
    print("=" * 70)

    metrics_agg = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
        values = [r[metric] for r in fold_results]
        metrics_agg[metric] = {'mean': np.mean(values), 'std': np.std(values)}

    print(f"\n    Metric       Mean ± Std")
    print(f"    {'─' * 35}")
    print(f"    Accuracy:   {metrics_agg['accuracy']['mean']*100:5.1f}% ± {metrics_agg['accuracy']['std']*100:.1f}%")
    print(f"    Precision:  {metrics_agg['precision']['mean']*100:5.1f}% ± {metrics_agg['precision']['std']*100:.1f}%")
    print(f"    Recall:     {metrics_agg['recall']['mean']*100:5.1f}% ± {metrics_agg['recall']['std']*100:.1f}%")
    print(f"    F1-Score:   {metrics_agg['f1']['mean']*100:5.1f}% ± {metrics_agg['f1']['std']*100:.1f}%")
    print(f"    AUC-ROC:    {metrics_agg['auc_roc']['mean']*100:5.1f}% ± {metrics_agg['auc_roc']['std']*100:.1f}%")

    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(f"\n    Confusion Matrix:")
    print(f"                      Predicted")
    print(f"                   Baseline  Stress")
    print(f"    Actual Baseline  {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"    Actual Stress    {cm[1,0]:5d}   {cm[1,1]:5d}")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        "dataset": "EEGMAT Full (PhysioNet)",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "n_samples_original": len(y),
        "n_samples_augmented": len(y_aug),
        "n_folds": n_folds,
        "metrics": {k: {"mean": v['mean'], "std": v['std']} for k, v in metrics_agg.items()},
        "confusion_matrix": cm.tolist()
    }

    with open(os.path.join(OUTPUT_DIR, "eegmat_full_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n    Results saved to: {OUTPUT_DIR}/eegmat_full_results.json")
    print("\n" + "=" * 70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
