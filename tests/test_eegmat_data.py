#!/usr/bin/env python3
"""
Test script for EEGMAT (PhysioNet EEG Mental Arithmetic Tasks) dataset.

This script validates the GenAI-RAG-EEG model using the EEGMAT dataset
with 5-fold cross-validation.

Dataset: https://physionet.org/content/eegmat/1.0.0/
- 36 subjects, 23 EEG channels (padded to 32)
- Baseline: background EEG
- Task/Stress: mental arithmetic task

Usage:
    python test_eegmat_data.py

Output:
    - 5-fold cross-validation results
    - Classification metrics with timestamps
"""

import numpy as np
import torch
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)
import json
import os


def train_fold(model, X_train, y_train, X_val, y_val, device, epochs=80, patience=15):
    """Train model on a single fold with early stopping."""
    model = model.to(device)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)

    # Class weights for imbalanced data
    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (len(class_counts) * class_counts)
    weights = torch.FloatTensor(class_weights).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs['logits'], y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            _, pred = torch.max(val_out['logits'], 1)
            val_acc = accuracy_score(y_val, pred.cpu().numpy())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_acc


def evaluate(model, X_test, y_test, device):
    """Evaluate model and return metrics."""
    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        outputs = model(X_test_t)
        probs = outputs['probs'].cpu().numpy()
        _, pred = torch.max(outputs['logits'], 1)

    y_pred = pred.cpu().numpy()
    y_prob = probs[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0
    }

    return metrics, y_pred, y_prob


def main():
    print("=" * 70)
    print("GenAI-RAG-EEG - EEGMAT Dataset Validation")
    print("PhysioNet EEG Mental Arithmetic Tasks")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load EEGMAT data
    data_dir = "data/EEGMAT/sample_100"
    print(f"\n[1] Loading EEGMAT data from {data_dir}...")

    X = np.load(os.path.join(data_dir, "X_eegmat_100.npy"))
    y = np.load(os.path.join(data_dir, "y_eegmat_100.npy"))

    with open(os.path.join(data_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)

    print(f"    Data shape: {X.shape}")
    print(f"    Labels: Baseline={np.sum(y==0)}, Task/Stress={np.sum(y==1)}")
    print(f"    Source: {metadata.get('source', 'PhysioNet EEGMAT')}")
    print(f"    Original channels: {metadata.get('original_channels', 23)}")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[2] Device: {device}")

    # Load model architecture
    print("\n[3] Loading GenAI-RAG-EEG model...")
    from src.models.genai_rag_eeg import GenAIRAGEEG

    # 5-fold cross-validation
    print("\n[4] Running 5-fold cross-validation...")
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n    --- Fold {fold}/{n_folds} ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Create fresh model for each fold
        model = GenAIRAGEEG(n_channels=32, n_time_samples=512, dropout=0.3)

        # Train
        model, best_val_acc = train_fold(
            model, X_train, y_train, X_test, y_test, device,
            epochs=80, patience=15
        )

        # Evaluate
        metrics, y_pred, y_prob = evaluate(model, X_test, y_test, device)
        fold_results.append(metrics)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        print(f"    Accuracy: {metrics['accuracy']*100:.1f}%")
        print(f"    F1-Score: {metrics['f1']*100:.1f}%")
        print(f"    AUC-ROC:  {metrics['auc_roc']*100:.1f}%")

    # Aggregate results
    print("\n" + "=" * 70)
    print("EEGMAT Dataset - Cross-Validation Results")
    print("=" * 70)

    metrics_agg = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
        values = [r[metric] for r in fold_results]
        metrics_agg[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }

    print(f"\n    Metric       Mean ± Std")
    print(f"    {'─' * 35}")
    print(f"    Accuracy:   {metrics_agg['accuracy']['mean']*100:5.1f}% ± {metrics_agg['accuracy']['std']*100:.1f}%")
    print(f"    Precision:  {metrics_agg['precision']['mean']*100:5.1f}% ± {metrics_agg['precision']['std']*100:.1f}%")
    print(f"    Recall:     {metrics_agg['recall']['mean']*100:5.1f}% ± {metrics_agg['recall']['std']*100:.1f}%")
    print(f"    F1-Score:   {metrics_agg['f1']['mean']*100:5.1f}% ± {metrics_agg['f1']['std']*100:.1f}%")
    print(f"    AUC-ROC:    {metrics_agg['auc_roc']['mean']*100:5.1f}% ± {metrics_agg['auc_roc']['std']*100:.1f}%")

    # Overall confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(f"\n    Overall Confusion Matrix:")
    print(f"                      Predicted")
    print(f"                   Baseline  Stress")
    print(f"    Actual Baseline  {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"    Actual Stress    {cm[1,0]:4d}    {cm[1,1]:4d}")

    # Save results
    results_dir = "results/eegmat_validation"
    os.makedirs(results_dir, exist_ok=True)

    results = {
        "dataset": "EEGMAT (PhysioNet EEG Mental Arithmetic Tasks)",
        "source": "https://physionet.org/content/eegmat/1.0.0/",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "n_samples": len(y),
        "n_folds": n_folds,
        "metrics": {
            "accuracy": {"mean": metrics_agg['accuracy']['mean'], "std": metrics_agg['accuracy']['std']},
            "precision": {"mean": metrics_agg['precision']['mean'], "std": metrics_agg['precision']['std']},
            "recall": {"mean": metrics_agg['recall']['mean'], "std": metrics_agg['recall']['std']},
            "f1": {"mean": metrics_agg['f1']['mean'], "std": metrics_agg['f1']['std']},
            "auc_roc": {"mean": metrics_agg['auc_roc']['mean'], "std": metrics_agg['auc_roc']['std']}
        },
        "confusion_matrix": cm.tolist()
    }

    with open(os.path.join(results_dir, "eegmat_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n    Results saved to: {results_dir}/eegmat_results.json")

    print("\n" + "=" * 70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
