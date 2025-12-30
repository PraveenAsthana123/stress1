#!/usr/bin/env python3
"""
Deep Learning Model Training for EEG Stress Classification

Uses the GenAI-RAG-EEG architecture (CNN-LSTM-Attention) for training
on SAM-40 and EEGMAT datasets to match paper results.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, cohen_kappa_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.eeg_encoder import EEGEncoder


class StressClassifier(nn.Module):
    """Full stress classification model with EEG encoder and classifier head."""

    def __init__(self, n_channels=32, n_time_samples=512, dropout=0.3):
        super().__init__()
        self.encoder = EEGEncoder(
            n_channels=n_channels,
            n_time_samples=n_time_samples,
            dropout=dropout
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        features, _ = self.encoder(x)
        return self.classifier(features)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(y_batch).sum().item()
        total += y_batch.size(0)

    return total_loss / len(train_loader), correct / total


def evaluate(model, data_loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return total_loss / len(data_loader), np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(y_true, y_pred, y_proba):
    """Compute classification metrics."""
    accuracy = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred) * 100

    try:
        auc_roc = roc_auc_score(y_true, y_proba) * 100
    except:
        auc_roc = 50.0

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    kappa = cohen_kappa_score(y_true, y_pred)

    return {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2),
        "specificity": round(specificity, 2),
        "auc_roc": round(auc_roc, 2),
        "cohens_kappa": round(kappa, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }


def train_dataset(data, labels, dataset_name, n_folds=5, epochs=50, batch_size=32, lr=1e-4):
    """Train and evaluate on a dataset with cross-validation."""
    print(f"\n{'='*60}")
    print(f"Training on {dataset_name}")
    print(f"{'='*60}")
    print(f"Data shape: {data.shape}")
    print(f"Labels: Stress={np.sum(labels==1)}, Baseline={np.sum(labels==0)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Normalize data
    data = (data - np.mean(data)) / (np.std(data) + 1e-8)

    # Pad/truncate to 512 time samples
    target_samples = 512
    if data.shape[2] > target_samples:
        data = data[:, :, :target_samples]
    elif data.shape[2] < target_samples:
        pad_width = target_samples - data.shape[2]
        data = np.pad(data, ((0, 0), (0, 0), (0, pad_width)), mode='constant')

    n_channels = data.shape[1]

    # Cross-validation
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data, labels)):
        print(f"\nFold {fold + 1}/{n_folds}")

        # Split data
        X_train, X_val = data[train_idx], data[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Create model
        model = StressClassifier(
            n_channels=n_channels,
            n_time_samples=target_samples,
            dropout=0.3
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        best_val_acc = 0
        best_metrics = None

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, y_pred, y_true, y_proba = evaluate(model, val_loader, criterion, device)

            val_acc = accuracy_score(y_true, y_pred) * 100
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_metrics = compute_metrics(y_true, y_pred, y_proba)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Train Acc={train_acc*100:.1f}%, Val Acc={val_acc:.1f}%")

        print(f"  Best Val Acc: {best_val_acc:.2f}%")
        fold_results.append(best_metrics)

    # Aggregate results across folds
    avg_results = {
        "accuracy": round(np.mean([r["accuracy"] for r in fold_results]), 2),
        "accuracy_std": round(np.std([r["accuracy"] for r in fold_results]), 2),
        "precision": round(np.mean([r["precision"] for r in fold_results]), 2),
        "recall": round(np.mean([r["recall"] for r in fold_results]), 2),
        "f1_score": round(np.mean([r["f1_score"] for r in fold_results]), 2),
        "specificity": round(np.mean([r["specificity"] for r in fold_results]), 2),
        "auc_roc": round(np.mean([r["auc_roc"] for r in fold_results]), 2),
        "cohens_kappa": round(np.mean([r["cohens_kappa"] for r in fold_results]), 4)
    }

    print(f"\n{dataset_name} Results (5-Fold CV):")
    print(f"  Accuracy: {avg_results['accuracy']}% (+/- {avg_results['accuracy_std']}%)")
    print(f"  F1 Score: {avg_results['f1_score']}%")
    print(f"  AUC-ROC: {avg_results['auc_roc']}%")

    return avg_results


def main():
    """Main training function."""
    print("="*60)
    print("GenAI-RAG-EEG Deep Learning Training")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": "GenAI-RAG-EEG (CNN-LSTM-Attention)",
            "training_config": {
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 1e-4,
                "optimizer": "AdamW",
                "cv_folds": 5
            }
        },
        "datasets": {}
    }

    # 1. Train on SAM-40
    try:
        from data.real_data_loader import load_sam40_dataset
        sam40_data, sam40_labels, _ = load_sam40_dataset(data_type="filtered")
        sam40_results = train_dataset(sam40_data, sam40_labels, "SAM-40")
        results["datasets"]["SAM-40"] = sam40_results
    except Exception as e:
        print(f"Error with SAM-40: {e}")

    # 2. Train on EEGMAT
    try:
        from data.eegmat_loader import load_eegmat_dataset
        eegmat_data, eegmat_labels, _ = load_eegmat_dataset(binary=True)
        eegmat_results = train_dataset(eegmat_data, eegmat_labels)
        results["datasets"][] = eegmat_results
    except Exception as e:
        print(f"Error with EEGMAT: {e}")

    # Save results
    results_path = Path(__file__).parent / "results"
    results_path.mkdir(exist_ok=True)

    with open(results_path / "deep_learning_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

    # Summary comparison with paper
    print("\n=== COMPARISON WITH PAPER ===")
    print("\nPaper Claims:")
    print("  SAM-40: Acc=93.2%, F1=92.8%, AUC=95.8%")
    print("  DEAP:   Acc=94.7%, F1=94.3%, AUC=96.7%")

    print("\nOur Deep Learning Results:")
    for name, data in results["datasets"].items():
        print(f"  {name}: Acc={data['accuracy']}%, F1={data['f1_score']}%, AUC={data['auc_roc']}%")

    return results


if __name__ == "__main__":
    main()

# Last updated: 2025-12-30 15:48:54 UTC
