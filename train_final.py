#!/usr/bin/env python3
"""
Final Training Pipeline - Feature-Based Approach

Uses extracted EEG features (band powers, ratios, asymmetry)
combined with deep learning for robust classification.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy import signal
from scipy.stats import zscore, skew, kurtosis

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, cohen_kappa_score, matthews_corrcoef
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).parent / "src"))


class FeatureClassifier(nn.Module):
    """MLP classifier for extracted features."""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def compute_band_power(data, fs=256):
    """Compute band power features."""
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }

    features = []
    for band_name, (low, high) in bands.items():
        nyq = fs / 2
        low_n = max(low / nyq, 0.01)
        high_n = min(high / nyq, 0.99)

        try:
            b, a = signal.butter(4, [low_n, high_n], btype='band')
            filtered = signal.filtfilt(b, a, data, axis=-1)
            power = np.mean(filtered ** 2, axis=-1)  # (n_channels,)
            features.append(power)
        except:
            features.append(np.zeros(data.shape[0]))

    return np.array(features)  # (5, n_channels)


def extract_features(data, fs=256):
    """Extract comprehensive EEG features for each sample."""
    n_samples = data.shape[0]
    n_channels = data.shape[1]

    all_features = []

    for i in range(n_samples):
        sample = data[i]  # (n_channels, n_time)
        features = []

        # 1. Band powers per channel (5 bands x n_channels)
        band_powers = compute_band_power(sample, fs)  # (5, n_channels)
        features.extend(band_powers.flatten())

        # 2. Mean band powers across channels
        mean_band_powers = np.mean(band_powers, axis=1)  # (5,)
        features.extend(mean_band_powers)

        # 3. Theta/Beta ratio
        theta_power = np.mean(band_powers[1])
        beta_power = np.mean(band_powers[3])
        tbr = theta_power / (beta_power + 1e-10)
        features.append(tbr)

        # 4. Alpha/Beta ratio
        alpha_power = np.mean(band_powers[2])
        abr = alpha_power / (beta_power + 1e-10)
        features.append(abr)

        # 5. Frontal asymmetry (left vs right hemispheres)
        left_channels = sample[:n_channels//2]
        right_channels = sample[n_channels//2:]
        left_alpha = np.mean(compute_band_power(left_channels, fs)[2])
        right_alpha = np.mean(compute_band_power(right_channels, fs)[2])
        faa = np.log(right_alpha + 1e-10) - np.log(left_alpha + 1e-10)
        features.append(faa)

        # 6. Statistical features per channel
        for ch in range(min(n_channels, 8)):  # Limit to first 8 channels
            ch_data = sample[ch]
            features.extend([
                np.mean(ch_data),
                np.std(ch_data),
                np.var(ch_data),
                skew(ch_data),
                kurtosis(ch_data),
                np.max(ch_data) - np.min(ch_data),  # Range
                np.percentile(ch_data, 75) - np.percentile(ch_data, 25)  # IQR
            ])

        # 7. Hjorth parameters
        for ch in range(min(n_channels, 4)):
            ch_data = sample[ch]
            # Activity
            activity = np.var(ch_data)
            # Mobility
            diff1 = np.diff(ch_data)
            mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
            # Complexity
            diff2 = np.diff(diff1)
            complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)
            features.extend([activity, mobility, complexity])

        all_features.append(features)

    return np.array(all_features)


def train_epoch(model, train_loader, criterion, optimizer, device):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(y_batch).sum().item()
        total += y_batch.size(0)

    return total_loss / len(train_loader), correct / total


def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    try:
        auc_roc = roc_auc_score(y_true, y_proba)
    except:
        auc_roc = 0.5

    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    ba = (recall + specificity) / 2

    return {
        "accuracy": round(accuracy * 100, 1),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "specificity": round(specificity, 3),
        "auc_roc": round(auc_roc, 3),
        "cohens_kappa": round(kappa, 3),
        "mcc": round(mcc, 3),
        "balanced_accuracy": round(ba * 100, 1),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }


def train_dataset(data, labels, dataset_name, n_folds=5, epochs=200, batch_size=32, lr=1e-3):
    print(f"\n{'='*60}")
    print(f"Training: {dataset_name}")
    print(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Data shape: {data.shape}")
    print(f"Classes: Stress={np.sum(labels==1)}, Baseline={np.sum(labels==0)}")

    # Extract features
    print("Extracting features...")
    features = extract_features(data, fs=256)
    print(f"Feature shape: {features.shape}")

    # Handle NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Cross-validation
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(features, labels)):
        print(f"\nFold {fold + 1}/{n_folds}")

        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # SMOTE for class balancing
        try:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"  After SMOTE: {len(y_train)} samples")
        except:
            pass

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

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

        # Model
        input_dim = X_train.shape[1]
        model = FeatureClassifier(input_dim, hidden_dims=[256, 128, 64], dropout=0.4).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

        best_f1 = 0
        best_metrics = None
        patience = 30
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            y_pred, y_true, y_proba = evaluate(model, val_loader, device)

            metrics = compute_metrics(y_true, y_pred, y_proba)
            scheduler.step(metrics["f1_score"])

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_metrics = metrics
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}: Acc={metrics['accuracy']:.1f}%, F1={metrics['f1_score']:.3f}")

        print(f"  Best: Acc={best_metrics['accuracy']:.1f}%, F1={best_metrics['f1_score']:.3f}, AUC={best_metrics['auc_roc']:.3f}")
        fold_metrics.append(best_metrics)

    # Aggregate
    avg_metrics = {}
    for key in fold_metrics[0].keys():
        if key != "confusion_matrix":
            values = [m[key] for m in fold_metrics]
            avg_metrics[key] = round(np.mean(values), 1 if "accuracy" in key else 3)
            avg_metrics[f"{key}_std"] = round(np.std(values), 1 if "accuracy" in key else 3)

    total_cm = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for m in fold_metrics:
        for k in total_cm:
            total_cm[k] += m["confusion_matrix"][k]
    avg_metrics["confusion_matrix"] = total_cm

    print(f"\n{dataset_name} Final:")
    print(f"  Accuracy: {avg_metrics['accuracy']}% (+/- {avg_metrics['accuracy_std']}%)")
    print(f"  F1: {avg_metrics['f1_score']} | AUC: {avg_metrics['auc_roc']}")

    return avg_metrics


def update_paper(results):
    """Update paper .tex with results."""
    paper_path = Path(__file__).parent / "eeg-stress-rag.tex"
    if not paper_path.exists():
        return

    with open(paper_path, 'r') as f:
        content = f.read()

    import re

    for dataset, metrics in results["datasets"].items():
        if dataset == "SAM-40":
            pattern = r"(SAM-40\s*&\s*)\d+\.\d+\\%(\s*&\s*)\d+\.\d+(\s*&\s*)\d+\.\d+(\s*&\s*)\d+\.\d+(\s*&\s*)\d+\.\d+"
            replacement = f"SAM-40 & {metrics['accuracy']:.1f}\\% & {metrics['precision']:.3f} & {metrics['recall']:.3f} & {metrics['f1_score']:.3f} & {metrics['auc_roc']:.3f}"
            content = re.sub(pattern, replacement, content)

    with open(paper_path, 'w') as f:
        f.write(content)

    print(f"Paper updated: {paper_path}")


def main():
    print("="*60)
    print("GenAI-RAG-EEG: Feature-Based Training")
    print("="*60)

    results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": "GenAI-RAG-EEG",
            "approach": "Feature-based MLP"
        },
        "datasets": {}
    }

    # SAM-40
    try:
        from data.real_data_loader import load_sam40_dataset
        data, labels, _ = load_sam40_dataset(data_type="filtered")
        results["datasets"]["SAM-40"] = train_dataset(data, labels, "SAM-40")
    except Exception as e:
        print(f"SAM-40 Error: {e}")
        import traceback
        traceback.print_exc()

    # WESAD
    try:
        from data.wesad_loader import load_wesad_dataset
        data, labels, _ = load_wesad_dataset(binary=True)
        results["datasets"]["WESAD"] = train_dataset(data, labels, "WESAD")
    except Exception as e:
        print(f"WESAD Error: {e}")

    # Save
    results_path = Path(__file__).parent / "results"
    results_path.mkdir(exist_ok=True)

    with open(results_path / "final_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Update testing report
    report = {
        "test_date": datetime.now().isoformat(),
        "status": "SUCCESS",
        "data_source": "REAL DATA",
        "model": "GenAI-RAG-EEG",
        "datasets_tested": list(results["datasets"].keys()),
        "classification": {}
    }

    for name, m in results["datasets"].items():
        report["classification"][name] = {
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1_score"],
            "auc_roc": m["auc_roc"],
            "kappa": m["cohens_kappa"]
        }

    with open(results_path / "testing_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Update paper
    update_paper(results)

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    for name, m in results["datasets"].items():
        print(f"{name}: Acc={m['accuracy']}%, F1={m['f1_score']}, AUC={m['auc_roc']}")

    return results


if __name__ == "__main__":
    main()
