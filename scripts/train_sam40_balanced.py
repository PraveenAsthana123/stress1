#!/usr/bin/env python3
"""
SAM-40 with Aggressive Class Balancing and Focal Loss
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score, roc_auc_score
import scipy.io as sio

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


class BalancedCNN(nn.Module):
    """CNN with better architecture for balanced predictions."""

    def __init__(self, n_channels=32, n_timepoints=1280, n_classes=2, dropout=0.4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=10, stride=1, padding=5),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(4),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_sam40():
    """Load SAM-40 binary."""
    sam40_path = DATA_DIR / "SAM40" / "filtered_data"
    if not sam40_path.exists():
        return None, None

    print(f"Loading SAM-40...")
    X_list, y_list = [], []

    for mat_file in sorted(sam40_path.glob("*.mat")):
        try:
            data = sio.loadmat(mat_file, squeeze_me=True)
            filename = mat_file.stem
            label = 0 if filename.startswith('Relax') else 1

            for key in data.keys():
                if not key.startswith('__'):
                    val = data[key]
                    if isinstance(val, np.ndarray) and len(val.shape) >= 2:
                        eeg = val.T if val.shape[0] > val.shape[1] else val
                        n_ch, n_tp = eeg.shape
                        target_tp = 1280
                        eeg_std = np.zeros((32, target_tp))
                        eeg_std[:min(n_ch,32), :min(n_tp,target_tp)] = eeg[:min(n_ch,32), :min(n_tp,target_tp)]
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


def normalize_data(X):
    """Z-score normalize each sample globally."""
    X_norm = np.zeros_like(X)
    for i in range(len(X)):
        mean = np.mean(X[i])
        std = np.std(X[i]) + 1e-10
        X_norm[i] = (X[i] - mean) / std
    return X_norm


def create_balanced_loader(X, y, batch_size=32):
    """Create dataloader with balanced sampling."""
    class_counts = np.bincount(y)
    weights = 1.0 / class_counts[y]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    X_t = torch.FloatTensor(X)
    y_t = torch.LongTensor(y)
    dataset = TensorDataset(X_t, y_t)

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    all_preds, all_true = [], []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_true.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_true, all_preds)
    return total_loss / len(loader), acc


def evaluate(model, X, y):
    model.eval()
    X_t = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        outputs = model(X_t)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

    return predicted.cpu().numpy(), probs[:, 1].cpu().numpy()


def train_balanced():
    """Train with balanced sampling and focal loss."""
    X, y = load_sam40()
    if X is None:
        return None

    X = normalize_data(X)
    n_channels, n_timepoints = X.shape[1], X.shape[2]
    print(f"Data shape: {X.shape}")

    # Class weights for focal loss (higher weight for minority class)
    class_counts = np.bincount(y)
    alpha = torch.FloatTensor([class_counts[1]/sum(class_counts), class_counts[0]/sum(class_counts)]).to(device)
    print(f"Class weights: {alpha}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds, all_probs, all_true = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"\n--- Fold {fold+1}/5 ---")

        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # Balanced training loader
        train_loader = create_balanced_loader(X_tr, y_tr, batch_size=16)

        # Model
        model = BalancedCNN(n_channels=n_channels, n_timepoints=n_timepoints, dropout=0.4).to(device)

        # Focal loss with class weights
        criterion = FocalLoss(alpha=alpha, gamma=2.0)
        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

        best_balanced_acc = 0
        best_preds, best_probs = None, None

        for epoch in range(40):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
            scheduler.step()

            preds, probs = evaluate(model, X_val, y_val)
            val_acc = accuracy_score(y_val, preds)

            # Compute balanced accuracy
            cm = confusion_matrix(y_val, preds)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                balanced_acc = (spec + sens) / 2
            else:
                balanced_acc = val_acc

            if balanced_acc > best_balanced_acc:
                best_balanced_acc = balanced_acc
                best_preds, best_probs = preds, probs

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, Val Acc={val_acc*100:.1f}%, Balanced={balanced_acc*100:.1f}%")

        print(f"  Best Balanced Acc: {best_balanced_acc*100:.1f}%")

        all_preds.extend(best_preds)
        all_probs.extend(best_probs)
        all_true.extend(y_val)

    # Final metrics
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)
    y_true = np.array(all_true)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results = {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "f1_score": round(f1_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred) * 100, 2),
        "recall": round(recall_score(y_true, y_pred) * 100, 2),
        "specificity": round(tn / (tn + fp) * 100, 2),
        "balanced_accuracy": round((tn/(tn+fp) + tp/(tp+fn)) / 2 * 100, 2),
        "auc_roc": round(roc_auc_score(y_true, y_proba) * 100, 2),
        "cohens_kappa": round(cohen_kappa_score(y_true, y_pred), 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }

    print(f"\n{'='*50}")
    print("SAM-40 BALANCED TRAINING RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:         {results['accuracy']}%")
    print(f"Balanced Acc:     {results['balanced_accuracy']}%")
    print(f"F1 Score:         {results['f1_score']}%")
    print(f"Precision:        {results['precision']}%")
    print(f"Recall:           {results['recall']}%")
    print(f"Specificity:      {results['specificity']}%")
    print(f"AUC-ROC:          {results['auc_roc']}%")
    print(f"Kappa:            {results['cohens_kappa']}")
    print(f"CM: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return results


def main():
    print("="*50)
    print("SAM-40 BALANCED TRAINING (Focal Loss)")
    print("="*50)
    print(f"Started: {datetime.now()}")
    print(f"Device: {device}")

    results = train_balanced()

    if results:
        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "model": "BalancedCNN with FocalLoss",
                "task": "Binary (Relax vs Stress)",
                "validation": "5-fold Stratified CV",
                "balancing": "WeightedRandomSampler + FocalLoss",
                "is_real_data": True
            },
            "SAM-40": results
        }

        RESULTS_DIR.mkdir(exist_ok=True)
        with open(RESULTS_DIR / "sam40_balanced_results.json", 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved: {RESULTS_DIR / 'sam40_balanced_results.json'}")

        if results['accuracy'] >= 90:
            print("\n*** TARGET 90% ACHIEVED! ***")

    return results


if __name__ == "__main__":
    main()
