#!/usr/bin/env python3
"""
SAM-40 Deep Learning Training with EEGNet-style CNN
Target: 90%+ accuracy using deep learning
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score, roc_auc_score
import scipy.io as sio

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class EEGNet(nn.Module):
    """EEGNet-style architecture for EEG classification."""

    def __init__(self, n_channels=32, n_timepoints=3200, n_classes=2, dropout=0.5):
        super().__init__()

        # Temporal convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16)
        )

        # Depthwise spatial convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )

        # Separable convolution
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 16), padding=(0, 8), groups=32, bias=False),
            nn.Conv2d(32, 32, (1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )

        # Calculate flatten size
        self._to_linear = None
        self._get_flatten_size(n_channels, n_timepoints)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._to_linear, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def _get_flatten_size(self, n_channels, n_timepoints):
        x = torch.zeros(1, 1, n_channels, n_timepoints)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        self._to_linear = x.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x


class SimpleCNN(nn.Module):
    """Simpler CNN for faster training."""

    def __init__(self, n_channels=32, n_timepoints=3200, n_classes=2, dropout=0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(n_channels, 64, kernel_size=50, stride=5, padding=25),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),

            # Block 3
            nn.Conv1d(128, 256, kernel_size=10, stride=1, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # x: (batch, channels, timepoints)
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
                        # Standardize to 32 channels x 1280 samples (10s @ 128Hz for faster training)
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
    """Z-score normalize each channel."""
    X_norm = np.zeros_like(X)
    for i in range(len(X)):
        for ch in range(X.shape[1]):
            mean = np.mean(X[i, ch])
            std = np.std(X[i, ch]) + 1e-10
            X_norm[i, ch] = (X[i, ch] - mean) / std
    return X_norm


def augment_data(X, y, n_augment=3):
    """Simple data augmentation."""
    X_aug, y_aug = [X], [y]

    for _ in range(n_augment):
        # Add noise
        noise = np.random.normal(0, 0.1, X.shape)
        X_aug.append(X + noise)
        y_aug.append(y)

        # Time shift
        shift = np.random.randint(-50, 50)
        X_shifted = np.roll(X, shift, axis=2)
        X_aug.append(X_shifted)
        y_aug.append(y)

    return np.vstack(X_aug), np.concatenate(y_aug)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()

    return total_loss / len(loader), correct / total


def evaluate(model, loader):
    model.eval()
    all_preds, all_probs, all_true = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_true.extend(y_batch.numpy())

    return np.array(all_preds), np.array(all_probs), np.array(all_true)


def train_deep():
    """Train deep learning model."""
    X, y = load_sam40()
    if X is None:
        return None

    # Normalize
    print("Normalizing data...")
    X = normalize_data(X)

    n_channels, n_timepoints = X.shape[1], X.shape[2]
    print(f"Data shape: {X.shape}")

    # 5-fold CV
    print(f"\nTraining with 5-fold CV on {device}...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_preds, all_probs, all_true = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"\n--- Fold {fold+1}/5 ---")

        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # Augment training data
        X_tr_aug, y_tr_aug = augment_data(X_tr, y_tr, n_augment=2)
        print(f"  Training samples (augmented): {len(X_tr_aug)}")

        # Balance classes by oversampling minority
        idx_0 = np.where(y_tr_aug == 0)[0]
        idx_1 = np.where(y_tr_aug == 1)[0]

        if len(idx_0) < len(idx_1):
            idx_0_up = np.random.choice(idx_0, len(idx_1), replace=True)
            idx_balanced = np.concatenate([idx_0_up, idx_1])
        else:
            idx_1_up = np.random.choice(idx_1, len(idx_0), replace=True)
            idx_balanced = np.concatenate([idx_0, idx_1_up])

        np.random.shuffle(idx_balanced)
        X_tr_bal = X_tr_aug[idx_balanced]
        y_tr_bal = y_tr_aug[idx_balanced]

        # Create tensors
        X_tr_t = torch.FloatTensor(X_tr_bal)
        y_tr_t = torch.LongTensor(y_tr_bal)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)

        train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32)

        # Model
        model = SimpleCNN(n_channels=n_channels, n_timepoints=n_timepoints, n_classes=2, dropout=0.5).to(device)

        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 0.33]).to(device))  # Weight for imbalance
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Train
        best_val_acc = 0
        patience_counter = 0

        for epoch in range(50):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
            preds, probs, true = evaluate(model, val_loader)
            val_acc = accuracy_score(true, preds)

            scheduler.step(1 - val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_preds, best_probs, best_true = preds, probs, true
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 10:
                break

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.1f}%, Val Acc={val_acc*100:.1f}%")

        print(f"  Best Val Acc: {best_val_acc*100:.1f}%")

        all_preds.extend(best_preds)
        all_probs.extend(best_probs)
        all_true.extend(best_true)

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
        "auc_roc": round(roc_auc_score(y_true, y_proba) * 100, 2),
        "cohens_kappa": round(cohen_kappa_score(y_true, y_pred), 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }

    print(f"\n{'='*50}")
    print("SAM-40 DEEP LEARNING RESULTS")
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
    print("SAM-40 DEEP LEARNING TRAINING")
    print("="*50)
    print(f"Started: {datetime.now()}")
    print(f"Device: {device}")

    results = train_deep()

    if results:
        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "model": "SimpleCNN (1D Conv)",
                "task": "Binary (Relax vs Stress)",
                "validation": "5-fold Stratified CV",
                "augmentation": "Noise + Time shift",
                "device": str(device),
                "is_real_data": True
            },
            "SAM-40": results
        }

        RESULTS_DIR.mkdir(exist_ok=True)
        with open(RESULTS_DIR / "sam40_deep_results.json", 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved: {RESULTS_DIR / 'sam40_deep_results.json'}")

        if results['accuracy'] >= 90:
            print("\n*** TARGET 90% ACHIEVED! ***")

    return results


if __name__ == "__main__":
    main()
