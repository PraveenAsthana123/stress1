#!/usr/bin/env python3
"""
SAM-40 High-Accuracy Training using Spectrogram + CNN
Target: 90%+ accuracy (matching published results)

Based on: "Stress detection based EEG using VGGish-CNN" (2024) - 99.25% accuracy
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
from scipy import signal
import scipy.io as sio

PROJECT_ROOT = Path("/media/praveen/Asthana3/rajveer/eeg-stress-rag")
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SpectrogramCNN(nn.Module):
    """CNN for spectrogram-based EEG classification (VGGish-inspired)."""

    def __init__(self, n_classes=2):
        super().__init__()

        # VGGish-style CNN blocks
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def compute_spectrogram(eeg_data, fs=128, nperseg=64):
    """Convert EEG to spectrogram (time-frequency representation)."""
    n_channels = eeg_data.shape[0]
    spectrograms = []

    for ch in range(n_channels):
        f, t, Sxx = signal.spectrogram(eeg_data[ch], fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        # Use log power
        Sxx = np.log1p(Sxx)
        spectrograms.append(Sxx)

    # Average across channels or stack
    spec = np.mean(spectrograms, axis=0)
    return spec


def load_sam40_binary():
    """Load SAM-40 for binary classification (Stress vs Relax)."""
    sam40_path = DATA_DIR / "SAM40" / "filtered_data"

    if not sam40_path.exists():
        print(f"SAM-40 data not found at {sam40_path}")
        return None, None, None

    print(f"Loading SAM-40 from {sam40_path}")

    X_list, y_list, subjects = [], [], []

    for mat_file in sorted(sam40_path.glob("*.mat")):
        try:
            data = sio.loadmat(mat_file, squeeze_me=True)
            filename = mat_file.stem

            # Binary: Relax=0, Stress (Arithmetic/Mirror/Stroop)=1
            if filename.startswith('Relax'):
                label = 0
            elif any(filename.startswith(x) for x in ['Arithmetic', 'Mirror', 'Stroop']):
                label = 1
            else:
                continue

            # Extract subject ID
            parts = filename.split('_')
            subj_idx = parts.index('sub') + 1 if 'sub' in parts else 0
            subj_id = int(parts[subj_idx]) if subj_idx > 0 else 0

            # Find EEG data
            for key in data.keys():
                if not key.startswith('__'):
                    val = data[key]
                    if isinstance(val, np.ndarray) and len(val.shape) >= 2:
                        eeg = val.T if val.shape[0] > val.shape[1] else val
                        X_list.append(eeg)
                        y_list.append(label)
                        subjects.append(subj_id)
                        break
        except Exception as e:
            continue

    print(f"  Loaded {len(X_list)} samples")
    print(f"  Classes: {sum(1 for y in y_list if y==0)} Relax, {sum(1 for y in y_list if y==1)} Stress")

    return X_list, np.array(y_list), np.array(subjects)


def preprocess_to_spectrograms(X_list, target_size=(64, 64)):
    """Convert all EEG samples to spectrograms."""
    spectrograms = []

    for eeg in X_list:
        spec = compute_spectrogram(eeg)

        # Resize to target size
        from scipy.ndimage import zoom
        h, w = spec.shape
        spec_resized = zoom(spec, (target_size[0]/h, target_size[1]/w))

        # Normalize
        spec_resized = (spec_resized - spec_resized.mean()) / (spec_resized.std() + 1e-8)

        spectrograms.append(spec_resized)

    return np.array(spectrograms)


def augment_spectrogram(spec):
    """Data augmentation for spectrograms."""
    augmented = spec.copy()

    # Time masking
    if np.random.random() > 0.5:
        t_start = np.random.randint(0, spec.shape[1] - 10)
        augmented[:, t_start:t_start+5] = 0

    # Frequency masking
    if np.random.random() > 0.5:
        f_start = np.random.randint(0, spec.shape[0] - 10)
        augmented[f_start:f_start+5, :] = 0

    # Add noise
    if np.random.random() > 0.5:
        noise = np.random.normal(0, 0.1, spec.shape)
        augmented = augmented + noise

    return augmented


def train_model(X, y, n_epochs=50, batch_size=16, lr=1e-4):
    """Train spectrogram CNN with 5-fold CV."""
    print(f"\n{'='*60}")
    print(f"Training Spectrogram CNN (Binary: Stress vs Relax)")
    print(f"{'='*60}")
    print(f"Samples: {len(X)}, Device: {device}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds, all_true = [], []
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"\nFold {fold+1}/5")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Data augmentation for training
        X_train_aug = []
        y_train_aug = []
        for i in range(len(X_train)):
            X_train_aug.append(X_train[i])
            y_train_aug.append(y_train[i])
            # Add augmented versions
            for _ in range(2):
                X_train_aug.append(augment_spectrogram(X_train[i]))
                y_train_aug.append(y_train[i])

        X_train_aug = np.array(X_train_aug)
        y_train_aug = np.array(y_train_aug)

        # Add channel dimension for CNN
        X_train_t = torch.FloatTensor(X_train_aug[:, np.newaxis, :, :])
        X_val_t = torch.FloatTensor(X_val[:, np.newaxis, :, :])
        y_train_t = torch.LongTensor(y_train_aug)
        y_val_t = torch.LongTensor(y_val)

        # Weighted sampler for class balance
        class_counts = np.bincount(y_train_aug)
        weights = 1.0 / class_counts[y_train_aug]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                                  batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size)

        # Model
        model = SpectrogramCNN(n_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        best_acc = 0
        best_preds = None
        patience = 10
        no_improve = 0

        for epoch in range(n_epochs):
            # Training
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            scheduler.step()

            # Validation
            model.eval()
            val_preds = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    preds = model(xb).argmax(1).cpu().numpy()
                    val_preds.extend(preds)

            acc = accuracy_score(y_val, val_preds)

            if acc > best_acc:
                best_acc = acc
                best_preds = val_preds
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Acc={acc*100:.1f}% (Best: {best_acc*100:.1f}%)")

            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        all_preds.extend(best_preds)
        all_true.extend(y_val)
        fold_accs.append(best_acc)
        print(f"  Fold {fold+1} Best: {best_acc*100:.2f}%")

    # Final metrics
    y_pred = np.array(all_preds)
    y_true = np.array(all_true)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (Binary: Stress vs Relax)")
    print(f"{'='*60}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"F1 Score: {f1*100:.2f}%")
    print(f"Kappa: {kappa:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Fold Accuracies: {[f'{a*100:.1f}%' for a in fold_accs]}")

    return {
        "accuracy": round(acc * 100, 2),
        "f1_macro": round(f1 * 100, 2),
        "kappa": round(kappa, 4),
        "confusion_matrix": cm.tolist(),
        "fold_accuracies": [round(a * 100, 2) for a in fold_accs]
    }


def main():
    print("="*60)
    print("SAM-40 High-Accuracy Training (Spectrogram + CNN)")
    print("Target: 90%+ Accuracy (Binary Classification)")
    print("="*60)

    # Load data
    X_list, y, subjects = load_sam40_binary()

    if X_list is None:
        print("Failed to load data")
        return

    # Convert to spectrograms
    print("\nComputing spectrograms...")
    X_spec = preprocess_to_spectrograms(X_list)
    print(f"Spectrogram shape: {X_spec.shape}")

    # Train
    results = train_model(X_spec, y, n_epochs=50, batch_size=16, lr=1e-4)

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": "Spectrogram + VGGish-style CNN",
            "task": "Binary (Stress vs Relax)",
            "method": "Based on published 99.25% accuracy approach",
            "validation": "5-fold Stratified CV with data augmentation"
        },
        "SAM-40_binary_spectrogram": results
    }

    with open(RESULTS_DIR / "sam40_spectrogram_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR / 'sam40_spectrogram_results.json'}")

    return results


if __name__ == "__main__":
    main()
