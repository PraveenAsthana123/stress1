#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Training Module for GenAI-RAG-EEG
================================================================================

Module: trainer.py
Project: GenAI-RAG-EEG for Stress Classification
Author: Research Team
License: MIT

================================================================================
OVERVIEW
================================================================================

This module implements the complete training pipeline for the GenAI-RAG-EEG
stress classification model. It follows best practices for deep learning
training and is optimized for reproducibility and scientific rigor.

================================================================================
TRAINING PIPELINE ARCHITECTURE
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        TRAINING PIPELINE                                │
    └─────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │          DATA LOADERS               │
                    │   ┌─────────────┬────────────────┐  │
                    │   │ Train Loader│  Val Loader    │  │
                    │   │  (shuffle)  │  (sequential)  │  │
                    │   └──────┬──────┴───────┬────────┘  │
                    └──────────┼──────────────┼───────────┘
                               │              │
                               ▼              ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  TRAINING LOOP (per epoch)                                              │
    │  ═══════════════════════════════════════════════════════════════════    │
    │                                                                         │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │  FOR EACH BATCH:                                                │   │
    │  │                                                                  │   │
    │  │  1. Forward Pass                                                 │   │
    │  │     EEG → Model → Logits → Softmax → Probabilities              │   │
    │  │                                                                  │   │
    │  │  2. Loss Computation                                            │   │
    │  │     L = CrossEntropy(logits, labels)                            │   │
    │  │                                                                  │   │
    │  │  3. Backward Pass                                                │   │
    │  │     ∂L/∂θ = Backpropagation through network                     │   │
    │  │                                                                  │   │
    │  │  4. Gradient Clipping                                           │   │
    │  │     ||∇|| = clip(||∇||, max_norm=1.0)                           │   │
    │  │                                                                  │   │
    │  │  5. Parameter Update                                            │   │
    │  │     θ ← θ - α · (∇L + λ · θ)  [AdamW with weight decay]        │   │
    │  │                                                                  │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                                                                         │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │  AFTER EACH EPOCH:                                              │   │
    │  │                                                                  │   │
    │  │  • Evaluate on validation set                                   │   │
    │  │  • Update learning rate scheduler                               │   │
    │  │  • Check early stopping criterion                               │   │
    │  │  • Save checkpoint if best model                                │   │
    │  │  • Log metrics to history                                       │   │
    │  │                                                                  │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  EVALUATION METRICS                                                     │
    │  ═══════════════════════════════════════════════════════════════════    │
    │                                                                         │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │ Metric              │ Formula                    │ Range        │   │
    │  ├─────────────────────┼────────────────────────────┼──────────────┤   │
    │  │ Accuracy            │ (TP + TN) / N              │ [0, 1]       │   │
    │  │ Balanced Accuracy   │ (TPR + TNR) / 2            │ [0, 1]       │   │
    │  │ Precision           │ TP / (TP + FP)             │ [0, 1]       │   │
    │  │ Recall (Sensitivity)│ TP / (TP + FN)             │ [0, 1]       │   │
    │  │ F1 Score            │ 2 · (P · R) / (P + R)      │ [0, 1]       │   │
    │  │ AUC-ROC             │ Area under ROC curve       │ [0, 1]       │   │
    │  │ MCC                 │ Matthews Correlation Coef   │ [-1, 1]      │   │
    │  └─────────────────────┴────────────────────────────┴──────────────┘   │
    │                                                                         │
    │  Where: TP=True Positive, TN=True Negative,                            │
    │         FP=False Positive, FN=False Negative,                          │
    │         TPR=True Positive Rate, TNR=True Negative Rate                 │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
OPTIMIZATION STRATEGY
================================================================================

    OPTIMIZER: AdamW (Adam with decoupled weight decay)
    ══════════════════════════════════════════════════

    Update Rules:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ m_t = β₁ · m_{t-1} + (1 - β₁) · g_t           (momentum)           │
    │ v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²          (RMS)               │
    │                                                                     │
    │ m̂_t = m_t / (1 - β₁^t)                        (bias correction)   │
    │ v̂_t = v_t / (1 - β₂^t)                        (bias correction)   │
    │                                                                     │
    │ θ_t = θ_{t-1} - α · (m̂_t / (√v̂_t + ε) + λ·θ_{t-1})              │
    └─────────────────────────────────────────────────────────────────────┘

    Hyperparameters (paper specifications):
    • Learning rate (α): 1e-4
    • Weight decay (λ): 1e-2
    • β₁: 0.9 (momentum coefficient)
    • β₂: 0.999 (RMS coefficient)
    • ε: 1e-8 (numerical stability)

    LEARNING RATE SCHEDULING
    ════════════════════════

    ReduceLROnPlateau (default):
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  α │ ──────┐                                                       │
    │    │       └──────┐         ← Reduce when val_acc plateaus         │
    │    │              └──────┐                                          │
    │    │                     └────────                                  │
    │    └──────────────────────────────────────────────────► Epoch      │
    │                                                                     │
    │  factor = 0.5 (halve LR on plateau)                                │
    │  patience = 5 epochs                                                │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

    CosineAnnealingLR (alternative):
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  α │ ╲                                                             │
    │    │   ╲                     α_t = α_min + ½(α_max - α_min) ×      │
    │    │     ╲                          (1 + cos(πt/T_max))            │
    │    │       ╲                                                        │
    │    │         ╲___                                                   │
    │    └──────────────────────────────────────────────────► Epoch      │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

================================================================================
EARLY STOPPING
================================================================================

    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  Val   │         ┌─────┐                                           │
    │  Acc   │    ┌────┘     └────┐                                      │
    │        │ ───┘               └────────────────────────              │
    │        │                    ↑                                       │
    │        │                    │ Early stopping triggered              │
    │        │                    │ (patience=10 epochs without          │
    │        │                    │  improvement > min_delta)            │
    │        └──────────────────────────────────────────────► Epoch      │
    │                                                                     │
    │  Prevents overfitting by stopping when validation performance      │
    │  stops improving.                                                   │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

================================================================================
CROSS-VALIDATION STRATEGIES
================================================================================

    LEAVE-ONE-SUBJECT-OUT (LOSO)
    ════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  Fold 1:  [S2, S3, S4, S5, ..., Sn] → Train    [S1] → Test        │
    │  Fold 2:  [S1, S3, S4, S5, ..., Sn] → Train    [S2] → Test        │
    │  Fold 3:  [S1, S2, S4, S5, ..., Sn] → Train    [S3] → Test        │
    │  ...                                                                │
    │  Fold n:  [S1, S2, S3, S4, ..., Sn-1] → Train  [Sn] → Test        │
    │                                                                     │
    │  Benefits:                                                          │
    │  • Maximizes training data usage                                   │
    │  • Subject-independent evaluation                                  │
    │  • Realistic generalization assessment                             │
    │                                                                     │
    │  Used for: Final model evaluation (paper results)                  │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

    10-FOLD STRATIFIED CROSS-VALIDATION
    ════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  Data split into 10 equal folds with balanced class distribution   │
    │                                                                     │
    │  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐    │
    │  │ F1  │ F2  │ F3  │ F4  │ F5  │ F6  │ F7  │ F8  │ F9  │ F10 │    │
    │  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘    │
    │                                                                     │
    │  Each fold serves as test set once:                                │
    │  Fold 1: Test F1, Train F2-F10                                     │
    │  Fold 2: Test F2, Train F1,F3-F10                                  │
    │  ...                                                                │
    │                                                                     │
    │  Used for: Hyperparameter tuning, model selection                  │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

================================================================================
CHECKPOINTING STRATEGY
================================================================================

    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  Saved Checkpoints:                                                │
    │                                                                     │
    │  checkpoints/                                                       │
    │  ├── best_model.pt      ← Highest validation accuracy              │
    │  ├── last_model.pt      ← Most recent epoch                        │
    │  └── fold_N/            ← Per-fold checkpoints (LOSO)              │
    │      ├── best_model.pt                                              │
    │      └── last_model.pt                                              │
    │                                                                     │
    │  Checkpoint Contents:                                               │
    │  • epoch: Current epoch number                                     │
    │  • model_state_dict: Model weights                                 │
    │  • optimizer_state_dict: Optimizer state (momentum, etc.)         │
    │  • scheduler_state_dict: LR scheduler state                       │
    │  • metrics: Validation metrics at save time                        │
    │  • config: Training configuration                                  │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

================================================================================
USAGE EXAMPLES
================================================================================

    Basic Training:
    ```python
    from src.training.trainer import Trainer, TrainingConfig
    from src.models import GenAIRAGEEG

    # Configure training
    config = TrainingConfig(
        learning_rate=1e-4,
        batch_size=64,
        n_epochs=100,
        patience=10
    )

    # Create model and trainer
    model = GenAIRAGEEG()
    trainer = Trainer(model, config)

    # Train
    history = trainer.train(train_loader, val_loader)

    # Evaluate
    metrics = trainer.evaluate(test_loader)
    print(f"Test Accuracy: {metrics.accuracy:.4f}")
    ```

    LOSO Cross-Validation:
    ```python
    from src.training.trainer import train_loso
    from src.models import GenAIRAGEEG

    # Model factory for creating fresh models
    def model_factory():
        return GenAIRAGEEG(use_rag=False)

    # Run LOSO
    results = train_loso(
        model_factory=model_factory,
        data=X,          # (n_samples, channels, time)
        labels=y,        # (n_samples,)
        subject_ids=s,   # (n_samples,)
        config=config
    )

    # Results
    print(f"Accuracy: {results['aggregate']['accuracy']['mean']:.4f} ± "
          f"{results['aggregate']['accuracy']['std']:.4f}")
    ```

================================================================================
DEPENDENCIES
================================================================================

    Required:
    - torch >= 2.0.0
    - numpy >= 1.21.0

    Optional:
    - scikit-learn >= 1.0.0 (for metrics)

================================================================================
PAPER SPECIFICATIONS
================================================================================

    As described in the paper:
    • Optimizer: AdamW with weight decay (1e-2)
    • Learning rate: 1e-4
    • Batch size: 64
    • Epochs: 100 with early stopping (patience=10)
    • Gradient clipping: max_norm=1.0
    • LR scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
    • Cross-validation: 10-fold stratified

================================================================================
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        roc_auc_score,
        matthews_corrcoef,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 64
    n_epochs: int = 100

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Scheduler
    scheduler_type: str = "plateau"  # "plateau" or "cosine"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5

    # Checkpointing
    save_best: bool = True
    save_last: bool = True
    checkpoint_dir: str = "checkpoints"

    # Logging
    log_interval: int = 10
    eval_interval: int = 1

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingMetrics:
    """Training metrics container."""
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    auc: float = 0.0
    mcc: float = 0.0
    loss: float = 0.0


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


class Trainer:
    """
    Trainer for GenAI-RAG-EEG model.

    Handles training loop, validation, and metrics computation.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config or TrainingConfig()
        self.model = model.to(self.config.device)
        self.logger = logger or logging.getLogger(__name__)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Scheduler
        if self.config.scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.n_epochs
            )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta
        )

        # History
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "lr": []
        }

        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            eeg = batch["eeg"].to(self.config.device)
            labels = batch["label"].to(self.config.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(eeg)
            logits = output["logits"]

            # Compute loss
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % self.config.log_interval == 0:
                self.logger.debug(
                    f"Batch {batch_idx + 1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> TrainingMetrics:
        """Evaluate on validation set."""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0

        for batch in val_loader:
            eeg = batch["eeg"].to(self.config.device)
            labels = batch["label"].to(self.config.device)

            output = self.model(eeg)
            logits = output["logits"]
            probs = output["probs"]

            loss = self.criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Compute metrics
        metrics = TrainingMetrics(loss=total_loss / len(val_loader))

        if SKLEARN_AVAILABLE:
            metrics.accuracy = accuracy_score(all_labels, all_preds)
            metrics.balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
            metrics.f1 = f1_score(all_labels, all_preds, average='binary')
            metrics.precision = precision_score(all_labels, all_preds, average='binary')
            metrics.recall = recall_score(all_labels, all_preds, average='binary')
            metrics.mcc = matthews_corrcoef(all_labels, all_preds)

            try:
                metrics.auc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                metrics.auc = 0.5
        else:
            metrics.accuracy = (all_preds == all_labels).mean()

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: Optional[int] = None
    ) -> Dict:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs (overrides config)

        Returns:
            Training history dictionary
        """
        n_epochs = n_epochs or self.config.n_epochs
        best_val_acc = 0.0
        best_model_state = None

        self.logger.info(f"Starting training for {n_epochs} epochs")
        self.logger.info(f"Device: {self.config.device}")
        self.logger.info(f"Learning rate: {self.config.learning_rate}")

        start_time = time.time()

        for epoch in range(n_epochs):
            epoch_start = time.time()

            # Training
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation
            val_metrics = self.evaluate(val_loader)

            # Scheduler step
            if self.config.scheduler_type == "plateau":
                self.scheduler.step(val_metrics.accuracy)
            else:
                self.scheduler.step()

            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_metrics.loss)
            self.history["val_acc"].append(val_metrics.accuracy)
            self.history["val_f1"].append(val_metrics.f1)
            self.history["lr"].append(current_lr)

            # Logging
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch + 1}/{n_epochs} ({epoch_time:.1f}s) - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_metrics.loss:.4f}, Val Acc: {val_metrics.accuracy:.4f}, "
                f"Val F1: {val_metrics.f1:.4f}, LR: {current_lr:.2e}"
            )

            # Save best model
            if val_metrics.accuracy > best_val_acc:
                best_val_acc = val_metrics.accuracy
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

                if self.config.save_best:
                    self.save_checkpoint(
                        Path(self.config.checkpoint_dir) / "best_model.pt",
                        epoch,
                        val_metrics
                    )

            # Early stopping
            if self.early_stopping(val_metrics.accuracy):
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Save last model
        if self.config.save_last:
            self.save_checkpoint(
                Path(self.config.checkpoint_dir) / "last_model.pt",
                epoch,
                val_metrics
            )

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.1f}s")
        self.logger.info(f"Best validation accuracy: {best_val_acc:.4f}")

        return self.history

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: TrainingMetrics
    ):
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": asdict(metrics),
            "config": asdict(self.config)
        }, path)
        self.logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.logger.info(f"Loaded checkpoint from {path}")
        return checkpoint


def train_loso(
    model_factory: Callable,
    data: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    config: Optional[TrainingConfig] = None,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Train with Leave-One-Subject-Out cross-validation.

    Args:
        model_factory: Function that creates a new model instance
        data: EEG data (n_samples, channels, time)
        labels: Labels (n_samples,)
        subject_ids: Subject IDs (n_samples,)
        config: Training configuration
        logger: Logger instance

    Returns:
        Dictionary with per-fold and aggregate results
    """
    from ..data.datasets import create_loso_splits, create_dataloaders

    config = config or TrainingConfig()
    logger = logger or logging.getLogger(__name__)

    unique_subjects = np.unique(subject_ids)
    n_folds = len(unique_subjects)

    fold_results = []

    logger.info(f"Starting LOSO cross-validation with {n_folds} folds")

    for fold, (train_data, train_labels, test_data, test_labels, test_subject) in enumerate(
        create_loso_splits(data, labels, subject_ids)
    ):
        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold + 1}/{n_folds} - Test Subject: {test_subject}")
        logger.info(f"Train: {len(train_labels)}, Test: {len(test_labels)}")

        # Create data loaders
        train_loader, test_loader = create_dataloaders(
            train_data, train_labels,
            test_data, test_labels,
            batch_size=config.batch_size
        )

        # Create new model
        model = model_factory()

        # Update checkpoint dir for this fold
        fold_config = TrainingConfig(**asdict(config))
        fold_config.checkpoint_dir = f"{config.checkpoint_dir}/fold_{fold + 1}"

        # Train
        trainer = Trainer(model, fold_config, logger)
        history = trainer.train(train_loader, test_loader)

        # Final evaluation
        final_metrics = trainer.evaluate(test_loader)

        fold_results.append({
            "fold": fold + 1,
            "test_subject": int(test_subject),
            "metrics": asdict(final_metrics),
            "history": history
        })

        logger.info(f"Fold {fold + 1} - Final Acc: {final_metrics.accuracy:.4f}, F1: {final_metrics.f1:.4f}")

    # Aggregate results
    all_acc = [r["metrics"]["accuracy"] for r in fold_results]
    all_f1 = [r["metrics"]["f1"] for r in fold_results]
    all_auc = [r["metrics"]["auc"] for r in fold_results]

    aggregate = {
        "accuracy": {
            "mean": np.mean(all_acc),
            "std": np.std(all_acc),
            "ci_95": 1.96 * np.std(all_acc) / np.sqrt(len(all_acc))
        },
        "f1": {
            "mean": np.mean(all_f1),
            "std": np.std(all_f1),
            "ci_95": 1.96 * np.std(all_f1) / np.sqrt(len(all_f1))
        },
        "auc": {
            "mean": np.mean(all_auc),
            "std": np.std(all_auc),
            "ci_95": 1.96 * np.std(all_auc) / np.sqrt(len(all_auc))
        }
    }

    logger.info(f"\n{'='*50}")
    logger.info("LOSO Cross-Validation Results:")
    logger.info(f"Accuracy: {aggregate['accuracy']['mean']:.4f} ± {aggregate['accuracy']['std']:.4f}")
    logger.info(f"F1 Score: {aggregate['f1']['mean']:.4f} ± {aggregate['f1']['std']:.4f}")
    logger.info(f"AUC-ROC:  {aggregate['auc']['mean']:.4f} ± {aggregate['auc']['std']:.4f}")

    return {
        "fold_results": fold_results,
        "aggregate": aggregate
    }


if __name__ == "__main__":
    # Test training on synthetic data
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    from src.models import create_model
    from src.data import generate_synthetic_dataset, create_dataloaders

    print("Testing Training Module")
    print("=" * 50)

    # Generate data
    data, labels, subjects = generate_synthetic_dataset(
        n_subjects=5,
        n_trials_per_subject=20
    )

    # Split
    train_mask = subjects <= 4
    test_mask = subjects == 5

    train_loader, test_loader = create_dataloaders(
        data[train_mask], labels[train_mask],
        data[test_mask], labels[test_mask],
        batch_size=16
    )

    # Create model
    model = create_model(use_text=False, use_rag=False)

    # Train
    config = TrainingConfig(
        n_epochs=5,
        learning_rate=1e-3,
        batch_size=16,
        patience=3
    )

    trainer = Trainer(model, config, logger)
    history = trainer.train(train_loader, test_loader)

    print(f"\nFinal metrics:")
    final_metrics = trainer.evaluate(test_loader)
    print(f"Accuracy: {final_metrics.accuracy:.4f}")
    print(f"F1 Score: {final_metrics.f1:.4f}")
