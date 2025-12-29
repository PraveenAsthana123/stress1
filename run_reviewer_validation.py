#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GenAI-RAG-EEG: Comprehensive Reviewer Validation Script
================================================================================

Title: Complete Validation Pipeline for Paper Review
Authors: GenAI-RAG-EEG Research Team
Version: 1.0.0
Date: December 2024

Description:
    This script performs comprehensive validation of the GenAI-RAG-EEG model
    for reviewer verification. It includes:
    - Full 5-fold cross-validation
    - Detailed logging with timestamps
    - Model checkpoint saving
    - Comprehensive metrics computation
    - Results export for reproducibility

Usage:
    python run_reviewer_validation.py

Output:
    - results/validation_YYYYMMDD_HHMMSS/
        - validation_log.txt
        - metrics_summary.json
        - confusion_matrices.png
        - training_curves.png
        - model_checkpoints/
        - predictions.csv

License: MIT
================================================================================
"""

import os
import sys
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# =============================================================================
# CONFIGURATION
# =============================================================================

class ValidationConfig:
    """Configuration for validation run."""

    # Data settings
    DATA_PATH = "data/SAM40/filtered_data"
    N_CHANNELS = 32
    SEGMENT_LENGTH = 512
    SAMPLING_RATE = 256

    # Training settings
    N_FOLDS = 5
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 15

    # Model settings
    DROPOUT = 0.3

    # Output settings
    RESULTS_BASE_DIR = "results"
    SAVE_CHECKPOINTS = True
    SAVE_PREDICTIONS = True


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(results_dir: Path) -> logging.Logger:
    """Setup comprehensive logging."""

    logger = logging.getLogger('GenAI-RAG-EEG-Validation')
    logger.setLevel(logging.DEBUG)

    # File handler - detailed logs
    fh = logging.FileHandler(results_dir / 'validation_log.txt')
    fh.setLevel(logging.DEBUG)

    # Console handler - summary logs
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Format with timestamps
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# =============================================================================
# DATA LOADING
# =============================================================================

def load_sam40_data(config: ValidationConfig, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load SAM-40 dataset from local directory."""

    import scipy.io as sio

    logger.info("=" * 70)
    logger.info("LOADING SAM-40 DATASET")
    logger.info("=" * 70)

    data_path = Path(config.DATA_PATH)

    if not data_path.exists():
        logger.error(f"Data path not found: {data_path}")
        raise FileNotFoundError(f"Data path not found: {data_path}")

    # Get all .mat files
    mat_files = sorted(data_path.glob("*.mat"))
    logger.info(f"Found {len(mat_files)} .mat files")

    all_data = []
    all_labels = []
    metadata = {
        "subjects": [],
        "conditions": [],
        "trials": [],
        "files_loaded": 0,
        "files_failed": 0,
        "file_list": []
    }

    # Stress conditions
    stress_conditions = ["Arithmetic", "Stroop", "Mirror_image"]

    for mat_file in mat_files:
        filename = mat_file.stem
        parts = filename.split('_')

        try:
            # Parse filename: Condition_sub_XX_trialY
            if len(parts) >= 4:
                condition = parts[0]
                if parts[1] == "image":  # Handle "Mirror_image"
                    condition = f"{parts[0]}_{parts[1]}"
                    subject_idx = int(parts[3])
                    trial_idx = int(parts[4].replace("trial", ""))
                else:
                    subject_idx = int(parts[2])
                    trial_idx = int(parts[3].replace("trial", ""))

                # Load data
                mat_data = sio.loadmat(str(mat_file))

                # Find EEG array
                eeg_data = None
                for key in mat_data.keys():
                    if not key.startswith('__'):
                        data = mat_data[key]
                        if isinstance(data, np.ndarray) and data.size > 100:
                            eeg_data = data
                            break

                if eeg_data is not None:
                    # Ensure shape is (channels, samples)
                    if eeg_data.shape[0] > eeg_data.shape[1]:
                        eeg_data = eeg_data.T

                    # Determine label
                    label = 1 if condition in stress_conditions else 0

                    all_data.append(eeg_data)
                    all_labels.append(label)
                    metadata["subjects"].append(subject_idx)
                    metadata["conditions"].append(condition)
                    metadata["trials"].append(trial_idx)
                    metadata["files_loaded"] += 1
                    metadata["file_list"].append(filename)
                else:
                    metadata["files_failed"] += 1

        except Exception as e:
            logger.warning(f"Error loading {filename}: {e}")
            metadata["files_failed"] += 1

    if len(all_data) == 0:
        raise ValueError("No data was loaded successfully")

    # Normalize shapes
    min_samples = min(d.shape[1] for d in all_data)
    max_channels = max(d.shape[0] for d in all_data)

    logger.info(f"Data dimensions: max_channels={max_channels}, min_samples={min_samples}")

    # Segment and normalize
    segment_length = config.SEGMENT_LENGTH
    segments = []
    segment_labels = []
    segment_metadata = {"subjects": [], "conditions": [], "trials": []}

    for i, (d, label) in enumerate(zip(all_data, all_labels)):
        # Truncate/pad channels
        if d.shape[0] < config.N_CHANNELS:
            padding = np.zeros((config.N_CHANNELS - d.shape[0], d.shape[1]))
            d = np.vstack([d, padding])
        elif d.shape[0] > config.N_CHANNELS:
            d = d[:config.N_CHANNELS, :]

        # Extract segments
        n_segs = d.shape[1] // segment_length
        for j in range(n_segs):
            start = j * segment_length
            end = start + segment_length
            seg = d[:, start:end]

            # Normalize segment
            seg = (seg - seg.mean()) / (seg.std() + 1e-8)

            segments.append(seg)
            segment_labels.append(label)
            segment_metadata["subjects"].append(metadata["subjects"][i])
            segment_metadata["conditions"].append(metadata["conditions"][i])
            segment_metadata["trials"].append(metadata["trials"][i])

    X = np.array(segments, dtype=np.float32)
    y = np.array(segment_labels, dtype=np.int64)

    # Log statistics
    logger.info(f"Loaded {metadata['files_loaded']} files, {metadata['files_failed']} failed")
    logger.info(f"Total segments: {len(X)}")
    logger.info(f"Data shape: {X.shape}")
    logger.info(f"Class distribution: Baseline={np.sum(y==0)}, Stress={np.sum(y==1)}")
    logger.info(f"Class balance: {np.sum(y==1)/len(y)*100:.1f}% stress")

    # Unique subjects and conditions
    unique_subjects = len(set(metadata["subjects"]))
    unique_conditions = set(metadata["conditions"])
    logger.info(f"Unique subjects: {unique_subjects}")
    logger.info(f"Conditions: {unique_conditions}")

    metadata["segment_metadata"] = segment_metadata
    metadata["shape"] = X.shape
    metadata["n_stress"] = int(np.sum(y == 1))
    metadata["n_baseline"] = int(np.sum(y == 0))

    return X, y, metadata


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ValidationConfig,
    logger: logging.Logger,
    fold: int
) -> Tuple[nn.Module, Dict]:
    """Train model for one fold."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "learning_rates": []
    }

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            logits = outputs['logits']
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(logits.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                logits = outputs['logits']
                loss = criterion(logits, batch_y)

                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(logits.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["learning_rates"].append(current_lr)

        # Check for best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Fold {fold+1} | Epoch {epoch+1:3d}/{config.EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}% | "
                f"LR: {current_lr:.6f}"
            )

        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    history["best_val_acc"] = best_val_acc
    history["final_epoch"] = epoch + 1

    return model, history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Comprehensive model evaluation."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = outputs['probs'].cpu().numpy()
            _, preds = torch.max(outputs['logits'], 1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1])
            all_labels.extend(batch_y.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }

    # ROC-AUC
    try:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    except:
        metrics["auc_roc"] = 0.5

    # Precision-Recall curve
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        metrics["pr_curve"] = {"precision": precision.tolist(), "recall": recall.tolist()}
    except:
        pass

    metrics["predictions"] = {
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist()
    }

    return metrics


# =============================================================================
# MAIN VALIDATION PIPELINE
# =============================================================================

def run_validation():
    """Run complete validation pipeline."""

    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/validation_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint directory
    checkpoint_dir = results_dir / "model_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Setup logging
    logger = setup_logging(results_dir)

    config = ValidationConfig()

    # Log start
    logger.info("=" * 70)
    logger.info("GENAI-RAG-EEG COMPREHENSIVE VALIDATION")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Random seed: {RANDOM_SEED}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Log configuration
    logger.info("\n--- Configuration ---")
    for attr in dir(config):
        if not attr.startswith('_'):
            logger.info(f"  {attr}: {getattr(config, attr)}")

    # Load data
    try:
        X, y, metadata = load_sam40_data(config, logger)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    # Save metadata
    with open(results_dir / "data_metadata.json", 'w') as f:
        # Convert non-serializable items
        save_meta = {k: v for k, v in metadata.items()
                     if k not in ['segment_metadata']}
        json.dump(save_meta, f, indent=2)

    # Import model
    from src.models.genai_rag_eeg import GenAIRAGEEG

    # Log model architecture
    logger.info("\n" + "=" * 70)
    logger.info("MODEL ARCHITECTURE")
    logger.info("=" * 70)

    sample_model = GenAIRAGEEG(
        n_channels=config.N_CHANNELS,
        n_time_samples=config.SEGMENT_LENGTH,
        dropout=config.DROPOUT
    )

    total_params = sum(p.numel() for p in sample_model.parameters())
    trainable_params = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # 5-Fold Cross-Validation
    logger.info("\n" + "=" * 70)
    logger.info(f"{config.N_FOLDS}-FOLD CROSS-VALIDATION")
    logger.info("=" * 70)

    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []
    all_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "auc_roc": []
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"\n--- Fold {fold+1}/{config.N_FOLDS} ---")
        logger.info(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")

        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        # Initialize model
        model = GenAIRAGEEG(
            n_channels=config.N_CHANNELS,
            n_time_samples=config.SEGMENT_LENGTH,
            dropout=config.DROPOUT
        )

        # Train
        model, history = train_fold(model, train_loader, val_loader, config, logger, fold)

        # Evaluate
        metrics = evaluate_model(model, val_loader, logger)

        # Log fold results
        logger.info(f"\nFold {fold+1} Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
        logger.info(f"  Precision: {metrics['precision']*100:.2f}%")
        logger.info(f"  Recall:    {metrics['recall']*100:.2f}%")
        logger.info(f"  F1-Score:  {metrics['f1_score']*100:.2f}%")
        logger.info(f"  AUC-ROC:   {metrics['auc_roc']*100:.2f}%")

        # Store results
        fold_results.append({
            "fold": fold + 1,
            "metrics": metrics,
            "history": history
        })

        for key in all_metrics:
            all_metrics[key].append(metrics[key])

        # Save checkpoint
        if config.SAVE_CHECKPOINTS:
            checkpoint_path = checkpoint_dir / f"model_fold{fold+1}.pt"
            torch.save({
                'fold': fold + 1,
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
                'history': history
            }, checkpoint_path)
            logger.info(f"  Checkpoint saved: {checkpoint_path}")

    # Compute summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("=" * 70)

    summary = {}
    for metric, values in all_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        summary[metric] = {
            "mean": float(mean_val),
            "std": float(std_val),
            "values": [float(v) for v in values]
        }
        logger.info(f"{metric.upper():12s}: {mean_val*100:.2f}% ± {std_val*100:.2f}%")

    # Create results table
    logger.info("\n--- Detailed Fold Results ---")
    logger.info(f"{'Fold':^6} | {'Accuracy':^10} | {'Precision':^10} | {'Recall':^10} | {'F1-Score':^10} | {'AUC-ROC':^10}")
    logger.info("-" * 70)
    for i, result in enumerate(fold_results):
        m = result['metrics']
        logger.info(f"{i+1:^6} | {m['accuracy']*100:>8.2f}% | {m['precision']*100:>8.2f}% | {m['recall']*100:>8.2f}% | {m['f1_score']*100:>8.2f}% | {m['auc_roc']*100:>8.2f}%")
    logger.info("-" * 70)
    logger.info(f"{'Mean':^6} | {summary['accuracy']['mean']*100:>8.2f}% | {summary['precision']['mean']*100:>8.2f}% | {summary['recall']['mean']*100:>8.2f}% | {summary['f1_score']['mean']*100:>8.2f}% | {summary['auc_roc']['mean']*100:>8.2f}%")
    logger.info(f"{'Std':^6} | {summary['accuracy']['std']*100:>8.2f}% | {summary['precision']['std']*100:>8.2f}% | {summary['recall']['std']*100:>8.2f}% | {summary['f1_score']['std']*100:>8.2f}% | {summary['auc_roc']['std']*100:>8.2f}%")

    # Compare with paper results
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON WITH PAPER-REPORTED RESULTS (SAM-40)")
    logger.info("=" * 70)

    paper_results = {
        "accuracy": 93.2,
        "precision": 92.8,
        "recall": 93.7,
        "f1_score": 93.2,
        "auc_roc": 95.8
    }

    logger.info(f"{'Metric':^12} | {'Paper':^12} | {'This Run':^12} | {'Difference':^12}")
    logger.info("-" * 55)
    for metric in paper_results:
        paper_val = paper_results[metric]
        run_val = summary[metric]['mean'] * 100
        diff = run_val - paper_val
        diff_str = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"
        logger.info(f"{metric:^12} | {paper_val:>10.1f}% | {run_val:>10.2f}% | {diff_str:>10}")

    # Save all results
    results_summary = {
        "timestamp": timestamp,
        "config": {attr: getattr(config, attr) for attr in dir(config) if not attr.startswith('_')},
        "data_info": {
            "n_samples": len(X),
            "n_channels": X.shape[1],
            "segment_length": X.shape[2],
            "n_stress": int(np.sum(y == 1)),
            "n_baseline": int(np.sum(y == 0))
        },
        "model_info": {
            "total_params": total_params,
            "trainable_params": trainable_params
        },
        "cv_summary": summary,
        "fold_results": [
            {
                "fold": r["fold"],
                "accuracy": r["metrics"]["accuracy"],
                "precision": r["metrics"]["precision"],
                "recall": r["metrics"]["recall"],
                "f1_score": r["metrics"]["f1_score"],
                "auc_roc": r["metrics"]["auc_roc"]
            }
            for r in fold_results
        ],
        "paper_comparison": paper_results
    }

    with open(results_dir / "validation_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"\n\nResults saved to: {results_dir}")
    logger.info("=" * 70)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 70)

    return results_summary


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" GENAI-RAG-EEG: Reviewer Validation Script")
    print("=" * 70)
    print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    try:
        results = run_validation()

        print("\n" + "=" * 70)
        print(" FINAL SUMMARY")
        print("=" * 70)
        print(f" Accuracy:  {results['cv_summary']['accuracy']['mean']*100:.2f}% ± {results['cv_summary']['accuracy']['std']*100:.2f}%")
        print(f" F1-Score:  {results['cv_summary']['f1_score']['mean']*100:.2f}% ± {results['cv_summary']['f1_score']['std']*100:.2f}%")
        print(f" AUC-ROC:   {results['cv_summary']['auc_roc']['mean']*100:.2f}% ± {results['cv_summary']['auc_roc']['std']*100:.2f}%")
        print("=" * 70)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
