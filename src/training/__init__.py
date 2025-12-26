"""
Training Package for GenAI-RAG-EEG.

Contains:
- Trainer class for model training
- LOSO cross-validation
- Metrics computation
- Early stopping and checkpointing
"""

from .trainer import (
    Trainer,
    TrainingConfig,
    TrainingMetrics,
    EarlyStopping,
    train_loso
)

__all__ = [
    "Trainer",
    "TrainingConfig",
    "TrainingMetrics",
    "EarlyStopping",
    "train_loso"
]
