"""
Data Processing Package for GenAI-RAG-EEG.

Contains:
- EEG preprocessing pipeline
- Dataset loaders (DEAP, SAM-40, EEGMAT)
- Data augmentation utilities
- LOSO cross-validation splits
"""

from .preprocessing import (
    EEGPreprocessor,
    PreprocessingConfig,
    BandpassFilter,
    NotchFilter,
    EpochSegmenter,
    Normalizer,
    ArtifactRejector,
    get_dataset_config,
    compute_psd_features
)

from .datasets import (
    EEGStressDataset,
    DEAPLoader,
    SAM40Loader,
    EEGMATLoader,
    DATASET_INFO,
    create_loso_splits,
    create_dataloaders,
    generate_synthetic_dataset
)

__all__ = [
    # Preprocessing
    "EEGPreprocessor",
    "PreprocessingConfig",
    "BandpassFilter",
    "NotchFilter",
    "EpochSegmenter",
    "Normalizer",
    "ArtifactRejector",
    "get_dataset_config",
    "compute_psd_features",
    # Datasets
    "EEGStressDataset",
    "DEAPLoader",
    "SAM40Loader",
    "EEGMATLoader",
    "DATASET_INFO",
    "create_loso_splits",
    "create_dataloaders",
    "generate_synthetic_dataset",
]
