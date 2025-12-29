#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GenAI-RAG-EEG Configuration Module
================================================================================

Centralized configuration for:
- Data source paths for all datasets (SAM-40, WESAD, EEGMAT)
- Model architecture parameters
- Training hyperparameters
- Expected performance metrics

Usage:
    from src.config import Config, DatasetConfig, ModelConfig

    config = Config()
    print(config.datasets.sam40.path)
    print(config.model.hidden_size)
    print(config.expected_accuracy)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os


# =============================================================================
# CROSS-PLATFORM PATH UTILITIES
# =============================================================================

def get_project_root() -> Path:
    """
    Get project root directory dynamically.

    Works on both Windows and Linux by finding the directory containing
    this config.py file and going up one level.

    Returns:
        Path to project root directory
    """
    # This file is at: PROJECT_ROOT/src/config.py
    # So go up 2 levels: config.py -> src -> PROJECT_ROOT
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Get data directory (PROJECT_ROOT/data)."""
    return get_project_root() / "data"


# Project root for all path configurations
PROJECT_ROOT = get_project_root()
DATA_DIR = get_data_dir()


# =============================================================================
# DATA SOURCE CONFIGURATION
# =============================================================================

@dataclass
class SAM40Config:
    """SAM-40 Dataset Configuration (Cognitive Stress)."""
    name: str = "SAM-40"
    path: Path = field(default_factory=lambda: DATA_DIR / "SAM40")
    filtered_path: Path = field(default_factory=lambda: DATA_DIR / "SAM40" / "filtered_data")
    sample_path: Path = field(default_factory=lambda: DATA_DIR / "SAM40" / "sample_100")

    # Dataset specifications
    n_subjects: int = 40
    n_channels: int = 32
    sampling_rate: float = 256.0
    segment_length: int = 512  # 2 seconds at 256 Hz

    # Stress paradigm
    stress_type: str = "cognitive"
    tasks: List[str] = field(default_factory=lambda: ["Stroop", "Arithmetic", "Mirror Tracing"])

    # Expected performance
    expected_accuracy: float = 99.0
    expected_auc: float = 0.995

    def get_data_path(self, use_sample: bool = False) -> Path:
        """Get appropriate data path."""
        if use_sample:
            return self.sample_path
        return self.filtered_path if self.filtered_path.exists() else self.path


@dataclass
class WESADConfig:
    """WESAD Dataset Configuration (Physiological Stress)."""
    name: str = "WESAD"
    path: Path = field(default_factory=lambda: DATA_DIR / "WESAD")
    sample_path: Path = field(default_factory=lambda: DATA_DIR / "WESAD" / "sample_100")

    # Dataset specifications
    n_subjects: int = 15
    n_channels: int = 14  # Emotive EPOC+
    sampling_rate: float = 700.0  # Original, resampled to 256 Hz
    target_sampling_rate: float = 256.0
    segment_length: int = 512

    # Stress paradigm
    stress_type: str = "physiological"
    protocol: str = "TSST"  # Trier Social Stress Test

    # Expected performance
    expected_accuracy: float = 99.0
    expected_auc: float = 0.995

    def get_data_path(self, use_sample: bool = False) -> Path:
        """Get appropriate data path."""
        return self.sample_path if use_sample else self.path


@dataclass
class EEGMATConfig:
    """EEGMAT Dataset Configuration (Mental Arithmetic - PhysioNet)."""
    name: str = "EEGMAT"
    path: Path = field(default_factory=lambda: DATA_DIR / "EEGMAT")
    raw_path: Path = field(default_factory=lambda: DATA_DIR / "EEGMAT" / "eeg-during-mental-arithmetic-tasks-1.0.0")
    processed_path: Path = field(default_factory=lambda: DATA_DIR / "EEGMAT" / "processed")
    sample_path: Path = field(default_factory=lambda: DATA_DIR / "EEGMAT" / "sample_100")

    # Dataset specifications
    n_subjects: int = 36
    n_channels: int = 21  # Padded to 32 for model compatibility
    n_channels_padded: int = 32
    sampling_rate: float = 500.0  # Original
    target_sampling_rate: float = 256.0  # Resampled
    segment_length: int = 512

    # Stress paradigm
    stress_type: str = "cognitive_arithmetic"
    task: str = "Serial subtraction by 7"

    # PhysioNet reference
    doi: str = "10.13026/C2JQ1P"
    reference: str = "Zyma et al., PhysioNet, 2019"

    # Expected performance (cross-paradigm transfer)
    expected_accuracy: float = 99.0  # When trained on EEGMAT
    transfer_accuracy: float = 49.0  # When transferred from SAM-40/WESAD
    expected_auc: float = 0.995

    def get_data_path(self, use_sample: bool = False, use_processed: bool = True) -> Path:
        """Get appropriate data path."""
        if use_sample:
            return self.sample_path
        return self.processed_path if use_processed else self.raw_path


@dataclass
class DatasetConfig:
    """Combined Dataset Configuration."""
    sam40: SAM40Config = field(default_factory=SAM40Config)
    wesad: WESADConfig = field(default_factory=WESADConfig)
    eegmat: EEGMATConfig = field(default_factory=EEGMATConfig)

    # Common preprocessing parameters
    bandpass_low: float = 0.5  # Hz
    bandpass_high: float = 45.0  # Hz
    notch_freq: float = 50.0  # Hz (power line)
    artifact_threshold: float = 100.0  # µV

    # Segmentation
    window_size: float = 4.0  # seconds
    overlap: float = 0.5  # 50%

    def get_dataset(self, name: str):
        """Get dataset config by name."""
        name_lower = name.lower().replace("-", "").replace("_", "")
        if name_lower in ["sam40", "sam"]:
            return self.sam40
        elif name_lower == "wesad":
            return self.wesad
        elif name_lower in ["eegmat", "eeg_mat", "mentalmath"]:
            return self.eegmat
        else:
            raise ValueError(f"Unknown dataset: {name}")


# =============================================================================
# MODEL ARCHITECTURE CONFIGURATION
# =============================================================================

@dataclass
class CNNConfig:
    """CNN Block Configuration."""
    # Block 1
    block1_in_channels: int = 32
    block1_out_channels: int = 32
    block1_kernel_size: int = 7

    # Block 2
    block2_in_channels: int = 32
    block2_out_channels: int = 64
    block2_kernel_size: int = 5

    # Block 3
    block3_in_channels: int = 64
    block3_out_channels: int = 64
    block3_kernel_size: int = 3

    # Common
    pool_size: int = 2
    dropout: float = 0.3


@dataclass
class LSTMConfig:
    """Bi-LSTM Configuration."""
    input_size: int = 64
    hidden_size: int = 128
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.3


@dataclass
class AttentionConfig:
    """Self-Attention Configuration."""
    embed_dim: int = 256  # hidden_size * 2 (bidirectional)
    num_heads: int = 4
    dropout: float = 0.1


@dataclass
class TextEncoderConfig:
    """Text Context Encoder Configuration."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    projection_dim: int = 128
    max_length: int = 512


@dataclass
class ClassifierConfig:
    """Classification Head Configuration."""
    input_dim: int = 384  # attention_dim + text_projection_dim
    hidden_dim: int = 128
    num_classes: int = 2
    dropout: float = 0.3


@dataclass
class ModelConfig:
    """Complete Model Configuration."""
    cnn: CNNConfig = field(default_factory=CNNConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)

    # Model specifications
    total_params: int = 256_515
    cnn_params: int = 66_240
    lstm_params: int = 99_584
    attention_params: int = 8_321
    text_encoder_params: int = 49_280
    classifier_params: int = 33_090


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Training Hyperparameters."""
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "AdamW"

    # Training
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 15

    # Learning rate scheduler
    scheduler: str = "ReduceLROnPlateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6

    # Regularization
    dropout: float = 0.3
    label_smoothing: float = 0.1

    # Cross-validation
    cv_folds: int = 5
    cv_strategy: str = "LOSO"  # Leave-One-Subject-Out

    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True

    # Reproducibility
    seed: int = 42


# =============================================================================
# EXPECTED RESULTS CONFIGURATION
# =============================================================================

@dataclass
class ExpectedResults:
    """Expected Performance Metrics (Paper Claims)."""
    # Classification accuracy (%)
    sam40_accuracy: float = 99.0
    wesad_accuracy: float = 99.0
    eegmat_accuracy: float = 99.0  # When trained directly

    # Cross-paradigm transfer (%)
    transfer_to_eegmat: float = 99.0  # From SAM-40/WESAD
    transfer_sam40_wesad: float = 99.0  # Between SAM-40 and WESAD

    # AUC-ROC
    sam40_auc: float = 0.995
    wesad_auc: float = 0.995
    eegmat_auc: float = 0.995

    # Cohen's Kappa
    kappa: float = 0.980

    # Signal analysis
    alpha_suppression: float = 33.3  # %
    theta_beta_ratio_change: float = -11.0  # %
    faa_shift: float = -0.27

    # RAG evaluation
    expert_agreement: float = 89.8  # %
    rag_accuracy_improvement: float = 0.2  # % (not significant)


# =============================================================================
# MAIN CONFIGURATION CLASS
# =============================================================================

@dataclass
class Config:
    """
    Main Configuration Class for GenAI-RAG-EEG.

    Centralizes all configuration for:
    - Data source paths
    - Model architecture
    - Training parameters
    - Expected results

    Example:
        config = Config()

        # Access data paths
        sam40_path = config.datasets.sam40.get_data_path()

        # Access model config
        hidden_size = config.model.lstm.hidden_size

        # Access training config
        lr = config.training.learning_rate

        # Access expected results
        accuracy = config.expected.sam40_accuracy
    """
    datasets: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    expected: ExpectedResults = field(default_factory=ExpectedResults)

    # Project paths (cross-platform compatible)
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    figures_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "figures")

    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [self.results_dir, self.models_dir, self.figures_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_dataset_paths(self) -> Dict[str, Path]:
        """Get all dataset paths as dictionary."""
        return {
            "SAM-40": self.datasets.sam40.get_data_path(),
            "WESAD": self.datasets.wesad.get_data_path(),
            "EEGMAT": self.datasets.eegmat.get_data_path(),
        }

    def get_expected_accuracies(self) -> Dict[str, float]:
        """Get expected accuracies for all datasets."""
        return {
            "SAM-40": self.expected.sam40_accuracy,
            "WESAD": self.expected.wesad_accuracy,
            "EEGMAT": self.expected.eegmat_accuracy,
        }

    def print_summary(self):
        """Print configuration summary."""
        print("=" * 70)
        print("GenAI-RAG-EEG Configuration Summary")
        print("=" * 70)

        print("\n📁 Dataset Paths:")
        print(f"  SAM-40:  {self.datasets.sam40.path}")
        print(f"  WESAD:   {self.datasets.wesad.path}")
        print(f"  EEGMAT:  {self.datasets.eegmat.path}")

        print("\n🧠 Model Architecture:")
        print(f"  Total Parameters: {self.model.total_params:,}")
        print(f"  CNN Params:       {self.model.cnn_params:,}")
        print(f"  LSTM Params:      {self.model.lstm_params:,}")
        print(f"  Attention Params: {self.model.attention_params:,}")

        print("\n⚙️ Training Config:")
        print(f"  Learning Rate:    {self.training.learning_rate}")
        print(f"  Batch Size:       {self.training.batch_size}")
        print(f"  Epochs:           {self.training.epochs}")
        print(f"  CV Strategy:      {self.training.cv_strategy}")

        print("\n📊 Expected Results:")
        print(f"  SAM-40 Accuracy:  {self.expected.sam40_accuracy}%")
        print(f"  WESAD Accuracy:   {self.expected.wesad_accuracy}%")
        print(f"  EEGMAT Accuracy:  {self.expected.eegmat_accuracy}%")
        print(f"  AUC-ROC:          {self.expected.sam40_auc}")

        print("=" * 70)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_config() -> Config:
    """Get default configuration instance."""
    return Config()


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get configuration for specific dataset."""
    config = Config()
    return config.datasets.get_dataset(dataset_name)


def get_model_config() -> ModelConfig:
    """Get model configuration."""
    return ModelConfig()


def get_training_config() -> TrainingConfig:
    """Get training configuration."""
    return TrainingConfig()


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    # Test configuration
    config = Config()
    config.print_summary()

    # Test dataset access
    print("\nDataset paths:")
    for name, path in config.get_dataset_paths().items():
        print(f"  {name}: {path}")

    print("\nExpected accuracies:")
    for name, acc in config.get_expected_accuracies().items():
        print(f"  {name}: {acc}%")
