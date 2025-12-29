#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GenAI-RAG-EEG: Main Entry Point
================================================================================

Project: GenAI-RAG-EEG for Stress Classification
Authors: Praveen Asthana, Rajveer Singh Lalawat, Sarita Singh Gond
License: MIT
Python: >= 3.8

================================================================================
OVERVIEW
================================================================================

This is the main entry point for the GenAI-RAG-EEG system. It provides a
command-line interface for training, evaluating, and running demonstrations
of the EEG-based stress classification model.

================================================================================
SYSTEM ARCHITECTURE
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         GenAI-RAG-EEG SYSTEM                            │
    └─────────────────────────────────────────────────────────────────────────┘

                              ┌────────────────┐
                              │    main.py     │
                              │  (Entry Point) │
                              └───────┬────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │   TRAIN MODE    │     │  EVALUATE MODE  │     │    DEMO MODE    │
    │                 │     │                 │     │                 │
    │ • Load data     │     │ • Load model    │     │ • Quick demo    │
    │ • Create model  │     │ • Run inference │     │ • Single sample │
    │ • LOSO CV       │     │ • Compute metrics│    │ • Explanation   │
    │ • Save results  │     │ • Generate report│    │ • Visualization │
    └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                           CORE MODULES                                  │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
    │  │   Models    │  │    Data     │  │  Training   │  │     RAG     │   │
    │  │ (EEG+Text)  │  │ (Preprocess)│  │  (Trainer)  │  │ (Pipeline)  │   │
    │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
USAGE EXAMPLES
================================================================================

    1. TRAINING WITH SYNTHETIC DATA (for testing):
       ─────────────────────────────────────────────
       $ python main.py --mode train --synthetic

       This generates synthetic EEG data and trains the model. Useful for
       verifying the installation and testing the pipeline.


    2. TRAINING WITH REAL DATASET:
       ────────────────────────────
       $ python main.py --mode train --dataset sam40 --config config.yaml

       Supported datasets:
       • DEAP: 32 participants, emotion/stress labels
       • SAM-40: 40 participants, stress induction protocol
       • EEGMAT: Mental arithmetic stress dataset

       Note: Dataset paths must be configured in config.yaml


    3. EVALUATION OF TRAINED MODEL:
       ─────────────────────────────
       $ python main.py --mode evaluate --checkpoint checkpoints/best_model.pt

       Loads a saved checkpoint and computes evaluation metrics on test data.


    4. INTERACTIVE DEMO:
       ──────────────────
       $ python main.py --mode demo

       Runs a quick demonstration with random EEG data, showing:
       • Model architecture
       • Prediction with confidence
       • Natural language explanation via RAG


    5. DEMO WITH CUSTOM INPUT:
       ────────────────────────
       $ python main.py --mode demo --input path/to/eeg_data.npy

       Input format: NumPy array with shape (batch, channels, samples)
       Example: (1, 32, 512) for single sample


    6. DEBUG MODE:
       ────────────
       $ python main.py --mode train --synthetic --log-level DEBUG

       Enables verbose logging for debugging purposes.

================================================================================
CONFIGURATION
================================================================================

    The system is configured via config.yaml:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ experiment:                                                             │
    │   name: "stress_classification"                                         │
    │   seed: 42                                                              │
    │   results_dir: "results/"                                               │
    │                                                                         │
    │ model:                                                                  │
    │   n_channels: 32                                                        │
    │   n_time_samples: 512                                                   │
    │   n_classes: 2                                                          │
    │   eeg_encoder:                                                          │
    │     hidden_dim: 128                                                     │
    │   text_encoder:                                                         │
    │     enabled: true                                                       │
    │     model_name: "all-MiniLM-L6-v2"                                      │
    │   rag:                                                                  │
    │     enabled: true                                                       │
    │                                                                         │
    │ training:                                                               │
    │   learning_rate: 1e-4                                                   │
    │   batch_size: 64                                                        │
    │   n_epochs: 100                                                         │
    │   patience: 10                                                          │
    │                                                                         │
    │ data:                                                                   │
    │   deap_path: "data/DEAP/"                                               │
    │   sam40_path: "data/SAM40/"                                             │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
OUTPUT FILES
================================================================================

    Training produces the following outputs:

    project_root/
    ├── checkpoints/
    │   ├── best_model.pt          # Best validation accuracy
    │   ├── last_model.pt          # Last epoch
    │   └── fold_N/                # Per-fold checkpoints (LOSO)
    │       ├── best_model.pt
    │       └── last_model.pt
    │
    ├── results/
    │   └── loso_results.json      # Cross-validation results
    │
    └── logs/
        └── training.log           # Training logs

================================================================================
QUICK START
================================================================================

    1. Install dependencies:
       $ pip install -r requirements.txt

    2. Run demo to verify installation:
       $ python main.py --mode demo

    3. Train on synthetic data:
       $ python main.py --mode train --synthetic

    4. (Optional) Configure real dataset paths and train:
       $ python main.py --mode train --dataset sam40

================================================================================
DEPENDENCIES
================================================================================

    Required:
    - Python >= 3.8
    - PyTorch >= 2.0.0
    - NumPy >= 1.21.0
    - PyYAML >= 6.0

    Optional:
    - transformers (for text encoder)
    - sentence-transformers (for BERT)
    - scikit-learn (for metrics)
    - faiss-cpu (for RAG)
    - openai (for LLM explanations)

================================================================================
"""

import argparse
import logging
import sys
import platform
from pathlib import Path
from typing import Optional
from datetime import datetime
import yaml

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import custom logger for detailed CLI output
try:
    from src.utils.logger import setup_logger, GenAILogger
    from src.utils.compatibility import CompatibilityLayer, check_system_requirements
    CUSTOM_LOGGER_AVAILABLE = True
except ImportError:
    CUSTOM_LOGGER_AVAILABLE = False

from src.models import create_model, create_context_string
from src.data import (
    generate_synthetic_dataset,
    create_loso_splits,
    create_dataloaders,
    DATASET_INFO
)
from src.training import Trainer, TrainingConfig, train_loso


def print_startup_banner():
    """Print detailed startup banner with system information."""
    print("\n" + "=" * 70)
    print("  ╔═══════════════════════════════════════════════════════════════╗")
    print("  ║           GenAI-RAG-EEG Stress Classification System          ║")
    print("  ║          Hybrid Deep Learning with RAG Explanations           ║")
    print("  ╚═══════════════════════════════════════════════════════════════╝")
    print("=" * 70)

    print(f"\n  System Information:")
    print(f"    Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    Platform: {platform.system()} {platform.release()}")
    print(f"    Python: {platform.python_version()}")
    print(f"    PyTorch: {torch.__version__}")
    print(f"    NumPy: {np.__version__}")

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"    GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print(f"    GPU: Not available (using CPU)")

    print("=" * 70)


def setup_logging(log_dir: str = "logs", level: int = logging.INFO, verbose: bool = True) -> logging.Logger:
    """
    Setup logging configuration with detailed CLI output.

    Args:
        log_dir: Directory for log files
        level: Logging level
        verbose: Enable verbose CLI output

    Returns:
        Logger instance
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Use custom logger if available
    if CUSTOM_LOGGER_AVAILABLE and verbose:
        logger = setup_logger(
            name="genai_rag_eeg",
            log_file=Path(log_dir) / "training.log",
            console_level="DEBUG" if level == logging.DEBUG else "INFO"
        )
        return logger

    # Fallback to standard logging
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/training.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(
    config: dict,
    dataset: str = "sam40",
    use_synthetic: bool = True,
    logger: Optional[logging.Logger] = None
):
    """
    Train GenAI-RAG-EEG model.

    Args:
        config: Configuration dictionary
        dataset: Dataset to use ("deap", "sam40", "eegmat")
        use_synthetic: Use synthetic data for testing
        logger: Logger instance
    """
    logger = logger or logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("GenAI-RAG-EEG Training")
    logger.info("=" * 60)

    # Set seed
    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")

    # Get device
    device = config.get("hardware", {}).get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load data
    if use_synthetic:
        logger.info("Using synthetic dataset for demonstration")
        data, labels, subjects = generate_synthetic_dataset(
            n_subjects=10,
            n_trials_per_subject=20,
            n_channels=config.get("model", {}).get("n_channels", 32),
            n_time_samples=config.get("model", {}).get("n_time_samples", 512)
        )
    else:
        # Load real dataset
        dataset_info = DATASET_INFO.get(dataset.lower())
        if dataset_info is None:
            raise ValueError(f"Unknown dataset: {dataset}")

        logger.info(f"Loading {dataset_info.name} dataset...")
        # Implement actual dataset loading here
        raise NotImplementedError(
            f"Real dataset loading not implemented. "
            f"Please place {dataset_info.name} data in {config['data'][f'{dataset.lower()}_path']}"
        )

    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Labels: {np.bincount(labels)}")
    logger.info(f"Subjects: {len(np.unique(subjects))}")

    # Model factory
    model_config = config.get("model", {})

    def create_model_fn():
        return create_model(
            n_channels=model_config.get("n_channels", 32),
            n_time_samples=model_config.get("n_time_samples", 512),
            n_classes=model_config.get("n_classes", 2),
            use_text=model_config.get("text_encoder", {}).get("enabled", True),
            use_rag=model_config.get("rag", {}).get("enabled", True),
            device=device
        )

    # Training config
    train_config = config.get("training", {})
    training_config = TrainingConfig(
        learning_rate=train_config.get("learning_rate", 1e-4),
        weight_decay=train_config.get("weight_decay", 1e-2),
        batch_size=train_config.get("batch_size", 64),
        n_epochs=train_config.get("n_epochs", 100),
        patience=train_config.get("patience", 10),
        checkpoint_dir=train_config.get("checkpoint_dir", "checkpoints"),
        device=device
    )

    # Run LOSO cross-validation
    results = train_loso(
        model_factory=create_model_fn,
        data=data,
        labels=labels,
        subject_ids=subjects,
        config=training_config,
        logger=logger
    )

    # Save results
    results_dir = Path(config.get("experiment", {}).get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    import json
    with open(results_dir / "loso_results.json", 'w') as f:
        json.dump({
            "aggregate": results["aggregate"],
            "fold_results": [
                {
                    "fold": r["fold"],
                    "test_subject": r["test_subject"],
                    "metrics": r["metrics"]
                }
                for r in results["fold_results"]
            ]
        }, f, indent=2)

    logger.info(f"Results saved to {results_dir / 'loso_results.json'}")

    return results


def evaluate(
    checkpoint_path: str,
    config: dict,
    logger: Optional[logging.Logger] = None
):
    """
    Evaluate a trained model.

    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        logger: Logger instance
    """
    logger = logger or logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("GenAI-RAG-EEG Evaluation")
    logger.info("=" * 60)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # Create model
    model_config = config.get("model", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model(
        n_channels=model_config.get("n_channels", 32),
        n_time_samples=model_config.get("n_time_samples", 512),
        n_classes=model_config.get("n_classes", 2),
        use_text=model_config.get("text_encoder", {}).get("enabled", True),
        use_rag=model_config.get("rag", {}).get("enabled", True),
        device=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info("Model loaded successfully")
    logger.info(f"Checkpoint metrics: {checkpoint.get('metrics', {})}")

    return model


def demo(
    config: dict,
    input_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None
):
    """
    Run a demonstration of the model.

    Args:
        config: Configuration dictionary
        input_path: Path to input EEG data (optional)
        logger: Logger instance
    """
    logger = logger or logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("GenAI-RAG-EEG Demo")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    model = create_model(
        n_channels=32,
        n_time_samples=512,
        n_classes=2,
        use_text=True,
        use_rag=True,
        device=device
    )

    model.eval()

    # Print model info
    param_counts = model.get_parameter_count()
    logger.info("\nModel Architecture:")
    for component, count in param_counts.items():
        logger.info(f"  {component}: {count:,} parameters")

    # Generate or load sample data
    if input_path:
        logger.info(f"\nLoading EEG data from {input_path}")
        eeg = np.load(input_path)
    else:
        logger.info("\nGenerating sample EEG data...")
        eeg = np.random.randn(1, 32, 512).astype(np.float32)

    eeg_tensor = torch.FloatTensor(eeg).to(device)

    # Context
    context = [create_context_string("Stroop", 25, "M")]

    # Run prediction with explanation
    logger.info("\nRunning prediction with explanation...")

    with torch.no_grad():
        explanations = model.predict_with_explanation(
            eeg_tensor,
            context,
            eeg_features={
                "alpha_power": 0.32,
                "beta_power": 0.71,
                "theta_power": 0.58,
                "frontal_asymmetry": 0.15
            }
        )

    # Display results
    result = explanations[0]

    logger.info("\n" + "=" * 60)
    logger.info("PREDICTION RESULT")
    logger.info("=" * 60)
    logger.info(f"Prediction: {result['prediction_label']}")
    logger.info(f"Confidence: {result['confidence']:.1%}")
    logger.info("\nExplanation:")
    logger.info("-" * 40)
    logger.info(result['explanation'])
    logger.info("=" * 60)

    return result


def main():
    """Main entry point with detailed CLI output."""
    parser = argparse.ArgumentParser(
        description="GenAI-RAG-EEG: Explainable EEG Stress Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train on synthetic data:
    python main.py --mode train --synthetic

  Train on SAM-40 dataset:
    python main.py --mode train --dataset sam40

  Evaluate checkpoint:
    python main.py --mode evaluate --checkpoint checkpoints/best_model.pt

  Run demo:
    python main.py --mode demo

  Verbose output:
    python main.py --mode train --synthetic --verbose
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "demo"],
        default="demo",
        help="Operation mode"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["deap", "sam40", "eegmat"],
        default="sam40",
        help="Dataset to use for training"
    )

    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for testing"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint for evaluation"
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Path to input EEG data for demo"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Enable verbose CLI output (default: True)"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Disable verbose output (minimal logging)"
    )

    args = parser.parse_args()

    # Determine verbosity
    verbose = args.verbose and not args.quiet

    # Print startup banner
    if verbose:
        print_startup_banner()
        print(f"\n  Mode: {args.mode.upper()}")
        print(f"  Log Level: {args.log_level}")
        if args.mode == "train":
            print(f"  Dataset: {args.dataset if not args.synthetic else 'Synthetic'}")
        print()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(level=log_level, verbose=verbose)

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = {}

    # Run selected mode
    try:
        if args.mode == "train":
            train(
                config=config,
                dataset=args.dataset,
                use_synthetic=args.synthetic,
                logger=logger
            )

        elif args.mode == "evaluate":
            if not args.checkpoint:
                logger.error("Checkpoint path required for evaluation")
                sys.exit(1)
            evaluate(
                checkpoint_path=args.checkpoint,
                config=config,
                logger=logger
            )

        elif args.mode == "demo":
            demo(
                config=config,
                input_path=args.input,
                logger=logger
            )

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.exception(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
