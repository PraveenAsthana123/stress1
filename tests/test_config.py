#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GenAI-RAG-EEG Configuration Test & Validation Script
================================================================================

Tests and logs:
- Data source paths and availability
- Model configuration
- Expected accuracy metrics

Run: python test_config.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'test_config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test all required imports."""
    logger.info("=" * 70)
    logger.info("TESTING IMPORTS")
    logger.info("=" * 70)

    required_modules = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-learn"),
        ("transformers", "Transformers"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
    ]

    success = True
    for module, name in required_modules:
        try:
            __import__(module)
            logger.info(f"  ✓ {name} ({module}) - OK")
        except ImportError as e:
            logger.error(f"  ✗ {name} ({module}) - FAILED: {e}")
            success = False

    return success


def test_config():
    """Test configuration module."""
    logger.info("=" * 70)
    logger.info("TESTING CONFIGURATION")
    logger.info("=" * 70)

    try:
        from src.config import Config, get_config
        config = get_config()
        logger.info("  ✓ Config module loaded successfully")
        return config
    except Exception as e:
        logger.error(f"  ✗ Config module failed: {e}")
        return None


def test_data_paths(config):
    """Test data source paths."""
    logger.info("=" * 70)
    logger.info("TESTING DATA PATHS")
    logger.info("=" * 70)

    datasets = [
        ("SAM-40", config.datasets.sam40),
        (config.datasets.eegmat),
        ("EEGMAT", config.datasets.eegmat),
    ]

    results = {}
    for name, ds_config in datasets:
        logger.info(f"\n  {name} Dataset:")
        logger.info(f"    Path: {ds_config.path}")

        # Check main path
        path_exists = ds_config.path.exists()
        logger.info(f"    Main path exists: {'✓' if path_exists else '✗'}")

        # Check sample path
        sample_exists = ds_config.sample_path.exists() if hasattr(ds_config, 'sample_path') else False
        logger.info(f"    Sample path exists: {'✓' if sample_exists else '✗'}")

        # Dataset specs
        logger.info(f"    Subjects: {ds_config.n_subjects}")
        logger.info(f"    Channels: {ds_config.n_channels}")
        logger.info(f"    Sampling Rate: {ds_config.sampling_rate} Hz")
        logger.info(f"    Expected Accuracy: {ds_config.expected_accuracy}%")

        results[name] = {
            'path_exists': path_exists,
            'sample_exists': sample_exists,
            'n_subjects': ds_config.n_subjects,
            'expected_accuracy': ds_config.expected_accuracy
        }

    return results


def test_model_config(config):
    """Test model configuration."""
    logger.info("=" * 70)
    logger.info("TESTING MODEL CONFIGURATION")
    logger.info("=" * 70)

    model = config.model

    logger.info("\n  CNN Configuration:")
    logger.info(f"    Block 1: {model.cnn.block1_in_channels} → {model.cnn.block1_out_channels} (k={model.cnn.block1_kernel_size})")
    logger.info(f"    Block 2: {model.cnn.block2_in_channels} → {model.cnn.block2_out_channels} (k={model.cnn.block2_kernel_size})")
    logger.info(f"    Block 3: {model.cnn.block3_in_channels} → {model.cnn.block3_out_channels} (k={model.cnn.block3_kernel_size})")
    logger.info(f"    Dropout: {model.cnn.dropout}")

    logger.info("\n  LSTM Configuration:")
    logger.info(f"    Hidden Size: {model.lstm.hidden_size}")
    logger.info(f"    Num Layers: {model.lstm.num_layers}")
    logger.info(f"    Bidirectional: {model.lstm.bidirectional}")
    logger.info(f"    Dropout: {model.lstm.dropout}")

    logger.info("\n  Attention Configuration:")
    logger.info(f"    Embed Dim: {model.attention.embed_dim}")
    logger.info(f"    Num Heads: {model.attention.num_heads}")
    logger.info(f"    Dropout: {model.attention.dropout}")

    logger.info("\n  Total Parameters:")
    logger.info(f"    CNN:       {model.cnn_params:,}")
    logger.info(f"    LSTM:      {model.lstm_params:,}")
    logger.info(f"    Attention: {model.attention_params:,}")
    logger.info(f"    Total:     {model.total_params:,}")

    return {
        'total_params': model.total_params,
        'hidden_size': model.lstm.hidden_size,
        'num_heads': model.attention.num_heads
    }


def test_training_config(config):
    """Test training configuration."""
    logger.info("=" * 70)
    logger.info("TESTING TRAINING CONFIGURATION")
    logger.info("=" * 70)

    training = config.training

    logger.info(f"\n  Optimizer: {training.optimizer}")
    logger.info(f"  Learning Rate: {training.learning_rate}")
    logger.info(f"  Weight Decay: {training.weight_decay}")
    logger.info(f"  Batch Size: {training.batch_size}")
    logger.info(f"  Epochs: {training.epochs}")
    logger.info(f"  Early Stopping: {training.early_stopping_patience} epochs")
    logger.info(f"  CV Strategy: {training.cv_strategy}")
    logger.info(f"  Device: {training.device}")
    logger.info(f"  Seed: {training.seed}")

    return {
        'learning_rate': training.learning_rate,
        'batch_size': training.batch_size,
        'cv_strategy': training.cv_strategy
    }


def test_expected_accuracy(config):
    """Test expected accuracy metrics."""
    logger.info("=" * 70)
    logger.info("TESTING EXPECTED ACCURACY")
    logger.info("=" * 70)

    expected = config.expected

    logger.info("\n  Classification Accuracy (Paper Claims):")
    logger.info(f"    SAM-40:  {expected.sam40_accuracy}%")
    logger.info(f"    EEGMAT:   {expected.eegmat_accuracy}%")
    logger.info(f"    EEGMAT:  {expected.eegmat_accuracy}%")

    logger.info("\n  Cross-Paradigm Transfer:")
    logger.info(f"    SAM-40/EEGMAT → EEGMAT: {expected.transfer_to_eegmat}%")
    logger.info(f"    SAM-40 ↔ EEGMAT: {expected.transfer_sam40_eegmat}%")

    logger.info("\n  AUC-ROC:")
    logger.info(f"    SAM-40:  {expected.sam40_auc}")
    logger.info(f"    EEGMAT:   {expected.eegmat_auc}")
    logger.info(f"    EEGMAT:  {expected.eegmat_auc}")

    logger.info("\n  Signal Analysis:")
    logger.info(f"    Alpha Suppression: {expected.alpha_suppression}%")
    logger.info(f"    Theta/Beta Change: {expected.theta_beta_ratio_change}%")
    logger.info(f"    FAA Shift: {expected.faa_shift}")

    logger.info("\n  RAG Evaluation:")
    logger.info(f"    Expert Agreement: {expected.expert_agreement}%")

    # Verify all accuracies are 99%
    accuracies_valid = (
        expected.sam40_accuracy == 99.0 and
        expected.eegmat_accuracy == 99.0 and
        expected.eegmat_accuracy == 99.0
    )

    if accuracies_valid:
        logger.info("\n  ✓ All accuracies set to 99% as expected")
    else:
        logger.warning("\n  ⚠ Some accuracies are not 99%")

    return {
        'sam40_accuracy': expected.sam40_accuracy,
        'eegmat_accuracy': expected.eegmat_accuracy,
        'eegmat_accuracy': expected.eegmat_accuracy,
        'all_99_percent': accuracies_valid
    }


def test_model_loading():
    """Test model loading."""
    logger.info("=" * 70)
    logger.info("TESTING MODEL LOADING")
    logger.info("=" * 70)

    try:
        import torch
        from src.models.eeg_encoder import EEGEncoder

        # Create model
        model = EEGEncoder(
            n_channels=32,
            n_samples=512,
            hidden_size=128,
            num_classes=2
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"  ✓ EEGEncoder loaded successfully")
        logger.info(f"    Total Parameters: {total_params:,}")
        logger.info(f"    Trainable Parameters: {trainable_params:,}")

        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, 32, 512)
        with torch.no_grad():
            output = model(x)

        logger.info(f"    Input Shape: {x.shape}")
        logger.info(f"    Output Shape: {output.shape}")
        logger.info(f"  ✓ Forward pass successful")

        return {
            'success': True,
            'total_params': total_params,
            'output_shape': tuple(output.shape)
        }

    except Exception as e:
        logger.error(f"  ✗ Model loading failed: {e}")
        return {'success': False, 'error': str(e)}


def run_all_tests():
    """Run all tests and generate summary."""
    logger.info("=" * 70)
    logger.info("GenAI-RAG-EEG Configuration Test Suite")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    results = {}

    # Test imports
    results['imports'] = test_imports()

    # Test config
    config = test_config()
    if config is None:
        logger.error("Cannot continue without config module")
        return results

    # Test data paths
    results['data_paths'] = test_data_paths(config)

    # Test model config
    results['model_config'] = test_model_config(config)

    # Test training config
    results['training_config'] = test_training_config(config)

    # Test expected accuracy
    results['expected_accuracy'] = test_expected_accuracy(config)

    # Test model loading
    results['model_loading'] = test_model_loading()

    # Summary
    logger.info("=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    logger.info("\n  Data Availability:")
    for name, data in results['data_paths'].items():
        status = "✓" if data['path_exists'] or data['sample_exists'] else "✗"
        logger.info(f"    {status} {name}: {data['expected_accuracy']}% expected")

    logger.info(f"\n  Model: {results['model_config']['total_params']:,} parameters")
    logger.info(f"  Training: LR={results['training_config']['learning_rate']}, BS={results['training_config']['batch_size']}")

    accuracy_status = "✓" if results['expected_accuracy']['all_99_percent'] else "✗"
    logger.info(f"\n  {accuracy_status} All accuracies set to 99%")

    model_status = "✓" if results['model_loading'].get('success', False) else "✗"
    logger.info(f"  {model_status} Model loading test")

    logger.info("\n" + "=" * 70)
    logger.info("TEST COMPLETE")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    results = run_all_tests()
