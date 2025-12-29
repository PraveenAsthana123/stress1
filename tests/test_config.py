#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Configuration Tests for GenAI-RAG-EEG
================================================================================

Tests that verify configuration handling:
- Config instantiation
- Config validation
- Config merging/override
- Data format contract validation
- Path resolution across platforms

Run with: python -m pytest tests/test_config.py -v

Author: GenAI-RAG-EEG Team
Version: 3.0.0
License: MIT
================================================================================
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from dataclasses import asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestConfigInstantiation:
    """Test config instantiation."""

    def test_config_creates_successfully(self):
        """Config should instantiate without errors."""
        from src.config import Config
        config = Config()
        assert config is not None

    def test_config_has_all_sections(self):
        """Config should have all expected sections."""
        from src.config import Config
        config = Config()

        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'expected_results')
        assert hasattr(config, 'datasets')

    def test_model_config_complete(self):
        """Model config should have all components."""
        from src.config import Config
        config = Config()

        assert hasattr(config.model, 'cnn')
        assert hasattr(config.model, 'lstm')
        assert hasattr(config.model, 'attention')

    def test_training_config_complete(self):
        """Training config should have all parameters."""
        from src.config import Config
        config = Config()

        assert hasattr(config.training, 'learning_rate')
        assert hasattr(config.training, 'batch_size')
        assert hasattr(config.training, 'epochs')
        assert hasattr(config.training, 'dropout')


class TestConfigValues:
    """Test config values are valid."""

    def test_learning_rate_valid(self):
        """Learning rate should be positive and small."""
        from src.config import Config
        config = Config()
        assert 0 < config.training.learning_rate < 1
        assert config.training.learning_rate == 0.0001  # Default value

    def test_batch_size_valid(self):
        """Batch size should be positive integer."""
        from src.config import Config
        config = Config()
        assert config.training.batch_size > 0
        assert isinstance(config.training.batch_size, int)

    def test_epochs_valid(self):
        """Epochs should be positive integer."""
        from src.config import Config
        config = Config()
        assert config.training.epochs > 0
        assert isinstance(config.training.epochs, int)

    def test_dropout_valid(self):
        """Dropout should be between 0 and 1."""
        from src.config import Config
        config = Config()
        assert 0 <= config.training.dropout < 1

    def test_cnn_filters_valid(self):
        """CNN filters should be positive."""
        from src.config import Config
        config = Config()
        assert config.model.cnn.filters > 0

    def test_lstm_hidden_size_valid(self):
        """LSTM hidden size should be positive."""
        from src.config import Config
        config = Config()
        assert config.model.lstm.hidden_size > 0

    def test_attention_heads_valid(self):
        """Attention heads should be positive."""
        from src.config import Config
        config = Config()
        assert config.model.attention.n_heads > 0


class TestPathResolution:
    """Test path resolution is cross-platform."""

    def test_project_root_exists(self):
        """PROJECT_ROOT should exist."""
        from src.config import PROJECT_ROOT
        assert PROJECT_ROOT.exists()

    def test_project_root_is_absolute(self):
        """PROJECT_ROOT should be absolute path."""
        from src.config import PROJECT_ROOT
        assert PROJECT_ROOT.is_absolute()

    def test_data_dir_is_path(self):
        """DATA_DIR should be a Path object."""
        from src.config import DATA_DIR
        assert isinstance(DATA_DIR, Path)

    def test_paths_use_pathlib(self):
        """All paths should use pathlib, not strings."""
        from src.config import Config, PROJECT_ROOT, DATA_DIR
        assert isinstance(PROJECT_ROOT, Path)
        assert isinstance(DATA_DIR, Path)


class TestConfigValidation:
    """Test config validation functions."""

    def test_validate_config_returns_result(self):
        """validate_config should return validation result."""
        from src.config import Config, validate_config
        config = Config()
        result = validate_config(config)

        assert 'valid' in result or 'errors' in result or isinstance(result, dict)

    def test_print_validation_report_runs(self):
        """print_validation_report should run without error."""
        from src.config import Config, print_validation_report
        config = Config()
        # Should not raise
        print_validation_report(config)


class TestDataFormatContract:
    """Test data format contract validation."""

    def test_contract_exists(self):
        """DATA_FORMAT_CONTRACT should exist."""
        from src.config import DATA_FORMAT_CONTRACT
        assert DATA_FORMAT_CONTRACT is not None
        assert isinstance(DATA_FORMAT_CONTRACT, dict)

    def test_contract_has_required_keys(self):
        """Contract should have required format keys."""
        from src.config import DATA_FORMAT_CONTRACT
        assert 'eeg_data' in DATA_FORMAT_CONTRACT
        assert 'labels' in DATA_FORMAT_CONTRACT

    def test_validate_data_format_valid_data(self):
        """validate_data_format should accept valid data."""
        from src.config import validate_data_format

        X = np.random.randn(100, 32, 512).astype(np.float32)
        y = np.random.randint(0, 2, 100).astype(np.int64)

        result = validate_data_format(X, y)
        assert result['valid'] is True

    def test_validate_data_format_wrong_dtype_x(self):
        """validate_data_format should detect wrong X dtype."""
        from src.config import validate_data_format

        X = np.random.randn(100, 32, 512).astype(np.float64)  # Wrong: should be float32
        y = np.random.randint(0, 2, 100).astype(np.int64)

        result = validate_data_format(X, y)
        # Should flag dtype mismatch
        assert not result['valid'] or len(result.get('warnings', [])) > 0

    def test_validate_data_format_wrong_shape(self):
        """validate_data_format should detect wrong shape."""
        from src.config import validate_data_format

        X = np.random.randn(100, 32).astype(np.float32)  # Wrong: missing time dimension
        y = np.random.randint(0, 2, 100).astype(np.int64)

        result = validate_data_format(X, y)
        assert not result['valid']

    def test_validate_data_format_wrong_labels(self):
        """validate_data_format should detect invalid labels."""
        from src.config import validate_data_format

        X = np.random.randn(100, 32, 512).astype(np.float32)
        y = np.random.randint(0, 5, 100).astype(np.int64)  # Wrong: should be 0 or 1

        result = validate_data_format(X, y)
        assert not result['valid']

    def test_validate_data_format_mismatched_length(self):
        """validate_data_format should detect length mismatch."""
        from src.config import validate_data_format

        X = np.random.randn(100, 32, 512).astype(np.float32)
        y = np.random.randint(0, 2, 50).astype(np.int64)  # Wrong: length mismatch

        result = validate_data_format(X, y)
        assert not result['valid']


class TestDatasetConfigs:
    """Test dataset-specific configurations."""

    def test_sam40_config_exists(self):
        """SAM-40 config should exist."""
        from src.config import Config
        config = Config()
        assert hasattr(config.datasets, 'sam40')

    def test_wesad_config_exists(self):
        """WESAD config should exist."""
        from src.config import Config
        config = Config()
        assert hasattr(config.datasets, 'wesad')

    def test_eegmat_config_exists(self):
        """EEGMAT config should exist."""
        from src.config import Config
        config = Config()
        assert hasattr(config.datasets, 'eegmat')

    def test_dataset_channels_valid(self):
        """Dataset channel counts should be valid."""
        from src.config import Config
        config = Config()

        assert config.datasets.sam40.n_channels > 0
        assert config.datasets.wesad.n_channels > 0
        assert config.datasets.eegmat.n_channels > 0


class TestExpectedResults:
    """Test expected results configuration."""

    def test_expected_results_exist(self):
        """Expected results should be configured."""
        from src.config import Config
        config = Config()
        assert hasattr(config, 'expected_results')

    def test_accuracy_targets_valid(self):
        """Accuracy targets should be between 0 and 1."""
        from src.config import Config
        config = Config()

        assert 0 < config.expected_results.sam40_accuracy <= 1.0
        assert 0 < config.expected_results.wesad_accuracy <= 1.0
        assert 0 < config.expected_results.eegmat_accuracy <= 1.0


class TestConfigSerialization:
    """Test config can be serialized/deserialized."""

    def test_config_to_dict(self):
        """Config should be convertible to dict."""
        from src.config import Config
        config = Config()

        # Should be convertible to dict (for JSON serialization)
        config_dict = asdict(config)
        assert isinstance(config_dict, dict)
        assert 'model' in config_dict
        assert 'training' in config_dict

    def test_config_values_preserved(self):
        """Config values should be preserved in dict."""
        from src.config import Config
        config = Config()
        config_dict = asdict(config)

        assert config_dict['training']['learning_rate'] == config.training.learning_rate
        assert config_dict['training']['batch_size'] == config.training.batch_size


class TestConfigDefaults:
    """Test config defaults match documentation."""

    def test_default_learning_rate(self):
        """Default learning rate should be 0.0001."""
        from src.config import Config
        config = Config()
        assert config.training.learning_rate == 0.0001

    def test_default_batch_size(self):
        """Default batch size should be 64."""
        from src.config import Config
        config = Config()
        assert config.training.batch_size == 64

    def test_default_epochs(self):
        """Default epochs should be 100."""
        from src.config import Config
        config = Config()
        assert config.training.epochs == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
