#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Smoke Test Suite for GenAI-RAG-EEG
================================================================================

Fast smoke tests that verify:
- Install → run on sample data → produce outputs
- Runs on CPU in <2 minutes
- Catches missing deps, path issues, config bugs

Run with: python -m pytest tests/test_smoke.py -v

Author: GenAI-RAG-EEG Team
Version: 3.0.0
License: MIT
================================================================================
"""

import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestImports:
    """Test that all critical imports work."""

    def test_torch_import(self):
        """PyTorch must be importable."""
        import torch
        assert torch is not None

    def test_numpy_import(self):
        """NumPy must be importable."""
        import numpy as np
        assert np is not None

    def test_sklearn_import(self):
        """Scikit-learn must be importable."""
        from sklearn.model_selection import train_test_split
        assert train_test_split is not None

    def test_config_import(self):
        """Config module must be importable."""
        from src.config import Config, PROJECT_ROOT, DATA_DIR
        assert Config is not None
        assert PROJECT_ROOT.exists()

    def test_model_import(self):
        """Model module must be importable."""
        from src.models.genai_rag_eeg import GenAIRAGEEG
        assert GenAIRAGEEG is not None

    def test_utils_import(self):
        """Utils package must be importable."""
        from src.utils import setup_logger, RunManager, CompatibilityLayer
        assert setup_logger is not None
        assert RunManager is not None
        assert CompatibilityLayer is not None


class TestConfig:
    """Test configuration loading and validation."""

    def test_config_instantiation(self):
        """Config should instantiate without errors."""
        from src.config import Config
        config = Config()
        assert config is not None

    def test_config_paths_exist(self):
        """Project root and data dir should exist."""
        from src.config import PROJECT_ROOT, DATA_DIR
        assert PROJECT_ROOT.exists(), f"PROJECT_ROOT not found: {PROJECT_ROOT}"
        # DATA_DIR may not exist if no data downloaded yet, but path should be valid
        assert DATA_DIR is not None

    def test_config_model_params(self):
        """Model config should have expected params."""
        from src.config import Config
        config = Config()
        assert hasattr(config.model, 'cnn')
        assert hasattr(config.model, 'lstm')
        assert hasattr(config.model, 'attention')
        assert config.model.cnn.filters > 0
        assert config.model.lstm.hidden_size > 0

    def test_config_training_params(self):
        """Training config should have expected params."""
        from src.config import Config
        config = Config()
        assert config.training.learning_rate > 0
        assert config.training.batch_size > 0
        assert config.training.epochs > 0


class TestModelCreation:
    """Test model instantiation and basic forward pass."""

    def test_model_instantiation(self):
        """Model should instantiate with default params."""
        from src.models.genai_rag_eeg import GenAIRAGEEG
        model = GenAIRAGEEG(n_channels=32, n_time_samples=512)
        assert model is not None

    def test_model_forward_cpu(self):
        """Model forward pass should work on CPU."""
        import torch
        from src.models.genai_rag_eeg import GenAIRAGEEG

        model = GenAIRAGEEG(n_channels=32, n_time_samples=512)
        model.eval()

        # Create dummy input: (batch, channels, time)
        x = torch.randn(2, 32, 512)

        with torch.no_grad():
            output = model(x)

        assert 'logits' in output
        assert 'probs' in output
        assert output['logits'].shape == (2, 2)  # 2 classes
        assert output['probs'].shape == (2, 2)

    def test_model_output_valid(self):
        """Model outputs should be valid probabilities."""
        import torch
        from src.models.genai_rag_eeg import GenAIRAGEEG

        model = GenAIRAGEEG(n_channels=32, n_time_samples=512)
        model.eval()

        x = torch.randn(4, 32, 512)

        with torch.no_grad():
            output = model(x)

        probs = output['probs']
        # Probabilities should sum to 1
        prob_sums = probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(4), atol=1e-5)
        # Probabilities should be in [0, 1]
        assert (probs >= 0).all()
        assert (probs <= 1).all()


class TestSampleData:
    """Test sample data loading (if available)."""

    def test_sample_data_exists(self):
        """Check if sample validation data exists."""
        from src.config import DATA_DIR
        sample_dir = DATA_DIR / "sample_validation"

        if not sample_dir.exists():
            pytest.skip("Sample data not generated yet")

        assert sample_dir.exists()

    def test_sample_data_loadable(self):
        """Sample data should be loadable."""
        from src.config import DATA_DIR
        sample_file = DATA_DIR / "sample_validation" / "combined_sample.npz"

        if not sample_file.exists():
            pytest.skip("Sample data not generated yet")

        data = np.load(sample_file)
        assert 'X' in data
        assert 'y' in data
        assert len(data['X']) == len(data['y'])

    def test_sample_data_shape(self):
        """Sample data should have correct shape."""
        from src.config import DATA_DIR
        sample_file = DATA_DIR / "sample_validation" / "combined_sample.npz"

        if not sample_file.exists():
            pytest.skip("Sample data not generated yet")

        data = np.load(sample_file)
        X = data['X']
        y = data['y']

        # Shape should be (N, channels, time)
        assert len(X.shape) == 3
        assert X.shape[1] == 32  # channels
        assert X.shape[2] == 512  # time samples

        # Labels should be binary
        assert set(np.unique(y)).issubset({0, 1})


class TestEndToEnd:
    """End-to-end smoke test with synthetic data."""

    def test_train_one_batch(self):
        """Train model for one batch on synthetic data."""
        import torch
        import torch.nn as nn
        from src.models.genai_rag_eeg import GenAIRAGEEG

        # Create model
        model = GenAIRAGEEG(n_channels=32, n_time_samples=512)
        model.train()

        # Synthetic data
        X = torch.randn(8, 32, 512)
        y = torch.randint(0, 2, (8,))

        # One training step
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output['logits'], y)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0  # Loss should be positive
        assert not torch.isnan(loss)  # Loss should not be NaN

    def test_inference_pipeline(self):
        """Test full inference pipeline."""
        import torch
        from src.models.genai_rag_eeg import GenAIRAGEEG

        model = GenAIRAGEEG(n_channels=32, n_time_samples=512)
        model.eval()

        # Simulate batch inference
        X = torch.randn(16, 32, 512)

        with torch.no_grad():
            output = model(X)
            predictions = output['logits'].argmax(dim=1)

        assert len(predictions) == 16
        assert all(p in [0, 1] for p in predictions.tolist())


class TestRunManager:
    """Test experiment tracking utilities."""

    def test_run_id_generation(self):
        """Run ID should be generated correctly."""
        from src.utils import generate_run_id

        run_id = generate_run_id()
        assert run_id is not None
        assert len(run_id) > 0
        # Should contain timestamp
        assert '_' in run_id or run_id.isalnum()

    def test_run_manager_creation(self):
        """RunManager should create output directory."""
        import tempfile
        from src.utils import RunManager

        with tempfile.TemporaryDirectory() as tmpdir:
            with RunManager("test_exp", output_base=tmpdir) as run:
                assert run.output_dir.exists()

    def test_data_fingerprint(self):
        """Data fingerprint should capture stats."""
        from src.utils.run_manager import DataFingerprint

        X = np.random.randn(100, 32, 512).astype(np.float32)
        y = np.random.randint(0, 2, 100).astype(np.int64)

        fingerprint = DataFingerprint.create(X, y, "test_data")

        assert fingerprint.name == "test_data"
        assert fingerprint.shape == (100, 32, 512)
        assert fingerprint.n_samples == 100
        assert fingerprint.schema_hash is not None


class TestCompatibility:
    """Test cross-platform compatibility utilities."""

    def test_compatibility_layer_creation(self):
        """CompatibilityLayer should instantiate."""
        from src.utils import CompatibilityLayer

        compat = CompatibilityLayer()
        assert compat is not None

    def test_device_detection(self):
        """Device detection should work."""
        from src.utils import CompatibilityLayer

        compat = CompatibilityLayer()
        device = compat.get_device()

        assert device in ['cpu', 'cuda', 'mps']

    def test_system_requirements_check(self):
        """System requirements check should return dict."""
        from src.utils import check_system_requirements

        result = check_system_requirements()

        assert isinstance(result, dict)
        assert 'python_version' in result or 'status' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
