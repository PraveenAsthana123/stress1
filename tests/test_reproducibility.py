#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Reproducibility Tests for GenAI-RAG-EEG
================================================================================

Tests that verify reproducibility:
- Same seed → same results
- Model initialization determinism
- Data loading determinism
- Training step determinism

Run with: python -m pytest tests/test_reproducibility.py -v

Author: GenAI-RAG-EEG Team
Version: 3.0.0
License: MIT
================================================================================
"""

import pytest
import torch
import numpy as np
import random
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TestModelInitialization:
    """Test model initialization is reproducible."""

    def test_same_seed_same_weights(self):
        """Same seed should produce identical model weights."""
        from src.models.genai_rag_eeg import GenAIRAGEEG

        # First initialization
        set_seed(42)
        model1 = GenAIRAGEEG(n_channels=32, n_time_samples=512)
        weights1 = {name: param.clone() for name, param in model1.named_parameters()}

        # Second initialization with same seed
        set_seed(42)
        model2 = GenAIRAGEEG(n_channels=32, n_time_samples=512)
        weights2 = {name: param.clone() for name, param in model2.named_parameters()}

        # Compare weights
        for name in weights1:
            assert torch.allclose(weights1[name], weights2[name]), f"Weights differ for {name}"

    def test_different_seed_different_weights(self):
        """Different seeds should produce different model weights."""
        from src.models.genai_rag_eeg import GenAIRAGEEG

        set_seed(42)
        model1 = GenAIRAGEEG(n_channels=32, n_time_samples=512)

        set_seed(123)
        model2 = GenAIRAGEEG(n_channels=32, n_time_samples=512)

        # At least some weights should differ
        any_different = False
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            if not torch.allclose(param1, param2):
                any_different = True
                break

        assert any_different, "Models with different seeds should have different weights"


class TestForwardPass:
    """Test forward pass is reproducible."""

    def test_same_input_same_output(self):
        """Same input should always produce same output."""
        from src.models.genai_rag_eeg import GenAIRAGEEG

        set_seed(42)
        model = GenAIRAGEEG(n_channels=32, n_time_samples=512)
        model.eval()

        # Create input
        x = torch.randn(4, 32, 512)

        # Multiple forward passes
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
            output3 = model(x)

        assert torch.allclose(output1['logits'], output2['logits'])
        assert torch.allclose(output2['logits'], output3['logits'])
        assert torch.allclose(output1['probs'], output2['probs'])

    def test_reproducible_with_seed(self):
        """Full pipeline should be reproducible with seed."""
        from src.models.genai_rag_eeg import GenAIRAGEEG

        def run_inference(seed):
            set_seed(seed)
            model = GenAIRAGEEG(n_channels=32, n_time_samples=512)
            model.eval()
            x = torch.randn(4, 32, 512)
            with torch.no_grad():
                output = model(x)
            return output['logits'].clone()

        # Run twice with same seed
        result1 = run_inference(42)
        result2 = run_inference(42)

        assert torch.allclose(result1, result2), "Same seed should produce same results"


class TestTrainingStep:
    """Test training step is reproducible."""

    def test_single_step_reproducible(self):
        """Single training step should be reproducible."""
        from src.models.genai_rag_eeg import GenAIRAGEEG

        def train_step(seed):
            set_seed(seed)
            model = GenAIRAGEEG(n_channels=32, n_time_samples=512)
            model.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()

            x = torch.randn(8, 32, 512)
            y = torch.randint(0, 2, (8,))

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output['logits'], y)
            loss.backward()
            optimizer.step()

            return loss.item(), {name: param.clone() for name, param in model.named_parameters()}

        # Run twice
        loss1, weights1 = train_step(42)
        loss2, weights2 = train_step(42)

        assert abs(loss1 - loss2) < 1e-6, f"Loss differs: {loss1} vs {loss2}"
        for name in weights1:
            assert torch.allclose(weights1[name], weights2[name], atol=1e-6), f"Weights differ for {name}"

    def test_multiple_steps_reproducible(self):
        """Multiple training steps should be reproducible."""
        from src.models.genai_rag_eeg import GenAIRAGEEG

        def train_multiple_steps(seed, n_steps=5):
            set_seed(seed)
            model = GenAIRAGEEG(n_channels=32, n_time_samples=512)
            model.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()

            losses = []
            for _ in range(n_steps):
                x = torch.randn(8, 32, 512)
                y = torch.randint(0, 2, (8,))

                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output['logits'], y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            return losses

        losses1 = train_multiple_steps(42, n_steps=5)
        losses2 = train_multiple_steps(42, n_steps=5)

        for i, (l1, l2) in enumerate(zip(losses1, losses2)):
            assert abs(l1 - l2) < 1e-5, f"Step {i}: Loss differs: {l1} vs {l2}"


class TestDataGeneration:
    """Test synthetic data generation is reproducible."""

    def test_numpy_random_reproducible(self):
        """NumPy random generation should be reproducible."""
        set_seed(42)
        data1 = np.random.randn(100, 32, 512)

        set_seed(42)
        data2 = np.random.randn(100, 32, 512)

        np.testing.assert_array_equal(data1, data2)

    def test_torch_random_reproducible(self):
        """PyTorch random generation should be reproducible."""
        set_seed(42)
        data1 = torch.randn(100, 32, 512)

        set_seed(42)
        data2 = torch.randn(100, 32, 512)

        assert torch.equal(data1, data2)

    def test_label_generation_reproducible(self):
        """Label generation should be reproducible."""
        set_seed(42)
        labels1 = np.random.randint(0, 2, 100)

        set_seed(42)
        labels2 = np.random.randint(0, 2, 100)

        np.testing.assert_array_equal(labels1, labels2)


class TestShuffling:
    """Test data shuffling is reproducible."""

    def test_numpy_shuffle_reproducible(self):
        """NumPy shuffling should be reproducible."""
        data = np.arange(100)

        set_seed(42)
        indices1 = np.random.permutation(len(data))

        set_seed(42)
        indices2 = np.random.permutation(len(data))

        np.testing.assert_array_equal(indices1, indices2)

    def test_train_test_split_reproducible(self):
        """Train/test split should be reproducible."""
        from sklearn.model_selection import train_test_split

        X = np.random.randn(100, 32, 512)
        y = np.random.randint(0, 2, 100)

        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(y_test1, y_test2)


class TestRunManager:
    """Test RunManager generates reproducible IDs."""

    def test_run_id_contains_timestamp(self):
        """Run ID should contain timestamp."""
        from src.utils import generate_run_id

        run_id = generate_run_id()
        # Should have timestamp format YYYYMMDD_HHMMSS
        assert len(run_id) >= 15  # At least timestamp part


class TestConfigReproducibility:
    """Test config produces reproducible results."""

    def test_config_values_consistent(self):
        """Config values should be consistent across loads."""
        from src.config import Config

        config1 = Config()
        config2 = Config()

        assert config1.training.learning_rate == config2.training.learning_rate
        assert config1.training.batch_size == config2.training.batch_size
        assert config1.model.cnn.filters == config2.model.cnn.filters
        assert config1.model.lstm.hidden_size == config2.model.lstm.hidden_size


class TestEndToEndReproducibility:
    """End-to-end reproducibility test."""

    def test_full_pipeline_reproducible(self):
        """Full training pipeline should be reproducible."""
        from src.models.genai_rag_eeg import GenAIRAGEEG
        from sklearn.model_selection import train_test_split

        def run_pipeline(seed):
            set_seed(seed)

            # Generate data
            X = np.random.randn(100, 32, 512).astype(np.float32)
            y = np.random.randint(0, 2, 100)

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed
            )

            # Create model
            model = GenAIRAGEEG(n_channels=32, n_time_samples=512)
            model.train()

            # Train 3 epochs
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()

            for epoch in range(3):
                for i in range(0, len(X_train), 16):
                    batch_x = torch.FloatTensor(X_train[i:i+16])
                    batch_y = torch.LongTensor(y_train[i:i+16])

                    optimizer.zero_grad()
                    output = model(batch_x)
                    loss = criterion(output['logits'], batch_y)
                    loss.backward()
                    optimizer.step()

            # Evaluate
            model.eval()
            with torch.no_grad():
                X_test_t = torch.FloatTensor(X_test)
                output = model(X_test_t)
                predictions = output['logits'].argmax(dim=1).numpy()

            return predictions

        pred1 = run_pipeline(42)
        pred2 = run_pipeline(42)

        np.testing.assert_array_equal(pred1, pred2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
