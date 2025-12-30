#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Model Shape Tests for GenAI-RAG-EEG
================================================================================

Tests that verify model input/output shapes are consistent:
- Input shape validation
- Output shape validation
- Intermediate layer shapes
- Various batch sizes
- Edge cases

Run with: python -m pytest tests/test_model_shapes.py -v

Author: GenAI-RAG-EEG Team
Version: 3.0.0
License: MIT
================================================================================
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestInputShapes:
    """Test model accepts various input configurations."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        from src.models.genai_rag_eeg import GenAIRAGEEG
        return GenAIRAGEEG(n_channels=32, n_time_samples=512)

    def test_standard_input(self, model):
        """Standard input shape (batch=8, channels=32, time=512)."""
        x = torch.randn(8, 32, 512)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output['logits'].shape == (8, 2)

    def test_batch_size_1(self, model):
        """Single sample batch."""
        x = torch.randn(1, 32, 512)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output['logits'].shape == (1, 2)

    def test_batch_size_32(self, model):
        """Larger batch size."""
        x = torch.randn(32, 32, 512)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output['logits'].shape == (32, 2)

    def test_batch_size_64(self, model):
        """Large batch size."""
        x = torch.randn(64, 32, 512)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output['logits'].shape == (64, 2)

    def test_various_batch_sizes(self, model):
        """Test multiple batch sizes."""
        model.eval()
        for batch_size in [1, 2, 4, 8, 16, 32]:
            x = torch.randn(batch_size, 32, 512)
            with torch.no_grad():
                output = model(x)
            assert output['logits'].shape == (batch_size, 2), f"Failed for batch_size={batch_size}"


class TestOutputShapes:
    """Test model output shapes and types."""

    @pytest.fixture
    def model(self):
        from src.models.genai_rag_eeg import GenAIRAGEEG
        return GenAIRAGEEG(n_channels=32, n_time_samples=512)

    def test_logits_shape(self, model):
        """Logits should have shape (batch, num_classes)."""
        x = torch.randn(8, 32, 512)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output['logits'].shape == (8, 2)
        assert output['logits'].dtype == torch.float32

    def test_probs_shape(self, model):
        """Probabilities should have shape (batch, num_classes)."""
        x = torch.randn(8, 32, 512)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output['probs'].shape == (8, 2)
        assert output['probs'].dtype == torch.float32

    def test_output_keys(self, model):
        """Output should contain expected keys."""
        x = torch.randn(4, 32, 512)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert 'logits' in output
        assert 'probs' in output

    def test_probs_sum_to_one(self, model):
        """Probabilities should sum to 1."""
        x = torch.randn(16, 32, 512)
        model.eval()
        with torch.no_grad():
            output = model(x)
        prob_sums = output['probs'].sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(16), atol=1e-5)

    def test_probs_valid_range(self, model):
        """Probabilities should be in [0, 1]."""
        x = torch.randn(16, 32, 512)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert (output['probs'] >= 0).all()
        assert (output['probs'] <= 1).all()


class TestChannelConfigurations:
    """Test different channel configurations."""

    def test_32_channels(self):
        """Standard 32 channel configuration."""
        from src.models.genai_rag_eeg import GenAIRAGEEG
        model = GenAIRAGEEG(n_channels=32, n_time_samples=512)
        x = torch.randn(4, 32, 512)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output['logits'].shape == (4, 2)

    def test_14_channels_eegmat(self):
        """14 channel configuration (EEGMAT-like)."""
        from src.models.genai_rag_eeg import GenAIRAGEEG
        model = GenAIRAGEEG(n_channels=14, n_time_samples=512)
        x = torch.randn(4, 14, 512)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output['logits'].shape == (4, 2)

    def test_21_channels_eegmat(self):
        """21 channel configuration (EEGMAT original)."""
        from src.models.genai_rag_eeg import GenAIRAGEEG
        model = GenAIRAGEEG(n_channels=21, n_time_samples=512)
        x = torch.randn(4, 21, 512)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output['logits'].shape == (4, 2)

    def test_64_channels(self):
        """64 channel configuration (high-density EEG)."""
        from src.models.genai_rag_eeg import GenAIRAGEEG
        model = GenAIRAGEEG(n_channels=64, n_time_samples=512)
        x = torch.randn(4, 64, 512)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output['logits'].shape == (4, 2)


class TestTimeSampleConfigurations:
    """Test different time sample configurations."""

    def test_512_samples(self):
        """Standard 512 time samples (2 seconds at 256 Hz)."""
        from src.models.genai_rag_eeg import GenAIRAGEEG
        model = GenAIRAGEEG(n_channels=32, n_time_samples=512)
        x = torch.randn(4, 32, 512)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output['logits'].shape == (4, 2)

    def test_256_samples(self):
        """256 time samples (1 second at 256 Hz)."""
        from src.models.genai_rag_eeg import GenAIRAGEEG
        model = GenAIRAGEEG(n_channels=32, n_time_samples=256)
        x = torch.randn(4, 32, 256)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output['logits'].shape == (4, 2)

    def test_1024_samples(self):
        """1024 time samples (4 seconds at 256 Hz)."""
        from src.models.genai_rag_eeg import GenAIRAGEEG
        model = GenAIRAGEEG(n_channels=32, n_time_samples=1024)
        x = torch.randn(4, 32, 1024)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output['logits'].shape == (4, 2)


class TestGradientFlow:
    """Test gradient flow through model."""

    @pytest.fixture
    def model(self):
        from src.models.genai_rag_eeg import GenAIRAGEEG
        return GenAIRAGEEG(n_channels=32, n_time_samples=512)

    def test_gradients_flow(self, model):
        """Gradients should flow through all layers."""
        model.train()
        x = torch.randn(4, 32, 512, requires_grad=True)
        y = torch.randint(0, 2, (4,))

        output = model(x)
        loss = torch.nn.functional.cross_entropy(output['logits'], y)
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_no_gradient_explosion(self, model):
        """Gradients should not explode."""
        model.train()
        x = torch.randn(4, 32, 512, requires_grad=True)
        y = torch.randint(0, 2, (4,))

        output = model(x)
        loss = torch.nn.functional.cross_entropy(output['logits'], y)
        loss.backward()

        # Check gradient magnitudes
        max_grad = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                max_grad = max(max_grad, param.grad.abs().max().item())

        assert max_grad < 1000, f"Gradient explosion: max_grad={max_grad}"


class TestModelParameters:
    """Test model parameter counts and configurations."""

    def test_parameter_count(self):
        """Model should have reasonable parameter count."""
        from src.models.genai_rag_eeg import GenAIRAGEEG
        model = GenAIRAGEEG(n_channels=32, n_time_samples=512)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Should be in reasonable range for research model
        assert 100_000 < total_params < 10_000_000, f"Unexpected param count: {total_params}"
        assert trainable_params == total_params  # All params should be trainable

    def test_model_dtype(self):
        """Model parameters should be float32."""
        from src.models.genai_rag_eeg import GenAIRAGEEG
        model = GenAIRAGEEG(n_channels=32, n_time_samples=512)

        for name, param in model.named_parameters():
            assert param.dtype == torch.float32, f"Unexpected dtype for {name}: {param.dtype}"


class TestNumericalStability:
    """Test numerical stability of model."""

    @pytest.fixture
    def model(self):
        from src.models.genai_rag_eeg import GenAIRAGEEG
        return GenAIRAGEEG(n_channels=32, n_time_samples=512)

    def test_zero_input(self, model):
        """Model should handle zero input."""
        x = torch.zeros(4, 32, 512)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert not torch.isnan(output['logits']).any()
        assert not torch.isinf(output['logits']).any()

    def test_normalized_input(self, model):
        """Model should handle normalized input (mean=0, std=1)."""
        x = torch.randn(4, 32, 512)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert not torch.isnan(output['logits']).any()
        assert not torch.isinf(output['logits']).any()

    def test_scaled_input(self, model):
        """Model should handle scaled input."""
        x = torch.randn(4, 32, 512) * 100  # Large values
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert not torch.isnan(output['logits']).any()
        assert not torch.isinf(output['logits']).any()

    def test_small_input(self, model):
        """Model should handle small input values."""
        x = torch.randn(4, 32, 512) * 0.001  # Small values
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert not torch.isnan(output['logits']).any()
        assert not torch.isinf(output['logits']).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
