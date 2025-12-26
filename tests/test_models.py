#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Unit Tests for GenAI-RAG-EEG Models
================================================================================

Tests for:
- EEG Encoder
- Text Context Encoder
- Fusion Layer
- Classification Head
- Complete GenAI-RAG-EEG Model

Run:
    pytest tests/test_models.py -v
    pytest tests/test_models.py -v -k "test_eeg"  # Run specific tests

================================================================================
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.eeg_encoder import EEGEncoder, SelfAttention, ConvBlock
from src.models.text_encoder import TextContextEncoder, create_context_string


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_eeg():
    """Generate sample EEG data."""
    batch_size = 4
    n_channels = 32
    n_time_samples = 512
    return torch.randn(batch_size, n_channels, n_time_samples)


@pytest.fixture
def sample_contexts():
    """Generate sample context strings."""
    return [
        create_context_string("Stroop", 25, "M"),
        create_context_string("Arithmetic", 30, "F"),
        create_context_string("Rest", 22, "M"),
        create_context_string("N-back", 28, "F")
    ]


# =============================================================================
# EEG ENCODER TESTS
# =============================================================================

class TestEEGEncoder:
    """Test cases for EEG Encoder."""

    def test_encoder_creation(self):
        """Test encoder instantiation."""
        encoder = EEGEncoder()
        assert encoder is not None
        assert encoder.output_dim == 128

    def test_encoder_forward_shape(self, sample_eeg):
        """Test forward pass output shape."""
        encoder = EEGEncoder()
        features, attention = encoder(sample_eeg, return_attention=True)

        assert features.shape == (4, 128), f"Expected (4, 128), got {features.shape}"
        assert attention.shape == (4, 64), f"Expected (4, 64), got {attention.shape}"

    def test_encoder_without_attention(self, sample_eeg):
        """Test forward pass without returning attention."""
        encoder = EEGEncoder()
        features, attention = encoder(sample_eeg, return_attention=False)

        assert features.shape == (4, 128)
        assert attention is None

    def test_encoder_parameter_count(self):
        """Test parameter count is reasonable."""
        encoder = EEGEncoder()
        total_params = sum(p.numel() for p in encoder.parameters())

        # Should be around 138K parameters
        assert 100000 < total_params < 200000, f"Unexpected param count: {total_params}"

    def test_encoder_different_input_sizes(self):
        """Test encoder with different input configurations."""
        # Different batch sizes
        for batch_size in [1, 8, 16]:
            encoder = EEGEncoder()
            x = torch.randn(batch_size, 32, 512)
            features, _ = encoder(x)
            assert features.shape == (batch_size, 128)

    def test_encoder_gradient_flow(self, sample_eeg):
        """Test gradients flow through encoder."""
        encoder = EEGEncoder()
        sample_eeg.requires_grad = True

        features, _ = encoder(sample_eeg)
        loss = features.sum()
        loss.backward()

        assert sample_eeg.grad is not None
        assert not torch.isnan(sample_eeg.grad).any()


class TestSelfAttention:
    """Test cases for Self-Attention mechanism."""

    def test_attention_creation(self):
        """Test attention instantiation."""
        attention = SelfAttention(hidden_size=128, attention_dim=64)
        assert attention is not None

    def test_attention_output_shape(self):
        """Test attention output shapes."""
        attention = SelfAttention(hidden_size=128, attention_dim=64)
        lstm_output = torch.randn(4, 64, 128)  # (batch, seq_len, hidden)

        context, weights = attention(lstm_output)

        assert context.shape == (4, 128)
        assert weights.shape == (4, 64)

    def test_attention_weights_sum_to_one(self):
        """Test attention weights sum to 1."""
        attention = SelfAttention(hidden_size=128, attention_dim=64)
        lstm_output = torch.randn(4, 64, 128)

        _, weights = attention(lstm_output)
        sums = weights.sum(dim=-1)

        assert torch.allclose(sums, torch.ones(4), atol=1e-5)


class TestConvBlock:
    """Test cases for Convolutional Block."""

    def test_conv_block_creation(self):
        """Test conv block instantiation."""
        block = ConvBlock(in_channels=32, out_channels=64, kernel_size=7)
        assert block is not None

    def test_conv_block_output_shape(self):
        """Test conv block output shape."""
        block = ConvBlock(in_channels=32, out_channels=64, kernel_size=7)
        x = torch.randn(4, 32, 512)

        out = block(x)

        # MaxPool(2) halves the temporal dimension
        assert out.shape == (4, 64, 256)


# =============================================================================
# TEXT ENCODER TESTS
# =============================================================================

class TestTextContextEncoder:
    """Test cases for Text Context Encoder."""

    @pytest.fixture
    def encoder(self):
        """Create text encoder fixture."""
        return TextContextEncoder()

    def test_encoder_creation(self, encoder):
        """Test encoder instantiation."""
        assert encoder is not None
        assert encoder.output_dim == 128
        assert encoder.bert_dim == 384

    def test_encoder_single_text(self, encoder):
        """Test encoding single text."""
        text = "Task: Stroop. Age: 25. Gender: M."
        embedding = encoder.encode_text(text)

        assert embedding.shape == (1, 128)

    def test_encoder_batch_text(self, encoder, sample_contexts):
        """Test encoding batch of texts."""
        embeddings = encoder.encode_text(sample_contexts)

        assert embeddings.shape == (4, 128)

    def test_encoder_trainable_params(self, encoder):
        """Test only projection layer is trainable."""
        trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in encoder.parameters())

        # Projection: 384 * 128 + 128 = 49,280
        assert 49000 < trainable < 50000
        # Total includes frozen BERT
        assert total > 20_000_000


class TestContextString:
    """Test cases for context string utilities."""

    def test_create_context_basic(self):
        """Test basic context string creation."""
        ctx = create_context_string("Stroop", 25, "M")
        assert "Task: Stroop" in ctx
        assert "Age: 25" in ctx
        assert "Gender: M" in ctx

    def test_create_context_minimal(self):
        """Test context with only task."""
        ctx = create_context_string("Rest")
        assert ctx == "Task: Rest."

    def test_create_context_full(self):
        """Test context with all fields."""
        ctx = create_context_string(
            task_type="Stroop",
            subject_age=25,
            subject_gender="F",
            stress_history="Low baseline",
            environment="Laboratory"
        )
        assert "Task: Stroop" in ctx
        assert "Age: 25" in ctx
        assert "Gender: F" in ctx
        assert "History: Low baseline" in ctx
        assert "Environment: Laboratory" in ctx


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestModelIntegration:
    """Integration tests for complete model."""

    def test_model_forward_pass(self, sample_eeg, sample_contexts):
        """Test complete model forward pass."""
        try:
            from src.models.genai_rag_eeg import GenAIRAGEEG

            model = GenAIRAGEEG(use_rag=False)  # Disable RAG for testing
            output = model(sample_eeg, sample_contexts)

            assert "logits" in output
            assert "probs" in output
            assert output["logits"].shape == (4, 2)
            assert output["probs"].shape == (4, 2)
        except ImportError:
            pytest.skip("GenAIRAGEEG not available")

    def test_model_prediction(self, sample_eeg):
        """Test model prediction method."""
        try:
            from src.models.genai_rag_eeg import GenAIRAGEEG

            model = GenAIRAGEEG(use_text_encoder=False, use_rag=False)
            predictions = model.predict(sample_eeg)

            assert predictions.shape == (4,)
            assert all(p in [0, 1] for p in predictions.tolist())
        except ImportError:
            pytest.skip("GenAIRAGEEG not available")

    def test_model_parameter_count(self):
        """Test model parameter count matches expectations."""
        try:
            from src.models.genai_rag_eeg import GenAIRAGEEG

            model = GenAIRAGEEG(use_rag=False)
            counts = model.get_parameter_count()

            assert "eeg_encoder" in counts
            assert "fusion" in counts
            assert "classifier" in counts
            assert counts["total"] > 200000
        except ImportError:
            pytest.skip("GenAIRAGEEG not available")


# =============================================================================
# NUMERICAL STABILITY TESTS
# =============================================================================

class TestNumericalStability:
    """Test numerical stability of models."""

    def test_no_nan_output(self, sample_eeg):
        """Test outputs don't contain NaN."""
        encoder = EEGEncoder()
        features, attention = encoder(sample_eeg, return_attention=True)

        assert not torch.isnan(features).any()
        assert not torch.isnan(attention).any()

    def test_no_inf_output(self, sample_eeg):
        """Test outputs don't contain Inf."""
        encoder = EEGEncoder()
        features, attention = encoder(sample_eeg, return_attention=True)

        assert not torch.isinf(features).any()
        assert not torch.isinf(attention).any()

    def test_extreme_input_handling(self):
        """Test handling of extreme input values."""
        encoder = EEGEncoder()

        # Large values
        large_input = torch.randn(2, 32, 512) * 1000
        features, _ = encoder(large_input)
        assert not torch.isnan(features).any()

        # Small values
        small_input = torch.randn(2, 32, 512) * 0.001
        features, _ = encoder(small_input)
        assert not torch.isnan(features).any()


# =============================================================================
# DEVICE TESTS
# =============================================================================

class TestDeviceCompatibility:
    """Test model works on different devices."""

    def test_cpu_execution(self, sample_eeg):
        """Test model runs on CPU."""
        encoder = EEGEncoder()
        sample_eeg_cpu = sample_eeg.cpu()

        features, _ = encoder(sample_eeg_cpu)
        assert features.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_execution(self, sample_eeg):
        """Test model runs on GPU."""
        encoder = EEGEncoder().cuda()
        sample_eeg_gpu = sample_eeg.cuda()

        features, _ = encoder(sample_eeg_gpu)
        assert features.device.type == "cuda"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
