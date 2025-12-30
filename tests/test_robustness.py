#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Robustness Tests for GenAI-RAG-EEG
================================================================================

Tests model robustness under various challenging conditions:
- Gaussian noise injection
- Missing/zeroed channels
- Downsampled input
- Artifact-like patterns
- Input scaling variations

Run with: python -m pytest tests/test_robustness.py -v

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


class TestNoiseRobustness:
    """Test model robustness to noise."""

    @pytest.fixture
    def model(self):
        from src.models.genai_rag_eeg import GenAIRAGEEG
        model = GenAIRAGEEG(n_channels=32, n_time_samples=512)
        model.eval()
        return model

    @pytest.fixture
    def clean_input(self):
        """Generate clean synthetic EEG."""
        torch.manual_seed(42)
        return torch.randn(8, 32, 512)

    def test_low_noise_robustness(self, model, clean_input):
        """Model should handle low noise (SNR ~20dB)."""
        noise = torch.randn_like(clean_input) * 0.1
        noisy_input = clean_input + noise

        with torch.no_grad():
            clean_out = model(clean_input)
            noisy_out = model(noisy_input)

        # Predictions should be mostly similar
        clean_preds = clean_out['logits'].argmax(dim=1)
        noisy_preds = noisy_out['logits'].argmax(dim=1)

        agreement = (clean_preds == noisy_preds).float().mean()
        assert agreement >= 0.5, f"Low noise agreement too low: {agreement}"

    def test_medium_noise_robustness(self, model, clean_input):
        """Model should handle medium noise (SNR ~10dB)."""
        noise = torch.randn_like(clean_input) * 0.3
        noisy_input = clean_input + noise

        with torch.no_grad():
            output = model(noisy_input)

        # Model should still produce valid outputs
        assert not torch.isnan(output['logits']).any()
        assert not torch.isinf(output['logits']).any()

    def test_high_noise_robustness(self, model, clean_input):
        """Model should not crash with high noise."""
        noise = torch.randn_like(clean_input) * 1.0
        noisy_input = clean_input + noise

        with torch.no_grad():
            output = model(noisy_input)

        # Model should still produce valid outputs
        assert not torch.isnan(output['logits']).any()
        assert not torch.isinf(output['logits']).any()
        assert output['probs'].sum(dim=1).allclose(torch.ones(8), atol=1e-5)


class TestMissingChannelRobustness:
    """Test model robustness to missing/zeroed channels."""

    @pytest.fixture
    def model(self):
        from src.models.genai_rag_eeg import GenAIRAGEEG
        return GenAIRAGEEG(n_channels=32, n_time_samples=512)

    @pytest.fixture
    def clean_input(self):
        torch.manual_seed(42)
        return torch.randn(8, 32, 512)

    def test_single_missing_channel(self, model, clean_input):
        """Model should handle one zeroed channel."""
        model.eval()
        x = clean_input.clone()
        x[:, 0, :] = 0  # Zero out first channel

        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output['logits']).any()
        assert output['probs'].shape == (8, 2)

    def test_multiple_missing_channels(self, model, clean_input):
        """Model should handle multiple zeroed channels."""
        model.eval()
        x = clean_input.clone()
        x[:, :5, :] = 0  # Zero out first 5 channels

        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output['logits']).any()
        assert output['probs'].shape == (8, 2)

    def test_random_missing_channels(self, model, clean_input):
        """Model should handle randomly missing channels."""
        model.eval()
        x = clean_input.clone()

        # Randomly zero 20% of channels
        mask = torch.rand(32) < 0.2
        x[:, mask, :] = 0

        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output['logits']).any()
        assert output['probs'].shape == (8, 2)

    def test_half_missing_channels(self, model, clean_input):
        """Model should handle 50% missing channels (extreme case)."""
        model.eval()
        x = clean_input.clone()
        x[:, ::2, :] = 0  # Zero every other channel

        with torch.no_grad():
            output = model(x)

        # Should still produce valid output
        assert not torch.isnan(output['logits']).any()
        assert not torch.isinf(output['logits']).any()


class TestScalingRobustness:
    """Test model robustness to input scaling variations."""

    @pytest.fixture
    def model(self):
        from src.models.genai_rag_eeg import GenAIRAGEEG
        return GenAIRAGEEG(n_channels=32, n_time_samples=512)

    def test_small_amplitude(self, model):
        """Model should handle small amplitude inputs."""
        model.eval()
        x = torch.randn(4, 32, 512) * 0.01

        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output['logits']).any()
        assert not torch.isinf(output['logits']).any()

    def test_large_amplitude(self, model):
        """Model should handle large amplitude inputs."""
        model.eval()
        x = torch.randn(4, 32, 512) * 100

        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output['logits']).any()
        assert not torch.isinf(output['logits']).any()

    def test_varying_amplitude_per_channel(self, model):
        """Model should handle different scales per channel."""
        model.eval()
        x = torch.randn(4, 32, 512)

        # Scale each channel differently
        scales = torch.linspace(0.1, 10, 32).view(1, 32, 1)
        x = x * scales

        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output['logits']).any()
        assert not torch.isinf(output['logits']).any()

    def test_unnormalized_input(self, model):
        """Model should handle unnormalized input with offset."""
        model.eval()
        x = torch.randn(4, 32, 512) + 50  # Mean of 50

        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output['logits']).any()
        assert not torch.isinf(output['logits']).any()


class TestArtifactRobustness:
    """Test model robustness to artifact-like patterns."""

    @pytest.fixture
    def model(self):
        from src.models.genai_rag_eeg import GenAIRAGEEG
        return GenAIRAGEEG(n_channels=32, n_time_samples=512)

    def test_spike_artifact(self, model):
        """Model should handle spike artifacts."""
        model.eval()
        x = torch.randn(4, 32, 512)

        # Add spike artifacts
        x[:, 0, 100:110] = 10  # Large spike
        x[:, 5, 200:205] = -10  # Negative spike

        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output['logits']).any()
        assert not torch.isinf(output['logits']).any()

    def test_drift_artifact(self, model):
        """Model should handle drift artifacts."""
        model.eval()
        x = torch.randn(4, 32, 512)

        # Add linear drift
        drift = torch.linspace(0, 5, 512).view(1, 1, 512)
        x[:, :5, :] = x[:, :5, :] + drift

        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output['logits']).any()
        assert not torch.isinf(output['logits']).any()

    def test_saturation_artifact(self, model):
        """Model should handle saturation (clipping) artifacts."""
        model.eval()
        x = torch.randn(4, 32, 512)

        # Clip values (simulate amplifier saturation)
        x = torch.clamp(x, -3, 3)

        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output['logits']).any()
        assert not torch.isinf(output['logits']).any()

    def test_flatline_artifact(self, model):
        """Model should handle flatline segments."""
        model.eval()
        x = torch.randn(4, 32, 512)

        # Add flatline segments
        x[:, 0, 100:200] = 0  # Flatline
        x[:, 5, 300:400] = x[:, 5, 300].unsqueeze(-1)  # Constant

        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output['logits']).any()
        assert not torch.isinf(output['logits']).any()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def model(self):
        from src.models.genai_rag_eeg import GenAIRAGEEG
        return GenAIRAGEEG(n_channels=32, n_time_samples=512)

    def test_all_zeros(self, model):
        """Model should handle all-zero input."""
        model.eval()
        x = torch.zeros(4, 32, 512)

        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output['logits']).any()
        assert output['probs'].shape == (4, 2)

    def test_all_ones(self, model):
        """Model should handle all-ones input."""
        model.eval()
        x = torch.ones(4, 32, 512)

        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output['logits']).any()

    def test_alternating_values(self, model):
        """Model should handle alternating pattern."""
        model.eval()
        x = torch.zeros(4, 32, 512)
        x[:, :, ::2] = 1
        x[:, :, 1::2] = -1

        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output['logits']).any()

    def test_single_sample_batch(self, model):
        """Model should handle batch size of 1."""
        model.eval()
        x = torch.randn(1, 32, 512)

        with torch.no_grad():
            output = model(x)

        assert output['logits'].shape == (1, 2)
        assert output['probs'].shape == (1, 2)


class TestPreprocessingRobustness:
    """Test preprocessing pipeline robustness."""

    def test_preprocessing_with_noise(self):
        """Preprocessing should handle noisy data."""
        from src.data.preprocessing import EEGPreprocessor, PreprocessingConfig

        config = PreprocessingConfig(
            artifact_threshold=200.0  # Higher threshold
        )
        preprocessor = EEGPreprocessor(config=config, fs=256.0)

        # Create noisy data
        np.random.seed(42)
        data = np.random.randn(32, 15360)  # 60s at 256Hz
        data += np.random.randn(32, 15360) * 2  # Add noise

        result = preprocessor.process(data, return_stats=True)

        assert result['epochs'].shape[0] > 0
        assert not np.isnan(result['epochs']).any()

    def test_preprocessing_with_artifacts(self):
        """Preprocessing should reject artifacts."""
        from src.data.preprocessing import EEGPreprocessor

        preprocessor = EEGPreprocessor(fs=256.0)

        # Create data with artifacts
        np.random.seed(42)
        data = np.random.randn(32, 15360) * 20  # Normal EEG

        # Add large artifacts
        data[0, 5000:5100] = 500  # Very large artifact

        result = preprocessor.process(data, return_stats=True)

        # Some epochs should be rejected
        assert result['stats']['n_epochs_rejected'] > 0

    def test_bandpass_filter_stability(self):
        """Bandpass filter should be stable."""
        from src.data.preprocessing import BandpassFilter

        bp = BandpassFilter(lowcut=0.5, highcut=45.0, fs=256.0)

        # Test with various inputs
        data = np.random.randn(32, 1024)
        filtered = bp(data)

        assert not np.isnan(filtered).any()
        assert not np.isinf(filtered).any()

    def test_notch_filter_stability(self):
        """Notch filter should be stable."""
        from src.data.preprocessing import NotchFilter

        notch = NotchFilter(freq=50.0, fs=256.0)

        # Test with various inputs
        data = np.random.randn(32, 1024)
        filtered = notch(data)

        assert not np.isnan(filtered).any()
        assert not np.isinf(filtered).any()


class TestPerformanceDegradation:
    """Test that performance degrades gracefully under stress."""

    @pytest.fixture
    def trained_model(self):
        """Get a trained model for degradation tests."""
        import torch
        from src.models.genai_rag_eeg import GenAIRAGEEG

        # Quick training
        torch.manual_seed(42)
        model = GenAIRAGEEG(n_channels=32, n_time_samples=512)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(5):
            x = torch.randn(16, 32, 512)
            y = torch.randint(0, 2, (16,))
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output['logits'], y)
            loss.backward()
            optimizer.step()

        model.eval()
        return model

    def test_gradual_noise_degradation(self, trained_model):
        """Performance should degrade gradually with increasing noise."""
        torch.manual_seed(42)
        x = torch.randn(100, 32, 512)
        y = torch.randint(0, 2, (100,))

        noise_levels = [0.0, 0.1, 0.5, 1.0, 2.0]
        accuracies = []

        for noise_std in noise_levels:
            x_noisy = x + torch.randn_like(x) * noise_std

            with torch.no_grad():
                output = trained_model(x_noisy)
                preds = output['logits'].argmax(dim=1)
                acc = (preds == y).float().mean().item()
                accuracies.append(acc)

        # Accuracy should not suddenly drop to 0
        assert min(accuracies) > 0.3, "Model collapsed under noise"

    def test_channel_dropout_degradation(self, trained_model):
        """Performance should degrade with more missing channels."""
        torch.manual_seed(42)
        x = torch.randn(100, 32, 512)

        dropout_rates = [0.0, 0.1, 0.3, 0.5]

        for rate in dropout_rates:
            x_dropped = x.clone()
            n_drop = int(32 * rate)
            if n_drop > 0:
                drop_idx = torch.randperm(32)[:n_drop]
                x_dropped[:, drop_idx, :] = 0

            with torch.no_grad():
                output = trained_model(x_dropped)

            # Should not produce NaN
            assert not torch.isnan(output['logits']).any()
            assert not torch.isinf(output['logits']).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
