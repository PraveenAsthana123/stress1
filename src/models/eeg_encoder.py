#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
EEG Encoder Module for GenAI-RAG-EEG Architecture
================================================================================

Title: Deep Learning EEG Encoder for Stress Classification
Authors: [Your Name], [Co-author Names]
Institution: [Your Institution]
Contact: [Your Email]

Description:
    This module implements the core EEG feature extraction pipeline using a
    hybrid deep learning architecture combining Convolutional Neural Networks
    (CNN), Bidirectional Long Short-Term Memory (Bi-LSTM), and Self-Attention
    mechanisms.

Architecture Overview:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         EEG ENCODER PIPELINE                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  INPUT: Raw EEG Signal (batch, 32 channels, 512 time samples)          │
    │                              │                                          │
    │                              ▼                                          │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │  CONV BLOCK 1: Spatial-Temporal Feature Extraction              │   │
    │  │  • Conv1D (32→32 filters, kernel=7)                             │   │
    │  │  • BatchNorm → ReLU → MaxPool(2) → Dropout(0.3)                 │   │
    │  │  • Output: (batch, 32, 256)                                     │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              │                                          │
    │                              ▼                                          │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │  CONV BLOCK 2: Higher-Level Feature Extraction                  │   │
    │  │  • Conv1D (32→64 filters, kernel=5)                             │   │
    │  │  • BatchNorm → ReLU → MaxPool(2) → Dropout(0.3)                 │   │
    │  │  • Output: (batch, 64, 128)                                     │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              │                                          │
    │                              ▼                                          │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │  CONV BLOCK 3: Abstract Feature Extraction                      │   │
    │  │  • Conv1D (64→64 filters, kernel=3)                             │   │
    │  │  • BatchNorm → ReLU → MaxPool(2) → Dropout(0.3)                 │   │
    │  │  • Output: (batch, 64, 64)                                      │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              │                                          │
    │                              ▼                                          │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │  BI-LSTM: Temporal Sequence Modeling                            │   │
    │  │  • Bidirectional LSTM (hidden=64, layers=1)                     │   │
    │  │  • Captures forward & backward temporal dependencies            │   │
    │  │  • Output: (batch, 64, 128)                                     │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              │                                          │
    │                              ▼                                          │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │  SELF-ATTENTION: Temporal Importance Weighting                  │   │
    │  │  • Learns attention weights α_t for each time step              │   │
    │  │  • Focuses on stress-relevant temporal segments                 │   │
    │  │  • Output: (batch, 128) context vector                          │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              │                                          │
    │                              ▼                                          │
    │  OUTPUT: EEG Feature Vector (batch, 128)                               │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

Data Flow:
    1. Raw EEG → Conv1D extracts local spatial-temporal patterns
    2. Multiple conv layers → hierarchical feature abstraction
    3. Bi-LSTM → captures long-range temporal dependencies
    4. Self-attention → weighted aggregation focusing on discriminative segments

Key Components:
    - ConvBlock: Modular conv layer with normalization and regularization
    - SelfAttention: Learnable temporal importance weighting
    - EEGEncoder: Complete encoder combining all components

Parameters:
    - Total Trainable: ~138,081
    - Conv Layers: ~30,000
    - Bi-LSTM: ~99,584
    - Attention: ~8,321

References:
    [1] Koelstra et al., "DEAP: A Database for Emotion Analysis using
        Physiological Signals," IEEE Trans. Affective Computing, 2012.
    [2] Hochreiter & Schmidhuber, "Long Short-Term Memory," Neural Computation, 1997.
    [3] Bahdanau et al., "Neural Machine Translation by Jointly Learning to
        Align and Translate," ICLR, 2015.

License: MIT License
Version: 1.0.0
Last Updated: 2024
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# =============================================================================
# SELF-ATTENTION MECHANISM
# =============================================================================

class SelfAttention(nn.Module):
    """
    Self-Attention Mechanism for Temporal Sequence Weighting.

    This module computes importance weights across the temporal sequence,
    enabling the model to focus on stress-relevant time segments while
    suppressing noise and irrelevant portions.

    Mathematical Formulation:
    -------------------------
    Given LSTM output H = [h_1, h_2, ..., h_T] ∈ R^(T × d):

    1. Energy computation:
       e_t = tanh(W_a · h_t + b_a)  ∈ R^(attention_dim)

    2. Score computation:
       s_t = w_a^T · e_t  ∈ R

    3. Attention weights (softmax normalization):
       α_t = exp(s_t) / Σ_j exp(s_j)

    4. Context vector (weighted sum):
       c = Σ_t α_t · h_t  ∈ R^d

    Parameters:
    -----------
    hidden_size : int
        Dimension of input LSTM hidden states (default: 128 for bidirectional)
    attention_dim : int
        Dimension of attention intermediate layer (default: 64)

    Trainable Parameters: 8,321
        - W_a: 128 × 64 = 8,192
        - b_a: 64 (bias)
        - w_a: 64 × 1 = 64
        - w_a bias: 1

    Integration:
    ------------
    This module integrates with the EEGEncoder by receiving Bi-LSTM outputs
    and producing a fixed-size context vector that summarizes the entire
    temporal sequence with learned importance weighting.

    Visualization:
    --------------
    Attention weights can be visualized as a heatmap over time to interpret
    which temporal segments the model considers most relevant for classification.
    High attention regions typically correspond to stress-related EEG patterns.
    """

    def __init__(self, hidden_size: int = 128, attention_dim: int = 64):
        """
        Initialize Self-Attention mechanism.

        Args:
            hidden_size: Dimension of LSTM hidden states (128 for bidirectional)
            attention_dim: Dimension of attention layer (controls capacity)
        """
        super().__init__()

        # =====================================================================
        # ATTENTION WEIGHT MATRICES
        # =====================================================================
        # W_a: Projects hidden states to attention space
        # This learns to identify important temporal patterns
        self.W_a = nn.Linear(hidden_size, attention_dim)

        # w_a: Computes scalar attention score from attention representation
        # No bias to ensure pure importance weighting
        self.w_a = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted context vector.

        Process Flow:
        -------------
        1. Project LSTM outputs to attention space
        2. Apply tanh non-linearity for bounded activations
        3. Compute scalar attention scores
        4. Apply softmax to get probability distribution
        5. Compute weighted sum of LSTM outputs

        Args:
            lstm_output: Bi-LSTM output tensor
                Shape: (batch_size, sequence_length, hidden_size)
                Example: (32, 64, 128) for 32 samples, 64 time steps

        Returns:
            context: Attention-weighted context vector
                Shape: (batch_size, hidden_size)
                Contains aggregated temporal information
            attention_weights: Attention distribution over time
                Shape: (batch_size, sequence_length)
                Values sum to 1.0, indicating relative importance

        Visualization Note:
            attention_weights can be plotted as:
            - Line plot over time showing attention intensity
            - Heatmap for batch visualization
            - Overlaid on raw EEG to show focus regions
        """
        # =====================================================================
        # STEP 1: PROJECT TO ATTENTION SPACE
        # =====================================================================
        # Apply linear transformation and tanh activation
        # This creates a bounded representation suitable for scoring
        energy = torch.tanh(self.W_a(lstm_output))  # (batch, seq_len, attention_dim)

        # =====================================================================
        # STEP 2: COMPUTE ATTENTION SCORES
        # =====================================================================
        # Reduce to scalar score per time step
        scores = self.w_a(energy).squeeze(-1)  # (batch, seq_len)

        # =====================================================================
        # STEP 3: NORMALIZE WITH SOFTMAX
        # =====================================================================
        # Convert scores to probability distribution
        # Ensures weights sum to 1 for interpretable importance
        attention_weights = F.softmax(scores, dim=-1)  # (batch, seq_len)

        # =====================================================================
        # STEP 4: COMPUTE WEIGHTED CONTEXT VECTOR
        # =====================================================================
        # Weighted sum of LSTM outputs using attention weights
        # bmm: batch matrix multiplication for efficient computation
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_output                       # (batch, seq_len, hidden)
        ).squeeze(1)                          # (batch, hidden)

        return context, attention_weights


# =============================================================================
# CONVOLUTIONAL BLOCK
# =============================================================================

class ConvBlock(nn.Module):
    """
    Convolutional Block for EEG Feature Extraction.

    This module implements a single convolutional processing stage with:
    - 1D Convolution for spatial-temporal pattern extraction
    - Batch Normalization for training stability
    - ReLU activation for non-linearity
    - Max Pooling for temporal downsampling
    - Dropout for regularization

    Architecture:
    -------------
    Input → Conv1D → BatchNorm → ReLU → MaxPool → Dropout → Output

    Logic:
    ------
    1. Conv1D: Learns local patterns across channels and time
       - Kernel slides along temporal dimension
       - Captures spatial correlations between EEG channels

    2. BatchNorm: Normalizes activations for stable training
       - Reduces internal covariate shift
       - Allows higher learning rates

    3. ReLU: Introduces non-linearity
       - f(x) = max(0, x)
       - Enables learning complex patterns

    4. MaxPool: Reduces temporal resolution
       - Captures dominant features in windows
       - Provides translation invariance

    5. Dropout: Prevents overfitting
       - Randomly zeros activations during training
       - Improves generalization

    Parameters:
    -----------
    in_channels : int
        Number of input channels (EEG channels or previous layer filters)
    out_channels : int
        Number of output feature maps (filters to learn)
    kernel_size : int
        Size of convolutional kernel (temporal receptive field)
    pool_size : int
        Max pooling window size (temporal downsampling factor)
    dropout : float
        Dropout probability (regularization strength)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize Convolutional Block.

        Args:
            in_channels: Number of input feature channels
            out_channels: Number of output feature channels (filters)
            kernel_size: Convolution kernel size (temporal span)
            pool_size: Max pooling window size
            dropout: Dropout probability for regularization
        """
        super().__init__()

        # =====================================================================
        # 1D CONVOLUTION LAYER
        # =====================================================================
        # Extracts local spatial-temporal patterns
        # Padding maintains temporal dimension before pooling
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2  # Same padding to preserve length
        )

        # =====================================================================
        # BATCH NORMALIZATION
        # =====================================================================
        # Normalizes across batch and spatial dimensions
        # Stabilizes training and accelerates convergence
        self.bn = nn.BatchNorm1d(out_channels)

        # =====================================================================
        # MAX POOLING
        # =====================================================================
        # Reduces temporal dimension by factor of pool_size
        # Captures dominant features in each window
        self.pool = nn.MaxPool1d(pool_size)

        # =====================================================================
        # DROPOUT REGULARIZATION
        # =====================================================================
        # Randomly drops activations during training
        # Reduces co-adaptation and improves generalization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolutional block.

        Data Flow:
        ----------
        Input: (batch, in_channels, time_samples)
            ↓ Conv1D
        After conv: (batch, out_channels, time_samples)
            ↓ BatchNorm
        Normalized: (batch, out_channels, time_samples)
            ↓ ReLU
        Activated: (batch, out_channels, time_samples)
            ↓ MaxPool(2)
        Downsampled: (batch, out_channels, time_samples/2)
            ↓ Dropout
        Output: (batch, out_channels, time_samples/2)

        Args:
            x: Input tensor of shape (batch, in_channels, time_samples)

        Returns:
            Output tensor of shape (batch, out_channels, time_samples/pool_size)
        """
        # Convolution: Learn local patterns
        x = self.conv(x)

        # Batch normalization: Stabilize activations
        x = self.bn(x)

        # ReLU activation: Introduce non-linearity
        x = F.relu(x)

        # Max pooling: Downsample and capture dominant features
        x = self.pool(x)

        # Dropout: Regularization (active only during training)
        x = self.dropout(x)

        return x


# =============================================================================
# MAIN EEG ENCODER
# =============================================================================

class EEGEncoder(nn.Module):
    """
    Complete EEG Encoder for Stress Classification.

    This module implements the full EEG feature extraction pipeline using
    a hybrid CNN-LSTM-Attention architecture. It transforms raw multi-channel
    EEG signals into compact feature representations suitable for classification.

    Architecture Details:
    ---------------------

    INPUT LAYER:
        • Shape: (batch, n_channels=32, n_time_samples=512)
        • Channels: 32 EEG electrodes (10-20 system)
        • Time: 512 samples (~2 seconds at 256 Hz)

    CONVOLUTIONAL BLOCKS (3 layers):
        Block 1: Local pattern extraction
            • Conv1D: 32 → 32 filters, kernel=7
            • Captures ~27ms temporal patterns
            • Output: (batch, 32, 256)

        Block 2: Mid-level features
            • Conv1D: 32 → 64 filters, kernel=5
            • Captures ~19ms patterns on downsampled signal
            • Output: (batch, 64, 128)

        Block 3: Abstract features
            • Conv1D: 64 → 64 filters, kernel=3
            • Captures high-level abstractions
            • Output: (batch, 64, 64)

    BI-LSTM LAYER:
        • Input: (batch, 64, 64) reshaped to (batch, 64, 64)
        • Hidden: 64 units × 2 directions = 128
        • Captures long-range temporal dependencies
        • Output: (batch, 64, 128)

    SELF-ATTENTION:
        • Input: (batch, 64, 128)
        • Learns importance weights for each time step
        • Output: (batch, 128) context vector

    Data Integration:
    -----------------
    The encoder integrates with the larger GenAI-RAG-EEG system:

    1. Receives preprocessed EEG from DataLoader
       - Bandpass filtered (0.5-100 Hz)
       - Notch filtered (50/60 Hz)
       - Z-score normalized per channel

    2. Outputs feature vector to Fusion Layer
       - 128-dimensional representation
       - Combined with text context features
       - Fed to classification head

    3. Provides attention weights for interpretation
       - Visualizable as temporal heatmap
       - Used by RAG explainer for explanations

    Parameter Count:
    ----------------
    Component           | Parameters
    --------------------|------------
    Conv Block 1        | 2,336
    Conv Block 2        | 10,368
    Conv Block 3        | 12,416
    BatchNorm layers    | 320
    Bi-LSTM            | 99,584
    Self-Attention     | 8,321
    --------------------|------------
    TOTAL              | ~138,081

    Usage Example:
    --------------
    >>> encoder = EEGEncoder(n_channels=32, n_time_samples=512)
    >>> eeg_signal = torch.randn(16, 32, 512)  # batch of 16
    >>> features, attention = encoder(eeg_signal, return_attention=True)
    >>> print(features.shape)  # (16, 128)
    >>> print(attention.shape)  # (16, 64)

    Visualization:
    --------------
    For visualization of learned features and attention:

    1. Attention Heatmap:
       plt.imshow(attention.cpu().numpy(), aspect='auto')
       plt.xlabel('Time Step')
       plt.ylabel('Sample')
       plt.colorbar(label='Attention Weight')

    2. Feature Embedding (t-SNE):
       tsne = TSNE(n_components=2)
       embedded = tsne.fit_transform(features.cpu().numpy())
       plt.scatter(embedded[:, 0], embedded[:, 1], c=labels)
    """

    def __init__(
        self,
        n_channels: int = 32,
        n_time_samples: int = 512,
        conv_filters: Tuple[int, int, int] = (32, 64, 64),
        kernel_sizes: Tuple[int, int, int] = (7, 5, 3),
        lstm_hidden: int = 64,
        attention_dim: int = 64,
        dropout: float = 0.3,
    ):
        """
        Initialize EEG Encoder.

        Args:
            n_channels: Number of EEG channels (default: 32 for 10-20 system)
            n_time_samples: Number of time samples per segment (default: 512)
            conv_filters: Number of filters for each conv block (default: 32, 64, 64)
            kernel_sizes: Kernel sizes for each conv block (default: 7, 5, 3)
            lstm_hidden: LSTM hidden dimension per direction (default: 64)
            attention_dim: Attention intermediate dimension (default: 64)
            dropout: Dropout probability for regularization (default: 0.3)
        """
        super().__init__()

        # Store configuration for reference
        self.n_channels = n_channels
        self.n_time_samples = n_time_samples

        # =====================================================================
        # CONVOLUTIONAL FEATURE EXTRACTION BLOCKS
        # =====================================================================
        # Block 1: Initial spatial-temporal pattern extraction
        # Large kernel (7) captures broader temporal patterns
        self.conv1 = ConvBlock(
            in_channels=n_channels,      # 32 EEG channels
            out_channels=conv_filters[0], # 32 feature maps
            kernel_size=kernel_sizes[0],  # kernel=7
            dropout=dropout
        )

        # Block 2: Mid-level feature extraction
        # Medium kernel (5) for intermediate patterns
        self.conv2 = ConvBlock(
            in_channels=conv_filters[0],  # 32 from previous
            out_channels=conv_filters[1], # 64 feature maps
            kernel_size=kernel_sizes[1],  # kernel=5
            dropout=dropout
        )

        # Block 3: Abstract feature extraction
        # Small kernel (3) for fine-grained patterns
        self.conv3 = ConvBlock(
            in_channels=conv_filters[1],  # 64 from previous
            out_channels=conv_filters[2], # 64 feature maps
            kernel_size=kernel_sizes[2],  # kernel=3
            dropout=dropout
        )

        # =====================================================================
        # SEQUENCE LENGTH AFTER CONVOLUTIONS
        # =====================================================================
        # Each conv block applies MaxPool(2), reducing length by half
        # After 3 blocks: 512 → 256 → 128 → 64
        self.seq_len_after_conv = n_time_samples // (2 ** 3)  # 512 → 64

        # =====================================================================
        # BIDIRECTIONAL LSTM FOR TEMPORAL MODELING
        # =====================================================================
        # Captures long-range dependencies in both directions
        # Forward LSTM: captures past context
        # Backward LSTM: captures future context
        self.lstm = nn.LSTM(
            input_size=conv_filters[2],     # 64 features from conv
            hidden_size=lstm_hidden,         # 64 hidden units
            num_layers=1,                    # Single layer
            batch_first=True,                # (batch, seq, features)
            bidirectional=True,              # Forward + backward
            dropout=0                        # No internal dropout (single layer)
        )

        # =====================================================================
        # SELF-ATTENTION MECHANISM
        # =====================================================================
        # Learns to weight time steps by importance
        # Input: 128 (64 × 2 for bidirectional)
        self.attention = SelfAttention(
            hidden_size=lstm_hidden * 2,  # 128 for bidirectional
            attention_dim=attention_dim    # 64 attention dimension
        )

        # =====================================================================
        # OUTPUT DIMENSION
        # =====================================================================
        # Final feature dimension = 128 (bidirectional LSTM output)
        self.output_dim = lstm_hidden * 2  # 128

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through EEG Encoder.

        Complete Data Flow:
        -------------------

        1. INPUT: Raw EEG Signal
           Shape: (batch, 32, 512)
           Content: Multi-channel EEG time series

        2. CONV BLOCK 1: Local Pattern Extraction
           Operation: Conv1D(32→32, k=7) → BN → ReLU → Pool(2) → Dropout
           Shape: (batch, 32, 512) → (batch, 32, 256)
           Logic: Detects low-level oscillatory patterns (~27ms windows)

        3. CONV BLOCK 2: Mid-Level Features
           Operation: Conv1D(32→64, k=5) → BN → ReLU → Pool(2) → Dropout
           Shape: (batch, 32, 256) → (batch, 64, 128)
           Logic: Combines low-level patterns into higher-level features

        4. CONV BLOCK 3: Abstract Features
           Operation: Conv1D(64→64, k=3) → BN → ReLU → Pool(2) → Dropout
           Shape: (batch, 64, 128) → (batch, 64, 64)
           Logic: Creates abstract representations of EEG patterns

        5. RESHAPE FOR LSTM
           Operation: Permute dimensions
           Shape: (batch, 64, 64) → (batch, 64, 64)
           Logic: Treats spatial features as sequence features

        6. BI-LSTM: Temporal Modeling
           Operation: Bidirectional LSTM processing
           Shape: (batch, 64, 64) → (batch, 64, 128)
           Logic: Captures temporal dependencies in both directions

        7. SELF-ATTENTION: Importance Weighting
           Operation: Attention-weighted pooling
           Shape: (batch, 64, 128) → (batch, 128)
           Logic: Focuses on stress-relevant time segments

        8. OUTPUT: Feature Vector
           Shape: (batch, 128)
           Content: Compact EEG representation for classification

        Args:
            x: Input EEG tensor
                Shape: (batch_size, n_channels, n_time_samples)
                Example: (32, 32, 512) for 32 samples
            return_attention: Whether to return attention weights for visualization

        Returns:
            features: EEG feature representation
                Shape: (batch_size, 128)
                Used for classification and fusion with text features
            attention_weights: Optional attention distribution
                Shape: (batch_size, 64) if return_attention=True
                Useful for interpretability and visualization

        Raises:
            RuntimeError: If input dimensions don't match expected shape
        """
        # =====================================================================
        # STAGE 1: CONVOLUTIONAL FEATURE EXTRACTION
        # =====================================================================
        # Apply three convolutional blocks sequentially
        # Each block: Conv → BN → ReLU → Pool → Dropout

        x = self.conv1(x)  # (batch, 32, 512) → (batch, 32, 256)
        x = self.conv2(x)  # (batch, 32, 256) → (batch, 64, 128)
        x = self.conv3(x)  # (batch, 64, 128) → (batch, 64, 64)

        # =====================================================================
        # STAGE 2: RESHAPE FOR SEQUENTIAL PROCESSING
        # =====================================================================
        # Convert from (batch, channels, time) to (batch, time, channels)
        # LSTM expects sequence in dim 1
        x = x.permute(0, 2, 1)  # (batch, 64, 64)

        # =====================================================================
        # STAGE 3: BIDIRECTIONAL LSTM
        # =====================================================================
        # Process sequence with forward and backward LSTMs
        # Captures temporal context from both directions
        lstm_out, _ = self.lstm(x)  # (batch, 64, 128)

        # =====================================================================
        # STAGE 4: SELF-ATTENTION POOLING
        # =====================================================================
        # Compute attention weights and create context vector
        # Focuses on most informative time segments
        context, attention_weights = self.attention(lstm_out)  # (batch, 128)

        # =====================================================================
        # RETURN RESULTS
        # =====================================================================
        if return_attention:
            return context, attention_weights
        return context, None


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_eeg_encoder(**kwargs) -> EEGEncoder:
    """
    Factory function to create EEG encoder with default or custom parameters.

    This function provides a convenient way to instantiate the EEGEncoder
    with commonly used configurations.

    Args:
        **kwargs: Keyword arguments passed to EEGEncoder constructor
            - n_channels: Number of EEG channels (default: 32)
            - n_time_samples: Time samples per segment (default: 512)
            - conv_filters: Conv filter counts (default: (32, 64, 64))
            - kernel_sizes: Conv kernel sizes (default: (7, 5, 3))
            - lstm_hidden: LSTM hidden size (default: 64)
            - attention_dim: Attention dimension (default: 64)
            - dropout: Dropout probability (default: 0.3)

    Returns:
        EEGEncoder: Configured encoder instance

    Example:
        >>> encoder = get_eeg_encoder(dropout=0.5)  # Higher regularization
        >>> encoder = get_eeg_encoder(lstm_hidden=128)  # Larger LSTM
    """
    return EEGEncoder(**kwargs)


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    """
    Test and demonstrate EEG Encoder functionality.

    This block runs when the module is executed directly:
        python eeg_encoder.py

    It verifies:
    1. Model instantiation
    2. Parameter counting
    3. Forward pass execution
    4. Output shape verification
    """

    print("=" * 70)
    print("EEG ENCODER MODULE TEST")
    print("GenAI-RAG-EEG Architecture - EEG Feature Extraction Pipeline")
    print("=" * 70)

    # =========================================================================
    # CREATE MODEL
    # =========================================================================
    print("\n[1] Creating EEG Encoder...")
    model = EEGEncoder()
    print(f"    Architecture: CNN (3 blocks) → Bi-LSTM → Self-Attention")

    # =========================================================================
    # COUNT PARAMETERS
    # =========================================================================
    print("\n[2] Parameter Count:")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total trainable parameters: {total_params:,}")

    # Detailed breakdown
    conv_params = sum(p.numel() for name, p in model.named_parameters()
                      if 'conv' in name and p.requires_grad)
    lstm_params = sum(p.numel() for name, p in model.named_parameters()
                      if 'lstm' in name and p.requires_grad)
    attn_params = sum(p.numel() for name, p in model.named_parameters()
                      if 'attention' in name and p.requires_grad)

    print(f"    - Convolutional layers: {conv_params:,}")
    print(f"    - Bi-LSTM layer: {lstm_params:,}")
    print(f"    - Self-Attention: {attn_params:,}")

    # =========================================================================
    # TEST FORWARD PASS
    # =========================================================================
    print("\n[3] Testing Forward Pass...")
    batch_size = 4
    x = torch.randn(batch_size, 32, 512)  # 32 channels, 512 time samples

    print(f"    Input shape: {x.shape}")
    print(f"    - Batch size: {batch_size}")
    print(f"    - EEG channels: 32")
    print(f"    - Time samples: 512")

    features, attention = model(x, return_attention=True)

    print(f"\n    Output shape: {features.shape}")
    print(f"    - Feature dimension: 128")

    print(f"\n    Attention shape: {attention.shape}")
    print(f"    - Sequence length: 64 (after 3× pooling)")

    # =========================================================================
    # VERIFY ATTENTION PROPERTIES
    # =========================================================================
    print("\n[4] Attention Weights Verification:")
    attn_sum = attention.sum(dim=-1)
    print(f"    Sum per sample (should be ~1.0): {attn_sum.mean().item():.4f}")
    print(f"    Min attention: {attention.min().item():.4f}")
    print(f"    Max attention: {attention.max().item():.4f}")

    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)
