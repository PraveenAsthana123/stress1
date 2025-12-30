#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GenAI-RAG-EEG: Complete Architecture Implementation
================================================================================

Title: Explainable EEG-Based Stress Classification using GenAI-RAG
Authors: [Your Name], [Co-author Names]
Institution: [Your Institution]
Contact: [Your Email]

Description:
    This module implements the complete GenAI-RAG-EEG architecture, a hybrid
    deep learning system for explainable EEG-based stress classification.
    The model combines multi-modal feature fusion with retrieval-augmented
    generation for interpretable predictions.

Architecture Overview:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      GenAI-RAG-EEG ARCHITECTURE                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │     EEG Signal                           Context Text                   │
    │    (32 × 512)                        "Task: Stroop..."                 │
    │         │                                    │                          │
    │         ▼                                    ▼                          │
    │  ┌─────────────┐                    ┌─────────────┐                    │
    │  │ EEG ENCODER │                    │TEXT ENCODER │                    │
    │  │ CNN+LSTM+   │                    │ SBERT +     │                    │
    │  │ Attention   │                    │ Projection  │                    │
    │  │ (~138K)     │                    │ (~49K)      │                    │
    │  └──────┬──────┘                    └──────┬──────┘                    │
    │         │ (128-dim)                        │ (128-dim)                 │
    │         │                                  │                           │
    │         └──────────────┬───────────────────┘                           │
    │                        │                                               │
    │                        ▼                                               │
    │               ┌─────────────────┐                                      │
    │               │  FUSION LAYER   │                                      │
    │               │  Concatenation  │                                      │
    │               │  or Attention   │                                      │
    │               │  (~16K)         │                                      │
    │               └────────┬────────┘                                      │
    │                        │ (128-dim)                                     │
    │                        ▼                                               │
    │               ┌─────────────────┐                                      │
    │               │ CLASSIFICATION  │                                      │
    │               │     HEAD        │                                      │
    │               │ FC → FC → Out   │                                      │
    │               │ (~10K)          │                                      │
    │               └────────┬────────┘                                      │
    │                        │                                               │
    │         ┌──────────────┼──────────────┐                               │
    │         │              │              │                               │
    │         ▼              ▼              ▼                               │
    │    [Prediction]   [Confidence]   [Attention]                          │
    │     Stress/No      0.0-1.0       Weights                              │
    │                        │                                               │
    │                        ▼                                               │
    │               ┌─────────────────┐                                      │
    │               │  RAG EXPLAINER  │                                      │
    │               │ Retrieval +     │                                      │
    │               │ Generation      │                                      │
    │               └────────┬────────┘                                      │
    │                        │                                               │
    │                        ▼                                               │
    │               [Natural Language                                        │
    │                Explanation]                                            │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

Components:
    1. EEG Encoder (eeg_encoder.py)
       - 3-layer 1D-CNN for spatial-temporal features
       - Bidirectional LSTM for sequence modeling
       - Self-attention for temporal importance weighting
       - Output: 128-dimensional feature vector

    2. Text Context Encoder (text_encoder.py)
       - Frozen Sentence-BERT (all-MiniLM-L6-v2)
       - Trainable projection layer (384 → 128)
       - Encodes task, demographic, and environmental context

    3. Fusion Layer
       - Combines EEG and text modalities
       - Supports: concatenation, attention, gated fusion
       - Output: 128-dimensional fused representation

    4. Classification Head
       - Multi-layer perceptron (128 → 64 → 32 → 2)
       - Binary stress classification
       - Dropout regularization

    5. RAG Explainer (rag_pipeline.py)
       - Retrieves relevant scientific evidence
       - Generates natural language explanations
       - Grounds predictions in literature

Data Flow:
    1. EEG signal (32 channels × 512 samples) → EEG Encoder → 128-dim features
    2. Context text → Text Encoder → 128-dim features
    3. EEG + Text features → Fusion Layer → 128-dim fused features
    4. Fused features → Classifier → logits + probabilities
    5. Prediction + features → RAG Explainer → explanation

Parameter Count (Actual Implementation):
    Component          | Trainable | Frozen
    -------------------|-----------|--------
    EEG Encoder        | 105,056   | 0
    Text Encoder       | 49,280    | 22.7M
    Fusion Layer       | 32,896    | 0
    Classification     | 10,402    | 0
    -------------------|-----------|--------
    Total              | 197,634   | 22.7M

    EEG-only variant: 131,970 trainable params (encoder + fusion + classifier)

    Note: Paper v2 Table XIX reports different values (138,081 for EEG encoder).
    Discrepancy due to internal paper inconsistencies in LSTM input dimensions.

Performance (10-fold CV):
    Dataset  | Accuracy | F1-Score | AUC
    ---------|----------|----------|------
    DEAP     | 94.7%    | 0.948    | 0.982
    SAM-40   | 81.9%    | 0.835    | 0.891
    EEGMAT    | 100.0%   | 1.000    | 1.000

Usage:
    >>> from src.models import GenAIRAGEEG, create_model
    >>> model = create_model(n_channels=32, use_text=True, use_rag=True)
    >>> output = model(eeg_tensor, context_text=["Task: Stroop. Age: 25."])
    >>> print(output["probs"])  # Stress probability

References:
    [1] "GenAI-RAG-EEG: A Novel Hybrid Deep Learning Architecture for
        Explainable EEG-Based Stress Classification using Generative AI
        and Retrieval-Augmented Generation," IEEE Sensors Journal, 2024.
    [2] Koelstra et al., "DEAP: A Database for Emotion Analysis using
        Physiological Signals," IEEE Trans. Affective Computing, 2012.
    [3] Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive
        NLP Tasks," NeurIPS 2020.

License: MIT License
Version: 1.0.0
Last Updated: 2024
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any
import numpy as np

from .eeg_encoder import EEGEncoder
from .text_encoder import TextContextEncoder, create_context_string
from .rag_pipeline import RAGExplainer, create_default_rag_explainer


class FusionLayer(nn.Module):
    """
    Multimodal fusion layer for combining EEG and text features.

    Supports multiple fusion strategies:
    - concatenation: Simple concatenation
    - attention: Cross-modal attention
    - gated: Gated fusion with learned weights
    """

    def __init__(
        self,
        eeg_dim: int = 128,
        text_dim: int = 128,
        output_dim: int = 128,
        fusion_type: str = "concatenation",
        dropout: float = 0.3
    ):
        super().__init__()

        self.fusion_type = fusion_type
        self.eeg_dim = eeg_dim
        self.text_dim = text_dim
        self.output_dim = output_dim

        if fusion_type == "concatenation":
            self.projection = nn.Sequential(
                nn.Linear(eeg_dim + text_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        elif fusion_type == "attention":
            self.eeg_proj = nn.Linear(eeg_dim, output_dim)
            self.text_proj = nn.Linear(text_dim, output_dim)
            self.attention = nn.Linear(output_dim * 2, 2)
        elif fusion_type == "gated":
            self.gate = nn.Sequential(
                nn.Linear(eeg_dim + text_dim, 2),
                nn.Softmax(dim=-1)
            )
            self.eeg_proj = nn.Linear(eeg_dim, output_dim)
            self.text_proj = nn.Linear(text_dim, output_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(
        self,
        eeg_features: torch.Tensor,
        text_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse EEG and text features.

        Args:
            eeg_features: (batch, eeg_dim)
            text_features: (batch, text_dim), optional

        Returns:
            fused_features: (batch, output_dim)
        """
        if text_features is None:
            # EEG-only mode
            if self.fusion_type == "concatenation":
                # Pad with zeros for text
                text_features = torch.zeros(
                    eeg_features.size(0), self.text_dim,
                    device=eeg_features.device
                )
                combined = torch.cat([eeg_features, text_features], dim=-1)
                return self.projection(combined)
            else:
                return eeg_features

        if self.fusion_type == "concatenation":
            combined = torch.cat([eeg_features, text_features], dim=-1)
            return self.projection(combined)

        elif self.fusion_type == "attention":
            eeg_proj = self.eeg_proj(eeg_features)
            text_proj = self.text_proj(text_features)
            combined = torch.cat([eeg_proj, text_proj], dim=-1)
            weights = F.softmax(self.attention(combined), dim=-1)
            return weights[:, 0:1] * eeg_proj + weights[:, 1:2] * text_proj

        elif self.fusion_type == "gated":
            combined = torch.cat([eeg_features, text_features], dim=-1)
            gates = self.gate(combined)
            eeg_proj = self.eeg_proj(eeg_features)
            text_proj = self.text_proj(text_features)
            return gates[:, 0:1] * eeg_proj + gates[:, 1:2] * text_proj


class ClassificationHead(nn.Module):
    """
    Classification head for stress prediction.

    Architecture:
    - FC1: 128 -> 64
    - FC2: 64 -> 32
    - Output: 32 -> n_classes

    Parameters: 10,402 (binary classification)
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: Tuple[int, ...] = (64, 32),
        n_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)

        Returns:
            logits: (batch, n_classes)
        """
        return self.classifier(x)


class GenAIRAGEEG(nn.Module):
    """
    GenAI-RAG-EEG: Complete model for explainable EEG stress classification.

    Reference: Paper v2, IEEE Sensors Journal 2024

    Components (Actual Implementation):
    1. EEG Encoder: 1D-CNN + Bi-LSTM + Attention (105,056 params)
    2. Text Encoder: Sentence-BERT + Projection (49,280 trainable, 22.7M frozen)
    3. Fusion Layer: Multimodal feature combination (32,896 params)
    4. Classification Head: Stress prediction (10,402 params)
    5. RAG Explainer: Evidence-based explanation generation

    Total Trainable Parameters: 197,634 (with text encoder)
    Total Trainable Parameters: 131,970 (EEG-only variant)
    """

    def __init__(
        self,
        n_channels: int = 32,
        n_time_samples: int = 512,
        n_classes: int = 2,
        use_text_encoder: bool = True,
        use_rag: bool = True,
        fusion_type: str = "concatenation",
        dropout: float = 0.3,
        device: Optional[str] = None
    ):
        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_classes = n_classes
        self.use_text_encoder = use_text_encoder
        self.use_rag = use_rag

        # EEG Encoder
        self.eeg_encoder = EEGEncoder(
            n_channels=n_channels,
            n_time_samples=n_time_samples,
            dropout=dropout
        )

        # Text Context Encoder (optional)
        if use_text_encoder:
            self.text_encoder = TextContextEncoder(device=self.device)
        else:
            self.text_encoder = None

        # Fusion Layer
        self.fusion = FusionLayer(
            eeg_dim=128,
            text_dim=128 if use_text_encoder else 0,
            output_dim=128,
            fusion_type=fusion_type,
            dropout=dropout
        )

        # Classification Head
        self.classifier = ClassificationHead(
            input_dim=128,
            hidden_dims=(64, 32),
            n_classes=n_classes,
            dropout=dropout
        )

        # RAG Explainer (optional, not part of trainable parameters)
        if use_rag:
            self.rag_explainer = create_default_rag_explainer(device=self.device)
        else:
            self.rag_explainer = None

    def forward(
        self,
        eeg: torch.Tensor,
        context_text: Optional[List[str]] = None,
        return_attention: bool = False,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            eeg: EEG input (batch, n_channels, n_time_samples)
            context_text: Optional list of context strings
            return_attention: Whether to return attention weights
            return_features: Whether to return intermediate features

        Returns:
            Dictionary with:
            - logits: (batch, n_classes)
            - probs: (batch, n_classes)
            - attention_weights: (batch, seq_len) if return_attention
            - features: dict of intermediate features if return_features
        """
        # EEG encoding
        eeg_features, attention_weights = self.eeg_encoder(eeg, return_attention=True)

        # Text encoding (if available)
        text_features = None
        if self.use_text_encoder and context_text is not None:
            text_features = self.text_encoder(texts=context_text)
            # Ensure text features are on the same device as EEG features
            text_features = text_features.to(eeg_features.device)

        # Fusion
        fused_features = self.fusion(eeg_features, text_features)

        # Classification
        logits = self.classifier(fused_features)
        probs = F.softmax(logits, dim=-1)

        output = {
            "logits": logits,
            "probs": probs
        }

        if return_attention:
            output["attention_weights"] = attention_weights

        if return_features:
            output["features"] = {
                "eeg": eeg_features,
                "text": text_features,
                "fused": fused_features
            }

        return output

    def predict(
        self,
        eeg: torch.Tensor,
        context_text: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Get predictions.

        Args:
            eeg: EEG input
            context_text: Optional context

        Returns:
            predictions: (batch,) class predictions
        """
        output = self.forward(eeg, context_text)
        return torch.argmax(output["probs"], dim=-1)

    def predict_with_explanation(
        self,
        eeg: torch.Tensor,
        context_text: Optional[List[str]] = None,
        eeg_features: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get predictions with RAG-generated explanations.

        Args:
            eeg: EEG input
            context_text: Optional context
            eeg_features: Optional EEG feature dict for explanation

        Returns:
            List of explanation dictionaries for each sample
        """
        if not self.use_rag or self.rag_explainer is None:
            raise RuntimeError("RAG is not enabled for this model")

        output = self.forward(eeg, context_text, return_attention=True)

        predictions = torch.argmax(output["probs"], dim=-1)
        confidences = torch.max(output["probs"], dim=-1).values
        attention = output.get("attention_weights")

        explanations = []
        for i in range(eeg.size(0)):
            pred = predictions[i].item()
            conf = confidences[i].item()
            attn = attention[i].detach().cpu().numpy() if attention is not None else None

            # Use provided features or generate placeholder
            features = eeg_features if eeg_features else {
                "alpha_power": 0.5,
                "beta_power": 0.5,
                "theta_power": 0.5
            }

            explanation = self.rag_explainer.generate_explanation(
                prediction=pred,
                confidence=conf,
                eeg_features=features,
                attention_weights=attn,
                context=context_text[i] if context_text else None
            )
            explanations.append(explanation)

        return explanations

    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count by component."""
        counts = {}

        # EEG encoder
        eeg_params = sum(p.numel() for p in self.eeg_encoder.parameters() if p.requires_grad)
        counts["eeg_encoder"] = eeg_params

        # Text encoder (only projection is trainable)
        if self.text_encoder:
            text_params = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
            counts["text_encoder"] = text_params

        # Fusion layer
        fusion_params = sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)
        counts["fusion"] = fusion_params

        # Classifier
        classifier_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        counts["classifier"] = classifier_params

        counts["total"] = sum(counts.values())

        return counts


def create_model(
    n_channels: int = 32,
    n_time_samples: int = 512,
    n_classes: int = 2,
    use_text: bool = True,
    use_rag: bool = True,
    device: Optional[str] = None
) -> GenAIRAGEEG:
    """Factory function to create GenAI-RAG-EEG model."""
    model = GenAIRAGEEG(
        n_channels=n_channels,
        n_time_samples=n_time_samples,
        n_classes=n_classes,
        use_text_encoder=use_text,
        use_rag=use_rag,
        device=device
    )
    return model.to(device if device else "cpu")


if __name__ == "__main__":
    print("=" * 60)
    print("GenAI-RAG-EEG Model Test")
    print("=" * 60)

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(device=device)

    # Print parameter counts
    print("\nParameter Counts:")
    for component, count in model.get_parameter_count().items():
        print(f"  {component}: {count:,}")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    eeg = torch.randn(batch_size, 32, 512).to(device)
    context = [
        create_context_string("Stroop", 25, "M"),
        create_context_string("Arithmetic", 30, "F"),
        create_context_string("Mirror Tracing", 22, "M"),
        create_context_string("Rest", 28, "F")
    ]

    output = model(eeg, context, return_attention=True, return_features=True)

    print(f"\nInput EEG shape: {eeg.shape}")
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"Output probs shape: {output['probs'].shape}")
    print(f"Attention weights shape: {output['attention_weights'].shape}")

    # Test prediction with explanation
    print("\nTesting prediction with explanation...")
    explanations = model.predict_with_explanation(
        eeg[:1],
        context[:1],
        eeg_features={
            "alpha_power": 0.32,
            "beta_power": 0.71,
            "theta_power": 0.58
        }
    )

    print(f"\nPrediction: {explanations[0]['prediction_label']}")
    print(f"Confidence: {explanations[0]['confidence']:.1%}")
    print(f"\nExplanation:\n{explanations[0]['explanation']}")
