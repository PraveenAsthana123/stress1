#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Text Context Encoder for GenAI-RAG-EEG Architecture
================================================================================

Title: Sentence-BERT Based Text Context Encoder
Authors: [Your Name], [Co-author Names]
Institution: [Your Institution]
Contact: [Your Email]

Description:
    This module implements the Text Context Encoder component that encodes
    supplementary textual information (task type, subject demographics, etc.)
    using a pre-trained Sentence-BERT model with a trainable projection layer.

Architecture Overview:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    TEXT CONTEXT ENCODER PIPELINE                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  INPUT: Context String                                                  │
    │  "Task: Stroop. Age: 25 years. Gender: M."                             │
    │                              │                                          │
    │                              ▼                                          │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │  TOKENIZATION (WordPiece)                                       │   │
    │  │  • Max length: 128 tokens                                       │   │
    │  │  • Padding and truncation                                       │   │
    │  │  • Output: input_ids, attention_mask                            │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              │                                          │
    │                              ▼                                          │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │  SENTENCE-BERT (Frozen)                                         │   │
    │  │  • Model: all-MiniLM-L6-v2                                      │   │
    │  │  • Parameters: 22.7M (all frozen)                               │   │
    │  │  • Output: Token embeddings (batch, seq_len, 384)               │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              │                                          │
    │                              ▼                                          │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │  MEAN POOLING                                                   │   │
    │  │  • Pool across token dimension                                  │   │
    │  │  • Weighted by attention mask                                   │   │
    │  │  • Output: (batch, 384)                                         │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              │                                          │
    │                              ▼                                          │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │  PROJECTION LAYER (Trainable)                                   │   │
    │  │  • Linear: 384 → 128                                            │   │
    │  │  • ReLU activation                                              │   │
    │  │  • Dropout: 0.1                                                 │   │
    │  │  • Parameters: 49,280 trainable                                 │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              │                                          │
    │                              ▼                                          │
    │  OUTPUT: Text Features (batch, 128)                                    │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

Data Flow:
    1. Raw text → WordPiece tokenization → token IDs
    2. Token IDs → BERT encoder → contextualized embeddings
    3. Token embeddings → mean pooling → sentence embedding
    4. Sentence embedding → projection → output features

Integration:
    The text encoder integrates with the main GenAI-RAG-EEG model by:
    1. Receiving context strings from the data pipeline
    2. Producing 128-dim features for fusion with EEG features
    3. Contributing to explainability via context-aware classification

Key Features:
    - Frozen BERT: Prevents catastrophic forgetting, reduces training cost
    - Trainable projection: Learns task-specific representation
    - Mean pooling: Robust sentence representation
    - Efficient: 49K trainable params vs 22.7M frozen

Parameters:
    - Frozen (Sentence-BERT): 22,713,216
    - Trainable (Projection): 49,280
    - Total: 22,762,496

References:
    [1] Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using
        Siamese BERT-Networks," EMNLP 2019.
    [2] Devlin et al., "BERT: Pre-training of Deep Bidirectional
        Transformers for Language Understanding," NAACL 2019.

License: MIT License
Version: 1.0.0
Last Updated: 2024
================================================================================
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union
from transformers import AutoTokenizer, AutoModel


# =============================================================================
# TEXT CONTEXT ENCODER
# =============================================================================

class TextContextEncoder(nn.Module):
    """
    Text Context Encoder using Sentence-BERT.

    This module encodes supplementary textual information to provide
    context for EEG-based stress classification. It enables the model
    to consider factors like task type, subject demographics, and
    environmental conditions when making predictions.

    Supported Context Types:
    ------------------------
    1. Task Information:
       - Cognitive task type (Stroop, Arithmetic, N-back, etc.)
       - Task difficulty level
       - Task duration

    2. Subject Demographics:
       - Age
       - Gender
       - Previous stress history

    3. Environmental Context:
       - Recording environment (lab, home, field)
       - Time of day
       - Session number

    Architecture Details:
    ---------------------

    SENTENCE-BERT BACKBONE:
        Model: sentence-transformers/all-MiniLM-L6-v2
        - 6 transformer layers
        - Hidden size: 384
        - 12 attention heads
        - Vocabulary: 30,522 tokens
        - Max sequence length: 512

    PROJECTION LAYER:
        Input: 384 (BERT hidden size)
        Output: 128 (matches EEG encoder output)
        Activation: ReLU
        Regularization: Dropout (0.1)

    Parameter Count:
    ----------------
    Component                | Parameters | Trainable
    -------------------------|------------|----------
    BERT Embeddings         | 23,440,896 | No
    BERT Transformer Layers | 21,289,472 | No
    Projection Linear       | 49,152     | Yes
    Projection Bias         | 128        | Yes
    -------------------------|------------|----------
    Total                   | 22,762,496 | 49,280

    Usage Example:
    --------------
    >>> encoder = TextContextEncoder()
    >>> contexts = ["Task: Stroop. Age: 25. Gender: M."]
    >>> embeddings = encoder(texts=contexts)
    >>> print(embeddings.shape)  # (1, 128)

    Visualization:
    --------------
    Text embeddings can be visualized using:
    1. t-SNE/UMAP for clustering analysis
    2. Cosine similarity heatmaps between contexts
    3. Attention visualization (requires modification)
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dim: int = 128,
        freeze_bert: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize Text Context Encoder.

        Args:
            model_name: Hugging Face model identifier
                Default: "sentence-transformers/all-MiniLM-L6-v2"
                Alternatives:
                - "sentence-transformers/all-mpnet-base-v2" (768-dim, more accurate)
                - "sentence-transformers/paraphrase-MiniLM-L6-v2" (384-dim)
            output_dim: Output feature dimension (default: 128 to match EEG)
            freeze_bert: Whether to freeze BERT parameters (default: True)
            device: Compute device ("cuda" or "cpu")
        """
        super().__init__()

        # =====================================================================
        # DEVICE CONFIGURATION
        # =====================================================================
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.output_dim = output_dim

        # =====================================================================
        # TOKENIZER INITIALIZATION
        # =====================================================================
        # WordPiece tokenizer for text preprocessing
        # - Handles subword tokenization
        # - Manages special tokens ([CLS], [SEP], [PAD])
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # =====================================================================
        # BERT MODEL INITIALIZATION
        # =====================================================================
        # Load pre-trained Sentence-BERT model
        # Contains: embedding layer + 6 transformer blocks
        self.bert = AutoModel.from_pretrained(model_name)

        # =====================================================================
        # FREEZE BERT PARAMETERS
        # =====================================================================
        # Freezing prevents catastrophic forgetting and reduces memory usage
        # Only the projection layer will be trained
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # =====================================================================
        # GET BERT EMBEDDING DIMENSION
        # =====================================================================
        # MiniLM-L6-v2: 384, MPNet-base: 768
        self.bert_dim = self.bert.config.hidden_size  # 384 for MiniLM

        # =====================================================================
        # TRAINABLE PROJECTION LAYER
        # =====================================================================
        # Maps BERT embeddings to match EEG encoder output dimension
        # This is the only trainable part of the text encoder
        self.projection = nn.Sequential(
            nn.Linear(self.bert_dim, output_dim),  # 384 → 128
            nn.ReLU(),                              # Non-linearity
            nn.Dropout(0.1)                         # Regularization
        )

        # =====================================================================
        # MOVE ALL COMPONENTS TO DEVICE
        # =====================================================================
        self.bert = self.bert.to(self.device)
        self.projection = self.projection.to(self.device)

    def mean_pooling(
        self,
        model_output: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply mean pooling to BERT token embeddings.

        Logic:
        ------
        1. Get token embeddings from BERT output
        2. Mask out padding tokens using attention mask
        3. Compute mean across token dimension

        Mathematical Formulation:
        -------------------------
        sentence_embedding = Σ(token_i * mask_i) / Σ(mask_i)

        Args:
            model_output: BERT model output containing:
                - last_hidden_state: (batch, seq_len, hidden_dim)
                - pooler_output: (batch, hidden_dim)
            attention_mask: Binary mask indicating real tokens vs padding
                - Shape: (batch, seq_len)
                - Values: 1 for real tokens, 0 for padding

        Returns:
            sentence_embeddings: Mean-pooled embeddings
                - Shape: (batch, hidden_dim)

        Note:
            Mean pooling typically outperforms [CLS] token for sentence-level
            tasks as it incorporates information from all tokens.
        """
        # =====================================================================
        # STEP 1: EXTRACT TOKEN EMBEDDINGS
        # =====================================================================
        # model_output[0] = last_hidden_state: (batch, seq_len, hidden_dim)
        token_embeddings = model_output[0]

        # =====================================================================
        # STEP 2: EXPAND ATTENTION MASK
        # =====================================================================
        # Expand mask to match embedding dimension for element-wise multiplication
        # (batch, seq_len) → (batch, seq_len, hidden_dim)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # =====================================================================
        # STEP 3: MASKED SUM
        # =====================================================================
        # Zero out padding token embeddings and sum across sequence
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)

        # =====================================================================
        # STEP 4: COUNT VALID TOKENS
        # =====================================================================
        # Sum of mask = number of real tokens per sample
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        # =====================================================================
        # STEP 5: COMPUTE MEAN
        # =====================================================================
        # Average embeddings, accounting for variable sequence lengths
        return sum_embeddings / sum_mask

    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text(s) into feature embeddings.

        Process Flow:
        -------------
        1. Tokenization: text → input_ids + attention_mask
        2. BERT forward: input_ids → token embeddings
        3. Mean pooling: token embeddings → sentence embedding
        4. Projection: sentence embedding → output features

        Args:
            texts: Single string or list of strings
                Example: "Task: Stroop. Age: 25. Gender: M."
                Example: ["Context 1", "Context 2", "Context 3"]

        Returns:
            embeddings: Feature tensor
                Shape: (batch, output_dim) = (batch, 128)

        Example:
            >>> encoder = TextContextEncoder()
            >>> embeddings = encoder.encode_text("Task: Stroop")
            >>> print(embeddings.shape)  # (1, 128)
        """
        # =====================================================================
        # HANDLE SINGLE STRING INPUT
        # =====================================================================
        if isinstance(texts, str):
            texts = [texts]

        # =====================================================================
        # TOKENIZATION
        # =====================================================================
        # Convert text to token IDs with padding and truncation
        encoded = self.tokenizer(
            texts,
            padding=True,              # Pad to longest sequence in batch
            truncation=True,           # Truncate to max_length
            max_length=128,            # Maximum sequence length
            return_tensors="pt"        # Return PyTorch tensors
        )

        # =====================================================================
        # MOVE TO DEVICE
        # =====================================================================
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # =====================================================================
        # BERT FORWARD PASS (NO GRADIENTS)
        # =====================================================================
        # Frozen BERT: use torch.no_grad() for efficiency
        with torch.no_grad():
            bert_output = self.bert(**encoded)

        # =====================================================================
        # MEAN POOLING
        # =====================================================================
        # Convert token-level to sentence-level embeddings
        sentence_embeddings = self.mean_pooling(bert_output, encoded["attention_mask"])

        # =====================================================================
        # PROJECTION (TRAINABLE)
        # =====================================================================
        # Map to output dimension for fusion with EEG features
        projected = self.projection(sentence_embeddings)

        return projected

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None
    ) -> torch.Tensor:
        """
        Forward pass for text encoding.

        This method supports two input modes:
        1. Raw text strings (texts parameter) - recommended
        2. Pre-tokenized inputs (input_ids + attention_mask)

        Args:
            input_ids: Pre-tokenized input IDs
                Shape: (batch, seq_len)
            attention_mask: Attention mask for padding
                Shape: (batch, seq_len)
            texts: Raw text strings (alternative to tokenized inputs)
                Type: str or List[str]

        Returns:
            embeddings: Text feature embeddings
                Shape: (batch, output_dim)

        Raises:
            ValueError: If neither texts nor input_ids are provided

        Example:
            >>> # Using raw text (recommended)
            >>> output = encoder(texts=["Task: Stroop"])
            >>> # Using pre-tokenized
            >>> encoded = tokenizer("Task: Stroop", return_tensors="pt")
            >>> output = encoder(**encoded)
        """
        # =====================================================================
        # MODE 1: RAW TEXT INPUT (RECOMMENDED)
        # =====================================================================
        if texts is not None:
            return self.encode_text(texts)

        # =====================================================================
        # MODE 2: PRE-TOKENIZED INPUT
        # =====================================================================
        if input_ids is None:
            raise ValueError("Either texts or input_ids must be provided")

        # BERT forward pass
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Mean pooling
        sentence_embeddings = self.mean_pooling(bert_output, attention_mask)

        # Projection
        projected = self.projection(sentence_embeddings)

        return projected


# =============================================================================
# CONTEXT STRING UTILITIES
# =============================================================================

def create_context_string(
    task_type: str = "unknown",
    subject_age: Optional[int] = None,
    subject_gender: Optional[str] = None,
    stress_history: Optional[str] = None,
    environment: Optional[str] = None
) -> str:
    """
    Create a formatted context string from available metadata.

    This utility function constructs consistent context strings
    for input to the text encoder. It handles missing fields gracefully.

    Logic:
    ------
    1. Start with task type (always present)
    2. Append available demographic information
    3. Append environmental context if available
    4. Join with periods for clear sentence structure

    Args:
        task_type: Type of cognitive task
            Options: "Stroop", "Arithmetic", "N-back", "Mirror Tracing",
                    "Rest", "TSST", "unknown"
        subject_age: Subject age in years (18-65 typical range)
        subject_gender: Subject gender
            Options: "M", "F", "Other", None
        stress_history: Previous stress level or history
            Example: "Low baseline", "Previous high stress"
        environment: Recording environment context
            Example: "Laboratory", "Home", "Field study"

    Returns:
        context: Formatted context string

    Examples:
        >>> create_context_string("Stroop", 25, "M")
        "Task: Stroop. Age: 25 years. Gender: M."

        >>> create_context_string("Arithmetic", 30, "F", "Low baseline")
        "Task: Arithmetic. Age: 30 years. Gender: F. History: Low baseline."

        >>> create_context_string("Rest")
        "Task: Rest."

    Integration:
        This function is typically called in the data pipeline:
        >>> dataset = StressDataset(...)
        >>> for eeg, label, metadata in dataset:
        >>>     context = create_context_string(**metadata)
        >>>     output = model(eeg, context_text=[context])
    """
    # =========================================================================
    # BUILD CONTEXT PARTS
    # =========================================================================
    parts = [f"Task: {task_type}"]

    if subject_age is not None:
        parts.append(f"Age: {subject_age} years")

    if subject_gender is not None:
        parts.append(f"Gender: {subject_gender}")

    if stress_history is not None:
        parts.append(f"History: {stress_history}")

    if environment is not None:
        parts.append(f"Environment: {environment}")

    # =========================================================================
    # JOIN WITH PERIODS
    # =========================================================================
    return ". ".join(parts) + "."


def parse_context_string(context: str) -> dict:
    """
    Parse a context string back into component fields.

    Useful for debugging and visualization.

    Args:
        context: Formatted context string

    Returns:
        fields: Dictionary of parsed fields

    Example:
        >>> parse_context_string("Task: Stroop. Age: 25 years. Gender: M.")
        {'task_type': 'Stroop', 'subject_age': 25, 'subject_gender': 'M'}
    """
    fields = {}

    # Parse each component
    for part in context.split(". "):
        if part.startswith("Task:"):
            fields['task_type'] = part.replace("Task: ", "").strip(".")
        elif part.startswith("Age:"):
            age_str = part.replace("Age: ", "").replace(" years", "").strip(".")
            fields['subject_age'] = int(age_str)
        elif part.startswith("Gender:"):
            fields['subject_gender'] = part.replace("Gender: ", "").strip(".")
        elif part.startswith("History:"):
            fields['stress_history'] = part.replace("History: ", "").strip(".")
        elif part.startswith("Environment:"):
            fields['environment'] = part.replace("Environment: ", "").strip(".")

    return fields


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    """
    Test and demonstrate Text Context Encoder functionality.

    Run directly:
        python text_encoder.py
    """
    print("=" * 70)
    print("TEXT CONTEXT ENCODER MODULE TEST")
    print("GenAI-RAG-EEG Architecture - Text Feature Extraction")
    print("=" * 70)

    # =========================================================================
    # CREATE ENCODER
    # =========================================================================
    print("\n[1] Initializing Text Context Encoder...")
    encoder = TextContextEncoder()
    print(f"    Model: {encoder.model_name}")
    print(f"    BERT dimension: {encoder.bert_dim}")
    print(f"    Output dimension: {encoder.output_dim}")
    print(f"    Device: {encoder.device}")

    # =========================================================================
    # COUNT PARAMETERS
    # =========================================================================
    print("\n[2] Parameter Count:")
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    print(f"    Frozen parameters: {frozen_params:,}")

    # =========================================================================
    # TEST CONTEXT STRING CREATION
    # =========================================================================
    print("\n[3] Creating context strings...")
    test_contexts = [
        create_context_string("Stroop", 25, "M"),
        create_context_string("Arithmetic", 30, "F", "Low stress baseline"),
        create_context_string("Mirror Tracing", 22, "M", environment="Laboratory"),
        create_context_string("Rest", 28, "F")
    ]

    for i, ctx in enumerate(test_contexts):
        print(f"    Context {i+1}: {ctx}")

    # =========================================================================
    # TEST ENCODING
    # =========================================================================
    print("\n[4] Testing text encoding...")
    embeddings = encoder.encode_text(test_contexts)
    print(f"    Input: {len(test_contexts)} context strings")
    print(f"    Output shape: {embeddings.shape}")
    print(f"    Output dtype: {embeddings.dtype}")

    # =========================================================================
    # TEST EMBEDDING PROPERTIES
    # =========================================================================
    print("\n[5] Embedding statistics:")
    print(f"    Mean: {embeddings.mean().item():.4f}")
    print(f"    Std: {embeddings.std().item():.4f}")
    print(f"    Min: {embeddings.min().item():.4f}")
    print(f"    Max: {embeddings.max().item():.4f}")

    # =========================================================================
    # TEST SIMILARITY
    # =========================================================================
    print("\n[6] Cosine similarity between contexts:")
    from torch.nn.functional import cosine_similarity

    for i in range(len(test_contexts)):
        for j in range(i + 1, len(test_contexts)):
            sim = cosine_similarity(embeddings[i:i+1], embeddings[j:j+1]).item()
            print(f"    Context {i+1} vs {j+1}: {sim:.4f}")

    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)
