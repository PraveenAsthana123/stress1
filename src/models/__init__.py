"""
GenAI-RAG-EEG Models Package.

Contains the core model components:
- EEGEncoder: 1D-CNN + Bi-LSTM + Attention for EEG feature extraction
- TextContextEncoder: Sentence-BERT based context encoding
- RAGExplainer: Retrieval-augmented explanation generation
- GenAIRAGEEG: Complete hybrid architecture
"""

from .eeg_encoder import EEGEncoder, SelfAttention, get_eeg_encoder
from .text_encoder import TextContextEncoder, create_context_string
from .rag_pipeline import RAGExplainer, create_default_rag_explainer, Document
from .genai_rag_eeg import GenAIRAGEEG, FusionLayer, ClassificationHead, create_model

__all__ = [
    # EEG Encoder
    "EEGEncoder",
    "SelfAttention",
    "get_eeg_encoder",
    # Text Encoder
    "TextContextEncoder",
    "create_context_string",
    # RAG Pipeline
    "RAGExplainer",
    "create_default_rag_explainer",
    "Document",
    # Complete Model
    "GenAIRAGEEG",
    "FusionLayer",
    "ClassificationHead",
    "create_model",
]
