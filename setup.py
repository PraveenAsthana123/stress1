#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GenAI-RAG-EEG: Setup Script for Package Installation
================================================================================

Installation:
    # Standard installation
    pip install .

    # Development installation (editable)
    pip install -e .

    # With optional dependencies
    pip install -e ".[dev,docs,gpu]"

================================================================================
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
def read_requirements(filename):
    """Read requirements from file, filtering comments and empty lines."""
    requirements = []
    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-"):
                    # Remove inline comments
                    if "#" in line:
                        line = line.split("#")[0].strip()
                    requirements.append(line)
    except FileNotFoundError:
        pass
    return requirements


# Core dependencies
INSTALL_REQUIRES = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.2.0",
    "transformers>=4.30.0",
    "sentence-transformers>=2.2.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "tqdm>=4.65.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
]

# Optional dependencies
EXTRAS_REQUIRE = {
    # Development dependencies
    "dev": [
        "pytest>=7.3.0",
        "pytest-cov>=4.1.0",
        "black>=23.3.0",
        "flake8>=6.0.0",
        "mypy>=1.3.0",
        "isort>=5.12.0",
        "pre-commit>=3.3.0",
    ],
    # Documentation dependencies
    "docs": [
        "sphinx>=6.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "myst-parser>=1.0.0",
    ],
    # GPU support
    "gpu": [
        "faiss-gpu>=1.7.4",
    ],
    # Full RAG support
    "rag": [
        "faiss-cpu>=1.7.4",
        "chromadb>=0.4.0",
        "openai>=1.0.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
    ],
    # EEG processing
    "eeg": [
        "mne>=1.4.0",
        "pyedflib>=0.1.30",
    ],
    # Web application
    "web": [
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "streamlit>=1.28.0",
    ],
    # Document generation (PDF, Word, PowerPoint)
    "docs-export": [
        "python-docx>=0.8.11",
        "python-pptx>=0.6.21",
        "reportlab>=4.0.0",
    ],
    # Experiment tracking
    "tracking": [
        "wandb>=0.15.0",
        "tensorboard>=2.13.0",
    ],
}

# All optional dependencies
EXTRAS_REQUIRE["all"] = list(set(
    dep for deps in EXTRAS_REQUIRE.values() for dep in deps
))

setup(
    # =========================================================================
    # PACKAGE METADATA
    # =========================================================================
    name="genai-rag-eeg",
    version="3.0.0",
    author="[Your Name]",
    author_email="[your.email@institution.edu]",
    description="Explainable EEG-Based Stress Classification using GenAI-RAG Architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/genai-rag-eeg",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/genai-rag-eeg/issues",
        "Documentation": "https://genai-rag-eeg.readthedocs.io/",
        "Source Code": "https://github.com/yourusername/genai-rag-eeg",
    },
    license="MIT",

    # =========================================================================
    # PACKAGE CONFIGURATION
    # =========================================================================
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },

    # =========================================================================
    # DEPENDENCIES
    # =========================================================================
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,

    # =========================================================================
    # ENTRY POINTS
    # =========================================================================
    entry_points={
        "console_scripts": [
            "genai-rag-eeg=main:main",
            "eeg-train=scripts.train:main",
            "eeg-evaluate=scripts.evaluate:main",
            "eeg-demo=scripts.demo:main",
        ],
    },

    # =========================================================================
    # CLASSIFIERS
    # =========================================================================
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],

    # =========================================================================
    # KEYWORDS
    # =========================================================================
    keywords=[
        "eeg",
        "stress-classification",
        "deep-learning",
        "explainable-ai",
        "rag",
        "retrieval-augmented-generation",
        "attention-mechanism",
        "cnn-lstm",
        "brain-computer-interface",
        "affective-computing",
    ],
)
