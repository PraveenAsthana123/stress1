#!/usr/bin/env python3
"""
GenAI-RAG-EEG Configuration File
=================================
IMPORTANT: Change the paths below to match YOUR system before running any scripts.

This is the ONLY file you need to edit to run the project on your machine.

Authors: Praveen Asthana, Rajveer Singh Lalawat, Sarita Singh Gond
Version: 6.0.0
Python: 3.7+ (Windows/Linux/Mac compatible)
"""

import os
from pathlib import Path

# ==============================================================================
# PROJECT ROOT PATH
# ==============================================================================
# Change this to your project folder path
# Examples:
#   Windows: r"C:\Users\YourName\Documents\eeg-stress-rag"
#   Linux:   "/home/yourname/eeg-stress-rag"
#   Mac:     "/Users/yourname/eeg-stress-rag"

PROJECT_ROOT = Path(os.environ.get(
    "EEG_PROJECT_ROOT",
    str(Path(__file__).parent.resolve())
))

# ==============================================================================
# DATA PATHS - CHANGE THESE TO YOUR DATA LOCATIONS
# ==============================================================================
# Main data directory (contains SAM40/ and EEGMAT/ folders)
DATA_DIR = Path(os.environ.get(
    "EEG_DATA_DIR",
    str(PROJECT_ROOT / "data")
))

# SAM-40 Dataset
# Download from: https://doi.org/10.6084/m9.figshare.13589538
# After download, place .mat files in this folder
SAM40_DIR = Path(os.environ.get(
    "EEG_SAM40_DIR",
    str(DATA_DIR / "SAM40" / "filtered_data")
))

# EEGMAT Dataset
# Download from: https://physionet.org/content/eegmat/1.0.0/
# After download, place .edf files in this folder
EEGMAT_DIR = Path(os.environ.get(
    "EEG_EEGMAT_DIR",
    str(DATA_DIR / "EEGMAT" / "eeg-during-mental-arithmetic-tasks-1.0.0")
))

# ==============================================================================
# OUTPUT PATHS
# ==============================================================================
RESULTS_DIR = Path(os.environ.get(
    "EEG_RESULTS_DIR",
    str(PROJECT_ROOT / "results")
))

PAPER_DIR = Path(os.environ.get(
    "EEG_PAPER_DIR",
    str(PROJECT_ROOT / "paper")
))

FIGURES_DIR = Path(os.environ.get(
    "EEG_FIGURES_DIR",
    str(PROJECT_ROOT / "figures_extracted")
))

CHECKPOINTS_DIR = Path(os.environ.get(
    "EEG_CHECKPOINTS_DIR",
    str(PROJECT_ROOT / "checkpoints")
))

LOGS_DIR = Path(os.environ.get(
    "EEG_LOGS_DIR",
    str(PROJECT_ROOT / "logs")
))

# ==============================================================================
# DATASET PARAMETERS (DO NOT CHANGE unless using different datasets)
# ==============================================================================
# SAM-40 Dataset Parameters
SAM40_SAMPLING_RATE = 128       # Hz
SAM40_N_CHANNELS = 32           # EEG channels
SAM40_SEGMENT_DURATION = 25     # seconds
SAM40_N_SAMPLES = SAM40_SAMPLING_RATE * SAM40_SEGMENT_DURATION  # 3200
SAM40_CLASSES = ["Arithmetic", "Mirror_image", "Stroop", "Relax"]
SAM40_STRESS_CLASSES = ["Arithmetic", "Mirror_image", "Stroop"]  # Binary: stress=1
SAM40_N_SUBJECTS = 40

# EEGMAT Dataset Parameters
EEGMAT_SAMPLING_RATE = 500      # Hz
EEGMAT_N_CHANNELS = 21          # EEG channels (original, resampled to 32)
EEGMAT_SEGMENT_DURATION = 60    # seconds
EEGMAT_N_SAMPLES = EEGMAT_SAMPLING_RATE * EEGMAT_SEGMENT_DURATION  # 30000
EEGMAT_CLASSES = ["Baseline", "Mental_Arithmetic"]
EEGMAT_N_SUBJECTS = 36

# ==============================================================================
# MODEL HYPERPARAMETERS
# ==============================================================================
# Ensemble Model Configuration
RF_N_ESTIMATORS = 500           # Random Forest trees
RF_MAX_DEPTH = 15               # Random Forest max depth
GB_N_ESTIMATORS = 300           # Gradient Boosting trees
GB_MAX_DEPTH = 5                # Gradient Boosting max depth
SVM_KERNEL = "rbf"              # SVM kernel type
SVM_C = 10                      # SVM regularization parameter

# Training Configuration
N_FOLDS = 5                     # Stratified K-Fold cross-validation
RANDOM_SEED = 42                # Reproducibility seed
TEST_SIZE = 0.2                 # Hold-out test set fraction

# Feature Extraction
N_FEATURES_SELECT = 100         # Top features via NMI selection
SEGMENT_WINDOW = 4              # seconds (for windowed segmentation)
SEGMENT_OVERLAP = 0.5           # 50% overlap

# Preprocessing
BANDPASS_LOW = 0.5              # Hz (high-pass cutoff)
BANDPASS_HIGH = 45.0            # Hz (low-pass cutoff)
BANDPASS_ORDER = 4              # Butterworth filter order
NOTCH_FREQ = 50.0               # Hz (power line, 50 for EU/India, 60 for US)
NOTCH_Q = 30                    # Notch filter quality factor

# Figure Generation
DPI = 300                       # Publication quality

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def setup_directories():
    """Create all output directories if they don't exist."""
    for d in [RESULTS_DIR, PAPER_DIR, FIGURES_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def validate_data_paths():
    """Check if data directories exist and print status."""
    print("=" * 60)
    print("DATA PATH VALIDATION")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"  Exists: {PROJECT_ROOT.exists()}")
    print()
    print(f"SAM-40 Data:  {SAM40_DIR}")
    if SAM40_DIR.exists():
        mat_files = list(SAM40_DIR.glob("*.mat"))
        print(f"  Exists: True ({len(mat_files)} .mat files found)")
    else:
        print(f"  Exists: False")
        print(f"  ACTION: Download SAM-40 dataset and place .mat files here")
    print()
    print(f"EEGMAT Data:  {EEGMAT_DIR}")
    if EEGMAT_DIR.exists():
        edf_files = list(EEGMAT_DIR.glob("*.edf"))
        print(f"  Exists: True ({len(edf_files)} .edf files found)")
    else:
        print(f"  Exists: False")
        print(f"  ACTION: Download EEGMAT dataset and place .edf files here")
    print()
    print(f"Results Dir:  {RESULTS_DIR}")
    print(f"  Exists: {RESULTS_DIR.exists()}")
    print(f"Paper Dir:    {PAPER_DIR}")
    print(f"  Exists: {PAPER_DIR.exists()}")
    print("=" * 60)

    return SAM40_DIR.exists() or EEGMAT_DIR.exists()


def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("GenAI-RAG-EEG Configuration")
    print("=" * 60)
    print(f"PROJECT_ROOT:    {PROJECT_ROOT}")
    print(f"DATA_DIR:        {DATA_DIR}")
    print(f"SAM40_DIR:       {SAM40_DIR}")
    print(f"EEGMAT_DIR:      {EEGMAT_DIR}")
    print(f"RESULTS_DIR:     {RESULTS_DIR}")
    print(f"PAPER_DIR:       {PAPER_DIR}")
    print()
    print("Model: Ensemble (RF + GB + SVM)")
    print(f"  RF: {RF_N_ESTIMATORS} trees, depth={RF_MAX_DEPTH}")
    print(f"  GB: {GB_N_ESTIMATORS} trees, depth={GB_MAX_DEPTH}")
    print(f"  SVM: kernel={SVM_KERNEL}, C={SVM_C}")
    print(f"  CV: {N_FOLDS}-fold stratified")
    print(f"  Seed: {RANDOM_SEED}")
    print()
    print("Preprocessing:")
    print(f"  Bandpass: {BANDPASS_LOW}-{BANDPASS_HIGH} Hz (order {BANDPASS_ORDER})")
    print(f"  Notch: {NOTCH_FREQ} Hz (Q={NOTCH_Q})")
    print(f"  Features: top {N_FEATURES_SELECT} via NMI")
    print("=" * 60)


# ==============================================================================
# RUN VALIDATION ON IMPORT (optional)
# ==============================================================================
if __name__ == "__main__":
    print_config()
    print()
    validate_data_paths()
