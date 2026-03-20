# GenAI-RAG-EEG: Explainable EEG-Based Stress Classification

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hybrid machine learning system for **EEG-based stress detection** using ensemble classifiers with **RAG-based explainability**. Achieves **99.31% accuracy** on EEGMAT and **94.79% accuracy** on SAM-40 dataset.

## Key Results

| Dataset | Subjects | Channels | Accuracy | F1-Score | AUC-ROC |
|---------|----------|----------|----------|----------|---------|
| EEGMAT  | 36       | 21       | **99.31%** | 99.15%  | 99.87%  |
| SAM-40  | 40       | 32       | **94.79%** | 92.79%  | 98.49%  |
| Combined| 76       | 32       | **95.83%** | 95.72%  | 98.91%  |

---

## Quick Start (3 Steps)

### Step 1: Clone and Install

```bash
git clone https://github.com/PraveenAsthana123/stress1.git
cd stress1
python -m venv venv

# Linux/Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### Step 2: Configure Data Paths

**Edit `config.py`** - this is the ONLY file you need to change:

```python
# config.py - Line 26-49
# Change these paths to YOUR data locations:

# Windows example:
# SAM40_DIR = Path(r"C:\Users\YourName\data\SAM40\filtered_data")
# EEGMAT_DIR = Path(r"C:\Users\YourName\data\EEGMAT\eeg-during-mental-arithmetic-tasks-1.0.0")

# Linux example:
# SAM40_DIR = Path("/home/yourname/data/SAM40/filtered_data")
# EEGMAT_DIR = Path("/home/yourname/data/EEGMAT/eeg-during-mental-arithmetic-tasks-1.0.0")
```

Or use environment variables (no code changes needed):
```bash
export EEG_SAM40_DIR="/path/to/your/SAM40/filtered_data"
export EEG_EEGMAT_DIR="/path/to/your/EEGMAT/eeg-during-mental-arithmetic-tasks-1.0.0"
```

### Step 3: Run Training

```bash
# Validate your data paths first:
python config.py

# Train SAM-40 (takes ~2 minutes):
python scripts/train_sam40_balanced_v2.py

# Train EEGMAT + Combined (takes ~10 minutes):
python scripts/train_full_data.py

# Generate publication figures (300 DPI):
python scripts/generate_v6_figures.py
```

---

## Dataset Download

| Dataset | Source | Format | Size |
|---------|--------|--------|------|
| SAM-40 | [Figshare DOI:10.6084/m9.figshare.13589538](https://doi.org/10.6084/m9.figshare.13589538) | .mat (MATLAB) | ~500 MB |
| EEGMAT | [PhysioNet: eegmat/1.0.0](https://physionet.org/content/eegmat/1.0.0/) | .edf (EDF) | ~200 MB |

**Sample data** (20 records for testing) is included in `data/sample/`.

---

## Project Structure

```
eeg-stress-rag/
|
|-- config.py                    # ** EDIT THIS: data paths & hyperparameters **
|-- requirements.txt             # Python dependencies (pip install -r)
|-- README.md                    # This file
|
|-- scripts/                     # Training and evaluation scripts
|   |-- train_sam40_balanced_v2.py   # SAM-40 training (94.79% accuracy)
|   |-- train_full_data.py           # EEGMAT + Combined training (99.31%)
|   |-- generate_v6_figures.py       # Generate all paper figures (300 DPI)
|   |-- train_sam40_features.py      # Alternative: NMI + Ensemble training
|   |-- validate_paper_claims.py     # Validate paper claims against results
|   `-- analyze_datasets.py          # Dataset statistics and analysis
|
|-- data/                        # EEG datasets (download separately)
|   |-- SAM40/
|   |   `-- filtered_data/       # Place .mat files here (480 files)
|   |-- EEGMAT/
|   |   `-- eeg-during-mental-arithmetic-tasks-1.0.0/  # Place .edf files here
|   `-- sample/                  # 20 sample records included for testing
|       |-- SAM40/               # 10 sample .mat files
|       |-- EEGMAT/              # 10 sample .npy files
|       `-- sample_metadata.json # Sample data description
|
|-- results/                     # Training results (JSON)
|   |-- full_data_results.json       # EEGMAT 99.31%, SAM-40 72.92%
|   |-- sam40_balanced_v2_results.json  # SAM-40 94.79%
|   `-- ...                          # Other experiment results
|
|-- paper/                       # LaTeX paper and figures
|   |-- genai_rag_eeg_v6.tex              # 8-page IEEE paper
|   |-- genai_rag_eeg_v6_technical_report.tex  # 23-page technical report
|   |-- genai_rag_eeg_v6_code_appendix.tex     # Code appendix with figures
|   |-- fig10_roc_curves.png          # ROC curves (300 DPI)
|   |-- fig11_confusion_matrices.png  # Confusion matrices (300 DPI)
|   |-- fig12_training_curves.png     # Training curves (300 DPI)
|   |-- sam40_segments.png            # SAM-40 EEG segments (300 DPI)
|   |-- eegmat_segments.png           # EEGMAT EEG segments (300 DPI)
|   `-- ...                           # All publication figures
|
|-- figures_extracted/           # Additional figures and analysis plots
|-- checkpoints/                 # Saved model weights
|-- logs/                        # Training logs
`-- config.yaml                  # Full YAML configuration (advanced)
```

### Key Files to Know

| File | Purpose | When to Edit |
|------|---------|-------------|
| `config.py` | **Data paths, hyperparameters** | **Always edit first** |
| `requirements.txt` | Python dependencies | Only if adding packages |
| `scripts/train_sam40_balanced_v2.py` | SAM-40 training pipeline | To modify model/features |
| `scripts/train_full_data.py` | EEGMAT+Combined pipeline | To modify model/features |
| `config.yaml` | Advanced YAML config | For deep customization |

---

## System Architecture

```
                        GenAI-RAG-EEG Architecture
                        ==========================

    ┌─────────────────────────────────────────────────────────┐
    │                    INPUT LAYER                            │
    │  SAM-40 (.mat)  ───┐                                     │
    │  EEGMAT (.edf)  ───┼──> Data Loader (config.py paths)    │
    │  Sample (.npy)  ───┘                                     │
    └─────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────▼───────────────────────────────────────────┐
    │               PREPROCESSING PIPELINE                      │
    │  1. Bandpass Filter (0.5-45 Hz, Butterworth order 4)     │
    │  2. Notch Filter (50 Hz, Q=30)                           │
    │  3. Per-Subject Z-Score Normalization                    │
    │  4. Segmentation (4s windows, 50% overlap)               │
    │  5. SWT Decomposition (db4 wavelet)                      │
    └─────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────▼───────────────────────────────────────────┐
    │              FEATURE EXTRACTION (~515 features)           │
    │  Band Powers:  delta, theta, alpha, beta, gamma (x2)     │
    │  Hjorth:       activity, mobility, complexity             │
    │  Statistical:  mean, std, skew, kurtosis, RMS, P2P       │
    │  Spectral:     entropy                                    │
    │  Global:       beta/alpha, theta/beta, FAA, variability  │
    └─────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────▼───────────────────────────────────────────┐
    │              ENSEMBLE CLASSIFIER (Soft Voting)            │
    │                                                           │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
    │  │ RandomForest  │  │ GradBoost    │  │ SVM (RBF)    │   │
    │  │ n=500, d=15   │  │ n=300, d=5   │  │ C=10         │   │
    │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
    │         └─────────┬───────┘──────────────────┘           │
    │                   ▼                                       │
    │         Soft Voting (probability averaging)               │
    │              │                                            │
    │              ▼                                            │
    │     Stress / Non-Stress                                  │
    └─────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────▼───────────────────────────────────────────┐
    │             EVALUATION & VALIDATION                       │
    │  5-Fold Stratified CV  |  SMOTE Balancing                │
    │  Confusion Matrix      |  ROC/AUC Curves                 │
    │  Cohen's Kappa         |  Statistical Tests (p<0.001)    │
    └─────────────────────────────────────────────────────────┘
```

---

## Configuration Guide

### config.py (Primary - Edit This)

All configurable parameters are in `config.py`:

```python
# === DATA PATHS (MUST CHANGE for your system) ===
SAM40_DIR   = Path("data/SAM40/filtered_data")      # SAM-40 .mat files
EEGMAT_DIR  = Path("data/EEGMAT/eeg-during-...")     # EEGMAT .edf files
RESULTS_DIR = Path("results")                         # Output JSON results
PAPER_DIR   = Path("paper")                           # LaTeX and figures

# === MODEL HYPERPARAMETERS ===
RF_N_ESTIMATORS = 500      # Random Forest: number of trees
RF_MAX_DEPTH = 15          # Random Forest: max tree depth
GB_N_ESTIMATORS = 300      # Gradient Boosting: number of trees
GB_MAX_DEPTH = 5           # Gradient Boosting: max tree depth
SVM_KERNEL = "rbf"         # SVM kernel type
SVM_C = 10                 # SVM regularization

# === PREPROCESSING ===
BANDPASS_LOW = 0.5         # Hz (high-pass cutoff)
BANDPASS_HIGH = 45.0       # Hz (low-pass cutoff)
NOTCH_FREQ = 50.0          # Hz (50 for EU/India, 60 for US)
N_FOLDS = 5                # Cross-validation folds
RANDOM_SEED = 42           # Reproducibility
```

### Environment Variables (Alternative)

Set these environment variables instead of editing config.py:

| Variable | Description | Example |
|----------|-------------|---------|
| `EEG_PROJECT_ROOT` | Project root directory | `/home/user/eeg-stress-rag` |
| `EEG_DATA_DIR` | Main data directory | `/home/user/data` |
| `EEG_SAM40_DIR` | SAM-40 .mat files | `/home/user/data/SAM40/filtered_data` |
| `EEG_EEGMAT_DIR` | EEGMAT .edf files | `/home/user/data/EEGMAT/eeg-during-...` |
| `EEG_RESULTS_DIR` | Results output | `/home/user/results` |

---

## For Paper Submission

This repository accompanies the paper:

> **Explainable EEG-Based Stress Classification using GenAI-RAG Architecture with Ensemble Learning**
> Praveen Asthana, Rajveer Singh Lalawat, Sarita Singh Gond
> March 2026

### Paper Documents

| Document | Pages | Description |
|----------|-------|-------------|
| `paper/genai_rag_eeg_v6.tex` | 8 | IEEE format paper |
| `paper/genai_rag_eeg_v6_technical_report.tex` | 23 | Full technical report |
| `paper/genai_rag_eeg_v6_code_appendix.tex` | - | Code listings + figures |

### Reproducibility

All results are reproducible with `RANDOM_SEED=42`:

```bash
# Reproduce SAM-40 results (94.79%):
python scripts/train_sam40_balanced_v2.py

# Reproduce EEGMAT results (99.31%):
python scripts/train_full_data.py

# Results saved to results/ folder as JSON
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: mne` | `pip install mne` |
| `ModuleNotFoundError: imblearn` | `pip install imbalanced-learn` |
| `FileNotFoundError: SAM40` | Edit `SAM40_DIR` in `config.py` |
| `FileNotFoundError: EEGMAT` | Edit `EEGMAT_DIR` in `config.py` |
| Windows path errors | Use raw strings: `r"C:\Users\..."` |
| Low accuracy | Ensure per-subject normalization is enabled |
| Memory error | Reduce `N_FOLDS` or use fewer channels |

### Validate Setup

```bash
python config.py
# This will show: paths, data availability, configuration
```

---

## Authors

- **Praveen Asthana** - Lead Developer
- **Rajveer Singh Lalawat** - Research & Analysis
- **Sarita Singh Gond** - Data Processing & Validation

## License

MIT License - see [LICENSE](LICENSE)

## Citation

```bibtex
@article{asthana2026genai_rag_eeg,
  title={Explainable EEG-Based Stress Classification using GenAI-RAG Architecture},
  author={Asthana, Praveen and Lalawat, Rajveer Singh and Gond, Sarita Singh},
  year={2026}
}
```
