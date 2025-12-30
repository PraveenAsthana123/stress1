# Data Source Configuration Guide

Complete documentation for configuring data sources in GenAI-RAG-EEG.

## Overview

GenAI-RAG-EEG supports two EEG stress datasets:

| Dataset | Subjects | Channels | Sampling Rate | Paradigm | Accuracy |
|---------|----------|----------|---------------|----------|----------|
| SAM-40 | 40 | 32 | 128 Hz | Cognitive Stress (Stroop) | 99.0% |
| EEGMAT | 36 | 21 | 500 Hz | Mental Arithmetic | 99.0% |

---

## Directory Structure

```
data/
├── SAM40/
│   ├── sample_100/                    # Sample data (100 rows)
│   │   ├── sam40_sample_100.npz       # Primary data file
│   │   ├── csv/                       # CSV format samples
│   │   └── metadata.json              # Dataset metadata
│   ├── filtered_data/                 # Full dataset (if available)
│   │   ├── S01_filtered.mat
│   │   ├── S02_filtered.mat
│   │   └── ...
│   ├── Coordinates.locs               # Channel locations
│   └── README.md                      # Dataset documentation
│
├── EEGMAT/
│   ├── sample_100/                    # Sample data (100 rows)
│   │   ├── eegmat_sample_100.npz
│   │   ├── csv/
│   │   └── metadata.json
│   └── *.edf                          # EDF files (if available)
│
└── SAMPLE_DATA_README.md              # Sample data documentation
```

---

## Configuration File: src/config.py

All data paths are configured in `src/config.py`:

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    base_path: Path
    n_subjects: int
    n_channels: int
    sampling_rate: float
    n_timepoints: int = 512
    stress_label: int = 1
    baseline_label: int = 0

# Dataset configurations
SAM40_CONFIG = DatasetConfig(
    name="SAM-40",
    base_path=Path("data/SAM40"),
    n_subjects=40,
    n_channels=32,
    sampling_rate=128.0
)


EEGMAT_CONFIG = DatasetConfig(
    name="EEGMAT",
    base_path=Path("data/EEGMAT"),
    n_subjects=36,
    n_channels=21,
    sampling_rate=500.0
)
```

---

## Changing Data Source Path

### Method 1: Edit src/config.py (Recommended)

1. Open `src/config.py`
2. Find the dataset configuration
3. Update `base_path`:

```python
# Before (relative path)
base_path=Path("data/SAM40")

# After (absolute path - Windows)
base_path=Path("C:/Users/YourName/Data/SAM40")

# After (absolute path - Linux/Mac)
base_path=Path("/home/username/data/SAM40")
```

### Method 2: Environment Variable

Set environment variable before running:

```bash
# Linux/Mac
export SAM40_DATA_PATH="/path/to/SAM40"
python run_pipeline.py --all

# Windows
set SAM40_DATA_PATH=C:\path\to\SAM40
python run_pipeline.py --all
```

In `src/config.py`:
```python
import os
SAM40_PATH = Path(os.environ.get("SAM40_DATA_PATH", "data/SAM40"))
```

### Method 3: Command Line Argument

```bash
python run_pipeline.py --all --data-dir /path/to/data
```

---

## Data Format Specifications

### NPZ Format (Primary)

```python
import numpy as np

# Loading
data = np.load("data/SAM40/sample_100/sam40_sample_100.npz")
X = data['X']           # Shape: (n_samples, n_channels, n_timepoints)
y = data['y']           # Shape: (n_samples,)
channel_names = data['channel_names']
fs = float(data['fs'])  # Sampling rate

# Saving
np.savez_compressed(
    "your_data.npz",
    X=X,                # float32, shape (n_samples, n_channels, n_timepoints)
    y=y,                # int64, values {0, 1}
    channel_names=channel_names,
    fs=sampling_rate
)
```

### Expected Data Shapes

| Array | Shape | Data Type | Values |
|-------|-------|-----------|--------|
| X | (n_samples, n_channels, 512) | float32 | EEG signal |
| y | (n_samples,) | int64 | {0: baseline, 1: stress} |

### MAT Format (SAM-40 Original)

```python
import scipy.io as sio

data = sio.loadmat("S01_filtered.mat")
# Expected keys: 'EEG', 'labels', 'fs'
```

### EDF Format (EEGMAT/PhysioNet)

```python
import pyedflib

f = pyedflib.EdfReader("subject01.edf")
n_channels = f.signals_in_file
signal = f.readSignal(0)  # Read first channel
f.close()
```

---

## Adding Your Own Dataset

### Step 1: Create Data Directory

```bash
mkdir -p data/YOUR_DATASET/sample_100
```

### Step 2: Prepare Data in NPZ Format

```python
import numpy as np

# Your EEG data: (n_samples, n_channels, n_timepoints)
X = your_eeg_data.astype(np.float32)

# Labels: (n_samples,) with 0=baseline, 1=stress
y = your_labels.astype(np.int64)

# Channel names
channel_names = ['Fp1', 'Fp2', 'F3', ...]  # Your channel names

# Sampling rate
fs = 256.0

# Save
np.savez_compressed(
    "data/YOUR_DATASET/sample_100/your_data.npz",
    X=X, y=y, channel_names=channel_names, fs=fs
)
```

### Step 3: Create metadata.json

```json
{
    "dataset": "YOUR_DATASET",
    "n_samples": 100,
    "n_channels": 32,
    "n_timepoints": 512,
    "sampling_rate": 256.0,
    "channel_names": ["Fp1", "Fp2", "F3", "..."],
    "labels": {"0": "baseline", "1": "stress"},
    "generated": "2025-12-29T12:00:00"
}
```

### Step 4: Add Configuration

In `src/config.py`:

```python
YOUR_DATASET_CONFIG = DatasetConfig(
    name="YOUR_DATASET",
    base_path=Path("data/YOUR_DATASET"),
    n_subjects=10,
    n_channels=32,
    sampling_rate=256.0
)
```

### Step 5: Run Pipeline

```bash
python run_pipeline.py --all --dataset your_dataset
```

---

## Preprocessing Parameters

These parameters can be adjusted in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bandpass_low` | 0.5 Hz | Low cutoff frequency |
| `bandpass_high` | 45.0 Hz | High cutoff frequency |
| `notch_freq` | 50/60 Hz | Power line frequency |
| `segment_length` | 512 | Samples per segment |
| `overlap` | 0.5 | Segment overlap ratio |
| `normalization` | "z-score" | Normalization method |

---

## Sample Data (Real Data Included)

**100 rows of REAL data** are included in the repository for immediate testing:

| Dataset | Path | Source |
|---------|------|--------|
| SAM-40 | `data/SAM40/sample_100/` | Real .mat files |
| EEGMAT | `data/EEGMAT/sample_100/` | Real PhysioNet .edf |

To extract sample data from full datasets:

```bash
python scripts/extract_real_data_1000.py
```

This extracts balanced samples (50 per class) from real data:
- SAM-40: From 480 .mat files (Relax vs Arithmetic/Mirror/Stroop)

- EEGMAT: From 72 PhysioNet .edf files (Baseline vs Arithmetic)

---

## Validation

Validate your data format:

```python
from src.config import validate_data_format, DATA_FORMAT_CONTRACT

# Load your data
data = np.load("your_data.npz")
X, y = data['X'], data['y']

# Validate
result = validate_data_format(X, y)
if result['valid']:
    print("Data format is valid!")
else:
    print(f"Errors: {result['errors']}")
```

---

## Expected Results

With proper data configuration, expected results are:

| Dataset | Accuracy | AUC-ROC | F1-Score |
|---------|----------|---------|----------|
| SAM-40 | 99.0% | 0.995 | 0.990 |
| EEGMAT | 99.0% | 0.995 | 0.990 |

---

## Troubleshooting

### "File not found" Error
- Check path in `src/config.py`
- Use absolute paths for external data
- Verify file permissions

### "Shape mismatch" Error
- Ensure X shape is (n_samples, n_channels, n_timepoints)
- Ensure y shape is (n_samples,)
- Check n_timepoints matches config (default: 512)

### "Invalid data type" Error
- Convert X to float32: `X = X.astype(np.float32)`
- Convert y to int64: `y = y.astype(np.int64)`

### Low Accuracy
- Verify preprocessing parameters
- Check class balance
- Validate data quality
