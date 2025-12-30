# Sample Data - 1000 Rows Per Dataset

Real EEG data extracted from SAM-40 and EEGMAT datasets.

## Datasets

| Dataset | Samples | Channels | Rate | Source |
|---------|---------|----------|------|--------|
| SAM-40 | 1000 | 32 | 128 Hz | Real (.mat files) |
| WESAD | 1000 | 14 | 700 Hz | Synthetic |
| EEGMAT | 720 | 21 | 500 Hz | Real (PhysioNet .edf) |

## Loading Data

```python
import numpy as np

# Load SAM-40
sam40 = np.load('data/SAM40/sample_1000/sam40_sample_1000.npz')
X, y = sam40['X'], sam40['y']
print(f"SAM-40: X={X.shape}, y={y.shape}")

# Load EEGMAT
eegmat = np.load('data/EEGMAT/sample_1000/eegmat_sample_1000.npz')
X, y = eegmat['X'], eegmat['y']
print(f"EEGMAT: X={X.shape}, y={y.shape}")

# Load WESAD
wesad = np.load('data/WESAD/sample_1000/wesad_sample_1000.npz')
X, y = wesad['X'], wesad['y']
print(f"WESAD: X={X.shape}, y={y.shape}")
```

## Data Format

- **X**: EEG signals, shape (n_samples, n_channels, 512)
- **y**: Labels, shape (n_samples,) - 0=relaxed/baseline, 1=stressed/arithmetic
- **channel_names**: List of electrode names
- **fs**: Sampling frequency in Hz

Generated: 2025-12-29 22:18:00
