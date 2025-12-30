#!/usr/bin/env python3
"""
Generate 1000-row sample data for each dataset.

Creates synthetic but realistic EEG data for testing and demonstration.
Windows compatible using pathlib.

Usage:
    python scripts/generate_sample_data.py
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

# Set random seed for reproducibility
np.random.seed(42)


def generate_eeg_signal(n_channels: int, n_samples: int, fs: float,
                       stressed: bool = False) -> np.ndarray:
    """
    Generate realistic EEG signal with physiologically plausible characteristics.
    """
    t = np.arange(n_samples) / fs
    signal = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        # Delta (0.5-4 Hz)
        delta = 0.3 * np.sin(2 * np.pi * 2 * t + np.random.rand() * 2 * np.pi)
        # Theta (4-8 Hz)
        theta_amp = 0.35 if stressed else 0.25
        theta = theta_amp * np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi)
        # Alpha (8-13 Hz) - Suppressed during stress
        alpha_amp = 0.15 if stressed else 0.4
        alpha = alpha_amp * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
        # Beta (13-30 Hz) - Increased during stress
        beta_amp = 0.25 if stressed else 0.15
        beta = beta_amp * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)
        # Gamma (30-45 Hz)
        gamma_amp = 0.1 if stressed else 0.05
        gamma = gamma_amp * np.sin(2 * np.pi * 35 * t + np.random.rand() * 2 * np.pi)
        # Combine
        signal[ch] = delta + theta + alpha + beta + gamma
        signal[ch] += 0.1 * np.random.randn(n_samples)
        signal[ch] += np.random.randn() * 0.05

    return signal.astype(np.float32)


def generate_sam40_sample_data(output_dir: Path, n_samples: int = 1000):
    """Generate SAM-40 format sample data."""
    print("\nGenerating SAM-40 sample data (1000 rows)...")
    sam40_dir = output_dir / "SAM40" / "sample_1000"
    sam40_dir.mkdir(parents=True, exist_ok=True)

    n_channels, n_timepoints, fs = 32, 512, 128.0
    channel_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 
                    'FC4', 'FT8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'TP7', 'CP3', 'CPz', 
                    'CP4', 'TP8', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2', 'A1', 'A2']

    X = np.zeros((n_samples, n_channels, n_timepoints), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        stressed = i % 2 == 1
        X[i] = generate_eeg_signal(n_channels, n_timepoints, fs, stressed)
        y[i] = 1 if stressed else 0

    npz_path = sam40_dir / "sam40_sample_1000.npz"
    np.savez_compressed(npz_path, X=X, y=y, channel_names=channel_names, fs=fs)

    csv_dir = sam40_dir / "csv"
    csv_dir.mkdir(exist_ok=True)
    for i in range(min(10, n_samples)):
        csv_path = csv_dir / f"sample_{i:03d}_{'stressed' if y[i] == 1 else 'relaxed'}.csv"
        np.savetxt(csv_path, X[i].T, delimiter=",", header=",".join(channel_names), comments="")

    metadata = {"dataset": "SAM-40", "n_samples": n_samples, "n_channels": n_channels,
                "sampling_rate": fs, "channel_names": channel_names, 
                "labels": {"0": "relaxed", "1": "stressed"}, "generated": datetime.now().isoformat()}
    with open(sam40_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved: {npz_path} | Shape: X={X.shape}, y={y.shape}")
    return X, y


def generate_eegmat_sample_data(output_dir: Path, n_samples: int = 100):
    """Generate EEGMAT format sample data."""
    print("\nGenerating EEGMAT sample data...")
    eegmat_dir = output_dir /  / "sample_100"
    eegmat_dir.mkdir(parents=True, exist_ok=True)

    n_channels, n_timepoints, fs = 14, 512, 700.0
    channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    X = np.zeros((n_samples, n_channels, n_timepoints), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        stressed = i % 2 == 1
        X[i] = generate_eeg_signal(n_channels, n_timepoints, fs, stressed)
        y[i] = 1 if stressed else 0

    npz_path = eegmat_dir / "eegmat_sample_100.npz"
    np.savez_compressed(npz_path, X=X, y=y, channel_names=channel_names, fs=fs)

    csv_dir = eegmat_dir / "csv"
    csv_dir.mkdir(exist_ok=True)
    for i in range(min(10, n_samples)):
        csv_path = csv_dir / f"sample_{i:03d}_{'stressed' if y[i] == 1 else 'relaxed'}.csv"
        np.savetxt(csv_path, X[i].T, delimiter=",", header=",".join(channel_names), comments="")

    metadata = {"dataset": "n_samples": n_samples, "n_channels": n_channels,
                "sampling_rate": fs, "channel_names": channel_names,
                "labels": {"0": "relaxed", "1": "stressed"}, "generated": datetime.now().isoformat()}
    with open(eegmat_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved: {npz_path} | Shape: X={X.shape}, y={y.shape}")
    return X, y


def generate_eegmat_sample_data(output_dir: Path, n_samples: int = 100):
    """Generate EEGMAT format sample data."""
    print("\nGenerating EEGMAT sample data...")
    eegmat_dir = output_dir / "EEGMAT" / "sample_100"
    eegmat_dir.mkdir(parents=True, exist_ok=True)

    n_channels, n_timepoints, fs = 21, 512, 500.0
    channel_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 
                    'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2', 'A1']

    X = np.zeros((n_samples, n_channels, n_timepoints), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        stressed = i % 2 == 1
        X[i] = generate_eeg_signal(n_channels, n_timepoints, fs, stressed)
        y[i] = 1 if stressed else 0

    X_padded = np.zeros((n_samples, 32, n_timepoints), dtype=np.float32)
    X_padded[:, :n_channels, :] = X

    npz_path = eegmat_dir / "eegmat_sample_100.npz"
    np.savez_compressed(npz_path, X=X, X_padded=X_padded, y=y, channel_names=channel_names, fs=fs)

    csv_dir = eegmat_dir / "csv"
    csv_dir.mkdir(exist_ok=True)
    for i in range(min(10, n_samples)):
        csv_path = csv_dir / f"sample_{i:03d}_{'stressed' if y[i] == 1 else 'relaxed'}.csv"
        np.savetxt(csv_path, X[i].T, delimiter=",", header=",".join(channel_names), comments="")

    metadata = {"dataset": "EEGMAT", "n_samples": n_samples, "n_channels": n_channels,
                "sampling_rate": fs, "channel_names": channel_names,
                "labels": {"0": "baseline", "1": "arithmetic"}, "generated": datetime.now().isoformat()}
    with open(eegmat_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved: {npz_path} | Shape: X={X.shape}, y={y.shape}")
    return X, y


def create_data_readme(output_dir: Path):
    """Create README for sample data."""
    readme = f"""# Sample Data for GenAI-RAG-EEG

100-row sample datasets for testing and demonstration.

## Datasets

| Dataset | Samples | Channels | Rate | Paradigm |
|---------|---------|----------|------|----------|
| SAM-40 | 100 | 32 | 128 Hz | Cognitive (Stroop) |
| EEGMAT | 100 | 14 | 700 Hz | TSST Protocol |
| EEGMAT | 100 | 21 | 500 Hz | Mental Arithmetic |

## Loading Data

```python
import numpy as np
data = np.load('data/SAM40/sample_100/sam40_sample_100.npz')
X, y = data['X'], data['y']
```

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    with open(output_dir / "SAMPLE_DATA_README.md", "w") as f:
        f.write(readme)
    print(f"\n✓ Created README")


def main():
    print("="*60)
    print("  Sample Data Generator - 100 rows per dataset")
    print("="*60)

    output_dir = project_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_sam40_sample_data(output_dir)
    generate_eegmat_sample_data(output_dir)
    generate_eegmat_sample_data(output_dir)
    create_data_readme(output_dir)

    print("\n" + "="*60)
    print(f"  Complete! Output: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
