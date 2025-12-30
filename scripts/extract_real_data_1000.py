#!/usr/bin/env python3
"""
Extract sample rows from real EEG datasets.

Extracts actual data from SAM-40, WESAD, and EEGMAT datasets.
Default: 100 rows per dataset (configurable via N_SAMPLES).
"""

import sys
from pathlib import Path
import json
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

try:
    import scipy.io as sio
except ImportError:
    print("Installing scipy...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    import scipy.io as sio

try:
    import pyedflib
except ImportError:
    print("Installing pyedflib...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyedflib"])
    import pyedflib


def extract_sam40_data(output_dir: Path, n_samples: int = 100):
    """Extract samples from real SAM-40 .mat files."""
    print("\n" + "="*60)
    print(f"  Extracting SAM-40 Real Data ({n_samples} rows)")
    print("="*60)

    sam40_raw = project_root / "data" / "SAM40" / "filtered_data"
    sam40_out = output_dir / "SAM40" / f"sample_{n_samples}"
    sam40_out.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(sam40_raw.glob("*.mat"))
    print(f"  Found {len(mat_files)} .mat files")

    all_X = []
    all_y = []
    n_channels = 32
    n_timepoints = 512

    for mat_file in mat_files:
        try:
            data = sio.loadmat(str(mat_file))

            # SAM-40 uses 'Clean_data' key with shape (32, N)
            if 'Clean_data' in data:
                eeg_data = data['Clean_data']
            else:
                # Fallback: find any array
                eeg_key = None
                for key in data.keys():
                    if not key.startswith('_') and isinstance(data[key], np.ndarray):
                        if len(data[key].shape) >= 2:
                            eeg_key = key
                            break
                if eeg_key is None:
                    continue
                eeg_data = data[eeg_key]

            # Determine label from filename
            # SAM-40 has: Arithmetic, Mirror, Stroop (stress=1), Relax (baseline=0)
            is_stress = "Arithmetic" in mat_file.name or "Mirror" in mat_file.name or "Stroop" in mat_file.name
            is_relax = "Relax" in mat_file.name
            label = 0 if is_relax else 1

            # Data shape is (32, N) where N >= 3200
            if eeg_data.shape[0] == n_channels:
                # Segment into n_timepoints chunks
                n_segments = eeg_data.shape[1] // n_timepoints
                for seg in range(min(n_segments, 6)):  # Max 6 segments per file
                    start = seg * n_timepoints
                    segment = eeg_data[:, start:start+n_timepoints]
                    if segment.shape == (n_channels, n_timepoints):
                        all_X.append(segment)
                        all_y.append(label)
            elif eeg_data.shape[1] == n_channels:
                # Transposed: (N, 32)
                eeg_data = eeg_data.T
                n_segments = eeg_data.shape[1] // n_timepoints
                for seg in range(min(n_segments, 6)):
                    start = seg * n_timepoints
                    segment = eeg_data[:, start:start+n_timepoints]
                    if segment.shape == (n_channels, n_timepoints):
                        all_X.append(segment)
                        all_y.append(label)

        except Exception as e:
            print(f"  Error reading {mat_file.name}: {e}")
            continue

    # Convert to arrays first
    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    print(f"  Total segments extracted: {len(X)}")
    print(f"  Class distribution: relaxed={np.sum(y==0)}, stressed={np.sum(y==1)}")

    # Balance classes (stratified sampling)
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    n_per_class = min(len(idx_0), len(idx_1), n_samples // 2)

    np.random.seed(42)
    np.random.shuffle(idx_0)
    np.random.shuffle(idx_1)

    balanced_idx = np.concatenate([idx_0[:n_per_class], idx_1[:n_per_class]])
    np.random.shuffle(balanced_idx)

    X = X[balanced_idx]
    y = y[balanced_idx]

    channel_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz',
                    'FC4', 'FT8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'TP7', 'CP3', 'CPz',
                    'CP4', 'TP8', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2', 'A1', 'A2']

    npz_path = sam40_out / f"sam40_sample_{n_samples}.npz"
    np.savez_compressed(npz_path, X=X, y=y, channel_names=channel_names, fs=128.0)

    # Save CSV samples
    csv_dir = sam40_out / "csv"
    csv_dir.mkdir(exist_ok=True)
    for i in range(min(10, len(X))):
        csv_path = csv_dir / f"sample_{i:03d}_{'stressed' if y[i] == 1 else 'relaxed'}.csv"
        np.savetxt(csv_path, X[i].T, delimiter=",", header=",".join(channel_names[:X.shape[1]]), comments="")

    metadata = {
        "dataset": "SAM-40",
        "source": "Real data from filtered_data/*.mat",
        "n_samples": len(X),
        "n_channels": X.shape[1],
        "n_timepoints": X.shape[2],
        "sampling_rate": 128.0,
        "class_distribution": {"relaxed": int(np.sum(y == 0)), "stressed": int(np.sum(y == 1))},
        "generated": datetime.now().isoformat()
    }
    with open(sam40_out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved: {npz_path}")
    print(f"    Shape: X={X.shape}, y={y.shape}")
    print(f"    Classes: relaxed={np.sum(y==0)}, stressed={np.sum(y==1)}")

    return X, y


def extract_eegmat_data(output_dir: Path, n_samples: int = 100):
    """Extract samples from real EEGMAT .edf files."""
    print("\n" + "="*60)
    print(f"  Extracting EEGMAT Real Data ({n_samples} rows)")
    print("="*60)

    eegmat_raw = project_root / "data" / "EEGMAT" / "eeg-during-mental-arithmetic-tasks-1.0.0"
    eegmat_out = output_dir / "EEGMAT" / f"sample_{n_samples}"
    eegmat_out.mkdir(parents=True, exist_ok=True)

    edf_files = sorted(eegmat_raw.glob("*.edf"))
    print(f"  Found {len(edf_files)} .edf files")

    all_X = []
    all_y = []
    n_channels = 21
    n_timepoints = 512

    for edf_file in edf_files:
        try:
            f = pyedflib.EdfReader(str(edf_file))
            n_signals = f.signals_in_file

            # Determine label: _1.edf = baseline (0), _2.edf = arithmetic (1)
            is_arithmetic = "_2.edf" in edf_file.name
            label = 1 if is_arithmetic else 0

            # Read signals
            signals = []
            for i in range(min(n_signals, n_channels)):
                signals.append(f.readSignal(i))
            f.close()

            if len(signals) < n_channels:
                continue

            eeg_data = np.array(signals[:n_channels])

            # Segment
            n_segments = eeg_data.shape[1] // n_timepoints
            for seg in range(min(n_segments, 10)):
                start = seg * n_timepoints
                segment = eeg_data[:, start:start+n_timepoints]
                if segment.shape == (n_channels, n_timepoints):
                    all_X.append(segment)
                    all_y.append(label)

        except Exception as e:
            continue

    # Convert to arrays first
    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    print(f"  Total segments extracted: {len(X)}")
    print(f"  Class distribution: baseline={np.sum(y==0)}, arithmetic={np.sum(y==1)}")

    # Balance classes (stratified sampling)
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    n_per_class = min(len(idx_0), len(idx_1), n_samples // 2)

    np.random.seed(42)
    np.random.shuffle(idx_0)
    np.random.shuffle(idx_1)

    balanced_idx = np.concatenate([idx_0[:n_per_class], idx_1[:n_per_class]])
    np.random.shuffle(balanced_idx)

    X = X[balanced_idx]
    y = y[balanced_idx]

    channel_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4',
                    'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2', 'A1']

    npz_path = eegmat_out / f"eegmat_sample_{n_samples}.npz"
    np.savez_compressed(npz_path, X=X, y=y, channel_names=channel_names, fs=500.0)

    csv_dir = eegmat_out / "csv"
    csv_dir.mkdir(exist_ok=True)
    for i in range(min(10, len(X))):
        csv_path = csv_dir / f"sample_{i:03d}_{'arithmetic' if y[i] == 1 else 'baseline'}.csv"
        np.savetxt(csv_path, X[i].T, delimiter=",", header=",".join(channel_names[:X.shape[1]]), comments="")

    metadata = {
        "dataset": "EEGMAT",
        "source": "Real data from PhysioNet EDF files",
        "n_samples": len(X),
        "n_channels": X.shape[1],
        "n_timepoints": X.shape[2],
        "sampling_rate": 500.0,
        "class_distribution": {"baseline": int(np.sum(y == 0)), "arithmetic": int(np.sum(y == 1))},
        "generated": datetime.now().isoformat()
    }
    with open(eegmat_out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved: {npz_path}")
    print(f"    Shape: X={X.shape}, y={y.shape}")
    print(f"    Classes: baseline={np.sum(y==0)}, arithmetic={np.sum(y==1)}")

    return X, y


def extract_wesad_data(output_dir: Path, n_samples: int = 100):
    """Extract samples from real WESAD data."""
    print("\n" + "="*60)
    print(f"  Extracting WESAD Real Data ({n_samples} rows)")
    print("="*60)

    # Real WESAD data path
    wesad_data_path = Path("/media/praveen/Asthana3/ upgrad/synopysis/thesis_code/data/chapter5_wesad")
    wesad_out = output_dir / "WESAD" / f"sample_{n_samples}"
    wesad_out.mkdir(parents=True, exist_ok=True)

    # Load real WESAD data
    data = np.load(wesad_data_path / "data.npy")  # Shape: (2000, 14, 256)
    labels = np.load(wesad_data_path / "labels.npy")  # Shape: (2000,)

    print(f"  Loaded real WESAD: data={data.shape}, labels={labels.shape}")
    print(f"  Original classes: {np.unique(labels, return_counts=True)}")

    # For binary classification: 0=baseline, 1=stress
    # Original: 0=baseline, 1=stress, 2=amusement, 3=meditation
    # Keep only classes 0 and 1
    mask = (labels == 0) | (labels == 1)
    X_binary = data[mask]
    y_binary = labels[mask]

    print(f"  Binary (baseline vs stress): {len(X_binary)} samples")

    # Balance classes
    idx_0 = np.where(y_binary == 0)[0]
    idx_1 = np.where(y_binary == 1)[0]
    n_per_class = min(len(idx_0), len(idx_1), n_samples // 2)

    np.random.seed(42)
    np.random.shuffle(idx_0)
    np.random.shuffle(idx_1)

    balanced_idx = np.concatenate([idx_0[:n_per_class], idx_1[:n_per_class]])
    np.random.shuffle(balanced_idx)

    X = X_binary[balanced_idx].astype(np.float32)
    y = y_binary[balanced_idx].astype(np.int64)

    n_channels = X.shape[1]
    n_timepoints = X.shape[2]
    fs = 700.0

    channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    npz_path = wesad_out / f"wesad_sample_{n_samples}.npz"
    np.savez_compressed(npz_path, X=X, y=y, channel_names=channel_names, fs=fs)

    csv_dir = wesad_out / "csv"
    csv_dir.mkdir(exist_ok=True)
    for i in range(min(10, len(X))):
        csv_path = csv_dir / f"sample_{i:03d}_{'stressed' if y[i] == 1 else 'relaxed'}.csv"
        np.savetxt(csv_path, X[i].T, delimiter=",", header=",".join(channel_names[:n_channels]), comments="")

    metadata = {
        "dataset": "WESAD",
        "source": "Real data from thesis_code/data/chapter5_wesad",
        "n_samples": len(X),
        "n_channels": n_channels,
        "n_timepoints": n_timepoints,
        "sampling_rate": fs,
        "class_distribution": {"relaxed": int(np.sum(y == 0)), "stressed": int(np.sum(y == 1))},
        "generated": datetime.now().isoformat()
    }
    with open(wesad_out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved: {npz_path}")
    print(f"    Shape: X={X.shape}, y={y.shape}")
    print(f"    Classes: relaxed={np.sum(y==0)}, stressed={np.sum(y==1)}")

    return X, y


def create_readme(output_dir: Path, n_samples: int = 100):
    """Create README for sample data."""
    readme = f"""# Sample Data - {n_samples} Rows Per Dataset

Real EEG data extracted from SAM-40, WESAD, and EEGMAT datasets.

## Datasets

| Dataset | Samples | Channels | Rate | Source |
|---------|---------|----------|------|--------|
| SAM-40 | {n_samples} | 32 | 128 Hz | Real (.mat files) |
| WESAD | {n_samples} | 14 | 700 Hz | Real (thesis data) |
| EEGMAT | {n_samples} | 21 | 500 Hz | Real (PhysioNet .edf) |

## Loading Data

```python
import numpy as np

# Load SAM-40
sam40 = np.load('data/SAM40/sample_{n_samples}/sam40_sample_{n_samples}.npz')
X, y = sam40['X'], sam40['y']
print(f"SAM-40: X={{X.shape}}, y={{y.shape}}")

# Load EEGMAT
eegmat = np.load('data/EEGMAT/sample_{n_samples}/eegmat_sample_{n_samples}.npz')
X, y = eegmat['X'], eegmat['y']
print(f"EEGMAT: X={{X.shape}}, y={{y.shape}}")

# Load WESAD
wesad = np.load('data/WESAD/sample_{n_samples}/wesad_sample_{n_samples}.npz')
X, y = wesad['X'], wesad['y']
print(f"WESAD: X={{X.shape}}, y={{y.shape}}")
```

## Data Format

- **X**: EEG signals, shape (n_samples, n_channels, timepoints)
- **y**: Labels, shape (n_samples,) - 0=relaxed/baseline, 1=stressed/arithmetic
- **channel_names**: List of electrode names
- **fs**: Sampling frequency in Hz

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    with open(output_dir / "SAMPLE_DATA_README.md", "w") as f:
        f.write(readme)
    print(f"\n✓ Created SAMPLE_DATA_README.md")


def main():
    # Default sample size
    N_SAMPLES = 100

    print("\n" + "="*70)
    print(f"  GenAI-RAG-EEG: Extract {N_SAMPLES} Rows from Real Data")
    print("="*70)

    output_dir = project_root / "data"

    # Extract from real data - all 3 datasets
    extract_sam40_data(output_dir, n_samples=N_SAMPLES)
    extract_eegmat_data(output_dir, n_samples=N_SAMPLES)
    extract_wesad_data(output_dir, n_samples=N_SAMPLES)

    create_readme(output_dir, n_samples=N_SAMPLES)

    print("\n" + "="*70)
    print(f"  Complete! {N_SAMPLES} rows extracted per dataset (all real data)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
