#!/usr/bin/env python3
"""
Process PhysioNet EEG Mental Arithmetic Tasks (EEGMAT) dataset.
Converts EDF files to numpy format compatible with GenAI-RAG-EEG model.

Dataset: https://physionet.org/content/eegmat/1.0.0/
- 36 subjects, 23 EEG channels
- _1 files: baseline (background EEG)
- _2 files: stress (mental arithmetic task)
"""

import os
import numpy as np
from glob import glob
import json
from datetime import datetime

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("MNE not available, trying pyedflib...")

try:
    import pyedflib
    PYEDFLIB_AVAILABLE = True
except ImportError:
    PYEDFLIB_AVAILABLE = False

# Configuration
DATA_DIR = "/media/praveen/Asthana3/rajveer/eeg-stress-rag/data/EEGMAT/eeg-during-mental-arithmetic-tasks-1.0.0"
OUTPUT_DIR = "/media/praveen/Asthana3/rajveer/eeg-stress-rag/data/EEGMAT/sample_100"
TARGET_CHANNELS = 32
TARGET_SAMPLES = 512
TARGET_SR = 256  # Hz


def load_edf_with_mne(filepath):
    """Load EDF file using MNE."""
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    data = raw.get_data()  # (channels, samples)
    sfreq = raw.info['sfreq']
    return data, sfreq


def load_edf_with_pyedflib(filepath):
    """Load EDF file using pyedflib."""
    f = pyedflib.EdfReader(filepath)
    n_channels = f.signals_in_file
    n_samples = f.getNSamples()[0]
    sfreq = f.getSampleFrequency(0)

    data = np.zeros((n_channels, n_samples))
    for i in range(n_channels):
        data[i, :] = f.readSignal(i)

    f.close()
    return data, sfreq


def load_edf(filepath):
    """Load EDF file using available library."""
    if MNE_AVAILABLE:
        return load_edf_with_mne(filepath)
    elif PYEDFLIB_AVAILABLE:
        return load_edf_with_pyedflib(filepath)
    else:
        raise ImportError("Neither MNE nor pyedflib available. Install with: pip install mne pyedflib")


def resample_data(data, orig_sr, target_sr):
    """Resample data to target sampling rate."""
    if orig_sr == target_sr:
        return data

    from scipy import signal
    n_samples_new = int(data.shape[1] * target_sr / orig_sr)
    resampled = signal.resample(data, n_samples_new, axis=1)
    return resampled


def pad_channels(data, target_channels):
    """Pad or truncate to target number of channels."""
    n_channels = data.shape[0]
    if n_channels == target_channels:
        return data
    elif n_channels < target_channels:
        # Pad with zeros
        padding = np.zeros((target_channels - n_channels, data.shape[1]))
        return np.vstack([data, padding])
    else:
        # Truncate
        return data[:target_channels, :]


def segment_data(data, segment_length, overlap=0):
    """Segment continuous data into fixed-length windows."""
    n_channels, n_samples = data.shape
    step = segment_length - overlap
    segments = []

    for start in range(0, n_samples - segment_length + 1, step):
        segment = data[:, start:start + segment_length]
        segments.append(segment)

    return np.array(segments)


def normalize_segment(segment):
    """Z-score normalization."""
    mean = segment.mean()
    std = segment.std()
    if std < 1e-8:
        return segment - mean
    return (segment - mean) / std


def process_dataset():
    """Process entire EEGMAT dataset."""
    print(f"Processing EEGMAT dataset from: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all EDF files
    edf_files = sorted(glob(os.path.join(DATA_DIR, "Subject*_*.edf")))
    print(f"Found {len(edf_files)} EDF files")

    baseline_files = [f for f in edf_files if "_1.edf" in f]
    task_files = [f for f in edf_files if "_2.edf" in f]

    print(f"Baseline files (_1): {len(baseline_files)}")
    print(f"Task/Stress files (_2): {len(task_files)}")

    all_baseline_segments = []
    all_task_segments = []
    metadata_samples = []

    # Process baseline files
    print("\nProcessing baseline files...")
    for filepath in baseline_files:
        filename = os.path.basename(filepath)
        subject = filename.split("_")[0]

        try:
            data, sfreq = load_edf(filepath)
            print(f"  {filename}: {data.shape}, {sfreq} Hz")

            # Resample if needed
            if sfreq != TARGET_SR:
                data = resample_data(data, sfreq, TARGET_SR)

            # Pad channels
            data = pad_channels(data, TARGET_CHANNELS)

            # Segment
            segments = segment_data(data, TARGET_SAMPLES)

            # Normalize each segment
            for i, seg in enumerate(segments):
                normalized = normalize_segment(seg)
                all_baseline_segments.append(normalized)
                metadata_samples.append({
                    "subject": subject,
                    "condition": "baseline",
                    "segment": i,
                    "file": filename
                })
        except Exception as e:
            print(f"  Error processing {filename}: {e}")

    # Process task files
    print("\nProcessing task/stress files...")
    for filepath in task_files:
        filename = os.path.basename(filepath)
        subject = filename.split("_")[0]

        try:
            data, sfreq = load_edf(filepath)
            print(f"  {filename}: {data.shape}, {sfreq} Hz")

            # Resample if needed
            if sfreq != TARGET_SR:
                data = resample_data(data, sfreq, TARGET_SR)

            # Pad channels
            data = pad_channels(data, TARGET_CHANNELS)

            # Segment
            segments = segment_data(data, TARGET_SAMPLES)

            # Normalize each segment
            for i, seg in enumerate(segments):
                normalized = normalize_segment(seg)
                all_task_segments.append(normalized)
                metadata_samples.append({
                    "subject": subject,
                    "condition": "task",
                    "segment": i,
                    "file": filename
                })
        except Exception as e:
            print(f"  Error processing {filename}: {e}")

    print(f"\nTotal baseline segments: {len(all_baseline_segments)}")
    print(f"Total task segments: {len(all_task_segments)}")

    # Create balanced 100-sample dataset
    n_per_class = 50

    # Randomly select samples
    np.random.seed(42)

    baseline_indices = np.random.choice(len(all_baseline_segments), n_per_class, replace=False)
    task_indices = np.random.choice(len(all_task_segments), n_per_class, replace=False)

    selected_baseline = [all_baseline_segments[i] for i in baseline_indices]
    selected_task = [all_task_segments[i] for i in task_indices]

    # Combine and create labels
    X = np.array(selected_baseline + selected_task, dtype=np.float32)
    y = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.int64)

    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    print(f"\nFinal dataset shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: baseline={np.sum(y==0)}, task/stress={np.sum(y==1)}")

    # Save numpy files
    np.save(os.path.join(OUTPUT_DIR, "X_eegmat_100.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y_eegmat_100.npy"), y)

    # Save metadata
    metadata = {
        "description": "EEGMAT (PhysioNet EEG Mental Arithmetic Tasks) balanced sample dataset",
        "source": "https://physionet.org/content/eegmat/1.0.0/",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": 100,
        "shape": list(X.shape),
        "original_channels": 23,
        "padded_channels": TARGET_CHANNELS,
        "segment_length": TARGET_SAMPLES,
        "sampling_rate": TARGET_SR,
        "class_distribution": {
            "baseline": int(np.sum(y==0)),
            "task_stress": int(np.sum(y==1))
        },
        "notes": "Baseline=background EEG, Task=mental arithmetic stress condition. Channels padded from 23 to 32."
    }

    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved files to {OUTPUT_DIR}:")
    print(f"  - X_eegmat_100.npy")
    print(f"  - y_eegmat_100.npy")
    print(f"  - metadata.json")

    return X, y, metadata


if __name__ == "__main__":
    X, y, metadata = process_dataset()
