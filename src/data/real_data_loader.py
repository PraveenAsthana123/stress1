#!/usr/bin/env python3
"""
Real Data Loader for SAM-40 EEG Stress Dataset

Loads the actual SAM-40 dataset from .mat files.
Dataset location: /media/praveen/Asthana3/ upgrad/synopysis/datasets/SAM40_Stress/
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import scipy.io as sio
from dataclasses import dataclass


@dataclass
class SAM40Config:
    """Configuration for SAM-40 dataset."""
    base_path: str = "/media/praveen/Asthana3/ upgrad/synopysis/datasets/SAM40_Stress"
    n_subjects: int = 40
    n_channels: int = 32
    sampling_rate: int = 256
    n_trials: int = 3
    conditions: Tuple[str, ...] = ("Arithmetic", "Stroop", "Mirror_image", "Relax")
    stress_conditions: Tuple[str, ...] = ("Arithmetic", "Stroop", "Mirror_image")
    baseline_condition: str = "Relax"


def load_mat_file(filepath: str) -> np.ndarray:
    """Load a single .mat file and extract EEG data."""
    try:
        mat_data = sio.loadmat(filepath)
        # Find the data array (usually the largest non-metadata array)
        for key in mat_data.keys():
            if not key.startswith('__'):
                data = mat_data[key]
                if isinstance(data, np.ndarray) and data.size > 100:
                    return data
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_sam40_dataset(
    data_type: str = "filtered",
    config: Optional[SAM40Config] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load the complete SAM-40 dataset.

    Args:
        data_type: "raw" or "filtered" data
        config: Dataset configuration

    Returns:
        data: EEG data array (n_epochs, n_channels, n_samples)
        labels: Binary labels (0=baseline, 1=stress)
        metadata: Additional information
    """
    if config is None:
        config = SAM40Config()

    data_folder = "filtered_data" if data_type == "filtered" else "raw_data"
    data_path = Path(config.base_path) / data_folder

    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    all_data = []
    all_labels = []
    metadata = {
        "subjects": [],
        "conditions": [],
        "trials": [],
        "files_loaded": 0,
        "files_failed": 0
    }

    print(f"Loading SAM-40 dataset from: {data_path}")

    # Get all .mat files
    mat_files = list(data_path.glob("*.mat"))
    print(f"Found {len(mat_files)} .mat files")

    for mat_file in sorted(mat_files):
        filename = mat_file.stem
        parts = filename.split('_')

        # Parse filename: Condition_sub_XX_trialY
        if len(parts) >= 4:
            condition = parts[0]
            if parts[1] == "image":  # Handle "Mirror_image"
                condition = f"{parts[0]}_{parts[1]}"
                subject_idx = int(parts[3])
                trial_idx = int(parts[4].replace("trial", ""))
            else:
                subject_idx = int(parts[2])
                trial_idx = int(parts[3].replace("trial", ""))

            # Load EEG data
            eeg_data = load_mat_file(str(mat_file))

            if eeg_data is not None:
                # Ensure correct shape (channels x samples or samples x channels)
                if eeg_data.shape[0] > eeg_data.shape[1]:
                    eeg_data = eeg_data.T  # Transpose to (channels, samples)

                # Determine label: stress (1) vs baseline (0)
                label = 1 if condition in config.stress_conditions else 0

                all_data.append(eeg_data)
                all_labels.append(label)
                metadata["subjects"].append(subject_idx)
                metadata["conditions"].append(condition)
                metadata["trials"].append(trial_idx)
                metadata["files_loaded"] += 1
            else:
                metadata["files_failed"] += 1

    if len(all_data) == 0:
        raise ValueError("No data was loaded successfully")

    # Find min length for padding/truncating
    min_samples = min(d.shape[1] for d in all_data)
    max_channels = max(d.shape[0] for d in all_data)

    # Normalize to same shape
    normalized_data = []
    for d in all_data:
        # Truncate to min_samples
        d_truncated = d[:, :min_samples]
        # Pad channels if needed
        if d_truncated.shape[0] < max_channels:
            padding = np.zeros((max_channels - d_truncated.shape[0], min_samples))
            d_truncated = np.vstack([d_truncated, padding])
        normalized_data.append(d_truncated)

    data = np.array(normalized_data)
    labels = np.array(all_labels)

    metadata["shape"] = data.shape
    metadata["n_stress"] = int(np.sum(labels == 1))
    metadata["n_baseline"] = int(np.sum(labels == 0))
    metadata["sampling_rate"] = config.sampling_rate
    metadata["n_channels"] = data.shape[1]
    metadata["n_samples"] = data.shape[2]

    print(f"\nDataset loaded successfully:")
    print(f"  Shape: {data.shape}")
    print(f"  Stress samples: {metadata['n_stress']}")
    print(f"  Baseline samples: {metadata['n_baseline']}")
    print(f"  Files loaded: {metadata['files_loaded']}")
    print(f"  Files failed: {metadata['files_failed']}")

    return data, labels, metadata


def load_stress_detection_dataset(
    data_type: str = "filtered"
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load the Stress Detection dataset (alternative location).
    """
    base_path = "/media/praveen/Asthana3/ upgrad/synopysis/datasets/Stress_Detection"

    config = SAM40Config(base_path=base_path)
    return load_sam40_dataset(data_type=data_type, config=config)


def get_subject_data(
    data: np.ndarray,
    labels: np.ndarray,
    metadata: Dict,
    subject_id: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Get data for a specific subject."""
    mask = np.array(metadata["subjects"]) == subject_id
    return data[mask], labels[mask]


def get_condition_data(
    data: np.ndarray,
    labels: np.ndarray,
    metadata: Dict,
    condition: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Get data for a specific condition."""
    mask = np.array(metadata["conditions"]) == condition
    return data[mask], labels[mask]


if __name__ == "__main__":
    # Test the data loader
    print("Testing SAM-40 data loader...")

    try:
        data, labels, metadata = load_sam40_dataset(data_type="filtered")
        print(f"\nTest successful!")
        print(f"Data shape: {data.shape}")
        print(f"Labels distribution: {np.bincount(labels)}")
        print(f"Unique conditions: {set(metadata['conditions'])}")
        print(f"Unique subjects: {len(set(metadata['subjects']))}")
    except Exception as e:
        print(f"Error: {e}")

        # Try alternative location
        print("\nTrying alternative location...")
        try:
            data, labels, metadata = load_stress_detection_dataset(data_type="filtered")
            print(f"Alternative location successful!")
        except Exception as e2:
            print(f"Alternative also failed: {e2}")
