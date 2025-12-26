#!/usr/bin/env python3
"""
WESAD (Wearable Stress and Affect Detection) Dataset Loader

Loads preprocessed WESAD data for stress detection.
Dataset location: /media/praveen/Asthana3/ upgrad/synopysis/thesis_code/data/chapter5_wesad/
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class WESADConfig:
    """Configuration for WESAD dataset."""
    base_path: str = "/media/praveen/Asthana3/ upgrad/synopysis/thesis_code/data/chapter5_wesad"
    n_channels: int = 14
    sampling_rate: int = 256
    # WESAD labels: 0=baseline, 1=stress, 2=amusement, 3=meditation
    stress_label: int = 1
    baseline_label: int = 0
    # For binary stress detection, we can use stress vs baseline


def load_wesad_dataset(
    config: Optional[WESADConfig] = None,
    binary: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load the WESAD dataset.

    Args:
        config: Dataset configuration
        binary: If True, convert to binary classification (stress vs non-stress)

    Returns:
        data: EEG data array (n_epochs, n_channels, n_samples)
        labels: Labels (0=non-stress/baseline, 1=stress)
        metadata: Additional information
    """
    if config is None:
        config = WESADConfig()

    data_path = Path(config.base_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    # Load preprocessed data
    data = np.load(data_path / "data.npy")
    labels = np.load(data_path / "labels.npy")

    print(f"Loading WESAD dataset from: {data_path}")
    print(f"Original data shape: {data.shape}")
    print(f"Original labels distribution: {np.bincount(labels.astype(int))}")

    metadata = {
        "original_shape": data.shape,
        "original_labels": np.unique(labels).tolist(),
        "sampling_rate": config.sampling_rate,
        "n_channels": data.shape[1],
        "n_samples": data.shape[2]
    }

    if binary:
        # For binary classification: stress (1) vs baseline (0)
        # Keep only stress and baseline samples
        mask = (labels == config.stress_label) | (labels == config.baseline_label)
        data = data[mask]
        labels = labels[mask]
        # Convert to binary: 1=stress, 0=baseline
        labels = (labels == config.stress_label).astype(int)

    metadata["shape"] = data.shape
    metadata["n_stress"] = int(np.sum(labels == 1))
    metadata["n_baseline"] = int(np.sum(labels == 0))
    metadata["binary"] = binary

    print(f"\nDataset loaded successfully:")
    print(f"  Shape: {data.shape}")
    print(f"  Stress samples: {metadata['n_stress']}")
    print(f"  Baseline samples: {metadata['n_baseline']}")

    return data, labels, metadata


if __name__ == "__main__":
    print("Testing WESAD data loader...")

    try:
        data, labels, metadata = load_wesad_dataset(binary=True)
        print(f"\nTest successful!")
        print(f"Data shape: {data.shape}")
        print(f"Labels distribution: {np.bincount(labels)}")
    except Exception as e:
        print(f"Error: {e}")
