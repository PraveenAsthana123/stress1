#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GenAI-RAG-EEG Sample Data Generator
================================================================================

DESCRIPTION:
    Generates realistic sample EEG data for each dataset (SAM-40, WESAD, EEGMAT)
    with 100 samples each. This ensures the code can run out-of-the-box on any
    system without needing to download the full datasets.

PURPOSE:
    1. Allow users to test the pipeline without downloading large datasets
    2. Provide consistent test data for CI/CD pipelines
    3. Enable quick demos and tutorials
    4. Facilitate debugging on systems with limited storage

SAMPLE DATA CHARACTERISTICS:
    - 100 samples per dataset (50 stress, 50 baseline)
    - Realistic EEG signal properties (frequency bands, noise)
    - Matching data format with original datasets
    - Includes metadata files for compatibility

OUTPUT STRUCTURE:
    data/
    ├── sample_validation/
    │   ├── sam40_sample.npz     # 100 SAM-40 samples
    │   ├── wesad_sample.npz     # 100 WESAD samples
    │   ├── eegmat_sample.npz    # 100 EEGMAT samples
    │   └── metadata.json        # Dataset specifications
    │
    ├── SAM40/sample_100/        # SAM-40 format samples
    ├── WESAD/sample_100/        # WESAD format samples
    └── EEGMAT/sample_100/       # EEGMAT format samples

USAGE:
    python scripts/generate_sample_data.py

    # Or with options:
    python scripts/generate_sample_data.py --n-samples 100 --output-dir ./data

Author: GenAI-RAG-EEG Team
Version: 3.0.0
License: MIT
================================================================================
"""

import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# CONSTANTS - DATASET SPECIFICATIONS
# =============================================================================

DATASET_SPECS = {
    'SAM-40': {
        'n_channels': 32,
        'sampling_rate': 256,
        'segment_length': 512,  # 2 seconds
        'channel_names': [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
            'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
            'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3',
            'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10'
        ],
        'stress_type': 'cognitive',
        'description': 'Cognitive stress from Stroop, Arithmetic, Mirror tasks'
    },
    'WESAD': {
        'n_channels': 14,
        'sampling_rate': 256,  # After resampling from 700 Hz
        'segment_length': 512,
        'channel_names': [
            'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
            'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
        ],
        'stress_type': 'physiological',
        'description': 'Physiological stress from Trier Social Stress Test (TSST)'
    },
    'EEGMAT': {
        'n_channels': 21,
        'n_channels_padded': 32,  # Padded for model compatibility
        'sampling_rate': 256,  # After resampling from 500 Hz
        'segment_length': 512,
        'channel_names': [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
            'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
            'Fz', 'Cz', 'Pz', 'A1', 'A2'
        ],
        'stress_type': 'cognitive_arithmetic',
        'description': 'Mental arithmetic stress (serial subtraction by 7)'
    }
}

# EEG frequency bands (Hz)
FREQUENCY_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}


# =============================================================================
# EEG SIGNAL GENERATION
# =============================================================================

def generate_eeg_oscillation(
    n_samples: int,
    freq: float,
    sampling_rate: float,
    amplitude: float = 1.0,
    phase_offset: float = 0.0
) -> np.ndarray:
    """
    Generate a sinusoidal oscillation at a specific frequency.

    Args:
        n_samples: Number of time samples
        freq: Frequency in Hz
        sampling_rate: Sampling rate in Hz
        amplitude: Signal amplitude
        phase_offset: Phase offset in radians

    Returns:
        1D array of oscillation values
    """
    t = np.arange(n_samples) / sampling_rate
    return amplitude * np.sin(2 * np.pi * freq * t + phase_offset)


def generate_realistic_eeg(
    n_channels: int,
    n_samples: int,
    sampling_rate: float,
    is_stress: bool,
    random_state: np.random.RandomState
) -> np.ndarray:
    """
    Generate realistic multi-channel EEG signal.

    The signal incorporates:
    1. Dominant alpha rhythm (8-13 Hz) - reduced during stress
    2. Theta activity (4-8 Hz)
    3. Beta activity (13-30 Hz) - increased during stress
    4. Pink noise (1/f) for realistic spectral shape
    5. Channel-specific variations
    6. Spatial correlations between nearby channels

    Args:
        n_channels: Number of EEG channels
        n_samples: Number of time samples
        sampling_rate: Sampling rate in Hz
        is_stress: Whether this is a stress condition
        random_state: NumPy random state for reproducibility

    Returns:
        2D array of shape (n_channels, n_samples)
    """
    eeg = np.zeros((n_channels, n_samples))

    # Stress modifies band power
    if is_stress:
        alpha_amp = 0.7  # Reduced alpha (stress biomarker)
        beta_amp = 1.3   # Increased beta
        theta_amp = 0.9
    else:
        alpha_amp = 1.0
        beta_amp = 1.0
        theta_amp = 1.0

    for ch in range(n_channels):
        # Channel-specific phase offset for spatial variation
        phase_offset = random_state.uniform(0, 2 * np.pi)

        # Alpha oscillation (dominant rhythm)
        alpha_freq = random_state.uniform(9, 11)  # Individual alpha frequency
        eeg[ch] += generate_eeg_oscillation(
            n_samples, alpha_freq, sampling_rate,
            amplitude=10 * alpha_amp,
            phase_offset=phase_offset
        )

        # Theta oscillation
        theta_freq = random_state.uniform(5, 7)
        eeg[ch] += generate_eeg_oscillation(
            n_samples, theta_freq, sampling_rate,
            amplitude=5 * theta_amp,
            phase_offset=random_state.uniform(0, 2 * np.pi)
        )

        # Beta oscillation
        beta_freq = random_state.uniform(18, 25)
        eeg[ch] += generate_eeg_oscillation(
            n_samples, beta_freq, sampling_rate,
            amplitude=3 * beta_amp,
            phase_offset=random_state.uniform(0, 2 * np.pi)
        )

        # Pink noise (1/f spectrum)
        white_noise = random_state.randn(n_samples)
        pink_noise = np.cumsum(white_noise) * 0.5
        pink_noise -= pink_noise.mean()
        eeg[ch] += pink_noise * 2

        # High-frequency noise
        eeg[ch] += random_state.randn(n_samples) * 1.5

    # Add spatial correlations (nearby channels are correlated)
    for ch in range(1, n_channels):
        correlation = 0.3 * random_state.uniform(0.5, 1.0)
        eeg[ch] += correlation * eeg[ch - 1]

    # Normalize to realistic amplitude range (µV)
    eeg = eeg / np.std(eeg) * 20  # ~20 µV standard deviation

    return eeg


# =============================================================================
# SAMPLE DATA GENERATION
# =============================================================================

def generate_dataset_samples(
    dataset_name: str,
    n_samples: int = 100,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate sample data for a specific dataset.

    Args:
        dataset_name: Name of dataset ('SAM-40', 'WESAD', 'EEGMAT')
        n_samples: Total number of samples (split evenly between classes)
        seed: Random seed for reproducibility

    Returns:
        X: EEG data of shape (n_samples, n_channels, segment_length)
        y: Labels of shape (n_samples,) where 0=baseline, 1=stress
        metadata: Dictionary with dataset information
    """
    print(f"\n{'=' * 60}")
    print(f"Generating {dataset_name} Sample Data")
    print(f"{'=' * 60}")

    spec = DATASET_SPECS[dataset_name]
    random_state = np.random.RandomState(seed)

    n_channels = spec.get('n_channels_padded', spec['n_channels'])
    segment_length = spec['segment_length']
    sampling_rate = spec['sampling_rate']

    # Initialize arrays
    X = np.zeros((n_samples, n_channels, segment_length), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)

    # Generate samples (balanced classes)
    n_per_class = n_samples // 2

    print(f"  Channels: {n_channels}")
    print(f"  Segment length: {segment_length} samples ({segment_length/sampling_rate:.2f}s)")
    print(f"  Generating {n_per_class} stress + {n_per_class} baseline samples...")

    for i in range(n_samples):
        is_stress = i < n_per_class
        y[i] = 1 if is_stress else 0

        # Generate EEG (use original channel count, then pad if needed)
        eeg = generate_realistic_eeg(
            spec['n_channels'],
            segment_length,
            sampling_rate,
            is_stress,
            random_state
        )

        # Pad channels if needed (for EEGMAT)
        if spec['n_channels'] < n_channels:
            padded = np.zeros((n_channels, segment_length), dtype=np.float32)
            padded[:spec['n_channels']] = eeg
            eeg = padded

        X[i] = eeg

        # Progress indicator
        if (i + 1) % 25 == 0:
            print(f"    Generated {i + 1}/{n_samples} samples")

    # Create metadata
    metadata = {
        'dataset_name': dataset_name,
        'n_samples': n_samples,
        'n_channels': n_channels,
        'n_channels_original': spec['n_channels'],
        'segment_length': segment_length,
        'sampling_rate': sampling_rate,
        'channel_names': spec['channel_names'],
        'stress_type': spec['stress_type'],
        'description': spec['description'],
        'class_distribution': {
            'stress': int(np.sum(y == 1)),
            'baseline': int(np.sum(y == 0))
        },
        'generation_date': datetime.now().isoformat(),
        'seed': seed,
        'is_synthetic': True,
        'data_shape': list(X.shape),
        'dtype': str(X.dtype)
    }

    # Verify data
    print(f"\n  Data shape: {X.shape}")
    print(f"  Labels: {np.sum(y==1)} stress, {np.sum(y==0)} baseline")
    print(f"  Value range: [{X.min():.2f}, {X.max():.2f}] µV")
    print(f"  Mean: {X.mean():.2f}, Std: {X.std():.2f}")

    return X, y, metadata


def save_sample_data(
    X: np.ndarray,
    y: np.ndarray,
    metadata: Dict[str, Any],
    output_dir: Path,
    dataset_name: str
):
    """
    Save sample data in multiple formats for compatibility.

    Args:
        X: EEG data array
        y: Label array
        metadata: Metadata dictionary
        output_dir: Output directory
        dataset_name: Dataset name
    """
    # Create directories
    sample_dir = output_dir / 'sample_validation'
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Save as compressed NPZ
    npz_file = sample_dir / f'{dataset_name.lower().replace("-", "")}_sample.npz'
    np.savez_compressed(
        npz_file,
        X=X,
        y=y,
        metadata=json.dumps(metadata)
    )
    print(f"  Saved: {npz_file}")

    # Save individual format in dataset-specific directory
    dataset_dir = output_dir / dataset_name.replace('-', '') / 'sample_100'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Save as individual files (for compatibility with different loaders)
    np.save(dataset_dir / 'X.npy', X)
    np.save(dataset_dir / 'y.npy', y)

    with open(dataset_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved: {dataset_dir}/")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Generate sample data for all datasets."""

    parser = argparse.ArgumentParser(
        description='Generate sample EEG data for GenAI-RAG-EEG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_sample_data.py
  python generate_sample_data.py --n-samples 50
  python generate_sample_data.py --output-dir /path/to/data

This script generates realistic synthetic EEG data for testing the
GenAI-RAG-EEG pipeline without requiring the original datasets.
        """
    )
    parser.add_argument(
        '--n-samples', type=int, default=100,
        help='Number of samples per dataset (default: 100)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory (default: PROJECT_ROOT/data)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / 'data'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("GenAI-RAG-EEG Sample Data Generator")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Samples per dataset: {args.n_samples}")
    print(f"Random seed: {args.seed}")

    # Generate for each dataset
    all_metadata = {}

    for dataset_name in ['SAM-40', 'WESAD', 'EEGMAT']:
        X, y, metadata = generate_dataset_samples(
            dataset_name,
            n_samples=args.n_samples,
            seed=args.seed
        )

        save_sample_data(X, y, metadata, output_dir, dataset_name)
        all_metadata[dataset_name] = metadata

    # Save combined metadata
    combined_metadata = {
        'generation_date': datetime.now().isoformat(),
        'generator_version': '3.0.0',
        'total_samples': args.n_samples * 3,
        'datasets': all_metadata
    }

    meta_file = output_dir / 'sample_validation' / 'all_metadata.json'
    with open(meta_file, 'w') as f:
        json.dump(combined_metadata, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Sample Data Generation Complete!")
    print(f"{'=' * 70}")
    print(f"\nGenerated {args.n_samples} samples for each dataset:")
    print(f"  - SAM-40:  {args.n_samples} samples (32 channels)")
    print(f"  - WESAD:   {args.n_samples} samples (14 channels)")
    print(f"  - EEGMAT:  {args.n_samples} samples (21→32 channels)")
    print(f"\nTotal: {args.n_samples * 3} samples")
    print(f"\nOutput: {output_dir}/sample_validation/")
    print("\nTo load sample data:")
    print("  >>> data = np.load('data/sample_validation/sam40_sample.npz')")
    print("  >>> X, y = data['X'], data['y']")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
