#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Sample EEG Data Generator for GenAI-RAG-EEG
================================================================================

Title: Synthetic EEG Data for Testing and Demonstration
Authors: [Your Name]
Version: 1.0.0

Description:
    This module generates synthetic EEG data for testing the GenAI-RAG-EEG
    pipeline without requiring access to real datasets (DEAP, SAM-40, EEGMAT).

    The synthetic data mimics key characteristics of real EEG:
    - Realistic frequency components (alpha, beta, theta, delta, gamma)
    - Multi-channel structure (32 channels, 10-20 system)
    - Stress vs. baseline differences in band power

Data Characteristics:
    - Sampling Rate: 256 Hz
    - Channels: 32 (standard 10-20 montage)
    - Segment Length: 512 samples (2 seconds)
    - Classes: 0 (Baseline/No Stress), 1 (Stress)

Usage:
    >>> from sample_eeg_data import generate_sample_data
    >>> X_train, y_train, X_test, y_test = generate_sample_data(n_samples=100)
    >>> print(X_train.shape)  # (80, 32, 512)

License: MIT
================================================================================
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import json
import os


# =============================================================================
# CONSTANTS
# =============================================================================

# EEG Configuration
N_CHANNELS = 32
SAMPLING_RATE = 256  # Hz
SEGMENT_LENGTH = 512  # samples (2 seconds)

# Standard 10-20 Electrode Names
CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC5', 'FC1', 'FC2', 'FC6',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'PO9', 'O1', 'Oz', 'O2', 'PO10'
]

# EEG Frequency Bands (Hz)
FREQUENCY_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100)
}

# Stress-related power changes (relative to baseline)
STRESS_POWER_CHANGES = {
    'delta': 0.85,   # Decreased during stress
    'theta': 0.80,   # Decreased during stress
    'alpha': 0.70,   # Significantly decreased (less relaxation)
    'beta': 1.40,    # Increased (cognitive load)
    'gamma': 1.10    # Slightly increased
}


# =============================================================================
# SIGNAL GENERATION FUNCTIONS
# =============================================================================

def generate_eeg_band(
    n_samples: int,
    freq_range: Tuple[float, float],
    amplitude: float = 1.0,
    fs: int = SAMPLING_RATE
) -> np.ndarray:
    """
    Generate a single EEG frequency band signal.

    Logic:
    ------
    Creates a signal by summing multiple sinusoids within the specified
    frequency range, with random phases for naturalistic appearance.

    Args:
        n_samples: Number of time samples
        freq_range: (low_freq, high_freq) in Hz
        amplitude: Signal amplitude scaling factor
        fs: Sampling frequency in Hz

    Returns:
        signal: 1D numpy array of shape (n_samples,)

    Example:
        >>> alpha_signal = generate_eeg_band(512, (8, 13), amplitude=1.5)
    """
    t = np.arange(n_samples) / fs
    signal = np.zeros(n_samples)

    # Generate multiple frequency components
    n_components = 5
    frequencies = np.linspace(freq_range[0], freq_range[1], n_components)

    for freq in frequencies:
        phase = np.random.uniform(0, 2 * np.pi)
        signal += np.sin(2 * np.pi * freq * t + phase)

    # Normalize and scale
    signal = signal / n_components * amplitude

    return signal


def generate_single_channel_eeg(
    n_samples: int = SEGMENT_LENGTH,
    is_stress: bool = False,
    base_amplitudes: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Generate synthetic EEG for a single channel.

    Logic:
    ------
    1. Generate each frequency band separately
    2. Apply stress-related power modifications if is_stress=True
    3. Sum all bands and add Gaussian noise

    Data Flow:
    ----------
    Base amplitudes → Generate bands → Apply stress modulation → Sum + noise

    Args:
        n_samples: Number of time samples
        is_stress: Whether to generate stress-like pattern
        base_amplitudes: Optional dict of band amplitudes

    Returns:
        eeg: 1D numpy array of shape (n_samples,)
    """
    if base_amplitudes is None:
        # Default amplitudes based on typical EEG power distribution
        base_amplitudes = {
            'delta': 1.0,
            'theta': 0.8,
            'alpha': 0.6,
            'beta': 0.4,
            'gamma': 0.2
        }

    signal = np.zeros(n_samples)

    for band_name, freq_range in FREQUENCY_BANDS.items():
        amplitude = base_amplitudes.get(band_name, 0.5)

        # Apply stress modulation
        if is_stress:
            amplitude *= STRESS_POWER_CHANGES.get(band_name, 1.0)

        # Generate band signal
        band_signal = generate_eeg_band(n_samples, freq_range, amplitude)
        signal += band_signal

    # Add realistic noise
    noise = np.random.normal(0, 0.1, n_samples)
    signal += noise

    return signal


def generate_multichannel_eeg(
    n_samples: int = SEGMENT_LENGTH,
    n_channels: int = N_CHANNELS,
    is_stress: bool = False
) -> np.ndarray:
    """
    Generate synthetic multi-channel EEG data.

    Logic:
    ------
    1. Generate each channel independently
    2. Add spatial correlation between nearby electrodes
    3. Apply realistic amplitude variations across scalp

    Integration:
    ------------
    This function integrates with the DataLoader to provide
    test data in the same format as real datasets.

    Args:
        n_samples: Number of time samples per channel
        n_channels: Number of EEG channels
        is_stress: Whether to generate stress-like pattern

    Returns:
        eeg: 2D numpy array of shape (n_channels, n_samples)
    """
    # Regional amplitude variations (frontal > occipital for stress)
    regional_scaling = {
        'frontal': 1.2 if is_stress else 1.0,      # Fp1, Fp2, F3, F4, Fz, F7, F8
        'central': 1.0,                             # C3, Cz, C4
        'parietal': 0.9,                            # P3, Pz, P4
        'occipital': 0.8 if is_stress else 1.0,     # O1, Oz, O2
        'temporal': 1.1 if is_stress else 1.0       # T7, T8
    }

    # Map channels to regions
    def get_region(ch_name: str) -> str:
        if ch_name.startswith('F') or ch_name.startswith('Fp'):
            return 'frontal'
        elif ch_name.startswith('C') or ch_name.startswith('FC'):
            return 'central'
        elif ch_name.startswith('P') or ch_name.startswith('CP') or ch_name.startswith('PO'):
            return 'parietal'
        elif ch_name.startswith('O'):
            return 'occipital'
        elif ch_name.startswith('T'):
            return 'temporal'
        return 'central'

    eeg = np.zeros((n_channels, n_samples))

    for i in range(n_channels):
        ch_name = CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else f'CH{i}'
        region = get_region(ch_name)
        scaling = regional_scaling.get(region, 1.0)

        # Generate channel with regional scaling
        base_amplitudes = {
            'delta': 1.0 * scaling,
            'theta': 0.8 * scaling,
            'alpha': 0.6 * scaling,
            'beta': 0.4 * scaling,
            'gamma': 0.2 * scaling
        }

        eeg[i] = generate_single_channel_eeg(
            n_samples=n_samples,
            is_stress=is_stress,
            base_amplitudes=base_amplitudes
        )

    # Add inter-channel correlation (nearby electrodes are correlated)
    correlation_matrix = np.eye(n_channels)
    for i in range(n_channels):
        for j in range(i + 1, min(i + 3, n_channels)):  # Correlate nearby channels
            correlation_matrix[i, j] = 0.3
            correlation_matrix[j, i] = 0.3

    # Apply correlation (simplified)
    correlated_eeg = eeg.copy()
    for i in range(n_channels):
        for j in range(n_channels):
            if i != j and correlation_matrix[i, j] > 0:
                correlated_eeg[i] += correlation_matrix[i, j] * 0.2 * eeg[j]

    return correlated_eeg


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_sample_data(
    n_samples: int = 100,
    test_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate complete sample dataset for training and testing.

    Logic:
    ------
    1. Generate balanced stress and baseline samples
    2. Split into train/test sets
    3. Shuffle data

    Data Integration:
    -----------------
    Returns data in the same format as real dataset loaders:
    - X: (n_samples, n_channels, n_time_samples)
    - y: (n_samples,) with labels 0 or 1

    Args:
        n_samples: Total number of samples to generate
        test_ratio: Fraction of data for testing
        random_seed: Random seed for reproducibility

    Returns:
        X_train: Training EEG data (n_train, 32, 512)
        y_train: Training labels (n_train,)
        X_test: Testing EEG data (n_test, 32, 512)
        y_test: Testing labels (n_test,)

    Example:
        >>> X_train, y_train, X_test, y_test = generate_sample_data(n_samples=100)
        >>> print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        Train: (80, 32, 512), Test: (20, 32, 512)
    """
    np.random.seed(random_seed)

    # Generate balanced dataset
    n_per_class = n_samples // 2

    # Generate stress samples (class 1)
    stress_samples = np.array([
        generate_multichannel_eeg(is_stress=True)
        for _ in range(n_per_class)
    ])
    stress_labels = np.ones(n_per_class, dtype=np.int64)

    # Generate baseline samples (class 0)
    baseline_samples = np.array([
        generate_multichannel_eeg(is_stress=False)
        for _ in range(n_per_class)
    ])
    baseline_labels = np.zeros(n_per_class, dtype=np.int64)

    # Combine
    X = np.concatenate([stress_samples, baseline_samples], axis=0)
    y = np.concatenate([stress_labels, baseline_labels], axis=0)

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Split
    n_test = int(len(X) * test_ratio)
    n_train = len(X) - n_test

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train.astype(np.float32), y_train, X_test.astype(np.float32), y_test


def generate_sample_contexts(
    n_samples: int,
    random_seed: int = 42
) -> List[str]:
    """
    Generate sample context strings for text encoder.

    Logic:
    ------
    Creates realistic context strings mimicking experimental conditions.

    Args:
        n_samples: Number of context strings to generate
        random_seed: Random seed for reproducibility

    Returns:
        contexts: List of context strings
    """
    np.random.seed(random_seed)

    tasks = ['Stroop', 'Arithmetic', 'Mirror Tracing', 'N-back', 'Rest']
    genders = ['M', 'F']
    ages = list(range(18, 45))

    contexts = []
    for _ in range(n_samples):
        task = np.random.choice(tasks)
        age = np.random.choice(ages)
        gender = np.random.choice(genders)
        context = f"Task: {task}. Age: {age} years. Gender: {gender}."
        contexts.append(context)

    return contexts


def save_sample_data(
    output_dir: str = 'data/sample',
    n_samples: int = 100
) -> Dict[str, str]:
    """
    Save sample data to files for reproducibility.

    Logic:
    ------
    1. Generate sample data
    2. Save as numpy files
    3. Save metadata as JSON

    Args:
        output_dir: Output directory path
        n_samples: Number of samples to generate

    Returns:
        file_paths: Dictionary of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate data
    X_train, y_train, X_test, y_test = generate_sample_data(n_samples)
    contexts = generate_sample_contexts(len(X_train) + len(X_test))

    # Save numpy arrays
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

    # Save metadata
    metadata = {
        'n_channels': N_CHANNELS,
        'sampling_rate': SAMPLING_RATE,
        'segment_length': SEGMENT_LENGTH,
        'channel_names': CHANNEL_NAMES,
        'frequency_bands': FREQUENCY_BANDS,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'class_distribution': {
            'train_stress': int(y_train.sum()),
            'train_baseline': int(len(y_train) - y_train.sum()),
            'test_stress': int(y_test.sum()),
            'test_baseline': int(len(y_test) - y_test.sum())
        },
        'contexts': contexts[:10]  # Sample contexts
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save contexts
    with open(os.path.join(output_dir, 'contexts.txt'), 'w') as f:
        for ctx in contexts:
            f.write(ctx + '\n')

    return {
        'X_train': os.path.join(output_dir, 'X_train.npy'),
        'y_train': os.path.join(output_dir, 'y_train.npy'),
        'X_test': os.path.join(output_dir, 'X_test.npy'),
        'y_test': os.path.join(output_dir, 'y_test.npy'),
        'metadata': os.path.join(output_dir, 'metadata.json'),
        'contexts': os.path.join(output_dir, 'contexts.txt')
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_sample_eeg(
    eeg: np.ndarray,
    channel_idx: int = 0,
    title: str = "Sample EEG Signal"
) -> None:
    """
    Visualize EEG signal for debugging.

    Args:
        eeg: EEG array of shape (n_channels, n_samples) or (n_samples,)
        channel_idx: Channel index to plot (if multi-channel)
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt

        if eeg.ndim == 2:
            signal = eeg[channel_idx]
        else:
            signal = eeg

        t = np.arange(len(signal)) / SAMPLING_RATE

        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        # Time domain
        axes[0].plot(t, signal)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'{title} - Time Domain')
        axes[0].grid(True, alpha=0.3)

        # Frequency domain
        from scipy.fft import fft, fftfreq
        n = len(signal)
        yf = np.abs(fft(signal))[:n//2]
        xf = fftfreq(n, 1/SAMPLING_RATE)[:n//2]

        axes[1].plot(xf, yf)
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Magnitude')
        axes[1].set_title(f'{title} - Frequency Domain')
        axes[1].set_xlim(0, 50)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('sample_eeg_visualization.png', dpi=300)
        plt.close()
        print("Saved visualization to sample_eeg_visualization.png")

    except ImportError:
        print("Matplotlib not available for visualization")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Generate and save sample data when run directly.

    Usage:
        python sample_eeg_data.py
    """
    print("=" * 70)
    print("SAMPLE EEG DATA GENERATOR")
    print("GenAI-RAG-EEG - Synthetic Data for Testing")
    print("=" * 70)

    # Generate data
    print("\n[1] Generating sample data...")
    X_train, y_train, X_test, y_test = generate_sample_data(n_samples=200)

    print(f"\n[2] Data shapes:")
    print(f"    X_train: {X_train.shape}")
    print(f"    y_train: {y_train.shape}")
    print(f"    X_test: {X_test.shape}")
    print(f"    y_test: {y_test.shape}")

    print(f"\n[3] Class distribution:")
    print(f"    Train - Stress: {y_train.sum()}, Baseline: {len(y_train) - y_train.sum()}")
    print(f"    Test - Stress: {y_test.sum()}, Baseline: {len(y_test) - y_test.sum()}")

    # Save data
    print("\n[4] Saving to disk...")
    file_paths = save_sample_data(output_dir='data/sample', n_samples=200)
    for name, path in file_paths.items():
        print(f"    {name}: {path}")

    # Visualize
    print("\n[5] Generating visualization...")
    visualize_sample_eeg(X_train[0], channel_idx=0, title="Sample Stress EEG")
    visualize_sample_eeg(X_train[-1], channel_idx=0, title="Sample Baseline EEG")

    print("\n" + "=" * 70)
    print("SAMPLE DATA GENERATION COMPLETE")
    print("=" * 70)
