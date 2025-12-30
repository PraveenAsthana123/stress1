#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Real EEG Dataset Loader for GenAI-RAG-EEG
================================================================================

Title: Real Dataset Loading and Preprocessing
Authors: GenAI-RAG-EEG Team
Version: 1.0.0

Description:
    This module provides loading and preprocessing functions for real EEG
    stress datasets: SAM-40 and EEGMAT.

Supported Datasets:
    1. DEAP (Database for Emotion Analysis using Physiological Signals)
       - 32 participants, 40 music video trials each
       - 32 EEG channels + 8 peripheral channels
       - Format: .dat (pickle), .bdf (biosemi)
       - URL: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/

    2. SAM-40 (Stress Analysis using EEG)
       - 40 subjects performing stress-inducing tasks
       - 32 EEG channels
       - Format: .mat (MATLAB)
       - Source: IIT Delhi dataset

Reference: GenAI-RAG-EEG Paper v2, IEEE Sensors Journal 2024
License: MIT
================================================================================
"""

import os
import pickle
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from pathlib import Path
import warnings

# Try importing optional dependencies
try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False


# =============================================================================
# CONSTANTS
# =============================================================================

# Standard EEG configuration
SAMPLING_RATE = 256
N_CHANNELS = 32
SEGMENT_LENGTH = 512  # 2 seconds at 256 Hz

# DEAP dataset configuration
DEAP_CHANNELS = [
    'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
    'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
    'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
    'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
]

# SAM-40 dataset configuration
SAM40_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC5', 'FC1', 'FC2', 'FC6',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'PO9', 'O1', 'Oz', 'O2', 'PO10'
]

# EEGMAT configuration (uses chest sensor primarily)
EEGMAT_MODALITIES = ['ACC', 'ECG', 'EDA', 'EMG', 'RESP', 'TEMP']


# =============================================================================
# DEAP DATASET LOADER
# =============================================================================

class DEAPLoader:
    """
    Load and preprocess DEAP dataset.

    DEAP contains physiological signals recorded while subjects watched
    music videos. Valence/Arousal ratings can be used for stress detection.

    Data Structure:
        - data: (40 trials, 40 channels, 8064 samples)
        - labels: (40 trials, 4) - valence, arousal, dominance, liking

    Stress Definition:
        High arousal (>5) + Low valence (<5) = Stress-like state
        Low arousal (<5) + High valence (>5) = Relaxed/Baseline state
    """

    def __init__(self, data_path: str):
        """
        Initialize DEAP loader.

        Args:
            data_path: Path to DEAP data_preprocessed folder
        """
        self.data_path = Path(data_path)
        self.sampling_rate = 128  # DEAP is recorded at 128 Hz
        self.n_channels = 32

    def load_subject(self, subject_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data for a single subject.

        Args:
            subject_id: Subject number (1-32)

        Returns:
            data: EEG data (40 trials, 32 channels, samples)
            labels: Rating labels (40 trials, 4)
        """
        filename = self.data_path / f's{subject_id:02d}.dat'

        if not filename.exists():
            raise FileNotFoundError(f"DEAP file not found: {filename}")

        with open(filename, 'rb') as f:
            subject_data = pickle.load(f, encoding='latin1')

        # Extract EEG channels (first 32 of 40 channels)
        data = subject_data['data'][:, :32, :]  # (40 trials, 32 channels, 8064 samples)
        labels = subject_data['labels']  # (40 trials, 4)

        return data, labels

    def extract_stress_labels(self, labels: np.ndarray,
                              arousal_threshold: float = 5.0,
                              valence_threshold: float = 5.0) -> np.ndarray:
        """
        Convert valence-arousal ratings to binary stress labels.

        Stress Logic:
            - High arousal + Low valence = Stress (1)
            - Otherwise = Baseline (0)

        Args:
            labels: Valence-arousal ratings (n_trials, 4)
            arousal_threshold: Threshold for arousal classification
            valence_threshold: Threshold for valence classification

        Returns:
            stress_labels: Binary stress labels (n_trials,)
        """
        valence = labels[:, 0]
        arousal = labels[:, 1]

        # Stress: high arousal + low valence
        stress_labels = ((arousal > arousal_threshold) &
                        (valence < valence_threshold)).astype(np.int64)

        return stress_labels

    def preprocess(self, data: np.ndarray,
                   target_fs: int = 256,
                   segment_length: int = SEGMENT_LENGTH) -> np.ndarray:
        """
        Preprocess DEAP data: resample and segment.

        Args:
            data: Raw EEG data (n_trials, n_channels, n_samples)
            target_fs: Target sampling rate
            segment_length: Target segment length in samples

        Returns:
            segments: Preprocessed segments (n_segments, n_channels, segment_length)
        """
        from scipy.signal import resample

        n_trials, n_channels, n_samples = data.shape

        # Resample to target sampling rate
        new_n_samples = int(n_samples * target_fs / self.sampling_rate)
        resampled = np.zeros((n_trials, n_channels, new_n_samples))

        for i in range(n_trials):
            for j in range(n_channels):
                resampled[i, j] = resample(data[i, j], new_n_samples)

        # Segment into fixed-length epochs
        n_segments_per_trial = new_n_samples // segment_length
        segments = []

        for trial in range(n_trials):
            for seg in range(n_segments_per_trial):
                start = seg * segment_length
                end = start + segment_length
                segments.append(resampled[trial, :, start:end])

        return np.array(segments)

    def load_all(self, subjects: Optional[List[int]] = None,
                 preprocess: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from all subjects.

        Args:
            subjects: List of subject IDs to load (default: all 1-32)
            preprocess: Whether to preprocess the data

        Returns:
            X: EEG data (n_samples, n_channels, segment_length)
            y: Stress labels (n_samples,)
        """
        if subjects is None:
            subjects = list(range(1, 33))

        all_data = []
        all_labels = []

        for subj in subjects:
            try:
                data, labels = self.load_subject(subj)
                stress_labels = self.extract_stress_labels(labels)

                if preprocess:
                    segments = self.preprocess(data)
                    # Repeat labels for each segment from same trial
                    n_segs_per_trial = len(segments) // len(stress_labels)
                    expanded_labels = np.repeat(stress_labels, n_segs_per_trial)
                    all_data.append(segments[:len(expanded_labels)])
                    all_labels.append(expanded_labels)
                else:
                    all_data.append(data)
                    all_labels.append(stress_labels)

            except FileNotFoundError:
                print(f"Warning: Subject {subj} data not found, skipping...")
                continue

        if not all_data:
            raise FileNotFoundError("No DEAP data files found")

        X = np.concatenate(all_data, axis=0)
        y = np.concatenate(all_labels, axis=0)

        return X.astype(np.float32), y.astype(np.int64)


# =============================================================================
# SAM-40 DATASET LOADER
# =============================================================================

class SAM40Loader:
    """
    Load and preprocess SAM-40 (Stress Analysis) dataset.

    SAM-40 contains EEG recordings from 40 subjects performing
    stress-inducing cognitive tasks.

    Tasks:
        - Stroop Color Word Test
        - Mirror Tracing
        - Arithmetic Tasks
        - Baseline/Rest

    Data Structure:
        - EEG: 32 channels at 256 Hz
        - Labels: Task-based (stress/no-stress)
    """

    def __init__(self, data_path: str):
        """
        Initialize SAM-40 loader.

        Args:
            data_path: Path to SAM-40 data folder
        """
        self.data_path = Path(data_path)
        self.sampling_rate = 256
        self.n_channels = 32

    def load_mat_file(self, filepath: str) -> Dict[str, Any]:
        """
        Load MATLAB .mat file.

        Args:
            filepath: Path to .mat file

        Returns:
            data_dict: Dictionary containing MATLAB variables
        """
        if not HAS_SCIPY:
            raise ImportError("scipy required for loading .mat files")

        return sio.loadmat(filepath)

    def load_subject(self, subject_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data for a single subject.

        Args:
            subject_id: Subject number (1-40)

        Returns:
            data: EEG data (n_trials, n_channels, n_samples)
            labels: Stress labels (n_trials,)
        """
        # Try different naming conventions
        possible_names = [
            f'S{subject_id:02d}.mat',
            f'Subject{subject_id}.mat',
            f'sub{subject_id:02d}.mat',
            f's{subject_id:02d}.mat'
        ]

        mat_data = None
        for name in possible_names:
            filepath = self.data_path / name
            if filepath.exists():
                mat_data = self.load_mat_file(str(filepath))
                break

        if mat_data is None:
            raise FileNotFoundError(f"SAM-40 data not found for subject {subject_id}")

        # Extract data (structure varies by dataset version)
        if 'EEG' in mat_data:
            eeg_data = mat_data['EEG']
        elif 'data' in mat_data:
            eeg_data = mat_data['data']
        elif 'eeg' in mat_data:
            eeg_data = mat_data['eeg']
        else:
            # Find the first large array
            for key, value in mat_data.items():
                if isinstance(value, np.ndarray) and value.size > 1000:
                    eeg_data = value
                    break

        # Extract labels
        if 'labels' in mat_data:
            labels = mat_data['labels'].flatten()
        elif 'stress' in mat_data:
            labels = mat_data['stress'].flatten()
        elif 'label' in mat_data:
            labels = mat_data['label'].flatten()
        else:
            # Generate labels based on data structure
            n_trials = eeg_data.shape[0] if eeg_data.ndim == 3 else 1
            labels = np.zeros(n_trials)  # Default to no stress

        return eeg_data, labels.astype(np.int64)

    def segment_continuous(self, data: np.ndarray,
                          segment_length: int = SEGMENT_LENGTH,
                          overlap: float = 0.5) -> np.ndarray:
        """
        Segment continuous EEG recording into epochs.

        Args:
            data: Continuous EEG (n_channels, n_samples)
            segment_length: Length of each segment
            overlap: Fraction of overlap between segments

        Returns:
            segments: Segmented data (n_segments, n_channels, segment_length)
        """
        n_channels, n_samples = data.shape
        step = int(segment_length * (1 - overlap))

        segments = []
        for start in range(0, n_samples - segment_length + 1, step):
            end = start + segment_length
            segments.append(data[:, start:end])

        return np.array(segments)

    def load_all(self, subjects: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from all subjects.

        Args:
            subjects: List of subject IDs (default: 1-40)

        Returns:
            X: EEG data (n_samples, n_channels, segment_length)
            y: Stress labels (n_samples,)
        """
        if subjects is None:
            subjects = list(range(1, 41))

        all_data = []
        all_labels = []

        for subj in subjects:
            try:
                data, labels = self.load_subject(subj)

                if data.ndim == 2:
                    # Continuous recording - segment it
                    segments = self.segment_continuous(data)
                    # Use single label for all segments
                    segment_labels = np.full(len(segments), labels[0] if len(labels) > 0 else 0)
                else:
                    segments = data
                    segment_labels = labels

                all_data.append(segments)
                all_labels.append(segment_labels)

            except FileNotFoundError:
                print(f"Warning: Subject {subj} data not found, skipping...")
                continue

        if not all_data:
            raise FileNotFoundError("No SAM-40 data files found")

        X = np.concatenate(all_data, axis=0)
        y = np.concatenate(all_labels, axis=0)

        return X.astype(np.float32), y.astype(np.int64)


# =============================================================================
# EEGMAT DATASET LOADER
# =============================================================================

class EEGMATLoader:
    """
    Load and preprocess EEGMAT (Wearable Stress and Affect Detection) dataset.

    EEGMAT contains physiological signals from chest-worn and wrist-worn
    sensors during stress induction protocol (TSST).

    Labels:
        0: Not defined / transient
        1: Baseline
        2: Stress
        3: Amusement
        4: Meditation

    Signals (Chest - RespiBAN):
        - ACC: Accelerometer (700 Hz, 3 axes)
        - ECG: Electrocardiogram (700 Hz)
        - EDA: Electrodermal Activity (700 Hz)
        - EMG: Electromyogram (700 Hz)
        - RESP: Respiration (700 Hz)
        - TEMP: Temperature (700 Hz)
    """

    def __init__(self, data_path: str):
        """
        Initialize EEGMAT loader.

        Args:
            data_path: Path to EEGMAT data folder
        """
        self.data_path = Path(data_path)
        self.chest_sampling_rate = 700
        self.wrist_sampling_rate = 64  # For E4 wristband

    def load_subject(self, subject_id: int) -> Dict[str, Any]:
        """
        Load data for a single subject.

        Args:
            subject_id: Subject number (2-17, excluding some)

        Returns:
            subject_data: Dictionary with signals and labels
        """
        subject_dir = self.data_path / f'S{subject_id}'
        pkl_file = subject_dir / f'S{subject_id}.pkl'

        if not pkl_file.exists():
            raise FileNotFoundError(f"EEGMAT file not found: {pkl_file}")

        with open(pkl_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        return data

    def extract_stress_baseline(self, data: Dict[str, Any],
                                modality: str = 'ECG',
                                segment_length: int = SEGMENT_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract stress and baseline segments from EEGMAT data.

        Uses binary classification: Stress (2) vs Baseline (1)

        Args:
            data: Subject data dictionary
            modality: Which signal to extract ('ECG', 'EDA', etc.)
            segment_length: Length of each segment

        Returns:
            segments: Signal segments
            labels: Binary labels (0=baseline, 1=stress)
        """
        # Get labels and signal
        labels = data['label']

        if modality in data['signal']['chest']:
            signal = data['signal']['chest'][modality]
            fs = self.chest_sampling_rate
        elif modality in data['signal']['wrist']:
            signal = data['signal']['wrist'][modality]
            fs = self.wrist_sampling_rate
        else:
            raise ValueError(f"Modality {modality} not found")

        # Ensure signal is 2D (samples, channels)
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

        # Resample to 256 Hz
        from scipy.signal import resample
        target_fs = 256
        n_samples_new = int(len(signal) * target_fs / fs)
        signal_resampled = resample(signal, n_samples_new, axis=0)

        # Resample labels to match
        labels_resampled = np.repeat(labels, target_fs // min(fs, target_fs) + 1)[:n_samples_new]

        # Extract stress (label=2) and baseline (label=1) segments
        stress_mask = labels_resampled == 2
        baseline_mask = labels_resampled == 1

        segments = []
        segment_labels = []

        # Extract stress segments
        stress_indices = np.where(stress_mask)[0]
        for i in range(0, len(stress_indices) - segment_length, segment_length):
            idx = stress_indices[i:i+segment_length]
            if len(idx) == segment_length and np.all(np.diff(idx) == 1):
                seg = signal_resampled[idx[0]:idx[-1]+1].T  # (channels, samples)
                segments.append(seg)
                segment_labels.append(1)  # Stress

        # Extract baseline segments
        baseline_indices = np.where(baseline_mask)[0]
        for i in range(0, len(baseline_indices) - segment_length, segment_length):
            idx = baseline_indices[i:i+segment_length]
            if len(idx) == segment_length and np.all(np.diff(idx) == 1):
                seg = signal_resampled[idx[0]:idx[-1]+1].T
                segments.append(seg)
                segment_labels.append(0)  # Baseline

        if not segments:
            # Return empty arrays with correct shape
            return np.zeros((0, signal.shape[1], segment_length)), np.zeros(0, dtype=np.int64)

        return np.array(segments), np.array(segment_labels, dtype=np.int64)

    def load_all(self, subjects: Optional[List[int]] = None,
                 modality: str = 'ECG') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from all subjects.

        Args:
            subjects: List of subject IDs (default: valid EEGMAT subjects)
            modality: Which signal modality to load

        Returns:
            X: Signal data (n_samples, n_channels, segment_length)
            y: Stress labels (n_samples,)
        """
        # Valid EEGMAT subjects (some are excluded)
        if subjects is None:
            subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

        all_data = []
        all_labels = []

        for subj in subjects:
            try:
                data = self.load_subject(subj)
                segments, labels = self.extract_stress_baseline(data, modality)

                if len(segments) > 0:
                    all_data.append(segments)
                    all_labels.append(labels)

            except FileNotFoundError:
                print(f"Warning: Subject {subj} data not found, skipping...")
                continue
            except Exception as e:
                print(f"Warning: Error loading subject {subj}: {e}")
                continue

        if not all_data:
            raise FileNotFoundError("No EEGMAT data files found")

        X = np.concatenate(all_data, axis=0)
        y = np.concatenate(all_labels, axis=0)

        return X.astype(np.float32), y.astype(np.int64)


# =============================================================================
# UNIFIED DATA LOADER
# =============================================================================

class RealDataLoader:
    """
    Unified loader for all supported real EEG datasets.

    Automatically detects dataset type and loads appropriately.
    """

    def __init__(self, data_root: str = 'data'):
        """
        Initialize the unified loader.

        Args:
            data_root: Root directory containing dataset folders
        """
        self.data_root = Path(data_root)

    def detect_dataset(self) -> Dict[str, bool]:
        """
        Detect which datasets are available.

        Returns:
            availability: Dict mapping dataset names to availability
        """
        availability = {
            'DEAP': False,
            'SAM40': False,
            : False
        }

        # Check for DEAP
        deap_path = self.data_root / 'DEAP' / 'data_preprocessed_python'
        if deap_path.exists() and any(deap_path.glob('s*.dat')):
            availability['DEAP'] = True

        # Alternative DEAP location
        deap_path2 = self.data_root / 'deap'
        if deap_path2.exists() and any(deap_path2.glob('*.dat')):
            availability['DEAP'] = True

        # Check for SAM-40
        sam40_path = self.data_root / 'SAM40'
        if sam40_path.exists() and any(sam40_path.glob('*.mat')):
            availability['SAM40'] = True

        sam40_path2 = self.data_root / 'sam40'
        if sam40_path2.exists() and any(sam40_path2.glob('*.mat')):
            availability['SAM40'] = True

        # Check for EEGMAT
        eegmat_path = self.data_root / 
        if eegmat_path.exists() and any(eegmat_path.glob('S*/*.pkl')):
            availability[] = True

        eegmat_path2 = self.data_root / 'eegmat'
        if eegmat_path2.exists() and any(eegmat_path2.glob('S*/*.pkl')):
            availability[] = True

        return availability

    def get_dataset_path(self, dataset: str) -> Path:
        """
        Get the path to a specific dataset.

        Args:
            dataset: Dataset name ('DEAP', 'SAM40')

        Returns:
            path: Path to dataset directory
        """
        # Try different naming conventions
        paths = {
            'DEAP': [
                self.data_root / 'DEAP' / 'data_preprocessed_python',
                self.data_root / 'deap',
                self.data_root / 'DEAP'
            ],
            'SAM40': [
                self.data_root / 'SAM40',
                self.data_root / 'sam40',
                self.data_root / 'SAM-40'
            ],
            : [
                self.data_root / ,
                self.data_root / 'eegmat'
            ]
        }

        for path in paths.get(dataset, []):
            if path.exists():
                return path

        raise FileNotFoundError(f"Dataset {dataset} not found in {self.data_root}")

    def load(self, dataset: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a specific dataset.

        Args:
            dataset: Dataset name ('DEAP', 'SAM40')
            **kwargs: Additional arguments for the specific loader

        Returns:
            X: Data array
            y: Labels
        """
        path = self.get_dataset_path(dataset)

        if dataset == 'DEAP':
            loader = DEAPLoader(str(path))
            return loader.load_all(**kwargs)
        elif dataset == 'SAM40':
            loader = SAM40Loader(str(path))
            return loader.load_all(**kwargs)
        elif dataset == :
            loader = EEGMATLoader(str(path))
            return loader.load_all(**kwargs)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def load_all_available(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load all available datasets.

        Returns:
            datasets: Dict mapping dataset names to (X, y) tuples
        """
        availability = self.detect_dataset()
        datasets = {}

        for name, available in availability.items():
            if available:
                try:
                    X, y = self.load(name)
                    datasets[name] = (X, y)
                    print(f"Loaded {name}: {X.shape[0]} samples")
                except Exception as e:
                    print(f"Error loading {name}: {e}")

        return datasets


# =============================================================================
# DATA DOWNLOAD INSTRUCTIONS
# =============================================================================

def print_download_instructions():
    """Print instructions for downloading real datasets."""

    instructions = """
================================================================================
                    REAL EEG DATASET DOWNLOAD INSTRUCTIONS
================================================================================

The GenAI-RAG-EEG system supports three real EEG datasets for stress analysis.
Follow the instructions below to download and set up each dataset.

--------------------------------------------------------------------------------
1. DEAP Dataset (Recommended - Emotion/Stress Analysis)
--------------------------------------------------------------------------------

   Website: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/

   Steps:
   1. Visit the DEAP website and register for dataset access
   2. Download "data_preprocessed_python.zip"
   3. Extract to: data/DEAP/data_preprocessed_python/

   Expected structure:
   data/DEAP/data_preprocessed_python/
   ├── s01.dat
   ├── s02.dat
   ...
   └── s32.dat

   Citation:
   Koelstra et al., "DEAP: A Database for Emotion Analysis Using
   Physiological Signals", IEEE Trans. on Affective Computing, 2012.

--------------------------------------------------------------------------------
2. SAM-40 Dataset (Stress Analysis)
--------------------------------------------------------------------------------

   Source: IIT Delhi / PhysioNet

   Steps:
   1. Search for "SAM-40 stress EEG dataset" or check PhysioNet
   2. Download the MATLAB (.mat) files
   3. Extract to: data/SAM40/

   Expected structure:
   data/SAM40/
   ├── S01.mat
   ├── S02.mat
   ...
   └── S40.mat

--------------------------------------------------------------------------------
   Website: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/

   Steps:
   1. Visit the EEGMAT website and download the dataset
   2. Extract to: data/EEGMAT/

   Expected structure:
   data/EEGMAT/
   ├── S2/
   │   └── S2.pkl
   ├── S3/
   │   └── S3.pkl
   ...
   └── S17/
       └── S17.pkl

   Note: EEGMAT contains ECG, EDA, EMG signals (not EEG), but is useful
   for multimodal stress detection.

   Citation:
   Schmidt et al., "Introducing EEGMAT, a Multimodal Dataset for
   Wearable Stress and Affect Detection", ICMI 2018.

--------------------------------------------------------------------------------
VERIFICATION
--------------------------------------------------------------------------------

After downloading, run this script to verify:

    python data/real_data_loader.py

Or in Python:

    from data.real_data_loader import RealDataLoader
    loader = RealDataLoader('data')
    availability = loader.detect_dataset()
    print(availability)

================================================================================
"""
    print(instructions)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("REAL DATA LOADER - GenAI-RAG-EEG")
    print("=" * 70)

    # Check for available datasets
    loader = RealDataLoader('data')
    availability = loader.detect_dataset()

    print("\nDataset Availability:")
    print("-" * 40)
    for name, available in availability.items():
        status = "✓ Available" if available else "✗ Not found"
        print(f"  {name}: {status}")

    # Load available datasets
    any_available = any(availability.values())

    if any_available:
        print("\nLoading available datasets...")
        print("-" * 40)
        datasets = loader.load_all_available()

        print("\nDataset Statistics:")
        print("-" * 40)
        for name, (X, y) in datasets.items():
            print(f"\n{name}:")
            print(f"  Shape: {X.shape}")
            print(f"  Stress samples: {(y == 1).sum()}")
            print(f"  Baseline samples: {(y == 0).sum()}")
            print(f"  Class balance: {(y == 1).sum() / len(y) * 100:.1f}% stress")
    else:
        print("\nNo datasets found!")
        print_download_instructions()

    print("\n" + "=" * 70)
