"""
Dataset Loaders for GenAI-RAG-EEG.

Supports loading and processing:
- DEAP Dataset (32 subjects, emotion-induced arousal)
- SAM-40 Dataset (40 subjects, cognitive stress)
- EEGMAT Dataset (25 subjects, mental workload)

All datasets are processed with Leave-One-Subject-Out (LOSO) cross-validation.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass
import warnings

import torch
from torch.utils.data import Dataset, DataLoader

from .preprocessing import EEGPreprocessor, get_dataset_config


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    n_subjects: int
    n_channels: int
    sampling_rate: float
    n_trials_per_subject: int
    label_type: str  # "stress_proxy", "cognitive_stress", "workload_proxy"
    description: str


DATASET_INFO = {
    "deap": DatasetInfo(
        name="DEAP",
        n_subjects=32,
        n_channels=32,
        sampling_rate=128.0,
        n_trials_per_subject=40,
        label_type="stress_proxy",
        description="Emotion-induced arousal as stress proxy (arousal >= 5)"
    ),
    "sam40": DatasetInfo(
        name="SAM-40",
        n_subjects=40,
        n_channels=32,
        sampling_rate=256.0,
        n_trials_per_subject=12,
        label_type="cognitive_stress",
        description="Cognitive stress from Stroop, Arithmetic, Mirror Tracing tasks"
    ),
    "eegmat": DatasetInfo(
        name="EEGMAT",
        n_subjects=25,
        n_channels=14,
        sampling_rate=128.0,
        n_trials_per_subject=20,
        label_type="workload_proxy",
        description="Mental workload from N-back and arithmetic tasks"
    )
}


class EEGStressDataset(Dataset):
    """
    PyTorch Dataset for EEG stress classification.

    Provides preprocessed EEG epochs with labels and optional context.
    """

    def __init__(
        self,
        epochs: np.ndarray,
        labels: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        context_info: Optional[List[Dict]] = None,
        transform=None
    ):
        """
        Args:
            epochs: EEG epochs (n_samples, n_channels, n_time)
            labels: Binary labels (n_samples,)
            subject_ids: Subject ID for each sample
            context_info: List of context dictionaries
            transform: Optional transform to apply
        """
        self.epochs = torch.FloatTensor(epochs)
        self.labels = torch.LongTensor(labels)
        self.subject_ids = subject_ids
        self.context_info = context_info
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        epoch = self.epochs[idx]
        label = self.labels[idx]

        if self.transform:
            epoch = self.transform(epoch)

        item = {
            "eeg": epoch,
            "label": label
        }

        if self.subject_ids is not None:
            item["subject_id"] = self.subject_ids[idx]

        if self.context_info is not None:
            item["context"] = self.context_info[idx]

        return item


class DEAPLoader:
    """
    Loader for DEAP dataset.

    DEAP uses arousal ratings (>= 5) as stress proxy.
    Original sampling: 512 Hz, downsampled to 128 Hz in preprocessed data.
    """

    def __init__(self, data_path: str, arousal_threshold: float = 5.0):
        """
        Args:
            data_path: Path to DEAP preprocessed data directory
            arousal_threshold: Threshold for binary arousal classification
        """
        self.data_path = Path(data_path)
        self.arousal_threshold = arousal_threshold
        self.info = DATASET_INFO["deap"]

    def load_subject(self, subject_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data for a single subject.

        Args:
            subject_id: Subject number (1-32)

        Returns:
            data: EEG data (n_trials, n_channels, n_samples)
            labels: Binary arousal labels (n_trials,)
        """
        filename = self.data_path / f"s{subject_id:02d}.dat"

        if not filename.exists():
            raise FileNotFoundError(f"DEAP file not found: {filename}")

        with open(filename, 'rb') as f:
            subject_data = pickle.load(f, encoding='latin1')

        # DEAP format: data (40, 40, 8064), labels (40, 4)
        # Channels 0-31 are EEG, 32-39 are peripheral
        eeg_data = subject_data['data'][:, :32, :]  # (40, 32, 8064)
        arousal = subject_data['labels'][:, 1]  # Arousal ratings

        # Convert to binary labels
        labels = (arousal >= self.arousal_threshold).astype(np.int64)

        return eeg_data, labels

    def load_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load all subjects.

        Returns:
            data: (n_total_trials, n_channels, n_samples)
            labels: (n_total_trials,)
            subject_ids: (n_total_trials,)
        """
        all_data = []
        all_labels = []
        all_subjects = []

        for subject_id in range(1, self.info.n_subjects + 1):
            try:
                data, labels = self.load_subject(subject_id)
                all_data.append(data)
                all_labels.append(labels)
                all_subjects.append(np.full(len(labels), subject_id))
            except FileNotFoundError:
                warnings.warn(f"Subject {subject_id} not found, skipping")

        return (
            np.concatenate(all_data),
            np.concatenate(all_labels),
            np.concatenate(all_subjects)
        )


class SAM40Loader:
    """
    Loader for SAM-40 dataset.

    SAM-40 contains validated cognitive stress from Stroop, Arithmetic,
    and Mirror Tracing tasks with NASA-TLX and physiological validation.
    """

    def __init__(self, data_path: str):
        """
        Args:
            data_path: Path to SAM-40 data directory
        """
        self.data_path = Path(data_path)
        self.info = DATASET_INFO["sam40"]

    def load_subject(self, subject_id: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load data for a single subject.

        Args:
            subject_id: Subject number (1-40)

        Returns:
            data: EEG data (n_trials, n_channels, n_samples)
            labels: Binary stress labels (n_trials,)
            task_types: Task type for each trial
        """
        subject_dir = self.data_path / f"sub-{subject_id:02d}"

        if not subject_dir.exists():
            raise FileNotFoundError(f"SAM-40 subject not found: {subject_dir}")

        # Load EEG data (format depends on actual dataset structure)
        # This is a placeholder - actual loading depends on SAM-40 format
        eeg_file = subject_dir / "eeg.npy"
        labels_file = subject_dir / "labels.npy"
        tasks_file = subject_dir / "tasks.txt"

        if eeg_file.exists():
            data = np.load(eeg_file)
            labels = np.load(labels_file)
            if tasks_file.exists():
                with open(tasks_file) as f:
                    task_types = [line.strip() for line in f]
            else:
                task_types = ["unknown"] * len(labels)
        else:
            # Generate placeholder data for testing
            warnings.warn(f"SAM-40 data not found, generating placeholder for subject {subject_id}")
            n_trials = self.info.n_trials_per_subject
            n_samples = int(4.0 * self.info.sampling_rate)  # 4 second epochs
            data = np.random.randn(n_trials, self.info.n_channels, n_samples)
            labels = np.random.randint(0, 2, n_trials)
            task_types = ["Stroop", "Arithmetic", "Mirror Tracing"] * 4

        return data, labels, task_types

    def load_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Load all subjects.

        Returns:
            data: (n_total_trials, n_channels, n_samples)
            labels: (n_total_trials,)
            subject_ids: (n_total_trials,)
            task_types: Task types for all trials
        """
        all_data = []
        all_labels = []
        all_subjects = []
        all_tasks = []

        for subject_id in range(1, self.info.n_subjects + 1):
            try:
                data, labels, tasks = self.load_subject(subject_id)
                all_data.append(data)
                all_labels.append(labels)
                all_subjects.append(np.full(len(labels), subject_id))
                all_tasks.extend(tasks)
            except FileNotFoundError:
                continue

        if not all_data:
            # Return placeholder if no data found
            warnings.warn("No SAM-40 data found, returning placeholder")
            n_samples = int(4.0 * self.info.sampling_rate)
            return (
                np.random.randn(100, self.info.n_channels, n_samples),
                np.random.randint(0, 2, 100),
                np.repeat(np.arange(1, 11), 10),
                ["Stroop"] * 100
            )

        return (
            np.concatenate(all_data),
            np.concatenate(all_labels),
            np.concatenate(all_subjects),
            all_tasks
        )


class EEGMATLoader:
    """
    Loader for EEGMAT dataset.

    EEGMAT uses mental arithmetic workload as stress proxy.
    Consumer-grade EEG (Emotiv EPOC+, 14 channels).
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.info = DATASET_INFO["eegmat"]

    def load_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load all EEGMAT data."""
        # Placeholder implementation
        warnings.warn("EEGMAT loader using placeholder data")
        n_samples = int(4.0 * self.info.sampling_rate)
        n_total = self.info.n_subjects * self.info.n_trials_per_subject

        return (
            np.random.randn(n_total, self.info.n_channels, n_samples),
            np.random.randint(0, 2, n_total),
            np.repeat(np.arange(1, self.info.n_subjects + 1), self.info.n_trials_per_subject)
        )


def create_loso_splits(
    data: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], None, None]:
    """
    Create Leave-One-Subject-Out cross-validation splits.

    Args:
        data: EEG data (n_samples, ...)
        labels: Labels (n_samples,)
        subject_ids: Subject IDs (n_samples,)

    Yields:
        train_data, train_labels, test_data, test_labels, test_subject_id
    """
    unique_subjects = np.unique(subject_ids)

    for test_subject in unique_subjects:
        test_mask = subject_ids == test_subject
        train_mask = ~test_mask

        yield (
            data[train_mask],
            labels[train_mask],
            data[test_mask],
            labels[test_mask],
            test_subject
        )


def create_dataloaders(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and testing.

    Args:
        train_data: Training EEG data
        train_labels: Training labels
        test_data: Test EEG data
        test_labels: Test labels
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        train_loader, test_loader
    """
    train_dataset = EEGStressDataset(train_data, train_labels)
    test_dataset = EEGStressDataset(test_data, test_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def generate_synthetic_dataset(
    n_subjects: int = 10,
    n_trials_per_subject: int = 20,
    n_channels: int = 32,
    n_time_samples: int = 512,
    class_balance: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic EEG dataset for testing.

    Creates data with simulated stress-related features:
    - Low stress: Higher alpha power
    - High stress: Lower alpha, higher beta

    Args:
        n_subjects: Number of subjects
        n_trials_per_subject: Trials per subject
        n_channels: Number of EEG channels
        n_time_samples: Time samples per trial
        class_balance: Proportion of high stress samples

    Returns:
        data: (n_total, n_channels, n_time_samples)
        labels: (n_total,)
        subject_ids: (n_total,)
    """
    np.random.seed(42)

    n_total = n_subjects * n_trials_per_subject
    data = np.zeros((n_total, n_channels, n_time_samples))
    labels = np.zeros(n_total)
    subject_ids = np.zeros(n_total)

    t = np.linspace(0, 4, n_time_samples)  # 4 seconds

    for s in range(n_subjects):
        subject_offset = s * n_trials_per_subject

        for trial in range(n_trials_per_subject):
            idx = subject_offset + trial

            # Determine label
            is_stress = np.random.rand() < class_balance
            labels[idx] = int(is_stress)
            subject_ids[idx] = s + 1

            # Generate EEG with stress-related features
            for ch in range(n_channels):
                # Base signal
                signal = np.random.randn(n_time_samples) * 5

                if is_stress:
                    # High stress: suppressed alpha, elevated beta
                    alpha = 5 * np.sin(2 * np.pi * 10 * t)
                    beta = 15 * np.sin(2 * np.pi * 20 * t)
                else:
                    # Low stress: elevated alpha, reduced beta
                    alpha = 20 * np.sin(2 * np.pi * 10 * t)
                    beta = 5 * np.sin(2 * np.pi * 20 * t)

                signal += alpha + beta
                data[idx, ch] = signal

    return data, labels.astype(np.int64), subject_ids.astype(np.int64)


if __name__ == "__main__":
    print("Testing Dataset Loaders")
    print("=" * 50)

    # Generate synthetic data
    print("\nGenerating synthetic dataset...")
    data, labels, subjects = generate_synthetic_dataset(
        n_subjects=10,
        n_trials_per_subject=20
    )

    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique subjects: {np.unique(subjects)}")
    print(f"Class balance: {labels.mean():.2%} high stress")

    # Test LOSO splits
    print("\nTesting LOSO cross-validation...")
    for train_data, train_labels, test_data, test_labels, subject in create_loso_splits(data, labels, subjects):
        print(f"Subject {subject}: Train={len(train_labels)}, Test={len(test_labels)}")
        break  # Just show first split

    # Test DataLoader
    print("\nTesting DataLoader...")
    train_loader, test_loader = create_dataloaders(
        data[:180], labels[:180],
        data[180:], labels[180:],
        batch_size=16
    )

    for batch in train_loader:
        print(f"Batch EEG shape: {batch['eeg'].shape}")
        print(f"Batch labels shape: {batch['label'].shape}")
        break
