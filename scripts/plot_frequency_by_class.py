#!/usr/bin/env python3
"""
Generate wave frequency distribution by class (Baseline vs Stress).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "paper"

def load_edf_file(filepath):
    """Load EDF file and return data."""
    if MNE_AVAILABLE:
        try:
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            data = raw.get_data()
            fs = raw.info['sfreq']
            return data, fs
        except:
            pass
    return None, None

def compute_average_psd(data, fs=500):
    """Compute average PSD across all channels."""
    n_channels = data.shape[0]
    all_psd = []

    for ch in range(min(n_channels, 32)):
        freqs, psd = signal.welch(data[ch], fs=fs, nperseg=min(1024, len(data[ch])))
        all_psd.append(psd)

    return freqs, np.mean(all_psd, axis=0)

def main():
    edf_dir = DATA_DIR / "EEGMAT" / "eeg-during-mental-arithmetic-tasks-1.0.0"

    if not edf_dir.exists():
        print(f"EDF directory not found: {edf_dir}")
        return

    print("Loading EEG data for frequency analysis...")

    baseline_psds = []
    stress_psds = []
    freqs = None

    # Load data from subjects
    for i in range(36):
        subj_id = f"Subject{i:02d}"

        # Baseline (condition 1)
        baseline_file = edf_dir / f"{subj_id}_1.edf"
        if baseline_file.exists():
            data, fs = load_edf_file(baseline_file)
            if data is not None:
                f, psd = compute_average_psd(data, fs)
                if freqs is None:
                    freqs = f
                baseline_psds.append(psd)

        # Stress (condition 2)
        stress_file = edf_dir / f"{subj_id}_2.edf"
        if stress_file.exists():
            data, fs = load_edf_file(stress_file)
            if data is not None:
                f, psd = compute_average_psd(data, fs)
                stress_psds.append(psd)

    print(f"Loaded {len(baseline_psds)} baseline and {len(stress_psds)} stress recordings")

    # Average across subjects
    baseline_avg = np.mean(baseline_psds, axis=0)
    stress_avg = np.mean(stress_psds, axis=0)
    baseline_std = np.std(baseline_psds, axis=0)
    stress_std = np.std(stress_psds, axis=0)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Full spectrum comparison
    ax1 = axes[0]
    freq_mask = freqs <= 50  # Up to 50 Hz

    ax1.semilogy(freqs[freq_mask], baseline_avg[freq_mask], 'b-', linewidth=2, label='Baseline (Relaxed)')
    ax1.fill_between(freqs[freq_mask],
                     baseline_avg[freq_mask] - baseline_std[freq_mask],
                     baseline_avg[freq_mask] + baseline_std[freq_mask],
                     alpha=0.3, color='blue')

    ax1.semilogy(freqs[freq_mask], stress_avg[freq_mask], 'r-', linewidth=2, label='Stress (Mental Arithmetic)')
    ax1.fill_between(freqs[freq_mask],
                     stress_avg[freq_mask] - stress_std[freq_mask],
                     stress_avg[freq_mask] + stress_std[freq_mask],
                     alpha=0.3, color='red')

    # Add band annotations
    bands = {'Delta\n(0.5-4Hz)': (0.5, 4), 'Theta\n(4-8Hz)': (4, 8),
             'Alpha\n(8-13Hz)': (8, 13), 'Beta\n(13-30Hz)': (13, 30),
             'Gamma\n(30-45Hz)': (30, 45)}

    colors = ['#FFE4E1', '#E6E6FA', '#E0FFE0', '#FFFACD', '#FFE4B5']
    for idx, (name, (low, high)) in enumerate(bands.items()):
        ax1.axvspan(low, high, alpha=0.2, color=colors[idx])
        ax1.text((low + high) / 2, ax1.get_ylim()[1] * 0.5, name,
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax1.set_xlabel('Frequency (Hz)', fontsize=12)
    ax1.set_ylabel('Power Spectral Density (V²/Hz)', fontsize=12)
    ax1.set_title('EEG Power Spectrum: Baseline vs Stress (EEGMAT Dataset)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 50)

    # Plot 2: Band power comparison
    ax2 = axes[1]

    band_names = ['Delta\n(0.5-4Hz)', 'Theta\n(4-8Hz)', 'Alpha\n(8-13Hz)', 'Beta\n(13-30Hz)', 'Gamma\n(30-45Hz)']
    band_ranges = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]

    baseline_powers = []
    stress_powers = []

    for low, high in band_ranges:
        mask = (freqs >= low) & (freqs <= high)
        baseline_powers.append(np.mean(baseline_avg[mask]))
        stress_powers.append(np.mean(stress_avg[mask]))

    x = np.arange(len(band_names))
    width = 0.35

    bars1 = ax2.bar(x - width/2, baseline_powers, width, label='Baseline', color='steelblue', edgecolor='black')
    bars2 = ax2.bar(x + width/2, stress_powers, width, label='Stress', color='indianred', edgecolor='black')

    ax2.set_xlabel('Frequency Band', fontsize=12)
    ax2.set_ylabel('Average Power (V²/Hz)', fontsize=12)
    ax2.set_title('Band Power Comparison: Baseline vs Stress', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(band_names)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add percentage change labels
    for i, (b, s) in enumerate(zip(baseline_powers, stress_powers)):
        pct_change = ((s - b) / b) * 100
        color = 'green' if pct_change > 0 else 'red'
        ax2.annotate(f'{pct_change:+.1f}%', xy=(i, max(b, s)),
                    ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')

    plt.tight_layout()

    # Save figure
    OUTPUT_DIR.mkdir(exist_ok=True)
    save_path = OUTPUT_DIR / "fig_frequency_by_class.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {save_path}")

    # Print band power summary
    print("\n" + "="*60)
    print("BAND POWER SUMMARY")
    print("="*60)
    print(f"{'Band':<20} {'Baseline':<15} {'Stress':<15} {'Change':<10}")
    print("-"*60)
    for name, bp, sp in zip(band_names, baseline_powers, stress_powers):
        name_clean = name.replace('\n', ' ')
        pct = ((sp - bp) / bp) * 100
        print(f"{name_clean:<20} {bp:<15.2e} {sp:<15.2e} {pct:+.1f}%")

if __name__ == "__main__":
    main()
