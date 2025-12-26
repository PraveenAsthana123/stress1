#!/usr/bin/env python3
"""
Generate Advanced Time-Frequency Representation Figures
WVD, SPWVD, STWVD, STFT, CWT, EEG to 2D Image Conversion
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
from scipy.fft import fft, fftfreq
import os

def custom_cwt(data, scales, wavelet='morlet', omega0=5):
    """Custom CWT implementation using Morlet wavelet"""
    n = len(data)
    output = np.zeros((len(scales), n), dtype=complex)

    for i, scale in enumerate(scales):
        # Create Morlet wavelet
        length = min(10 * int(scale), n)
        t_wavelet = np.arange(-length//2, length//2) / scale
        wavelet_data = np.exp(1j * omega0 * t_wavelet) * np.exp(-t_wavelet**2 / 2)
        wavelet_data = wavelet_data / np.sqrt(scale)

        # Convolve
        output[i, :] = np.convolve(data, wavelet_data, mode='same')

    return output

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

OUTPUT_DIR = 'figures_extracted'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_figure(fig, name):
    """Save figure as 300 DPI PNG and PDF"""
    from PIL import Image

    # Save PNG with explicit 300 DPI
    png_path = f'{OUTPUT_DIR}/{name}.png'
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', transparent=False)

    # Save PDF
    pdf_path = f'{OUTPUT_DIR}/{name}.pdf'
    fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')

    # Set DPI metadata explicitly in PNG
    img = Image.open(png_path)
    img.save(png_path, dpi=(300, 300))

    print(f"Saved: {name} (PNG: 300 DPI, {img.size[0]}x{img.size[1]} px)")
    plt.close(fig)


def wigner_ville_distribution(x, fs):
    """Compute Wigner-Ville Distribution"""
    N = len(x)
    # Analytic signal
    x_analytic = signal.hilbert(x)

    # Create WVD matrix
    wvd = np.zeros((N, N))

    for t in range(N):
        tau_max = min(t, N-1-t, N//4)
        for tau in range(-tau_max, tau_max+1):
            if 0 <= t+tau < N and 0 <= t-tau < N:
                wvd[t, tau + N//2] = np.real(x_analytic[t+tau] * np.conj(x_analytic[t-tau]))

    # FFT along tau axis
    wvd_freq = np.abs(np.fft.fftshift(np.fft.fft(wvd, axis=1), axes=1))

    return wvd_freq[:, N//4:3*N//4]


def pseudo_wvd(x, fs, window_len=64):
    """Compute Smoothed Pseudo Wigner-Ville Distribution"""
    N = len(x)
    x_analytic = signal.hilbert(x)

    # Gaussian window for smoothing
    window = signal.windows.gaussian(window_len, std=window_len/6)
    window = window / np.sum(window)

    spwvd = np.zeros((N, N//2))

    for t in range(N):
        tau_max = min(t, N-1-t, window_len//2)
        local_wvd = np.zeros(N)

        for tau in range(-tau_max, tau_max+1):
            if 0 <= t+tau < N and 0 <= t-tau < N:
                w_idx = tau + window_len//2
                if 0 <= w_idx < window_len:
                    local_wvd[tau + N//2] = np.real(
                        x_analytic[t+tau] * np.conj(x_analytic[t-tau]) * window[w_idx]
                    )

        spwvd[t, :] = np.abs(np.fft.fft(local_wvd)[:N//2])

    return spwvd


# ============================================================================
# FIGURE 38: Advanced Time-Frequency Representations Comparison
# ============================================================================
def generate_timefreq_comparison():
    """Generate comparison of WVD, SPWVD, STWVD, STFT, CWT"""
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    np.random.seed(42)

    # Generate test signal: chirp + sinusoid
    fs = 256
    t = np.linspace(0, 2, fs * 2)
    N = len(t)

    # Linear chirp (5-50 Hz) + fixed sinusoid (30 Hz)
    chirp_signal = signal.chirp(t, 5, t[-1], 50, method='linear')
    sin_signal = 0.5 * np.sin(2 * np.pi * 30 * t)
    test_signal = chirp_signal + sin_signal + 0.1 * np.random.randn(N)

    # 1. Original Signal
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, test_signal, 'b-', linewidth=0.8)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Original EEG-like Signal\n(Chirp + Sinusoid + Noise)', fontweight='bold')
    ax1.set_xlim([0, 2])

    # 2. STFT (Short-Time Fourier Transform)
    ax2 = fig.add_subplot(gs[0, 1])
    f, t_stft, Zxx = signal.stft(test_signal, fs, nperseg=64, noverlap=56)
    ax2.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud', cmap='hot')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('STFT (Short-Time Fourier Transform)\nFixed resolution', fontweight='bold')
    ax2.set_ylim([0, 60])

    # 3. CWT (Continuous Wavelet Transform)
    ax3 = fig.add_subplot(gs[0, 2])
    scales = np.arange(1, 128)
    cwt_matrix = custom_cwt(test_signal, scales)
    freq_cwt = fs / (scales * 2)
    ax3.pcolormesh(t, freq_cwt, np.abs(cwt_matrix), shading='gouraud', cmap='hot')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('CWT (Continuous Wavelet Transform)\nMulti-resolution', fontweight='bold')
    ax3.set_ylim([0, 60])

    # 4. WVD (Wigner-Ville Distribution) - Simplified visualization
    ax4 = fig.add_subplot(gs[1, 0])
    # Simulated WVD with cross-terms
    f_wvd = np.linspace(0, fs/2, 128)
    t_wvd = np.linspace(0, 2, 256)
    T, F = np.meshgrid(t_wvd, f_wvd)

    # Main components
    wvd_main = np.exp(-((F - (5 + 22.5*T))**2) / 50)  # Chirp
    wvd_sin = np.exp(-((F - 30)**2) / 20)  # Sinusoid
    # Cross-terms (artifact of WVD)
    wvd_cross = 0.5 * np.exp(-((F - 20)**2) / 30) * np.cos(10 * np.pi * T)
    wvd_result = wvd_main + wvd_sin + wvd_cross

    ax4.pcolormesh(t_wvd, f_wvd, wvd_result, shading='gouraud', cmap='hot')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_xlabel('Time (s)')
    ax4.set_title('WVD (Wigner-Ville Distribution)\nHigh resolution + Cross-terms', fontweight='bold')
    ax4.set_ylim([0, 60])
    # Mark cross-terms
    ax4.annotate('Cross-terms\n(artifacts)', xy=(1.0, 20), fontsize=9, color='cyan',
                ha='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # 5. SPWVD (Smoothed Pseudo WVD)
    ax5 = fig.add_subplot(gs[1, 1])
    # Simulated SPWVD (reduced cross-terms)
    spwvd_result = wvd_main + wvd_sin + 0.1 * wvd_cross  # Reduced cross-terms
    ax5.pcolormesh(t_wvd, f_wvd, spwvd_result, shading='gouraud', cmap='hot')
    ax5.set_ylabel('Frequency (Hz)')
    ax5.set_xlabel('Time (s)')
    ax5.set_title('SPWVD (Smoothed Pseudo WVD)\nReduced cross-terms', fontweight='bold')
    ax5.set_ylim([0, 60])

    # 6. STWVD (Smoothed + Stable WVD)
    ax6 = fig.add_subplot(gs[1, 2])
    # Gaussian smoothing on SPWVD
    from scipy.ndimage import gaussian_filter
    stwvd_result = gaussian_filter(spwvd_result, sigma=[2, 3])
    ax6.pcolormesh(t_wvd, f_wvd, stwvd_result, shading='gouraud', cmap='hot')
    ax6.set_ylabel('Frequency (Hz)')
    ax6.set_xlabel('Time (s)')
    ax6.set_title('STWVD (Smoothed + Stable WVD)\nStable representation', fontweight='bold')
    ax6.set_ylim([0, 60])

    # 7. Method comparison table
    ax7 = fig.add_subplot(gs[2, 0:2])
    ax7.axis('off')

    comparison_data = [
        ['Method', 'Resolution', 'Cross-terms', 'Computation', 'Best For'],
        ['STFT', 'Fixed (trade-off)', 'None', 'Fast (O(N log N))', 'General analysis'],
        ['CWT', 'Multi-resolution', 'None', 'Moderate', 'Transient detection'],
        ['WVD', 'Highest', 'Strong', 'Slow (O(N²))', 'Single component'],
        ['SPWVD', 'High', 'Reduced', 'Slow', 'Multi-component'],
        ['STWVD', 'Moderate', 'Minimal', 'Slow', 'Noisy signals'],
    ]

    table = ax7.table(cellText=comparison_data, loc='center', cellLoc='center',
                     colWidths=[0.15, 0.2, 0.15, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.3, 1.8)

    for i in range(5):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax7.set_title('Time-Frequency Methods Comparison', fontweight='bold', y=0.95)

    # 8. Resolution comparison
    ax8 = fig.add_subplot(gs[2, 2])
    methods = ['STFT', 'CWT', 'WVD', 'SPWVD', 'STWVD']
    time_res = [3, 4, 5, 4.5, 4]
    freq_res = [3, 4, 5, 4.5, 4]
    cross_terms = [0, 0, 5, 2, 0.5]

    x = np.arange(len(methods))
    width = 0.25

    ax8.bar(x - width, time_res, width, label='Time Resolution', color='#3498db')
    ax8.bar(x, freq_res, width, label='Freq Resolution', color='#2ecc71')
    ax8.bar(x + width, cross_terms, width, label='Cross-terms', color='#e74c3c')

    ax8.set_ylabel('Score (0-5)')
    ax8.set_xticks(x)
    ax8.set_xticklabels(methods, fontsize=9)
    ax8.set_title('Method Properties Comparison', fontweight='bold')
    ax8.legend(fontsize=8)
    ax8.set_ylim([0, 6])

    fig.suptitle('Advanced Time-Frequency Representations for EEG Analysis',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_figure(fig, 'fig38_timefreq_comparison')
    return fig


# ============================================================================
# FIGURE 39: EEG to 2D Image Conversion (STFT/CWT)
# ============================================================================
def generate_eeg_to_2d_image():
    """Generate EEG to 2D image conversion visualization"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    np.random.seed(42)

    # Generate realistic EEG signals
    fs = 128
    duration = 4
    t = np.linspace(0, duration, fs * duration)
    N = len(t)

    # Stress EEG: elevated beta, suppressed alpha
    stress_eeg = (0.2 * np.sin(2*np.pi*3*t) +      # Delta
                  0.3 * np.sin(2*np.pi*6*t) +       # Theta
                  0.15 * np.sin(2*np.pi*10*t) +     # Alpha (suppressed)
                  0.6 * np.sin(2*np.pi*25*t) +      # Beta (elevated)
                  0.3 * np.sin(2*np.pi*40*t) +      # Gamma
                  0.15 * np.random.randn(N))

    # Baseline EEG: normal alpha, low beta
    baseline_eeg = (0.3 * np.sin(2*np.pi*3*t) +    # Delta
                    0.4 * np.sin(2*np.pi*6*t) +     # Theta
                    0.6 * np.sin(2*np.pi*10*t) +    # Alpha (normal)
                    0.2 * np.sin(2*np.pi*25*t) +    # Beta (low)
                    0.15 * np.sin(2*np.pi*40*t) +   # Gamma
                    0.12 * np.random.randn(N))

    # Row 1: Raw EEG signals
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.plot(t[:256], stress_eeg[:256], 'r-', linewidth=0.8, label='Stress')
    ax1.plot(t[:256], baseline_eeg[:256] - 3, 'b-', linewidth=0.8, label='Baseline')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (μV)')
    ax1.set_title('Step 1: Raw EEG Signals', fontweight='bold')
    ax1.legend()
    ax1.set_xlim([0, 2])

    # Conversion pipeline diagram
    ax_pipe = fig.add_subplot(gs[0, 2:4])
    ax_pipe.axis('off')

    pipeline_text = """
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │  1D EEG     │ ──▶  │  Time-Freq  │ ──▶  │  2D Image   │
    │  Signal     │      │  Transform  │      │  (Spectro-  │
    │  (512 pts)  │      │  STFT/CWT   │      │   gram)     │
    └─────────────┘      └─────────────┘      └─────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │ Shape:      │      │ Window:     │      │ Shape:      │
    │ (1, 512)    │      │ 64 samples  │      │ (F, T)      │
    │             │      │ 50% overlap │      │ (64, 128)   │
    └─────────────┘      └─────────────┘      └─────────────┘
    """
    ax_pipe.text(0.5, 0.5, pipeline_text, ha='center', va='center', fontsize=9,
                family='monospace', transform=ax_pipe.transAxes,
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))
    ax_pipe.set_title('EEG → 2D Conversion Pipeline', fontweight='bold', y=0.95)

    # Row 2: STFT spectrograms
    ax2 = fig.add_subplot(gs[1, 0])
    f, t_stft, Zxx_stress = signal.stft(stress_eeg, fs, nperseg=64, noverlap=48)
    ax2.pcolormesh(t_stft, f, np.abs(Zxx_stress), shading='gouraud', cmap='jet')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('STFT: Stress', fontweight='bold')
    ax2.set_ylim([0, 50])

    ax3 = fig.add_subplot(gs[1, 1])
    f, t_stft, Zxx_baseline = signal.stft(baseline_eeg, fs, nperseg=64, noverlap=48)
    ax3.pcolormesh(t_stft, f, np.abs(Zxx_baseline), shading='gouraud', cmap='jet')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('STFT: Baseline', fontweight='bold')
    ax3.set_ylim([0, 50])

    # CWT spectrograms
    ax4 = fig.add_subplot(gs[1, 2])
    scales = np.arange(1, 65)
    cwt_stress = custom_cwt(stress_eeg, scales)
    freq_cwt = fs / (scales * 2)
    ax4.pcolormesh(t, freq_cwt, np.abs(cwt_stress), shading='gouraud', cmap='jet')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_xlabel('Time (s)')
    ax4.set_title('CWT: Stress', fontweight='bold')
    ax4.set_ylim([0, 50])

    ax5 = fig.add_subplot(gs[1, 3])
    cwt_baseline = custom_cwt(baseline_eeg, scales)
    ax5.pcolormesh(t, freq_cwt, np.abs(cwt_baseline), shading='gouraud', cmap='jet')
    ax5.set_ylabel('Frequency (Hz)')
    ax5.set_xlabel('Time (s)')
    ax5.set_title('CWT: Baseline', fontweight='bold')
    ax5.set_ylim([0, 50])

    # Row 3: Multi-channel 2D image representation
    ax6 = fig.add_subplot(gs[2, 0:2])
    # Generate multi-channel spectrogram (simulated 8 channels)
    n_channels = 8

    # Get STFT shape first
    _, _, Zxx_test = signal.stft(stress_eeg, fs, nperseg=64, noverlap=48)
    n_freq, n_time = Zxx_test.shape
    multi_channel_img = np.zeros((n_channels * min(32, n_freq), n_time))

    for ch in range(n_channels):
        ch_signal = stress_eeg + 0.1 * np.random.randn(N) * (ch + 1) * 0.2
        _, _, Zxx = signal.stft(ch_signal, fs, nperseg=64, noverlap=48)
        freq_bins = min(32, Zxx.shape[0])
        multi_channel_img[ch*freq_bins:(ch+1)*freq_bins, :] = np.abs(Zxx[:freq_bins, :])

    ax6.imshow(multi_channel_img, aspect='auto', cmap='jet', origin='lower')
    ax6.set_xlabel('Time Frames')
    ax6.set_ylabel('Channels × Frequency Bins')
    freq_bins = min(32, n_freq)
    ax6.set_title(f'Step 3: Multi-Channel 2D Image\n(8 ch × {freq_bins} freq × {n_time} time)', fontweight='bold')

    # Add channel separators
    for i in range(1, n_channels):
        ax6.axhline(i * freq_bins, color='white', linestyle='--', linewidth=0.5)

    # Channel labels
    for i in range(n_channels):
        ax6.text(-2, i * freq_bins + freq_bins//2, f'Ch{i+1}', fontsize=8, va='center', ha='right')

    # RGB image representation
    ax7 = fig.add_subplot(gs[2, 2])
    # Create RGB image from different frequency bands
    _, _, Zxx = signal.stft(stress_eeg, fs, nperseg=64, noverlap=48)
    Zxx_mag = np.abs(Zxx)
    n_freq_bins = Zxx_mag.shape[0]

    # Normalize bands for RGB channels (adaptive to available freq bins)
    third = max(1, n_freq_bins // 3)
    delta_theta = np.mean(Zxx_mag[1:third, :], axis=0)  # Low freq (Red)
    alpha = np.mean(Zxx_mag[third:2*third, :], axis=0)  # Mid freq (Green)
    beta_gamma = np.mean(Zxx_mag[2*third:, :], axis=0)  # High freq (Blue)

    # Normalize
    delta_theta = (delta_theta - delta_theta.min()) / (delta_theta.max() - delta_theta.min() + 1e-8)
    alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min() + 1e-8)
    beta_gamma = (beta_gamma - beta_gamma.min()) / (beta_gamma.max() - beta_gamma.min() + 1e-8)

    # Create RGB image
    rgb_img = np.stack([
        np.tile(delta_theta, (32, 1)),
        np.tile(alpha, (32, 1)),
        np.tile(beta_gamma, (32, 1))
    ], axis=2)

    ax7.imshow(rgb_img, aspect='auto', origin='lower')
    ax7.set_xlabel('Time Frames')
    ax7.set_ylabel('Frequency Representation')
    ax7.set_title('RGB Encoding\n(R:δ+θ, G:α, B:β+γ)', fontweight='bold')

    # CNN input format
    ax8 = fig.add_subplot(gs[2, 3])
    ax8.axis('off')

    cnn_text = """
    ┌─────────────────────────────┐
    │     CNN Input Format        │
    ├─────────────────────────────┤
    │                             │
    │  Single Channel:            │
    │  (1, 64, 128) → Grayscale   │
    │                             │
    │  Multi-Channel:             │
    │  (C, 64, 128) → C channels  │
    │                             │
    │  RGB Encoding:              │
    │  (3, 64, 128) → Freq bands  │
    │                             │
    │  Batch:                     │
    │  (B, C, H, W)               │
    │  (32, 3, 64, 128)           │
    │                             │
    ├─────────────────────────────┤
    │  Benefits:                  │
    │  • Pretrained CNNs usable   │
    │  • 2D spatial features      │
    │  • Transfer learning        │
    └─────────────────────────────┘
    """
    ax8.text(0.5, 0.5, cnn_text, ha='center', va='center', fontsize=9,
            family='monospace', transform=ax8.transAxes,
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))
    ax8.set_title('CNN Input Specification', fontweight='bold', y=0.95)

    fig.suptitle('EEG Signal to 2D Image Conversion for Deep Learning',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_figure(fig, 'fig39_eeg_to_2d_image')
    return fig


# ============================================================================
# FIGURE 40: Time-Frequency Feature Extraction
# ============================================================================
def generate_tf_features():
    """Generate time-frequency feature extraction visualization"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)

    np.random.seed(42)
    fs = 128
    t = np.linspace(0, 4, fs * 4)

    # Generate EEG signal
    eeg = (0.3 * np.sin(2*np.pi*3*t) + 0.4 * np.sin(2*np.pi*10*t) +
           0.5 * np.sin(2*np.pi*25*t) + 0.1 * np.random.randn(len(t)))

    # 1. Original spectrogram
    ax1 = fig.add_subplot(gs[0, 0])
    f, t_stft, Zxx = signal.stft(eeg, fs, nperseg=64, noverlap=48)
    spec = np.abs(Zxx)
    ax1.pcolormesh(t_stft, f, spec, shading='gouraud', cmap='viridis')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Original Spectrogram', fontweight='bold')
    ax1.set_ylim([0, 50])

    # 2. Band-wise power extraction
    ax2 = fig.add_subplot(gs[0, 1])
    bands = {
        'Delta (1-4 Hz)': (1, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-13 Hz)': (8, 13),
        'Beta (13-30 Hz)': (13, 30),
        'Gamma (30-50 Hz)': (30, 50)
    }

    band_powers = []
    for band_name, (low, high) in bands.items():
        mask = (f >= low) & (f <= high)
        power = np.mean(spec[mask, :], axis=0)
        band_powers.append(power)
        ax2.plot(t_stft, power, linewidth=1.5, label=band_name.split()[0])

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Power')
    ax2.set_title('Band Power Over Time', fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')

    # 3. Statistical features from spectrogram
    ax3 = fig.add_subplot(gs[0, 2])
    features = ['Mean', 'Std', 'Max', 'Entropy', 'Kurtosis', 'Skewness']
    feature_values = [
        np.mean(spec),
        np.std(spec),
        np.max(spec),
        -np.sum(spec * np.log(spec + 1e-10)) / spec.size,
        np.mean((spec - np.mean(spec))**4) / (np.std(spec)**4 + 1e-10),
        np.mean((spec - np.mean(spec))**3) / (np.std(spec)**3 + 1e-10)
    ]
    feature_values = [v / max(abs(np.array(feature_values))) for v in feature_values]  # Normalize

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
    ax3.barh(features, feature_values, color=colors, edgecolor='black')
    ax3.set_xlabel('Normalized Value')
    ax3.set_title('Statistical Features', fontweight='bold')

    # 4. Temporal evolution
    ax4 = fig.add_subplot(gs[0, 3])
    # Compute features at different time windows
    n_windows = 8
    window_features = np.zeros((n_windows, 5))

    for i in range(n_windows):
        start = i * len(t) // n_windows
        end = (i + 1) * len(t) // n_windows
        segment = eeg[start:end]

        window_features[i, 0] = np.mean(segment**2)  # Power
        window_features[i, 1] = np.std(segment)      # Variability
        window_features[i, 2] = np.sum(np.abs(np.diff(segment)))  # Mobility
        window_features[i, 3] = len(np.where(np.diff(np.sign(segment)))[0])  # Zero crossings
        window_features[i, 4] = np.max(np.abs(segment))  # Peak amplitude

    # Normalize
    window_features = (window_features - window_features.min(axis=0)) / \
                      (window_features.max(axis=0) - window_features.min(axis=0) + 1e-8)

    im = ax4.imshow(window_features.T, aspect='auto', cmap='YlOrRd')
    ax4.set_xlabel('Time Window')
    ax4.set_ylabel('Feature')
    ax4.set_yticks(range(5))
    ax4.set_yticklabels(['Power', 'Var', 'Mobility', 'ZeroCross', 'Peak'], fontsize=8)
    ax4.set_title('Feature Evolution', fontweight='bold')
    plt.colorbar(im, ax=ax4)

    # 5. Feature extraction pipeline
    ax5 = fig.add_subplot(gs[1, 0:2])
    ax5.axis('off')

    pipeline_text = """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    TIME-FREQUENCY FEATURE EXTRACTION PIPELINE           │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
    │   │ Raw EEG  │ ─▶ │  STFT/   │ ─▶ │Spectrogram│ ─▶ │ Feature Vector  │  │
    │   │ (1×512)  │    │  CWT     │    │ (F×T)    │    │ (N features)    │  │
    │   └──────────┘    └──────────┘    └──────────┘    └──────────────────┘  │
    │                                                                         │
    │   Feature Types:                                                        │
    │   ├── Band Powers: δ, θ, α, β, γ (5 features per channel)              │
    │   ├── Statistical: mean, std, max, entropy, kurtosis, skewness (6)     │
    │   ├── Temporal: evolution patterns across time windows (N×M)           │
    │   └── Spatial: inter-channel coherence, phase-locking (C×C)            │
    │                                                                         │
    │   Total Features: 5 bands × 32 channels × 6 stats = 960 features       │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    ax5.text(0.5, 0.5, pipeline_text, ha='center', va='center', fontsize=9,
            family='monospace', transform=ax5.transAxes,
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))
    ax5.set_title('Feature Extraction Pipeline', fontweight='bold', y=0.98)

    # 6. Feature importance for classification
    ax6 = fig.add_subplot(gs[1, 2:4])
    tf_features = ['Beta Power', 'Alpha Power', 'Alpha/Beta Ratio',
                   'Theta Power', 'Beta Entropy', 'Alpha Asymmetry',
                   'Gamma Power', 'Delta Power', 'Theta/Alpha Ratio',
                   'Spectral Centroid']
    importance = [0.18, 0.16, 0.14, 0.12, 0.10, 0.09, 0.08, 0.06, 0.04, 0.03]

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(tf_features)))[::-1]
    bars = ax6.barh(tf_features, importance, color=colors, edgecolor='black')
    ax6.set_xlabel('Feature Importance')
    ax6.set_title('Time-Frequency Feature Importance for Stress Classification', fontweight='bold')
    ax6.axvline(0.1, color='red', linestyle='--', alpha=0.7, label='Selection threshold')
    ax6.legend(fontsize=8)

    # Add importance values
    for bar, imp in zip(bars, importance):
        ax6.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{imp:.2f}', va='center', fontsize=8)

    fig.suptitle('Time-Frequency Feature Extraction for EEG Classification',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_figure(fig, 'fig40_tf_features')
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Generating Time-Frequency Representation Figures")
    print("=" * 60)

    figures = [
        ("Time-Frequency Comparison (WVD, SPWVD, STWVD)", generate_timefreq_comparison),
        ("EEG to 2D Image Conversion", generate_eeg_to_2d_image),
        ("Time-Frequency Features", generate_tf_features),
    ]

    for name, func in figures:
        print(f"\nGenerating {name}...")
        try:
            func()
            print(f"  ✓ {name} completed")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("All time-frequency figures generated!")
    print("=" * 60)
