#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Research-Grade Visualization Module for Medical/Scientific Publications
================================================================================

This module provides publication-ready visualization tools following journal
standards for Nature, IEEE, Elsevier, and Springer publications.

Categories:
    1. Trend Charts & Time-Series (EEG, Vitals, Signals)
    2. Medical & Clinical Data Analysis
    3. EEG / BCI / Neuroinformatics
    4. Statistical Analysis (PhD-Level)
    5. Research-Paper-Grade Figures
    6. Flowcharts, Pipelines & Methodology Diagrams

Reference Standards:
    - Matplotlib for deterministic, journal-safe figures
    - Seaborn for statistical visualizations with CI
    - MNE-Python for medical-grade EEG/MEG
    - PyWavelets for time-frequency analysis
    - Lifelines for survival analysis
    - Pingouin for modern statistical analysis
    - Graphviz for flowcharts and pipelines

================================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass
import warnings

# Core visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Statistical analysis
try:
    from scipy import stats, signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.stats import power as sm_power
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False

# EEG processing
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

# Signal processing
try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False

# Survival analysis
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

# Dimensionality reduction
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Graphviz
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False


# =============================================================================
# PUBLICATION SETTINGS
# =============================================================================

# IEEE/Nature/Elsevier standard settings
PUBLICATION_RCPARAMS = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'text.usetex': False,  # Set True if LaTeX available
}

# Color schemes for publications
PUBLICATION_COLORS = {
    'primary': '#2C3E50',
    'secondary': '#E74C3C',
    'tertiary': '#3498DB',
    'quaternary': '#27AE60',
    'neutral': '#95A5A6',
    'stress': '#E74C3C',
    'baseline': '#3498DB',
    'highlight': '#F39C12',
}

# EEG frequency bands
EEG_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45),
}


def apply_publication_style():
    """Apply publication-ready matplotlib style."""
    if MATPLOTLIB_AVAILABLE:
        plt.rcParams.update(PUBLICATION_RCPARAMS)
        if SEABORN_AVAILABLE:
            sns.set_style("whitegrid")
            sns.set_context("paper")


# =============================================================================
# 1. TREND CHARTS & TIME-SERIES
# =============================================================================

class TrendChartGenerator:
    """Generate trend charts and time-series visualizations."""

    def __init__(self, output_dir: str = './figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        apply_publication_style()

    def plot_eeg_signal(
        self,
        data: np.ndarray,
        fs: float = 128.0,  # SAM-40: 128 Hz
        channel_names: List[str] = None,
        title: str = "EEG Signal",
        save_path: str = None
    ) -> plt.Figure:
        """Plot multi-channel EEG signal (MNE-style)."""
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels = data.shape[0]
        n_samples = data.shape[1]
        time = np.arange(n_samples) / fs

        if channel_names is None:
            channel_names = [f'Ch{i+1}' for i in range(n_channels)]

        fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2*n_channels), sharex=True)
        if n_channels == 1:
            axes = [axes]

        for i, (ax, ch_name) in enumerate(zip(axes, channel_names)):
            ax.plot(time, data[i], color=PUBLICATION_COLORS['primary'], linewidth=0.5)
            ax.set_ylabel(ch_name, fontsize=9)
            ax.set_xlim([0, time[-1]])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        axes[-1].set_xlabel('Time (s)')
        plt.suptitle(title, fontsize=12, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=600, bbox_inches='tight')

        return fig

    def plot_psd_bands(
        self,
        data: np.ndarray,
        fs: float = 128.0,  # SAM-40: 128 Hz
        title: str = "Power Spectral Density",
        save_path: str = None
    ) -> plt.Figure:
        """Plot PSD with frequency band annotations."""
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for PSD computation")

        freqs, psd = signal.welch(data, fs=fs, nperseg=min(256, len(data)))

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.semilogy(freqs, psd, color=PUBLICATION_COLORS['primary'], linewidth=1.5)

        # Add band shading
        colors = ['#9B59B6', '#3498DB', '#27AE60', '#F39C12', '#E74C3C']
        for (band_name, (low, high)), color in zip(EEG_BANDS.items(), colors):
            ax.axvspan(low, high, alpha=0.2, color=color, label=f'{band_name} ({low}-{high} Hz)')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density (μV²/Hz)')
        ax.set_title(title)
        ax.set_xlim([0, 50])
        ax.legend(loc='upper right', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=600, bbox_inches='tight')

        return fig

    def plot_time_frequency(
        self,
        data: np.ndarray,
        fs: float = 128.0,  # SAM-40: 128 Hz
        wavelet: str = 'morl',
        title: str = "Time-Frequency Analysis",
        save_path: str = None
    ) -> plt.Figure:
        """Plot wavelet time-frequency representation."""
        if not PYWAVELETS_AVAILABLE:
            raise ImportError("PyWavelets required for time-frequency analysis")

        scales = np.arange(1, 128)
        coef, freqs = pywt.cwt(data, scales, wavelet, sampling_period=1/fs)

        fig, ax = plt.subplots(figsize=(12, 6))

        time = np.arange(len(data)) / fs
        im = ax.pcolormesh(time, freqs, np.abs(coef), cmap='jet', shading='auto')

        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_title(title)
        ax.set_ylim([0, 50])

        plt.colorbar(im, ax=ax, label='Magnitude')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=600, bbox_inches='tight')

        return fig


# =============================================================================
# 2. MEDICAL & CLINICAL DATA ANALYSIS
# =============================================================================

class ClinicalVisualization:
    """Clinical and medical data visualization tools."""

    def __init__(self, output_dir: str = './figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        apply_publication_style()

    def plot_bland_altman(
        self,
        method1: np.ndarray,
        method2: np.ndarray,
        method1_name: str = "Method 1",
        method2_name: str = "Method 2",
        title: str = "Bland-Altman Plot",
        save_path: str = None
    ) -> plt.Figure:
        """Create Bland-Altman agreement plot."""
        mean = (method1 + method2) / 2
        diff = method1 - method2
        md = np.mean(diff)
        sd = np.std(diff)

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(mean, diff, alpha=0.6, color=PUBLICATION_COLORS['primary'])
        ax.axhline(md, color=PUBLICATION_COLORS['secondary'], linestyle='-', label=f'Mean: {md:.3f}')
        ax.axhline(md + 1.96*sd, color=PUBLICATION_COLORS['tertiary'], linestyle='--', label=f'+1.96 SD: {md+1.96*sd:.3f}')
        ax.axhline(md - 1.96*sd, color=PUBLICATION_COLORS['tertiary'], linestyle='--', label=f'-1.96 SD: {md-1.96*sd:.3f}')

        ax.set_xlabel(f'Mean of {method1_name} and {method2_name}')
        ax.set_ylabel(f'{method1_name} - {method2_name}')
        ax.set_title(title)
        ax.legend(loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=600, bbox_inches='tight')

        return fig

    def plot_kaplan_meier(
        self,
        durations: np.ndarray,
        events: np.ndarray,
        groups: np.ndarray = None,
        group_names: List[str] = None,
        title: str = "Kaplan-Meier Survival Curve",
        save_path: str = None
    ) -> plt.Figure:
        """Create Kaplan-Meier survival plot."""
        if not LIFELINES_AVAILABLE:
            raise ImportError("lifelines required for survival analysis")

        fig, ax = plt.subplots(figsize=(10, 8))

        if groups is None:
            kmf = KaplanMeierFitter()
            kmf.fit(durations, events, label='All subjects')
            kmf.plot_survival_function(ax=ax, ci_show=True)
        else:
            unique_groups = np.unique(groups)
            if group_names is None:
                group_names = [f'Group {g}' for g in unique_groups]

            colors = [PUBLICATION_COLORS['primary'], PUBLICATION_COLORS['secondary'],
                     PUBLICATION_COLORS['tertiary'], PUBLICATION_COLORS['quaternary']]

            for i, (group, name) in enumerate(zip(unique_groups, group_names)):
                mask = groups == group
                kmf = KaplanMeierFitter()
                kmf.fit(durations[mask], events[mask], label=name)
                kmf.plot_survival_function(ax=ax, ci_show=True, color=colors[i % len(colors)])

            # Log-rank test
            if len(unique_groups) == 2:
                mask1 = groups == unique_groups[0]
                mask2 = groups == unique_groups[1]
                result = logrank_test(durations[mask1], durations[mask2], events[mask1], events[mask2])
                ax.text(0.95, 0.95, f'Log-rank p = {result.p_value:.4f}',
                       transform=ax.transAxes, ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Time')
        ax.set_ylabel('Survival Probability')
        ax.set_title(title)
        ax.legend(loc='lower left')

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=600, bbox_inches='tight')

        return fig

    def plot_forest_meta_analysis(
        self,
        studies: List[Dict],
        title: str = "Forest Plot (Meta-Analysis)",
        save_path: str = None
    ) -> plt.Figure:
        """Create forest plot for meta-analysis."""
        fig, ax = plt.subplots(figsize=(12, len(studies) * 0.5 + 2))

        y_positions = np.arange(len(studies))

        for i, study in enumerate(studies):
            effect = study['effect']
            ci_low = study.get('ci_low', effect - 0.3)
            ci_high = study.get('ci_high', effect + 0.3)
            weight = study.get('weight', 1.0)

            # Plot confidence interval
            ax.hlines(i, ci_low, ci_high, colors=PUBLICATION_COLORS['primary'], linewidth=2)
            # Plot effect size
            ax.scatter(effect, i, s=weight*100, color=PUBLICATION_COLORS['primary'], zorder=3)

        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([s['name'] for s in studies])
        ax.set_xlabel('Effect Size')
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=600, bbox_inches='tight')

        return fig


# =============================================================================
# 3. STATISTICAL ANALYSIS VISUALIZATIONS
# =============================================================================

class StatisticalVisualization:
    """PhD-level statistical visualizations."""

    def __init__(self, output_dir: str = './figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        apply_publication_style()

    def plot_effect_size_comparison(
        self,
        groups: Dict[str, Tuple[np.ndarray, np.ndarray]],
        title: str = "Effect Size Comparison",
        save_path: str = None
    ) -> plt.Figure:
        """Plot with effect sizes (Cohen's d) - mandatory for modern papers."""
        if not SEABORN_AVAILABLE:
            raise ImportError("seaborn required for statistical plots")

        fig, axes = plt.subplots(1, len(groups), figsize=(4*len(groups), 6))
        if len(groups) == 1:
            axes = [axes]

        for ax, (name, (group1, group2)) in zip(axes, groups.items()):
            # Cohen's d
            pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
            cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std

            # Create violin plot
            data = pd.DataFrame({
                'Value': np.concatenate([group1, group2]),
                'Group': ['Control']*len(group1) + ['Treatment']*len(group2)
            })
            sns.violinplot(data=data, x='Group', y='Value', ax=ax, palette=[PUBLICATION_COLORS['baseline'], PUBLICATION_COLORS['stress']])

            # Add effect size annotation
            ax.text(0.5, 0.95, f"Cohen's d = {cohens_d:.2f}", transform=ax.transAxes,
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_title(name)

        plt.suptitle(title, fontsize=12, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=600, bbox_inches='tight')

        return fig

    def plot_power_analysis(
        self,
        effect_sizes: np.ndarray = None,
        sample_sizes: List[int] = None,
        alpha: float = 0.05,
        title: str = "Statistical Power Analysis",
        save_path: str = None
    ) -> plt.Figure:
        """Plot power analysis curves - increasingly required by editors."""
        if effect_sizes is None:
            effect_sizes = np.linspace(0.1, 2.0, 50)
        if sample_sizes is None:
            sample_sizes = [20, 40, 60, 80, 100]

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.viridis(np.linspace(0, 1, len(sample_sizes)))

        for n, color in zip(sample_sizes, colors):
            power = 1 - stats.norm.cdf(stats.norm.ppf(1 - alpha/2) - effect_sizes * np.sqrt(n/2))
            ax.plot(effect_sizes, power, color=color, linewidth=2, label=f'n = {n}')

        ax.axhline(y=0.8, color='red', linestyle='--', label='Power = 0.80')
        ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label="Cohen's d = 0.5 (medium)")
        ax.axvline(x=0.8, color='gray', linestyle=':', alpha=0.5, label="Cohen's d = 0.8 (large)")

        ax.set_xlabel("Effect Size (Cohen's d)")
        ax.set_ylabel('Statistical Power (1 - β)')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.set_xlim([0, 2])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=600, bbox_inches='tight')

        return fig

    def plot_regression_diagnostics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        residuals: np.ndarray = None,
        title: str = "Regression Diagnostics",
        save_path: str = None
    ) -> plt.Figure:
        """Create regression diagnostic plots."""
        if residuals is None:
            residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Actual vs Predicted
        ax = axes[0, 0]
        ax.scatter(y_true, y_pred, alpha=0.6, color=PUBLICATION_COLORS['primary'])
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Actual vs Predicted')

        # 2. Residuals vs Predicted
        ax = axes[0, 1]
        ax.scatter(y_pred, residuals, alpha=0.6, color=PUBLICATION_COLORS['primary'])
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Predicted')

        # 3. Residual histogram
        ax = axes[1, 0]
        ax.hist(residuals, bins=30, color=PUBLICATION_COLORS['primary'], alpha=0.7, edgecolor='black')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Residual Distribution')

        # 4. Q-Q plot
        ax = axes[1, 1]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=600, bbox_inches='tight')

        return fig


# =============================================================================
# 4. FLOWCHARTS & METHODOLOGY DIAGRAMS
# =============================================================================

class FlowchartGenerator:
    """Generate flowcharts and pipeline diagrams using Graphviz."""

    def __init__(self, output_dir: str = './figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_study_workflow(
        self,
        stages: List[Dict[str, Any]],
        title: str = "Study Workflow",
        save_path: str = None,
        format: str = 'svg'
    ) -> str:
        """Create study workflow diagram."""
        if not GRAPHVIZ_AVAILABLE:
            warnings.warn("graphviz not available. Install with: pip install graphviz")
            return None

        dot = graphviz.Digraph(comment=title)
        dot.attr(rankdir='TB', size='10,10')
        dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')

        for i, stage in enumerate(stages):
            node_id = f'stage_{i}'
            label = stage.get('name', f'Stage {i+1}')
            color = stage.get('color', 'lightblue')
            dot.node(node_id, label, fillcolor=color)

            if i > 0:
                prev_id = f'stage_{i-1}'
                edge_label = stage.get('edge_label', '')
                dot.edge(prev_id, node_id, label=edge_label)

        filepath = self.output_dir / (save_path or 'workflow')
        dot.render(filepath, format=format, cleanup=True)

        return str(filepath) + f'.{format}'

    def create_ml_pipeline(
        self,
        components: List[Dict[str, Any]],
        title: str = "ML Pipeline",
        save_path: str = None,
        format: str = 'svg'
    ) -> str:
        """Create ML pipeline diagram."""
        if not GRAPHVIZ_AVAILABLE:
            warnings.warn("graphviz not available")
            return None

        dot = graphviz.Digraph(comment=title)
        dot.attr(rankdir='LR', size='12,8')

        # Define node styles
        styles = {
            'data': {'shape': 'cylinder', 'fillcolor': '#E8F4FD'},
            'process': {'shape': 'box', 'fillcolor': '#FFF3E0'},
            'model': {'shape': 'box3d', 'fillcolor': '#E8F5E9'},
            'output': {'shape': 'note', 'fillcolor': '#FCE4EC'},
        }

        for i, comp in enumerate(components):
            node_id = f'comp_{i}'
            label = comp.get('name', f'Component {i+1}')
            comp_type = comp.get('type', 'process')
            style = styles.get(comp_type, styles['process'])

            dot.node(node_id, label, style='filled', **style)

            if i > 0:
                prev_id = f'comp_{i-1}'
                dot.edge(prev_id, node_id)

        filepath = self.output_dir / (save_path or 'pipeline')
        dot.render(filepath, format=format, cleanup=True)

        return str(filepath) + f'.{format}'

    def create_architecture_diagram(
        self,
        layers: List[Dict[str, Any]],
        title: str = "Model Architecture",
        save_path: str = None,
        format: str = 'svg'
    ) -> str:
        """Create neural network architecture diagram."""
        if not GRAPHVIZ_AVAILABLE:
            warnings.warn("graphviz not available")
            return None

        dot = graphviz.Digraph(comment=title)
        dot.attr(rankdir='TB', size='10,12')

        for i, layer in enumerate(layers):
            node_id = f'layer_{i}'
            name = layer.get('name', f'Layer {i+1}')
            params = layer.get('params', '')
            label = f'{name}\n{params}' if params else name
            color = layer.get('color', '#E3F2FD')

            dot.node(node_id, label, shape='box', style='filled,rounded', fillcolor=color)

            if i > 0:
                prev_id = f'layer_{i-1}'
                dot.edge(prev_id, node_id)

        filepath = self.output_dir / (save_path or 'architecture')
        dot.render(filepath, format=format, cleanup=True)

        return str(filepath) + f'.{format}'


# =============================================================================
# 5. MASTER VISUALIZATION CLASS
# =============================================================================

class ResearchVisualizationSuite:
    """
    Master class for all research-grade visualizations.

    Usage:
        from src.analysis import ResearchVisualizationSuite

        viz = ResearchVisualizationSuite(output_dir='./figures')

        # Time-series
        viz.trend.plot_eeg_signal(data, fs=256, save_path='eeg.pdf')

        # Clinical
        viz.clinical.plot_kaplan_meier(durations, events, save_path='survival.pdf')

        # Statistical
        viz.stats.plot_effect_size_comparison(groups, save_path='effects.pdf')

        # Flowcharts
        viz.flowchart.create_ml_pipeline(components, save_path='pipeline', format='svg')
    """

    def __init__(self, output_dir: str = './figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sub-modules
        self.trend = TrendChartGenerator(output_dir)
        self.clinical = ClinicalVisualization(output_dir)
        self.stats = StatisticalVisualization(output_dir)
        self.flowchart = FlowchartGenerator(output_dir)

        # Apply publication style
        apply_publication_style()

    def check_dependencies(self) -> Dict[str, bool]:
        """Check which visualization dependencies are available."""
        return {
            'matplotlib': MATPLOTLIB_AVAILABLE,
            'seaborn': SEABORN_AVAILABLE,
            'pandas': PANDAS_AVAILABLE,
            'scipy': SCIPY_AVAILABLE,
            'statsmodels': STATSMODELS_AVAILABLE,
            'pingouin': PINGOUIN_AVAILABLE,
            'mne': MNE_AVAILABLE,
            'pywavelets': PYWAVELETS_AVAILABLE,
            'lifelines': LIFELINES_AVAILABLE,
            'umap': UMAP_AVAILABLE,
            'graphviz': GRAPHVIZ_AVAILABLE,
        }

    def print_dependency_status(self):
        """Print status of all visualization dependencies."""
        deps = self.check_dependencies()
        print("=" * 50)
        print("VISUALIZATION DEPENDENCIES STATUS")
        print("=" * 50)
        for name, available in deps.items():
            status = "✓ Installed" if available else "✗ Missing"
            print(f"  {name:15} {status}")
        print("=" * 50)

    def generate_demo_figures(self) -> Dict[str, str]:
        """Generate all demo figures."""
        paths = {}

        # Generate sample data (SAM-40: 128 Hz)
        np.random.seed(42)
        fs = 128  # SAM-40 sampling rate
        t = np.arange(0, 2, 1/fs)
        eeg_data = 10*np.sin(2*np.pi*10*t) + 5*np.sin(2*np.pi*20*t) + np.random.randn(len(t))*2

        # EEG signal
        try:
            fig = self.trend.plot_eeg_signal(eeg_data.reshape(1, -1), fs=fs, save_path='demo_eeg.pdf')
            paths['eeg'] = 'demo_eeg.pdf'
            plt.close(fig)
        except Exception as e:
            print(f"EEG plot failed: {e}")

        # PSD
        try:
            fig = self.trend.plot_psd_bands(eeg_data, fs=fs, save_path='demo_psd.pdf')
            paths['psd'] = 'demo_psd.pdf'
            plt.close(fig)
        except Exception as e:
            print(f"PSD plot failed: {e}")

        # Power analysis
        try:
            fig = self.stats.plot_power_analysis(save_path='demo_power.pdf')
            paths['power'] = 'demo_power.pdf'
            plt.close(fig)
        except Exception as e:
            print(f"Power analysis failed: {e}")

        return paths


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RESEARCH VISUALIZATION SUITE")
    print("=" * 60)

    viz = ResearchVisualizationSuite(output_dir='./results/figures')
    viz.print_dependency_status()

    print("\nGenerating demo figures...")
    paths = viz.generate_demo_figures()
    print(f"\nGenerated {len(paths)} figures:")
    for name, path in paths.items():
        print(f"  - {name}: {path}")
