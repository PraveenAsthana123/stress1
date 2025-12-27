#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Research-Grade Format Configuration & Export Suite
================================================================================

Complete configuration for PhD thesis, journal papers, and medical research
visualizations with reviewer-safe formats and colorblind-safe palettes.

FORMAT SCORECARD:
    SVG (vector)     : 10 - Scales infinitely, editable
    PDF (vector)     : 10 - Journal standard, LaTeX-ready
    EPS (vector)     : 9  - Legacy but accepted
    PNG (600 DPI)    : 8  - For raster images only
    TIFF (600 DPI)   : 7  - Medical imaging standard
    JPG/JPEG         : 2  - AVOID (compression artifacts)

RESOLUTION STANDARDS:
    Line art         : 600 DPI
    Mixed content    : 600 DPI
    Photos/scans     : 300 DPI
    EEG topomaps     : ≥600 DPI

COLOR MODES:
    Online/PDF       : RGB
    Print journals   : CMYK (if requested)
    Colorblind-safe  : Viridis, ColorBrewer, Tableau

PUBLISHER REQUIREMENTS:
    IEEE             : PDF/EPS figures
    Nature           : PDF/SVG/PNG
    Elsevier         : TIFF/EPS/PDF
    Springer         : EPS/PDF

================================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import warnings

# =============================================================================
# IMPORTS
# =============================================================================

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

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

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

try:
    from prettytable import PrettyTable
    PRETTYTABLE_AVAILABLE = True
except ImportError:
    PRETTYTABLE_AVAILABLE = False


# =============================================================================
# FORMAT DEFINITIONS
# =============================================================================

class FigureFormat(Enum):
    """Figure format with quality scores."""
    SVG = ("svg", 10, "vector", "Scales infinitely, editable")
    PDF = ("pdf", 10, "vector", "Journal standard, LaTeX-ready")
    EPS = ("eps", 9, "vector", "Legacy but accepted")
    PNG = ("png", 8, "raster", "For images only (600 DPI)")
    TIFF = ("tiff", 7, "raster", "Medical imaging standard")
    JPG = ("jpg", 2, "raster", "AVOID - compression artifacts")

    def __init__(self, ext, score, type_, description):
        self.ext = ext
        self.score = score
        self.type_ = type_
        self.description = description


class Resolution(Enum):
    """DPI standards for different content types."""
    LINE_ART = 600
    MIXED = 600
    PHOTO = 300
    EEG_TOPOMAP = 600
    PRINT_HIGH = 1200
    WEB = 150


class ColorMode(Enum):
    """Color mode for different outputs."""
    RGB = "RGB"      # Online/PDF
    CMYK = "CMYK"    # Print journals
    GRAYSCALE = "L"  # B&W printing


# =============================================================================
# COLORBLIND-SAFE PALETTES
# =============================================================================

COLORBLIND_PALETTES = {
    # Viridis family (perceptually uniform)
    'viridis': ['#440154', '#482878', '#3E4A89', '#31688E', '#26838E',
                '#1F9E89', '#35B779', '#6ECE58', '#B5DE2B', '#FDE725'],

    # ColorBrewer qualitative (colorblind-safe)
    'colorbrewer_qual': ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3',
                         '#FF7F00', '#FFFF33', '#A65628', '#F781BF'],

    # Tableau 10 (widely used)
    'tableau10': ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
                  '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC'],

    # IBM Design (accessible)
    'ibm': ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000'],

    # Paul Tol's (optimized for colorblindness)
    'tol_bright': ['#4477AA', '#EE6677', '#228833', '#CCBB44',
                   '#66CCEE', '#AA3377', '#BBBBBB'],

    # Safe for deuteranopia (red-green colorblind)
    'deuteranopia_safe': ['#0072B2', '#E69F00', '#CC79A7', '#009E73',
                          '#F0E442', '#56B4E9', '#D55E00'],

    # Research standard (high contrast)
    'research': ['#2C3E50', '#E74C3C', '#3498DB', '#27AE60',
                 '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22'],
}

# Default palette for research
DEFAULT_PALETTE = COLORBLIND_PALETTES['tableau10']


# =============================================================================
# PUBLISHER CONFIGURATIONS
# =============================================================================

PUBLISHER_CONFIGS = {
    'IEEE': {
        'figure_formats': ['pdf', 'eps'],
        'table_format': 'latex',
        'dpi': 600,
        'color_mode': 'RGB',
        'font_family': 'Times New Roman',
        'column_width_inches': 3.5,
        'page_width_inches': 7.16,
    },
    'Nature': {
        'figure_formats': ['pdf', 'svg', 'png'],
        'table_format': 'word',
        'dpi': 300,
        'color_mode': 'RGB',
        'font_family': 'Arial',
        'column_width_inches': 3.5,
        'page_width_inches': 7.2,
    },
    'Elsevier': {
        'figure_formats': ['tiff', 'eps', 'pdf'],
        'table_format': 'word',
        'dpi': 600,
        'color_mode': 'RGB',
        'font_family': 'Times New Roman',
        'column_width_inches': 3.3,
        'page_width_inches': 6.85,
    },
    'Springer': {
        'figure_formats': ['eps', 'pdf'],
        'table_format': 'latex',
        'dpi': 600,
        'color_mode': 'RGB',
        'font_family': 'Times New Roman',
        'column_width_inches': 3.3,
        'page_width_inches': 6.69,
    },
    'PLOS': {
        'figure_formats': ['tiff', 'eps'],
        'table_format': 'word',
        'dpi': 300,
        'color_mode': 'RGB',
        'font_family': 'Arial',
    },
    'MDPI': {
        'figure_formats': ['pdf', 'png', 'eps'],
        'table_format': 'latex',
        'dpi': 600,
        'color_mode': 'RGB',
        'font_family': 'Palatino',
    },
}


# =============================================================================
# FIGURE TYPE RECOMMENDATIONS
# =============================================================================

FIGURE_TYPE_FORMAT = {
    'line_plot': ['svg', 'pdf'],
    'trend_chart': ['svg', 'pdf'],
    'bar_chart': ['svg', 'pdf'],
    'box_plot': ['svg', 'pdf'],
    'violin_plot': ['svg', 'pdf'],
    'scatter_plot': ['svg', 'pdf'],
    'heatmap': ['svg', 'png'],
    'eeg_timeseries': ['pdf', 'svg'],
    'eeg_topomap': ['png', 'pdf'],
    'psd_plot': ['svg', 'pdf'],
    'spectrogram': ['png', 'pdf'],
    'brain_map': ['png', 'pdf'],
    'microscopy': ['tiff', 'png'],
    'mri_scan': ['tiff', 'png'],
    'flowchart': ['svg', 'pdf'],
    'architecture': ['svg', 'pdf'],
    'network_graph': ['svg', 'pdf'],
    'roc_curve': ['svg', 'pdf'],
    'confusion_matrix': ['svg', 'pdf'],
    'forest_plot': ['svg', 'pdf'],
    'kaplan_meier': ['svg', 'pdf'],
}


# =============================================================================
# MATPLOTLIB CONFIGURATION
# =============================================================================

def get_publication_rcparams(publisher: str = 'IEEE') -> Dict:
    """Get matplotlib rcParams for specific publisher."""
    config = PUBLISHER_CONFIGS.get(publisher, PUBLISHER_CONFIGS['IEEE'])

    return {
        'font.family': 'serif',
        'font.serif': [config['font_family'], 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': config['dpi'],
        'savefig.format': config['figure_formats'][0],
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.5,
        'figure.figsize': (config.get('column_width_inches', 3.5), 2.5),
    }


def apply_publisher_style(publisher: str = 'IEEE'):
    """Apply publication style for specific publisher."""
    if MATPLOTLIB_AVAILABLE:
        plt.rcParams.update(get_publication_rcparams(publisher))
        if SEABORN_AVAILABLE:
            sns.set_palette(DEFAULT_PALETTE)


# =============================================================================
# FILE NAMING CONVENTION
# =============================================================================

class FileNamer:
    """Generate standardized file names for research figures."""

    @staticmethod
    def figure_name(
        fig_number: int,
        description: str,
        format: str = 'pdf'
    ) -> str:
        """
        Generate standard figure filename.

        Format: Fig{N}_{Description}.{ext}
        Example: Fig1_EEG_TimeSeries.pdf
        """
        # Clean description
        clean_desc = description.replace(' ', '_').replace('-', '_')
        clean_desc = ''.join(c for c in clean_desc if c.isalnum() or c == '_')
        return f"Fig{fig_number}_{clean_desc}.{format}"

    @staticmethod
    def table_name(
        table_number: int,
        description: str,
        format: str = 'tex'
    ) -> str:
        """
        Generate standard table filename.

        Format: Table{N}_{Description}.{ext}
        Example: Table2_Statistical_Results.tex
        """
        clean_desc = description.replace(' ', '_').replace('-', '_')
        clean_desc = ''.join(c for c in clean_desc if c.isalnum() or c == '_')
        return f"Table{table_number}_{clean_desc}.{format}"

    @staticmethod
    def supplement_name(
        supp_id: str,
        description: str,
        format: str = 'pdf'
    ) -> str:
        """
        Generate supplementary material filename.

        Format: Supp_{ID}_{Description}.{ext}
        Example: Supp_S1_Additional_Analysis.pdf
        """
        clean_desc = description.replace(' ', '_')
        return f"Supp_{supp_id}_{clean_desc}.{format}"


# =============================================================================
# MULTI-FORMAT EXPORTER
# =============================================================================

class ResearchFigureExporter:
    """
    Export figures in multiple research-grade formats.

    Supports all major publishers and format requirements.
    """

    def __init__(
        self,
        output_dir: str = './figures',
        publisher: str = 'IEEE'
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.publisher = publisher
        self.config = PUBLISHER_CONFIGS.get(publisher, PUBLISHER_CONFIGS['IEEE'])
        self.namer = FileNamer()

        if MATPLOTLIB_AVAILABLE:
            apply_publisher_style(publisher)

    def save(
        self,
        fig: 'plt.Figure',
        fig_number: int,
        description: str,
        formats: List[str] = None,
        dpi: int = None
    ) -> Dict[str, str]:
        """
        Save figure in multiple formats.

        Args:
            fig: matplotlib Figure
            fig_number: Figure number (e.g., 1 for Fig1)
            description: Brief description (e.g., "EEG_TimeSeries")
            formats: List of formats, defaults to publisher's preferred
            dpi: Resolution, defaults to publisher's standard

        Returns:
            Dict of format -> filepath
        """
        if formats is None:
            formats = self.config['figure_formats']
        if dpi is None:
            dpi = self.config['dpi']

        paths = {}
        for fmt in formats:
            filename = self.namer.figure_name(fig_number, description, fmt)
            filepath = self.output_dir / filename

            save_kwargs = {
                'dpi': dpi,
                'bbox_inches': 'tight',
                'pad_inches': 0.05,
                'facecolor': 'white',
                'edgecolor': 'none',
            }

            if fmt == 'pdf':
                save_kwargs['backend'] = 'pdf'
            elif fmt == 'tiff':
                save_kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}

            try:
                fig.savefig(filepath, format=fmt, **save_kwargs)
                paths[fmt] = str(filepath)
                print(f"  ✓ Saved: {filepath}")
            except Exception as e:
                print(f"  ✗ Failed {fmt}: {e}")

        return paths

    def save_for_publisher(
        self,
        fig: 'plt.Figure',
        fig_number: int,
        description: str
    ) -> Dict[str, str]:
        """Save in all formats required by the configured publisher."""
        print(f"Saving for {self.publisher}...")
        return self.save(fig, fig_number, description)

    def recommend_format(self, figure_type: str) -> List[str]:
        """Get recommended formats for a figure type."""
        return FIGURE_TYPE_FORMAT.get(figure_type, ['pdf', 'svg'])


# =============================================================================
# TABLE FORMATTERS
# =============================================================================

class ResearchTableFormatter:
    """Format tables for research papers."""

    @staticmethod
    def to_latex(
        df: 'pd.DataFrame',
        caption: str = None,
        label: str = None,
        precision: int = 3,
        highlight_best: bool = True
    ) -> str:
        """Convert DataFrame to publication-quality LaTeX."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required")

        # Start table
        lines = []
        lines.append(r'\begin{table}[htbp]')
        lines.append(r'\centering')
        if caption:
            lines.append(f'\\caption{{{caption}}}')
        if label:
            lines.append(f'\\label{{{label}}}')

        # Column format with siunitx
        n_cols = len(df.columns)
        col_fmt = 'l' + ' '.join(['S[table-format=1.' + str(precision) + ']'] * (n_cols - 1 if df.index.name else n_cols))
        lines.append(r'\begin{tabular}{' + col_fmt + '}')
        lines.append(r'\toprule')

        # Header
        if df.index.name:
            headers = [f'{{\\textbf{{{df.index.name}}}}}'] + [f'{{\\textbf{{{c}}}}}' for c in df.columns]
        else:
            headers = [f'{{\\textbf{{{c}}}}}' for c in df.columns]
        lines.append(' & '.join(headers) + r' \\')
        lines.append(r'\midrule')

        # Find best values
        best_idx = {}
        if highlight_best:
            for col in df.select_dtypes(include=[np.number]).columns:
                best_idx[col] = df[col].idxmax()

        # Data rows
        for idx, row in df.iterrows():
            cells = [str(idx)] if df.index.name or True else []
            for col in df.columns:
                val = row[col]
                if isinstance(val, float):
                    formatted = f'{val:.{precision}f}'
                else:
                    formatted = str(val)

                if highlight_best and col in best_idx and idx == best_idx[col]:
                    formatted = f'\\textbf{{{formatted}}}'
                cells.append(formatted)
            lines.append(' & '.join(cells) + r' \\')

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(r'\end{table}')

        return '\n'.join(lines)

    @staticmethod
    def to_terminal(df: 'pd.DataFrame', format: str = 'grid') -> str:
        """Format table for terminal display."""
        if TABULATE_AVAILABLE:
            return tabulate(df, headers='keys', tablefmt=format, floatfmt='.3f')
        elif PRETTYTABLE_AVAILABLE:
            pt = PrettyTable()
            pt.field_names = [''] + list(df.columns)
            for idx, row in df.iterrows():
                pt.add_row([idx] + list(row))
            return str(pt)
        else:
            return df.to_string()

    @staticmethod
    def to_csv(df: 'pd.DataFrame', filepath: str) -> str:
        """Export to CSV for supplementary data."""
        df.to_csv(filepath)
        return filepath


# =============================================================================
# COLORBLIND CHECK
# =============================================================================

def check_colorblind_safety(colors: List[str]) -> Dict[str, Any]:
    """
    Check if color palette is colorblind-safe.

    Returns recommendations if issues found.
    """
    # Convert to RGB
    rgb_colors = []
    for c in colors:
        try:
            rgb = mcolors.to_rgb(c)
            rgb_colors.append(rgb)
        except:
            continue

    # Simple heuristic: check for similar red-green values
    issues = []
    for i, c1 in enumerate(rgb_colors):
        for j, c2 in enumerate(rgb_colors[i+1:], i+1):
            # Deuteranopia simulation (simplified)
            rg_diff1 = abs(c1[0] - c1[1])
            rg_diff2 = abs(c2[0] - c2[1])
            if rg_diff1 < 0.2 and rg_diff2 < 0.2:
                if abs(c1[2] - c2[2]) < 0.3:
                    issues.append(f"Colors {i} and {j} may be confused by colorblind viewers")

    return {
        'is_safe': len(issues) == 0,
        'issues': issues,
        'recommendation': 'viridis' if issues else None
    }


def get_colorblind_palette(n_colors: int = 8, palette_name: str = 'tableau10') -> List[str]:
    """Get a colorblind-safe palette with n colors."""
    palette = COLORBLIND_PALETTES.get(palette_name, DEFAULT_PALETTE)
    return palette[:n_colors]


# =============================================================================
# MASTER CONFIGURATION CLASS
# =============================================================================

class ResearchFormatConfig:
    """
    Master configuration for research-grade visualizations.

    Usage:
        from src.analysis import ResearchFormatConfig

        config = ResearchFormatConfig(publisher='IEEE')
        config.print_format_guide()
        config.apply_style()

        # Export figures
        config.exporter.save(fig, 1, "EEG_Analysis")

        # Format tables
        latex = config.tables.to_latex(df, caption="Results")
    """

    def __init__(
        self,
        publisher: str = 'IEEE',
        output_dir: str = './output'
    ):
        self.publisher = publisher
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sub-modules
        self.exporter = ResearchFigureExporter(
            self.output_dir / 'figures',
            publisher
        )
        self.tables = ResearchTableFormatter()
        self.namer = FileNamer()

        # Apply style
        self.apply_style()

    def apply_style(self):
        """Apply publication style."""
        apply_publisher_style(self.publisher)

    def get_palette(self, n_colors: int = 8) -> List[str]:
        """Get colorblind-safe palette."""
        return get_colorblind_palette(n_colors)

    def check_dependencies(self) -> Dict[str, bool]:
        """Check available visualization libraries."""
        return {
            'matplotlib': MATPLOTLIB_AVAILABLE,
            'seaborn': SEABORN_AVAILABLE,
            'pandas': PANDAS_AVAILABLE,
            'plotly': PLOTLY_AVAILABLE,
            'networkx': NETWORKX_AVAILABLE,
            'tabulate': TABULATE_AVAILABLE,
            'prettytable': PRETTYTABLE_AVAILABLE,
        }

    def print_format_guide(self):
        """Print comprehensive format guide."""
        print("=" * 70)
        print("RESEARCH FORMAT GUIDE")
        print(f"Publisher: {self.publisher}")
        print("=" * 70)

        print("\n📊 FIGURE FORMATS (by score):")
        for fmt in FigureFormat:
            print(f"  {fmt.ext.upper():5} : {fmt.score:2}/10 - {fmt.description}")

        print("\n📏 RESOLUTION STANDARDS:")
        for res in Resolution:
            print(f"  {res.name:15} : {res.value} DPI")

        print("\n🎨 COLORBLIND-SAFE PALETTES:")
        for name in COLORBLIND_PALETTES:
            print(f"  - {name}")

        print(f"\n📋 {self.publisher} REQUIREMENTS:")
        config = PUBLISHER_CONFIGS[self.publisher]
        print(f"  Figures : {', '.join(config['figure_formats'])}")
        print(f"  Tables  : {config['table_format']}")
        print(f"  DPI     : {config['dpi']}")
        print(f"  Font    : {config['font_family']}")

        print("\n✅ DEPENDENCIES:")
        deps = self.check_dependencies()
        for name, available in deps.items():
            status = "✓" if available else "✗"
            print(f"  {status} {name}")

        print("=" * 70)

    def print_figure_recommendations(self):
        """Print format recommendations by figure type."""
        print("\n📈 RECOMMENDED FORMATS BY FIGURE TYPE:")
        print("-" * 40)
        for fig_type, formats in FIGURE_TYPE_FORMAT.items():
            print(f"  {fig_type:20} : {', '.join(formats)}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def setup_research_environment(publisher: str = 'IEEE') -> ResearchFormatConfig:
    """
    Quick setup for research visualization environment.

    Usage:
        from src.analysis import setup_research_environment

        config = setup_research_environment('Nature')
        # Now matplotlib is configured for Nature publications
    """
    config = ResearchFormatConfig(publisher=publisher)
    config.print_format_guide()
    return config


def get_publisher_config(publisher: str) -> Dict:
    """Get configuration for a specific publisher."""
    return PUBLISHER_CONFIGS.get(publisher, PUBLISHER_CONFIGS['IEEE'])


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RESEARCH FORMAT CONFIGURATION")
    print("=" * 70)

    config = ResearchFormatConfig(publisher='IEEE')
    config.print_format_guide()
    config.print_figure_recommendations()

    print("\n🎨 Sample Colorblind-Safe Palette:")
    palette = config.get_palette(8)
    for i, color in enumerate(palette):
        print(f"  {i+1}. {color}")
