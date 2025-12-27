#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PhD-Grade LaTeX Research Tools
================================================================================

Complete toolkit for generating publication-ready LaTeX content:
- Tables (booktabs, siunitx, tabularx format)
- Figures (PDF vector export for LaTeX inclusion)
- PGFPlots/TikZ code generation
- Beamer presentations
- Format conversion utilities

Format Scorecard (Reviewer-Safe):
    PDF (vector)     : 10 - Best for LaTeX + journals
    SVG (vector)     : 9  - Perfect scaling, easy editing
    EPS (vector)     : 8  - Widely accepted, older workflow
    PNG (600 DPI)    : 7  - OK when vector not possible
    TIFF (600 DPI)   : 6  - Sometimes required for medical
    Native LaTeX     : 10 - Tables should be text, not images

Workflow: Python (analysis) → PDF (figures) → LaTeX (tables + paper)

================================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import warnings
import re

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for PDF
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

# Publication-quality matplotlib settings for PDF export
PDF_RCPARAMS = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,  # TrueType fonts (editable in Illustrator)
    'ps.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'text.usetex': False,  # Set True if LaTeX installed
}


def apply_latex_style():
    """Apply publication-ready style for LaTeX figures."""
    if MATPLOTLIB_AVAILABLE:
        plt.rcParams.update(PDF_RCPARAMS)


# =============================================================================
# 1. LATEX TABLE GENERATOR (booktabs + siunitx)
# =============================================================================

class LaTeXTableGenerator:
    """
    Generate publication-quality LaTeX tables.

    Supports:
    - booktabs formatting (toprule, midrule, bottomrule)
    - siunitx S columns for decimal alignment
    - tabularx for auto-width
    - longtable for multi-page tables
    - threeparttable for table notes

    Usage:
        from src.analysis import LaTeXTableGenerator

        gen = LaTeXTableGenerator()
        latex = gen.results_table(df, caption="Classification Results")
        gen.save_table(latex, "results.tex")
    """

    def __init__(self, output_dir: str = './tables'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _format_number(self, val: Any, precision: int = 3) -> str:
        """Format number for siunitx S columns."""
        if pd.isna(val):
            return '{--}'
        if isinstance(val, (int, np.integer)):
            return str(val)
        if isinstance(val, (float, np.floating)):
            if abs(val) < 0.001 and val != 0:
                return f'{val:.2e}'
            return f'{val:.{precision}f}'
        return str(val)

    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters."""
        if not isinstance(text, str):
            return str(text)
        replacements = [
            ('_', r'\_'),
            ('%', r'\%'),
            ('&', r'\&'),
            ('#', r'\#'),
            ('$', r'\$'),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text

    def results_table(
        self,
        df: 'pd.DataFrame',
        caption: str = "Results",
        label: str = "tab:results",
        precision: int = 3,
        highlight_best: bool = True,
        notes: List[str] = None
    ) -> str:
        """
        Generate results comparison table with booktabs + siunitx.

        Args:
            df: DataFrame with results
            caption: Table caption
            label: LaTeX label
            precision: Decimal precision
            highlight_best: Bold the best value in each column
            notes: Table footnotes

        Returns:
            Complete LaTeX table code
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for table generation")

        n_cols = len(df.columns)

        # Determine column types
        col_specs = []
        for col in df.columns:
            if df[col].dtype in [np.float64, np.float32]:
                col_specs.append(f'S[table-format=1.{precision}]')
            else:
                col_specs.append('l')

        # Find best values for highlighting
        best_idx = {}
        if highlight_best:
            for col in df.columns:
                if df[col].dtype in [np.float64, np.float32]:
                    best_idx[col] = df[col].idxmax()

        # Build table
        lines = []
        lines.append(r'\begin{table}[htbp]')
        lines.append(r'\centering')
        lines.append(f'\\caption{{{caption}}}')
        lines.append(f'\\label{{{label}}}')

        if notes:
            lines.append(r'\begin{threeparttable}')

        lines.append(r'\begin{tabular}{' + ' '.join(col_specs) + '}')
        lines.append(r'\toprule')

        # Header
        header = ' & '.join([f'{{\\textbf{{{self._escape_latex(col)}}}}}' for col in df.columns])
        lines.append(header + r' \\')
        lines.append(r'\midrule')

        # Data rows
        for idx, row in df.iterrows():
            cells = []
            for col in df.columns:
                val = self._format_number(row[col], precision)
                if highlight_best and col in best_idx and idx == best_idx[col]:
                    val = f'\\textbf{{{val}}}'
                cells.append(val)
            lines.append(' & '.join(cells) + r' \\')

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')

        if notes:
            lines.append(r'\begin{tablenotes}')
            lines.append(r'\small')
            for i, note in enumerate(notes):
                lines.append(f'\\item[{chr(97+i)}] {note}')
            lines.append(r'\end{tablenotes}')
            lines.append(r'\end{threeparttable}')

        lines.append(r'\end{table}')

        return '\n'.join(lines)

    def comparison_table(
        self,
        data: Dict[str, Dict[str, float]],
        caption: str = "Method Comparison",
        label: str = "tab:comparison",
        metrics: List[str] = None,
        precision: int = 3
    ) -> str:
        """
        Generate method comparison table.

        Args:
            data: {method_name: {metric: value}}
            caption: Table caption
            label: LaTeX label
            metrics: List of metrics to include
            precision: Decimal precision

        Returns:
            LaTeX table code
        """
        methods = list(data.keys())
        if metrics is None:
            metrics = list(data[methods[0]].keys())

        lines = []
        lines.append(r'\begin{table}[htbp]')
        lines.append(r'\centering')
        lines.append(f'\\caption{{{caption}}}')
        lines.append(f'\\label{{{label}}}')

        # Column spec: Method name + S columns for each metric
        col_spec = 'l' + ''.join([f' S[table-format=1.{precision}]' for _ in metrics])
        lines.append(r'\begin{tabular}{' + col_spec + '}')
        lines.append(r'\toprule')

        # Header
        header = r'{\textbf{Method}}' + ' & '.join([''] + [f'{{\\textbf{{{m}}}}}' for m in metrics])
        lines.append(header + r' \\')
        lines.append(r'\midrule')

        # Find best for each metric
        best_vals = {m: max(data[method].get(m, 0) for method in methods) for m in metrics}

        # Data rows
        for method in methods:
            cells = [self._escape_latex(method)]
            for metric in metrics:
                val = data[method].get(metric, 0)
                formatted = self._format_number(val, precision)
                if val == best_vals[metric]:
                    formatted = f'\\textbf{{{formatted}}}'
                cells.append(formatted)
            lines.append(' & '.join(cells) + r' \\')

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(r'\end{table}')

        return '\n'.join(lines)

    def statistical_table(
        self,
        tests: List[Dict[str, Any]],
        caption: str = "Statistical Analysis",
        label: str = "tab:stats"
    ) -> str:
        """
        Generate statistical analysis table with p-values and effect sizes.

        Args:
            tests: List of {name, statistic, p_value, effect_size, ci_lower, ci_upper}
            caption: Table caption
            label: LaTeX label

        Returns:
            LaTeX table code
        """
        lines = []
        lines.append(r'\begin{table}[htbp]')
        lines.append(r'\centering')
        lines.append(f'\\caption{{{caption}}}')
        lines.append(f'\\label{{{label}}}')
        lines.append(r'\begin{tabular}{l S[table-format=2.3] S[table-format=1.4] S[table-format=1.3] c}')
        lines.append(r'\toprule')
        lines.append(r'{\textbf{Comparison}} & {\textbf{Statistic}} & {\textbf{p-value}} & {\textbf{Effect Size}} & {\textbf{95\% CI}} \\')
        lines.append(r'\midrule')

        for test in tests:
            name = self._escape_latex(test['name'])
            stat = self._format_number(test.get('statistic', 0), 3)
            p = test.get('p_value', 1)

            # Format p-value with significance markers
            if p < 0.001:
                p_str = r'$<$0.001***'
            elif p < 0.01:
                p_str = f'{p:.3f}**'
            elif p < 0.05:
                p_str = f'{p:.3f}*'
            else:
                p_str = f'{p:.3f}'

            effect = self._format_number(test.get('effect_size', 0), 3)
            ci_low = test.get('ci_lower', 0)
            ci_high = test.get('ci_upper', 0)
            ci_str = f'[{ci_low:.2f}, {ci_high:.2f}]'

            lines.append(f'{name} & {stat} & {p_str} & {effect} & {ci_str} \\\\')

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(r'\begin{tablenotes}')
        lines.append(r'\small')
        lines.append(r'\item Note: * p $<$ 0.05, ** p $<$ 0.01, *** p $<$ 0.001')
        lines.append(r'\end{tablenotes}')
        lines.append(r'\end{table}')

        return '\n'.join(lines)

    def save_table(self, latex_code: str, filename: str) -> str:
        """Save LaTeX table to file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(latex_code)
        print(f"Table saved: {filepath}")
        return str(filepath)


# =============================================================================
# 2. PDF FIGURE EXPORTER (Vector Graphics)
# =============================================================================

class PDFFigureExporter:
    """
    Export matplotlib figures as publication-quality PDF.

    Features:
    - 600 DPI for print quality
    - Embedded fonts (Type 42)
    - Tight bounding box
    - Multiple format support (PDF, SVG, EPS, PNG)

    Usage:
        from src.analysis import PDFFigureExporter

        exporter = PDFFigureExporter('./figures')
        fig = plt.figure()
        # ... create plot ...
        exporter.save(fig, 'my_figure')  # Saves as PDF
        exporter.save_all_formats(fig, 'my_figure')  # PDF, SVG, EPS, PNG
    """

    FORMAT_SCORES = {
        'pdf': 10,   # Best for LaTeX + journals
        'svg': 9,    # Perfect scaling, editable
        'eps': 8,    # Widely accepted, legacy
        'png': 7,    # OK when vector not possible
        'tiff': 6,   # Medical imaging standard
    }

    def __init__(self, output_dir: str = './figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        apply_latex_style()

    def save(
        self,
        fig: plt.Figure,
        filename: str,
        format: str = 'pdf',
        dpi: int = 600
    ) -> str:
        """
        Save figure in specified format.

        Args:
            fig: matplotlib Figure
            filename: Output filename (without extension)
            format: Output format (pdf, svg, eps, png, tiff)
            dpi: Resolution for raster formats

        Returns:
            Path to saved file
        """
        filepath = self.output_dir / f'{filename}.{format}'

        save_kwargs = {
            'dpi': dpi,
            'bbox_inches': 'tight',
            'pad_inches': 0.05,
            'facecolor': 'white',
            'edgecolor': 'none',
        }

        if format == 'pdf':
            save_kwargs['backend'] = 'pdf'
        elif format == 'svg':
            save_kwargs['format'] = 'svg'

        fig.savefig(filepath, **save_kwargs)
        print(f"Figure saved ({format.upper()}, score={self.FORMAT_SCORES.get(format, 0)}): {filepath}")
        return str(filepath)

    def save_all_formats(
        self,
        fig: plt.Figure,
        filename: str,
        formats: List[str] = None
    ) -> Dict[str, str]:
        """Save figure in multiple formats."""
        if formats is None:
            formats = ['pdf', 'svg', 'eps', 'png']

        paths = {}
        for fmt in formats:
            try:
                paths[fmt] = self.save(fig, filename, format=fmt)
            except Exception as e:
                print(f"Warning: Failed to save {fmt}: {e}")

        return paths

    def create_trend_chart(
        self,
        x: np.ndarray,
        y_data: Dict[str, np.ndarray],
        xlabel: str = "X",
        ylabel: str = "Y",
        title: str = None,
        filename: str = "trend",
        show_ci: bool = True
    ) -> Tuple[plt.Figure, str]:
        """
        Create publication-ready trend chart.

        Args:
            x: X-axis values
            y_data: {label: y_values} or {label: (mean, std)}
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Optional title
            filename: Output filename
            show_ci: Show confidence intervals

        Returns:
            (Figure, filepath)
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(y_data)))

        for (label, data), color in zip(y_data.items(), colors):
            if isinstance(data, tuple):
                mean, std = data
                ax.plot(x, mean, label=label, color=color, linewidth=2)
                if show_ci:
                    ax.fill_between(x, mean - 1.96*std, mean + 1.96*std, alpha=0.2, color=color)
            else:
                ax.plot(x, data, label=label, color=color, linewidth=2)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.save(fig, filename)

        return fig, filepath

    def create_bar_chart(
        self,
        categories: List[str],
        values: Dict[str, List[float]],
        xlabel: str = "Category",
        ylabel: str = "Value",
        title: str = None,
        filename: str = "bar_chart",
        show_error: bool = True,
        errors: Dict[str, List[float]] = None
    ) -> Tuple[plt.Figure, str]:
        """
        Create publication-ready bar chart.

        Args:
            categories: X-axis category labels
            values: {group: [values]}
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Optional title
            filename: Output filename
            show_error: Show error bars
            errors: {group: [error_values]}

        Returns:
            (Figure, filepath)
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        n_groups = len(values)
        n_categories = len(categories)
        bar_width = 0.8 / n_groups
        x = np.arange(n_categories)

        colors = plt.cm.Set2(np.linspace(0, 1, n_groups))

        for i, (label, vals) in enumerate(values.items()):
            offset = (i - n_groups/2 + 0.5) * bar_width
            err = errors.get(label) if errors and show_error else None
            ax.bar(x + offset, vals, bar_width, label=label, color=colors[i],
                   yerr=err, capsize=3, edgecolor='black', linewidth=0.5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        if title:
            ax.set_title(title)
        ax.legend(loc='best')

        plt.tight_layout()
        filepath = self.save(fig, filename)

        return fig, filepath


# =============================================================================
# 3. PGFPLOTS CODE GENERATOR
# =============================================================================

class PGFPlotsGenerator:
    """
    Generate native LaTeX PGFPlots code.

    For reviewers who prefer LaTeX-native figures.
    Score: 10 (IEEE, Springer, Elsevier love it)

    Usage:
        from src.analysis import PGFPlotsGenerator

        gen = PGFPlotsGenerator()
        code = gen.line_plot(x, y, xlabel="Time", ylabel="Accuracy")
        gen.save(code, "accuracy_plot.tex")
    """

    def __init__(self, output_dir: str = './figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _data_to_table(self, x: np.ndarray, y: np.ndarray) -> str:
        """Convert data to PGFPlots table format."""
        lines = []
        for xi, yi in zip(x, y):
            lines.append(f'            {xi} {yi}')
        return '\n'.join(lines)

    def line_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        xlabel: str = "x",
        ylabel: str = "y",
        title: str = None,
        color: str = "blue",
        legend: str = None
    ) -> str:
        """Generate PGFPlots line plot code."""
        data = self._data_to_table(x, y)

        code = f'''\\begin{{tikzpicture}}
\\begin{{axis}}[
    xlabel={{{xlabel}}},
    ylabel={{{ylabel}}},
    {'title={' + title + '},' if title else ''}
    grid=major,
    legend pos=north west,
]
\\addplot[{color}, thick, mark=*] coordinates {{
{data}
}};
{'\\addlegendentry{' + legend + '}' if legend else ''}
\\end{{axis}}
\\end{{tikzpicture}}'''

        return code

    def bar_plot(
        self,
        categories: List[str],
        values: List[float],
        xlabel: str = "Category",
        ylabel: str = "Value",
        title: str = None,
        color: str = "blue!70"
    ) -> str:
        """Generate PGFPlots bar chart code."""
        coords = ', '.join([f'({cat},{val})' for cat, val in zip(categories, values)])

        code = f'''\\begin{{tikzpicture}}
\\begin{{axis}}[
    ybar,
    xlabel={{{xlabel}}},
    ylabel={{{ylabel}}},
    {'title={' + title + '},' if title else ''}
    symbolic x coords={{{','.join(categories)}}},
    xtick=data,
    nodes near coords,
    nodes near coords align={{vertical}},
    bar width=0.6cm,
]
\\addplot[fill={color}] coordinates {{{coords}}};
\\end{{axis}}
\\end{{tikzpicture}}'''

        return code

    def grouped_bar_plot(
        self,
        categories: List[str],
        groups: Dict[str, List[float]],
        xlabel: str = "Category",
        ylabel: str = "Value",
        title: str = None
    ) -> str:
        """Generate grouped bar chart code."""
        colors = ['blue!70', 'red!70', 'green!70', 'orange!70', 'purple!70']

        legend_entries = []
        addplots = []

        for i, (group_name, values) in enumerate(groups.items()):
            coords = ', '.join([f'({cat},{val})' for cat, val in zip(categories, values)])
            color = colors[i % len(colors)]
            addplots.append(f'\\addplot[fill={color}] coordinates {{{coords}}};')
            legend_entries.append(group_name)

        code = f'''\\begin{{tikzpicture}}
\\begin{{axis}}[
    ybar,
    xlabel={{{xlabel}}},
    ylabel={{{ylabel}}},
    {'title={' + title + '},' if title else ''}
    symbolic x coords={{{','.join(categories)}}},
    xtick=data,
    legend style={{at={{(0.5,-0.15)}}, anchor=north, legend columns=-1}},
    bar width=0.3cm,
    enlarge x limits=0.15,
]
{chr(10).join(addplots)}
\\legend{{{','.join(legend_entries)}}}
\\end{{axis}}
\\end{{tikzpicture}}'''

        return code

    def error_bar_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        yerr: np.ndarray,
        xlabel: str = "x",
        ylabel: str = "y",
        title: str = None
    ) -> str:
        """Generate line plot with error bars."""
        data_lines = []
        for xi, yi, ei in zip(x, y, yerr):
            data_lines.append(f'            ({xi}, {yi}) +- (0, {ei})')

        code = f'''\\begin{{tikzpicture}}
\\begin{{axis}}[
    xlabel={{{xlabel}}},
    ylabel={{{ylabel}}},
    {'title={' + title + '},' if title else ''}
    grid=major,
]
\\addplot[blue, thick, mark=*, error bars/.cd, y dir=both, y explicit]
    coordinates {{
{chr(10).join(data_lines)}
    }};
\\end{{axis}}
\\end{{tikzpicture}}'''

        return code

    def save(self, code: str, filename: str) -> str:
        """Save PGFPlots code to file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(code)
        print(f"PGFPlots saved: {filepath}")
        return str(filepath)


# =============================================================================
# 4. BEAMER PRESENTATION GENERATOR
# =============================================================================

class BeamerGenerator:
    """
    Generate Beamer presentations.

    Score: 10 (Academic standard for talks and defense)

    Usage:
        from src.analysis import BeamerGenerator

        beamer = BeamerGenerator()
        beamer.add_title_slide("My Research", "Author Name")
        beamer.add_content_slide("Results", ["Point 1", "Point 2"])
        beamer.add_figure_slide("Accuracy", "figures/accuracy.pdf")
        beamer.save("presentation.tex")
    """

    def __init__(self, output_dir: str = './presentations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.slides = []
        self.title = "Presentation"
        self.author = "Author"
        self.institute = "Institute"
        self.date = datetime.now().strftime("%B %d, %Y")

    def set_metadata(
        self,
        title: str,
        author: str,
        institute: str = None,
        date: str = None
    ):
        """Set presentation metadata."""
        self.title = title
        self.author = author
        if institute:
            self.institute = institute
        if date:
            self.date = date

    def add_title_slide(self):
        """Add title slide."""
        self.slides.append(r'\begin{frame}' + '\n' + r'\titlepage' + '\n' + r'\end{frame}')

    def add_content_slide(
        self,
        title: str,
        items: List[str],
        columns: int = 1
    ):
        """Add content slide with bullet points."""
        content = []
        content.append(f'\\begin{{frame}}{{{title}}}')

        if columns > 1:
            content.append(r'\begin{columns}')
            per_col = len(items) // columns
            for c in range(columns):
                content.append(r'\begin{column}{0.5\textwidth}')
                content.append(r'\begin{itemize}')
                start = c * per_col
                end = start + per_col if c < columns - 1 else len(items)
                for item in items[start:end]:
                    content.append(f'    \\item {item}')
                content.append(r'\end{itemize}')
                content.append(r'\end{column}')
            content.append(r'\end{columns}')
        else:
            content.append(r'\begin{itemize}')
            for item in items:
                content.append(f'    \\item {item}')
            content.append(r'\end{itemize}')

        content.append(r'\end{frame}')
        self.slides.append('\n'.join(content))

    def add_figure_slide(
        self,
        title: str,
        figure_path: str,
        caption: str = None,
        width: str = "0.8\\textwidth"
    ):
        """Add slide with figure."""
        content = []
        content.append(f'\\begin{{frame}}{{{title}}}')
        content.append(r'\centering')
        content.append(f'\\includegraphics[width={width}]{{{figure_path}}}')
        if caption:
            content.append(f'\\\\\\small {caption}')
        content.append(r'\end{frame}')
        self.slides.append('\n'.join(content))

    def add_table_slide(
        self,
        title: str,
        table_content: str
    ):
        """Add slide with table (paste LaTeX table code)."""
        content = []
        content.append(f'\\begin{{frame}}{{{title}}}')
        content.append(r'\centering')
        content.append(r'\small')
        content.append(table_content)
        content.append(r'\end{frame}')
        self.slides.append('\n'.join(content))

    def add_two_column_slide(
        self,
        title: str,
        left_content: str,
        right_content: str
    ):
        """Add two-column slide."""
        content = []
        content.append(f'\\begin{{frame}}{{{title}}}')
        content.append(r'\begin{columns}')
        content.append(r'\begin{column}{0.5\textwidth}')
        content.append(left_content)
        content.append(r'\end{column}')
        content.append(r'\begin{column}{0.5\textwidth}')
        content.append(right_content)
        content.append(r'\end{column}')
        content.append(r'\end{columns}')
        content.append(r'\end{frame}')
        self.slides.append('\n'.join(content))

    def generate(self) -> str:
        """Generate complete Beamer document."""
        document = f'''\\documentclass[aspectratio=169]{{beamer}}

% Theme
\\usetheme{{Madrid}}
\\usecolortheme{{default}}

% Packages
\\usepackage{{booktabs}}
\\usepackage{{siunitx}}
\\usepackage{{graphicx}}
\\usepackage{{tikz}}
\\usepackage{{pgfplots}}
\\pgfplotsset{{compat=1.18}}

% Metadata
\\title{{{self.title}}}
\\author{{{self.author}}}
\\institute{{{self.institute}}}
\\date{{{self.date}}}

\\begin{{document}}

{chr(10).join(self.slides)}

\\end{{document}}
'''
        return document

    def save(self, filename: str = "presentation.tex") -> str:
        """Save Beamer presentation to file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(self.generate())
        print(f"Beamer presentation saved: {filepath}")
        return str(filepath)


# =============================================================================
# 5. MASTER LATEX RESEARCH SUITE
# =============================================================================

class LaTeXResearchSuite:
    """
    Master class for all LaTeX research tools.

    Usage:
        from src.analysis import LaTeXResearchSuite

        latex = LaTeXResearchSuite(output_dir='./output')

        # Tables
        latex.tables.results_table(df, caption="Results")

        # Figures (PDF export)
        latex.figures.save(fig, 'my_plot')

        # PGFPlots
        latex.pgfplots.line_plot(x, y)

        # Beamer
        latex.beamer.add_title_slide("Talk Title", "Author")
    """

    def __init__(self, output_dir: str = './latex_output'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sub-modules
        self.tables = LaTeXTableGenerator(self.output_dir / 'tables')
        self.figures = PDFFigureExporter(self.output_dir / 'figures')
        self.pgfplots = PGFPlotsGenerator(self.output_dir / 'pgfplots')
        self.beamer = BeamerGenerator(self.output_dir / 'presentations')

    def generate_preamble(self) -> str:
        """Generate recommended LaTeX preamble."""
        return r'''\usepackage{booktabs}      % Professional tables
\usepackage{tabularx}      % Auto-width columns
\usepackage{longtable}     % Multi-page tables
\usepackage{siunitx}       % Number alignment
\usepackage{threeparttable} % Table notes
\usepackage{multirow}      % Multi-row cells
\usepackage{graphicx}      % Figures
\usepackage{tikz}          % Diagrams
\usepackage{pgfplots}      % Plots
\pgfplotsset{compat=1.18}
\usepackage{pgfplotstable} % Data tables
'''

    def print_format_guide(self):
        """Print format recommendation guide."""
        print("=" * 60)
        print("FORMAT GUIDE (PhD-Grade LaTeX)")
        print("=" * 60)
        print("\nFIGURES:")
        print("  PDF (vector)     : 10 - Best for LaTeX + journals")
        print("  SVG (vector)     : 9  - Perfect scaling, editable")
        print("  EPS (vector)     : 8  - Widely accepted")
        print("  PNG (600 DPI)    : 7  - OK when vector not possible")
        print("\nTABLES:")
        print("  Native LaTeX     : 10 - Always use booktabs + siunitx")
        print("  Never use images : 0  - Accessibility issues")
        print("\nWORKFLOW:")
        print("  Python → PDF → LaTeX (tables + paper)")
        print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LATEX RESEARCH TOOLS")
    print("=" * 60)

    suite = LaTeXResearchSuite('./latex_output')
    suite.print_format_guide()

    # Demo: Generate sample table
    if PANDAS_AVAILABLE:
        df = pd.DataFrame({
            'Method': ['CNN', 'LSTM', 'CNN-LSTM', 'Ours'],
            'Accuracy': [0.892, 0.901, 0.923, 0.947],
            'F1 Score': [0.885, 0.894, 0.918, 0.943],
            'AUC': [0.912, 0.921, 0.945, 0.967]
        })
        df = df.set_index('Method')
        table_code = suite.tables.results_table(df, caption="Classification Performance")
        suite.tables.save_table(table_code, 'demo_results.tex')
        print("\nSample table generated: latex_output/tables/demo_results.tex")
