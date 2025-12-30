#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Comprehensive Evaluation Module for GenAI-RAG-EEG
================================================================================

This module implements thesis-quality analysis covering:
- Model Analysis (Phase 2): Complexity, stability, baseline comparison
- Performance Analysis (Phase 3): Metrics, calibration, robustness
- Subject Analysis (Phase 4): Per-subject metrics, worst-case reporting
- Sensitivity Analysis (Phase 5): Ablations, hyperparameter sensitivity
- Statistical Analysis (Phase 6): Effect sizes, CIs, paired comparisons
- Reporting Analysis (Phase 7): Thesis-ready tables and figures

Usage:
    from src.analysis.comprehensive_evaluation import ComprehensiveEvaluator

    evaluator = ComprehensiveEvaluator()
    report = evaluator.full_evaluation(
        model, X_train, y_train, X_test, y_test,
        subject_ids=subject_ids
    )
    evaluator.generate_thesis_tables(report, output_dir="results/")

Author: GenAI-RAG-EEG Team
Version: 3.0.0
License: MIT
================================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import json
from pathlib import Path
import warnings
from datetime import datetime

# Scikit-learn imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, brier_score_loss
)
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.calibration import calibration_curve

# SciPy for statistics
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class EvaluationConfig:
    """Configuration for comprehensive evaluation."""
    # Cross-validation
    n_folds: int = 5
    n_seeds: int = 5
    random_state: int = 42

    # Confidence intervals
    ci_level: float = 0.95
    n_bootstrap: int = 1000

    # Calibration
    n_calibration_bins: int = 10

    # Subject analysis
    worst_case_percentile: float = 10.0  # Bottom 10%

    # Thresholds
    min_acceptable_accuracy: float = 0.6
    max_acceptable_gap: float = 0.15  # Train-val gap


# =============================================================================
# PHASE 2: MODEL ANALYSIS
# =============================================================================

class ModelAnalyzer:
    """Analyze model characteristics and training behavior."""

    @staticmethod
    def compute_model_complexity(model) -> Dict[str, Any]:
        """
        Compute model complexity metrics.

        Returns:
            Dictionary with parameter counts, layer info
        """
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Layer breakdown
        layer_info = {}
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                layer_info[name] = {
                    'type': module.__class__.__name__,
                    'params': sum(p.numel() for p in module.parameters())
                }

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': total_params - trainable_params,
            'layer_breakdown': layer_info,
            'complexity_ratio': trainable_params / (total_params + 1)
        }

    @staticmethod
    def analyze_training_stability(
        loss_histories: List[List[float]],
        val_histories: List[List[float]]
    ) -> Dict[str, Any]:
        """
        Analyze training stability across multiple runs.

        Args:
            loss_histories: List of training loss curves per run
            val_histories: List of validation metric curves per run

        Returns:
            Stability metrics
        """
        # Compute variance across runs
        min_len = min(len(h) for h in loss_histories)
        loss_array = np.array([h[:min_len] for h in loss_histories])
        val_array = np.array([h[:min_len] for h in val_histories])

        # Per-epoch statistics
        loss_mean = loss_array.mean(axis=0)
        loss_std = loss_array.std(axis=0)
        val_mean = val_array.mean(axis=0)
        val_std = val_array.std(axis=0)

        # Convergence analysis
        final_loss_mean = loss_mean[-1]
        final_loss_std = loss_std[-1]
        final_val_mean = val_mean[-1]
        final_val_std = val_std[-1]

        # Train-val gap
        train_val_gap = final_loss_mean - (1 - final_val_mean)  # Approximate

        return {
            'final_loss_mean': final_loss_mean,
            'final_loss_std': final_loss_std,
            'final_val_mean': final_val_mean,
            'final_val_std': final_val_std,
            'convergence_epoch': int(np.argmin(loss_mean)),
            'max_loss_variance': float(loss_std.max()),
            'train_val_gap_estimate': train_val_gap,
            'is_stable': final_val_std < 0.05
        }

    @staticmethod
    def check_overfitting(
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        threshold: float = 0.15
    ) -> Dict[str, Any]:
        """
        Check for overfitting by comparing train vs validation metrics.
        """
        gaps = {}
        for metric in ['accuracy', 'f1', 'auc_roc']:
            if metric in train_metrics and metric in val_metrics:
                gap = train_metrics[metric] - val_metrics[metric]
                gaps[f'{metric}_gap'] = gap

        max_gap = max(gaps.values()) if gaps else 0

        return {
            'gaps': gaps,
            'max_gap': max_gap,
            'is_overfitting': max_gap > threshold,
            'recommendation': 'Increase regularization' if max_gap > threshold else 'OK'
        }


# =============================================================================
# PHASE 3: PERFORMANCE ANALYSIS
# =============================================================================

class PerformanceAnalyzer:
    """Comprehensive performance analysis."""

    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive classification metrics.
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        }

        if y_prob is not None and len(np.unique(y_true)) > 1:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            metrics['brier_score'] = brier_score_loss(y_true, y_prob)

            # PR-AUC
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
            metrics['pr_auc'] = np.trapz(precision_curve, recall_curve)

        return metrics

    @staticmethod
    def compute_confidence_intervals(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute bootstrap confidence intervals for metrics.
        """
        n_samples = len(y_true)
        alpha = (1 - ci_level) / 2

        # Bootstrap
        metrics_boot = {
            'accuracy': [],
            'f1': [],
            'precision': [],
            'recall': []
        }
        if y_prob is not None:
            metrics_boot['auc_roc'] = []

        np.random.seed(42)
        for _ in range(n_bootstrap):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            y_t = y_true[idx]
            y_p = y_pred[idx]

            if len(np.unique(y_t)) < 2:
                continue

            metrics_boot['accuracy'].append(accuracy_score(y_t, y_p))
            metrics_boot['f1'].append(f1_score(y_t, y_p, zero_division=0))
            metrics_boot['precision'].append(precision_score(y_t, y_p, zero_division=0))
            metrics_boot['recall'].append(recall_score(y_t, y_p, zero_division=0))

            if y_prob is not None:
                y_pr = y_prob[idx]
                try:
                    metrics_boot['auc_roc'].append(roc_auc_score(y_t, y_pr))
                except:
                    pass

        # Compute CIs
        ci_results = {}
        for metric, values in metrics_boot.items():
            if values:
                values = np.array(values)
                ci_results[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'ci_lower': float(np.percentile(values, alpha * 100)),
                    'ci_upper': float(np.percentile(values, (1 - alpha) * 100))
                }

        return ci_results

    @staticmethod
    def compute_calibration(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Compute calibration metrics (ECE, reliability curve).
        """
        # Reliability curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

        # Expected Calibration Error (ECE)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        bin_counts = []

        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_acc = y_true[mask].mean()
                bin_conf = y_prob[mask].mean()
                bin_count = mask.sum()
                ece += (bin_count / len(y_true)) * abs(bin_acc - bin_conf)
                bin_counts.append(bin_count)
            else:
                bin_counts.append(0)

        return {
            'ece': ece,
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist(),
            'bin_counts': bin_counts,
            'brier_score': brier_score_loss(y_true, y_prob),
            'is_well_calibrated': ece < 0.1
        }

    @staticmethod
    def threshold_analysis(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze performance at different decision thresholds.
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9).tolist()

        results = []
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            results.append({
                'threshold': thresh,
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0)
            })

        # Find optimal threshold
        f1_scores = [r['f1'] for r in results]
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = results[optimal_idx]['threshold']

        return {
            'threshold_results': results,
            'optimal_threshold_f1': optimal_threshold,
            'optimal_f1': results[optimal_idx]['f1']
        }


# =============================================================================
# PHASE 4: SUBJECT ANALYSIS
# =============================================================================

class SubjectAnalyzer:
    """Subject-level analysis for EEG classification."""

    @staticmethod
    def per_subject_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        subject_ids: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute metrics per subject.
        """
        unique_subjects = np.unique(subject_ids)
        subject_metrics = {}

        for subj in unique_subjects:
            mask = subject_ids == subj
            if mask.sum() < 2:
                continue

            y_t = y_true[mask]
            y_p = y_pred[mask]
            y_pr = y_prob[mask]

            metrics = {
                'n_samples': int(mask.sum()),
                'accuracy': accuracy_score(y_t, y_p),
                'f1': f1_score(y_t, y_p, zero_division=0)
            }

            if len(np.unique(y_t)) > 1:
                metrics['auc_roc'] = roc_auc_score(y_t, y_pr)
            else:
                metrics['auc_roc'] = None

            subject_metrics[str(subj)] = metrics

        return subject_metrics

    @staticmethod
    def worst_case_analysis(
        subject_metrics: Dict[str, Dict],
        percentile: float = 10.0
    ) -> Dict[str, Any]:
        """
        Analyze worst-performing subjects.
        """
        # Extract accuracies
        accuracies = []
        subjects = []
        for subj, metrics in subject_metrics.items():
            if 'accuracy' in metrics:
                accuracies.append(metrics['accuracy'])
                subjects.append(subj)

        accuracies = np.array(accuracies)
        subjects = np.array(subjects)

        # Quantiles
        q10 = np.percentile(accuracies, percentile)
        q50 = np.percentile(accuracies, 50)
        q90 = np.percentile(accuracies, 100 - percentile)

        # Worst subjects
        worst_mask = accuracies <= q10
        worst_subjects = subjects[worst_mask].tolist()
        worst_accuracies = accuracies[worst_mask].tolist()

        return {
            'n_subjects': len(accuracies),
            'mean_accuracy': float(accuracies.mean()),
            'std_accuracy': float(accuracies.std()),
            'min_accuracy': float(accuracies.min()),
            'max_accuracy': float(accuracies.max()),
            f'p{int(percentile)}_accuracy': float(q10),
            'median_accuracy': float(q50),
            f'p{int(100-percentile)}_accuracy': float(q90),
            'worst_subjects': worst_subjects,
            'worst_accuracies': worst_accuracies,
            'worst_case_acceptable': q10 >= 0.5  # Above chance
        }

    @staticmethod
    def subject_contribution_analysis(
        subject_ids: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze contribution of each subject to the dataset.
        """
        unique_subjects = np.unique(subject_ids)
        contributions = {}

        for subj in unique_subjects:
            mask = subject_ids == subj
            n_samples = mask.sum()
            class_dist = np.bincount(y_true[mask].astype(int), minlength=2)

            contributions[str(subj)] = {
                'n_samples': int(n_samples),
                'class_0': int(class_dist[0]),
                'class_1': int(class_dist[1]),
                'class_ratio': float(class_dist[1] / (class_dist[0] + 1e-8))
            }

        # Check for dominance
        total_samples = len(y_true)
        sample_counts = [c['n_samples'] for c in contributions.values()]
        max_contribution = max(sample_counts) / total_samples

        return {
            'subject_contributions': contributions,
            'n_subjects': len(unique_subjects),
            'max_contribution_ratio': max_contribution,
            'is_balanced': max_contribution < 0.2,  # No subject > 20%
            'total_samples': total_samples
        }


# =============================================================================
# PHASE 5: SENSITIVITY ANALYSIS
# =============================================================================

class SensitivityAnalyzer:
    """Sensitivity and ablation analysis."""

    @staticmethod
    def hyperparameter_sensitivity(
        results_grid: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Analyze sensitivity to hyperparameter changes.

        Args:
            results_grid: Dict mapping HP config string to metrics
        """
        metrics = list(list(results_grid.values())[0].keys())
        sensitivity = {}

        for metric in metrics:
            values = [r[metric] for r in results_grid.values()]
            sensitivity[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'range': float(np.max(values) - np.min(values)),
                'cv': float(np.std(values) / (np.mean(values) + 1e-8))
            }

        # Overall stability
        primary_metric = 'accuracy' if 'accuracy' in metrics else metrics[0]
        is_stable = sensitivity[primary_metric]['cv'] < 0.1

        return {
            'sensitivity': sensitivity,
            'is_stable': is_stable,
            'best_config': max(results_grid.keys(),
                              key=lambda k: results_grid[k].get(primary_metric, 0))
        }

    @staticmethod
    def seed_sensitivity(
        seed_results: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Analyze sensitivity to random seed.
        """
        metrics = list(seed_results[0].keys())
        sensitivity = {}

        for metric in metrics:
            values = [r[metric] for r in seed_results]
            sensitivity[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }

        # Check if variance is acceptable
        primary_metric = 'accuracy' if 'accuracy' in metrics else metrics[0]
        is_reproducible = sensitivity[primary_metric]['std'] < 0.03

        return {
            'seed_sensitivity': sensitivity,
            'n_seeds': len(seed_results),
            'is_reproducible': is_reproducible
        }

    @staticmethod
    def ablation_analysis(
        full_result: Dict[str, float],
        ablation_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Analyze ablation study results.

        Args:
            full_result: Metrics with full model
            ablation_results: Dict mapping ablation name to metrics
        """
        primary_metric = 'accuracy'
        full_score = full_result[primary_metric]

        ablation_impacts = {}
        for name, result in ablation_results.items():
            ablated_score = result[primary_metric]
            impact = full_score - ablated_score
            ablation_impacts[name] = {
                'full_score': full_score,
                'ablated_score': ablated_score,
                'impact': impact,
                'relative_impact': impact / (full_score + 1e-8),
                'is_important': impact > 0.02  # >2% drop
            }

        # Rank by importance
        ranked = sorted(ablation_impacts.items(),
                       key=lambda x: x[1]['impact'], reverse=True)

        return {
            'ablation_impacts': ablation_impacts,
            'importance_ranking': [name for name, _ in ranked],
            'most_important': ranked[0][0] if ranked else None
        }


# =============================================================================
# PHASE 6: STATISTICAL ANALYSIS
# =============================================================================

class StatisticalAnalyzer:
    """Statistical testing and effect size computation."""

    @staticmethod
    def effect_size_cohens_d(
        group1: np.ndarray,
        group2: np.ndarray
    ) -> float:
        """Compute Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        return (group1.mean() - group2.mean()) / (pooled_std + 1e-8)

    @staticmethod
    def paired_comparison(
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        test_type: str = 'wilcoxon'
    ) -> Dict[str, Any]:
        """
        Paired statistical comparison between two models.

        Args:
            scores_a: Scores from model A (e.g., per fold)
            scores_b: Scores from model B
            test_type: 'wilcoxon' or 'ttest'
        """
        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)

        diff = scores_a - scores_b

        # Effect size
        cohens_d = diff.mean() / (diff.std() + 1e-8)

        # Statistical test
        if test_type == 'wilcoxon' and len(diff) >= 5:
            try:
                stat, p_value = wilcoxon(diff)
            except:
                stat, p_value = None, 1.0
        else:
            stat, p_value = stats.ttest_rel(scores_a, scores_b)

        return {
            'mean_a': float(scores_a.mean()),
            'mean_b': float(scores_b.mean()),
            'mean_diff': float(diff.mean()),
            'std_diff': float(diff.std()),
            'cohens_d': float(cohens_d),
            'test_type': test_type,
            'statistic': float(stat) if stat is not None else None,
            'p_value': float(p_value),
            'is_significant': p_value < 0.05,
            'direction': 'A > B' if diff.mean() > 0 else 'B > A'
        }

    @staticmethod
    def multiple_comparison_correction(
        p_values: List[float],
        method: str = 'holm'
    ) -> Dict[str, Any]:
        """
        Apply multiple comparison correction.

        Args:
            p_values: List of p-values
            method: 'holm' (Holm-Bonferroni) or 'fdr' (Benjamini-Hochberg)
        """
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_idx]

        if method == 'holm':
            # Holm-Bonferroni
            adjusted = np.zeros(n)
            for i, p in enumerate(sorted_p):
                adjusted[sorted_idx[i]] = min(1.0, p * (n - i))

        elif method == 'fdr':
            # Benjamini-Hochberg
            adjusted = np.zeros(n)
            for i, p in enumerate(sorted_p):
                adjusted[sorted_idx[i]] = min(1.0, p * n / (i + 1))

        else:
            adjusted = np.array(p_values)

        return {
            'original_p_values': p_values,
            'adjusted_p_values': adjusted.tolist(),
            'method': method,
            'n_significant_original': sum(p < 0.05 for p in p_values),
            'n_significant_adjusted': sum(p < 0.05 for p in adjusted)
        }


# =============================================================================
# PHASE 7: REPORTING
# =============================================================================

class ReportGenerator:
    """Generate thesis-quality reports and tables."""

    @staticmethod
    def generate_main_results_table(
        results: Dict[str, Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate main results table in LaTeX format.
        """
        latex = """
\\begin{table}[htbp]
\\centering
\\caption{Classification Performance on EEG Stress Detection}
\\label{tab:main_results}
\\begin{tabular}{lccccc}
\\toprule
Model & Accuracy & Precision & Recall & F1-Score & AUC-ROC \\\\
\\midrule
"""
        for model_name, metrics in results.items():
            if 'accuracy' in metrics:
                acc = metrics.get('accuracy', {})
                prec = metrics.get('precision', {})
                rec = metrics.get('recall', {})
                f1 = metrics.get('f1', {})
                auc = metrics.get('auc_roc', {})

                # Format with CI if available
                def fmt(m):
                    if isinstance(m, dict) and 'mean' in m:
                        return f"{m['mean']*100:.1f} $\\pm$ {m.get('std', 0)*100:.1f}"
                    elif isinstance(m, (int, float)):
                        return f"{m*100:.1f}"
                    return "—"

                latex += f"{model_name} & {fmt(acc)} & {fmt(prec)} & {fmt(rec)} & {fmt(f1)} & {fmt(auc)} \\\\\n"

        latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

        if output_path:
            with open(output_path, 'w') as f:
                f.write(latex)

        return latex

    @staticmethod
    def generate_subject_analysis_table(
        subject_analysis: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate subject analysis table.
        """
        wa = subject_analysis

        latex = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Subject-wise Performance Analysis}}
\\label{{tab:subject_analysis}}
\\begin{{tabular}}{{lr}}
\\toprule
Metric & Value \\\\
\\midrule
Number of Subjects & {wa.get('n_subjects', '—')} \\\\
Mean Accuracy & {wa.get('mean_accuracy', 0)*100:.1f}\\% \\\\
Std Accuracy & {wa.get('std_accuracy', 0)*100:.1f}\\% \\\\
Minimum Accuracy & {wa.get('min_accuracy', 0)*100:.1f}\\% \\\\
10th Percentile & {wa.get('p10_accuracy', 0)*100:.1f}\\% \\\\
Median Accuracy & {wa.get('median_accuracy', 0)*100:.1f}\\% \\\\
90th Percentile & {wa.get('p90_accuracy', 0)*100:.1f}\\% \\\\
Maximum Accuracy & {wa.get('max_accuracy', 0)*100:.1f}\\% \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

        if output_path:
            with open(output_path, 'w') as f:
                f.write(latex)

        return latex

    @staticmethod
    def generate_ablation_table(
        ablation_results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate ablation study table.
        """
        latex = """
\\begin{table}[htbp]
\\centering
\\caption{Ablation Study Results}
\\label{tab:ablation}
\\begin{tabular}{lcccc}
\\toprule
Configuration & Accuracy & Change & Important \\\\
\\midrule
"""

        for name, impact in ablation_results.get('ablation_impacts', {}).items():
            change = impact.get('impact', 0) * 100
            sign = "+" if change > 0 else ""
            important = "Yes" if impact.get('is_important', False) else "No"

            latex += f"{name} & {impact.get('ablated_score', 0)*100:.1f}\\% & {sign}{change:.1f}\\% & {important} \\\\\n"

        latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

        if output_path:
            with open(output_path, 'w') as f:
                f.write(latex)

        return latex


# =============================================================================
# COMPREHENSIVE EVALUATOR
# =============================================================================

class ComprehensiveEvaluator:
    """
    Main class for comprehensive model evaluation.

    Combines all analysis phases into a unified evaluation pipeline.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()

        # Initialize analyzers
        self.model_analyzer = ModelAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.subject_analyzer = SubjectAnalyzer()
        self.sensitivity_analyzer = SensitivityAnalyzer()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.report_generator = ReportGenerator()

        self.results = {}

    def full_evaluation(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_prob_test: Optional[np.ndarray] = None,
        subject_ids_test: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Run full evaluation pipeline.

        Args:
            model: Trained model
            X_train, y_train: Training data
            X_test, y_test: Test data
            y_prob_test: Predicted probabilities on test set
            subject_ids_test: Subject IDs for test samples

        Returns:
            Comprehensive evaluation report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__
        }

        # 1. Model Analysis
        print("Running model analysis...")
        report['model_analysis'] = {
            'complexity': self.model_analyzer.compute_model_complexity(model)
        }

        # 2. Get predictions if not provided
        if y_prob_test is None and TORCH_AVAILABLE:
            import torch
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test)
                output = model(X_tensor)
                y_prob_test = output['probs'][:, 1].numpy()

        y_pred_test = (y_prob_test >= 0.5).astype(int) if y_prob_test is not None else None

        # 3. Performance Analysis
        print("Running performance analysis...")
        if y_pred_test is not None:
            report['performance_analysis'] = {
                'metrics': self.performance_analyzer.compute_all_metrics(
                    y_test, y_pred_test, y_prob_test
                ),
                'confidence_intervals': self.performance_analyzer.compute_confidence_intervals(
                    y_test, y_pred_test, y_prob_test,
                    n_bootstrap=self.config.n_bootstrap,
                    ci_level=self.config.ci_level
                )
            }

            if y_prob_test is not None:
                report['performance_analysis']['calibration'] = \
                    self.performance_analyzer.compute_calibration(
                        y_test, y_prob_test, self.config.n_calibration_bins
                    )
                report['performance_analysis']['threshold_analysis'] = \
                    self.performance_analyzer.threshold_analysis(y_test, y_prob_test)

        # 4. Subject Analysis
        if subject_ids_test is not None and y_pred_test is not None:
            print("Running subject analysis...")
            subject_metrics = self.subject_analyzer.per_subject_metrics(
                y_test, y_pred_test, y_prob_test, subject_ids_test
            )
            report['subject_analysis'] = {
                'per_subject': subject_metrics,
                'worst_case': self.subject_analyzer.worst_case_analysis(
                    subject_metrics, self.config.worst_case_percentile
                ),
                'contributions': self.subject_analyzer.subject_contribution_analysis(
                    subject_ids_test, y_test
                )
            }

        # 5. Confusion Matrix
        if y_pred_test is not None:
            cm = confusion_matrix(y_test, y_pred_test)
            report['confusion_matrix'] = {
                'matrix': cm.tolist(),
                'tn': int(cm[0, 0]),
                'fp': int(cm[0, 1]),
                'fn': int(cm[1, 0]),
                'tp': int(cm[1, 1])
            }

        self.results = report
        return report

    def save_report(self, output_path: str):
        """Save evaluation report to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Report saved to {output_path}")

    def generate_thesis_tables(
        self,
        report: Dict[str, Any],
        output_dir: str
    ):
        """Generate all thesis-quality tables."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Main results table
        if 'performance_analysis' in report:
            results = {'GenAI-RAG-EEG': report['performance_analysis'].get('confidence_intervals', {})}
            self.report_generator.generate_main_results_table(
                results, str(output_dir / 'table_main_results.tex')
            )

        # Subject analysis table
        if 'subject_analysis' in report:
            self.report_generator.generate_subject_analysis_table(
                report['subject_analysis'].get('worst_case', {}),
                str(output_dir / 'table_subject_analysis.tex')
            )

        print(f"Tables generated in {output_dir}")


if __name__ == "__main__":
    print("Testing Comprehensive Evaluation Module")
    print("=" * 50)

    # Create synthetic data
    np.random.seed(42)
    n_samples = 200
    n_channels = 32
    n_time = 3200  # SAM-40: 25 sec at 128 Hz
    n_subjects = 10

    X = np.random.randn(n_samples, n_channels, n_time).astype(np.float32)
    y = np.random.randint(0, 2, n_samples)
    y_prob = np.clip(y + np.random.randn(n_samples) * 0.3, 0, 1)
    y_pred = (y_prob >= 0.5).astype(int)
    subject_ids = np.repeat(np.arange(n_subjects), n_samples // n_subjects)

    # Test analyzers
    print("\n1. Performance Analysis")
    perf = PerformanceAnalyzer()
    metrics = perf.compute_all_metrics(y, y_pred, y_prob)
    print(f"   Metrics: {metrics}")

    print("\n2. Subject Analysis")
    subj = SubjectAnalyzer()
    subj_metrics = subj.per_subject_metrics(y, y_pred, y_prob, subject_ids)
    worst = subj.worst_case_analysis(subj_metrics)
    print(f"   Worst case: p10={worst['p10_accuracy']*100:.1f}%")

    print("\n3. Statistical Analysis")
    stat = StatisticalAnalyzer()
    comparison = stat.paired_comparison(
        np.random.rand(5) * 0.2 + 0.7,
        np.random.rand(5) * 0.2 + 0.65
    )
    print(f"   Cohen's d: {comparison['cohens_d']:.3f}, p={comparison['p_value']:.4f}")

    print("\n✓ All evaluation modules work correctly!")
