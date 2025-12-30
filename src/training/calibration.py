#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Model Calibration Module for GenAI-RAG-EEG
================================================================================

This module provides probability calibration methods:
- Temperature Scaling (post-hoc)
- Platt Scaling
- Isotonic Regression
- Calibration metrics (ECE, MCE, Brier)

Usage:
    from src.training.calibration import TemperatureScaling, CalibrationMetrics

    # Calibrate model
    calibrator = TemperatureScaling()
    calibrator.fit(val_logits, val_labels)
    calibrated_probs = calibrator.calibrate(test_logits)

    # Evaluate calibration
    metrics = CalibrationMetrics()
    ece = metrics.expected_calibration_error(y_true, y_prob)

Author: GenAI-RAG-EEG Team
Version: 3.0.0
License: MIT
================================================================================
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve


@dataclass
class CalibrationConfig:
    """Configuration for calibration."""
    n_bins: int = 15
    strategy: str = 'uniform'  # 'uniform' or 'quantile'


class CalibrationMetrics:
    """
    Compute calibration metrics.

    Metrics:
    - ECE (Expected Calibration Error)
    - MCE (Maximum Calibration Error)
    - Brier Score
    - Reliability Diagram data
    """

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins

    def expected_calibration_error(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        ECE = sum(|bin_accuracy - bin_confidence| * bin_weight)
        """
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0

        for i in range(self.n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_accuracy = y_true[mask].mean()
                bin_confidence = y_prob[mask].mean()
                bin_weight = mask.sum() / len(y_true)
                ece += bin_weight * abs(bin_accuracy - bin_confidence)

        return float(ece)

    def maximum_calibration_error(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        """
        Compute Maximum Calibration Error (MCE).

        MCE = max(|bin_accuracy - bin_confidence|)
        """
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        mce = 0.0

        for i in range(self.n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_accuracy = y_true[mask].mean()
                bin_confidence = y_prob[mask].mean()
                mce = max(mce, abs(bin_accuracy - bin_confidence))

        return float(mce)

    def brier_score(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        """
        Compute Brier Score.

        Brier = mean((y_prob - y_true)^2)
        """
        return float(np.mean((y_prob - y_true) ** 2))

    def reliability_diagram_data(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get data for reliability diagram.
        """
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=self.n_bins, strategy='uniform'
        )

        return {
            'prob_true': prob_true,
            'prob_pred': prob_pred,
            'perfect_calibration': np.linspace(0, 1, 100)
        }

    def compute_all(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Compute all calibration metrics."""
        return {
            'ece': self.expected_calibration_error(y_true, y_prob),
            'mce': self.maximum_calibration_error(y_true, y_prob),
            'brier': self.brier_score(y_true, y_prob),
            'is_well_calibrated': self.expected_calibration_error(y_true, y_prob) < 0.1
        }


class TemperatureScaling:
    """
    Temperature Scaling for neural network calibration.

    Learns a single temperature parameter T to scale logits:
    calibrated_prob = softmax(logits / T)
    """

    def __init__(self, max_iter: int = 100, lr: float = 0.01):
        self.max_iter = max_iter
        self.lr = lr
        self.temperature = 1.0

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray
    ) -> 'TemperatureScaling':
        """
        Fit temperature parameter on validation set.

        Args:
            logits: Raw model logits (n_samples, n_classes) or (n_samples,)
            labels: True labels (n_samples,)
        """
        if not TORCH_AVAILABLE:
            warnings.warn("PyTorch not available. Using default temperature=1.0")
            return self

        # Convert to tensors
        if logits.ndim == 1:
            logits = np.column_stack([1 - logits, logits])

        logits_tensor = torch.FloatTensor(logits)
        labels_tensor = torch.LongTensor(labels)

        # Initialize temperature
        temperature = nn.Parameter(torch.ones(1) * 1.5)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([temperature], lr=self.lr, max_iter=self.max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits_tensor / temperature
            loss = criterion(scaled_logits, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)

        self.temperature = float(temperature.item())
        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Calibrate logits using learned temperature.

        Args:
            logits: Raw model logits

        Returns:
            Calibrated probabilities
        """
        if logits.ndim == 1:
            # Binary case: logits are log-odds
            scaled = logits / self.temperature
            return 1 / (1 + np.exp(-scaled))
        else:
            # Multi-class case
            scaled = logits / self.temperature
            exp_scaled = np.exp(scaled - scaled.max(axis=1, keepdims=True))
            return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)

    def get_temperature(self) -> float:
        """Get learned temperature."""
        return self.temperature


class PlattScaling:
    """
    Platt Scaling for probability calibration.

    Fits a logistic regression on the model outputs.
    """

    def __init__(self):
        self.calibrator = LogisticRegression(
            solver='lbfgs',
            max_iter=1000
        )

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> 'PlattScaling':
        """
        Fit Platt scaling on validation set.

        Args:
            scores: Model scores/probabilities (n_samples,)
            labels: True labels (n_samples,)
        """
        scores = scores.reshape(-1, 1)
        self.calibrator.fit(scores, labels)
        return self

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """
        Calibrate scores.

        Returns:
            Calibrated probabilities
        """
        scores = scores.reshape(-1, 1)
        return self.calibrator.predict_proba(scores)[:, 1]


class IsotonicCalibration:
    """
    Isotonic Regression for probability calibration.

    Non-parametric calibration that preserves ordering.
    """

    def __init__(self):
        self.calibrator = IsotonicRegression(
            y_min=0,
            y_max=1,
            out_of_bounds='clip'
        )

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> 'IsotonicCalibration':
        """
        Fit isotonic regression on validation set.
        """
        self.calibrator.fit(scores, labels)
        return self

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """
        Calibrate scores.
        """
        return self.calibrator.predict(scores)


class ModelCalibrator:
    """
    Complete calibration pipeline for models.

    Supports multiple calibration methods and automatic selection.
    """

    def __init__(
        self,
        method: str = 'temperature',
        n_bins: int = 15
    ):
        """
        Args:
            method: 'temperature', 'platt', 'isotonic', or 'auto'
            n_bins: Number of bins for calibration evaluation
        """
        self.method = method
        self.metrics = CalibrationMetrics(n_bins=n_bins)
        self.calibrator = None
        self.calibration_results = {}

    def fit(
        self,
        logits_or_probs: np.ndarray,
        labels: np.ndarray,
        is_logits: bool = True
    ) -> 'ModelCalibrator':
        """
        Fit calibrator on validation data.

        Args:
            logits_or_probs: Model outputs (logits or probabilities)
            labels: True labels
            is_logits: Whether inputs are logits (True) or probabilities (False)
        """
        if is_logits:
            # Convert logits to probabilities for evaluation
            if logits_or_probs.ndim == 1:
                probs = 1 / (1 + np.exp(-logits_or_probs))
            else:
                exp_logits = np.exp(logits_or_probs - logits_or_probs.max(axis=1, keepdims=True))
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                probs = probs[:, 1] if probs.shape[1] == 2 else probs
        else:
            probs = logits_or_probs

        # Compute pre-calibration metrics
        self.calibration_results['pre_calibration'] = self.metrics.compute_all(labels, probs)

        # Fit calibrator based on method
        if self.method == 'temperature':
            self.calibrator = TemperatureScaling()
            if is_logits:
                self.calibrator.fit(logits_or_probs, labels)
            else:
                # Convert probs to logits
                eps = 1e-7
                logits = np.log(probs + eps) - np.log(1 - probs + eps)
                self.calibrator.fit(logits, labels)

        elif self.method == 'platt':
            self.calibrator = PlattScaling()
            self.calibrator.fit(probs, labels)

        elif self.method == 'isotonic':
            self.calibrator = IsotonicCalibration()
            self.calibrator.fit(probs, labels)

        elif self.method == 'auto':
            # Try all methods and pick best
            best_ece = float('inf')
            best_method = None

            for method_name, calibrator_class in [
                ('temperature', TemperatureScaling),
                ('platt', PlattScaling),
                ('isotonic', IsotonicCalibration)
            ]:
                try:
                    cal = calibrator_class()
                    if method_name == 'temperature' and is_logits:
                        cal.fit(logits_or_probs, labels)
                        calib_probs = cal.calibrate(logits_or_probs)
                    else:
                        cal.fit(probs, labels)
                        calib_probs = cal.calibrate(probs)

                    ece = self.metrics.expected_calibration_error(labels, calib_probs)
                    if ece < best_ece:
                        best_ece = ece
                        best_method = method_name
                        self.calibrator = cal
                except Exception:
                    continue

            self.calibration_results['best_method'] = best_method

        return self

    def calibrate(
        self,
        logits_or_probs: np.ndarray,
        is_logits: bool = True
    ) -> np.ndarray:
        """
        Calibrate model outputs.

        Returns:
            Calibrated probabilities
        """
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        if isinstance(self.calibrator, TemperatureScaling):
            if is_logits:
                return self.calibrator.calibrate(logits_or_probs)
            else:
                eps = 1e-7
                logits = np.log(logits_or_probs + eps) - np.log(1 - logits_or_probs + eps)
                return self.calibrator.calibrate(logits)
        else:
            if is_logits:
                if logits_or_probs.ndim == 1:
                    probs = 1 / (1 + np.exp(-logits_or_probs))
                else:
                    exp_logits = np.exp(logits_or_probs - logits_or_probs.max(axis=1, keepdims=True))
                    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                    probs = probs[:, 1]
            else:
                probs = logits_or_probs

            return self.calibrator.calibrate(probs)

    def evaluate(
        self,
        logits_or_probs: np.ndarray,
        labels: np.ndarray,
        is_logits: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate calibration on test set.
        """
        calibrated_probs = self.calibrate(logits_or_probs, is_logits)
        metrics = self.metrics.compute_all(labels, calibrated_probs)
        self.calibration_results['post_calibration'] = metrics
        return metrics

    def get_improvement(self) -> Dict[str, float]:
        """Get calibration improvement."""
        if 'pre_calibration' not in self.calibration_results:
            return {}

        pre = self.calibration_results['pre_calibration']
        post = self.calibration_results.get('post_calibration', pre)

        return {
            'ece_improvement': pre['ece'] - post['ece'],
            'mce_improvement': pre['mce'] - post['mce'],
            'brier_improvement': pre['brier'] - post['brier'],
            'relative_ece_improvement': (pre['ece'] - post['ece']) / (pre['ece'] + 1e-8)
        }


if __name__ == "__main__":
    print("Testing Calibration Module")
    print("=" * 50)

    # Create synthetic data
    np.random.seed(42)
    n_samples = 500

    # Simulate overconfident model
    y_true = np.random.randint(0, 2, n_samples)
    # Generate overconfident probabilities
    y_prob_raw = np.clip(y_true + np.random.randn(n_samples) * 0.2, 0.1, 0.9)
    y_prob_raw = np.where(y_prob_raw > 0.5, y_prob_raw * 1.2, y_prob_raw * 0.8)
    y_prob_raw = np.clip(y_prob_raw, 0, 1)

    print(f"Samples: {n_samples}")
    print(f"Class balance: {y_true.mean():.2f}")

    # Test calibration metrics
    metrics = CalibrationMetrics()
    print(f"\nPre-calibration metrics:")
    pre_metrics = metrics.compute_all(y_true, y_prob_raw)
    print(f"  ECE: {pre_metrics['ece']:.4f}")
    print(f"  MCE: {pre_metrics['mce']:.4f}")
    print(f"  Brier: {pre_metrics['brier']:.4f}")

    # Split data
    split = int(0.6 * n_samples)
    y_val, y_test = y_true[:split], y_true[split:]
    prob_val, prob_test = y_prob_raw[:split], y_prob_raw[split:]

    # Test temperature scaling
    print("\n1. Temperature Scaling:")
    temp_scaler = TemperatureScaling()
    logits = np.log(prob_val + 1e-7) - np.log(1 - prob_val + 1e-7)
    temp_scaler.fit(logits, y_val)
    print(f"   Learned temperature: {temp_scaler.get_temperature():.3f}")

    # Test Platt scaling
    print("\n2. Platt Scaling:")
    platt = PlattScaling()
    platt.fit(prob_val, y_val)
    platt_probs = platt.calibrate(prob_test)
    platt_metrics = metrics.compute_all(y_test, platt_probs)
    print(f"   Post-calibration ECE: {platt_metrics['ece']:.4f}")

    # Test full calibrator
    print("\n3. Model Calibrator (auto):")
    calibrator = ModelCalibrator(method='auto')
    calibrator.fit(prob_val, y_val, is_logits=False)
    post_metrics = calibrator.evaluate(prob_test, y_test, is_logits=False)
    improvement = calibrator.get_improvement()
    print(f"   Best method: {calibrator.calibration_results.get('best_method', 'N/A')}")
    print(f"   Post-calibration ECE: {post_metrics['ece']:.4f}")
    print(f"   ECE improvement: {improvement.get('ece_improvement', 0):.4f}")

    print("\n✓ Calibration module works!")
