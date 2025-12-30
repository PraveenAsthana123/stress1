#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Classical ML Baselines for EEG Stress Classification
================================================================================

This module implements classical machine learning baselines for comparison
with the deep learning GenAI-RAG-EEG model.

Baselines implemented:
1. Logistic Regression (LR) - Linear baseline
2. Support Vector Machine (SVM) - RBF kernel
3. Random Forest (RF) - Ensemble baseline
4. Linear Discriminant Analysis (LDA) - Classic EEG/BCI baseline
5. XGBoost (XGB) - Gradient boosting baseline
6. Riemannian Geometry (optional) - State-of-the-art EEG baseline

Usage:
    from src.models.baselines import BaselineComparison

    comparison = BaselineComparison()
    results = comparison.run_all(X_train, y_train, X_test, y_test)

Author: GenAI-RAG-EEG Team
Version: 3.0.0
License: MIT
================================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
from pathlib import Path
import json

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Optional imports
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from pyriemann.classification import MDM
    RIEMANN_AVAILABLE = True
except ImportError:
    RIEMANN_AVAILABLE = False


@dataclass
class BaselineConfig:
    """Configuration for baseline models."""
    # General
    random_state: int = 42
    n_jobs: int = -1

    # Logistic Regression
    lr_C: float = 1.0
    lr_max_iter: int = 1000

    # SVM
    svm_C: float = 1.0
    svm_kernel: str = "rbf"
    svm_gamma: str = "scale"

    # Random Forest
    rf_n_estimators: int = 100
    rf_max_depth: int = 10

    # LDA
    lda_solver: str = "svd"

    # XGBoost
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1


def extract_features(X: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from EEG epochs for classical ML.

    Features per channel:
    - Mean, Std, Min, Max
    - Skewness, Kurtosis
    - RMS, Peak-to-peak

    Args:
        X: EEG data (n_samples, n_channels, n_time)

    Returns:
        features: (n_samples, n_channels * n_features)
    """
    from scipy import stats

    n_samples, n_channels, n_time = X.shape

    features_list = []

    for i in range(n_samples):
        sample_features = []
        for ch in range(n_channels):
            signal = X[i, ch, :]

            # Basic statistics
            sample_features.append(np.mean(signal))
            sample_features.append(np.std(signal))
            sample_features.append(np.min(signal))
            sample_features.append(np.max(signal))

            # Higher-order statistics
            sample_features.append(stats.skew(signal))
            sample_features.append(stats.kurtosis(signal))

            # Signal features
            sample_features.append(np.sqrt(np.mean(signal**2)))  # RMS
            sample_features.append(np.max(signal) - np.min(signal))  # Peak-to-peak

        features_list.append(sample_features)

    return np.array(features_list, dtype=np.float32)


def extract_bandpower_features(
    X: np.ndarray,
    fs: float = 256.0,
    bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Extract band power features from EEG epochs.

    Args:
        X: EEG data (n_samples, n_channels, n_time)
        fs: Sampling frequency
        bands: Frequency band definitions

    Returns:
        features: (n_samples, n_channels * n_bands)
    """
    from scipy import signal as sig

    if bands is None:
        bands = {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 45.0)
        }

    n_samples, n_channels, n_time = X.shape
    n_bands = len(bands)

    features = np.zeros((n_samples, n_channels * n_bands))

    for i in range(n_samples):
        feat_idx = 0
        for ch in range(n_channels):
            # Compute PSD
            freqs, psd = sig.welch(X[i, ch], fs=fs, nperseg=min(256, n_time))

            for band_name, (low, high) in bands.items():
                idx = np.where((freqs >= low) & (freqs <= high))[0]
                features[i, feat_idx] = np.trapz(psd[idx], freqs[idx])
                feat_idx += 1

    return features


class LogisticRegressionBaseline:
    """Logistic Regression baseline."""

    def __init__(self, config: Optional[BaselineConfig] = None):
        self.config = config or BaselineConfig()
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(
                C=self.config.lr_C,
                max_iter=self.config.lr_max_iter,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            ))
        ])
        self.name = "Logistic Regression"

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model."""
        if X.ndim == 3:
            X = extract_features(X)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if X.ndim == 3:
            X = extract_features(X)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if X.ndim == 3:
            X = extract_features(X)
        return self.model.predict_proba(X)


class SVMBaseline:
    """Support Vector Machine baseline with RBF kernel."""

    def __init__(self, config: Optional[BaselineConfig] = None):
        self.config = config or BaselineConfig()
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                C=self.config.svm_C,
                kernel=self.config.svm_kernel,
                gamma=self.config.svm_gamma,
                probability=True,
                random_state=self.config.random_state
            ))
        ])
        self.name = "SVM (RBF)"

    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 3:
            X = extract_features(X)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = extract_features(X)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = extract_features(X)
        return self.model.predict_proba(X)


class RandomForestBaseline:
    """Random Forest baseline."""

    def __init__(self, config: Optional[BaselineConfig] = None):
        self.config = config or BaselineConfig()
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            ))
        ])
        self.name = "Random Forest"

    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 3:
            X = extract_features(X)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = extract_features(X)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = extract_features(X)
        return self.model.predict_proba(X)


class LDABaseline:
    """Linear Discriminant Analysis baseline (classic BCI baseline)."""

    def __init__(self, config: Optional[BaselineConfig] = None):
        self.config = config or BaselineConfig()
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis(
                solver=self.config.lda_solver
            ))
        ])
        self.name = "LDA"

    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 3:
            X = extract_features(X)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = extract_features(X)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = extract_features(X)
        return self.model.predict_proba(X)


class XGBoostBaseline:
    """XGBoost baseline."""

    def __init__(self, config: Optional[BaselineConfig] = None):
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")

        self.config = config or BaselineConfig()
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', xgb.XGBClassifier(
                n_estimators=self.config.xgb_n_estimators,
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                use_label_encoder=False,
                eval_metric='logloss'
            ))
        ])
        self.name = "XGBoost"

    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 3:
            X = extract_features(X)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = extract_features(X)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = extract_features(X)
        return self.model.predict_proba(X)


class RiemannianBaseline:
    """
    Riemannian Geometry baseline using covariance matrices.

    This is a state-of-the-art baseline for EEG classification
    that leverages the geometry of symmetric positive definite matrices.
    """

    def __init__(self, config: Optional[BaselineConfig] = None):
        if not RIEMANN_AVAILABLE:
            raise ImportError(
                "pyriemann not available. Install with: pip install pyriemann"
            )

        self.config = config or BaselineConfig()
        self.cov = Covariances(estimator='lwf')
        self.ts = TangentSpace(metric='riemann')
        self.clf = LogisticRegression(
            max_iter=1000,
            random_state=self.config.random_state
        )
        self.name = "Riemannian (Tangent Space)"

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Compute covariance matrices
        cov_matrices = self.cov.fit_transform(X)
        # Project to tangent space
        ts_features = self.ts.fit_transform(cov_matrices)
        # Fit classifier
        self.clf.fit(ts_features, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        cov_matrices = self.cov.transform(X)
        ts_features = self.ts.transform(cov_matrices)
        return self.clf.predict(ts_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        cov_matrices = self.cov.transform(X)
        ts_features = self.ts.transform(cov_matrices)
        return self.clf.predict_proba(ts_features)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate a trained model.

    Args:
        model: Trained model with predict and predict_proba methods
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0
    }


class BaselineComparison:
    """
    Run all baseline models for comparison.

    Usage:
        comparison = BaselineComparison()
        results = comparison.run_all(X_train, y_train, X_test, y_test)
        comparison.print_results()
        comparison.save_results("baseline_results.json")
    """

    def __init__(self, config: Optional[BaselineConfig] = None):
        self.config = config or BaselineConfig()
        self.models = {}
        self.results = {}

        # Initialize all available models
        self.models['Logistic Regression'] = LogisticRegressionBaseline(self.config)
        self.models['SVM (RBF)'] = SVMBaseline(self.config)
        self.models['Random Forest'] = RandomForestBaseline(self.config)
        self.models['LDA'] = LDABaseline(self.config)

        if XGB_AVAILABLE:
            self.models['XGBoost'] = XGBoostBaseline(self.config)

        if RIEMANN_AVAILABLE:
            self.models['Riemannian'] = RiemannianBaseline(self.config)

    def run_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Run all baseline models and collect results.

        Args:
            X_train: Training data (n_samples, n_channels, n_time)
            y_train: Training labels
            X_test: Test data
            y_test: Test labels
            verbose: Print progress

        Returns:
            Dictionary of model results
        """
        self.results = {}

        for name, model in self.models.items():
            if verbose:
                print(f"Training {name}...", end=" ")

            try:
                model.fit(X_train, y_train)
                metrics = evaluate_model(model, X_test, y_test)
                self.results[name] = metrics

                if verbose:
                    print(f"Accuracy: {metrics['accuracy']*100:.1f}%, "
                          f"F1: {metrics['f1']*100:.1f}%")
            except Exception as e:
                if verbose:
                    print(f"Failed: {e}")
                self.results[name] = {'error': str(e)}

        return self.results

    def run_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        verbose: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run cross-validation for all baselines.

        Args:
            X: All data
            y: All labels
            n_folds: Number of CV folds
            verbose: Print progress

        Returns:
            CV results with mean ± std
        """
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                             random_state=self.config.random_state)

        self.results = {}

        # Pre-extract features once
        if X.ndim == 3:
            X_features = extract_features(X)
        else:
            X_features = X

        for name, model in self.models.items():
            if verbose:
                print(f"CV for {name}...", end=" ")

            try:
                if name == 'Riemannian' and RIEMANN_AVAILABLE:
                    # Riemannian needs raw epochs
                    scores = []
                    for train_idx, test_idx in cv.split(X, y):
                        model.fit(X[train_idx], y[train_idx])
                        y_pred = model.predict(X[test_idx])
                        scores.append(accuracy_score(y[test_idx], y_pred))
                    scores = np.array(scores)
                else:
                    scores = cross_val_score(
                        model.model, X_features, y, cv=cv, scoring='accuracy'
                    )

                self.results[name] = {
                    'accuracy_mean': scores.mean(),
                    'accuracy_std': scores.std(),
                    'scores': scores.tolist()
                }

                if verbose:
                    print(f"Accuracy: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

            except Exception as e:
                if verbose:
                    print(f"Failed: {e}")
                self.results[name] = {'error': str(e)}

        return self.results

    def print_results(self):
        """Print results as a formatted table."""
        print("\n" + "=" * 70)
        print("Baseline Comparison Results")
        print("=" * 70)
        print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} "
              f"{'Recall':>10} {'F1':>10} {'AUC':>10}")
        print("-" * 70)

        for name, metrics in self.results.items():
            if 'error' in metrics:
                print(f"{name:<25} {'ERROR':>10}")
            elif 'accuracy_mean' in metrics:
                print(f"{name:<25} "
                      f"{metrics['accuracy_mean']*100:>9.1f}% ± {metrics['accuracy_std']*100:.1f}%")
            else:
                print(f"{name:<25} {metrics['accuracy']*100:>9.1f}% "
                      f"{metrics['precision']*100:>9.1f}% "
                      f"{metrics['recall']*100:>9.1f}% "
                      f"{metrics['f1']*100:>9.1f}% "
                      f"{metrics['auc_roc']*100:>9.1f}%")

        print("=" * 70)

    def save_results(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filepath}")

    def get_best_baseline(self) -> Tuple[str, float]:
        """Get the best performing baseline."""
        best_name = None
        best_acc = 0

        for name, metrics in self.results.items():
            if 'error' not in metrics:
                acc = metrics.get('accuracy', metrics.get('accuracy_mean', 0))
                if acc > best_acc:
                    best_acc = acc
                    best_name = name

        return best_name, best_acc


if __name__ == "__main__":
    print("Testing Baseline Models")
    print("=" * 50)

    # Create synthetic data
    np.random.seed(42)
    n_samples = 200
    n_channels = 32
    n_time = 512

    X = np.random.randn(n_samples, n_channels, n_time).astype(np.float32)
    y = np.random.randint(0, 2, n_samples)

    # Split
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Run comparison
    comparison = BaselineComparison()
    results = comparison.run_all(X_train, y_train, X_test, y_test)
    comparison.print_results()

    # Get best
    best_name, best_acc = comparison.get_best_baseline()
    print(f"\nBest baseline: {best_name} ({best_acc*100:.1f}%)")
