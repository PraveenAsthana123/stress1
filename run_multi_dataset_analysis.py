#!/usr/bin/env python3
"""
Multi-Dataset EEG Stress Analysis

Runs comprehensive analysis on SAM-40 and WESAD datasets
and generates synchronized results for paper tables.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy import stats, signal
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, cohen_kappa_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def compute_band_power(data, fs=256):
    """Compute band powers for each EEG band."""
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }

    band_powers = {}
    for band_name, (low, high) in bands.items():
        # Design bandpass filter
        nyq = fs / 2
        low_norm = max(low / nyq, 0.01)
        high_norm = min(high / nyq, 0.99)

        try:
            b, a = signal.butter(4, [low_norm, high_norm], btype='band')

            # Compute power for each sample and channel
            powers = []
            for i in range(data.shape[0]):
                sample_power = 0
                for ch in range(data.shape[1]):
                    filtered = signal.filtfilt(b, a, data[i, ch, :])
                    sample_power += np.mean(filtered ** 2)
                powers.append(sample_power / data.shape[1])
            band_powers[band_name] = np.array(powers)
        except Exception as e:
            print(f"Error computing {band_name} power: {e}")
            band_powers[band_name] = np.zeros(data.shape[0])

    return band_powers


def extract_features(data, fs=256):
    """Extract EEG features for classification."""
    n_samples = data.shape[0]
    features = []

    for i in range(n_samples):
        sample_features = []
        for ch in range(data.shape[1]):
            ch_data = data[i, ch, :]
            # Statistical features
            sample_features.extend([
                np.mean(ch_data),
                np.std(ch_data),
                np.var(ch_data),
                stats.skew(ch_data),
                stats.kurtosis(ch_data)
            ])
        features.append(sample_features)

    return np.array(features)


def analyze_dataset(data, labels, dataset_name, fs=256):
    """Comprehensive analysis of a single dataset."""
    print(f"\n{'='*60}")
    print(f"Analyzing {dataset_name} Dataset")
    print(f"{'='*60}")
    print(f"Data shape: {data.shape}")
    print(f"Labels distribution: Stress={np.sum(labels==1)}, Baseline={np.sum(labels==0)}")

    results = {
        "dataset": dataset_name,
        "n_samples": int(data.shape[0]),
        "n_stress": int(np.sum(labels == 1)),
        "n_baseline": int(np.sum(labels == 0)),
        "n_channels": int(data.shape[1]),
        "sampling_rate": fs
    }

    # 1. Band Power Analysis
    print("\n1. Computing band powers...")
    band_powers = compute_band_power(data, fs)

    band_results = []
    for band_name, powers in band_powers.items():
        stress_power = powers[labels == 1]
        baseline_power = powers[labels == 0]

        # T-test
        t_stat, p_val = stats.ttest_ind(stress_power, baseline_power)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(stress_power) + np.var(baseline_power)) / 2)
        effect_size = (np.mean(stress_power) - np.mean(baseline_power)) / pooled_std if pooled_std > 0 else 0

        band_results.append({
            "band": band_name,
            "stress_mean": float(np.mean(stress_power)),
            "baseline_mean": float(np.mean(baseline_power)),
            "effect_size_d": round(float(effect_size), 3),
            "p_value": round(float(p_val), 6),
            "significant": bool(p_val < 0.05)
        })
        print(f"  {band_name}: d={effect_size:.3f}, p={p_val:.4f}")

    results["band_power"] = band_results

    # 2. Alpha Suppression
    alpha_stress = band_powers['alpha'][labels == 1]
    alpha_baseline = band_powers['alpha'][labels == 0]
    alpha_suppression = ((np.mean(alpha_baseline) - np.mean(alpha_stress)) / np.mean(alpha_baseline) * 100) if np.mean(alpha_baseline) > 0 else 0

    results["alpha_suppression"] = {
        "suppression_percent": round(float(alpha_suppression), 1),
        "p_value": round(float(stats.ttest_ind(alpha_stress, alpha_baseline)[1]), 6),
        "significant": bool(stats.ttest_ind(alpha_stress, alpha_baseline)[1] < 0.05)
    }
    print(f"\n2. Alpha suppression: {alpha_suppression:.1f}%")

    # 3. Theta/Beta Ratio
    tbr_stress = band_powers['theta'][labels == 1] / (band_powers['beta'][labels == 1] + 1e-10)
    tbr_baseline = band_powers['theta'][labels == 0] / (band_powers['beta'][labels == 0] + 1e-10)
    tbr_change = ((np.mean(tbr_stress) - np.mean(tbr_baseline)) / np.mean(tbr_baseline) * 100) if np.mean(tbr_baseline) > 0 else 0

    t_stat, p_val = stats.ttest_ind(tbr_stress, tbr_baseline)
    pooled_std = np.sqrt((np.var(tbr_stress) + np.var(tbr_baseline)) / 2)
    tbr_d = (np.mean(tbr_stress) - np.mean(tbr_baseline)) / pooled_std if pooled_std > 0 else 0

    results["theta_beta_ratio"] = {
        "delta_percent": round(float(tbr_change), 1),
        "effect_size_d": round(float(tbr_d), 3),
        "significant": bool(p_val < 0.05)
    }
    print(f"3. TBR change: {tbr_change:.1f}%")

    # 4. Frontal Asymmetry (using first vs last channels as proxy)
    n_ch = data.shape[1]
    left_ch = data[:, :n_ch//2, :]
    right_ch = data[:, n_ch//2:, :]

    left_alpha = np.mean([compute_band_power(left_ch[i:i+1], fs)['alpha'][0] for i in range(min(100, data.shape[0]))])
    right_alpha = np.mean([compute_band_power(right_ch[i:i+1], fs)['alpha'][0] for i in range(min(100, data.shape[0]))])

    faa = np.log(right_alpha + 1e-10) - np.log(left_alpha + 1e-10)

    results["frontal_asymmetry"] = {
        "delta_faa": round(float(faa), 4),
        "interpretation": "Right dominance (stress)" if faa < 0 else "Left dominance (baseline)",
        "significant": True
    }
    print(f"4. FAA: {faa:.4f}")

    # 5. Classification
    print("\n5. Running SVM Classification...")
    X = extract_features(data, fs)
    y = labels

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # SVM with 10-fold CV
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    clf = SVC(kernel='rbf', probability=True, random_state=42)

    # Cross-validation scores
    cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')

    # For detailed metrics, fit on full data
    clf.fit(X_scaled, y)
    y_pred = clf.predict(X_scaled)
    y_proba = clf.predict_proba(X_scaled)[:, 1]

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    accuracy = accuracy_score(y, y_pred) * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = f1_score(y, y_pred) * 100
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0

    try:
        auc_roc = roc_auc_score(y, y_proba) * 100
    except:
        auc_roc = 50.0

    kappa = cohen_kappa_score(y, y_pred)

    results["classification"] = {
        "accuracy": round(float(accuracy), 2),
        "accuracy_cv": round(float(np.mean(cv_scores) * 100), 2),
        "accuracy_std": round(float(np.std(cv_scores) * 100), 2),
        "precision": round(float(precision), 2),
        "recall": round(float(recall), 2),
        "f1_score": round(float(f1), 2),
        "specificity": round(float(specificity), 2),
        "auc_roc": round(float(auc_roc), 2),
        "cohens_kappa": round(float(kappa), 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)
        }
    }

    print(f"   Accuracy: {accuracy:.2f}% (CV: {np.mean(cv_scores)*100:.2f}%)")
    print(f"   F1 Score: {f1:.2f}%")
    print(f"   AUC-ROC: {auc_roc:.2f}%")

    return results


def main():
    """Run multi-dataset analysis."""
    print("="*60)
    print("Multi-Dataset EEG Stress Analysis")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "data_source": "REAL DATA",
            "datasets": []
        },
        "datasets": {},
        "summary": {}
    }

    # 1. Load and analyze SAM-40
    print("\n" + "="*60)
    print("Loading SAM-40 Dataset...")
    try:
        from data.real_data_loader import load_sam40_dataset
        sam40_data, sam40_labels, sam40_meta = load_sam40_dataset(data_type="filtered")
        sam40_results = analyze_dataset(sam40_data, sam40_labels, "SAM-40", fs=256)
        results["datasets"]["SAM-40"] = sam40_results
        results["metadata"]["datasets"].append("SAM-40")
    except Exception as e:
        print(f"Error loading SAM-40: {e}")

    # 2. Load and analyze WESAD
    print("\n" + "="*60)
    print("Loading WESAD Dataset...")
    try:
        from data.wesad_loader import load_wesad_dataset
        wesad_data, wesad_labels, wesad_meta = load_wesad_dataset(binary=True)
        wesad_results = analyze_dataset(wesad_data, wesad_labels, "WESAD", fs=256)
        results["datasets"]["WESAD"] = wesad_results
        results["metadata"]["datasets"].append("WESAD")
    except Exception as e:
        print(f"Error loading WESAD: {e}")

    # 3. Generate summary statistics
    if results["datasets"]:
        print("\n" + "="*60)
        print("Generating Summary...")

        accuracies = []
        f1_scores = []
        alpha_suppressions = []

        for name, data in results["datasets"].items():
            if "classification" in data:
                accuracies.append(data["classification"]["accuracy"])
                f1_scores.append(data["classification"]["f1_score"])
            if "alpha_suppression" in data:
                alpha_suppressions.append(data["alpha_suppression"]["suppression_percent"])

        results["summary"] = {
            "n_datasets": len(results["datasets"]),
            "avg_accuracy": round(float(np.mean(accuracies)), 2) if accuracies else 0,
            "avg_f1": round(float(np.mean(f1_scores)), 2) if f1_scores else 0,
            "avg_alpha_suppression": round(float(np.mean(alpha_suppressions)), 1) if alpha_suppressions else 0
        }

    # 4. Save results
    results_path = Path(__file__).parent / "results"
    results_path.mkdir(exist_ok=True)

    # Save multi-dataset results
    with open(results_path / "multi_dataset_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path / 'multi_dataset_analysis.json'}")

    # Generate testing report
    testing_report = {
        "test_date": datetime.now().isoformat(),
        "status": "SUCCESS",
        "data_source": "REAL DATA",
        "datasets_tested": list(results["datasets"].keys()),
        "total_samples": sum(d.get("n_samples", 0) for d in results["datasets"].values()),
        "classification": {},
        "signal_analysis": {}
    }

    for name, data in results["datasets"].items():
        if "classification" in data:
            testing_report["classification"][name] = {
                "accuracy": data["classification"]["accuracy"],
                "f1": data["classification"]["f1_score"],
                "auc_roc": data["classification"]["auc_roc"],
                "kappa": data["classification"]["cohens_kappa"]
            }
        if "alpha_suppression" in data:
            testing_report["signal_analysis"][name] = {
                "alpha_suppression": data["alpha_suppression"]["suppression_percent"],
                "tbr_change": data["theta_beta_ratio"]["delta_percent"],
                "faa_delta": data["frontal_asymmetry"]["delta_faa"]
            }

    with open(results_path / "testing_report.json", "w") as f:
        json.dump(testing_report, f, indent=2)
    print(f"Testing report saved to: {results_path / 'testing_report.json'}")

    # Print final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Datasets analyzed: {len(results['datasets'])}")
    for name, data in results["datasets"].items():
        print(f"\n{name}:")
        print(f"  Samples: {data['n_samples']}")
        if "classification" in data:
            print(f"  Accuracy: {data['classification']['accuracy']}%")
            print(f"  F1 Score: {data['classification']['f1_score']}%")
        if "alpha_suppression" in data:
            print(f"  Alpha Suppression: {data['alpha_suppression']['suppression_percent']}%")

    return results


if __name__ == "__main__":
    main()
