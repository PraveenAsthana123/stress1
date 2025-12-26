# FIGURE SPECIFICATIONS DATABASE — EEG-BCI STRESS PAPER

## Version: 3.0.0

---

## MASTER FIGURE LIST

| Fig. No. | Type | Title | Content | Section | Mandatory | Status |
|----------|------|-------|---------|---------|-----------|--------|
| Fig. 1 | Architecture | Overall EEG-BCI Stress Framework | End-to-end: EEG → preprocessing → features → classifier → stats/explainability | Methods – System Overview | ✅ YES | ⚠️ Need |
| Fig. 2 | Flowchart | LOSO Validation Workflow | Training/testing separation, no data leakage | Methods – Validation | ✅ YES | ⚠️ Need |
| Fig. 3 | Flowchart | Dataset & Labeling Pipeline | Raw EEG → task/rating → normalization → labels | Methods – Labeling | ✅ YES | ⚠️ Need |
| Fig. 4 | Flow Diagram | Feature Extraction Pipeline | Bandpower, TF, Riemannian → feature vector | Methods – Features | ✅ YES | ⚠️ Need |
| Fig. 5 | Architecture | RAG-LLM Explanation Framework | EEG classifier → retriever → LLM → explanation | Methods – RAG | ⚠️ If RAG | ⚠️ Need |
| Fig. 6 | Boxplot | Subject-wise LOSO Performance | Distribution of accuracies (median, IQR) | Results – Performance | ✅ YES | ⚠️ Need |
| Fig. 7 | Matrix | Confusion Matrix | Stress vs non-stress errors | Results – Classification | ✅ YES | ⚠️ Need |
| Fig. 8 | Curve | ROC Curve | Discriminative ability (AUC) | Results – Performance | ⚠️ Strong | ⚠️ Need |
| Fig. 9 | TF Map | Stress vs Non-Stress Spectral | TF representation showing stress markers | Results – Signal | ⚠️ Strong | ⚠️ Need |
| Fig. 10 | Topography | Scalp Distribution | Spatial EEG patterns (α, θ) | Results – Signal | ⚠️ Strong | ⚠️ Need |
| Fig. 11 | Heatmap | Feature Importance | Channel × Band importance | Results – Explainability | ⚠️ Strong | ⚠️ Need |
| Fig. 12 | Bar/Line | Cross-Dataset Comparison | DEAP vs SAM-40 trends | Results/Discussion | ⚠️ Optional | ⚠️ Need |
| Fig. 13 | Bar | Ablation Study | With/without components | Supplementary | ⚠️ Optional | ⚠️ Need |
| Fig. 14 | Error | Failure Case Analysis | Low-performance subjects | Supplementary | ⚠️ Optional | ⚠️ Need |

---

## FIGURE CAPTIONS (CAMERA-READY)

### Fig. 1 — System Architecture
**Caption:**
> Overview of the proposed EEG-based stress detection framework. Raw EEG signals are preprocessed, segmented into fixed-length windows, and transformed into feature representations for subject-independent classification. Performance evaluation, statistical validation, and interpretability analyses are conducted on the classifier outputs.

### Fig. 2 — LOSO Validation Workflow
**Caption:**
> Illustration of the leave-one-subject-out (LOSO) validation protocol. For each fold, data from one subject are held out for testing, while all preprocessing, feature selection, and model training steps are performed exclusively on the remaining subjects to prevent data leakage.

### Fig. 3 — Dataset & Labeling Pipeline
**Caption:**
> Dataset-specific labeling workflow. Raw EEG data and task or rating information are combined with subject-wise normalization to derive stress and non-stress labels before window-level assignment.

### Fig. 4 — Feature Extraction Pipeline
**Caption:**
> Feature extraction process illustrating the computation of band-power, time–frequency, and covariance-based representations from windowed EEG segments, resulting in a unified feature vector for classification.

### Fig. 5 — RAG-LLM Explanation Architecture
**Caption:**
> Retrieval-augmented generation (RAG) framework used for explanation and reasoning. The EEG classifier provides predictions and symbolic summaries, which are grounded via retrieval from a domain-specific knowledge base before generating structured explanations. The RAG module does not influence classification decisions.

### Fig. 6 — Subject-Wise LOSO Performance
**Caption:**
> Distribution of subject-wise classification performance under LOSO validation. Boxplots report median accuracy, interquartile range, and outliers, highlighting inter-subject variability and generalization robustness.

### Fig. 7 — Confusion Matrix
**Caption:**
> Confusion matrix summarizing stress and non-stress classification outcomes aggregated across LOSO folds, illustrating class-specific error patterns.

### Fig. 8 — ROC Curve
**Caption:**
> Receiver operating characteristic (ROC) curve for binary stress classification, demonstrating discriminative performance independent of decision threshold.

### Fig. 9 — Time–Frequency Representation
**Caption:**
> Time–frequency representations comparing stress and non-stress conditions, revealing task-related spectral dynamics associated with stress.

### Fig. 10 — Scalp Topography
**Caption:**
> Scalp topographies of key frequency bands illustrating spatial distributions of EEG activity associated with stress.

### Fig. 11 — Feature Importance Heatmap
**Caption:**
> Channel-by-frequency importance heatmap highlighting EEG features that contribute most to stress classification, supporting interpretability of the proposed model.

### Fig. 12 — Cross-Dataset Performance
**Caption:**
> Comparison of classification performance trends across datasets, demonstrating consistency of the proposed approach under different stress paradigms.

### Fig. 13 — Ablation Study
**Caption:**
> Ablation analysis evaluating the contribution of individual components of the proposed framework to overall performance.

### Fig. 14 — Failure Case Analysis
**Caption:**
> Analysis of misclassified subjects or conditions, highlighting limitations and sources of performance degradation.

---

## ARCHITECTURE DIAGRAMS (ASCII SPECIFICATIONS)

### Fig. 1 — System Architecture

```
EEG Acquisition
  (SAM-40 / DEAP)
        |
        v
Preprocessing
(Band-pass, Notch, ICA/ASR)
        |
        v
Windowing
(L sec, overlap)
        |
        v
Feature Extraction
(Bandpower / TF / Riemannian)
        |
        v
Classifier
(LDA / SVM / Proposed)
        |
        +--------------------+
        |                    |
        v                    v
Prediction Output      Explainability
(Stress / Non-stress)  (Channel×Band)
        |
        v
Statistics & Validation
(LOSO, CI, Wilcoxon)
```

### Fig. 2 — LOSO Validation Flow

```
For each subject s:
  Test Set  ← Subject s
  Train Set ← All subjects except s
      |
      v
 Train-only operations
 (Scaling, FS, Hyperparams)
      |
      v
  Train Model
      |
      v
  Test on Subject s
      |
      v
  Store Subject-wise Metrics
```

### Fig. 3 — Labeling Flow

```
Raw EEG + Task Info
        |
        v
Rating / Task Rules
(Stress / Workload)
        |
        v
Per-Subject Normalization
        |
        v
Window Label Assignment
(Stress / Non-stress)
        |
        v
Final Labeled Segments
```

### Fig. 4 — Feature Extraction

```
Windowed EEG
     |
     +--> Bandpower (θ, α, β)
     |
     +--> Time–Frequency (Wavelet)
     |
     +--> Covariance → Riemannian
                 |
                 v
          Feature Vector
```

### Fig. 5 — RAG Architecture

```
EEG Classifier Output
(Stress / Confidence)
        |
        v
Symbolic Summary
(Top channels, bands)
        |
        v
Retriever
(EEG stress KB)
        |
        v
LLM Reasoning
(Constrained, grounded)
        |
        v
Structured Explanation
(No effect on prediction)
```

---

## FIGURE COUNT RECOMMENDATIONS

| Scenario | Figures in Main | Move to Supplement |
|----------|-----------------|-------------------|
| Page-limited | 6–7 | TF maps, ablation |
| Top-conference safe | 9–10 | Failure cases |
| With RAG | 10–11 | Prompt ablations |

---

## REVIEWER WARNINGS IF MISSING

| Missing Figure | Reviewer Comment |
|----------------|------------------|
| No Fig. 1 | "Method unclear" |
| No Fig. 2 | "Possible data leakage" |
| No Fig. 6 | "Weak generalization" |
| No Fig. 9-10 | "No neurophysiological grounding" |
| No Fig. 5 (with RAG) | "LLM role unclear" |

---

*Version: 3.0.0*
*Last Updated: 2025-12-25*
