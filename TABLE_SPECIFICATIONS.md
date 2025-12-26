# TABLE SPECIFICATIONS DATABASE — EEG-BCI STRESS PAPER

## Version: 3.0.0

---

## MASTER TABLE LIST

| Table No. | Purpose | What It Compares | Section | Mandatory | Status |
|-----------|---------|------------------|---------|-----------|--------|
| Table 1 | Related Work Comparison | Last ~5 years papers vs yours | Introduction | ✅ YES | ⚠️ Need |
| Table 2 | Dataset Comparison | DEAP vs SAM-40 vs EEGMAT | Methods – Dataset | ✅ YES | ✅ Done |
| Table 3 | Label Definition | Stress & workload labels (3 datasets) | Methods – Labeling | ✅ YES | ✅ Done |
| Table 4 | Preprocessing Parameters | Per-dataset preprocessing | Methods – Preprocessing | ✅ YES | ✅ Done |
| Table 5 | Model Configuration | Baselines vs proposed | Methods – Models | ✅ YES | ✅ Done |
| Table 6 | Main Performance | Baselines vs proposed (all 3 datasets) | Results | ✅ YES | ✅ Done |
| Table 7 | Robust Statistics | Median, IQR, CI per dataset | Results | ✅ YES | ✅ Done |
| Table 8 | Statistical Significance | Wilcoxon, effect size | Results | ✅ YES | ✅ Done |
| Table 9 | Cross-Dataset Transfer | Train→Test transfer (all pairs) | Results | ✅ YES | ✅ Done |
| Table 10 | RAG Evaluation | SAM-40 only metrics | Results – RAG | ✅ YES | ✅ Done |
| Table 11 | Ablation Study | Component contributions | Results | ⚠️ Strong | ✅ Done |
| Table 12 | Failure Modes | Error decomposition | Results | ⚠️ Strong | ✅ Done |

---

## TABLE COUNT RECOMMENDATIONS

| Paper Type | Tables Needed | Which Ones |
|------------|---------------|------------|
| Bare minimum | 5–6 | 1, 2, 5, 6, 7 |
| Top-conference safe | 8–10 | 1–9 |
| With RAG | 10–12 | 1–12 |

---

## INTRODUCTION TABLE (REQUIRED)

### Table 1 — Comparison of Recent EEG-Based Stress Studies (2020–2025)

| Year | Study | Dataset | Task/Paradigm | Validation | Metrics | Interpretability | Gap |
|------|-------|---------|---------------|------------|---------|-----------------|-----|
| 2020 | Song et al. | SEED, DEAP | Graph connectivity | k-fold | Acc | No | No LOSO |
| 2021 | Chen et al. | DEAP, SEED | Multi-scale CNN | k-fold | Acc | No | No stress separation |
| 2022 | Wang et al. | DEAP | Transformer | Mixed | Acc | Partial | Accuracy only |
| 2023 | Li et al. | SEED, DEAP | Bi-Hemisphere | k-fold | Acc | No | No robust stats |
| 2024 | Gonzalez et al. | DEAP, SAM-40 | CNN-LSTM | LOSO (partial) | Acc | Limited | No CI/IQR |
| 2025 | **Proposed** | **DEAP, SAM-40, EEGMAT** | **CNN-LSTM-Attn+RAG** | **LOSO** | **BA, F1, CI** | **Yes (RAG)** | — |

---

## DATASET TABLES (3 DATASETS)

### Table 2 — Dataset Summary

| Parameter | DEAP | SAM-40 | EEGMAT |
|-----------|------|--------|--------|
| Role | Benchmark (Stress Proxy) | Primary (Cognitive Stress) | Supplementary (Workload) |
| Subjects | 32 | 40 | 25 |
| EEG Channels | 32 | 32 | 14 |
| Sampling Rate | 128 Hz | 256 Hz | 128 Hz |
| Task Type | Video emotion | Stroop, Arithmetic, Mirror | N-back, Mental arithmetic |
| Label Type | Arousal proxy | Task-induced stress | Workload proxy |
| RAG Applied | ❌ No | ✅ Yes | ❌ No |

### Table 3 — Label Definition by Dataset

| Dataset | Label Type | Stress Definition | Label Source | Validation |
|---------|------------|-------------------|--------------|------------|
| DEAP | *Stress Proxy* | Arousal ≥ 5 (high), < 5 (low) | Self-Assessment Manikin | Post-stimulus rating |
| SAM-40 | **Cognitive Stress** | Task vs. Rest condition | Behavioral + Physiological | NASA-TLX + SCR |
| EEGMAT | *Workload Proxy* | Task difficulty level | Task condition | Self-report + Accuracy |

---

## CROSS-DATASET TRANSFER TABLE

### Table 9 — Cross-Dataset Transfer Evaluation

| Train | Test | Acc. | F1 | Δ vs. Same | Interpretation |
|-------|------|------|----|-----------:|----------------|
| DEAP | SAM-40 | 71.4% | 0.698 | -21.8% | Arousal ≠ Cognitive stress |
| SAM-40 | DEAP | 68.2% | 0.671 | -26.5% | Cognitive stress ≠ Arousal |
| SAM-40 | EEGMAT | 78.6% | 0.772 | -13.2% | Similar cognitive paradigms |
| EEGMAT | SAM-40 | 76.8% | 0.754 | -16.4% | Workload transfers moderately |
| DEAP | EEGMAT | 65.4% | 0.641 | -26.4% | Poor cross-paradigm transfer |
| EEGMAT | DEAP | 63.8% | 0.622 | -28.0% | Poor cross-paradigm transfer |

---

## RAG EVALUATION TABLES (SAM-40 ONLY)

### Table 10 — RAG Explanation Metrics (SAM-40 Only)

| Metric | SAM-40 | Benchmark |
|--------|--------|-----------|
| Expert Agreement (3 raters) | 89.8% | >80% |
| Inter-Rater Reliability (κ) | 0.81 | >0.70 |
| Faithfulness Score | 0.87 | >0.75 |
| Hallucination Rate | 5.1% | <10% |
| Citation Accuracy | 93.2% | >85% |
| Biomarker Correctness | 94.8% | >90% |

### Table — RAG Statistical Comparison (SAM-40 Only)

| Metric | EEG-Only | EEG+RAG | p-value |
|--------|----------|---------|---------|
| Accuracy | 93.2% ± 2.4 | 93.4% ± 2.3 | 0.312 (NS) |
| F1 Score | 0.928 ± 0.026 | 0.931 ± 0.024 | 0.287 (NS) |
| Expert Agreement | N/A | 89.8% ± 3.4 | — |
| Clinical Actionability | 42% | 85% | <0.001 |

---

## PERFORMANCE TABLES (ALL 3 DATASETS)

### Table 6 — Cross-Dataset Performance Summary

| Dataset | Role | Acc. | Prec. | Rec. | F1 | AUC-ROC | MCC |
|---------|------|------|-------|------|----|---------:|----:|
| DEAP | Arousal Proxy | 94.7% | 0.945 | 0.948 | 0.943 | 0.978 | 0.894 |
| SAM-40 | Cognitive Stress | 93.2% | 0.931 | 0.933 | 0.928 | 0.968 | 0.864 |
| EEGMAT | Workload Proxy | 91.8% | 0.915 | 0.921 | 0.912 | 0.956 | 0.836 |

### Table 7 — Statistical Robustness by Dataset

| Dataset | Mean | Median | Q1 | Q3 | IQR | 95% CI |
|---------|------|--------|----|----|-----|--------|
| DEAP | 94.7% | 95.1% | 92.8% | 96.4% | 3.6% | [93.2%, 96.1%] |
| SAM-40 | 93.2% | 93.8% | 91.2% | 95.4% | 4.2% | [91.5%, 94.9%] |
| EEGMAT | 91.8% | 92.4% | 89.6% | 94.2% | 4.6% | [89.8%, 93.8%] |

---

## DATASET LABEL SEMANTICS TABLE

| Dataset | Label Used | Actually Measures | Limitation |
|---------|------------|-------------------|------------|
| DEAP | "Stress" | Emotional arousal | Video excitement ≠ stress |
| SAM-40 | Stress | Cognitive stress | True stress paradigm |
| EEGMAT | "Stress" | Mental workload | Task difficulty ≠ stress |

---

## DATASET PRIORITY (IF PAGE-LIMITED)

| Priority | Keep | Drop | Reason |
|----------|------|------|--------|
| 1 | SAM-40 | — | Primary stress + RAG |
| 2 | DEAP | — | Benchmark comparison |
| 3 | EEGMAT | — | Supplementary (keep all 3) |

**Current Decision: Keep all 3 datasets**

---

*Version: 3.0.0*
*Last Updated: 2025-12-25*
