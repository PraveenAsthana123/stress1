# EEG-Stress-RAG Paper Instructions Database

## Version: 3.0.0 (Camera-Ready)

---

## QUICK REFERENCE FILES

| File | Purpose |
|------|---------|
| `CAMERA_READY_CHECKLIST.md` | Final pre-submission audit |
| `FIGURE_SPECIFICATIONS.md` | All figures with captions |
| `TABLE_SPECIFICATIONS.md` | All tables with specs |
| `PASTE_READY_SECTIONS.md` | Copy-paste text |
| `VERSION_HISTORY.md` | Version tracking |
| `CHANGELOG_DETAILED.md` | Detailed changes |

---

## Table 1: Dataset Role Definitions

| Dataset | Role | Label Type | Stress Definition | RAG Applied | Primary Use |
|---------|------|------------|-------------------|-------------|-------------|
| SAM-40 | Primary | Cognitive Stress | Task vs. Rest condition | YES | Stress classification + RAG evaluation |
| DEAP | Benchmark | Stress Proxy | Arousal ≥ 5 (high), < 5 (low) | NO | EEG classification only |
| EEGMAT | Supplementary | Workload Proxy | Task difficulty level | NO | Cross-paradigm validation |

## Table 2: Dataset-Specific Label Rules

| Dataset | What Label Represents | Ground Truth Source | Validation Method | Limitation |
|---------|----------------------|---------------------|-------------------|------------|
| DEAP | Emotional arousal | Self-Assessment Manikin | Post-stimulus rating (1-9) | Video excitement ≠ stress |
| SAM-40 | Cognitive stress | Behavioral + Physiological | NASA-TLX + SCR | True stress paradigm |
| EEGMAT | Mental workload | Task condition | Self-report + Accuracy | Task difficulty ≠ stress |

## Table 3: RAG Scope Rules

| Rule | Description | Justification |
|------|-------------|---------------|
| RAG Only on SAM-40 | All RAG evaluation restricted to SAM-40 | Validated stress labels enable meaningful assessment |
| No RAG on DEAP | DEAP reports EEG classification only | Arousal-based proxy labels unsuitable for stress explanations |
| No RAG on EEGMAT | EEGMAT reports EEG classification only | Workload focus inappropriate for stress-specific RAG |
| Conservative Claims | RAG does NOT improve accuracy (p=0.312) | Contribution is explainability, not prediction |

## Table 4: Complete Analysis Checklist - Data & Labels

| Category | Item | Priority | Status |
|----------|------|----------|--------|
| Data & Labels | Stress label definitions per dataset | MANDATORY | Done |
| Data & Labels | Workload vs stress analysis | MANDATORY | Done |
| Data & Labels | Class balance analysis | MANDATORY | Done |
| Data & Labels | Demographics table | MANDATORY | Done |
| Data & Labels | Label validation method | MANDATORY | Done |
| Data & Labels | Ground truth source | MANDATORY | Done |

## Table 5: Complete Analysis Checklist - Preprocessing

| Category | Item | Priority | Status |
|----------|------|----------|--------|
| Preprocessing | Band-pass filtering parameters | MANDATORY | Done |
| Preprocessing | Artifact removal method | MANDATORY | Done |
| Preprocessing | ICA/ASR parameters | MANDATORY | Done |
| Preprocessing | Epoch rejection statistics | MANDATORY | Done |
| Preprocessing | Re-referencing method | MANDATORY | Done |
| Preprocessing | Normalization approach | MANDATORY | Done |

## Table 6: Complete Analysis Checklist - Signal-Level EEG

| Category | Item | Priority | Status |
|----------|------|----------|--------|
| Signal-Level | Band power analysis (all bands) | MANDATORY | Done |
| Signal-Level | Alpha suppression analysis | MANDATORY | Done |
| Signal-Level | Theta/Beta ratio (TBR) | STRONGLY EXPECTED | Done |
| Signal-Level | Statistical tests per band | MANDATORY | Done |
| Signal-Level | Effect sizes | STRONGLY EXPECTED | Done |

## Table 7: Complete Analysis Checklist - Time-Frequency

| Category | Item | Priority | Status |
|----------|------|----------|--------|
| Time-Frequency | Wavelet analysis | STRONGLY EXPECTED | Done |
| Time-Frequency | Baseline normalization | MANDATORY | Done |
| Time-Frequency | Window length effects | Expected | Done |

## Table 8: Complete Analysis Checklist - Spatial/Topographic

| Category | Item | Priority | Status |
|----------|------|----------|--------|
| Spatial | Frontal asymmetry analysis | STRONGLY EXPECTED | Done |
| Spatial | Channel significance table | MANDATORY | Done |
| Spatial | Regional contributions | Expected | Done |

## Table 9: Complete Analysis Checklist - Feature Engineering

| Category | Item | Priority | Status |
|----------|------|----------|--------|
| Features | Feature importance ranking | MANDATORY | Done |
| Features | Channel × Band importance matrix | STRONGLY EXPECTED | Done |
| Features | Feature selection method | Expected | Done |

## Table 10: Complete Analysis Checklist - Error/Sensitivity

| Category | Item | Priority | Status |
|----------|------|----------|--------|
| Error Analysis | Confusion matrices | MANDATORY | Done |
| Error Analysis | Misclassification analysis | STRONGLY EXPECTED | Done |
| Error Analysis | Noise robustness testing | Expected | Done |
| Error Analysis | Artifact impact analysis | Expected | Done |

## Table 11: Complete Analysis Checklist - BCI Practicality

| Category | Item | Priority | Status |
|----------|------|----------|--------|
| BCI | Window length analysis | MANDATORY | Done |
| BCI | Latency measurements | MANDATORY | Done |
| BCI | Real-time feasibility | STRONGLY EXPECTED | Done |
| BCI | Deployment specifications | Expected | Done |

## Table 12: Complete Analysis Checklist - Statistical Robustness

| Category | Item | Priority | Status |
|----------|------|----------|--------|
| Statistics | Mean, Median, Q1, Q3, IQR | MANDATORY | Done |
| Statistics | 95% Confidence Intervals | MANDATORY | Done |
| Statistics | Standard deviation | MANDATORY | Done |
| Statistics | Normality testing | Expected | Done |
| Statistics | FDR correction | STRONGLY EXPECTED | Done |

## Table 13: Complete Analysis Checklist - Statistical Significance

| Category | Item | Priority | Status |
|----------|------|----------|--------|
| Significance | Wilcoxon signed-rank test | MANDATORY | Done |
| Significance | Paired t-test | MANDATORY | Done |
| Significance | Mann-Whitney U test | Expected | Done |
| Significance | Effect size (Cohen's d) | MANDATORY | Done |
| Significance | Bonferroni correction | STRONGLY EXPECTED | Done |

## Table 14: RAG-Specific Requirements

| Category | Item | Priority | Status |
|----------|------|----------|--------|
| RAG Core | Research question definition | MANDATORY | Done |
| RAG Core | LLM role definition | MANDATORY | Done |
| RAG Core | Prediction vs Reasoning separation | MANDATORY | Done |
| RAG Core | Output schema (JSON fields) | MANDATORY | Done |
| RAG Eval | Explanation quality metrics | MANDATORY | Done |
| RAG Eval | Expert agreement rate | MANDATORY | Done |
| RAG Eval | Faithfulness score | STRONGLY EXPECTED | Done |
| RAG Eval | Hallucination rate | MANDATORY | Done |
| RAG Eval | Citation accuracy | STRONGLY EXPECTED | Done |
| RAG Eval | Biomarker correctness | STRONGLY EXPECTED | Done |

## Table 15: RAG Ablation Requirements

| Category | Item | Priority | Status |
|----------|------|----------|--------|
| Ablation | RAG ablation (with/without) | MANDATORY | Done |
| Ablation | Prompt ablation | STRONGLY EXPECTED | Done |
| Ablation | Knowledge base ablation | STRONGLY EXPECTED | Done |
| Ablation | LLM size ablation | Expected | Done |

## Table 16: RAG Additional Analysis

| Category | Item | Priority | Status |
|----------|------|----------|--------|
| Analysis | Failure mode decomposition | MANDATORY | Done |
| Analysis | Negative results table | STRONGLY EXPECTED | Done |
| Analysis | Non-LLM baseline comparison | MANDATORY | Done |
| Analysis | Confidence calibration (ECE, Brier) | STRONGLY EXPECTED | Done |
| Analysis | Subject-wise RAG benefit | Expected | Done |

## Table 17: RAG Computational & Ethics

| Category | Item | Priority | Status |
|----------|------|----------|--------|
| Compute | Latency breakdown | MANDATORY | Done |
| Compute | Token usage analysis | STRONGLY EXPECTED | Done |
| Compute | Cost analysis | Expected | Done |
| Ethics | Over-reliance risks | MANDATORY | Done |
| Ethics | Privacy protection | MANDATORY | Done |
| Ethics | Mislabeling risks | STRONGLY EXPECTED | Done |

## Table 18: Cross-Dataset Evaluation Requirements

| Requirement | Description | Status |
|-------------|-------------|--------|
| Transfer experiments | Train on Dataset A, Test on Dataset B | Done |
| Accuracy drop analysis | Measure performance degradation | Done |
| Semantic validation | Confirm arousal ≠ cognitive stress | Done |
| Justification for RAG scope | Explain why SAM-40 only | Done |

## Table 19: Reviewer-Safe Framing Checklist

| Section | Requirement | Status |
|---------|-------------|--------|
| Abstract | State dataset roles clearly | Done |
| Abstract | Mention RAG is SAM-40 only | Done |
| Abstract | Include conservative RAG claim (p=0.312) | Done |
| Introduction | Updated contributions with transparency | Done |
| Methodology | Dataset role definitions | Done |
| Methodology | Label type distinctions | Done |
| Results | Cross-dataset transfer table | Done |
| Discussion | Dataset-specific results interpretation | Done |
| Discussion | Dataset Label Semantics limitation | Done |
| Discussion | Explainability (SAM-40) section | Done |

## Table 20: Paper Output Checklist

| Step | Task | Status |
|------|------|--------|
| 1 | Freeze problem scope | Done |
| 2 | Finalize stress label definitions | Done |
| 3 | Lock windowing parameters | Done |
| 4 | Complete preprocessing tables | Done |
| 5 | Run signal-level EEG analysis | Done |
| 6 | Complete time-frequency analysis | Done |
| 7 | Complete spatial analysis | Done |
| 8 | Feature importance analysis | Done |
| 9 | Error/sensitivity analysis | Done |
| 10 | BCI practicality analysis | Done |
| 11 | Statistical robustness tables | Done |
| 12 | Statistical significance tables | Done |
| 13 | RAG evaluation (SAM-40 only) | Done |
| 14 | Cross-dataset evaluation | Done |
| 15 | Limitations section | Done |
| 16 | Ethical considerations | Done |
| 17 | Reproducibility (prompts, config) | Done |
| 18 | Future directions | Done |
| 19 | Reviewer-safe framing | Done |

## Table 21: Dataset Dropping Priority (If Page-Limited)

| Priority | Dataset to Keep | Reason |
|----------|-----------------|--------|
| 1 | SAM-40 | Primary stress + RAG |
| 2 | DEAP | Benchmark comparison |
| 3 | EEGMAT | Supplementary (KEEP ALL 3) |

**Current Decision: Keep all 3 datasets**

## Table 22: Key Claims and Caveats

| Claim | Caveat | Evidence |
|-------|--------|----------|
| 94.7% accuracy | On DEAP arousal proxy, not true stress | Cross-dataset transfer shows 21-28% drop |
| 93.2% accuracy | On SAM-40 cognitive stress (primary result) | Validated stress paradigm |
| RAG improves explainability | Does NOT improve accuracy (p=0.312) | Statistical testing shows NS |
| 89.8% expert agreement | SAM-40 only | DEAP/EEGMAT labels unsuitable for evaluation |
| Cross-dataset generalization | Poor between arousal and cognitive stress | Transfer experiments validate distinction |

---

## Table 23: Camera-Ready Figure Requirements

| Fig. | Type | Title | Mandatory | Status |
|------|------|-------|-----------|--------|
| 1 | Architecture | System Overview | ✅ YES | ⚠️ Need |
| 2 | Flowchart | LOSO Validation | ✅ YES | ⚠️ Need |
| 3 | Flowchart | Dataset & Labeling | ✅ YES | ⚠️ Need |
| 4 | Flow | Feature Extraction | ✅ YES | ⚠️ Need |
| 5 | Architecture | RAG-LLM | ⚠️ If RAG | ⚠️ Need |
| 6 | Boxplot | Subject-wise Performance | ✅ YES | ⚠️ Need |
| 7 | Matrix | Confusion Matrix | ✅ YES | ⚠️ Need |
| 8 | Curve | ROC Curve | ⚠️ Strong | ⚠️ Need |
| 9 | TF Map | Time-Frequency | ⚠️ Strong | ⚠️ Need |
| 10 | Topography | Scalp Distribution | ⚠️ Strong | ⚠️ Need |
| 11 | Heatmap | Feature Importance | ⚠️ Strong | ⚠️ Need |
| 12 | Bar/Line | Cross-Dataset | ⚠️ Optional | ⚠️ Need |

## Table 24: Camera-Ready Table Requirements

| Table | Purpose | Mandatory | Status |
|-------|---------|-----------|--------|
| 1 | Last-5-years comparison | ✅ YES | ⚠️ Need in Intro |
| 2 | Dataset summary | ✅ YES | ✅ Done |
| 3 | Label definition | ✅ YES | ✅ Done |
| 4 | Model configuration | ✅ YES | ✅ Done |
| 5 | Baseline comparison | ✅ YES | ✅ Done |
| 6 | Robust statistics | ✅ YES | ✅ Done |
| 7 | Statistical significance | ✅ YES | ✅ Done |
| 8 | Cross-dataset transfer | ✅ YES | ✅ Done |
| 9 | RAG evaluation | ✅ YES | ✅ Done |

## Table 25: Rejection Risk Audit

| Area | Question | Status |
|------|----------|--------|
| Problem Definition | Stress clearly defined? | ✅ |
| Dataset Choice | Datasets justified? | ✅ |
| Label Validity | Labels defensible? | ✅ |
| Evaluation Protocol | LOSO used? | ✅ |
| Data Leakage | No leakage? | ✅ |
| Baselines | ≥2 strong baselines? | ⚠️ Verify |
| Metrics | Beyond accuracy? | ✅ |
| Robust Statistics | Median, IQR, CI? | ✅ |
| Statistical Testing | Improvements tested? | ✅ |
| RAG Role | LLM constrained? | ✅ |
| RAG Evaluation | Evaluated separately? | ✅ |
| Negative Results | Limitations discussed? | ✅ |

## Table 26: Final Verdict

| Configuration | Rating | Notes |
|---------------|--------|-------|
| EEG-only | ⭐⭐⭐⭐☆ | Strong Accept potential |
| EEG + careful RAG | ⭐⭐⭐⭐☆ | Accept / Weak Accept |
| EEG + poor RAG | ⭐⭐☆☆☆ | Higher reject risk |

---

## Usage Notes

1. **For Reviewers**: All dataset roles are explicitly defined; arousal ≠ cognitive stress is acknowledged
2. **For RAG Evaluation**: Only SAM-40 results should be cited for RAG metrics
3. **For Performance Claims**: Always specify which dataset and its label type
4. **Conservative Claims**: RAG enhances interpretation, NOT prediction
5. **Figures Needed**: 10-12 figures required for camera-ready
6. **All 3 Datasets**: SAM-40 (primary), DEAP (benchmark), EEGMAT (supplementary)

---

*Database Version: 3.0.0*
*Last Updated: 2025-12-25*
*Paper: GenAI-RAG-EEG (Camera-Ready)*
