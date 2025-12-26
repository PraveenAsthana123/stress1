# CAMERA-READY SUBMISSION CHECKLIST — EEG-BCI STRESS PAPER

## Final Pre-Submission Audit (v3.0.0)

---

## A. CORE SCIENTIFIC VALIDITY

| Item | Check | Why Reviewers Care | Status |
|------|-------|-------------------|--------|
| Stress definition is explicit | ⬜ | Avoids stress/workload confusion | ✅ Done |
| Stress proxy clearly stated (DEAP) | ⬜ | Prevents overclaiming | ✅ Done |
| Workload modeled separately | ⬜ | Cognitive validity | ✅ Done |
| Subject-independent evaluation (LOSO) | ⬜ | Real-world generalization | ✅ Done |
| No data leakage (train-only preprocessing) | ⬜ | #1 EEG rejection reason | ✅ Done |
| At least 2 strong baselines | ⬜ | Fair comparison | ⚠️ Verify |
| Same pipeline across datasets | ⬜ | Scientific fairness | ✅ Done |

---

## B. RESULTS & STATISTICS

| Item | Check | Why Reviewers Care | Status |
|------|-------|-------------------|--------|
| Balanced Accuracy + F1 (not accuracy only) | ⬜ | Class imbalance | ✅ Done |
| Median + Q1/Q3 + IQR | ⬜ | EEG is non-Gaussian | ✅ Done |
| 95% confidence interval | ⬜ | Reliability | ✅ Done |
| Subject-wise performance plot | ⬜ | Generalization | ✅ Done |
| Wilcoxon signed-rank test | ⬜ | Proper EEG stats | ✅ Done |
| Effect size reported | ⬜ | Practical significance | ✅ Done |
| FDR / correction (if multiple tests) | ⬜ | Statistical rigor | ✅ Done |

---

## C. NEUROPHYSIOLOGICAL EVIDENCE

| Item | Check | Why Reviewers Care | Status |
|------|-------|-------------------|--------|
| Time–frequency analysis | ⬜ | Stress dynamics | ✅ Done |
| Alpha suppression / frontal theta discussed | ⬜ | Cognitive grounding | ✅ Done |
| Scalp topography shown | ⬜ | Spatial interpretation | ✅ Done |
| Results align with literature | ⬜ | Not noise-driven | ✅ Done |

---

## D. FIGURES & ARCHITECTURE

| Figure | Description | Present? | Mandatory? | Status |
|--------|-------------|----------|------------|--------|
| Fig. 1 | System architecture diagram | ⬜ | ✅ Yes | ⚠️ Need |
| Fig. 2 | LOSO flowchart (no leakage) | ⬜ | ✅ Yes | ⚠️ Need |
| Fig. 3 | Dataset & labeling flow | ⬜ | ✅ Yes | ⚠️ Need |
| Fig. 4 | Feature extraction flow | ⬜ | ✅ Yes | ⚠️ Need |
| Fig. 5 | Subject-wise boxplot | ⬜ | ✅ Yes | ⚠️ Need |
| Fig. 6 | Confusion matrix | ⬜ | ✅ Yes | ⚠️ Need |
| Fig. 7 | ROC curve | ⬜ | ⚠️ Strong | ⚠️ Need |
| Fig. 8 | TF map + topography | ⬜ | ⚠️ Strong | ⚠️ Need |
| Fig. 9 | Explainability heatmap | ⬜ | ⚠️ Strong | ⚠️ Need |
| Fig. 10 | RAG architecture | ⬜ | ⚠️ Required if RAG | ⚠️ Need |

---

## E. TABLES (COMPARISON & RIGOR)

| Table | Description | Check | Section | Status |
|-------|-------------|-------|---------|--------|
| Last-5-years comparison | ⬜ | Introduction | ⚠️ Need |
| Dataset summary | ⬜ | Methods | ✅ Done |
| Label definition | ⬜ | Methods | ✅ Done |
| Model configuration | ⬜ | Methods | ✅ Done |
| Baseline comparison | ⬜ | Results | ✅ Done |
| Robust statistics | ⬜ | Results | ✅ Done |
| Statistical significance | ⬜ | Results | ✅ Done |
| Cross-dataset comparison | ⬜ | Results/Discussion | ✅ Done |

---

## F. RAG / LLM (ONLY IF USED)

| Item | Check | Why Reviewers Care | Status |
|------|-------|-------------------|--------|
| LLM does NOT see raw EEG | ⬜ | Prevents misuse | ✅ Done |
| LLM role = explanation only | ⬜ | Avoids hype | ✅ Done |
| EEG accuracy unchanged with RAG | ⬜ | Safety | ✅ Done |
| RAG evaluated separately | ⬜ | Scientific value | ✅ Done |
| Failure cases discussed | ⬜ | Trust | ✅ Done |
| Cost / latency mentioned | ⬜ | Practicality | ✅ Done |

---

## G. LIMITATIONS & ETHICS

| Item | Check | Why Reviewers Care | Status |
|------|-------|-------------------|--------|
| DEAP stress proxy limitation stated | ⬜ | Honesty | ✅ Done |
| Offline analysis limitation | ⬜ | Realism | ✅ Done |
| Inter-subject variability discussed | ⬜ | Transparency | ✅ Done |
| Stress misclassification risk mentioned | ⬜ | Ethics | ✅ Done |
| Mental-state privacy acknowledged | ⬜ | Modern review focus | ✅ Done |

---

## H. WRITING & PRESENTATION

| Item | Check | Status |
|------|-------|--------|
| Claims match results (no overclaiming) | ⬜ | ✅ Done |
| Figures referenced in correct order | ⬜ | ⚠️ Verify |
| Captions are self-contained | ⬜ | ⚠️ Verify |
| Contributions ≤ 5, very clear | ⬜ | ✅ Done |
| Page limit strictly respected | ⬜ | ⚠️ Verify |
| Supplementary properly referenced | ⬜ | ⚠️ Verify |

---

## FINAL GO / NO-GO DECISION

| Status | Meaning |
|--------|---------|
| ✅ All critical items checked | SUBMIT |
| ⚠️ Minor gaps (figures, latency) | Fix before submit |
| ❌ Missing LOSO / stats / architecture | Do NOT submit yet |

---

## REJECTION RISK AUDIT

| Area | Reviewer Question | Status | Action If Weak |
|------|------------------|--------|----------------|
| Problem Definition | Is stress clearly defined? | ✅ | — |
| Dataset Choice | Are datasets justified? | ✅ | — |
| Label Validity | Are stress labels defensible? | ✅ | — |
| Evaluation Protocol | Is LOSO used? | ✅ | — |
| Data Leakage | Any chance of leakage? | ✅ | — |
| Baselines | Are ≥2 strong baselines included? | ⚠️ | Verify LDA, SVM |
| Metrics | Beyond accuracy reported? | ✅ | — |
| Robust Statistics | Median, IQR, CI reported? | ✅ | — |
| Statistical Testing | Improvements tested? | ✅ | — |
| Multiple Testing | FDR applied? | ✅ | — |
| Subject Variability | Inter-subject shown? | ✅ | — |
| Neuro Evidence | EEG stress markers shown? | ✅ | — |
| Explainability | Model explained? | ✅ | — |
| RAG Role | LLM constrained? | ✅ | — |
| RAG Evaluation | RAG evaluated separately? | ✅ | — |
| Cross-Dataset | Generalization discussed? | ✅ | — |
| Negative Results | Limitations discussed? | ✅ | — |
| Real-World BCI | Latency discussed? | ✅ | — |
| Ethics | Risks acknowledged? | ✅ | — |
| Novelty | Clear vs last 5 years? | ✅ | — |
| Writing Tone | Conservative, not hype? | ✅ | — |

---

## TOP 5 LAST-MINUTE FIXES

1. ⬜ Confirm ≥2 strong baselines (Bandpower+LDA, Bandpower+SVM)
2. ⬜ Add figures (architecture, LOSO flow, subject-wise boxplot)
3. ⬜ Verify ethical considerations sentence
4. ⬜ RAG: expert agreement rate included (89.8%)
5. ⬜ Double-check figures match captions

---

## COMMON REJECTION TRIGGERS (AVOIDED)

| Trigger | Status |
|---------|--------|
| ❌ Claiming DEAP is "stress dataset" | ✅ Avoided - marked as proxy |
| ❌ Accuracy-only claims | ✅ Avoided - BA, F1, CI included |
| ❌ Vague LLM contribution | ✅ Avoided - SAM-40 only, explanation role |
| ❌ No subject-wise plots | ⚠️ Need figure |
| ❌ Missing architecture diagram | ⚠️ Need figure |

---

## HONEST FINAL VERDICT

| Configuration | Rating | Notes |
|---------------|--------|-------|
| EEG-only version | ⭐⭐⭐⭐☆ | Strong Accept potential |
| EEG + carefully scoped RAG | ⭐⭐⭐⭐☆ | Accept / Weak Accept |
| EEG + poorly evaluated RAG | ⭐⭐☆☆☆ | Higher reject risk |

**Your EEG core is strong. RAG is optional, not necessary for acceptance.**

---

*Checklist Version: 3.0.0*
*Last Updated: 2025-12-25*
