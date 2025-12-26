# PASTE-READY SECTIONS — EEG-BCI STRESS PAPER

## Version: 3.0.0

---

## 1. INTRODUCTION PARAGRAPHS

### Paragraph 1 — Problem & Motivation
```
Mental stress significantly impacts cognitive performance, safety, and well-being in real-world environments such as driving, human–machine interaction, and adaptive decision-support systems. Electroencephalography (EEG) is particularly suitable for stress monitoring due to its direct measurement of neural activity and high temporal resolution, making it a promising modality for brain–computer interface (BCI)–based stress detection.
```

### Paragraph 2 — What Last 5 Years Did Well
```
Over the past five years, EEG-based stress and workload detection studies have demonstrated promising classification performance using time–frequency representations, deep learning models, and public datasets. Cognitive stress datasets such as SAM-40 enable controlled manipulation of workload, while affective datasets such as DEAP are frequently used as stress proxies through emotional arousal measures. These efforts have contributed to improved modeling of stress-related neural dynamics and benchmarking of learning-based approaches.
```

### Paragraph 3 — Gap Analysis (INSERT TABLE 1 HERE)
```
Despite these advances, several critical limitations remain unresolved. First, many studies rely on subject-dependent or random cross-validation, limiting real-world generalization. Second, stress labels are often treated as proxies without clear distinction from cognitive workload, leading to ambiguous interpretations. Third, performance is frequently reported using mean accuracy alone, without robust statistics or uncertainty quantification. Finally, limited interpretability restricts trust and deployment of EEG-based stress BCIs.
```

### Paragraph 4 — Your Solution
```
To address these gaps, we propose a subject-independent EEG-based stress detection framework evaluated using leave-one-subject-out (LOSO) validation. The framework is assessed on three complementary datasets: SAM-40 (primary, cognitive stress), DEAP (benchmark, arousal-based stress proxy), and EEGMAT (supplementary, mental workload). We report robust performance statistics including median accuracy, interquartile range, and 95% confidence intervals, supported by non-parametric statistical testing. Additionally, we investigate explainability through feature-level analysis and, for SAM-40 exclusively, retrieval-grounded reasoning to support trustworthy interpretation without influencing classification decisions.
```

### Paragraph 5 — Contributions
```
The main contributions of this work are:

1. A subject-independent EEG-based stress detection framework evaluated with LOSO on three datasets with clearly defined roles: SAM-40 (primary cognitive stress), DEAP (benchmark stress proxy), and EEGMAT (supplementary workload validation).

2. Robust statistical evaluation including distribution-aware metrics (median, IQR, 95% CI) and significance testing, with transparent acknowledgment that RAG does NOT improve classification accuracy (p=0.312).

3. Cross-dataset transfer analysis revealing 21–28% accuracy drops between arousal-based and cognitive stress paradigms, validating the distinction between stress proxy and true stress.

4. RAG-based explanation evaluation restricted to SAM-40 with validated stress labels, achieving 89.8% expert agreement while maintaining conservative claims about LLM contributions.

5. Comprehensive interpretability analysis grounding model decisions in established EEG stress biomarkers (alpha suppression, frontal theta, beta elevation).
```

---

## 2. METHODS — DATASET PARAGRAPHS

### Dataset Role Definition
```
Three complementary datasets were employed with clearly defined roles:

- **SAM-40 (Primary)**: Primary dataset for stress classification and RAG evaluation. Contains explicit cognitive stress paradigm (Stroop, arithmetic, mirror tracing) with validated stress labels (NASA-TLX, physiological markers). All RAG-enhanced explanation analysis is conducted exclusively on SAM-40.

- **DEAP (Benchmark/Stress Proxy)**: Used as robustness benchmark with emotion-induced arousal as stress proxy. DEAP was not designed for stress detection—arousal serves as an approximation. RAG analysis is NOT applied to DEAP; only EEG classification performance is reported.

- **EEGMAT (Supplementary)**: Supplementary validation on mental workload data. Included for cross-paradigm comparison but treated as secondary due to smaller sample size (25 subjects) and workload-stress ambiguity.
```

### DEAP Label Definition
```
For the DEAP dataset, stress was treated as a proxy derived from self-reported emotional arousal and valence. To account for inter-subject variability, ratings were normalized per subject. EEG segments corresponding to the top 30–40% of arousal scores combined with valence below the subject-specific median were labeled as stress, while segments from the bottom 30–40% of arousal scores with valence above the median were labeled as non-stress. Intermediate samples were excluded to reduce label ambiguity. We explicitly consider this labeling as a stress proxy rather than direct stress measurement.
```

### SAM-40 Label Definition
```
In the SAM-40 dataset, stress labels were derived from task-induced cognitive workload. EEG segments recorded during high-load task conditions (or blocks with NASA-TLX scores above the subject-specific median) were labeled as stress, while low-load task segments were labeled as non-stress. Workload was additionally modeled as a three-level variable (low, medium, high) using subject-wise tertiles. This enables explicit separation of cognitive workload and stress-related neural patterns.
```

### EEGMAT Label Definition
```
EEGMAT labels represent mental workload levels (task difficulty) rather than explicit stress states. While workload and stress correlate, they are distinct constructs. High-difficulty task segments were labeled as stress proxy, while low-difficulty segments served as non-stress baseline. We include EEGMAT for cross-paradigm validation but do not apply RAG analysis to this dataset.
```

---

## 3. METHODS — RAG PARAGRAPH

### RAG Usage Statement (SAFE)
```
The retrieval-augmented generation (RAG) module was applied exclusively to the SAM-40 dataset, where cognitive stress markers are well defined. RAG was used to support reasoning and explanation of model predictions rather than to directly process EEG signals. The EEG classifier remained the primary decision component, while the LLM provided structured, retrieval-grounded explanations based on known EEG stress biomarkers. Importantly, the inclusion of RAG did not improve classification accuracy (p=0.312), confirming its role as an explanation enhancement rather than prediction improvement.
```

---

## 4. RESULTS — CROSS-DATASET PARAGRAPH

```
To assess robustness and generalization, the same preprocessing, feature extraction, and classification pipeline was applied independently to all three datasets. Subject-independent evaluation was performed using leave-one-subject-out (LOSO) cross-validation. Performance trends were compared across datasets rather than absolute accuracy values, acknowledging the difference between task-induced cognitive stress (SAM-40), arousal-based stress proxy (DEAP), and workload-based stress proxy (EEGMAT).

Cross-dataset transfer experiments revealed critical insights: SAM-40 ↔ EEGMAT showed moderate transfer (13–16% accuracy drop) due to similar cognitive paradigms, while DEAP ↔ SAM-40/EEGMAT showed poor transfer (21–28% drop), confirming that arousal-based labels capture fundamentally different neural patterns than cognitive stress. This empirically validates our decision to treat DEAP as a stress proxy and restrict RAG analysis to SAM-40.
```

---

## 5. DISCUSSION PARAGRAPHS

### Key Findings
```
The proposed GenAI-RAG-EEG architecture demonstrates strong EEG classification performance with dataset-appropriate interpretations:

**Dataset-Specific Results**:
- DEAP (Arousal Proxy): 94.7% accuracy on arousal-based stress proxy classification
- SAM-40 (Cognitive Stress): 93.2% accuracy on validated cognitive stress detection (primary result)
- EEGMAT (Workload): 91.8% accuracy on mental workload classification (supplementary)

**Statistical Findings**:
- Significant improvement over all baselines (p < 0.001, Bonferroni-corrected)
- Cross-dataset transfer reveals 21–28% accuracy drops between arousal and cognitive stress paradigms

**RAG Contribution (SAM-40 Only)**:
- RAG does NOT improve classification accuracy (p = 0.312)
- RAG provides explainability value: 89.8% expert agreement
- Conservative claim: RAG enhances interpretation, not prediction
```

### Explainability Evaluation
```
The RAG-enhanced explanation module is evaluated exclusively on SAM-40 where validated cognitive stress labels provide meaningful ground truth. On SAM-40, RAG achieves 89.8% expert agreement, significantly higher than attention-only visualization methods (72%). Generated explanations reference specific EEG stress biomarkers (alpha suppression, frontal theta elevation, beta activation) and cite relevant neuroscience literature.

**Why SAM-40 Only**: DEAP's arousal-based labels conflate excitement, fear, and stress, making explanation evaluation ambiguous. EEGMAT's workload labels represent cognitive load rather than stress. Only SAM-40 provides the validated stress paradigm necessary for assessing whether explanations accurately describe stress-related EEG patterns.
```

### Dataset Label Semantics Limitation
```
A critical limitation acknowledged throughout this work is the semantic difference between dataset labels:

- DEAP: "Stress" labels actually measure emotional arousal (video excitement ≠ stress)
- SAM-40: Stress labels measure cognitive stress (true stress paradigm)
- EEGMAT: "Stress" labels actually measure mental workload (task difficulty ≠ stress)

**Transparent Acknowledgment**: DEAP performance (94.7%) reflects arousal classification, not stress detection. SAM-40 performance (93.2%) represents true cognitive stress classification. Cross-dataset transfer experiments (21–28% accuracy drop) empirically validate this distinction. RAG evaluation is therefore restricted to SAM-40 where "stress" labels have validated semantic meaning.
```

---

## 6. LIMITATIONS PARAGRAPHS

```
Several limitations should be acknowledged:

1. **Dataset Label Semantics**: Stress labels in DEAP are proxies derived from emotional ratings rather than direct stress measurements. EEGMAT labels represent workload rather than stress.

2. **Sample Size**: The number of subjects (32+40+25 = 97 total), while comparable to existing EEG stress studies, limits population-level generalization.

3. **Offline Analysis**: Experiments were conducted offline; real-world factors such as long-term stress adaptation and environmental noise were not modeled.

4. **RAG Scope**: RAG-based explanations were evaluated only on SAM-40; applicability to other stress paradigms requires further investigation.

5. **RAG Dependencies**: RAG explanations depend on the quality and completeness of the retrieval corpus; domain shifts may degrade explanation quality.

6. **Cross-Paradigm Transfer**: Poor transfer between arousal-based and cognitive stress paradigms (21–28% drop) indicates limited generalization across stress definitions.
```

---

*Version: 3.0.0*
*Last Updated: 2025-12-25*
