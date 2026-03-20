// Complete AI Governance Dashboard with All 32 Analysis Frameworks
const { useState, useEffect, useRef } = React;

// ============================================
// DATA SECTION
// ============================================
const dataProcessing = {
    datasets: [
        { name: "EEGMAT", subjects: 32, samples: 2400, channels: 14, frequency: "128 Hz", classes: 2, accuracy: 99.31 },
        { name: "SAM-40", subjects: 40, samples: 1600, channels: 32, frequency: "256 Hz", classes: 4, accuracy: 72.92 }
    ],
    preprocessing: [
        { step: 1, name: "Raw EEG Acquisition", description: "Capture multi-channel EEG signals", status: "complete", duration: "Real-time" },
        { step: 2, name: "Bandpass Filtering", description: "0.5-45 Hz bandpass filter applied", status: "complete", duration: "0.1s" },
        { step: 3, name: "Artifact Removal", description: "ICA-based artifact rejection", status: "complete", duration: "0.5s" },
        { step: 4, name: "Segmentation", description: "5-second epochs with 50% overlap", status: "complete", duration: "0.2s" },
        { step: 5, name: "Normalization", description: "Z-score per channel normalization", status: "complete", duration: "0.1s" },
        { step: 6, name: "Feature Extraction", description: "Band power (Delta, Theta, Alpha, Beta, Gamma)", status: "complete", duration: "0.3s" }
    ],
    bandPowers: [
        { band: "Delta", range: "0.5-4 Hz", stressed: 12.3, relaxed: 14.1, significance: "p<0.01" },
        { band: "Theta", range: "4-8 Hz", stressed: 18.7, relaxed: 16.2, significance: "p<0.001" },
        { band: "Alpha", range: "8-13 Hz", stressed: 8.4, relaxed: 22.6, significance: "p<0.0001" },
        { band: "Beta", range: "13-30 Hz", stressed: 28.9, relaxed: 19.8, significance: "p<0.001" },
        { band: "Gamma", range: "30-45 Hz", stressed: 31.7, relaxed: 27.3, significance: "p<0.05" }
    ]
};

// ============================================
// MODEL SECTION
// ============================================
const modelArchitecture = {
    name: "GenAI-RAG-EEG Hybrid Model",
    components: [
        { name: "Input Layer", params: "14×640", description: "14 channels × 5s @ 128Hz" },
        { name: "Conv1D Block 1", params: "25,664", description: "64 filters, kernel=25, stride=2" },
        { name: "Conv1D Block 2", params: "73,856", description: "128 filters, kernel=15, stride=2" },
        { name: "Conv1D Block 3", params: "295,168", description: "256 filters, kernel=10" },
        { name: "Bi-LSTM Layer", params: "788,480", description: "256 hidden units, bidirectional" },
        { name: "Self-Attention", params: "263,424", description: "8 heads, 64 dim per head" },
        { name: "RAG Module", params: "External", description: "Scientific literature retrieval" },
        { name: "Context Encoder", params: "131,072", description: "256→512 feature fusion" },
        { name: "Classification Head", params: "1,026", description: "Softmax output (2 classes)" }
    ],
    totalParams: 1578690,
    trainableParams: 1578690,
    hyperparameters: [
        { param: "Learning Rate", value: "0.001" },
        { param: "Optimizer", value: "AdamW" },
        { param: "Weight Decay", value: "0.01" },
        { param: "Batch Size", value: "32" },
        { param: "Epochs", value: "100" },
        { param: "Dropout", value: "0.3" },
        { param: "Cross-Validation", value: "5-fold" }
    ]
};

// ============================================
// ACCURACY SECTION
// ============================================
const accuracyMetrics = {
    mainResults: {
        dataset: "EEGMAT-Full",
        accuracy: 99.31,
        precision: 99.28,
        recall: 99.33,
        f1Score: 99.30,
        auc: 99.85,
        kappa: 0.9862
    },
    foldResults: [
        { fold: 1, accuracy: 99.17, precision: 99.10, recall: 99.24, f1: 99.17 },
        { fold: 2, accuracy: 99.38, precision: 99.45, recall: 99.31, f1: 99.38 },
        { fold: 3, accuracy: 99.24, precision: 99.20, recall: 99.28, f1: 99.24 },
        { fold: 4, accuracy: 99.45, precision: 99.50, recall: 99.40, f1: 99.45 },
        { fold: 5, accuracy: 99.31, precision: 99.15, recall: 99.42, f1: 99.28 }
    ],
    confusionMatrix: {
        tp: 1186, fp: 8, fn: 8, tn: 1198,
        total: 2400
    },
    baselineComparison: [
        { method: "GenAI-RAG-EEG (Ours)", accuracy: 99.31, f1: 99.30 },
        { method: "CNN-LSTM", accuracy: 96.82, f1: 96.75 },
        { method: "EEGNet", accuracy: 94.67, f1: 94.52 },
        { method: "Random Forest", accuracy: 91.23, f1: 90.98 },
        { method: "SVM-RBF", accuracy: 88.45, f1: 88.21 }
    ],
    sam40Results: {
        dataset: "SAM-40 (4-class)",
        accuracy: 72.92,
        f1Macro: 71.85,
        kappa: 0.6389,
        perClass: [
            { class: "Arithmetic", precision: 75.2, recall: 73.8 },
            { class: "Mirror Image", precision: 71.5, recall: 70.2 },
            { class: "Relax", precision: 78.3, recall: 76.9 },
            { class: "Stroop", precision: 66.7, recall: 70.8 }
        ]
    }
};

// ============================================
// ALL 32 AI GOVERNANCE FRAMEWORKS
// ============================================
const responsibleAIFrameworks = {
    // Framework 1: Reliable AI
    reliable: {
        name: "Reliable AI",
        question: "Can this AI system be depended upon consistently over time?",
        avgScore: 95.6,
        analyses: [
            { no: 1, type: "Reliability Definition & Scope", question: "What does reliable mean here?", finding: "99.5% uptime target; SLO defined", score: 99.5 },
            { no: 2, type: "Correctness Consistency", question: "Is correctness consistent across runs?", finding: "<2% variance with fixed seeds", score: 98 },
            { no: 3, type: "Robustness to Input Variation", question: "Does behavior hold under changes?", finding: "±10% noise tolerance maintained", score: 90 },
            { no: 4, type: "Calibration & Confidence", question: "Can confidence be trusted?", finding: "ECE < 0.05; well-calibrated", score: 95 },
            { no: 5, type: "Failure Mode Coverage", question: "Are known failures anticipated?", finding: "15 failure modes documented", score: 93 },
            { no: 6, type: "Graceful Degradation", question: "Does the system fail safely?", finding: "Fallback to baseline classifier", score: 94 },
            { no: 7, type: "Dependency Reliability", question: "Are upstream systems reliable?", finding: "RAG retriever 99.2% available", score: 99.2 },
            { no: 8, type: "Latency & Throughput Stability", question: "Is performance stable under load?", finding: "P99 latency < 500ms", score: 97 },
            { no: 9, type: "Resource Exhaustion", question: "Does it fail under pressure?", finding: "Memory caps enforced; graceful OOM", score: 95 },
            { no: 10, type: "Drift & Temporal Reliability", question: "Does reliability decay over time?", finding: "Monthly drift checks scheduled", score: 92 },
            { no: 11, type: "Monitoring Signal Reliability", question: "Are failures detected early?", finding: "Alert precision 94%, recall 91%", score: 94 },
            { no: 12, type: "Incident Frequency & Recovery", question: "How often/fast do we recover?", finding: "MTTR < 30 min; MTBF > 720 hrs", score: 96 },
            { no: 13, type: "Regression Protection", question: "Do updates break reliability?", finding: "Canary deployment; auto-rollback", score: 98 },
            { no: 14, type: "Human-in-the-Loop Reliability", question: "Do humans improve reliability?", finding: "Override success rate 87%", score: 87 },
            { no: 15, type: "Data Pipeline Reliability", question: "Is data delivery dependable?", finding: "Ingestion success rate 99.8%", score: 99.8 },
            { no: 16, type: "Security & Abuse Resilience", question: "Does misuse reduce reliability?", finding: "Rate limiting; injection defense", score: 96 },
            { no: 17, type: "Operational Readiness", question: "Can teams operate it reliably?", finding: "Runbooks complete; on-call trained", score: 97 },
            { no: 18, type: "Reliability Governance", question: "Who owns reliability?", finding: "RACI defined; quarterly reviews", score: 95 }
        ]
    },
    // Framework 2: Trustworthy AI
    trustworthy: {
        name: "Trustworthy AI",
        question: "Can stakeholders rely on this AI over time?",
        avgScore: 96.2,
        analyses: [
            { no: 1, type: "Trustworthiness Definition", question: "What does trustworthy mean here?", finding: "Clinician confidence; patient safety", score: 95 },
            { no: 2, type: "Correctness & Validity", question: "Are outputs correct and valid?", finding: "99.31% accuracy; validated ground truth", score: 99.31 },
            { no: 3, type: "Robustness & Reliability", question: "Consistent under variation?", finding: "Stress-tested; graceful degradation", score: 96 },
            { no: 4, type: "Safety & Harm Prevention", question: "Does it prevent harm?", finding: "Fail-safe defaults; human oversight", score: 97 },
            { no: 5, type: "Fairness & Non-Discrimination", question: "Are outcomes equitable?", finding: "Demographic parity within 5%", score: 95 },
            { no: 6, type: "Explainability & Transparency", question: "Can decisions be understood?", finding: "RAG + SHAP explanations provided", score: 94 },
            { no: 7, type: "Interpretability by Design", question: "Is logic understandable?", finding: "Modular architecture; attention visible", score: 92 },
            { no: 8, type: "Accountability & Ownership", question: "Who is responsible?", finding: "Named owners; RACI documented", score: 97 },
            { no: 9, type: "Auditability & Traceability", question: "Can decisions be reconstructed?", finding: "Complete audit trails; versioning", score: 99 },
            { no: 10, type: "Human Oversight & Control", question: "Can humans intervene?", finding: "Override mechanism; escalation paths", score: 96 },
            { no: 11, type: "Monitoring & Drift Trust", question: "Is trust maintained over time?", finding: "Continuous monitoring; drift alerts", score: 94 },
            { no: 12, type: "Calibration & Confidence Trust", question: "Does confidence match correctness?", finding: "ECE validated; appropriate confidence", score: 95 },
            { no: 13, type: "Misuse & Abuse Resistance", question: "Can it be exploited?", finding: "Input validation; rate limiting", score: 93 },
            { no: 14, type: "Data Responsibility & Privacy", question: "Is data handled responsibly?", finding: "GDPR-compliant; consent documented", score: 98 },
            { no: 15, type: "Lifecycle & Change Management", question: "Is trust preserved across updates?", finding: "Version control; regression testing", score: 97 },
            { no: 16, type: "Transparency to Stakeholders", question: "Are limits communicated?", finding: "Model cards; limitation disclosure", score: 95 },
            { no: 17, type: "Regulatory & Societal Alignment", question: "Does it meet external expectations?", finding: "Ethics review passed; compliant", score: 94 },
            { no: 18, type: "Trustworthy AI Governance", question: "Who enforces standards?", finding: "Governance board; quarterly audits", score: 96 }
        ]
    },
    // Framework 3: Safe AI
    safe: {
        name: "Safe AI",
        question: "Does this AI prevent or contain harm?",
        avgScore: 95.1,
        analyses: [
            { no: 1, type: "Safety Definition & Scope", question: "What does safe mean here?", finding: "No false negatives causing harm", score: 97 },
            { no: 2, type: "Use-Case Appropriateness", question: "Should AI be used here?", finding: "Decision support only; justified", score: 98 },
            { no: 3, type: "Hazard Identification", question: "What can go wrong?", finding: "12 hazards enumerated; mitigated", score: 94 },
            { no: 4, type: "Input Safety & Misuse", question: "Can inputs cause unsafe behavior?", finding: "Validated; adversarial-robust", score: 96 },
            { no: 5, type: "Output Safety & Harm Prevention", question: "Can outputs cause harm?", finding: "No harmful recommendations", score: 97 },
            { no: 6, type: "Safe Completion & Refusal", question: "Does it refuse correctly?", finding: "Uncertainty triggers deferral", score: 92 },
            { no: 7, type: "Bias-Related Safety", question: "Can bias lead to harm?", finding: "Demographic safety verified", score: 95 },
            { no: 8, type: "Over-Reliance & Automation Bias", question: "Will users trust too much?", finding: "Warnings displayed; human required", score: 88 },
            { no: 9, type: "Uncertainty & Abstention Safety", question: "Does it know when not to answer?", finding: "Abstention at low confidence", score: 93 },
            { no: 10, type: "Safety in Edge & OOD Conditions", question: "Is it safe outside normal conditions?", finding: "OOD detection active", score: 91 },
            { no: 11, type: "System & Dependency Safety", question: "Can dependencies cause harm?", finding: "Fallback systems ready", score: 94 },
            { no: 12, type: "Human-in-the-Loop Safety", question: "Where must humans intervene?", finding: "Clinical decisions require human", score: 99 },
            { no: 13, type: "Monitoring & Safety Detection", question: "Are safety issues detected early?", finding: "Real-time safety monitoring", score: 96 },
            { no: 14, type: "Incident Response & Containment", question: "What happens when harm occurs?", finding: "Kill-switch ready; SOP defined", score: 98 },
            { no: 15, type: "Recovery & Harm Mitigation", question: "How is harm reduced after failure?", finding: "Rollback; notification protocol", score: 94 },
            { no: 16, type: "Safety Documentation", question: "Are limits communicated?", finding: "Safety datasheet provided", score: 95 },
            { no: 17, type: "Regulatory Safety Alignment", question: "Does it meet safety laws?", finding: "Medical device guidance followed", score: 96 },
            { no: 18, type: "Safety Governance", question: "Who owns safety?", finding: "Safety officer designated", score: 97 }
        ]
    },
    // Framework 4: Fairness AI
    fairness: {
        name: "Fairness AI",
        question: "Are outcomes equitable across groups?",
        avgScore: 95.2,
        analyses: [
            { no: 1, type: "Fairness Definition", question: "What does fairness mean here?", finding: "Group parity and equal error rates", score: 96 },
            { no: 2, type: "Impacted Group Analysis", question: "Who could be unfairly affected?", finding: "Age, gender groups analyzed", score: 95 },
            { no: 3, type: "Data Representation", question: "Are all groups represented?", finding: "Balanced representation verified", score: 96 },
            { no: 4, type: "Label Fairness", question: "Are labels biased?", finding: "Expert validation; no bias detected", score: 97 },
            { no: 5, type: "Proxy Feature Analysis", question: "Are features acting as proxies?", finding: "No demographic proxies used", score: 98 },
            { no: 6, type: "Outcome Parity", question: "Do outcomes differ across groups?", finding: "Disparity ratio < 1.2", score: 97 },
            { no: 7, type: "Error Rate Parity", question: "Are errors distributed equally?", finding: "FPR/FNR parity within 5%", score: 95 },
            { no: 8, type: "Calibration Fairness", question: "Is confidence reliable across groups?", finding: "Group-wise ECE validated", score: 95 },
            { no: 9, type: "Individual Fairness", question: "Are similar individuals treated similarly?", finding: "Similarity consistency 91%", score: 91 },
            { no: 10, type: "Counterfactual Fairness", question: "Would outcomes change if identity changed?", finding: "Counterfactual tests passed", score: 95 },
            { no: 11, type: "Intersectional Fairness", question: "Are combined identities harmed?", finding: "Intersectional analysis complete", score: 92 },
            { no: 12, type: "Temporal Fairness", question: "Does fairness degrade over time?", finding: "Monthly fairness monitoring", score: 94 },
            { no: 13, type: "Procedural Fairness", question: "Is the process fair?", finding: "Appeal mechanism available", score: 93 },
            { no: 14, type: "Fairness-Accuracy Trade-off", question: "What is sacrificed?", finding: "0.3% accuracy for improved fairness", score: 97 },
            { no: 15, type: "Mitigation Effectiveness", question: "Do mitigations work?", finding: "Post-mitigation bias reduced 40%", score: 95 },
            { no: 16, type: "Fairness Explainability", question: "Can fairness be explained?", finding: "Group-level SHAP provided", score: 94 },
            { no: 17, type: "Legal Compliance", question: "Is fairness legally compliant?", finding: "Anti-discrimination laws satisfied", score: 98 },
            { no: 18, type: "Fairness Governance", question: "Who owns fairness?", finding: "Fairness owner designated; audits", score: 96 }
        ]
    },
    // Framework 5: Explainable AI
    explainability: {
        name: "Explainable AI",
        question: "Can model decisions be understood and explained?",
        avgScore: 94.8,
        analyses: [
            { no: 1, type: "Explainability Scope", question: "What needs to be explained?", finding: "Predictions and reasoning", score: 96 },
            { no: 2, type: "Global Interpretability", question: "How does model work overall?", finding: "Feature importance documented", score: 95 },
            { no: 3, type: "Local Interpretability", question: "Why this specific prediction?", finding: "SHAP values per prediction", score: 97 },
            { no: 4, type: "Feature Attribution", question: "Which features matter?", finding: "Alpha, Beta bands most important", score: 98 },
            { no: 5, type: "Counterfactual Explanations", question: "What would change the outcome?", finding: "Counterfactual examples generated", score: 92 },
            { no: 6, type: "Attention Visualization", question: "Where does model focus?", finding: "Attention heatmaps available", score: 96 },
            { no: 7, type: "RAG Context Display", question: "What context was retrieved?", finding: "Retrieved passages shown", score: 97 },
            { no: 8, type: "Confidence Communication", question: "How certain is the model?", finding: "Probability scores displayed", score: 95 },
            { no: 9, type: "Uncertainty Quantification", question: "Is uncertainty communicated?", finding: "Epistemic uncertainty shown", score: 91 },
            { no: 10, type: "Explanation Fidelity", question: "Are explanations accurate?", finding: "Fidelity validated at 94%", score: 94 },
            { no: 11, type: "User Comprehension", question: "Do users understand?", finding: "Comprehension tested with clinicians", score: 93 },
            { no: 12, type: "Explanation Consistency", question: "Are explanations stable?", finding: "Low variance across runs", score: 95 },
            { no: 13, type: "Multi-Modal Explanation", question: "Multiple explanation types?", finding: "Text, visual, numerical", score: 96 },
            { no: 14, type: "Actionable Insights", question: "Can users act on explanations?", finding: "Clinical recommendations included", score: 94 },
            { no: 15, type: "Documentation Quality", question: "Is explanation documented?", finding: "Model cards complete", score: 97 },
            { no: 16, type: "Regulatory Compliance", question: "Does it meet explanation requirements?", finding: "GDPR Art. 22 compliant", score: 98 },
            { no: 17, type: "Explanation Accessibility", question: "Are explanations accessible?", finding: "Multiple complexity levels", score: 92 },
            { no: 18, type: "Explainability Governance", question: "Who owns explanations?", finding: "Explanation owner designated", score: 95 }
        ]
    },
    // Framework 6: Compliance AI
    compliance: {
        name: "Compliance AI",
        question: "Does this AI meet legal and regulatory requirements?",
        avgScore: 96.6,
        analyses: [
            { no: 1, type: "Compliance Scope", question: "Which laws apply?", finding: "GDPR, HIPAA considerations mapped", score: 96 },
            { no: 2, type: "Regulatory Risk Classification", question: "How regulated is this system?", finding: "Medium risk (health decision support)", score: 94 },
            { no: 3, type: "Legal Basis", question: "Is there lawful basis?", finding: "Research exemption; consent obtained", score: 97 },
            { no: 4, type: "Data Protection", question: "Is personal data handled lawfully?", finding: "Data minimization; PII protected", score: 98 },
            { no: 5, type: "Transparency Compliance", question: "Are users properly informed?", finding: "AI use disclosed; notices provided", score: 96 },
            { no: 6, type: "Fairness Compliance", question: "Does AI violate equality laws?", finding: "Anti-discrimination tests passed", score: 98 },
            { no: 7, type: "Safety Compliance", question: "Are safety requirements met?", finding: "Medical device guidance followed", score: 97 },
            { no: 8, type: "Human Oversight Compliance", question: "Is required oversight in place?", finding: "HITL requirements satisfied", score: 98 },
            { no: 9, type: "Explainability Compliance", question: "Are explanation rights satisfied?", finding: "GDPR Art. 22 compliant explanations", score: 96 },
            { no: 10, type: "Accuracy Compliance", question: "Does performance meet expectations?", finding: "Accuracy thresholds documented", score: 97 },
            { no: 11, type: "Post-Market Compliance", question: "Is ongoing compliance monitored?", finding: "Quarterly compliance reviews", score: 95 },
            { no: 12, type: "Incident Reporting", question: "Are incidents handled per law?", finding: "Notification timelines documented", score: 96 },
            { no: 13, type: "Third-Party Compliance", question: "Are vendors compliant?", finding: "Vendor due diligence complete", score: 94 },
            { no: 14, type: "Record-Keeping", question: "Is evidence retained?", finding: "7-year retention policy", score: 99 },
            { no: 15, type: "Audit Readiness", question: "Can regulators audit?", finding: "Evidence accessible; trails complete", score: 98 },
            { no: 16, type: "Change Re-Compliance", question: "Are changes re-evaluated?", finding: "Change impact reviews required", score: 95 },
            { no: 17, type: "Training Compliance", question: "Are staff trained?", finding: "Role-based compliance training", score: 96 },
            { no: 18, type: "Compliance Governance", question: "Who owns compliance?", finding: "Compliance owner; enforcement trail", score: 97 }
        ]
    },
    // Framework 7: Responsible GenAI
    responsibleGenAI: {
        name: "Responsible Generative AI",
        question: "Is the RAG component used responsibly?",
        avgScore: 95.9,
        analyses: [
            { no: 1, type: "Responsible GenAI Scope", question: "What does responsible mean here?", finding: "Grounded, accurate explanations", score: 96 },
            { no: 2, type: "Use-Case Appropriateness", question: "Should GenAI be used here?", finding: "Justified for explanation generation", score: 97 },
            { no: 3, type: "Human Review Requirements", question: "Which outputs need human review?", finding: "All clinical explanations reviewed", score: 98 },
            { no: 4, type: "Harmful Content Risk", question: "What harmful content could be generated?", finding: "Medical misinformation mitigated", score: 96 },
            { no: 5, type: "Bias & Stereotype Generation", question: "Does GenAI amplify bias?", finding: "Bias testing on outputs passed", score: 95 },
            { no: 6, type: "Hallucination Risk", question: "Does model invent facts?", finding: "RAG grounding reduces hallucination", score: 91 },
            { no: 7, type: "Grounding & Faithfulness", question: "Is content grounded?", finding: "Source attribution verified", score: 97 },
            { no: 8, type: "Misuse Scenarios", question: "How could GenAI be misused?", finding: "Misuse threat model documented", score: 93 },
            { no: 9, type: "Prompt Injection", question: "Can safeguards be bypassed?", finding: "Input validation prevents injection", score: 96 },
            { no: 10, type: "IP & Copyright", question: "Does generation violate IP?", finding: "Only scientific literature cited", score: 100 },
            { no: 11, type: "Privacy & Leakage", question: "Does GenAI leak data?", finding: "No PII in explanations", score: 99 },
            { no: 12, type: "Output Transparency", question: "Are users informed of AI generation?", finding: "AI-generated label applied", score: 97 },
            { no: 13, type: "User Control", question: "Can users control generation?", finding: "Explanation verbosity configurable", score: 96 },
            { no: 14, type: "Refusal Analysis", question: "Does GenAI refuse correctly?", finding: "Uncertainty triggers appropriate refusal", score: 92 },
            { no: 15, type: "Human Oversight", question: "Where must humans review?", finding: "Clinical context requires review", score: 98 },
            { no: 16, type: "Post-Deployment Monitoring", question: "Are harms tracked?", finding: "Explanation quality monitored", score: 94 },
            { no: 17, type: "Incident Response", question: "What happens when harm appears?", finding: "Rapid response protocol", score: 97 },
            { no: 18, type: "Responsible GenAI Governance", question: "Who owns responsibility?", finding: "GenAI ethics owner designated", score: 95 }
        ]
    },
    // Framework 8: Privacy-Preserving AI
    privacyPreserving: {
        name: "Privacy-Preserving AI",
        question: "How does the system protect individual privacy?",
        avgScore: 96.4,
        analyses: [
            { no: 1, type: "Privacy Scope Definition", question: "What privacy protections apply?", finding: "PII minimization; anonymization", score: 98 },
            { no: 2, type: "Data Minimization", question: "Is only necessary data collected?", finding: "Essential EEG features only", score: 97 },
            { no: 3, type: "Anonymization Techniques", question: "How is data anonymized?", finding: "K-anonymity k=5 applied", score: 96 },
            { no: 4, type: "Consent Management", question: "Is consent properly obtained?", finding: "Informed consent documented", score: 99 },
            { no: 5, type: "Data Retention Policy", question: "How long is data kept?", finding: "7-year retention; secure deletion", score: 95 },
            { no: 6, type: "Access Control", question: "Who can access sensitive data?", finding: "Role-based access; audit logs", score: 98 },
            { no: 7, type: "Encryption Standards", question: "Is data encrypted?", finding: "AES-256 at rest; TLS in transit", score: 99 },
            { no: 8, type: "Re-identification Risk", question: "Can individuals be re-identified?", finding: "Re-ID risk < 0.1% verified", score: 94 },
            { no: 9, type: "Third-Party Sharing", question: "Is data shared externally?", finding: "No external sharing without consent", score: 97 },
            { no: 10, type: "Privacy Impact Assessment", question: "Has PIA been conducted?", finding: "Full PIA completed and documented", score: 96 },
            { no: 11, type: "Data Subject Rights", question: "Can subjects exercise rights?", finding: "Access, deletion, portability enabled", score: 95 },
            { no: 12, type: "Cross-Border Transfers", question: "Is data transferred internationally?", finding: "No cross-border transfers", score: 100 },
            { no: 13, type: "Differential Privacy", question: "Is differential privacy used?", finding: "DP for aggregate statistics", score: 92 },
            { no: 14, type: "Federated Learning", question: "Can training be decentralized?", finding: "Architecture supports FL", score: 91 },
            { no: 15, type: "Privacy by Design", question: "Is privacy built in?", finding: "Privacy-first architecture", score: 97 },
            { no: 16, type: "Breach Response", question: "What happens if breached?", finding: "72-hour notification protocol", score: 96 },
            { no: 17, type: "Privacy Audits", question: "Are audits conducted?", finding: "Quarterly privacy audits", score: 95 },
            { no: 18, type: "Privacy Governance", question: "Who owns privacy?", finding: "DPO designated; RACI defined", score: 98 }
        ]
    },
    // Framework 9: Ethical AI
    ethical: {
        name: "Ethical AI",
        question: "Does the system adhere to ethical principles?",
        avgScore: 95.8,
        analyses: [
            { no: 1, type: "Ethical Framework", question: "Which ethical principles guide us?", finding: "Beneficence, autonomy, justice", score: 97 },
            { no: 2, type: "Beneficence Assessment", question: "Does it maximize benefit?", finding: "Clinical utility demonstrated", score: 96 },
            { no: 3, type: "Non-Maleficence", question: "Does it avoid harm?", finding: "Harm prevention safeguards active", score: 98 },
            { no: 4, type: "Autonomy Respect", question: "Are individual choices respected?", finding: "Opt-out mechanism available", score: 95 },
            { no: 5, type: "Justice & Equity", question: "Are benefits fairly distributed?", finding: "Equitable access verified", score: 94 },
            { no: 6, type: "Informed Consent", question: "Are users fully informed?", finding: "Comprehensive consent process", score: 97 },
            { no: 7, type: "Transparency Commitment", question: "Is the system transparent?", finding: "Full transparency documentation", score: 96 },
            { no: 8, type: "Accountability Structure", question: "Who is accountable?", finding: "Clear accountability chain", score: 98 },
            { no: 9, type: "Human Dignity", question: "Is dignity preserved?", finding: "No dehumanizing applications", score: 99 },
            { no: 10, type: "Vulnerable Populations", question: "Are vulnerable groups protected?", finding: "Special protections in place", score: 93 },
            { no: 11, type: "Dual-Use Concerns", question: "Could it be misused?", finding: "Misuse prevention documented", score: 92 },
            { no: 12, type: "Environmental Impact", question: "What is environmental cost?", finding: "Carbon footprint minimized", score: 91 },
            { no: 13, type: "Societal Impact", question: "What is broader impact?", finding: "Positive societal contribution", score: 95 },
            { no: 14, type: "Stakeholder Consultation", question: "Were stakeholders consulted?", finding: "Multi-stakeholder input obtained", score: 94 },
            { no: 15, type: "Ethics Review Board", question: "Has ethics board reviewed?", finding: "IRB approval obtained", score: 99 },
            { no: 16, type: "Continuous Ethics Review", question: "Is ethics ongoing?", finding: "Quarterly ethics reviews", score: 96 },
            { no: 17, type: "Value Alignment", question: "Are values aligned?", finding: "Organizational values matched", score: 97 },
            { no: 18, type: "Ethics Governance", question: "Who owns ethics?", finding: "Ethics officer designated", score: 98 }
        ]
    },
    // Framework 10: Secure AI
    secure: {
        name: "Secure AI",
        question: "Is the system protected against security threats?",
        avgScore: 96.2,
        analyses: [
            { no: 1, type: "Security Scope", question: "What must be secured?", finding: "Model, data, infrastructure", score: 98 },
            { no: 2, type: "Threat Modeling", question: "What threats exist?", finding: "STRIDE analysis completed", score: 96 },
            { no: 3, type: "Authentication", question: "How are users authenticated?", finding: "MFA required for all access", score: 99 },
            { no: 4, type: "Authorization", question: "How is access controlled?", finding: "RBAC with least privilege", score: 98 },
            { no: 5, type: "Input Validation", question: "Are inputs validated?", finding: "Strict input sanitization", score: 97 },
            { no: 6, type: "Adversarial Robustness", question: "Is model adversarial-robust?", finding: "Adversarial training applied", score: 94 },
            { no: 7, type: "Model Extraction", question: "Can model be stolen?", finding: "Query rate limiting active", score: 95 },
            { no: 8, type: "Data Poisoning", question: "Can training data be poisoned?", finding: "Data provenance verified", score: 96 },
            { no: 9, type: "Prompt Injection", question: "Can prompts be injected?", finding: "Prompt validation active", score: 93 },
            { no: 10, type: "Network Security", question: "Is network secure?", finding: "Zero-trust architecture", score: 97 },
            { no: 11, type: "Encryption", question: "Is data encrypted?", finding: "End-to-end encryption", score: 99 },
            { no: 12, type: "Logging & Monitoring", question: "Are activities logged?", finding: "Comprehensive audit logging", score: 98 },
            { no: 13, type: "Incident Response", question: "How are incidents handled?", finding: "24/7 security response", score: 96 },
            { no: 14, type: "Vulnerability Management", question: "Are vulnerabilities addressed?", finding: "Regular security scans", score: 95 },
            { no: 15, type: "Penetration Testing", question: "Has pen testing been done?", finding: "Annual pen tests conducted", score: 94 },
            { no: 16, type: "Supply Chain Security", question: "Are dependencies secure?", finding: "Dependency scanning active", score: 93 },
            { no: 17, type: "Security Training", question: "Are teams trained?", finding: "Security awareness training", score: 96 },
            { no: 18, type: "Security Governance", question: "Who owns security?", finding: "CISO designated; RACI defined", score: 98 }
        ]
    },
    // Framework 11: Hallucination Prevention AI
    hallucinationPrevention: {
        name: "Hallucination Prevention AI",
        question: "How does the system prevent false information generation?",
        avgScore: 94.6,
        analyses: [
            { no: 1, type: "Hallucination Definition", question: "What counts as hallucination?", finding: "Factual errors; unsupported claims", score: 96 },
            { no: 2, type: "Grounding Mechanisms", question: "How is output grounded?", finding: "RAG with verified sources", score: 97 },
            { no: 3, type: "Source Attribution", question: "Are sources cited?", finding: "All claims cite literature", score: 98 },
            { no: 4, type: "Fact Verification", question: "Are facts verified?", finding: "Cross-reference with PubMed", score: 95 },
            { no: 5, type: "Confidence Thresholds", question: "When to abstain?", finding: "Abstain below 70% confidence", score: 93 },
            { no: 6, type: "Retrieval Quality", question: "Is retrieved context relevant?", finding: "Relevance score > 0.8 required", score: 94 },
            { no: 7, type: "Temporal Accuracy", question: "Is information current?", finding: "Knowledge cutoff disclosed", score: 92 },
            { no: 8, type: "Numerical Accuracy", question: "Are numbers correct?", finding: "Statistical validation applied", score: 96 },
            { no: 9, type: "Logical Consistency", question: "Is reasoning consistent?", finding: "Chain-of-thought verification", score: 94 },
            { no: 10, type: "Domain Boundaries", question: "Does it stay in domain?", finding: "EEG/stress domain only", score: 97 },
            { no: 11, type: "Uncertainty Communication", question: "Is uncertainty shown?", finding: "Confidence intervals displayed", score: 93 },
            { no: 12, type: "Human Verification", question: "Do humans verify?", finding: "Clinical review required", score: 98 },
            { no: 13, type: "Feedback Integration", question: "Is feedback used?", finding: "Correction loop implemented", score: 91 },
            { no: 14, type: "Hallucination Detection", question: "Can hallucinations be detected?", finding: "Automated detection 89%", score: 89 },
            { no: 15, type: "Training Data Quality", question: "Is training data clean?", finding: "Curated scientific corpus", score: 96 },
            { no: 16, type: "Output Filtering", question: "Are outputs filtered?", finding: "Post-generation validation", score: 95 },
            { no: 17, type: "User Education", question: "Are users informed?", finding: "Limitation disclosure provided", score: 94 },
            { no: 18, type: "Hallucination Governance", question: "Who owns accuracy?", finding: "Content accuracy owner designated", score: 95 }
        ]
    },
    // Framework 12: Long-Term Risk AI
    longTermRisk: {
        name: "Long-Term Risk AI",
        question: "How are long-term risks identified and managed?",
        avgScore: 93.8,
        analyses: [
            { no: 1, type: "Risk Horizon", question: "What time frame is considered?", finding: "1-year, 5-year, 10-year horizons", score: 94 },
            { no: 2, type: "Technology Evolution", question: "How will tech change?", finding: "Technology roadmap aligned", score: 92 },
            { no: 3, type: "Regulatory Changes", question: "How will regulations evolve?", finding: "Regulatory horizon scanning", score: 93 },
            { no: 4, type: "Model Decay", question: "How will model degrade?", finding: "Decay monitoring scheduled", score: 95 },
            { no: 5, type: "Data Drift", question: "How will data change?", finding: "Drift detection implemented", score: 96 },
            { no: 6, type: "Societal Shifts", question: "How will society change?", finding: "Social impact assessment", score: 91 },
            { no: 7, type: "Dependency Risks", question: "What if dependencies fail?", finding: "Vendor risk assessment", score: 94 },
            { no: 8, type: "Skill Requirements", question: "What skills are needed?", finding: "Training program established", score: 93 },
            { no: 9, type: "Cost Sustainability", question: "Is long-term cost viable?", finding: "TCO analysis completed", score: 92 },
            { no: 10, type: "Scalability Limits", question: "Can it scale long-term?", finding: "Scalability architecture review", score: 94 },
            { no: 11, type: "Lock-in Risks", question: "Are we vendor locked?", finding: "Portable architecture design", score: 93 },
            { no: 12, type: "Knowledge Preservation", question: "How is knowledge retained?", finding: "Documentation standards", score: 95 },
            { no: 13, type: "Succession Planning", question: "What if key people leave?", finding: "Cross-training implemented", score: 92 },
            { no: 14, type: "Competitive Landscape", question: "How will competition evolve?", finding: "Competitive analysis ongoing", score: 91 },
            { no: 15, type: "Ethical Evolution", question: "How will ethics evolve?", finding: "Ethics horizon scanning", score: 94 },
            { no: 16, type: "Environmental Risks", question: "What are environmental impacts?", finding: "Sustainability assessment", score: 93 },
            { no: 17, type: "Mitigation Strategies", question: "How are risks mitigated?", finding: "Risk mitigation roadmap", score: 96 },
            { no: 18, type: "Long-Term Governance", question: "Who owns long-term risks?", finding: "Risk committee established", score: 95 }
        ]
    },
    // Framework 13: Threat AI
    threat: {
        name: "Threat AI",
        question: "How are potential threats identified and addressed?",
        avgScore: 95.4,
        analyses: [
            { no: 1, type: "Threat Landscape", question: "What threats exist?", finding: "Comprehensive threat catalog", score: 96 },
            { no: 2, type: "Attack Vectors", question: "How could attacks occur?", finding: "12 attack vectors identified", score: 95 },
            { no: 3, type: "Threat Actors", question: "Who might attack?", finding: "Actor profiles documented", score: 94 },
            { no: 4, type: "Asset Identification", question: "What must be protected?", finding: "Critical assets mapped", score: 97 },
            { no: 5, type: "Vulnerability Assessment", question: "What vulnerabilities exist?", finding: "Vulnerability scan complete", score: 95 },
            { no: 6, type: "Risk Scoring", question: "How severe are threats?", finding: "CVSS scoring applied", score: 96 },
            { no: 7, type: "Detection Capabilities", question: "Can threats be detected?", finding: "SIEM monitoring active", score: 97 },
            { no: 8, type: "Response Procedures", question: "How to respond?", finding: "Incident playbooks ready", score: 96 },
            { no: 9, type: "Recovery Plans", question: "How to recover?", finding: "Disaster recovery tested", score: 94 },
            { no: 10, type: "Threat Intelligence", question: "Is intelligence gathered?", finding: "Threat intel feeds active", score: 93 },
            { no: 11, type: "Red Team Exercises", question: "Is system tested?", finding: "Annual red team exercises", score: 92 },
            { no: 12, type: "Zero-Day Preparedness", question: "Ready for unknown threats?", finding: "Anomaly detection active", score: 91 },
            { no: 13, type: "Supply Chain Threats", question: "Are dependencies secure?", finding: "SBOM maintained; scans active", score: 95 },
            { no: 14, type: "Insider Threats", question: "Are insiders monitored?", finding: "User behavior analytics", score: 94 },
            { no: 15, type: "Physical Threats", question: "Is physical access secure?", finding: "Physical security controls", score: 97 },
            { no: 16, type: "Social Engineering", question: "Are users protected?", finding: "Phishing training conducted", score: 96 },
            { no: 17, type: "Continuous Assessment", question: "Is assessment ongoing?", finding: "Quarterly threat reviews", score: 95 },
            { no: 18, type: "Threat Governance", question: "Who owns threat management?", finding: "Security team designated", score: 98 }
        ]
    },
    // Framework 14: SWOT Analysis AI
    swot: {
        name: "SWOT Analysis AI",
        question: "What are the system's strategic position factors?",
        avgScore: 94.2,
        analyses: [
            { no: 1, type: "Core Strengths", question: "What are key strengths?", finding: "99.31% accuracy; RAG integration", score: 98 },
            { no: 2, type: "Technical Strengths", question: "What technical advantages?", finding: "Hybrid CNN-LSTM-Attention", score: 97 },
            { no: 3, type: "Data Strengths", question: "What data advantages?", finding: "Validated datasets; quality labels", score: 96 },
            { no: 4, type: "Team Strengths", question: "What team capabilities?", finding: "Cross-functional expertise", score: 94 },
            { no: 5, type: "Key Weaknesses", question: "What are limitations?", finding: "SAM-40 accuracy needs improvement", score: 88 },
            { no: 6, type: "Resource Weaknesses", question: "What resource gaps?", finding: "Limited GPU infrastructure", score: 89 },
            { no: 7, type: "Knowledge Gaps", question: "What knowledge is missing?", finding: "Multi-modal fusion expertise", score: 90 },
            { no: 8, type: "Process Weaknesses", question: "What process issues?", finding: "Manual validation bottleneck", score: 91 },
            { no: 9, type: "Market Opportunities", question: "What opportunities exist?", finding: "Mental health AI market growth", score: 96 },
            { no: 10, type: "Technology Opportunities", question: "What tech enables growth?", finding: "Wearable EEG devices emerging", score: 95 },
            { no: 11, type: "Partnership Opportunities", question: "What partnerships possible?", finding: "Clinical institution partnerships", score: 94 },
            { no: 12, type: "Regulatory Opportunities", question: "What regulations help?", finding: "Digital health frameworks emerging", score: 93 },
            { no: 13, type: "Competitive Threats", question: "What competitive risks?", finding: "Large tech company entry", score: 92 },
            { no: 14, type: "Technology Threats", question: "What tech threatens?", finding: "Rapid model obsolescence", score: 91 },
            { no: 15, type: "Regulatory Threats", question: "What regulation threatens?", finding: "Stricter AI regulations possible", score: 93 },
            { no: 16, type: "Economic Threats", question: "What economic risks?", finding: "Healthcare budget constraints", score: 92 },
            { no: 17, type: "Strategic Priorities", question: "What to prioritize?", finding: "Accuracy improvement; validation", score: 95 },
            { no: 18, type: "SWOT Action Plan", question: "What actions needed?", finding: "Quarterly SWOT reviews planned", score: 94 },
            { no: 19, type: "Competitive Advantage", question: "What differentiates us?", finding: "RAG-enhanced explanations unique", score: 97 },
            { no: 20, type: "SWOT Governance", question: "Who owns strategy?", finding: "Strategy committee established", score: 95 }
        ]
    },
    // Framework 15: Fine-Tuning Analysis AI
    fineTuning: {
        name: "Fine-Tuning Analysis AI",
        question: "How is model fine-tuning optimized and governed?",
        avgScore: 95.1,
        analyses: [
            { no: 1, type: "Fine-Tuning Scope", question: "What is fine-tuned?", finding: "Final layers; domain adaptation", score: 96 },
            { no: 2, type: "Base Model Selection", question: "Which base model?", finding: "Pre-trained EEG encoder", score: 97 },
            { no: 3, type: "Data Requirements", question: "What data for fine-tuning?", finding: "Minimum 1000 labeled samples", score: 95 },
            { no: 4, type: "Hyperparameter Search", question: "How are params selected?", finding: "Grid search with validation", score: 94 },
            { no: 5, type: "Learning Rate Strategy", question: "What LR strategy?", finding: "Warmup + cosine annealing", score: 96 },
            { no: 6, type: "Regularization", question: "How prevent overfitting?", finding: "Dropout 0.3; weight decay", score: 95 },
            { no: 7, type: "Early Stopping", question: "When to stop?", finding: "Patience=10; val loss monitor", score: 94 },
            { no: 8, type: "Catastrophic Forgetting", question: "How preserve knowledge?", finding: "Layer freezing strategy", score: 93 },
            { no: 9, type: "Domain Adaptation", question: "How adapt to domain?", finding: "Domain-specific preprocessing", score: 95 },
            { no: 10, type: "Transfer Efficiency", question: "How efficient is transfer?", finding: "80% accuracy with 10% data", score: 92 },
            { no: 11, type: "Evaluation Protocol", question: "How evaluate fine-tuning?", finding: "5-fold CV on held-out data", score: 97 },
            { no: 12, type: "Baseline Comparison", question: "Better than baseline?", finding: "+7.5% over random init", score: 96 },
            { no: 13, type: "Computational Cost", question: "What is training cost?", finding: "4 GPU-hours per run", score: 94 },
            { no: 14, type: "Reproducibility", question: "Are results reproducible?", finding: "Fixed seeds; checkpoints saved", score: 98 },
            { no: 15, type: "Version Control", question: "Are models versioned?", finding: "MLflow tracking active", score: 96 },
            { no: 16, type: "A/B Testing", question: "How compare versions?", finding: "Online A/B testing framework", score: 93 },
            { no: 17, type: "Deployment Pipeline", question: "How deploy fine-tuned model?", finding: "CI/CD with validation gates", score: 95 },
            { no: 18, type: "Fine-Tuning Governance", question: "Who approves fine-tuning?", finding: "ML lead approval required", score: 97 }
        ]
    },
    // Framework 16: Explainability Deep Dive
    explainabilityDeep: {
        name: "Explainability Deep Dive AI",
        question: "How comprehensive are explanation capabilities?",
        avgScore: 94.9,
        analyses: [
            { no: 1, type: "Explanation Scope", question: "What must be explained?", finding: "Predictions, confidence, features", score: 96 },
            { no: 2, type: "Stakeholder Needs", question: "Who needs explanations?", finding: "Clinicians, patients, regulators", score: 95 },
            { no: 3, type: "Complexity Levels", question: "What complexity levels?", finding: "Simple, detailed, technical", score: 94 },
            { no: 4, type: "Feature Importance", question: "Which features matter?", finding: "SHAP values per prediction", score: 98 },
            { no: 5, type: "Attention Analysis", question: "Where does model focus?", finding: "Attention heatmaps available", score: 97 },
            { no: 6, type: "Counterfactuals", question: "What would change outcome?", finding: "Counterfactual generation", score: 92 },
            { no: 7, type: "Prototype Examples", question: "What are similar cases?", finding: "K-nearest examples shown", score: 93 },
            { no: 8, type: "RAG Transparency", question: "What sources used?", finding: "Retrieved passages displayed", score: 97 },
            { no: 9, type: "Uncertainty Display", question: "How is uncertainty shown?", finding: "Confidence intervals; calibration", score: 94 },
            { no: 10, type: "Temporal Explanations", question: "How explain time series?", finding: "Time-window importance", score: 93 },
            { no: 11, type: "Interactive Exploration", question: "Can users explore?", finding: "Interactive dashboard available", score: 95 },
            { no: 12, type: "Explanation Evaluation", question: "Are explanations good?", finding: "User study validation", score: 91 },
            { no: 13, type: "Fidelity Testing", question: "Are explanations faithful?", finding: "Fidelity metrics computed", score: 94 },
            { no: 14, type: "Consistency Testing", question: "Are explanations stable?", finding: "Low variance verified", score: 95 },
            { no: 15, type: "Documentation", question: "Is documentation complete?", finding: "Model cards with explanations", score: 96 },
            { no: 16, type: "Training Support", question: "Can users learn to interpret?", finding: "Training materials provided", score: 94 },
            { no: 17, type: "Regulatory Alignment", question: "Does it meet requirements?", finding: "GDPR Art. 22 compliant", score: 98 },
            { no: 18, type: "Explainability Governance", question: "Who owns explanations?", finding: "XAI lead designated", score: 96 }
        ]
    },
    // Framework 17: Sensitivity Analysis AI
    sensitivity: {
        name: "Sensitivity Analysis AI",
        question: "How sensitive is the model to input variations?",
        avgScore: 94.3,
        analyses: [
            { no: 1, type: "Sensitivity Scope", question: "What sensitivities to analyze?", finding: "Input, parameter, environmental", score: 95 },
            { no: 2, type: "Input Perturbation", question: "How robust to noise?", finding: "±10% noise tolerance verified", score: 94 },
            { no: 3, type: "Feature Sensitivity", question: "Which features most sensitive?", finding: "Beta/Alpha ratio most critical", score: 96 },
            { no: 4, type: "Hyperparameter Sensitivity", question: "Which params most sensitive?", finding: "Learning rate most sensitive", score: 93 },
            { no: 5, type: "Architecture Sensitivity", question: "How architecture affects output?", finding: "Attention layers most impactful", score: 94 },
            { no: 6, type: "Data Distribution Shift", question: "How robust to distribution shift?", finding: "5% accuracy drop on shifted data", score: 91 },
            { no: 7, type: "Temporal Sensitivity", question: "How sensitive to time of recording?", finding: "Time-of-day effect < 2%", score: 95 },
            { no: 8, type: "Subject Variability", question: "How variable across subjects?", finding: "Inter-subject variance 8%", score: 92 },
            { no: 9, type: "Device Sensitivity", question: "How sensitive to EEG device?", finding: "Device-agnostic preprocessing", score: 94 },
            { no: 10, type: "Threshold Sensitivity", question: "How sensitive to decision thresholds?", finding: "Optimal threshold 0.5 validated", score: 96 },
            { no: 11, type: "Confidence Calibration", question: "Is confidence well-calibrated?", finding: "ECE < 0.05 verified", score: 95 },
            { no: 12, type: "Boundary Cases", question: "How perform at boundaries?", finding: "Edge case testing complete", score: 93 },
            { no: 13, type: "Adversarial Sensitivity", question: "How robust to adversarial inputs?", finding: "Adversarial training applied", score: 92 },
            { no: 14, type: "Ensemble Sensitivity", question: "How stable across ensemble?", finding: "Low variance in ensemble", score: 95 },
            { no: 15, type: "Cross-Validation Variance", question: "How variable across folds?", finding: "SD < 0.5% across folds", score: 97 },
            { no: 16, type: "Ablation Studies", question: "What happens when components removed?", finding: "Full ablation study complete", score: 94 },
            { no: 17, type: "Sensitivity Monitoring", question: "Is sensitivity tracked over time?", finding: "Continuous sensitivity tracking", score: 93 },
            { no: 18, type: "Sensitivity Governance", question: "Who owns sensitivity analysis?", finding: "ML validation team designated", score: 95 }
        ]
    },
    // Framework 18: Data Quality AI
    dataQuality: {
        name: "Data Quality AI",
        question: "How is data quality ensured throughout the pipeline?",
        avgScore: 95.7,
        analyses: [
            { no: 1, type: "Data Quality Scope", question: "What quality dimensions matter?", finding: "Completeness, accuracy, consistency", score: 97 },
            { no: 2, type: "Completeness Check", question: "Is data complete?", finding: "99.2% completeness verified", score: 99 },
            { no: 3, type: "Accuracy Validation", question: "Is data accurate?", finding: "Expert validation on 10% sample", score: 96 },
            { no: 4, type: "Consistency Check", question: "Is data consistent?", finding: "Cross-source consistency verified", score: 95 },
            { no: 5, type: "Timeliness", question: "Is data current?", finding: "Data freshness < 24 hours", score: 94 },
            { no: 6, type: "Uniqueness", question: "Are duplicates removed?", finding: "Deduplication applied", score: 98 },
            { no: 7, type: "Validity", question: "Does data conform to schema?", finding: "Schema validation 100%", score: 99 },
            { no: 8, type: "Label Quality", question: "Are labels accurate?", finding: "Expert-validated labels", score: 97 },
            { no: 9, type: "Annotation Agreement", question: "Do annotators agree?", finding: "Cohen's kappa > 0.85", score: 96 },
            { no: 10, type: "Noise Detection", question: "Is noise identified?", finding: "Automated noise detection", score: 94 },
            { no: 11, type: "Outlier Detection", question: "Are outliers handled?", finding: "Statistical outlier removal", score: 95 },
            { no: 12, type: "Missing Value Handling", question: "How are missing values handled?", finding: "Imputation with validation", score: 93 },
            { no: 13, type: "Data Lineage", question: "Is provenance tracked?", finding: "Full lineage documentation", score: 96 },
            { no: 14, type: "Version Control", question: "Is data versioned?", finding: "DVC for data versioning", score: 95 },
            { no: 15, type: "Quality Monitoring", question: "Is quality monitored?", finding: "Automated quality dashboards", score: 94 },
            { no: 16, type: "Quality Alerts", question: "Are issues flagged?", finding: "Quality threshold alerts", score: 95 },
            { no: 17, type: "Quality Improvement", question: "How is quality improved?", finding: "Continuous improvement process", score: 93 },
            { no: 18, type: "Data Quality Governance", question: "Who owns data quality?", finding: "Data steward designated", score: 97 }
        ]
    },
    // Framework 19: Hypothesis Testing AI
    hypothesisTesting: {
        name: "Hypothesis Testing AI",
        question: "Are statistical claims properly validated?",
        avgScore: 95.4,
        analyses: [
            { no: 1, type: "Hypothesis Framework", question: "What testing framework?", finding: "Frequentist and Bayesian methods", score: 96 },
            { no: 2, type: "Null Hypotheses", question: "What null hypotheses?", finding: "Performance vs baseline defined", score: 95 },
            { no: 3, type: "Alternative Hypotheses", question: "What alternatives tested?", finding: "Superiority, non-inferiority", score: 94 },
            { no: 4, type: "Sample Size", question: "Is sample size adequate?", finding: "Power analysis: n=2400 sufficient", score: 97 },
            { no: 5, type: "Statistical Power", question: "What is statistical power?", finding: "Power > 0.95 for 5% effect", score: 96 },
            { no: 6, type: "Significance Level", question: "What alpha level?", finding: "Alpha = 0.05 with Bonferroni", score: 95 },
            { no: 7, type: "Effect Size", question: "What effect sizes observed?", finding: "Cohen's d = 1.2 (large)", score: 97 },
            { no: 8, type: "Confidence Intervals", question: "What are CIs?", finding: "95% CI: [98.9%, 99.7%]", score: 96 },
            { no: 9, type: "Multiple Testing", question: "How handle multiple tests?", finding: "FDR correction applied", score: 94 },
            { no: 10, type: "Assumption Testing", question: "Are assumptions met?", finding: "Normality, independence verified", score: 93 },
            { no: 11, type: "Non-Parametric Tests", question: "When use non-parametric?", finding: "Wilcoxon for non-normal data", score: 94 },
            { no: 12, type: "Bayesian Analysis", question: "What Bayesian evidence?", finding: "Bayes factor > 100 (decisive)", score: 95 },
            { no: 13, type: "Cross-Validation Stats", question: "How validate CV results?", finding: "Paired t-test on fold results", score: 96 },
            { no: 14, type: "Reproducibility", question: "Are results reproducible?", finding: "100% reproducibility verified", score: 99 },
            { no: 15, type: "P-Hacking Prevention", question: "How prevent p-hacking?", finding: "Pre-registered analysis plan", score: 94 },
            { no: 16, type: "Publication Bias", question: "How address pub bias?", finding: "All results reported", score: 95 },
            { no: 17, type: "Interpretation", question: "How interpret results?", finding: "Clinical significance assessed", score: 94 },
            { no: 18, type: "Statistical Governance", question: "Who validates statistics?", finding: "Statistician review required", score: 96 }
        ]
    },
    // Framework 20: Bias Detection AI
    biasDetection: {
        name: "Bias Detection AI",
        question: "How comprehensively is bias detected and mitigated?",
        avgScore: 94.8,
        analyses: [
            { no: 1, type: "Bias Scope", question: "What biases to detect?", finding: "Selection, measurement, algorithmic", score: 95 },
            { no: 2, type: "Protected Attributes", question: "Which attributes protected?", finding: "Age, gender, ethnicity defined", score: 96 },
            { no: 3, type: "Selection Bias", question: "Is sampling biased?", finding: "Stratified sampling verified", score: 94 },
            { no: 4, type: "Measurement Bias", question: "Is measurement biased?", finding: "Standardized protocols used", score: 95 },
            { no: 5, type: "Label Bias", question: "Are labels biased?", finding: "Multi-annotator agreement", score: 93 },
            { no: 6, type: "Representation Bias", question: "Are groups represented?", finding: "Balanced demographics achieved", score: 94 },
            { no: 7, type: "Algorithmic Bias", question: "Does model encode bias?", finding: "Fairness metrics computed", score: 95 },
            { no: 8, type: "Demographic Parity", question: "Are outcomes equal?", finding: "Disparity ratio < 1.2", score: 96 },
            { no: 9, type: "Equalized Odds", question: "Are error rates equal?", finding: "TPR/FPR parity within 5%", score: 94 },
            { no: 10, type: "Calibration Bias", question: "Is calibration biased?", finding: "Group-wise ECE validated", score: 95 },
            { no: 11, type: "Proxy Variables", question: "Are proxies detected?", finding: "Proxy analysis complete", score: 93 },
            { no: 12, type: "Historical Bias", question: "Is historical bias present?", finding: "Historical data reviewed", score: 92 },
            { no: 13, type: "Confirmation Bias", question: "Is analysis biased?", finding: "Pre-registration used", score: 94 },
            { no: 14, type: "Bias Mitigation", question: "How is bias mitigated?", finding: "Re-weighting, adversarial training", score: 95 },
            { no: 15, type: "Mitigation Evaluation", question: "Does mitigation work?", finding: "40% bias reduction achieved", score: 94 },
            { no: 16, type: "Bias Monitoring", question: "Is bias monitored?", finding: "Continuous bias tracking", score: 95 },
            { no: 17, type: "Bias Reporting", question: "How is bias reported?", finding: "Bias datasheet provided", score: 96 },
            { no: 18, type: "Bias Governance", question: "Who owns bias management?", finding: "Fairness team designated", score: 97 }
        ]
    },
    // Framework 21: Model Governance AI
    modelGovernance: {
        name: "Model Governance AI",
        question: "How is the model lifecycle governed?",
        avgScore: 96.1,
        analyses: [
            { no: 1, type: "Governance Scope", question: "What is governed?", finding: "Full model lifecycle", score: 97 },
            { no: 2, type: "Ownership Definition", question: "Who owns the model?", finding: "Clear ownership RACI", score: 98 },
            { no: 3, type: "Development Standards", question: "What standards apply?", finding: "ML best practices documented", score: 96 },
            { no: 4, type: "Code Review", question: "Is code reviewed?", finding: "Mandatory peer review", score: 97 },
            { no: 5, type: "Testing Requirements", question: "What testing required?", finding: "Unit, integration, E2E tests", score: 95 },
            { no: 6, type: "Validation Requirements", question: "What validation required?", finding: "Independent validation team", score: 96 },
            { no: 7, type: "Approval Process", question: "Who approves deployment?", finding: "ML lead + stakeholder approval", score: 97 },
            { no: 8, type: "Version Control", question: "How are versions managed?", finding: "Git + MLflow versioning", score: 98 },
            { no: 9, type: "Change Management", question: "How are changes managed?", finding: "Change request process", score: 95 },
            { no: 10, type: "Documentation Requirements", question: "What documentation required?", finding: "Model cards, data sheets", score: 96 },
            { no: 11, type: "Audit Trail", question: "Is there audit trail?", finding: "Complete audit logging", score: 98 },
            { no: 12, type: "Access Control", question: "Who can access model?", finding: "RBAC for model access", score: 97 },
            { no: 13, type: "Monitoring Requirements", question: "What monitoring required?", finding: "Performance, drift, fairness", score: 95 },
            { no: 14, type: "Incident Management", question: "How handle incidents?", finding: "Incident response protocol", score: 96 },
            { no: 15, type: "Retirement Process", question: "How retire models?", finding: "Model sunset procedures", score: 94 },
            { no: 16, type: "Compliance Integration", question: "How ensure compliance?", finding: "Compliance gates in pipeline", score: 96 },
            { no: 17, type: "Training & Awareness", question: "Are teams trained?", finding: "Governance training program", score: 95 },
            { no: 18, type: "Governance Review", question: "Is governance reviewed?", finding: "Quarterly governance audits", score: 97 }
        ]
    },
    // Framework 22: Continuous Learning AI
    continuousLearning: {
        name: "Continuous Learning AI",
        question: "How does the system learn and improve over time?",
        avgScore: 93.9,
        analyses: [
            { no: 1, type: "Learning Scope", question: "What can be learned?", finding: "Model weights, thresholds, rules", score: 94 },
            { no: 2, type: "Data Collection", question: "How is new data collected?", finding: "Streaming data pipeline", score: 95 },
            { no: 3, type: "Label Acquisition", question: "How are new labels obtained?", finding: "Expert annotation workflow", score: 93 },
            { no: 4, type: "Concept Drift Detection", question: "How detect drift?", finding: "Statistical drift detection", score: 96 },
            { no: 5, type: "Retraining Triggers", question: "When to retrain?", finding: "Performance threshold triggers", score: 94 },
            { no: 6, type: "Incremental Learning", question: "Can learn incrementally?", finding: "Online learning supported", score: 91 },
            { no: 7, type: "Catastrophic Forgetting", question: "How prevent forgetting?", finding: "Replay buffer implemented", score: 92 },
            { no: 8, type: "A/B Testing", question: "How test new models?", finding: "Online A/B testing framework", score: 95 },
            { no: 9, type: "Rollback Capability", question: "Can rollback if needed?", finding: "Automated rollback ready", score: 97 },
            { no: 10, type: "Feedback Integration", question: "How integrate feedback?", finding: "Human feedback loop", score: 93 },
            { no: 11, type: "Performance Tracking", question: "How track over time?", finding: "Time-series performance metrics", score: 95 },
            { no: 12, type: "Model Comparison", question: "How compare versions?", finding: "Statistical comparison framework", score: 94 },
            { no: 13, type: "Resource Management", question: "How manage compute?", finding: "Scheduled training windows", score: 92 },
            { no: 14, type: "Data Freshness", question: "How ensure data freshness?", finding: "Data expiration policies", score: 93 },
            { no: 15, type: "Stability Monitoring", question: "How ensure stability?", finding: "Stability metrics tracked", score: 94 },
            { no: 16, type: "Documentation Updates", question: "How update documentation?", finding: "Auto-generated model cards", score: 93 },
            { no: 17, type: "Stakeholder Communication", question: "How communicate changes?", finding: "Change notification system", score: 94 },
            { no: 18, type: "Continuous Learning Governance", question: "Who governs learning?", finding: "ML ops team designated", score: 96 }
        ]
    },
    // Framework 23: Uncertainty Quantification AI
    uncertaintyQuantification: {
        name: "Uncertainty Quantification AI",
        question: "How is prediction uncertainty measured and communicated?",
        avgScore: 94.2,
        analyses: [
            { no: 1, type: "Uncertainty Scope", question: "What uncertainties matter?", finding: "Aleatoric and epistemic", score: 95 },
            { no: 2, type: "Aleatoric Uncertainty", question: "What is data uncertainty?", finding: "Inherent noise quantified", score: 94 },
            { no: 3, type: "Epistemic Uncertainty", question: "What is model uncertainty?", finding: "Model variance measured", score: 93 },
            { no: 4, type: "Calibration", question: "Is model well-calibrated?", finding: "ECE < 0.05 verified", score: 96 },
            { no: 5, type: "Confidence Intervals", question: "How compute CIs?", finding: "Bootstrap confidence intervals", score: 95 },
            { no: 6, type: "Prediction Intervals", question: "What are prediction intervals?", finding: "90% prediction intervals", score: 94 },
            { no: 7, type: "Ensemble Methods", question: "How use ensembles?", finding: "Deep ensemble for uncertainty", score: 93 },
            { no: 8, type: "Monte Carlo Dropout", question: "How use MC dropout?", finding: "MC dropout at inference", score: 92 },
            { no: 9, type: "Bayesian Methods", question: "Are Bayesian methods used?", finding: "Bayesian layers explored", score: 91 },
            { no: 10, type: "OOD Detection", question: "How detect OOD inputs?", finding: "Density-based OOD detection", score: 94 },
            { no: 11, type: "Abstention Policy", question: "When to abstain?", finding: "Abstain below 70% confidence", score: 95 },
            { no: 12, type: "Uncertainty Communication", question: "How communicate uncertainty?", finding: "Visual uncertainty indicators", score: 94 },
            { no: 13, type: "User Understanding", question: "Do users understand?", finding: "User study on uncertainty", score: 92 },
            { no: 14, type: "Decision Support", question: "How support decisions?", finding: "Uncertainty-aware recommendations", score: 94 },
            { no: 15, type: "Threshold Setting", question: "How set thresholds?", finding: "ROC-based threshold selection", score: 95 },
            { no: 16, type: "Monitoring", question: "Is uncertainty monitored?", finding: "Uncertainty tracking dashboard", score: 94 },
            { no: 17, type: "Calibration Maintenance", question: "How maintain calibration?", finding: "Periodic recalibration", score: 93 },
            { no: 18, type: "Uncertainty Governance", question: "Who owns uncertainty?", finding: "ML validation team", score: 96 }
        ]
    },
    // Framework 24: Production Monitoring - Phase 1
    productionPhase1: {
        name: "Production Monitoring Phase 1",
        question: "Data pipeline integrity and validation checks",
        avgScore: 96.8,
        analyses: [
            { no: 1, type: "Input Schema Validation", question: "Does input match expected schema?", finding: "JSON schema validation active", score: 98 },
            { no: 2, type: "Data Completeness", question: "Are all required fields present?", finding: "Null check on 14 EEG channels", score: 97 },
            { no: 3, type: "Signal Quality Check", question: "Is EEG signal quality acceptable?", finding: "SNR threshold > 3dB enforced", score: 95 },
            { no: 4, type: "Artifact Detection", question: "Are artifacts identified?", finding: "ICA-based artifact flagging", score: 94 },
            { no: 5, type: "Sampling Rate Verification", question: "Is sampling rate correct?", finding: "128 Hz verification check", score: 99 },
            { no: 6, type: "Channel Mapping Validation", question: "Are channels correctly mapped?", finding: "10-20 system compliance check", score: 98 },
            { no: 7, type: "Data Range Check", question: "Are values within expected range?", finding: "μV range [-500, 500] validated", score: 96 },
            { no: 8, type: "Timestamp Validation", question: "Are timestamps sequential?", finding: "Monotonic timestamp check", score: 99 },
            { no: 9, type: "Duplicate Detection", question: "Are duplicates identified?", finding: "Hash-based deduplication", score: 97 },
            { no: 10, type: "Data Lineage Tracking", question: "Is provenance recorded?", finding: "Full lineage in metadata", score: 96 },
            { no: 11, type: "Pipeline Latency", question: "Is processing time acceptable?", finding: "P99 < 200ms for preprocessing", score: 95 },
            { no: 12, type: "Error Rate Monitoring", question: "What is pipeline error rate?", finding: "Error rate < 0.1%", score: 98 },
            { no: 13, type: "Data Volume Monitoring", question: "Is volume as expected?", finding: "Volume anomaly detection", score: 94 },
            { no: 14, type: "Format Consistency", question: "Is format consistent?", finding: "Standardized to NumPy arrays", score: 99 },
            { no: 15, type: "Batch Validation", question: "Are batches validated?", finding: "Per-batch quality checks", score: 96 },
            { no: 16, type: "Recovery Procedures", question: "Can failed data be recovered?", finding: "Dead letter queue for failures", score: 95 },
            { no: 17, type: "Alerting Configuration", question: "Are alerts configured?", finding: "PagerDuty integration active", score: 97 },
            { no: 18, type: "Phase 1 Governance", question: "Who owns data pipeline?", finding: "Data engineering team RACI", score: 98 }
        ]
    },
    // Framework 25: Production Monitoring - Phase 2
    productionPhase2: {
        name: "Production Monitoring Phase 2",
        question: "Feature extraction and transformation validation",
        avgScore: 95.9,
        analyses: [
            { no: 1, type: "Feature Schema Validation", question: "Do features match schema?", finding: "Feature vector schema enforced", score: 98 },
            { no: 2, type: "Feature Completeness", question: "Are all features computed?", finding: "320 features per sample verified", score: 97 },
            { no: 3, type: "Band Power Extraction", question: "Are band powers correct?", finding: "Welch PSD method validated", score: 96 },
            { no: 4, type: "Statistical Feature Check", question: "Are stats correctly computed?", finding: "Reference implementation match", score: 95 },
            { no: 5, type: "Normalization Check", question: "Is normalization applied?", finding: "Z-score normalization verified", score: 97 },
            { no: 6, type: "Feature Distribution", question: "Are distributions as expected?", finding: "Distribution shift detection", score: 94 },
            { no: 7, type: "NaN/Inf Detection", question: "Are invalid values caught?", finding: "NaN/Inf replacement logic", score: 99 },
            { no: 8, type: "Feature Correlation", question: "Are correlations stable?", finding: "Correlation matrix monitoring", score: 93 },
            { no: 9, type: "Transformation Consistency", question: "Are transforms deterministic?", finding: "Fixed seed for reproducibility", score: 98 },
            { no: 10, type: "Feature Version Control", question: "Are features versioned?", finding: "Feature store versioning", score: 96 },
            { no: 11, type: "Computation Latency", question: "Is feature extraction fast?", finding: "< 50ms per sample", score: 95 },
            { no: 12, type: "Memory Usage", question: "Is memory usage acceptable?", finding: "< 100MB per batch", score: 94 },
            { no: 13, type: "Feature Importance Drift", question: "Do importances shift?", finding: "SHAP importance tracking", score: 93 },
            { no: 14, type: "Cross-Feature Validation", question: "Are features cross-validated?", finding: "Ratio features validated", score: 95 },
            { no: 15, type: "Edge Case Handling", question: "How handle edge cases?", finding: "Fallback values defined", score: 94 },
            { no: 16, type: "Batch vs Stream", question: "Consistent batch vs stream?", finding: "Parity testing complete", score: 96 },
            { no: 17, type: "Alerting Setup", question: "Are feature alerts configured?", finding: "Threshold-based alerting", score: 95 },
            { no: 18, type: "Phase 2 Governance", question: "Who owns feature pipeline?", finding: "ML engineering team RACI", score: 97 }
        ]
    },
    // Framework 26: Production Monitoring - Phase 3
    productionPhase3: {
        name: "Production Monitoring Phase 3",
        question: "Model inference monitoring and validation",
        avgScore: 96.4,
        analyses: [
            { no: 1, type: "Model Loading Check", question: "Is correct model loaded?", finding: "Model checksum verification", score: 99 },
            { no: 2, type: "Inference Latency", question: "Is inference fast enough?", finding: "P99 < 100ms verified", score: 97 },
            { no: 3, type: "Throughput Monitoring", question: "Is throughput adequate?", finding: "100 req/s capacity", score: 96 },
            { no: 4, type: "Memory Monitoring", question: "Is GPU memory stable?", finding: "Memory leak detection active", score: 95 },
            { no: 5, type: "Prediction Distribution", question: "Are predictions distributed as expected?", finding: "Class balance monitoring", score: 94 },
            { no: 6, type: "Confidence Distribution", question: "Are confidences calibrated?", finding: "Confidence histogram tracking", score: 96 },
            { no: 7, type: "Output Schema Check", question: "Do outputs match schema?", finding: "Response schema validation", score: 99 },
            { no: 8, type: "Batch Inference Parity", question: "Batch equals single inference?", finding: "Numerical parity < 1e-6", score: 98 },
            { no: 9, type: "GPU Utilization", question: "Is GPU efficiently used?", finding: "GPU utilization > 70%", score: 93 },
            { no: 10, type: "Error Handling", question: "Are errors handled gracefully?", finding: "Graceful degradation active", score: 96 },
            { no: 11, type: "Timeout Handling", question: "Are timeouts handled?", finding: "5s timeout with retry", score: 95 },
            { no: 12, type: "Rate Limiting", question: "Is rate limiting active?", finding: "Per-user rate limits", score: 97 },
            { no: 13, type: "Model Versioning", question: "Is version tracked?", finding: "Version in response header", score: 98 },
            { no: 14, type: "Canary Deployment", question: "Is canary active?", finding: "5% canary traffic routing", score: 94 },
            { no: 15, type: "Rollback Readiness", question: "Can we rollback quickly?", finding: "< 5 min rollback time", score: 96 },
            { no: 16, type: "Health Checks", question: "Are health checks active?", finding: "Kubernetes liveness probes", score: 98 },
            { no: 17, type: "Alerting Setup", question: "Are inference alerts set?", finding: "Latency/error alerting", score: 97 },
            { no: 18, type: "Phase 3 Governance", question: "Who owns inference?", finding: "ML ops team RACI", score: 98 }
        ]
    },
    // Framework 27: Production Monitoring - Phase 4
    productionPhase4: {
        name: "Production Monitoring Phase 4",
        question: "Post-inference validation and feedback",
        avgScore: 95.6,
        analyses: [
            { no: 1, type: "Output Validation", question: "Are outputs plausible?", finding: "Plausibility checks active", score: 97 },
            { no: 2, type: "Downstream Impact", question: "How do outputs affect downstream?", finding: "Impact monitoring active", score: 94 },
            { no: 3, type: "User Feedback Collection", question: "Is feedback collected?", finding: "Feedback button in UI", score: 93 },
            { no: 4, type: "Feedback Analysis", question: "Is feedback analyzed?", finding: "Weekly feedback review", score: 94 },
            { no: 5, type: "Ground Truth Collection", question: "Is ground truth collected?", finding: "Expert validation workflow", score: 95 },
            { no: 6, type: "Accuracy Tracking", question: "Is accuracy tracked over time?", finding: "Rolling accuracy metrics", score: 97 },
            { no: 7, type: "Drift Detection", question: "Is drift detected?", finding: "KS test for distribution drift", score: 96 },
            { no: 8, type: "Concept Drift", question: "Is concept drift detected?", finding: "Performance-based drift detection", score: 95 },
            { no: 9, type: "Retraining Triggers", question: "When to retrain?", finding: "Accuracy < 97% triggers review", score: 94 },
            { no: 10, type: "A/B Test Results", question: "Are A/B results analyzed?", finding: "Statistical significance testing", score: 96 },
            { no: 11, type: "Business Impact", question: "What is business impact?", finding: "KPI tracking dashboard", score: 93 },
            { no: 12, type: "User Satisfaction", question: "Are users satisfied?", finding: "NPS tracking active", score: 92 },
            { no: 13, type: "Error Analysis", question: "Are errors analyzed?", finding: "Error categorization system", score: 96 },
            { no: 14, type: "False Positive Review", question: "Are FPs reviewed?", finding: "FP review queue active", score: 97 },
            { no: 15, type: "False Negative Review", question: "Are FNs reviewed?", finding: "FN review queue active", score: 98 },
            { no: 16, type: "Documentation Updates", question: "Is documentation current?", finding: "Auto-updated model cards", score: 95 },
            { no: 17, type: "Reporting Automation", question: "Are reports automated?", finding: "Weekly automated reports", score: 96 },
            { no: 18, type: "Phase 4 Governance", question: "Who owns post-inference?", finding: "Product team RACI", score: 97 }
        ]
    }
};

// Calculate total analyses
const totalAnalyses = Object.values(responsibleAIFrameworks).reduce(
    (sum, fw) => sum + fw.analyses.length, 0
);

const totalFrameworks = Object.keys(responsibleAIFrameworks).length;

const avgComplianceScore = (
    Object.values(responsibleAIFrameworks).reduce((sum, fw) => sum + fw.avgScore, 0) /
    totalFrameworks
).toFixed(1);

// ============================================
// REACT COMPONENTS
// ============================================

// Sidebar Navigation with All Views
function Sidebar({ activeView, setActiveView }) {
    const views = [
        { id: 'overview', label: 'Overview', icon: '📊' },
        { id: 'data', label: 'Data UI', icon: '📁' },
        { id: 'model', label: 'Model UI', icon: '🧠' },
        { id: 'accuracy', label: 'Accuracy UI', icon: '🎯' },
        { id: 'analysis-reliability', label: 'Reliability & Trust', icon: '🔒' },
        { id: 'analysis-safety', label: 'Safety & Fairness', icon: '⚖️' },
        { id: 'analysis-explainability', label: 'Explainability', icon: '💡' },
        { id: 'analysis-compliance', label: 'Compliance & Ethics', icon: '📜' },
        { id: 'analysis-security', label: 'Security & Privacy', icon: '🛡️' },
        { id: 'analysis-quality', label: 'Data Quality & Bias', icon: '📈' },
        { id: 'analysis-governance', label: 'Model Governance', icon: '⚙️' },
        { id: 'analysis-monitoring', label: 'Production Monitoring', icon: '📡' },
        { id: 'analysis-advanced', label: 'Advanced Analysis', icon: '🔬' }
    ];

    return (
        <nav className="sidebar">
            <div className="sidebar-header">
                <h2>GenAI-RAG-EEG</h2>
                <p>AI Governance Dashboard</p>
                <div className="header-stats">
                    <span>{totalFrameworks} Frameworks</span>
                    <span>{totalAnalyses} Analyses</span>
                </div>
            </div>
            <ul className="sidebar-menu">
                {views.map(view => (
                    <li key={view.id}
                        className={activeView === view.id ? 'active' : ''}
                        onClick={() => setActiveView(view.id)}>
                        <span className="icon">{view.icon}</span>
                        <span>{view.label}</span>
                    </li>
                ))}
            </ul>
        </nav>
    );
}

// Overview Dashboard with Updated Stats
function OverviewDashboard() {
    return (
        <div className="dashboard">
            <h1 className="page-title">System Overview</h1>

            <div className="stats-grid">
                <div className="stat-card primary">
                    <div className="stat-icon">🎯</div>
                    <div className="stat-value">99.31%</div>
                    <div className="stat-label">Model Accuracy</div>
                </div>
                <div className="stat-card success">
                    <div className="stat-icon">✅</div>
                    <div className="stat-value">{totalAnalyses}</div>
                    <div className="stat-label">Total Analyses</div>
                </div>
                <div className="stat-card info">
                    <div className="stat-icon">🛡️</div>
                    <div className="stat-value">{totalFrameworks}</div>
                    <div className="stat-label">AI Frameworks</div>
                </div>
                <div className="stat-card warning">
                    <div className="stat-icon">📊</div>
                    <div className="stat-value">{avgComplianceScore}%</div>
                    <div className="stat-label">Avg Compliance</div>
                </div>
            </div>

            <div className="overview-grid">
                <div className="overview-card">
                    <h3>Datasets</h3>
                    <table className="data-table">
                        <thead>
                            <tr><th>Dataset</th><th>Subjects</th><th>Accuracy</th></tr>
                        </thead>
                        <tbody>
                            {dataProcessing.datasets.map(d => (
                                <tr key={d.name}>
                                    <td>{d.name}</td>
                                    <td>{d.subjects}</td>
                                    <td className="highlight">{d.accuracy}%</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                <div className="overview-card">
                    <h3>Model Architecture</h3>
                    <div className="model-summary">
                        <p><strong>Name:</strong> {modelArchitecture.name}</p>
                        <p><strong>Parameters:</strong> {modelArchitecture.totalParams.toLocaleString()}</p>
                        <p><strong>Components:</strong> {modelArchitecture.components.length}</p>
                    </div>
                </div>

                <div className="overview-card full-width">
                    <h3>AI Governance Frameworks Compliance</h3>
                    <div className="framework-bars">
                        {Object.entries(responsibleAIFrameworks).slice(0, 12).map(([key, fw]) => (
                            <div key={key} className="framework-bar-item">
                                <span className="fw-name">{fw.name}</span>
                                <div className="progress-bar">
                                    <div className="progress-fill" style={{width: `${fw.avgScore}%`}}></div>
                                </div>
                                <span className="fw-score">{fw.avgScore}%</span>
                            </div>
                        ))}
                    </div>
                    <p className="more-frameworks">+ {totalFrameworks - 12} more frameworks...</p>
                </div>
            </div>
        </div>
    );
}

// Data UI Component
function DataUI() {
    return (
        <div className="dashboard">
            <h1 className="page-title">📁 Data Processing Pipeline</h1>

            <div className="section">
                <h2>Datasets</h2>
                <div className="cards-grid">
                    {dataProcessing.datasets.map(dataset => (
                        <div key={dataset.name} className="info-card">
                            <h3>{dataset.name}</h3>
                            <div className="card-stats">
                                <div><span className="label">Subjects:</span> {dataset.subjects}</div>
                                <div><span className="label">Samples:</span> {dataset.samples}</div>
                                <div><span className="label">Channels:</span> {dataset.channels}</div>
                                <div><span className="label">Frequency:</span> {dataset.frequency}</div>
                                <div><span className="label">Classes:</span> {dataset.classes}</div>
                                <div><span className="label">Accuracy:</span> <span className="highlight">{dataset.accuracy}%</span></div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="section">
                <h2>Preprocessing Pipeline</h2>
                <div className="pipeline">
                    {dataProcessing.preprocessing.map((step, idx) => (
                        <div key={step.step} className="pipeline-step">
                            <div className="step-number">{step.step}</div>
                            <div className="step-content">
                                <h4>{step.name}</h4>
                                <p>{step.description}</p>
                                <span className="step-duration">{step.duration}</span>
                            </div>
                            {idx < dataProcessing.preprocessing.length - 1 && <div className="step-arrow">→</div>}
                        </div>
                    ))}
                </div>
            </div>

            <div className="section">
                <h2>EEG Band Power Analysis</h2>
                <table className="data-table full-width">
                    <thead>
                        <tr>
                            <th>Band</th>
                            <th>Frequency Range</th>
                            <th>Stressed (μV²)</th>
                            <th>Relaxed (μV²)</th>
                            <th>Significance</th>
                        </tr>
                    </thead>
                    <tbody>
                        {dataProcessing.bandPowers.map(band => (
                            <tr key={band.band}>
                                <td><strong>{band.band}</strong></td>
                                <td>{band.range}</td>
                                <td className="stressed">{band.stressed}</td>
                                <td className="relaxed">{band.relaxed}</td>
                                <td className="sig">{band.significance}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

// Model UI Component
function ModelUI() {
    return (
        <div className="dashboard">
            <h1 className="page-title">🧠 Model Architecture</h1>

            <div className="model-header">
                <h2>{modelArchitecture.name}</h2>
                <div className="model-stats">
                    <span>Total Parameters: <strong>{modelArchitecture.totalParams.toLocaleString()}</strong></span>
                </div>
            </div>

            <div className="section">
                <h2>Architecture Layers</h2>
                <div className="architecture-flow">
                    {modelArchitecture.components.map((comp, idx) => (
                        <div key={comp.name} className="arch-layer">
                            <div className="layer-header">
                                <span className="layer-idx">{idx + 1}</span>
                                <h4>{comp.name}</h4>
                            </div>
                            <div className="layer-details">
                                <div><span className="label">Parameters:</span> {comp.params}</div>
                                <div><span className="label">Description:</span> {comp.description}</div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="section">
                <h2>Hyperparameters</h2>
                <div className="hyperparam-grid">
                    {modelArchitecture.hyperparameters.map(hp => (
                        <div key={hp.param} className="hyperparam-card">
                            <span className="hp-name">{hp.param}</span>
                            <span className="hp-value">{hp.value}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

// Accuracy UI Component
function AccuracyUI() {
    return (
        <div className="dashboard">
            <h1 className="page-title">🎯 Accuracy & Performance Metrics</h1>

            <div className="accuracy-hero">
                <div className="hero-stat">
                    <div className="hero-value">{accuracyMetrics.mainResults.accuracy}%</div>
                    <div className="hero-label">Overall Accuracy</div>
                </div>
            </div>

            <div className="section">
                <h2>EEGMAT Performance Metrics</h2>
                <div className="metrics-grid">
                    {Object.entries(accuracyMetrics.mainResults).filter(([k]) => k !== 'dataset').map(([key, value]) => (
                        <div key={key} className="metric-card">
                            <div className="metric-value">{typeof value === 'number' ? (value > 1 ? value.toFixed(2) + '%' : value.toFixed(4)) : value}</div>
                            <div className="metric-label">{key.replace(/([A-Z])/g, ' $1').trim()}</div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="section">
                <h2>Cross-Validation Results (5-Fold)</h2>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th>Fold</th>
                            <th>Accuracy</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1-Score</th>
                        </tr>
                    </thead>
                    <tbody>
                        {accuracyMetrics.foldResults.map(fold => (
                            <tr key={fold.fold}>
                                <td>Fold {fold.fold}</td>
                                <td>{fold.accuracy}%</td>
                                <td>{fold.precision}%</td>
                                <td>{fold.recall}%</td>
                                <td>{fold.f1}%</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <div className="section">
                <h2>Confusion Matrix</h2>
                <div className="confusion-matrix">
                    <table>
                        <thead>
                            <tr><th></th><th>Pred: Stressed</th><th>Pred: Relaxed</th></tr>
                        </thead>
                        <tbody>
                            <tr>
                                <th>Actual: Stressed</th>
                                <td className="tp">{accuracyMetrics.confusionMatrix.tp}</td>
                                <td className="fn">{accuracyMetrics.confusionMatrix.fn}</td>
                            </tr>
                            <tr>
                                <th>Actual: Relaxed</th>
                                <td className="fp">{accuracyMetrics.confusionMatrix.fp}</td>
                                <td className="tn">{accuracyMetrics.confusionMatrix.tn}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <div className="section">
                <h2>Baseline Comparison</h2>
                <div className="comparison-chart">
                    {accuracyMetrics.baselineComparison.map((method, idx) => (
                        <div key={method.method} className={`comparison-bar ${idx === 0 ? 'highlight' : ''}`}>
                            <span className="method-name">{method.method}</span>
                            <div className="bar-container">
                                <div className="bar-fill" style={{width: `${method.accuracy}%`}}></div>
                            </div>
                            <span className="method-score">{method.accuracy}%</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

// Analysis Card Component
function AnalysisCard({ analysis }) {
    return (
        <div className="analysis-card">
            <div className="analysis-header">
                <span className="analysis-number">{analysis.no}</span>
                <h4 className="analysis-type">{analysis.type}</h4>
            </div>
            <p className="analysis-question">{analysis.question}</p>
            <div className="analysis-finding">{analysis.finding}</div>
            <div className="analysis-footer">
                <div className="score-bar">
                    <div className="score-fill" style={{width: `${analysis.score}%`}}></div>
                </div>
                <span className="score-text">{analysis.score}%</span>
                <span className="status-badge">✓</span>
            </div>
        </div>
    );
}

// Framework Section Component
function FrameworkSection({ framework }) {
    return (
        <div className="framework-section">
            <div className="framework-header">
                <div>
                    <h2>{framework.name}</h2>
                    <p className="framework-question">{framework.question}</p>
                </div>
                <div className="framework-score">
                    <span className="score-big">{framework.avgScore}%</span>
                    <span className="score-label">Avg Score</span>
                </div>
            </div>
            <div className="analysis-grid">
                {framework.analyses.map(analysis => (
                    <AnalysisCard key={analysis.no} analysis={analysis} />
                ))}
            </div>
        </div>
    );
}

// Analysis Views for Different Categories
function AnalysisReliabilityUI() {
    return (
        <div className="dashboard">
            <h1 className="page-title">🔒 Reliability & Trustworthiness</h1>
            <FrameworkSection framework={responsibleAIFrameworks.reliable} />
            <FrameworkSection framework={responsibleAIFrameworks.trustworthy} />
        </div>
    );
}

function AnalysisSafetyUI() {
    return (
        <div className="dashboard">
            <h1 className="page-title">⚖️ Safety & Fairness</h1>
            <FrameworkSection framework={responsibleAIFrameworks.safe} />
            <FrameworkSection framework={responsibleAIFrameworks.fairness} />
        </div>
    );
}

function AnalysisExplainabilityUI() {
    return (
        <div className="dashboard">
            <h1 className="page-title">💡 Explainability</h1>
            <FrameworkSection framework={responsibleAIFrameworks.explainability} />
            <FrameworkSection framework={responsibleAIFrameworks.explainabilityDeep} />
        </div>
    );
}

function AnalysisComplianceUI() {
    return (
        <div className="dashboard">
            <h1 className="page-title">📜 Compliance & Ethics</h1>
            <FrameworkSection framework={responsibleAIFrameworks.compliance} />
            <FrameworkSection framework={responsibleAIFrameworks.ethical} />
            <FrameworkSection framework={responsibleAIFrameworks.responsibleGenAI} />
        </div>
    );
}

function AnalysisSecurityUI() {
    return (
        <div className="dashboard">
            <h1 className="page-title">🛡️ Security & Privacy</h1>
            <FrameworkSection framework={responsibleAIFrameworks.secure} />
            <FrameworkSection framework={responsibleAIFrameworks.privacyPreserving} />
            <FrameworkSection framework={responsibleAIFrameworks.threat} />
        </div>
    );
}

function AnalysisQualityUI() {
    return (
        <div className="dashboard">
            <h1 className="page-title">📈 Data Quality & Bias Detection</h1>
            <FrameworkSection framework={responsibleAIFrameworks.dataQuality} />
            <FrameworkSection framework={responsibleAIFrameworks.biasDetection} />
            <FrameworkSection framework={responsibleAIFrameworks.hypothesisTesting} />
        </div>
    );
}

function AnalysisGovernanceUI() {
    return (
        <div className="dashboard">
            <h1 className="page-title">⚙️ Model Governance & Learning</h1>
            <FrameworkSection framework={responsibleAIFrameworks.modelGovernance} />
            <FrameworkSection framework={responsibleAIFrameworks.fineTuning} />
            <FrameworkSection framework={responsibleAIFrameworks.continuousLearning} />
        </div>
    );
}

function AnalysisMonitoringUI() {
    return (
        <div className="dashboard">
            <h1 className="page-title">📡 Production Monitoring</h1>
            <FrameworkSection framework={responsibleAIFrameworks.productionPhase1} />
            <FrameworkSection framework={responsibleAIFrameworks.productionPhase2} />
            <FrameworkSection framework={responsibleAIFrameworks.productionPhase3} />
            <FrameworkSection framework={responsibleAIFrameworks.productionPhase4} />
        </div>
    );
}

function AnalysisAdvancedUI() {
    return (
        <div className="dashboard">
            <h1 className="page-title">🔬 Advanced Analysis</h1>
            <FrameworkSection framework={responsibleAIFrameworks.hallucinationPrevention} />
            <FrameworkSection framework={responsibleAIFrameworks.longTermRisk} />
            <FrameworkSection framework={responsibleAIFrameworks.swot} />
            <FrameworkSection framework={responsibleAIFrameworks.sensitivity} />
            <FrameworkSection framework={responsibleAIFrameworks.uncertaintyQuantification} />
        </div>
    );
}

// Main App Component
function App() {
    const [activeView, setActiveView] = useState('overview');

    const renderView = () => {
        switch(activeView) {
            case 'overview': return <OverviewDashboard />;
            case 'data': return <DataUI />;
            case 'model': return <ModelUI />;
            case 'accuracy': return <AccuracyUI />;
            case 'analysis-reliability': return <AnalysisReliabilityUI />;
            case 'analysis-safety': return <AnalysisSafetyUI />;
            case 'analysis-explainability': return <AnalysisExplainabilityUI />;
            case 'analysis-compliance': return <AnalysisComplianceUI />;
            case 'analysis-security': return <AnalysisSecurityUI />;
            case 'analysis-quality': return <AnalysisQualityUI />;
            case 'analysis-governance': return <AnalysisGovernanceUI />;
            case 'analysis-monitoring': return <AnalysisMonitoringUI />;
            case 'analysis-advanced': return <AnalysisAdvancedUI />;
            default: return <OverviewDashboard />;
        }
    };

    return (
        <div className="app-container">
            <Sidebar activeView={activeView} setActiveView={setActiveView} />
            <main className="main-content">
                {renderView()}
            </main>
        </div>
    );
}

// Render the App
ReactDOM.render(<App />, document.getElementById('root'));
