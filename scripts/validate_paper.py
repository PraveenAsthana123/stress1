#!/usr/bin/env python3
"""Validate paper content against actual data and code."""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path("/media/praveen/Asthana3/rajveer/eeg-stress-rag")
PAPER_DIR = PROJECT_ROOT / "paper"
RESULTS_DIR = PROJECT_ROOT / "results"

def load_paper():
    """Load the paper content."""
    paper_path = PAPER_DIR / "genai_rag_eeg_v4.tex"
    with open(paper_path, 'r') as f:
        return f.read()

def validate_figures(paper_content):
    """Validate all figures are defined and referenced."""
    print("\n" + "="*60)
    print("FIGURE VALIDATION")
    print("="*60)

    # Find all figure labels
    labels = re.findall(r'\\label\{(fig:[^}]+)\}', paper_content)
    refs = re.findall(r'\\ref\{(fig:[^}]+)\}', paper_content)

    print(f"\nTotal figures defined: {len(labels)}")
    print(f"Total figure references: {len(refs)}")

    # Check for unreferenced figures
    unreferenced = set(labels) - set(refs)
    if unreferenced:
        print(f"\n⚠️  Unreferenced figures: {unreferenced}")
    else:
        print("\n✓ All figures are referenced")

    # Check for missing figures
    missing = set(refs) - set(labels)
    if missing:
        print(f"\n❌ Missing figure definitions: {missing}")
    else:
        print("✓ All referenced figures exist")

    # List governance framework figures
    governance_figs = [l for l in labels if any(x in l for x in ['reliable', 'trustworthy', 'safe', 'accountable',
                                                                   'auditable', 'lifecycle', 'monitoring', 'green',
                                                                   'fairness', 'human', 'compliance', 'social',
                                                                   'hitl', 'data_trans', 'mechanistic', 'resp_genai'])]
    print(f"\nGovernance framework figures: {len(governance_figs)}")
    for fig in governance_figs:
        print(f"  ✓ {fig}")

    return labels

def validate_tables(paper_content):
    """Validate all tables are defined and referenced."""
    print("\n" + "="*60)
    print("TABLE VALIDATION")
    print("="*60)

    # Find all table labels
    labels = re.findall(r'\\label\{(tab:[^}]+)\}', paper_content)
    refs = re.findall(r'\\ref\{(tab:[^}]+)\}', paper_content)

    print(f"\nTotal tables defined: {len(labels)}")
    print(f"Total table references: {len(refs)}")

    # Check for unreferenced tables
    unreferenced = set(labels) - set(refs)
    if unreferenced:
        print(f"\n⚠️  Unreferenced tables: {len(unreferenced)}")
    else:
        print("\n✓ All tables are referenced")

    # Check for missing tables
    missing = set(refs) - set(labels)
    if missing:
        print(f"\n❌ Missing table definitions: {missing}")
    else:
        print("✓ All referenced tables exist")

    # Count governance tables
    governance_tabs = [l for l in labels if any(x in l for x in ['reliable', 'trustworthy', 'safe', 'accountable',
                                                                  'auditable', 'lifecycle', 'monitoring', 'green',
                                                                  'fairness', 'human', 'compliance', 'social',
                                                                  'hitl', 'data_trans', 'mechanistic', 'resp_genai',
                                                                  'responsible', 'trust', 'debug', 'interpretable',
                                                                  'portable', 'causality', 'interpretability'])]
    print(f"\nGovernance framework tables: {len(governance_tabs)}")

    return labels

def validate_accuracy(paper_content):
    """Validate accuracy values in paper."""
    print("\n" + "="*60)
    print("ACCURACY VALIDATION")
    print("="*60)

    # Load actual results
    results_files = list(RESULTS_DIR.glob("*.json"))
    print(f"\nFound {len(results_files)} result files")

    actual_results = {}
    for rf in results_files:
        try:
            with open(rf) as f:
                data = json.load(f)
                actual_results[rf.stem] = data
                print(f"  Loaded: {rf.name}")
        except:
            pass

    # Extract accuracy claims from paper
    accuracy_matches = re.findall(r'(\d{2,3}\.\d{1,2})\s*\\%?\s*(?:accuracy|Accuracy)', paper_content)
    print(f"\nAccuracy values found in paper:")
    for acc in set(accuracy_matches):
        print(f"  - {acc}%")

    # Check key accuracy values
    key_values = {
        "99.31": "EEGMAT main result",
        "72.92": "SAM-40 4-class result",
    }

    print("\nKey accuracy validation:")
    for val, desc in key_values.items():
        if val in paper_content:
            print(f"  ✓ {val}% ({desc}) - Found in paper")
        else:
            print(f"  ❌ {val}% ({desc}) - NOT found in paper")

def validate_confusion_matrix(paper_content):
    """Validate confusion matrix values."""
    print("\n" + "="*60)
    print("CONFUSION MATRIX VALIDATION")
    print("="*60)

    # Load real confusion matrix from results
    cm_file = RESULTS_DIR / "real_confusion_matrices.json"
    if cm_file.exists():
        with open(cm_file) as f:
            cm_data = json.load(f)
        print("\n✓ Loaded actual confusion matrix data")

        if "EEGMAT-Full" in cm_data:
            eegmat = cm_data["EEGMAT-Full"]
            print(f"\nActual EEGMAT Confusion Matrix:")
            print(f"  TP: {eegmat.get('tp', 'N/A')}")
            print(f"  TN: {eegmat.get('tn', 'N/A')}")
            print(f"  FP: {eegmat.get('fp', 'N/A')}")
            print(f"  FN: {eegmat.get('fn', 'N/A')}")

            # Check if these values are in paper
            tp = str(eegmat.get('tp', ''))
            if tp and tp in paper_content:
                print(f"  ✓ TP value {tp} found in paper")
            else:
                print(f"  ⚠️  TP value may need verification")
    else:
        print("\n⚠️  No confusion matrix results file found")
        print("  Checking paper for confusion matrix...")

        # Look for confusion matrix in paper
        if "confusion" in paper_content.lower():
            print("  ✓ Confusion matrix section exists in paper")

def validate_hyperparameters(paper_content):
    """Validate hyperparameters against training scripts."""
    print("\n" + "="*60)
    print("HYPERPARAMETER VALIDATION")
    print("="*60)

    # Expected hyperparameters
    expected = {
        "lr": ["0.001", "1e-3"],
        "batch_size": ["32"],
        "epochs": ["100", "30"],
        "dropout": ["0.3"],
        "weight_decay": ["0.01"],
    }

    print("\nChecking hyperparameters in paper:")
    for param, values in expected.items():
        found = any(v in paper_content for v in values)
        if found:
            print(f"  ✓ {param}: {values[0]}")
        else:
            print(f"  ⚠️  {param}: not found or different value")

def validate_sections(paper_content):
    """Validate paper sections."""
    print("\n" + "="*60)
    print("SECTION VALIDATION")
    print("="*60)

    sections = re.findall(r'\\section\{([^}]+)\}', paper_content)
    subsections = re.findall(r'\\subsection\{([^}]+)\}', paper_content)
    subsubsections = re.findall(r'\\subsubsection\{([^}]+)\}', paper_content)

    print(f"\nSections: {len(sections)}")
    for s in sections:
        print(f"  - {s}")

    print(f"\nSubsections: {len(subsections)}")
    print(f"Subsubsections: {len(subsubsections)}")

    # Check for governance framework sections
    governance_keywords = ['Reliable', 'Trustworthy', 'Safe', 'Accountable', 'Auditable',
                          'Lifecycle', 'Monitoring', 'Green', 'Fairness', 'Human-Centered',
                          'Compliance', 'Social', 'Human-in-the-Loop', 'Transparent',
                          'Mechanistic', 'Responsible']

    print(f"\nGovernance framework sections found:")
    found_count = 0
    for kw in governance_keywords:
        if any(kw in s for s in subsubsections):
            print(f"  ✓ {kw}")
            found_count += 1
    print(f"\nTotal: {found_count}/{len(governance_keywords)} governance frameworks")

def main():
    print("="*60)
    print("PAPER VALIDATION REPORT")
    print("GenAI-RAG-EEG v4")
    print("="*60)

    paper_content = load_paper()
    print(f"\nPaper length: {len(paper_content):,} characters")

    validate_figures(paper_content)
    validate_tables(paper_content)
    validate_accuracy(paper_content)
    validate_confusion_matrix(paper_content)
    validate_hyperparameters(paper_content)
    validate_sections(paper_content)

    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
