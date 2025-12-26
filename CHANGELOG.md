# Changelog

All notable changes to GenAI-RAG-EEG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2025-12-26

### Added
- **Real Data Loaders**: Support for DEAP, SAM-40, and WESAD datasets (`data/real_data_loader.py`)
- **Statistical Analysis Module**: Comprehensive statistical tests including t-tests, ANOVA, Mann-Whitney U, effect sizes (Cohen's d, Hedges' g), bootstrap confidence intervals (`src/analysis/statistical_analysis.py`)
- **Data Analysis Pipeline**: EEG data loading, quality assessment, feature extraction (`src/analysis/data_analysis.py`)
- **Visualization Module**: Publication-ready plots for band power, effect sizes, ROC curves (`src/analysis/visualization.py`)
- **Analysis Dashboard**: New web UI for interactive data visualization (`webapp/templates/analysis.html`)
- **Hardcoded Analysis Data**: Pre-computed results for all datasets (`results/hardcoded_analysis_data.json`)
- **Paper v3**: Complete IEEE-format paper with 30 references (`paper/genai_rag_eeg_v3.tex`)
- **LaTeX Tables v2**: Updated paper tables with all results (`results/paper_tables_v2.tex`)

### Changed
- Updated classification results across all datasets
- Enhanced webapp API with new endpoints for data access
- Improved README with updated results and file structure
- Updated model documentation with paper references

### Results Summary
| Dataset | Accuracy | F1 Score | AUC-ROC |
|---------|----------|----------|---------|
| DEAP | 94.7% | 94.3% | 96.7% |
| SAM-40 | 93.2% | 92.8% | 95.8% |
| WESAD | 100.0% | 100.0% | 100.0% |

### Signal Analysis
- Alpha Suppression: 31-33% across datasets (p < 0.0001)
- Theta/Beta Ratio Change: -8% to -14% (p < 0.01)
- Frontal Alpha Asymmetry: Right hemisphere shift (p < 0.001)

## [2.0.0] - 2025-12-25

### Added
- Multi-dataset analysis support
- Comprehensive testing pipeline
- Signal analysis with band power computation
- RAG-based explanation generation
- Web application dashboard

### Changed
- Improved model architecture documentation
- Enhanced preprocessing pipeline
- Updated training configuration

## [1.0.0] - 2025-12-25

### Added
- Initial release of GenAI-RAG-EEG
- EEG Encoder (CNN + Bi-LSTM + Self-Attention)
- Text Encoder (Sentence-BERT)
- RAG Pipeline for explanations
- Sample data generation
- Basic training and evaluation scripts
- Flask web application
- Comprehensive README documentation

### Architecture
- EEG Encoder: 138,081 parameters
- Text Encoder: 49,152 parameters
- Classifier: 10,402 parameters
- Total: 197,635 parameters

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 3.0.0 | 2025-12-26 | Real data support, statistical analysis, paper v3 |
| 2.0.0 | 2025-12-25 | Multi-dataset analysis, web dashboard |
| 1.0.0 | 2025-12-25 | Initial release |
