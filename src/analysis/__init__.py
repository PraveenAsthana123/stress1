"""
Analysis Module for GenAI-RAG-EEG.

This module provides comprehensive data analysis and statistical testing
capabilities for EEG-based stress classification.

Modules:
    - signal_analysis: EEG signal processing and band power analysis
    - statistical_analysis: Advanced statistical tests and effect sizes
    - data_analysis: Comprehensive data loading and analysis pipeline

Reference: GenAI-RAG-EEG Paper v2, IEEE Sensors Journal 2024
"""

from .signal_analysis import (
    compute_psd,
    compute_band_power,
    band_power_analysis,
    alpha_suppression_analysis,
    theta_beta_ratio_analysis,
    frontal_asymmetry_analysis,
    compute_all_metrics,
    run_complete_signal_analysis,
    BandPowerResult,
    ClassificationMetrics,
    FREQUENCY_BANDS,
    CHANNEL_GROUPS
)

from .statistical_analysis import (
    # Effect sizes
    cohens_d,
    hedges_g,
    glass_delta,
    common_language_effect_size,
    eta_squared,
    omega_squared,
    compute_all_effect_sizes,

    # Normality tests
    test_normality,
    check_assumptions,

    # Parametric tests
    independent_ttest,
    paired_ttest,
    one_way_anova,

    # Non-parametric tests
    mann_whitney_u,
    wilcoxon_signed_rank,
    kruskal_wallis,
    friedman_test,
    mcnemar_test,

    # Multiple comparisons
    bonferroni_correction,
    holm_bonferroni_correction,
    benjamini_hochberg_fdr,

    # Correlation
    comprehensive_correlation,

    # Bootstrap and permutation
    bootstrap_ci,
    permutation_test,

    # Comprehensive analysis
    comprehensive_two_group_analysis,
    compare_cv_results,

    # Power analysis
    power_analysis_ttest,
    achieved_power,

    # Report generation
    generate_statistical_report,

    # Data classes
    StatisticalTestResult,
    NormalityTestResult,
    CorrelationResult,
    EffectSizeResult,
    MultipleComparisonResult,
    ComprehensiveAnalysisResult
)

from .data_analysis import (
    EEGDataLoader,
    QualityAssessor,
    FeatureExtractor,
    EEGAnalyzer,
    DatasetInfo,
    QualityReport,
    FeatureSet,
    AnalysisResult
)

# Comprehensive Analysis Framework (v4)
from .comprehensive_analysis import (
    # Data classes
    FeatureEngineeringResult,
    ClinicalMetrics,
    SubjectWiseResult,
    ReliabilityMetrics,
    ModelAnalysisResult,
    CognitiveWorkloadResult,

    # Analysis classes
    FeatureEngineeringAnalysis,
    ClinicalValidationAnalysis,
    SubjectWiseLOSOAnalysis,
    ReliabilityRobustnessAnalysis,
    ModelAnalysisFramework,
    CognitiveWorkloadAnalysis,
    PerformanceMetricsMatrix,
    DataQualityAnalysis,
    AccuracyAnalysis,
    SubjectAnalysis,

    # Main orchestrator
    ComprehensiveAnalysisOrchestrator,

    # Demo functions
    generate_demo_data,
    run_demo_analysis
)

from .visualization import (
    # Core plot functions
    plot_band_power_comparison,
    plot_band_power_heatmap,
    plot_violin_comparison,
    plot_effect_size_forest,
    plot_significance_volcano,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_multi_roc,
    plot_cross_validation_results,
    plot_psd,
    plot_spectrogram,
    plot_analysis_summary,

    # Advanced plot functions
    plot_precision_recall_curves,
    plot_calibration_curves,
    plot_shap_importance,
    plot_topographical_maps,
    plot_learning_curves,
    plot_component_importance,
    plot_cumulative_ablation,
    plot_power_analysis,
    plot_cross_subject_generalization,

    # Global chart generator (PNG/SVG/PDF/EPS)
    AnalysisChartGenerator,

    # Document exporter (PDF/Word/PPT)
    AnalysisReportExporter
)

# Research-Grade Visualization (Publication-Ready)
from .research_visualization import (
    # Main suite
    ResearchVisualizationSuite,

    # Sub-modules for specialized visualization
    TrendChartGenerator,
    ClinicalVisualization,
    StatisticalVisualization,
    FlowchartGenerator,

    # Publication settings
    apply_publication_style,
    PUBLICATION_RCPARAMS,
    PUBLICATION_COLORS,
    EEG_BANDS
)

# LaTeX Research Tools (PhD-Grade)
from .latex_research_tools import (
    # Master suite
    LaTeXResearchSuite,

    # Sub-modules
    LaTeXTableGenerator,
    PDFFigureExporter,
    PGFPlotsGenerator,
    BeamerGenerator,

    # Utilities
    apply_latex_style,
    PDF_RCPARAMS
)

# Research Format Configuration (Publisher-Ready)
from .research_formats import (
    # Enums
    FigureFormat,
    Resolution,
    ColorMode,

    # Configuration dictionaries
    COLORBLIND_PALETTES,
    PUBLISHER_CONFIGS,
    FIGURE_TYPE_FORMAT,
    DEFAULT_PALETTE,

    # Classes
    FileNamer,
    ResearchFigureExporter,
    ResearchTableFormatter,
    ResearchFormatConfig,

    # Functions
    get_publication_rcparams,
    apply_publisher_style,
    check_colorblind_safety,
    get_colorblind_palette,
    setup_research_environment,
    get_publisher_config
)

__all__ = [
    # Signal analysis
    'compute_psd',
    'compute_band_power',
    'band_power_analysis',
    'alpha_suppression_analysis',
    'theta_beta_ratio_analysis',
    'frontal_asymmetry_analysis',
    'compute_all_metrics',
    'run_complete_signal_analysis',
    'BandPowerResult',
    'ClassificationMetrics',
    'FREQUENCY_BANDS',
    'CHANNEL_GROUPS',

    # Statistical analysis
    'cohens_d',
    'hedges_g',
    'glass_delta',
    'common_language_effect_size',
    'eta_squared',
    'omega_squared',
    'compute_all_effect_sizes',
    'test_normality',
    'check_assumptions',
    'independent_ttest',
    'paired_ttest',
    'one_way_anova',
    'mann_whitney_u',
    'wilcoxon_signed_rank',
    'kruskal_wallis',
    'friedman_test',
    'mcnemar_test',
    'bonferroni_correction',
    'holm_bonferroni_correction',
    'benjamini_hochberg_fdr',
    'comprehensive_correlation',
    'bootstrap_ci',
    'permutation_test',
    'comprehensive_two_group_analysis',
    'compare_cv_results',
    'power_analysis_ttest',
    'achieved_power',
    'generate_statistical_report',
    'StatisticalTestResult',
    'NormalityTestResult',
    'CorrelationResult',
    'EffectSizeResult',
    'MultipleComparisonResult',
    'ComprehensiveAnalysisResult',

    # Data analysis
    'EEGDataLoader',
    'QualityAssessor',
    'FeatureExtractor',
    'EEGAnalyzer',
    'DatasetInfo',
    'QualityReport',
    'FeatureSet',
    'AnalysisResult',

    # Comprehensive Analysis Framework (v4)
    'FeatureEngineeringResult',
    'ClinicalMetrics',
    'SubjectWiseResult',
    'ReliabilityMetrics',
    'ModelAnalysisResult',
    'CognitiveWorkloadResult',
    'FeatureEngineeringAnalysis',
    'ClinicalValidationAnalysis',
    'SubjectWiseLOSOAnalysis',
    'ReliabilityRobustnessAnalysis',
    'ModelAnalysisFramework',
    'CognitiveWorkloadAnalysis',
    'PerformanceMetricsMatrix',
    'DataQualityAnalysis',
    'AccuracyAnalysis',
    'SubjectAnalysis',
    'ComprehensiveAnalysisOrchestrator',
    'generate_demo_data',
    'run_demo_analysis',

    # Advanced Visualization
    'plot_precision_recall_curves',
    'plot_calibration_curves',
    'plot_shap_importance',
    'plot_topographical_maps',
    'plot_learning_curves',
    'plot_component_importance',
    'plot_cumulative_ablation',
    'plot_power_analysis',
    'plot_cross_subject_generalization',

    # Global Chart Generator (PNG/SVG/PDF/EPS)
    'AnalysisChartGenerator',

    # Document Exporter (PDF/Word/PPT)
    'AnalysisReportExporter',

    # Research-Grade Visualization Suite
    'ResearchVisualizationSuite',
    'TrendChartGenerator',
    'ClinicalVisualization',
    'StatisticalVisualization',
    'FlowchartGenerator',
    'apply_publication_style',
    'PUBLICATION_RCPARAMS',
    'PUBLICATION_COLORS',
    'EEG_BANDS',

    # LaTeX Research Tools (PhD-Grade)
    'LaTeXResearchSuite',
    'LaTeXTableGenerator',
    'PDFFigureExporter',
    'PGFPlotsGenerator',
    'BeamerGenerator',

    # Research Format Configuration (Publisher-Ready)
    'FigureFormat',
    'Resolution',
    'ColorMode',
    'COLORBLIND_PALETTES',
    'PUBLISHER_CONFIGS',
    'FIGURE_TYPE_FORMAT',
    'DEFAULT_PALETTE',
    'FileNamer',
    'ResearchFigureExporter',
    'ResearchTableFormatter',
    'ResearchFormatConfig',
    'get_publication_rcparams',
    'apply_publisher_style',
    'check_colorblind_safety',
    'get_colorblind_palette',
    'setup_research_environment',
    'get_publisher_config',
]
