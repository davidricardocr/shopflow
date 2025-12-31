"""Evaluation metrics and visualization for ShopFlow."""

from .metrics import (
    evaluate_model,
    evaluate_by_category,
    get_confusion_matrix_details,
    compare_models,
    identify_failure_modes,
    format_classification_report,
    get_curve_data,
)
from .threshold import (
    ThresholdOptimizer,
    find_threshold_for_target_metric,
)
from .visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_threshold_analysis,
    plot_category_performance,
    plot_model_comparison,
    create_evaluation_dashboard,
)

__all__ = [
    # Metrics
    "evaluate_model",
    "evaluate_by_category",
    "get_confusion_matrix_details",
    "compare_models",
    "identify_failure_modes",
    "format_classification_report",
    "get_curve_data",
    # Threshold
    "ThresholdOptimizer",
    "find_threshold_for_target_metric",
    # Visualization
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_threshold_analysis",
    "plot_category_performance",
    "plot_model_comparison",
    "create_evaluation_dashboard",
]
