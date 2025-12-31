"""Business cost-benefit analysis for ShopFlow."""

from .cost_analysis import (
    PredictionOutcome,
    calculate_confusion_matrix_from_predictions,
    calculate_financial_impact,
    calculate_expected_value,
    calculate_expected_value_at_threshold,
    find_optimal_threshold,
    compare_thresholds,
    generate_cost_benefit_report,
)

__all__ = [
    "PredictionOutcome",
    "calculate_confusion_matrix_from_predictions",
    "calculate_financial_impact",
    "calculate_expected_value",
    "calculate_expected_value_at_threshold",
    "find_optimal_threshold",
    "compare_thresholds",
    "generate_cost_benefit_report",
]
