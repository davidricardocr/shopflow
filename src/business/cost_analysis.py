"""
Business cost-benefit analysis for ShopFlow.

This module implements financial impact calculations based on the
challenge's cost structure:
- Return cost: $18
- Intervention cost: $3
- Intervention reduces return probability by 35%

Key metric: Expected Value (EV) per customer
"""

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from ..config import business_config


@dataclass
class PredictionOutcome:
    """
    Financial outcome of a prediction decision.
    
    Attributes:
        true_positives: Count of correctly predicted returns.
        false_positives: Count of incorrectly predicted returns.
        true_negatives: Count of correctly predicted non-returns.
        false_negatives: Count of missed returns.
    """
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    @property
    def total(self) -> int:
        """Total number of predictions."""
        return (
            self.true_positives + 
            self.false_positives + 
            self.true_negatives + 
            self.false_negatives
        )
    
    @property
    def precision(self) -> float:
        """Precision (positive predictive value)."""
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """Recall (sensitivity, true positive rate)."""
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator > 0 else 0.0
    
    @property
    def f1_score(self) -> float:
        """F1 score (harmonic mean of precision and recall)."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


def calculate_confusion_matrix_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> PredictionOutcome:
    """
    Calculate confusion matrix values from predictions.
    
    Args:
        y_true: Actual labels (0 or 1).
        y_pred: Predicted labels (0 or 1).
    
    Returns:
        PredictionOutcome with TP, FP, TN, FN counts.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    return PredictionOutcome(
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn
    )


def calculate_financial_impact(
    outcome: PredictionOutcome,
    config: Optional[business_config.__class__] = None
) -> Dict[str, float]:
    """
    Calculate the financial impact of model predictions.
    
    Based on the cost matrix:
    - True Positive: Save $15 (return prevented, cost $3 intervention)
    - False Positive: Lose $3 (wasted intervention)
    - True Negative: $0 (correct, no action needed)
    - False Negative: Lose $18 (missed return)
    
    Args:
        outcome: PredictionOutcome with confusion matrix values.
        config: Business configuration (uses default if None).
    
    Returns:
        Dictionary with financial metrics.
    
    Example:
        >>> outcome = PredictionOutcome(tp=100, fp=50, tn=800, fn=50)
        >>> impact = calculate_financial_impact(outcome)
        >>> print(f"Net savings: ${impact['net_savings']:.2f}")
    """
    config = config or business_config
    
    # Calculate individual impacts
    savings_from_tp = outcome.true_positives * config.savings_per_true_positive
    cost_from_fp = outcome.false_positives * config.intervention_cost
    cost_from_fn = outcome.false_negatives * config.return_cost
    
    # Net financial impact
    net_savings = savings_from_tp - cost_from_fp - cost_from_fn
    
    # Per-customer metrics
    total_customers = outcome.total
    expected_value_per_customer = net_savings / total_customers if total_customers > 0 else 0
    
    # Monthly projection (assuming test set represents monthly volume)
    monthly_savings = net_savings
    annual_savings = monthly_savings * 12
    
    return {
        "savings_from_true_positives": savings_from_tp,
        "cost_from_false_positives": cost_from_fp,
        "cost_from_false_negatives": cost_from_fn,
        "net_savings": net_savings,
        "expected_value_per_customer": expected_value_per_customer,
        "monthly_savings": monthly_savings,
        "annual_savings_projection": annual_savings,
        "total_customers": total_customers,
        "return_cost": config.return_cost,
        "intervention_cost": config.intervention_cost,
    }


def calculate_expected_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: Optional[business_config.__class__] = None
) -> float:
    """
    Calculate expected value per customer from predictions.
    
    Formula: EV = (TP_rate × $15) - (FP_rate × $3) - (FN_rate × $18)
    
    Args:
        y_true: Actual labels.
        y_pred: Predicted labels.
        config: Business configuration.
    
    Returns:
        Expected value per customer in dollars.
    """
    outcome = calculate_confusion_matrix_from_predictions(y_true, y_pred)
    impact = calculate_financial_impact(outcome, config)
    return impact["expected_value_per_customer"]


def calculate_expected_value_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    config: Optional[business_config.__class__] = None
) -> Dict[str, float]:
    """
    Calculate expected value at a specific probability threshold.
    
    Args:
        y_true: Actual labels.
        y_proba: Predicted probabilities for positive class.
        threshold: Decision threshold.
        config: Business configuration.
    
    Returns:
        Dictionary with metrics at this threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)
    outcome = calculate_confusion_matrix_from_predictions(y_true, y_pred)
    impact = calculate_financial_impact(outcome, config)
    
    return {
        "threshold": threshold,
        "precision": outcome.precision,
        "recall": outcome.recall,
        "f1_score": outcome.f1_score,
        "expected_value": impact["expected_value_per_customer"],
        "monthly_savings": impact["monthly_savings"],
        "true_positives": outcome.true_positives,
        "false_positives": outcome.false_positives,
        "false_negatives": outcome.false_negatives,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    metric: str = "expected_value",
    config: Optional[business_config.__class__] = None
) -> Tuple[float, pd.DataFrame]:
    """
    Find the optimal decision threshold by maximizing expected value.
    
    Args:
        y_true: Actual labels.
        y_proba: Predicted probabilities.
        thresholds: Array of thresholds to evaluate. Defaults to 0.01-0.99.
        metric: Metric to optimize ('expected_value', 'f1_score').
        config: Business configuration.
    
    Returns:
        Tuple of (optimal_threshold, DataFrame with all threshold results).
    
    Example:
        >>> optimal_th, results_df = find_optimal_threshold(y_true, y_proba)
        >>> print(f"Optimal threshold: {optimal_th:.2f}")
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)
    
    results = []
    for threshold in thresholds:
        metrics = calculate_expected_value_at_threshold(
            y_true, y_proba, threshold, config
        )
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    
    # Find optimal threshold
    optimal_idx = results_df[metric].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, "threshold"]
    
    logger.info(
        f"Optimal threshold: {optimal_threshold:.2f} "
        f"(EV: ${results_df.loc[optimal_idx, 'expected_value']:.2f})"
    )
    
    return optimal_threshold, results_df


def compare_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: List[float] = [0.30, 0.42, 0.50, 0.60],
    config: Optional[business_config.__class__] = None
) -> pd.DataFrame:
    """
    Compare performance at specific thresholds.
    
    Useful for presenting threshold analysis in reports.
    
    Args:
        y_true: Actual labels.
        y_proba: Predicted probabilities.
        thresholds: List of thresholds to compare.
        config: Business configuration.
    
    Returns:
        DataFrame with comparison.
    """
    results = []
    for th in thresholds:
        metrics = calculate_expected_value_at_threshold(y_true, y_proba, th, config)
        results.append(metrics)
    
    return pd.DataFrame(results)


def generate_cost_benefit_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> str:
    """
    Generate a formatted cost-benefit report.
    
    Args:
        y_true: Actual labels.
        y_pred: Predicted labels.
        model_name: Name for the report header.
    
    Returns:
        Formatted string report.
    """
    outcome = calculate_confusion_matrix_from_predictions(y_true, y_pred)
    impact = calculate_financial_impact(outcome)
    
    report = f"""
{'=' * 60}
COST-BENEFIT ANALYSIS: {model_name}
{'=' * 60}

CONFUSION MATRIX:
  True Positives:  {outcome.true_positives:,} (Returns correctly predicted)
  False Positives: {outcome.false_positives:,} (Wasted interventions)
  True Negatives:  {outcome.true_negatives:,} (Correct non-return predictions)
  False Negatives: {outcome.false_negatives:,} (Missed returns)

PERFORMANCE METRICS:
  Precision: {outcome.precision:.4f}
  Recall:    {outcome.recall:.4f}
  F1 Score:  {outcome.f1_score:.4f}

FINANCIAL IMPACT:
  Savings from prevented returns (TP × $15): ${impact['savings_from_true_positives']:,.2f}
  Cost of wasted interventions (FP × $3):    -${impact['cost_from_false_positives']:,.2f}
  Cost of missed returns (FN × $18):         -${impact['cost_from_false_negatives']:,.2f}
  {'─' * 50}
  NET SAVINGS:                               ${impact['net_savings']:,.2f}

EXPECTED VALUE:
  Per Customer: ${impact['expected_value_per_customer']:.2f}
  Monthly:      ${impact['monthly_savings']:,.2f}
  Annual:       ${impact['annual_savings_projection']:,.2f}

{'=' * 60}
"""
    return report
