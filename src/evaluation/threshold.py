"""
Threshold optimization for ShopFlow.

This module provides utilities for finding and analyzing optimal
decision thresholds based on business objectives.
"""

from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from loguru import logger

from src.config import business_config, deployment_criteria
from src.business.cost_analysis import (
    calculate_expected_value_at_threshold,
    find_optimal_threshold as find_optimal_ev_threshold,
)


class ThresholdOptimizer:
    """
    Optimizer for finding the best decision threshold.
    
    Supports multiple optimization objectives:
    - Expected Value (business profit)
    - F1 Score (balanced precision/recall)
    - Custom metrics
    
    Attributes:
        y_true: Ground truth labels.
        y_proba: Predicted probabilities.
        results_df: DataFrame with metrics at each threshold.
        optimal_threshold: Best threshold found.
    """
    
    def __init__(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ):
        """
        Initialize the optimizer.
        
        Args:
            y_true: Actual labels.
            y_proba: Predicted probabilities for positive class.
            thresholds: Thresholds to evaluate. Defaults to 0.01-0.99.
        """
        self.y_true = np.array(y_true)
        self.y_proba = np.array(y_proba)
        self.thresholds = thresholds if thresholds is not None else np.arange(0.05, 0.95, 0.01)
        
        self.results_df: Optional[pd.DataFrame] = None
        self.optimal_threshold: Optional[float] = None
        self._is_analyzed = False
    
    def analyze(self) -> "ThresholdOptimizer":
        """
        Analyze all thresholds and compute metrics.
        
        Returns:
            self
        """
        results = []
        
        for threshold in self.thresholds:
            metrics = calculate_expected_value_at_threshold(
                self.y_true,
                self.y_proba,
                threshold
            )
            results.append(metrics)
        
        self.results_df = pd.DataFrame(results)
        self._is_analyzed = True
        
        logger.info(f"Analyzed {len(self.thresholds)} thresholds")
        return self
    
    def find_optimal(
        self,
        metric: str = "expected_value",
        constraints: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Find the optimal threshold for a given metric.
        
        Args:
            metric: Metric to optimize ('expected_value', 'f1_score', 'recall').
            constraints: Optional constraints like {'precision': 0.65}.
        
        Returns:
            Optimal threshold value.
        """
        if not self._is_analyzed:
            self.analyze()
        
        df = self.results_df.copy()
        
        # Apply constraints
        if constraints:
            for constraint_metric, min_value in constraints.items():
                df = df[df[constraint_metric] >= min_value]
        
        if len(df) == 0:
            logger.warning("No thresholds satisfy the constraints")
            return 0.5
        
        # Find optimal
        optimal_idx = df[metric].idxmax()
        self.optimal_threshold = df.loc[optimal_idx, "threshold"]
        
        logger.info(
            f"Optimal threshold for {metric}: {self.optimal_threshold:.2f} "
            f"(value: {df.loc[optimal_idx, metric]:.4f})"
        )
        
        return self.optimal_threshold
    
    def find_business_optimal(self) -> float:
        """
        Find threshold optimizing expected value with deployment constraints.
        
        Applies the deployment criteria from the challenge:
        - EV > $2.00/customer
        - Precision > 0.65
        - Recall > 0.50
        
        Returns:
            Optimal threshold satisfying business constraints.
        """
        return self.find_optimal(
            metric="expected_value",
            constraints={
                "precision": deployment_criteria.min_precision,
                "recall": deployment_criteria.min_recall,
            }
        )
    
    def get_metrics_at_threshold(self, threshold: float) -> Dict[str, float]:
        """
        Get all metrics at a specific threshold.
        
        Args:
            threshold: Threshold to evaluate.
        
        Returns:
            Dictionary of metrics.
        """
        if not self._is_analyzed:
            self.analyze()
        
        # Find closest threshold
        idx = (self.results_df["threshold"] - threshold).abs().idxmin()
        return self.results_df.loc[idx].to_dict()
    
    def compare_thresholds(
        self,
        thresholds: List[float] = [0.3, 0.4, 0.5, 0.6]
    ) -> pd.DataFrame:
        """
        Compare specific thresholds side by side.
        
        Args:
            thresholds: List of thresholds to compare.
        
        Returns:
            DataFrame with comparison.
        """
        if not self._is_analyzed:
            self.analyze()
        
        results = []
        for th in thresholds:
            metrics = self.get_metrics_at_threshold(th)
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def check_deployment_readiness(
        self,
        threshold: Optional[float] = None
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Check if model meets deployment criteria at given threshold.
        
        Args:
            threshold: Threshold to check. Uses optimal if None.
        
        Returns:
            Tuple of (is_ready, individual_checks).
        """
        threshold = threshold or self.optimal_threshold or 0.5
        metrics = self.get_metrics_at_threshold(threshold)
        
        checks = {
            "expected_value_ok": metrics["expected_value"] >= deployment_criteria.min_expected_value_per_customer,
            "precision_ok": metrics["precision"] >= deployment_criteria.min_precision,
            "recall_ok": metrics["recall"] >= deployment_criteria.min_recall,
        }
        
        is_ready = all(checks.values())
        
        return is_ready, checks
    
    def generate_report(self) -> str:
        """
        Generate a threshold analysis report.
        
        Returns:
            Formatted report string.
        """
        if not self._is_analyzed:
            self.analyze()
        
        # Find key thresholds
        ev_optimal = self.find_optimal("expected_value")
        f1_optimal = self.find_optimal("f1_score")
        
        ev_metrics = self.get_metrics_at_threshold(ev_optimal)
        f1_metrics = self.get_metrics_at_threshold(f1_optimal)
        default_metrics = self.get_metrics_at_threshold(0.5)
        
        is_ready, checks = self.check_deployment_readiness(ev_optimal)
        
        report = f"""
{'=' * 60}
THRESHOLD OPTIMIZATION REPORT
{'=' * 60}

DEFAULT THRESHOLD (0.50):
  Precision: {default_metrics['precision']:.4f}
  Recall:    {default_metrics['recall']:.4f}
  F1 Score:  {default_metrics['f1_score']:.4f}
  Expected Value: ${default_metrics['expected_value']:.2f}/customer

OPTIMAL FOR EXPECTED VALUE ({ev_optimal:.2f}):
  Precision: {ev_metrics['precision']:.4f}
  Recall:    {ev_metrics['recall']:.4f}
  F1 Score:  {ev_metrics['f1_score']:.4f}
  Expected Value: ${ev_metrics['expected_value']:.2f}/customer
  Monthly Savings: ${ev_metrics['monthly_savings']:,.2f}

OPTIMAL FOR F1 SCORE ({f1_optimal:.2f}):
  Precision: {f1_metrics['precision']:.4f}
  Recall:    {f1_metrics['recall']:.4f}
  F1 Score:  {f1_metrics['f1_score']:.4f}
  Expected Value: ${f1_metrics['expected_value']:.2f}/customer

DEPLOYMENT READINESS (at EV-optimal threshold):
  ✓ EV > $2.00:       {'PASS' if checks['expected_value_ok'] else 'FAIL'}
  ✓ Precision > 0.65: {'PASS' if checks['precision_ok'] else 'FAIL'}
  ✓ Recall > 0.50:    {'PASS' if checks['recall_ok'] else 'FAIL'}
  
  OVERALL: {'✅ READY FOR DEPLOYMENT' if is_ready else '❌ NOT READY'}

{'=' * 60}
"""
        return report


def find_threshold_for_target_metric(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_metric: str,
    target_value: float,
    tolerance: float = 0.01
) -> Optional[float]:
    """
    Find threshold that achieves a target metric value.
    
    Useful for setting thresholds like "I need at least 70% recall".
    
    Args:
        y_true: Actual labels.
        y_proba: Predicted probabilities.
        target_metric: Metric to target ('precision', 'recall').
        target_value: Desired value for the metric.
        tolerance: Acceptable deviation from target.
    
    Returns:
        Threshold achieving the target, or None if not possible.
    """
    optimizer = ThresholdOptimizer(y_true, y_proba)
    optimizer.analyze()
    
    # Find thresholds meeting the target
    df = optimizer.results_df
    candidates = df[
        (df[target_metric] >= target_value - tolerance) &
        (df[target_metric] <= target_value + tolerance)
    ]
    
    if len(candidates) == 0:
        # Find closest
        idx = (df[target_metric] - target_value).abs().idxmin()
        logger.warning(
            f"Could not achieve {target_metric}={target_value}. "
            f"Closest: {df.loc[idx, target_metric]:.4f} at threshold {df.loc[idx, 'threshold']:.2f}"
        )
        return df.loc[idx, "threshold"]
    
    # Among candidates, pick the one with best expected value
    best_idx = candidates["expected_value"].idxmax()
    return candidates.loc[best_idx, "threshold"]
