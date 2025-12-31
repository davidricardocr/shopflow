"""
Evaluation metrics for ShopFlow.

This module provides comprehensive evaluation functions beyond standard
sklearn metrics, including business-aligned metrics and segment analysis.
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
from loguru import logger

from src.config import data_config, business_config
from src.business.cost_analysis import (
    calculate_expected_value,
    calculate_confusion_matrix_from_predictions,
    calculate_financial_impact,
)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    include_business_metrics: bool = True
) -> Dict[str, float]:
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        y_true: Actual labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (optional, for AUC metrics).
        include_business_metrics: Whether to include EV calculation.
    
    Returns:
        Dictionary of metric names to values.
    
    Example:
        >>> metrics = evaluate_model(y_true, y_pred, y_proba)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
        >>> print(f"Expected Value: ${metrics['expected_value']:.2f}")
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Add probability-based metrics if available
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["average_precision"] = average_precision_score(y_true, y_proba)
    
    # Add business metrics
    if include_business_metrics:
        outcome = calculate_confusion_matrix_from_predictions(y_true, y_pred)
        impact = calculate_financial_impact(outcome)
        metrics["expected_value"] = impact["expected_value_per_customer"]
        metrics["net_savings"] = impact["net_savings"]
    
    return metrics


def evaluate_by_category(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    category_column: str = "product_category"
) -> pd.DataFrame:
    """
    Evaluate model performance by product category.
    
    This is crucial for identifying failure modes - the challenge mentions
    Fashion has a 30% return rate vs 15% overall.
    
    Args:
        df: Original DataFrame with category information.
        y_true: Actual labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities.
        category_column: Name of the category column.
    
    Returns:
        DataFrame with metrics per category.
    
    Example:
        >>> category_metrics = evaluate_by_category(test_df, y_true, y_pred)
        >>> print(category_metrics)
    """
    results = []
    
    for category in df[category_column].unique():
        mask = df[category_column] == category
        
        if mask.sum() == 0:
            continue
        
        y_true_cat = y_true[mask]
        y_pred_cat = y_pred[mask]
        
        metrics = {
            "category": category,
            "n_samples": int(mask.sum()),
            "return_rate": float(y_true_cat.mean()),
            "accuracy": accuracy_score(y_true_cat, y_pred_cat),
            "precision": precision_score(y_true_cat, y_pred_cat, zero_division=0),
            "recall": recall_score(y_true_cat, y_pred_cat, zero_division=0),
            "f1_score": f1_score(y_true_cat, y_pred_cat, zero_division=0),
        }
        
        if y_proba is not None:
            y_proba_cat = y_proba[mask]
            try:
                metrics["roc_auc"] = roc_auc_score(y_true_cat, y_proba_cat)
            except ValueError:
                metrics["roc_auc"] = np.nan
        
        # Business metric
        metrics["expected_value"] = calculate_expected_value(y_true_cat, y_pred_cat)
        
        results.append(metrics)
    
    return pd.DataFrame(results).sort_values("return_rate", ascending=False)


def get_confusion_matrix_details(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Get detailed confusion matrix with interpretations.
    
    Args:
        y_true: Actual labels.
        y_pred: Predicted labels.
    
    Returns:
        Dictionary with confusion matrix and interpretations.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    total = len(y_true)
    actual_positives = int(np.sum(y_true))
    actual_negatives = total - actual_positives
    
    return {
        "confusion_matrix": cm,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_samples": total,
        "actual_positives": actual_positives,
        "actual_negatives": actual_negatives,
        "positive_rate": actual_positives / total,
        "interpretation": {
            "TP": f"{tp} returns correctly identified → Save ${tp * 15:,.0f}",
            "FP": f"{fp} false alarms → Waste ${fp * 3:,.0f} on interventions",
            "TN": f"{tn} correctly identified as non-returns → No cost",
            "FN": f"{fn} missed returns → Lose ${fn * 18:,.0f}",
        }
    }


def compare_models(
    y_true: np.ndarray,
    predictions: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
) -> pd.DataFrame:
    """
    Compare multiple models side by side.
    
    Args:
        y_true: Actual labels.
        predictions: Dictionary of model_name -> (y_pred, y_proba).
    
    Returns:
        DataFrame comparing all models.
    
    Example:
        >>> predictions = {
        ...     "Baseline": (y_pred_baseline, y_proba_baseline),
        ...     "Improved": (y_pred_improved, y_proba_improved),
        ... }
        >>> comparison = compare_models(y_true, predictions)
    """
    results = []
    
    for model_name, (y_pred, y_proba) in predictions.items():
        metrics = evaluate_model(y_true, y_pred, y_proba)
        metrics["model"] = model_name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ["model"] + [c for c in df.columns if c != "model"]
    return df[cols]


def identify_failure_modes(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_columns: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Identify where the model fails most.
    
    Analyzes false positives and false negatives to find patterns.
    
    Args:
        df: Original DataFrame with features.
        y_true: Actual labels.
        y_pred: Predicted labels.
        feature_columns: Columns to analyze.
    
    Returns:
        Dictionary with 'false_positives' and 'false_negatives' DataFrames.
    """
    df = df.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    
    # Identify misclassifications
    false_positives = df[(df["y_true"] == 0) & (df["y_pred"] == 1)]
    false_negatives = df[(df["y_true"] == 1) & (df["y_pred"] == 0)]
    
    feature_columns = feature_columns or (
        data_config.numeric_features + 
        data_config.categorical_features
    )
    
    # Analyze patterns
    fp_analysis = {}
    fn_analysis = {}
    
    for col in feature_columns:
        if col in df.columns:
            if df[col].dtype in ["object", "category"]:
                # Categorical: count distribution
                fp_analysis[col] = false_positives[col].value_counts().to_dict()
                fn_analysis[col] = false_negatives[col].value_counts().to_dict()
            else:
                # Numeric: summary stats
                fp_analysis[col] = false_positives[col].describe().to_dict()
                fn_analysis[col] = false_negatives[col].describe().to_dict()
    
    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "fp_count": len(false_positives),
        "fn_count": len(false_negatives),
        "fp_analysis": fp_analysis,
        "fn_analysis": fn_analysis,
    }


def format_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = ["Kept", "Returned"]
) -> str:
    """
    Generate a formatted classification report.
    
    Args:
        y_true: Actual labels.
        y_pred: Predicted labels.
        target_names: Names for the classes.
    
    Returns:
        Formatted classification report string.
    """
    return classification_report(
        y_true, 
        y_pred, 
        target_names=target_names,
        digits=4
    )


def get_curve_data(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """
    Get data for ROC and Precision-Recall curves.
    
    Args:
        y_true: Actual labels.
        y_proba: Predicted probabilities.
    
    Returns:
        Dictionary with 'roc' and 'pr' curve data.
    """
    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    
    # Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    
    return {
        "roc": (fpr, tpr, roc_thresholds),
        "pr": (precision, recall, pr_thresholds),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
    }
