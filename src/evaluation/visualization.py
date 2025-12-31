"""
Visualization utilities for ShopFlow.

This module provides plotting functions for model evaluation,
threshold analysis, and business impact visualization.
"""

from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

from src.config import business_config


# Set style
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "success": "#28A745",
    "warning": "#FFC107",
    "danger": "#DC3545",
    "neutral": "#6C757D",
}


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    labels: List[str] = ["Kept", "Returned"],
    figsize: Tuple[int, int] = (8, 6),
    show_financial_impact: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot confusion matrix with optional financial annotations.
    
    Args:
        y_true: Actual labels.
        y_pred: Predicted labels.
        title: Plot title.
        labels: Class labels.
        figsize: Figure size.
        show_financial_impact: Add financial impact annotations.
        ax: Matplotlib axes (creates new if None).
    
    Returns:
        Matplotlib Figure object.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Count"}
    )
    
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    # Add financial impact annotations
    if show_financial_impact:
        tn, fp, fn, tp = cm.ravel()
        
        annotations = [
            f"TN: ${0:,}",                                    # True Negative
            f"FP: -${fp * business_config.intervention_cost:,.0f}",  # False Positive
            f"FN: -${fn * business_config.return_cost:,.0f}",        # False Negative
            f"TP: +${tp * business_config.savings_per_true_positive:,.0f}",  # True Positive
        ]
        
        # Add text below the matrix
        net = (tp * business_config.savings_per_true_positive - 
               fp * business_config.intervention_cost - 
               fn * business_config.return_cost)
        
        ax.text(
            0.5, -0.15,
            f"Net Financial Impact: ${net:,.0f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=COLORS["success"] if net > 0 else COLORS["danger"]
        )
    
    plt.tight_layout()
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 6),
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot ROC curve with AUC score.
    
    Args:
        y_true: Actual labels.
        y_proba: Predicted probabilities.
        title: Plot title.
        figsize: Figure size.
        ax: Matplotlib axes.
    
    Returns:
        Matplotlib Figure object.
    """
    from sklearn.metrics import roc_auc_score
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    ax.plot(fpr, tpr, color=COLORS["primary"], lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=COLORS["neutral"], lw=1, linestyle="--", label="Random")
    
    ax.fill_between(fpr, tpr, alpha=0.2, color=COLORS["primary"])
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Precision-Recall Curve",
    figsize: Tuple[int, int] = (8, 6),
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: Actual labels.
        y_proba: Predicted probabilities.
        title: Plot title.
        figsize: Figure size.
        ax: Matplotlib axes.
    
    Returns:
        Matplotlib Figure object.
    """
    from sklearn.metrics import average_precision_score
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    ax.plot(recall, precision, color=COLORS["secondary"], lw=2, label=f"PR (AP = {ap:.3f})")
    ax.fill_between(recall, precision, alpha=0.2, color=COLORS["secondary"])
    
    # Baseline (random classifier)
    baseline = y_true.mean()
    ax.axhline(y=baseline, color=COLORS["neutral"], linestyle="--", label=f"Baseline ({baseline:.2f})")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    return fig


def plot_threshold_analysis(
    results_df: pd.DataFrame,
    optimal_threshold: Optional[float] = None,
    title: str = "Threshold Analysis",
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot metrics across different thresholds.
    
    Args:
        results_df: DataFrame with threshold analysis results.
        optimal_threshold: Highlight this threshold.
        title: Plot title.
        figsize: Figure size.
    
    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Precision, Recall, F1
    ax1 = axes[0]
    ax1.plot(results_df["threshold"], results_df["precision"], label="Precision", color=COLORS["primary"], lw=2)
    ax1.plot(results_df["threshold"], results_df["recall"], label="Recall", color=COLORS["secondary"], lw=2)
    ax1.plot(results_df["threshold"], results_df["f1_score"], label="F1 Score", color=COLORS["success"], lw=2)
    
    if optimal_threshold:
        ax1.axvline(x=optimal_threshold, color=COLORS["warning"], linestyle="--", lw=2, label=f"Optimal ({optimal_threshold:.2f})")
    
    ax1.set_xlabel("Threshold", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title("Classification Metrics vs Threshold", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Expected Value
    ax2 = axes[1]
    ax2.plot(results_df["threshold"], results_df["expected_value"], color=COLORS["success"], lw=2)
    ax2.fill_between(
        results_df["threshold"],
        results_df["expected_value"],
        alpha=0.3,
        color=COLORS["success"]
    )
    
    if optimal_threshold:
        ax2.axvline(x=optimal_threshold, color=COLORS["warning"], linestyle="--", lw=2, label=f"Optimal ({optimal_threshold:.2f})")
        
        # Mark optimal point
        opt_idx = (results_df["threshold"] - optimal_threshold).abs().idxmin()
        opt_ev = results_df.loc[opt_idx, "expected_value"]
        ax2.scatter([optimal_threshold], [opt_ev], color=COLORS["warning"], s=100, zorder=5)
        ax2.annotate(
            f"${opt_ev:.2f}",
            (optimal_threshold, opt_ev),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold"
        )
    
    ax2.axhline(y=2.0, color=COLORS["danger"], linestyle=":", label="Min Deployment ($2.00)")
    
    ax2.set_xlabel("Threshold", fontsize=12)
    ax2.set_ylabel("Expected Value ($/customer)", fontsize=12)
    ax2.set_title("Expected Value vs Threshold", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_category_performance(
    category_metrics: pd.DataFrame,
    metric: str = "expected_value",
    title: str = "Performance by Category",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot performance metrics by product category.
    
    Args:
        category_metrics: DataFrame from evaluate_by_category.
        metric: Metric to visualize.
        title: Plot title.
        figsize: Figure size.
    
    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Return rate by category
    ax1 = axes[0]
    colors = [COLORS["danger"] if r > 0.2 else COLORS["warning"] if r > 0.15 else COLORS["success"] 
              for r in category_metrics["return_rate"]]
    
    bars = ax1.barh(category_metrics["category"], category_metrics["return_rate"], color=colors)
    ax1.set_xlabel("Return Rate", fontsize=12)
    ax1.set_title("Return Rate by Category", fontsize=12, fontweight="bold")
    ax1.axvline(x=0.15, color=COLORS["neutral"], linestyle="--", label="Overall avg")
    
    # Add percentage labels
    for bar, rate in zip(bars, category_metrics["return_rate"]):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{rate:.1%}", va="center", fontsize=10)
    
    # Right: Expected value by category
    ax2 = axes[1]
    colors = [COLORS["success"] if ev > 2 else COLORS["warning"] if ev > 0 else COLORS["danger"]
              for ev in category_metrics["expected_value"]]
    
    bars = ax2.barh(category_metrics["category"], category_metrics["expected_value"], color=colors)
    ax2.set_xlabel("Expected Value ($/customer)", fontsize=12)
    ax2.set_title("Expected Value by Category", fontsize=12, fontweight="bold")
    ax2.axvline(x=2.0, color=COLORS["neutral"], linestyle="--", label="Min deployment")
    
    # Add value labels
    for bar, ev in zip(bars, category_metrics["expected_value"]):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f"${ev:.2f}", va="center", fontsize=10)
    
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metrics: List[str] = ["accuracy", "precision", "recall", "f1_score", "expected_value"],
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot comparison of multiple models.
    
    Args:
        comparison_df: DataFrame from compare_models.
        metrics: Metrics to compare.
        title: Plot title.
        figsize: Figure size.
    
    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Classification metrics (bar chart)
    ax1 = axes[0]
    classification_metrics = [m for m in metrics if m != "expected_value"]
    
    x = np.arange(len(classification_metrics))
    width = 0.35
    
    for i, (_, row) in enumerate(comparison_df.iterrows()):
        offset = (i - len(comparison_df)/2 + 0.5) * width
        values = [row[m] for m in classification_metrics]
        ax1.bar(x + offset, values, width, label=row["model"])
    
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(classification_metrics, rotation=45)
    ax1.set_title("Classification Metrics", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Right: Expected value (horizontal bar)
    ax2 = axes[1]
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["success"]][:len(comparison_df)]
    
    bars = ax2.barh(comparison_df["model"], comparison_df["expected_value"], color=colors)
    ax2.set_xlabel("Expected Value ($/customer)", fontsize=12)
    ax2.set_title("Business Impact", fontsize=12, fontweight="bold")
    ax2.axvline(x=2.0, color=COLORS["danger"], linestyle="--", label="Min deployment")
    
    # Add value labels
    for bar, ev in zip(bars, comparison_df["expected_value"]):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f"${ev:.2f}", va="center", fontsize=10, fontweight="bold")
    
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def create_evaluation_dashboard(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    results_df: Optional[pd.DataFrame] = None,
    optimal_threshold: Optional[float] = None,
    title: str = "Model Evaluation Dashboard",
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create a comprehensive evaluation dashboard.
    
    Args:
        y_true: Actual labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities.
        results_df: Threshold analysis results.
        optimal_threshold: Optimal threshold value.
        title: Dashboard title.
        figsize: Figure size.
    
    Returns:
        Matplotlib Figure object.
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    plot_confusion_matrix(y_true, y_pred, ax=ax1)
    
    # ROC Curve
    ax2 = fig.add_subplot(gs[0, 1])
    plot_roc_curve(y_true, y_proba, ax=ax2)
    
    # PR Curve
    ax3 = fig.add_subplot(gs[0, 2])
    plot_precision_recall_curve(y_true, y_proba, ax=ax3)
    
    # Threshold analysis (if provided)
    if results_df is not None:
        ax4 = fig.add_subplot(gs[1, :2])
        ax4.plot(results_df["threshold"], results_df["precision"], label="Precision", lw=2)
        ax4.plot(results_df["threshold"], results_df["recall"], label="Recall", lw=2)
        ax4.plot(results_df["threshold"], results_df["f1_score"], label="F1", lw=2)
        
        if optimal_threshold:
            ax4.axvline(x=optimal_threshold, color="red", linestyle="--", label=f"Optimal ({optimal_threshold:.2f})")
        
        ax4.set_xlabel("Threshold")
        ax4.set_ylabel("Score")
        ax4.set_title("Metrics vs Threshold")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Expected Value
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(results_df["threshold"], results_df["expected_value"], color="green", lw=2)
        ax5.axhline(y=2.0, color="red", linestyle="--")
        
        if optimal_threshold:
            ax5.axvline(x=optimal_threshold, color="red", linestyle="--")
        
        ax5.set_xlabel("Threshold")
        ax5.set_ylabel("Expected Value ($)")
        ax5.set_title("Expected Value vs Threshold")
        ax5.grid(True, alpha=0.3)
    
    # Summary metrics text
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis("off")
    
    from src.evaluation.metrics import evaluate_model
    metrics = evaluate_model(y_true, y_pred, y_proba)
    
    summary_text = (
        f"SUMMARY METRICS\n"
        f"{'─' * 50}\n"
        f"Accuracy: {metrics['accuracy']:.4f}    "
        f"Precision: {metrics['precision']:.4f}    "
        f"Recall: {metrics['recall']:.4f}    "
        f"F1: {metrics['f1_score']:.4f}\n"
        f"ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}    "
        f"Expected Value: ${metrics['expected_value']:.2f}/customer    "
        f"Net Savings: ${metrics['net_savings']:,.0f}"
    )
    
    ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes,
             fontsize=12, ha="center", va="center", family="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    
    return fig
