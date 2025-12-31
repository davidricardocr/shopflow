"""
Baseline logistic regression model for ShopFlow.

This module replicates the baseline model from the challenge,
wrapped in our BaseModel interface for consistent evaluation.
"""

from typing import Optional, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from loguru import logger

from src.config import model_config
from src.models.base import BaseModel, SklearnModelWrapper


class BaselineLogisticRegression(SklearnModelWrapper):
    """
    Baseline logistic regression matching the challenge's starting point.
    
    This model replicates the exact configuration from baseline_model.py:
    - LogisticRegression with max_iter=1000
    - random_state=42
    - Default solver (lbfgs)
    
    Use this as the benchmark for all model improvements.
    """
    
    def __init__(self, name: str = "Baseline Logistic Regression"):
        """
        Initialize the baseline model.
        
        Args:
            name: Model name for logging and saving.
        """
        super().__init__(
            model=LogisticRegression,
            name=name,
            max_iter=1000,
            random_state=model_config.random_state,
        )
        
        self.metadata["model_type"] = "baseline"
        self.metadata["description"] = "Exact replica of challenge baseline"


class OptimizedLogisticRegression(SklearnModelWrapper):
    """
    Optimized logistic regression with tunable hyperparameters.
    
    Improvements over baseline:
    - Class weight balancing for imbalanced data
    - Regularization strength tuning (C parameter)
    - Different solvers for optimization
    """
    
    def __init__(
        self,
        name: str = "Optimized Logistic Regression",
        C: float = 1.0,
        class_weight: Optional[str] = "balanced",
        solver: str = "lbfgs",
        max_iter: int = 1000,
    ):
        """
        Initialize the optimized model.
        
        Args:
            name: Model name.
            C: Inverse regularization strength. Smaller = stronger regularization.
            class_weight: 'balanced' to handle class imbalance, None for default.
            solver: Optimization algorithm.
            max_iter: Maximum iterations for solver.
        """
        super().__init__(
            model=LogisticRegression,
            name=name,
            C=C,
            class_weight=class_weight,
            solver=solver,
            max_iter=max_iter,
            random_state=model_config.random_state,
        )
        
        self.metadata["model_type"] = "optimized"
        self.metadata["improvements"] = [
            "class_weight=balanced for handling imbalance",
            f"C={C} for regularization control",
        ]


def create_baseline_model() -> BaselineLogisticRegression:
    """
    Factory function to create the baseline model.
    
    Returns:
        Configured baseline logistic regression model.
    
    Example:
        >>> model = create_baseline_model()
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    return BaselineLogisticRegression()


def create_optimized_lr(
    C: float = 1.0,
    class_weight: str = "balanced"
) -> OptimizedLogisticRegression:
    """
    Factory function to create an optimized logistic regression.
    
    Args:
        C: Regularization strength.
        class_weight: Class weighting strategy.
    
    Returns:
        Configured optimized logistic regression model.
    """
    return OptimizedLogisticRegression(C=C, class_weight=class_weight)
