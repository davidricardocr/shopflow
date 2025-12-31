"""
Improved models for ShopFlow.

This module contains model implementations that aim to improve
upon the baseline logistic regression through:
- Different algorithms (Random Forest, XGBoost, LightGBM)
- Ensemble methods
- Hyperparameter optimization with Optuna
"""

from typing import Optional, Dict, Any, List, Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from loguru import logger

from src.config import model_config
from src.models.base import BaseModel, SklearnModelWrapper


class RandomForestModel(SklearnModelWrapper):
    """
    Random Forest classifier for returns prediction.
    
    Random Forest is useful for:
    - Handling non-linear relationships
    - Feature importance analysis
    - Robustness to overfitting
    """
    
    def __init__(
        self,
        name: str = "Random Forest",
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        class_weight: Optional[str] = "balanced",
        **kwargs
    ):
        """
        Initialize Random Forest model.
        
        Args:
            name: Model name.
            n_estimators: Number of trees.
            max_depth: Maximum depth of trees (None for unlimited).
            min_samples_split: Minimum samples to split a node.
            min_samples_leaf: Minimum samples in a leaf node.
            class_weight: 'balanced' for handling class imbalance.
            **kwargs: Additional parameters.
        """
        super().__init__(
            model=RandomForestClassifier,
            name=name,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=-1,  # Use all cores
            random_state=model_config.random_state,
            **kwargs
        )
        
        self.metadata["model_type"] = "random_forest"
        self.metadata["improvements"] = [
            "Non-linear decision boundaries",
            "Built-in feature importance",
            "Reduced overfitting through bagging",
        ]


class GradientBoostingModel(SklearnModelWrapper):
    """
    Gradient Boosting classifier for returns prediction.
    
    Gradient Boosting provides:
    - Strong predictive performance
    - Handles complex interactions
    - Good for tabular data
    """
    
    def __init__(
        self,
        name: str = "Gradient Boosting",
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 5,
        subsample: float = 0.8,
        **kwargs
    ):
        """
        Initialize Gradient Boosting model.
        
        Args:
            name: Model name.
            n_estimators: Number of boosting stages.
            learning_rate: Shrinkage rate.
            max_depth: Maximum depth of trees.
            min_samples_split: Minimum samples to split.
            subsample: Fraction of samples for each tree.
            **kwargs: Additional parameters.
        """
        super().__init__(
            model=GradientBoostingClassifier,
            name=name,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            subsample=subsample,
            random_state=model_config.random_state,
            **kwargs
        )
        
        self.metadata["model_type"] = "gradient_boosting"
        self.metadata["improvements"] = [
            "Sequential error correction",
            "Handles complex patterns",
            "Regularization through learning rate",
        ]


class XGBoostModel(BaseModel):
    """
    XGBoost classifier for returns prediction.
    
    XGBoost is a state-of-the-art gradient boosting library that provides:
    - Regularization to prevent overfitting
    - Handling of missing values
    - Fast training with parallel processing
    """
    
    def __init__(
        self,
        name: str = "XGBoost",
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        scale_pos_weight: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize XGBoost model.
        
        Args:
            name: Model name.
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Boosting learning rate.
            subsample: Row sampling ratio.
            colsample_bytree: Column sampling ratio per tree.
            reg_alpha: L1 regularization.
            reg_lambda: L2 regularization.
            scale_pos_weight: Balance between classes.
            **kwargs: Additional XGBoost parameters.
        """
        super().__init__(name)
        
        try:
            import xgboost as xgb
            self._xgb = xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": model_config.random_state,
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "verbosity": 0,
            **kwargs
        }
        
        if scale_pos_weight is not None:
            self.params["scale_pos_weight"] = scale_pos_weight
        
        self.model = self._xgb.XGBClassifier(**self.params)
        
        self.metadata["model_type"] = "xgboost"
        self.metadata["model_params"] = self.params
        self.metadata["improvements"] = [
            "L1/L2 regularization",
            "Column subsampling",
            "Native missing value handling",
        ]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        """Train the XGBoost model."""
        logger.info(f"Training {self.name}...")
        
        # Calculate scale_pos_weight if not set
        if "scale_pos_weight" not in self.params:
            neg_count = np.sum(y == 0)
            pos_count = np.sum(y == 1)
            self.model.set_params(scale_pos_weight=neg_count / pos_count)
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        self.metadata["n_samples"] = len(y)
        self.metadata["n_features"] = X.shape[1]
        
        logger.info(f"{self.name} trained successfully")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make binary predictions."""
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted first")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability estimates."""
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted first")
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importances(self) -> np.ndarray:
        """Get feature importances from XGBoost."""
        return self.model.feature_importances_


class LightGBMModel(BaseModel):
    """
    LightGBM classifier for returns prediction.
    
    LightGBM is optimized for:
    - Fast training speed
    - Lower memory usage
    - Better accuracy on large datasets
    """
    
    def __init__(
        self,
        name: str = "LightGBM",
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        class_weight: Optional[str] = "balanced",
        **kwargs
    ):
        """
        Initialize LightGBM model.
        
        Args:
            name: Model name.
            n_estimators: Number of boosting iterations.
            max_depth: Maximum tree depth (-1 for no limit).
            learning_rate: Boosting learning rate.
            num_leaves: Maximum number of leaves per tree.
            subsample: Row sampling ratio.
            colsample_bytree: Column sampling ratio.
            reg_alpha: L1 regularization.
            reg_lambda: L2 regularization.
            class_weight: 'balanced' for handling imbalance.
            **kwargs: Additional LightGBM parameters.
        """
        super().__init__(name)
        
        try:
            import lightgbm as lgb
            self._lgb = lgb
        except ImportError:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "class_weight": class_weight,
            "random_state": model_config.random_state,
            "verbosity": -1,
            **kwargs
        }
        
        self.model = self._lgb.LGBMClassifier(**self.params)
        
        self.metadata["model_type"] = "lightgbm"
        self.metadata["model_params"] = self.params
        self.metadata["improvements"] = [
            "Histogram-based algorithm",
            "Leaf-wise tree growth",
            "Efficient memory usage",
        ]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMModel":
        """Train the LightGBM model."""
        logger.info(f"Training {self.name}...")
        self.model.fit(X, y)
        self.is_fitted = True
        
        self.metadata["n_samples"] = len(y)
        self.metadata["n_features"] = X.shape[1]
        
        logger.info(f"{self.name} trained successfully")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make binary predictions."""
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted first")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability estimates."""
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted first")
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importances(self) -> np.ndarray:
        """Get feature importances from LightGBM."""
        return self.model.feature_importances_


def tune_with_optuna(
    model_class: type,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    metric: str = "expected_value",
    param_space: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Tune model hyperparameters using Optuna.
    
    Args:
        model_class: Model class to tune.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        n_trials: Number of optimization trials.
        metric: Metric to optimize.
        param_space: Custom parameter search space.
    
    Returns:
        Dictionary with best parameters and trial results.
    
    Example:
        >>> results = tune_with_optuna(XGBoostModel, X_train, y_train, X_val, y_val)
        >>> best_params = results["best_params"]
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        raise ImportError("Optuna not installed. Run: pip install optuna")
    
    from src.business.cost_analysis import calculate_expected_value
    
    def objective(trial: optuna.Trial) -> float:
        # Default parameter space for XGBoost
        if param_space is None:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
        else:
            params = {k: trial.suggest_categorical(k, v) if isinstance(v, list)
                      else v for k, v in param_space.items()}
        
        # Train model
        model = model_class(**params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val)
        
        if metric == "expected_value":
            score = calculate_expected_value(y_val, y_pred)
        else:
            from sklearn.metrics import f1_score, roc_auc_score
            if metric == "f1":
                score = f1_score(y_val, y_pred)
            elif metric == "roc_auc":
                score = roc_auc_score(y_val, model.predict_proba(X_val))
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        return score
    
    # Run optimization
    sampler = TPESampler(seed=model_config.random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    logger.info(f"Starting Optuna optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best {metric}: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
    }
