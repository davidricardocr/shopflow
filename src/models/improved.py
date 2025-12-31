"""
Improved models for ShopFlow.

This module provides advanced model implementations including
tree-based models and gradient boosting algorithms.
"""

from typing import Optional, Dict, Any, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from loguru import logger

from .base import BaseModel

# Try to import optional dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


class RandomForestModel(BaseModel):
    """Random Forest classifier wrapper."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(name="RandomForest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "RandomForestModel":
        """Train the model."""
        logger.info(f"Training {self.name}...")
        self.model.fit(X, y)
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self._check_is_fitted()
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        self._check_is_fitted()
        return self.model.predict_proba(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params()
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importances."""
        self._check_is_fitted()
        return self.model.feature_importances_


class GradientBoostingModel(BaseModel):
    """Gradient Boosting classifier wrapper."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(name="GradientBoosting")
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "GradientBoostingModel":
        """Train the model."""
        logger.info(f"Training {self.name}...")
        self.model.fit(X, y)
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self._check_is_fitted()
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        self._check_is_fitted()
        return self.model.predict_proba(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params()
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importances."""
        self._check_is_fitted()
        return self.model.feature_importances_


class XGBoostModel(BaseModel):
    """XGBoost classifier wrapper."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        random_state: int = 42,
        **kwargs
    ):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed. Install it with: pip install xgboost")
        
        super().__init__(name="XGBoost")
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "XGBoostModel":
        """Train the model."""
        logger.info(f"Training {self.name}...")
        self.model.fit(X, y, **kwargs)
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self._check_is_fitted()
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        self._check_is_fitted()
        return self.model.predict_proba(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params()
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importances."""
        self._check_is_fitted()
        return self.model.feature_importances_


class LightGBMModel(BaseModel):
    """LightGBM classifier wrapper."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = -1,
        num_leaves: int = 31,
        random_state: int = 42,
        **kwargs
    ):
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM is not installed. Install it with: pip install lightgbm")
        
        super().__init__(name="LightGBM")
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=random_state,
            verbose=-1,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "LightGBMModel":
        """Train the model."""
        logger.info(f"Training {self.name}...")
        self.model.fit(X, y, **kwargs)
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self._check_is_fitted()
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        self._check_is_fitted()
        return self.model.predict_proba(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params()
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importances."""
        self._check_is_fitted()
        return self.model.feature_importances_


def tune_with_optuna(
    model_class: type,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    metric: str = "roc_auc"
) -> Dict[str, Any]:
    """
    Tune model hyperparameters using Optuna.
    
    Args:
        model_class: Model class to tune
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of optimization trials
        metric: Metric to optimize
    
    Returns:
        Dictionary with best parameters and score
    """
    if not HAS_OPTUNA:
        raise ImportError("Optuna is not installed. Install it with: pip install optuna")
    
    from sklearn.metrics import roc_auc_score, f1_score
    
    def objective(trial):
        if model_class == XGBoostModel:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
            }
        elif model_class == LightGBMModel:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            }
        elif model_class == RandomForestModel:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            }
        else:
            params = {}
        
        model = model_class(**params)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        if metric == "roc_auc":
            return roc_auc_score(y_val, y_proba)
        elif metric == "f1":
            y_pred = (y_proba >= 0.5).astype(int)
            return f1_score(y_val, y_pred)
        else:
            return roc_auc_score(y_val, y_proba)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "study": study
    }
