"""
Abstract base model for ShopFlow.

This module implements the Strategy pattern for interchangeable models,
allowing easy experimentation with different algorithms while maintaining
a consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger

from src.config import MODELS_DIR, model_config


class BaseModel(ABC):
    """
    Abstract base class for all models.
    
    Implements the Strategy pattern, allowing different model implementations
    to be swapped without changing the client code.
    
    All models must implement:
    - fit(): Train the model
    - predict(): Make binary predictions
    - predict_proba(): Get probability estimates
    
    Attributes:
        name: Human-readable model name.
        model: The underlying sklearn model.
        is_fitted: Whether the model has been trained.
        metadata: Additional model information.
    """
    
    def __init__(self, name: str = "BaseModel"):
        """
        Initialize the base model.
        
        Args:
            name: Descriptive name for the model.
        """
        self.name = name
        self.model: Any = None
        self.is_fitted = False
        self.metadata: Dict[str, Any] = {
            "name": name,
            "random_state": model_config.random_state,
        }
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """
        Train the model on the provided data.
        
        Args:
            X: Training features.
            y: Training labels.
        
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            X: Features to predict on.
        
        Returns:
            Array of binary predictions (0 or 1).
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability estimates for the positive class.
        
        Args:
            X: Features to predict on.
        
        Returns:
            Array of probabilities for class 1.
        """
        pass
    
    def predict_at_threshold(
        self,
        X: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Make predictions at a custom threshold.
        
        Args:
            X: Features to predict on.
            threshold: Decision threshold.
        
        Returns:
            Array of binary predictions.
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def save(self, filepath: Optional[Path] = None) -> Path:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save to. Defaults to MODELS_DIR/{name}.pkl
        
        Returns:
            Path where model was saved.
        """
        if filepath is None:
            filepath = MODELS_DIR / f"{self.name.lower().replace(' ', '_')}.pkl"
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> "BaseModel":
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model.
        
        Returns:
            Loaded model instance.
        """
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of model parameters.
        """
        if hasattr(self.model, "get_params"):
            return self.model.get_params()
        return {}
    
    def set_params(self, **params) -> "BaseModel":
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set.
        
        Returns:
            self
        """
        if hasattr(self.model, "set_params"):
            self.model.set_params(**params)
        return self
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"


class SklearnModelWrapper(BaseModel):
    """
    Wrapper for any sklearn-compatible model.
    
    Provides a consistent interface for sklearn models while adding
    logging, metadata, and custom threshold predictions.
    """
    
    def __init__(
        self,
        model: Any,
        name: str = "SklearnModel",
        **model_params
    ):
        """
        Initialize the wrapper.
        
        Args:
            model: An sklearn model class (not instance).
            name: Model name.
            **model_params: Parameters to pass to the model.
        """
        super().__init__(name)
        
        # Add random state if supported
        if "random_state" not in model_params:
            model_params["random_state"] = model_config.random_state
        
        self.model = model(**model_params)
        self.model_params = model_params
        self.metadata["model_class"] = model.__name__
        self.metadata["model_params"] = model_params
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnModelWrapper":
        """
        Train the sklearn model.
        
        Args:
            X: Training features.
            y: Training labels.
        
        Returns:
            self
        """
        logger.info(f"Training {self.name}...")
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Store training info
        self.metadata["n_samples"] = len(y)
        self.metadata["n_features"] = X.shape[1]
        self.metadata["class_distribution"] = {
            "class_0": int(np.sum(y == 0)),
            "class_1": int(np.sum(y == 1)),
        }
        
        logger.info(f"{self.name} trained on {len(y)} samples")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            X: Features to predict on.
        
        Returns:
            Array of binary predictions.
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before predicting")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability estimates for the positive class.
        
        Args:
            X: Features to predict on.
        
        Returns:
            Array of probabilities for class 1.
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before predicting")
        
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        elif hasattr(self.model, "decision_function"):
            # For models like SVM that don't have predict_proba by default
            from scipy.special import expit
            return expit(self.model.decision_function(X))
        else:
            raise ValueError(f"{self.name} does not support probability predictions")
    
    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        Get feature importances if available.
        
        Returns:
            Array of feature importances or None.
        """
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            return np.abs(self.model.coef_[0])
        return None
