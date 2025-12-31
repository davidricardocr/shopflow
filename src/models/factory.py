"""
Model factory for ShopFlow.

This module implements the Factory pattern for creating model instances,
providing a clean interface for model selection and configuration.
"""

from typing import Dict, Any, Optional, Type, List



from loguru import logger

from .base import BaseModel
from .baseline import (
    BaselineLogisticRegression,
    OptimizedLogisticRegression,
)
from .improved import (
    RandomForestModel,
    GradientBoostingModel,
    XGBoostModel,
    LightGBMModel,
)


# Registry of available models
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "baseline": BaselineLogisticRegression,
    "logistic_regression": BaselineLogisticRegression,
    "optimized_lr": OptimizedLogisticRegression,
    "random_forest": RandomForestModel,
    "gradient_boosting": GradientBoostingModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
}


class ModelFactory:
    """
    Factory for creating model instances.
    
    Provides a unified interface for instantiating different model types
    with consistent configuration and logging.
    
    Example:
        >>> model = ModelFactory.create("xgboost", n_estimators=200)
        >>> model.fit(X_train, y_train)
    """
    
    @staticmethod
    def create(
        model_type: str,
        name: Optional[str] = None,
        **kwargs
    ) -> BaseModel:
        """
        Create a model instance by type.
        
        Args:
            model_type: Type of model to create (e.g., 'baseline', 'xgboost').
            name: Optional custom name for the model.
            **kwargs: Model-specific parameters.
        
        Returns:
            Configured model instance.
        
        Raises:
            ValueError: If model_type is not recognized.
        
        Example:
            >>> model = ModelFactory.create("xgboost", n_estimators=200)
            >>> model = ModelFactory.create("random_forest", max_depth=10)
        """
        model_type = model_type.lower().replace("-", "_").replace(" ", "_")
        
        if model_type not in MODEL_REGISTRY:
            available = ", ".join(MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown model type: '{model_type}'. "
                f"Available models: {available}"
            )
        
        model_class = MODEL_REGISTRY[model_type]
        
        if name:
            kwargs["name"] = name
        
        model = model_class(**kwargs)
        logger.info(f"Created model: {model}")
        
        return model
    
    @staticmethod
    def list_available() -> List[str]:
        """
        List all available model types.
        
        Returns:
            List of model type names.
        """
        return list(MODEL_REGISTRY.keys())
    
    @staticmethod
    def get_model_class(model_type: str) -> Type[BaseModel]:
        """
        Get the model class for a given type.
        
        Args:
            model_type: Type of model.
        
        Returns:
            Model class (not instance).
        """
        model_type = model_type.lower().replace("-", "_").replace(" ", "_")
        
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return MODEL_REGISTRY[model_type]
    
    @staticmethod
    def register(name: str, model_class: Type[BaseModel]) -> None:
        """
        Register a new model type.
        
        Allows extending the factory with custom model implementations.
        
        Args:
            name: Name for the model type.
            model_class: Model class to register.
        
        Example:
            >>> class MyCustomModel(BaseModel):
            ...     pass
            >>> ModelFactory.register("custom", MyCustomModel)
        """
        MODEL_REGISTRY[name.lower()] = model_class
        logger.info(f"Registered model: {name}")


def create_model_ensemble(
    model_configs: List[Dict[str, Any]],
    voting: str = "soft"
) -> "EnsembleModel":
    """
    Create an ensemble of models.
    
    Args:
        model_configs: List of model configurations.
            Each dict should have 'type' and optionally 'params'.
        voting: 'soft' for probability averaging, 'hard' for majority vote.
    
    Returns:
        EnsembleModel instance.
    
    Example:
        >>> configs = [
        ...     {"type": "xgboost", "params": {"n_estimators": 100}},
        ...     {"type": "lightgbm", "params": {"n_estimators": 100}},
        ...     {"type": "random_forest", "params": {"n_estimators": 100}},
        ... ]
        >>> ensemble = create_model_ensemble(configs)
    """
    
    
    estimators = []
    for i, config in enumerate(model_configs):
        model_type = config["type"]
        params = config.get("params", {})
        
        model = ModelFactory.create(model_type, **params)
        estimators.append((f"model_{i}_{model_type}", model.model))
    
    return VotingClassifier(estimators=estimators, voting=voting)


# Convenience functions
def get_baseline() -> BaselineLogisticRegression:
    """Get the baseline logistic regression model."""
    return ModelFactory.create("baseline")


def get_best_model(
    model_type: str = "xgboost",
    optimized: bool = True
) -> BaseModel:
    """
    Get a pre-configured high-performance model.
    
    Args:
        model_type: Type of model ('xgboost', 'lightgbm', 'random_forest').
        optimized: Whether to use optimized hyperparameters.
    
    Returns:
        Configured model instance.
    """
    if model_type == "xgboost":
        # Reasonable defaults for e-commerce returns prediction
        return ModelFactory.create(
            "xgboost",
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
        )
    elif model_type == "lightgbm":
        return ModelFactory.create(
            "lightgbm",
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            class_weight="balanced",
        )
    elif model_type == "random_forest":
        return ModelFactory.create(
            "random_forest",
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
        )
    else:
        return ModelFactory.create(model_type)
