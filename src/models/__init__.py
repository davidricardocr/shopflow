"""Model implementations for ShopFlow."""

from src.models.base import BaseModel, SklearnModelWrapper
from src.models.baseline import (
    BaselineLogisticRegression,
    OptimizedLogisticRegression,
    create_baseline_model,
    create_optimized_lr,
)
from src.models.improved import (
    RandomForestModel,
    GradientBoostingModel,
    XGBoostModel,
    LightGBMModel,
    tune_with_optuna,
)
from src.models.factory import (
    ModelFactory,
    MODEL_REGISTRY,
    get_baseline,
    get_best_model,
    create_model_ensemble,
)

__all__ = [
    # Base
    "BaseModel",
    "SklearnModelWrapper",
    # Baseline
    "BaselineLogisticRegression",
    "OptimizedLogisticRegression",
    "create_baseline_model",
    "create_optimized_lr",
    # Improved
    "RandomForestModel",
    "GradientBoostingModel",
    "XGBoostModel",
    "LightGBMModel",
    "tune_with_optuna",
    # Factory
    "ModelFactory",
    "MODEL_REGISTRY",
    "get_baseline",
    "get_best_model",
    "create_model_ensemble",
]
