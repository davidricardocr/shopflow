"""
ShopFlow - E-commerce Returns Prediction

A machine learning project for predicting product returns in e-commerce,
with business-aligned metrics and cost-benefit optimization.
"""

from src.config import (
    business_config,
    data_config,
    model_config,
    deployment_criteria,
    ROOT_DIR,
    DATA_DIR,
    MODELS_DIR,
)

__version__ = "1.0.0"
__author__ = "David Mora"

__all__ = [
    "business_config",
    "data_config", 
    "model_config",
    "deployment_criteria",
    "ROOT_DIR",
    "DATA_DIR",
    "MODELS_DIR",
]
