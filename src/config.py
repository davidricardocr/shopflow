"""
ShopFlow - E-commerce Returns Prediction

Configuration module with business constants and settings.
Uses Pydantic for type validation and configuration management.
"""

from pydantic import BaseModel, Field
from pathlib import Path
from typing import List


# =============================================================================
# Path Configuration
# =============================================================================

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"


# =============================================================================
# Business Constants
# =============================================================================

class BusinessConfig(BaseModel):
    """
    Financial parameters for cost-benefit analysis.
    
    These values are derived from the challenge requirements:
    - Returns cost the company $18 per item
    - Interventions cost $3 but reduce return probability by 35%
    - True Positive saves $15 ($18 - $3)
    """
    
    return_cost: float = Field(
        default=18.0,
        description="Cost when an item is returned ($)"
    )
    intervention_cost: float = Field(
        default=3.0,
        description="Cost of preventive intervention ($)"
    )
    intervention_effectiveness: float = Field(
        default=0.35,
        description="Reduction in return probability from intervention"
    )
    
    @property
    def savings_per_true_positive(self) -> float:
        """Net savings when correctly predicting and preventing a return."""
        return self.return_cost - self.intervention_cost
    
    @property
    def cost_ratio(self) -> float:
        """Ratio of FN cost to FP cost, useful for threshold optimization."""
        return self.return_cost / self.intervention_cost


# =============================================================================
# Data Configuration
# =============================================================================

class DataConfig(BaseModel):
    """Dataset configuration and feature definitions."""
    
    train_file: str = "ecommerce_returns_train.csv"
    test_file: str = "ecommerce_returns_test.csv"
    
    target_column: str = "is_return"
    id_column: str = "order_id"
    
    numeric_features: List[str] = Field(default=[
        "customer_age",
        "customer_tenure_days",
        "product_price",
        "days_since_last_purchase",
        "previous_returns",
        "product_rating",
    ])
    
    categorical_features: List[str] = Field(default=[
        "product_category",
        "size_purchased",
    ])
    
    binary_features: List[str] = Field(default=[
        "discount_applied",
    ])


# =============================================================================
# Model Configuration
# =============================================================================

class ModelConfig(BaseModel):
    """Model training configuration."""
    
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    
    # Deployment thresholds (from challenge example)
    min_expected_value: float = 2.0
    min_precision: float = 0.65
    min_recall: float = 0.50


# =============================================================================
# Deployment Criteria
# =============================================================================

class DeploymentCriteria(BaseModel):
    """
    Criteria for model deployment readiness.
    Based on the challenge's example of a strong Part 2 submission.
    """
    
    min_expected_value_per_customer: float = 2.0
    min_precision: float = 0.65
    min_recall: float = 0.50
    confidence_level: float = 0.95  # For A/B testing
    
    def is_deployment_ready(
        self,
        expected_value: float,
        precision: float,
        recall: float
    ) -> bool:
        """Check if model meets all deployment criteria."""
        return (
            expected_value >= self.min_expected_value_per_customer
            and precision >= self.min_precision
            and recall >= self.min_recall
        )


# =============================================================================
# Global Instances
# =============================================================================

business_config = BusinessConfig()
data_config = DataConfig()
model_config = ModelConfig()
deployment_criteria = DeploymentCriteria()
