"""
Feature engineering for ShopFlow.

This module provides functions for creating new features from the raw data,
with documented hypotheses for each feature to explain the business rationale.
"""

from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from loguru import logger

from src.config import data_config


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer for e-commerce returns data.
    
    Creates new features based on domain knowledge and hypotheses:
    - Customer behavior features
    - Product risk indicators
    - Temporal patterns
    
    All features have documented hypotheses to explain the reasoning.
    
    Attributes:
        features_created: List of new features created.
        feature_hypotheses: Dictionary mapping features to their hypotheses.
    """
    
    def __init__(
        self,
        create_customer_features: bool = True,
        create_product_features: bool = True,
        create_interaction_features: bool = True,
        create_temporal_features: bool = False,
    ):
        """
        Initialize the feature engineer.
        
        Args:
            create_customer_features: Create customer behavior features.
            create_product_features: Create product risk features.
            create_interaction_features: Create feature interactions.
            create_temporal_features: Create time-based features.
        """
        self.create_customer_features = create_customer_features
        self.create_product_features = create_product_features
        self.create_interaction_features = create_interaction_features
        self.create_temporal_features = create_temporal_features
        
        self.features_created: List[str] = []
        self.feature_hypotheses: Dict[str, str] = {}
        
        # Store computed statistics for transform
        self._category_return_rates: Dict[str, float] = {}
        self._price_stats: Dict[str, float] = {}
        self._is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """
        Fit the feature engineer by computing statistics.
        
        Args:
            X: Training DataFrame.
            y: Target variable (used for computing category return rates).
        
        Returns:
            self
        """
        X = pd.DataFrame(X)
        
        # Compute category return rates if target is available
        if y is not None and "product_category" in X.columns:
            df_temp = X.copy()
            df_temp["target"] = y
            self._category_return_rates = (
                df_temp.groupby("product_category")["target"]
                .mean()
                .to_dict()
            )
        
        # Compute price statistics per category
        if "product_price" in X.columns and "product_category" in X.columns:
            self._price_stats = {
                "overall_median": X["product_price"].median(),
                "overall_std": X["product_price"].std(),
            }
            
            for cat in X["product_category"].unique():
                cat_prices = X[X["product_category"] == cat]["product_price"]
                self._price_stats[f"{cat}_median"] = cat_prices.median()
                self._price_stats[f"{cat}_std"] = cat_prices.std()
        
        self._is_fitted = True
        logger.info("FeatureEngineer fitted successfully")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from the input data.
        
        Args:
            X: Input DataFrame.
        
        Returns:
            DataFrame with new features added.
        """
        if not self._is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        X = pd.DataFrame(X).copy()
        self.features_created = []
        
        if self.create_customer_features:
            X = self._add_customer_features(X)
        
        if self.create_product_features:
            X = self._add_product_features(X)
        
        if self.create_interaction_features:
            X = self._add_interaction_features(X)
        
        if self.create_temporal_features:
            X = self._add_temporal_features(X)
        
        logger.info(f"Created {len(self.features_created)} new features")
        return X
    
    def _add_customer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add customer behavior features."""
        
        # Feature: Customer Risk Score
        # Hypothesis: Customers with high previous returns and short tenure are higher risk
        if "previous_returns" in X.columns and "customer_tenure_days" in X.columns:
            # Normalize tenure to 0-1 (more tenure = lower risk)
            max_tenure = X["customer_tenure_days"].max()
            tenure_normalized = X["customer_tenure_days"] / max_tenure if max_tenure > 0 else 0
            
            X["customer_risk_score"] = (
                X["previous_returns"] * 0.7 - 
                tenure_normalized * 0.3
            )
            
            self.features_created.append("customer_risk_score")
            self.feature_hypotheses["customer_risk_score"] = (
                "Customers with many previous returns and short tenure "
                "are more likely to return items. Weighted combination "
                "of returns (positive) and tenure (negative)."
            )
        
        # Feature: Is New Customer
        # Hypothesis: New customers (< 30 days) may have different return behavior
        if "customer_tenure_days" in X.columns:
            X["is_new_customer"] = (X["customer_tenure_days"] < 30).astype(int)
            
            self.features_created.append("is_new_customer")
            self.feature_hypotheses["is_new_customer"] = (
                "New customers (tenure < 30 days) may have higher return rates "
                "due to unfamiliarity with sizing or product quality."
            )
        
        # Feature: Is Frequent Returner
        # Hypothesis: Customers with 3+ returns in 6 months are habitual returners
        if "previous_returns" in X.columns:
            X["is_frequent_returner"] = (X["previous_returns"] >= 3).astype(int)
            
            self.features_created.append("is_frequent_returner")
            self.feature_hypotheses["is_frequent_returner"] = (
                "Customers with 3+ returns in the last 6 months show a pattern "
                "of habitual returning that predicts future returns."
            )
        
        # Feature: Days Since Last Purchase Bucket
        # Hypothesis: Very recent or very old last purchases have different patterns
        if "days_since_last_purchase" in X.columns:
            X["recency_bucket"] = pd.cut(
                X["days_since_last_purchase"],
                bins=[0, 7, 30, 90, float("inf")],
                labels=[0, 1, 2, 3]  # 0=very recent, 3=very old
            ).astype(int)
            
            self.features_created.append("recency_bucket")
            self.feature_hypotheses["recency_bucket"] = (
                "Customer recency affects purchase behavior. Very recent buyers "
                "may be making impulse purchases (higher return risk)."
            )
        
        return X
    
    def _add_product_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add product risk features."""
        
        # Feature: Category Return Rate
        # Hypothesis: Some categories (e.g., Fashion) have inherently higher return rates
        if "product_category" in X.columns and self._category_return_rates:
            X["category_return_rate"] = X["product_category"].map(
                self._category_return_rates
            )
            # Fill missing with overall mean
            X["category_return_rate"] = X["category_return_rate"].fillna(
                np.mean(list(self._category_return_rates.values()))
            )
            
            self.features_created.append("category_return_rate")
            self.feature_hypotheses["category_return_rate"] = (
                "Historical return rate by category. Fashion has ~30% vs "
                "15% overall. This encodes category-specific risk."
            )
        
        # Feature: Is Fashion
        # Hypothesis: Fashion items have special return patterns due to sizing
        if "product_category" in X.columns:
            X["is_fashion"] = (X["product_category"] == "Fashion").astype(int)
            
            self.features_created.append("is_fashion")
            self.feature_hypotheses["is_fashion"] = (
                "Fashion items have 2x the return rate of other categories "
                "due to sizing issues. Binary indicator for this high-risk category."
            )
        
        # Feature: Price Deviation from Category Median
        # Hypothesis: Unusually expensive items within a category may have higher returns
        if "product_price" in X.columns and "product_category" in X.columns:
            def price_deviation(row):
                cat = row["product_category"]
                median_key = f"{cat}_median"
                std_key = f"{cat}_std"
                
                if median_key in self._price_stats and std_key in self._price_stats:
                    median = self._price_stats[median_key]
                    std = self._price_stats[std_key]
                    if std > 0:
                        return (row["product_price"] - median) / std
                return 0
            
            X["price_deviation"] = X.apply(price_deviation, axis=1)
            
            self.features_created.append("price_deviation")
            self.feature_hypotheses["price_deviation"] = (
                "Standardized price deviation from category median. "
                "Unusually priced items may have different return patterns."
            )
        
        # Feature: Low Rating
        # Hypothesis: Products with ratings < 3.5 are more likely to be returned
        if "product_rating" in X.columns:
            X["is_low_rating"] = (X["product_rating"] < 3.5).astype(int)
            
            self.features_created.append("is_low_rating")
            self.feature_hypotheses["is_low_rating"] = (
                "Products with ratings below 3.5 indicate quality concerns "
                "that increase return likelihood."
            )
        
        return X
    
    def _add_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add feature interactions."""
        
        # Feature: Fashion with Discount
        # Hypothesis: Discounted fashion items may be impulse buys with higher returns
        if "product_category" in X.columns and "discount_applied" in X.columns:
            X["fashion_with_discount"] = (
                (X["product_category"] == "Fashion") & 
                (X["discount_applied"] == 1)
            ).astype(int)
            
            self.features_created.append("fashion_with_discount")
            self.feature_hypotheses["fashion_with_discount"] = (
                "Fashion items bought on discount may be impulse purchases "
                "with less consideration of fit, leading to higher returns."
            )
        
        # Feature: High Price + New Customer
        # Hypothesis: New customers buying expensive items may be testing the waters
        if "product_price" in X.columns and "customer_tenure_days" in X.columns:
            high_price_threshold = self._price_stats.get("overall_median", 50) * 1.5
            X["expensive_new_customer"] = (
                (X["product_price"] > high_price_threshold) & 
                (X["customer_tenure_days"] < 30)
            ).astype(int)
            
            self.features_created.append("expensive_new_customer")
            self.feature_hypotheses["expensive_new_customer"] = (
                "New customers making expensive purchases may have higher "
                "return rates as they test the retailer's products/service."
            )
        
        # Feature: Previous Returns × Discount
        # Hypothesis: Frequent returners buying on discount = very high risk
        if "previous_returns" in X.columns and "discount_applied" in X.columns:
            X["returner_on_discount"] = X["previous_returns"] * X["discount_applied"]
            
            self.features_created.append("returner_on_discount")
            self.feature_hypotheses["returner_on_discount"] = (
                "Interaction between return history and discount purchasing. "
                "Known returners buying discounted items are high risk."
            )
        
        return X
    
    def _add_temporal_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features (if date columns exist)."""
        # Placeholder for temporal features
        # Would require order_date column which isn't in the dataset
        return X
    
    def get_feature_documentation(self) -> pd.DataFrame:
        """
        Get documentation for all created features.
        
        Returns:
            DataFrame with feature names and hypotheses.
        """
        return pd.DataFrame([
            {"feature": k, "hypothesis": v}
            for k, v in self.feature_hypotheses.items()
        ])
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)


def create_all_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, FeatureEngineer]:
    """
    Create all engineered features for train and test sets.
    
    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        y_train: Training target (for computing category rates).
    
    Returns:
        Tuple of (train_with_features, test_with_features, feature_engineer).
    
    Example:
        >>> train_fe, test_fe, fe = create_all_features(train, test, y_train)
        >>> print(fe.get_feature_documentation())
    """
    feature_engineer = FeatureEngineer(
        create_customer_features=True,
        create_product_features=True,
        create_interaction_features=True,
        create_temporal_features=False,
    )
    
    train_fe = feature_engineer.fit_transform(train_df, y_train)
    test_fe = feature_engineer.transform(test_df)
    
    return train_fe, test_fe, feature_engineer


def get_feature_importance_analysis(
    model: Any,
    feature_names: List[str],
    top_n: int = 15
) -> pd.DataFrame:
    """
    Analyze feature importances from a trained model.
    
    Args:
        model: Trained model with feature_importances_ or coef_.
        feature_names: List of feature names.
        top_n: Number of top features to return.
    
    Returns:
        DataFrame with features ranked by importance.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not have feature importances")
    
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    
    # Normalize to percentages
    df["importance_pct"] = df["importance"] / df["importance"].sum() * 100
    
    return df.head(top_n).reset_index(drop=True)
