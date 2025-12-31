"""
Preprocessing pipeline for ShopFlow.

This module provides factory functions for creating sklearn pipelines
that handle the complete preprocessing workflow for the returns prediction model.
"""

from typing import Tuple, Optional, List

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from loguru import logger

from ..config import data_config
from .transformers import (
    SafeLabelEncoder,
    MissingValueImputer,
    FeatureSelector,
    DataFrameScaler,
)


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for e-commerce returns data.
    
    This class encapsulates the full preprocessing workflow:
    1. Handle missing values
    2. Encode categorical features
    3. Scale numeric features
    
    Follows the Factory pattern for creating different pipeline configurations.
    
    Attributes:
        pipeline: The fitted sklearn Pipeline.
        feature_names_: List of feature names after transformation.
    """
    
    def __init__(
        self,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        binary_features: Optional[List[str]] = None,
        encoding_strategy: str = "label"  # 'label' or 'onehot'
    ):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            numeric_features: List of numeric column names.
            categorical_features: List of categorical column names.
            binary_features: List of binary column names.
            encoding_strategy: How to encode categoricals ('label' or 'onehot').
        """
        self.numeric_features = numeric_features or data_config.numeric_features
        self.categorical_features = categorical_features or data_config.categorical_features
        self.binary_features = binary_features or data_config.binary_features
        self.encoding_strategy = encoding_strategy
        
        self.pipeline: Optional[Pipeline] = None
        self.feature_names_: List[str] = []
        self._is_fitted = False
    
    def _create_pipeline(self) -> Pipeline:
        """
        Create the sklearn pipeline based on configuration.
        
        Returns:
            Configured Pipeline object.
        """
        # Numeric transformer
        numeric_transformer = Pipeline(steps=[
            ("imputer", MissingValueImputer(numeric_strategy="median")),
            ("scaler", DataFrameScaler(columns=self.numeric_features))
        ])
        
        # Categorical transformer
        if self.encoding_strategy == "onehot":
            categorical_transformer = Pipeline(steps=[
                ("imputer", MissingValueImputer(categorical_strategy="mode")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ))
            ])
        else:
            categorical_transformer = Pipeline(steps=[
                ("imputer", MissingValueImputer(categorical_strategy="mode")),
                ("encoder", SafeLabelEncoder(columns=self.categorical_features))
            ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, self.numeric_features),
                ("categorical", categorical_transformer, self.categorical_features),
                ("binary", "passthrough", self.binary_features)
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )
        
        return Pipeline(steps=[
            ("preprocessor", preprocessor)
        ])
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "PreprocessingPipeline":
        """
        Fit the preprocessing pipeline.
        
        Args:
            X: Training features DataFrame.
            y: Target variable (ignored, for API compatibility).
        
        Returns:
            self
        """
        self.pipeline = self._create_pipeline()
        self.pipeline.fit(X, y)
        
        # Store feature names
        self._compute_feature_names()
        self._is_fitted = True
        
        logger.info(f"Pipeline fitted with {len(self.feature_names_)} features")
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted pipeline.
        
        Args:
            X: Features DataFrame to transform.
        
        Returns:
            Transformed numpy array.
        
        Raises:
            ValueError: If pipeline is not fitted.
        """
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        return self.pipeline.transform(X)
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Training features DataFrame.
            y: Target variable (ignored).
        
        Returns:
            Transformed numpy array.
        """
        self.fit(X, y)
        return self.transform(X)
    
    def _compute_feature_names(self) -> None:
        """Compute feature names after transformation."""
        self.feature_names_ = (
            self.numeric_features.copy() +
            [f"{col}_encoded" for col in self.categorical_features] +
            self.binary_features.copy()
        )
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names after transformation.
        
        Returns:
            List of feature names.
        """
        return self.feature_names_


def create_baseline_pipeline() -> PreprocessingPipeline:
    """
    Create a preprocessing pipeline matching the baseline model.
    
    This replicates the preprocessing from baseline_model.py for
    fair comparison.
    
    Returns:
        PreprocessingPipeline configured for baseline compatibility.
    """
    return PreprocessingPipeline(
        encoding_strategy="label"
    )


def create_improved_pipeline() -> PreprocessingPipeline:
    """
    Create an improved preprocessing pipeline with one-hot encoding.
    
    One-hot encoding may improve model performance for categorical
    features with no ordinal relationship.
    
    Returns:
        PreprocessingPipeline with one-hot encoding.
    """
    return PreprocessingPipeline(
        encoding_strategy="onehot"
    )


def preprocess_for_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, PreprocessingPipeline]:
    """
    Convenience function to preprocess train/test data.
    
    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, pipeline).
    
    Example:
        >>> X_train, X_test, y_train, y_test, pipeline = preprocess_for_baseline(train, test)
        >>> X_train.shape
        (8000, 9)
    """
    target_col = data_config.target_column
    feature_cols = (
        data_config.numeric_features + 
        data_config.categorical_features + 
        data_config.binary_features
    )
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    
    pipeline = create_baseline_pipeline()
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train, y_test, pipeline
