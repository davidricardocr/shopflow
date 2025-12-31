"""
Custom sklearn transformers for ShopFlow preprocessing.

This module implements reusable transformers following sklearn's
BaseEstimator and TransformerMixin interfaces for seamless pipeline integration.
"""

from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler


class SafeLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Label encoder that handles unseen categories gracefully.
    
    Unlike sklearn's LabelEncoder, this transformer:
    - Works with DataFrames
    - Handles unseen categories by mapping to a special value
    - Supports multiple columns
    
    Attributes:
        columns: List of columns to encode.
        encoders_: Dictionary mapping column names to fitted LabelEncoders.
        unknown_value_: Value used for unseen categories.
    """
    
    def __init__(self, columns: Optional[List[str]] = None):
        """
        Initialize the encoder.
        
        Args:
            columns: Columns to encode. If None, encodes all object columns.
        """
        self.columns = columns
        self.encoders_: Dict[str, LabelEncoder] = {}
        self.unknown_value_ = -1
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "SafeLabelEncoder":
        """
        Fit label encoders for each column.
        
        Args:
            X: Input DataFrame.
            y: Ignored, present for API compatibility.
        
        Returns:
            self
        """
        X = pd.DataFrame(X)
        columns = self.columns or X.select_dtypes(include=["object"]).columns.tolist()
        
        for col in columns:
            le = LabelEncoder()
            # Handle NaN by converting to string
            values = X[col].fillna("__MISSING__").astype(str)
            le.fit(values)
            self.encoders_[col] = le
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform columns using fitted encoders.
        
        Args:
            X: Input DataFrame.
        
        Returns:
            DataFrame with encoded columns.
        """
        X = pd.DataFrame(X).copy()
        
        for col, encoder in self.encoders_.items():
            if col in X.columns:
                values = X[col].fillna("__MISSING__").astype(str)
                # Handle unseen categories
                known_classes = set(encoder.classes_)
                X[col] = values.apply(
                    lambda x: encoder.transform([x])[0] if x in known_classes else self.unknown_value_
                )
        
        return X
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Return feature names for encoded columns."""
        return list(self.encoders_.keys())


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Flexible missing value imputer with different strategies per column.
    
    Supports:
    - Numeric: mean, median, constant
    - Categorical: mode, constant
    
    Attributes:
        strategy: Default strategy for all columns.
        fill_values_: Dictionary of computed fill values per column.
    """
    
    def __init__(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
        fill_value: Any = None
    ):
        """
        Initialize the imputer.
        
        Args:
            numeric_strategy: Strategy for numeric columns ('mean', 'median', 'constant').
            categorical_strategy: Strategy for categorical columns ('mode', 'constant').
            fill_value: Value to use when strategy is 'constant'.
        """
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.fill_value = fill_value
        self.fill_values_: Dict[str, Any] = {}
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "MissingValueImputer":
        """
        Compute fill values for each column.
        
        Args:
            X: Input DataFrame.
            y: Ignored.
        
        Returns:
            self
        """
        X = pd.DataFrame(X)
        
        for col in X.columns:
            if X[col].dtype in ["float64", "int64", "float32", "int32"]:
                # Numeric column
                if self.numeric_strategy == "mean":
                    self.fill_values_[col] = X[col].mean()
                elif self.numeric_strategy == "median":
                    self.fill_values_[col] = X[col].median()
                else:
                    self.fill_values_[col] = self.fill_value or 0
            else:
                # Categorical column
                if self.categorical_strategy == "mode":
                    mode_result = X[col].mode()
                    self.fill_values_[col] = mode_result[0] if len(mode_result) > 0 else "Unknown"
                else:
                    self.fill_values_[col] = self.fill_value or "Unknown"
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values using computed fill values.
        
        Args:
            X: Input DataFrame.
        
        Returns:
            DataFrame with missing values filled.
        """
        X = pd.DataFrame(X).copy()
        
        for col, fill_value in self.fill_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(fill_value)
        
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select specific columns from a DataFrame.
    
    Useful at the start of a pipeline to ensure only relevant
    features are passed through.
    """
    
    def __init__(self, columns: List[str]):
        """
        Initialize selector.
        
        Args:
            columns: List of column names to select.
        """
        self.columns = columns
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureSelector":
        """Fit method (no-op, returns self)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select specified columns.
        
        Args:
            X: Input DataFrame.
        
        Returns:
            DataFrame with only selected columns.
        """
        X = pd.DataFrame(X)
        missing = set(self.columns) - set(X.columns)
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")
        return X[self.columns].copy()
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Return selected column names."""
        return self.columns


class DataFrameScaler(BaseEstimator, TransformerMixin):
    """
    StandardScaler wrapper that preserves DataFrame structure.
    
    sklearn's StandardScaler returns numpy arrays. This wrapper
    maintains column names and returns DataFrames.
    """
    
    def __init__(self, columns: Optional[List[str]] = None):
        """
        Initialize scaler.
        
        Args:
            columns: Columns to scale. If None, scales all numeric columns.
        """
        self.columns = columns
        self.scaler_ = StandardScaler()
        self.columns_to_scale_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataFrameScaler":
        """
        Fit the scaler.
        
        Args:
            X: Input DataFrame.
            y: Ignored.
        
        Returns:
            self
        """
        X = pd.DataFrame(X)
        
        if self.columns:
            self.columns_to_scale_ = self.columns
        else:
            self.columns_to_scale_ = X.select_dtypes(
                include=["float64", "int64", "float32", "int32"]
            ).columns.tolist()
        
        if self.columns_to_scale_:
            self.scaler_.fit(X[self.columns_to_scale_])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale specified columns.
        
        Args:
            X: Input DataFrame.
        
        Returns:
            DataFrame with scaled columns.
        """
        X = pd.DataFrame(X).copy()
        
        if self.columns_to_scale_:
            X[self.columns_to_scale_] = self.scaler_.transform(X[self.columns_to_scale_])
        
        return X
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Return column names."""
        return self.columns_to_scale_
