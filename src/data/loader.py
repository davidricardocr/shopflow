"""
Data loading utilities for ShopFlow.

This module provides functions for loading and validating the e-commerce
returns dataset with proper error handling and type safety.
"""

from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
from loguru import logger

from ..config import DATA_DIR, data_config


class DataLoadError(Exception):
    """Raised when data loading fails."""
    pass


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


def load_dataset(
    filename: str,
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load a CSV dataset with validation.
    
    Args:
        filename: Name of the CSV file to load.
        data_dir: Directory containing the data. Defaults to DATA_DIR.
    
    Returns:
        DataFrame with the loaded data.
    
    Raises:
        DataLoadError: If the file cannot be loaded.
        DataValidationError: If required columns are missing.
    
    Example:
        >>> train_df = load_dataset("ecommerce_returns_train.csv")
        >>> train_df.shape
        (8000, 11)
    """
    data_dir = data_dir or DATA_DIR
    filepath = data_dir / filename
    
    if not filepath.exists():
        raise DataLoadError(f"File not found: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        raise DataLoadError(f"Failed to load {filename}: {e}")
    
    _validate_dataframe(df, filename)
    return df


def _validate_dataframe(df: pd.DataFrame, filename: str) -> None:
    """
    Validate that DataFrame has required columns.
    
    Args:
        df: DataFrame to validate.
        filename: Filename for error messages.
    
    Raises:
        DataValidationError: If required columns are missing.
    """
    required_columns = (
        data_config.numeric_features 
        + data_config.categorical_features 
        + data_config.binary_features
        + [data_config.target_column, data_config.id_column]
    )
    
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise DataValidationError(
            f"Missing columns in {filename}: {missing}"
        )


def load_train_test() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both training and test datasets.
    
    Returns:
        Tuple of (train_df, test_df).
    
    Example:
        >>> train, test = load_train_test()
        >>> print(f"Train: {train.shape}, Test: {test.shape}")
        Train: (8000, 11), Test: (2000, 11)
    """
    train = load_dataset(data_config.train_file)
    test = load_dataset(data_config.test_file)
    
    logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
    logger.info(f"Target distribution (train): {train[data_config.target_column].value_counts(normalize=True).to_dict()}")
    
    return train, test


def get_feature_target_split(
    df: pd.DataFrame,
    include_id: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features (X) and target (y).
    
    Args:
        df: Input DataFrame.
        include_id: Whether to include order_id in features.
    
    Returns:
        Tuple of (X, y) where X is features and y is target.
    
    Example:
        >>> X, y = get_feature_target_split(train_df)
        >>> X.shape, y.shape
        ((8000, 9), (8000,))
    """
    feature_cols = (
        data_config.numeric_features 
        + data_config.categorical_features 
        + data_config.binary_features
    )
    
    if include_id:
        feature_cols = [data_config.id_column] + feature_cols
    
    X = df[feature_cols].copy()
    y = df[data_config.target_column].copy()
    
    return X, y


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary of the dataset for exploratory analysis.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        Dictionary with dataset statistics.
    """
    target_col = data_config.target_column
    
    summary = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "target_distribution": df[target_col].value_counts().to_dict(),
        "target_rate": df[target_col].mean(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_stats": df[data_config.numeric_features].describe().to_dict(),
        "category_counts": {
            col: df[col].value_counts().to_dict()
            for col in data_config.categorical_features
        }
    }
    
    return summary
