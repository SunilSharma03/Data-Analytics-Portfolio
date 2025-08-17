"""
Regression Models

This module provides implementations of various regression algorithms
including Linear Regression, Ridge, Lasso, Elastic Net, Random Forest, XGBoost, and LightGBM.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Union
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from loguru import logger

from .base import BaseModel


class RegressionModel(BaseModel):
    """
    Regression model wrapper that supports multiple algorithms.
    
    Supported algorithms:
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - Elastic Net
    - Random Forest
    - XGBoost
    - LightGBM
    """
    
    def __init__(self, algorithm: str = 'random_forest', config: Optional[Dict[str, Any]] = None):
        """
        Initialize the regression model.
        
        Args:
            algorithm: Algorithm to use ('linear', 'ridge', 'lasso', 'elastic_net',
                       'random_forest', 'xgboost', 'lightgbm')
            config: Configuration dictionary
        """
        super().__init__(config)
        self.algorithm = algorithm.lower()
        
    def _initialize_model(self) -> Any:
        """Initialize the specific regression model."""
        model_config = self.config.get('models', {}).get('regression', {})
        
        if self.algorithm == 'linear':
            params = model_config.get('linear_regression', {})
            return LinearRegression(**params)
            
        elif self.algorithm == 'ridge':
            params = model_config.get('ridge', {})
            return Ridge(**params)
            
        elif self.algorithm == 'lasso':
            params = model_config.get('lasso', {})
            return Lasso(**params)
            
        elif self.algorithm == 'elastic_net':
            params = model_config.get('elastic_net', {})
            return ElasticNet(**params)
            
        elif self.algorithm == 'random_forest':
            params = model_config.get('random_forest', {})
            return RandomForestRegressor(**params)
            
        elif self.algorithm == 'xgboost':
            params = model_config.get('xgboost', {})
            return xgb.XGBRegressor(**params)
            
        elif self.algorithm == 'lightgbm':
            params = model_config.get('lightgbm', {})
            return lgb.LGBMRegressor(**params)
            
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def _train_model(self, X: Union[np.ndarray, pd.DataFrame], 
                    y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """Train the regression model."""
        # Train the model
        self.model.fit(X, y)
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='neg_mean_squared_error')
        
        training_history = {
            'algorithm': self.algorithm,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_r2': r2_score(y, y_pred),
            'training_mse': mean_squared_error(y, y_pred),
            'training_mae': mean_absolute_error(y, y_pred)
        }
        
        return training_history
    
    def _predict_model(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the trained regression model."""
        return self.model.predict(X)
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], 
                y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Evaluate the regression model.
        
        Args:
            X: Test features
            y: True targets
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X)
        
        # Calculate metrics
        metrics = {
            'r2_score': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'explained_variance': self.model.score(X, y) if hasattr(self.model, 'score') else None
        }
        
        # Additional metrics for linear models
        if hasattr(self.model, 'coef_'):
            metrics['coefficients'] = self.model.coef_.tolist()
            metrics['intercept'] = float(self.model.intercept_)
        
        return metrics


class LinearRegressionModel(RegressionModel):
    """Linear Regression model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(algorithm='linear', config=config)


class RidgeModel(RegressionModel):
    """Ridge Regression model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(algorithm='ridge', config=config)


class LassoModel(RegressionModel):
    """Lasso Regression model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(algorithm='lasso', config=config)


class ElasticNetModel(RegressionModel):
    """Elastic Net Regression model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(algorithm='elastic_net', config=config)


class RandomForestRegressorModel(RegressionModel):
    """Random Forest Regressor model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(algorithm='random_forest', config=config)


class XGBoostRegressorModel(RegressionModel):
    """XGBoost Regressor model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(algorithm='xgboost', config=config)


class LightGBMRegressorModel(RegressionModel):
    """LightGBM Regressor model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(algorithm='lightgbm', config=config)
