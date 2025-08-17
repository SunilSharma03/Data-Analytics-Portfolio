"""
Base Model Class

This module provides the base class for all predictive models in the package.
It defines the common interface and functionality that all models should implement.
"""

import os
import pickle
import yaml
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from loguru import logger


class BaseModel(ABC, BaseEstimator):
    """
    Abstract base class for all predictive models.
    
    This class provides common functionality for model training, prediction,
    evaluation, saving, and loading. All specific model implementations
    should inherit from this class.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base model.
        
        Args:
            config: Configuration dictionary for model parameters
        """
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.target_name = None
        self.training_history = {}
        
        # Load configuration from file if not provided
        if not config:
            self._load_config()
    
    def _load_config(self):
        """Load configuration from config file."""
        try:
            config_path = "config/config.yaml"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                logger.info("Configuration loaded from config.yaml")
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")
            self.config = {}
    
    @abstractmethod
    def _initialize_model(self) -> Any:
        """
        Initialize the specific model implementation.
        
        Returns:
            The initialized model object
        """
        pass
    
    @abstractmethod
    def _train_model(self, X: Union[np.ndarray, pd.DataFrame], 
                    y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Dictionary containing training history and metrics
        """
        pass
    
    @abstractmethod
    def _predict_model(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        pass
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'BaseModel':
        """
        Fit the model to the training data.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training {self.__class__.__name__}")
        
        # Store feature and target names
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        if hasattr(y, 'name'):
            self.target_name = y.name
        
        # Initialize model if not already done
        if self.model is None:
            self.model = self._initialize_model()
        
        # Train the model
        self.training_history = self._train_model(X, y)
        self.is_trained = True
        
        logger.info(f"Training completed for {self.__class__.__name__}")
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"Making predictions with {self.__class__.__name__}")
        return self._predict_model(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities (for classification models).
        
        Args:
            X: Features to predict on
            
        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("This model does not support probability predictions")
    
    def score(self, X: Union[np.ndarray, pd.DataFrame], 
              y: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate the score of the model on the given data.
        
        Args:
            X: Features
            y: True targets
            
        Returns:
            Model score
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring")
        
        if hasattr(self.model, 'score'):
            return self.model.score(X, y)
        else:
            # Default implementation using predictions
            predictions = self.predict(X)
            from sklearn.metrics import accuracy_score, r2_score
            if len(np.unique(y)) <= 10:  # Classification
                return accuracy_score(y, predictions)
            else:  # Regression
                return r2_score(y, predictions)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.
        
        Args:
            deep: If True, will return the parameters for this estimator and
                  contained subobjects that are estimators.
                  
        Returns:
            Parameter names mapped to their values
        """
        return self.config.copy()
    
    def set_params(self, **params) -> 'BaseModel':
        """
        Set the parameters of this estimator.
        
        Args:
            **params: Parameter names and values to set
            
        Returns:
            Self for method chaining
        """
        self.config.update(params)
        return self
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'config': self.config,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> 'BaseModel':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Self with loaded model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.config = model_data['config']
        self.feature_names = model_data['feature_names']
        self.target_name = model_data['target_name']
        self.training_history = model_data['training_history']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
        return self
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores if available.
        
        Returns:
            Dictionary mapping feature names to importance scores, or None if not available
        """
        if not self.is_trained or self.feature_names is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            return dict(zip(self.feature_names, np.abs(self.model.coef_)))
        else:
            return None
    
    def get_training_history(self) -> Dict[str, Any]:
        """
        Get the training history and metrics.
        
        Returns:
            Dictionary containing training history
        """
        return self.training_history.copy()
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(config={self.config})"
    
    def __str__(self) -> str:
        """String representation of the model."""
        return self.__repr__()
