"""
Classification Models

This module provides implementations of various classification algorithms
including Logistic Regression, Random Forest, XGBoost, LightGBM, and SVM.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from loguru import logger

from .base import BaseModel


class ClassificationModel(BaseModel):
    """
    Classification model wrapper that supports multiple algorithms.
    
    Supported algorithms:
    - Logistic Regression
    - Random Forest
    - XGBoost
    - LightGBM
    - Support Vector Machine
    """
    
    def __init__(self, algorithm: str = 'random_forest', config: Optional[Dict[str, Any]] = None):
        """
        Initialize the classification model.
        
        Args:
            algorithm: Algorithm to use ('logistic_regression', 'random_forest', 
                       'xgboost', 'lightgbm', 'svm')
            config: Configuration dictionary
        """
        super().__init__(config)
        self.algorithm = algorithm.lower()
        self.classes_ = None
        
    def _initialize_model(self) -> Any:
        """Initialize the specific classification model."""
        model_config = self.config.get('models', {}).get('classification', {})
        
        if self.algorithm == 'logistic_regression':
            params = model_config.get('logistic_regression', {})
            return LogisticRegression(**params)
            
        elif self.algorithm == 'random_forest':
            params = model_config.get('random_forest', {})
            return RandomForestClassifier(**params)
            
        elif self.algorithm == 'xgboost':
            params = model_config.get('xgboost', {})
            return xgb.XGBClassifier(**params)
            
        elif self.algorithm == 'lightgbm':
            params = model_config.get('lightgbm', {})
            return lgb.LGBMClassifier(**params)
            
        elif self.algorithm == 'svm':
            params = model_config.get('svm', {})
            return SVC(probability=True, **params)
            
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def _train_model(self, X: Union[np.ndarray, pd.DataFrame], 
                    y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """Train the classification model."""
        # Store classes
        self.classes_ = np.unique(y)
        
        # Train the model
        self.model.fit(X, y)
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X) if hasattr(self.model, 'predict_proba') else None
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        
        training_history = {
            'algorithm': self.algorithm,
            'classes': self.classes_.tolist(),
            'n_classes': len(self.classes_),
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_accuracy': (y_pred == y).mean()
        }
        
        return training_history
    
    def _predict_model(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the trained classification model."""
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("This model does not support probability predictions")
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], 
                y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Evaluate the classification model.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X) if hasattr(self.model, 'predict_proba') else None
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1_score': f1_score(y, y_pred, average='weighted')
        }
        
        # ROC AUC for binary classification
        if len(self.classes_) == 2 and y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y, y_proba[:, 1])
        
        # Classification report
        report = classification_report(y, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics


class LogisticRegressionModel(ClassificationModel):
    """Logistic Regression classifier."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(algorithm='logistic_regression', config=config)


class RandomForestModel(ClassificationModel):
    """Random Forest classifier."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(algorithm='random_forest', config=config)


class XGBoostModel(ClassificationModel):
    """XGBoost classifier."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(algorithm='xgboost', config=config)


class LightGBMModel(ClassificationModel):
    """LightGBM classifier."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(algorithm='lightgbm', config=config)


class SVMModel(ClassificationModel):
    """Support Vector Machine classifier."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(algorithm='svm', config=config)
