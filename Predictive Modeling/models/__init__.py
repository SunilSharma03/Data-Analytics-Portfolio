"""
Predictive Modeling Package

This package contains implementations of various predictive modeling techniques
including classification, regression, clustering, deep learning, and time series forecasting.
"""

from .base import BaseModel
from .classification import ClassificationModel
from .regression import RegressionModel
from .clustering import ClusteringModel
from .deep_learning import NeuralNetwork, LSTMModel
from .time_series import ProphetForecaster, ARIMAModel

__version__ = "1.0.0"
__author__ = "Predictive Modeling Team"

__all__ = [
    "BaseModel",
    "ClassificationModel", 
    "RegressionModel",
    "ClusteringModel",
    "NeuralNetwork",
    "LSTMModel",
    "ProphetForecaster",
    "ARIMAModel"
]
