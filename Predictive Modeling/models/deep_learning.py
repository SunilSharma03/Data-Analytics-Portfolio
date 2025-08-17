"""
Deep Learning Models

This module provides implementations of various deep learning architectures
including Neural Networks and LSTM models using TensorFlow/Keras.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from loguru import logger

from .base import BaseModel


class NeuralNetwork(BaseModel):
    """
    Neural Network model using TensorFlow/Keras.
    
    Supports both classification and regression tasks.
    """
    
    def __init__(self, layers: List[int] = [64, 32, 16], 
                 task: str = 'regression',
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the neural network.
        
        Args:
            layers: List of hidden layer sizes
            task: 'classification' or 'regression'
            config: Configuration dictionary
        """
        super().__init__(config)
        self.layers = layers
        self.task = task.lower()
        self.scaler = StandardScaler()
        self.label_encoder = None
        self.history = None
        
    def _initialize_model(self) -> Any:
        """Initialize the neural network model."""
        model_config = self.config.get('models', {}).get('deep_learning', {}).get('neural_network', {})
        
        # Get parameters from config
        activation = model_config.get('activation', 'relu')
        dropout_rate = model_config.get('dropout_rate', 0.2)
        learning_rate = model_config.get('learning_rate', 0.001)
        
        # Build the model
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Dense(self.layers[0], activation=activation, input_shape=(self.n_features_,)))
        model.add(layers.Dropout(dropout_rate))
        
        # Hidden layers
        for units in self.layers[1:]:
            model.add(layers.Dense(units, activation=activation))
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        if self.task == 'classification':
            if self.n_classes_ == 2:
                model.add(layers.Dense(1, activation='sigmoid'))
            else:
                model.add(layers.Dense(self.n_classes_, activation='softmax'))
        else:  # regression
            model.add(layers.Dense(1, activation='linear'))
        
        # Compile the model
        if self.task == 'classification':
            if self.n_classes_ == 2:
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
        else:  # regression
            loss = 'mse'
            metrics = ['mae']
        
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model
    
    def _train_model(self, X: Union[np.ndarray, pd.DataFrame], 
                    y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """Train the neural network."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels for classification
        if self.task == 'classification':
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
                self.n_classes_ = len(self.label_encoder.classes_)
            else:
                y_encoded = self.label_encoder.transform(y)
        else:
            y_encoded = y
            self.n_classes_ = 1
        
        # Store number of features
        self.n_features_ = X_scaled.shape[1]
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42
        )
        
        # Get training parameters from config
        model_config = self.config.get('models', {}).get('deep_learning', {}).get('neural_network', {})
        batch_size = model_config.get('batch_size', 32)
        epochs = model_config.get('epochs', 100)
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks_list,
            verbose=0
        )
        
        # Calculate training metrics
        train_loss, train_metric = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_metric = self.model.evaluate(X_val, y_val, verbose=0)
        
        training_history = {
            'task': self.task,
            'layers': self.layers,
            'n_features': self.n_features_,
            'n_classes': self.n_classes_,
            'train_loss': float(train_loss),
            'train_metric': float(train_metric),
            'val_loss': float(val_loss),
            'val_metric': float(val_metric),
            'history': {
                'loss': [float(x) for x in self.history.history['loss']],
                'val_loss': [float(x) for x in self.history.history['val_loss']],
                'accuracy': [float(x) for x in self.history.history.get('accuracy', [])],
                'val_accuracy': [float(x) for x in self.history.history.get('val_accuracy', [])],
                'mae': [float(x) for x in self.history.history.get('mae', [])],
                'val_mae': [float(x) for x in self.history.history.get('val_mae', [])]
            }
        }
        
        return training_history
    
    def _predict_model(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the trained neural network."""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        
        if self.task == 'classification':
            if self.n_classes_ == 2:
                return (predictions > 0.5).astype(int)
            else:
                return np.argmax(predictions, axis=1)
        else:  # regression
            return predictions.flatten()
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities for classification."""
        if self.task != 'classification':
            raise ValueError("Probability predictions only available for classification tasks")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0)


class LSTMModel(BaseModel):
    """
    LSTM model for time series and sequence data.
    
    Supports both classification and regression tasks.
    """
    
    def __init__(self, sequence_length: int = 10,
                 task: str = 'regression',
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LSTM model.
        
        Args:
            sequence_length: Length of input sequences
            task: 'classification' or 'regression'
            config: Configuration dictionary
        """
        super().__init__(config)
        self.sequence_length = sequence_length
        self.task = task.lower()
        self.scaler = StandardScaler()
        self.label_encoder = None
        self.history = None
        
    def _initialize_model(self) -> Any:
        """Initialize the LSTM model."""
        model_config = self.config.get('models', {}).get('deep_learning', {}).get('lstm', {})
        
        # Get parameters from config
        units = model_config.get('units', 50)
        dropout = model_config.get('dropout', 0.2)
        recurrent_dropout = model_config.get('recurrent_dropout', 0.2)
        
        # Build the model
        model = models.Sequential()
        
        # LSTM layer
        model.add(layers.LSTM(
            units, 
            dropout=dropout, 
            recurrent_dropout=recurrent_dropout,
            input_shape=(self.sequence_length, self.n_features_)
        ))
        
        # Output layer
        if self.task == 'classification':
            if self.n_classes_ == 2:
                model.add(layers.Dense(1, activation='sigmoid'))
            else:
                model.add(layers.Dense(self.n_classes_, activation='softmax'))
        else:  # regression
            model.add(layers.Dense(1, activation='linear'))
        
        # Compile the model
        if self.task == 'classification':
            if self.n_classes_ == 2:
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
        else:  # regression
            loss = 'mse'
            metrics = ['mae']
        
        optimizer = optimizers.Adam()
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model
    
    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM input."""
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X) - self.sequence_length):
            X_sequences.append(X[i:(i + self.sequence_length)])
            y_sequences.append(y[i + self.sequence_length])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def _train_model(self, X: Union[np.ndarray, pd.DataFrame], 
                    y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """Train the LSTM model."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels for classification
        if self.task == 'classification':
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
                self.n_classes_ = len(self.label_encoder.classes_)
            else:
                y_encoded = self.label_encoder.transform(y)
        else:
            y_encoded = y
            self.n_classes_ = 1
        
        # Store number of features
        self.n_features_ = X_scaled.shape[1]
        
        # Prepare sequences
        X_sequences, y_sequences = self._prepare_sequences(X_scaled, y_encoded)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_sequences, y_sequences, test_size=0.2, random_state=42
        )
        
        # Get training parameters from config
        model_config = self.config.get('models', {}).get('deep_learning', {}).get('lstm', {})
        batch_size = model_config.get('batch_size', 32)
        epochs = model_config.get('epochs', 100)
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks_list,
            verbose=0
        )
        
        # Calculate training metrics
        train_loss, train_metric = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_metric = self.model.evaluate(X_val, y_val, verbose=0)
        
        training_history = {
            'task': self.task,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features_,
            'n_classes': self.n_classes_,
            'train_loss': float(train_loss),
            'train_metric': float(train_metric),
            'val_loss': float(val_loss),
            'val_metric': float(val_metric),
            'history': {
                'loss': [float(x) for x in self.history.history['loss']],
                'val_loss': [float(x) for x in self.history.history['val_loss']],
                'accuracy': [float(x) for x in self.history.history.get('accuracy', [])],
                'val_accuracy': [float(x) for x in self.history.history.get('val_accuracy', [])],
                'mae': [float(x) for x in self.history.history.get('mae', [])],
                'val_mae': [float(x) for x in self.history.history.get('val_mae', [])]
            }
        }
        
        return training_history
    
    def _predict_model(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the trained LSTM model."""
        X_scaled = self.scaler.transform(X)
        X_sequences, _ = self._prepare_sequences(X_scaled, np.zeros(len(X_scaled)))
        
        predictions = self.model.predict(X_sequences, verbose=0)
        
        if self.task == 'classification':
            if self.n_classes_ == 2:
                return (predictions > 0.5).astype(int)
            else:
                return np.argmax(predictions, axis=1)
        else:  # regression
            return predictions.flatten()
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities for classification."""
        if self.task != 'classification':
            raise ValueError("Probability predictions only available for classification tasks")
        
        X_scaled = self.scaler.transform(X)
        X_sequences, _ = self._prepare_sequences(X_scaled, np.zeros(len(X_scaled)))
        return self.model.predict(X_sequences, verbose=0)
