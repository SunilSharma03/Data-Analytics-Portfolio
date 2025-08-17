"""
Time Series Models

This module provides implementations of various time series forecasting algorithms
including Prophet and ARIMA models.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Union, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Prophet for time series forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None

# ARIMA models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from pmdarima import auto_arima
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    ARIMA = None
    auto_arima = None

from sklearn.metrics import mean_squared_error, mean_absolute_error
from loguru import logger

from .base import BaseModel


class ProphetForecaster(BaseModel):
    """
    Facebook Prophet time series forecaster.
    
    Prophet is designed for forecasting time series data that has strong seasonal effects
    and several seasons of historical data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Prophet forecaster.
        
        Args:
            config: Configuration dictionary
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not available. Install it with: pip install prophet")
        
        super().__init__(config)
        self.date_column = 'ds'
        self.value_column = 'y'
        
    def _initialize_model(self) -> Any:
        """Initialize the Prophet model."""
        model_config = self.config.get('models', {}).get('time_series', {}).get('prophet', {})
        
        # Get parameters from config
        changepoint_prior_scale = model_config.get('changepoint_prior_scale', 0.05)
        seasonality_prior_scale = model_config.get('seasonality_prior_scale', 10.0)
        holidays_prior_scale = model_config.get('holidays_prior_scale', 10.0)
        
        return Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            interval_width=0.95
        )
    
    def _prepare_data(self, data: Union[pd.DataFrame, pd.Series], 
                     date_column: Optional[str] = None,
                     value_column: Optional[str] = None) -> pd.DataFrame:
        """Prepare data for Prophet format."""
        if isinstance(data, pd.Series):
            # If data is a Series, assume index is dates
            df = pd.DataFrame({
                'ds': data.index,
                'y': data.values
            })
        else:
            # If data is DataFrame, use specified columns
            df = data.copy()
            
            if date_column is not None:
                df = df.rename(columns={date_column: 'ds'})
            if value_column is not None:
                df = df.rename(columns={value_column: 'y'})
            
            # Ensure we have the required columns
            if 'ds' not in df.columns or 'y' not in df.columns:
                raise ValueError("Data must have 'ds' (date) and 'y' (value) columns")
        
        # Ensure date column is datetime
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Sort by date
        df = df.sort_values('ds').reset_index(drop=True)
        
        return df
    
    def _train_model(self, data: Union[pd.DataFrame, pd.Series], 
                    y: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
        """Train the Prophet model."""
        # Prepare data
        df = self._prepare_data(data)
        
        # Fit the model
        self.model.fit(df)
        
        # Make predictions on training data
        forecast = self.model.predict(df)
        
        # Calculate training metrics
        mse = mean_squared_error(df['y'], forecast['yhat'])
        mae = mean_absolute_error(df['y'], forecast['yhat'])
        
        training_history = {
            'algorithm': 'prophet',
            'n_samples': len(df),
            'date_range': {
                'start': df['ds'].min().isoformat(),
                'end': df['ds'].max().isoformat()
            },
            'training_mse': mse,
            'training_mae': mae,
            'training_rmse': np.sqrt(mse)
        }
        
        return training_history
    
    def _predict_model(self, periods: int = 30, freq: str = 'D') -> pd.DataFrame:
        """Make predictions using the trained Prophet model."""
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        
        # Make predictions
        forecast = self.model.predict(future)
        
        return forecast
    
    def fit(self, data: Union[pd.DataFrame, pd.Series], 
            y: Optional[Union[np.ndarray, pd.Series]] = None) -> 'ProphetForecaster':
        """
        Fit the Prophet model to the time series data.
        
        Args:
            data: Time series data (DataFrame with 'ds' and 'y' columns, or Series with datetime index)
            y: Not used (kept for compatibility)
            
        Returns:
            Self for method chaining
        """
        logger.info("Training Prophet model")
        
        # Initialize model if not already done
        if self.model is None:
            self.model = self._initialize_model()
        
        # Train the model
        self.training_history = self._train_model(data, y)
        self.is_trained = True
        
        logger.info("Prophet training completed")
        return self
    
    def predict(self, periods: int = 30, freq: str = 'D') -> pd.DataFrame:
        """
        Make predictions for future periods.
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency of the time series ('D' for daily, 'M' for monthly, etc.)
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"Making Prophet predictions for {periods} periods")
        return self._predict_model(periods, freq)
    
    def evaluate(self, test_data: Union[pd.DataFrame, pd.Series]) -> Dict[str, Any]:
        """
        Evaluate the Prophet model on test data.
        
        Args:
            test_data: Test time series data
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Prepare test data
        test_df = self._prepare_data(test_data)
        
        # Make predictions
        forecast = self.model.predict(test_df)
        
        # Calculate metrics
        mse = mean_squared_error(test_df['y'], forecast['yhat'])
        mae = mean_absolute_error(test_df['y'], forecast['yhat'])
        rmse = np.sqrt(mse)
        
        # Calculate MAPE
        mape = np.mean(np.abs((test_df['y'] - forecast['yhat']) / test_df['y'])) * 100
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
        
        return metrics


class ARIMAModel(BaseModel):
    """
    ARIMA (Autoregressive Integrated Moving Average) time series model.
    
    ARIMA is a statistical model for analyzing and forecasting time series data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ARIMA model.
        
        Args:
            config: Configuration dictionary
        """
        if not ARIMA_AVAILABLE:
            raise ImportError("ARIMA is not available. Install it with: pip install statsmodels pmdarima")
        
        super().__init__(config)
        self.order = None
        self.seasonal_order = None
        
    def _initialize_model(self) -> Any:
        """Initialize the ARIMA model."""
        model_config = self.config.get('models', {}).get('time_series', {}).get('arima', {})
        
        # Get parameters from config
        self.order = model_config.get('order', (1, 1, 1))
        self.seasonal_order = model_config.get('seasonal_order', (1, 1, 1, 12))
        
        return None  # Model will be created during training
    
    def _check_stationarity(self, data: pd.Series) -> Dict[str, Any]:
        """Check if the time series is stationary."""
        result = adfuller(data.dropna())
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    
    def _find_best_order(self, data: pd.Series) -> tuple:
        """Find the best ARIMA order using auto_arima."""
        try:
            model = auto_arima(
                data,
                seasonal=True,
                m=12,  # Monthly seasonality
                suppress_warnings=True,
                error_action='ignore',
                stepwise=True
            )
            return model.order, model.seasonal_order
        except:
            # Fallback to default values
            return (1, 1, 1), (1, 1, 1, 12)
    
    def _train_model(self, data: Union[pd.DataFrame, pd.Series], 
                    y: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
        """Train the ARIMA model."""
        # Prepare data
        if isinstance(data, pd.DataFrame):
            # Assume first column is the time series
            ts_data = data.iloc[:, 0]
        else:
            ts_data = data
        
        # Convert to pandas Series if needed
        if not isinstance(ts_data, pd.Series):
            ts_data = pd.Series(ts_data)
        
        # Check stationarity
        stationarity = self._check_stationarity(ts_data)
        
        # Find best order if not specified
        if self.order is None:
            self.order, self.seasonal_order = self._find_best_order(ts_data)
        
        # Create and fit the model
        if self.seasonal_order is not None:
            self.model = ARIMA(ts_data, order=self.order, seasonal_order=self.seasonal_order)
        else:
            self.model = ARIMA(ts_data, order=self.order)
        
        self.fitted_model = self.model.fit()
        
        # Make predictions on training data
        predictions = self.fitted_model.predict(start=1, end=len(ts_data))
        
        # Calculate training metrics
        actual = ts_data.iloc[1:]  # Skip first value due to differencing
        predicted = predictions.iloc[:-1]  # Skip last prediction
        
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        
        training_history = {
            'algorithm': 'arima',
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'n_samples': len(ts_data),
            'is_stationary': stationarity['is_stationary'],
            'adf_p_value': stationarity['p_value'],
            'training_mse': mse,
            'training_mae': mae,
            'training_rmse': np.sqrt(mse),
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic
        }
        
        return training_history
    
    def _predict_model(self, steps: int = 30) -> pd.Series:
        """Make predictions using the trained ARIMA model."""
        # Make predictions
        forecast = self.fitted_model.forecast(steps=steps)
        
        return forecast
    
    def fit(self, data: Union[pd.DataFrame, pd.Series], 
            y: Optional[Union[np.ndarray, pd.Series]] = None) -> 'ARIMAModel':
        """
        Fit the ARIMA model to the time series data.
        
        Args:
            data: Time series data
            y: Not used (kept for compatibility)
            
        Returns:
            Self for method chaining
        """
        logger.info("Training ARIMA model")
        
        # Initialize model if not already done
        if self.model is None:
            self._initialize_model()
        
        # Train the model
        self.training_history = self._train_model(data, y)
        self.is_trained = True
        
        logger.info("ARIMA training completed")
        return self
    
    def predict(self, steps: int = 30) -> pd.Series:
        """
        Make predictions for future steps.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Series with predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"Making ARIMA predictions for {steps} steps")
        return self._predict_model(steps)
    
    def evaluate(self, test_data: Union[pd.DataFrame, pd.Series]) -> Dict[str, Any]:
        """
        Evaluate the ARIMA model on test data.
        
        Args:
            test_data: Test time series data
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Prepare test data
        if isinstance(test_data, pd.DataFrame):
            test_ts = test_data.iloc[:, 0]
        else:
            test_ts = test_data
        
        if not isinstance(test_ts, pd.Series):
            test_ts = pd.Series(test_ts)
        
        # Make predictions
        predictions = self.fitted_model.predict(start=len(test_ts), end=len(test_ts) + len(test_ts) - 1)
        
        # Calculate metrics
        mse = mean_squared_error(test_ts, predictions)
        mae = mean_absolute_error(test_ts, predictions)
        rmse = np.sqrt(mse)
        
        # Calculate MAPE
        mape = np.mean(np.abs((test_ts - predictions) / test_ts)) * 100
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
        
        return metrics
