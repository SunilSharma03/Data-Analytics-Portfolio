#!/usr/bin/env python3
"""
Time Series Forecasting Example

This script demonstrates how to use the time series forecasting models in the predictive modeling package.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our models
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.time_series import ProphetForecaster, ARIMAModel

def generate_sample_time_series(n_days=365):
    """Generate sample time series data for demonstration."""
    # Create date range
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Generate time series with trend and seasonality
    np.random.seed(42)
    trend = np.linspace(100, 150, n_days)  # Upward trend
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n_days) / 365)  # Annual seasonality
    noise = np.random.normal(0, 5, n_days)  # Random noise
    
    values = trend + seasonality + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'ds': dates,
        'y': values
    })
    
    return df

def main():
    """Main function demonstrating time series forecasting."""
    print("📈 Time Series Forecasting Example")
    print("=" * 50)
    
    # 1. Generate sample data
    print("\n1. Generating sample time series data...")
    data = generate_sample_time_series(365)
    
    print(f"Dataset shape: {data.shape}")
    print(f"Date range: {data['ds'].min()} to {data['ds'].max()}")
    print(f"Value range: {data['y'].min():.2f} to {data['y'].max():.2f}")
    
    # Split data for training and testing
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    print(f"Training set: {train_data.shape}")
    print(f"Test set: {test_data.shape}")
    
    # 2. Prophet Forecasting
    print("\n2. Training Prophet model...")
    
    try:
        prophet_model = ProphetForecaster()
        prophet_model.fit(train_data)
        
        # Make predictions
        prophet_forecast = prophet_model.predict(periods=len(test_data))
        
        # Evaluate
        prophet_metrics = prophet_model.evaluate(test_data)
        
        print(f"Prophet Results:")
        print(f"  MSE: {prophet_metrics['mse']:.2f}")
        print(f"  MAE: {prophet_metrics['mae']:.2f}")
        print(f"  RMSE: {prophet_metrics['rmse']:.2f}")
        print(f"  MAPE: {prophet_metrics['mape']:.2f}%")
        
        prophet_success = True
        
    except ImportError as e:
        print(f"Prophet not available: {e}")
        print("Install with: pip install prophet")
        prophet_success = False
    
    # 3. ARIMA Forecasting
    print("\n3. Training ARIMA model...")
    
    try:
        arima_model = ARIMAModel()
        arima_model.fit(train_data['y'])
        
        # Make predictions
        arima_forecast = arima_model.predict(steps=len(test_data))
        
        # Evaluate
        arima_metrics = arima_model.evaluate(test_data['y'])
        
        print(f"ARIMA Results:")
        print(f"  MSE: {arima_metrics['mse']:.2f}")
        print(f"  MAE: {arima_metrics['mae']:.2f}")
        print(f"  RMSE: {arima_metrics['rmse']:.2f}")
        print(f"  MAPE: {arima_metrics['mape']:.2f}%")
        
        arima_success = True
        
    except ImportError as e:
        print(f"ARIMA not available: {e}")
        print("Install with: pip install statsmodels pmdarima")
        arima_success = False
    
    # 4. Compare results
    if prophet_success and arima_success:
        print("\n4. Model Comparison:")
        print("-" * 60)
        print(f"{'Model':<15} {'MSE':<10} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
        print("-" * 60)
        
        prophet_metrics = prophet_model.evaluate(test_data)
        arima_metrics = arima_model.evaluate(test_data['y'])
        
        print(f"{'Prophet':<15} {prophet_metrics['mse']:<10.2f} {prophet_metrics['mae']:<10.2f} "
              f"{prophet_metrics['rmse']:<10.2f} {prophet_metrics['mape']:<10.2f}%")
        print(f"{'ARIMA':<15} {arima_metrics['mse']:<10.2f} {arima_metrics['mae']:<10.2f} "
              f"{arima_metrics['rmse']:<10.2f} {arima_metrics['mape']:<10.2f}%")
    
    # 5. Save best model
    print("\n5. Saving models...")
    
    if prophet_success:
        prophet_path = "../models/saved/prophet_forecaster.pkl"
        os.makedirs(os.path.dirname(prophet_path), exist_ok=True)
        prophet_model.save(prophet_path)
        print(f"Prophet model saved to: {prophet_path}")
    
    if arima_success:
        arima_path = "../models/saved/arima_model.pkl"
        os.makedirs(os.path.dirname(arima_path), exist_ok=True)
        arima_model.save(arima_path)
        print(f"ARIMA model saved to: {arima_path}")
    
    # 6. Future forecasting
    print("\n6. Making future forecasts...")
    
    if prophet_success:
        future_forecast = prophet_model.predict(periods=30)
        print(f"Prophet 30-day forecast range: {future_forecast['yhat'].min():.2f} to {future_forecast['yhat'].max():.2f}")
    
    if arima_success:
        future_arima = arima_model.predict(steps=30)
        print(f"ARIMA 30-day forecast range: {future_arima.min():.2f} to {future_arima.max():.2f}")
    
    print("\n✅ Time series forecasting example completed successfully!")
    
    return {
        'prophet_success': prophet_success,
        'arima_success': arima_success,
        'data': data,
        'train_data': train_data,
        'test_data': test_data
    }

if __name__ == "__main__":
    main()
