#!/usr/bin/env python3
"""
Classification Example

This script demonstrates how to use the classification models in the predictive modeling package.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import our models
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classification import (
    ClassificationModel, 
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    SVMModel
)

def main():
    """Main function demonstrating classification."""
    print("🔍 Classification Example")
    print("=" * 50)
    
    # 1. Load and prepare data
    print("\n1. Loading Iris dataset...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target classes: {iris.target_names}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # 2. Train different models
    print("\n2. Training classification models...")
    
    models = {
        'Logistic Regression': LogisticRegressionModel(),
        'Random Forest': RandomForestModel(),
        'XGBoost': XGBoostModel(),
        'LightGBM': LightGBMModel(),
        'SVM': SVMModel()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        results[name] = metrics
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
    
    # 3. Compare results
    print("\n3. Model Comparison:")
    print("-" * 80)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 80)
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f}")
    
    # 4. Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_model = models[best_model_name]
    best_metrics = results[best_model_name]
    
    print(f"\n4. Best Model: {best_model_name}")
    print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
    
    # 5. Feature importance
    print("\n5. Feature Importance Analysis:")
    feature_importance = best_model.get_feature_importance()
    
    if feature_importance:
        importance_df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        }).sort_values('Importance', ascending=False)
        
        print("\nTop features:")
        for _, row in importance_df.head().iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    # 6. Save best model
    print("\n6. Saving best model...")
    model_path = f"../models/saved/{best_model_name.lower().replace(' ', '_')}_iris.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    best_model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # 7. Test model loading
    print("\n7. Testing model loading...")
    loaded_model = ClassificationModel()
    loaded_model.load(model_path)
    
    # Make a test prediction
    test_sample = X_test.iloc[:1]
    prediction = loaded_model.predict(test_sample)
    actual = y_test.iloc[0]
    
    print(f"Test prediction: {prediction[0]} ({iris.target_names[prediction[0]]})")
    print(f"Actual value: {actual} ({iris.target_names[actual]})")
    
    print("\n✅ Classification example completed successfully!")
    
    return results

if __name__ == "__main__":
    main()
