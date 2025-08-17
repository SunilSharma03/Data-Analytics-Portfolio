"""
Test Classification Models

This module contains tests for the classification models.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Import models to test
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


class TestClassificationModels(unittest.TestCase):
    """Test cases for classification models."""
    
    def setUp(self):
        """Set up test data."""
        # Load iris dataset
        iris = load_iris()
        self.X = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.y = pd.Series(iris.target, name='target')
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
    
    def test_logistic_regression(self):
        """Test Logistic Regression model."""
        model = LogisticRegressionModel()
        
        # Test training
        model.fit(self.X_train, self.y_train)
        self.assertTrue(model.is_trained)
        
        # Test prediction
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        
        # Test evaluation
        metrics = model.evaluate(self.X_test, self.y_test)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        
        # Test score
        score = model.score(self.X_test, self.y_test)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_random_forest(self):
        """Test Random Forest model."""
        model = RandomForestModel()
        
        # Test training
        model.fit(self.X_train, self.y_train)
        self.assertTrue(model.is_trained)
        
        # Test prediction
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        
        # Test feature importance
        importance = model.get_feature_importance()
        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), len(self.X.columns))
    
    def test_xgboost(self):
        """Test XGBoost model."""
        model = XGBoostModel()
        
        # Test training
        model.fit(self.X_train, self.y_train)
        self.assertTrue(model.is_trained)
        
        # Test prediction
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        
        # Test probability prediction
        proba = model.predict_proba(self.X_test)
        self.assertEqual(proba.shape[0], len(self.y_test))
    
    def test_lightgbm(self):
        """Test LightGBM model."""
        model = LightGBMModel()
        
        # Test training
        model.fit(self.X_train, self.y_train)
        self.assertTrue(model.is_trained)
        
        # Test prediction
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
    
    def test_svm(self):
        """Test SVM model."""
        model = SVMModel()
        
        # Test training
        model.fit(self.X_train, self.y_train)
        self.assertTrue(model.is_trained)
        
        # Test prediction
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        model = RandomForestModel()
        model.fit(self.X_train, self.y_train)
        
        # Save model
        model_path = "test_model.pkl"
        model.save(model_path)
        
        # Load model
        loaded_model = ClassificationModel()
        loaded_model.load(model_path)
        
        # Test predictions are the same
        original_pred = model.predict(self.X_test)
        loaded_pred = loaded_model.predict(self.X_test)
        
        np.testing.assert_array_equal(original_pred, loaded_pred)
        
        # Clean up
        os.remove(model_path)
    
    def test_invalid_predictions(self):
        """Test that untrained models raise errors."""
        model = RandomForestModel()
        
        with self.assertRaises(ValueError):
            model.predict(self.X_test)
        
        with self.assertRaises(ValueError):
            model.evaluate(self.X_test, self.y_test)
    
    def test_training_history(self):
        """Test that training history is recorded."""
        model = RandomForestModel()
        model.fit(self.X_train, self.y_train)
        
        history = model.get_training_history()
        self.assertIsInstance(history, dict)
        self.assertIn('algorithm', history)
        self.assertIn('cv_mean', history)


if __name__ == '__main__':
    unittest.main()
