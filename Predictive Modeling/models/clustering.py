"""
Clustering Models

This module provides implementations of various clustering algorithms
including K-Means, DBSCAN, Hierarchical clustering, and Gaussian Mixture Models.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Union
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from loguru import logger

from .base import BaseModel


class ClusteringModel(BaseModel):
    """
    Clustering model wrapper that supports multiple algorithms.
    
    Supported algorithms:
    - K-Means
    - DBSCAN
    - Hierarchical Clustering
    - Gaussian Mixture Model
    """
    
    def __init__(self, algorithm: str = 'kmeans', config: Optional[Dict[str, Any]] = None):
        """
        Initialize the clustering model.
        
        Args:
            algorithm: Algorithm to use ('kmeans', 'dbscan', 'hierarchical', 'gmm')
            config: Configuration dictionary
        """
        super().__init__(config)
        self.algorithm = algorithm.lower()
        self.scaler = StandardScaler()
        self.labels_ = None
        self.n_clusters_ = None
        
    def _initialize_model(self) -> Any:
        """Initialize the specific clustering model."""
        model_config = self.config.get('models', {}).get('clustering', {})
        
        if self.algorithm == 'kmeans':
            params = model_config.get('kmeans', {})
            return KMeans(**params)
            
        elif self.algorithm == 'dbscan':
            params = model_config.get('dbscan', {})
            return DBSCAN(**params)
            
        elif self.algorithm == 'hierarchical':
            params = model_config.get('hierarchical', {})
            return AgglomerativeClustering(**params)
            
        elif self.algorithm == 'gmm':
            params = model_config.get('gmm', {})
            return GaussianMixture(**params)
            
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def _train_model(self, X: Union[np.ndarray, pd.DataFrame], 
                    y: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
        """Train the clustering model."""
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled)
        
        # Get cluster labels
        if hasattr(self.model, 'labels_'):
            self.labels_ = self.model.labels_
        elif hasattr(self.model, 'predict'):
            self.labels_ = self.model.predict(X_scaled)
        
        # Get number of clusters
        if hasattr(self.model, 'n_clusters_'):
            self.n_clusters_ = self.model.n_clusters_
        elif hasattr(self.model, 'n_components_'):
            self.n_clusters_ = self.model.n_components_
        else:
            self.n_clusters_ = len(np.unique(self.labels_))
        
        # Calculate clustering metrics
        metrics = self._calculate_clustering_metrics(X_scaled, self.labels_)
        
        training_history = {
            'algorithm': self.algorithm,
            'n_clusters': self.n_clusters_,
            'n_samples': len(X),
            'metrics': metrics
        }
        
        return training_history
    
    def _predict_model(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict cluster labels for new data."""
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict'):
            return self.model.predict(X_scaled)
        elif hasattr(self.model, 'labels_'):
            # For models that don't have predict method, use fit_predict
            return self.model.fit_predict(X_scaled)
        else:
            raise NotImplementedError("This model does not support prediction")
    
    def _calculate_clustering_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering evaluation metrics."""
        metrics = {}
        
        # Silhouette score (higher is better)
        try:
            metrics['silhouette_score'] = silhouette_score(X, labels)
        except:
            metrics['silhouette_score'] = None
        
        # Calinski-Harabasz score (higher is better)
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
        except:
            metrics['calinski_harabasz_score'] = None
        
        # Davies-Bouldin score (lower is better)
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
        except:
            metrics['davies_bouldin_score'] = None
        
        return metrics
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Optional[Union[np.ndarray, pd.Series]] = None) -> 'ClusteringModel':
        """
        Fit the clustering model to the data.
        
        Args:
            X: Training features
            y: Not used for clustering (kept for compatibility)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training {self.__class__.__name__}")
        
        # Store feature names
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        
        # Initialize model if not already done
        if self.model is None:
            self.model = self._initialize_model()
        
        # Train the model
        self.training_history = self._train_model(X, y)
        self.is_trained = True
        
        logger.info(f"Training completed for {self.__class__.__name__}")
        return self
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        Evaluate the clustering model.
        
        Args:
            X: Data to evaluate on
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        X_scaled = self.scaler.transform(X)
        labels = self.predict(X)
        
        metrics = self._calculate_clustering_metrics(X_scaled, labels)
        
        # Add additional information
        metrics['n_clusters'] = self.n_clusters_
        metrics['n_samples'] = len(X)
        metrics['cluster_sizes'] = np.bincount(labels).tolist()
        
        return metrics
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get cluster centers if available."""
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'cluster_centers_'):
            # Transform back to original scale
            return self.scaler.inverse_transform(self.model.cluster_centers_)
        elif hasattr(self.model, 'means_'):
            # For GMM
            return self.scaler.inverse_transform(self.model.means_)
        else:
            return None
    
    def get_cluster_labels(self) -> Optional[np.ndarray]:
        """Get cluster labels from training."""
        return self.labels_


class KMeansModel(ClusteringModel):
    """K-Means clustering model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(algorithm='kmeans', config=config)


class DBSCANModel(ClusteringModel):
    """DBSCAN clustering model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(algorithm='dbscan', config=config)


class HierarchicalModel(ClusteringModel):
    """Hierarchical clustering model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(algorithm='hierarchical', config=config)


class GaussianMixtureModel(ClusteringModel):
    """Gaussian Mixture Model clustering."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(algorithm='gmm', config=config)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict cluster probabilities for GMM."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
