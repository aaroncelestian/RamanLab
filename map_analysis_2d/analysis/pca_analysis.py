"""
Principal Component Analysis (PCA) module for Raman spectroscopy data.

This module provides PCA functionality extracted from the main UI module,
making it reusable and testable independently.
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple, Any
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class PCAAnalyzer:
    """PCA analysis functionality for Raman spectroscopy data."""
    
    def __init__(self):
        """Initialize the PCA analyzer."""
        self.pca = None
        self.scaler = None
        self.components = None
        self.explained_variance_ratio = None
        
    def run_analysis(self, data: np.ndarray, n_components: int = 5, 
                    batch_size: int = 5000, remove_nans: bool = True) -> Dict[str, Any]:
        """
        Run PCA analysis on the provided data.
        
        Args:
            data: Input data matrix (samples x features)
            n_components: Number of principal components to compute
            batch_size: Maximum samples to use for fitting (for large datasets)
            remove_nans: Whether to clean NaN/infinite values
            
        Returns:
            Dictionary containing PCA results
        """
        try:
            # Data validation and cleaning
            if remove_nans:
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            if data.size == 0:
                raise ValueError("No data available for PCA analysis")
                
            # For large datasets, subsample for fitting
            if len(data) > batch_size:
                np.random.seed(42)  # Reproducible sampling
                indices = np.random.choice(len(data), size=batch_size, replace=False)
                data_sample = data[indices]
            else:
                data_sample = data
            
            # Initialize and fit PCA
            self.pca = PCA(n_components=n_components, random_state=42)
            self.pca.fit(data_sample)
            
            # Transform all data
            self.components = self.pca.transform(data)
            self.explained_variance_ratio = self.pca.explained_variance_ratio_
            
            # Store scaler for consistency
            self.scaler = StandardScaler()
            self.scaler.fit(data_sample)
            
            return {
                'success': True,
                'components': self.components,
                'explained_variance_ratio': self.explained_variance_ratio,
                'n_components': n_components,
                'n_samples': len(data),
                'n_features': data.shape[1]
            }
            
        except Exception as e:
            logger.error(f"PCA analysis failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def transform_data(self, data: np.ndarray, fallback_to_full: bool = True) -> Tuple[Optional[np.ndarray], str]:
        """
        Transform new data using the fitted PCA model.
        
        Args:
            data: Input data to transform
            fallback_to_full: Whether to return full data if transformation fails
            
        Returns:
            Tuple of (transformed_data, feature_type_used)
        """
        if self.pca is None or self.scaler is None:
            logger.warning("PCA model not fitted")
            if fallback_to_full:
                return data, 'full'
            return None, 'none'
        
        try:
            # Check feature dimension compatibility
            expected_features = (self.scaler.n_features_in_ 
                               if hasattr(self.scaler, 'n_features_in_') 
                               else self.scaler.mean_.shape[0])
            actual_features = data.shape[1]
            
            if expected_features != actual_features:
                logger.warning(f"Feature dimension mismatch: expected {expected_features}, "
                             f"got {actual_features}")
                if fallback_to_full:
                    return data, 'full'
                return None, 'none'
            
            # Transform data
            transformed = self.pca.transform(data)
            return transformed, 'pca'
            
        except Exception as e:
            logger.error(f"PCA transformation failed: {str(e)}")
            if fallback_to_full:
                return data, 'full'
            return None, 'none'
    
    def get_loadings(self) -> Optional[np.ndarray]:
        """Get PCA loadings (components)."""
        if self.pca is None:
            return None
        return self.pca.components_
    
    def get_explained_variance(self) -> Optional[np.ndarray]:
        """Get explained variance ratios."""
        return self.explained_variance_ratio
    
    def save_results(self, filepath: str) -> bool:
        """
        Save PCA results to file.
        
        Args:
            filepath: Path to save the results
            
        Returns:
            Success status
        """
        if self.pca is None or self.components is None:
            logger.error("No PCA results to save")
            return False
        
        try:
            import pickle
            results = {
                'pca': self.pca,
                'components': self.components,
                'explained_variance_ratio': self.explained_variance_ratio,
                'scaler': self.scaler
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)
                
            logger.info(f"PCA results saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save PCA results: {str(e)}")
            return False
    
    def load_results(self, filepath: str) -> bool:
        """
        Load PCA results from file.
        
        Args:
            filepath: Path to load the results from
            
        Returns:
            Success status
        """
        try:
            import pickle
            with open(filepath, 'rb') as f:
                results = pickle.load(f)
            
            self.pca = results['pca']
            self.components = results['components']
            self.explained_variance_ratio = results['explained_variance_ratio']
            self.scaler = results.get('scaler')
            
            logger.info(f"PCA results loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load PCA results: {str(e)}")
            return False 