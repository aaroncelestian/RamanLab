"""
Non-negative Matrix Factorization (NMF) module for Raman spectroscopy data.

This module provides NMF functionality extracted from the main UI module,
making it reusable and testable independently.
"""

import numpy as np
import logging
import os
from typing import Dict, Optional, Tuple, Any
from sklearn.decomposition import NMF

logger = logging.getLogger(__name__)


class NMFAnalyzer:
    """NMF analysis functionality for Raman spectroscopy data."""
    
    def __init__(self):
        """Initialize the NMF analyzer."""
        self.nmf = None
        self.components = None  # W matrix (samples x components)
        self.feature_components = None  # H matrix (components x features)
        self.reconstruction_error = None
        
    def run_analysis(self, data: np.ndarray, n_components: int = 5, 
                    batch_size: int = 2000, max_iter: int = 200) -> Dict[str, Any]:
        """
        Run NMF analysis on the provided data.
        
        Args:
            data: Input data matrix (samples x features) - must be non-negative
            n_components: Number of components to extract
            batch_size: Maximum samples to use for fitting (for large datasets)
            max_iter: Maximum iterations for NMF convergence
            
        Returns:
            Dictionary containing NMF results
        """
        try:
            # Set environment variables to prevent threading issues
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['NUMEXPR_NUM_THREADS'] = '1'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            
            # Data validation and cleaning
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Make values non-negative (NMF requirement)
            data = np.abs(data)
            
            if data.size == 0:
                raise ValueError("No data available for NMF analysis")
            
            # Enhanced data cleaning for stability
            original_shape = data.shape
            logger.info(f"Original data shape: {original_shape}")
            
            # Remove zero-variance columns
            col_variances = np.var(data, axis=0)
            non_zero_cols = col_variances > 1e-10
            
            if not np.any(non_zero_cols):
                raise ValueError("All features have zero variance after preprocessing")
            
            if np.sum(non_zero_cols) < len(non_zero_cols):
                logger.info(f"Removing {len(non_zero_cols) - np.sum(non_zero_cols)} zero-variance columns")
                data = data[:, non_zero_cols]
            
            # Remove rows that are all zeros
            row_sums = np.sum(data, axis=1)
            non_zero_rows = row_sums > 1e-10
            
            if not np.any(non_zero_rows):
                raise ValueError("All samples have zero intensity after preprocessing")
            
            if np.sum(non_zero_rows) < len(non_zero_rows):
                logger.info(f"Removing {len(non_zero_rows) - np.sum(non_zero_rows)} zero-intensity rows")
                data = data[non_zero_rows, :]
            
            logger.info(f"Cleaned data shape: {data.shape}")
            
            # Validate dimensions for NMF
            if data.shape[0] < n_components:
                raise ValueError(f"Not enough samples ({data.shape[0]}) for {n_components} components")
            if data.shape[1] < n_components:
                raise ValueError(f"Not enough features ({data.shape[1]}) for {n_components} components")
            
            # Apply safety limits for memory and stability
            max_samples = min(len(data), max(batch_size * 2, 20000))
            max_features = min(data.shape[1], 3000)
            
            if len(data) > max_samples or data.shape[1] > max_features:
                logger.info(f"Reducing data size for stability: {data.shape} -> ({max_samples}, {max_features})")
                
                # Sample rows
                if len(data) > max_samples:
                    np.random.seed(42)
                    row_indices = np.random.choice(len(data), size=max_samples, replace=False)
                    data = data[row_indices, :]
                
                # Sample features
                if data.shape[1] > max_features:
                    np.random.seed(42)
                    col_indices = np.random.choice(data.shape[1], size=max_features, replace=False)
                    data = data[:, col_indices]
                
                logger.info(f"Reduced data shape: {data.shape}")
            
            # Use user's batch size for fitting
            fit_batch_size = min(batch_size, len(data))
            
            if len(data) > fit_batch_size:
                np.random.seed(42)  # Reproducible sampling
                indices = np.random.choice(len(data), size=fit_batch_size, replace=False)
                data_sample = data[indices]
            else:
                data_sample = data
            
            # Add small epsilon to prevent zero matrices
            epsilon = 1e-10
            data_sample = data_sample + epsilon
            data = data + epsilon
            
            # Initialize NMF with safe parameters
            self.nmf = NMF(
                n_components=min(n_components, min(data.shape) - 1),
                random_state=42,
                max_iter=max_iter,
                tol=1e-3,
                init='random',
                solver='mu',  # Multiplicative update solver (more stable)
                beta_loss='frobenius'
            )
            
            logger.info(f"Fitting NMF with {self.nmf.n_components} components on {data_sample.shape[0]} samples...")
            
            try:
                # Fit NMF
                W_sample = self.nmf.fit_transform(data_sample)
                H = self.nmf.components_
                logger.info(f"NMF fit successful - converged in {self.nmf.n_iter_} iterations")
            except Exception as e:
                logger.warning(f"NMF fit failed: {str(e)}")
                # Try fallback with minimal parameters
                logger.info("Trying fallback NMF with minimal parameters...")
                self.nmf = NMF(
                    n_components=min(5, min(data_sample.shape) - 1),
                    random_state=42,
                    max_iter=100,
                    tol=1e-2,
                    init='random',
                    solver='mu'
                )
                W_sample = self.nmf.fit_transform(data_sample)
                H = self.nmf.components_
                logger.info("Fallback NMF successful")
            
            # Transform all data
            logger.info("Transforming full dataset...")
            
            # Process data in batches to avoid memory issues
            transform_batch_size = min(1000, len(data))
            n_batches = (len(data) + transform_batch_size - 1) // transform_batch_size
            
            W_full = np.zeros((len(data), self.nmf.n_components))
            
            for i in range(n_batches):
                start_idx = i * transform_batch_size
                end_idx = min((i + 1) * transform_batch_size, len(data))
                batch_data = data[start_idx:end_idx]
                
                try:
                    W_full[start_idx:end_idx] = self.nmf.transform(batch_data)
                except Exception as e:
                    logger.warning(f"Transform failed for batch {i+1}/{n_batches}: {str(e)}")
                    # Use zeros for failed batch
                    W_full[start_idx:end_idx] = 0.0
            
            self.components = W_full
            self.feature_components = H
            self.reconstruction_error = self.nmf.reconstruction_err_
            
            return {
                'success': True,
                'components': self.components,
                'feature_components': self.feature_components,
                'reconstruction_error': self.reconstruction_error,
                'n_components': self.nmf.n_components,
                'n_samples': len(data),
                'n_features': data.shape[1],
                'n_iterations': self.nmf.n_iter_
            }
            
        except Exception as e:
            logger.error(f"NMF analysis failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def transform_data(self, data: np.ndarray, fallback_to_full: bool = True) -> Tuple[Optional[np.ndarray], str]:
        """
        Transform new data using the fitted NMF model.
        
        Args:
            data: Input data to transform
            fallback_to_full: Whether to return full data if transformation fails
            
        Returns:
            Tuple of (transformed_data, feature_type_used)
        """
        if self.nmf is None:
            logger.warning("NMF model not fitted")
            if fallback_to_full:
                return data, 'full'
            return None, 'none'
        
        try:
            # Ensure data is non-negative and clean
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            data = np.abs(data) + 1e-10  # Add epsilon
            
            # Check feature dimensions
            expected_features = self.nmf.n_features_in_ if hasattr(self.nmf, 'n_features_in_') else self.feature_components.shape[1]
            actual_features = data.shape[1]
            
            logger.info(f"NMF transform: Expected {expected_features} features, got {actual_features}")
            
            if actual_features != expected_features:
                logger.warning(f"Feature dimension mismatch: X has {actual_features} features, but NMF is expecting {expected_features} features as input.")
                
                # Try to align features
                if actual_features > expected_features:
                    # Truncate excess features
                    logger.info(f"Truncating {actual_features - expected_features} excess features")
                    data = data[:, :expected_features]
                elif actual_features < expected_features:
                    # Pad with zeros
                    logger.info(f"Padding with {expected_features - actual_features} zero features")
                    padding = np.zeros((data.shape[0], expected_features - actual_features))
                    data = np.hstack([data, padding])
                
                # Verify dimensions after alignment
                if data.shape[1] != expected_features:
                    raise ValueError(f"Could not align features: still have dimension mismatch after alignment")
            
            # Transform data in batches to avoid memory issues
            batch_size = min(1000, data.shape[0])
            n_batches = (data.shape[0] + batch_size - 1) // batch_size
            
            transformed_batches = []
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, data.shape[0])
                batch_data = data[start_idx:end_idx]
                
                try:
                    batch_transformed = self.nmf.transform(batch_data)
                    transformed_batches.append(batch_transformed)
                except Exception as batch_error:
                    logger.warning(f"NMF transform failed for batch {i+1}/{n_batches}: {str(batch_error)}")
                    # Use zeros for failed batch
                    batch_transformed = np.zeros((batch_data.shape[0], self.nmf.n_components))
                    transformed_batches.append(batch_transformed)
            
            transformed = np.vstack(transformed_batches)
            logger.info(f"NMF transformation successful: {data.shape} -> {transformed.shape}")
            return transformed, 'nmf'
            
        except Exception as e:
            logger.error(f"NMF transformation failed: {str(e)}")
            if fallback_to_full:
                logger.info("Falling back to full spectrum features")
                return data, 'full'
            return None, 'none'
    
    def get_components(self) -> Optional[np.ndarray]:
        """Get NMF components (W matrix)."""
        return self.components
    
    def get_feature_components(self) -> Optional[np.ndarray]:
        """Get NMF feature components (H matrix)."""
        return self.feature_components
    
    def get_reconstruction_error(self) -> Optional[float]:
        """Get reconstruction error."""
        return self.reconstruction_error
    
    def save_results(self, filepath: str) -> bool:
        """
        Save NMF results to file.
        
        Args:
            filepath: Path to save the results
            
        Returns:
            Success status
        """
        if self.nmf is None or self.components is None:
            logger.error("No NMF results to save")
            return False
        
        try:
            import pickle
            results = {
                'nmf': self.nmf,
                'components': self.components,
                'feature_components': self.feature_components,
                'reconstruction_error': self.reconstruction_error
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)
                
            logger.info(f"NMF results saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save NMF results: {str(e)}")
            return False
    
    def load_results(self, filepath: str) -> bool:
        """
        Load NMF results from file.
        
        Args:
            filepath: Path to load the results from
            
        Returns:
            Success status
        """
        try:
            import pickle
            with open(filepath, 'rb') as f:
                results = pickle.load(f)
            
            self.nmf = results['nmf']
            self.components = results['components']
            self.feature_components = results['feature_components']
            self.reconstruction_error = results['reconstruction_error']
            
            logger.info(f"NMF results loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load NMF results: {str(e)}")
            return False 