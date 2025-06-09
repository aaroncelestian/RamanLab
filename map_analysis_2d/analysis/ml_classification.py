"""
Machine Learning Classification module for Raman spectroscopy data.

This module provides ML classification functionality extracted from the main UI module,
making it reusable and testable independently.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class MLTrainingDataManager:
    """Manager for ML training data from multiple classes."""
    
    def __init__(self):
        """Initialize the training data manager."""
        self.class_data = {}  # class_name -> list of spectra
        self.class_names = []
        
    def load_class_data(self, class_directories: Dict[str, str], 
                       cosmic_ray_manager=None) -> Dict[str, Any]:
        """
        Load training data from multiple class directories.
        
        Args:
            class_directories: Dictionary mapping class names to directory paths
            cosmic_ray_manager: Optional cosmic ray removal manager
            
        Returns:
            Dictionary containing loading results
        """
        try:
            self.class_data = {}
            self.class_names = list(class_directories.keys())
            total_spectra = 0
            
            for class_name, directory in class_directories.items():
                class_spectra = SpectrumLoader.load_from_directory(
                    directory, label=class_name, cosmic_ray_manager=cosmic_ray_manager
                )
                self.class_data[class_name] = class_spectra
                total_spectra += len(class_spectra)
                
                logger.info(f"Loaded {len(class_spectra)} spectra for class '{class_name}'")
            
            return {
                'success': True,
                'n_classes': len(self.class_names),
                'class_names': self.class_names,
                'total_spectra': total_spectra,
                'class_counts': {name: len(data) for name, data in self.class_data.items()}
            }
            
        except Exception as e:
            logger.error(f"Failed to load training data: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_training_data(self, preprocessor=None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Get formatted training data for ML algorithms.
        
        Args:
            preprocessor: Optional preprocessing function
            
        Returns:
            Tuple of (features, labels, class_names)
        """
        if not self.class_data:
            raise ValueError("No training data loaded")
        
        # First pass: collect all wavenumbers and find common range
        all_wavenumbers = []
        all_spectra_info = []
        
        for class_name, spectra_data in self.class_data.items():
            for spectrum_data, _ in spectra_data:
                wavenumbers = spectrum_data['wavenumbers']
                intensities = spectrum_data['intensities']
                all_wavenumbers.append(wavenumbers)
                all_spectra_info.append((class_name, wavenumbers, intensities))
        
        # Find common wavenumber range
        min_wn = max(wn.min() for wn in all_wavenumbers)
        max_wn = min(wn.max() for wn in all_wavenumbers)
        
        if min_wn >= max_wn:
            raise ValueError("No overlapping wavenumber range found between spectra")
        
        # Create common wavenumber grid (use the spectrum with most points in range as reference)
        best_spectrum_idx = 0
        best_points_in_range = 0
        
        for i, wavenumbers in enumerate(all_wavenumbers):
            points_in_range = np.sum((wavenumbers >= min_wn) & (wavenumbers <= max_wn))
            if points_in_range > best_points_in_range:
                best_points_in_range = points_in_range
                best_spectrum_idx = i
        
        reference_wn = all_wavenumbers[best_spectrum_idx]
        mask = (reference_wn >= min_wn) & (reference_wn <= max_wn)
        common_wavenumbers = reference_wn[mask]
        
        logger.info(f"Using common wavenumber range: {float(min_wn):.1f} - {float(max_wn):.1f} cm⁻¹")
        logger.info(f"Common grid has {len(common_wavenumbers)} points")
        
        # Second pass: interpolate all spectra to common grid
        X = []
        y = []
        
        # Create label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(self.class_names)
        
        from scipy.interpolate import interp1d
        
        for class_name, wavenumbers, intensities in all_spectra_info:
            class_label = label_encoder.transform([class_name])[0]
            
            try:
                # Preprocess spectrum if preprocessor is available
                if preprocessor:
                    processed_spectrum = preprocessor(wavenumbers, intensities)
                else:
                    processed_spectrum = intensities
                
                # Interpolate to common wavenumber grid
                if len(wavenumbers) == len(common_wavenumbers) and np.allclose(wavenumbers, common_wavenumbers):
                    # Already on the right grid
                    interpolated_spectrum = processed_spectrum
                else:
                    # Need to interpolate
                    interp_func = interp1d(wavenumbers, processed_spectrum, 
                                         kind='linear', bounds_error=False, 
                                         fill_value='extrapolate')
                    interpolated_spectrum = interp_func(common_wavenumbers)
                
                # Remove any NaN or infinite values
                interpolated_spectrum = np.nan_to_num(interpolated_spectrum, nan=0.0, posinf=0.0, neginf=0.0)
                
                X.append(interpolated_spectrum)
                y.append(class_label)
                
            except Exception as e:
                logger.warning(f"Failed to process spectrum from class {class_name}: {str(e)}")
                continue
        
        if len(X) == 0:
            raise ValueError("No valid spectra could be processed")
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        logger.info(f"Successfully processed {len(X_array)} spectra")
        logger.info(f"Feature matrix shape: {X_array.shape}")
        
        return X_array, y_array, self.class_names, common_wavenumbers
    
    def get_class_info(self) -> Dict[str, Any]:
        """Get information about loaded classes."""
        if not self.class_data:
            return {'n_classes': 0, 'class_names': [], 'class_counts': {}}
        
        return {
            'n_classes': len(self.class_names),
            'class_names': self.class_names,
            'class_counts': {name: len(data) for name, data in self.class_data.items()}
        }
    
    def validate_training_data(self) -> Dict[str, Any]:
        """
        Validate training data and provide diagnostic information.
        
        Returns:
            Dictionary containing validation results and diagnostics
        """
        if not self.class_data:
            return {'valid': False, 'error': 'No training data loaded'}
        
        diagnostics = {
            'valid': True,
            'total_spectra': 0,
            'classes': {},
            'wavenumber_ranges': {},
            'spectrum_lengths': {},
            'issues': []
        }
        
        all_wavenumbers = []
        
        for class_name, spectra_data in self.class_data.items():
            class_info = {
                'count': len(spectra_data),
                'wavenumber_ranges': [],
                'spectrum_lengths': [],
                'files': []
            }
            
            for spectrum_data, _ in spectra_data:
                wavenumbers = spectrum_data['wavenumbers']
                intensities = spectrum_data['intensities']
                filename = spectrum_data.get('filename', 'unknown')
                
                wn_min, wn_max = wavenumbers.min(), wavenumbers.max()
                class_info['wavenumber_ranges'].append((wn_min, wn_max))
                class_info['spectrum_lengths'].append(len(wavenumbers))
                class_info['files'].append(filename)
                all_wavenumbers.append(wavenumbers)
                
                diagnostics['total_spectra'] += 1
            
            diagnostics['classes'][class_name] = class_info
        
        # Check for common issues
        all_lengths = []
        all_min_wn = []
        all_max_wn = []
        
        for class_name, class_info in diagnostics['classes'].items():
            all_lengths.extend(class_info['spectrum_lengths'])
            for wn_min, wn_max in class_info['wavenumber_ranges']:
                all_min_wn.append(wn_min)
                all_max_wn.append(wn_max)
        
        # Check for length consistency
        unique_lengths = set(all_lengths)
        if len(unique_lengths) > 1:
            diagnostics['issues'].append(f"Inconsistent spectrum lengths: {sorted(unique_lengths)}")
        
        # Check for wavenumber range overlap
        if all_min_wn and all_max_wn:  # Ensure we have data
            overall_min = max(all_min_wn)
            overall_max = min(all_max_wn)
            
            if overall_min >= overall_max:
                diagnostics['issues'].append("No overlapping wavenumber range between all spectra")
                diagnostics['valid'] = False
            else:
                diagnostics['common_range'] = (float(overall_min), float(overall_max))
        else:
            diagnostics['issues'].append("No wavenumber data found")
            diagnostics['valid'] = False
        
        # Check for very small classes
        for class_name, class_info in diagnostics['classes'].items():
            if class_info['count'] < 3:
                diagnostics['issues'].append(f"Class '{class_name}' has only {class_info['count']} spectra (recommend ≥5)")
        
        return diagnostics


class SpectrumLoader:
    """Utility class for loading spectra from directories."""
    
    @staticmethod
    def load_from_directory(directory: str, label: int, 
                           cosmic_ray_manager=None) -> List[Tuple[Dict, int]]:
        """
        Load all spectra from a directory.
        
        Args:
            directory: Path to directory containing spectrum files
            label: Class label to assign to loaded spectra
            cosmic_ray_manager: Optional cosmic ray removal manager
            
        Returns:
            List of (spectrum_data, label) tuples
        """
        dir_path = Path(directory)
        spectra = []
        
        # Look for spectrum files (support multiple formats)
        supported_extensions = ['*.csv', '*.txt', '*.dat', '*.asc', '*.spc']
        for ext in supported_extensions:
            for file_path in dir_path.glob(ext):
                try:
                    # Load spectrum data
                    if file_path.suffix.lower() == '.csv':
                        # Try to read CSV, handling headers appropriately
                        try:
                            # First try with header=0 (assuming header exists)
                            df = pd.read_csv(file_path, header=0)
                            if len(df.columns) >= 2:
                                data = df.values
                            else:
                                # Fallback to no header
                                data = pd.read_csv(file_path, header=None).values
                        except:
                            # If all else fails, read without header
                            data = pd.read_csv(file_path, header=None).values
                    else:
                        # Handle space-delimited and other text formats
                        try:
                            data = np.loadtxt(file_path)
                        except ValueError:
                            # Try with different delimiters if space doesn't work
                            try:
                                data = np.loadtxt(file_path, delimiter='\t')  # Tab-delimited
                            except ValueError:
                                try:
                                    data = np.loadtxt(file_path, delimiter=',')  # Comma-delimited
                                except ValueError:
                                    logger.warning(f"Could not parse file format: {file_path}")
                                    continue
                    
                    if data.shape[1] >= 2:
                        wavenumbers = data[:, 0]
                        intensities = data[:, 1]
                        
                        # Apply cosmic ray removal if available
                        if cosmic_ray_manager and cosmic_ray_manager.config.enabled:
                            spectrum_id = f"loaded_{file_path.name}"
                            _, intensities, _ = cosmic_ray_manager.detect_and_remove_cosmic_rays(
                                wavenumbers, intensities, spectrum_id
                            )
                        
                        spectrum_data = {
                            'wavenumbers': wavenumbers,
                            'intensities': intensities,
                            'filename': file_path.name
                        }
                        
                        spectra.append((spectrum_data, label))
                        
                except Exception as e:
                    logger.warning(f"Failed to load spectrum {file_path}: {str(e)}")
                    continue
        
        return spectra


class SupervisedMLAnalyzer:
    """Supervised ML classification functionality for Raman spectroscopy data."""
    
    def __init__(self, preprocessor=None):
        """
        Initialize the supervised ML analyzer.
        
        Args:
            preprocessor: Function to preprocess individual spectra
        """
        self.model = None
        self.scaler = None
        self.feature_type = 'full'
        self.model_type = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.training_results = None
        self.preprocessor = preprocessor
        self.label_encoder = None
        self.class_names = None
        
    def train(self, class_a_dir: str, class_b_dir: str, 
              model_type: str = 'Random Forest',
              feature_transformer=None, feature_type: str = 'full',
              test_size: float = 0.2, n_estimators: int = 100, 
              max_depth: int = 10, cosmic_ray_manager=None) -> Dict[str, Any]:
        """
        Train supervised ML classifier on spectral data.
        
        Args:
            class_a_dir: Directory containing class A spectra
            class_b_dir: Directory containing class B spectra
            model_type: Type of ML model ('Random Forest', 'Support Vector Machine', 'Gradient Boosting')
            feature_transformer: Optional feature transformer (PCA/NMF)
            feature_type: Type of features to use ('full', 'pca', 'nmf')
            test_size: Fraction of data to use for testing
            n_estimators: Number of trees in the forest (for tree-based models)
            max_depth: Maximum depth of trees (for tree-based models)
            cosmic_ray_manager: Optional cosmic ray removal manager
            
        Returns:
            Dictionary containing training results
        """
        try:
            # Load spectra from both directories
            class_a_spectra = SpectrumLoader.load_from_directory(
                class_a_dir, label=1, cosmic_ray_manager=cosmic_ray_manager
            )
            class_b_spectra = SpectrumLoader.load_from_directory(
                class_b_dir, label=0, cosmic_ray_manager=cosmic_ray_manager
            )
            
            if len(class_a_spectra) == 0 or len(class_b_spectra) == 0:
                raise ValueError("No spectra found in one or both directories")
            
            # Combine data
            all_spectra = class_a_spectra + class_b_spectra
            
            # Extract features and labels
            X = []
            y = []
            
            for spectrum_data, label in all_spectra:
                # Preprocess spectrum if preprocessor is available
                if self.preprocessor:
                    processed_spectrum = self.preprocessor(
                        spectrum_data['wavenumbers'], 
                        spectrum_data['intensities']
                    )
                else:
                    processed_spectrum = spectrum_data['intensities']
                
                X.append(processed_spectrum)
                y.append(label)
            
            X = np.array(X)
            y = np.array(y)
            
            # Remove any NaN or infinite values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply feature transformation if provided
            if feature_transformer is not None:
                X_transformed, actual_type = feature_transformer.transform_data(X, fallback_to_full=True)
                if X_transformed is not None:
                    X = X_transformed
                    self.feature_type = actual_type
                else:
                    self.feature_type = 'full'
            else:
                self.feature_type = feature_type
            
            # Split data
            X_train, self.X_test, y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Store model type
            self.model_type = model_type
            
            # Create and train model based on type
            if model_type == 'Random Forest':
                self.model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
            elif model_type == 'Support Vector Machine':
                # Scale features for SVM
                self.scaler = StandardScaler()
                X_train = self.scaler.fit_transform(X_train)
                self.X_test = self.scaler.transform(self.X_test)
                
                self.model = SVC(
                    kernel='rbf',
                    probability=True,  # Enable probability prediction
                    random_state=42
                )
            elif model_type == 'Gradient Boosting':
                self.model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            self.model.fit(X_train, y_train)
            
            # Make predictions (handle scaling for SVM)
            X_test_pred = self.X_test
            if self.model_type == 'Support Vector Machine' and self.scaler is not None:
                # X_test is already scaled above
                pass
            self.y_pred = self.model.predict(X_test_pred)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, self.y_pred)
            report = classification_report(self.y_test, self.y_pred, output_dict=True)
            conf_matrix = confusion_matrix(self.y_test, self.y_pred)
            
            # Cross-validation score (adjust cv folds for small datasets)
            n_samples = len(X_train)
            cv_folds = min(5, n_samples) if n_samples > 2 else 2
            try:
                cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds)
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
            except ValueError as e:
                logger.warning(f"Cross-validation failed with {cv_folds} folds: {str(e)}")
                # Fallback: use simple train/test split accuracy
                cv_mean = accuracy
                cv_std = 0.0
            
            self.training_results = {
                'success': True,
                'model_type': model_type,
                'accuracy': accuracy,
                'cv_accuracy': cv_mean,
                'cv_std': cv_std,
                'report': report,
                'confusion_matrix': conf_matrix,
                'n_class_a': len(class_a_spectra),
                'n_class_b': len(class_b_spectra),
                'feature_type': self.feature_type,
                'n_features': X.shape[1]
            }
            
            logger.info(f"{model_type} training completed - Accuracy: {accuracy:.3f}, CV: {cv_mean:.3f}±{cv_std:.3f}")
            return self.training_results
            
        except Exception as e:
            logger.error(f"Random Forest training failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def classify_data(self, data: np.ndarray, 
                     feature_transformer=None) -> Dict[str, Any]:
        """
        Classify new data using the trained model.
        
        Args:
            data: Input data to classify
            feature_transformer: Optional feature transformer to match training
            
        Returns:
            Dictionary containing classification results
        """
        if self.model is None:
            return {
                'success': False,
                'error': 'Model not trained'
            }
        
        try:
            # Clean data
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply feature transformation if provided
            if feature_transformer is not None:
                data_transformed, _ = feature_transformer.transform_data(data, fallback_to_full=True)
                if data_transformed is not None:
                    data = data_transformed
            
            # Apply scaling if needed (for SVM)
            if self.model_type == 'Support Vector Machine' and self.scaler is not None:
                data = self.scaler.transform(data)
            
            # Make predictions
            predictions = self.model.predict(data)
            probabilities = self.model.predict_proba(data)
            
            return {
                'success': True,
                'predictions': predictions,
                'probabilities': probabilities,
                'n_samples': len(data)
            }
            
        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance from the trained model."""
        if self.model is None:
            return None
        return self.model.feature_importances_
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model to file."""
        if self.model is None:
            logger.error("No model to save")
            return False
        
        try:
            import pickle
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'feature_type': self.feature_type,
                'training_results': self.training_results
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model from file."""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data.get('scaler')
            self.model_type = model_data.get('model_type', 'Random Forest')
            self.feature_type = model_data.get('feature_type', 'full')
            self.training_results = model_data.get('training_results')
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False


class UnsupervisedAnalyzer:
    """Unsupervised ML analysis functionality for Raman spectroscopy data."""
    
    def __init__(self):
        """Initialize the unsupervised analyzer."""
        self.model = None
        self.scaler = None
        self.labels = None
        self.cluster_centers = None
        self.method = None
        
    def train_clustering(self, data: np.ndarray, method: str = 'K-Means', 
                        n_clusters: int = 3, eps: float = 0.5, min_samples: int = 5,
                        feature_transformer=None) -> Dict[str, Any]:
        """
        Train unsupervised clustering model.
        
        Args:
            data: Input data matrix
            method: Clustering method ('K-Means', 'Gaussian Mixture Model', 'DBSCAN', 'Hierarchical Clustering')
            n_clusters: Number of clusters (for K-Means, GMM, Hierarchical)
            eps: DBSCAN neighborhood distance parameter
            min_samples: DBSCAN minimum samples per cluster
            feature_transformer: Optional feature transformer
            
        Returns:
            Dictionary containing clustering results
        """
        try:
            # Clean data
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply feature transformation if provided
            if feature_transformer is not None:
                data_transformed, _ = feature_transformer.transform_data(data, fallback_to_full=True)
                if data_transformed is not None:
                    data = data_transformed
            
            # Standardize data
            self.scaler = StandardScaler()
            data_scaled = self.scaler.fit_transform(data)
            
            # Apply clustering
            if method == 'K-Means':
                self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                self.labels = self.model.fit_predict(data_scaled)
                self.cluster_centers = self.model.cluster_centers_
                n_clusters_found = n_clusters
                
            elif method == 'Gaussian Mixture Model':
                self.model = GaussianMixture(n_components=n_clusters, random_state=42)
                self.model.fit(data_scaled)
                self.labels = self.model.predict(data_scaled)
                self.cluster_centers = self.model.means_
                n_clusters_found = n_clusters
                
            elif method == 'DBSCAN':
                self.model = DBSCAN(eps=eps, min_samples=min_samples)
                self.labels = self.model.fit_predict(data_scaled)
                self.cluster_centers = None  # DBSCAN doesn't have explicit centers
                n_clusters_found = len(set(self.labels)) - (1 if -1 in self.labels else 0)
                
            elif method == 'Hierarchical Clustering':
                self.model = AgglomerativeClustering(n_clusters=n_clusters)
                self.labels = self.model.fit_predict(data_scaled)
                self.cluster_centers = None  # No explicit centers
                n_clusters_found = n_clusters
                
            else:
                raise ValueError(f"Unsupported clustering method: {method}")
            
            self.method = method
            
            # Calculate clustering metrics
            from sklearn.metrics import silhouette_score, adjusted_rand_score
            try:
                if len(set(self.labels)) > 1:  # Need at least 2 clusters for silhouette
                    silhouette = silhouette_score(data_scaled, self.labels)
                else:
                    silhouette = -1
            except:
                silhouette = -1  # Fallback if calculation fails
            
            # Additional metrics for DBSCAN
            n_noise = np.sum(self.labels == -1) if method == 'DBSCAN' else 0
            
            return {
                'success': True,
                'labels': self.labels,
                'cluster_centers': self.cluster_centers,
                'silhouette_score': silhouette,
                'n_clusters': n_clusters_found,
                'n_noise': n_noise,
                'method': method,
                'n_samples': len(data)
            }
            
        except Exception as e:
            logger.error(f"Unsupervised training failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict_clusters(self, data: np.ndarray, 
                        feature_transformer=None) -> Dict[str, Any]:
        """
        Predict clusters for new data.
        
        Args:
            data: Input data to cluster
            feature_transformer: Optional feature transformer
            
        Returns:
            Dictionary containing clustering results
        """
        if self.model is None or self.scaler is None:
            return {
                'success': False,
                'error': 'Model not trained'
            }
        
        try:
            # Clean data
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply feature transformation if provided
            if feature_transformer is not None:
                data_transformed, _ = feature_transformer.transform_data(data, fallback_to_full=True)
                if data_transformed is not None:
                    data = data_transformed
            
            # Scale data
            data_scaled = self.scaler.transform(data)
            
            # Predict clusters based on method
            if self.method in ['K-Means', 'Gaussian Mixture Model']:
                predictions = self.model.predict(data_scaled)
            elif self.method == 'DBSCAN':
                # DBSCAN doesn't have a predict method, use fit_predict on new data
                # This is a limitation - for true prediction, we'd need a different approach
                predictions = self.model.fit_predict(data_scaled)
            elif self.method == 'Hierarchical Clustering':
                # Hierarchical clustering doesn't have a predict method
                # Use the closest cluster center approach for new data
                if self.cluster_centers is not None:
                    from sklearn.metrics.pairwise import euclidean_distances
                    distances = euclidean_distances(data_scaled, self.cluster_centers)
                    predictions = np.argmin(distances, axis=1)
                else:
                    # Fallback: re-fit on combined data
                    predictions = self.model.fit_predict(data_scaled)
            else:
                predictions = self.model.predict(data_scaled)
            
            return {
                'success': True,
                'labels': predictions,
                'n_samples': len(data)
            }
            
        except Exception as e:
            logger.error(f"Cluster prediction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get cluster centers."""
        return self.cluster_centers
    
    def get_labels(self) -> Optional[np.ndarray]:
        """Get cluster labels."""
        return self.labels


# Backward compatibility alias
RandomForestAnalyzer = SupervisedMLAnalyzer 