"""
Model management for multiple trained ML models.

This module provides functionality to manage multiple named ML models,
allowing users to save, load, and switch between different trained models.
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages multiple trained ML models with metadata."""
    
    def __init__(self):
        """Initialize the model manager."""
        self.models: Dict[str, Dict[str, Any]] = {}
        self.active_model: Optional[str] = None
        
    def add_model(self, name: str, analyzer, model_type: str, 
                  training_info: Optional[Dict] = None) -> bool:
        """
        Add a trained model to the manager.
        
        Args:
            name: Model name/identifier
            analyzer: Trained ML analyzer (SupervisedMLAnalyzer or UnsupervisedAnalyzer)
            model_type: Type of model ('supervised' or 'unsupervised')
            training_info: Optional training information dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not name or not name.strip():
                logger.error("Model name cannot be empty")
                return False
                
            if name in self.models:
                logger.warning(f"Model '{name}' already exists, overwriting")
            
            # Extract model components based on type
            if model_type == 'supervised':
                if analyzer.model is None:
                    logger.error("No trained model found in analyzer")
                    return False
                    
                model_data = {
                    'model': analyzer.model,
                    'scaler': analyzer.scaler,
                    'model_algorithm': analyzer.model_type,
                    'feature_type': analyzer.feature_type,
                    'training_results': analyzer.training_results,
                    'label_encoder': analyzer.label_encoder,
                    'class_names': analyzer.class_names
                }
            else:  # unsupervised
                if analyzer.model is None and analyzer.labels is None:
                    logger.error("No trained clustering model found in analyzer")
                    return False
                    
                model_data = {
                    'model': analyzer.model,
                    'scaler': analyzer.scaler,
                    'labels': analyzer.labels,
                    'cluster_centers': analyzer.cluster_centers,
                    'method': analyzer.method
                }
            
            # Add metadata
            metadata = {
                'model_type': model_type,
                'created_at': datetime.now().isoformat(),
                'training_info': training_info or {},
                'algorithm': model_data.get('model_algorithm') or model_data.get('method', 'Unknown')
            }
            
            self.models[name] = {
                'model_data': model_data,
                'metadata': metadata
            }
            
            # Set as active model
            self.active_model = name
            
            logger.info(f"Added {model_type} model '{name}' to manager")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add model '{name}': {str(e)}")
            return False
    
    def get_model(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a model by name.
        
        Args:
            name: Model name
            
        Returns:
            Model dictionary or None if not found
        """
        return self.models.get(name)
    
    def get_model_names(self) -> List[str]:
        """Get list of all model names."""
        return list(self.models.keys())
    
    def get_models_by_type(self, model_type: str) -> List[str]:
        """
        Get model names filtered by type.
        
        Args:
            model_type: 'supervised' or 'unsupervised'
            
        Returns:
            List of model names of the specified type
        """
        return [name for name, model_info in self.models.items() 
                if model_info['metadata']['model_type'] == model_type]
    
    def remove_model(self, name: str) -> bool:
        """
        Remove a model from the manager.
        
        Args:
            name: Model name to remove
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.models:
            logger.warning(f"Model '{name}' not found")
            return False
            
        del self.models[name]
        
        # Update active model if necessary
        if self.active_model == name:
            remaining_models = self.get_model_names()
            self.active_model = remaining_models[0] if remaining_models else None
        
        logger.info(f"Removed model '{name}' from manager")
        return True
    
    def set_active_model(self, name: str) -> bool:
        """
        Set the active model.
        
        Args:
            name: Model name to set as active
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.models:
            logger.error(f"Model '{name}' not found")
            return False
            
        self.active_model = name
        logger.info(f"Set active model to '{name}'")
        return True
    
    def get_active_model(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Get the active model.
        
        Returns:
            Tuple of (model_name, model_dict) or None if no active model
        """
        if self.active_model and self.active_model in self.models:
            return self.active_model, self.models[self.active_model]
        return None
    
    def load_model_into_analyzer(self, name: str, analyzer) -> bool:
        """
        Load a model from the manager into an analyzer.
        
        Args:
            name: Model name
            analyzer: Target analyzer to load into
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.models:
            logger.error(f"Model '{name}' not found")
            return False
            
        try:
            model_info = self.models[name]
            model_data = model_info['model_data']
            model_type = model_info['metadata']['model_type']
            
            if model_type == 'supervised':
                analyzer.model = model_data['model']
                analyzer.scaler = model_data.get('scaler')
                analyzer.model_type = model_data.get('model_algorithm', 'Unknown')
                analyzer.feature_type = model_data.get('feature_type', 'full')
                analyzer.training_results = model_data.get('training_results')
                analyzer.label_encoder = model_data.get('label_encoder')
                analyzer.class_names = model_data.get('class_names')
            else:  # unsupervised
                analyzer.model = model_data.get('model')
                analyzer.scaler = model_data.get('scaler')
                analyzer.labels = model_data.get('labels')
                analyzer.cluster_centers = model_data.get('cluster_centers')
                analyzer.method = model_data.get('method', 'Unknown')
            
            logger.info(f"Loaded model '{name}' into analyzer")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model '{name}' into analyzer: {str(e)}")
            return False
    
    def save_models_to_file(self, filepath: str) -> bool:
        """
        Save all models to a file.
        
        Args:
            filepath: File path to save to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            save_data = {
                'models': self.models,
                'active_model': self.active_model,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"Saved {len(self.models)} models to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save models to {filepath}: {str(e)}")
            return False
    
    def load_models_from_file(self, filepath: str) -> bool:
        """
        Load models from a file.
        
        Args:
            filepath: File path to load from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            self.models = save_data.get('models', {})
            self.active_model = save_data.get('active_model')
            
            logger.info(f"Loaded {len(self.models)} models from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models from {filepath}: {str(e)}")
            return False
    
    def save_single_model(self, name: str, filepath: str) -> bool:
        """
        Save a single model to a file.
        
        Args:
            name: Model name to save
            filepath: File path to save to
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.models:
            logger.error(f"Model '{name}' not found")
            return False
            
        try:
            model_info = self.models[name]
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_info, f)
            
            logger.info(f"Saved model '{name}' to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model '{name}' to {filepath}: {str(e)}")
            return False
    
    def load_single_model(self, filepath: str, name: Optional[str] = None) -> Optional[str]:
        """
        Load a single model from a file.
        
        Args:
            filepath: File path to load from
            name: Optional name to assign to the model
            
        Returns:
            Model name if successful, None otherwise
        """
        try:
            with open(filepath, 'rb') as f:
                model_info = pickle.load(f)
            
            # Handle both old format (direct model data) and new format (with metadata)
            if 'model_data' in model_info and 'metadata' in model_info:
                # New format
                model_name = name or Path(filepath).stem
            else:
                # Old format - convert to new format
                model_name = name or Path(filepath).stem
                model_info = {
                    'model_data': model_info,
                    'metadata': {
                        'model_type': 'supervised' if 'model' in model_info else 'unsupervised',
                        'created_at': datetime.now().isoformat(),
                        'training_info': {},
                        'algorithm': model_info.get('model_type', 'Unknown')
                    }
                }
            
            # Avoid name conflicts
            original_name = model_name
            counter = 1
            while model_name in self.models:
                model_name = f"{original_name}_{counter}"
                counter += 1
            
            self.models[model_name] = model_info
            self.active_model = model_name
            
            logger.info(f"Loaded model '{model_name}' from {filepath}")
            return model_name
            
        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {str(e)}")
            return None
    
    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata information about a model.
        
        Args:
            name: Model name
            
        Returns:
            Model metadata dictionary or None if not found
        """
        if name not in self.models:
            return None
            
        model_info = self.models[name]
        metadata = model_info['metadata']
        
        return {
            'name': name,
            'type': metadata['model_type'],
            'algorithm': metadata['algorithm'],
            'created_at': metadata['created_at'],
            'training_info': metadata.get('training_info', {})
        }
    
    def clear_all_models(self):
        """Clear all models from the manager."""
        self.models.clear()
        self.active_model = None
        logger.info("Cleared all models from manager")
    
    def count(self) -> int:
        """Get the number of models in the manager."""
        return len(self.models) 