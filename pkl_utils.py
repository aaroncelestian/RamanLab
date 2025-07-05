#!/usr/bin/env python3
"""
PKL Utilities Module
Provides safe loading functions for PKL files with proper module resolution.
"""

import os
import sys
import pickle
import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

def ensure_module_path():
    """
    Ensures the current directory is in Python path for module imports.
    This is needed when loading PKL files that reference local modules.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        logger.debug(f"Added {current_dir} to Python path")

def safe_pickle_load(file_path, ensure_path=True):
    """
    Safely load a pickle file with proper module path resolution.
    
    Args:
        file_path (str or Path): Path to the pickle file
        ensure_path (bool): Whether to ensure module path is set up
        
    Returns:
        object: The loaded data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pickle.UnpicklingError: If the file cannot be unpickled
        ModuleNotFoundError: If required modules cannot be imported
    """
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"PKL file not found: {file_path}")
    
    if ensure_path:
        ensure_module_path()
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Successfully loaded PKL file: {file_path}")
        return data
        
    except ModuleNotFoundError as e:
        logger.error(f"Module not found when loading {file_path}: {e}")
        logger.error("Try running from the directory containing the required modules")
        raise
        
    except Exception as e:
        logger.error(f"Error loading PKL file {file_path}: {e}")
        raise

def safe_pickle_save(data, file_path):
    """
    Safely save data to a pickle file.
    
    Args:
        data: The data to save
        file_path (str or Path): Path where to save the pickle file
    """
    
    file_path = Path(file_path)
    
    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Successfully saved PKL file: {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving PKL file {file_path}: {e}")
        raise

def load_map_data(file_path):
    """
    Convenience function specifically for loading map data PKL files.
    This ensures the map_analysis_2d module is available.
    
    Args:
        file_path (str or Path): Path to the map data PKL file
        
    Returns:
        object: The loaded map data
    """
    
    # Ensure module path is set up
    ensure_module_path()
    
    # Try to import the required modules from the new modular structure
    try:
        from map_analysis_2d.core import RamanMapData, CosmicRayConfig, SimpleCosmicRayManager
        logger.debug("Successfully imported map_analysis_2d.core modules")
    except ImportError as e:
        logger.error(f"Cannot import map_analysis_2d.core modules: {e}")
        raise ModuleNotFoundError(
            "map_analysis_2d.core modules not found. "
            "Make sure you're running from the correct directory and the modular structure is available."
        )
    
    # Load the data
    return safe_pickle_load(file_path, ensure_path=False)

# Convenience functions for common use cases
def load_raman_database(db_path="RamanLab_Database_20250602.pkl"):
    """Load the RamanLab database PKL file."""
    return safe_pickle_load(db_path)

def load_mineral_modes(modes_path="mineral_modes.pkl"):
    """Load the mineral modes PKL file."""
    return safe_pickle_load(modes_path)

def load_ml_models(models_path="saved_models/ml_models.pkl"):
    """Load the ML models PKL file."""
    return safe_pickle_load(models_path) 