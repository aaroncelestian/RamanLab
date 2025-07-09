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

def get_workspace_root():
    """
    Safely detect the RamanLab workspace root directory.
    
    Returns:
        Path: The workspace root directory
    """
    # Get the directory of this file
    current_file = Path(__file__).resolve()
    
    # Look for RamanLab workspace indicators
    workspace_indicators = [
        'RamanLab_Database_20250602.pkl',
        'mineral_modes.pkl',
        '__exampleData',
        'test_data',
        'test_batch_data',
        'requirements_qt6.txt'
    ]
    
    # Start from current directory and walk up
    current_dir = current_file.parent
    max_levels = 5  # Prevent infinite loops
    
    for _ in range(max_levels):
        # Check if this looks like the workspace root
        indicators_found = sum(1 for indicator in workspace_indicators 
                             if (current_dir / indicator).exists())
        
        if indicators_found >= 3:  # Need at least 3 indicators
            logger.debug(f"Found workspace root: {current_dir}")
            return current_dir
        
        # Move up one level
        parent = current_dir.parent
        if parent == current_dir:  # Reached filesystem root
            break
        current_dir = parent
    
    # Fallback to current file's directory
    logger.warning(f"Could not find workspace root, using: {current_file.parent}")
    return current_file.parent

def get_example_data_paths():
    """
    Get safe paths to example data files.
    
    Returns:
        dict: Dictionary of example data paths
    """
    workspace_root = get_workspace_root()
    
    paths = {
        'workspace_root': workspace_root,
        'example_data': workspace_root / '__exampleData',
        'test_data': workspace_root / 'test_data',
        'test_batch_data': workspace_root / 'test_batch_data',
        'database_file': workspace_root / 'RamanLab_Database_20250602.pkl',
        'mineral_modes': workspace_root / 'mineral_modes.pkl'
    }
    
    # Find specific example files
    example_files = {}
    
    # Check __exampleData
    example_data_dir = paths['example_data']
    if example_data_dir.exists():
        for file_path in example_data_dir.glob('*.txt'):
            key = f"example_{file_path.stem.lower()}"
            example_files[key] = file_path
    
    # Check test_data
    test_data_dir = paths['test_data']
    if test_data_dir.exists():
        for file_path in test_data_dir.glob('*.txt'):
            key = f"test_{file_path.stem.lower()}"
            example_files[key] = file_path
        for file_path in test_data_dir.glob('*.pkl'):
            key = f"test_{file_path.stem.lower()}"
            example_files[key] = file_path
    
    # Check test_batch_data
    test_batch_dir = paths['test_batch_data']
    if test_batch_dir.exists():
        for file_path in test_batch_dir.glob('*.txt'):
            key = f"batch_{file_path.stem.lower()}"
            example_files[key] = file_path
    
    paths.update(example_files)
    
    return paths

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
def load_raman_database(db_path=None):
    """Load the RamanLab database PKL file."""
    if db_path is None:
        paths = get_example_data_paths()
        db_path = paths['database_file']
    return safe_pickle_load(db_path)

def load_mineral_modes(modes_path=None):
    """Load the mineral modes PKL file."""
    if modes_path is None:
        paths = get_example_data_paths()
        modes_path = paths['mineral_modes']
    return safe_pickle_load(modes_path)

def load_ml_models(models_path=None):
    """Load the ML models PKL file."""
    if models_path is None:
        workspace_root = get_workspace_root()
        models_path = workspace_root / "saved_models" / "ml_models.pkl"
    return safe_pickle_load(models_path)

def get_example_spectrum_file(mineral_name=None):
    """
    Get a safe path to an example spectrum file.
    
    Args:
        mineral_name (str, optional): Name of mineral to look for
        
    Returns:
        Path: Path to an example spectrum file
    """
    paths = get_example_data_paths()
    
    if mineral_name:
        # Look for specific mineral
        mineral_key = f"batch_{mineral_name.lower()}_sample"
        if mineral_key in paths:
            return paths[mineral_key]
        
        # Look in example data
        example_key = f"example_{mineral_name.lower()}"
        if example_key in paths:
            return paths[example_key]
    
    # Return first available example file
    for key, path in paths.items():
        if key.startswith(('batch_', 'example_', 'test_')) and str(path).endswith('.txt'):
            return path
    
    # Fallback to None
    return None

def print_available_example_files():
    """Print all available example files for debugging."""
    paths = get_example_data_paths()
    
    print("🔍 Available Example Data Files:")
    print("=" * 50)
    
    for key, path in paths.items():
        if isinstance(path, Path):
            if path.exists():
                print(f"✅ {key}: {path}")
            else:
                print(f"❌ {key}: {path} (not found)")
    
    print("\n📁 Directory Structure:")
    workspace_root = paths['workspace_root']
    for data_dir in ['__exampleData', 'test_data', 'test_batch_data']:
        dir_path = workspace_root / data_dir
        if dir_path.exists():
            print(f"📂 {data_dir}/")
            for file_path in sorted(dir_path.glob('*')):
                print(f"   📄 {file_path.name}")
        else:
            print(f"❌ {data_dir}/ (not found)") 