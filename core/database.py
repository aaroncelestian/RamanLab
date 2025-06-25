"""
Mineral Database Management Module for Raman Polarization Analyzer.

This module handles loading, searching, and managing the mineral database
used for spectrum generation and reference comparisons.
"""

import os
import pickle
import importlib.util
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


class MineralDatabase:
    """
    Comprehensive mineral database manager for Raman spectroscopy analysis.
    """
    
    def __init__(self):
        """Initialize the mineral database manager."""
        self.database = {}
        self.mineral_list = []
        self.loaded_from = None
        
    def load_database(self, database_paths: Optional[List[str]] = None) -> bool:
        """
        Load mineral database from various sources.
        
        Parameters:
        -----------
        database_paths : list, optional
            List of paths to search for database files
            
        Returns:
        --------
        bool
            True if database loaded successfully, False otherwise
        """
        if database_paths is None:
            database_paths = self._get_default_database_paths()
        
        for db_path in database_paths:
            if os.path.exists(db_path):
                try:
                    if db_path.endswith('.pkl'):
                        success = self._load_pickle_database(db_path)
                    elif db_path.endswith('.py'):
                        success = self._load_python_database(db_path)
                    else:
                        continue
                    
                    if success:
                        self.loaded_from = db_path
                        self._update_mineral_list()
                        print(f"✓ Loaded mineral database from {db_path}")
                        print(f"  Contains {len(self.mineral_list)} minerals")
                        return True
                        
                except Exception as e:
                    print(f"Error loading database from {db_path}: {e}")
                    continue
        
        # If no database found, create minimal one
        print("⚠ No database found, creating minimal database")
        self._create_minimal_database()
        return True
    
    def _get_default_database_paths(self) -> List[str]:
        """Get default paths to search for database files."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        
        return [
            'mineral_database.pkl',
            'mineral_database.py',
            os.path.join(script_dir, 'mineral_database.pkl'),
            os.path.join(script_dir, 'mineral_database.py'),
            os.path.join(parent_dir, 'mineral_database.pkl'),
            os.path.join(parent_dir, 'mineral_database.py'),
        ]
    
    def _load_pickle_database(self, file_path: str) -> bool:
        """Load database from pickle file."""
        try:
            with open(file_path, 'rb') as f:
                self.database = pickle.load(f)
            return bool(self.database)
        except Exception as e:
            print(f"Error loading pickle database: {e}")
            return False
    
    def _load_python_database(self, file_path: str) -> bool:
        """Load database from Python module."""
        try:
            spec = importlib.util.spec_from_file_location("mineral_database", file_path)
            mineral_db_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mineral_db_module)
            
            if hasattr(mineral_db_module, 'get_mineral_database'):
                self.database = mineral_db_module.get_mineral_database()
                return bool(self.database)
            elif hasattr(mineral_db_module, 'MINERAL_DATABASE'):
                self.database = mineral_db_module.MINERAL_DATABASE
                return bool(self.database)
            else:
                print("No recognized database function/variable found in module")
                return False
                
        except Exception as e:
            print(f"Error loading Python database module: {e}")
            return False
    
    def _create_minimal_database(self):
        """Create a minimal database with common minerals."""
        self.database = {
            'QUARTZ': {
                'name': 'Quartz',
                'formula': 'SiO2',
                'crystal_system': 'Hexagonal',
                'space_group': 'P3121',
                'space_group_number': 152,
                'point_group': '32',
                'raman_modes': [
                    {'frequency': 128, 'character': 'E', 'intensity': 'medium', 'symmetry': 'E'},
                    {'frequency': 206, 'character': 'A1', 'intensity': 'weak', 'symmetry': 'A1'},
                    {'frequency': 464, 'character': 'A1', 'intensity': 'very_strong', 'symmetry': 'A1'},
                    {'frequency': 696, 'character': 'E', 'intensity': 'weak', 'symmetry': 'E'},
                    {'frequency': 808, 'character': 'E', 'intensity': 'weak', 'symmetry': 'E'},
                    {'frequency': 1085, 'character': 'E', 'intensity': 'weak', 'symmetry': 'E'}
                ],
                'lattice_parameters': {
                    'a': 4.913, 'b': 4.913, 'c': 5.405,
                    'alpha': 90, 'beta': 90, 'gamma': 120
                }
            },
            'CALCITE': {
                'name': 'Calcite',
                'formula': 'CaCO3',
                'crystal_system': 'Hexagonal',
                'space_group': 'R3c',
                'space_group_number': 167,
                'point_group': '3m',
                'raman_modes': [
                    {'frequency': 155, 'character': 'Eg', 'intensity': 'medium', 'symmetry': 'Eg'},
                    {'frequency': 282, 'character': 'Eg', 'intensity': 'medium', 'symmetry': 'Eg'},
                    {'frequency': 714, 'character': 'Eg', 'intensity': 'medium', 'symmetry': 'Eg'},
                    {'frequency': 1086, 'character': 'A1g', 'intensity': 'very_strong', 'symmetry': 'A1g'}
                ],
                'lattice_parameters': {
                    'a': 4.989, 'b': 4.989, 'c': 17.061,
                    'alpha': 90, 'beta': 90, 'gamma': 120
                }
            },
            'GYPSUM': {
                'name': 'Gypsum',
                'formula': 'CaSO4·2H2O',
                'crystal_system': 'Monoclinic',
                'space_group': 'C2/c',
                'space_group_number': 15,
                'point_group': '2/m',
                'raman_modes': [
                    {'frequency': 415, 'character': 'Ag', 'intensity': 'medium', 'symmetry': 'Ag'},
                    {'frequency': 493, 'character': 'Ag', 'intensity': 'strong', 'symmetry': 'Ag'},
                    {'frequency': 618, 'character': 'Ag', 'intensity': 'medium', 'symmetry': 'Ag'},
                    {'frequency': 670, 'character': 'Ag', 'intensity': 'weak', 'symmetry': 'Ag'},
                    {'frequency': 1008, 'character': 'Ag', 'intensity': 'very_strong', 'symmetry': 'Ag'}
                ],
                'lattice_parameters': {
                    'a': 5.68, 'b': 15.18, 'c': 6.29,
                    'alpha': 90, 'beta': 118.43, 'gamma': 90
                }
            },
            'HILAIRITE': {
                'name': 'Hilairite',
                'formula': 'Na2ZrSi3O9·3H2O',
                'crystal_system': 'Triclinic',
                'space_group': 'P-1',
                'space_group_number': 2,
                'point_group': '-1',
                'raman_modes': [
                    {'frequency': 295, 'character': 'Ag', 'intensity': 'strong', 'symmetry': 'Ag'},
                    {'frequency': 415, 'character': 'Ag', 'intensity': 'medium', 'symmetry': 'Ag'},
                    {'frequency': 520, 'character': 'Ag', 'intensity': 'very_strong', 'symmetry': 'Ag'},
                    {'frequency': 580, 'character': 'Ag', 'intensity': 'medium', 'symmetry': 'Ag'},
                    {'frequency': 740, 'character': 'Ag', 'intensity': 'weak', 'symmetry': 'Ag'},
                    {'frequency': 924, 'character': 'Ag', 'intensity': 'strong', 'symmetry': 'Ag'},
                    {'frequency': 1050, 'character': 'Ag', 'intensity': 'medium', 'symmetry': 'Ag'}
                ],
                'lattice_parameters': {
                    'a': 10.37, 'b': 10.73, 'c': 8.99,
                    'alpha': 90.2, 'beta': 95.3, 'gamma': 90.1
                }
            },
            'CORUNDUM': {
                'name': 'Corundum',
                'formula': 'Al2O3',
                'crystal_system': 'Hexagonal',
                'space_group': 'R-3c',
                'space_group_number': 167,
                'point_group': '-3m',
                'raman_modes': [
                    {'frequency': 378, 'character': 'Eg', 'intensity': 'medium', 'symmetry': 'Eg'},
                    {'frequency': 418, 'character': 'A1g', 'intensity': 'strong', 'symmetry': 'A1g'},
                    {'frequency': 432, 'character': 'Eg', 'intensity': 'weak', 'symmetry': 'Eg'},
                    {'frequency': 451, 'character': 'A1g', 'intensity': 'weak', 'symmetry': 'A1g'},
                    {'frequency': 578, 'character': 'Eg', 'intensity': 'strong', 'symmetry': 'Eg'},
                    {'frequency': 645, 'character': 'Eg', 'intensity': 'medium', 'symmetry': 'Eg'},
                    {'frequency': 751, 'character': 'Eg', 'intensity': 'weak', 'symmetry': 'Eg'}
                ],
                'lattice_parameters': {
                    'a': 4.759, 'b': 4.759, 'c': 12.991,
                    'alpha': 90, 'beta': 90, 'gamma': 120
                }
            },
            'RUTILE': {
                'name': 'Rutile',
                'formula': 'TiO2',
                'crystal_system': 'Tetragonal',
                'space_group': 'P42/mnm',
                'space_group_number': 136,
                'point_group': '4/mmm',
                'raman_modes': [
                    {'frequency': 143, 'character': 'B1g', 'intensity': 'medium', 'symmetry': 'B1g'},
                    {'frequency': 235, 'character': 'Eg', 'intensity': 'weak', 'symmetry': 'Eg'},
                    {'frequency': 447, 'character': 'Eg', 'intensity': 'medium', 'symmetry': 'Eg'},
                    {'frequency': 612, 'character': 'A1g', 'intensity': 'very_strong', 'symmetry': 'A1g'}
                ],
                'lattice_parameters': {
                    'a': 4.593, 'b': 4.593, 'c': 2.959,
                    'alpha': 90, 'beta': 90, 'gamma': 90
                }
            },
            'FELDSPAR': {
                'name': 'Feldspar',
                'formula': 'KAlSi3O8',
                'crystal_system': 'Triclinic',
                'space_group': 'C-1',
                'space_group_number': 2,
                'point_group': '-1',
                'raman_modes': [
                    {'frequency': 288, 'character': 'Ag', 'intensity': 'medium', 'symmetry': 'Ag'},
                    {'frequency': 456, 'character': 'Ag', 'intensity': 'strong', 'symmetry': 'Ag'},
                    {'frequency': 478, 'character': 'Ag', 'intensity': 'weak', 'symmetry': 'Ag'},
                    {'frequency': 508, 'character': 'Ag', 'intensity': 'medium', 'symmetry': 'Ag'},
                    {'frequency': 580, 'character': 'Ag', 'intensity': 'weak', 'symmetry': 'Ag'},
                    {'frequency': 760, 'character': 'Ag', 'intensity': 'medium', 'symmetry': 'Ag'}
                ],
                'lattice_parameters': {
                    'a': 8.584, 'b': 12.96, 'c': 7.22,
                    'alpha': 90.3, 'beta': 115.9, 'gamma': 87.7
                }
            }
        }
        self.loaded_from = "built-in minimal database"
    
    def _update_mineral_list(self):
        """Update the sorted list of mineral names."""
        self.mineral_list = sorted(list(self.database.keys()))
    
    def search_minerals(self, query: str, max_results: int = 20) -> List[str]:
        """
        Search for minerals matching a query string.
        
        Parameters:
        -----------
        query : str
            Search query (case-insensitive)
        max_results : int
            Maximum number of results to return
            
        Returns:
        --------
        list
            List of matching mineral names
        """
        if not query:
            return self.mineral_list[:max_results]
        
        query_lower = query.lower()
        matches = []
        
        # Exact matches first
        for mineral in self.mineral_list:
            if mineral.lower() == query_lower:
                matches.append(mineral)
        
        # Starts with query
        for mineral in self.mineral_list:
            if mineral.lower().startswith(query_lower) and mineral not in matches:
                matches.append(mineral)
        
        # Contains query
        for mineral in self.mineral_list:
            if query_lower in mineral.lower() and mineral not in matches:
                matches.append(mineral)
        
        return matches[:max_results]
    
    def get_mineral_data(self, mineral_name: str) -> Optional[Dict[str, Any]]:
        """
        Get data for a specific mineral.
        
        Parameters:
        -----------
        mineral_name : str
            Name of the mineral
            
        Returns:
        --------
        dict or None
            Mineral data dictionary or None if not found
        """
        return self.database.get(mineral_name)
    
    def generate_spectrum(self, mineral_name: str, 
                         wavenumber_range: Tuple[float, float] = (100, 1200),
                         num_points: int = 2200,
                         peak_width: float = 10.0,
                         intensity_scale: float = 1.0,
                         add_noise: bool = True,
                         noise_level: float = 0.02) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generate a synthetic Raman spectrum for a mineral.
        
        Parameters:
        -----------
        mineral_name : str
            Name of the mineral
        wavenumber_range : tuple
            (min, max) wavenumber range in cm⁻¹
        num_points : int
            Number of data points in spectrum
        peak_width : float
            Default peak width in cm⁻¹
        intensity_scale : float
            Global intensity scaling factor
        add_noise : bool
            Whether to add random noise
        noise_level : float
            Noise amplitude as fraction of signal
            
        Returns:
        --------
        tuple
            (wavenumbers, intensities) or (None, None) if failed
        """
        mineral_data = self.get_mineral_data(mineral_name)
        if not mineral_data or 'raman_modes' not in mineral_data:
            return None, None
        
        try:
            # Create wavenumber array
            wavenumbers = np.linspace(wavenumber_range[0], wavenumber_range[1], num_points)
            intensities = np.zeros_like(wavenumbers)
            
            # Intensity mapping
            intensity_map = {
                'very_weak': 0.1,
                'weak': 0.3,
                'medium': 0.6,
                'strong': 0.8,
                'very_strong': 1.0
            }
            
            # Add peaks for each Raman mode
            for mode in mineral_data['raman_modes']:
                frequency = mode['frequency']
                intensity_label = mode.get('intensity', 'medium')
                intensity = intensity_map.get(intensity_label, 0.5) * intensity_scale
                
                # Use mode-specific width if available
                mode_width = mode.get('width', peak_width)
                
                # Add Lorentzian peak
                peak_intensities = intensity / (1 + ((wavenumbers - frequency) / mode_width) ** 2)
                intensities += peak_intensities
            
            # Add baseline
            baseline_slope = np.random.uniform(-0.01, 0.01) if add_noise else 0
            baseline = baseline_slope * (wavenumbers - wavenumber_range[0]) / (wavenumber_range[1] - wavenumber_range[0])
            intensities += baseline
            
            # Add noise if requested
            if add_noise:
                noise = np.random.normal(0, noise_level * np.max(intensities), len(intensities))
                intensities += noise
            
            # Ensure non-negative
            intensities = np.maximum(intensities, 0)
            
            return wavenumbers, intensities
            
        except Exception as e:
            print(f"Error generating spectrum for {mineral_name}: {e}")
            return None, None
    
    def get_crystal_system_info(self, mineral_name: str) -> Optional[Dict[str, str]]:
        """
        Get crystal system information for a mineral.
        
        Parameters:
        -----------
        mineral_name : str
            Name of the mineral
            
        Returns:
        --------
        dict or None
            Crystal system information
        """
        mineral_data = self.get_mineral_data(mineral_name)
        if not mineral_data:
            return None
        
        return {
            'crystal_system': mineral_data.get('crystal_system', 'Unknown'),
            'space_group': mineral_data.get('space_group', 'Unknown'),
            'point_group': mineral_data.get('point_group', 'Unknown'),
            'space_group_number': mineral_data.get('space_group_number', 'Unknown')
        }
    
    def get_raman_modes(self, mineral_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get Raman active modes for a mineral.
        
        Parameters:
        -----------
        mineral_name : str
            Name of the mineral
            
        Returns:
        --------
        list or None
            List of Raman mode dictionaries
        """
        mineral_data = self.get_mineral_data(mineral_name)
        if not mineral_data:
            return None
        
        return mineral_data.get('raman_modes', [])
    
    def save_database(self, file_path: str, format: str = 'pickle') -> bool:
        """
        Save the current database to file.
        
        Parameters:
        -----------
        file_path : str
            Output file path
        format : str
            File format ('pickle' or 'json')
            
        Returns:
        --------
        bool
            True if saved successfully
        """
        try:
            if format == 'pickle':
                with open(file_path, 'wb') as f:
                    pickle.dump(self.database, f)
            elif format == 'json':
                import json
                with open(file_path, 'w') as f:
                    json.dump(self.database, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            print(f"Database saved to {file_path}")
            return True
            
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
    
    def add_mineral(self, mineral_name: str, mineral_data: Dict[str, Any]) -> bool:
        """
        Add a new mineral to the database.
        
        Parameters:
        -----------
        mineral_name : str
            Name of the mineral
        mineral_data : dict
            Mineral data dictionary
            
        Returns:
        --------
        bool
            True if added successfully
        """
        try:
            self.database[mineral_name.upper()] = mineral_data
            self._update_mineral_list()
            print(f"Added mineral: {mineral_name}")
            return True
        except Exception as e:
            print(f"Error adding mineral {mineral_name}: {e}")
            return False
    
    def remove_mineral(self, mineral_name: str) -> bool:
        """
        Remove a mineral from the database.
        
        Parameters:
        -----------
        mineral_name : str
            Name of the mineral to remove
            
        Returns:
        --------
        bool
            True if removed successfully
        """
        try:
            if mineral_name in self.database:
                del self.database[mineral_name]
                self._update_mineral_list()
                print(f"Removed mineral: {mineral_name}")
                return True
            else:
                print(f"Mineral not found: {mineral_name}")
                return False
        except Exception as e:
            print(f"Error removing mineral {mineral_name}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
        --------
        dict
            Database statistics
        """
        if not self.database:
            return {'total_minerals': 0}
        
        crystal_systems = {}
        total_modes = 0
        
        for mineral_data in self.database.values():
            # Count crystal systems
            crystal_system = mineral_data.get('crystal_system', 'Unknown')
            crystal_systems[crystal_system] = crystal_systems.get(crystal_system, 0) + 1
            
            # Count Raman modes
            raman_modes = mineral_data.get('raman_modes', [])
            total_modes += len(raman_modes)
        
        return {
            'total_minerals': len(self.database),
            'total_raman_modes': total_modes,
            'crystal_systems': crystal_systems,
            'loaded_from': self.loaded_from
        }


# Utility functions for database operations
def infer_crystal_system_from_name(mineral_name: str) -> str:
    """
    Infer crystal system from mineral name using common mineral knowledge.
    
    Parameters:
    -----------
    mineral_name : str
        Name of the mineral
        
    Returns:
    --------
    str
        Inferred crystal system or "Unknown"
    """
    crystal_systems = {
        # Cubic minerals
        'HALITE': 'Cubic', 'FLUORITE': 'Cubic', 'PYRITE': 'Cubic', 
        'GALENA': 'Cubic', 'MAGNETITE': 'Cubic', 'SPINEL': 'Cubic',
        'GARNET': 'Cubic', 'DIAMOND': 'Cubic', 'PERICLASE': 'Cubic',
        
        # Tetragonal minerals
        'ZIRCON': 'Tetragonal', 'RUTILE': 'Tetragonal', 'CASSITERITE': 'Tetragonal',
        'ANATASE': 'Tetragonal', 'SCHEELITE': 'Tetragonal',
        
        # Hexagonal minerals
        'QUARTZ': 'Hexagonal', 'CALCITE': 'Hexagonal', 'HEMATITE': 'Hexagonal',
        'CORUNDUM': 'Hexagonal', 'APATITE': 'Hexagonal', 'GRAPHITE': 'Hexagonal',
        
        # Orthorhombic minerals
        'OLIVINE': 'Orthorhombic', 'ARAGONITE': 'Orthorhombic', 'BARITE': 'Orthorhombic',
        'CELESTITE': 'Orthorhombic', 'TOPAZ': 'Orthorhombic',
        
        # Monoclinic minerals
        'GYPSUM': 'Monoclinic', 'MICA': 'Monoclinic', 'PYROXENE': 'Monoclinic',
        'AMPHIBOLE': 'Monoclinic', 'FELDSPAR': 'Monoclinic',
        
        # Triclinic minerals
        'PLAGIOCLASE': 'Triclinic', 'KAOLINITE': 'Triclinic'
    }
    
    mineral_upper = mineral_name.upper()
    
    # Direct lookup
    if mineral_upper in crystal_systems:
        return crystal_systems[mineral_upper]
    
    # Partial matching
    for mineral_key, system in crystal_systems.items():
        if mineral_key in mineral_upper or mineral_upper in mineral_key:
            return system
    
    return "Unknown" 