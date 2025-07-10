#!/usr/bin/env python3
"""
Universal Analysis State Manager for RamanLab
Handles persistent state management for all analysis modules with a plugin-like architecture
"""

import json
import pickle
import os
import time
import hashlib
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable, Union
import numpy as np
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import threading
import weakref
from PySide6.QtWidgets import QSplitter

# Version for backward compatibility
STATE_MANAGER_VERSION = "1.0.0"

def sanitize_path_for_windows(path_str: str) -> str:
    """
    Sanitize a path string for Windows compatibility.
    
    Args:
        path_str: The path string to sanitize
        
    Returns:
        Sanitized path string safe for Windows
    """
    if not isinstance(path_str, str):
        return path_str
    
    # Handle special SQLite memory databases
    if path_str == ":memory:" or path_str.startswith(":memory:"):
        return path_str  # Keep memory databases as-is
    
    # Handle other special database URIs (must be proper URIs with //)
    if (path_str.startswith("file://") or 
        path_str.startswith("sqlite://") or 
        path_str.startswith("http://") or 
        path_str.startswith("https://")):
        return path_str  # Keep URI schemes as-is
    
    # Replace problematic characters for Windows
    # But preserve drive letters (C:, D:, etc.)
    if len(path_str) >= 2 and path_str[1] == ':' and path_str[0].isalpha():
        # This is a Windows drive path, keep the drive colon
        drive_part = path_str[:2]
        rest_part = path_str[2:]
        # Sanitize only the rest of the path (including any additional colons)
        sanitized_rest = re.sub(r'[<>:"|?*]', '_', rest_part)
        return drive_part + sanitized_rest
    else:
        # No drive letter, sanitize all reserved characters including colons
        # Replace reserved characters for Windows
        sanitized = re.sub(r'[<>:"|?*]', '_', path_str)
        return sanitized

def is_memory_database(path_str: str) -> bool:
    """
    Check if a path string represents an in-memory database.
    
    Args:
        path_str: The path string to check
        
    Returns:
        True if this is a memory database path
    """
    if not isinstance(path_str, str):
        return False
    
    return (path_str == ":memory:" or 
            path_str.startswith(":memory:") or
            "memory:" in path_str.lower())

def validate_and_sanitize_database_path(db_path: Any) -> Optional[str]:
    """
    Validate and sanitize a database path for cross-platform compatibility.
    
    Args:
        db_path: Database path (can be string, Path, or None)
        
    Returns:
        Sanitized path string or None if invalid
    """
    if db_path is None:
        return None
    
    # Convert Path objects to strings
    if isinstance(db_path, Path):
        db_path_str = str(db_path)
    elif isinstance(db_path, str):
        db_path_str = db_path
    else:
        # Convert other types to string
        db_path_str = str(db_path)
    
    # Handle empty strings
    if not db_path_str.strip():
        return None
    
    # Check for memory databases (these should not be file operations)
    if is_memory_database(db_path_str):
        return db_path_str  # Return as-is for memory databases
    
    # Sanitize for Windows compatibility
    sanitized_path = sanitize_path_for_windows(db_path_str)
    
    return sanitized_path

def safe_path_join(*args) -> str:
    """
    Safely join path components with cross-platform compatibility.
    
    Args:
        *args: Path components to join
        
    Returns:
        Joined path string
    """
    # Filter out None values and convert all to strings
    clean_args = []
    for arg in args:
        if arg is not None:
            if isinstance(arg, Path):
                clean_args.append(str(arg))
            else:
                clean_args.append(str(arg))
    
    if not clean_args:
        return ""
    
    # Use pathlib for cross-platform path joining
    result_path = Path(*clean_args)
    return str(result_path)

@dataclass
class StateMetadata:
    """Metadata for analysis state"""
    module_name: str
    version: str
    created: str
    last_modified: str
    file_hash: str
    user_notes: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class StateSerializerInterface(ABC):
    """Interface for module-specific state serializers"""
    
    @abstractmethod
    def serialize_state(self, module_instance) -> Dict[str, Any]:
        """Convert module state to serializable dictionary"""
        pass
    
    @abstractmethod
    def deserialize_state(self, state_data: Dict[str, Any], module_instance) -> bool:
        """Restore module state from dictionary. Returns success status."""
        pass
    
    @abstractmethod
    def get_state_summary(self, state_data: Dict[str, Any]) -> str:
        """Generate human-readable summary of state"""
        pass
    
    @abstractmethod
    def validate_state(self, state_data: Dict[str, Any]) -> bool:
        """Validate state data integrity"""
        pass

class BatchPeakFittingSerializer(StateSerializerInterface):
    """Serializer for batch peak fitting module"""
    
    def serialize_state(self, module_instance) -> Dict[str, Any]:
        """Extract all relevant state from BatchPeakFittingQt6 instance"""
        try:
            state = {
                'module_type': 'batch_peak_fitting',
                'version': '1.0.0',
                
                # File management
                'spectra_files': getattr(module_instance, 'spectra_files', []),
                'current_spectrum_index': getattr(module_instance, 'current_spectrum_index', -1),
                
                # Reference data
                'reference_spectrum': {
                    'file': getattr(module_instance, 'reference_file', None),
                    'peaks': self._array_to_list(getattr(module_instance, 'reference_peaks', [])),
                    'fit_params': self._array_to_list(getattr(module_instance, 'reference_fit_params', [])),
                    'model': getattr(module_instance, 'reference_model', 'Gaussian'),
                    'background_method': getattr(module_instance, 'reference_bg_method', 'ALS')
                },
                
                # Batch results
                'batch_results': self._serialize_batch_results(getattr(module_instance, 'batch_results', [])),
                
                # Manual fits (your custom adjustments)
                'manual_fits': self._serialize_manual_fits(module_instance),
                
                # Current spectrum state
                'current_spectrum': {
                    'wavenumbers': self._array_to_list(getattr(module_instance, 'wavenumbers', [])),
                    'intensities': self._array_to_list(getattr(module_instance, 'intensities', [])),
                    'original_intensities': self._array_to_list(getattr(module_instance, 'original_intensities', [])),
                    'background': self._array_to_list(getattr(module_instance, 'background', [])),
                    'fit_params': self._array_to_list(getattr(module_instance, 'fit_params', [])),
                    'peaks': self._array_to_list(getattr(module_instance, 'peaks', [])),
                    'manual_peaks': self._array_to_list(getattr(module_instance, 'manual_peaks', []))
                },
                
                # UI settings
                'ui_settings': self._serialize_ui_settings(module_instance),
                
                # Analysis metadata
                'analysis_metadata': {
                    'total_spectra': len(getattr(module_instance, 'spectra_files', [])),
                    'successful_fits': len([r for r in getattr(module_instance, 'batch_results', []) if not r.get('fit_failed', True)]),
                    'manual_adjustments': len([r for r in getattr(module_instance, 'batch_results', []) if r.get('manual_fit', False)]),
                    'last_batch_run': getattr(module_instance, '_last_batch_timestamp', None)
                }
            }
            
            return state
            
        except Exception as e:
            print(f"Error serializing batch peak fitting state: {e}")
            return {}
    
    def deserialize_state(self, state_data: Dict[str, Any], module_instance) -> bool:
        """Restore BatchPeakFittingQt6 state"""
        try:
            # Restore file management
            module_instance.spectra_files = state_data.get('spectra_files', [])
            module_instance.current_spectrum_index = state_data.get('current_spectrum_index', -1)
            
            # Restore reference data
            ref_data = state_data.get('reference_spectrum', {})
            module_instance.reference_file = ref_data.get('file')
            module_instance.reference_peaks = np.array(ref_data.get('peaks', []))
            module_instance.reference_fit_params = ref_data.get('fit_params', [])
            module_instance.reference_model = ref_data.get('model', 'Gaussian')
            module_instance.reference_bg_method = ref_data.get('background_method', 'ALS')
            
            # Restore batch results
            module_instance.batch_results = self._deserialize_batch_results(state_data.get('batch_results', []))
            
            # Restore current spectrum
            current = state_data.get('current_spectrum', {})
            module_instance.wavenumbers = np.array(current.get('wavenumbers', []))
            module_instance.intensities = np.array(current.get('intensities', []))
            module_instance.original_intensities = np.array(current.get('original_intensities', []))
            module_instance.background = np.array(current.get('background', [])) if current.get('background') else None
            module_instance.fit_params = current.get('fit_params', [])
            module_instance.peaks = np.array(current.get('peaks', []))
            module_instance.manual_peaks = np.array(current.get('manual_peaks', []))
            
            # Restore UI settings
            self._deserialize_ui_settings(state_data.get('ui_settings', {}), module_instance)
            
            # Update UI to reflect restored state
            if hasattr(module_instance, 'update_file_status'):
                module_instance.update_file_status()
            if hasattr(module_instance, 'update_current_plot'):
                module_instance.update_current_plot()
            if hasattr(module_instance, 'update_all_plots'):
                module_instance.update_all_plots()
            
            return True
            
        except Exception as e:
            print(f"Error deserializing batch peak fitting state: {e}")
            return False
    
    def get_state_summary(self, state_data: Dict[str, Any]) -> str:
        """Generate summary of batch peak fitting state"""
        metadata = state_data.get('analysis_metadata', {})
        ref_data = state_data.get('reference_spectrum', {})
        
        summary = f"Batch Peak Fitting Analysis\n"
        summary += f"├─ Total Spectra: {metadata.get('total_spectra', 0)}\n"
        summary += f"├─ Successful Fits: {metadata.get('successful_fits', 0)}\n"
        summary += f"├─ Manual Adjustments: {metadata.get('manual_adjustments', 0)}\n"
        summary += f"├─ Reference Model: {ref_data.get('model', 'Unknown')}\n"
        summary += f"├─ Reference Peaks: {len(ref_data.get('peaks', []))}\n"
        summary += f"└─ Background Method: {ref_data.get('background_method', 'Unknown')}"
        
        return summary
    
    def validate_state(self, state_data: Dict[str, Any]) -> bool:
        """Validate batch peak fitting state data"""
        required_keys = ['module_type', 'version', 'spectra_files', 'reference_spectrum']
        
        for key in required_keys:
            if key not in state_data:
                return False
        
        # Validate module type
        if state_data['module_type'] != 'batch_peak_fitting':
            return False
        
        # Validate reference spectrum structure
        ref_data = state_data['reference_spectrum']
        if not isinstance(ref_data.get('peaks', []), list):
            return False
        
        return True
    
    def _array_to_list(self, arr):
        """Convert numpy array to list for JSON serialization"""
        if arr is None:
            return []
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        if isinstance(arr, list):
            return arr
        return []
    
    def _serialize_batch_results(self, batch_results):
        """Serialize batch results with numpy array handling"""
        serialized = []
        for result in batch_results:
            if isinstance(result, dict):
                serialized_result = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        serialized_result[key] = value.tolist()
                    else:
                        serialized_result[key] = value
                serialized.append(serialized_result)
        return serialized
    
    def _deserialize_batch_results(self, batch_results_data):
        """Deserialize batch results, converting lists back to numpy arrays where appropriate"""
        deserialized = []
        array_keys = ['wavenumbers', 'intensities', 'original_intensities', 'background', 'fitted_curve', 'residuals', 'reference_peaks', 'peaks']
        
        for result_data in batch_results_data:
            if isinstance(result_data, dict):
                result = {}
                for key, value in result_data.items():
                    if key in array_keys and isinstance(value, list):
                        result[key] = np.array(value) if value else None
                    else:
                        result[key] = value
                deserialized.append(result)
        return deserialized
    
    def _serialize_manual_fits(self, module_instance):
        """Extract manual fit data"""
        if hasattr(module_instance, 'saved_fits'):
            return module_instance.saved_fits
        return {}
    
    def _serialize_ui_settings(self, module_instance):
        """Serialize UI control settings"""
        settings = {}
        
        # Background settings
        if hasattr(module_instance, 'bg_method_combo'):
            settings['background_method'] = module_instance.bg_method_combo.currentText()
        if hasattr(module_instance, 'lambda_slider'):
            settings['als_lambda'] = module_instance.lambda_slider.value()
        if hasattr(module_instance, 'p_slider'):
            settings['als_p'] = module_instance.p_slider.value()
        
        # Peak fitting settings
        if hasattr(module_instance, 'model_combo'):
            settings['peak_model'] = module_instance.model_combo.currentText()
        
        return settings
    
    def _deserialize_ui_settings(self, settings, module_instance):
        """Restore UI control settings"""
        if hasattr(module_instance, 'bg_method_combo') and 'background_method' in settings:
            module_instance.bg_method_combo.setCurrentText(settings['background_method'])
        if hasattr(module_instance, 'lambda_slider') and 'als_lambda' in settings:
            module_instance.lambda_slider.setValue(settings['als_lambda'])
        if hasattr(module_instance, 'p_slider') and 'als_p' in settings:
            module_instance.p_slider.setValue(settings['als_p'])
        if hasattr(module_instance, 'model_combo') and 'peak_model' in settings:
            module_instance.model_combo.setCurrentText(settings['peak_model'])


class PolarizationAnalyzerSerializer(StateSerializerInterface):
    """Serializer for Raman Polarization Analyzer module state."""
    
    def serialize_state(self, module_instance) -> Dict[str, Any]:
        """Extract all critical state from polarization analyzer module."""
        try:
            state = {
                'module_type': 'polarization_analyzer',
                'version': '1.0.0',
                'timestamp': datetime.now().isoformat(),
                
                # Spectrum data
                'current_spectrum': self._array_to_list(getattr(module_instance, 'current_spectrum', None)),
                'original_spectrum': self._array_to_list(getattr(module_instance, 'original_spectrum', None)),
                'imported_spectrum': self._array_to_list(getattr(module_instance, 'imported_spectrum', None)),
                
                # Peak fitting data
                'selected_peaks': getattr(module_instance, 'selected_peaks', []),
                'fitted_peaks': getattr(module_instance, 'fitted_peaks', []),
                'peak_assignments': getattr(module_instance, 'peak_assignments', {}),
                'frequency_shifts': getattr(module_instance, 'frequency_shifts', {}),
                'matched_peaks': getattr(module_instance, 'matched_peaks', []),
                'peak_labels': getattr(module_instance, 'peak_labels', {}),
                'calculated_peaks': getattr(module_instance, 'calculated_peaks', []),
                
                # Polarization data
                'polarization_data': getattr(module_instance, 'polarization_data', {}),
                'depolarization_ratios': getattr(module_instance, 'depolarization_ratios', {}),
                'angular_data': getattr(module_instance, 'angular_data', {}),
                'current_polarization_config': getattr(module_instance, 'current_polarization_config', None),
                
                # Crystal structure data
                'selected_reference_mineral': getattr(module_instance, 'selected_reference_mineral', None),
                'current_crystal_structure': getattr(module_instance, 'current_crystal_structure', None),
                'current_crystal_bonds': getattr(module_instance, 'current_crystal_bonds', []),
                
                # Optimization results
                'orientation_results': getattr(module_instance, 'orientation_results', {}),
                'optimization_parameters': getattr(module_instance, 'optimization_parameters', {}),
                'stage_results': getattr(module_instance, 'stage_results', {'stage1': None, 'stage2': None, 'stage3': None}),
                'optimized_orientation': getattr(module_instance, 'optimized_orientation', None),
                
                # UI state
                'peak_selection_mode': getattr(module_instance, 'peak_selection_mode', False),
                'peak_matching_tolerance': getattr(module_instance, 'peak_matching_tolerance', 50),
            }
            
            return state
            
        except Exception as e:
            print(f"Warning: Could not serialize polarization analyzer state: {e}")
            return {'module_type': 'polarization_analyzer', 'error': str(e)}
    
    def deserialize_state(self, state_data: Dict[str, Any], module_instance) -> bool:
        """Restore state to polarization analyzer module."""
        try:
            if state_data.get('module_type') != 'polarization_analyzer':
                return False
            
            # Restore spectrum data
            if 'current_spectrum' in state_data:
                module_instance.current_spectrum = np.array(state_data['current_spectrum']) if state_data['current_spectrum'] else None
            if 'original_spectrum' in state_data:
                module_instance.original_spectrum = np.array(state_data['original_spectrum']) if state_data['original_spectrum'] else None
            if 'imported_spectrum' in state_data:
                module_instance.imported_spectrum = np.array(state_data['imported_spectrum']) if state_data['imported_spectrum'] else None
            
            # Restore peak data
            for attr in ['selected_peaks', 'fitted_peaks', 'peak_assignments', 'frequency_shifts', 
                        'matched_peaks', 'peak_labels', 'calculated_peaks']:
                if attr in state_data:
                    setattr(module_instance, attr, state_data[attr])
            
            # Restore polarization data
            for attr in ['polarization_data', 'depolarization_ratios', 'angular_data', 'current_polarization_config']:
                if attr in state_data:
                    setattr(module_instance, attr, state_data[attr])
            
            # Restore crystal structure data
            for attr in ['selected_reference_mineral', 'current_crystal_structure', 'current_crystal_bonds']:
                if attr in state_data:
                    setattr(module_instance, attr, state_data[attr])
            
            # Restore optimization results
            for attr in ['orientation_results', 'optimization_parameters', 'stage_results', 'optimized_orientation']:
                if attr in state_data:
                    setattr(module_instance, attr, state_data[attr])
            
            # Restore UI state
            if 'peak_selection_mode' in state_data:
                module_instance.peak_selection_mode = state_data['peak_selection_mode']
            if 'peak_matching_tolerance' in state_data:
                module_instance.peak_matching_tolerance = state_data['peak_matching_tolerance']
            
            # Refresh UI displays
            if hasattr(module_instance, 'update_spectrum_plot'):
                module_instance.update_spectrum_plot()
            if hasattr(module_instance, 'update_mineral_lists'):
                module_instance.update_mineral_lists()
                
            return True
            
        except Exception as e:
            print(f"Warning: Could not restore polarization analyzer state: {e}")
            return False
    
    def get_state_summary(self, state_data: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the state."""
        summary = f"Polarization Analysis State\n"
        summary += f"  Timestamp: {state_data.get('timestamp', 'Unknown')}\n"
        summary += f"  Has Current Spectrum: {'Yes' if state_data.get('current_spectrum') else 'No'}\n"
        summary += f"  Selected Peaks: {len(state_data.get('selected_peaks', []))}\n"
        summary += f"  Fitted Peaks: {len(state_data.get('fitted_peaks', []))}\n"
        summary += f"  Reference Mineral: {state_data.get('selected_reference_mineral', 'None')}\n"
        summary += f"  Has Optimization Results: {'Yes' if state_data.get('orientation_results') else 'No'}\n"
        return summary
    
    def validate_state(self, state_data: Dict[str, Any]) -> bool:
        """Validate that the state data is complete and consistent."""
        return (isinstance(state_data, dict) and 
                state_data.get('module_type') == 'polarization_analyzer' and
                'version' in state_data)
    
    def _array_to_list(self, arr):
        """Convert numpy array to list for JSON serialization."""
        if arr is None:
            return None
        if hasattr(arr, 'tolist'):
            return arr.tolist()
        return arr


class ClusterAnalysisSerializer(StateSerializerInterface):
    """Serializer for Raman Cluster Analysis module state."""
    
    def serialize_state(self, module_instance) -> Dict[str, Any]:
        """Extract all critical state from cluster analysis module."""
        try:
            state = {
                'module_type': 'cluster_analysis',
                'version': '1.0.0',
                'timestamp': datetime.now().isoformat(),
                
                # File and data configuration
                'selected_folder': getattr(module_instance, 'selected_folder', None),
                'visualization_method': getattr(module_instance, 'visualization_method', 'PCA'),
                'n_subclusters': getattr(module_instance, 'n_subclusters', 2),
                'split_method': getattr(module_instance, 'split_method', 'kmeans'),
                
                # Main cluster data (arrays converted to lists)
                'cluster_data': self._serialize_cluster_data(getattr(module_instance, 'cluster_data', {})),
                
                # Analysis results
                'analysis_results': self._serialize_analysis_results(getattr(module_instance, 'analysis_results', {})),
                
                # UI state
                'selected_points': list(getattr(module_instance, 'selected_points', set())),
                'refinement_mode': getattr(module_instance, 'refinement_mode', False),
                
                # Undo stack for refinement (keep last 5 only)
                'undo_stack': getattr(module_instance, 'undo_stack', [])[-5:],
                
                # Database configuration
                'custom_db_path': validate_and_sanitize_database_path(getattr(module_instance, 'custom_db_path', None)),
            }
            
            return state
            
        except Exception as e:
            print(f"Warning: Could not serialize cluster analysis state: {e}")
            return {'module_type': 'cluster_analysis', 'error': str(e)}
    
    def deserialize_state(self, state_data: Dict[str, Any], module_instance) -> bool:
        """Restore state to cluster analysis module."""
        try:
            if state_data.get('module_type') != 'cluster_analysis':
                return False
            
            # Restore basic configuration
            for attr in ['selected_folder', 'visualization_method', 'n_subclusters', 'split_method']:
                if attr in state_data:
                    setattr(module_instance, attr, state_data[attr])
            
            # Handle database path specially with validation
            if 'custom_db_path' in state_data:
                db_path = state_data['custom_db_path']
                if db_path is not None and not is_memory_database(str(db_path)):
                    # Only set if it's not a memory database or if the path exists
                    try:
                        path_obj = Path(db_path)
                        if path_obj.exists() or is_memory_database(str(db_path)):
                            setattr(module_instance, 'custom_db_path', db_path)
                        else:
                            print(f"Warning: Database path {db_path} does not exist, skipping restore")
                            setattr(module_instance, 'custom_db_path', None)
                    except (OSError, ValueError) as e:
                        print(f"Warning: Invalid database path {db_path}: {e}")
                        setattr(module_instance, 'custom_db_path', None)
                else:
                    setattr(module_instance, 'custom_db_path', db_path)
            
            # Restore cluster data
            if 'cluster_data' in state_data:
                module_instance.cluster_data = self._deserialize_cluster_data(state_data['cluster_data'])
            
            # Restore analysis results
            if 'analysis_results' in state_data:
                module_instance.analysis_results = self._deserialize_analysis_results(state_data['analysis_results'])
            
            # Restore UI state
            if 'selected_points' in state_data:
                module_instance.selected_points = set(state_data['selected_points'])
            if 'refinement_mode' in state_data:
                module_instance.refinement_mode = state_data['refinement_mode']
            if 'undo_stack' in state_data:
                module_instance.undo_stack = state_data['undo_stack']
            
            # Refresh UI
            if hasattr(module_instance, 'update_visualizations'):
                module_instance.update_visualizations()
            if hasattr(module_instance, 'update_clustering_controls'):
                module_instance.update_clustering_controls()
                
            return True
            
        except Exception as e:
            print(f"Warning: Could not restore cluster analysis state: {e}")
            return False
    
    def get_state_summary(self, state_data: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the state."""
        summary = f"Cluster Analysis State\n"
        summary += f"  Timestamp: {state_data.get('timestamp', 'Unknown')}\n"
        summary += f"  Selected Folder: {state_data.get('selected_folder', 'None')}\n"
        summary += f"  Visualization Method: {state_data.get('visualization_method', 'PCA')}\n"
        summary += f"  Has Cluster Data: {'Yes' if state_data.get('cluster_data', {}).get('features') else 'No'}\n"
        summary += f"  Selected Points: {len(state_data.get('selected_points', []))}\n"
        summary += f"  Refinement Mode: {'Yes' if state_data.get('refinement_mode') else 'No'}\n"
        return summary
    
    def validate_state(self, state_data: Dict[str, Any]) -> bool:
        """Validate that the state data is complete and consistent."""
        return (isinstance(state_data, dict) and 
                state_data.get('module_type') == 'cluster_analysis' and
                'version' in state_data)
    
    def _serialize_cluster_data(self, cluster_data):
        """Safely serialize cluster data with numpy arrays."""
        safe_data = {}
        for key, value in cluster_data.items():
            if key in ['wavenumbers', 'intensities', 'features', 'features_scaled', 'umap_embedding']:
                safe_data[key] = self._array_to_list(value)
            elif key in ['labels', 'silhouette_scores']:
                safe_data[key] = self._array_to_list(value)
            elif key == 'linkage_matrix':
                safe_data[key] = self._array_to_list(value)
            elif key == 'distance_matrix':
                safe_data[key] = self._array_to_list(value)
            else:
                safe_data[key] = value
        return safe_data
    
    def _deserialize_cluster_data(self, safe_data):
        """Safely deserialize cluster data with numpy arrays."""
        cluster_data = {}
        for key, value in safe_data.items():
            if key in ['wavenumbers', 'intensities', 'features', 'features_scaled', 'umap_embedding',
                      'labels', 'silhouette_scores', 'linkage_matrix', 'distance_matrix']:
                cluster_data[key] = np.array(value) if value is not None else None
            else:
                cluster_data[key] = value
        return cluster_data
    
    def _serialize_analysis_results(self, analysis_results):
        """Safely serialize analysis results."""
        safe_results = {}
        for key, value in analysis_results.items():
            if key == 'cluster_centroids':
                safe_results[key] = self._array_to_list(value)
            else:
                safe_results[key] = value
        return safe_results
    
    def _deserialize_analysis_results(self, safe_results):
        """Safely deserialize analysis results."""
        analysis_results = {}
        for key, value in safe_results.items():
            if key == 'cluster_centroids':
                analysis_results[key] = np.array(value) if value is not None else None
            else:
                analysis_results[key] = value
        return analysis_results
    
    def _array_to_list(self, arr):
        """Convert numpy array to list for JSON serialization."""
        if arr is None:
            return None
        if hasattr(arr, 'tolist'):
            return arr.tolist()
        return arr


class MapAnalysisSerializer(StateSerializerInterface):
    """Serializer for 2D Map Analysis module state."""
    
    def serialize_state(self, module_instance) -> Dict[str, Any]:
        """Extract all critical state from 2D map analysis module."""
        try:
            state = {
                'module_type': 'map_analysis_2d',
                'version': '1.0.0',
                'timestamp': datetime.now().isoformat(),
                
                # Map data - basic info only due to size
                'has_map_data': hasattr(module_instance, 'map_data') and module_instance.map_data is not None,
                'map_data_path': validate_and_sanitize_database_path(
                    getattr(module_instance.map_data, 'data_path', None) 
                    if hasattr(module_instance, 'map_data') and module_instance.map_data else None
                ),
                
                # Analysis state
                'current_feature': getattr(module_instance, 'current_feature', 'Integrated Intensity'),
                'use_processed': getattr(module_instance, 'use_processed', True),
                'intensity_vmin': getattr(module_instance, 'intensity_vmin', None),
                'intensity_vmax': getattr(module_instance, 'intensity_vmax', None),
                'integration_center': getattr(module_instance, 'integration_center', None),
                'integration_width': getattr(module_instance, 'integration_width', None),
                
                # Analysis completion status
                'pca_completed': hasattr(module_instance, 'pca_analyzer') and hasattr(module_instance.pca_analyzer, 'results_'),
                'nmf_completed': hasattr(module_instance, 'nmf_analyzer') and hasattr(module_instance.nmf_analyzer, 'results_'),
                
                # Template data
                'template_count': len(getattr(module_instance.template_manager, 'templates', {})) if hasattr(module_instance, 'template_manager') else 0,
                'template_names': list(getattr(module_instance.template_manager, 'templates', {}).keys()) if hasattr(module_instance, 'template_manager') else [],
                
                # ML model information
                'saved_models': list(getattr(module_instance.model_manager, 'models', {}).keys()) if hasattr(module_instance, 'model_manager') else [],
                
                # UI state
                'current_tab_index': getattr(module_instance, 'tab_widget', None).currentIndex() if hasattr(module_instance, 'tab_widget') else 0,
                'template_extraction_mode': getattr(module_instance, 'template_extraction_mode', False),
                'current_marker_position': getattr(module_instance, 'current_marker_position', None),
                
                # Cosmic ray detection settings
                'cosmic_ray_enabled': getattr(module_instance.cosmic_ray_config, 'enabled', False) if hasattr(module_instance, 'cosmic_ray_config') else False,
            }
            
            return state
            
        except Exception as e:
            print(f"Warning: Could not serialize 2D map analysis state: {e}")
            return {'module_type': 'map_analysis_2d', 'error': str(e)}
    
    def deserialize_state(self, state_data: Dict[str, Any], module_instance) -> bool:
        """Restore state to 2D map analysis module."""
        try:
            if state_data.get('module_type') != 'map_analysis_2d':
                return False
            
            # Restore analysis parameters
            for attr in ['current_feature', 'use_processed', 'intensity_vmin', 'intensity_vmax',
                        'integration_center', 'integration_width', 'template_extraction_mode']:
                if attr in state_data:
                    setattr(module_instance, attr, state_data[attr])
            
            # Restore UI state
            if 'current_tab_index' in state_data and hasattr(module_instance, 'tab_widget'):
                module_instance.tab_widget.setCurrentIndex(state_data['current_tab_index'])
            
            if 'current_marker_position' in state_data:
                module_instance.current_marker_position = state_data['current_marker_position']
            
            # Restore cosmic ray settings
            if 'cosmic_ray_enabled' in state_data and hasattr(module_instance, 'cosmic_ray_config'):
                module_instance.cosmic_ray_config.enabled = state_data['cosmic_ray_enabled']
            
            # Note: Large data like map_data, PCA results, templates, and ML models
            # are not automatically restored due to size - user needs to reload files
            # but settings and parameters are preserved
            
            # Refresh UI
            if hasattr(module_instance, 'update_map'):
                try:
                    module_instance.update_map()
                except:
                    pass  # Map data might not be loaded yet
            
            return True
            
        except Exception as e:
            print(f"Warning: Could not restore 2D map analysis state: {e}")
            return False
    
    def get_state_summary(self, state_data: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the state."""
        summary = f"2D Map Analysis State\n"
        summary += f"  Timestamp: {state_data.get('timestamp', 'Unknown')}\n"
        summary += f"  Has Map Data: {'Yes' if state_data.get('has_map_data') else 'No'}\n"
        summary += f"  Current Feature: {state_data.get('current_feature', 'Unknown')}\n"
        summary += f"  Template Count: {state_data.get('template_count', 0)}\n"
        summary += f"  Saved Models: {len(state_data.get('saved_models', []))}\n"
        summary += f"  PCA Completed: {'Yes' if state_data.get('pca_completed') else 'No'}\n"
        summary += f"  NMF Completed: {'Yes' if state_data.get('nmf_completed') else 'No'}\n"
        return summary
    
    def validate_state(self, state_data: Dict[str, Any]) -> bool:
        """Validate that the state data is complete and consistent."""
        return (isinstance(state_data, dict) and 
                state_data.get('module_type') == 'map_analysis_2d' and
                'version' in state_data)


class RamanAnalysisAppSerializer(StateSerializerInterface):
    """Serializer for the main RamanLab analysis application"""
    
    def serialize_state(self, module_instance) -> Dict[str, Any]:
        """Extract all relevant state from RamanAnalysisAppQt6 instance"""
        try:
            state = {
                'module_type': 'raman_analysis_app',
                'version': '1.0.0',
                
                # Current spectrum data
                'current_spectrum': {
                    'wavenumbers': self._array_to_list(getattr(module_instance, 'current_wavenumbers', [])),
                    'intensities': self._array_to_list(getattr(module_instance, 'current_intensities', [])),
                    'processed_intensities': self._array_to_list(getattr(module_instance, 'processed_intensities', [])),
                    'original_wavenumbers': self._array_to_list(getattr(module_instance, 'original_wavenumbers', [])),
                    'original_intensities': self._array_to_list(getattr(module_instance, 'original_intensities', [])),
                    'file_path': getattr(module_instance, 'spectrum_file_path', None),
                    'metadata': getattr(module_instance, 'metadata', {})
                },
                
                # Processing state
                'processing_state': {
                    'detected_peaks': self._array_to_list(getattr(module_instance, 'detected_peaks', [])),
                    'manual_peaks': getattr(module_instance, 'manual_peaks', []),
                    'peak_selection_mode': getattr(module_instance, 'peak_selection_mode', False),
                    'peak_selection_tolerance': getattr(module_instance, 'peak_selection_tolerance', 20),
                    'background_preview': self._array_to_list(getattr(module_instance, 'background_preview', [])),
                    'smoothing_preview': self._array_to_list(getattr(module_instance, 'smoothing_preview', [])),
                    'background_preview_active': getattr(module_instance, 'background_preview_active', False),
                    'smoothing_preview_active': getattr(module_instance, 'smoothing_preview_active', False),
                    'preview_background': self._array_to_list(getattr(module_instance, 'preview_background', [])),
                    'preview_corrected': self._array_to_list(getattr(module_instance, 'preview_corrected', [])),
                    'preview_smoothed': self._array_to_list(getattr(module_instance, 'preview_smoothed', []))
                },
                
                # UI state
                'ui_state': {
                    'window_geometry': self._get_window_geometry(module_instance),
                    'current_tab': getattr(module_instance.tab_widget, 'currentIndex', lambda: 0)() if hasattr(module_instance, 'tab_widget') else 0,
                    'splitter_sizes': self._get_splitter_sizes(module_instance)
                },
                
                # Search state (if any recent searches)
                'search_state': {
                    'last_search_results': self._serialize_search_results(module_instance),
                    'search_parameters': self._serialize_search_parameters(module_instance)
                },
                
                # Database state
                'database_state': {
                    'database_loaded': hasattr(module_instance, 'raman_db') and module_instance.raman_db is not None,
                    'database_size': len(getattr(module_instance, 'database', {}))
                },
                
                # Analysis metadata
                'analysis_metadata': {
                    'session_created': datetime.now(timezone.utc).isoformat(),
                    'has_spectrum': module_instance.current_wavenumbers is not None,
                    'spectrum_points': len(module_instance.current_wavenumbers) if module_instance.current_wavenumbers is not None else 0,
                    'has_peaks': module_instance.detected_peaks is not None or len(getattr(module_instance, 'manual_peaks', [])) > 0,
                    'processing_applied': module_instance.processed_intensities is not None and not np.array_equal(
                        module_instance.processed_intensities, 
                        module_instance.current_intensities
                    ) if module_instance.processed_intensities is not None and module_instance.current_intensities is not None else False
                }
            }
            
            return state
            
        except Exception as e:
            print(f"Error serializing main app state: {e}")
            return {}
    
    def deserialize_state(self, state_data: Dict[str, Any], module_instance) -> bool:
        """Restore RamanAnalysisAppQt6 state"""
        try:
            # Restore current spectrum data
            spectrum_data = state_data.get('current_spectrum', {})
            if spectrum_data.get('wavenumbers') and spectrum_data.get('intensities'):
                module_instance.current_wavenumbers = np.array(spectrum_data['wavenumbers'])
                module_instance.current_intensities = np.array(spectrum_data['intensities'])
                module_instance.processed_intensities = np.array(spectrum_data.get('processed_intensities', spectrum_data['intensities']))
                module_instance.original_wavenumbers = np.array(spectrum_data.get('original_wavenumbers', spectrum_data['wavenumbers']))
                module_instance.original_intensities = np.array(spectrum_data.get('original_intensities', spectrum_data['intensities']))
                module_instance.spectrum_file_path = spectrum_data.get('file_path')
                module_instance.metadata = spectrum_data.get('metadata', {})
            
            # Restore processing state
            processing_state = state_data.get('processing_state', {})
            if processing_state.get('detected_peaks'):
                module_instance.detected_peaks = np.array(processing_state['detected_peaks'])
            module_instance.manual_peaks = processing_state.get('manual_peaks', [])
            module_instance.peak_selection_mode = processing_state.get('peak_selection_mode', False)
            module_instance.peak_selection_tolerance = processing_state.get('peak_selection_tolerance', 20)
            module_instance.background_preview_active = processing_state.get('background_preview_active', False)
            module_instance.smoothing_preview_active = processing_state.get('smoothing_preview_active', False)
            
            # Restore preview data if available
            if processing_state.get('background_preview'):
                module_instance.background_preview = np.array(processing_state['background_preview'])
            if processing_state.get('smoothing_preview'):
                module_instance.smoothing_preview = np.array(processing_state['smoothing_preview'])
            if processing_state.get('preview_background'):
                module_instance.preview_background = np.array(processing_state['preview_background'])
            if processing_state.get('preview_corrected'):
                module_instance.preview_corrected = np.array(processing_state['preview_corrected'])
            if processing_state.get('preview_smoothed'):
                module_instance.preview_smoothed = np.array(processing_state['preview_smoothed'])
            
            # Restore UI state
            ui_state = state_data.get('ui_state', {})
            if ui_state.get('window_geometry'):
                self._restore_window_geometry(module_instance, ui_state['window_geometry'])
            if hasattr(module_instance, 'tab_widget') and 'current_tab' in ui_state:
                module_instance.tab_widget.setCurrentIndex(ui_state['current_tab'])
            if ui_state.get('splitter_sizes'):
                self._restore_splitter_sizes(module_instance, ui_state['splitter_sizes'])
            
            # Update UI to reflect restored state
            if hasattr(module_instance, 'update_plot'):
                module_instance.update_plot()
            if hasattr(module_instance, 'update_info_display') and module_instance.spectrum_file_path:
                module_instance.update_info_display(module_instance.spectrum_file_path)
            if hasattr(module_instance, 'update_peak_count_display'):
                module_instance.update_peak_count_display()
            if hasattr(module_instance, 'update_window_title'):
                filename = Path(module_instance.spectrum_file_path).name if module_instance.spectrum_file_path else None
                module_instance.update_window_title(filename)
            
            return True
            
        except Exception as e:
            print(f"Error deserializing main app state: {e}")
            return False
    
    def get_state_summary(self, state_data: Dict[str, Any]) -> str:
        """Generate summary of main app state"""
        metadata = state_data.get('analysis_metadata', {})
        spectrum_data = state_data.get('current_spectrum', {})
        
        summary = f"RamanLab Main Application Session\n"
        summary += f"├─ Has Spectrum: {'Yes' if metadata.get('has_spectrum', False) else 'No'}\n"
        if metadata.get('has_spectrum', False):
            summary += f"├─ Spectrum Points: {metadata.get('spectrum_points', 0)}\n"
            summary += f"├─ Source File: {Path(spectrum_data.get('file_path', 'Unknown')).name if spectrum_data.get('file_path') else 'Unknown'}\n"
        summary += f"├─ Has Peaks: {'Yes' if metadata.get('has_peaks', False) else 'No'}\n"
        summary += f"├─ Processing Applied: {'Yes' if metadata.get('processing_applied', False) else 'No'}\n"
        summary += f"├─ Database Loaded: {'Yes' if state_data.get('database_state', {}).get('database_loaded', False) else 'No'}\n"
        summary += f"└─ Session Created: {metadata.get('session_created', 'Unknown')}"
        
        return summary
    
    def validate_state(self, state_data: Dict[str, Any]) -> bool:
        """Validate main app state data"""
        required_keys = ['module_type', 'version', 'current_spectrum', 'processing_state']
        
        for key in required_keys:
            if key not in state_data:
                return False
        
        # Validate module type
        if state_data['module_type'] != 'raman_analysis_app':
            return False
        
        # Validate spectrum data structure
        spectrum_data = state_data['current_spectrum']
        if not isinstance(spectrum_data, dict):
            return False
        
        return True
    
    def _array_to_list(self, arr):
        """Convert numpy array to list for JSON serialization"""
        if arr is None:
            return []
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        elif isinstance(arr, list):
            return arr
        else:
            return []
    
    def _get_window_geometry(self, module_instance):
        """Get window geometry for saving"""
        try:
            if hasattr(module_instance, 'geometry'):
                geom = module_instance.geometry()
                return {
                    'x': geom.x(),
                    'y': geom.y(),
                    'width': geom.width(),
                    'height': geom.height()
                }
        except:
            pass
        return {}
    
    def _restore_window_geometry(self, module_instance, geometry_data):
        """Restore window geometry"""
        try:
            if geometry_data and hasattr(module_instance, 'setGeometry'):
                module_instance.setGeometry(
                    geometry_data.get('x', 100),
                    geometry_data.get('y', 100),
                    geometry_data.get('width', 1600),
                    geometry_data.get('height', 1000)
                )
        except Exception as e:
            print(f"Could not restore window geometry: {e}")
    
    def _get_splitter_sizes(self, module_instance):
        """Get splitter sizes for saving"""
        try:
            # Look for the main splitter in the central widget
            central_widget = module_instance.centralWidget()
            if central_widget:
                for child in central_widget.findChildren(QSplitter):
                    return child.sizes()
        except:
            pass
        return []
    
    def _restore_splitter_sizes(self, module_instance, sizes):
        """Restore splitter sizes"""
        try:
            if sizes:
                central_widget = module_instance.centralWidget()
                if central_widget:
                    for child in central_widget.findChildren(QSplitter):
                        child.setSizes(sizes)
                        break
        except Exception as e:
            print(f"Could not restore splitter sizes: {e}")
    
    def _serialize_search_results(self, module_instance):
        """Serialize recent search results if available"""
        try:
            if hasattr(module_instance, 'search_results_text'):
                return module_instance.search_results_text.toPlainText()
        except:
            pass
        return ""
    
    def _serialize_search_parameters(self, module_instance):
        """Serialize current search parameters"""
        # This could be expanded to save search algorithm, thresholds, etc.
        return {}

class MultiSpectrumManagerSerializer(StateSerializerInterface):
    """Serializer for Multi-Spectrum Manager module"""
    
    def serialize_state(self, module_instance) -> Dict[str, Any]:
        """Extract all relevant state from MultiSpectrumManagerQt6 instance"""
        try:
            state = {
                'module_type': 'multi_spectrum_manager',
                'version': '1.0.0',
                
                # Core spectrum data
                'loaded_spectra': self._serialize_loaded_spectra(getattr(module_instance, 'loaded_spectra', {})),
                'spectrum_settings': getattr(module_instance, 'spectrum_settings', {}),
                'global_settings': getattr(module_instance, 'global_settings', {}),
                
                # UI state
                'ui_settings': self._serialize_ui_settings(module_instance),
                'window_geometry': self._get_window_geometry(module_instance),
                'splitter_sizes': self._get_splitter_sizes(module_instance),
                
                # Current selection
                'current_spectrum_selection': self._get_current_selection(module_instance),
                
                # Database search state
                'database_search_state': self._serialize_database_search_state(module_instance),
                
                # Analysis metadata
                'analysis_metadata': {
                    'total_spectra': len(getattr(module_instance, 'loaded_spectra', {})),
                    'last_modified': datetime.now(timezone.utc).isoformat(),
                    'session_duration': getattr(module_instance, '_session_start_time', None)
                }
            }
            
            return state
            
        except Exception as e:
            print(f"Error serializing multi-spectrum manager state: {e}")
            return {}
    
    def deserialize_state(self, state_data: Dict[str, Any], module_instance) -> bool:
        """Restore MultiSpectrumManagerQt6 state"""
        try:
            # Restore core spectrum data
            loaded_spectra = self._deserialize_loaded_spectra(state_data.get('loaded_spectra', {}))
            module_instance.loaded_spectra = loaded_spectra
            module_instance.spectrum_settings = state_data.get('spectrum_settings', {})
            module_instance.global_settings = state_data.get('global_settings', {})
            
            # Restore UI settings
            ui_settings = state_data.get('ui_settings', {})
            self._deserialize_ui_settings(ui_settings, module_instance)
            
            # Restore window geometry
            geometry_data = state_data.get('window_geometry', {})
            self._restore_window_geometry(module_instance, geometry_data)
            
            # Restore splitter sizes
            splitter_sizes = state_data.get('splitter_sizes', [])
            self._restore_splitter_sizes(module_instance, splitter_sizes)
            
            # Restore current selection
            current_selection = state_data.get('current_spectrum_selection', {})
            self._restore_current_selection(module_instance, current_selection)
            
            # Restore database search state
            db_search_state = state_data.get('database_search_state', {})
            self._restore_database_search_state(module_instance, db_search_state)
            
            # Update UI to reflect restored state
            self._update_ui_after_restore(module_instance)
            
            return True
            
        except Exception as e:
            print(f"Error deserializing multi-spectrum manager state: {e}")
            return False
    
    def get_state_summary(self, state_data: Dict[str, Any]) -> str:
        """Generate summary of multi-spectrum manager state"""
        metadata = state_data.get('analysis_metadata', {})
        global_settings = state_data.get('global_settings', {})
        
        summary = f"Multi-Spectrum Manager Session\n"
        summary += f"├─ Total Spectra: {metadata.get('total_spectra', 0)}\n"
        summary += f"├─ Normalize: {global_settings.get('normalize', 'Unknown')}\n"
        summary += f"├─ Waterfall Mode: {global_settings.get('waterfall_mode', 'Unknown')}\n"
        summary += f"├─ Show Legend: {global_settings.get('show_legend', 'Unknown')}\n"
        summary += f"├─ Colormap: {global_settings.get('colormap', 'Unknown')}\n"
        summary += f"└─ Last Modified: {metadata.get('last_modified', 'Unknown')[:19]}"
        
        return summary
    
    def validate_state(self, state_data: Dict[str, Any]) -> bool:
        """Validate multi-spectrum manager state data"""
        required_keys = ['module_type', 'version', 'loaded_spectra', 'global_settings']
        
        for key in required_keys:
            if key not in state_data:
                return False
        
        # Validate module type
        if state_data['module_type'] != 'multi_spectrum_manager':
            return False
        
        # Validate loaded spectra structure
        loaded_spectra = state_data['loaded_spectra']
        if not isinstance(loaded_spectra, dict):
            return False
        
        # Validate global settings
        global_settings = state_data['global_settings']
        if not isinstance(global_settings, dict):
            return False
        
        return True
    
    def _serialize_loaded_spectra(self, loaded_spectra):
        """Serialize loaded spectra with numpy arrays converted to lists"""
        if not loaded_spectra:
            return {}
        
        safe_spectra = {}
        for name, spectrum_data in loaded_spectra.items():
            if isinstance(spectrum_data, dict):
                safe_data = {}
                for key, value in spectrum_data.items():
                    if isinstance(value, np.ndarray):
                        safe_data[key] = self._array_to_list(value)
                    else:
                        safe_data[key] = value
                safe_spectra[name] = safe_data
            else:
                safe_spectra[name] = spectrum_data
        
        return safe_spectra
    
    def _deserialize_loaded_spectra(self, loaded_spectra_data):
        """Deserialize loaded spectra with lists converted back to numpy arrays"""
        if not loaded_spectra_data:
            return {}
        
        spectra = {}
        for name, spectrum_data in loaded_spectra_data.items():
            if isinstance(spectrum_data, dict):
                restored_data = {}
                for key, value in spectrum_data.items():
                    if isinstance(value, list) and key in ['wavenumbers', 'intensities']:
                        restored_data[key] = np.array(value)
                    else:
                        restored_data[key] = value
                spectra[name] = restored_data
            else:
                spectra[name] = spectrum_data
        
        return spectra
    
    def _serialize_ui_settings(self, module_instance):
        """Serialize UI control settings"""
        settings = {}
        
        # Control tab state
        if hasattr(module_instance, 'control_tabs'):
            settings['control_tabs_current'] = module_instance.control_tabs.currentIndex()
        
        # Global control values
        for attr_name in ['normalize_check', 'legend_check', 'grid_check', 'waterfall_check']:
            if hasattr(module_instance, attr_name):
                widget = getattr(module_instance, attr_name)
                if hasattr(widget, 'isChecked'):
                    settings[attr_name] = widget.isChecked()
        
        # Slider values
        for attr_name in ['line_width_slider', 'waterfall_spacing_slider']:
            if hasattr(module_instance, attr_name):
                widget = getattr(module_instance, attr_name)
                if hasattr(widget, 'value'):
                    settings[attr_name] = widget.value()
        
        # Combo box values
        for attr_name in ['colormap_combo']:
            if hasattr(module_instance, attr_name):
                widget = getattr(module_instance, attr_name)
                if hasattr(widget, 'currentText'):
                    settings[attr_name] = widget.currentText()
        
        return settings
    
    def _deserialize_ui_settings(self, settings, module_instance):
        """Restore UI control settings"""
        # Control tab state
        if 'control_tabs_current' in settings and hasattr(module_instance, 'control_tabs'):
            module_instance.control_tabs.setCurrentIndex(settings['control_tabs_current'])
        
        # Global control values
        for attr_name in ['normalize_check', 'legend_check', 'grid_check', 'waterfall_check']:
            if attr_name in settings and hasattr(module_instance, attr_name):
                widget = getattr(module_instance, attr_name)
                if hasattr(widget, 'setChecked'):
                    widget.setChecked(settings[attr_name])
        
        # Slider values
        for attr_name in ['line_width_slider', 'waterfall_spacing_slider']:
            if attr_name in settings and hasattr(module_instance, attr_name):
                widget = getattr(module_instance, attr_name)
                if hasattr(widget, 'setValue'):
                    widget.setValue(settings[attr_name])
        
        # Combo box values
        for attr_name in ['colormap_combo']:
            if attr_name in settings and hasattr(module_instance, attr_name):
                widget = getattr(module_instance, attr_name)
                if hasattr(widget, 'setCurrentText'):
                    widget.setCurrentText(settings[attr_name])
    
    def _get_window_geometry(self, module_instance):
        """Get window geometry data"""
        geometry = {}
        if hasattr(module_instance, 'geometry'):
            rect = module_instance.geometry()
            geometry = {
                'x': rect.x(),
                'y': rect.y(),
                'width': rect.width(),
                'height': rect.height()
            }
        return geometry
    
    def _restore_window_geometry(self, module_instance, geometry_data):
        """Restore window geometry"""
        if geometry_data and hasattr(module_instance, 'setGeometry'):
            module_instance.setGeometry(
                geometry_data.get('x', 100),
                geometry_data.get('y', 100),
                geometry_data.get('width', 1600),
                geometry_data.get('height', 1000)
            )
    
    def _get_splitter_sizes(self, module_instance):
        """Get splitter sizes from the main splitter"""
        sizes = []
        if hasattr(module_instance, 'centralWidget'):
            central_widget = module_instance.centralWidget()
            if central_widget:
                for child in central_widget.findChildren(QSplitter):
                    sizes.append(child.sizes())
                    break  # Just get the first splitter
        return sizes
    
    def _restore_splitter_sizes(self, module_instance, sizes):
        """Restore splitter sizes"""
        if sizes and hasattr(module_instance, 'centralWidget'):
            central_widget = module_instance.centralWidget()
            if central_widget:
                for child in central_widget.findChildren(QSplitter):
                    if sizes:
                        child.setSizes(sizes[0])
                    break  # Just set the first splitter
    
    def _get_current_selection(self, module_instance):
        """Get current spectrum selection"""
        selection = {}
        if hasattr(module_instance, 'multi_spectrum_list'):
            current_item = module_instance.multi_spectrum_list.currentItem()
            if current_item:
                selection['current_spectrum'] = current_item.text()
        return selection
    
    def _restore_current_selection(self, module_instance, selection):
        """Restore current spectrum selection"""
        if selection.get('current_spectrum') and hasattr(module_instance, 'multi_spectrum_list'):
            list_widget = module_instance.multi_spectrum_list
            for i in range(list_widget.count()):
                if list_widget.item(i).text() == selection['current_spectrum']:
                    list_widget.setCurrentRow(i)
                    break
    
    def _serialize_database_search_state(self, module_instance):
        """Serialize database search state"""
        state = {}
        if hasattr(module_instance, 'db_search_field'):
            state['search_text'] = module_instance.db_search_field.text()
        return state
    
    def _restore_database_search_state(self, module_instance, state):
        """Restore database search state"""
        if state.get('search_text') and hasattr(module_instance, 'db_search_field'):
            module_instance.db_search_field.setText(state['search_text'])
            # Trigger the search to populate results
            if hasattr(module_instance, 'filter_database_results'):
                module_instance.filter_database_results()
    
    def _update_ui_after_restore(self, module_instance):
        """Update UI elements after restoring state"""
        # Update the spectrum list
        if hasattr(module_instance, 'multi_spectrum_list'):
            list_widget = module_instance.multi_spectrum_list
            list_widget.clear()
            for spectrum_name in module_instance.loaded_spectra.keys():
                list_widget.addItem(spectrum_name)
        
        # Update the plot
        if hasattr(module_instance, 'update_multi_plot'):
            module_instance.update_multi_plot()
        
        # Update individual controls if there's a current selection
        if hasattr(module_instance, 'update_individual_controls'):
            current_item = getattr(module_instance, 'multi_spectrum_list', None)
            if current_item and current_item.currentItem():
                spectrum_name = current_item.currentItem().text()
                module_instance.update_individual_controls(spectrum_name)
    
    def _array_to_list(self, arr):
        """Convert numpy array to list, handling None values"""
        if arr is None:
            return None
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        return arr

class UniversalStateManager:
    """
    Universal state manager for RamanLab analysis modules.
    
    Provides a centralized system for saving and restoring analysis states
    across different modules with automatic serialization and data management.
    """
    
    def __init__(self, base_directory: str = None):
        """
        Initialize the universal state manager with cross-platform path handling.
        
        Args:
            base_directory: Base directory for state storage. If None, uses ~/RamanLab_Projects
        """
        if base_directory:
            # Sanitize the provided base directory
            sanitized_base = sanitize_path_for_windows(base_directory)
            self.base_directory = Path(sanitized_base)
        else:
            self.base_directory = Path.home() / "RamanLab_Projects"
        
        # Create directory with proper error handling
        try:
            self.base_directory.mkdir(exist_ok=True, parents=True)
        except (OSError, PermissionError) as e:
            print(f"Warning: Could not create state directory {self.base_directory}: {e}")
            # Fallback to a temporary directory in the current working directory
            self.base_directory = Path.cwd() / "RamanLab_State_Temp"
            self.base_directory.mkdir(exist_ok=True)
        
        # Module registration
        self._active_modules = {}
        self._serializers = {
            'batch_peak_fitting': BatchPeakFittingSerializer(),
            'polarization_analyzer': PolarizationAnalyzerSerializer(),
            'cluster_analysis': ClusterAnalysisSerializer(),
            'map_analysis_2d': MapAnalysisSerializer(),
            'raman_analysis_app': RamanAnalysisAppSerializer(),
            'multi_spectrum_manager': MultiSpectrumManagerSerializer(),
        }
        
        # Auto-save settings
        self._auto_save_enabled = True
        self._auto_save_interval = 300  # 5 minutes
        self._last_save_time = {}
        
        # Threading for auto-save
        self._save_lock = threading.Lock()
        
        # Project metadata
        self._current_project = None
        self._project_metadata = {}
    
    def register_module(self, module_name: str, module_instance, serializer: StateSerializerInterface = None):
        """Register a module instance for state management"""
        if serializer:
            self._serializers[module_name] = serializer
        
        # Store weak reference to avoid circular dependencies
        self._active_modules[module_name] = weakref.ref(module_instance)
        
        print(f"Registered module: {module_name}")
    
    def save_module_state(self, module_name: str, notes: str = "", tags: List[str] = None) -> bool:
        """Save state for a specific module"""
        if module_name not in self._active_modules:
            print(f"Module {module_name} not registered")
            return False
        
        module_ref = self._active_modules[module_name]
        module_instance = module_ref()
        
        if module_instance is None:
            print(f"Module {module_name} instance no longer exists")
            return False
        
        if module_name not in self._serializers:
            print(f"No serializer registered for module {module_name}")
            return False
        
        with self._save_lock:
            try:
                # Serialize module state
                serializer = self._serializers[module_name]
                state_data = serializer.serialize_state(module_instance)
                
                if not state_data:
                    print(f"Failed to serialize state for module {module_name}")
                    return False
                
                # Create metadata
                metadata = StateMetadata(
                    module_name=module_name,
                    version=STATE_MANAGER_VERSION,
                    created=datetime.now(timezone.utc).isoformat(),
                    last_modified=datetime.now(timezone.utc).isoformat(),
                    file_hash=self._calculate_hash(state_data),
                    user_notes=notes,
                    tags=tags or []
                )
                
                # Prepare complete state package
                state_package = {
                    'metadata': asdict(metadata),
                    'state_data': state_data
                }
                
                # Determine save location with proper path sanitization
                if self._current_project:
                    sanitized_project = sanitize_path_for_windows(self._current_project)
                    save_dir = self.base_directory / sanitized_project
                else:
                    save_dir = self.base_directory / "auto_saves"
                
                save_dir.mkdir(exist_ok=True)
                
                # Save state with sanitized filename
                sanitized_module_name = sanitize_path_for_windows(module_name)
                state_file = save_dir / f"{sanitized_module_name}_state.pkl"
                with open(state_file, 'wb') as f:
                    pickle.dump(state_package, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                self._last_save_time[module_name] = time.time()
                
                print(f"Successfully saved state for {module_name} to {state_file}")
                return True
                
            except Exception as e:
                print(f"Error saving state for {module_name}: {e}")
                return False
    
    def load_module_state(self, module_name: str, state_file: str = None) -> bool:
        """Load state for a specific module"""
        if module_name not in self._active_modules:
            print(f"Module {module_name} not registered")
            return False
        
        module_ref = self._active_modules[module_name]
        module_instance = module_ref()
        
        if module_instance is None:
            print(f"Module {module_name} instance no longer exists")
            return False
        
        if module_name not in self._serializers:
            print(f"No serializer registered for module {module_name}")
            return False
        
        # Determine state file location with proper path sanitization
        if not state_file:
            sanitized_module_name = sanitize_path_for_windows(module_name)
            if self._current_project:
                sanitized_project = sanitize_path_for_windows(self._current_project)
                state_file = self.base_directory / sanitized_project / f"{sanitized_module_name}_state.pkl"
            else:
                state_file = self.base_directory / "auto_saves" / f"{sanitized_module_name}_state.pkl"
        else:
            # Validate the provided state file path
            if isinstance(state_file, str) and not is_memory_database(state_file):
                state_file = Path(sanitize_path_for_windows(state_file))
            else:
                state_file = Path(state_file)
        
        if not state_file.exists():
            print(f"State file not found: {state_file}")
            return False
        
        try:
            # Load state package
            with open(state_file, 'rb') as f:
                state_package = pickle.load(f)
            
            # Validate state
            serializer = self._serializers[module_name]
            state_data = state_package['state_data']
            
            if not serializer.validate_state(state_data):
                print(f"Invalid state data for module {module_name}")
                return False
            
            # Restore state
            success = serializer.deserialize_state(state_data, module_instance)
            
            if success:
                print(f"Successfully loaded state for {module_name}")
                # Show state summary
                summary = serializer.get_state_summary(state_data)
                print(f"State Summary:\n{summary}")
            else:
                print(f"Failed to restore state for {module_name}")
            
            return success
            
        except Exception as e:
            print(f"Error loading state for {module_name}: {e}")
            return False
    
    def create_project(self, project_name: str, description: str = "") -> str:
        """Create a new analysis project with Windows-safe naming"""
        # Sanitize project name for cross-platform compatibility
        sanitized_project_name = sanitize_path_for_windows(project_name)
        project_dir = self.base_directory / sanitized_project_name
        project_dir.mkdir(exist_ok=True)
        
        # Create project metadata
        metadata = {
            'name': project_name,  # Keep original name in metadata
            'sanitized_name': sanitized_project_name,  # Track sanitized version
            'description': description,
            'created': datetime.now(timezone.utc).isoformat(),
            'last_modified': datetime.now(timezone.utc).isoformat(),
            'version': STATE_MANAGER_VERSION,
            'modules': {}
        }
        
        metadata_file = project_dir / "project_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self._current_project = sanitized_project_name  # Store sanitized name
        self._project_metadata = metadata
        
        print(f"Created project: {project_name} (sanitized: {sanitized_project_name}) at {project_dir}")
        return str(project_dir)
    
    def auto_save_module(self, module_name: str, force: bool = False):
        """Auto-save module state if conditions are met"""
        if not self._auto_save_enabled and not force:
            return
        
        current_time = time.time()
        last_save = self._last_save_time.get(module_name, 0)
        
        if force or (current_time - last_save) >= self._auto_save_interval:
            self.save_module_state(module_name, notes="Auto-save", tags=["auto-save"])
    
    def _calculate_hash(self, data) -> str:
        """Calculate hash of data for integrity checking"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def validate_paths(self) -> Dict[str, Any]:
        """
        Validate all paths used by the state manager for cross-platform compatibility.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'base_directory': {
                'path': str(self.base_directory),
                'exists': self.base_directory.exists(),
                'writable': os.access(self.base_directory, os.W_OK) if self.base_directory.exists() else False,
                'is_memory_db': is_memory_database(str(self.base_directory))
            },
            'current_project': None,
            'auto_saves_directory': None
        }
        
        if self._current_project:
            project_dir = self.base_directory / self._current_project
            results['current_project'] = {
                'name': self._current_project,
                'path': str(project_dir),
                'exists': project_dir.exists(),
                'writable': os.access(project_dir, os.W_OK) if project_dir.exists() else False
            }
        
        auto_saves_dir = self.base_directory / "auto_saves"
        results['auto_saves_directory'] = {
            'path': str(auto_saves_dir),
            'exists': auto_saves_dir.exists(),
            'writable': os.access(auto_saves_dir, os.W_OK) if auto_saves_dir.exists() else False
        }
        
        return results

# Global state manager instance
_global_state_manager = None

def get_state_manager() -> UniversalStateManager:
    """Get the global state manager instance"""
    global _global_state_manager
    if _global_state_manager is None:
        _global_state_manager = UniversalStateManager()
    return _global_state_manager

def register_module(module_name: str, module_instance, serializer: StateSerializerInterface = None):
    """Convenience function to register a module with the global state manager"""
    return get_state_manager().register_module(module_name, module_instance, serializer)

def save_module_state(module_name: str, notes: str = "", tags: List[str] = None) -> bool:
    """Convenience function to save module state"""
    return get_state_manager().save_module_state(module_name, notes, tags)

def load_module_state(module_name: str, state_file: str = None) -> bool:
    """Convenience function to load module state"""
    return get_state_manager().load_module_state(module_name, state_file)

def auto_save_module(module_name: str, force: bool = False):
    """Convenience function for auto-save"""
    return get_state_manager().auto_save_module(module_name, force)

# Example usage and integration helpers
def integrate_with_batch_peak_fitting(batch_peak_fitting_instance):
    """Easy integration with batch peak fitting module"""
    state_manager = get_state_manager()
    state_manager.register_module('batch_peak_fitting', batch_peak_fitting_instance)
    
    # Add auto-save hooks to key methods
    original_save_method = batch_peak_fitting_instance.update_batch_results_with_manual_fit
    
    def enhanced_save_method(*args, **kwargs):
        result = original_save_method(*args, **kwargs)
        # Auto-save after manual fit updates
        auto_save_module('batch_peak_fitting')
        return result
    
    batch_peak_fitting_instance.update_batch_results_with_manual_fit = enhanced_save_method
    
    return state_manager

if __name__ == "__main__":
    # Example usage and testing
    print("Testing Universal State Manager with Windows compatibility fixes...")
    
    # Test path sanitization functions
    test_paths = [
        ":memory:",
        ":memory:test",
        "C:\\Users\\Test:File.db",
        "regular_file.pkl",
        "file<with>invalid|chars?.txt",
        "sqlite://test.db"
    ]
    
    print("\nTesting path sanitization:")
    for path in test_paths:
        sanitized = sanitize_path_for_windows(path)
        is_memory = is_memory_database(path)
        validated = validate_and_sanitize_database_path(path)
        print(f"  Original: {path}")
        print(f"  Sanitized: {sanitized}")
        print(f"  Is memory DB: {is_memory}")
        print(f"  Validated: {validated}")
        print()
    
    # Test state manager
    state_manager = UniversalStateManager()
    
    # Validate paths
    path_validation = state_manager.validate_paths()
    print("Path validation results:")
    for key, value in path_validation.items():
        if value:
            print(f"  {key}: {value}")
    
    # Test project creation with special characters
    test_project_names = [
        "My Raman Analysis",
        "Project:With:Colons",
        "Project<With>Invalid|Chars?",
        "Normal_Project_Name"
    ]
    
    print("\nTesting project creation:")
    for project_name in test_project_names:
        try:
            project_path = state_manager.create_project(project_name, f"Test project: {project_name}")
            print(f"  ✓ Created: {project_name} -> {project_path}")
        except Exception as e:
            print(f"  ✗ Failed: {project_name} -> {e}")
    
    print("\nUniversal State Manager Windows compatibility test completed!") 