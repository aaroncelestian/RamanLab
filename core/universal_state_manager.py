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
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable, Union
import numpy as np
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import threading
import weakref

# Version for backward compatibility
STATE_MANAGER_VERSION = "1.0.0"

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
                'custom_db_path': getattr(module_instance, 'custom_db_path', None),
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
            for attr in ['selected_folder', 'visualization_method', 'n_subclusters', 'split_method', 'custom_db_path']:
                if attr in state_data:
                    setattr(module_instance, attr, state_data[attr])
            
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
                'map_data_path': getattr(module_instance.map_data, 'data_path', None) if hasattr(module_instance, 'map_data') and module_instance.map_data else None,
                
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


class UniversalStateManager:
    """Universal state manager for all RamanLab analysis modules"""
    
    def __init__(self, base_directory: str = None):
        if base_directory:
            self.base_directory = Path(base_directory)
        else:
            from .config_manager import get_config_manager
            config = get_config_manager()
            self.base_directory = config.get_projects_folder()
        self.base_directory.mkdir(exist_ok=True)
        
        # Registry of module serializers
        self._serializers: Dict[str, StateSerializerInterface] = {
            'batch_peak_fitting': BatchPeakFittingSerializer(),
            'polarization_analyzer': PolarizationAnalyzerSerializer(),
            'cluster_analysis': ClusterAnalysisSerializer(),
            'map_analysis_2d': MapAnalysisSerializer(),
        }
        
        # Active module instances (weak references to avoid circular dependencies)
        self._active_modules: Dict[str, weakref.ref] = {}
        
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
                
                # Determine save location
                if self._current_project:
                    save_dir = self.base_directory / self._current_project
                else:
                    save_dir = self.base_directory / "auto_saves"
                
                save_dir.mkdir(exist_ok=True)
                
                # Save state
                state_file = save_dir / f"{module_name}_state.pkl"
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
        
        # Determine state file location
        if not state_file:
            if self._current_project:
                state_file = self.base_directory / self._current_project / f"{module_name}_state.pkl"
            else:
                state_file = self.base_directory / "auto_saves" / f"{module_name}_state.pkl"
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
        """Create a new analysis project"""
        project_dir = self.base_directory / project_name
        project_dir.mkdir(exist_ok=True)
        
        # Create project metadata
        metadata = {
            'name': project_name,
            'description': description,
            'created': datetime.now(timezone.utc).isoformat(),
            'last_modified': datetime.now(timezone.utc).isoformat(),
            'version': STATE_MANAGER_VERSION,
            'modules': {}
        }
        
        metadata_file = project_dir / "project_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self._current_project = project_name
        self._project_metadata = metadata
        
        print(f"Created project: {project_name} at {project_dir}")
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
    # Example usage
    with UniversalStateManager() as state_manager:
        # Create a project
        project_path = state_manager.create_project("My_Raman_Analysis", "Comprehensive analysis of mineral samples")
        
        # Register modules (this would be done by each module)
        # state_manager.register_module('batch_peak_fitting', batch_instance)
        # state_manager.register_module('polarization_analysis', polarization_instance)
        
        # Save states
        # state_manager.save_module_state('batch_peak_fitting', "After manual adjustments")
        
        # List projects
        projects = state_manager.list_projects()
        for project in projects:
            print(f"Project: {project['name']} - {project['description']}") 