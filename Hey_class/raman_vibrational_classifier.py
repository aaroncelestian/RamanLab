#!/usr/bin/env python3
"""
Hey-Celestian Classification System

A novel mineral classification approach that organizes minerals based on their characteristic
vibrational signatures as observed in Raman spectroscopy. This system builds upon Hey's 
foundational work but reorganizes minerals by their dominant vibrational modes rather than 
purely chemical composition, making it more intuitive and practical for Raman spectroscopists.

The Hey-Celestian system provides better groupings for spectral identification, predictive 
analysis capabilities, and targeted measurement strategies for each vibrational class.

Author: Aaron Celestian & RamanLab Development Team
Based on: Hey's Chemical Index of Minerals (1962)
Innovation: Vibrational mode-based reorganization for Raman spectroscopy
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from scipy.signal import find_peaks


class SpectralPeakMatcher:
    """
    Enhanced spectral peak matching with constrained peak labeling.
    Provides robust peak identification and assignment to vibrational modes.
    """
    
    def __init__(self, tolerance: float = 15.0):
        """
        Initialize peak matcher.
        
        Parameters:
        -----------
        tolerance : float
            Default tolerance for peak matching in cm⁻¹
        """
        self.tolerance = tolerance
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
    
    def match_peaks_to_modes(self, detected_peaks: List[float], 
                           expected_modes: List[Dict], 
                           tolerance: Optional[float] = None) -> List[Dict]:
        """
        Match detected peaks to expected vibrational modes with confidence scoring.
        
        Parameters:
        -----------
        detected_peaks : List[float]
            List of detected peak positions in cm⁻¹
        expected_modes : List[Dict]
            List of expected modes with 'peak', 'range', 'description'
        tolerance : float, optional
            Custom tolerance for this matching
            
        Returns:
        --------
        List[Dict]
            Matched peaks with confidence scores and assignments
        """
        if not detected_peaks or not expected_modes:
            return []
        
        tol = tolerance or self.tolerance
        matches = []
        used_peaks = set()
        
        # Sort expected modes by priority (narrower ranges first for specificity)
        sorted_modes = sorted(expected_modes, 
                            key=lambda x: x['range'][1] - x['range'][0])
        
        for mode in sorted_modes:
            best_match = None
            best_distance = float('inf')
            
            # Find closest peak within tolerance
            for i, peak in enumerate(detected_peaks):
                if i in used_peaks:
                    continue
                    
                # Check if peak is within the expected range
                in_range = mode['range'][0] <= peak <= mode['range'][1]
                distance = abs(peak - mode['peak'])
                
                if distance <= tol and distance < best_distance:
                    best_match = i
                    best_distance = distance
            
            if best_match is not None:
                peak_pos = detected_peaks[best_match]
                confidence = self._calculate_peak_confidence(
                    peak_pos, mode, best_distance, tol
                )
                
                matches.append({
                    'detected_peak': peak_pos,
                    'expected_peak': mode['peak'],
                    'mode_description': mode['description'],
                    'distance': best_distance,
                    'confidence': confidence,
                    'in_range': mode['range'][0] <= peak_pos <= mode['range'][1],
                    'assignment_quality': self._get_assignment_quality(confidence)
                })
                
                used_peaks.add(best_match)
        
        return matches
    
    def _calculate_peak_confidence(self, detected: float, mode: Dict, 
                                 distance: float, tolerance: float) -> float:
        """Calculate confidence score for peak assignment."""
        # Base confidence from distance
        distance_confidence = max(0.0, 1.0 - (distance / tolerance))
        
        # Range bonus - peaks closer to center of range get higher confidence
        range_center = (mode['range'][0] + mode['range'][1]) / 2
        range_width = mode['range'][1] - mode['range'][0]
        range_distance = abs(detected - range_center)
        range_confidence = max(0.0, 1.0 - (range_distance / (range_width / 2)))
        
        # Combined confidence (weighted average)
        confidence = 0.7 * distance_confidence + 0.3 * range_confidence
        
        return min(1.0, confidence)
    
    def _get_assignment_quality(self, confidence: float) -> str:
        """Get qualitative assessment of assignment quality."""
        if confidence >= self.confidence_thresholds['high']:
            return 'excellent'
        elif confidence >= self.confidence_thresholds['medium']:
            return 'good'
        elif confidence >= self.confidence_thresholds['low']:
            return 'tentative'
        else:
            return 'poor'
    
    def constrained_peak_labeling(self, detected_peaks: List[float],
                                 group_modes: Dict[str, List[Dict]],
                                 chemistry: str = "") -> Dict:
        """
        Constrained peak labeling approach that considers multiple vibrational groups
        and enforces chemical constraints.
        
        Parameters:
        -----------
        detected_peaks : List[float]
            Detected peak positions
        group_modes : Dict[str, List[Dict]]
            Dictionary of group_id -> expected modes
        chemistry : str
            Chemical formula for additional constraints
            
        Returns:
        --------
        Dict
            Comprehensive peak labeling results
        """
        results = {
            'total_peaks': len(detected_peaks),
            'group_matches': {},
            'best_group': None,
            'confidence_score': 0.0,
            'peak_assignments': [],
            'unassigned_peaks': [],
            'chemical_constraints': []
        }
        
        if not detected_peaks:
            return results
        
        # Apply chemical constraints
        chemical_constraints = self._get_chemical_constraints(chemistry)
        results['chemical_constraints'] = chemical_constraints
        
        # Test each vibrational group
        for group_id, expected_modes in group_modes.items():
            matches = self.match_peaks_to_modes(detected_peaks, expected_modes)
            
            if matches:
                # Calculate group score
                group_score = self._calculate_group_score(matches, expected_modes)
                
                # Apply chemical constraint modifiers
                constrained_score = self._apply_chemical_constraints(
                    group_score, group_id, chemical_constraints
                )
                
                results['group_matches'][group_id] = {
                    'matches': matches,
                    'raw_score': group_score,
                    'constrained_score': constrained_score,
                    'matched_peaks': len(matches),
                    'expected_peaks': len(expected_modes)
                }
        
        # Determine best group
        if results['group_matches']:
            best_group_id = max(results['group_matches'].keys(),
                              key=lambda x: results['group_matches'][x]['constrained_score'])
            
            results['best_group'] = best_group_id
            results['confidence_score'] = results['group_matches'][best_group_id]['constrained_score']
            results['peak_assignments'] = results['group_matches'][best_group_id]['matches']
        
        # Identify unassigned peaks
        assigned_peaks = set()
        if results['peak_assignments']:
            assigned_peaks = {match['detected_peak'] for match in results['peak_assignments']}
        
        results['unassigned_peaks'] = [peak for peak in detected_peaks 
                                     if peak not in assigned_peaks]
        
        return results
    
    def _calculate_group_score(self, matches: List[Dict], expected_modes: List[Dict]) -> float:
        """Calculate overall group score from peak matches."""
        if not matches or not expected_modes:
            return 0.0
        
        # Base score from match quality
        total_confidence = sum(match['confidence'] for match in matches)
        avg_confidence = total_confidence / len(matches)
        
        # Coverage bonus (fraction of expected modes found)
        coverage = len(matches) / len(expected_modes)
        
        # High-confidence match bonus
        excellent_matches = sum(1 for match in matches 
                              if match['assignment_quality'] == 'excellent')
        excellence_bonus = excellent_matches / len(expected_modes)
        
        # Combined score
        score = 0.5 * avg_confidence + 0.3 * coverage + 0.2 * excellence_bonus
        
        return min(1.0, score)
    
    def _get_chemical_constraints(self, chemistry: str) -> List[str]:
        """Extract chemical constraints from formula."""
        constraints = []
        
        if not chemistry:
            return constraints
        
        # Check for specific chemical groups
        if re.search(r'CO_?3', chemistry, re.IGNORECASE):
            constraints.append('carbonate_required')
        if re.search(r'SO_?4', chemistry, re.IGNORECASE):
            constraints.append('sulfate_required')
        if re.search(r'PO_?4', chemistry, re.IGNORECASE):
            constraints.append('phosphate_required')
        if re.search(r'SiO_?2', chemistry, re.IGNORECASE):
            constraints.append('silicate_framework')
        if re.search(r'OH', chemistry, re.IGNORECASE):
            constraints.append('hydroxide_present')
        
        # Check for organic compounds
        # Look for organic patterns (C-H, C-N combinations without typical inorganic groups)
        has_carbon = 'C' in chemistry
        has_hydrogen = 'H' in chemistry
        has_nitrogen = 'N' in chemistry
        has_carbonate = re.search(r'CO_?3', chemistry, re.IGNORECASE)
        
        if has_carbon and (has_hydrogen or has_nitrogen) and not has_carbonate:
            constraints.append('organic_compound')
        
        # Check for sheet silicate patterns
        sheet_silicate_indicators = [
            r'KAl.*Si.*O.*OH',    # Mica patterns
            r'Al.*Si.*O.*OH.*H2O', # Clay patterns
            r'Mg.*Si.*O.*OH',     # Talc/serpentine patterns
        ]
        
        for pattern in sheet_silicate_indicators:
            if re.search(pattern, chemistry, re.IGNORECASE):
                constraints.append('sheet_silicate_structure')
                break
        
        # Check for simple oxide patterns (metal + oxygen, minimal other elements)
        simple_oxide_indicators = [
            r'^Fe_?[0-9]*_?O_?[0-9]*$',    # Iron oxides
            r'^Ti_?O_?2$',                  # Titanium dioxide
            r'^Al_?2_?O_?3$',              # Aluminum oxide
            r'^Si_?O_?2$',                 # Silicon dioxide
            r'^[A-Z][a-z]?_?[0-9]*_?O_?[0-9]*$'  # General simple oxide
        ]
        
        for pattern in simple_oxide_indicators:
            if re.search(pattern, chemistry.replace(' ', ''), re.IGNORECASE):
                constraints.append('simple_oxide_structure')
                break
        
        # Check for octahedral framework patterns
        octahedral_indicators = [
            r'^Ti_?O_?2$',                  # Rutile/anatase
            r'^Al_?2_?O_?3$',              # Corundum
            r'^[A-Z][a-z]?[A-Z][a-z]?_?2_?O_?4$',  # Spinel structures
            r'^[A-Z][a-z]?[A-Z][a-z]?O_?3$'   # Perovskite structures
        ]
        
        for pattern in octahedral_indicators:
            if re.search(pattern, chemistry.replace(' ', ''), re.IGNORECASE):
                constraints.append('octahedral_framework')
                break
        
        # Check for single chain silicate patterns
        single_chain_indicators = [
            r'[A-Z][a-z]?Si_?O_?3',         # Simple pyroxenes (MSiO3)
            r'[A-Z][a-z]?[A-Z][a-z]?Si_?2_?O_?6',  # Complex pyroxenes (XYSi2O6)
            r'Ca_?Si_?O_?3',                # Wollastonite
            r'Mn_?Si_?O_?3'                 # Rhodonite
        ]
        
        for pattern in single_chain_indicators:
            if re.search(pattern, chemistry.replace(' ', ''), re.IGNORECASE):
                constraints.append('single_chain_silicate')
                break
        
        # Check for double chain silicate patterns (amphiboles)
        double_chain_indicators = [
            r'.*Si_?8_?O_?22_?.*OH',        # Classic amphibole formula
            r'Ca.*Mg.*Si_?8_?O_?22',        # Tremolite-actinolite
            r'Na.*Al.*Si_?8_?O_?22',        # Glaucophane-type
            r'.*OH.*Si_?8',                 # OH-bearing silicates with 8 Si
            r'Ca_?2_?.*Si_?8_?O_?22',       # Calcic amphiboles
            r'Na_?2_?.*Si_?8_?O_?22'        # Sodic amphiboles
        ]
        
        for pattern in double_chain_indicators:
            if re.search(pattern, chemistry.replace(' ', ''), re.IGNORECASE):
                constraints.append('double_chain_silicate')
                break
        
        # Check for ring silicate patterns (cyclosilicates)
        ring_silicate_indicators = [
            r'.*B.*Si_?6_?O_?18',           # Tourmaline group
            r'Be_?3_?Al_?2_?Si_?6_?O_?18',  # Beryl formula
            r'.*Si_?6_?O_?18',              # General 6-membered rings
            r'Ba_?Ti_?Si_?3_?O_?9',         # Benitoite
            r'.*Si_?3_?O_?9',               # 3-membered rings
            r'.*Al_?4_?Si_?5_?O_?18'        # Cordierite formula
        ]
        
        for pattern in ring_silicate_indicators:
            if re.search(pattern, chemistry.replace(' ', ''), re.IGNORECASE):
                constraints.append('ring_silicate')
                break
        
        return constraints
    
    def _apply_chemical_constraints(self, score: float, group_id: str, 
                                  constraints: List[str]) -> float:
        """Apply chemical constraints to modify group scores."""
        modified_score = score
        
        # Group-specific constraint bonuses/penalties
        constraint_modifiers = {
            '3': {'carbonate_required': 1.5, 'sulfate_required': 0.1, 'phosphate_required': 0.1, 'organic_compound': 0.05, 'sheet_silicate_structure': 0.1, 'simple_oxide_structure': 0.1, 'octahedral_framework': 0.1, 'single_chain_silicate': 0.1, 'double_chain_silicate': 0.1, 'ring_silicate': 0.1},
            '4': {'sulfate_required': 1.5, 'carbonate_required': 0.1, 'phosphate_required': 0.1, 'organic_compound': 0.05, 'sheet_silicate_structure': 0.1, 'simple_oxide_structure': 0.1, 'octahedral_framework': 0.1, 'single_chain_silicate': 0.1, 'double_chain_silicate': 0.1, 'ring_silicate': 0.1},  
            '5': {'phosphate_required': 1.5, 'carbonate_required': 0.1, 'sulfate_required': 0.1, 'organic_compound': 0.1, 'sheet_silicate_structure': 0.1, 'simple_oxide_structure': 0.1, 'octahedral_framework': 0.1, 'single_chain_silicate': 0.1, 'double_chain_silicate': 0.1, 'ring_silicate': 0.1},
            '1': {'silicate_framework': 1.3, 'carbonate_required': 0.2, 'organic_compound': 0.1, 'sheet_silicate_structure': 0.3, 'simple_oxide_structure': 1.2, 'octahedral_framework': 0.3, 'single_chain_silicate': 0.1, 'double_chain_silicate': 0.1, 'ring_silicate': 0.1},  # Quartz is both framework and simple oxide
            '2': {'octahedral_framework': 2.0, 'simple_oxide_structure': 1.2, 'carbonate_required': 0.05, 'sulfate_required': 0.05, 'phosphate_required': 0.05, 'organic_compound': 0.05, 'sheet_silicate_structure': 0.1, 'single_chain_silicate': 0.05, 'double_chain_silicate': 0.05, 'ring_silicate': 0.05},  # Group 2 is octahedral frameworks
            '6': {'single_chain_silicate': 2.0, 'silicate_framework': 0.6, 'carbonate_required': 0.05, 'sulfate_required': 0.05, 'phosphate_required': 0.05, 'organic_compound': 0.05, 'sheet_silicate_structure': 0.1, 'simple_oxide_structure': 0.05, 'octahedral_framework': 0.1, 'double_chain_silicate': 0.1, 'ring_silicate': 0.1},  # Group 6 is single chain silicates
            '7': {'double_chain_silicate': 2.0, 'single_chain_silicate': 0.5, 'silicate_framework': 0.6, 'carbonate_required': 0.05, 'sulfate_required': 0.05, 'phosphate_required': 0.05, 'organic_compound': 0.05, 'sheet_silicate_structure': 0.2, 'simple_oxide_structure': 0.05, 'octahedral_framework': 0.1, 'ring_silicate': 0.1},  # Group 7 is double chain silicates
            '8': {'ring_silicate': 2.0, 'silicate_framework': 0.6, 'carbonate_required': 0.05, 'sulfate_required': 0.05, 'phosphate_required': 0.05, 'organic_compound': 0.05, 'sheet_silicate_structure': 0.1, 'simple_oxide_structure': 0.05, 'octahedral_framework': 0.1, 'single_chain_silicate': 0.1, 'double_chain_silicate': 0.1},  # Group 8 is ring silicates
            '9': {'sheet_silicate_structure': 2.0, 'silicate_framework': 0.8, 'carbonate_required': 0.1, 'sulfate_required': 0.05, 'organic_compound': 0.1, 'simple_oxide_structure': 0.1, 'octahedral_framework': 0.1, 'single_chain_silicate': 0.1, 'double_chain_silicate': 0.1, 'ring_silicate': 0.1},  # Group 9 is sheet silicates
            '10': {'non_silicate_layer': 2.0, 'carbonate_required': 0.05, 'sulfate_required': 0.05, 'phosphate_required': 0.05, 'organic_compound': 0.1, 'sheet_silicate_structure': 0.05, 'simple_oxide_structure': 0.3, 'octahedral_framework': 0.2, 'silicate_framework': 0.05},  # Group 10 is non-silicate layers
            '11': {'simple_oxide_structure': 2.0, 'carbonate_required': 0.05, 'sulfate_required': 0.05, 'phosphate_required': 0.05, 'organic_compound': 0.05, 'sheet_silicate_structure': 0.1, 'octahedral_framework': 1.3, 'single_chain_silicate': 0.05, 'double_chain_silicate': 0.05, 'ring_silicate': 0.05},  # Group 11 is simple oxides, but some overlap with octahedral
            '12': {'complex_oxide_structure': 2.0, 'simple_oxide_structure': 0.8, 'octahedral_framework': 0.6, 'carbonate_required': 0.05, 'sulfate_required': 0.05, 'phosphate_required': 0.05, 'organic_compound': 0.05, 'sheet_silicate_structure': 0.05, 'silicate_framework': 0.3},  # Group 12 is complex oxides
            '13': {'hydroxide_present': 2.0, 'organic_compound': 0.2, 'sheet_silicate_structure': 0.9, 'simple_oxide_structure': 0.8, 'octahedral_framework': 0.7, 'single_chain_silicate': 0.1, 'double_chain_silicate': 0.1, 'ring_silicate': 0.1, 'non_silicate_layer': 0.5},  # Group 13 is hydroxides
            '14': {'organic_compound': 2.0, 'carbonate_required': 0.1, 'sulfate_required': 0.05, 'phosphate_required': 0.1, 'sheet_silicate_structure': 0.05, 'simple_oxide_structure': 0.05, 'octahedral_framework': 0.05, 'single_chain_silicate': 0.05, 'double_chain_silicate': 0.05, 'ring_silicate': 0.05},  # Group 14 is organic
            '15': {'mixed_mode_structure': 2.0, 'silicate_framework': 1.2, 'carbonate_required': 0.3, 'sulfate_required': 0.3, 'phosphate_required': 0.2, 'organic_compound': 0.1, 'sheet_silicate_structure': 0.4, 'simple_oxide_structure': 0.2, 'octahedral_framework': 0.3}  # Group 15 is mixed modes
        }
        
        if group_id in constraint_modifiers:
            for constraint in constraints:
                if constraint in constraint_modifiers[group_id]:
                    modifier = constraint_modifiers[group_id][constraint]
                    modified_score *= modifier
        
        return min(1.0, modified_score)


class HeyCelestianClassifier:
    """
    Hey-Celestian Classification System with Spectral Analysis
    
    Classify minerals based on their dominant vibrational modes and structural units
    as observed in Raman spectroscopy. Now includes actual spectral peak analysis
    and constrained peak labeling for robust identification.
    """
    
    def __init__(self, peak_tolerance: float = 15.0):
        """Initialize the vibrational classifier with spectral analysis capabilities."""
        self.vibrational_groups = self._define_vibrational_groups()
        self.characteristic_modes = self._define_characteristic_modes()
        self.structural_indicators = self._define_structural_indicators()
        self.peak_matcher = SpectralPeakMatcher(tolerance=peak_tolerance)
        
        # Define expected modes for each group
        self.group_expected_modes = self._define_group_expected_modes()
    
    def _define_vibrational_groups(self) -> Dict:
        """Define the main vibrational classification groups."""
        return {
            "1": {
                "name": "Framework Modes - Tetrahedral Networks",
                "description": "Minerals with 3D tetrahedral frameworks (SiO4, AlO4, PO4)",
                "typical_range": "400-1200 cm⁻¹",
                "examples": ["Quartz", "Feldspar", "Zeolites", "Cristobalite"]
            },
            "2": {
                "name": "Framework Modes - Octahedral Networks", 
                "description": "Minerals with octahedral coordination frameworks",
                "typical_range": "200-800 cm⁻¹",
                "examples": ["Rutile", "Anatase", "Spinel", "Corundum"]
            },
            "3": {
                "name": "Characteristic Vibrational Mode - Carbonate Groups",
                "description": "Minerals with discrete CO3²⁻ molecular units",
                "typical_range": "1050-1100 cm⁻¹ (ν1), 700-900 cm⁻¹ (ν4)",
                "examples": ["Calcite", "Aragonite", "Dolomite", "Malachite"]
            },
            "4": {
                "name": "Characteristic Vibrational Mode - Sulfate Groups",
                "description": "Minerals with discrete SO4²⁻ molecular units", 
                "typical_range": "980-1020 cm⁻¹ (ν1), 400-700 cm⁻¹ (ν2,ν4)",
                "examples": ["Gypsum", "Anhydrite", "Barite", "Celestine"]
            },
            "5": {
                "name": "Characteristic Vibrational Mode - Phosphate Groups",
                "description": "Minerals with discrete PO4³⁻ molecular units",
                "typical_range": "950-980 cm⁻¹ (ν1), 400-650 cm⁻¹ (ν2,ν4)",
                "examples": ["Apatite", "Vivianite", "Turquoise", "Monazite"]
            },
            "6": {
                "name": "Chain Modes - Single Chain Silicates",
                "description": "Minerals with single chains of SiO4 tetrahedra",
                "typical_range": "650-700 cm⁻¹ (Si-O-Si), 300-500 cm⁻¹ (M-O)",
                "examples": ["Pyroxenes", "Wollastonite", "Rhodonite"]
            },
            "7": {
                "name": "Chain Modes - Double Chain Silicates", 
                "description": "Minerals with double chains of SiO4 tetrahedra",
                "typical_range": "660-680 cm⁻¹ (Si-O-Si), 200-400 cm⁻¹ (M-O)",
                "examples": ["Amphiboles", "Actinolite", "Hornblende"]
            },
            "8": {
                "name": "Ring Modes - Cyclosilicates",
                "description": "Minerals with ring structures of SiO4 tetrahedra",
                "typical_range": "500-800 cm⁻¹ (ring breathing), 200-500 cm⁻¹",
                "examples": ["Tourmaline", "Beryl", "Cordierite", "Benitoite"]
            },
            "9": {
                "name": "Layer Modes - Sheet Silicates",
                "description": "Minerals with layered silicate structures",
                "typical_range": "100-600 cm⁻¹ (layer modes), 3500-3700 cm⁻¹ (OH)",
                "examples": ["Micas", "Clays", "Talc", "Chlorite"]
            },
            "10": {
                "name": "Layer Modes - Non-Silicate Layers",
                "description": "Minerals with layered non-silicate structures",
                "typical_range": "100-500 cm⁻¹ (layer modes)",
                "examples": ["Graphite", "Molybdenite", "Brucite", "Gibbsite"]
            },
            "11": {
                "name": "Metal-Oxygen Modes - Simple Oxides",
                "description": "Simple metal oxide structures",
                "typical_range": "200-800 cm⁻¹ (M-O stretching)",
                "examples": ["Hematite", "Magnetite", "Cuprite", "Zincite"]
            },
            "12": {
                "name": "Metal-Oxygen Modes - Complex Oxides",
                "description": "Complex metal oxide structures and spinels",
                "typical_range": "200-800 cm⁻¹ (M-O stretching)",
                "examples": ["Chromite", "Franklinite", "Gahnite", "Hercynite"]
            },
            "13": {
                "name": "Hydroxide Modes",
                "description": "Minerals with prominent OH⁻ groups",
                "typical_range": "3200-3700 cm⁻¹ (OH stretch), 200-800 cm⁻¹ (M-OH)",
                "examples": ["Goethite", "Lepidocrocite", "Diaspore", "Boehmite"]
            },
            "14": {
                "name": "Characteristic Vibrational Mode - Organic Groups",
                "description": "Organic minerals and biominerals",
                "typical_range": "1000-1800 cm⁻¹ (C-C, C-O), 2800-3000 cm⁻¹ (C-H)",
                "examples": ["Whewellite", "Weddellite", "Amber", "Jet"]
            },
            "15": {
                "name": "Mixed Modes",
                "description": "Minerals with multiple distinct vibrational units",
                "typical_range": "Variable - multiple characteristic regions",
                "examples": ["Epidote", "Vesuvianite", "Sodalite", "Scapolite"]
            }
        }
    
    def _define_characteristic_modes(self) -> Dict:
        """Define characteristic vibrational modes for identification."""
        return {
            # Tetrahedral framework modes
            "si_o_framework": {"range": (400, 1200), "peak": 460, "description": "Si-O framework vibrations"},
            "al_o_framework": {"range": (400, 800), "peak": 500, "description": "Al-O framework vibrations"},
            
            # Molecular group modes
            "carbonate_v1": {"range": (1050, 1100), "peak": 1085, "description": "CO3 symmetric stretch"},
            "carbonate_v4": {"range": (700, 900), "peak": 712, "description": "CO3 bending"},
            "sulfate_v1": {"range": (980, 1020), "peak": 1008, "description": "SO4 symmetric stretch"},
            "phosphate_v1": {"range": (950, 980), "peak": 960, "description": "PO4 symmetric stretch"},
            
            # Chain and ring modes
            "pyroxene_chain": {"range": (650, 700), "peak": 665, "description": "Single chain Si-O-Si"},
            "amphibole_chain": {"range": (660, 680), "peak": 670, "description": "Double chain Si-O-Si"},
            "ring_breathing": {"range": (500, 800), "peak": 640, "description": "Ring breathing modes"},
            
            # Layer modes
            "layer_bending": {"range": (100, 300), "peak": 200, "description": "Layer bending modes"},
            "oh_stretch": {"range": (3200, 3700), "peak": 3620, "description": "OH stretching"},
            
            # Metal-oxygen modes
            "fe_o_stretch": {"range": (200, 400), "peak": 300, "description": "Fe-O stretching"},
            "ti_o_stretch": {"range": (400, 700), "peak": 515, "description": "Ti-O stretching"},
        }
    
    def _define_structural_indicators(self) -> Dict:
        """Define structural indicators from chemical formulas."""
        return {
            # Framework indicators
            "tetrahedral_framework": [
                r"SiO_?2", r"AlO_?2", r"PO_?2", r"GeO_?2",  # Framework formers
                r"Si\d*O\d*", r"Al\d*O\d*"  # General tetrahedral
            ],
            
            # Molecular group indicators
            "carbonate_groups": [
                r"CO_?3", r"\(CO_?3\)", r"C\d*O\d*"
            ],
            "sulfate_groups": [
                r"SO_?4", r"\(SO_?4\)", r"S\d*O\d*"
            ],
            "phosphate_groups": [
                r"PO_?4", r"\(PO_?4\)", r"P\d*O\d*", r"AsO_?4", r"VO_?4"
            ],
            
            # Chain indicators
            "chain_silicates": [
                r"Si_?2_?O_?6", r"Si_?4_?O_?11", r"SiO_?3"  # Chain stoichiometries
            ],
            
            # Layer indicators  
            "layer_silicates": [
                r"Si_?2_?O_?5", r"Si_?4_?O_?10", r"Al_?2_?Si_?2_?O_?5"  # Layer stoichiometries
            ],
            
            # Hydroxide indicators
            "hydroxide_groups": [
                r"OH", r"\(OH\)", r"H_?2_?O"
            ],
            
            # Organic indicators
            "organic_groups": [
                r"C_?2_?O_?4", r"C_?2_?H_?2", r"CH", r"COOH"
            ]
        }
    
    def _define_group_expected_modes(self) -> Dict[str, List[Dict]]:
        """
        Define expected vibrational modes for each classification group.
        This is the core of the spectral analysis system.
        """
        return {
            "1": [  # Framework Modes - Tetrahedral Networks
                {"range": (440, 480), "peak": 460, "description": "Si-O-Si bending (quartz main peak)"},
                {"range": (1050, 1120), "peak": 1085, "description": "Si-O symmetric stretch"},
                {"range": (790, 810), "peak": 798, "description": "Si-O-Si symmetric stretch"},
                {"range": (350, 370), "peak": 360, "description": "Framework deformation"},
                {"range": (120, 140), "peak": 128, "description": "Framework lattice mode"}
            ],
            "2": [  # Framework Modes - Octahedral Networks
                {"range": (430, 450), "peak": 440, "description": "Ti-O stretch (anatase)"},
                {"range": (510, 530), "peak": 515, "description": "Ti-O stretch (rutile)"},
                {"range": (190, 210), "peak": 200, "description": "Ti-O-Ti bending"},
                {"range": (600, 620), "peak": 610, "description": "Octahedral symmetric stretch"},
                {"range": (290, 310), "peak": 300, "description": "Octahedral deformation"}
            ],
            "3": [  # Characteristic Vibrational Mode - Carbonate Groups
                {"range": (1080, 1090), "peak": 1085, "description": "CO3 symmetric stretch (ν1)"},
                {"range": (710, 720), "peak": 712, "description": "CO3 in-plane bending (ν4)"},
                {"range": (1430, 1450), "peak": 1435, "description": "CO3 antisymmetric stretch (ν3)"},
                {"range": (870, 890), "peak": 879, "description": "CO3 out-of-plane bending (ν2)"},
                {"range": (150, 170), "peak": 160, "description": "Lattice mode"}
            ],
            "4": [  # Characteristic Vibrational Mode - Sulfate Groups
                {"range": (1000, 1020), "peak": 1008, "description": "SO4 symmetric stretch (ν1)"},
                {"range": (1110, 1130), "peak": 1120, "description": "SO4 antisymmetric stretch (ν3)"},
                {"range": (620, 640), "peak": 630, "description": "SO4 antisymmetric bending (ν4)"},
                {"range": (450, 470), "peak": 460, "description": "SO4 symmetric bending (ν2)"},
                {"range": (410, 430), "peak": 420, "description": "SO4 translation"}
            ],
            "5": [  # Characteristic Vibrational Mode - Phosphate Groups
                {"range": (955, 975), "peak": 965, "description": "PO4 symmetric stretch (ν1)"},
                {"range": (1020, 1080), "peak": 1050, "description": "PO4 antisymmetric stretch (ν3)"},
                {"range": (560, 620), "peak": 590, "description": "PO4 antisymmetric bending (ν4)"},
                {"range": (420, 460), "peak": 440, "description": "PO4 symmetric bending (ν2)"},
                {"range": (250, 270), "peak": 260, "description": "PO4 translation"}
            ],
            "6": [  # Chain Modes - Single Chain Silicates
                {"range": (650, 700), "peak": 665, "description": "Si-O-Si chain stretch"},
                {"range": (980, 1020), "peak": 1000, "description": "Si-O terminal stretch"},
                {"range": (300, 400), "peak": 350, "description": "M-O stretch (metal-oxygen)"},
                {"range": (520, 560), "peak": 540, "description": "Si-O bending"},
                {"range": (180, 220), "peak": 200, "description": "Chain deformation"}
            ],
            "7": [  # Chain Modes - Double Chain Silicates
                {"range": (660, 680), "peak": 670, "description": "Si-O-Si double chain stretch"},
                {"range": (920, 960), "peak": 940, "description": "Si-O terminal stretch"},
                {"range": (300, 350), "peak": 325, "description": "M-O stretch"},
                {"range": (480, 520), "peak": 500, "description": "Si-O bending"},
                {"range": (150, 190), "peak": 170, "description": "Chain deformation"}
            ],
            "8": [  # Ring Modes - Cyclosilicates
                {"range": (620, 660), "peak": 640, "description": "Ring breathing mode"},
                {"range": (500, 540), "peak": 520, "description": "Si-O-Si ring stretch"},
                {"range": (900, 950), "peak": 920, "description": "Si-O terminal stretch"},
                {"range": (350, 390), "peak": 370, "description": "Ring deformation"},
                {"range": (200, 240), "peak": 220, "description": "Lattice mode"}
            ],
            "9": [  # Layer Modes - Sheet Silicates
                {"range": (3600, 3640), "peak": 3620, "description": "OH stretch"},
                {"range": (460, 480), "peak": 470, "description": "Si-O stretch in tetrahedral sheet"},
                {"range": (350, 370), "peak": 360, "description": "Al-OH deformation"},
                {"range": (260, 280), "peak": 270, "description": "Si-O-Al bending"},
                {"range": (100, 120), "peak": 110, "description": "Layer bending mode"}
            ],
            "10": [  # Layer Modes - Non-Silicate Layers
                {"range": (1560, 1580), "peak": 1570, "description": "C=C stretch (graphite)"},
                {"range": (380, 400), "peak": 390, "description": "Mo-S stretch (molybdenite)"},
                {"range": (3650, 3670), "peak": 3660, "description": "OH stretch (brucite)"},
                {"range": (440, 460), "peak": 450, "description": "M-OH bending"},
                {"range": (200, 220), "peak": 210, "description": "Layer interaction"}
            ],
            "11": [  # Metal-Oxygen Modes - Simple Oxides
                {"range": (220, 240), "peak": 230, "description": "Fe-O stretch (hematite)"},
                {"range": (290, 310), "peak": 300, "description": "Fe-O stretch (magnetite)"},
                {"range": (405, 415), "peak": 410, "description": "Fe-O stretch (hematite)"},
                {"range": (660, 680), "peak": 670, "description": "Fe-O stretch (magnetite)"},
                {"range": (540, 560), "peak": 550, "description": "Fe-O bending"}
            ],
            "12": [  # Metal-Oxygen Modes - Complex Oxides
                {"range": (660, 680), "peak": 670, "description": "Spinel A1g mode"},
                {"range": (450, 470), "peak": 460, "description": "Spinel Eg mode"},
                {"range": (310, 330), "peak": 320, "description": "Spinel F2g mode"},
                {"range": (200, 220), "peak": 210, "description": "Spinel lattice mode"},
                {"range": (580, 600), "peak": 590, "description": "Metal-oxygen stretch"}
            ],
            "13": [  # Hydroxide Modes
                {"range": (3100, 3200), "peak": 3150, "description": "OH stretch (strong H-bonding)"},
                {"range": (3400, 3500), "peak": 3450, "description": "OH stretch (weak H-bonding)"},
                {"range": (1000, 1100), "peak": 1050, "description": "OH bending"},
                {"range": (400, 500), "peak": 450, "description": "M-OH stretch"},
                {"range": (200, 300), "peak": 250, "description": "M-OH bending"}
            ],
            "14": [  # Characteristic Vibrational Mode - Organic Groups
                {"range": (1460, 1480), "peak": 1470, "description": "COO symmetric stretch"},
                {"range": (1620, 1640), "peak": 1630, "description": "COO antisymmetric stretch"},
                {"range": (2900, 2950), "peak": 2925, "description": "C-H stretch"},
                {"range": (1350, 1370), "peak": 1360, "description": "C-H bending"},
                {"range": (880, 920), "peak": 900, "description": "C-C stretch"}
            ],
            "15": [  # Mixed Modes - Complex structures
                {"range": (900, 1100), "peak": 1000, "description": "Mixed silicate modes"},
                {"range": (500, 700), "peak": 600, "description": "Mixed framework modes"},
                {"range": (300, 500), "peak": 400, "description": "Mixed M-O modes"},
                {"range": (150, 300), "peak": 200, "description": "Mixed lattice modes"},
                {"range": (1200, 1400), "peak": 1300, "description": "Mixed molecular modes"}
            ]
        }
    
    def classify_mineral(self, chemistry: str, elements: str = "", mineral_name: str = "",
                        wavenumbers: Optional[Union[List, np.ndarray]] = None,
                        intensities: Optional[Union[List, np.ndarray]] = None,
                        detected_peaks: Optional[List[float]] = None) -> Dict:
        """
        Classify a mineral based on its vibrational characteristics using spectral analysis.
        
        Parameters:
        -----------
        chemistry : str
            Chemical formula of the mineral
        elements : str
            Comma-separated list of elements
        mineral_name : str
            Name of the mineral (for additional context)
        wavenumbers : array-like, optional
            Wavenumber array for the spectrum
        intensities : array-like, optional
            Intensity array for the spectrum
        detected_peaks : List[float], optional
            Pre-detected peak positions in cm⁻¹
            
        Returns:
        --------
        Dict
            Enhanced classification result with spectral analysis
        """
        if not chemistry:
            return {
                "best_group_id": "0", 
                "best_group_name": "Unclassified", 
                "confidence": 0.0, 
                "reasoning": "No chemical formula provided",
                "spectral_analysis": None
            }
        
        # Clean and prepare the chemistry string
        clean_chemistry = self._clean_chemistry_formula(chemistry)
        element_list = self._parse_elements(elements) if elements else []
        
        # Extract or detect peaks from spectral data
        peaks_to_use = self._get_or_detect_peaks(wavenumbers, intensities, detected_peaks)
        
        # Perform constrained peak labeling if peaks are available
        spectral_analysis = None
        if peaks_to_use:
            spectral_analysis = self.peak_matcher.constrained_peak_labeling(
                detected_peaks=peaks_to_use,
                group_modes=self.group_expected_modes,
                chemistry=clean_chemistry
            )
        
        # Score each vibrational group using combined chemical + spectral evidence
        group_scores = {}
        for group_id, group_info in self.vibrational_groups.items():
            # Get chemical score
            chemical_score, chemical_reasoning = self._score_vibrational_group(
                clean_chemistry, element_list, mineral_name, group_id
            )
            
            # Get spectral score if spectral analysis is available
            spectral_score = 0.0
            spectral_reasoning = "No spectral data"
            
            if spectral_analysis and group_id in spectral_analysis['group_matches']:
                spectral_score = spectral_analysis['group_matches'][group_id]['constrained_score']
                matches = spectral_analysis['group_matches'][group_id]['matches']
                # Fix: Handle empty matches to avoid array boolean context issues
                if matches:
                    avg_confidence = sum(m['confidence'] for m in matches) / len(matches)
                    spectral_reasoning = f"Matched {len(matches)} peaks with avg confidence {avg_confidence:.3f}"
                else:
                    spectral_reasoning = "No peak matches found"
            
            # Combine chemical and spectral scores
            if spectral_analysis:
                # Weight spectral evidence more heavily when available
                combined_score = 0.3 * chemical_score + 0.7 * spectral_score
                reasoning = f"Chemical: {chemical_reasoning}; Spectral: {spectral_reasoning}"
            else:
                # Fall back to chemical-only scoring
                combined_score = chemical_score
                reasoning = f"Chemical only: {chemical_reasoning} (no spectral data)"
            
            group_scores[group_id] = {
                "score": combined_score,
                "chemical_score": chemical_score,
                "spectral_score": spectral_score,
                "reasoning": reasoning,
                "name": group_info["name"]
            }
        
        # Find the best match
        best_group = max(group_scores.items(), key=lambda x: x[1]["score"])
        best_id, best_data = best_group
        
        # Calculate confidence based on score separation and spectral evidence
        scores = [data["score"] for data in group_scores.values()]
        scores.sort(reverse=True)
        confidence = scores[0]
        
        if len(scores) > 1 and scores[1] > 0:
            # Enhanced confidence calculation considering spectral evidence
            separation = scores[0] - scores[1]
            confidence = min(1.0, scores[0] * (1.0 + separation))
        
        # Add spectral confidence boost if available
        if spectral_analysis and spectral_analysis['best_group'] == best_id:
            confidence = min(1.0, confidence * 1.2)  # 20% boost for spectral confirmation
        
        return {
            "best_group_id": best_id,
            "best_group_name": best_data["name"],
            "confidence": confidence,
            "reasoning": best_data["reasoning"],
            "chemical_score": best_data["chemical_score"],
            "spectral_score": best_data["spectral_score"],
            "all_scores": group_scores,
            "spectral_analysis": spectral_analysis,
            "detected_peaks": peaks_to_use,
            "peak_assignments": spectral_analysis['peak_assignments'] if spectral_analysis else [],
            "unassigned_peaks": spectral_analysis['unassigned_peaks'] if spectral_analysis else []
        }
    
    def _get_or_detect_peaks(self, wavenumbers: Optional[Union[List, np.ndarray]], 
                           intensities: Optional[Union[List, np.ndarray]], 
                           detected_peaks: Optional[List[float]]) -> List[float]:
        """
        Get peaks from provided list or detect them from spectral data.
        """
        # Use provided peaks if available
        if detected_peaks is not None and len(detected_peaks) > 0:
            return list(detected_peaks)
        
        # Detect peaks from spectral data if available
        # Fix array boolean context issue by checking length instead of None comparison
        try:
            if (wavenumbers is not None and intensities is not None and 
                len(wavenumbers) > 0 and len(intensities) > 0):
                wn_array = np.array(wavenumbers)
                int_array = np.array(intensities)
                
                if len(wn_array) > 0 and len(int_array) > 0:
                    # Use scipy.signal.find_peaks for automatic peak detection
                    # Parameters tuned for typical Raman spectra
                    height_threshold = 0.1 * np.max(int_array)
                    prominence_threshold = 0.05 * np.max(int_array)
                    
                    peak_indices, _ = find_peaks(
                        int_array,
                        height=height_threshold,
                        prominence=prominence_threshold,
                        distance=10  # Minimum distance between peaks
                    )
                    
                    # Convert indices to wavenumber positions
                    return wn_array[peak_indices].tolist()
        except (TypeError, ValueError, AttributeError) as e:
            # Handle cases where wavenumbers/intensities are not array-like
            print(f"DEBUG: Error in peak detection: {e}")
            pass
        
        return []
    
    def _clean_chemistry_formula(self, chemistry: str) -> str:
        """Clean and normalize the chemistry formula."""
        if not chemistry:
            return ""
        
        # Remove common prefixes/suffixes that don't affect vibrational classification
        cleaned = chemistry.strip()
        
        # Normalize subscripts and superscripts
        cleaned = re.sub(r'[₀-₉]', lambda m: str(ord(m.group()) - ord('₀')), cleaned)
        cleaned = re.sub(r'[⁰-⁹]', lambda m: str(ord(m.group()) - ord('⁰')), cleaned)
        
        return cleaned
    
    def _parse_elements(self, elements: str) -> List[str]:
        """Parse comma-separated elements list."""
        if not elements:
            return []
        return [elem.strip() for elem in elements.split(',') if elem.strip()]
    
    def _score_vibrational_group(self, chemistry: str, elements: List[str], 
                                mineral_name: str, group_id: str) -> Tuple[float, str]:
        """Score how well a mineral fits a vibrational group."""
        score = 0.0
        reasoning = []
        
        # Get group-specific scoring logic
        if group_id == "1":  # Tetrahedral Framework
            score, reasoning = self._score_tetrahedral_framework(chemistry, elements, mineral_name)
        elif group_id == "2":  # Octahedral Framework
            score, reasoning = self._score_octahedral_framework(chemistry, elements, mineral_name)
        elif group_id == "3":  # Carbonate Groups
            score, reasoning = self._score_carbonate_groups(chemistry, elements, mineral_name)
        elif group_id == "4":  # Sulfate Groups
            score, reasoning = self._score_sulfate_groups(chemistry, elements, mineral_name)
        elif group_id == "5":  # Phosphate Groups
            score, reasoning = self._score_phosphate_groups(chemistry, elements, mineral_name)
        elif group_id == "6":  # Single Chain Silicates
            score, reasoning = self._score_single_chain_silicates(chemistry, elements, mineral_name)
        elif group_id == "7":  # Double Chain Silicates
            score, reasoning = self._score_double_chain_silicates(chemistry, elements, mineral_name)
        elif group_id == "8":  # Ring Silicates
            score, reasoning = self._score_ring_silicates(chemistry, elements, mineral_name)
        elif group_id == "9":  # Sheet Silicates
            score, reasoning = self._score_sheet_silicates(chemistry, elements, mineral_name)
        elif group_id == "10":  # Non-Silicate Layers
            score, reasoning = self._score_nonsilicate_layers(chemistry, elements, mineral_name)
        elif group_id == "11":  # Simple Oxides
            score, reasoning = self._score_simple_oxides(chemistry, elements, mineral_name)
        elif group_id == "12":  # Complex Oxides
            score, reasoning = self._score_complex_oxides(chemistry, elements, mineral_name)
        elif group_id == "13":  # Hydroxides
            score, reasoning = self._score_hydroxides(chemistry, elements, mineral_name)
        elif group_id == "14":  # Organic
            score, reasoning = self._score_organic(chemistry, elements, mineral_name)
        elif group_id == "15":  # Mixed Mode
            score, reasoning = self._score_mixed_mode(chemistry, elements, mineral_name)
        else:
            score, reasoning = 0.0, "Unknown group"
        
        # Ensure reasoning is returned as string, not iterated character by character
        return score, reasoning if isinstance(reasoning, str) else "No reasoning provided"
    
    def _score_tetrahedral_framework(self, chemistry: str, elements: List[str], 
                                   mineral_name: str) -> Tuple[float, str]:
        """Score tetrahedral framework minerals."""
        score = 0.0
        reasoning = []
        
        # Check for framework-forming elements
        framework_elements = {'Si', 'Al', 'P', 'Ge', 'B'}
        framework_present = framework_elements.intersection(set(elements))
        
        if framework_present:
            score += 0.4
            reasoning.append(f"Framework elements present: {', '.join(framework_present)}")
        
        # Check for framework stoichiometry patterns
        framework_patterns = [
            r'SiO_?2', r'AlO_?2', r'Si\d*Al\d*O\d*',
            r'SiO2', r'AlSiO', r'NaAlSi'  # Common framework patterns
        ]
        
        for pattern in framework_patterns:
            if re.search(pattern, chemistry, re.IGNORECASE):
                score += 0.3
                reasoning.append(f"Framework stoichiometry pattern: {pattern}")
                break
        
        # Check mineral name indicators
        framework_names = ['quartz', 'feldspar', 'zeolite', 'cristobalite', 'tridymite']
        name_lower = mineral_name.lower()
        for name_indicator in framework_names:
            if name_indicator in name_lower:
                score += 0.3
                reasoning.append(f"Framework mineral name indicator: {name_indicator}")
                break
        
        return score, "; ".join(reasoning) if reasoning else "No framework indicators"
    
    def _score_carbonate_groups(self, chemistry: str, elements: List[str], 
                              mineral_name: str) -> Tuple[float, str]:
        """Score carbonate group minerals."""
        score = 0.0
        reasoning = []
        
        # Strong indicator: CO3 groups
        carbonate_patterns = [r'CO_?3', r'\(CO_?3\)', r'C\d*O\d*']
        for pattern in carbonate_patterns:
            if re.search(pattern, chemistry, re.IGNORECASE):
                score += 0.6
                reasoning.append(f"Carbonate group detected: {pattern}")
                break
        
        # Element indicators
        if 'C' in elements and 'O' in elements:
            score += 0.2
            reasoning.append("Carbon and oxygen present")
        
        # Common carbonate mineral names
        carbonate_names = ['calcite', 'aragonite', 'dolomite', 'malachite', 'azurite']
        name_lower = mineral_name.lower()
        for name_indicator in carbonate_names:
            if name_indicator in name_lower:
                score += 0.2
                reasoning.append(f"Carbonate mineral name: {name_indicator}")
                break
        
        return score, "; ".join(reasoning) if reasoning else "No carbonate indicators"
    
    # Additional scoring methods would be implemented here for each group...
    # For brevity, I'll implement a few key ones and indicate where others would go
    
    def _score_sulfate_groups(self, chemistry: str, elements: List[str], 
                            mineral_name: str) -> Tuple[float, str]:
        """Score sulfate group minerals."""
        score = 0.0
        reasoning = []
        
        # Strong indicator: SO4 groups
        sulfate_patterns = [r'SO_?4', r'\(SO_?4\)', r'S\d*O\d*']
        for pattern in sulfate_patterns:
            if re.search(pattern, chemistry, re.IGNORECASE):
                score += 0.6
                reasoning.append(f"Sulfate group detected: {pattern}")
                break
        
        # Element indicators
        if 'S' in elements and 'O' in elements:
            score += 0.2
            reasoning.append("Sulfur and oxygen present")
        
        return score, "; ".join(reasoning) if reasoning else "No sulfate indicators"
    
    def _score_phosphate_groups(self, chemistry: str, elements: List[str], 
                              mineral_name: str) -> Tuple[float, str]:
        """Score phosphate group minerals."""
        score = 0.0
        reasoning = []
        
        # Strong indicators: PO4, AsO4, VO4 groups
        phosphate_patterns = [r'PO_?4', r'AsO_?4', r'VO_?4', r'\(PO_?4\)', r'P\d*O\d*']
        for pattern in phosphate_patterns:
            if re.search(pattern, chemistry, re.IGNORECASE):
                score += 0.6
                reasoning.append(f"Phosphate-type group detected: {pattern}")
                break
        
        # Element indicators
        phosphate_elements = {'P', 'As', 'V'}
        phosphate_present = phosphate_elements.intersection(set(elements))
        if phosphate_present and 'O' in elements:
            score += 0.2
            reasoning.append(f"Phosphate-type elements with oxygen: {', '.join(phosphate_present)}")
        
        return score, "; ".join(reasoning) if reasoning else "No phosphate indicators"
    
    # Placeholder methods for other groups - these would be fully implemented
    def _score_octahedral_framework(self, chemistry: str, elements: List[str], mineral_name: str) -> Tuple[float, str]:
        """Score octahedral framework minerals (rutile, anatase, corundum, spinels)."""
        score = 0.0
        reasoning = []
        
        # Essential: Must have oxygen for octahedral coordination
        if 'O' not in elements:
            return 0.0, "No oxygen - cannot have octahedral framework"
        
        score += 0.3
        reasoning.append("Oxygen present - essential for octahedral coordination")
        
        # High-coordination metals that prefer octahedral sites
        octahedral_metals = {
            'Ti': 0.5,   # Rutile, anatase, brookite (TiO2) - classic octahedral
            'Al': 0.4,   # Corundum (Al2O3) - octahedral Al
            'Fe': 0.4,   # Spinel structures, perovskites
            'Mg': 0.4,   # Spinel structures, periclase-type
            'Cr': 0.4,   # Chromium in octahedral sites
            'V': 0.3,    # Vanadium oxides with octahedral coordination
            'Mn': 0.3,   # Manganese in octahedral sites
            'Ni': 0.3,   # Nickel in octahedral coordination
            'Co': 0.3,   # Cobalt in octahedral sites
            'Zn': 0.2,   # Zinc in some octahedral structures
            'Cu': 0.2,   # Copper in some octahedral sites
            'Zr': 0.3,   # Zirconia and related structures
            'Nb': 0.3,   # Niobium oxides
            'Ta': 0.3,   # Tantalum oxides
            'W': 0.2,    # Tungsten oxides
            'Mo': 0.2,   # Molybdenum oxides
            'Sn': 0.2    # Tin in octahedral coordination
        }
        
        found_octahedral_metals = []
        for metal, weight in octahedral_metals.items():
            if metal in elements:
                score += weight
                found_octahedral_metals.append(metal)
        
        if found_octahedral_metals:
            reasoning.append(f"Octahedral metals present: {', '.join(found_octahedral_metals)}")
        else:
            score = max(0.0, score - 0.3)
            reasoning.append("No typical octahedral framework metals")
        
        # Octahedral framework formula patterns
        octahedral_patterns = [
            # Rutile structure family (MO2)
            (r'^Ti_?O_?2$', 0.6, "Rutile structure (TiO2)"),
            (r'^Sn_?O_?2$', 0.5, "Cassiterite structure (SnO2)"),
            (r'^Pb_?O_?2$', 0.4, "Plattnerite structure (PbO2)"),
            (r'^Mn_?O_?2$', 0.5, "Pyrolusite structure (MnO2)"),
            
            # Corundum structure (M2O3)
            (r'^Al_?2_?O_?3$', 0.6, "Corundum structure (Al2O3)"),
            (r'^Cr_?2_?O_?3$', 0.5, "Eskolaite structure (Cr2O3)"),
            (r'^Fe_?2_?O_?3$', 0.5, "Hematite structure (Fe2O3)"),
            (r'^V_?2_?O_?3$', 0.4, "Karelianite structure (V2O3)"),
            
            # Spinel structures (AB2O4)
            (r'^[A-Z][a-z]?[A-Z][a-z]?_?2_?O_?4$', 0.5, "Spinel structure (AB2O4)"),
            (r'^Mg_?Al_?2_?O_?4$', 0.6, "Spinel structure (MgAl2O4)"),
            (r'^Fe_?[A-Z][a-z]?_?2_?O_?4$', 0.5, "Iron-bearing spinel"),
            (r'^[A-Z][a-z]?_?Fe_?2_?O_?4$', 0.5, "Ferrite spinel structure"),
            
            # Perovskite structures (ABO3)
            (r'^[A-Z][a-z]?[A-Z][a-z]?O_?3$', 0.4, "Perovskite structure (ABO3)"),
            (r'^Ca_?Ti_?O_?3$', 0.5, "Perovskite (CaTiO3)"),
            
            # Rock salt structures (MO)
            (r'^[A-Z][a-z]?O$', 0.3, "Rock salt structure (MO)"),
            (r'^MgO$', 0.4, "Periclase structure (MgO)"),
            (r'^NiO$', 0.4, "Bunsenite structure (NiO)"),
            (r'^CoO$', 0.4, "Cobalt oxide structure"),
            
            # Fluorite/antifluorite structures (MO2, M2O)
            (r'^Zr_?O_?2$', 0.5, "Baddeleyite structure (ZrO2)"),
            (r'^Hf_?O_?2$', 0.4, "Hafnia structure (HfO2)"),
            
            # General octahedral patterns
            (r'^[A-Z][a-z]?_?[0-9]*_?O_?[0-9]*$', 0.2, "Simple metal oxide (potential octahedral)")
        ]
        
        for pattern, weight, description in octahedral_patterns:
            if re.search(pattern, chemistry.replace(' ', ''), re.IGNORECASE):
                score += weight
                reasoning.append(f"Octahedral framework pattern: {description}")
                break  # Only count the best match
        
        # Known octahedral framework mineral names
        octahedral_names = [
            # Rutile structure group
            ('rutile', 0.7, "Rutile (TiO2) - rutile structure"),
            ('anatase', 0.7, "Anatase (TiO2) - anatase structure"),
            ('brookite', 0.7, "Brookite (TiO2) - brookite structure"),
            ('cassiterite', 0.6, "Cassiterite (SnO2) - rutile structure"),
            ('pyrolusite', 0.6, "Pyrolusite (MnO2) - rutile structure"),
            ('plattnerite', 0.5, "Plattnerite (PbO2) - rutile structure"),
            
            # Corundum structure group
            ('corundum', 0.7, "Corundum (Al2O3) - corundum structure"),
            ('hematite', 0.6, "Hematite (Fe2O3) - corundum structure"),
            ('eskolaite', 0.6, "Eskolaite (Cr2O3) - corundum structure"),
            ('karelianite', 0.5, "Karelianite (V2O3) - corundum structure"),
            ('ruby', 0.6, "Ruby (Cr:Al2O3) - corundum structure"),
            ('sapphire', 0.6, "Sapphire (Al2O3) - corundum structure"),
            
            # Spinel structure group
            ('spinel', 0.7, "Spinel (MgAl2O4) - spinel structure"),
            ('magnetite', 0.6, "Magnetite (Fe3O4) - inverse spinel"),
            ('chromite', 0.6, "Chromite (FeCr2O4) - spinel structure"),
            ('franklinite', 0.6, "Franklinite (ZnFe2O4) - spinel structure"),
            ('gahnite', 0.6, "Gahnite (ZnAl2O4) - spinel structure"),
            ('hercynite', 0.6, "Hercynite (FeAl2O4) - spinel structure"),
            ('jacobsite', 0.5, "Jacobsite (MnFe2O4) - spinel structure"),
            ('ulvospinel', 0.5, "Ulvöspinel (Fe2TiO4) - spinel structure"),
            
            # Perovskite structure group
            ('perovskite', 0.6, "Perovskite (CaTiO3) - perovskite structure"),
            ('loparite', 0.5, "Loparite (Na,Ce,Ca)(Ti,Nb)O3 - perovskite"),
            ('lueshite', 0.5, "Lueshite (NaNbO3) - perovskite structure"),
            
            # Rock salt structure group
            ('periclase', 0.6, "Periclase (MgO) - rock salt structure"),
            ('bunsenite', 0.6, "Bunsenite (NiO) - rock salt structure"),
            ('monteponite', 0.5, "Monteponite (CdO) - rock salt structure"),
            ('lime', 0.4, "Lime (CaO) - rock salt structure"),
            
            # Fluorite/baddeleyite structures
            ('baddeleyite', 0.6, "Baddeleyite (ZrO2) - baddeleyite structure"),
            ('hafnia', 0.5, "Hafnia (HfO2) - fluorite structure"),
            
            # General terms
            ('oxide', 0.1, "Oxide mineral (potential octahedral)")
        ]
        
        name_lower = mineral_name.lower()
        for name_pattern, weight, description in octahedral_names:
            if name_pattern in name_lower:
                score += weight
                reasoning.append(f"Known octahedral framework mineral: {description}")
                break  # Only count the first match
        
        # Structural analysis bonuses
        # Multiple metals suggest spinel or perovskite structures
        if len([elem for elem in elements if elem in octahedral_metals]) >= 2:
            score += 0.2
            reasoning.append("Multiple octahedral metals - suggests spinel/perovskite structure")
        
        # Ti-O system gets extra bonus (rutile/anatase are archetypal octahedral structures)
        if 'Ti' in elements and len(elements) <= 3:
            score += 0.3
            reasoning.append("Ti-O system - classic octahedral framework")
        
        # Al2O3 system gets bonus (corundum is archetypal)
        if 'Al' in elements and 'O' in elements and len(elements) <= 3:
            score += 0.2
            reasoning.append("Al-O system - corundum-type octahedral framework")
        
        # Penalty for elements that suggest other structural types
        incompatible_elements = {
            'Si': "silicate frameworks (tetrahedral)",
            'P': "phosphate tetrahedra",
            'S': "sulfate tetrahedra",
            'C': "carbonate groups",
            'B': "borate groups"
        }
        
        for elem, reason in incompatible_elements.items():
            if elem in elements:
                penalty = 0.15
                # Special case: Si with simple chemistry might still be octahedral (stishovite)
                if elem == 'Si' and len(elements) <= 2:
                    penalty = 0.05  # Reduced penalty for simple Si-O systems
                score = max(0.0, score - penalty)
                reasoning.append(f"Penalty for {elem} - suggests {reason}")
        
        # Penalty for too many elements (suggests complex structures)
        if len(elements) > 5:
            penalty = (len(elements) - 5) * 0.05
            score = max(0.0, score - penalty)
            reasoning.append(f"Penalty for complex chemistry ({len(elements)} elements)")
        
        # Bonus for perfect octahedral stoichiometry
        # This is simplified - a full implementation would parse formulas exactly
        if 'Ti' in elements and 'O' in elements and len(elements) == 2:
            score += 0.2
            reasoning.append("Perfect TiO2 stoichiometry - archetypal octahedral")
        elif 'Al' in elements and 'O' in elements and len(elements) == 2:
            score += 0.2
            reasoning.append("Perfect Al2O3 stoichiometry - corundum structure")
        
        return min(1.0, score), "; ".join(reasoning) if reasoning else "No octahedral framework indicators"
    
    def _score_single_chain_silicates(self, chemistry: str, elements: List[str], mineral_name: str) -> Tuple[float, str]:
        """Score single chain silicate minerals (pyroxenes, wollastonite, rhodonite)."""
        score = 0.0
        reasoning = []
        
        # Essential elements for silicates
        if 'Si' not in elements or 'O' not in elements:
            return 0.0, "Missing Si or O - cannot be silicate"
        
        score += 0.4
        reasoning.append("Silicon and oxygen present - essential for silicates")
        
        # Chain-forming cations that prefer single chain structures
        single_chain_cations = {
            'Mg': 0.3,   # Enstatite, diopside
            'Fe': 0.3,   # Ferrosilite, hedenbergite, aegirine
            'Ca': 0.3,   # Diopside, wollastonite, hedenbergite
            'Mn': 0.2,   # Rhodonite, johannsenite
            'Li': 0.2,   # Spodumene
            'Na': 0.2,   # Jadeite, aegirine
            'Al': 0.2,   # Some pyroxenes (jadeite, aegirine)
            'Ti': 0.1,   # Titanium-bearing pyroxenes
            'Cr': 0.1,   # Chrome-bearing pyroxenes
            'Zn': 0.1,   # Zinc-bearing pyroxenes
            'Ni': 0.1    # Nickel-bearing pyroxenes
        }
        
        found_chain_cations = []
        for cation, weight in single_chain_cations.items():
            if cation in elements:
                score += weight
                found_chain_cations.append(cation)
        
        if found_chain_cations:
            reasoning.append(f"Single chain cations present: {', '.join(found_chain_cations)}")
        else:
            score = max(0.0, score - 0.2)
            reasoning.append("No typical single chain silicate cations")
        
        # Single chain silicate formula patterns
        single_chain_patterns = [
            # Pyroxene patterns (XYSi2O6 general formula)
            (r'[A-Z][a-z]?[A-Z][a-z]?Si_?2_?O_?6', 0.6, "Pyroxene formula (XYSi2O6)"),
            (r'Mg_?Si_?O_?3', 0.6, "Enstatite formula (MgSiO3)"),
            (r'Fe_?Si_?O_?3', 0.6, "Ferrosilite formula (FeSiO3)"),
            (r'Ca_?Mg_?Si_?2_?O_?6', 0.6, "Diopside formula (CaMgSi2O6)"),
            (r'Ca_?Fe_?Si_?2_?O_?6', 0.6, "Hedenbergite formula (CaFeSi2O6)"),
            (r'Na_?Al_?Si_?2_?O_?6', 0.6, "Jadeite formula (NaAlSi2O6)"),
            (r'Na_?Fe_?Si_?2_?O_?6', 0.5, "Aegirine formula (NaFeSi2O6)"),
            (r'Li_?Al_?Si_?2_?O_?6', 0.6, "Spodumene formula (LiAlSi2O6)"),
            
            # Wollastonite pattern
            (r'Ca_?Si_?O_?3', 0.5, "Wollastonite formula (CaSiO3)"),
            (r'Ca_?3_?Si_?3_?O_?9', 0.5, "Wollastonite formula (Ca3Si3O9)"),
            
            # Rhodonite pattern
            (r'Mn_?Si_?O_?3', 0.5, "Rhodonite formula (MnSiO3)"),
            (r'Mn_?5_?Si_?5_?O_?15', 0.5, "Rhodonite formula (Mn5Si5O15)"),
            
            # Bustamite pattern
            (r'Ca_?Mn_?Si_?2_?O_?6', 0.4, "Bustamite formula (CaMnSi2O6)"),
            
            # General single chain patterns
            (r'[A-Z][a-z]?_?[0-9]*_?Si_?O_?3', 0.3, "Simple pyroxene pattern (MSiO3)"),
            (r'[A-Z][a-z]?_?[0-9]*_?[A-Z][a-z]?_?[0-9]*_?Si_?2_?O_?6', 0.4, "Complex pyroxene pattern"),
        ]
        
        for pattern, weight, description in single_chain_patterns:
            if re.search(pattern, chemistry.replace(' ', ''), re.IGNORECASE):
                score += weight
                reasoning.append(f"Single chain pattern: {description}")
                break  # Only count the best match
        
        # Known single chain silicate mineral names
        single_chain_names = [
            # Orthopyroxenes
            ('enstatite', 0.7, "Enstatite (MgSiO3) - orthopyroxene"),
            ('ferrosilite', 0.7, "Ferrosilite (FeSiO3) - orthopyroxene"),
            ('bronzite', 0.6, "Bronzite (Mg,Fe)SiO3 - orthopyroxene"),
            ('hypersthene', 0.6, "Hypersthene (Mg,Fe)SiO3 - orthopyroxene"),
            
            # Clinopyroxenes
            ('diopside', 0.7, "Diopside (CaMgSi2O6) - clinopyroxene"),
            ('hedenbergite', 0.7, "Hedenbergite (CaFeSi2O6) - clinopyroxene"),
            ('augite', 0.7, "Augite (Ca,Mg,Fe,Al)(Si,Al)2O6 - clinopyroxene"),
            ('jadeite', 0.7, "Jadeite (NaAlSi2O6) - clinopyroxene"),
            ('aegirine', 0.7, "Aegirine (NaFeSi2O6) - clinopyroxene"),
            ('spodumene', 0.7, "Spodumene (LiAlSi2O6) - clinopyroxene"),
            ('omphacite', 0.6, "Omphacite (Ca,Na)(Mg,Fe,Al)Si2O6 - clinopyroxene"),
            ('johannsenite', 0.6, "Johannsenite (CaMnSi2O6) - clinopyroxene"),
            ('petedunnite', 0.5, "Petedunnite (CaZnSi2O6) - clinopyroxene"),
            
            # Pyroxenoids (single chain, but slightly different structure)
            ('wollastonite', 0.7, "Wollastonite (CaSiO3) - pyroxenoid"),
            ('rhodonite', 0.7, "Rhodonite (MnSiO3) - pyroxenoid"),
            ('bustamite', 0.6, "Bustamite (CaMnSi2O6) - pyroxenoid"),
            ('pectolite', 0.6, "Pectolite (NaCa2Si3O8OH) - pyroxenoid"),
            ('babingtonite', 0.5, "Babingtonite (Ca2Fe2+Fe3+Si5O14OH) - pyroxenoid"),
            
            # Group terms
            ('pyroxene', 0.5, "Pyroxene group mineral"),
            ('pyroxenoid', 0.4, "Pyroxenoid group mineral")
        ]
        
        name_lower = mineral_name.lower()
        for name_pattern, weight, description in single_chain_names:
            if name_pattern in name_lower:
                score += weight
                reasoning.append(f"Known single chain silicate: {description}")
                break  # Only count the first match
        
        # Structural chemistry analysis
        # Look for characteristic Si:O ratios
        # Single chains typically have Si:O ratios around 1:3 or 2:6
        if 'Si' in elements and 'O' in elements:
            # Check for pyroxene-like Si:O ratios
            pyroxene_ratios = [
                r'Si_?O_?3',      # 1:3 ratio (simple pyroxenes)
                r'Si_?2_?O_?6',   # 2:6 ratio (complex pyroxenes)
                r'Si_?3_?O_?9',   # 3:9 ratio (wollastonite)
                r'Si_?5_?O_?15'   # 5:15 ratio (rhodonite)
            ]
            
            for ratio_pattern in pyroxene_ratios:
                if re.search(ratio_pattern, chemistry, re.IGNORECASE):
                    score += 0.2
                    reasoning.append("Si:O ratio consistent with single chain silicates")
                    break
        
        # Bonus for classic pyroxene chemistry
        # Look for combinations that strongly suggest pyroxenes
        if 'Ca' in elements and 'Mg' in elements and 'Si' in elements:
            score += 0.3
            reasoning.append("Ca-Mg-Si system - classic clinopyroxene chemistry")
        
        if 'Mg' in elements and 'Fe' in elements and 'Si' in elements and 'Ca' not in elements:
            score += 0.2
            reasoning.append("Mg-Fe-Si system - orthopyroxene chemistry")
        
        if 'Na' in elements and 'Al' in elements and 'Si' in elements:
            score += 0.2
            reasoning.append("Na-Al-Si system - jadeite-type pyroxene")
        
        if 'Li' in elements and 'Al' in elements and 'Si' in elements:
            score += 0.3
            reasoning.append("Li-Al-Si system - spodumene chemistry")
        
        if 'Mn' in elements and 'Si' in elements and 'Ca' not in elements:
            score += 0.2
            reasoning.append("Mn-Si system - rhodonite-type chemistry")
        
        # Penalty for elements that suggest other silicate types
        incompatible_elements = {
            'K': "sheet silicates (micas)",
            'B': "borosilicates",
            'Be': "beryllosilicates"
        }
        
        for elem, reason in incompatible_elements.items():
            if elem in elements:
                score = max(0.0, score - 0.1)
                reasoning.append(f"Minor penalty for {elem} - more typical of {reason}")
        
        # Penalty for molecular anion groups (suggests other mineral classes)
        molecular_anions = ['S', 'P', 'C']  # Sulfates, phosphates, carbonates
        for anion in molecular_anions:
            if anion in elements:
                score = max(0.0, score - 0.15)
                reasoning.append(f"Penalty for {anion} - suggests non-silicate mineral class")
        
        # Penalty for too many elements (suggests complex structures like amphiboles)
        if len(elements) > 6:
            penalty = (len(elements) - 6) * 0.05
            score = max(0.0, score - penalty)
            reasoning.append(f"Penalty for complex chemistry ({len(elements)} elements) - suggests amphiboles or other complex silicates")
        
        return min(1.0, score), "; ".join(reasoning) if reasoning else "No single chain silicate indicators"
    
    def _score_double_chain_silicates(self, chemistry: str, elements: List[str], mineral_name: str) -> Tuple[float, str]:
        """Score double chain silicate minerals (amphiboles: hornblende, actinolite, tremolite)."""
        score = 0.0
        reasoning = []
        
        # Essential elements for silicates
        if 'Si' not in elements or 'O' not in elements:
            return 0.0, "Missing Si or O - cannot be silicate"
        
        score += 0.3
        reasoning.append("Silicon and oxygen present - essential for silicates")
        
        # Amphiboles typically contain OH groups
        if re.search(r'OH', chemistry, re.IGNORECASE) or 'H' in elements:
            score += 0.4
            reasoning.append("OH groups present - characteristic of amphiboles")
        else:
            score = max(0.0, score - 0.2)
            reasoning.append("No OH groups detected - unusual for amphiboles")
        
        # Double chain cations (amphiboles have complex multi-site chemistry)
        # A-site cations (large, 10-12 coordination)
        a_site_cations = {'Na': 0.2, 'K': 0.2, 'Ca': 0.2}
        
        # B-site cations (8-9 coordination)
        b_site_cations = {'Ca': 0.2, 'Na': 0.1, 'Mn': 0.15, 'Fe': 0.2, 'Mg': 0.2}
        
        # C-site cations (octahedral, 6 coordination)
        c_site_cations = {'Mg': 0.3, 'Fe': 0.3, 'Al': 0.2, 'Ti': 0.1, 'Mn': 0.1, 'Cr': 0.1, 'V': 0.05}
        
        # T-site cations (tetrahedral, 4 coordination)
        t_site_cations = {'Si': 0.4, 'Al': 0.1}  # Si already counted above
        
        found_amphibole_cations = []
        
        # Score A-site cations
        for cation, weight in a_site_cations.items():
            if cation in elements:
                score += weight
                found_amphibole_cations.append(f"{cation}(A-site)")
        
        # Score B-site cations  
        for cation, weight in b_site_cations.items():
            if cation in elements:
                score += weight
                if f"{cation}(A-site)" not in [x.split('(')[0] for x in found_amphibole_cations]:
                    found_amphibole_cations.append(f"{cation}(B-site)")
        
        # Score C-site cations
        for cation, weight in c_site_cations.items():
            if cation in elements:
                score += weight
                if cation not in [x.split('(')[0] for x in found_amphibole_cations]:
                    found_amphibole_cations.append(f"{cation}(C-site)")
        
        if found_amphibole_cations:
            reasoning.append(f"Amphibole cations present: {', '.join(found_amphibole_cations)}")
        
        # Complexity bonus - amphiboles typically have many elements (4-8 elements common)
        if 4 <= len(elements) <= 8:
            score += 0.3
            reasoning.append(f"Complex chemistry ({len(elements)} elements) - typical of amphiboles")
        elif len(elements) > 8:
            score += 0.1
            reasoning.append(f"Very complex chemistry ({len(elements)} elements) - possible amphibole")
        else:
            score = max(0.0, score - 0.2)
            reasoning.append(f"Simple chemistry ({len(elements)} elements) - unusual for amphiboles")
        
        # Double chain silicate formula patterns
        double_chain_patterns = [
            # Classic amphibole patterns (general formula A0-1B2C5T8O22(OH)2)
            (r'Ca_?2_?Mg_?5_?Si_?8_?O_?22_?.*OH', 0.7, "Tremolite formula (Ca2Mg5Si8O22(OH)2)"),
            (r'Ca_?2_?.*Fe.*_?5_?Si_?8_?O_?22_?.*OH', 0.7, "Actinolite formula (Ca2(Mg,Fe)5Si8O22(OH)2)"),
            (r'Na_?2_?.*Al.*Si_?8_?O_?22_?.*OH', 0.6, "Glaucophane-type formula"),
            (r'.*Mg.*Fe.*Si_?8_?O_?22_?.*OH', 0.6, "Anthophyllite-type formula"),
            (r'K.*Ca.*.*Si_?8_?O_?22_?.*OH', 0.6, "Potassic amphibole formula"),
            
            # General amphibole patterns
            (r'.*Si_?8_?O_?22_?.*OH', 0.5, "General amphibole formula (T8O22(OH)2)"),
            (r'.*Si_?7_?.*O_?22_?.*OH', 0.4, "Al-bearing amphibole formula"),
            (r'.*Si_?6_?.*O_?22_?.*OH', 0.4, "High-Al amphibole formula"),
            
            # Simplified patterns
            (r'Ca.*Mg.*Si.*O.*OH', 0.3, "Ca-Mg-Si-OH system (amphibole-like)"),
            (r'Na.*.*Si.*O.*OH', 0.3, "Na-bearing silicate with OH"),
            (r'.*Fe.*Mg.*Si.*O.*OH', 0.3, "Fe-Mg-Si-OH system (amphibole-like)")
        ]
        
        for pattern, weight, description in double_chain_patterns:
            if re.search(pattern, chemistry.replace(' ', ''), re.IGNORECASE):
                score += weight
                reasoning.append(f"Double chain pattern: {description}")
                break  # Only count the best match
        
        # Known double chain silicate mineral names
        double_chain_names = [
            # Calcic amphiboles
            ('tremolite', 0.8, "Tremolite (Ca2Mg5Si8O22(OH)2) - calcic amphibole"),
            ('actinolite', 0.8, "Actinolite (Ca2(Mg,Fe)5Si8O22(OH)2) - calcic amphibole"),
            ('hornblende', 0.8, "Hornblende - complex calcic amphibole"),
            ('edenite', 0.7, "Edenite (NaCa2Mg5Si7AlO22(OH)2) - calcic amphibole"),
            ('pargasite', 0.7, "Pargasite (NaCa2(Mg,Fe)4AlSi6Al2O22(OH)2) - calcic amphibole"),
            ('tschermakite', 0.7, "Tschermakite (Ca2(Mg,Fe)3Al2Si6Al2O22(OH)2) - calcic amphibole"),
            
            # Sodic amphiboles
            ('glaucophane', 0.8, "Glaucophane (Na2(Mg,Fe)3Al2Si8O22(OH)2) - sodic amphibole"),
            ('riebeckite', 0.8, "Riebeckite (Na2(Fe,Mg)3Fe2Si8O22(OH)2) - sodic amphibole"),
            ('arfvedsonite', 0.7, "Arfvedsonite (NaNa2(Fe,Mg)4FeSi8O22(OH)2) - sodic amphibole"),
            ('eckermannite', 0.7, "Eckermannite (NaNa2(Mg,Fe)4AlSi8O22(OH)2) - sodic amphibole"),
            
            # Orthoamphiboles (Mg-Fe series)
            ('anthophyllite', 0.8, "Anthophyllite ((Mg,Fe)7Si8O22(OH)2) - orthoamphibole"),
            ('gedrite', 0.7, "Gedrite ((Mg,Fe)5Al2Si6Al2O22(OH)2) - orthoamphibole"),
            ('holmquistite', 0.6, "Holmquistite (Li2(Mg,Fe)3Al2Si8O22(OH)2) - orthoamphibole"),
            
            # Other amphiboles
            ('richterite', 0.7, "Richterite (Na(CaNa)(Mg,Fe)5Si8O22(OH)2) - alkali amphibole"),
            ('winchite', 0.6, "Winchite (CaNa(Mg,Fe)4AlSi8O22(OH)2)"),
            ('barroisite', 0.6, "Barroisite (CaNa(Mg,Fe)3Al2Si7AlO22(OH)2)"),
            ('katophorite', 0.6, "Katophorite (Na2Ca(Mg,Fe)4AlSi7AlO22(OH)2)"),
            ('kaersutite', 0.6, "Kaersutite (NaCa2(Mg,Fe)4TiSi6Al2O23(OH)2)"),
            
            # Group terms
            ('amphibole', 0.5, "Amphibole group mineral"),
            ('hornblend', 0.4, "Hornblende-type amphibole")  # Partial match for "hornblende"
        ]
        
        name_lower = mineral_name.lower()
        for name_pattern, weight, description in double_chain_names:
            if name_pattern in name_lower:
                score += weight
                reasoning.append(f"Known double chain silicate: {description}")
                break  # Only count the first match
        
        # Structural chemistry analysis
        # Amphiboles have characteristic element combinations
        
        # Classic tremolite-actinolite series (Ca-Mg-Fe-Si-OH)
        if all(elem in elements for elem in ['Ca', 'Mg', 'Si']) and 'H' in elements:
            score += 0.4
            reasoning.append("Ca-Mg-Si-OH system - tremolite-actinolite series")
        
        # Hornblende-type complex chemistry (Ca + Na + Mg/Fe + Al + Si + OH)
        if all(elem in elements for elem in ['Ca', 'Na', 'Al', 'Si']) and 'H' in elements:
            score += 0.4
            reasoning.append("Ca-Na-Al-Si-OH system - hornblende-type amphibole")
        
        # Glaucophane-type (Na + Al + Si + OH, typically with Fe/Mg)
        if all(elem in elements for elem in ['Na', 'Al', 'Si']) and 'H' in elements:
            score += 0.3
            reasoning.append("Na-Al-Si-OH system - glaucophane-type amphibole")
        
        # Anthophyllite-type (Mg + Fe + Si + OH, no Ca or Na)
        if all(elem in elements for elem in ['Mg', 'Fe', 'Si']) and 'H' in elements and 'Ca' not in elements and 'Na' not in elements:
            score += 0.3
            reasoning.append("Mg-Fe-Si-OH system without Ca/Na - anthophyllite-type")
        
        # Penalty for elements that are rare in amphiboles
        incompatible_elements = {
            'Li': "lithium minerals (but holmquistite exists)",
            'B': "borosilicates", 
            'Be': "beryllosilicates",
            'Zr': "zirconosilicates",
            'P': "phosphates",
            'S': "sulfates/sulfides",
            'C': "carbonates"
        }
        
        for elem, reason in incompatible_elements.items():
            if elem in elements:
                penalty = 0.1 if elem == 'Li' else 0.15  # Li is less penalized (holmquistite)
                score = max(0.0, score - penalty)
                reasoning.append(f"Minor penalty for {elem} - more typical of {reason}")
        
        # Penalty if no Ca, Na, or K (amphiboles almost always have large cations)
        if not any(elem in elements for elem in ['Ca', 'Na', 'K']):
            score = max(0.0, score - 0.2)
            reasoning.append("No large cations (Ca/Na/K) - unusual for amphiboles")
        
        # Penalty for too simple chemistry (amphiboles are complex)
        if len(elements) < 4:
            penalty = (4 - len(elements)) * 0.1
            score = max(0.0, score - penalty)
            reasoning.append(f"Chemistry too simple ({len(elements)} elements) - amphiboles are typically complex")
        
        # Bonus for perfect amphibole stoichiometry indicators
        # Look for Si8, O22, etc. patterns
        stoich_patterns = [
            r'Si_?8',     # 8 Si atoms typical
            r'O_?22',     # 22 oxygen atoms typical
            r'OH_?2'      # 2 OH groups typical
        ]
        
        stoich_matches = 0
        for pattern in stoich_patterns:
            if re.search(pattern, chemistry, re.IGNORECASE):
                stoich_matches += 1
        
        if stoich_matches >= 2:
            score += 0.3
            reasoning.append("Perfect amphibole stoichiometry (Si8O22(OH)2 pattern)")
        elif stoich_matches == 1:
            score += 0.1
            reasoning.append("Partial amphibole stoichiometry match")
        
        return min(1.0, score), "; ".join(reasoning) if reasoning else "No double chain silicate indicators"
    
    def _score_ring_silicates(self, chemistry: str, elements: List[str], mineral_name: str) -> Tuple[float, str]:
        """Score ring silicate minerals (tourmaline, beryl, cordierite, benitoite)."""
        score = 0.0
        reasoning = []
        
        # Essential elements for silicates
        if 'Si' not in elements or 'O' not in elements:
            return 0.0, "Missing Si or O - cannot be silicate"
        
        score += 0.3
        reasoning.append("Silicon and oxygen present - essential for silicates")
        
        # Ring silicates often contain specific ring-forming elements
        ring_forming_elements = {
            'B': 0.5,    # Tourmaline group (essential for most ring silicates)
            'Be': 0.6,   # Beryl group (Be3Al2Si6O18)
            'Al': 0.3,   # Common in most ring silicates
            'Mg': 0.2,   # Cordierite, some tourmalines
            'Fe': 0.2,   # Iron-bearing tourmalines, cordierite
            'Mn': 0.15,  # Manganese tourmalines
            'Li': 0.15,  # Lithium tourmalines
            'Na': 0.2,   # Sodium-bearing ring silicates
            'Ca': 0.2,   # Calcium-bearing ring silicates
            'Ti': 0.1,   # Benitoite, some tourmalines
            'Cr': 0.1,   # Chrome tourmalines
            'V': 0.1,    # Vanadium-bearing ring silicates
            'Zn': 0.1    # Zinc-bearing tourmalines
        }
        
        found_ring_elements = []
        for element, weight in ring_forming_elements.items():
            if element in elements:
                score += weight
                found_ring_elements.append(element)
        
        if found_ring_elements:
            reasoning.append(f"Ring-forming elements present: {', '.join(found_ring_elements)}")
        
        # Special bonus for boron (tourmaline group is the largest ring silicate family)
        if 'B' in elements:
            score += 0.3
            reasoning.append("Boron present - strong indicator of tourmaline group")
        
        # Special bonus for beryllium (beryl group)
        if 'Be' in elements:
            score += 0.4
            reasoning.append("Beryllium present - strong indicator of beryl group")
        
        # Ring silicate formula patterns
        ring_silicate_patterns = [
            # Tourmaline group patterns
            (r'.*B.*Si_?6_?O_?18', 0.7, "Tourmaline group formula (BSi6O18 framework)"),
            (r'Na.*B.*Al.*Si.*O.*OH', 0.6, "Alkali tourmaline formula"),
            (r'Ca.*B.*Al.*Si.*O.*OH', 0.6, "Calcic tourmaline formula"),
            (r'.*B.*Al.*Si.*O.*OH', 0.5, "General tourmaline formula"),
            (r'Li.*B.*Al.*Si.*O.*OH', 0.6, "Lithium tourmaline formula"),
            (r'Fe.*B.*Al.*Si.*O.*OH', 0.6, "Iron tourmaline formula"),
            
            # Beryl group patterns
            (r'Be_?3_?Al_?2_?Si_?6_?O_?18', 0.8, "Beryl formula (Be3Al2Si6O18)"),
            (r'Be.*Al.*Si_?6_?O_?18', 0.7, "Beryl group formula"),
            
            # Cordierite patterns
            (r'Mg_?2_?Al_?4_?Si_?5_?O_?18', 0.7, "Cordierite formula (Mg2Al4Si5O18)"),
            (r'.*Al_?4_?Si_?5_?O_?18', 0.5, "Cordierite-type formula"),
            
            # Benitoite patterns
            (r'Ba_?Ti_?Si_?3_?O_?9', 0.7, "Benitoite formula (BaTiSi3O9)"),
            (r'Ba.*Ti.*Si.*O', 0.5, "Barium titanium silicate"),
            
            # Milarite group patterns
            (r'K.*Ca.*Be.*Al.*Si.*O', 0.5, "Milarite group pattern"),
            
            # General ring patterns
            (r'.*Si_?6_?O_?18', 0.4, "Six-membered silicate ring (Si6O18)"),
            (r'.*Si_?3_?O_?9', 0.3, "Three-membered silicate ring (Si3O9)"),
            (r'.*Si_?4_?O_?12', 0.3, "Four-membered silicate ring (Si4O12)"),
            (r'.*Si_?5_?O_?15', 0.3, "Five-membered silicate ring (Si5O15)")
        ]
        
        for pattern, weight, description in ring_silicate_patterns:
            if re.search(pattern, chemistry.replace(' ', ''), re.IGNORECASE):
                score += weight
                reasoning.append(f"Ring silicate pattern: {description}")
                break  # Only count the best match
        
        # Known ring silicate mineral names
        ring_silicate_names = [
            # Tourmaline group
            ('tourmaline', 0.8, "Tourmaline group - ring silicate"),
            ('elbaite', 0.8, "Elbaite (Na(Li,Al)3Al6(BO3)3Si6O18(OH)4) - tourmaline"),
            ('schorl', 0.8, "Schorl (NaFe3Al6(BO3)3Si6O18(OH)4) - tourmaline"),
            ('dravite', 0.8, "Dravite (NaMg3Al6(BO3)3Si6O18(OH)4) - tourmaline"),
            ('uvite', 0.7, "Uvite (CaMg3(Al5Mg)(BO3)3Si6O18(OH)4) - tourmaline"),
            ('liddicoatite', 0.7, "Liddicoatite (Ca(Li,Al)3Al6(BO3)3Si6O18(OH)4) - tourmaline"),
            ('foitite', 0.7, "Foitite (□(Fe,Al)3Al6(BO3)3Si6O18(OH)4) - tourmaline"),
            ('buergerite', 0.7, "Buergerite (NaFe3Al6(BO3)3Si6O18O3F) - tourmaline"),
            
            # Beryl group
            ('beryl', 0.8, "Beryl (Be3Al2Si6O18) - ring silicate"),
            ('emerald', 0.8, "Emerald (Cr-bearing beryl) - ring silicate"),
            ('aquamarine', 0.8, "Aquamarine (Fe-bearing beryl) - ring silicate"),
            ('morganite', 0.8, "Morganite (Mn-bearing beryl) - ring silicate"),
            ('heliodor', 0.7, "Heliodor (Fe-bearing beryl) - ring silicate"),
            ('bixbite', 0.7, "Bixbite (Mn-bearing beryl) - ring silicate"),
            
            # Cordierite group
            ('cordierite', 0.8, "Cordierite (Mg2Al4Si5O18) - ring silicate"),
            ('iolite', 0.7, "Iolite (cordierite) - ring silicate"),
            ('sekaninaite', 0.7, "Sekaninaite (Fe-cordierite) - ring silicate"),
            
            # Other ring silicates
            ('benitoite', 0.8, "Benitoite (BaTiSi3O9) - ring silicate"),
            ('milarite', 0.7, "Milarite (KCa2Be2AlSi12O30·H2O) - ring silicate"),
            ('osumilite', 0.7, "Osumilite (K(Fe,Mg)2(Al,Fe)3(Si,Al)12O30) - ring silicate"),
            ('sugilite', 0.6, "Sugilite (KNa2(Fe,Mn,Al)2Li3Si12O30) - ring silicate"),
            ('eudialyte', 0.6, "Eudialyte (Na15Ca6(Fe,Mn)3Zr3Si(Si25O73)(OH,Cl)2) - ring silicate"),
            ('dioptase', 0.6, "Dioptase (CuSiO3·H2O) - ring silicate"),
            
            # Partial matches
            ('cyclo', 0.3, "Cyclosilicate mineral")
        ]
        
        name_lower = mineral_name.lower()
        for name_pattern, weight, description in ring_silicate_names:
            if name_pattern in name_lower:
                score += weight
                reasoning.append(f"Known ring silicate: {description}")
                break  # Only count the first match
        
        # Structural chemistry analysis
        
        # Tourmaline chemistry (B + Al + Si + OH, usually with Na/Ca and other cations)
        if 'B' in elements and 'Al' in elements and 'Si' in elements:
            if re.search(r'OH', chemistry, re.IGNORECASE) or 'H' in elements:
                score += 0.5
                reasoning.append("B-Al-Si-OH system - classic tourmaline chemistry")
            else:
                score += 0.3
                reasoning.append("B-Al-Si system - tourmaline-type chemistry")
        
        # Beryl chemistry (Be + Al + Si, simple formula)
        if 'Be' in elements and 'Al' in elements and 'Si' in elements:
            score += 0.5
            reasoning.append("Be-Al-Si system - beryl group chemistry")
            
            # Bonus for simple beryl chemistry (few elements)
            if len(elements) <= 4:
                score += 0.2
                reasoning.append("Simple Be-Al-Si chemistry - pure beryl type")
        
        # Cordierite chemistry (Mg + Al + Si, often with Fe)
        if 'Mg' in elements and 'Al' in elements and 'Si' in elements and 'B' not in elements and 'Be' not in elements:
            score += 0.3
            reasoning.append("Mg-Al-Si system without B/Be - cordierite-type chemistry")
        
        # Benitoite chemistry (Ba + Ti + Si)
        if 'Ba' in elements and 'Ti' in elements and 'Si' in elements:
            score += 0.4
            reasoning.append("Ba-Ti-Si system - benitoite-type chemistry")
        
        # Ring structure indicators in formula
        # Look for characteristic Si:O ratios typical of rings
        ring_ratios = [
            (r'Si_?6_?O_?18', 0.3, "Si6O18 six-membered ring"),
            (r'Si_?3_?O_?9', 0.2, "Si3O9 three-membered ring"), 
            (r'Si_?4_?O_?12', 0.2, "Si4O12 four-membered ring"),
            (r'Si_?5_?O_?15', 0.2, "Si5O15 five-membered ring")
        ]
        
        for pattern, weight, description in ring_ratios:
            if re.search(pattern, chemistry, re.IGNORECASE):
                score += weight
                reasoning.append(f"Ring structure: {description}")
                break
        
        # Penalty for elements that are uncommon in ring silicates
        incompatible_elements = {
            'P': "phosphates",
            'S': "sulfates/sulfides", 
            'C': "carbonates"
        }
        
        for elem, reason in incompatible_elements.items():
            if elem in elements:
                score = max(0.0, score - 0.15)
                reasoning.append(f"Penalty for {elem} - more typical of {reason}")
        
        # Penalty for very complex chemistry (most ring silicates are relatively simple)
        if len(elements) > 8:
            penalty = (len(elements) - 8) * 0.05
            score = max(0.0, score - penalty)
            reasoning.append(f"Penalty for very complex chemistry ({len(elements)} elements)")
        
        # Bonus for moderate complexity (ring silicates often have 4-6 elements)
        if 4 <= len(elements) <= 6:
            score += 0.1
            reasoning.append(f"Moderate complexity ({len(elements)} elements) - typical of ring silicates")
        
        return min(1.0, score), "; ".join(reasoning) if reasoning else "No ring silicate indicators"
    
    def _score_sheet_silicates(self, chemistry: str, elements: List[str], mineral_name: str) -> Tuple[float, str]:
        """Score sheet silicate minerals (micas, clays, talc, serpentine, chlorite)."""
        score = 0.0
        reasoning = []
        
        # Essential elements for sheet silicates
        if 'Si' in elements:
            score += 0.4
            reasoning.append("Silicon present - essential for silicate sheets")
        else:
            return 0.0, "No silicon - cannot be sheet silicate"
        
        # Hydroxyl groups are characteristic of most sheet silicates
        if re.search(r'OH', chemistry, re.IGNORECASE) or 'H' in elements:
            score += 0.3
            reasoning.append("Hydroxyl groups present - characteristic of sheet silicates")
        
        # Aluminum is common in octahedral layers
        if 'Al' in elements:
            score += 0.2
            reasoning.append("Aluminum present - common in octahedral layers")
        
        # Layer-forming and octahedral cations
        sheet_silicate_elements = {
            'Mg': 0.15,  # Talc, serpentine, chlorite
            'Fe': 0.15,  # Biotite, chlorite
            'K': 0.1,    # Micas (interlayer)
            'Ca': 0.1,   # Some micas and clays
            'Na': 0.1,   # Some micas and clays
            'Ti': 0.05,  # Biotite
            'Mn': 0.05,  # Some micas
            'Li': 0.05   # Lepidolite
        }
        
        found_cations = []
        for element, weight in sheet_silicate_elements.items():
            if element in elements:
                score += weight
                found_cations.append(element)
        
        if found_cations:
            reasoning.append(f"Sheet silicate cations present: {', '.join(found_cations)}")
        
        # Chemical formula patterns characteristic of sheet silicates
        sheet_silicate_patterns = [
            # Mica patterns
            (r'KAl[2-3]Si[3-4]O[10-12]', 0.5, "Mica-type formula (muscovite/biotite)"),
            (r'K.*Al.*Si.*O.*OH', 0.4, "Mica-type formula with K-Al-Si-OH"),
            (r'Na.*Al.*Si.*O.*OH', 0.3, "Sodium mica formula"),
            
            # Clay patterns  
            (r'Al[2-4]Si[2-4]O[5-10].*OH', 0.4, "Clay mineral formula (kaolinite/montmorillonite)"),
            (r'Al.*Si.*O.*OH.*H2O', 0.3, "Hydrated clay mineral"),
            
            # Talc and serpentine patterns
            (r'Mg[3-6]Si[4-8]O[10-20].*OH', 0.4, "Talc/serpentine-type formula"),
            (r'Mg.*Si.*O.*OH', 0.3, "Magnesium sheet silicate"),
            
            # Chlorite patterns
            (r'.*Mg.*Al.*Si.*O.*OH.*', 0.4, "Chlorite-type formula"),
            (r'.*Fe.*Al.*Si.*O.*OH.*', 0.4, "Iron-bearing chlorite"),
            
            # General sheet silicate indicators
            (r'Si[2-8]O[5-20]', 0.2, "Silicate sheet ratio"),
            (r'.*Si.*O.*OH', 0.2, "Silicate with hydroxyl groups")
        ]
        
        for pattern, weight, description in sheet_silicate_patterns:
            if re.search(pattern, chemistry, re.IGNORECASE):
                score += weight
                reasoning.append(f"Sheet silicate pattern: {description}")
                break  # Only count the first (best) match
        
        # Known sheet silicate mineral names
        sheet_silicate_names = [
            # Micas
            ('muscovite', 0.6, "Muscovite mica"),
            ('biotite', 0.6, "Biotite mica"),
            ('phlogopite', 0.6, "Phlogopite mica"),
            ('lepidolite', 0.6, "Lepidolite mica"),
            ('glauconite', 0.5, "Glauconite mica"),
            
            # Clays
            ('kaolinite', 0.6, "Kaolinite clay"),
            ('montmorillonite', 0.6, "Montmorillonite clay"),
            ('illite', 0.6, "Illite clay"),
            ('vermiculite', 0.6, "Vermiculite clay"),
            ('smectite', 0.5, "Smectite clay group"),
            
            # Other sheet silicates
            ('talc', 0.6, "Talc"),
            ('pyrophyllite', 0.6, "Pyrophyllite"),
            ('serpentine', 0.6, "Serpentine group"),
            ('antigorite', 0.6, "Antigorite serpentine"),
            ('chrysotile', 0.6, "Chrysotile serpentine"),
            ('lizardite', 0.6, "Lizardite serpentine"),
            ('chlorite', 0.6, "Chlorite group"),
            ('clinochlore', 0.6, "Clinochlore chlorite"),
            ('chamosite', 0.6, "Chamosite chlorite"),
            
            # Partial matches
            ('mica', 0.4, "Mica group mineral"),
            ('clay', 0.4, "Clay mineral")
        ]
        
        name_lower = mineral_name.lower()
        for name_pattern, weight, description in sheet_silicate_names:
            if name_pattern in name_lower:
                score += weight
                reasoning.append(f"Known sheet silicate: {description}")
                break  # Only count the first match
        
        # Chemical composition analysis for sheet silicates
        # Look for characteristic Si:O ratios and layer chemistry
        if 'Si' in elements and 'O' in elements:
            # Check for Si:O ratios typical of sheet silicates (around 1:2.5-3.0)
            # This is a simplified check - a full implementation would parse the formula
            si_o_indicators = [
                r'Si[2-4]O[5-12]',  # Common sheet silicate ratios
                r'Si[4-8]O[10-24]'  # Larger sheet structures
            ]
            for pattern in si_o_indicators:
                if re.search(pattern, chemistry, re.IGNORECASE):
                    score += 0.2
                    reasoning.append("Si:O ratio consistent with sheet silicates")
                    break
        
        # Penalty for elements that rarely occur in sheet silicates
        incompatible_elements = ['S', 'P', 'B', 'Zr', 'U', 'Th']
        incompatible_found = [elem for elem in incompatible_elements if elem in elements]
        if incompatible_found:
            penalty = len(incompatible_found) * 0.1
            score = max(0.0, score - penalty)
            reasoning.append(f"Penalty for incompatible elements: {', '.join(incompatible_found)}")
        
        # Boost for perfect mica chemistry (K-Al-Si system with OH)
        if all(elem in elements for elem in ['K', 'Al', 'Si']) and 'H' in elements:
            score += 0.2
            reasoning.append("Perfect mica chemistry (K-Al-Si-OH system)")
        
        # Boost for perfect clay chemistry (Al-Si system with OH and water)
        if all(elem in elements for elem in ['Al', 'Si']) and 'H' in elements:
            if any(pattern in chemistry.upper() for pattern in ['H2O', 'H20']):
                score += 0.2
                reasoning.append("Perfect clay chemistry (Al-Si-OH-H2O system)")
        
        return min(1.0, score), "; ".join(reasoning) if reasoning else "No sheet silicate indicators"
    
    def _score_nonsilicate_layers(self, chemistry: str, elements: List[str], mineral_name: str) -> Tuple[float, str]:
        """Score non-silicate layered minerals (graphite, molybdenite, brucite, gibbsite)."""
        score = 0.0
        reasoning = []
        
        # Must NOT contain silicon for non-silicate layers
        if 'Si' in elements:
            return 0.0, "Contains silicon - cannot be non-silicate layer"
        
        score += 0.2
        reasoning.append("No silicon present - compatible with non-silicate layers")
        
        # Layer-forming elements and structures
        layer_forming_systems = {
            # Carbon layers (graphite, diamond, organic)
            'carbon_layers': {
                'elements': ['C'],
                'weight': 0.8,
                'description': "Carbon layers (graphite-type)"
            },
            
            # Metal dichalcogenide layers (MX2 structures)
            'dichalcogenide_layers': {
                'elements': [('Mo', 'S'), ('W', 'S'), ('Mo', 'Se'), ('W', 'Se'), ('Mo', 'Te'), ('W', 'Te'),
                           ('Nb', 'S'), ('Ta', 'S'), ('Ti', 'S'), ('Zr', 'S'), ('Hf', 'S'), ('V', 'S'),
                           ('Re', 'S'), ('Tc', 'S'), ('Pt', 'S'), ('Pd', 'S')],
                'weight': 0.7,
                'description': "Metal dichalcogenide layers (MX2)"
            },
            
            # Hydroxide/oxide layers
            'hydroxide_layers': {
                'elements': [('Mg', 'O', 'H'), ('Al', 'O', 'H'), ('Fe', 'O', 'H'), ('Mn', 'O', 'H'),
                           ('Ca', 'O', 'H'), ('Ni', 'O', 'H'), ('Co', 'O', 'H'), ('Zn', 'O', 'H')],
                'weight': 0.6,
                'description': "Metal hydroxide layers"
            },
            
            # Halide layers
            'halide_layers': {
                'elements': [('Bi', 'I'), ('Bi', 'Br'), ('Sb', 'I'), ('Sb', 'Br'), ('Ga', 'S'), ('In', 'S')],
                'weight': 0.5,
                'description': "Metal halide/chalcogenide layers"
            }
        }
        
        # Check for layer-forming element combinations
        found_layer_systems = []
        
        # Carbon layers
        if 'C' in elements:
            score += 0.8
            found_layer_systems.append("Carbon layers")
            reasoning.append("Carbon present - graphite/carbon layer structure")
            
            # Bonus for pure carbon or carbon with minimal other elements
            if len(elements) == 1:  # Pure carbon
                score += 0.3
                reasoning.append("Pure carbon - likely graphite")
            elif len(elements) <= 3:  # Carbon with few other elements
                score += 0.1
                reasoning.append("Simple carbon-based composition")
        
        # Metal dichalcogenides (MX2 structures)
        dichalcogens = {'S', 'Se', 'Te'}
        transition_metals = {'Mo', 'W', 'Nb', 'Ta', 'Ti', 'Zr', 'Hf', 'V', 'Re', 'Tc', 'Pt', 'Pd', 'Ni', 'Co'}
        
        found_dichalcogens = [elem for elem in elements if elem in dichalcogens]
        found_metals = [elem for elem in elements if elem in transition_metals]
        
        if found_dichalcogens and found_metals:
            score += 0.7
            found_layer_systems.append(f"Metal dichalcogenide ({'+'.join(found_metals)}-{'+'.join(found_dichalcogens)})")
            reasoning.append(f"Metal dichalcogenide system: {'+'.join(found_metals)} with {'+'.join(found_dichalcogens)}")
            
            # Bonus for classic MX2 stoichiometry indicators
            if len(found_metals) == 1 and len(found_dichalcogens) == 1:
                score += 0.2
                reasoning.append("Simple MX2 stoichiometry - classic layer structure")
        
        # Metal hydroxide layers
        hydroxide_metals = {'Mg', 'Al', 'Fe', 'Mn', 'Ca', 'Ni', 'Co', 'Zn', 'Cu', 'Cr'}
        found_hydroxide_metals = [elem for elem in elements if elem in hydroxide_metals]
        
        if found_hydroxide_metals and 'O' in elements and 'H' in elements:
            score += 0.6
            found_layer_systems.append(f"Metal hydroxide ({'+'.join(found_hydroxide_metals)}-OH)")
            reasoning.append(f"Metal hydroxide layers: {'+'.join(found_hydroxide_metals)} with OH groups")
            
            # Specific hydroxide bonuses
            if 'Mg' in elements and len(elements) <= 4:  # Brucite-type
                score += 0.2
                reasoning.append("Mg-OH system - brucite-type structure")
            if 'Al' in elements and len(elements) <= 4:  # Gibbsite-type
                score += 0.2
                reasoning.append("Al-OH system - gibbsite-type structure")
        
        if found_layer_systems:
            reasoning.append(f"Layer-forming systems detected: {', '.join(found_layer_systems)}")
        
        # Non-silicate layer formula patterns
        layer_patterns = [
            # Carbon structures
            (r'^C$', 0.9, "Pure carbon (graphite)"),
            (r'^C_?\d*$', 0.8, "Carbon allotrope"),
            
            # Metal dichalcogenides
            (r'^Mo_?S_?2$', 0.8, "Molybdenite (MoS2)"),
            (r'^W_?S_?2$', 0.8, "Tungstenite (WS2)"),
            (r'^Mo_?Se_?2$', 0.7, "Molybdenum diselenide"),
            (r'^W_?Se_?2$', 0.7, "Tungsten diselenide"),
            (r'^Nb_?S_?2$', 0.7, "Niobium disulfide"),
            (r'^Ta_?S_?2$', 0.7, "Tantalum disulfide"),
            (r'^Ti_?S_?2$', 0.6, "Titanium disulfide"),
            (r'^V_?S_?2$', 0.6, "Vanadium disulfide"),
            (r'^Re_?S_?2$', 0.6, "Rhenium disulfide"),
            
            # Metal hydroxides
            (r'^Mg_?.*OH.*_?2$', 0.7, "Brucite (Mg(OH)2)"),
            (r'^Al_?.*OH.*_?3$', 0.7, "Gibbsite (Al(OH)3)"),
            (r'^Fe_?.*OH.*', 0.6, "Iron hydroxide layers"),
            (r'^Ca_?.*OH.*_?2$', 0.6, "Portlandite (Ca(OH)2)"),
            (r'^Ni_?.*OH.*_?2$', 0.6, "Nickel hydroxide"),
            
            # General patterns
            (r'.*_?S_?2$', 0.4, "Disulfide structure (possible layer)"),
            (r'.*_?Se_?2$', 0.4, "Diselenide structure (possible layer)"),
            (r'.*OH.*', 0.3, "Hydroxide groups (possible layer)")
        ]
        
        for pattern, weight, description in layer_patterns:
            if re.search(pattern, chemistry.replace(' ', ''), re.IGNORECASE):
                score += weight
                reasoning.append(f"Layer pattern: {description}")
                break  # Only count the best match
        
        # Known non-silicate layered mineral names
        layer_mineral_names = [
            # Carbon structures
            ('graphite', 0.9, "Graphite (C) - carbon layers"),
            ('diamond', 0.3, "Diamond (C) - 3D carbon structure"),
            ('lonsdaleite', 0.7, "Lonsdaleite - hexagonal diamond"),
            ('carbon', 0.6, "Carbon mineral"),
            
            # Metal dichalcogenides
            ('molybdenite', 0.9, "Molybdenite (MoS2) - molybdenum disulfide layers"),
            ('tungstenite', 0.8, "Tungstenite (WS2) - tungsten disulfide layers"),
            ('jordisite', 0.7, "Jordisite (MoS2) - molybdenite polytype"),
            ('rheniite', 0.7, "Rheniite (ReS2) - rhenium disulfide"),
            
            # Metal hydroxides
            ('brucite', 0.8, "Brucite (Mg(OH)2) - magnesium hydroxide layers"),
            ('gibbsite', 0.8, "Gibbsite (Al(OH)3) - aluminum hydroxide layers"),
            ('portlandite', 0.7, "Portlandite (Ca(OH)2) - calcium hydroxide"),
            ('theophrastite', 0.7, "Theophrastite (Ni(OH)2) - nickel hydroxide"),
            
            # Other layered structures
            ('bismuthinite', 0.6, "Bismuthinite (Bi2S3) - layered bismuth sulfide"),
            ('stibnite', 0.5, "Stibnite (Sb2S3) - layered antimony sulfide"),
            ('orpiment', 0.6, "Orpiment (As2S3) - layered arsenic sulfide"),
            ('realgar', 0.5, "Realgar (As4S4) - layered arsenic sulfide"),
            
            # Group terms
            ('hydroxide', 0.4, "Hydroxide mineral (possible layer structure)"),
            ('dichalcogenide', 0.5, "Metal dichalcogenide")
        ]
        
        name_lower = mineral_name.lower()
        for name_pattern, weight, description in layer_mineral_names:
            if name_pattern in name_lower:
                score += weight
                reasoning.append(f"Known layered mineral: {description}")
                break  # Only count the first match
        
        # Structural chemistry analysis
        
        # Pure elemental layers (graphite-type)
        if len(elements) == 1:
            if 'C' in elements:
                score += 0.4
                reasoning.append("Pure carbon - graphite layer structure")
            elif elements[0] in transition_metals:
                score += 0.2
                reasoning.append("Pure transition metal - possible layered structure")
        
        # Simple binary compounds (MX, MX2 patterns)
        elif len(elements) == 2:
            elem1, elem2 = elements
            
            # Metal-chalcogen combinations
            if (elem1 in transition_metals and elem2 in dichalcogens) or (elem2 in transition_metals and elem1 in dichalcogens):
                score += 0.4
                reasoning.append("Binary metal-chalcogen compound - likely layered")
            
            # Metal-oxygen combinations (simple oxides can have layer structures)
            elif (elem1 in transition_metals and elem2 == 'O') or (elem1 == 'O' and elem2 in transition_metals):
                score += 0.2
                reasoning.append("Binary metal oxide - possible layered structure")
        
        # Ternary hydroxides (M-O-H systems)
        elif len(elements) == 3 and 'O' in elements and 'H' in elements:
            metal = [elem for elem in elements if elem not in ['O', 'H']][0]
            if metal in hydroxide_metals:
                score += 0.4
                reasoning.append(f"Ternary {metal}-O-H system - hydroxide layer structure")
        
        # Penalty for elements uncommon in layered structures
        incompatible_elements = {
            'Si': "silicate minerals",  # Already handled above
            'P': "phosphate minerals",
            'S': "sulfate minerals (when with O)",  # But S alone is OK for dichalcogenides
            'C': "carbonate minerals (when with O)"  # But C alone is OK for graphite
        }
        
        # Only penalize S and C if they appear with O (indicating sulfates/carbonates)
        for elem, reason in incompatible_elements.items():
            if elem in elements:
                if elem == 'S' and 'O' in elements and len([e for e in elements if e in transition_metals]) == 0:
                    score = max(0.0, score - 0.2)
                    reasoning.append(f"Penalty for S with O - more typical of sulfates")
                elif elem == 'C' and 'O' in elements and len(elements) > 2:
                    score = max(0.0, score - 0.15)
                    reasoning.append(f"Penalty for C with O in complex formula - more typical of carbonates")
                elif elem == 'P':
                    score = max(0.0, score - 0.15)
                    reasoning.append(f"Penalty for {elem} - more typical of {reason}")
        
        # Bonus for elements that strongly indicate layered structures
        layer_indicators = {
            'Mo': 0.2,  # Molybdenite
            'W': 0.15,  # Tungstenite
            'Nb': 0.1,  # Niobium dichalcogenides
            'Ta': 0.1,  # Tantalum dichalcogenides
            'Re': 0.1   # Rhenium dichalcogenides
        }
        
        for elem, bonus in layer_indicators.items():
            if elem in elements:
                score += bonus
                reasoning.append(f"Bonus for {elem} - commonly forms layered structures")
        
        # Penalty for too many elements (layered structures are often simple)
        if len(elements) > 5:
            penalty = (len(elements) - 5) * 0.05
            score = max(0.0, score - penalty)
            reasoning.append(f"Penalty for complex chemistry ({len(elements)} elements) - layered structures are typically simple")
        
        # Bonus for appropriate complexity
        if 1 <= len(elements) <= 3:
            score += 0.1
            reasoning.append(f"Simple chemistry ({len(elements)} elements) - typical of layered structures")
        
        return min(1.0, score), "; ".join(reasoning) if reasoning else "No non-silicate layer indicators"
    
    def _score_simple_oxides(self, chemistry: str, elements: List[str], mineral_name: str) -> Tuple[float, str]:
        """Score simple oxide minerals (hematite, magnetite, rutile, quartz, etc.)."""
        score = 0.0
        reasoning = []
        
        # Essential: Must have oxygen
        if 'O' not in elements:
            return 0.0, "No oxygen - cannot be oxide"
        
        score += 0.3
        reasoning.append("Oxygen present - essential for oxides")
        
        # Common oxide-forming metals with their typical oxidation states
        oxide_metals = {
            'Fe': 0.4,   # Hematite (Fe2O3), magnetite (Fe3O4), wüstite (FeO)
            'Ti': 0.4,   # Rutile, anatase, brookite (TiO2)
            'Al': 0.4,   # Corundum (Al2O3)
            'Cr': 0.3,   # Chromite (Cr2O3)
            'Mn': 0.3,   # Pyrolusite (MnO2), hausmannite (Mn3O4)
            'Zn': 0.3,   # Zincite (ZnO)
            'Cu': 0.3,   # Cuprite (Cu2O), tenorite (CuO)
            'Mg': 0.3,   # Periclase (MgO)
            'Ca': 0.2,   # Lime (CaO) - less common as simple oxide
            'Ni': 0.2,   # Bunsenite (NiO)
            'Co': 0.2,   # CoO
            'V': 0.2,    # Various vanadium oxides
            'W': 0.2,    # Tungsten oxides
            'Mo': 0.2,   # Molybdenum oxides
            'Sn': 0.2,   # Cassiterite (SnO2)
            'Pb': 0.2,   # Litharge (PbO), massicot (PbO)
            'U': 0.2     # Uraninite (UO2)
        }
        
        found_metals = []
        for metal, weight in oxide_metals.items():
            if metal in elements:
                score += weight
                found_metals.append(metal)
        
        if found_metals:
            reasoning.append(f"Oxide-forming metals present: {', '.join(found_metals)}")
        else:
            # If no typical oxide metals, check for Si (quartz group)
            if 'Si' in elements:
                score += 0.3
                reasoning.append("Silicon present - silica group oxides")
            else:
                score = max(0.0, score - 0.2)
                reasoning.append("No typical oxide-forming metals")
        
        # Simple oxide formula patterns (Metal + Oxygen only, minimal other elements)
        simple_oxide_patterns = [
            # Iron oxides
            (r'^Fe_?2_?O_?3$', 0.6, "Hematite formula (Fe2O3)"),
            (r'^Fe_?3_?O_?4$', 0.6, "Magnetite formula (Fe3O4)"),
            (r'^FeO$', 0.6, "Wüstite formula (FeO)"),
            
            # Titanium oxides
            (r'^Ti_?O_?2$', 0.6, "Titanium dioxide (TiO2 - rutile/anatase)"),
            
            # Aluminum oxides
            (r'^Al_?2_?O_?3$', 0.6, "Corundum formula (Al2O3)"),
            
            # Silicon oxides
            (r'^Si_?O_?2$', 0.6, "Silica formula (SiO2 - quartz group)"),
            
            # Other simple oxides
            (r'^Cr_?2_?O_?3$', 0.5, "Chromium oxide (Cr2O3)"),
            (r'^Mn_?O_?2$', 0.5, "Pyrolusite formula (MnO2)"),
            (r'^Mn_?3_?O_?4$', 0.5, "Hausmannite formula (Mn3O4)"),
            (r'^ZnO$', 0.5, "Zincite formula (ZnO)"),
            (r'^Cu_?2_?O$', 0.5, "Cuprite formula (Cu2O)"),
            (r'^CuO$', 0.5, "Tenorite formula (CuO)"),
            (r'^MgO$', 0.5, "Periclase formula (MgO)"),
            (r'^NiO$', 0.4, "Bunsenite formula (NiO)"),
            (r'^CoO$', 0.4, "Cobalt oxide (CoO)"),
            (r'^SnO_?2$', 0.4, "Cassiterite formula (SnO2)"),
            
            # General patterns
            (r'^[A-Z][a-z]?_?[0-9]*_?O_?[0-9]*$', 0.3, "Simple metal oxide pattern"),
            (r'^[A-Z][a-z]?_?[0-9]*_?O$', 0.3, "Simple 1:1 metal oxide")
        ]
        
        for pattern, weight, description in simple_oxide_patterns:
            if re.search(pattern, chemistry.replace(' ', ''), re.IGNORECASE):
                score += weight
                reasoning.append(f"Simple oxide pattern: {description}")
                break  # Only count the best match
        
        # Known simple oxide mineral names
        simple_oxide_names = [
            # Iron oxides
            ('hematite', 0.7, "Hematite (Fe2O3)"),
            ('magnetite', 0.7, "Magnetite (Fe3O4)"),
            ('wüstite', 0.6, "Wüstite (FeO)"),
            ('wustite', 0.6, "Wüstite (FeO)"),
            
            # Titanium oxides
            ('rutile', 0.7, "Rutile (TiO2)"),
            ('anatase', 0.7, "Anatase (TiO2)"),
            ('brookite', 0.7, "Brookite (TiO2)"),
            
            # Aluminum oxides
            ('corundum', 0.7, "Corundum (Al2O3)"),
            ('ruby', 0.6, "Ruby (Cr-bearing Al2O3)"),
            ('sapphire', 0.6, "Sapphire (Al2O3)"),
            
            # Silicon oxides (quartz group)
            ('quartz', 0.7, "Quartz (SiO2)"),
            ('cristobalite', 0.6, "Cristobalite (SiO2)"),
            ('tridymite', 0.6, "Tridymite (SiO2)"),
            ('coesite', 0.6, "Coesite (SiO2)"),
            ('stishovite', 0.6, "Stishovite (SiO2)"),
            
            # Other simple oxides
            ('chromite', 0.6, "Chromite (FeCr2O4)"),  # Actually a spinel, but often grouped with simple oxides
            ('pyrolusite', 0.6, "Pyrolusite (MnO2)"),
            ('hausmannite', 0.6, "Hausmannite (Mn3O4)"),
            ('zincite', 0.6, "Zincite (ZnO)"),
            ('cuprite', 0.6, "Cuprite (Cu2O)"),
            ('tenorite', 0.6, "Tenorite (CuO)"),
            ('periclase', 0.6, "Periclase (MgO)"),
            ('bunsenite', 0.5, "Bunsenite (NiO)"),
            ('lime', 0.5, "Lime (CaO)"),
            ('cassiterite', 0.6, "Cassiterite (SnO2)"),
            ('uraninite', 0.6, "Uraninite (UO2)"),
            
            # Partial matches
            ('oxide', 0.2, "Oxide mineral")
        ]
        
        name_lower = mineral_name.lower()
        for name_pattern, weight, description in simple_oxide_names:
            if name_pattern in name_lower:
                score += weight
                reasoning.append(f"Known simple oxide: {description}")
                break  # Only count the first match
        
        # Penalty for complex chemistry (many different elements suggest complex oxide or other mineral class)
        if len(elements) > 4:
            penalty = (len(elements) - 4) * 0.05
            score = max(0.0, score - penalty)
            reasoning.append(f"Penalty for complex chemistry ({len(elements)} elements)")
        
        # Penalty for elements that suggest other mineral classes
        incompatible_elements = {
            'S': "sulfides/sulfates",
            'P': "phosphates", 
            'C': "carbonates/organic",
            'Si': "silicates",  # Except for SiO2
            'B': "borates",
            'As': "arsenates",
            'V': "vanadates",
            'W': "tungstates",
            'Mo': "molybdates"
        }
        
        # Special handling for Si - allow it only if it's likely SiO2
        if 'Si' in elements:
            if len(elements) <= 2 and 'O' in elements:  # Likely SiO2
                # Already handled above, no penalty
                pass
            else:
                score = max(0.0, score - 0.2)
                reasoning.append("Silicon with other elements suggests complex silicate")
        
        # Check other incompatible elements
        for elem in ['S', 'P', 'C', 'B', 'As', 'V', 'W', 'Mo']:
            if elem in elements:
                score = max(0.0, score - 0.15)
                reasoning.append(f"Penalty for {elem} - suggests {incompatible_elements[elem]}")
        
        # Bonus for perfect simple oxide chemistry (only metal + oxygen)
        if len(elements) == 2 and 'O' in elements:
            score += 0.3
            reasoning.append("Perfect simple oxide chemistry (metal + oxygen only)")
        elif len(elements) == 1 and 'O' in elements:
            score += 0.2
            reasoning.append("Very simple chemistry (oxygen only - might be O2 or peroxide)")
        
        return min(1.0, score), "; ".join(reasoning) if reasoning else "No simple oxide indicators"
    
    def _score_complex_oxides(self, chemistry: str, elements: List[str], mineral_name: str) -> Tuple[float, str]:
        """Score complex oxide minerals (spinels, chromites, garnets, perovskites)."""
        score = 0.0
        reasoning = []
        
        # Must contain oxygen for oxides
        if 'O' not in elements:
            return 0.0, "Missing oxygen - cannot be oxide"
        
        score += 0.2
        reasoning.append("Oxygen present - essential for oxides")
        
        # Must have multiple metals for complex oxides (vs simple oxides)
        metal_elements = [elem for elem in elements if elem not in ['O', 'H', 'F', 'Cl', 'Br', 'I']]
        if len(metal_elements) < 2:
            return 0.0, "Insufficient metals - complex oxides require multiple metal cations"
        
        score += 0.3
        reasoning.append(f"Multiple metals present ({len(metal_elements)}) - compatible with complex oxides")
        
        # Complex oxide structural families
        
        # Spinel group (AB2O4 structures)
        spinel_a_site = {'Mg', 'Fe', 'Mn', 'Zn', 'Co', 'Ni', 'Cu', 'Cd'}  # Tetrahedral sites
        spinel_b_site = {'Al', 'Fe', 'Cr', 'V', 'Mn', 'Ti', 'Ga', 'In'}   # Octahedral sites
        
        found_a_site = [elem for elem in elements if elem in spinel_a_site]
        found_b_site = [elem for elem in elements if elem in spinel_b_site]
        
        if found_a_site and found_b_site:
            score += 0.6
            reasoning.append(f"Spinel-type chemistry: A-site ({'+'.join(found_a_site)}) + B-site ({'+'.join(found_b_site)})")
            
            # Bonus for classic spinel combinations
            if 'Mg' in elements and 'Al' in elements:
                score += 0.2
                reasoning.append("Mg-Al combination - spinel (MgAl2O4)")
            if 'Fe' in elements and 'Cr' in elements:
                score += 0.2
                reasoning.append("Fe-Cr combination - chromite (FeCr2O4)")
            if 'Zn' in elements and 'Al' in elements:
                score += 0.15
                reasoning.append("Zn-Al combination - gahnite (ZnAl2O4)")
            if 'Fe' in elements and 'Al' in elements:
                score += 0.15
                reasoning.append("Fe-Al combination - hercynite (FeAl2O4)")
        
        # Garnet group (A3B2Si3O12 or complex A3B2C3O12)
        garnet_a_site = {'Ca', 'Mg', 'Fe', 'Mn'}  # Dodecahedral sites
        garnet_b_site = {'Al', 'Fe', 'Cr', 'V', 'Ti'}  # Octahedral sites
        garnet_c_site = {'Si', 'Al', 'Fe', 'Ti'}  # Tetrahedral sites
        
        found_garnet_a = [elem for elem in elements if elem in garnet_a_site]
        found_garnet_b = [elem for elem in elements if elem in garnet_b_site]
        found_garnet_c = [elem for elem in elements if elem in garnet_c_site]
        
        if found_garnet_a and found_garnet_b and found_garnet_c and len(elements) >= 4:
            score += 0.5
            reasoning.append(f"Garnet-type chemistry: A-site ({'+'.join(found_garnet_a)}), B-site ({'+'.join(found_garnet_b)}), C-site ({'+'.join(found_garnet_c)})")
            
            # Bonus for classic garnet combinations
            if all(elem in elements for elem in ['Ca', 'Al', 'Si']):
                score += 0.2
                reasoning.append("Ca-Al-Si system - grossular garnet")
            if all(elem in elements for elem in ['Mg', 'Al', 'Si']):
                score += 0.2
                reasoning.append("Mg-Al-Si system - pyrope garnet")
            if all(elem in elements for elem in ['Fe', 'Al', 'Si']):
                score += 0.2
                reasoning.append("Fe-Al-Si system - almandine garnet")
        
        # Perovskite group (ABO3 structures)
        perovskite_a_site = {'Ca', 'Sr', 'Ba', 'Na', 'K', 'La', 'Ce', 'Nd'}  # Large cations
        perovskite_b_site = {'Ti', 'Nb', 'Ta', 'Zr', 'Hf', 'Fe', 'Al', 'Mg'}  # Small cations
        
        found_perov_a = [elem for elem in elements if elem in perovskite_a_site]
        found_perov_b = [elem for elem in elements if elem in perovskite_b_site]
        
        if found_perov_a and found_perov_b:
            score += 0.5
            reasoning.append(f"Perovskite-type chemistry: A-site ({'+'.join(found_perov_a)}) + B-site ({'+'.join(found_perov_b)})")
            
            # Bonus for classic perovskite combinations
            if 'Ca' in elements and 'Ti' in elements:
                score += 0.2
                reasoning.append("Ca-Ti combination - perovskite (CaTiO3)")
            if 'Sr' in elements and 'Ti' in elements:
                score += 0.15
                reasoning.append("Sr-Ti combination - tausonite (SrTiO3)")
        
        # Complex oxide formula patterns
        complex_oxide_patterns = [
            # Spinel patterns (AB2O4)
            (r'.*Al_?2_?O_?4$', 0.7, "Spinel formula (MAl2O4)"),
            (r'.*Cr_?2_?O_?4$', 0.7, "Chromite formula (MCr2O4)"),
            (r'.*Fe_?2_?O_?4$', 0.6, "Ferrite formula (MFe2O4)"),
            (r'Mg_?Al_?2_?O_?4$', 0.8, "Spinel (MgAl2O4)"),
            (r'Fe_?Cr_?2_?O_?4$', 0.8, "Chromite (FeCr2O4)"),
            (r'Zn_?Al_?2_?O_?4$', 0.7, "Gahnite (ZnAl2O4)"),
            (r'Fe_?Al_?2_?O_?4$', 0.7, "Hercynite (FeAl2O4)"),
            (r'.*_?2_?O_?4$', 0.5, "General AB2O4 spinel structure"),
            
            # Garnet patterns (A3B2Si3O12)
            (r'.*_?3_?.*_?2_?Si_?3_?O_?12$', 0.7, "Garnet formula (A3B2Si3O12)"),
            (r'Ca_?3_?Al_?2_?Si_?3_?O_?12$', 0.8, "Grossular garnet"),
            (r'Mg_?3_?Al_?2_?Si_?3_?O_?12$', 0.8, "Pyrope garnet"),
            (r'Fe_?3_?Al_?2_?Si_?3_?O_?12$', 0.8, "Almandine garnet"),
            (r'.*_?3_?.*_?2_?.*_?3_?O_?12$', 0.6, "General garnet structure"),
            
            # Perovskite patterns (ABO3)
            (r'Ca_?Ti_?O_?3$', 0.8, "Perovskite (CaTiO3)"),
            (r'Sr_?Ti_?O_?3$', 0.7, "Tausonite (SrTiO3)"),
            (r'Ba_?Ti_?O_?3$', 0.7, "Barium titanate"),
            (r'.*Ti_?O_?3$', 0.5, "Titanate perovskite"),
            (r'.*Nb_?O_?3$', 0.5, "Niobate perovskite"),
            
            # Other complex oxides
            (r'.*_?2_?.*_?4_?O_?7$', 0.5, "Pyrochlore structure (A2B2O7)"),
            (r'.*_?4_?O_?7$', 0.4, "Complex oxide with 4:7 ratio"),
            (r'.*_?3_?O_?8$', 0.4, "Complex oxide with 3:8 ratio")
        ]
        
        for pattern, weight, description in complex_oxide_patterns:
            if re.search(pattern, chemistry.replace(' ', ''), re.IGNORECASE):
                score += weight
                reasoning.append(f"Complex oxide pattern: {description}")
                break  # Only count the best match
        
        # Known complex oxide mineral names
        complex_oxide_names = [
            # Spinel group
            ('spinel', 0.8, "Spinel (MgAl2O4) - complex oxide"),
            ('chromite', 0.8, "Chromite (FeCr2O4) - complex oxide"),
            ('magnetite', 0.7, "Magnetite (Fe3O4) - inverse spinel"),
            ('gahnite', 0.7, "Gahnite (ZnAl2O4) - spinel group"),
            ('hercynite', 0.7, "Hercynite (FeAl2O4) - spinel group"),
            ('franklinite', 0.7, "Franklinite (ZnFe2O4) - spinel group"),
            ('jacobsite', 0.6, "Jacobsite (MnFe2O4) - spinel group"),
            ('trevorite', 0.6, "Trevorite (NiFe2O4) - spinel group"),
            ('ulvospinel', 0.6, "Ulvöspinel (Fe2TiO4) - spinel group"),
            
            # Garnet group
            ('garnet', 0.7, "Garnet group - complex silicate oxide"),
            ('almandine', 0.8, "Almandine (Fe3Al2Si3O12) - garnet"),
            ('pyrope', 0.8, "Pyrope (Mg3Al2Si3O12) - garnet"),
            ('grossular', 0.8, "Grossular (Ca3Al2Si3O12) - garnet"),
            ('spessartine', 0.7, "Spessartine (Mn3Al2Si3O12) - garnet"),
            ('andradite', 0.7, "Andradite (Ca3Fe2Si3O12) - garnet"),
            ('uvarovite', 0.7, "Uvarovite (Ca3Cr2Si3O12) - garnet"),
            
            # Perovskite group
            ('perovskite', 0.8, "Perovskite (CaTiO3) - complex oxide"),
            ('tausonite', 0.7, "Tausonite (SrTiO3) - perovskite group"),
            ('loparite', 0.6, "Loparite ((Na,Ce,Ca)(Ti,Nb)O3) - perovskite group"),
            
            # Other complex oxides
            ('ilmenite', 0.6, "Ilmenite (FeTiO3) - complex oxide"),
            ('pyrochlore', 0.6, "Pyrochlore (A2B2O7) - complex oxide"),
            ('columbite', 0.6, "Columbite ((Fe,Mn)Nb2O6) - complex oxide"),
            ('tantalite', 0.6, "Tantalite ((Fe,Mn)Ta2O6) - complex oxide"),
            ('chrysoberyl', 0.6, "Chrysoberyl (BeAl2O4) - complex oxide"),
            
            # Group terms
            ('ferrite', 0.5, "Ferrite - complex iron oxide"),
            ('titanate', 0.5, "Titanate - complex titanium oxide"),
            ('aluminate', 0.4, "Aluminate - complex aluminum oxide")
        ]
        
        name_lower = mineral_name.lower()
        for name_pattern, weight, description in complex_oxide_names:
            if name_pattern in name_lower:
                score += weight
                reasoning.append(f"Known complex oxide: {description}")
                break  # Only count the first match
        
        # Structural chemistry analysis
        
        # Check for appropriate metal combinations
        transition_metals = {'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Nb', 'Ta', 'Mo', 'W'}
        alkaline_earth = {'Mg', 'Ca', 'Sr', 'Ba'}
        alkali = {'Na', 'K', 'Rb', 'Cs'}
        main_group = {'Al', 'Ga', 'In', 'Sn', 'Pb', 'Bi'}
        
        found_transition = [elem for elem in elements if elem in transition_metals]
        found_alkaline_earth = [elem for elem in elements if elem in alkaline_earth]
        found_alkali = [elem for elem in elements if elem in alkali]
        found_main_group = [elem for elem in elements if elem in main_group]
        
        # Bonus for diverse metal chemistry (complex oxides often have multiple metal types)
        metal_diversity = len([group for group in [found_transition, found_alkaline_earth, found_alkali, found_main_group] if group])
        if metal_diversity >= 2:
            score += 0.2 * metal_diversity
            reasoning.append(f"Diverse metal chemistry ({metal_diversity} metal groups) - typical of complex oxides")
        
        # Specific structural bonuses
        
        # High-field-strength cations (Ti, Nb, Ta, Zr, Hf)
        hfse = {'Ti', 'Nb', 'Ta', 'Zr', 'Hf'}
        found_hfse = [elem for elem in elements if elem in hfse]
        if found_hfse:
            score += 0.3
            reasoning.append(f"High-field-strength elements ({'+'.join(found_hfse)}) - typical of complex oxides")
        
        # Chromium-bearing systems (chromites, garnets)
        if 'Cr' in elements:
            score += 0.2
            reasoning.append("Chromium present - common in complex oxides (chromites, Cr-garnets)")
        
        # Multiple oxidation state elements (Fe, Mn, Ti, V)
        multi_valent = {'Fe', 'Mn', 'Ti', 'V', 'Cr', 'Co', 'Ni'}
        found_multi_valent = [elem for elem in elements if elem in multi_valent]
        if found_multi_valent:
            score += 0.1 * len(found_multi_valent)
            reasoning.append(f"Multi-valent elements ({'+'.join(found_multi_valent)}) - enable complex oxide structures")
        
        # Penalty for elements uncommon in complex oxides
        incompatible_elements = {
            'Si': "silicate minerals (unless garnet)",
            'P': "phosphate minerals",
            'S': "sulfate/sulfide minerals",
            'C': "carbonate minerals",
            'B': "borate minerals"
        }
        
        for elem, reason in incompatible_elements.items():
            if elem in elements:
                # Special case: Si is OK in garnets
                if elem == 'Si' and len(elements) >= 4 and any(g_elem in elements for g_elem in garnet_a_site):
                    continue  # Don't penalize Si in garnet-like compositions
                else:
                    penalty = 0.1 if elem == 'Si' else 0.15
                    score = max(0.0, score - penalty)
                    reasoning.append(f"Minor penalty for {elem} - more typical of {reason}")
        
        # Bonus for appropriate complexity (complex oxides typically have 3-6 elements)
        if 3 <= len(elements) <= 6:
            score += 0.2
            reasoning.append(f"Appropriate complexity ({len(elements)} elements) - typical of complex oxides")
        elif len(elements) > 6:
            penalty = (len(elements) - 6) * 0.05
            score = max(0.0, score - penalty)
            reasoning.append(f"High complexity ({len(elements)} elements) - may be too complex for typical complex oxides")
        else:
            score = max(0.0, score - 0.1)
            reasoning.append(f"Low complexity ({len(elements)} elements) - complex oxides typically have more elements")
        
        # Stoichiometry bonuses for common complex oxide ratios
        if re.search(r'.*_?2_?O_?4$', chemistry, re.IGNORECASE):  # AB2O4 spinels
            score += 0.2
            reasoning.append("AB2O4 stoichiometry - spinel structure")
        elif re.search(r'.*_?3_?O_?12$', chemistry, re.IGNORECASE):  # Garnets
            score += 0.2
            reasoning.append("Complex O12 stoichiometry - garnet-like structure")
        elif re.search(r'.*O_?3$', chemistry, re.IGNORECASE):  # ABO3 perovskites
            score += 0.15
            reasoning.append("ABO3 stoichiometry - perovskite structure")
        
        return min(1.0, score), "; ".join(reasoning) if reasoning else "No complex oxide indicators"
    
    def _score_hydroxides(self, chemistry: str, elements: List[str], mineral_name: str) -> Tuple[float, str]:
        """Score hydroxide minerals (goethite, lepidocrocite, diaspore, boehmite)."""
        score = 0.0
        reasoning = []
        
        # Must contain both oxygen and hydrogen for hydroxides
        if 'O' not in elements or 'H' not in elements:
            return 0.0, "Missing O or H - cannot be hydroxide"
        
        score += 0.4
        reasoning.append("Both oxygen and hydrogen present - essential for hydroxides")
        
        # Hydroxide-forming metals
        hydroxide_metals = {
            # Common hydroxide formers
            'Fe': {'weight': 0.6, 'description': "Iron hydroxides (goethite, lepidocrocite)"},
            'Al': {'weight': 0.6, 'description': "Aluminum hydroxides (gibbsite, boehmite, diaspore)"},
            'Mg': {'weight': 0.5, 'description': "Magnesium hydroxides (brucite)"},
            'Ca': {'weight': 0.4, 'description': "Calcium hydroxides (portlandite)"},
            'Mn': {'weight': 0.5, 'description': "Manganese hydroxides (manganite, pyrochroite)"},
            'Ni': {'weight': 0.4, 'description': "Nickel hydroxides (theophrastite)"},
            'Co': {'weight': 0.3, 'description': "Cobalt hydroxides"},
            'Zn': {'weight': 0.3, 'description': "Zinc hydroxides"},
            'Cu': {'weight': 0.3, 'description': "Copper hydroxides (spertiniite)"},
            'Cr': {'weight': 0.3, 'description': "Chromium hydroxides"},
            'Ti': {'weight': 0.2, 'description': "Titanium hydroxides"},
            'V': {'weight': 0.2, 'description': "Vanadium hydroxides"}
        }
        
        found_hydroxide_metals = []
        for metal, info in hydroxide_metals.items():
            if metal in elements:
                score += info['weight']
                found_hydroxide_metals.append(metal)
                reasoning.append(info['description'])
        
        if not found_hydroxide_metals:
            return 0.0, "No hydroxide-forming metals present"
        
        # Hydroxide formula patterns
        hydroxide_patterns = [
            # Iron hydroxides
            (r'^Fe_?O_?.*OH.*$', 0.8, "Iron oxyhydroxide (FeOOH) - goethite/lepidocrocite"),
            (r'^Fe_?.*OH.*_?3$', 0.7, "Iron hydroxide (Fe(OH)3)"),
            (r'Fe_?O_?OH$', 0.8, "Goethite/lepidocrocite formula (FeOOH)"),
            
            # Aluminum hydroxides
            (r'^Al_?O_?.*OH.*$', 0.8, "Aluminum oxyhydroxide (AlOOH) - boehmite/diaspore"),
            (r'^Al_?.*OH.*_?3$', 0.7, "Aluminum hydroxide (Al(OH)3) - gibbsite"),
            (r'Al_?O_?OH$', 0.8, "Boehmite/diaspore formula (AlOOH)"),
            
            # Magnesium hydroxides
            (r'^Mg_?.*OH.*_?2$', 0.7, "Magnesium hydroxide (Mg(OH)2) - brucite"),
            
            # Calcium hydroxides
            (r'^Ca_?.*OH.*_?2$', 0.6, "Calcium hydroxide (Ca(OH)2) - portlandite"),
            
            # Manganese hydroxides
            (r'^Mn_?O_?.*OH.*$', 0.7, "Manganese oxyhydroxide (MnOOH) - manganite"),
            (r'^Mn_?.*OH.*_?2$', 0.6, "Manganese hydroxide (Mn(OH)2) - pyrochroite"),
            
            # General hydroxide patterns
            (r'.*O_?OH$', 0.5, "Oxyhydroxide structure (MOOH)"),
            (r'.*OH.*_?2$', 0.4, "Divalent hydroxide (M(OH)2)"),
            (r'.*OH.*_?3$', 0.4, "Trivalent hydroxide (M(OH)3)"),
            (r'.*OH.*', 0.3, "Contains hydroxide groups")
        ]
        
        for pattern, weight, description in hydroxide_patterns:
            if re.search(pattern, chemistry.replace(' ', ''), re.IGNORECASE):
                score += weight
                reasoning.append(f"Hydroxide pattern: {description}")
                break  # Only count the best match
        
        # Known hydroxide mineral names
        hydroxide_names = [
            # Iron hydroxides/oxyhydroxides
            ('goethite', 0.9, "Goethite (α-FeOOH) - iron oxyhydroxide"),
            ('lepidocrocite', 0.9, "Lepidocrocite (γ-FeOOH) - iron oxyhydroxide"),
            ('akaganeite', 0.8, "Akaganeite (β-FeOOH) - iron oxyhydroxide"),
            ('feroxyhyte', 0.8, "Feroxyhyte (δ-FeOOH) - iron oxyhydroxide"),
            ('bernalite', 0.7, "Bernalite (Fe(OH)3) - iron hydroxide"),
            
            # Aluminum hydroxides/oxyhydroxides
            ('gibbsite', 0.9, "Gibbsite (Al(OH)3) - aluminum hydroxide"),
            ('boehmite', 0.9, "Boehmite (γ-AlOOH) - aluminum oxyhydroxide"),
            ('diaspore', 0.9, "Diaspore (α-AlOOH) - aluminum oxyhydroxide"),
            ('bayerite', 0.8, "Bayerite (α-Al(OH)3) - aluminum hydroxide"),
            ('nordstrandite', 0.7, "Nordstrandite (Al(OH)3) - aluminum hydroxide"),
            
            # Magnesium hydroxides
            ('brucite', 0.9, "Brucite (Mg(OH)2) - magnesium hydroxide"),
            
            # Calcium hydroxides
            ('portlandite', 0.8, "Portlandite (Ca(OH)2) - calcium hydroxide"),
            
            # Manganese hydroxides/oxyhydroxides
            ('manganite', 0.8, "Manganite (MnOOH) - manganese oxyhydroxide"),
            ('pyrochroite', 0.8, "Pyrochroite (Mn(OH)2) - manganese hydroxide"),
            ('groutite', 0.7, "Groutite (α-MnOOH) - manganese oxyhydroxide"),
            ('feitknechtite', 0.7, "Feitknechtite (β-MnOOH) - manganese oxyhydroxide"),
            
            # Other hydroxides
            ('theophrastite', 0.8, "Theophrastite (Ni(OH)2) - nickel hydroxide"),
            ('spertiniite', 0.7, "Spertiniite (Cu(OH)2) - copper hydroxide"),
            ('sweetite', 0.7, "Sweetite (Zn(OH)2) - zinc hydroxide"),
            ('bracewellite', 0.6, "Bracewellite (CrOOH) - chromium oxyhydroxide"),
            
            # Group terms
            ('hydroxide', 0.5, "Hydroxide mineral"),
            ('oxyhydroxide', 0.6, "Oxyhydroxide mineral")
        ]
        
        name_lower = mineral_name.lower()
        for name_pattern, weight, description in hydroxide_names:
            if name_pattern in name_lower:
                score += weight
                reasoning.append(f"Known hydroxide: {description}")
                break  # Only count the first match
        
        # Structural chemistry analysis
        
        # Simple hydroxides (M(OH)n patterns)
        if len(elements) == 3 and 'O' in elements and 'H' in elements:
            metal = [elem for elem in elements if elem not in ['O', 'H']][0]
            if metal in hydroxide_metals:
                score += 0.4
                reasoning.append(f"Simple {metal}-O-H hydroxide system")
                
                # Bonus for metals that commonly form simple hydroxides
                if metal in ['Mg', 'Ca', 'Mn', 'Ni', 'Co', 'Zn', 'Cu']:
                    score += 0.2
                    reasoning.append(f"{metal} commonly forms simple hydroxides")
        
        # Oxyhydroxides (MOOH patterns)
        elif len(elements) == 3 and 'O' in elements and 'H' in elements:
            metal = [elem for elem in elements if elem not in ['O', 'H']][0]
            if metal in ['Fe', 'Al', 'Mn', 'Cr']:
                score += 0.5
                reasoning.append(f"{metal} oxyhydroxide system - common structure")
        
        # Check for appropriate metal:OH ratios
        # Look for stoichiometric indicators in formula
        stoich_indicators = [
            (r'.*_?2.*OH.*_?2', 0.2, "M(OH)2 stoichiometry - divalent hydroxide"),
            (r'.*_?3.*OH.*_?3', 0.2, "M(OH)3 stoichiometry - trivalent hydroxide"),
            (r'.*OH_?2', 0.15, "Suggests M(OH)2 structure"),
            (r'.*OH_?3', 0.15, "Suggests M(OH)3 structure")
        ]
        
        for pattern, weight, description in stoich_indicators:
            if re.search(pattern, chemistry, re.IGNORECASE):
                score += weight
                reasoning.append(f"Stoichiometry: {description}")
                break
        
        # Bonus for metals that preferentially form hydroxides
        priority_hydroxide_metals = ['Fe', 'Al', 'Mn']
        found_priority = [metal for metal in found_hydroxide_metals if metal in priority_hydroxide_metals]
        if found_priority:
            score += 0.2
            reasoning.append(f"Priority hydroxide metals ({'+'.join(found_priority)}) - commonly form hydroxides")
        
        # Penalty for elements incompatible with hydroxides
        incompatible_elements = {
            'Si': "silicate minerals",
            'P': "phosphate minerals",
            'S': "sulfate/sulfide minerals",
            'C': "carbonate minerals",
            'B': "borate minerals"
        }
        
        for elem, reason in incompatible_elements.items():
            if elem in elements:
                score = max(0.0, score - 0.15)
                reasoning.append(f"Penalty for {elem} - more typical of {reason}")
        
        # Penalty for too many elements (hydroxides are typically simple)
        if len(elements) > 4:
            penalty = (len(elements) - 4) * 0.1
            score = max(0.0, score - penalty)
            reasoning.append(f"Penalty for complex chemistry ({len(elements)} elements) - hydroxides are typically simple")
        
        # Bonus for appropriate simplicity (hydroxides usually have 3-4 elements)
        if len(elements) == 3:
            score += 0.2
            reasoning.append("Simple ternary composition - typical of hydroxides")
        elif len(elements) == 4:
            score += 0.1
            reasoning.append("Quaternary composition - possible for complex hydroxides")
        
        # Special bonuses for classic hydroxide chemistry
        
        # Iron oxyhydroxides (very common and important)
        if 'Fe' in elements and len(elements) == 3:
            score += 0.3
            reasoning.append("Iron oxyhydroxide chemistry - goethite/lepidocrocite type")
        
        # Aluminum oxyhydroxides (also very common)
        if 'Al' in elements and len(elements) == 3:
            score += 0.3
            reasoning.append("Aluminum oxyhydroxide chemistry - boehmite/diaspore type")
        
        # Check for water content indicators
        if re.search(r'H_?2_?O', chemistry, re.IGNORECASE):
            score += 0.1
            reasoning.append("Water content indicated - common in hydroxide minerals")
        
        return min(1.0, score), "; ".join(reasoning) if reasoning else "No hydroxide indicators"
    
    def _score_organic(self, chemistry: str, elements: List[str], mineral_name: str) -> Tuple[float, str]:
        """Score organic minerals based on chemistry and known organic compounds."""
        score = 0.0
        reasoning = []
        
        # Strong indicators for organic minerals
        organic_indicators = {
            'C': 0.4,  # Carbon is essential for organic compounds
            'H': 0.3,  # Hydrogen commonly present
            'N': 0.2,  # Nitrogen in some organic minerals
            'O': 0.1   # Oxygen in many organic compounds
        }
        
        # Check for organic elements
        organic_elements_present = []
        for element, weight in organic_indicators.items():
            if element in elements:
                score += weight
                organic_elements_present.append(element)
        
        if organic_elements_present:
            reasoning.append(f"Organic elements present: {', '.join(organic_elements_present)}")
        
        # Check for typical organic mineral patterns
        organic_patterns = [
            r'C\d*H\d*',  # Hydrocarbon patterns
            r'C\d*H\d*N\d*',  # Carbon-hydrogen-nitrogen
            r'C\d*H\d*O\d*',  # Carbon-hydrogen-oxygen
            r'C\d*H\d*N\d*O\d*'  # Complex organic patterns
        ]
        
        for pattern in organic_patterns:
            if re.search(pattern, chemistry, re.IGNORECASE):
                score += 0.3
                reasoning.append(f"Organic compound pattern detected: {pattern}")
                break
        
        # Known organic minerals
        organic_mineral_names = [
            'abelsonite',  # NiC31H32N4 - the specific mineral mentioned
            'urea', 'acetamide', 'oxalate', 'mellite', 'whewellite',
            'weddellite', 'glushinskite', 'zhemchuzhnikovite'
        ]
        
        name_lower = mineral_name.lower()
        for organic_name in organic_mineral_names:
            if organic_name in name_lower:
                score += 0.5
                reasoning.append(f"Known organic mineral: {organic_name}")
                break
        
        # Special case for Abelsonite (Ni-porphyrin complex)
        if 'abelsonite' in name_lower or ('Ni' in elements and 'C' in elements and 'H' in elements and 'N' in elements):
            score += 0.4
            reasoning.append("Abelsonite-type organic complex (Ni-porphyrin)")
        
        # Boost score if carbon is present without typical inorganic indicators
        if 'C' in elements:
            # Check if it's likely carbonate (which would be inorganic)
            carbonate_indicators = ['CO3', 'CaCO3', 'MgCO3', 'FeCO3']
            is_likely_carbonate = any(indicator in chemistry for indicator in carbonate_indicators)
            
            if not is_likely_carbonate:
                score += 0.3
                reasoning.append("Carbon present without carbonate indicators - likely organic")
        
        return score, "; ".join(reasoning) if reasoning else "No organic indicators"
    
    def _score_mixed_mode(self, chemistry: str, elements: List[str], mineral_name: str) -> Tuple[float, str]:
        """Score mixed mode minerals with multiple distinct vibrational units (epidote, vesuvianite, sodalite)."""
        score = 0.0
        reasoning = []
        
        # Mixed mode minerals typically have complex chemistry with multiple structural units
        if len(elements) < 4:
            return 0.0, "Insufficient complexity - mixed mode minerals require multiple structural units"
        
        score += 0.2
        reasoning.append(f"Complex chemistry ({len(elements)} elements) - compatible with mixed modes")
        
        # Mixed mode indicators - minerals that combine multiple structural units
        mixed_mode_indicators = {
            # Sorosilicates with additional groups
            'epidote_group': {
                'required': ['Ca', 'Al', 'Si', 'O'],
                'optional': ['Fe', 'Mn', 'Sr'],
                'description': "Epidote group - sorosilicates with octahedral chains",
                'weight': 0.6
            },
            
            # Complex framework silicates with cages/channels
            'zeolite_group': {
                'required': ['Al', 'Si', 'O'],
                'optional': ['Na', 'Ca', 'K', 'H'],
                'description': "Zeolite group - framework silicates with channels",
                'weight': 0.5
            },
            
            # Sodalite group (framework + anions)
            'sodalite_group': {
                'required': ['Na', 'Al', 'Si', 'O'],
                'optional': ['Cl', 'S', 'OH'],
                'description': "Sodalite group - framework silicates with anion cages",
                'weight': 0.6
            },
            
            # Scapolite group (framework + anions)
            'scapolite_group': {
                'required': ['Ca', 'Al', 'Si', 'O'],
                'optional': ['Na', 'Cl', 'CO3', 'SO4'],
                'description': "Scapolite group - framework silicates with anion channels",
                'weight': 0.6
            },
            
            # Vesuvianite group (complex mixed structures)
            'vesuvianite_group': {
                'required': ['Ca', 'Al', 'Si', 'O'],
                'optional': ['Mg', 'Fe', 'Ti', 'F', 'OH'],
                'description': "Vesuvianite group - complex mixed silicate structures",
                'weight': 0.7
            }
        }
        
        found_mixed_groups = []
        for group_name, group_info in mixed_mode_indicators.items():
            required_present = all(elem in elements for elem in group_info['required'])
            if required_present:
                optional_count = sum(1 for elem in group_info['optional'] if elem in elements)
                if optional_count > 0:  # At least one optional element
                    score += group_info['weight']
                    found_mixed_groups.append(group_name)
                    reasoning.append(f"{group_info['description']} - {optional_count} optional elements present")
        
        # Structural complexity indicators
        
        # Multiple anion groups (indicates mixed modes)
        anion_groups = {
            'silicate': any(elem in elements for elem in ['Si']),
            'carbonate': any(elem in elements for elem in ['C']) and 'O' in elements,
            'sulfate': any(elem in elements for elem in ['S']) and 'O' in elements,
            'phosphate': any(elem in elements for elem in ['P']) and 'O' in elements,
            'halide': any(elem in elements for elem in ['F', 'Cl', 'Br', 'I']),
            'hydroxide': any(elem in elements for elem in ['H']) and 'O' in elements
        }
        
        anion_count = sum(1 for present in anion_groups.values() if present)
        if anion_count >= 2:
            score += 0.3 * anion_count
            present_anions = [name for name, present in anion_groups.items() if present]
            reasoning.append(f"Multiple anion groups ({', '.join(present_anions)}) - indicates mixed modes")
        
        # Framework + molecular group combinations
        if anion_groups['silicate'] and (anion_groups['carbonate'] or anion_groups['sulfate'] or anion_groups['halide']):
            score += 0.4
            reasoning.append("Silicate framework with molecular anions - classic mixed mode")
        
        # Mixed mode formula patterns
        mixed_patterns = [
            # Epidote group patterns
            (r'Ca_?2_?.*Al.*Si.*O.*OH', 0.7, "Epidote group formula (Ca2Al2Si3O12OH)"),
            (r'Ca_?2_?Fe.*Al.*Si.*O', 0.6, "Iron-bearing epidote formula"),
            (r'Ca_?2_?Mn.*Al.*Si.*O', 0.6, "Piemontite formula (Mn-epidote)"),
            
            # Sodalite group patterns
            (r'Na.*Al.*Si.*Cl', 0.7, "Sodalite formula (Na8Al6Si6O24Cl2)"),
            (r'Na.*Al.*Si.*S', 0.6, "Nosean formula (Na8Al6Si6O24SO4)"),
            (r'.*Al.*Si.*Cl', 0.5, "Chloride-bearing aluminosilicate"),
            
            # Scapolite patterns
            (r'.*Ca.*Al.*Si.*Cl', 0.6, "Scapolite formula with chloride"),
            (r'.*Na.*Ca.*Al.*Si.*Cl', 0.7, "Mixed Na-Ca scapolite"),
            
            # Vesuvianite patterns
            (r'Ca_?19_?.*Al.*Si.*O.*OH', 0.8, "Vesuvianite formula (Ca19Al13Si18O68(OH)10)"),
            (r'Ca.*Al.*Si.*F.*OH', 0.6, "F-OH bearing complex silicate"),
            
            # Zeolite patterns
            (r'.*Al.*Si.*H_?2_?O', 0.5, "Hydrated aluminosilicate (zeolite-like)"),
            (r'Na.*Ca.*Al.*Si.*H_?2_?O', 0.6, "Mixed alkali zeolite"),
            
            # General mixed patterns
            (r'.*Si.*CO_?3', 0.4, "Silicate-carbonate mixed system"),
            (r'.*Si.*SO_?4', 0.4, "Silicate-sulfate mixed system"),
            (r'.*Si.*Cl', 0.4, "Silicate-chloride mixed system"),
            (r'.*Al.*Si.*F.*OH', 0.5, "F-OH mixed anion silicate")
        ]
        
        for pattern, weight, description in mixed_patterns:
            if re.search(pattern, chemistry.replace(' ', ''), re.IGNORECASE):
                score += weight
                reasoning.append(f"Mixed mode pattern: {description}")
                break  # Only count the best match
        
        # Known mixed mode mineral names
        mixed_mode_names = [
            # Epidote group
            ('epidote', 0.8, "Epidote (Ca2Al2Si3O12OH) - sorosilicate with chains"),
            ('clinozoisite', 0.8, "Clinozoisite (Ca2Al3Si3O12OH) - epidote group"),
            ('zoisite', 0.8, "Zoisite (Ca2Al3Si3O12OH) - epidote group"),
            ('piemontite', 0.7, "Piemontite (Ca2MnAl2Si3O12OH) - Mn-epidote"),
            ('allanite', 0.7, "Allanite (REE-epidote) - epidote group"),
            
            # Sodalite group
            ('sodalite', 0.8, "Sodalite (Na8Al6Si6O24Cl2) - framework with Cl cages"),
            ('nosean', 0.8, "Nosean (Na8Al6Si6O24SO4) - framework with SO4 cages"),
            ('hauyne', 0.7, "Hauyne ((Na,Ca)4-8Al6Si6O24(SO4,Cl)1-2) - sodalite group"),
            ('lazurite', 0.7, "Lazurite (sodalite group with S) - lapis lazuli"),
            
            # Scapolite group
            ('scapolite', 0.8, "Scapolite group - framework silicates with anion channels"),
            ('marialite', 0.8, "Marialite (Na4Al3Si9O24Cl) - Na-scapolite"),
            ('meionite', 0.8, "Meionite (Ca4Al6Si6O24CO3) - Ca-scapolite"),
            
            # Vesuvianite group
            ('vesuvianite', 0.9, "Vesuvianite (Ca19Al13Si18O68(OH)10) - complex mixed structure"),
            ('idocrase', 0.9, "Idocrase (vesuvianite) - complex mixed structure"),
            
            # Zeolite group
            ('zeolite', 0.6, "Zeolite group - framework silicates with channels"),
            ('analcime', 0.7, "Analcime (NaAlSi2O6·H2O) - zeolite"),
            ('natrolite', 0.7, "Natrolite (Na2Al2Si3O10·2H2O) - zeolite"),
            ('chabazite', 0.6, "Chabazite (zeolite group)"),
            ('stilbite', 0.6, "Stilbite (zeolite group)"),
            
            # Other mixed mode minerals
            ('cancrinite', 0.7, "Cancrinite (Na6Ca2Al6Si6O24(CO3)2) - framework with CO3"),
            ('vishnevite', 0.6, "Vishnevite (cancrinite group with SO4)"),
            ('davyne', 0.6, "Davyne ((Na,K)6Ca2Al6Si6O24(Cl,SO4)2) - cancrinite group"),
            
            # Group terms
            ('mixed', 0.3, "Mixed structural modes"),
            ('complex', 0.2, "Complex mineral structure")
        ]
        
        name_lower = mineral_name.lower()
        for name_pattern, weight, description in mixed_mode_names:
            if name_pattern in name_lower:
                score += weight
                reasoning.append(f"Known mixed mode mineral: {description}")
                break  # Only count the first match
        
        # Structural chemistry analysis
        
        # High complexity bonus (mixed modes are typically very complex)
        if len(elements) >= 6:
            complexity_bonus = (len(elements) - 5) * 0.1
            score += min(0.4, complexity_bonus)  # Cap at 0.4
            reasoning.append(f"High complexity ({len(elements)} elements) - typical of mixed mode minerals")
        
        # Multiple coordination environments
        coordination_indicators = {
            'tetrahedral': ['Si', 'Al', 'P'],
            'octahedral': ['Al', 'Fe', 'Mg', 'Ti', 'Cr'],
            'large_cation': ['Ca', 'Na', 'K', 'Ba', 'Sr'],
            'anion_groups': ['Cl', 'F', 'S', 'C', 'P']
        }
        
        found_coordinations = []
        for coord_type, coord_elements in coordination_indicators.items():
            if any(elem in elements for elem in coord_elements):
                found_coordinations.append(coord_type)
        
        if len(found_coordinations) >= 3:
            score += 0.3
            reasoning.append(f"Multiple coordination types ({', '.join(found_coordinations)}) - mixed mode structure")
        
        # Specific mixed mode chemistry bonuses
        
        # Epidote-type (sorosilicate + chains)
        if all(elem in elements for elem in ['Ca', 'Al', 'Si', 'O']) and ('Fe' in elements or 'Mn' in elements):
            score += 0.4
            reasoning.append("Ca-Al-Si-O with Fe/Mn - epidote-type mixed structure")
        
        # Sodalite-type (framework + cages)
        if all(elem in elements for elem in ['Na', 'Al', 'Si', 'O']) and any(elem in elements for elem in ['Cl', 'S']):
            score += 0.4
            reasoning.append("Na-Al-Si framework with Cl/S - sodalite-type mixed structure")
        
        # Scapolite-type (framework + channels)
        if all(elem in elements for elem in ['Ca', 'Al', 'Si', 'O']) and ('Cl' in elements or 'C' in elements):
            score += 0.4
            reasoning.append("Ca-Al-Si framework with Cl/CO3 - scapolite-type mixed structure")
        
        # Vesuvianite-type (ultra-complex)
        if all(elem in elements for elem in ['Ca', 'Al', 'Si', 'O']) and len(elements) >= 6:
            if any(elem in elements for elem in ['F', 'H', 'Ti', 'Mg', 'Fe']):
                score += 0.5
                reasoning.append("Ultra-complex Ca-Al-Si system - vesuvianite-type mixed structure")
        
        # Penalty for too simple chemistry
        if len(elements) < 5:
            penalty = (5 - len(elements)) * 0.1
            score = max(0.0, score - penalty)
            reasoning.append(f"Penalty for simple chemistry ({len(elements)} elements) - mixed modes are typically complex")
        
        # Penalty for pure single-group chemistry
        if not found_mixed_groups and anion_count < 2:
            score = max(0.0, score - 0.3)
            reasoning.append("No clear mixed mode indicators - may be single structural type")
        
        # Bonus for rare element combinations that indicate mixed modes
        rare_combinations = [
            (['Na', 'Al', 'Si', 'Cl'], 0.2, "Na-Al-Si-Cl system"),
            (['Ca', 'Al', 'Si', 'F', 'OH'], 0.2, "Ca-Al-Si-F-OH system"),
            (['Ca', 'Al', 'Si', 'CO3'], 0.2, "Ca-Al-Si-CO3 system"),
            (['Na', 'Ca', 'Al', 'Si', 'SO4'], 0.2, "Na-Ca-Al-Si-SO4 system")
        ]
        
        for combination, bonus, description in rare_combinations:
            if all(elem in elements for elem in combination):
                score += bonus
                reasoning.append(f"Rare combination: {description} - indicates mixed modes")
        
        return min(1.0, score), "; ".join(reasoning) if reasoning else "No mixed mode indicators"
    
    def get_classification_info(self, group_id: str) -> Dict:
        """Get detailed information about a classification group."""
        return self.vibrational_groups.get(group_id, {})
    
    def get_characteristic_modes_for_group(self, group_id: str) -> List[Dict]:
        """Get characteristic vibrational modes for a specific group."""
        # This would return the expected Raman peaks for minerals in this group
        # Implementation would depend on the specific group
        return []
    
    def suggest_raman_analysis_strategy(self, group_id: str) -> Dict:
        """Suggest optimal Raman analysis parameters for a vibrational group."""
        group_info = self.vibrational_groups.get(group_id, {})
        
        suggestions = {
            "laser_wavelength": "785 nm (recommended for most groups)",
            "spectral_range": group_info.get("typical_range", "200-4000 cm⁻¹"),
            "key_regions": [],
            "potential_interferences": [],
            "analysis_notes": []
        }
        
        # Group-specific suggestions would be implemented here
        return suggestions


def create_hey_celestian_classification_report(input_csv: str, output_file: str):
    """
    Create a comprehensive report comparing traditional Hey classification
    with the new Hey-Celestian vibrational mode-based classification.
    """
    classifier = HeyCelestianClassifier()
    
    # Read the input data
    df = pd.read_csv(input_csv)
    
    results = []
    for _, row in df.iterrows():
        mineral_name = row.get('Mineral Name', '')
        chemistry = row.get('RRUFF Chemistry (concise)', '')
        elements = row.get('Chemistry Elements', '')
        hey_class = row.get('Hey Classification Name', '')
        
        # Classify with vibrational system
        vib_result = classifier.classify_mineral(chemistry, elements, mineral_name)
        
        results.append({
            'Mineral Name': mineral_name,
            'Chemistry': chemistry,
                    'Hey Classification': hey_class,
        'Hey-Celestian Group ID': vib_result['best_group_id'],
        'Hey-Celestian Group Name': vib_result['best_group_name'],
            'Confidence': vib_result['confidence'],
            'Reasoning': vib_result['reasoning']
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    # Generate summary statistics
    summary = {
        'Total Minerals': len(results),
        'Hey-Celestian Groups Used': len(results_df['Hey-Celestian Group ID'].unique()),
        'Average Confidence': results_df['Confidence'].mean(),
        'High Confidence (>0.8)': len(results_df[results_df['Confidence'] > 0.8]),
        'Group Distribution': results_df['Hey-Celestian Group Name'].value_counts().to_dict()
    }
    
    return results_df, summary


if __name__ == "__main__":
    # Example usage
    classifier = HeyCelestianClassifier()
    
    # Test with some example minerals
    test_minerals = [
        {"name": "Quartz", "chemistry": "SiO2", "elements": "Si, O"},
        {"name": "Calcite", "chemistry": "CaCO3", "elements": "Ca, C, O"},
        {"name": "Gypsum", "chemistry": "CaSO4·2H2O", "elements": "Ca, S, O, H"},
        {"name": "Apatite", "chemistry": "Ca5(PO4)3(F,Cl,OH)", "elements": "Ca, P, O, F, Cl, H"},
    ]
    
    print("Hey-Celestian Classification System - Test Results")
    print("=" * 60)
    
    for mineral in test_minerals:
        result = classifier.classify_mineral(
            mineral["chemistry"], 
            mineral["elements"], 
            mineral["name"]
        )
        
        print(f"\nMineral: {mineral['name']}")
        print(f"Chemistry: {mineral['chemistry']}")
        print(f"Classification: {result['best_group_name']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Reasoning: {result['reasoning']}") 