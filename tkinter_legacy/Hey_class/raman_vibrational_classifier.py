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
from typing import Dict, List, Tuple, Optional
import pandas as pd


class HeyCelestianClassifier:
    """
    Hey-Celestian Classification System
    
    Classify minerals based on their dominant vibrational modes and structural units
    as observed in Raman spectroscopy. This system builds upon Hey's foundational 
    chemical classification but reorganizes minerals by vibrational characteristics
    for enhanced Raman analysis workflows.
    """
    
    def __init__(self):
        """Initialize the vibrational classifier with mode definitions."""
        self.vibrational_groups = self._define_vibrational_groups()
        self.characteristic_modes = self._define_characteristic_modes()
        self.structural_indicators = self._define_structural_indicators()
    
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
    
    def classify_mineral(self, chemistry: str, elements: str = "", mineral_name: str = "") -> Dict:
        """
        Classify a mineral based on its vibrational characteristics.
        
        Parameters:
        -----------
        chemistry : str
            Chemical formula of the mineral
        elements : str
            Comma-separated list of elements
        mineral_name : str
            Name of the mineral (for additional context)
            
        Returns:
        --------
        Dict
            Classification result with ID, name, confidence, and reasoning
        """
        if not chemistry:
            return {"id": "0", "name": "Unclassified", "confidence": 0.0, "reasoning": "No chemical formula provided"}
        
        # Clean and prepare the chemistry string
        clean_chemistry = self._clean_chemistry_formula(chemistry)
        element_list = self._parse_elements(elements) if elements else []
        
        # Score each vibrational group
        group_scores = {}
        for group_id, group_info in self.vibrational_groups.items():
            score, reasoning = self._score_vibrational_group(
                clean_chemistry, element_list, mineral_name, group_id
            )
            group_scores[group_id] = {
                "score": score,
                "reasoning": reasoning,
                "name": group_info["name"]
            }
        
        # Find the best match
        best_group = max(group_scores.items(), key=lambda x: x[1]["score"])
        best_id, best_data = best_group
        
        # Calculate confidence based on score separation
        scores = [data["score"] for data in group_scores.values()]
        scores.sort(reverse=True)
        confidence = scores[0]
        if len(scores) > 1 and scores[1] > 0:
            confidence = min(1.0, scores[0] / (scores[0] + scores[1]))
        
        return {
            "id": best_id,
            "name": best_data["name"],
            "confidence": confidence,
            "reasoning": best_data["reasoning"],
            "all_scores": group_scores
        }
    
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
        reasoning_parts = []
        
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
        
        return score, reasoning
    
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
        return 0.0, "Octahedral framework scoring not yet implemented"
    
    def _score_single_chain_silicates(self, chemistry: str, elements: List[str], mineral_name: str) -> Tuple[float, str]:
        return 0.0, "Single chain silicate scoring not yet implemented"
    
    def _score_double_chain_silicates(self, chemistry: str, elements: List[str], mineral_name: str) -> Tuple[float, str]:
        return 0.0, "Double chain silicate scoring not yet implemented"
    
    def _score_ring_silicates(self, chemistry: str, elements: List[str], mineral_name: str) -> Tuple[float, str]:
        return 0.0, "Ring silicate scoring not yet implemented"
    
    def _score_sheet_silicates(self, chemistry: str, elements: List[str], mineral_name: str) -> Tuple[float, str]:
        return 0.0, "Sheet silicate scoring not yet implemented"
    
    def _score_nonsilicate_layers(self, chemistry: str, elements: List[str], mineral_name: str) -> Tuple[float, str]:
        return 0.0, "Non-silicate layer scoring not yet implemented"
    
    def _score_simple_oxides(self, chemistry: str, elements: List[str], mineral_name: str) -> Tuple[float, str]:
        return 0.0, "Simple oxide scoring not yet implemented"
    
    def _score_complex_oxides(self, chemistry: str, elements: List[str], mineral_name: str) -> Tuple[float, str]:
        return 0.0, "Complex oxide scoring not yet implemented"
    
    def _score_hydroxides(self, chemistry: str, elements: List[str], mineral_name: str) -> Tuple[float, str]:
        return 0.0, "Hydroxide scoring not yet implemented"
    
    def _score_organic(self, chemistry: str, elements: List[str], mineral_name: str) -> Tuple[float, str]:
        return 0.0, "Organic mineral scoring not yet implemented"
    
    def _score_mixed_mode(self, chemistry: str, elements: List[str], mineral_name: str) -> Tuple[float, str]:
        return 0.0, "Mixed mode scoring not yet implemented"
    
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
        'Hey-Celestian Group ID': vib_result['id'],
        'Hey-Celestian Group Name': vib_result['name'],
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
        print(f"Classification: {result['name']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Reasoning: {result['reasoning']}") 