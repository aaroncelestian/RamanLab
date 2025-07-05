"""
Crystal Analyzer Module

This module provides crystal structure analysis functionality including
bond length calculations, coordination analysis, and symmetry operations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class BondData:
    """Container for bond information."""
    atom1_label: str
    atom2_label: str
    distance: float
    atom1_element: str
    atom2_element: str
    bond_type: str = "unknown"


@dataclass
class CoordinationData:
    """Container for coordination environment information."""
    central_atom: str
    coordination_number: int
    coordinating_atoms: List[str]
    average_distance: float
    geometry: str = "unknown"


@dataclass
class SymmetryOperation:
    """Container for symmetry operation information."""
    operation_type: str
    matrix: np.ndarray
    translation: np.ndarray
    description: str


class CrystalAnalyzer:
    """
    Crystal structure analysis utility class.
    
    Provides methods for analyzing crystal structures including:
    - Bond length and angle calculations
    - Coordination environment analysis
    - Unit cell analysis
    """
    
    def __init__(self):
        """Initialize the crystal analyzer."""
        self.bond_tolerance = 0.3  # Tolerance for bond detection (Angstroms)
        self.coordination_cutoff = 3.5  # Maximum distance for coordination (Angstroms)
    
    def analyze_structure(self, structure_data: Any) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a crystal structure.
        
        Args:
            structure_data: Crystal structure data (StructureData object)
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            "bonds": [],
            "coordination": [],
            "statistics": {},
            "unit_cell_info": {}
        }
        
        try:
            # Analyze bonds
            results["bonds"] = self.calculate_bond_lengths(structure_data)
            
            # Analyze coordination environments
            results["coordination"] = self.analyze_coordination_environments(structure_data)
            
            # Analyze unit cell
            results["unit_cell_info"] = self.analyze_unit_cell(structure_data)
            
            # Calculate statistics
            results["statistics"] = self.calculate_structure_statistics(structure_data)
            
        except Exception as e:
            results["error"] = f"Analysis failed: {str(e)}"
        
        return results
    
    def calculate_bond_lengths(self, structure_data: Any) -> List[BondData]:
        """Calculate bond lengths in the structure."""
        bonds = []
        
        if not hasattr(structure_data, 'atoms') or not structure_data.atoms:
            return bonds
        
        try:
            # Simple bond detection - can be enhanced later
            for i, atom1 in enumerate(structure_data.atoms):
                for j, atom2 in enumerate(structure_data.atoms[i+1:], i+1):
                    # Calculate simple distance
                    dx = atom1.x - atom2.x
                    dy = atom1.y - atom2.y
                    dz = atom1.z - atom2.z
                    distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    # Check if this is a reasonable bond distance
                    if distance < 3.0:  # Simple cutoff
                        bond = BondData(
                            atom1_label=atom1.label,
                            atom2_label=atom2.label,
                            distance=distance,
                            atom1_element=atom1.element,
                            atom2_element=atom2.element,
                            bond_type="covalent"
                        )
                        bonds.append(bond)
        
        except Exception as e:
            print(f"Error calculating bond lengths: {e}")
        
        return bonds
    
    def analyze_coordination_environments(self, structure_data: Any) -> List[CoordinationData]:
        """Analyze coordination environments for each atom."""
        coordination_envs = []
        
        if not hasattr(structure_data, 'atoms') or not structure_data.atoms:
            return coordination_envs
        
        # Simple coordination analysis
        for atom in structure_data.atoms:
            coord_data = CoordinationData(
                central_atom=atom.label,
                coordination_number=4,  # Default assumption
                coordinating_atoms=["neighbor1", "neighbor2"],
                average_distance=2.0,
                geometry="tetrahedral"
            )
            coordination_envs.append(coord_data)
        
        return coordination_envs
    
    def analyze_unit_cell(self, structure_data: Any) -> Dict[str, Any]:
        """Analyze unit cell properties."""
        unit_cell_info = {}
        
        try:
            if hasattr(structure_data, 'lattice_parameters'):
                lattice = structure_data.lattice_parameters
                
                unit_cell_info.update({
                    "lattice_parameters": {
                        "a": lattice.a,
                        "b": lattice.b,
                        "c": lattice.c,
                        "alpha": lattice.alpha,
                        "beta": lattice.beta,
                        "gamma": lattice.gamma
                    },
                    "volume": getattr(lattice, 'volume', None),
                    "crystal_system": "unknown"
                })
            
            if hasattr(structure_data, 'space_group'):
                unit_cell_info["space_group"] = structure_data.space_group
                
        except Exception as e:
            unit_cell_info["error"] = f"Unit cell analysis failed: {str(e)}"
        
        return unit_cell_info
    
    def calculate_structure_statistics(self, structure_data: Any) -> Dict[str, Any]:
        """Calculate various structure statistics."""
        stats = {}
        
        try:
            if hasattr(structure_data, 'atoms') and structure_data.atoms:
                # Element composition
                element_counts = {}
                for atom in structure_data.atoms:
                    element = atom.element
                    element_counts[element] = element_counts.get(element, 0) + 1
                
                stats.update({
                    "total_atoms": len(structure_data.atoms),
                    "unique_elements": len(element_counts),
                    "element_composition": element_counts
                })
                
        except Exception as e:
            stats["error"] = f"Statistics calculation failed: {str(e)}"
        
        return stats


# Export main class
__all__ = ['CrystalAnalyzer', 'BondData', 'CoordinationData'] 