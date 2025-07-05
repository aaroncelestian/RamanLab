"""
CIF (Crystallographic Information File) Parser Module

This module provides comprehensive CIF file parsing capabilities using pymatgen
for professional-grade structure analysis and fallback parsers for basic functionality.

Features:
- Professional CIF parsing with pymatgen
- Fallback simple CIF parser
- Space group analysis
- Symmetry operations
- Atomic position generation

Author: RamanLab Development Team
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

# Try to import pymatgen for advanced CIF parsing
PYMATGEN_AVAILABLE = False
try:
    from pymatgen.io.cif import CifParser
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.core import Structure, Lattice
    PYMATGEN_AVAILABLE = True
    print("Pymatgen available for advanced CIF parsing")
except ImportError:
    print("Pymatgen not available, using fallback CIF parser")


@dataclass
class AtomData:
    """Container for atomic site information."""
    label: str
    element: str
    x: float  # Fractional coordinate
    y: float
    z: float
    occupancy: float = 1.0
    cartesian_coords: Optional[np.ndarray] = None
    wyckoff_symbol: Optional[str] = None
    site_index: Optional[int] = None


@dataclass
class LatticeParameters:
    """Container for unit cell lattice parameters."""
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    volume: Optional[float] = None


@dataclass
class StructureData:
    """Container for complete crystal structure information."""
    name: str
    lattice_parameters: LatticeParameters
    space_group: str
    space_group_number: Optional[int] = None
    crystal_system: Optional[str] = None
    point_group: Optional[str] = None
    atoms: List[AtomData] = None
    symmetry_operations: Optional[List] = None
    pymatgen_structure: Optional[Any] = None  # Store pymatgen Structure object
    
    def __post_init__(self):
        if self.atoms is None:
            self.atoms = []


class CifStructureParser:
    """
    Comprehensive CIF file parser with pymatgen integration and fallback support.
    
    This parser provides both professional-grade parsing using pymatgen and
    a simplified fallback parser for basic CIF structure extraction.
    """
    
    def __init__(self, use_pymatgen: bool = True):
        """
        Initialize the CIF parser.
        
        Args:
            use_pymatgen: Whether to use pymatgen for advanced parsing
        """
        self.use_pymatgen = use_pymatgen and PYMATGEN_AVAILABLE
        self.last_parsed_structure: Optional[StructureData] = None
    
    def parse_cif_file(self, file_path: str, get_conventional: bool = True) -> Optional[StructureData]:
        """
        Parse a CIF file and extract crystal structure information.
        
        Args:
            file_path: Path to the CIF file
            get_conventional: Whether to get conventional standard structure
            
        Returns:
            StructureData object containing parsed structure information
            
        Raises:
            FileNotFoundError: If the CIF file doesn't exist
            ValueError: If the CIF file cannot be parsed
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CIF file not found: {file_path}")
        
        if self.use_pymatgen:
            try:
                structure = self._parse_with_pymatgen(file_path, get_conventional)
                self.last_parsed_structure = structure
                return structure
            except Exception as e:
                print(f"Pymatgen parsing failed: {e}")
                print("Falling back to simple parser...")
        
        # Fallback to simple parser
        structure = self._parse_with_fallback(file_path)
        self.last_parsed_structure = structure
        return structure
    
    def _parse_with_pymatgen(self, file_path: str, get_conventional: bool = True) -> StructureData:
        """
        Parse CIF file using pymatgen for professional-grade parsing.
        
        Args:
            file_path: Path to the CIF file
            get_conventional: Whether to get conventional standard structure
            
        Returns:
            StructureData object with complete structure information
        """
        # Parse CIF file with pymatgen
        parser = CifParser(file_path)
        structures = parser.get_structures()
        
        if not structures:
            raise ValueError("No structures found in CIF file")
        
        # Use the first structure (most CIF files contain one structure)
        original_structure = structures[0]
        
        # Get space group analysis
        sga = SpacegroupAnalyzer(original_structure)
        
        # Get the conventional standard structure to ensure we have all atoms
        pmg_structure = original_structure
        if get_conventional:
            try:
                conventional_structure = sga.get_conventional_standard_structure()
                print(f"Original structure: {len(original_structure.sites)} sites")
                print(f"Conventional structure: {len(conventional_structure.sites)} sites")
                
                # Use conventional structure if it has more atoms (complete unit cell)
                if len(conventional_structure.sites) > len(original_structure.sites):
                    pmg_structure = conventional_structure
                    print(f"Using conventional structure with {len(pmg_structure.sites)} atoms")
                else:
                    print(f"Using original structure with {len(pmg_structure.sites)} atoms")
                    
            except Exception as e:
                print(f"Error getting conventional structure: {e}")
                pmg_structure = original_structure
        
        # Re-analyze with the chosen structure
        sga = SpacegroupAnalyzer(pmg_structure)
        
        # Extract lattice parameters
        lattice_params = LatticeParameters(
            a=pmg_structure.lattice.a,
            b=pmg_structure.lattice.b,
            c=pmg_structure.lattice.c,
            alpha=pmg_structure.lattice.alpha,
            beta=pmg_structure.lattice.beta,
            gamma=pmg_structure.lattice.gamma,
            volume=pmg_structure.lattice.volume
        )
        
        # Extract atomic positions
        atoms = []
        symmetry_dataset = sga.get_symmetry_dataset()
        
        for i, site in enumerate(pmg_structure.sites):
            wyckoff_symbol = 'unknown'
            if symmetry_dataset and 'wyckoffs' in symmetry_dataset:
                if i < len(symmetry_dataset['wyckoffs']):
                    wyckoff_symbol = symmetry_dataset['wyckoffs'][i]
            
            atom = AtomData(
                label=f"{site.specie.symbol}{i+1}",
                element=str(site.specie.symbol),
                x=site.frac_coords[0],
                y=site.frac_coords[1],
                z=site.frac_coords[2],
                occupancy=getattr(site, 'occupancy', 1.0),
                cartesian_coords=site.coords,
                wyckoff_symbol=wyckoff_symbol,
                site_index=i
            )
            atoms.append(atom)
        
        # Create structure data
        structure = StructureData(
            name=os.path.basename(file_path).replace('.cif', ''),
            lattice_parameters=lattice_params,
            space_group=sga.get_space_group_symbol(),
            space_group_number=sga.get_space_group_number(),
            crystal_system=sga.get_crystal_system(),
            point_group=sga.get_point_group_symbol(),
            atoms=atoms,
            symmetry_operations=sga.get_symmetry_operations(),
            pymatgen_structure=pmg_structure
        )
        
        # Print summary
        self._print_structure_summary(structure)
        
        return structure
    
    def _parse_with_fallback(self, file_path: str) -> Optional[StructureData]:
        """
        Parse CIF file using simplified fallback parser.
        
        Args:
            file_path: Path to the CIF file
            
        Returns:
            StructureData object with basic structure information
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Initialize structure data
            lattice_params = LatticeParameters(a=1.0, b=1.0, c=1.0, alpha=90.0, beta=90.0, gamma=90.0)
            space_group = 'Unknown'
            atoms = []
            
            # Parse basic structure information
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Lattice parameters
                if '_cell_length_a' in line:
                    lattice_params.a = self._extract_numeric_value(line)
                elif '_cell_length_b' in line:
                    lattice_params.b = self._extract_numeric_value(line)
                elif '_cell_length_c' in line:
                    lattice_params.c = self._extract_numeric_value(line)
                elif '_cell_angle_alpha' in line:
                    lattice_params.alpha = self._extract_numeric_value(line)
                elif '_cell_angle_beta' in line:
                    lattice_params.beta = self._extract_numeric_value(line)
                elif '_cell_angle_gamma' in line:
                    lattice_params.gamma = self._extract_numeric_value(line)
                
                # Space group
                elif '_space_group_name_H-M_alt' in line or '_symmetry_space_group_name_H-M' in line:
                    space_group = self._extract_space_group(line)
                
                # Atomic positions
                elif line.startswith('_atom_site_label') or (line == 'loop_' and 
                     any('_atom_site' in lines[j] for j in range(i, min(i+10, len(lines))))):
                    atoms = self._parse_atom_site_loop(lines, i)
                    break
            
            # Calculate volume (simplified)
            lattice_params.volume = self._calculate_unit_cell_volume(lattice_params)
            
            # Create structure data
            structure = StructureData(
                name=os.path.basename(file_path).replace('.cif', ''),
                lattice_parameters=lattice_params,
                space_group=space_group,
                atoms=atoms
            )
            
            # Print summary
            self._print_structure_summary(structure)
            
            return structure
            
        except Exception as e:
            print(f"Error parsing CIF file with fallback parser: {e}")
            return None
    
    def _extract_numeric_value(self, line: str) -> float:
        """Extract numeric value from CIF data line."""
        try:
            parts = line.split()
            if len(parts) >= 2:
                # Remove uncertainty notation if present (e.g., "5.123(4)" -> "5.123")
                value_str = parts[1].split('(')[0]
                return float(value_str)
        except (ValueError, IndexError):
            pass
        return 1.0
    
    def _extract_space_group(self, line: str) -> str:
        """Extract space group from CIF data line."""
        try:
            if '"' in line:
                # Extract text between quotes
                return line.split('"')[1]
            else:
                # Extract after tag
                parts = line.split()
                if len(parts) >= 2:
                    return ' '.join(parts[1:])
        except (IndexError, ValueError):
            pass
        return 'Unknown'
    
    def _parse_atom_site_loop(self, lines: List[str], start_idx: int) -> List[AtomData]:
        """
        Parse the atom site loop from CIF file.
        
        Args:
            lines: All lines from the CIF file
            start_idx: Starting index of the loop
            
        Returns:
            List of AtomData objects
        """
        atoms = []
        
        # Find column headers
        headers = []
        i = start_idx
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('_atom_site'):
                headers.append(line)
            elif line == 'loop_':
                pass  # Skip loop declaration
            elif line and not line.startswith('_'):
                break  # Start of data
            i += 1
        
        # Create column mapping
        col_map = self._create_column_mapping(headers)
        
        # Parse data rows
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('_') or line.startswith('loop_') or line.startswith('#'):
                break
            
            parts = line.split()
            if len(parts) >= 4:  # At least label, x, y, z
                try:
                    atom = self._create_atom_from_parts(parts, col_map)
                    atoms.append(atom)
                except (ValueError, IndexError) as e:
                    print(f"Error parsing atom line '{line}': {e}")
                    continue
            i += 1
        
        return atoms
    
    def _create_column_mapping(self, headers: List[str]) -> Dict[str, int]:
        """Create mapping of column types to indices."""
        col_map = {}
        for i, header in enumerate(headers):
            if '_atom_site_label' in header:
                col_map['label'] = i
            elif '_atom_site_type_symbol' in header:
                col_map['element'] = i
            elif '_atom_site_fract_x' in header:
                col_map['x'] = i
            elif '_atom_site_fract_y' in header:
                col_map['y'] = i
            elif '_atom_site_fract_z' in header:
                col_map['z'] = i
            elif '_atom_site_occupancy' in header:
                col_map['occupancy'] = i
        return col_map
    
    def _create_atom_from_parts(self, parts: List[str], col_map: Dict[str, int]) -> AtomData:
        """Create AtomData from parsed line parts."""
        # Get label
        label = parts[col_map.get('label', 0)]
        
        # Get element (try type_symbol first, then extract from label)
        if 'element' in col_map:
            element = parts[col_map['element']]
        else:
            element = ''.join([c for c in label if c.isalpha()])
        
        # Get coordinates
        x = float(parts[col_map.get('x', 1)].split('(')[0])  # Remove uncertainty
        y = float(parts[col_map.get('y', 2)].split('(')[0])
        z = float(parts[col_map.get('z', 3)].split('(')[0])
        
        # Get occupancy
        occupancy = 1.0
        if 'occupancy' in col_map and col_map['occupancy'] < len(parts):
            try:
                occupancy = float(parts[col_map['occupancy']].split('(')[0])
            except ValueError:
                occupancy = 1.0
        
        return AtomData(
            label=label,
            element=element,
            x=x,
            y=y,
            z=z,
            occupancy=occupancy
        )
    
    def _calculate_unit_cell_volume(self, lattice: LatticeParameters) -> float:
        """Calculate unit cell volume from lattice parameters."""
        try:
            # Convert angles to radians
            alpha_rad = np.radians(lattice.alpha)
            beta_rad = np.radians(lattice.beta)
            gamma_rad = np.radians(lattice.gamma)
            
            # Calculate volume using standard formula
            volume = (lattice.a * lattice.b * lattice.c * 
                     np.sqrt(1 + 2*np.cos(alpha_rad)*np.cos(beta_rad)*np.cos(gamma_rad) -
                            np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2))
            
            return volume
        except Exception:
            return lattice.a * lattice.b * lattice.c  # Simplified approximation
    
    def _print_structure_summary(self, structure: StructureData) -> None:
        """Print a summary of the parsed structure."""
        print(f"\nParsed structure: {structure.name}")
        print(f"Space group: {structure.space_group}")
        print(f"Crystal system: {structure.crystal_system}")
        print(f"Lattice parameters:")
        print(f"  a = {structure.lattice_parameters.a:.4f} Å")
        print(f"  b = {structure.lattice_parameters.b:.4f} Å")
        print(f"  c = {structure.lattice_parameters.c:.4f} Å")
        print(f"  α = {structure.lattice_parameters.alpha:.2f}°")
        print(f"  β = {structure.lattice_parameters.beta:.2f}°")
        print(f"  γ = {structure.lattice_parameters.gamma:.2f}°")
        if structure.lattice_parameters.volume:
            print(f"  V = {structure.lattice_parameters.volume:.2f} Å³")
        
        # Count elements
        element_counts = {}
        for atom in structure.atoms:
            element = atom.element
            element_counts[element] = element_counts.get(element, 0) + 1
        
        print(f"Composition:")
        for element, count in element_counts.items():
            print(f"  {element}: {count} atoms")
        print(f"Total atoms: {len(structure.atoms)}")
    
    def create_supercell(self, structure: StructureData, 
                        scale_factors: Tuple[int, int, int] = (2, 2, 2)) -> Optional[StructureData]:
        """
        Create a supercell from the given structure.
        
        Args:
            structure: Original structure
            scale_factors: Scaling factors for (a, b, c) directions
            
        Returns:
            New StructureData object representing the supercell
        """
        if structure.pymatgen_structure is not None and PYMATGEN_AVAILABLE:
            try:
                # Use pymatgen for supercell creation
                supercell_pmg = structure.pymatgen_structure.copy()
                supercell_pmg.make_supercell(scale_factors)
                
                # Convert back to StructureData
                return self._convert_pymatgen_to_structure_data(
                    supercell_pmg, 
                    f"{structure.name}_supercell_{scale_factors[0]}x{scale_factors[1]}x{scale_factors[2]}"
                )
            except Exception as e:
                print(f"Error creating supercell with pymatgen: {e}")
        
        # Fallback: simple supercell creation
        return self._create_simple_supercell(structure, scale_factors)
    
    def _convert_pymatgen_to_structure_data(self, pmg_structure, name: str) -> StructureData:
        """Convert pymatgen Structure to StructureData."""
        # This is a simplified conversion - could be expanded
        atoms = []
        for i, site in enumerate(pmg_structure.sites):
            atom = AtomData(
                label=f"{site.specie.symbol}{i+1}",
                element=str(site.specie.symbol),
                x=site.frac_coords[0],
                y=site.frac_coords[1],
                z=site.frac_coords[2],
                occupancy=getattr(site, 'occupancy', 1.0),
                cartesian_coords=site.coords,
                site_index=i
            )
            atoms.append(atom)
        
        lattice_params = LatticeParameters(
            a=pmg_structure.lattice.a,
            b=pmg_structure.lattice.b,
            c=pmg_structure.lattice.c,
            alpha=pmg_structure.lattice.alpha,
            beta=pmg_structure.lattice.beta,
            gamma=pmg_structure.lattice.gamma,
            volume=pmg_structure.lattice.volume
        )
        
        return StructureData(
            name=name,
            lattice_parameters=lattice_params,
            space_group="Unknown",  # Would need additional analysis
            atoms=atoms,
            pymatgen_structure=pmg_structure
        )
    
    def _create_simple_supercell(self, structure: StructureData, 
                               scale_factors: Tuple[int, int, int]) -> StructureData:
        """Create supercell using simple replication."""
        supercell_atoms = []
        
        for na in range(scale_factors[0]):
            for nb in range(scale_factors[1]):
                for nc in range(scale_factors[2]):
                    for atom in structure.atoms:
                        new_atom = AtomData(
                            label=f"{atom.element}{len(supercell_atoms)+1}",
                            element=atom.element,
                            x=(atom.x + na) / scale_factors[0],
                            y=(atom.y + nb) / scale_factors[1],
                            z=(atom.z + nc) / scale_factors[2],
                            occupancy=atom.occupancy
                        )
                        supercell_atoms.append(new_atom)
        
        # Scale lattice parameters
        new_lattice = LatticeParameters(
            a=structure.lattice_parameters.a * scale_factors[0],
            b=structure.lattice_parameters.b * scale_factors[1],
            c=structure.lattice_parameters.c * scale_factors[2],
            alpha=structure.lattice_parameters.alpha,
            beta=structure.lattice_parameters.beta,
            gamma=structure.lattice_parameters.gamma
        )
        new_lattice.volume = self._calculate_unit_cell_volume(new_lattice)
        
        return StructureData(
            name=f"{structure.name}_supercell_{scale_factors[0]}x{scale_factors[1]}x{scale_factors[2]}",
            lattice_parameters=new_lattice,
            space_group=structure.space_group,
            atoms=supercell_atoms
        )
    
    def get_element_composition(self, structure: StructureData) -> Dict[str, int]:
        """Get element composition from structure."""
        composition = {}
        for atom in structure.atoms:
            element = atom.element
            composition[element] = composition.get(element, 0) + 1
        return composition


def parse_cif_file(file_path: str, use_pymatgen: bool = True, 
                  get_conventional: bool = True) -> Optional[StructureData]:
    """
    Convenience function to parse a CIF file.
    
    Args:
        file_path: Path to the CIF file
        use_pymatgen: Whether to use pymatgen for parsing
        get_conventional: Whether to get conventional standard structure
        
    Returns:
        StructureData object or None if parsing fails
    """
    parser = CifStructureParser(use_pymatgen=use_pymatgen)
    return parser.parse_cif_file(file_path, get_conventional=get_conventional)


# Create aliases for backward compatibility with expected import names
CIFParser = CifStructureParser
CrystalStructure = StructureData

# Export main classes and functions
__all__ = ['CifStructureParser', 'CIFParser', 'StructureData', 'CrystalStructure', 
           'AtomData', 'LatticeParameters', 'parse_cif_file'] 