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
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

# Optional spglib for advanced symmetry analysis
try:
    import spglib
    SPGLIB_AVAILABLE = True
except ImportError:
    SPGLIB_AVAILABLE = False

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

        Extracts lattice parameters, space group, crystal system, point group,
        symmetry operations, and a fully expanded unit cell.
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            lattice_params = LatticeParameters(a=1.0, b=1.0, c=1.0, alpha=90.0, beta=90.0, gamma=90.0)
            space_group = 'Unknown'

            # --- Pass 1: scalar key-value pairs ---
            for line in lines:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                if '_cell_length_a' in s and not s.startswith('loop_'):
                    lattice_params.a = self._extract_numeric_value(s)
                elif '_cell_length_b' in s and not s.startswith('loop_'):
                    lattice_params.b = self._extract_numeric_value(s)
                elif '_cell_length_c' in s and not s.startswith('loop_'):
                    lattice_params.c = self._extract_numeric_value(s)
                elif '_cell_angle_alpha' in s:
                    lattice_params.alpha = self._extract_numeric_value(s)
                elif '_cell_angle_beta' in s:
                    lattice_params.beta = self._extract_numeric_value(s)
                elif '_cell_angle_gamma' in s:
                    lattice_params.gamma = self._extract_numeric_value(s)
                elif '_space_group_name_H-M_alt' in s or '_symmetry_space_group_name_H-M' in s:
                    space_group = self._extract_space_group(s)

            # --- Pass 2: find atom site loop ---
            atoms = []
            for i, line in enumerate(lines):
                s = line.strip()
                if (s.startswith('_atom_site_label') or
                        (s == 'loop_' and
                         any('_atom_site' in lines[j] for j in range(i, min(i + 10, len(lines)))))):
                    atoms = self._parse_atom_site_loop(lines, i)
                    break

            # --- Pass 3: parse symmetry ops and expand unit cell ---
            sym_ops = self._parse_symmetry_ops(lines)
            if sym_ops and atoms:
                atoms = self._apply_symmetry_ops(atoms, sym_ops)

            # Compute Cartesian coordinates
            self._compute_cartesian_coords(atoms, lattice_params)
            lattice_params.volume = self._calculate_unit_cell_volume(lattice_params)

            # Derive crystal system and point group from H-M name
            crystal_system = self._crystal_system_from_sg_name(space_group)
            point_group = self._point_group_from_sg_name(space_group, crystal_system)

            structure = StructureData(
                name=os.path.basename(file_path).replace('.cif', ''),
                lattice_parameters=lattice_params,
                space_group=space_group,
                crystal_system=crystal_system,
                point_group=point_group,
                atoms=atoms
            )

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
        """Extract space group from CIF data line, handling quoted values."""
        try:
            if '"' in line:
                return line.split('"')[1]
            elif "'" in line:
                parts = line.split("'")
                if len(parts) >= 2:
                    return parts[1].strip()
            else:
                parts = line.split()
                if len(parts) >= 2:
                    return ' '.join(parts[1:]).strip()
        except (IndexError, ValueError):
            pass
        return 'Unknown'

    def _crystal_system_from_sg_name(self, sg_name: str) -> str:
        """Determine crystal system from H-M space group name."""
        sg = sg_name.strip().strip("'\"")
        if not sg or sg == 'Unknown':
            return 'Unknown'
        # Remove centering letter
        sym = sg[1:].strip() if sg[0].upper() in 'PIFABCRH' else sg
        sym_cmp = sym.replace(' ', '').replace('_', '').lower()
        # Cubic: 4+3 together, or m-3 pattern
        if ('4' in sym and '3' in sym) or 'm-3' in sym_cmp or 'm3' in sym_cmp:
            return 'cubic'
        if '6' in sym:
            return 'hexagonal'
        if '4' in sym:
            return 'tetragonal'
        if '3' in sym or (sg and sg[0].upper() in 'RH'):
            return 'trigonal'
        parts = [p for p in sym.split() if p not in ('1', '-1')]
        if len(parts) >= 3:
            return 'orthorhombic'
        if len(parts) >= 1 and any('2' in p or 'm' in p for p in parts):
            return 'monoclinic'
        return 'triclinic'

    def _point_group_from_sg_name(self, sg_name: str, crystal_system: str = None) -> str:
        """Derive point group from H-M space group name."""
        _highest = {
            'cubic': 'm-3m', 'hexagonal': '6/mmm', 'tetragonal': '4/mmm',
            'trigonal': '-3m', 'orthorhombic': 'mmm', 'monoclinic': '2/m', 'triclinic': '-1'
        }
        sg = sg_name.strip().strip("'\"")
        if not sg or sg == 'Unknown':
            return _highest.get(crystal_system or '', 'Unknown')
        # Remove centering letter
        sym = sg[1:].strip() if sg and sg[0].upper() in 'PIFABCRH' else sg
        # Strip underscores used for subscripts in some CIF files
        sym = sym.replace('_', '')
        # Replace screw axes (e.g. 41 -> 4, 63 -> 6, 21 -> 2)
        sym = re.sub(r'([1-6])([1-6])', r'\1', sym)
        # Replace glide planes in '/' context: /a /b /c /n /d /e -> /m
        sym = re.sub(r'/[abcnde]', '/m', sym)
        # Replace standalone glide planes
        parts = sym.split()
        sym = ' '.join(re.sub(r'^([abcnde])$', 'm', p) for p in parts)
        sym_compact = sym.replace(' ', '').lower()
        _pg_map = {
            '4/mmm': '4/mmm', '4/mm': '4/mmm', '4mmm': '4/mmm',
            '4/m': '4/m', '4mm': '4mm', '422': '422',
            '-42m': '-42m', '-4m2': '-4m2', '-4': '-4', '4': '4',
            '6/mmm': '6/mmm', '6/mm': '6/mmm', '6mmm': '6/mmm',
            '6/m': '6/m', '6mm': '6mm', '622': '622',
            '-6m2': '-6m2', '-62m': '-6m2', '-6': '-6', '6': '6',
            'm-3m': 'm-3m', 'm3m': 'm-3m', '-43m': '-43m', '-4 3m': '-43m',
            'm-3': 'm-3', 'm3': 'm-3', '432': '432', '23': '23',
            'mmm': 'mmm', 'mm2': 'mm2', '2mm': 'mm2', '222': '222',
            '2/m': '2/m', 'm': 'm', '2': '2',
            '-3m': '-3m', '-3m1': '-3m', '3m': '3m', '3m1': '3m',
            '32': '32', '321': '32', '312': '32', '-3': '-3', '3': '3',
            '-1': '-1', '1': '1',
        }
        if sym_compact in _pg_map:
            return _pg_map[sym_compact]
        for k, v in _pg_map.items():
            if sym.lower().replace(' ', '') == k.lower():
                return v
        cs = crystal_system or self._crystal_system_from_sg_name(sg_name)
        return _highest.get(cs.lower(), 'Unknown')

    def _parse_symmetry_ops(self, lines: list) -> list:
        """Parse symmetry operations from _space_group_symop_operation_xyz block."""
        ops = []
        in_block = False
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if ('_space_group_symop_operation_xyz' in s or
                    '_symmetry_equiv_pos_as_xyz' in s):
                in_block = True
                continue
            if in_block:
                if s.startswith('_') or (s == 'loop_' and ops):
                    break
                if s == 'loop_':
                    continue
                op = s.strip("'\" ")
                if ',' in op:
                    ops.append(op)
        return ops if ops else ['x,y,z']

    def _apply_sym_op(self, op_str: str, x: float, y: float, z: float) -> list:
        """Apply a symmetry operation string to fractional coordinates using eval."""
        op = op_str.strip("'\" ").lower().replace(' ', '')
        ns = {'x': float(x), 'y': float(y), 'z': float(z)}
        try:
            result = [float(eval(c, {'__builtins__': {}}, ns)) for c in op.split(',')]
            return [v % 1.0 for v in result]
        except Exception:
            return [x, y, z]

    def _apply_symmetry_ops(self, atoms: list, sym_ops: list) -> list:
        """Generate all symmetry-equivalent positions in the unit cell."""
        all_atoms = []
        tol = 1e-3
        for atom in atoms:
            for op in sym_ops:
                nx, ny, nz = self._apply_sym_op(op, atom.x, atom.y, atom.z)
                duplicate = any(
                    a.element == atom.element and
                    abs(a.x - nx) < tol and abs(a.y - ny) < tol and abs(a.z - nz) < tol
                    for a in all_atoms
                )
                if not duplicate:
                    all_atoms.append(AtomData(
                        label=atom.label, element=atom.element,
                        x=nx, y=ny, z=nz, occupancy=atom.occupancy
                    ))
        return all_atoms if all_atoms else atoms

    def _compute_cartesian_coords(self, atoms: list, lattice: 'LatticeParameters') -> None:
        """Convert fractional to Cartesian coordinates (general triclinic formula)."""
        a, b, c = lattice.a, lattice.b, lattice.c
        cg = np.cos(np.radians(lattice.gamma))
        sg = np.sin(np.radians(lattice.gamma))
        cb = np.cos(np.radians(lattice.beta))
        ca = np.cos(np.radians(lattice.alpha))
        cy = (ca - cb * cg) / max(sg, 1e-10)
        cz = np.sqrt(max(0.0, 1.0 - cb**2 - cy**2))
        M = np.array([[a, b * cg, c * cb],
                      [0.0, b * sg, c * cy],
                      [0.0, 0.0,   c * cz]])
        for atom in atoms:
            atom.cartesian_coords = M @ np.array([atom.x, atom.y, atom.z])
    
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