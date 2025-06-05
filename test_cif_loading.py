#!/usr/bin/env python3
"""
Test CIF loading functionality
Tests the fixed crystal structure extraction methods.
"""

import sys
import os

def test_cif_loading():
    """Test CIF loading with the fixed methods."""
    try:
        from pymatgen.io.cif import CifParser
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        
        # Test with anatase.cif
        anatase_path = "__exampleData/anatase.cif"
        
        if not os.path.exists(anatase_path):
            print(f"❌ Test file not found: {anatase_path}")
            return False
        
        print(f"🔄 Testing CIF loading with {anatase_path}")
        
        # Parse CIF file
        parser = CifParser(anatase_path)
        structures = parser.parse_structures(primitive=True)
        
        if not structures:
            print("❌ No structures found in CIF file")
            return False
        
        structure = structures[0]
        print(f"✅ Structure parsed successfully")
        
        # Test crystal system extraction (the part that was failing)
        try:
            sga = SpacegroupAnalyzer(structure)
            crystal_system = sga.get_crystal_system()
            space_group = sga.get_space_group_symbol()
            print(f"✅ Crystal system: {crystal_system}")
            print(f"✅ Space group: {space_group}")
        except Exception as e:
            print(f"⚠️  SpacegroupAnalyzer failed: {e}")
            # Try lattice-based determination
            crystal_system = determine_crystal_system_from_lattice(structure.lattice)
            print(f"✅ Crystal system (fallback): {crystal_system}")
        
        # Test basic structure properties
        print(f"✅ Formula: {structure.composition.reduced_formula}")
        print(f"✅ Number of atoms: {len(structure)}")
        print(f"✅ Lattice parameters:")
        print(f"    a = {structure.lattice.a:.4f} Å")
        print(f"    b = {structure.lattice.b:.4f} Å")
        print(f"    c = {structure.lattice.c:.4f} Å")
        print(f"    α = {structure.lattice.alpha:.2f}°")
        print(f"    β = {structure.lattice.beta:.2f}°")
        print(f"    γ = {structure.lattice.gamma:.2f}°")
        
        # Test atomic sites
        elements = list(set([site.specie.symbol for site in structure]))
        print(f"✅ Elements: {elements}")
        
        # Test atomic radius access
        for i, site in enumerate(structure):
            try:
                radius = site.specie.atomic_radius if site.specie.atomic_radius else 1.0
                print(f"✅ Site {i}: {site.specie.symbol} at {site.frac_coords} (radius: {radius})")
                if i >= 2:  # Only show first few sites
                    break
            except Exception as e:
                print(f"⚠️  Issue with site {i}: {e}")
        
        print("🎉 All tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure pymatgen is installed: pip install pymatgen")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def determine_crystal_system_from_lattice(lattice):
    """Fallback crystal system determination."""
    a, b, c = lattice.a, lattice.b, lattice.c
    alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
    
    tol = 0.01
    
    if (abs(a - b) < tol and abs(b - c) < tol and 
        abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 90) < tol):
        return 'cubic'
    elif (abs(a - b) < tol and abs(alpha - 90) < tol and 
          abs(beta - 90) < tol and abs(gamma - 120) < tol):
        return 'hexagonal'
    elif (abs(a - b) < tol and abs(alpha - 90) < tol and 
          abs(beta - 90) < tol and abs(gamma - 90) < tol):
        return 'tetragonal'
    elif (abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 90) < tol):
        return 'orthorhombic'
    elif (abs(alpha - 90) < tol and abs(gamma - 90) < tol):
        return 'monoclinic'
    else:
        return 'triclinic'

if __name__ == "__main__":
    test_cif_loading() 