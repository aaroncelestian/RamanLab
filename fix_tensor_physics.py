#!/usr/bin/env python3
"""
Fix tensor physics by implementing proper crystal symmetry constraints.
This addresses the fundamental issue where tensors don't respect vibrational mode symmetries.
"""

import numpy as np
import matplotlib.pyplot as plt

class CrystalSymmetryTensorGenerator:
    """Generate physically correct Raman tensors based on crystal symmetry and vibrational modes."""
    
    def __init__(self):
        """Initialize the tensor generator with symmetry rules."""
        # Define Raman tensor forms for different point groups and irreducible representations
        self.tensor_forms = {
            # Trigonal D3d (calcite)
            'D3d': {
                'A1g': np.array([  # Totally symmetric, Raman active
                    [1, 0, 0],
                    [0, 1, 0], 
                    [0, 0, 1]
                ]),
                'A1u': np.array([  # Totally symmetric, Raman active (same as A1g for Raman)
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ]),
                'Eg': np.array([   # Doubly degenerate, Raman active
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 0]
                ]),
                'A2g': np.array([  # Raman inactive (zero tensor)
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ]),
                'A2u': np.array([  # Raman inactive (zero tensor)
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ]),
                'Eu': np.array([   # Raman inactive (zero tensor)
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ])
            },
            
            # Cubic Oh (many minerals)
            'Oh': {
                'A1g': np.array([  # Totally symmetric
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ]),
                'Eg': np.array([   # Doubly degenerate
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, -2]
                ]),
                'T2g': np.array([  # Triply degenerate
                    [0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 0]
                ])
            },
            
            # Tetragonal D4h
            'D4h': {
                'A1g': np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ]),
                'B1g': np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 0]
                ]),
                'Eg': np.array([
                    [0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 0]
                ])
            }
        }
    
    def create_correct_tensor(self, crystal_system, point_group, mode_symmetry, wavenumber, intensity):
        """Create a physically correct Raman tensor."""
        
        # Normalize inputs
        crystal_system = crystal_system.lower()
        point_group = point_group.upper() if point_group else self._infer_point_group(crystal_system)
        mode_symmetry = mode_symmetry.upper() if mode_symmetry else 'A1G'
        
        print(f"Creating tensor: {crystal_system} {point_group} {mode_symmetry} @ {wavenumber:.1f} cm⁻¹")
        
        # Get the base tensor form
        if point_group in self.tensor_forms and mode_symmetry in self.tensor_forms[point_group]:
            base_tensor = self.tensor_forms[point_group][mode_symmetry].copy()
        else:
            # Fallback: create tensor based on crystal system
            base_tensor = self._create_crystal_system_tensor(crystal_system, mode_symmetry)
        
        # Apply anisotropy based on crystal system and mode
        tensor = self._apply_anisotropy(base_tensor, crystal_system, mode_symmetry, wavenumber)
        
        # Scale by intensity
        tensor = tensor * intensity
        
        # Ensure symmetry (Raman tensors must be symmetric)
        tensor = (tensor + tensor.T) / 2.0
        
        # Validate physics
        self._validate_tensor_physics(tensor, crystal_system, point_group, mode_symmetry)
        
        return tensor
    
    def _infer_point_group(self, crystal_system):
        """Infer most common point group for crystal system."""
        point_groups = {
            'cubic': 'OH',
            'tetragonal': 'D4H', 
            'orthorhombic': 'D2H',
            'hexagonal': 'D6H',
            'trigonal': 'D3D',
            'monoclinic': 'C2H',
            'triclinic': 'CI'
        }
        return point_groups.get(crystal_system, 'OH')
    
    def _create_crystal_system_tensor(self, crystal_system, mode_symmetry):
        """Create tensor based on crystal system when point group is unknown."""
        
        if crystal_system == 'trigonal':
            if 'A1' in mode_symmetry:
                # A1 modes: totally symmetric, diagonal tensor with xx = yy ≠ zz
                return np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0.3]  # Strong anisotropy
                ])
            elif 'EG' in mode_symmetry or 'E' in mode_symmetry:
                # E modes: doubly degenerate, different tensor form
                return np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 0]
                ])
            else:
                # Default A1-like
                return np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0.3]
                ])
                
        elif crystal_system == 'cubic':
            # Cubic: isotropic
            return np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
            
        elif crystal_system == 'tetragonal':
            # Tetragonal: xx = yy ≠ zz
            return np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 0.7]
            ])
            
        elif crystal_system == 'orthorhombic':
            # Orthorhombic: all different
            return np.array([
                [1, 0, 0],
                [0, 0.8, 0],
                [0, 0, 0.6]
            ])
            
        else:
            # Default to isotropic
            return np.eye(3)
    
    def _apply_anisotropy(self, base_tensor, crystal_system, mode_symmetry, wavenumber):
        """Apply realistic anisotropy while preserving symmetry."""
        
        tensor = base_tensor.copy()
        
        # Frequency-dependent anisotropy (higher frequencies often more anisotropic)
        freq_factor = 1.0 + 0.3 * (wavenumber / 1000.0)
        
        if crystal_system == 'trigonal':
            if 'A1' in mode_symmetry:
                # A1 modes: enhance z-axis anisotropy
                tensor[2,2] *= (0.2 + 0.1 * np.sin(wavenumber / 200.0))  # 0.1-0.3 range
            elif 'E' in mode_symmetry:
                # E modes: enhance xy anisotropy
                tensor[0,0] *= freq_factor
                tensor[1,1] *= -freq_factor  # Opposite sign for E modes
        
        elif crystal_system == 'tetragonal':
            # Enhance c-axis anisotropy
            tensor[2,2] *= (0.5 + 0.3 * np.cos(wavenumber / 150.0))
        
        return tensor
    
    def _validate_tensor_physics(self, tensor, crystal_system, point_group, mode_symmetry):
        """Validate that tensor obeys physics constraints."""
        
        # Check symmetry
        if not np.allclose(tensor, tensor.T, atol=1e-10):
            print(f"⚠️  Warning: Tensor not symmetric for {mode_symmetry}")
        
        # Check crystal system constraints
        if crystal_system == 'trigonal' and 'A1' in mode_symmetry:
            # A1 in trigonal: xx = yy, off-diagonal xy terms should be zero
            if not np.isclose(tensor[0,0], tensor[1,1], atol=1e-6):
                print(f"⚠️  Warning: A1 trigonal tensor should have xx = yy")
            if not np.isclose(tensor[0,1], 0, atol=1e-6):
                print(f"⚠️  Warning: A1 trigonal tensor should have xy = 0")
        
        elif crystal_system == 'cubic':
            # Cubic: should be isotropic (xx = yy = zz)
            diag = np.diag(tensor)
            if not np.allclose(diag, diag[0], atol=1e-6):
                print(f"⚠️  Warning: Cubic tensor should be isotropic")

def fix_calcite_a1u_tensor():
    """Demonstrate the fix for calcite A1u tensor."""
    
    generator = CrystalSymmetryTensorGenerator()
    
    # Current problematic tensor (from screenshot)
    problematic_tensor = np.array([
        [1.855, 0.393, 0.000],
        [0.393, 0.995, 0.000],
        [0.000, 0.000, 0.309]
    ])
    
    # Create correct A1u tensor for trigonal calcite
    correct_tensor = generator.create_correct_tensor(
        crystal_system='trigonal',
        point_group='D3d', 
        mode_symmetry='A1u',
        wavenumber=1082.9,
        intensity=1.0
    )
    
    print("\n🔬 TENSOR COMPARISON:")
    print("=" * 50)
    print("\n❌ CURRENT (Problematic):")
    print(problematic_tensor)
    print(f"   xx ≠ yy: {problematic_tensor[0,0]:.3f} ≠ {problematic_tensor[1,1]:.3f}")
    print(f"   xy ≠ 0: {problematic_tensor[0,1]:.3f}")
    print("   ❌ Violates A1u trigonal symmetry")
    
    print("\n✅ CORRECT (Physics-based):")
    print(correct_tensor)
    print(f"   xx = yy: {correct_tensor[0,0]:.3f} = {correct_tensor[1,1]:.3f}")
    print(f"   xy = 0: {correct_tensor[0,1]:.3f}")
    print("   ✅ Respects A1u trigonal symmetry")
    
    return problematic_tensor, correct_tensor

def create_tensor_fix_patch():
    """Create a patch for the main analyzer to fix tensor physics."""
    
    patch_code = '''
def create_physics_correct_tensor(self, wavenumber, intensity, symmetry=None, crystal_system=None, point_group=None):
    """Create a Raman tensor that respects proper crystal physics and symmetry."""
    
    # Get crystal information
    if not crystal_system:
        crystal_system = getattr(self, 'crystal_system', 'trigonal')
        if hasattr(self, 'crystal_structure') and self.crystal_structure:
            crystal_system = self.crystal_structure.get('crystal_system', 'trigonal')
    
    if not point_group:
        point_group = getattr(self, 'point_group', 'D3d')
        if hasattr(self, 'crystal_structure') and self.crystal_structure:
            point_group = self.crystal_structure.get('point_group', 'D3d')
    
    # Infer symmetry from frequency if not provided
    if not symmetry:
        symmetry = self._infer_mode_symmetry(wavenumber, crystal_system)
    
    print(f"Creating physics-correct tensor: {crystal_system} {point_group} {symmetry} @ {wavenumber:.1f} cm⁻¹")
    
    # Create tensor based on proper symmetry
    if crystal_system.lower() == 'trigonal' and point_group.upper() in ['D3D', 'D3']:
        if 'A1' in str(symmetry).upper():
            # A1 modes in trigonal: diagonal with xx = yy ≠ zz
            base_tensor = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.3]  # Strong c-axis anisotropy
            ])
        elif 'E' in str(symmetry).upper():
            # E modes in trigonal: different form
            base_tensor = np.array([
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0]
            ])
        else:
            # Default to A1-like
            base_tensor = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.3]
            ])
    
    elif crystal_system.lower() == 'cubic':
        # Cubic: isotropic
        base_tensor = np.eye(3)
    
    elif crystal_system.lower() == 'tetragonal':
        # Tetragonal: xx = yy ≠ zz
        base_tensor = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.7]
        ])
    
    else:
        # Default fallback
        base_tensor = np.eye(3)
    
    # Apply frequency-dependent scaling while preserving symmetry
    freq_factor = 1.0 + 0.1 * np.sin(wavenumber / 200.0)
    tensor = base_tensor * intensity * freq_factor
    
    # Ensure tensor is symmetric
    tensor = (tensor + tensor.T) / 2.0
    
    # Validate physics
    self._validate_tensor_symmetry(tensor, crystal_system, symmetry)
    
    return tensor

def _infer_mode_symmetry(self, wavenumber, crystal_system):
    """Infer vibrational mode symmetry from frequency."""
    
    if crystal_system.lower() == 'trigonal':
        # Calcite mode assignments
        if wavenumber < 300:
            return 'Eg'  # Lattice modes
        elif 300 <= wavenumber < 500:
            return 'A1g'  # CO3 bending
        elif 700 <= wavenumber < 900:
            return 'Eg'   # CO3 bending
        elif 1000 <= wavenumber < 1200:
            return 'A1u'  # CO3 symmetric stretch
        else:
            return 'A1g'  # Default
    
    return 'A1g'  # Default totally symmetric

def _validate_tensor_symmetry(self, tensor, crystal_system, symmetry):
    """Validate tensor obeys symmetry constraints."""
    
    if crystal_system.lower() == 'trigonal' and 'A1' in str(symmetry).upper():
        # Check A1 trigonal constraints
        if not np.isclose(tensor[0,0], tensor[1,1], atol=1e-6):
            print(f"⚠️  Warning: A1 trigonal tensor should have xx = yy")
        if not np.isclose(tensor[0,1], 0, atol=1e-6):
            print(f"⚠️  Warning: A1 trigonal tensor should have xy = 0")
        if not np.isclose(tensor[1,0], 0, atol=1e-6):
            print(f"⚠️  Warning: A1 trigonal tensor should have yx = 0")
'''
    
    return patch_code

def main():
    """Main function to demonstrate the tensor physics fix."""
    
    print("🧪 FIXING RAMAN TENSOR PHYSICS")
    print("=" * 60)
    
    # Demonstrate the fix
    problematic, correct = fix_calcite_a1u_tensor()
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Matrix visualizations
    im1 = ax1.imshow(problematic, cmap='RdBu_r', vmin=-2, vmax=2)
    ax1.set_title('❌ CURRENT (Problematic)\nA1u Calcite 1082.9 cm⁻¹')
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, f'{problematic[i,j]:.3f}', ha='center', va='center')
    
    im2 = ax2.imshow(correct, cmap='RdBu_r', vmin=-2, vmax=2)
    ax2.set_title('✅ CORRECT (Physics-based)\nA1u Calcite 1082.9 cm⁻¹')
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f'{correct[i,j]:.3f}', ha='center', va='center')
    
    # Polar plots
    angles = np.linspace(0, 2*np.pi, 100)
    
    # Problematic tensor polar plot
    intensities_prob = []
    for angle in angles:
        pol = [np.cos(angle), np.sin(angle), 0]
        intensity = np.abs(np.dot(pol, np.dot(problematic, pol))) ** 2
        intensities_prob.append(intensity)
    
    ax3 = plt.subplot(2, 2, 3, projection='polar')
    ax3.plot(angles, intensities_prob, 'r-', linewidth=2, label='Problematic')
    ax3.set_title('❌ Complex Lobed Pattern\n(Violates A1u symmetry)')
    
    # Correct tensor polar plot
    intensities_correct = []
    for angle in angles:
        pol = [np.cos(angle), np.sin(angle), 0]
        intensity = np.abs(np.dot(pol, np.dot(correct, pol))) ** 2
        intensities_correct.append(intensity)
    
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    ax4.plot(angles, intensities_correct, 'g-', linewidth=2, label='Correct')
    ax4.set_title('✅ Circular Pattern\n(Correct A1u symmetry)')
    
    plt.tight_layout()
    plt.savefig('tensor_physics_fix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate patch code
    patch = create_tensor_fix_patch()
    
    print(f"\n🔧 IMPLEMENTATION STEPS:")
    print("=" * 40)
    print("1. Replace create_mode_specific_tensor() with create_physics_correct_tensor()")
    print("2. Add symmetry validation functions")
    print("3. Use proper crystal physics constraints")
    print("4. Validate all generated tensors")
    
    print(f"\n📝 PATCH CODE GENERATED:")
    print("Save the following code to implement the fix...")
    
    with open('tensor_physics_patch.py', 'w') as f:
        f.write(patch)
    
    print("✅ Patch saved to 'tensor_physics_patch.py'")
    
    print(f"\n🎯 EXPECTED RESULTS AFTER FIX:")
    print("• A1u modes: Circular polar patterns, diagonal tensors")
    print("• E modes: Different tensor shapes, proper degeneracy")
    print("• All modes: Respect crystal symmetry constraints")
    print("• Peak-by-peak analysis: Physically meaningful tensor shapes")

if __name__ == "__main__":
    main() 