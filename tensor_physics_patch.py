
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
