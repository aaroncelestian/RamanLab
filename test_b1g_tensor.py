#!/usr/bin/env python3
"""
Test script to verify B1g tensor creation and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_b1g_tensor(intensity=1.0):
    """Create a proper B1g tensor for tetragonal crystals."""
    # B1g: (x²-y²) character - creates 4-lobed dumbbell pattern
    tensor = np.array([
        [intensity * 3.0, 0, 0],        # Strong xx component
        [0, -intensity * 3.0, 0],       # Opposite yy component (creates nodes)
        [0, 0, 0]                       # No zz component
    ])
    return tensor

def test_b1g_intensity_pattern(tensor):
    """Test the intensity pattern for B1g tensor."""
    print("Testing B1g intensity pattern:")
    print("Tensor matrix:")
    print(tensor)
    print()
    
    # Test key directions
    test_directions = [
        ([1, 0, 0], "x-direction"),
        ([0, 1, 0], "y-direction"), 
        ([1/np.sqrt(2), 1/np.sqrt(2), 0], "45° in xy-plane"),
        ([1/np.sqrt(2), -1/np.sqrt(2), 0], "-45° in xy-plane")
    ]
    
    for direction, name in test_directions:
        e_vec = np.array(direction)
        amplitude = np.dot(e_vec, np.dot(tensor, e_vec))
        intensity = amplitude**2
        
        print(f"{name:20s}: amplitude = {amplitude:8.3f}, intensity = {intensity:8.3f}")
    
    print()

def visualize_b1g_tensor(tensor, character="B1g"):
    """Visualize the B1g tensor pattern."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create spherical coordinate grid
    theta = np.linspace(0, np.pi, 40)  # polar angle
    phi = np.linspace(0, 2*np.pi, 60)  # azimuthal angle
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # Convert to Cartesian unit vectors
    x_dir = np.sin(theta_grid) * np.cos(phi_grid)
    y_dir = np.sin(theta_grid) * np.sin(phi_grid)
    z_dir = np.cos(theta_grid)
    
    # Calculate Raman amplitude for each direction
    intensity = np.zeros_like(x_dir)
    
    for j in range(x_dir.shape[0]):
        for k in range(x_dir.shape[1]):
            # Unit vector in this direction
            e_vec = np.array([x_dir[j,k], y_dir[j,k], z_dir[j,k]])
            
            # Raman amplitude: e^T · R · e (keep sign for B1g visualization)
            raman_amplitude = np.dot(e_vec, np.dot(tensor, e_vec))
            intensity[j,k] = raman_amplitude
    
    # Handle the signed intensity for B1g (preserve nodes)
    max_abs_intensity = np.max(np.abs(intensity))
    if max_abs_intensity > 0:
        intensity_norm = intensity / max_abs_intensity
    else:
        intensity_norm = np.ones_like(intensity) * 0.1
    
    # Scale radius: show both positive and negative lobes
    radius_scale = 0.3 + 0.7 * np.abs(intensity_norm)
    
    # Calculate surface coordinates
    x_surface = radius_scale * x_dir
    y_surface = radius_scale * y_dir
    z_surface = radius_scale * z_dir
    
    # Use diverging colormap to show positive (red) and negative (blue) lobes
    colors = plt.cm.RdBu_r(0.5 + 0.5 * intensity_norm)
    
    # Plot the 3D surface
    surf = ax.plot_surface(x_surface, y_surface, z_surface, 
                         facecolors=colors, alpha=0.8, 
                         linewidth=0, antialiased=True)
    
    # Add wireframe for structure
    ax.plot_wireframe(x_surface, y_surface, z_surface, 
                    color='gray', alpha=0.3, linewidth=0.5)
    
    # Add coordinate system arrows
    arrow_length = 1.2
    ax.quiver(0, 0, 0, arrow_length, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2, alpha=0.8)
    ax.quiver(0, 0, 0, 0, arrow_length, 0, color='green', arrow_length_ratio=0.1, linewidth=2, alpha=0.8)
    ax.quiver(0, 0, 0, 0, 0, arrow_length, color='blue', arrow_length_ratio=0.1, linewidth=2, alpha=0.8)
    
    # Set equal aspect ratio
    max_extent = 1.5
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_zlim(-max_extent, max_extent)
    
    # Labels and title
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    
    title = f'{character} Raman Tensor\n4-Lobed Pattern (Red=Positive, Blue=Negative)'
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("Testing B1g Tensor Visualization")
    print("=" * 40)
    
    # Create B1g tensor
    b1g_tensor = create_b1g_tensor(intensity=1.0)
    
    # Test intensity pattern
    test_b1g_intensity_pattern(b1g_tensor)
    
    # Create visualization
    fig = visualize_b1g_tensor(b1g_tensor)
    
    print("Generated B1g tensor visualization.")
    print("Expected pattern: 4-lobed shape with positive lobes along x-axis (red)")
    print("                  and negative lobes along y-axis (blue)")
    print("                  with nodes at 45° angles")
    
    plt.show() 