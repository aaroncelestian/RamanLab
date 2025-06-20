#!/usr/bin/env python3
"""
Debug script for 3D tensor surface rendering

This script tests the tensor surface rendering to identify why
the tensor surfaces are not appearing in the 3D visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test_tensor_surface_rendering():
    """Test basic tensor surface rendering."""
    print("Testing 3D tensor surface rendering...")
    
    # Create test tensor (B1g-like)
    tensor = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    
    print(f"Test tensor shape: {tensor.shape}")
    print(f"Test tensor:\n{tensor}")
    
    # Create spherical coordinate grid
    theta = np.linspace(0, np.pi, 20)
    phi = np.linspace(0, 2*np.pi, 30)
    theta_mesh, phi_mesh = np.meshgrid(theta, phi)
    
    # Convert to Cartesian
    x_dir = np.sin(theta_mesh) * np.cos(phi_mesh)
    y_dir = np.sin(theta_mesh) * np.sin(phi_mesh)
    z_dir = np.cos(theta_mesh)
    
    print(f"Grid shapes: x_dir={x_dir.shape}, y_dir={y_dir.shape}, z_dir={z_dir.shape}")
    
    # Calculate Raman intensity for each direction
    intensity = np.zeros_like(x_dir)
    
    for i in range(x_dir.shape[0]):
        for j in range(x_dir.shape[1]):
            e_vec = np.array([x_dir[i,j], y_dir[i,j], z_dir[i,j]])
            # Raman intensity: e · tensor · e
            raman_amplitude = np.dot(e_vec, np.dot(tensor, e_vec))
            intensity[i,j] = raman_amplitude  # Keep sign for B1g
    
    print(f"Intensity range: {np.min(intensity):.3f} to {np.max(intensity):.3f}")
    
    # Normalize intensity
    intensity = np.real(intensity)
    max_abs = np.max(np.abs(intensity))
    if max_abs > 0:
        intensity_norm = intensity / max_abs
    else:
        intensity_norm = np.ones_like(intensity) * 0.1
        
    print(f"Normalized intensity range: {np.min(intensity_norm):.3f} to {np.max(intensity_norm):.3f}")
    
    # Scale radius
    radius_scale = 0.4 + 0.6 * np.abs(intensity_norm)
    
    # Calculate surface coordinates
    x_surface = radius_scale * x_dir
    y_surface = radius_scale * y_dir
    z_surface = radius_scale * z_dir
    
    print(f"Surface coordinate ranges:")
    print(f"  X: {np.min(x_surface):.3f} to {np.max(x_surface):.3f}")
    print(f"  Y: {np.min(y_surface):.3f} to {np.max(y_surface):.3f}")
    print(f"  Z: {np.min(z_surface):.3f} to {np.max(z_surface):.3f}")
    
    # Create plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color mapping
    colors = plt.cm.RdBu_r(0.5 + 0.5 * intensity_norm)
    
    # Plot the tensor surface
    surf = ax.plot_surface(x_surface, y_surface, z_surface, 
                          facecolors=colors, alpha=0.7, 
                          linewidth=0, antialiased=True)
    
    # Add coordinate system
    ax.quiver(0, 0, 0, 1.5, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2)
    ax.quiver(0, 0, 0, 0, 1.5, 0, color='green', arrow_length_ratio=0.1, linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, 1.5, color='blue', arrow_length_ratio=0.1, linewidth=2)
    
    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Test B1g Tensor Surface\n(Should show 4-lobed pattern)')
    
    # Set equal aspect ratio
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('test_tensor_surface.png', dpi=150, bbox_inches='tight')
    print("✓ Test plot saved as 'test_tensor_surface.png'")
    
    plt.show()
    
    return True

def test_isotropic_tensor():
    """Test isotropic tensor surface (A1g-like)."""
    print("\nTesting isotropic tensor (A1g-like)...")
    
    # Create isotropic tensor
    tensor = np.eye(3)
    
    # Create spherical coordinate grid
    theta = np.linspace(0, np.pi, 20)
    phi = np.linspace(0, 2*np.pi, 30)
    theta_mesh, phi_mesh = np.meshgrid(theta, phi)
    
    # Convert to Cartesian
    x_dir = np.sin(theta_mesh) * np.cos(phi_mesh)
    y_dir = np.sin(theta_mesh) * np.sin(phi_mesh)
    z_dir = np.cos(theta_mesh)
    
    # Calculate intensity
    intensity = np.zeros_like(x_dir)
    for i in range(x_dir.shape[0]):
        for j in range(x_dir.shape[1]):
            e_vec = np.array([x_dir[i,j], y_dir[i,j], z_dir[i,j]])
            raman_amplitude = np.dot(e_vec, np.dot(tensor, e_vec))
            intensity[i,j] = raman_amplitude**2  # Square for A1g
    
    # This should be constant (spherical)
    print(f"A1g intensity range: {np.min(intensity):.3f} to {np.max(intensity):.3f}")
    print(f"Expected: should be constant (~1.0)")
    
    return True

if __name__ == "__main__":
    print("=== 3D Tensor Surface Debug Test ===")
    test_tensor_surface_rendering()
    test_isotropic_tensor()
    print("Debug test complete!") 