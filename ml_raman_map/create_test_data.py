#!/usr/bin/env python3
"""
Script to generate test data for the unknown classification feature.
This creates a directory with sample spectrum files in the required format.
"""

import os
import numpy as np
import argparse
from pathlib import Path

def generate_spectrum(center=1000, intensity=100, noise_level=0.1):
    """Generate a synthetic Raman spectrum with a Gaussian peak."""
    # Create wavenumber range from 200 to 3000 cm^-1
    wavenumbers = np.linspace(200, 3000, 500)
    
    # Generate a Gaussian peak
    intensities = intensity * np.exp(-0.5 * ((wavenumbers - center) / 100)**2)
    
    # Add some noise
    intensities += noise_level * intensity * np.random.randn(len(wavenumbers))
    
    # Ensure positive values
    intensities = np.maximum(intensities, 0)
    
    return wavenumbers, intensities

def write_spectrum(filename, wavenumbers, intensities):
    """Write a spectrum to a tab-delimited text file."""
    with open(filename, 'w') as f:
        for w, i in zip(wavenumbers, intensities):
            f.write(f"{w:.1f}\t{i:.1f}\n")

def create_test_directory(output_dir, grid_size=5, class_a_center=1000, class_b_center=1500):
    """
    Create a directory with test spectrum files.
    
    Parameters:
    -----------
    output_dir : str
        Path to output directory
    grid_size : int
        Size of the grid (creates grid_size x grid_size spectra)
    class_a_center : float
        Center wavenumber for Class A spectra
    class_b_center : float
        Center wavenumber for Class B spectra
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate spectra for each grid position
    for y in range(grid_size):
        for x in range(grid_size):
            # Determine class based on position (checkerboard pattern)
            is_class_a = (x + y) % 2 == 0
            
            # Generate spectrum
            if is_class_a:
                wavenumbers, intensities = generate_spectrum(center=class_a_center, intensity=100)
            else:
                wavenumbers, intensities = generate_spectrum(center=class_b_center, intensity=80)
            
            # Create filename with position encoded
            filename = os.path.join(output_dir, f"spectrum_Y{y+1:03d}_X{x+1:03d}.txt")
            
            # Write spectrum to file
            write_spectrum(filename, wavenumbers, intensities)
            
            print(f"Created {filename} (Class {'A' if is_class_a else 'B'})")
    
    # Create README file
    readme_path = os.path.join(output_dir, "README.txt")
    with open(readme_path, 'w') as f:
        f.write("Test Raman Spectra\n")
        f.write("=================\n\n")
        f.write(f"This directory contains {grid_size*grid_size} test spectra.\n")
        f.write(f"- Class A spectra have peaks centered at {class_a_center} cm^-1\n")
        f.write(f"- Class B spectra have peaks centered at {class_b_center} cm^-1\n\n")
        f.write("The spectra are arranged in a checkerboard pattern (Class A and B alternate).\n")
        f.write("Filenames encode the X,Y coordinates for visualization.\n\n")
        f.write("To use these files with RamanLab:\n")
        f.write("1. Open the 2D Map Analysis window\n")
        f.write("2. Click 'Classify Unknowns & Visualize'\n")
        f.write("3. Select this directory\n")
    
    print(f"\nCreated {grid_size*grid_size} test spectra in {output_dir}")
    print(f"See {readme_path} for details")

def main():
    parser = argparse.ArgumentParser(description="Generate test data for Raman classification")
    parser.add_argument("output_dir", help="Directory to create test files in")
    parser.add_argument("--grid-size", type=int, default=5, help="Size of grid (default: 5)")
    parser.add_argument("--class-a-center", type=float, default=1000, help="Center wavenumber for Class A (default: 1000)")
    parser.add_argument("--class-b-center", type=float, default=1500, help="Center wavenumber for Class B (default: 1500)")
    
    args = parser.parse_args()
    
    create_test_directory(
        args.output_dir,
        grid_size=args.grid_size,
        class_a_center=args.class_a_center,
        class_b_center=args.class_b_center
    )

if __name__ == "__main__":
    main() 