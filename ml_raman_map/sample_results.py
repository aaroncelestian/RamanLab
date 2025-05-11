#!/usr/bin/env python3
"""
Create a sample CSV file for testing the 2D map visualization features.
This script generates a CSV with x, y coordinates and classification results.
"""

import os
import pandas as pd
import numpy as np
import argparse
import re
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import glob

def extract_coordinates_from_filename(filename):
    """Extract X and Y coordinates from a filename with format _Y123_X456."""
    x_match = re.search(r'_X(\d+)', filename)
    y_match = re.search(r'_Y(\d+)', filename)
    
    if x_match and y_match:
        x = int(x_match.group(1))
        y = int(y_match.group(1))
        return x, y
    
    return None, None

def create_sample_spectra(directory, num_files=25, overwrite=False):
    """
    Create sample spectra files for testing.
    
    Parameters:
    -----------
    directory : str
        Directory to save the spectra files
    num_files : int
        Number of files to create
    overwrite : bool
        Whether to overwrite existing files
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    grid_size = int(np.sqrt(num_files))
    if grid_size**2 != num_files:
        grid_size = int(np.ceil(np.sqrt(num_files)))
        print(f"Adjusting to grid size {grid_size}x{grid_size} for {num_files} files")
    
    wavenumbers = np.linspace(200, 3000, 500)
    
    # Create some peak positions that differentiate classes
    class_a_peaks = [500, 1000, 1500]
    class_b_peaks = [700, 1200, 1800]
    
    files_created = 0
    
    for y in range(grid_size):
        for x in range(grid_size):
            if files_created >= num_files:
                break
                
            # Create filename with coordinates
            filename = f"sample_Y{y:03d}_X{x:03d}.txt"
            filepath = os.path.join(directory, filename)
            
            # Skip if file exists and not overwriting
            if os.path.exists(filepath) and not overwrite:
                print(f"Skipping existing file: {filepath}")
                files_created += 1
                continue
            
            # Determine class based on position (e.g., diagonal pattern)
            is_class_a = x > y
            
            # Create spectra with appropriate peaks
            intensities = np.zeros_like(wavenumbers)
            baseline = 0.1 + 0.05 * np.random.random()
            
            # Add baseline and noise
            intensities += baseline + 0.02 * np.random.randn(len(wavenumbers))
            
            # Add peaks based on class
            if is_class_a:
                for peak in class_a_peaks:
                    peak_height = 0.5 + 0.3 * np.random.random()
                    peak_width = 30 + 10 * np.random.random()
                    peak_center = peak + 20 * np.random.randn()
                    intensities += peak_height * np.exp(-(wavenumbers - peak_center)**2 / (2 * peak_width**2))
            else:
                for peak in class_b_peaks:
                    peak_height = 0.4 + 0.4 * np.random.random()
                    peak_width = 25 + 15 * np.random.random()
                    peak_center = peak + 20 * np.random.randn()
                    intensities += peak_height * np.exp(-(wavenumbers - peak_center)**2 / (2 * peak_width**2))
            
            # Normalize
            intensities = intensities / np.max(intensities)
            
            # Save to file
            with open(filepath, 'w') as f:
                for wn, intensity in zip(wavenumbers, intensities):
                    f.write(f"{wn:.2f}\t{intensity:.6f}\n")
            
            files_created += 1
    
    print(f"Created {files_created} sample spectra files in {directory}")
    return files_created

def create_sample_results(directory, output_csv, grid_size=5, create_spectra=False):
    """
    Create a sample results CSV file with the expected format.
    
    Parameters:
    -----------
    directory : str
        Directory containing the spectra files (or where they would be)
    output_csv : str
        Path to the output CSV file
    grid_size : int
        Size of the 2D grid to generate if no files are found
    create_spectra : bool
        Whether to create sample spectra files if none exist
    """
    files = []
    
    # Check if directory exists
    if os.path.exists(directory):
        # Look for .txt files in the directory
        for filename in os.listdir(directory):
            if filename.lower().endswith('.txt'):
                files.append(filename)
    
    # Create sample spectra if requested and no files found
    if not files and create_spectra:
        print(f"Creating sample spectra files in {directory}")
        num_files = grid_size * grid_size
        create_sample_spectra(directory, num_files=num_files)
        
        # Refresh file list
        if os.path.exists(directory):
            files = [f for f in os.listdir(directory) if f.lower().endswith('.txt')]
    
    # Create results DataFrame
    if files:
        # Use real files
        print(f"Found {len(files)} spectra files in directory")
        results = []
        
        for filename in files:
            # Try to extract x, y from filename
            x, y = extract_coordinates_from_filename(filename)
            
            # If couldn't extract coordinates, assign random positions
            if x is None or y is None:
                x = np.random.randint(0, grid_size)
                y = np.random.randint(0, grid_size)
            
            # Process the file to determine spectral features
            # This is just simulated for the example
            filepath = os.path.join(directory, filename)
            if os.path.exists(filepath):
                try:
                    # Try to read the actual file
                    data = np.loadtxt(filepath)
                    if data.shape[1] >= 2:
                        wavenumbers = data[:, 0]
                        intensities = data[:, 1]
                        
                        # Calculate some spectral features
                        max_intensity = np.max(intensities)
                        max_position = wavenumbers[np.argmax(intensities)]
                        mean_intensity = np.mean(intensities)
                        
                        # Determine class based on spectral features
                        # This is a simple rule - in real life, use a trained model
                        if max_position < 1000:
                            prediction = "Class A"
                            class_a_prob = 0.7 + 0.2 * np.random.random()
                        else:
                            prediction = "Class B"
                            class_a_prob = 0.3 * np.random.random()
                            
                        confidence = max(class_a_prob, 1 - class_a_prob)
                        
                        # Add extra metrics for spectrum verification
                        dominant_peak = max_position
                        secondary_peak = wavenumbers[np.argsort(intensities)[-2]]
                        intensity_ratio = intensities[np.argmax(intensities)] / np.mean(intensities)
                    else:
                        # Fallback for invalid data
                        max_intensity = np.random.random()
                        mean_intensity = max_intensity * 0.7
                        dominant_peak = 500 + 1000 * np.random.random()
                        secondary_peak = 500 + 1000 * np.random.random()
                        intensity_ratio = 1.5 + np.random.random()
                        
                        # Randomly assign class and probability
                        class_a_prob = np.random.random()
                        prediction = "Class A" if class_a_prob > 0.5 else "Class B"
                        confidence = max(class_a_prob, 1 - class_a_prob)
                except Exception as e:
                    print(f"Error reading {filepath}: {str(e)}")
                    # Fallback for error
                    max_intensity = np.random.random()
                    mean_intensity = max_intensity * 0.7
                    dominant_peak = 500 + 1000 * np.random.random()
                    secondary_peak = 500 + 1000 * np.random.random()
                    intensity_ratio = 1.5 + np.random.random()
                    
                    # Randomly assign class and probability
                    class_a_prob = np.random.random()
                    prediction = "Class A" if class_a_prob > 0.5 else "Class B"
                    confidence = max(class_a_prob, 1 - class_a_prob)
            else:
                # Fallback if file doesn't exist
                max_intensity = np.random.random()
                mean_intensity = max_intensity * 0.7
                dominant_peak = 500 + 1000 * np.random.random()
                secondary_peak = 500 + 1000 * np.random.random()
                intensity_ratio = 1.5 + np.random.random()
                
                # Randomly assign class and probability
                class_a_prob = np.random.random()
                prediction = "Class A" if class_a_prob > 0.5 else "Class B"
                confidence = max(class_a_prob, 1 - class_a_prob)
            
            results.append({
                'Filename': filename,
                'x': x,
                'y': y,
                'Prediction': prediction,
                'Confidence': confidence,
                'Class A Probability': class_a_prob,
                'Class B Probability': 1 - class_a_prob,
                'Max Intensity': max_intensity,
                'Mean Intensity': mean_intensity,
                'Dominant Peak': dominant_peak,
                'Secondary Peak': secondary_peak,
                'Intensity Ratio': intensity_ratio,
                'Full Path': os.path.abspath(os.path.join(directory, filename))
            })
        
        df = pd.DataFrame(results)
        
    else:
        # Generate synthetic data in a grid pattern
        print(f"No files found, generating synthetic {grid_size}x{grid_size} grid data")
        results = []
        
        for y in range(grid_size):
            for x in range(grid_size):
                filename = f"sample_Y{y:03d}_X{x:03d}.txt"
                
                # Create patterns in the data (e.g., diagonal split between classes)
                if x > y:
                    prediction = "Class A"
                    class_a_prob = 0.7 + 0.2 * np.random.random()
                    dominant_peak = 500 + 200 * np.random.random()  # Class A peaks
                else:
                    prediction = "Class B"
                    class_a_prob = 0.3 * np.random.random()
                    dominant_peak = 1200 + 300 * np.random.random()  # Class B peaks
                
                confidence = max(class_a_prob, 1 - class_a_prob)
                
                # Generate additional metrics for verification
                max_intensity = 0.8 + 0.2 * np.random.random() 
                mean_intensity = 0.3 + 0.2 * np.random.random()
                secondary_peak = 800 + 1000 * np.random.random()
                intensity_ratio = 1.5 + 1.0 * np.random.random()
                
                results.append({
                    'Filename': filename,
                    'x': x,
                    'y': y,
                    'Prediction': prediction,
                    'Confidence': confidence,
                    'Class A Probability': class_a_prob,
                    'Class B Probability': 1 - class_a_prob,
                    'Max Intensity': max_intensity,
                    'Mean Intensity': mean_intensity,
                    'Dominant Peak': dominant_peak,
                    'Secondary Peak': secondary_peak,
                    'Intensity Ratio': intensity_ratio,
                    'Full Path': os.path.abspath(os.path.join(directory, filename))
                })
        
        df = pd.DataFrame(results)
    
    # Save the DataFrame to CSV
    df.to_csv(output_csv, index=False)
    print(f"Created sample results CSV with {len(df)} entries at {output_csv}")
    
    # Display summary
    class_counts = df['Prediction'].value_counts()
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"- {class_name}: {count} ({count/len(df):.1%})")
    
    # Create visualization of the results
    create_visualization(df, directory)
    
    return df

def create_visualization(df, directory):
    """Create visualization of the classification results."""
    # Check if we have enough data for a proper visualization
    if len(df) < 4:
        print("Not enough data points for visualization")
        return
        
    # Extract x, y coordinates and class information
    x_coords = df['x'].values
    y_coords = df['y'].values
    classes = df['Prediction'].values
    confidences = df['Confidence'].values
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Class map
    ax1 = axes[0, 0]
    class_values = np.zeros_like(classes, dtype=int)
    class_values[classes == 'Class A'] = 1
    
    # Determine grid size
    grid_size_x = int(np.max(x_coords)) + 1
    grid_size_y = int(np.max(y_coords)) + 1
    
    # Create grid for plotting
    grid = np.zeros((grid_size_y, grid_size_x))
    for i in range(len(df)):
        x, y = int(x_coords[i]), int(y_coords[i])
        if 0 <= x < grid_size_x and 0 <= y < grid_size_y:
            grid[y, x] = class_values[i]
    
    # Plot class map
    cmap = ListedColormap(['blue', 'red'])
    im1 = ax1.imshow(grid, cmap=cmap, interpolation='nearest', origin='lower')
    ax1.set_title('Class Map (Blue: Class B, Red: Class A)')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    
    # Add text labels
    for i in range(grid_size_y):
        for j in range(grid_size_x):
            if j < grid.shape[1] and i < grid.shape[0]:
                text = 'A' if grid[i, j] == 1 else 'B'
                ax1.text(j, i, text, ha='center', va='center', color='white')
    
    # 2. Confidence map
    ax2 = axes[0, 1]
    confidence_grid = np.zeros((grid_size_y, grid_size_x))
    for i in range(len(df)):
        x, y = int(x_coords[i]), int(y_coords[i])
        if 0 <= x < grid_size_x and 0 <= y < grid_size_y:
            confidence_grid[y, x] = confidences[i]
    
    im2 = ax2.imshow(confidence_grid, cmap='viridis', interpolation='nearest', origin='lower')
    fig.colorbar(im2, ax=ax2)
    ax2.set_title('Confidence Map')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    
    # 3. Dominant peak map
    ax3 = axes[1, 0]
    peak_grid = np.zeros((grid_size_y, grid_size_x))
    for i in range(len(df)):
        x, y = int(x_coords[i]), int(y_coords[i])
        if 0 <= x < grid_size_x and 0 <= y < grid_size_y:
            peak_grid[y, x] = df['Dominant Peak'].iloc[i]
    
    im3 = ax3.imshow(peak_grid, cmap='plasma', interpolation='nearest', origin='lower')
    fig.colorbar(im3, ax=ax3)
    ax3.set_title('Dominant Peak Map (cm⁻¹)')
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    
    # 4. Intensity ratio map
    ax4 = axes[1, 1]
    ratio_grid = np.zeros((grid_size_y, grid_size_x))
    for i in range(len(df)):
        x, y = int(x_coords[i]), int(y_coords[i])
        if 0 <= x < grid_size_x and 0 <= y < grid_size_y:
            ratio_grid[y, x] = df['Intensity Ratio'].iloc[i]
    
    im4 = ax4.imshow(ratio_grid, cmap='inferno', interpolation='nearest', origin='lower')
    fig.colorbar(im4, ax=ax4)
    ax4.set_title('Intensity Ratio Map')
    ax4.set_xlabel('X Position')
    ax4.set_ylabel('Y Position')
    
    # Adjust layout and save
    plt.tight_layout()
    output_dir = os.path.dirname(os.path.abspath(directory))
    vis_path = os.path.join(output_dir, 'classification_visualization.png')
    plt.savefig(vis_path, dpi=150)
    print(f"Saved visualization to {vis_path}")
    
    # Close the figure to free memory
    plt.close(fig)

def main():
    """Main function to parse arguments and create sample results."""
    parser = argparse.ArgumentParser(description="Create sample classification results CSV")
    parser.add_argument("--directory", "-d", default="./unknown_dir",
                       help="Directory containing spectral files (or where they would be)")
    parser.add_argument("--output", "-o", default="unknown_spectra_results.csv",
                       help="Output CSV file path")
    parser.add_argument("--grid-size", "-g", type=int, default=5,
                       help="Size of grid for synthetic data (default: 5)")
    parser.add_argument("--create-spectra", "-c", action="store_true",
                       help="Create sample spectra files if none exist")
    
    args = parser.parse_args()
    
    create_sample_results(args.directory, args.output, args.grid_size, args.create_spectra)

if __name__ == "__main__":
    main() 