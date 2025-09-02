#!/usr/bin/env python3
"""
Map Visualizer for ML Raman Map Results
Handles proper coordinate scaling and visualization of classification results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import re
import os
from pathlib import Path

def extract_coordinates_from_filename(filename):
    """
    Extract X and Y coordinates from various filename formats.
    
    Supports formats like:
    - _Y123_X456
    - Y123_X456
    - horiz_edge_01_Y02_X063
    """
    # Try multiple regex patterns
    patterns = [
        r'_Y(\d+)_X(\d+)',  # Standard format
        r'Y(\d+)_X(\d+)',   # Without leading underscore
        r'_Y(\d+).*_X(\d+)', # With text between Y and X
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            y = int(match.group(1))
            x = int(match.group(2))
            return x, y
    
    # If no pattern matches, try to parse position from hash
    print(f"Could not parse position from filename {filename}, using hash-based position {hash(filename) % 1000}")
    return None, None

def create_proper_map_visualization(results_df, output_path=None, title_prefix=""):
    """
    Create properly scaled map visualizations from classification results.
    
    Parameters:
    -----------
    results_df : DataFrame
        Results with x, y coordinates and classification data
    output_path : str, optional
        Path to save the visualization
    title_prefix : str
        Prefix for plot titles
    """
    if len(results_df) < 4:
        print("Not enough data points for visualization")
        return
    
    # Extract coordinates and ensure they're numeric
    x_coords = pd.to_numeric(results_df['x'], errors='coerce')
    y_coords = pd.to_numeric(results_df['y'], errors='coerce')
    
    # Remove rows with invalid coordinates
    valid_coords = ~(np.isnan(x_coords) | np.isnan(y_coords))
    if not valid_coords.any():
        print("No valid coordinates found")
        return
    
    df_valid = results_df[valid_coords].copy()
    x_coords = x_coords[valid_coords]
    y_coords = y_coords[valid_coords]
    
    print(f"Visualizing {len(df_valid)} valid data points")
    print(f"X range: {x_coords.min()} to {x_coords.max()}")
    print(f"Y range: {y_coords.min()} to {y_coords.max()}")
    
    # Determine actual coordinate ranges
    x_min, x_max = int(x_coords.min()), int(x_coords.max())
    y_min, y_max = int(y_coords.min()), int(y_coords.max())
    
    # Calculate grid dimensions
    grid_width = x_max - x_min + 1
    grid_height = y_max - y_min + 1
    
    print(f"Grid dimensions: {grid_width} x {grid_height}")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{title_prefix}ML Raman Map Analysis Results', fontsize=16)
    
    # Define extent for proper coordinate scaling
    extent = [x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5]
    
    # 1. Classification Map
    ax1 = axes[0, 0]
    if 'Prediction' in df_valid.columns:
        class_grid = np.full((grid_height, grid_width), np.nan)
        
        for _, row in df_valid.iterrows():
            x, y = int(row['x']), int(row['y'])
            grid_x, grid_y = x - x_min, y - y_min
            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                class_value = 1 if row['Prediction'] == 'Class A' else 0
                class_grid[grid_y, grid_x] = class_value
        
        # Create custom colormap that handles NaN
        colors = ['blue', 'red']
        cmap = ListedColormap(colors)
        cmap.set_bad('lightgray', 1.0)  # Color for NaN values
        
        im1 = ax1.imshow(class_grid, cmap=cmap, interpolation='nearest', 
                        origin='lower', extent=extent, vmin=0, vmax=1)
        ax1.set_title('Classification Map\n(Blue: Class B, Red: Class A)')
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1])
        cbar1.set_ticklabels(['Class B', 'Class A'])
    
    # 2. Confidence Map
    ax2 = axes[0, 1]
    if 'Confidence' in df_valid.columns:
        conf_grid = np.full((grid_height, grid_width), np.nan)
        
        for _, row in df_valid.iterrows():
            x, y = int(row['x']), int(row['y'])
            grid_x, grid_y = x - x_min, y - y_min
            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                conf_grid[grid_y, grid_x] = row['Confidence']
        
        im2 = ax2.imshow(conf_grid, cmap='viridis', interpolation='nearest',
                        origin='lower', extent=extent)
        plt.colorbar(im2, ax=ax2, label='Confidence')
        ax2.set_title('Confidence Map')
    
    # 3. Dominant Peak Map (if available)
    ax3 = axes[1, 0]
    if 'Dominant Peak' in df_valid.columns:
        peak_grid = np.full((grid_height, grid_width), np.nan)
        
        for _, row in df_valid.iterrows():
            x, y = int(row['x']), int(row['y'])
            grid_x, grid_y = x - x_min, y - y_min
            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                peak_grid[grid_y, grid_x] = row['Dominant Peak']
        
        im3 = ax3.imshow(peak_grid, cmap='plasma', interpolation='nearest',
                        origin='lower', extent=extent)
        plt.colorbar(im3, ax=ax3, label='Wavenumber (cm⁻¹)')
        ax3.set_title('Dominant Peak Map')
    else:
        ax3.text(0.5, 0.5, 'Dominant Peak\nData Not Available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Dominant Peak Map')
    
    # 4. Data Coverage Map
    ax4 = axes[1, 1]
    coverage_grid = np.zeros((grid_height, grid_width))
    
    for _, row in df_valid.iterrows():
        x, y = int(row['x']), int(row['y'])
        grid_x, grid_y = x - x_min, y - y_min
        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
            coverage_grid[grid_y, grid_x] = 1
    
    im4 = ax4.imshow(coverage_grid, cmap='Greys', interpolation='nearest',
                    origin='lower', extent=extent, vmin=0, vmax=1)
    ax4.set_title(f'Data Coverage\n({len(df_valid)} points)')
    
    # Set consistent labels for all subplots
    for ax in axes.flat:
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    return fig

def process_and_visualize_results(csv_file, output_dir=None):
    """
    Load results CSV and create proper visualizations.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file with results
    output_dir : str, optional
        Directory to save visualizations
    """
    try:
        # Load results
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} results from {csv_file}")
        
        # Extract coordinates from filenames if x,y columns don't exist
        if 'x' not in df.columns or 'y' not in df.columns:
            print("Extracting coordinates from filenames...")
            coords = []
            for filename in df['Filename'] if 'Filename' in df.columns else df['file_path']:
                x, y = extract_coordinates_from_filename(filename)
                coords.append({'x': x, 'y': y})
            
            coord_df = pd.DataFrame(coords)
            df = pd.concat([df, coord_df], axis=1)
        
        # Remove rows with invalid coordinates
        df = df.dropna(subset=['x', 'y'])
        
        if len(df) == 0:
            print("No valid coordinates found in data")
            return None
        
        # Create visualization
        if output_dir is None:
            output_dir = os.path.dirname(csv_file)
        
        output_path = os.path.join(output_dir, 'proper_map_visualization.png')
        fig = create_proper_map_visualization(df, output_path)
        
        # Print summary
        print(f"\nVisualization Summary:")
        print(f"- Data points: {len(df)}")
        print(f"- X range: {df['x'].min()} to {df['x'].max()}")
        print(f"- Y range: {df['y'].min()} to {df['y'].max()}")
        
        if 'Prediction' in df.columns:
            class_counts = df['Prediction'].value_counts()
            print(f"- Class distribution:")
            for class_name, count in class_counts.items():
                print(f"  {class_name}: {count} ({count/len(df):.1%})")
        
        return fig
        
    except Exception as e:
        print(f"Error processing results: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        process_and_visualize_results(csv_file, output_dir)
    else:
        print("Usage: python map_visualizer.py <csv_file> [output_dir]")
        print("Example: python map_visualizer.py unknown_spectra_results.csv ./output/")
