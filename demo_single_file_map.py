"""Demo script to load and analyze single-file 2D Raman map data.

This script demonstrates how to:
1. Load a single-file Raman map
2. Visualize the map dimensions and data
3. Prepare data for cluster analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import the single-file loader
from map_analysis_2d.core.single_file_map_loader import SingleFileRamanMapData
from map_analysis_2d.core.cosmic_ray_detection import CosmicRayConfig


def progress_callback(progress, message):
    """Simple progress callback."""
    print(f"[{progress:3d}%] {message}")


def visualize_map_overview(map_data):
    """Create overview visualization of the map."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Map dimensions info
    ax = axes[0, 0]
    ax.axis('off')
    info_text = f"""
Map Information:
━━━━━━━━━━━━━━━━━━━━━━
Total Spectra: {len(map_data.spectra)}
X Positions: {len(map_data.x_positions)}
Y Positions: {len(map_data.y_positions)}
Grid: {map_data.width} × {map_data.height}

X Range: {map_data.x_positions[0]:.2f} to {map_data.x_positions[-1]:.2f}
Y Range: {map_data.y_positions[0]:.2f} to {map_data.y_positions[-1]:.2f}

Wavenumber Range: {map_data.target_wavenumbers[0]:.1f} to {map_data.target_wavenumbers[-1]:.1f} cm⁻¹
Spectral Points: {len(map_data.target_wavenumbers)}
"""
    ax.text(0.1, 0.5, info_text, fontsize=11, family='monospace', 
            verticalalignment='center')
    ax.set_title('Map Overview', fontsize=14, fontweight='bold')
    
    # 2. Sample spectrum
    ax = axes[0, 1]
    # Get a spectrum from the middle of the map
    mid_x = map_data.x_positions[len(map_data.x_positions)//2]
    mid_y = map_data.y_positions[len(map_data.y_positions)//2]
    spectrum = map_data.get_spectrum(mid_x, mid_y)
    
    if spectrum:
        ax.plot(spectrum.wavenumbers, spectrum.intensities, 'b-', linewidth=0.8)
        ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=10)
        ax.set_ylabel('Intensity (a.u.)', fontsize=10)
        ax.set_title(f'Sample Spectrum\nPosition: ({mid_x:.1f}, {mid_y:.1f})', 
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 3. Integrated intensity map
    ax = axes[1, 0]
    intensity_map = np.zeros((len(map_data.y_positions), len(map_data.x_positions)))
    
    for i, y_pos in enumerate(map_data.y_positions):
        for j, x_pos in enumerate(map_data.x_positions):
            spectrum = map_data.get_spectrum(x_pos, y_pos)
            if spectrum and spectrum.processed_intensities is not None:
                intensity_map[i, j] = np.sum(spectrum.processed_intensities)
    
    im = ax.imshow(intensity_map, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xlabel('X Position Index', fontsize=10)
    ax.set_ylabel('Y Position Index', fontsize=10)
    ax.set_title('Total Integrated Intensity Map', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Total Intensity')
    
    # 4. Intensity at specific wavenumber
    ax = axes[1, 1]
    # Find wavenumber closest to 1000 cm-1 (common Raman feature region)
    target_wn = 1000
    wn_idx = np.argmin(np.abs(map_data.target_wavenumbers - target_wn))
    actual_wn = map_data.target_wavenumbers[wn_idx]
    
    wn_map = np.zeros((len(map_data.y_positions), len(map_data.x_positions)))
    
    for i, y_pos in enumerate(map_data.y_positions):
        for j, x_pos in enumerate(map_data.x_positions):
            spectrum = map_data.get_spectrum(x_pos, y_pos)
            if spectrum and spectrum.processed_intensities is not None:
                wn_map[i, j] = spectrum.processed_intensities[wn_idx]
    
    im = ax.imshow(wn_map, aspect='auto', cmap='hot', origin='lower')
    ax.set_xlabel('X Position Index', fontsize=10)
    ax.set_ylabel('Y Position Index', fontsize=10)
    ax.set_title(f'Intensity Map at {actual_wn:.1f} cm⁻¹', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Intensity')
    
    plt.tight_layout()
    return fig


def compare_spectra_across_map(map_data, num_samples=5):
    """Compare several spectra from different positions."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sample spectra from different positions
    x_indices = np.linspace(0, len(map_data.x_positions)-1, num_samples, dtype=int)
    
    colors = plt.cm.viridis(np.linspace(0, 1, num_samples))
    
    for idx, x_idx in enumerate(x_indices):
        x_pos = map_data.x_positions[x_idx]
        y_pos = map_data.y_positions[len(map_data.y_positions)//2]  # Middle Y
        
        spectrum = map_data.get_spectrum(x_pos, y_pos)
        if spectrum:
            # Normalize for comparison
            intensities = spectrum.processed_intensities
            if intensities is not None:
                normalized = intensities / np.max(intensities) + idx * 0.5
                ax.plot(spectrum.wavenumbers, normalized, 
                       color=colors[idx], linewidth=1.0,
                       label=f'X={x_pos:.1f}, Y={y_pos:.1f}')
    
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Normalized Intensity (offset)', fontsize=12)
    ax.set_title('Spectra Comparison Across X Positions', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def prepare_for_cluster_analysis(map_data):
    """Prepare data matrix for cluster analysis."""
    logger.info("\n" + "="*60)
    logger.info("PREPARING DATA FOR CLUSTER ANALYSIS")
    logger.info("="*60)
    
    # Get the data matrix
    data_matrix = map_data.get_processed_data_matrix()
    
    logger.info(f"Data matrix shape: {data_matrix.shape}")
    logger.info(f"  - {data_matrix.shape[0]} spectra")
    logger.info(f"  - {data_matrix.shape[1]} wavenumber points")
    
    # Get position mapping
    positions = map_data.get_position_list()
    logger.info(f"\nPosition mapping: {len(positions)} positions")
    
    # Basic statistics
    logger.info(f"\nData Statistics:")
    logger.info(f"  - Mean intensity: {np.mean(data_matrix):.2f}")
    logger.info(f"  - Std intensity: {np.std(data_matrix):.2f}")
    logger.info(f"  - Min intensity: {np.min(data_matrix):.2f}")
    logger.info(f"  - Max intensity: {np.max(data_matrix):.2f}")
    
    return data_matrix, positions


def main():
    """Main demo function."""
    print("\n" + "="*70)
    print("SINGLE-FILE 2D RAMAN MAP LOADER DEMO")
    print("="*70 + "\n")
    
    # Path to your data file
    data_file = "demo_data/Cymato_LA57434-2(94)GI01.txt"
    
    if not Path(data_file).exists():
        logger.error(f"Data file not found: {data_file}")
        logger.info("Please update the 'data_file' path in the script.")
        return
    
    # Configure cosmic ray detection (optional)
    cosmic_config = CosmicRayConfig(
        apply_during_load=False,  # Set to True to enable cosmic ray removal
        enabled=True,
        absolute_threshold=1000,
        neighbor_ratio=5.0
    )
    
    # Load the map data
    logger.info(f"Loading map from: {data_file}\n")
    
    try:
        map_data = SingleFileRamanMapData(
            filepath=data_file,
            cosmic_ray_config=cosmic_config,
            progress_callback=progress_callback
        )
        
        logger.info("\n✓ Map loaded successfully!\n")
        
        # Create visualizations
        logger.info("Creating overview visualization...")
        fig1 = visualize_map_overview(map_data)
        
        logger.info("Creating spectra comparison...")
        fig2 = compare_spectra_across_map(map_data, num_samples=6)
        
        # Prepare for cluster analysis
        data_matrix, positions = prepare_for_cluster_analysis(map_data)
        
        # Save the data matrix for later use
        output_file = "demo_data/map_data_matrix.npy"
        np.save(output_file, data_matrix)
        logger.info(f"\n✓ Data matrix saved to: {output_file}")
        
        logger.info("\n" + "="*60)
        logger.info("NEXT STEPS FOR CLUSTER ANALYSIS")
        logger.info("="*60)
        logger.info("""
1. The data is now ready for cluster analysis
2. You can use this data with:
   - K-means clustering
   - Hierarchical clustering
   - PCA (Principal Component Analysis)
   - NMF (Non-negative Matrix Factorization)
   - UMAP dimensionality reduction

3. To use with RamanLab's cluster analysis tool:
   - The data matrix is in the correct format
   - Each row is a spectrum
   - Each column is a wavenumber point
        """)
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error loading map: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
