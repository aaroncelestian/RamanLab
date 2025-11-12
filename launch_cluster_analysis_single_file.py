"""Launch cluster analysis for single-file 2D Raman map data.

This script loads a single-file Raman map and launches the cluster analysis GUI.
"""

import sys
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import QApplication
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import the single-file loader
from map_analysis_2d.core.single_file_map_loader import SingleFileRamanMapData
from map_analysis_2d.core.cosmic_ray_detection import CosmicRayConfig

# Import cluster analysis (refactored module)
from cluster_analysis import RamanClusterAnalysisQt6


def load_single_file_map(filepath, cosmic_ray_removal=False):
    """Load a single-file Raman map."""
    logger.info(f"Loading map from: {filepath}")
    
    # Configure cosmic ray detection
    cosmic_config = CosmicRayConfig(
        apply_during_load=cosmic_ray_removal,
        enabled=cosmic_ray_removal,
        absolute_threshold=1000,
        neighbor_ratio=5.0
    )
    
    # Progress callback
    def progress_callback(progress, message):
        print(f"[{progress:3d}%] {message}")
    
    # Load the map
    map_data = SingleFileRamanMapData(
        filepath=filepath,
        cosmic_ray_config=cosmic_config,
        progress_callback=progress_callback
    )
    
    logger.info(f"✓ Loaded {len(map_data.spectra)} spectra")
    logger.info(f"  Grid: {map_data.width} X × {map_data.height} Y positions")
    logger.info(f"  Wavenumber range: {map_data.target_wavenumbers[0]:.1f} to {map_data.target_wavenumbers[-1]:.1f} cm⁻¹")
    
    return map_data


def main():
    """Main function to launch cluster analysis."""
    print("\n" + "="*70)
    print("CLUSTER ANALYSIS FOR SINGLE-FILE 2D RAMAN MAP")
    print("="*70 + "\n")
    
    # Path to your data file
    data_file = "demo_data/Cymato_LA57434-2(94)GI01.txt"
    
    if not Path(data_file).exists():
        logger.error(f"Data file not found: {data_file}")
        logger.info("Please update the 'data_file' path in the script.")
        return
    
    try:
        # Load the map data
        map_data = load_single_file_map(data_file, cosmic_ray_removal=False)
        
        # Get data matrix and metadata
        data_matrix = map_data.get_processed_data_matrix()
        wavenumbers = map_data.target_wavenumbers
        positions = map_data.get_position_list()
        
        logger.info(f"\nData matrix shape: {data_matrix.shape}")
        logger.info(f"  - {data_matrix.shape[0]} spectra")
        logger.info(f"  - {data_matrix.shape[1]} wavenumber points")
        
        # Create filenames list for the cluster analysis tool
        filenames = [f"pos_{x:.1f}_{y:.1f}" for x, y in positions]
        
        # Launch Qt application
        logger.info("\nLaunching cluster analysis GUI...")
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # Create cluster analysis window
        window = RamanClusterAnalysisQt6()
        
        # Load the data into the cluster analysis tool
        window.wavenumbers = wavenumbers
        window.intensities = data_matrix
        window.filenames = filenames
        window.original_intensities = data_matrix.copy()
        
        # Store map metadata for spatial visualization
        window.map_metadata = {
            'x_positions': map_data.x_positions,
            'y_positions': map_data.y_positions,
            'width': map_data.width,
            'height': map_data.height,
            'position_map': {filename: pos for filename, pos in zip(filenames, positions)}
        }
        
        # Update UI
        window.update_data_info()
        
        logger.info("✓ Data loaded into cluster analysis tool")
        logger.info("\nYou can now:")
        logger.info("  1. Perform PCA analysis")
        logger.info("  2. Run K-means clustering")
        logger.info("  3. Apply hierarchical clustering")
        logger.info("  4. Use UMAP for dimensionality reduction")
        logger.info("  5. Visualize cluster maps")
        
        window.show()
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
