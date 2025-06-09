#!/usr/bin/env python3
"""
Example script showing how to load map data PKL files safely
"""

import pkl_utils
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    """Main example function"""
    
    print("🗺️  Map Data Loading Example")
    print("=" * 40)
    
    # Example 1: Load a specific map data file
    try:
        print("\n1️⃣  Loading specific map data file...")
        map_data_path = "./__exampleData/ML Plastics copy/mapData.pkl"
        data = pkl_utils.load_map_data(map_data_path)
        
        print(f"✅ Successfully loaded: {map_data_path}")
        print(f"   Data type: {type(data)}")
        
        # If it's a RamanMapData object, show some info
        if hasattr(data, 'spectra'):
            print(f"   Number of spectra: {len(data.spectra) if data.spectra else 'None'}")
        if hasattr(data, 'x_positions'):
            print(f"   X positions: {len(data.x_positions) if data.x_positions else 'None'}")
        if hasattr(data, 'y_positions'):
            print(f"   Y positions: {len(data.y_positions) if data.y_positions else 'None'}")
            
    except Exception as e:
        print(f"❌ Error loading map data: {e}")
    
    # Example 2: Load using the generic safe_pickle_load function
    try:
        print("\n2️⃣  Loading using generic safe_pickle_load...")
        data2 = pkl_utils.safe_pickle_load("./__exampleData/ML Plastics copy/mapData3.pkl")
        
        print(f"✅ Successfully loaded mapData3.pkl")
        print(f"   Data type: {type(data2)}")
        if isinstance(data2, dict):
            print(f"   Dictionary keys: {list(data2.keys())}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 3: Load other database files
    try:
        print("\n3️⃣  Loading mineral modes database...")
        mineral_data = pkl_utils.load_mineral_modes()
        print(f"✅ Successfully loaded mineral modes")
        print(f"   Data type: {type(mineral_data)}")
        
    except Exception as e:
        print(f"❌ Error loading mineral modes: {e}")

if __name__ == "__main__":
    main() 