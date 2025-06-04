#!/usr/bin/env python3
"""
Helper script to load CALCITE_3 data for 3D visualization.
Run this script to automatically load CALCITE_3 data into the application.
"""

import sys
import os
import pickle
import tkinter as tk
from tkinter import messagebox

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_calcite3_data():
    """Load CALCITE_3 data into the application."""
    try:
        # Import the main application
        from raman_polarization_analyzer import RamanPolarizationAnalyzer
        
        print("🔬 Loading CALCITE_3 Data for 3D Visualization")
        print("=" * 50)
        
        # Load database
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mineral_modes.pkl")
        if not os.path.exists(db_path):
            print(f"❌ Database file not found: {db_path}")
            return None
        
        with open(db_path, 'rb') as f:
            database = pickle.load(f)
        
        if 'CALCITE_3' not in database:
            print(f"❌ CALCITE_3 not found in database")
            return None
        
        calcite3_data = database['CALCITE_3']
        print(f"✅ CALCITE_3 data loaded from database")
        print(f"   Crystal system: {calcite3_data.get('crystal_system', 'Unknown')}")
        print(f"   Point group: {calcite3_data.get('point_group', 'Unknown')}")
        
        # Create application instance
        root = tk.Tk()
        app = RamanPolarizationAnalyzer(root)
        
        # Load the CALCITE_3 data
        app.mineral_data = calcite3_data
        
        # Create tensors from the mineral data
        print(f"\n🧮 Creating Raman tensors...")
        success = app.create_tensors_from_mineral_data()
        
        if success:
            print(f"✅ Successfully created tensors for 3D visualization!")
            
            # Switch to 3D Visualization tab
            try:
                # Find the 3D Visualization tab
                for i in range(app.notebook.index("end")):
                    tab_text = app.notebook.tab(i, "text")
                    if "3D" in tab_text:
                        app.notebook.select(i)
                        print(f"✅ Switched to 3D Visualization tab")
                        break
            except Exception as e:
                print(f"⚠️  Could not switch to 3D tab: {e}")
            
            # Show success message
            messagebox.showinfo(
                "CALCITE_3 Loaded Successfully!", 
                f"CALCITE_3 data has been loaded with {len(app.tensor_data_3d['wavenumbers'])} Raman-active modes.\n\n" +
                "You can now:\n" +
                "• View the 3D crystal orientation\n" +
                "• See Raman tensor ellipsoids\n" +
                "• Rotate the crystal and observe intensity changes\n" +
                "• Use the orientation sliders to explore anisotropy\n\n" +
                "The strongest modes are at:\n" +
                f"• {app.tensor_data_3d['wavenumbers'][0]:.0f} cm⁻¹\n" +
                f"• {app.tensor_data_3d['wavenumbers'][1]:.0f} cm⁻¹\n" +
                f"• {app.tensor_data_3d['wavenumbers'][2]:.0f} cm⁻¹"
            )
            
            print(f"\n🎯 Instructions:")
            print(f"   1. Go to the 3D Visualization tab")
            print(f"   2. Enable 'Show Tensor Ellipsoid' checkbox")
            print(f"   3. Use the orientation sliders (φ, θ, ψ) to rotate the crystal")
            print(f"   4. Watch how the tensor ellipsoid changes shape")
            print(f"   5. Enable 'Real-time Calculation' to see spectrum updates")
            
            return app
        else:
            print(f"❌ Failed to create tensors")
            messagebox.showerror("Error", "Failed to create Raman tensors from CALCITE_3 data.")
            return None
            
    except Exception as e:
        print(f"❌ Error loading CALCITE_3: {e}")
        import traceback
        traceback.print_exc()
        if 'root' in locals():
            messagebox.showerror("Error", f"Failed to load CALCITE_3 data:\n{str(e)}")
        return None

if __name__ == "__main__":
    app = load_calcite3_data()
    if app:
        print(f"\n🚀 Application started with CALCITE_3 data!")
        print(f"   Close this terminal window when done.")
        app.root.mainloop()
    else:
        print(f"\n❌ Failed to start application")
        input("Press Enter to exit...") 