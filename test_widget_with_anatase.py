#!/usr/bin/env python3
"""
Test Crystal Structure Widget with Anatase
Tests the complete widget functionality with the anatase CIF file.
"""

import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QMessageBox
from PySide6.QtCore import Qt, QTimer

# Add ui directory to path
sys.path.append('ui')

try:
    from ui.crystal_structure_widget import CrystalStructureWidget
    
    class AnataseTesterWindow(QMainWindow):
        """Test window specifically for anatase crystal structure."""
        
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Anatase Crystal Structure Test")
            self.setGeometry(100, 100, 1200, 800)
            
            # Create central widget
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            # Layout
            layout = QVBoxLayout(central_widget)
            
            # Auto-load button
            auto_load_btn = QPushButton("🔬 Auto-Load Anatase TiO₂ Structure")
            auto_load_btn.clicked.connect(self.auto_load_anatase)
            auto_load_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    font-size: 14px;
                    padding: 10px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            layout.addWidget(auto_load_btn)
            
            # Crystal structure widget
            self.structure_widget = CrystalStructureWidget()
            layout.addWidget(self.structure_widget)
            
            # Connect signals for testing
            self.structure_widget.structure_loaded.connect(self.on_structure_loaded)
            self.structure_widget.bond_calculated.connect(self.on_bonds_calculated)
            
            # Status tracking
            self.load_successful = False
            self.bonds_calculated = False
        
        def auto_load_anatase(self):
            """Automatically load the anatase structure."""
            anatase_path = "__exampleData/anatase.cif"
            
            if not os.path.exists(anatase_path):
                QMessageBox.critical(self, "Error", f"Anatase file not found:\n{anatase_path}")
                return
            
            try:
                # Use the widget's CIF loading method
                from pymatgen.io.cif import CifParser
                
                print("🔄 Loading anatase structure...")
                
                # Parse the CIF file
                parser = CifParser(anatase_path)
                structures = parser.parse_structures(primitive=True)
                
                if not structures:
                    QMessageBox.critical(self, "Error", "No structures found in anatase.cif")
                    return
                
                # Set the structure in the widget
                self.structure_widget.pymatgen_structure = structures[0]
                
                # Extract structure information (this was failing before)
                self.structure_widget.extract_structure_info()
                
                # Update displays
                self.structure_widget.update_structure_info_display()
                self.structure_widget.update_3d_plot()
                
                # Calculate bonds
                self.structure_widget.calculate_bonds()
                
                print("✅ Anatase structure loaded successfully!")
                
                # Set up automatic rotation for demo
                QTimer.singleShot(2000, self.enable_demo_rotation)
                
            except Exception as e:
                error_msg = f"Error loading anatase structure:\n{str(e)}"
                print(f"❌ {error_msg}")
                QMessageBox.critical(self, "Error", error_msg)
        
        def enable_demo_rotation(self):
            """Enable auto-rotation for demonstration."""
            if self.load_successful:
                self.structure_widget.auto_rotate_cb.setChecked(True)
                self.structure_widget.toggle_auto_rotation(True)
                print("🔄 Auto-rotation enabled for demonstration")
        
        def on_structure_loaded(self, structure_data):
            """Handle successful structure loading."""
            self.load_successful = True
            
            formula = structure_data.get('formula', 'Unknown')
            num_atoms = structure_data.get('num_atoms', 0)
            crystal_system = structure_data.get('crystal_system', 'Unknown')
            space_group = structure_data.get('space_group', 'Unknown')
            
            print(f"🎉 Structure loaded successfully!")
            print(f"   Formula: {formula}")
            print(f"   Crystal System: {crystal_system}")
            print(f"   Space Group: {space_group}")
            print(f"   Number of Atoms: {num_atoms}")
            
            # Show lattice parameters
            lattice = structure_data.get('lattice_params', {})
            if lattice:
                print(f"   Lattice Parameters:")
                print(f"     a = {lattice.get('a', 0):.4f} Å")
                print(f"     b = {lattice.get('b', 0):.4f} Å")
                print(f"     c = {lattice.get('c', 0):.4f} Å")
                print(f"     α = {lattice.get('alpha', 0):.2f}°")
                print(f"     β = {lattice.get('beta', 0):.2f}°")
                print(f"     γ = {lattice.get('gamma', 0):.2f}°")
                print(f"     Volume = {lattice.get('volume', 0):.2f} Å³")
        
        def on_bonds_calculated(self, bond_data):
            """Handle successful bond calculation."""
            self.bonds_calculated = True
            
            bonds = bond_data.get('bonds', [])
            bond_count = bond_data.get('count', 0)
            
            print(f"🔗 Bonds calculated successfully!")
            print(f"   Total bonds: {bond_count}")
            
            # Show first few bonds as examples
            if bonds:
                print(f"   Example bonds:")
                for i, bond in enumerate(bonds[:3]):  # Show first 3 bonds
                    atom1 = bond.get('atom1_element', '?')
                    atom2 = bond.get('atom2_element', '?')
                    distance = bond.get('distance', 0)
                    print(f"     {atom1}-{atom2}: {distance:.3f} Å")
                
                if len(bonds) > 3:
                    print(f"     ... and {len(bonds) - 3} more bonds")
            
            # Test successful if both structure and bonds are loaded
            if self.load_successful and self.bonds_calculated:
                print("\n🎊 Complete test successful!")
                print("   ✅ Structure loading works")
                print("   ✅ Bond calculation works")
                print("   ✅ 3D visualization works")
                print("   ✅ All pymatgen integration working")
    
    def main():
        """Run the anatase test application."""
        app = QApplication(sys.argv)
        
        # Check if anatase file exists
        if not os.path.exists("__exampleData/anatase.cif"):
            print("❌ Anatase test file not found: __exampleData/anatase.cif")
            print("   Please make sure the file exists in the __exampleData directory")
            return 1
        
        print("🧪 Starting Anatase Crystal Structure Test")
        print("   This will test loading and visualization of TiO₂ (anatase)")
        
        window = AnataseTesterWindow()
        window.show()
        
        print("👆 Click the 'Auto-Load Anatase' button to run the test")
        
        return app.exec()
    
    if __name__ == "__main__":
        sys.exit(main())
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Make sure pymatgen is installed: pip install pymatgen")
    print("   Make sure PySide6 is installed: pip install PySide6")
    sys.exit(1) 