#!/usr/bin/env python3
"""
Test script for Crystal Structure Widget
Tests loading and visualization of the anatase.cif file.
"""

import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PySide6.QtCore import Qt

# Add ui directory to path
sys.path.append('ui')

try:
    from ui.crystal_structure_widget import CrystalStructureWidget
    
    class TestWindow(QMainWindow):
        """Test window for crystal structure widget."""
        
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Crystal Structure Widget Test")
            self.setGeometry(100, 100, 1000, 700)
            
            # Create central widget
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            # Layout
            layout = QVBoxLayout(central_widget)
            
            # Test button to load anatase
            load_button = QPushButton("Load Anatase (TiO2) Structure")
            load_button.clicked.connect(self.load_anatase)
            layout.addWidget(load_button)
            
            # Crystal structure widget
            self.structure_widget = CrystalStructureWidget()
            layout.addWidget(self.structure_widget)
            
            # Connect signals
            self.structure_widget.structure_loaded.connect(self.on_structure_loaded)
            self.structure_widget.bond_calculated.connect(self.on_bonds_calculated)
        
        def load_anatase(self):
            """Load the anatase structure automatically."""
            anatase_path = "__exampleData/anatase.cif"
            if os.path.exists(anatase_path):
                # Simulate file selection by calling the load method directly
                try:
                    from pymatgen.io.cif import CifParser
                    
                    # Parse the file
                    parser = CifParser(anatase_path)
                    structures = parser.parse_structures(primitive=True)
                    
                    if structures:
                        self.structure_widget.pymatgen_structure = structures[0]
                        self.structure_widget.extract_structure_info()
                        self.structure_widget.update_structure_info_display()
                        self.structure_widget.update_3d_plot()
                        self.structure_widget.calculate_bonds()
                        
                        print("✓ Anatase structure loaded successfully")
                    else:
                        print("✗ No structures found in anatase.cif")
                
                except Exception as e:
                    print(f"✗ Error loading anatase: {e}")
            else:
                print(f"✗ Anatase file not found at {anatase_path}")
        
        def on_structure_loaded(self, structure_data):
            """Handle structure loading."""
            formula = structure_data.get('formula', 'Unknown')
            num_atoms = structure_data.get('num_atoms', 0)
            crystal_system = structure_data.get('crystal_system', 'Unknown')
            print(f"✓ Structure loaded: {formula} ({crystal_system}) with {num_atoms} atoms")
        
        def on_bonds_calculated(self, bond_data):
            """Handle bond calculation."""
            bond_count = bond_data.get('count', 0)
            print(f"✓ Calculated {bond_count} bonds")
    
    def main():
        """Run the test application."""
        app = QApplication(sys.argv)
        
        window = TestWindow()
        window.show()
        
        return app.exec()
    
    if __name__ == "__main__":
        sys.exit(main())
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure pymatgen is installed: pip install pymatgen")
    sys.exit(1) 