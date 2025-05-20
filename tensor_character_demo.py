#!/usr/bin/env python3
"""
Raman Tensor Character Visualization Demo

This script demonstrates the visualization of Raman tensors with different
symmetry characters (A1, E, T2, etc.) to show how the tensor shape varies
with symmetry type.
"""

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from raman_tensor_3d_visualization import RamanTensor3DVisualizer

class DummyCrystalStructure:
    """Dummy crystal structure for demonstration purposes."""
    def __init__(self, point_group="m-3m"):
        self.point_group = point_group
        print(f"Creating dummy structure with point group: {point_group}")
        self.create_tensors()
        
    def create_tensors(self):
        """Create tensors with different symmetry characters."""
        self.raman_tensor = []
        self.activities = []
        
        # Create different tensors based on point group
        if self.point_group in ['32', '3m', '3']:  # Trigonal (quartz)
            print("Creating tensors for trigonal system (e.g., quartz)")
            # A1 mode (totally symmetric)
            a1_tensor = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.5]
            ])
            self.raman_tensor.append(a1_tensor)
            self.activities.append("A1")
            
            # E modes (doubly degenerate)
            e_tensor1 = np.array([
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0]
            ])
            self.raman_tensor.append(e_tensor1)
            self.activities.append("E")
            
            e_tensor2 = np.array([
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]
            ])
            self.raman_tensor.append(e_tensor2)
            self.activities.append("E")
            
        elif self.point_group in ['6/mmm', '6/m', '622', '6mm', '-6m2', '-62m', '6', '-6']:  # Hexagonal
            print("Creating tensors for hexagonal system")
            # A1g mode (totally symmetric)
            a1g_tensor = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0]
            ])
            self.raman_tensor.append(a1g_tensor)
            self.activities.append("A1g")
            
            # E1g modes (doubly degenerate)
            e1g_tensor1 = np.array([
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0]
            ])
            self.raman_tensor.append(e1g_tensor1)
            self.activities.append("E1g")
            
            e1g_tensor2 = np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0]
            ])
            self.raman_tensor.append(e1g_tensor2)
            self.activities.append("E1g")
            
            # E2g modes (doubly degenerate)
            e2g_tensor1 = np.array([
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0]
            ])
            self.raman_tensor.append(e2g_tensor1)
            self.activities.append("E2g")
            
            e2g_tensor2 = np.array([
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]
            ])
            self.raman_tensor.append(e2g_tensor2)
            self.activities.append("E2g")
            
        elif self.point_group in ['m-3m', 'm-3', '432', '-43m', '23']:  # Cubic
            print("Creating tensors for cubic system (e.g., diamond)")
            # A1g mode
            a1g_tensor = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ])
            self.raman_tensor.append(a1g_tensor)
            self.activities.append("A1g")
            
            # Eg modes
            eg_tensor1 = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -2.0]
            ])
            self.raman_tensor.append(eg_tensor1)
            self.activities.append("Eg")
            
            eg_tensor2 = np.array([
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0]
            ])
            self.raman_tensor.append(eg_tensor2)
            self.activities.append("Eg")
            
            # T2g modes
            t2g_tensor1 = np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0]
            ])
            self.raman_tensor.append(t2g_tensor1)
            self.activities.append("T2g")
            
            t2g_tensor2 = np.array([
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0]
            ])
            self.raman_tensor.append(t2g_tensor2)
            self.activities.append("T2g")
            
            t2g_tensor3 = np.array([
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]
            ])
            self.raman_tensor.append(t2g_tensor3)
            self.activities.append("T2g")
            
        elif self.point_group in ['4/mmm', '4/m', '422', '4mm', '-42m', '-4m2', '4', '-4']:  # Tetragonal
            print("Creating tensors for tetragonal system")
            # A1g mode
            a1g_tensor = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0]
            ])
            self.raman_tensor.append(a1g_tensor)
            self.activities.append("A1g")
            
            # B1g mode
            b1g_tensor = np.array([
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0]
            ])
            self.raman_tensor.append(b1g_tensor)
            self.activities.append("B1g")
            
            # B2g mode
            b2g_tensor = np.array([
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]
            ])
            self.raman_tensor.append(b2g_tensor)
            self.activities.append("B2g")
            
            # Eg modes
            eg_tensor = np.array([
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0]
            ])
            self.raman_tensor.append(eg_tensor)
            self.activities.append("Eg")
        
        elif self.point_group in ['mmm', '222', 'mm2']:  # Orthorhombic
            print("Creating tensors for orthorhombic system")
            # Ag modes (3 independent components)
            ag1_tensor = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 0.05, 0.0],  # Small non-zero value
                [0.0, 0.0, 0.05]   # Small non-zero value
            ])
            self.raman_tensor.append(ag1_tensor)
            self.activities.append("Ag (xx)")
            
            ag2_tensor = np.array([
                [0.05, 0.0, 0.0],  # Small non-zero value
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.05]   # Small non-zero value
            ])
            self.raman_tensor.append(ag2_tensor)
            self.activities.append("Ag (yy)")
            
            ag3_tensor = np.array([
                [0.05, 0.0, 0.0],  # Small non-zero value
                [0.0, 0.05, 0.0],  # Small non-zero value
                [0.0, 0.0, 1.0]
            ])
            self.raman_tensor.append(ag3_tensor)
            self.activities.append("Ag (zz)")
            
            # B1g, B2g, B3g modes (off-diagonal components)
            b1g_tensor = np.array([
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]
            ])
            self.raman_tensor.append(b1g_tensor)
            self.activities.append("B1g")
            
            b2g_tensor = np.array([
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0]
            ])
            self.raman_tensor.append(b2g_tensor)
            self.activities.append("B2g")
            
            b3g_tensor = np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0]
            ])
            self.raman_tensor.append(b3g_tensor)
            self.activities.append("B3g")
            
        elif self.point_group in ['2/m', '2', 'm']:  # Monoclinic
            print("Creating tensors for monoclinic system")
            # A' (in-plane) modes
            a1_tensor = np.array([
                [1.0, 0.0, 0.1],
                [0.0, 0.7, 0.0],
                [0.1, 0.0, 0.5]
            ])
            self.raman_tensor.append(a1_tensor)
            self.activities.append("A'")
            
            # A'' (out-of-plane) modes
            a2_tensor = np.array([
                [0.0, 0.4, 0.0],
                [0.4, 0.0, 0.3],
                [0.0, 0.3, 0.0]
            ])
            self.raman_tensor.append(a2_tensor)
            self.activities.append("A''")
            
        else:  # Default or Triclinic
            print("Creating tensors for triclinic system")
            # General case: All components can be non-zero
            tensor = np.array([
                [0.8, 0.3, 0.2],
                [0.3, 0.6, 0.4],
                [0.2, 0.4, 0.5]
            ])
            self.raman_tensor.append(tensor)
            self.activities.append("A (general)")
            
            # A few more components to show the general case
            tensor2 = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.3]
            ])
            self.raman_tensor.append(tensor2)
            self.activities.append("A (diagonal)")
            
            tensor3 = np.array([
                [0.0, 0.6, 0.4],
                [0.6, 0.0, 0.0],
                [0.4, 0.0, 0.0]
            ])
            self.raman_tensor.append(tensor3)
            self.activities.append("A (off-diag)")
        
        print(f"Created {len(self.activities)} tensors with activities: {self.activities}")
        # For debug: print first tensor
        if self.raman_tensor:
            print(f"First tensor:\n{self.raman_tensor[0]}")

def main():
    # Create the main window
    root = tk.Tk()
    root.title("Raman Tensor Character Visualization Demo")
    root.geometry("1200x800")
    
    # Create a frame for the controls
    control_frame = ttk.Frame(root, padding=10)
    control_frame.pack(side=tk.TOP, fill=tk.X)
    
    # Point group selection
    ttk.Label(control_frame, text="Crystal Point Group:").pack(side=tk.LEFT, padx=5)
    point_group_var = tk.StringVar(value="m-3m")
    point_groups = [
        "m-3m (Cubic)", 
        "4/mmm (Tetragonal)", 
        "6/mmm (Hexagonal)",
        "32 (Trigonal)", 
        "mmm (Orthorhombic)",
        "2/m (Monoclinic)",
        "1 (Triclinic)"
    ]
    point_group_combo = ttk.Combobox(control_frame, textvariable=point_group_var, 
                                    values=point_groups, width=25)
    point_group_combo.pack(side=tk.LEFT, padx=5)
    
    # Create notebook for multiple visualizers
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Create the visualizers for different symmetry characters
    visualizers = {}
    crystal_structure = None
    crystal_point_groups = {
        "m-3m (Cubic)": "m-3m", 
        "4/mmm (Tetragonal)": "4/mmm",
        "6/mmm (Hexagonal)": "6/mmm",
        "32 (Trigonal)": "32", 
        "mmm (Orthorhombic)": "mmm",
        "2/m (Monoclinic)": "2/m",
        "1 (Triclinic)": "1"
    }
    
    def update_point_group():
        # Get point group and create new dummy structure
        pg_display = point_group_var.get()
        pg = crystal_point_groups.get(pg_display, "m-3m")
        
        print(f"Updating to point group: {pg_display} ({pg})")
        
        # Create new crystal structure
        nonlocal crystal_structure
        crystal_structure = DummyCrystalStructure(point_group=pg)
        
        # Clear existing tabs
        for i in range(notebook.index('end')):
            notebook.forget(0)  # Remove first tab (index 0) repeatedly
        
        visualizers.clear()  # Clear the dictionary of visualizers
        
        # Create new tabs for each activity in the new structure
        for i, activity in enumerate(crystal_structure.activities):
            # Create a frame for this visualizer
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=f"{activity} Mode")
            
            # Create the visualizer
            visualizer = RamanTensor3DVisualizer(frame, crystal_structure)
            # Make crystal more transparent by default
            visualizer.crystal_alpha_var.set(0.25)
            visualizer.frame.pack(fill=tk.BOTH, expand=True)
            visualizers[activity] = visualizer
            
            # Set the character for this visualizer
            visualizer.character_var.set(activity)
            visualizer.update_visualization()
        
        # Force update of the UI
        root.update()
    
    # Button to update the point group
    ttk.Button(control_frame, text="Update Point Group", 
              command=update_point_group).pack(side=tk.LEFT, padx=20)
    
    # Create initial crystal structure
    crystal_structure = DummyCrystalStructure(point_group="m-3m")
    
    # Create a visualizer for each symmetry character
    for i, activity in enumerate(crystal_structure.activities):
        # Create a frame for this visualizer
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=f"{activity} Mode")
        
        # Create the visualizer
        visualizer = RamanTensor3DVisualizer(frame, crystal_structure)
        # Make crystal more transparent by default
        visualizer.crystal_alpha_var.set(0.25)
        visualizer.frame.pack(fill=tk.BOTH, expand=True)
        visualizers[activity] = visualizer
        
        # Set the character for this visualizer
        visualizer.character_var.set(activity)
        visualizer.update_visualization()
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main() 