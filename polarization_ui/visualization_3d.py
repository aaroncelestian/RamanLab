"""
3D Visualization Module for Raman Polarization Analyzer

Provides comprehensive 3D visualization capabilities including:
- User selectable 3D tensor shapes oriented relative to laser beam
- Optical axis vectors based on crystal symmetry  
- Crystal shapes with optimized orientation
- Crystal structure visualization optimized for plot fitting
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure

# Use Qt6 backend for matplotlib with PySide6
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
# Use PySide6 as the official Qt for Python binding
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import sys
import os

# Import matplotlib config if available
try:
    from core.matplotlib_config import configure_compact_ui, CompactNavigationToolbar
    configure_compact_ui()
    COMPACT_UI_AVAILABLE = True
except ImportError:
    COMPACT_UI_AVAILABLE = False
    CompactNavigationToolbar = None


class Advanced3DVisualizationWidget(QWidget):
    """Advanced 3D visualization widget with comprehensive controls."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_analyzer = parent
        
        # Visualization data
        self.tensor_data = {}
        self.crystal_system = "Unknown"
        self.optimization_results = None
        self.crystal_structure = None
        
        # Visualization options
        self.show_optic_axes = True
        self.show_crystal_shape = True
        self.show_crystal_structure = False
        self.selected_tensor_freq = None
        self.laser_direction = np.array([0, 0, 1])  # Z-axis (from top)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QHBoxLayout(self)
        
        # Left panel for controls
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel, 1)
        
        # Right panel for 3D plot
        plot_panel = self.create_plot_panel()
        layout.addWidget(plot_panel, 3)
        
    def create_control_panel(self):
        """Create the control panel with all options."""
        panel = QGroupBox("3D Visualization Controls")
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Advanced 3D Visualization")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Tensor selection
        tensor_group = QGroupBox("Tensor Selection")
        tensor_layout = QVBoxLayout(tensor_group)
        
        self.tensor_combo = QComboBox()
        self.tensor_combo.addItem("Select Tensor...")
        self.tensor_combo.currentTextChanged.connect(self.on_tensor_selection_changed)
        tensor_layout.addWidget(QLabel("Raman Tensor:"))
        tensor_layout.addWidget(self.tensor_combo)
        
        layout.addWidget(tensor_group)
        
        # Visualization options
        options_group = QGroupBox("Display Options")
        options_layout = QVBoxLayout(options_group)
        
        # Tensor shape checkbox
        self.show_tensor_cb = QCheckBox("Show 3D Tensor Shape")
        self.show_tensor_cb.setChecked(True)
        self.show_tensor_cb.toggled.connect(self.update_visualization)
        options_layout.addWidget(self.show_tensor_cb)
        
        # Optic axes checkbox
        self.show_optic_axes_cb = QCheckBox("Show Optical Axis Vectors")
        self.show_optic_axes_cb.setChecked(True)
        self.show_optic_axes_cb.toggled.connect(self.update_visualization)
        options_layout.addWidget(self.show_optic_axes_cb)
        
        # Crystal shape checkbox
        self.show_crystal_shape_cb = QCheckBox("Show Crystal Shape")
        self.show_crystal_shape_cb.setChecked(True)
        self.show_crystal_shape_cb.toggled.connect(self.update_visualization)
        options_layout.addWidget(self.show_crystal_shape_cb)
        
        # Crystal structure checkbox
        self.show_crystal_structure_cb = QCheckBox("Show Crystal Structure")
        self.show_crystal_structure_cb.setChecked(False)
        self.show_crystal_structure_cb.toggled.connect(self.update_visualization)
        options_layout.addWidget(self.show_crystal_structure_cb)
        
        layout.addWidget(options_group)
        
        # Laser direction controls
        laser_group = QGroupBox("Laser Direction")
        laser_layout = QVBoxLayout(laser_group)
        
        laser_layout.addWidget(QLabel("Direction relative to crystal:"))
        
        # X, Y, Z sliders for laser direction
        self.laser_x_slider = QSlider(Qt.Orientation.Horizontal)
        self.laser_x_slider.setRange(-100, 100)
        self.laser_x_slider.setValue(0)
        self.laser_x_slider.valueChanged.connect(self.update_laser_direction)
        laser_layout.addWidget(QLabel("X:"))
        laser_layout.addWidget(self.laser_x_slider)
        
        self.laser_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.laser_y_slider.setRange(-100, 100)
        self.laser_y_slider.setValue(0)
        self.laser_y_slider.valueChanged.connect(self.update_laser_direction)
        laser_layout.addWidget(QLabel("Y:"))
        laser_layout.addWidget(self.laser_y_slider)
        
        self.laser_z_slider = QSlider(Qt.Orientation.Horizontal)
        self.laser_z_slider.setRange(-100, 100)
        self.laser_z_slider.setValue(100)
        self.laser_z_slider.valueChanged.connect(self.update_laser_direction)
        laser_layout.addWidget(QLabel("Z:"))
        laser_layout.addWidget(self.laser_z_slider)
        
        # Reset button
        reset_laser_btn = QPushButton("Reset to Z-axis")
        reset_laser_btn.clicked.connect(self.reset_laser_direction)
        laser_layout.addWidget(reset_laser_btn)
        
        layout.addWidget(laser_group)
        
        # Crystal orientation controls
        orientation_group = QGroupBox("Crystal Orientation")
        orientation_layout = QVBoxLayout(orientation_group)
        
        self.phi_slider = QSlider(Qt.Orientation.Horizontal)
        self.phi_slider.setRange(0, 360)
        self.phi_slider.setValue(0)
        self.phi_slider.valueChanged.connect(self.update_visualization)
        orientation_layout.addWidget(QLabel("φ (Phi):"))
        orientation_layout.addWidget(self.phi_slider)
        
        self.theta_slider = QSlider(Qt.Orientation.Horizontal)
        self.theta_slider.setRange(0, 180)
        self.theta_slider.setValue(0)
        self.theta_slider.valueChanged.connect(self.update_visualization)
        orientation_layout.addWidget(QLabel("θ (Theta):"))
        orientation_layout.addWidget(self.theta_slider)
        
        self.psi_slider = QSlider(Qt.Orientation.Horizontal)
        self.psi_slider.setRange(0, 360)
        self.psi_slider.setValue(0)
        self.psi_slider.valueChanged.connect(self.update_visualization)
        orientation_layout.addWidget(QLabel("ψ (Psi):"))
        orientation_layout.addWidget(self.psi_slider)
        
        # Use optimized orientation button
        use_optimized_btn = QPushButton("Use Optimized Orientation")
        use_optimized_btn.clicked.connect(self.use_optimized_orientation)
        orientation_layout.addWidget(use_optimized_btn)
        
        layout.addWidget(orientation_group)
        
        # Action buttons
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        refresh_btn = QPushButton("Refresh Visualization")
        refresh_btn.clicked.connect(self.refresh_data)
        actions_layout.addWidget(refresh_btn)
        
        export_btn = QPushButton("Export 3D View")
        export_btn.clicked.connect(self.export_visualization)
        actions_layout.addWidget(export_btn)
        
        layout.addWidget(actions_group)
        
        # Stretch
        layout.addStretch()
        
        return panel
        
    def create_plot_panel(self):
        """Create the matplotlib 3D plot panel."""
        panel = QGroupBox("3D Visualization")
        layout = QVBoxLayout(panel)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        
        # Add toolbar if available
        if COMPACT_UI_AVAILABLE and CompactNavigationToolbar:
            self.toolbar = CompactNavigationToolbar(self.canvas, panel)
        else:
            self.toolbar = NavigationToolbar(self.canvas, panel)
            
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Initialize the 3D axes
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.setup_3d_axes()
        
        return panel
    
    def setup_3d_axes(self):
        """Setup the 3D axes with proper labels and orientation."""
        self.ax.set_xlabel('X', fontsize=10)
        self.ax.set_ylabel('Y', fontsize=10)
        self.ax.set_zlabel('Z', fontsize=10)
        
        # Set equal aspect ratio
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_zlim(-2, 2)
        
        # Set viewing angle - looking down Z-axis (laser direction)
        self.ax.view_init(elev=20, azim=45)
        
        # Add coordinate system arrows
        self.draw_coordinate_system()
        
    def draw_coordinate_system(self):
        """Draw the coordinate system with laser direction indicator."""
        # Clear existing arrows
        self.ax.clear()
        self.setup_basic_axes()
        
        # Draw coordinate axes
        arrow_length = 1.5
        self.ax.quiver(0, 0, 0, arrow_length, 0, 0, 
                      color='red', arrow_length_ratio=0.1, linewidth=2, alpha=0.8, label='X')
        self.ax.quiver(0, 0, 0, 0, arrow_length, 0, 
                      color='green', arrow_length_ratio=0.1, linewidth=2, alpha=0.8, label='Y')
        self.ax.quiver(0, 0, 0, 0, 0, arrow_length, 
                      color='blue', arrow_length_ratio=0.1, linewidth=2, alpha=0.8, label='Z')
        
        # Draw laser direction (distinctive arrow)
        laser_norm = self.laser_direction / np.linalg.norm(self.laser_direction)
        self.ax.quiver(0, 0, 0, laser_norm[0] * 1.8, laser_norm[1] * 1.8, laser_norm[2] * 1.8, 
                      color='gold', arrow_length_ratio=0.15, linewidth=3, alpha=0.9, label='Laser')
        
        # Add labels
        self.ax.text(arrow_length + 0.1, 0, 0, 'X', fontsize=12, color='red')
        self.ax.text(0, arrow_length + 0.1, 0, 'Y', fontsize=12, color='green')
        self.ax.text(0, 0, arrow_length + 0.1, 'Z', fontsize=12, color='blue')
        self.ax.text(laser_norm[0] * 2, laser_norm[1] * 2, laser_norm[2] * 2, 'Laser', 
                    fontsize=10, color='gold', weight='bold')
        
    def setup_basic_axes(self):
        """Setup basic axes properties."""
        self.ax.set_xlabel('X', fontsize=10)
        self.ax.set_ylabel('Y', fontsize=10)
        self.ax.set_zlabel('Z', fontsize=10)
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_zlim(-2, 2)
        
    def update_laser_direction(self):
        """Update laser direction from sliders."""
        x = self.laser_x_slider.value() / 100.0
        y = self.laser_y_slider.value() / 100.0
        z = self.laser_z_slider.value() / 100.0
        
        # Normalize
        norm = np.sqrt(x*x + y*y + z*z)
        if norm > 0:
            self.laser_direction = np.array([x, y, z]) / norm
        else:
            self.laser_direction = np.array([0, 0, 1])
            
        self.update_visualization()
        
    def reset_laser_direction(self):
        """Reset laser direction to Z-axis."""
        self.laser_x_slider.setValue(0)
        self.laser_y_slider.setValue(0)
        self.laser_z_slider.setValue(100)
        self.laser_direction = np.array([0, 0, 1])
        self.update_visualization()
        
    def on_tensor_selection_changed(self, text):
        """Handle tensor selection change."""
        if text and text != "Select Tensor...":
            # Extract frequency from text (format: "frequency cm-1 - character")
            try:
                freq_str = text.split()[0]
                self.selected_tensor_freq = float(freq_str)
                print(f"Selected tensor frequency: {self.selected_tensor_freq}")
                print(f"Available tensor data: {list(self.tensor_data.keys()) if self.tensor_data else 'None'}")
                self.update_visualization()
            except (ValueError, IndexError) as e:
                print(f"Error parsing tensor selection '{text}': {e}")
                pass
                
    def use_optimized_orientation(self):
        """Use the optimized orientation from the analyzer."""
        if self.parent_analyzer and hasattr(self.parent_analyzer, 'stage_results'):
            # Find best available results
            best_stage = None
            for stage in ['stage3', 'stage2', 'stage1']:
                if self.parent_analyzer.stage_results.get(stage):
                    best_stage = stage
                    break
                    
            if best_stage:
                results = self.parent_analyzer.stage_results[best_stage]
                orientation = results.get('best_orientation', [0, 0, 0])
                
                # Set sliders to optimized values
                self.phi_slider.setValue(int(orientation[0]) % 360)
                self.theta_slider.setValue(int(orientation[1]) % 180)
                self.psi_slider.setValue(int(orientation[2]) % 360)
                
                self.update_visualization()
                
                QMessageBox.information(self, "Orientation Applied", 
                                      f"Applied optimized orientation from {best_stage}:\n"
                                      f"φ = {orientation[0]:.1f}°\n"
                                      f"θ = {orientation[1]:.1f}°\n" 
                                      f"ψ = {orientation[2]:.1f}°")
            else:
                QMessageBox.warning(self, "No Optimization Results", 
                                  "No optimization results available. Please run orientation optimization first.")
        
    def update_visualization(self):
        """Update the complete 3D visualization."""
        # Clear the plot
        self.ax.clear()
        self.setup_basic_axes()
        
        # Draw coordinate system
        self.draw_coordinate_system()
        
        # Get current orientation
        phi = np.deg2rad(self.phi_slider.value())
        theta = np.deg2rad(self.theta_slider.value())
        psi = np.deg2rad(self.psi_slider.value())
        
        # Draw tensor shape if selected and enabled
        print(f"Tensor checkbox checked: {self.show_tensor_cb.isChecked()}")
        print(f"Selected tensor freq: {self.selected_tensor_freq}")
        print(f"Tensor data available: {bool(self.tensor_data and self.selected_tensor_freq in self.tensor_data)}")
        
        if self.show_tensor_cb.isChecked() and self.selected_tensor_freq:
            print("Drawing tensor shape...")
            self.draw_tensor_shape(phi, theta, psi)
            
        # Draw optical axes if enabled
        if self.show_optic_axes_cb.isChecked():
            self.draw_optical_axes(phi, theta, psi)
            
        # Draw crystal shape if enabled
        if self.show_crystal_shape_cb.isChecked():
            self.draw_crystal_shape(phi, theta, psi)
            
        # Draw crystal structure if enabled and available
        if self.show_crystal_structure_cb.isChecked():
            self.draw_crystal_structure(phi, theta, psi)
            
        # Update title
        title = "3D Raman Polarization Visualization"
        if self.selected_tensor_freq:
            title += f"\nTensor: {self.selected_tensor_freq:.1f} cm⁻¹"
        if self.crystal_system != "Unknown":
            title += f" | Crystal System: {self.crystal_system}"
            
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Refresh canvas
        self.canvas.draw()
        
    def draw_tensor_shape(self, phi, theta, psi):
        """Draw the 3D tensor shape with proper orientation."""
        if not self.tensor_data or self.selected_tensor_freq not in self.tensor_data:
            print(f"No tensor data available. tensor_data keys: {list(self.tensor_data.keys()) if self.tensor_data else 'None'}")
            return
            
        tensor_info = self.tensor_data[self.selected_tensor_freq]
        tensor = tensor_info['tensor']
        character = tensor_info.get('character', 'Unknown')
        
        print(f"Drawing tensor: {self.selected_tensor_freq} cm⁻¹ ({character})")
        print(f"Tensor shape: {tensor.shape}")
        
        # Apply orientation rotation to tensor
        R = self.get_rotation_matrix(phi, theta, psi)
        rotated_tensor = R @ tensor @ R.T
        
        # Create spherical coordinate grid
        theta_grid = np.linspace(0, np.pi, 30)
        phi_grid = np.linspace(0, 2*np.pi, 40)
        theta_mesh, phi_mesh = np.meshgrid(theta_grid, phi_grid)
        
        # Convert to Cartesian
        x_dir = np.sin(theta_mesh) * np.cos(phi_mesh)
        y_dir = np.sin(theta_mesh) * np.sin(phi_mesh)
        z_dir = np.cos(theta_mesh)
        
        # Calculate Raman intensity for each direction
        intensity = np.zeros_like(x_dir)
        
        for i in range(x_dir.shape[0]):
            for j in range(x_dir.shape[1]):
                e_vec = np.array([x_dir[i,j], y_dir[i,j], z_dir[i,j]])
                # Raman intensity: |e · R · e|²
                raman_amplitude = np.dot(e_vec, np.dot(rotated_tensor, e_vec))
                
                # Handle different mode types
                if 'B1g' in character or 'B2g' in character:
                    intensity[i,j] = raman_amplitude  # Keep sign for node structure
                else:
                    intensity[i,j] = raman_amplitude**2
                    
        # Normalize and create surface
        intensity = np.real(intensity)
        
        if 'B1g' in character or 'B2g' in character:
            max_abs = np.max(np.abs(intensity))
            if max_abs > 0:
                intensity_norm = intensity / max_abs
            else:
                intensity_norm = np.ones_like(intensity) * 0.1
            radius_scale = 0.4 + 0.6 * np.abs(intensity_norm)
            colors = plt.cm.RdBu_r(0.5 + 0.5 * intensity_norm)
        else:
            min_intensity = np.min(intensity)
            if min_intensity < 0:
                intensity = intensity - min_intensity
            max_intensity = np.max(intensity)
            if max_intensity > 0:
                intensity_norm = intensity / max_intensity
            else:
                intensity_norm = np.ones_like(intensity) * 0.1
            radius_scale = 0.5 + 0.5 * intensity_norm
            colors = plt.cm.RdYlBu_r(0.5 + 0.5 * intensity_norm)
            
        # Calculate surface coordinates
        x_surface = radius_scale * x_dir
        y_surface = radius_scale * y_dir
        z_surface = radius_scale * z_dir
        
        # Plot the tensor surface
        self.ax.plot_surface(x_surface, y_surface, z_surface, 
                           facecolors=colors, alpha=0.7, 
                           linewidth=0, antialiased=True)
        
        # Add wireframe for structure
        self.ax.plot_wireframe(x_surface, y_surface, z_surface, 
                              color='gray', alpha=0.2, linewidth=0.5)
        
    def draw_optical_axes(self, phi, theta, psi):
        """Draw optical axis vectors based on crystal symmetry and optimization results."""
        if self.crystal_system == "Unknown":
            return
            
        # Define optical axes based on crystal system
        axes = self.get_optical_axes_for_crystal_system()
        
        if not axes:
            return
        
        # Check if we have optimization results to orient the optic axis properly
        optimized_orientation = None
        if self.parent_analyzer and hasattr(self.parent_analyzer, 'stage_results'):
            # Find best available results
            for stage in ['stage3', 'stage2', 'stage1']:
                if self.parent_analyzer.stage_results.get(stage):
                    results = self.parent_analyzer.stage_results[stage]
                    optimized_orientation = results.get('best_orientation', None)
                    break
        
        # Use either current slider values or optimized orientation
        if optimized_orientation is not None:
            # Use optimized orientation for optic axis
            opt_phi = np.deg2rad(optimized_orientation[0])
            opt_theta = np.deg2rad(optimized_orientation[1]) 
            opt_psi = np.deg2rad(optimized_orientation[2])
            R_opt = self.get_rotation_matrix(opt_phi, opt_theta, opt_psi)
            
            # Show that this is the optimized orientation
            axis_label_prefix = "Optimized "
        else:
            # Use current orientation from sliders
            R_opt = self.get_rotation_matrix(phi, theta, psi)
            axis_label_prefix = ""
            
        colors = ['orange', 'purple', 'cyan']
        for i, axis in enumerate(axes):
            rotated_axis = R_opt @ axis
            
            # Draw axis vector
            self.ax.quiver(0, 0, 0, rotated_axis[0] * 1.2, rotated_axis[1] * 1.2, rotated_axis[2] * 1.2,
                          color=colors[i % len(colors)], arrow_length_ratio=0.1, 
                          linewidth=2, alpha=0.8, label=f'{axis_label_prefix}Optic Axis {i+1}')
                          
            # Add text label
            self.ax.text(rotated_axis[0] * 1.3, rotated_axis[1] * 1.3, rotated_axis[2] * 1.3,
                        f'{axis_label_prefix}Axis {i+1}', 
                        color=colors[i % len(colors)], fontsize=9, fontweight='bold')
                          
    def draw_crystal_shape(self, phi, theta, psi):
        """Draw crystal shape based on real crystallographic data."""
        # Get crystal structure data from parent analyzer
        structure_data = None
        if self.parent_analyzer:
            if hasattr(self.parent_analyzer, 'current_crystal_structure'):
                structure_data = self.parent_analyzer.current_crystal_structure
            elif hasattr(self.parent_analyzer, 'crystal_structure'):
                structure_data = self.parent_analyzer.crystal_structure
                
        # Use real lattice parameters if available
        if structure_data and 'lattice_params' in structure_data:
            shape_vertices = self.get_crystal_shape_from_lattice(structure_data['lattice_params'])
        else:
            shape_vertices = self.get_crystal_shape_vertices()
        
        if not shape_vertices:
            return
            
        # Apply orientation rotation
        R = self.get_rotation_matrix(phi, theta, psi)
        rotated_vertices = np.array([R @ v for v in shape_vertices])
        
        # Draw crystal faces/edges
        self.draw_crystal_wireframe(rotated_vertices)
        
    def draw_crystal_structure(self, phi, theta, psi):
        """Draw crystal structure if available."""
        # Get crystal structure from parent analyzer
        structure_data = None
        
        if self.parent_analyzer:
            # Try multiple sources for crystal structure data
            if hasattr(self.parent_analyzer, 'current_crystal_structure'):
                structure_data = self.parent_analyzer.current_crystal_structure
            elif hasattr(self.parent_analyzer, 'crystal_structure'):
                structure_data = self.parent_analyzer.crystal_structure
            elif hasattr(self.parent_analyzer, 'crystal_structure_widget'):
                widget = self.parent_analyzer.crystal_structure_widget
                if hasattr(widget, 'current_structure'):
                    structure_data = widget.current_structure
                    
        if not structure_data:
            return
            
        print(f"Drawing crystal structure: {structure_data.get('formula', 'Unknown')}")
        
        # Apply rotation matrix
        R = self.get_rotation_matrix(phi, theta, psi)
        
        # Draw atomic positions if available
        if 'atomic_positions' in structure_data:
            positions = structure_data['atomic_positions']
            elements = structure_data.get('elements', [])
            
            # Standard atomic colors
            atomic_colors = {
                'H': 'white', 'C': 'gray', 'N': 'blue', 'O': 'red', 'S': 'yellow',
                'P': 'orange', 'F': 'lightgreen', 'Cl': 'green', 'Si': 'tan',
                'Ca': 'green', 'Ti': 'gray', 'Fe': 'orange', 'Zr': 'gray'
            }
            
            for i, pos in enumerate(positions):
                if len(pos) >= 3:
                    # Apply rotation
                    rotated_pos = R @ np.array(pos[:3])
                    
                    # Scale to fit in plot
                    scaled_pos = rotated_pos * 0.8
                    
                    element = elements[i] if i < len(elements) else 'C'
                    color = atomic_colors.get(element, 'gray')
                    
                    self.ax.scatter(scaled_pos[0], scaled_pos[1], scaled_pos[2], 
                                   c=color, s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
                                   
        # Draw unit cell if available
        if 'lattice_vectors' in structure_data:
            self.draw_unit_cell(structure_data['lattice_vectors'], R)
        elif 'lattice_params' in structure_data:
            self.draw_unit_cell_from_params(structure_data['lattice_params'], R)
        
    def get_rotation_matrix(self, phi, theta, psi):
        """Get 3D rotation matrix for Euler angles."""
        # Z-Y-Z convention (common in crystallography)
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)  
        cos_psi, sin_psi = np.cos(psi), np.sin(psi)
        
        R_z1 = np.array([[cos_phi, -sin_phi, 0],
                        [sin_phi, cos_phi, 0],
                        [0, 0, 1]])
                        
        R_y = np.array([[cos_theta, 0, sin_theta],
                       [0, 1, 0],
                       [-sin_theta, 0, cos_theta]])
                       
        R_z2 = np.array([[cos_psi, -sin_psi, 0],
                        [sin_psi, cos_psi, 0],
                        [0, 0, 1]])
                        
        return R_z2 @ R_y @ R_z1
        
    def get_optical_axes_for_crystal_system(self):
        """Get optical axes based on crystal system."""
        if self.crystal_system.lower() in ['cubic', 'isometric']:
            return []  # Isotropic
        elif self.crystal_system.lower() in ['tetragonal', 'hexagonal', 'trigonal']:
            return [np.array([0, 0, 1])]  # Uniaxial - c-axis
        elif self.crystal_system.lower() in ['orthorhombic', 'monoclinic', 'triclinic']:
            # Biaxial - approximate axes
            return [np.array([1, 0, 0]), np.array([0, 1, 0])]
        else:
            return [np.array([0, 0, 1])]  # Default
            
    def get_crystal_shape_vertices(self):
        """Get vertices for basic crystal shapes."""
        if self.crystal_system.lower() == 'cubic':
            # Cube vertices
            return [np.array([x, y, z]) * 0.8 for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]]
        elif self.crystal_system.lower() == 'tetragonal':
            # Tetragonal prism
            return [np.array([x, y, z]) * np.array([0.8, 0.8, 1.2]) 
                   for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]]
        elif self.crystal_system.lower() == 'hexagonal':
            # Hexagonal prism - approximate with octagon
            angles = np.linspace(0, 2*np.pi, 7)
            vertices = []
            for z in [-1, 1]:
                for angle in angles[:-1]:  # Skip last to avoid duplication
                    x, y = 0.8 * np.cos(angle), 0.8 * np.sin(angle)
                    vertices.append(np.array([x, y, z * 1.2]))
            return vertices
        else:
            # Default orthorhombic-like shape
            return [np.array([x, y, z]) * np.array([1.0, 0.8, 1.2]) 
                   for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]]
                   
    def draw_crystal_wireframe(self, vertices):
        """Draw wireframe representation of crystal."""
        if len(vertices) == 8:  # Cubic/orthorhombic
            # Define edges for a box
            edges = [
                (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
                (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
                (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
            ]
            
            for edge in edges:
                p1, p2 = vertices[edge[0]], vertices[edge[1]]
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                           'k-', alpha=0.6, linewidth=1.5)
        else:
            # For other shapes, draw a simple outline
            for i, vertex in enumerate(vertices[:6]):  # Limit for performance
                self.ax.scatter(vertex[0], vertex[1], vertex[2], 
                              c='black', s=20, alpha=0.6)
                              
    def get_crystal_shape_from_lattice(self, lattice_params):
        """Get crystal shape vertices from real lattice parameters."""
        a = lattice_params.get('a', 1.0)
        b = lattice_params.get('b', 1.0) 
        c = lattice_params.get('c', 1.0)
        alpha = np.deg2rad(lattice_params.get('alpha', 90.0))
        beta = np.deg2rad(lattice_params.get('beta', 90.0))
        gamma = np.deg2rad(lattice_params.get('gamma', 90.0))
        
        # Scale to fit in visualization (normalize to max dimension = 1)
        max_dim = max(a, b, c)
        a, b, c = a/max_dim, b/max_dim, c/max_dim
        
        # Calculate unit cell vectors
        # a-axis along x
        ax, ay, az = a, 0, 0
        
        # b-axis in xy plane
        bx = b * np.cos(gamma)
        by = b * np.sin(gamma)
        bz = 0
        
        # c-axis completing the unit cell
        cx = c * np.cos(beta)
        cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        cz = c * np.sqrt(1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 + 
                        2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        
        # Unit cell vertices
        vertices = [
            np.array([0, 0, 0]),                           # Origin
            np.array([ax, ay, az]),                        # a
            np.array([ax + bx, ay + by, az + bz]),         # a + b
            np.array([bx, by, bz]),                        # b
            np.array([cx, cy, cz]),                        # c
            np.array([ax + cx, ay + cy, az + cz]),         # a + c
            np.array([ax + bx + cx, ay + by + cy, az + bz + cz]),  # a + b + c
            np.array([bx + cx, by + cy, bz + cz])          # b + c
        ]
        
        return vertices
        
    def draw_unit_cell(self, lattice_vectors, rotation_matrix):
        """Draw unit cell from lattice vectors."""
        # Apply rotation and scaling
        scale = 0.8
        rotated_vectors = [rotation_matrix @ (v * scale) for v in lattice_vectors]
        
        # Draw lattice vectors as colored arrows
        colors = ['red', 'green', 'blue']
        labels = ['a', 'b', 'c']
        
        for i, (vec, color, label) in enumerate(zip(rotated_vectors, colors, labels)):
            self.ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], 
                          color=color, arrow_length_ratio=0.1, linewidth=2, alpha=0.7)
            self.ax.text(vec[0]*1.1, vec[1]*1.1, vec[2]*1.1, label, 
                        color=color, fontsize=10, fontweight='bold')
                        
    def draw_unit_cell_from_params(self, lattice_params, rotation_matrix):
        """Draw unit cell from lattice parameters."""
        vertices = self.get_crystal_shape_from_lattice(lattice_params)
        rotated_vertices = np.array([rotation_matrix @ v for v in vertices])
        self.draw_crystal_wireframe(rotated_vertices)
        
    def refresh_data(self):
        """Refresh data from parent analyzer."""
        if self.parent_analyzer:
            # Update tensor data
            if hasattr(self.parent_analyzer, 'calculated_raman_tensors'):
                self.tensor_data = self.parent_analyzer.calculated_raman_tensors.copy()
                print(f"Refreshed tensor data: {len(self.tensor_data)} tensors")
                print(f"Tensor frequencies: {list(self.tensor_data.keys())}")
                self.update_tensor_combo()
                
            # Update crystal system
            if hasattr(self.parent_analyzer, 'current_crystal_system'):
                self.crystal_system = self.parent_analyzer.current_crystal_system
                print(f"Crystal system: {self.crystal_system}")
            elif hasattr(self.parent_analyzer, 'selected_reference_mineral'):
                mineral = self.parent_analyzer.selected_reference_mineral
                if mineral and self.parent_analyzer.mineral_database.get(mineral):
                    self.crystal_system = self.parent_analyzer.mineral_database[mineral].get('crystal_system', 'Unknown')
                    print(f"Crystal system from mineral {mineral}: {self.crystal_system}")
                    
            # Update crystal structure
            if hasattr(self.parent_analyzer, 'current_crystal_structure'):
                self.crystal_structure = self.parent_analyzer.current_crystal_structure
                if self.crystal_structure:
                    print(f"Crystal structure available: {self.crystal_structure.get('formula', 'Unknown')}")
                
            # Update optimization results
            if hasattr(self.parent_analyzer, 'stage_results'):
                self.optimization_results = self.parent_analyzer.stage_results
                if self.optimization_results:
                    print(f"Optimization results available for stages: {list(self.optimization_results.keys())}")
                
        self.update_visualization()
        
    def update_tensor_combo(self):
        """Update the tensor selection combo box."""
        self.tensor_combo.clear()
        self.tensor_combo.addItem("Select Tensor...")
        
        if self.tensor_data:
            for freq, data in sorted(self.tensor_data.items()):
                character = data.get('character', 'Unknown')
                self.tensor_combo.addItem(f"{freq:.1f} cm⁻¹ - {character}")
                
            # Auto-select the first tensor if none is selected
            if self.tensor_combo.count() > 1 and not self.selected_tensor_freq:
                self.tensor_combo.setCurrentIndex(1)  # Select first tensor
                print(f"Auto-selected first tensor: {self.tensor_combo.currentText()}")
                
    def export_visualization(self):
        """Export the current 3D visualization."""
        filename, _ = QFileDialog.getSaveFileName(self, "Export 3D Visualization", 
                                                 "3d_visualization.png", 
                                                 "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)")
        if filename:
            try:
                self.figure.savefig(filename, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Export Successful", f"3D visualization saved to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Failed to export visualization:\n{str(e)}") 