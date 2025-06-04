import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource, LinearSegmentedColormap
import tkinter as tk
from tkinter import ttk, messagebox
import re

class RamanTensor3DVisualizer:
    """
    A class for visualizing Raman tensors and crystal shapes in 3D.
    This can be used as a standalone component or integrated as a tab in the main application.
    """
    
    def __init__(self, parent, crystal_structure=None, optimized_orientation=None):
        self.parent = parent
        self.crystal_structure = crystal_structure
        self.optimized_orientation = optimized_orientation if optimized_orientation else (0, 0, 0)
        
        # Current visualization settings
        self.show_tensor = True
        self.show_crystal = True
        self.tensor_scale = 1.0
        self.crystal_scale = 1.0
        self.current_character = "All"
        self.laser_direction = np.array([0, 0, 1])  # Z-axis
        
        # Point group to crystal shape mapping
        self.point_group_shapes = {
            # Cubic system
            "m-3m": "cube",
            "m-3": "cube",
            "432": "octahedron",
            "-43m": "tetrahedron",
            "23": "cube",
            
            # Hexagonal system
            "6/mmm": "hexagonal_prism",
            "6/m": "hexagonal_prism",
            "622": "hexagonal_prism",
            "6mm": "hexagonal_prism",
            "-6m2": "hexagonal_prism",
            "-62m": "hexagonal_prism",
            "6": "hexagonal_prism",
            "-6": "hexagonal_prism",
            
            # Trigonal system
            "-3m": "trigonal_prism",
            "-3": "trigonal_prism",
            "32": "trigonal_prism",
            "3m": "trigonal_prism",
            "3": "trigonal_prism",
            
            # Tetragonal system
            "4/mmm": "tetragonal_prism",
            "4/m": "tetragonal_prism",
            "422": "tetragonal_prism",
            "4mm": "tetragonal_prism",
            "-42m": "tetragonal_prism",
            "-4m2": "tetragonal_prism",
            "4": "tetragonal_prism",
            "-4": "tetragonal_prism",
            
            # Orthorhombic system
            "mmm": "orthorhombic_prism",
            "222": "orthorhombic_prism",
            "mm2": "orthorhombic_prism",
            
            # Monoclinic system
            "2/m": "monoclinic_prism",
            "2": "monoclinic_prism",
            "m": "monoclinic_prism",
            
            # Triclinic system
            "-1": "triclinic_prism",
            "1": "triclinic_prism"
        }
        
        # Create the main frame
        self.frame = ttk.Frame(parent)
        self.setup_gui()
    
    def setup_gui(self):
        """Set up the GUI elements for the 3D visualization tab"""
        # Create a PanedWindow to allow resizing between panels
        main_paned = ttk.PanedWindow(self.frame, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left frame for controls
        left_frame = ttk.Frame(main_paned, width=300)
        left_frame.pack_propagate(False)  # Prevent the frame from shrinking
        
        # Right frame for 3D visualization
        right_frame = ttk.Frame(main_paned)
        
        # Add frames to the paned window
        main_paned.add(left_frame, weight=0)  # Fixed weight for left panel
        main_paned.add(right_frame, weight=1)  # Expandable weight for right panel
        
        # === Left frame - Controls ===
        ttk.Label(left_frame, text="3D Visualization Controls", font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        # Visualization toggles
        toggle_frame = ttk.LabelFrame(left_frame, text="Display Options")
        toggle_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        # Show/hide tensor
        tensor_toggle_frame = ttk.Frame(toggle_frame)
        tensor_toggle_frame.pack(fill=tk.X, pady=5)
        self.show_tensor_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tensor_toggle_frame, text="Show Raman Tensor", variable=self.show_tensor_var, 
                       command=self.update_visualization).pack(side=tk.LEFT)
        ttk.Label(tensor_toggle_frame, text="Scale:").pack(side=tk.LEFT, padx=(10, 0))
        self.tensor_scale_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(tensor_toggle_frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.tensor_scale_var, 
                   width=5, command=self.update_visualization).pack(side=tk.LEFT, padx=5)
        
        # Show/hide crystal
        crystal_toggle_frame = ttk.Frame(toggle_frame)
        crystal_toggle_frame.pack(fill=tk.X, pady=5)
        self.show_crystal_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(crystal_toggle_frame, text="Show Crystal", variable=self.show_crystal_var, 
                       command=self.update_visualization).pack(side=tk.LEFT)
        ttk.Label(crystal_toggle_frame, text="Scale:").pack(side=tk.LEFT, padx=(22, 0))
        self.crystal_scale_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(crystal_toggle_frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.crystal_scale_var, 
                   width=5, command=self.update_visualization).pack(side=tk.LEFT, padx=5)
        
        # Crystal transparency control
        crystal_alpha_frame = ttk.Frame(toggle_frame)
        crystal_alpha_frame.pack(fill=tk.X, pady=5)
        ttk.Label(crystal_alpha_frame, text="Crystal Opacity:").pack(side=tk.LEFT, padx=5)
        self.crystal_alpha_var = tk.DoubleVar(value=0.25)
        alpha_slider = ttk.Scale(crystal_alpha_frame, from_=0.1, to=1.0, 
                               variable=self.crystal_alpha_var, 
                               orient=tk.HORIZONTAL)
        alpha_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        alpha_slider.bind("<ButtonRelease-1>", lambda e: self.update_visualization())
        
        # Enhanced visualization options
        enhanced_frame = ttk.Frame(toggle_frame)
        enhanced_frame.pack(fill=tk.X, pady=5)
        self.show_ellipses_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(enhanced_frame, text="Show Principal Ellipses", 
                      variable=self.show_ellipses_var, 
                      command=self.update_visualization).pack(side=tk.LEFT)
        
        self.enhanced_shading_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(enhanced_frame, text="Enhanced Shading", 
                      variable=self.enhanced_shading_var, 
                      command=self.update_visualization).pack(side=tk.LEFT, padx=(10, 0))
        
        # Character selection
        character_frame = ttk.LabelFrame(left_frame, text="Raman Tensor Character")
        character_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.character_var = tk.StringVar(value="All")
        characters = ["All", "A1", "A2", "E", "T1", "T2"]  # Default characters
        self.character_combo = ttk.Combobox(character_frame, textvariable=self.character_var, 
                                           values=characters, state="readonly")
        self.character_combo.pack(fill=tk.X, pady=5, padx=5)
        self.character_combo.bind("<<ComboboxSelected>>", lambda e: self.update_visualization())
        
        # Orientation controls
        orientation_frame = ttk.LabelFrame(left_frame, text="Crystal Orientation (Euler Angles)")
        orientation_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        # Phi angle
        phi_frame = ttk.Frame(orientation_frame)
        phi_frame.pack(fill=tk.X, pady=5)
        ttk.Label(phi_frame, text="φ (deg):").pack(side=tk.LEFT, padx=5)
        self.phi_var = tk.DoubleVar(value=self.optimized_orientation[0])
        phi_spinbox = ttk.Spinbox(phi_frame, from_=0, to=360, increment=5, textvariable=self.phi_var, width=5)
        phi_spinbox.pack(side=tk.LEFT, padx=5)
        phi_spinbox.bind("<Return>", lambda e: self.update_visualization())
        phi_spinbox.bind("<FocusOut>", lambda e: self.update_visualization())
        
        # Theta angle
        theta_frame = ttk.Frame(orientation_frame)
        theta_frame.pack(fill=tk.X, pady=5)
        ttk.Label(theta_frame, text="θ (deg):").pack(side=tk.LEFT, padx=5)
        self.theta_var = tk.DoubleVar(value=self.optimized_orientation[1])
        theta_spinbox = ttk.Spinbox(theta_frame, from_=0, to=180, increment=5, textvariable=self.theta_var, width=5)
        theta_spinbox.pack(side=tk.LEFT, padx=5)
        theta_spinbox.bind("<Return>", lambda e: self.update_visualization())
        theta_spinbox.bind("<FocusOut>", lambda e: self.update_visualization())
        
        # Psi angle
        psi_frame = ttk.Frame(orientation_frame)
        psi_frame.pack(fill=tk.X, pady=5)
        ttk.Label(psi_frame, text="ψ (deg):").pack(side=tk.LEFT, padx=5)
        self.psi_var = tk.DoubleVar(value=self.optimized_orientation[2])
        psi_spinbox = ttk.Spinbox(psi_frame, from_=0, to=360, increment=5, textvariable=self.psi_var, width=5)
        psi_spinbox.pack(side=tk.LEFT, padx=5)
        psi_spinbox.bind("<Return>", lambda e: self.update_visualization())
        psi_spinbox.bind("<FocusOut>", lambda e: self.update_visualization())
        
        # View controls
        view_frame = ttk.LabelFrame(left_frame, text="Viewpoint")
        view_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        # Elevation (vertical) angle
        elev_frame = ttk.Frame(view_frame)
        elev_frame.pack(fill=tk.X, pady=5)
        ttk.Label(elev_frame, text="Elevation:").pack(side=tk.LEFT, padx=5)
        self.elevation_var = tk.IntVar(value=20)
        elev_slider = ttk.Scale(elev_frame, from_=0, to=90, 
                               variable=self.elevation_var, 
                               orient=tk.HORIZONTAL)
        elev_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        elev_slider.bind("<ButtonRelease-1>", lambda e: self.update_camera_view())
        
        # Azimuth (horizontal) angle
        azim_frame = ttk.Frame(view_frame)
        azim_frame.pack(fill=tk.X, pady=5)
        ttk.Label(azim_frame, text="Azimuth:").pack(side=tk.LEFT, padx=5)
        self.azimuth_var = tk.IntVar(value=30)
        azim_slider = ttk.Scale(azim_frame, from_=0, to=360, 
                               variable=self.azimuth_var, 
                               orient=tk.HORIZONTAL)
        azim_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        azim_slider.bind("<ButtonRelease-1>", lambda e: self.update_camera_view())
        
        # View preset buttons
        view_presets_frame = ttk.Frame(view_frame)
        view_presets_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(view_presets_frame, text="Front", width=6,
                  command=lambda: self.set_view_preset('front')).pack(side=tk.LEFT, padx=2)
        ttk.Button(view_presets_frame, text="Side", width=6,
                  command=lambda: self.set_view_preset('side')).pack(side=tk.LEFT, padx=2)
        ttk.Button(view_presets_frame, text="Top", width=6,
                  command=lambda: self.set_view_preset('top')).pack(side=tk.LEFT, padx=2)
        ttk.Button(view_presets_frame, text="Iso", width=6,
                  command=lambda: self.set_view_preset('iso')).pack(side=tk.LEFT, padx=2)
        
        # Action buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Reset View", command=self.reset_view).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Use Optimized Orientation", 
                  command=self.use_optimized_orientation).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Apply to Main View", 
                  command=self.apply_to_main_view).pack(side=tk.LEFT, padx=5)
        
        # Information display
        info_frame = ttk.LabelFrame(left_frame, text="Tensor Information")
        info_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.tensor_info_text = tk.Text(info_frame, height=8, width=35)
        self.tensor_info_text.pack(fill=tk.X, pady=5, padx=5)
        
        # === Right frame - 3D Plot ===
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, right_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add a screenshot button to the toolbar
        screenshot_button = ttk.Button(self.toolbar, text="Screenshot", 
                                     command=self.save_screenshot)
        screenshot_button.pack(side=tk.RIGHT, padx=5)
        
        # Initialize the 3D plot
        self.initialize_plot()
    
    def initialize_plot(self):
        """Initialize the 3D plot with basic settings"""
        if not hasattr(self, 'ax') or not hasattr(self, 'canvas'):
            return
            
        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Raman Tensor & Crystal Visualization')
        
        # Set equal aspect ratio
        self.ax.set_box_aspect([1, 1, 1])
        
        # Enable proper 3D depth rendering
        self.ax.set_proj_type('persp')  # Use perspective projection
        
        # Set view angle from the current elevation and azimuth values
        self.ax.view_init(elev=self.elevation_var.get(), azim=self.azimuth_var.get())
        
        # Add grid for better depth perception
        self.ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add coordinate axes
        self.plot_coordinate_axes()
        
        # Add laser direction
        self.plot_laser_direction()
    
    def plot_coordinate_axes(self):
        """Plot the coordinate axes"""
        # Draw the coordinate axes
        origin = np.zeros(3)
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])
        
        # Plot the axes
        self.ax.quiver(origin[0], origin[1], origin[2], 
                      x_axis[0], x_axis[1], x_axis[2], 
                      color='r', alpha=0.5, linestyle='-', 
                      arrow_length_ratio=0.15)
        self.ax.quiver(origin[0], origin[1], origin[2], 
                      y_axis[0], y_axis[1], y_axis[2], 
                      color='g', alpha=0.5, linestyle='-', 
                      arrow_length_ratio=0.15)
        self.ax.quiver(origin[0], origin[1], origin[2], 
                      z_axis[0], z_axis[1], z_axis[2], 
                      color='b', alpha=0.5, linestyle='-', 
                      arrow_length_ratio=0.15)
        
        # Add labels for the axes
        self.ax.text(1.1, 0, 0, "X", color='r', fontsize=12)
        self.ax.text(0, 1.1, 0, "Y", color='g', fontsize=12)
        self.ax.text(0, 0, 1.1, "Z", color='b', fontsize=12)
    
    def plot_laser_direction(self):
        """Plot the laser direction"""
        origin = np.zeros(3)
        length = 2.0
        
        # Laser direction vector (along z-axis)
        self.ax.quiver(origin[0], origin[1], origin[2], 
                      self.laser_direction[0], self.laser_direction[1], self.laser_direction[2], 
                      length=length, color='orange', arrow_length_ratio=0.1, linestyle='--', alpha=0.7)
        
        # Add text label
        self.ax.text(self.laser_direction[0]*length*1.1, 
                    self.laser_direction[1]*length*1.1, 
                    self.laser_direction[2]*length*1.1, 
                    "Laser", color='orange')
    
    def update_visualization(self):
        """Update the 3D visualization with current settings"""
        # Clear previous plot
        self.ax.clear()
        
        # Initialize the plot with axes and labels
        self.initialize_plot()
        
        # Get current settings
        show_tensor = self.show_tensor_var.get()
        show_crystal = self.show_crystal_var.get()
        tensor_scale = self.tensor_scale_var.get()
        crystal_scale = self.crystal_scale_var.get()
        
        # Get current orientation
        phi = self.phi_var.get()
        theta = self.theta_var.get()
        psi = self.psi_var.get()
        
        # Convert to radians for calculation
        phi_rad = np.radians(phi)
        theta_rad = np.radians(theta)
        psi_rad = np.radians(psi)
        
        # Calculate rotation matrix
        rotation_matrix = self.euler_to_rotation_matrix(phi_rad, theta_rad, psi_rad)
        
        # Check if we have a valid crystal structure
        has_valid_structure = False
        if hasattr(self, 'crystal_structure') and self.crystal_structure is not None:
            if hasattr(self.crystal_structure, 'raman_tensor'):
                has_valid_structure = True
        
        # If we don't have a valid structure, create a dummy one for visualization
        if not has_valid_structure:
            # Use a diagonal tensor as example
            tensor = np.diag([1.0, 0.5, 0.3])
            
            # Display tensor info
            self.display_tensor_info(tensor)
            
            # Plot tensor if enabled
            if show_tensor:
                self.plot_tensor_ellipsoid(tensor, scale=tensor_scale)
            
            # Plot a default cubic crystal if enabled
            if show_crystal:
                try:
                    self.plot_crystal_shape("m-3m", rotation_matrix, scale=crystal_scale)
                except Exception as e:
                    print(f"Error plotting default crystal shape: {e}")
                    
            # Set reasonable limits
            max_range = 2.0 * max(tensor_scale, crystal_scale)
            self.ax.set_xlim(-max_range, max_range)
            self.ax.set_ylim(-max_range, max_range)
            self.ax.set_zlim(-max_range, max_range)
            
            # Draw the plot
            self.canvas.draw()
            return
            
        # Get the selected character
        character = self.character_var.get()
        
        # Get the Raman tensor (use the first one by default)
        tensor = None
        
        # Try to get tensor based on character
        if character != "All" and hasattr(self.crystal_structure, 'activities'):
            activities = self.crystal_structure.activities
            if activities and hasattr(self.crystal_structure, 'raman_tensor'):
                tensor = self.crystal_structure.raman_tensor
                
                # If the tensor is a list/array of tensors, find the one with matching character
                if isinstance(tensor, list) or (isinstance(tensor, np.ndarray) and len(tensor.shape) > 2):
                    for i, activity in enumerate(activities):
                        if activity and character in activity:
                            if i < len(tensor):
                                tensor = tensor[i]
                                break
        else:
            # Get the full Raman tensor
            if hasattr(self.crystal_structure, 'raman_tensor'):
                tensor = self.crystal_structure.raman_tensor
                
                # If it's a list/array of tensors, use the first one
                if isinstance(tensor, list) or (isinstance(tensor, np.ndarray) and len(tensor.shape) > 2):
                    tensor = tensor[0]
        
        # Validate tensor
        valid_tensor = False
        if tensor is not None:
            if isinstance(tensor, np.ndarray) and tensor.shape == (3, 3):
                valid_tensor = True
            
        # Display tensor info
        if valid_tensor:
            # Apply the rotation to the tensor: R * T * R^T
            rotated_tensor = rotation_matrix @ tensor @ rotation_matrix.T
            self.display_tensor_info(rotated_tensor)
            
            # Plot the tensor if enabled
            if show_tensor:
                try:
                    self.plot_tensor_ellipsoid(rotated_tensor, scale=tensor_scale)
                except Exception as e:
                    print(f"Error plotting tensor ellipsoid: {e}")
        else:
            # Display a message that no valid tensor was found
            self.tensor_info_text.delete(1.0, tk.END)
            self.tensor_info_text.insert(tk.END, "No valid Raman tensor found for the selected character.")
        
        # Plot the crystal shape if enabled
        if show_crystal:
            # Get crystal point group (or use cubic as default)
            point_group = "m-3m"  # Default to cubic
            if hasattr(self, 'crystal_structure') and self.crystal_structure:
                if hasattr(self.crystal_structure, 'point_group'):
                    point_group = self.crystal_structure.point_group
            
            # Plot the crystal shape
            try:
                self.plot_crystal_shape(point_group, rotation_matrix, scale=crystal_scale)
            except Exception as e:
                print(f"Error plotting crystal shape: {e}")
                # Fall back to a simple cube if there's an error
                try:
                    self.plot_crystal_shape("m-3m", rotation_matrix, scale=crystal_scale)
                except:
                    pass
        
        # Set reasonable limits
        max_range = 2.0 * max(tensor_scale, crystal_scale)
        self.ax.set_xlim(-max_range, max_range)
        self.ax.set_ylim(-max_range, max_range)
        self.ax.set_zlim(-max_range, max_range)
        
        # Draw the plot
        self.canvas.draw()
    
    def plot_tensor_ellipsoid(self, tensor, scale=1.0):
        """Plot the Raman tensor as an ellipsoid with improved 3D appearance"""
        # Check if this is a T-type tensor (zeros on diagonal, non-zero off-diagonal)
        is_t_type = False
        tensor_abs = np.abs(tensor)
        diag_sum = np.trace(tensor_abs)
        offdiag_sum = np.sum(tensor_abs) - diag_sum
        
        # Detect T-type pattern: mostly zeros on diagonal, significant off-diagonal elements
        if diag_sum < 0.2 * offdiag_sum and offdiag_sum > 0:
            is_t_type = True
            print("T-type tensor detected, using specialized visualization")
        
        if is_t_type:
            # Special visualization for T-type tensors
            self.plot_t_type_tensor(tensor, scale)
        else:
            # Standard eigendecomposition approach for other tensor types
            # Perform eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(tensor)
            
            # Create a sphere with higher resolution for smoother appearance
            u = np.linspace(0, 2 * np.pi, 40)
            v = np.linspace(0, np.pi, 30)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones_like(u), np.cos(v))
            
            # Stack the coordinates
            points = np.vstack([x.flatten(), y.flatten(), z.flatten()])
            
            # Transform the sphere to an ellipsoid using the tensor
            # Create scaling matrix from eigenvalues
            scaling_matrix = np.diag(np.abs(eigenvalues))
            
            # Transform points
            transformed_points = eigenvectors @ scaling_matrix @ eigenvectors.T @ points
            
            # Apply scaling
            transformed_points *= scale
            
            # Reshape back
            x_new = transformed_points[0].reshape(x.shape)
            y_new = transformed_points[1].reshape(y.shape)
            z_new = transformed_points[2].reshape(z.shape)
            
            # Plot surface with enhanced shading if enabled
            if self.enhanced_shading_var.get():
                try:
                    # Create custom colormap with a gradient for better depth perception
                    # Blue to cyan gradient
                    tensor_cmap = LinearSegmentedColormap.from_list('tensor_cmap', 
                                                                ['darkblue', 'royalblue', 'skyblue', 'cyan'], 
                                                                N=256)
                    
                    # Add a light source for 3D shading
                    ls = LightSource(azdeg=225, altdeg=45)
                    
                    # Calculate the surface normals for shading
                    nx = np.zeros_like(x_new)
                    ny = np.zeros_like(y_new)
                    nz = np.zeros_like(z_new)
                    
                    # Approximation of normals (this is a simplification, better results would need proper normal calculation)
                    for i in range(1, x_new.shape[0]-1):
                        for j in range(1, x_new.shape[1]-1):
                            vx1 = np.array([x_new[i+1,j] - x_new[i-1,j], 
                                        y_new[i+1,j] - y_new[i-1,j], 
                                        z_new[i+1,j] - z_new[i-1,j]])
                            vx2 = np.array([x_new[i,j+1] - x_new[i,j-1], 
                                        y_new[i,j+1] - y_new[i,j-1], 
                                        z_new[i,j+1] - z_new[i,j-1]])
                            normal = np.cross(vx1, vx2)
                            norm = np.linalg.norm(normal)
                            if norm > 0:
                                normal = normal / norm
                                nx[i,j], ny[i,j], nz[i,j] = normal
                    
                    # Color data based on z-coordinate for better visual contrast
                    illumination = 0.5 + 0.5 * nz  # Simple illumination model
                    
                    # Plot the ellipsoid with illuminated shading
                    surf = self.ax.plot_surface(
                        x_new, y_new, z_new, 
                        facecolors=tensor_cmap(illumination),
                        alpha=0.6,
                        rstride=1, cstride=1,
                        linewidth=0.1,
                        antialiased=True,
                        shade=True
                    )
                except Exception as e:
                    # Fallback to standard rendering if enhanced shading fails
                    print(f"Enhanced tensor shading error: {e}")
                    surf = self.ax.plot_surface(
                        x_new, y_new, z_new, 
                        color='royalblue',
                        alpha=0.5,
                        rstride=2, cstride=2,
                        linewidth=0.1,
                        antialiased=True
                    )
            else:
                # Standard rendering without enhanced shading
                surf = self.ax.plot_surface(
                    x_new, y_new, z_new, 
                    color='royalblue',
                    alpha=0.5,
                    rstride=2, cstride=2,
                    linewidth=0.1,
                    antialiased=True
                )
            
            # Add principal ellipses (intersections with principal planes) if enabled
            if self.show_ellipses_var.get():
                # XY plane
                theta_xy = np.linspace(0, 2 * np.pi, 100)
                xy_ellipse_x = np.zeros(100)
                xy_ellipse_y = np.zeros(100)
                
                # Calculate ellipse in XY plane
                for i, angle in enumerate(theta_xy):
                    vec = np.array([np.cos(angle), np.sin(angle), 0])
                    transformed = eigenvectors @ np.sqrt(scaling_matrix) @ eigenvectors.T @ vec
                    xy_ellipse_x[i] = transformed[0] * scale
                    xy_ellipse_y[i] = transformed[1] * scale
                
                # Plot XY principal ellipse with dashes
                self.ax.plot(xy_ellipse_x, xy_ellipse_y, np.zeros_like(xy_ellipse_x), 
                            'b--', linewidth=1, alpha=0.8)
                
                # XZ plane
                theta_xz = np.linspace(0, 2 * np.pi, 100)
                xz_ellipse_x = np.zeros(100)
                xz_ellipse_z = np.zeros(100)
                
                # Calculate ellipse in XZ plane
                for i, angle in enumerate(theta_xz):
                    vec = np.array([np.cos(angle), 0, np.sin(angle)])
                    transformed = eigenvectors @ np.sqrt(scaling_matrix) @ eigenvectors.T @ vec
                    xz_ellipse_x[i] = transformed[0] * scale
                    xz_ellipse_z[i] = transformed[2] * scale
                
                # Plot XZ principal ellipse with dashes
                self.ax.plot(xz_ellipse_x, np.zeros_like(xz_ellipse_x), xz_ellipse_z, 
                            'g--', linewidth=1, alpha=0.8)
                
                # YZ plane
                theta_yz = np.linspace(0, 2 * np.pi, 100)
                yz_ellipse_y = np.zeros(100)
                yz_ellipse_z = np.zeros(100)
                
                # Calculate ellipse in YZ plane
                for i, angle in enumerate(theta_yz):
                    vec = np.array([0, np.cos(angle), np.sin(angle)])
                    transformed = eigenvectors @ np.sqrt(scaling_matrix) @ eigenvectors.T @ vec
                    yz_ellipse_y[i] = transformed[1] * scale
                    yz_ellipse_z[i] = transformed[2] * scale
                
                # Plot YZ principal ellipse with dashes
                self.ax.plot(np.zeros_like(yz_ellipse_y), yz_ellipse_y, yz_ellipse_z, 
                            'r--', linewidth=1, alpha=0.8)
            
            # Plot the principal axes
            origin = np.zeros(3)
            for i in range(3):
                # Get eigenvector and scale by eigenvalue
                direction = eigenvectors[:, i] * np.abs(eigenvalues[i]) * scale
                
                # Color based on eigenvalue sign
                color = 'r' if eigenvalues[i] < 0 else 'g'
                
                # Plot bidirectional arrow
                self.ax.quiver(origin[0], origin[1], origin[2], 
                            direction[0], direction[1], direction[2], 
                            color=color, arrow_length_ratio=0.1, linewidth=2)
                self.ax.quiver(origin[0], origin[1], origin[2], 
                            -direction[0], -direction[1], -direction[2], 
                            color=color, arrow_length_ratio=0.1, linewidth=2)
                
                # Add small spheres at the ends for better visibility
                self.ax.scatter(direction[0], direction[1], direction[2], 
                            color=color, s=50, edgecolor='black')
                self.ax.scatter(-direction[0], -direction[1], -direction[2], 
                            color=color, s=50, edgecolor='black')
    
    def plot_t_type_tensor(self, tensor, scale=1.0):
        """Specialized plotting for T-type tensors (T1g, T2g, etc.) with volumetric lobes"""
        # For T-type tensor, we use a specialized visualization to show the directional lobes
        # Find the major off-diagonal component
        abs_tensor = np.abs(tensor)
        max_val = np.max(abs_tensor)
        
        # Detect which off-diagonal components are present
        has_xy = abs_tensor[0, 1] > 0.1 * max_val or abs_tensor[1, 0] > 0.1 * max_val
        has_xz = abs_tensor[0, 2] > 0.1 * max_val or abs_tensor[2, 0] > 0.1 * max_val
        has_yz = abs_tensor[1, 2] > 0.1 * max_val or abs_tensor[2, 1] > 0.1 * max_val
        
        # Create a custom grid for spherical coordinates
        n_phi = 50    # Number of points in phi direction (latitude)
        n_theta = 100  # Number of points in theta direction (longitude)
        
        phi = np.linspace(0, np.pi, n_phi)
        theta = np.linspace(0, 2 * np.pi, n_theta)
        
        # Create meshgrid of spherical coordinates
        PHI, THETA = np.meshgrid(phi, theta)
        
        # Base scaling factor
        R = scale
        
        # Calculate color based on tensor sign
        def tensor_color(val):
            return 'r' if val < 0 else 'g'
        
        # Create tensor-based shape with 4 lobes (for T2g mode)
        if has_xy:
            xy_val = (tensor[0, 1] + tensor[1, 0]) / 2  # Average the symmetric components
            color = tensor_color(xy_val)
            
            # Create a four-leaf clover pattern in the XY plane
            # This is the 3D equivalent of d_xy orbital
            r = R * (1.0 + 2.0 * np.sin(2*THETA) * np.sin(PHI)**2)
            
            # Convert to Cartesian coordinates
            X = r * np.sin(PHI) * np.cos(THETA)
            Y = r * np.sin(PHI) * np.sin(THETA)
            Z = r * np.cos(PHI)
            
            # Create colormap for depth perception
            cmap = plt.cm.viridis if xy_val >= 0 else plt.cm.plasma
            
            # Calculate normalized radius for coloring
            radius_norm = np.sqrt(X**2 + Y**2 + Z**2) / (R * 3)
            
            # Plot the 3D surface with colormap
            surf_xy = self.ax.plot_surface(
                X, Y, Z,
                facecolors=cmap(radius_norm),
                alpha=0.7,
                rstride=1, cstride=1,
                linewidth=0.1,
                antialiased=True,
                shade=True
            )
        
        elif has_xz:
            xz_val = (tensor[0, 2] + tensor[2, 0]) / 2
            color = tensor_color(xz_val)
            
            # Create a four-leaf clover pattern in the XZ plane
            # This is the 3D equivalent of d_xz orbital
            r = R * (1.0 + 2.0 * np.sin(THETA) * np.cos(THETA) * np.sin(PHI) * np.cos(PHI))
            
            # Convert to Cartesian coordinates
            X = r * np.sin(PHI) * np.cos(THETA)
            Y = r * np.sin(PHI) * np.sin(THETA)
            Z = r * np.cos(PHI)
            
            # Create colormap for depth perception
            cmap = plt.cm.viridis if xz_val >= 0 else plt.cm.plasma
            
            # Calculate normalized radius for coloring
            radius_norm = np.sqrt(X**2 + Y**2 + Z**2) / (R * 3)
            
            # Plot the 3D surface with colormap
            surf_xz = self.ax.plot_surface(
                X, Y, Z,
                facecolors=cmap(radius_norm),
                alpha=0.7,
                rstride=1, cstride=1,
                linewidth=0.1,
                antialiased=True,
                shade=True
            )
            
        elif has_yz:
            yz_val = (tensor[1, 2] + tensor[2, 1]) / 2
            color = tensor_color(yz_val)
            
            # Create a four-leaf clover pattern in the YZ plane
            # This is the 3D equivalent of d_yz orbital
            r = R * (1.0 + 2.0 * np.sin(THETA) * np.cos(THETA) * np.sin(PHI) * np.cos(PHI))
            
            # Rotate to align with YZ plane by swapping coordinates
            X = r * np.sin(PHI) * np.sin(THETA)  # Y coordinate
            Y = r * np.cos(PHI)                  # Z coordinate
            Z = r * np.sin(PHI) * np.cos(THETA)  # X coordinate
            
            # Create colormap for depth perception
            cmap = plt.cm.viridis if yz_val >= 0 else plt.cm.plasma
            
            # Calculate normalized radius for coloring
            radius_norm = np.sqrt(X**2 + Y**2 + Z**2) / (R * 3)
            
            # Plot the 3D surface with colormap
            surf_yz = self.ax.plot_surface(
                Z, X, Y,  # Reordering to get YZ plane
                facecolors=cmap(radius_norm),
                alpha=0.7,
                rstride=1, cstride=1,
                linewidth=0.1,
                antialiased=True,
                shade=True
            )
        
        # Create a more accurate multiple-component T2g representation when all components present
        if has_xy and has_yz and has_xz:
            # Clear existing plots to create a combined visualization
            self.ax.clear()
            self.initialize_plot()
            
            # Create a complete T2g shape with all 3 components
            # This creates a shape similar to the academic literature tensor visualization
            
            # Define d-orbital-like functions for T2g components
            def d_xy(phi, theta):
                return np.sin(phi)**2 * np.sin(2*theta)
                
            def d_xz(phi, theta):
                return np.sin(phi) * np.cos(phi) * np.cos(theta)
                
            def d_yz(phi, theta):
                return np.sin(phi) * np.cos(phi) * np.sin(theta)
            
            # Combine all components to create the T2g shape
            # Using the tensor values to weight each component
            t_xy = abs(tensor[0, 1] + tensor[1, 0]) / 2
            t_xz = abs(tensor[0, 2] + tensor[2, 0]) / 2
            t_yz = abs(tensor[1, 2] + tensor[2, 1]) / 2
            
            # Normalize weights
            total = t_xy + t_xz + t_yz
            if total > 0:
                t_xy /= total
                t_xz /= total
                t_yz /= total
            
            # Calculate the combined radius function
            r = R * (1.0 + 2.0 * (t_xy * d_xy(PHI, THETA) + 
                                  t_xz * d_xz(PHI, THETA) + 
                                  t_yz * d_yz(PHI, THETA)))
            
            # Convert to Cartesian coordinates
            X = r * np.sin(PHI) * np.cos(THETA)
            Y = r * np.sin(PHI) * np.sin(THETA)
            Z = r * np.cos(PHI)
            
            # Create colormap for enhanced visualization
            # Use viridis colormap for positive tensor, plasma for negative
            tensor_sign = np.sign(tensor[0, 1] + tensor[1, 0] + tensor[0, 2] + tensor[2, 0] + tensor[1, 2] + tensor[2, 1])
            cmap = plt.cm.viridis if tensor_sign >= 0 else plt.cm.plasma
            
            # Calculate normalized radius for coloring
            radius_norm = np.sqrt(X**2 + Y**2 + Z**2) / (R * 3)
            
            # Plot the 3D surface with colormap
            surf = self.ax.plot_surface(
                X, Y, Z,
                facecolors=cmap(radius_norm),
                alpha=0.8,
                rstride=1, cstride=1,
                linewidth=0.2,
                antialiased=True,
                shade=True
            )
            
            # Add coordinate axes back
            self.plot_coordinate_axes()
            
            # Add laser direction
            self.plot_laser_direction()
        
        # Add directional arrows showing tensor components
        arrow_scale = scale * 1.2
        
        # Show XY component if significant
        if has_xy:
            self.ax.quiver(0, 0, 0, arrow_scale, 0, 0, color='r', arrow_length_ratio=0.1, alpha=0.7)
            self.ax.quiver(0, 0, 0, 0, arrow_scale, 0, color='g', arrow_length_ratio=0.1, alpha=0.7)
            self.ax.text(arrow_scale*1.1, 0, 0, "X", color='r', fontsize=10)
            self.ax.text(0, arrow_scale*1.1, 0, "Y", color='g', fontsize=10)
        
        # Show XZ component if significant
        if has_xz:
            self.ax.quiver(0, 0, 0, arrow_scale, 0, 0, color='r', arrow_length_ratio=0.1, alpha=0.7)
            self.ax.quiver(0, 0, 0, 0, 0, arrow_scale, color='b', arrow_length_ratio=0.1, alpha=0.7)
            if not has_xy:  # Only add X label if not added already
                self.ax.text(arrow_scale*1.1, 0, 0, "X", color='r', fontsize=10)
            self.ax.text(0, 0, arrow_scale*1.1, "Z", color='b', fontsize=10)
        
        # Show YZ component if significant
        if has_yz:
            if not has_xy:  # Only add Y arrow if not added already
                self.ax.quiver(0, 0, 0, 0, arrow_scale, 0, color='g', arrow_length_ratio=0.1, alpha=0.7)
                self.ax.text(0, arrow_scale*1.1, 0, "Y", color='g', fontsize=10)
            if not has_xz:  # Only add Z arrow if not added already
                self.ax.quiver(0, 0, 0, 0, 0, arrow_scale, color='b', arrow_length_ratio=0.1, alpha=0.7)
                self.ax.text(0, 0, arrow_scale*1.1, "Z", color='b', fontsize=10)
    
    def plot_crystal_shape(self, point_group, rotation_matrix, scale=1.0):
        """Plot a basic crystal shape based on point group symmetry with improved 3D appearance"""
        # Determine the shape based on point group
        shape = self.point_group_shapes.get(point_group, "cube")
        
        # Create vertices based on shape
        vertices = []
        
        if shape == "cube" or shape == "orthorhombic_prism":
            # For cube and orthorhombic, use a rectangular prism
            # Adjust dimensions for orthorhombic
            a, b, c = (1.0, 1.0, 1.0) if shape == "cube" else (1.0, 0.8, 0.6)
            vertices = np.array([
                [ a,  b,  c], [ a,  b, -c], [ a, -b,  c], [ a, -b, -c],
                [-a,  b,  c], [-a,  b, -c], [-a, -b,  c], [-a, -b, -c]
            ])
        
        elif shape == "tetragonal_prism":
            # A square prism
            a, c = 1.0, 1.3
            vertices = np.array([
                [ a,  a,  c], [ a,  a, -c], [ a, -a,  c], [ a, -a, -c],
                [-a,  a,  c], [-a,  a, -c], [-a, -a,  c], [-a, -a, -c]
            ])
        
        elif shape == "hexagonal_prism":
            # Hexagonal prism
            c = 1.3
            for i in range(6):
                angle = i * 2 * np.pi / 6
                x = np.cos(angle)
                y = np.sin(angle)
                vertices.append([x, y, c])
                vertices.append([x, y, -c])
            vertices = np.array(vertices)
        
        elif shape == "trigonal_prism":
            # Trigonal prism
            c = 1.3
            for i in range(3):
                angle = i * 2 * np.pi / 3
                x = np.cos(angle)
                y = np.sin(angle)
                vertices.append([x, y, c])
                vertices.append([x, y, -c])
            vertices = np.array(vertices)
        
        elif shape == "monoclinic_prism":
            # Monoclinic prism (parallelogram base)
            a, b, c = 1.0, 0.8, 1.0
            beta = np.radians(110)  # Monoclinic angle
            
            # Create vertices with the monoclinic angle
            vertices = np.array([
                [a, 0, c], [a, 0, -c], 
                [a, b, c], [a, b, -c],
                [0, 0, c], [0, 0, -c], 
                [0, b, c], [0, b, -c]
            ])
            
            # Apply monoclinic transformation
            monoclinic_transform = np.array([
                [1, 0, np.cos(beta)],
                [0, 1, 0],
                [0, 0, np.sin(beta)]
            ])
            vertices = vertices @ monoclinic_transform.T
        
        elif shape == "triclinic_prism":
            # Triclinic prism (general parallelepiped)
            # Create a basis with three non-orthogonal vectors
            alpha = np.radians(80)  # angle between b and c
            beta = np.radians(85)   # angle between a and c
            gamma = np.radians(75)  # angle between a and b
            
            a = 1.0
            b = 0.8
            c = 0.6
            
            # Triclinic basis vectors
            basis = np.array([
                [a, 0, 0],
                [b * np.cos(gamma), b * np.sin(gamma), 0],
                [c * np.cos(beta), 
                 c * (np.cos(alpha) - np.cos(beta)*np.cos(gamma))/np.sin(gamma),
                 c * np.sqrt(1 - np.cos(beta)**2 - 
                           ((np.cos(alpha) - np.cos(beta)*np.cos(gamma))/np.sin(gamma))**2)]
            ])
            
            # Create the parallelepiped vertices
            corners = np.array([
                [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
            ])
            
            # Transform using the basis
            vertices = corners @ basis
        
        elif shape == "octahedron":
            # Octahedron
            r = 1.0
            vertices = np.array([
                [r, 0, 0], [-r, 0, 0], [0, r, 0], 
                [0, -r, 0], [0, 0, r], [0, 0, -r]
            ])
        
        elif shape == "tetrahedron":
            # Tetrahedron
            r = 1.0
            vertices = np.array([
                [r, r, r], [-r, -r, r], [-r, r, -r], [r, -r, -r]
            ])
        
        # Scale vertices
        vertices *= scale
        
        # Apply rotation
        rotated_vertices = vertices @ rotation_matrix.T
        
        # Define the faces based on the shape
        faces = []
        
        if shape in ["cube", "tetragonal_prism", "orthorhombic_prism", "monoclinic_prism", "triclinic_prism"]:
            # For prism shapes - 6 faces (top, bottom, 4 sides)
            # Bottom face (vertices 0,1,3,2)
            faces.append([0, 1, 3, 2])
            # Top face (vertices 4,5,7,6)
            faces.append([4, 6, 7, 5])
            # Side faces
            faces.append([0, 2, 6, 4])
            faces.append([1, 5, 7, 3])
            faces.append([0, 4, 5, 1])
            faces.append([2, 3, 7, 6])
        
        elif shape == "hexagonal_prism":
            # Bottom hexagon
            faces.append([i for i in range(0, 12, 2)])
            # Top hexagon
            faces.append([i for i in range(1, 12, 2)])
            # Side rectangles
            for i in range(6):
                faces.append([2*i, 2*i+1, 2*((i+1)%6)+1, 2*((i+1)%6)])
        
        elif shape == "trigonal_prism":
            # Bottom triangle
            faces.append([0, 2, 4])
            # Top triangle
            faces.append([1, 3, 5])
            # Side rectangles
            for i in range(3):
                faces.append([2*i, 2*i+1, 2*((i+1)%3)+1, 2*((i+1)%3)])
        
        elif shape == "octahedron":
            # 8 triangular faces
            faces.append([0, 2, 4])
            faces.append([0, 4, 3])
            faces.append([0, 3, 5])
            faces.append([0, 5, 2])
            faces.append([1, 2, 4])
            faces.append([1, 4, 3])
            faces.append([1, 3, 5])
            faces.append([1, 5, 2])
        
        elif shape == "tetrahedron":
            # 4 triangular faces
            faces.append([0, 1, 2])
            faces.append([0, 2, 3])
            faces.append([0, 3, 1])
            faces.append([1, 3, 2])
        
        # Create colormap for faces
        n_faces = len(faces)
        crystal_colors = plt.cm.Blues(np.linspace(0.4, 0.8, n_faces))
        
        # Add a light source for 3D shading
        ls = LightSource(azdeg=225, altdeg=45)
        
        # Plot each face as a filled polygon with proper shading
        for i, face in enumerate(faces):
            # Extract face vertices
            face_vertices = rotated_vertices[face]
            
            # Create a polygon for this face
            crystal_alpha = self.crystal_alpha_var.get() if hasattr(self, 'crystal_alpha_var') else 0.25
            poly = Poly3DCollection([face_vertices], alpha=crystal_alpha)
            
            # Set face color with lighting
            base_color = crystal_colors[i]
            
            # Prevent hillshade error for small surfaces
            if self.enhanced_shading_var.get() and len(face_vertices) >= 3:
                try:
                    # Use a simple lighting model instead of hillshade
                    # Calculate face normal
                    face_normal = np.cross(face_vertices[1]-face_vertices[0], face_vertices[-1]-face_vertices[0])
                    if np.linalg.norm(face_normal) > 0:
                        face_normal = face_normal / np.linalg.norm(face_normal)
                    
                    # Light direction (45° elevation, 45° azimuth)
                    light_dir = np.array([0.5, 0.5, 0.7071])
                    light_dir = light_dir / np.linalg.norm(light_dir)
                    
                    # Ambient and diffuse lighting
                    ambient = 0.3
                    diffuse = max(0, np.dot(face_normal, light_dir))
                    illumination = ambient + (1 - ambient) * diffuse
                    
                    # Apply lighting to base color
                    face_color = np.array(base_color[:3]) * illumination
                except Exception as e:
                    # Fallback to simple coloring if shading fails
                    print(f"Simple shading error: {e}")
                    face_color = base_color[:3]
            else:
                # No enhanced shading or too few vertices
                face_color = base_color[:3]
                
            poly.set_facecolor(face_color)
            
            # Add edge highlighting
            poly.set_edgecolor('k')
            poly.set_linewidth(1.0)
            
            # Add the face to the plot
            self.ax.add_collection3d(poly)
            
            # Calculate face center for normal vector
            face_center = np.mean(face_vertices, axis=0)
            
            # Calculate face normal (simplified)
            if len(face) >= 3:
                v1 = face_vertices[1] - face_vertices[0]
                v2 = face_vertices[2] - face_vertices[0]
                normal = np.cross(v1, v2)
                normal = normal / np.linalg.norm(normal) * 0.2 * scale  # Scale normal vector
                
                # Plot normal vector
                self.ax.quiver(face_center[0], face_center[1], face_center[2],
                              normal[0], normal[1], normal[2],
                              color='r', arrow_length_ratio=0.1, alpha=0.5, linewidth=1.5)
    
    def euler_to_rotation_matrix(self, phi, theta, psi):
        """Convert Euler angles (degrees) to rotation matrix"""
        # Convert to radians
        phi = np.radians(phi)
        theta = np.radians(theta)
        psi = np.radians(psi)
        
        # Calculate rotation matrices
        R_z1 = np.array([
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1]
        ])
        
        R_y = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        
        R_z2 = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        
        # Full rotation matrix
        R = R_z2 @ R_y @ R_z1
        
        return R
    
    def display_tensor_info(self, tensor):
        """Display tensor information in the text box"""
        # Clear existing text
        self.tensor_info_text.delete(1.0, tk.END)
        
        # Format the tensor
        tensor_text = "Raman Tensor:\n"
        for i in range(3):
            tensor_text += "[{:6.3f} {:6.3f} {:6.3f}]\n".format(
                tensor[i, 0], tensor[i, 1], tensor[i, 2])
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(tensor)
        
        # Add eigenvalue information
        tensor_text += "\nEigenvalues:\n"
        for i, val in enumerate(eigenvalues):
            tensor_text += "λ{}: {:6.3f}\n".format(i+1, val)
        
        # Insert the text
        self.tensor_info_text.insert(tk.END, tensor_text)
    
    def reset_view(self):
        """Reset the view to default orientation"""
        # Reset Euler angles
        self.phi_var.set(0.0)
        self.theta_var.set(0.0)
        self.psi_var.set(0.0)
        
        # Update the visualization
        self.update_visualization()
    
    def use_optimized_orientation(self):
        """Set the orientation to the optimized values"""
        if hasattr(self, 'optimized_orientation') and self.optimized_orientation:
            # Set Euler angles to optimized values
            self.phi_var.set(self.optimized_orientation[0])
            self.theta_var.set(self.optimized_orientation[1])
            self.psi_var.set(self.optimized_orientation[2])
            
            # Update the visualization
            self.update_visualization()
        else:
            messagebox.showinfo("Orientation", "No optimized orientation available.")
    
    def apply_to_main_view(self):
        """Apply the current orientation to the main application view"""
        # This method should be overridden by the main application
        messagebox.showinfo("Not Implemented", 
                           "This function should be connected to the main application.")
    
    def set_crystal_structure(self, crystal_structure):
        """Set the crystal structure for visualization"""
        self.crystal_structure = crystal_structure
        self.update_visualization()
    
    def set_optimized_orientation(self, orientation):
        """Set the optimized orientation"""
        self.optimized_orientation = orientation
        
        # Update UI to reflect new orientation
        self.phi_var.set(orientation[0])
        self.theta_var.set(orientation[1])
        self.psi_var.set(orientation[2])
        
        # Update visualization
        self.update_visualization()
        
    def update_character_options(self, characters):
        """Update the available character options in the dropdown"""
        if not characters:
            characters = ["All", "A1", "A2", "E", "T1", "T2"]
        else:
            characters = ["All"] + list(set(characters))
        
        # Update combobox values
        self.character_combo['values'] = characters
        
        # Set to 'All' if current value not in list
        if self.character_var.get() not in characters:
            self.character_var.set("All")

    # Add setup method for custom lighting
    def setup_lighting(self):
        """Set up custom lighting for better 3D appearance"""
        # This is a helper method that could be expanded in the future
        pass

    def update_camera_view(self):
        """Update the camera view based on slider values"""
        if hasattr(self, 'ax'):
            self.ax.view_init(elev=self.elevation_var.get(), 
                            azim=self.azimuth_var.get())
            self.canvas.draw_idle()
    
    def set_view_preset(self, preset):
        """Set a view preset for the 3D plot"""
        if preset == 'front':
            self.elevation_var.set(0)
            self.azimuth_var.set(0)
        elif preset == 'side':
            self.elevation_var.set(0)
            self.azimuth_var.set(90)
        elif preset == 'top':
            self.elevation_var.set(90)
            self.azimuth_var.set(0)
        elif preset == 'iso':
            self.elevation_var.set(30)
            self.azimuth_var.set(45)
        
        self.update_camera_view()
    
    def save_screenshot(self):
        """Save a screenshot of the current 3D visualization"""
        try:
            from tkinter import filedialog
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                title="Save Screenshot"
            )
            
            if file_path:
                # Save the figure
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Screenshot Saved", 
                                   f"Screenshot saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save screenshot: {str(e)}")


# For testing the 3D visualizer as a standalone
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Raman Tensor 3D Visualization")
    root.geometry("1000x800")
    
    # Create a sample tensor
    class DummyCrystalStructure:
        def __init__(self):
            self.raman_tensor = np.array([
                [0.8, 0.2, 0.0],
                [0.2, 0.5, 0.1],
                [0.0, 0.1, 0.3]
            ])
            self.point_group = "m-3m"  # Cubic
    
    # Create the visualizer with sample data
    visualizer = RamanTensor3DVisualizer(root, 
                                        crystal_structure=DummyCrystalStructure(),
                                        optimized_orientation=(30, 45, 60))
    visualizer.frame.pack(fill=tk.BOTH, expand=True)
    
    # Update the visualization
    visualizer.update_visualization()
    
    root.mainloop() 