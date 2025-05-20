import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import pdist
import glob
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
from matplotlib.widgets import RectangleSelector
import csv
import matplotlib.pyplot as plt

# Try to import UMAP for advanced visualization
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with 'pip install umap-learn' for additional visualization options.")

class RamanGroupAnalysisWindow:
    """Window for Raman group analysis with hierarchical clustering."""
    
    def __init__(self, parent, raman_app):
        """
        Initialize the Raman Group Analysis Window.
        
        Parameters:
        -----------
        parent : tk.Tk or tk.Toplevel
            Parent window
        raman_app : RamanAnalysisApp
            Main application instance
        """
        self.window = tk.Toplevel(parent)
        self.window.title("Raman Group Analysis")
        self.window.geometry("800x600")
        
        # Store reference to main app
        self.raman_app = raman_app
        
        # Initialize database path
        self.custom_db_path = None
        
        # Initialize variables
        self.selected_folder = None
        self.visualization_method = tk.StringVar(value='PCA')
        self.selected_points = set()
        self.refinement_mode = False
        self.n_subclusters = tk.IntVar(value=2)
        self.split_method = tk.StringVar(value='kmeans')
        
        # Initialize undo stack for refinement operations
        self.undo_stack = []
        self.max_undo_steps = 10  # Maximum number of undo steps to store
        
        # Initialize cluster data
        self.cluster_data = {
            'wavenumbers': None,
            'intensities': None,
            'features': None,
            'features_scaled': None,
            'labels': None,
            'linkage_matrix': None,
            'distance_matrix': None
        }
        
        # Create main frame
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill='both', expand=True)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_import_tab()
        self.create_clustering_tab()
        self.create_visualization_tab()
        self.create_analysis_tab()
        self.create_refinement_tab()
        
        # Create status bar
        self.status_bar = ttk.Label(self.window, text="No database loaded", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initially disable clustering tab
        self.notebook.tab(1, state='disabled')  # Disable clustering tab
        
        # Initialize UI state
        self.initialize_ui_state()
        
        # Update database status
        self.update_database_status()
        
    def initialize_ui_state(self):
        """Initialize the UI state."""
        # Disable clustering controls initially
        self.n_clusters.config(state='disabled')
        self.linkage_method_combo.config(state='disabled')
        self.distance_metric_combo.config(state='disabled')
        
        # Set initial status messages
        self.import_status.config(text="No data imported")
        self.clustering_status.config(text="No data available for clustering")

    def update_clustering_controls(self):
        """Update the state of clustering controls based on data availability."""
        try:
            print("\nDEBUG: Updating clustering controls")
            print(f"DEBUG: Cluster data keys: {self.cluster_data.keys()}")
            
            if self.cluster_data is None or self.cluster_data['features'] is None:
                print("DEBUG: No data available")
                self.n_clusters.config(state='disabled')
                self.linkage_method_combo.config(state='disabled')
                self.distance_metric_combo.config(state='disabled')
                self.clustering_status.config(text="No data available for clustering")
                return
            
            print("DEBUG: Enabling clustering controls")
            # Enable controls
            self.n_clusters.config(state='normal')
            self.linkage_method_combo.config(state='normal')
            self.distance_metric_combo.config(state='normal')
            self.clustering_status.config(text=f"Ready to cluster {len(self.cluster_data['features'])} spectra")
            
        except Exception as e:
            print(f"DEBUG: Error updating controls: {str(e)}")
            raise

    def update_ui_after_import(self, num_spectra):
        """Update UI after successful data import."""
        try:
            print("\nDEBUG: Updating UI after import")
            print(f"DEBUG: Number of spectra: {num_spectra}")
            
            # Update import status
            self.import_status.config(text=f"Successfully imported {num_spectra} spectra")
            self.import_progress['value'] = 100
            
            # Enable clustering tab
            self.notebook.tab(1, state='normal')  # Enable clustering tab
            
            # Update clustering controls
            self.update_clustering_controls()
            
            # Switch to clustering tab
            self.notebook.select(1)  # Switch to clustering tab
            
        except Exception as e:
            print(f"DEBUG: Error updating UI: {str(e)}")
            raise

    def create_gui(self):
        """Create the GUI elements."""
        # Main container with notebook for different views
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_import_tab()
        self.create_clustering_tab()
        self.create_visualization_tab()
        self.create_analysis_tab()
        self.create_refinement_tab()
        
    def create_import_tab(self):
        """Create the import tab."""
        import_frame = ttk.Frame(self.notebook)
        self.notebook.add(import_frame, text="Import")
        
        # Create folder selection frame
        folder_frame = ttk.LabelFrame(import_frame, text="Folder Selection")
        folder_frame.pack(fill="x", padx=5, pady=5)
        
        # Add folder path display
        self.folder_path_var = tk.StringVar()
        folder_path_label = ttk.Label(folder_frame, textvariable=self.folder_path_var, wraplength=400)
        folder_path_label.pack(fill="x", padx=5, pady=2)
        
        # Add select folder button
        select_folder_btn = ttk.Button(folder_frame, text="Select Folder", command=self.select_import_folder)
        select_folder_btn.pack(pady=5)
        
        # Add import from database button
        import_db_btn = ttk.Button(folder_frame, text="Import from Database", command=self.open_database_import_dialog)
        import_db_btn.pack(pady=5)
        
        # Create data configuration frame
        config_frame = ttk.LabelFrame(import_frame, text="Data Configuration")
        config_frame.pack(fill="x", padx=5, pady=5)
        
        # Add wavenumber column selection
        ttk.Label(config_frame, text="Wavenumber Column:").pack(anchor="w", padx=5, pady=2)
        self.wavenumber_col = ttk.Entry(config_frame)
        self.wavenumber_col.pack(fill="x", padx=5, pady=2)
        self.wavenumber_col.insert(0, "0")  # Default to first column
        
        # Add intensity column selection
        ttk.Label(config_frame, text="Intensity Column:").pack(anchor="w", padx=5, pady=2)
        self.intensity_col = ttk.Entry(config_frame)
        self.intensity_col.pack(fill="x", padx=5, pady=2)
        self.intensity_col.insert(0, "1")  # Default to second column
        
        # Add start import button
        start_import_btn = ttk.Button(import_frame, text="Start Import", command=self.start_batch_import)
        start_import_btn.pack(pady=10)
        
        # Add progress bar
        self.import_progress = ttk.Progressbar(import_frame, mode='determinate')
        self.import_progress.pack(fill="x", padx=5, pady=5)
        
        # Add status label
        self.import_status = ttk.Label(import_frame, text="")
        self.import_status.pack(pady=5)

    def create_clustering_tab(self):
        """Create the clustering tab."""
        clustering_frame = ttk.Frame(self.notebook)
        self.notebook.add(clustering_frame, text="Clustering")
        
        # Create controls frame
        controls_frame = ttk.LabelFrame(clustering_frame, text="Clustering Controls")
        controls_frame.pack(fill="x", padx=5, pady=5)
        
        # Add number of clusters selection
        ttk.Label(controls_frame, text="Number of Clusters:").pack(anchor="w", padx=5, pady=2)
        self.n_clusters = ttk.Spinbox(controls_frame, from_=2, to=20, width=10)
        self.n_clusters.pack(anchor="w", padx=5, pady=2)
        self.n_clusters.set(5)  # Default value
        
        # Add linkage method selection
        ttk.Label(controls_frame, text="Linkage Method:").pack(anchor="w", padx=5, pady=2)
        self.linkage_method_combo = ttk.Combobox(controls_frame, values=['ward', 'complete', 'average', 'single'])
        self.linkage_method_combo.pack(anchor="w", padx=5, pady=2)
        self.linkage_method_combo.set('ward')  # Default value
        
        # Add distance metric selection
        ttk.Label(controls_frame, text="Distance Metric:").pack(anchor="w", padx=5, pady=2)
        self.distance_metric_combo = ttk.Combobox(controls_frame, values=['euclidean', 'cosine', 'correlation'])
        self.distance_metric_combo.pack(anchor="w", padx=5, pady=2)
        self.distance_metric_combo.set('euclidean')  # Default value
        
        # Add run clustering button
        run_clustering_btn = ttk.Button(controls_frame, text="Run Clustering", command=self.run_clustering)
        run_clustering_btn.pack(pady=10)
        
        # Add progress bar
        self.clustering_progress = ttk.Progressbar(controls_frame, mode='determinate')
        self.clustering_progress.pack(fill="x", padx=5, pady=5)
        
        # Add status label
        self.clustering_status = ttk.Label(controls_frame, text="")
        self.clustering_status.pack(pady=5)
        
        # Create visualization frame
        viz_frame = ttk.LabelFrame(clustering_frame, text="Visualization")
        viz_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create notebook for different visualizations
        viz_notebook = ttk.Notebook(viz_frame)
        viz_notebook.pack(fill="both", expand=True)
        
        # Create dendrogram tab
        dendro_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(dendro_frame, text="Dendrogram")
        
        # Create dendrogram figure
        self.dendro_fig = Figure(figsize=(6, 4))
        self.dendrogram_ax = self.dendro_fig.add_subplot(111)
        self.dendro_canvas = FigureCanvasTkAgg(self.dendro_fig, master=dendro_frame)
        self.dendro_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add dendrogram toolbar
        self.dendro_toolbar = NavigationToolbar2Tk(self.dendro_canvas, dendro_frame)
        self.dendro_toolbar.update()
        
        # Create heatmap tab
        heatmap_frame = ttk.LabelFrame(viz_notebook, text="Cluster Heatmap", padding=5)
        viz_notebook.add(heatmap_frame, text="Heatmap")
        
        # Add controls for heatmap appearance
        controls_frame = ttk.Frame(heatmap_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        
        # Colormap selection
        ttk.Label(controls_frame, text="Colormap:").pack(side=tk.LEFT, padx=(5, 2))
        self.heatmap_colormap = tk.StringVar(value="viridis")
        colormap_combo = ttk.Combobox(controls_frame, textvariable=self.heatmap_colormap, width=10)
        colormap_combo['values'] = ('viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'RdBu_r', 'jet')
        colormap_combo.pack(side=tk.LEFT, padx=2)
        colormap_combo.bind('<<ComboboxSelected>>', lambda e: self.update_heatmap())
        
        # Normalization method
        ttk.Label(controls_frame, text="Normalization:").pack(side=tk.LEFT, padx=(10, 2))
        self.heatmap_norm = tk.StringVar(value="linear")
        norm_combo = ttk.Combobox(controls_frame, textvariable=self.heatmap_norm, width=10)
        norm_combo['values'] = ('linear', 'log', 'sqrt', 'row', 'column')
        norm_combo.pack(side=tk.LEFT, padx=2)
        norm_combo.bind('<<ComboboxSelected>>', lambda e: self.update_heatmap())
        
        # Contrast adjustment
        ttk.Label(controls_frame, text="Contrast:").pack(side=tk.LEFT, padx=(10, 2))
        self.heatmap_contrast = tk.DoubleVar(value=1.0)
        contrast_slider = ttk.Scale(controls_frame, variable=self.heatmap_contrast, 
                                  from_=0.1, to=2.0, length=100, orient=tk.HORIZONTAL)
        contrast_slider.pack(side=tk.LEFT, padx=2)
        contrast_slider.bind("<ButtonRelease-1>", lambda e: self.update_heatmap())
        
        # Create heatmap figure
        self.heatmap_fig = Figure(figsize=(6, 4))
        self.heatmap_ax = self.heatmap_fig.add_subplot(111)
        self.heatmap_canvas = FigureCanvasTkAgg(self.heatmap_fig, master=heatmap_frame)
        self.heatmap_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add heatmap toolbar
        heatmap_toolbar_frame = ttk.Frame(heatmap_frame)
        heatmap_toolbar_frame.pack(fill=tk.X)
        self.heatmap_toolbar = NavigationToolbar2Tk(self.heatmap_canvas, heatmap_toolbar_frame)
        self.heatmap_toolbar.update()
        
        # Create scatter tab
        scatter_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(scatter_frame, text="Scatter Plot")
        
        # Add visualization method selector
        method_frame = ttk.Frame(scatter_frame)
        method_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(method_frame, text="Visualization Method:").pack(side=tk.LEFT, padx=5)
        self.visualization_method = tk.StringVar(value="PCA")
        method_combo = ttk.Combobox(method_frame, textvariable=self.visualization_method, width=10)
        method_combo['values'] = ('PCA',)
        method_combo.pack(side=tk.LEFT, padx=5)
        method_combo.bind('<<ComboboxSelected>>', lambda e: self.update_scatter_plot())
        
        # Create scatter figure
        self.viz_fig = Figure(figsize=(6, 4))
        self.viz_ax = self.viz_fig.add_subplot(111)
        self.viz_canvas = FigureCanvasTkAgg(self.viz_fig, master=scatter_frame)
        self.viz_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add scatter toolbar
        self.viz_toolbar = NavigationToolbar2Tk(self.viz_canvas, scatter_frame)
        self.viz_toolbar.update()

    def create_visualization_tab(self):
        """Create the visualization tab."""
        visualization_frame = ttk.Frame(self.notebook)
        self.notebook.add(visualization_frame, text="Visualization")
        
        # Create main controls area
        controls_frame = ttk.LabelFrame(visualization_frame, text="Visualization Controls")
        controls_frame.pack(fill="x", padx=5, pady=5)
        
        # Visualization method selector
        method_frame = ttk.Frame(controls_frame)
        method_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(method_frame, text="Visualization Method:").pack(side=tk.LEFT, padx=5)
        
        # Determine available methods
        viz_methods = ['PCA']
        if UMAP_AVAILABLE:
            viz_methods.append('UMAP')
            
        viz_method_combo = ttk.Combobox(method_frame, textvariable=self.visualization_method, width=15)
        viz_method_combo['values'] = viz_methods
        viz_method_combo.pack(side=tk.LEFT, padx=5)
        viz_method_combo.bind('<<ComboboxSelected>>', lambda e: self.update_scatter_plot())
        
        # Add options for visualization customization
        options_frame = ttk.Frame(controls_frame)
        options_frame.pack(fill="x", padx=5, pady=5)
        
        # Color scheme
        ttk.Label(options_frame, text="Color Scheme:").pack(side=tk.LEFT, padx=5)
        self.color_scheme = tk.StringVar(value="viridis")
        color_combo = ttk.Combobox(options_frame, textvariable=self.color_scheme, width=10)
        color_combo['values'] = ('viridis', 'plasma', 'inferno', 'tab10', 'Set1', 'jet')
        color_combo.pack(side=tk.LEFT, padx=5)
        color_combo.bind('<<ComboboxSelected>>', lambda e: self.update_scatter_plot())
        
        # Add update button
        update_btn = ttk.Button(controls_frame, text="Update Visualization", command=self.update_scatter_plot)
        update_btn.pack(pady=10)
        
        # Create visualization display area
        plot_frame = ttk.LabelFrame(visualization_frame, text="Visualization Plot")
        plot_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create figure for visualization if it doesn't exist yet
        if not hasattr(self, 'viz_fig'):
            self.viz_fig = Figure(figsize=(6, 4))
            self.viz_ax = self.viz_fig.add_subplot(111)
            
        # Create canvas for the figure
        visualization_canvas = FigureCanvasTkAgg(self.viz_fig, master=plot_frame)
        visualization_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X)
        visualization_toolbar = NavigationToolbar2Tk(visualization_canvas, toolbar_frame)
        visualization_toolbar.update()
        
        # Add export button
        export_frame = ttk.Frame(visualization_frame)
        export_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(export_frame, text="Export Visualization", 
                  command=lambda: self.export_visualization()).pack(side=tk.RIGHT, padx=5)

    def create_analysis_tab(self):
        """Create the analysis results tab."""
        analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(analysis_tab, text="Analysis Results")
        
        # Results frame
        results_frame = ttk.LabelFrame(analysis_tab, text="Cluster Analysis Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a frame for text widget with scrollbars
        text_frame = ttk.Frame(results_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbars
        y_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        x_scrollbar = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add text widget for results with scrollbars
        self.results_text = tk.Text(
            text_frame, 
            wrap=tk.NONE,  # Changed to NONE to allow horizontal scrolling
            height=20,
            yscrollcommand=y_scrollbar.set,
            xscrollcommand=x_scrollbar.set
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure scrollbars
        y_scrollbar.config(command=self.results_text.yview)
        x_scrollbar.config(command=self.results_text.xview)
        
        # Add export button frame
        export_frame = ttk.Frame(results_frame)
        export_frame.pack(fill=tk.X, pady=5)
        
        # Add export button
        export_btn = ttk.Button(export_frame, text="Export Results to CSV", 
                              command=self.export_analysis_results)
        export_btn.pack(side=tk.RIGHT, padx=5)
        
        # Add a note about what will appear here
        self.results_text.insert(tk.END, "Cluster analysis results will appear here after running clustering.\n\n")
        self.results_text.insert(tk.END, "To populate this tab:\n")
        self.results_text.insert(tk.END, "1. Import data from the Import tab\n")
        self.results_text.insert(tk.END, "2. Run clustering from the Clustering tab\n")

    def export_analysis_results(self):
        """Export cluster analysis results to a CSV file."""
        if self.cluster_data is None or 'labels' not in self.cluster_data or self.cluster_data['labels'] is None:
            messagebox.showinfo("Info", "No clustering results available to export.")
            return
            
        try:
            # Get file path from user
            file_path = filedialog.asksaveasfilename(
                title="Export Analysis Results",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if not file_path:
                return  # User cancelled
                
            # Prepare data for export
            export_data = []
            
            # Get unique clusters
            unique_clusters = np.unique(self.cluster_data['labels'])
            
            # For each cluster
            for cluster in unique_clusters:
                cluster_points = np.where(self.cluster_data['labels'] == cluster)[0]
                
                # For each point in the cluster
                for point_idx in cluster_points:
                    row_data = {
                        'Cluster': cluster,
                        'NAME': 'Unknown',
                        'Chemical Formula': 'Unknown'
                    }
                    
                    # Get sample metadata if available
                    if 'sample_metadata' in self.cluster_data and len(self.cluster_data['sample_metadata']) > point_idx:
                        metadata = self.cluster_data['sample_metadata'][point_idx]
                        if metadata:
                            # Create case-insensitive lookup
                            metadata_lower = {k.lower(): k for k in metadata.keys()}
                            
                            # Get NAME (try different variations)
                            if 'NAME' in metadata and metadata['NAME']:
                                row_data['NAME'] = metadata['NAME']
                            elif 'name' in metadata_lower:
                                actual_key = metadata_lower['name']
                                row_data['NAME'] = metadata[actual_key]
                            elif 'MINERAL NAME' in metadata and metadata['MINERAL NAME']:
                                row_data['NAME'] = metadata['MINERAL NAME']
                            elif 'mineral name' in metadata_lower:
                                actual_key = metadata_lower['mineral name']
                                row_data['NAME'] = metadata[actual_key]
                            
                            # Get Chemical Formula (try different variations)
                            if 'CHEMICAL FORMULA' in metadata and metadata['CHEMICAL FORMULA']:
                                row_data['Chemical Formula'] = metadata['CHEMICAL FORMULA']
                            elif 'chemical formula' in metadata_lower:
                                actual_key = metadata_lower['chemical formula']
                                row_data['Chemical Formula'] = metadata[actual_key]
                            elif 'FORMULA' in metadata and metadata['FORMULA']:
                                row_data['Chemical Formula'] = metadata['FORMULA']
                            elif 'formula' in metadata_lower:
                                actual_key = metadata_lower['formula']
                                row_data['Chemical Formula'] = metadata[actual_key]
                    
                    export_data.append(row_data)
            
            # Convert to DataFrame and export
            df = pd.DataFrame(export_data)
            df.to_csv(file_path, index=False)
            
            messagebox.showinfo("Success", f"Analysis results exported to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {str(e)}")
            print(f"DEBUG: Export error: {str(e)}")

    def create_refinement_tab(self):
        """Create the refinement tab for user-guided cluster refinement."""
        refinement_tab = ttk.Frame(self.notebook)
        self.notebook.add(refinement_tab, text="Cluster Refinement")
        
        # Control frame
        control_frame = ttk.LabelFrame(refinement_tab, text="Refinement Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Refinement instructions
        ttk.Label(control_frame, text="Instructions:").pack(anchor=tk.W)
        ttk.Label(control_frame, text="1. Select points in the scatter plot to merge clusters\n"
                 "2. Use the dendrogram to adjust cluster boundaries\n"
                 "3. Select a cluster to split into subclusters\n"
                 "4. Preview changes before applying\n"
                 "5. Use undo to revert changes\n"
                 "6. Apply changes to update the analysis").pack(anchor=tk.W)
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Merge Selected Clusters", 
                  command=self.merge_selected_clusters).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Split Selected Cluster", 
                  command=self.preview_split).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Reset Selection", 
                  command=self.reset_selection).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Undo", 
                  command=self.undo_last_action).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Apply Changes", 
                  command=self.apply_refinement).pack(side=tk.LEFT, padx=2)
        
        # Split controls frame
        split_frame = ttk.LabelFrame(refinement_tab, text="Split Controls", padding=10)
        split_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Number of subclusters
        ttk.Label(split_frame, text="Number of Subclusters:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(split_frame, from_=2, to=5, textvariable=self.n_subclusters,
                   width=5).pack(side=tk.LEFT, padx=5)
        
        # Split method
        ttk.Label(split_frame, text="Split Method:").pack(side=tk.LEFT, padx=5)
        ttk.Combobox(split_frame, textvariable=self.split_method,
                    values=['kmeans', 'hierarchical', 'spectral'],
                    width=10).pack(side=tk.LEFT, padx=5)
        
        # Preview controls
        preview_frame = ttk.LabelFrame(refinement_tab, text="Preview Controls", padding=10)
        preview_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(preview_frame, text="Apply Preview", 
                  command=self.apply_preview).pack(side=tk.LEFT, padx=2)
        ttk.Button(preview_frame, text="Cancel Preview", 
                  command=self.cancel_preview).pack(side=tk.LEFT, padx=2)
        
        # Results frame
        results_frame = ttk.LabelFrame(refinement_tab, text="Refinement Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add text widget for results
        self.refinement_text = tk.Text(results_frame, wrap=tk.WORD, height=10)
        self.refinement_text.pack(fill=tk.BOTH, expand=True)
        
    def create_dendrogram_frame(self, parent):
        """Create frame for dendrogram plot."""
        dendro_frame = ttk.LabelFrame(parent, text="Hierarchical Clustering Dendrogram", padding=5)
        dendro_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create figure and canvas
        self.dendro_fig = plt.Figure(figsize=(6, 4))
        self.dendro_canvas = FigureCanvasTkAgg(self.dendro_fig, master=dendro_frame)
        self.dendro_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(dendro_frame)
        toolbar_frame.pack(fill=tk.X)
        self.dendro_toolbar = NavigationToolbar2Tk(self.dendro_canvas, toolbar_frame)
        self.dendro_toolbar.update()
        
    def run_clustering(self):
        """Run the clustering analysis."""
        if self.cluster_data is None or self.cluster_data['features'] is None:
            messagebox.showinfo("Info", "No data available for clustering")
            return
            
        try:
            # Update progress
            self.clustering_progress['value'] = 0
            self.clustering_status.config(text="Running clustering...")
            self.window.update()
            
            # Get parameters and ensure n_clusters is an integer
            try:
                n_clusters = int(self.n_clusters.get())
                if n_clusters < 2:
                    raise ValueError("Number of clusters must be at least 2")
            except ValueError as e:
                messagebox.showerror("Error", "Please enter a valid number of clusters (minimum 2)")
                self.clustering_status.config(text="Invalid number of clusters")
                self.clustering_progress['value'] = 0
                return
                
            linkage_method = self.linkage_method_combo.get()
            distance_metric = self.distance_metric_combo.get()
            
            print("\nDEBUG: Starting clustering")
            print(f"DEBUG: Using {n_clusters} clusters")
            print(f"DEBUG: Using {linkage_method} linkage")
            print(f"DEBUG: Using {distance_metric} distance")
            
            # Run clustering
            self.cluster_data['labels'] = self.perform_hierarchical_clustering(
                self.cluster_data['features'],
                n_clusters=n_clusters,
                linkage_method=linkage_method,
                distance_metric=distance_metric
            )
            
            # Update progress
            self.clustering_progress['value'] = 100
            self.clustering_status.config(text="Clustering complete")
            
            # Update visualizations
            self.update_visualizations()
            
            # Update analysis results tab
            self.update_analysis_results()
            
        except Exception as e:
            print(f"DEBUG: Clustering failed with error: {str(e)}")
            messagebox.showerror("Error", f"Clustering failed: {str(e)}")
            self.clustering_status.config(text="Clustering failed")
            self.clustering_progress['value'] = 0
        
    def perform_hierarchical_clustering(self, features, n_clusters, linkage_method, distance_metric):
        """Perform hierarchical clustering on the features."""
        print("\nDEBUG: Starting hierarchical clustering")
        print(f"DEBUG: Features shape: {features.shape}")
        print(f"DEBUG: Number of clusters: {n_clusters}")
        print(f"DEBUG: Linkage method: {linkage_method}")
        print(f"DEBUG: Distance metric: {distance_metric}")
        
        try:
            # Ensure features are properly scaled
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            
            # Check for NaN or infinite values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                print("DEBUG: Found NaN or infinite values in features")
                features = np.nan_to_num(features)
            
            # Standardize features if not already scaled
            if not hasattr(self, '_scaler') or self._scaler is None:
                self._scaler = StandardScaler()
                features_scaled = self._scaler.fit_transform(features)
            else:
                features_scaled = self._scaler.transform(features)
            
            print(f"DEBUG: Scaled features shape: {features_scaled.shape}")
            
            # Compute distance matrix
            if distance_metric == 'cosine':
                # For cosine distance, normalize the features first
                features_normalized = features_scaled / np.linalg.norm(features_scaled, axis=1)[:, np.newaxis]
                distance_matrix = pdist(features_normalized, metric='cosine')
            else:
                distance_matrix = pdist(features_scaled, metric=distance_metric)
            
            print(f"DEBUG: Distance matrix shape: {distance_matrix.shape}")
            
            # Compute linkage matrix
            linkage_matrix = linkage(distance_matrix, method=linkage_method)
            print(f"DEBUG: Linkage matrix shape: {linkage_matrix.shape}")
            
            # Get cluster assignments
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            print(f"DEBUG: Cluster labels shape: {cluster_labels.shape}")
            print(f"DEBUG: Number of unique clusters: {len(np.unique(cluster_labels))}")
            
            # Store data for visualization
            self.cluster_data.update({
                'features': features,
                'features_scaled': features_scaled,
                'linkage_matrix': linkage_matrix,
                'distance_matrix': distance_matrix,
                'labels': cluster_labels
            })
            
            return cluster_labels
            
        except Exception as e:
            print(f"DEBUG: Clustering error: {str(e)}")
            raise

    def extract_vibrational_features(self, intensities, wavenumbers):
        """Extract features from spectral data.
        
        Parameters:
        -----------
        intensities : ndarray
            2D array of spectral intensities (n_spectra, n_points)
        wavenumbers : ndarray
            1D array of common wavenumber scale or 2D array matching intensities
            
        Returns:
        --------
        ndarray
            2D array of features
        """
        try:
            print("\nDEBUG: Extracting features")
            print(f"DEBUG: Intensities shape: {intensities.shape}")
            
            # Convert arrays to numpy if they aren't already
            intensities = np.asarray(intensities)
            wavenumbers = np.asarray(wavenumbers)
            
            # Ensure wavenumbers is proper shape
            if wavenumbers.ndim == 1:
                print(f"DEBUG: Wavenumbers shape: {wavenumbers.shape}")
            elif wavenumbers.ndim == 2:
                print(f"DEBUG: Wavenumbers shape: {wavenumbers.shape}")
                # Use first row if 2D (should all be the same after interpolation)
                wavenumbers = wavenumbers[0]
            
            # Ensure intensities is 2D
            if intensities.ndim == 1:
                intensities = intensities.reshape(1, -1)
                
            # Get dimensions
            n_spectra = intensities.shape[0]
            
            # Initialize feature array using the intensity values directly as features
            features = intensities.copy()
            
            print(f"DEBUG: Extracted features shape: {features.shape}")
            
            return features
        except Exception as e:
            print(f"DEBUG: Feature extraction error: {str(e)}")
            raise
        
    def update_visualizations(self):
        """Update all visualization plots."""
        try:
            print("\nDEBUG: Updating visualizations")
            print(f"DEBUG: Cluster data keys: {self.cluster_data.keys()}")
            
            if 'linkage_matrix' not in self.cluster_data:
                print("DEBUG: No linkage matrix found in cluster data")
                return
                
            self.update_dendrogram()
            self.update_heatmap()
            self.update_scatter_plot()
            
        except Exception as e:
            print(f"DEBUG: Visualization update error: {str(e)}")
            raise

    def update_dendrogram(self):
        """Update the dendrogram plot."""
        try:
            print("\nDEBUG: Updating dendrogram")
            print(f"DEBUG: Linkage matrix shape: {self.cluster_data['linkage_matrix'].shape}")
            
            # Clear the current plot
            self.dendrogram_ax.clear()
            
            # Use filenames as labels if available
            labels = self.cluster_data.get('sample_labels', None)
            
            # Plot the dendrogram
            dendrogram(self.cluster_data['linkage_matrix'],
                      ax=self.dendrogram_ax,
                      truncate_mode='lastp',
                      p=30,
                      show_leaf_counts=True,
                      leaf_rotation=90,
                      labels=labels)
            
            # Update the plot
            self.dendrogram_ax.set_title('Hierarchical Clustering Dendrogram')
            self.dendro_canvas.draw()
            
        except Exception as e:
            print(f"DEBUG: Dendrogram update error: {str(e)}")
            raise

    def update_heatmap(self):
        """Update the heatmap visualization."""
        try:
            print("\nDEBUG: Updating heatmap")
            
            if self.cluster_data is None or self.cluster_data['features'] is None:
                print("DEBUG: No data available for heatmap")
                return
                
            # Clear the figure and recreate the axis
            self.heatmap_fig.clf()
            self.heatmap_ax = self.heatmap_fig.add_subplot(111)
            
            # Get the data
            features = self.cluster_data['features']
            labels = self.cluster_data['labels']
            
            # Sort data by cluster labels
            sort_idx = np.argsort(labels)
            sorted_features = features[sort_idx]
            sorted_labels = labels[sort_idx]
            
            # Apply normalization based on selected method
            norm_method = self.heatmap_norm.get()
            display_data = sorted_features.copy()
            
            if norm_method == 'log':
                # Log normalization (add small value to avoid log(0))
                display_data = np.log1p(np.abs(sorted_features))
            elif norm_method == 'sqrt':
                # Square root normalization (preserve sign)
                display_data = np.sign(sorted_features) * np.sqrt(np.abs(sorted_features))
            elif norm_method == 'row':
                # Row-wise normalization (per sample)
                row_max = np.max(np.abs(sorted_features), axis=1, keepdims=True)
                row_max[row_max == 0] = 1.0  # Avoid division by zero
                display_data = sorted_features / row_max
            elif norm_method == 'column':
                # Column-wise normalization (per feature)
                col_max = np.max(np.abs(sorted_features), axis=0, keepdims=True)
                col_max[col_max == 0] = 1.0  # Avoid division by zero
                display_data = sorted_features / col_max
                
            # Apply contrast adjustment
            contrast = self.heatmap_contrast.get()
            if contrast != 1.0:
                # Enhance contrast by applying power function
                display_data = np.sign(display_data) * np.power(np.abs(display_data), contrast)
            
            # Get selected colormap
            colormap = self.heatmap_colormap.get()
            
            # Create heatmap with selected colormap
            im = self.heatmap_ax.imshow(display_data, aspect='auto', cmap=colormap)
            
            # Add colorbar, anchored to the correct axis
            self.heatmap_fig.colorbar(im, ax=self.heatmap_ax)
            
            # Update the plot
            norm_info = f" ({norm_method} norm, contrast={contrast:.1f})" if norm_method != 'linear' or contrast != 1.0 else ""
            self.heatmap_ax.set_title(f'Feature Heatmap{norm_info}')
            self.heatmap_ax.set_xlabel('Features')
            self.heatmap_ax.set_ylabel('Samples')
            
            # Add cluster boundaries
            unique_labels = np.unique(sorted_labels)
            for label in unique_labels[:-1]:
                boundary = np.where(sorted_labels == label)[0][-1] + 0.5
                self.heatmap_ax.axhline(y=boundary, color='white', linestyle='--', alpha=0.5)
            
            self.heatmap_fig.tight_layout()
            self.heatmap_canvas.draw()
            
        except Exception as e:
            print(f"DEBUG: Heatmap update error: {str(e)}")
            raise

    def update_scatter_plot(self):
        """Update the scatter plot visualization."""
        try:
            print("\nDEBUG: Updating scatter plot")
            
            if self.cluster_data is None or self.cluster_data['features'] is None:
                print("DEBUG: No data available for scatter plot")
                return
                
            # Clear the figure and recreate the axis
            self.viz_fig.clf()
            self.viz_ax = self.viz_fig.add_subplot(111)
            
            # Get visualization method
            method = self.visualization_method.get()
            
            # Safety check - get number of data points
            n_samples = self.cluster_data['features'].shape[0]
            print(f"DEBUG: Number of samples for dimensionality reduction: {n_samples}")
            
            # Prepare data - ensure it's properly scaled
            if self.cluster_data['features_scaled'] is None:
                print("DEBUG: Scaling features for visualization")
                self.cluster_data['features_scaled'] = StandardScaler().fit_transform(self.cluster_data['features'])
            
            features_to_use = self.cluster_data['features_scaled']
            
            # Perform dimensionality reduction with proper error handling
            try:
                if method == 'PCA':
                    print("DEBUG: Using PCA for dimensionality reduction")
                    reducer = PCA(n_components=2)
                    features_2d = reducer.fit_transform(features_to_use)
                elif method == 'UMAP' and UMAP_AVAILABLE:
                    print("DEBUG: Using UMAP for dimensionality reduction")
                    # Calculate appropriate n_neighbors (between 5 and 50)
                    n_neighbors = min(30, max(5, n_samples // 10))
                    print(f"DEBUG: Using n_neighbors={n_neighbors} for UMAP")
                    
                    # Configure UMAP
                    reducer = umap.UMAP(
                        n_components=2,
                        n_neighbors=n_neighbors,
                        min_dist=0.1,
                        metric='euclidean',
                        random_state=42
                    )
                    
                    # Show user that this may take some time
                    self.status_bar.config(text="Calculating UMAP visualization (this may take a moment)...")
                    self.window.update()
                    
                    features_2d = reducer.fit_transform(features_to_use)
                else:  # Unsupported method or UMAP not available
                    if method == 'UMAP' and not UMAP_AVAILABLE:
                        print("DEBUG: UMAP requested but not available")
                        messagebox.showinfo("Info", 
                            "UMAP is not available. Install with 'pip install umap-learn'. Using PCA instead.")
                    else:
                        print(f"DEBUG: Unsupported visualization method: {method}")
                        messagebox.showinfo("Info", f"Visualization method '{method}' not supported. Using PCA instead.")
                    
                    reducer = PCA(n_components=2)
                    features_2d = reducer.fit_transform(features_to_use)
            except MemoryError:
                print("DEBUG: Memory error during dimensionality reduction")
                messagebox.showerror("Memory Error", 
                    f"Insufficient memory for {method} with {n_samples} samples. Try PCA instead.")
                # Fall back to PCA which is more memory efficient
                reducer = PCA(n_components=2)
                features_2d = reducer.fit_transform(features_to_use)
            except Exception as e:
                print(f"DEBUG: Error during dimensionality reduction: {str(e)}")
                messagebox.showerror("Error", 
                    f"Error calculating {method} visualization: {str(e)}\nFalling back to PCA.")
                # Fall back to PCA
                reducer = PCA(n_components=2)
                features_2d = reducer.fit_transform(features_to_use)
            
            # Reset status bar
            self.status_bar.config(text="Visualization complete")
            
            # Get colormap from color_scheme if available
            colormap = 'viridis'
            if hasattr(self, 'color_scheme'):
                colormap = self.color_scheme.get()
            
            # Plot the data with picking enabled
            scatter = self.viz_ax.scatter(
                features_2d[:, 0], 
                features_2d[:, 1],
                c=self.cluster_data['labels'],
                cmap=colormap,
                picker=True,  # Enable picking
                alpha=0.8     # Slight transparency for better visibility
            )
            
            # Add colorbar, anchored to the correct axis
            self.viz_fig.colorbar(scatter, ax=self.viz_ax)
            
            # Add grid for better readability
            self.viz_ax.grid(True, linestyle='--', alpha=0.5)
            
            # Update the plot
            self.viz_ax.set_title(f'{method} Visualization')
            self.viz_ax.set_xlabel('Component 1')
            self.viz_ax.set_ylabel('Component 2')
            self.viz_fig.tight_layout()
            
            # Add tooltips showing mineral names using mplcursors
            try:
                import mplcursors
                
                # Check if sample_labels are available
                if 'sample_labels' in self.cluster_data and self.cluster_data['sample_labels'] is not None:
                    # Create tooltips using mplcursors
                    cursor = mplcursors.cursor(scatter, hover=True)
                    
                    @cursor.connect("add")
                    def on_add(sel):
                        try:
                            # Get point index and sample label
                            point_idx = sel.index
                            
                            # Also handle if point_idx isn't directly usable (in case it's a tuple)
                            if isinstance(point_idx, tuple):
                                point_idx = point_idx[0]
                            
                            # Add check for sample_labels before accessing
                            if 'sample_labels' in self.cluster_data and self.cluster_data['sample_labels'] is not None:
                                sample_name = self.cluster_data['sample_labels'][point_idx]
                            else:
                                sample_name = f"Point {point_idx}"
                            
                            # Check if we have metadata with MINERAL NAME or NAME/ID to use instead
                            if 'sample_metadata' in self.cluster_data and len(self.cluster_data['sample_metadata']) > point_idx:
                                metadata = self.cluster_data['sample_metadata'][point_idx]
                                if metadata:
                                    # Create case-insensitive lookup
                                    metadata_lower = {k.lower(): k for k in metadata.keys()}
                                    
                                    # Try with 'NAME' first (seems to be the most common based on debug output)
                                    if 'NAME' in metadata and metadata['NAME']:
                                        sample_name = metadata['NAME']
                                    # Then try different case variations for MINERAL NAME
                                    elif 'MINERAL NAME' in metadata and metadata['MINERAL NAME']:
                                        sample_name = metadata['MINERAL NAME']
                                    elif 'mineral name' in metadata_lower:
                                        actual_key = metadata_lower['mineral name']
                                        sample_name = metadata[actual_key]
                                    
                                    # Try different case variations for NAME/ID
                                    elif 'NAME/ID' in metadata and metadata['NAME/ID']:
                                        sample_name = metadata['NAME/ID']
                                    elif 'name/id' in metadata_lower:
                                        actual_key = metadata_lower['name/id']
                                        sample_name = metadata[actual_key]
                            
                            # Add a check to ensure cluster_data and labels are available
                            if 'labels' in self.cluster_data and self.cluster_data['labels'] is not None:
                                cluster_id = self.cluster_data['labels'][point_idx]
                            else:
                                cluster_id = "N/A"  # Or some default value
                            
                            # Check if we have enhanced metadata
                            tooltip_text = f"Mineral: {sample_name}\nCluster: {cluster_id}"
                            
                            # Add extended metadata if available
                            if 'sample_metadata' in self.cluster_data and len(self.cluster_data['sample_metadata']) > point_idx:
                                metadata = self.cluster_data['sample_metadata'][point_idx]
                                if metadata:
                                    # Create case-insensitive lookup
                                    metadata_lower = {k.lower(): k for k in metadata.keys()}
                                    
                                    # Add key metadata items if available - check case variations
                                    if 'CHEMICAL FORMULA' in metadata and metadata['CHEMICAL FORMULA']:
                                        tooltip_text += f"\nFormula: {metadata['CHEMICAL FORMULA']}"
                                    elif 'chemical formula' in metadata_lower:
                                        actual_key = metadata_lower['chemical formula']
                                        tooltip_text += f"\nFormula: {metadata[actual_key]}"
                                        
                                    if 'HEY CLASSIFICATION' in metadata and metadata['HEY CLASSIFICATION']:
                                        tooltip_text += f"\nClass: {metadata['HEY CLASSIFICATION']}"
                                    elif 'hey classification' in metadata_lower:
                                        actual_key = metadata_lower['hey classification']
                                        tooltip_text += f"\nClass: {metadata[actual_key]}"
                                        
                                    if 'LOCALITY' in metadata and metadata['LOCALITY']:
                                        tooltip_text += f"\nLocality: {metadata['LOCALITY']}"
                                    elif 'locality' in metadata_lower:
                                        actual_key = metadata_lower['locality']
                                        tooltip_text += f"\nLocality: {metadata[actual_key]}"
                                        
                                    if 'CHEMISTRY' in metadata and metadata['CHEMISTRY']:
                                        chemistry = metadata['CHEMISTRY']
                                        if len(chemistry) > 30:
                                            chemistry = chemistry[:30] + "..."
                                        tooltip_text += f"\nChem: {chemistry}"
                                    elif 'chemistry' in metadata_lower:
                                        actual_key = metadata_lower['chemistry']
                                        chemistry = metadata[actual_key]
                                        if len(chemistry) > 30:
                                            chemistry = chemistry[:30] + "..."
                                        tooltip_text += f"\nChem: {chemistry}"
                            
                            # Format tooltip text
                            sel.annotation.set_text(tooltip_text)
                            
                            # Customize tooltip appearance
                            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)
                            sel.annotation.set_visible(True)
                            
                        except Exception as e:
                            print(f"ERROR in tooltip display: {str(e)}")
                            # Set a basic tooltip with error information
                            sel.annotation.set_text(f"Error displaying data: {str(e)[:50]}")
                    
                    @cursor.connect("remove")
                    def on_remove(sel):
                        if sel.annotation:
                            sel.annotation.set_visible(False)
                            self.viz_canvas.draw()
                
                else:
                    print("DEBUG: No sample labels available for tooltips")
            except ImportError:
                print("WARNING: mplcursors not available for tooltips. Install with 'pip install mplcursors'")
            except Exception as e:
                print(f"ERROR setting up tooltips: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Draw the canvas
            self.viz_canvas.draw()
            
            # Reconnect pick event handler for interactive selection
            self.viz_canvas.mpl_connect('pick_event', self.on_pick)
            
        except Exception as e:
            print(f"DEBUG: Scatter plot update error: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Visualization Error", f"Error updating scatter plot: {str(e)}")

    def export_visualization(self):
        """Export the current visualization to a file."""
        try:
            if not hasattr(self, 'viz_fig') or self.viz_fig is None:
                messagebox.showinfo("Info", "No visualization available to export.")
                return
            
            # Ask user for file type and location
            file_path = filedialog.asksaveasfilename(
                title="Save Visualization",
                defaultextension=".png",
                filetypes=[
                    ("PNG image", "*.png"),
                    ("PDF document", "*.pdf"),
                    ("SVG image", "*.svg"),
                    ("JPEG image", "*.jpg"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                return  # User cancelled
            
            # Save the figure
            self.viz_fig.savefig(
                file_path, 
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            
            messagebox.showinfo("Success", f"Visualization saved to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export visualization: {str(e)}")
            print(f"DEBUG: Export visualization error: {str(e)}")

    def start_refinement(self):
        """Start the refinement mode."""
        if self.cluster_data is None:
            messagebox.showerror("Error", "Please run clustering first.")
            return
            
        self.refinement_mode = True
        self.selected_points.clear()
        self.update_scatter_plot()
        messagebox.showinfo("Refinement Mode", 
                          "Refinement mode activated. Select points in the scatter plot to merge clusters.")
        
    def save_state(self):
        """Save current state for undo."""
        if self.cluster_data['labels'] is not None:
            state = {
                'cluster_labels': self.cluster_data['labels'].copy(),
                'selected_points': self.selected_points.copy()
            }
            self.undo_stack.append(state)
            if len(self.undo_stack) > self.max_undo_steps:
                self.undo_stack.pop(0)
            print("Refinement mode activated. Select points in the scatter plot to merge clusters.")

    def undo_last_action(self):
        """Undo the last action."""
        if not self.undo_stack:
            messagebox.showinfo("Info", "No actions to undo.")
            return
            
        state = self.undo_stack.pop()
        self.cluster_data['labels'] = state['cluster_labels']
        self.selected_points = state['selected_points']
        
        # Update visualizations
        self.update_visualizations()
        self.update_refinement_results()
        
        messagebox.showinfo("Success", "Last action undone.")

    def preview_split(self):
        """Preview the split of selected clusters."""
        if not self.selected_points:
            messagebox.showinfo("Info", "No points selected for splitting.")
            return
        # Use cluster labels from cluster_data
        selected_clusters = set(np.array(self.cluster_data['labels'])[list(self.selected_points)])
        if len(selected_clusters) != 1:
            messagebox.showinfo("Info", "Please select points from only one cluster to split.")
            return
            
        # Save current state
        self.save_state()
        
        # Perform split
        self.split_selected_cluster(preview=True)
        
        # Show preview message
        messagebox.showinfo("Preview", 
                          "Split preview applied. Use 'Apply Preview' to keep changes or 'Cancel Preview' to revert.")

    def apply_preview(self):
        """Apply the previewed changes."""
        if not self.undo_stack:
            messagebox.showinfo("Info", "No preview to apply.")
            return
            
        # Keep the current state
        self.undo_stack.pop()  # Remove the saved state since we're keeping the changes
        messagebox.showinfo("Success", "Preview changes applied.")

    def cancel_preview(self):
        """Cancel the previewed changes."""
        if not self.undo_stack:
            messagebox.showinfo("Info", "No preview to cancel.")
            return
            
        # Restore the previous state
        state = self.undo_stack.pop()
        self.cluster_data['labels'] = state['cluster_labels']
        self.selected_points = state['selected_points']
        
        # Update visualizations
        self.update_visualizations()
        self.update_refinement_results()
        
        messagebox.showinfo("Success", "Preview cancelled.")

    def split_selected_cluster(self, preview=False):
        """Split the selected cluster into subclusters."""
        if not self.selected_points:
            messagebox.showinfo("Info", "Please select points from a single cluster to split.")
            return
            
        # Get the cluster label of selected points
        selected_clusters = set(np.array(self.cluster_data['labels'])[list(self.selected_points)])
        if len(selected_clusters) != 1:
            messagebox.showinfo("Info", "Please select points from only one cluster to split.")
            return
            
        cluster_to_split = selected_clusters.pop()
        n_subclusters = self.n_subclusters.get()
        split_method = self.split_method.get()
        
        # Get the points in the selected cluster
        cluster_points = np.where(self.cluster_data['labels'] == cluster_to_split)[0]
        cluster_features = self.cluster_data['features_scaled'][cluster_points]
        
        # Perform splitting based on selected method
        if split_method == 'kmeans':
            splitter = KMeans(n_clusters=n_subclusters, random_state=42)
            subcluster_labels = splitter.fit_predict(cluster_features)
        elif split_method == 'hierarchical':
            from scipy.cluster.hierarchy import fcluster
            linkage_matrix = linkage(cluster_features, method='ward')
            subcluster_labels = fcluster(linkage_matrix, n_subclusters, criterion='maxclust')
        else:  # spectral
            splitter = SpectralClustering(n_clusters=n_subclusters, random_state=42)
            subcluster_labels = splitter.fit_predict(cluster_features)
        
        # Update cluster labels
        max_label = np.max(self.cluster_data['labels'])
        for i, point_idx in enumerate(cluster_points):
            self.cluster_data['labels'][point_idx] = max_label + subcluster_labels[i] + 1
        
        # Update visualizations
        self.update_visualizations()
        self.update_refinement_results()
        
        if not preview:
            messagebox.showinfo("Success", 
                              f"Cluster {cluster_to_split} split into {n_subclusters} subclusters.")

    def merge_selected_clusters(self):
        """Merge selected clusters based on selected points."""
        if not self.selected_points:
            messagebox.showinfo("Info", "No points selected for merging.")
            return
        # Use cluster labels from cluster_data
        selected_clusters = set(np.array(self.cluster_data['labels'])[list(self.selected_points)])
        if len(selected_clusters) < 2:
            messagebox.showinfo("Info", "Please select points from different clusters to merge.")
            return
            
        # Merge clusters
        new_label = min(selected_clusters)
        for old_label in selected_clusters:
            if old_label != new_label:
                self.cluster_data['labels'][self.cluster_data['labels'] == old_label] = new_label
                
        # Update visualizations
        self.update_visualizations()
        self.update_refinement_results()

    def reset_selection(self):
        """Reset the point selection."""
        self.selected_points.clear()
        self.update_scatter_plot()
        
    def apply_refinement(self):
        """Apply the refinement changes."""
        if not self.refinement_mode:
            return
            
        # Update analysis results
        self.update_analysis_results()
        
        # Reset refinement mode
        self.refinement_mode = False
        self.selected_points.clear()
        
        messagebox.showinfo("Success", "Refinement changes applied successfully.")
        
    def update_refinement_results(self):
        """Update the refinement results text."""
        if self.cluster_data is None:
            return
            
        # Calculate cluster statistics
        unique_clusters = np.unique(self.cluster_data['labels'])
        cluster_stats = []
        
        for cluster in unique_clusters:
            cluster_points = np.where(self.cluster_data['labels'] == cluster)[0]
            cluster_stats.append({
                'Cluster': cluster,
                'Size': len(cluster_points),
                'Percentage': len(cluster_points) / len(self.cluster_data['labels']) * 100
            })
            
        # Create results text
        results_text = "Cluster Refinement Results:\n\n"
        for stat in cluster_stats:
            results_text += f"Cluster {stat['Cluster']}:\n"
            results_text += f"  Size: {stat['Size']} points\n"
            results_text += f"  Percentage: {stat['Percentage']:.1f}%\n\n"
            
        self.refinement_text.delete(1.0, tk.END)
        self.refinement_text.insert(tk.END, results_text)
        
    def update_analysis_results(self):
        """Update the analysis results with detailed statistics."""
        print("\nDEBUG: Updating analysis results tab")
        
        if self.cluster_data is None or 'labels' not in self.cluster_data or self.cluster_data['labels'] is None:
            print("DEBUG: No cluster data available for analysis tab")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "No clustering results available.\n\nPlease import data and run clustering first.")
            return
            
        try:
            # Calculate cluster statistics
            unique_clusters = np.unique(self.cluster_data['labels'])
            cluster_stats = []
            
            for cluster in unique_clusters:
                cluster_points = np.where(self.cluster_data['labels'] == cluster)[0]
                cluster_features = self.cluster_data['features'][cluster_points]
                
                stats = {
                    'Cluster': cluster,
                    'Size': len(cluster_points),
                    'Percentage': len(cluster_points) / len(self.cluster_data['labels']) * 100,
                    'Mean Features': np.mean(cluster_features, axis=0),
                    'Std Features': np.std(cluster_features, axis=0)
                }
                
                # Add filenames/sample names if available
                if 'sample_labels' in self.cluster_data and self.cluster_data['sample_labels'] is not None:
                    sample_names = [self.cluster_data['sample_labels'][i] for i in cluster_points]
                    stats['Samples'] = sample_names
                
                cluster_stats.append(stats)
                
            # Create results text
            results_text = "Cluster Analysis Results:\n\n"
            
            # Overall statistics
            results_text += "Overall Statistics:\n"
            results_text += f"Number of Clusters: {len(unique_clusters)}\n"
            results_text += f"Total Points: {len(self.cluster_data['labels'])}\n\n"
            
            # Per-cluster statistics
            results_text += "Cluster Statistics:\n"
            for stat in cluster_stats:
                results_text += f"\nCluster {stat['Cluster']}:\n"
                results_text += f"  Size: {stat['Size']} points\n"
                results_text += f"  Percentage: {stat['Percentage']:.1f}%\n"
                results_text += f"  Mean Features: {stat['Mean Features'].mean():.3f}\n"
                results_text += f"  Feature Std: {stat['Std Features'].mean():.3f}\n"
                
                # List samples in this cluster if available
                if 'Samples' in stat:
                    results_text += "  Samples in this cluster:\n"
                    # List up to 15 samples, then summarize if there are more
                    if len(stat['Samples']) <= 15:
                        for sample in stat['Samples']:
                            results_text += f"    - {sample}\n"
                    else:
                        for sample in stat['Samples'][:15]:
                            results_text += f"    - {sample}\n"
                        results_text += f"    ... and {len(stat['Samples']) - 15} more samples\n"
            
            # Clear previous content and insert new results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, results_text)
            
            # Update status bar to inform user
            self.status_bar.config(text=f"Analysis Results updated with {len(unique_clusters)} clusters")
            
            print("DEBUG: Successfully updated analysis results")
            
        except Exception as e:
            print(f"DEBUG: Error updating analysis results: {str(e)}")
            import traceback
            traceback.print_exc()
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Error updating analysis results: {str(e)}")

    def import_from_main_app(self):
        """Import data from the main application."""
        try:
            # Get current spectrum data from main app
            if hasattr(self.raman_app, 'current_spectrum') and self.raman_app.current_spectrum is not None:
                wavenumbers = self.raman_app.current_wavenumbers
                intensities = self.raman_app.current_spectrum
                
                # Ensure intensities is 2D
                if intensities.ndim == 1:
                    intensities = intensities.reshape(1, -1)
                
                # Create feature matrix
                features = self.extract_vibrational_features(intensities, wavenumbers)
                
                # Extract sample name and metadata from main app
                sample_name = "Current Spectrum"
                sample_metadata = {}
                
                # Try to get metadata from main app
                if hasattr(self.raman_app, 'current_metadata') and self.raman_app.current_metadata:
                    sample_metadata = self.raman_app.current_metadata
                    
                    # Extract better name if available
                    if 'MINERAL NAME' in sample_metadata and sample_metadata['MINERAL NAME']:
                        sample_name = sample_metadata['MINERAL NAME']
                    elif 'NAME/ID' in sample_metadata and sample_metadata['NAME/ID']:
                        sample_name = sample_metadata['NAME/ID']
                
                # Try to get filename from main app
                if hasattr(self.raman_app, 'current_filename') and self.raman_app.current_filename:
                    if sample_name == "Current Spectrum":  # Only use if we don't have a better name
                        sample_name = os.path.basename(self.raman_app.current_filename)
                        base_name, _ = os.path.splitext(sample_name)
                        sample_name = base_name
                    
                    # Add filename to metadata
                    sample_metadata['FILENAME'] = self.raman_app.current_filename
                
                # Add import source to metadata
                sample_metadata['IMPORT_SOURCE'] = 'main_app'
                sample_metadata['IMPORT_DATE'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Store data with enhanced metadata
                self.cluster_data = {
                    'wavenumbers': wavenumbers,
                    'intensities': intensities,
                    'features': features,
                    'features_scaled': StandardScaler().fit_transform(features),
                    'sample_labels': [sample_name],
                    'sample_metadata': [sample_metadata]
                }
                
                # Update status
                if intensities.shape[0] == 1:
                    self.status_bar.config(text=f"Imported single spectrum: {len(wavenumbers)} points")
                    messagebox.showinfo("Info", 
                        "Single spectrum imported. For better clustering results, consider importing multiple spectra.")
                else:
                    self.status_bar.config(text=f"Imported {intensities.shape[0]} spectra: {len(wavenumbers)} points each")
                
                self.update_preview()
                
                # Enable clustering tab
                self.notebook.tab(1, state='normal')  # Enable clustering tab
            else:
                messagebox.showinfo("Info", "No spectrum data available in main application.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import data: {str(e)}")

    def import_from_file(self):
        """Import data from a file."""
        try:
            # Get file path from user
            file_path = filedialog.askopenfilename(
                title="Select Data File",
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("Text files", "*.txt"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                return
                
            # Read data based on file extension
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path, header=0 if self.has_header.get() else None)
            else:
                data = pd.read_csv(file_path, sep='\t', header=0 if self.has_header.get() else None)
                
            # Extract wavenumbers and intensities
            wavenumbers = data.iloc[:, self.wavenumber_col.get()].values
            intensities = data.iloc[:, self.intensity_col.get()].values
            
            # If only one intensity column, reshape to 2D
            if intensities.ndim == 1:
                intensities = intensities.reshape(1, -1)
            
            # Check if we need to interpolate to a common scale
            if self.cluster_data is not None and self.cluster_data['wavenumbers'] is not None:
                # Get existing wavenumber scale
                existing_wavenumbers = self.cluster_data['wavenumbers']
                
                # Check if wavenumber scales are different
                if not np.array_equal(wavenumbers, existing_wavenumbers):
                    # Ask user if they want to interpolate
                    response = messagebox.askyesno(
                        "Different Wavenumber Scales",
                        "The imported spectra have a different wavenumber scale than the existing data.\n\n"
                        "Would you like to interpolate the new spectra to match the existing scale?\n\n"
                        "Yes: Interpolate to existing scale\n"
                        "No: Create new scale (may affect clustering results)"
                    )
                    
                    if response:
                        # Interpolate to existing scale
                        new_intensities = np.zeros((intensities.shape[0], len(existing_wavenumbers)))
                        for i in range(intensities.shape[0]):
                            new_intensities[i] = np.interp(
                                existing_wavenumbers,
                                wavenumbers,
                                intensities[i],
                                left=0,   # Use 0 for values outside the original range
                                right=0
                            )
                        intensities = new_intensities
                        wavenumbers = existing_wavenumbers
                    else:
                        # Create new scale by finding common range
                        min_wavenumber = max(np.min(wavenumbers), np.min(existing_wavenumbers))
                        max_wavenumber = min(np.max(wavenumbers), np.max(existing_wavenumbers))
                        
                        # Create new common scale
                        num_points = min(len(wavenumbers), len(existing_wavenumbers))
                        common_wavenumbers = np.linspace(min_wavenumber, max_wavenumber, num_points)
                        
                        # Interpolate both existing and new data to common scale
                        # First, interpolate new data
                        new_intensities = np.zeros((intensities.shape[0], num_points))
                        for i in range(intensities.shape[0]):
                            new_intensities[i] = np.interp(
                                common_wavenumbers,
                                wavenumbers,
                                intensities[i],
                                left=0,
                                right=0
                            )
                        intensities = new_intensities
                        wavenumbers = common_wavenumbers
                        
                        # Then interpolate existing data
                        existing_intensities = self.cluster_data['intensities']
                        new_existing_intensities = np.zeros((existing_intensities.shape[0], num_points))
                        for i in range(existing_intensities.shape[0]):
                            new_existing_intensities[i] = np.interp(
                                common_wavenumbers,
                                existing_wavenumbers,
                                existing_intensities[i],
                                left=0,
                                right=0
                            )
                        self.cluster_data['intensities'] = new_existing_intensities
                        self.cluster_data['wavenumbers'] = common_wavenumbers
            
            # Create feature matrix
            features = self.extract_vibrational_features(intensities, wavenumbers)
            
            # Store data
            if self.cluster_data is None:
                self.cluster_data = {
                    'wavenumbers': wavenumbers,
                    'intensities': intensities,
                    'features': features,
                    'features_scaled': StandardScaler().fit_transform(features)
                }
            else:
                # Append new data to existing data
                self.cluster_data['intensities'] = np.vstack([
                    self.cluster_data['intensities'], intensities
                ])
                # Update features from combined dataset
                features = self.extract_vibrational_features(
                    self.cluster_data['intensities'],
                    self.cluster_data['wavenumbers']
                )
                self.cluster_data['features'] = features
                self.cluster_data['features_scaled'] = StandardScaler().fit_transform(features)
            
            # Update status
            self.status_bar.config(text=f"Imported data from file: {len(wavenumbers)} points, {intensities.shape[0]} spectra")
            self.update_preview()
            
            # Enable clustering tab
            self.notebook.tab(1, state='normal')  # Enable clustering tab
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import data: {str(e)}")
            print(f"DEBUG: Import error: {str(e)}")

    def import_from_database(self):
        """Import data from the database."""
        try:
            # Get database items
            db_items = self.raman_app.get_database_items()
            
            if not db_items:
                messagebox.showinfo("Info", "No items in database.")
                return
                
            # Create selection window
            select_window = tk.Toplevel(self.window)
            select_window.title("Select Database Items")
            select_window.geometry("400x500")
            
            # Create listbox
            listbox = tk.Listbox(select_window, selectmode=tk.MULTIPLE)
            listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Add items to listbox
            for item in db_items:
                listbox.insert(tk.END, item)
                
            # Add buttons
            button_frame = ttk.Frame(select_window)
            button_frame.pack(fill=tk.X, padx=5, pady=5)
            
            def import_selected():
                selected_indices = listbox.curselection()
                if not selected_indices:
                    messagebox.showinfo("Info", "Please select items to import.")
                    return
                    
                # Get selected items
                selected_items = [listbox.get(i) for i in selected_indices]
                
                # Import data for each item
                all_wavenumbers = []
                all_intensities = []
                
                for item in selected_items:
                    data = self.raman_app.get_database_item_data(item)
                    if data:
                        all_wavenumbers.append(data['wavenumbers'])
                        all_intensities.append(data['intensities'])
                
                if all_wavenumbers:
                    # Stack intensities
                    intensities = np.vstack(all_intensities)
                    # Append data
                    self.append_spectra(all_wavenumbers[0], intensities)
                
                select_window.destroy()
            
            ttk.Button(button_frame, text="Import Selected", 
                      command=import_selected).pack(side=tk.LEFT, padx=2)
            ttk.Button(button_frame, text="Cancel", 
                      command=select_window.destroy).pack(side=tk.LEFT, padx=2)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import data: {str(e)}")

    def update_preview(self):
        """Update the data preview text."""
        if self.cluster_data is None:
            return
            
        preview_text = "Data Preview:\n\n"
        
        # Add basic information
        preview_text += f"Number of spectra: {self.cluster_data['intensities'].shape[0]}\n"
        preview_text += f"Points per spectrum: {len(self.cluster_data['wavenumbers'])}\n"
        preview_text += f"Number of features: {self.cluster_data['features'].shape[1]}\n\n"
        
        # Add feature statistics
        preview_text += "Feature Statistics:\n"
        for i, feature in enumerate(self.cluster_data['features'].T):
            preview_text += f"Feature {i+1}:\n"
            preview_text += f"  Mean: {np.mean(feature):.3f}\n"
            preview_text += f"  Std: {np.std(feature):.3f}\n"
            preview_text += f"  Min: {np.min(feature):.3f}\n"
            preview_text += f"  Max: {np.max(feature):.3f}\n\n"
        
        # Add recommendation based on number of spectra
        if self.cluster_data['intensities'].shape[0] == 1:
            preview_text += "\nRecommendation: For better clustering results, consider importing multiple spectra.\n"
            preview_text += "You can import additional spectra from the database or from files."
        
        self.preview_text.delete(1.0, tk.END)
        self.preview_text.insert(tk.END, preview_text)

    def select_import_folder(self):
        """Select folder for batch import."""
        folder = filedialog.askdirectory(title="Select Folder with Spectrum Files")
        if folder:
            self.selected_folder = folder
            # Display full path in status bar and shortened path in label
            self.status_bar.config(text=f"Selected folder: {folder}")
            display_path = folder
            if len(display_path) > 50:  # Truncate long paths
                display_path = "..." + display_path[-47:]
            self.folder_path_var.set(folder)
            self.folder_path_var.set(display_path)
            
            # List files in the folder (check for both .csv and .txt)
            files = []
            for f in os.listdir(folder):
                if f.lower().endswith(('.csv', '.txt')):
                    files.append(f)
            
            if files:
                self.status_bar.config(
                    text=f"Found {len(files)} files in {folder}"
                )
            else:
                self.status_bar.config(
                    text=f"No .csv or .txt files found in {folder}"
                )

    def start_batch_import(self):
        """Start the batch import process."""
        if not hasattr(self, 'selected_folder') or not self.selected_folder:
            messagebox.showinfo("Info", "Please select a folder first")
            return
            
        try:
            # Get column indices
            wavenumber_col = int(self.wavenumber_col.get())
            intensity_col = int(self.intensity_col.get())
            
            # Get all files in the folder
            files = []
            for ext in ['.csv', '.txt']:
                files.extend(glob.glob(os.path.join(self.selected_folder, f'*{ext}')))
                files.extend(glob.glob(os.path.join(self.selected_folder, f'*{ext.upper()}')))
            
            if not files:
                messagebox.showinfo("Info", f"No valid files found in {self.selected_folder}")
                return
                
            # Initialize data storage
            all_wavenumbers = []
            all_intensities = []
            all_filenames = []  # Store filenames for labels
            all_metadata = []   # Store any extracted metadata
            
            # First pass: collect all wavenumber ranges to find common scale
            min_wavenumber = float('inf')
            max_wavenumber = float('-inf')
            wavenumber_counts = []
            
            # Process each file to find wavenumber ranges
            total_files = len(files)
            for i, file_path in enumerate(files):
                try:
                    # Update progress
                    progress = (i / total_files) * 30  # First phase is 30% of progress
                    self.import_progress['value'] = progress
                    self.import_status.config(text=f"Analyzing {os.path.basename(file_path)}...")
                    self.window.update()
                    
                    # Read file based on extension
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext == '.csv':
                        data = pd.read_csv(file_path, header=None)
                    else:  # .txt
                        try:
                            data = pd.read_csv(file_path, sep='\t', header=None)
                        except:
                            data = pd.read_csv(file_path, sep=' ', header=None)
                    
                    # Extract wavenumbers
                    wavenumbers = data.iloc[:, wavenumber_col].values
                    
                    # Update min/max wavenumbers
                    min_wavenumber = min(min_wavenumber, np.min(wavenumbers))
                    max_wavenumber = max(max_wavenumber, np.max(wavenumbers))
                    wavenumber_counts.append(len(wavenumbers))
                    
                except Exception as e:
                    print(f"Error analyzing {file_path}: {str(e)}")
                    continue
            
            # Create common wavenumber scale using median number of points
            num_points = int(np.median(wavenumber_counts))
            common_wavenumbers = np.linspace(min_wavenumber, max_wavenumber, num_points)
            
            # Second pass: interpolate and collect spectra
            for i, file_path in enumerate(files):
                try:
                    # Update progress
                    progress = 30 + (i / total_files) * 70  # Remaining 70% of progress
                    self.import_progress['value'] = progress
                    self.import_status.config(text=f"Processing {os.path.basename(file_path)}...")
                    self.window.update()
                    
                    # Read file based on extension
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext == '.csv':
                        data = pd.read_csv(file_path, header=None)
                    else:  # .txt
                        try:
                            data = pd.read_csv(file_path, sep='\t', header=None)
                        except:
                            data = pd.read_csv(file_path, sep=' ', header=None)
                    
                    # Extract wavenumbers and intensities
                    wavenumbers = data.iloc[:, wavenumber_col].values
                    intensities = data.iloc[:, intensity_col].values
                    
                    # Interpolate intensities to common wavenumber scale
                    interp_intensities = np.interp(
                        common_wavenumbers,
                        wavenumbers,
                        intensities,
                        left=0,   # Use 0 for values outside the original range
                        right=0
                    )
                    
                    # Extract metadata from filename
                    filename = os.path.basename(file_path)
                    base_name, _ = os.path.splitext(filename)
                    
                    # Store data
                    all_wavenumbers.append(common_wavenumbers)  # Use common scale
                    all_intensities.append(interp_intensities)
                    all_filenames.append(base_name)  # Use basename without extension
                    
                    # Create metadata dictionary for this file
                    file_metadata = {
                        'FILENAME': filename,
                        'FILEPATH': file_path,
                        'IMPORT_SOURCE': 'file',
                        'IMPORT_DATE': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Try to extract additional metadata from filename
                    # Check for common patterns like mineral_location_ID
                    parts = base_name.replace('-', '_').split('_')
                    if len(parts) >= 2:
                        file_metadata['NAME/ID'] = parts[0]
                        if len(parts) >= 3:
                            file_metadata['LOCALITY'] = parts[1]
                    
                    all_metadata.append(file_metadata)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    continue
            
            if not all_wavenumbers:
                messagebox.showinfo("Info", "No valid data found in any files")
                return
            
            # Stack intensities (now they should all have the same length)
            all_intensities = np.vstack(all_intensities)
            
            # Extract features
            features = self.extract_vibrational_features(all_intensities, common_wavenumbers)
            
            # Store data with enhanced metadata
            self.cluster_data = {
                'wavenumbers': common_wavenumbers,
                'intensities': all_intensities,
                'features': features,
                'features_scaled': None,
                'labels': None,
                'linkage_matrix': None,
                'distance_matrix': None,
                'sample_labels': all_filenames,  # Store filenames as labels
                'sample_metadata': all_metadata  # Store extracted metadata
            }
            
            # Update UI
            self.update_ui_after_import(len(all_wavenumbers))
            
        except Exception as e:
            messagebox.showerror("Error", f"Import failed: {str(e)}")
            self.import_status.config(text="Import failed")
            self.import_progress['value'] = 0
            print(f"DEBUG: Import error: {str(e)}")
            import traceback
            traceback.print_exc()

    def append_data(self):
        """Append more data to existing dataset."""
        if self.cluster_data is None:
            messagebox.showinfo("Info", "Please import initial data first.")
            return
            
        # Create append window
        append_window = tk.Toplevel(self.window)
        append_window.title("Append Data")
        append_window.geometry("400x300")
        
        # Create controls
        control_frame = ttk.Frame(append_window, padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_frame, text="Select Import Method:").pack(anchor=tk.W)
        
        def append_from_file():
            file_path = filedialog.askopenfilename(
                title="Select Data File",
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("Text files", "*.txt"),
                    ("All files", "*.*")
                ]
            )
            if file_path:
                try:
                    # Read file
                    if file_path.endswith('.csv'):
                        data = pd.read_csv(file_path, header=0 if self.has_header.get() else None)
                    else:
                        data = pd.read_csv(file_path, sep='\t', header=0 if self.has_header.get() else None)
                    
                    # Extract data
                    wavenumbers = data.iloc[:, self.wavenumber_col.get()].values
                    intensities = data.iloc[:, self.intensity_col.get()].values
                    
                    # Append data
                    self.append_spectra(wavenumbers, intensities)
                    append_window.destroy()
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to append data: {str(e)}")
        
        def append_from_folder():
            folder = filedialog.askdirectory(title="Select Folder with Spectrum Files")
            if folder:
                try:
                    # Get files
                    file_ext = '.csv' if self.file_format.get() == 'csv' else '.txt'
                    files = [f for f in os.listdir(folder) if f.endswith(file_ext)]
                    
                    if not files:
                        messagebox.showinfo("Info", "No valid files found in selected folder.")
                        return
                    
                    # Process files
                    all_wavenumbers = []
                    all_intensities = []
                    failed_files = []
                    
                    for filename in files:
                        try:
                            file_path = os.path.join(folder, filename)
                            if self.file_format.get() == 'csv':
                                data = pd.read_csv(file_path, header=0 if self.has_header.get() else None)
                            else:
                                data = pd.read_csv(file_path, sep='\t', header=0 if self.has_header.get() else None)
                            
                            wavenumbers = data.iloc[:, self.wavenumber_col.get()].values
                            intensities = data.iloc[:, self.intensity_col.get()].values
                            
                            all_wavenumbers.append(wavenumbers)
                            all_intensities.append(intensities)
                            
                        except Exception as e:
                            failed_files.append((filename, str(e)))
                            continue
                    
                    if all_wavenumbers:
                        # Stack intensities
                        intensities = np.vstack(all_intensities)
                        # Append data
                        self.append_spectra(all_wavenumbers[0], intensities)
                        
                        if failed_files:
                            error_msg = "Some files failed to import:\n\n"
                            for filename, error in failed_files:
                                error_msg += f"{filename}: {error}\n"
                            messagebox.showwarning("Import Warnings", error_msg)
                    
                    append_window.destroy()
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to append data: {str(e)}")
        
        def append_from_database():
            # Get database items
            db_items = self.raman_app.get_database_items()
            
            if not db_items:
                messagebox.showinfo("Info", "No items in database.")
                return
            
            # Create selection window
            select_window = tk.Toplevel(append_window)
            select_window.title("Select Database Items")
            select_window.geometry("400x500")
            
            # Create listbox
            listbox = tk.Listbox(select_window, selectmode=tk.MULTIPLE)
            listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Add items to listbox
            for item in db_items:
                listbox.insert(tk.END, item)
            
            def import_selected():
                selected_indices = listbox.curselection()
                if not selected_indices:
                    messagebox.showinfo("Info", "Please select items to import.")
                    return
                
                # Get selected items
                selected_items = [listbox.get(i) for i in selected_indices]
                
                # Import data for each item
                all_wavenumbers = []
                all_intensities = []
                
                for item in selected_items:
                    data = self.raman_app.get_database_item_data(item)
                    if data:
                        all_wavenumbers.append(data['wavenumbers'])
                        all_intensities.append(data['intensities'])
                
                if all_wavenumbers:
                    # Stack intensities
                    intensities = np.vstack(all_intensities)
                    # Append data
                    self.append_spectra(all_wavenumbers[0], intensities)
                
                select_window.destroy()
                append_window.destroy()
            
            # Add buttons
            button_frame = ttk.Frame(select_window)
            button_frame.pack(fill=tk.X, padx=5, pady=5)
            ttk.Button(button_frame, text="Import Selected", 
                      command=import_selected).pack(side=tk.LEFT, padx=2)
            ttk.Button(button_frame, text="Cancel", 
                      command=select_window.destroy).pack(side=tk.LEFT, padx=2)
        
        # Add buttons
        ttk.Button(control_frame, text="Append from File", 
                  command=append_from_file).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Append from Folder", 
                  command=append_from_folder).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Append from Database", 
                  command=append_from_database).pack(fill=tk.X, pady=2)
        
        # Add cancel button
        ttk.Button(append_window, text="Cancel", 
                  command=append_window.destroy).pack(pady=10)

    def append_spectra(self, wavenumbers, intensities):
        """Append new spectra to existing dataset.
        
        Parameters:
        -----------
        wavenumbers : ndarray
            Array of wavenumbers for the new spectra
        intensities : ndarray
            Array of intensities for the new spectra
        """
        try:
            # Ensure arrays are numpy arrays
            wavenumbers = np.asarray(wavenumbers)
            intensities = np.asarray(intensities)
            
            # Ensure intensities is 2D
            if intensities.ndim == 1:
                intensities = intensities.reshape(1, -1)
            
            # Check if cluster_data already has data
            if self.cluster_data['wavenumbers'] is None or self.cluster_data['intensities'] is None:
                # This is the first data, just store it directly
                self.cluster_data['wavenumbers'] = wavenumbers
                self.cluster_data['intensities'] = intensities
                
                # Extract features
                features = self.extract_vibrational_features(intensities, wavenumbers)
                self.cluster_data['features'] = features
                self.cluster_data['features_scaled'] = StandardScaler().fit_transform(features)
                
                messagebox.showinfo("Success", f"Successfully imported {intensities.shape[0]} spectra.")
                return
            
            # Check wavenumber compatibility and interpolate if needed
            existing_wavenumbers = self.cluster_data['wavenumbers']
            
            if not np.array_equal(wavenumbers, existing_wavenumbers):
                messagebox.showinfo(
                    "Info", 
                    "Wavenumber ranges don't match. Interpolating new spectra to match existing range."
                )
                
                # Interpolate new intensities to match existing wavenumbers
                new_intensities = np.zeros((intensities.shape[0], len(existing_wavenumbers)))
                
                for i in range(intensities.shape[0]):
                    new_intensities[i] = np.interp(
                        existing_wavenumbers,
                        wavenumbers, 
                        intensities[i],
                        left=0,   # Use 0 for values outside the original range
                        right=0
                    )
                
                intensities = new_intensities
            
            # Append intensities to existing data
            self.cluster_data['intensities'] = np.vstack([
                self.cluster_data['intensities'], intensities
            ])
            
            # Update features from combined dataset
            features = self.extract_vibrational_features(
                self.cluster_data['intensities'],
                self.cluster_data['wavenumbers']
            )
            
            # Update cluster data
            self.cluster_data['features'] = features
            self.cluster_data['features_scaled'] = StandardScaler().fit_transform(features)
            
            # Update status
            self.status_bar.config(
                text=f"Total spectra: {self.cluster_data['intensities'].shape[0]}"
            )
            
            # Update preview if method exists
            try:
                if hasattr(self, 'update_preview'):
                    self.update_preview()
            except Exception as e:
                print(f"Warning: Failed to update preview: {str(e)}")
            
            messagebox.showinfo("Success", 
                              f"Successfully appended {intensities.shape[0]} spectra.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to append spectra: {str(e)}")
            raise

    def on_pick(self, event):
        """Handle point selection in scatter plot."""
        if not hasattr(self, 'selected_points'):
            self.selected_points = set()
        ind = event.ind[0]  # Get the index of the selected point
        if ind in self.selected_points:
            self.selected_points.remove(ind)
        else:
            self.selected_points.add(ind)
        self.update_scatter_plot()

    def open_database_import_dialog(self):
        """Open a dialog to filter and import spectra from the database."""
        dialog = tk.Toplevel(self.window)
        dialog.title("Import from Database")
        dialog.geometry("650x700")
        
        # Add option to locate database file manually
        ttk.Button(dialog, text="Locate Database File", command=self.locate_database_file).pack(fill="x", padx=10, pady=5)
        
        # Status label to show database status
        db_status_var = tk.StringVar(value="Checking database...")
        status_label = ttk.Label(dialog, textvariable=db_status_var, foreground="blue")
        status_label.pack(fill="x", padx=10, pady=2)
        
        # Force database check and get classifications
        db = self.get_database()
        hey_classes = self.get_hey_classifications()
        
        # Update status label
        if db:
            db_status_var.set(f"Database connected with {len(db)} entries")
            status_label.config(foreground="green")
        else:
            db_status_var.set("No database found. Use 'Locate Database File' button.")
            status_label.config(foreground="red")
        
        # Create a frame for Hey classification search
        hey_frame = ttk.LabelFrame(dialog, text="Step 1: Select Hey Classification (optional)")
        hey_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Add search field for Hey classification
        search_frame = ttk.Frame(hey_frame)
        search_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(search_frame, text="Filter classifications:").pack(side=tk.LEFT, padx=5, pady=2)
        hey_search_var = tk.StringVar()
        hey_search_entry = ttk.Entry(search_frame, textvariable=hey_search_var)
        hey_search_entry.pack(side=tk.LEFT, fill="x", expand=True, padx=5, pady=2)
        
        # Add clear button
        def clear_search():
            hey_search_var.set("")
            hey_search_entry.focus()
        
        ttk.Button(search_frame, text="Clear", command=clear_search).pack(side=tk.RIGHT, padx=5)
        
        # Function to filter Hey classifications
        def filter_hey_classes(*args):
            search_term = hey_search_var.get().lower().strip()
            hey_listbox.delete(0, tk.END)
            
            # Get all Hey classifications from the database
            db = self.get_database()
            all_classes = set()
            
            if db:
                for entry in db.values():
                    # Look in metadata first (standard location)
                    if 'metadata' in entry and isinstance(entry['metadata'], dict):
                        hey_class = entry['metadata'].get('HEY CLASSIFICATION', '')
                        if hey_class:
                            all_classes.add(hey_class)
                    # Legacy format
                    elif 'hey_classification' in entry and entry['hey_classification']:
                        all_classes.add(entry['hey_classification'])
            
            # If no classes found in database, use default hey_classes list
            filtered_classes = hey_classes if not all_classes else sorted(all_classes)
            
            if search_term:
                # Match terms that contain the search term or terms where individual words match
                filtered_classes = []
                classes_to_filter = all_classes if all_classes else hey_classes
                for cls in classes_to_filter:
                    cls_lower = cls.lower()
                    # Direct match
                    if search_term in cls_lower:
                        filtered_classes.append(cls)
                        continue
                    
                    # Match individual words
                    words = search_term.split()
                    if all(word in cls_lower for word in words):
                        filtered_classes.append(cls)
            
            # Sort results alphabetically
            filtered_classes.sort()
            
            # Update the listbox
            for cls in filtered_classes:
                hey_listbox.insert(tk.END, cls)
                
            # Show count
            count_text = f"{len(filtered_classes)} classifications found"
            if search_term:
                count_text += f" matching '{search_term}'"
            selected_hey_var.set(count_text)
        
        # Create a frame with scrollbars for the listbox
        list_frame = ttk.Frame(hey_frame)
        list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add listbox for Hey classifications
        hey_listbox = tk.Listbox(list_frame, selectmode=tk.SINGLE, height=10, 
                               yscrollcommand=scrollbar.set, exportselection=False)
        hey_listbox.pack(side=tk.LEFT, fill="both", expand=True)
        scrollbar.config(command=hey_listbox.yview)
        
        # Populate the listbox and show initial count
        selected_hey_var = tk.StringVar()
        selected_hey_var.set(f"{len(hey_classes)} classifications available")
        
        # Call filter_hey_classes to initialize the listbox
        filter_hey_classes()
        
        # Selected classification variable
        selected_hey_label = ttk.Label(hey_frame, textvariable=selected_hey_var, 
                                      font=("", 10, "bold"))
        selected_hey_label.pack(fill="x", padx=5, pady=2)
        
        # Function to handle listbox selection
        def on_hey_select(event):
            try:
                selection = hey_listbox.curselection()
                if selection:
                    index = selection[0]
                    value = hey_listbox.get(index)
                    selected_hey_var.set(f"Selected: {value}")
                    
                    # Get database entries with this classification for further info
                    db = self.get_database()
                    matching_count = 0
                    
                    if db:
                        for entry in db.values():
                            # Check metadata first (standard location)
                            entry_class = ""
                            if 'metadata' in entry and isinstance(entry['metadata'], dict):
                                entry_class = entry['metadata'].get('HEY CLASSIFICATION', '')
                            # Legacy format
                            if not entry_class and 'hey_classification' in entry:
                                entry_class = entry.get('hey_classification', '')
                                
                            if entry_class and entry_class.lower() == value.lower():
                                matching_count += 1
                                
                        if matching_count > 0:
                            selected_hey_var.set(f"Selected: {value} ({matching_count} database entries)")
            except Exception as e:
                print(f"Error in hey selection: {str(e)}")
                
        hey_listbox.bind('<<ListboxSelect>>', on_hey_select)
        
        # Bind search entry to filter function
        hey_search_var.trace("w", filter_hey_classes)
        
        # Elements filters
        elements_frame = ttk.LabelFrame(dialog, text="Step 2: Filter by Elements (optional)")
        elements_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(elements_frame, text="Must have elements (comma separated):").pack(anchor="w", padx=5, pady=2)
        must_have_entry = ttk.Entry(elements_frame)
        must_have_entry.pack(fill="x", padx=5, pady=2)
        must_have_entry.insert(0, "")
        
        ttk.Label(elements_frame, text="Only these elements:").pack(anchor="w", padx=5, pady=2)
        only_entry = ttk.Entry(elements_frame)
        only_entry.pack(fill="x", padx=5, pady=2)
        only_entry.insert(0, "")
        
        ttk.Label(elements_frame, text="Exclude these elements:").pack(anchor="w", padx=5, pady=2)
        not_entry = ttk.Entry(elements_frame)
        not_entry.pack(fill="x", padx=5, pady=2)
        not_entry.insert(0, "")
        
        # Function to refresh Hey classifications when database changes
        def refresh_hey_classes():
            nonlocal hey_classes
            hey_classes = self.get_hey_classifications()
            filter_hey_classes()  # Apply any current filter
            if db := self.get_database():
                db_status_var.set(f"Database refreshed: {len(db)} entries")
                status_label.config(foreground="green")
            else:
                db_status_var.set("No database found")
                status_label.config(foreground="red")
        
        # Add refresh button
        ttk.Button(dialog, text="Refresh Database", command=refresh_hey_classes).pack(fill="x", padx=10, pady=5)
        
        # Buttons frame
        button_frame = ttk.LabelFrame(dialog, text="Step 3: Search and Import")
        button_frame.pack(fill="x", padx=10, pady=10)
        
        # Import button
        def do_import():
            selection = hey_listbox.curselection()
            hey_class = ""
            if selection:
                hey_class = hey_listbox.get(selection[0])
            
            must_have = [e.strip() for e in must_have_entry.get().split(",") if e.strip()]
            only = [e.strip() for e in only_entry.get().split(",") if e.strip()]
            not_these = [e.strip() for e in not_entry.get().split(",") if e.strip()]
            
            # Show summary of search criteria
            criteria_msg = "Search with the following criteria:\n\n"
            if hey_class:
                criteria_msg += f"Hey Classification: {hey_class}\n"
            else:
                criteria_msg += "Hey Classification: Any\n"
                
            if must_have:
                criteria_msg += f"Must have elements: {', '.join(must_have)}\n"
            if only:
                criteria_msg += f"Only these elements: {', '.join(only)}\n"
            if not_these:
                criteria_msg += f"Exclude elements: {', '.join(not_these)}\n"
                
            if not (hey_class or must_have or only or not_these):
                criteria_msg += "No filters applied - will import all spectra."
                
            proceed = messagebox.askyesno("Confirm Search", criteria_msg + "\n\nProceed with search?")
            if not proceed:
                return
                
            dialog.destroy()
            self.import_from_database_with_filters(hey_class, must_have, only, not_these)
        
        import_btn = ttk.Button(button_frame, text="Search & Import", command=do_import)
        import_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Update the locate_database_file method to also update this dialog
        original_locate_fn = self.locate_database_file
        
        def wrapped_locate_fn():
            result = original_locate_fn()
            # After loading database, refresh the Hey classification dropdown
            refresh_hey_classes()
            return result
            
        # Replace the method temporarily while dialog is open
        self.locate_database_file = wrapped_locate_fn
        
        # Restore original method when dialog is closed
        def on_dialog_close():
            self.locate_database_file = original_locate_fn
            dialog.destroy()
            
        dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)

    def locate_database_file(self):
        """Allow user to manually locate the database.pkl file."""
        try:
            import pickle
            import os
            from tkinter import filedialog
            
            # Prompt user to select database file
            db_path = filedialog.askopenfilename(
                title="Select Database File",
                filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")]
            )
            
            if db_path:
                # Try to load the database
                try:
                    with open(db_path, 'rb') as f:
                        database = pickle.load(f)
                    
                    # Check if it's a valid database
                    if isinstance(database, dict) and database:
                        # Store database location for future use
                        self.custom_db_path = db_path
                        
                        # Add reference to database in raman_app if it exists
                        if hasattr(self.raman_app, 'raman'):
                            self.raman_app.raman.database = database
                        
                        messagebox.showinfo(
                            "Database Loaded", 
                            f"Successfully loaded database with {len(database)} entries from {db_path}"
                        )
                        
                        # Update status bar
                        self.status_bar.config(text=f"Using custom database: {db_path} ({len(database)} entries)")
                        
                        # Update Hey classification combobox if it exists
                        hey_classes = set()
                        for entry in database.values():
                            # Check in metadata first (standard location)
                            if 'metadata' in entry and isinstance(entry['metadata'], dict):
                                hey_class = entry['metadata'].get('HEY CLASSIFICATION', '')
                                if hey_class:
                                    hey_classes.add(hey_class)
                            # Check legacy format
                            elif 'hey_classification' in entry and entry['hey_classification']:
                                hey_classes.add(entry['hey_classification'])
                                
                        # Sort the Hey classifications for display
                        hey_classes_list = sorted(hey_classes)
                        
                        # Find all comboboxes in open windows and update them
                        for widget in self.window.winfo_children():
                            if isinstance(widget, tk.Toplevel):
                                for child in widget.winfo_children():
                                    if isinstance(child, ttk.Combobox) and 'Hey Classification' in str(child):
                                        child['values'] = hey_classes_list
                        
                        return True  # Success
                    else:
                        messagebox.showerror("Invalid Database", "The selected file is not a valid Raman database")
                        self.status_bar.config(text="Invalid database file selected")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load database: {str(e)}")
                    self.status_bar.config(text=f"Error loading database: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_bar.config(text=f"Error: {str(e)}")
            
        return False  # Failed or cancelled

    def get_database(self):
        """Return the database from the main app, handling nested raman attribute."""
        # First try to get from main app
        if hasattr(self.raman_app, 'raman') and hasattr(self.raman_app.raman, 'database'):
            db = self.raman_app.raman.database
            if db:
                self.status_bar.config(text=f"Using database from main app: {len(db)} entries")
            return db
        
        # If not available, try to load from home directory
        try:
            import pickle
            import os
            home_dir = os.path.expanduser("~")
            db_path = os.path.join(home_dir, "raman_database.pkl")
            
            if os.path.exists(db_path):
                with open(db_path, 'rb') as f:
                    database = pickle.load(f)
                messagebox.showinfo("Database Found", f"Using database from home directory: {db_path}")
                self.status_bar.config(text=f"Using database from home directory: {len(database)} entries")
                return database
            else:
                messagebox.showwarning("Database Not Found", f"Database file not found in home directory: {db_path}")
                self.status_bar.config(text="No database found in home directory")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load database from home directory: {str(e)}")
            self.status_bar.config(text="Error loading database from home directory")
            
        # If we have a custom path, try that
        if hasattr(self, 'custom_db_path') and self.custom_db_path and os.path.exists(self.custom_db_path):
            try:
                with open(self.custom_db_path, 'rb') as f:
                    database = pickle.load(f)
                self.status_bar.config(text=f"Using custom database: {self.custom_db_path} ({len(database)} entries)")
                return database
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load database from custom path: {str(e)}")
                self.status_bar.config(text="Error loading database from custom path")
            
        self.status_bar.config(text="No database loaded")
        return None

    def get_hey_classifications(self):
        """Return a list of Hey classifications primarily from the database entries."""
        # Default Hey classifications if database is empty or missing
        default_classifications = [
            "Nesosilicate", "Sorosilicate", "Cyclosilicate", "Inosilicate", 
            "Phyllosilicate", "Tectosilicate", "Carbonates", "Oxides", "Sulfides", 
            "Sulfates", "Phosphates", "Halides", "Elements", "Borate", "Vanadate",
            "Tungstate", "Molybdate", "Hydroxide", "Arsenate", "Chromate", "Nitrate",
            "Native Elements", "Silicate", "Organic Minerals", 
            # Add singular versions for compatibility
            "Carbonate", "Oxide", "Sulfide", "Sulfate", "Phosphate", "Halide", "Element"
        ]
        
        # Create a mapping for common variations to standardized names
        standardize_map = {
            "carbonate": "Carbonates",
            "carbonates": "Carbonates",
            "oxide": "Oxides",
            "oxides": "Oxides",
            "sulfide": "Sulfides",
            "sulfides": "Sulfides",
            "sulphide": "Sulfides", 
            "sulphides": "Sulfides",
            "sulfate": "Sulfates",
            "sulfates": "Sulfates",
            "sulphate": "Sulfates",
            "sulphates": "Sulfates",
            "phosphate": "Phosphates", 
            "phosphates": "Phosphates",
            "silicate": "Silicates",
            "silicates": "Silicates",
            "halide": "Halides",
            "halides": "Halides",
            "element": "Elements", 
            "elements": "Elements",
            "native element": "Native Elements",
            "native elements": "Native Elements"
        }
        
        # Get CSV classifications only as fallback
        csv_hey_classes = set()
        if hasattr(self, '_csv_hey_classes') and self._csv_hey_classes:
            csv_hey_classes = self._csv_hey_classes
        
        db = self.get_database()
        db_hey_classes = set()
        
        if db is not None and db:
            # Extract Hey classifications from database
            db_classes = set()  # For debugging
            
            for entry in db.values():
                # Look for Hey Classification in metadata (which is the standard location)
                if 'metadata' in entry and isinstance(entry['metadata'], dict):
                    metadata = entry['metadata']
                    if 'HEY CLASSIFICATION' in metadata and metadata['HEY CLASSIFICATION']:
                        classification = metadata['HEY CLASSIFICATION']
                        db_classes.add(classification)
                        
                        # Standardize the classification
                        classification_lower = classification.lower()
                        if classification_lower in standardize_map:
                            classification = standardize_map[classification_lower]
                        
                        db_hey_classes.add(classification)
                # Also check for legacy format directly in entry
                elif 'hey_classification' in entry and entry['hey_classification']:
                    classification = entry['hey_classification']
                    db_classes.add(classification)
                    
                    # Standardize the classification
                    classification_lower = classification.lower()
                    if classification_lower in standardize_map:
                        classification = standardize_map[classification_lower]
                    
                    db_hey_classes.add(classification)
            
            # Debug output
            print("Hey Classifications found in database:")
            for cls in sorted(db_classes):
                print(f"  - '{cls}'")
            
            print("Standardized Hey Classifications from database:")
            for cls in sorted(db_hey_classes):
                print(f"  - '{cls}'")
        
        # Combine all sources of classifications without duplicates
        all_hey_classes = set()
        
        # Add database classifications first (they're the most up-to-date)
        all_hey_classes.update(db_hey_classes)
        
        # Add CSV classifications as fallback
        all_hey_classes.update(csv_hey_classes)
        
        # Add defaults for any that might be missing
        for cls in default_classifications:
            if cls not in all_hey_classes:
                all_hey_classes.add(cls)
        
        # Sort and return as a list
        return sorted(all_hey_classes)

    def import_from_database_with_filters(self, hey_class, must_have, only, not_these):
        """Import spectra from the main app database with filtering."""
        # Try to get database from main app or home directory
        db = self.get_database()
        
        # If no database found, check if we have a custom database path
        if db is None and hasattr(self, 'custom_db_path') and os.path.exists(self.custom_db_path):
            try:
                import pickle
                with open(self.custom_db_path, 'rb') as f:
                    db = pickle.load(f)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load database from custom path: {str(e)}")
        
        if db is None:
            messagebox.showerror("Error", "No database found. Please locate a database file using 'Locate Database File' button.")
            return
        
        # Debug info
        print(f"Filtering with: hey_class='{hey_class}', must_have={must_have}, only={only}, not_these={not_these}")
        print(f"Database has {len(db)} entries")
        
        # Create standardization map for Hey classifications
        standardize_map = {
            "carbonate": "Carbonates",
            "carbonates": "Carbonates",
            "carbonate with other anions": "Carbonates with other anions",
            "carbonates with other anions": "Carbonates with other anions",
            "oxide": "Oxides",
            "oxides": "Oxides",
            "sulfide": "Sulfides",
            "sulfides": "Sulfides",
            "sulphide": "Sulfides", 
            "sulphides": "Sulfides",
            "sulfate": "Sulfates",
            "sulfates": "Sulfates",
            "sulphate": "Sulfates",
            "sulphates": "Sulfates",
            "phosphate": "Phosphates", 
            "phosphates": "Phosphates",
            "silicate": "Silicates",
            "silicates": "Silicates",
            "halide": "Halides",
            "halides": "Halides",
            "element": "Elements", 
            "elements": "Elements",
            "native element": "Native Elements",
            "native elements": "Native Elements"
        }
        
        # Get a more complete list of possible variations from the RRUFF CSV
        if hasattr(self, '_csv_hey_classes'):
            csv_classes = self._csv_hey_classes
            for cls in csv_classes:
                if 'carbonate' in cls.lower() or 'carbonates' in cls.lower():
                    standardize_map[cls.lower()] = cls
        
        # Create a set of search terms for Hey classification using both the original term and standardized versions
        hey_search_terms = set()
        if hey_class:
            hey_search_terms.add(hey_class)
            # Add lowercase version
            hey_search_terms.add(hey_class.lower())
            # Check for singular/plural variations
            if hey_class.endswith('s'):
                hey_search_terms.add(hey_class[:-1])  # Remove 's' at the end
            else:
                hey_search_terms.add(hey_class + 's')  # Add 's' at the end
                
            # Check standardization map in both directions
            hey_class_lower = hey_class.lower()
            if hey_class_lower in standardize_map:
                hey_search_terms.add(standardize_map[hey_class_lower])
            
            # Also check reverse mapping (standardized to variations)
            for var, std in standardize_map.items():
                if std.lower() == hey_class.lower():
                    hey_search_terms.add(var)
                elif std == hey_class:
                    hey_search_terms.add(var)
            
            # Special case for "with other anions"
            if hey_class.lower() == "carbonates":
                hey_search_terms.add("Carbonates with other anions")
            elif hey_class.lower() == "carbonate":
                hey_search_terms.add("Carbonates with other anions")
            
            print(f"Hey classification search terms: {hey_search_terms}")
        
        # Create progress dialog
        progress_window = tk.Toplevel(self.window)
        progress_window.title("Importing Spectra")
        progress_window.geometry("400x150")
        progress_window.grab_set()  # Make it modal
        
        # Add progress information
        ttk.Label(progress_window, text="Filtering database entries...").pack(padx=10, pady=5)
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
        progress_bar.pack(fill="x", padx=10, pady=5)
        
        status_var = tk.StringVar(value="Scanning database...")
        status_label = ttk.Label(progress_window, textvariable=status_var)
        status_label.pack(padx=10, pady=5)
        
        # Update the window
        progress_window.update()
        
        # Filter database entries
        filtered = []
        total_entries = len(db)
        
        # Collect all Hey classifications in database for debugging
        all_db_hey = set()
        
        # Debug: Print structure of a couple of database entries
        print("\nDEBUG: Database entry structure:")
        entry_keys = list(db.keys())
        for i in range(min(2, len(entry_keys))):
            key = entry_keys[i]
            entry = db[key]
            print(f"\nEntry {i}: {key}")
            print(f"Entry keys: {list(entry.keys())}")
            
            # Check for metadata
            if 'metadata' in entry:
                print(f"Metadata type: {type(entry['metadata'])}")
                if isinstance(entry['metadata'], dict):
                    print(f"Metadata keys: {list(entry['metadata'].keys())}")
                    # Check for specific fields
                    for field in ['MINERAL NAME', 'NAME/ID', 'HEY CLASSIFICATION']:
                        if field in entry['metadata']:
                            print(f"{field}: {entry['metadata'][field]}")
                    
                    # Check for lowercase variants
                    for field in ['mineral name', 'name/id', 'hey classification']:
                        if field in entry['metadata']:
                            print(f"Lowercase {field}: {entry['metadata'][field]}")
        
        # For debugging rejections
        rejection_reasons = {
            'hey_class': 0,
            'must_have': 0,
            'only': 0,
            'not_these': 0
        }
        
        for i, (key, entry) in enumerate(db.items()):
            # Update progress
            progress_var.set((i / total_entries) * 30)  # First phase is 30% of progress
            status_var.set(f"Scanning entry {i+1} of {total_entries}: {key}")
            if i % 10 == 0:  # Update periodically to avoid freezing
                progress_window.update()
            
            # Get Hey Classification from metadata or from direct entry
            db_hey_class = ""
            if 'metadata' in entry and isinstance(entry['metadata'], dict):
                db_hey_class = entry['metadata'].get('HEY CLASSIFICATION', '').strip()
            if not db_hey_class and 'hey_classification' in entry:  # Legacy format
                db_hey_class = entry.get('hey_classification', '').strip()
            
            # Collect all Hey classifications for debugging
            if db_hey_class:
                all_db_hey.add(db_hey_class)
            
            # Filter by Hey classification if provided
            if hey_class:
                if not db_hey_class:
                    rejection_reasons['hey_class'] += 1
                    continue
                
                # Check against all search terms
                if db_hey_class not in hey_search_terms and db_hey_class.lower() not in hey_search_terms:
                    # Try standardized version
                    db_hey_lower = db_hey_class.lower()
                    if db_hey_lower in standardize_map:
                        standardized = standardize_map[db_hey_lower]
                        if standardized not in hey_search_terms and standardized.lower() not in hey_search_terms:
                            rejection_reasons['hey_class'] += 1
                            continue
                    else:
                        # Last attempt: check for partial matches 
                        matches_any = False
                        for term in hey_search_terms:
                            if term.lower() in db_hey_lower or db_hey_lower in term.lower():
                                matches_any = True
                                break
                        
                        if not matches_any:
                            rejection_reasons['hey_class'] += 1
                            continue
            
            # Filter by must-have elements
            if must_have:
                elements = set(entry.get('elements', []))
                if not all(e in elements for e in must_have):
                    rejection_reasons['must_have'] += 1
                    continue
            
            # Filter by only these elements
            if only:
                elements = set(entry.get('elements', []))
                if elements and (elements - set(only)):
                    rejection_reasons['only'] += 1
                    continue
            
            # Filter by not these elements
            if not_these:
                elements = set(entry.get('elements', []))
                if any(e in elements for e in not_these):
                    rejection_reasons['not_these'] += 1
                    continue
            
            filtered.append(entry)
        
        # Debug all Hey classifications found
        print(f"All Hey classifications found in database: {all_db_hey}")
        
        # Update status
        progress_var.set(30)
        status_var.set(f"Found {len(filtered)} matching spectra")
        progress_window.update()
        
        # Debug info
        print(f"Filtering resulted in {len(filtered)} matches")
        print(f"Rejection reasons: {rejection_reasons}")
        
        if not filtered:
            progress_window.destroy()
            # More detailed message about why no matches were found
            no_match_msg = "No spectra matched the selected filters.\n\n"
            if hey_class:
                no_match_msg += f"• {rejection_reasons['hey_class']} entries were rejected due to Hey Classification not matching '{hey_class}'.\n"
            if must_have:
                no_match_msg += f"• {rejection_reasons['must_have']} entries were rejected due to missing required elements: {', '.join(must_have)}.\n"
            if only:
                no_match_msg += f"• {rejection_reasons['only']} entries were rejected due to having elements outside the allowed set: {', '.join(only)}.\n"
            if not_these:
                no_match_msg += f"• {rejection_reasons['not_these']} entries were rejected due to containing excluded elements: {', '.join(not_these)}.\n"
                
            no_match_msg += "\nDatabase Hey classifications include: " + ", ".join(sorted(list(all_db_hey)[:10]))
                
            messagebox.showinfo("No Matches Found", no_match_msg)
            return
        
        # Extract wavenumbers, intensities, and labels
        all_spectra = []
        all_filenames = []
        all_metadata = []  # New: Store detailed metadata for each spectrum
        
        # Find the min and max wavenumber across all spectra to establish a common range
        progress_var.set(35)
        status_var.set("Analyzing wavenumber ranges...")
        progress_window.update()
        
        min_wavenumber = float('inf')
        max_wavenumber = float('-inf')
        wavenumber_counts = []
        
        # First pass to determine common wavenumber range
        for i, entry in enumerate(filtered):
            # Update progress periodically
            if i % 10 == 0:
                progress_var.set(35 + (i / len(filtered)) * 15)  # 35-50% progress
                status_var.set(f"Analyzing spectrum {i+1} of {len(filtered)}")
                progress_window.update()
                
            wavenumbers = np.array(entry['wavenumbers'])
            if len(wavenumbers) > 0:
                min_wavenumber = min(min_wavenumber, np.min(wavenumbers))
                max_wavenumber = max(max_wavenumber, np.max(wavenumbers))
                wavenumber_counts.append(len(wavenumbers))
        
        # Create common wavenumber scale - use the median number of points
        num_points = int(np.median(wavenumber_counts))
        common_wavenumbers = np.linspace(min_wavenumber, max_wavenumber, num_points)
        
        # Update progress
        progress_var.set(50)
        status_var.set(f"Interpolating {len(filtered)} spectra to common scale ({num_points} points)...")
        progress_window.update()
        
        # Second pass to interpolate intensities to common wavenumber scale
        failed_spectra = []
        for i, entry in enumerate(filtered):
            try:
                # Update progress periodically
                if i % 5 == 0:
                    progress_var.set(50 + (i / len(filtered)) * 40)  # 50-90% progress
                    status_var.set(f"Processing spectrum {i+1} of {len(filtered)}: {entry.get('name', 'Unknown')}")
                    progress_window.update()
                
                wavenumbers = np.array(entry['wavenumbers'])
                intensities = np.array(entry['intensities'])
                
                # Interpolate to common wavenumber scale
                if wavenumbers.size != common_wavenumbers.size or not np.array_equal(wavenumbers, common_wavenumbers):
                    interp_intensities = np.interp(
                        common_wavenumbers, 
                        wavenumbers, 
                        intensities,
                        left=0,   # Use 0 for values outside the original range
                        right=0
                    )
                else:
                    interp_intensities = intensities
                
                # Extract best sample label with priority order:
                # 1. Mineral Name from metadata
                # 2. Name/ID from metadata
                # 3. name field in entry
                # 4. filename field
                # 5. default to "Unknown"
                sample_name = "Unknown"
                sample_metadata = {}
                
                # Check for metadata
                if 'metadata' in entry and isinstance(entry['metadata'], dict):
                    metadata = entry['metadata']
                    # First priority: NAME (based on debug output)
                    if 'NAME' in metadata and metadata['NAME']:
                        sample_name = metadata['NAME']
                    # Second priority: Mineral Name
                    elif 'MINERAL NAME' in metadata and metadata['MINERAL NAME']:
                        sample_name = metadata['MINERAL NAME']
                    # Third priority: Name/ID
                    elif 'NAME/ID' in metadata and metadata['NAME/ID']:
                        sample_name = metadata['NAME/ID']
                    
                    # Create case-insensitive lookup for metadata
                    metadata_lower = {k.lower(): k for k in metadata.keys()}
                    
                    # Check for lowercase variants if needed
                    if sample_name == "Unknown":
                        if 'name' in metadata_lower:
                            actual_key = metadata_lower['name']
                            sample_name = metadata[actual_key]
                        elif 'mineral name' in metadata_lower:
                            actual_key = metadata_lower['mineral name']
                            sample_name = metadata[actual_key]
                        elif 'name/id' in metadata_lower:
                            actual_key = metadata_lower['name/id']
                            sample_name = metadata[actual_key]
                    
                    # Standardize metadata keys to uppercase
                    sample_metadata = {}
                    for key, value in metadata.items():
                        if isinstance(key, str):
                            # Use uppercase key for standard fields
                            upper_key = key.upper()
                            sample_metadata[upper_key] = value
                        else:
                            # Keep non-string keys as is
                            sample_metadata[key] = value
                else:
                    sample_metadata = {}
                
                # If not found in metadata, check other locations
                if sample_name == "Unknown":
                    # Third priority: name field
                    if 'name' in entry and entry['name']:
                        sample_name = entry['name']
                    # Fourth priority: filename
                    elif 'filename' in entry and entry['filename']:
                        sample_name = entry['filename']
                
                # Store the data
                all_spectra.append(interp_intensities)
                all_filenames.append(sample_name)
                all_metadata.append(sample_metadata)
                
            except Exception as e:
                failed_spectra.append((entry.get('name', 'Unknown'), str(e)))
                continue
        
        if not all_spectra:
            progress_window.destroy()
            messagebox.showinfo("Info", "No valid spectra could be processed.")
            return
        
        # Stack the interpolated spectra
        progress_var.set(90)
        status_var.set("Combining spectra and extracting features...")
        progress_window.update()
        
        all_intensities = np.vstack(all_spectra)
        
        # Extract features from standardized data
        features = self.extract_vibrational_features(all_intensities, common_wavenumbers)
        
        # Update cluster data with enhanced metadata
        self.cluster_data = {
            'wavenumbers': common_wavenumbers,
            'intensities': all_intensities,
            'features': features,
            'features_scaled': None,
            'labels': None,
            'linkage_matrix': None,
            'distance_matrix': None,
            'sample_labels': all_filenames,
            'sample_metadata': all_metadata  # New: Store detailed metadata for tooltips
        }
        
        # Debug: Print sample of metadata to check structure
        print("\nDEBUG: Sample metadata structure:")
        for i in range(min(3, len(all_metadata))):
            print(f"\nSample {i} - Filename: {all_filenames[i]}")
            print(f"Metadata keys: {list(all_metadata[i].keys() if all_metadata[i] else [])}")
            if all_metadata[i]:
                if 'MINERAL NAME' in all_metadata[i]:
                    print(f"MINERAL NAME: {all_metadata[i]['MINERAL NAME']}")
                if 'NAME/ID' in all_metadata[i]:
                    print(f"NAME/ID: {all_metadata[i]['NAME/ID']}")
                if 'HEY CLASSIFICATION' in all_metadata[i]:
                    print(f"HEY CLASSIFICATION: {all_metadata[i]['HEY CLASSIFICATION']}")
        
        # Update preview if method exists
        try:
            if hasattr(self, 'update_preview'):
                self.update_preview()
        except Exception as e:
            print(f"Warning: Failed to update preview: {str(e)}")
        
        # Complete progress
        progress_var.set(100)
        status_var.set(f"Import complete: {len(all_spectra)} spectra successfully imported")
        progress_window.update()
        
        # Close progress window after a short delay
        self.window.after(1000, progress_window.destroy)
        
        # Show warnings about failed spectra
        if failed_spectra:
            warning_msg = f"{len(failed_spectra)} spectra could not be processed:\n\n"
            for name, error in failed_spectra[:10]:  # Show first 10 failures
                warning_msg += f"• {name}: {error}\n"
            if len(failed_spectra) > 10:
                warning_msg += f"\n...and {len(failed_spectra) - 10} more."
            messagebox.showwarning("Import Warnings", warning_msg)
        
        # Update UI
        self.update_ui_after_import(len(all_spectra))

    def update_database_status(self):
        """Update the database status in the status bar."""
        db = self.get_database()
        if db:
            self.status_bar.config(text=f"Database loaded: {len(db)} entries")
        else:
            self.status_bar.config(text="No database loaded") 

    def load_hey_classifications_from_csv(self):
        """Load Hey Classifications directly from the RRUFF export CSV file."""
        try:
            import os
            import csv
            
            # Define possible file paths
            possible_paths = [
                "RRUFF_Export_with_Hey_Classification.csv",  # Current directory
                os.path.join(os.path.expanduser("~"), "RRUFF_Export_with_Hey_Classification.csv"),  # Home directory
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "RRUFF_Export_with_Hey_Classification.csv"),  # Script directory
            ]
            
            # Find the file
            csv_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
            
            if not csv_path:
                print("Could not find RRUFF_Export_with_Hey_Classification.csv")
                return set()
            
            # Read classifications from the CSV
            hey_classes = set()
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                # Skip header
                next(reader)
                
                # The Hey Classification is in the last column (21)
                for row in reader:
                    if len(row) >= 21:
                        # Get the classification name from the 21st column (0-indexed)
                        hey_class = row[20].strip()
                        if hey_class and hey_class != "Hey Classification Name":
                            hey_classes.add(hey_class)
            
            print(f"Loaded {len(hey_classes)} Hey Classifications from CSV file")
            print("Sample classifications:", list(hey_classes)[:10])
            return hey_classes
            
        except Exception as e:
            print(f"Error loading Hey Classifications from CSV: {str(e)}")
            return set()