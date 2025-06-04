#!/usr/bin/env python3
"""
Enhanced Mixed Mineral Analysis Module
User-guided analysis with NNLS fitting and peak intensity matching
"""

import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.patches as patches


class EnhancedMixedMineralAnalysis:
    """User-guided mixed mineral analysis with NNLS fitting and peak matching."""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.components = []  # List of component dictionaries
        self.component_weights = []  # Weights from NNLS fitting
        self.original_spectrum = None
        self.current_wavenumbers = None
        self.current_residual = None
        self.analysis_window = None
        self.roi_selector = None
        
    def launch_analysis(self):
        """Launch the user-guided mixed mineral analysis interface."""
        # Check for spectrum availability - prioritize processed spectrum
        spectrum_to_use = None
        spectrum_source = ""
        
        if (hasattr(self.parent_app.raman, 'processed_spectra') and 
            self.parent_app.raman.processed_spectra is not None and
            len(self.parent_app.raman.processed_spectra) > 0):
            spectrum_to_use = self.parent_app.raman.processed_spectra
            spectrum_source = "processed spectrum"
        elif (self.parent_app.raman.current_spectra is not None and
              len(self.parent_app.raman.current_spectra) > 0):
            spectrum_to_use = self.parent_app.raman.current_spectra
            spectrum_source = "raw spectrum"
        else:
            messagebox.showwarning("No Data", 
                "Please load and process a spectrum first.")
            return
        
        # Store original data
        self.original_spectrum = spectrum_to_use.copy()
        self.current_wavenumbers = self.parent_app.raman.current_wavenumbers.copy()
        self.current_residual = self.original_spectrum.copy()
        self.components = []
        self.component_weights = []
        
        # Create main analysis window
        self.analysis_window = tk.Toplevel(self.parent_app.root)
        self.analysis_window.title(f"Mixed Mineral Analysis - Peak Intensity Matching ({spectrum_source})")
        self.analysis_window.geometry("1400x900")
        
        self._create_analysis_interface()
        
        # Start with manual mineral selection
        self._start_manual_selection()
    
    def _create_analysis_interface(self):
        """Create the main analysis interface."""
        # Main paned window
        main_paned = ttk.PanedWindow(self.analysis_window, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel (controls)
        left_frame = ttk.Frame(main_paned, width=400)
        main_paned.add(left_frame, weight=0)
        
        # Right panel (plots)
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        # === LEFT PANEL ===
        
        # Workflow status
        status_frame = ttk.LabelFrame(left_frame, text="Analysis Workflow", padding=10)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.workflow_text = tk.Text(status_frame, height=4, wrap=tk.WORD)
        self.workflow_text.pack(fill=tk.X)
        self.workflow_text.insert(tk.END, "1. Select starting mineral\n2. Match peak intensities\n3. Search residual/ROI\n4. Refine contributions")
        self.workflow_text.config(state=tk.DISABLED)
        
        # Components list
        components_frame = ttk.LabelFrame(left_frame, text="Identified Components", padding=10)
        components_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Components treeview with weights
        columns = ("Mineral", "Weight", "Contribution%", "R²")
        self.components_tree = ttk.Treeview(components_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.components_tree.heading(col, text=col)
            if col == "Mineral":
                self.components_tree.column(col, width=150)
            else:
                self.components_tree.column(col, width=70)
        
        components_scrollbar = ttk.Scrollbar(components_frame, orient=tk.VERTICAL, 
                                           command=self.components_tree.yview)
        self.components_tree.configure(yscrollcommand=components_scrollbar.set)
        
        self.components_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        components_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Control buttons
        controls_frame = ttk.Frame(left_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Row 1 buttons
        controls_row1 = ttk.Frame(controls_frame)
        controls_row1.pack(fill=tk.X, pady=2)
        
        ttk.Button(controls_row1, text="Add Mineral", 
                  command=self._add_mineral_manual).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_row1, text="Manual Add", 
                  command=self._manual_add_known_mineral).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_row1, text="Auto-Suggest", 
                  command=self._auto_suggest_next_component).pack(side=tk.LEFT, padx=2)
        
        # Row 2 buttons  
        controls_row2 = ttk.Frame(controls_frame)
        controls_row2.pack(fill=tk.X, pady=2)
        
        ttk.Button(controls_row2, text="Search ROI", 
                  command=self._search_roi).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_row2, text="Clear ROI", 
                  command=self._clear_roi).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_row2, text="Refine NNLS", 
                  command=self._refine_nnls).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_row2, text="Remove", 
                  command=self._remove_component).pack(side=tk.LEFT, padx=2)
        
        # Row 3 buttons (new)
        controls_row3 = ttk.Frame(controls_frame)
        controls_row3.pack(fill=tk.X, pady=2)
        
        ttk.Button(controls_row3, text="Reorder", 
                  command=self._reorder_components).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_row3, text="Reset Analysis", 
                  command=self._reset_analysis).pack(side=tk.LEFT, padx=2)
        
        # Analysis summary
        summary_frame = ttk.LabelFrame(left_frame, text="Fit Quality", padding=10)
        summary_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.fit_quality_text = tk.Text(summary_frame, height=6, wrap=tk.WORD)
        self.fit_quality_text.pack(fill=tk.X)
        
        # === RIGHT PANEL - PLOTS ===
        
        # Create matplotlib figure with subplots
        self.fig = Figure(figsize=(12, 10), dpi=100)
        
        # Four subplots: Original+Model, Residual, Individual Components, Peak Analysis
        self.ax_main = self.fig.add_subplot(4, 1, 1)
        self.ax_residual = self.fig.add_subplot(4, 1, 2)
        self.ax_components = self.fig.add_subplot(4, 1, 3)
        self.ax_peaks = self.fig.add_subplot(4, 1, 4)
        
        self.fig.tight_layout(pad=3.0)
        
        # Create canvas with ROI selection capability
        self.canvas = FigureCanvasTkAgg(self.fig, right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, right_frame)
        toolbar.update()
        
        # Initialize plots
        self._initialize_plots()
        
        # Set up ROI selector
        self._setup_roi_selector()
    
    def _initialize_plots(self):
        """Initialize the plot displays."""
        # Main comparison plot
        self.ax_main.clear()
        self.ax_main.plot(self.current_wavenumbers, self.original_spectrum, 
                         'b-', label='Original Spectrum', linewidth=2)
        self.ax_main.set_ylabel('Intensity')
        self.ax_main.set_title('Original Spectrum vs Mixed Mineral Model')
        self.ax_main.legend()
        self.ax_main.grid(True, alpha=0.3)
        
        # Residual plot
        self.ax_residual.clear()
        self.ax_residual.plot(self.current_wavenumbers, self.current_residual, 
                             'g-', label='Current Residual', linewidth=1.5)
        self.ax_residual.set_ylabel('Intensity')
        self.ax_residual.set_title('Residual (Original - Model) - Click and drag to select ROI')
        self.ax_residual.legend()
        self.ax_residual.grid(True, alpha=0.3)
        
        # Components plot
        self.ax_components.clear()
        self.ax_components.set_ylabel('Intensity')
        self.ax_components.set_title('Individual Weighted Components')
        self.ax_components.grid(True, alpha=0.3)
        
        # Peak analysis plot
        self.ax_peaks.clear()
        self.ax_peaks.set_ylabel('Intensity')
        self.ax_peaks.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax_peaks.set_title('Peak Intensity Analysis')
        self.ax_peaks.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def _setup_roi_selector(self):
        """Set up region of interest selector on residual plot."""
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None
        self.is_dragging = False
        
        def on_press(event):
            # Only respond to left mouse button on residual plot
            if event.inaxes == self.ax_residual and event.button == 1 and event.xdata is not None:
                self.roi_start = event.xdata
                self.roi_end = None
                self.is_dragging = True
                
                # Remove existing rectangle
                if self.roi_rect:
                    try:
                        self.roi_rect.remove()
                    except:
                        pass
                    self.roi_rect = None
                
                print(f"ROI selection started at {self.roi_start:.1f} cm⁻¹")
        
        def on_motion(event):
            # Only respond during active dragging on residual plot
            if (self.is_dragging and event.inaxes == self.ax_residual and 
                self.roi_start is not None and event.xdata is not None):
                
                # Remove previous rectangle
                if self.roi_rect:
                    try:
                        self.roi_rect.remove()
                    except:
                        pass
                
                # Calculate rectangle parameters
                current_pos = event.xdata
                x_start = min(self.roi_start, current_pos)
                width = abs(current_pos - self.roi_start)
                
                # Get current y limits for rectangle height
                y_min, y_max = self.ax_residual.get_ylim()
                
                # Create new rectangle
                self.roi_rect = patches.Rectangle(
                    (x_start, y_min),
                    width, 
                    y_max - y_min,
                    alpha=0.3, facecolor='yellow', edgecolor='orange', linewidth=2
                )
                self.ax_residual.add_patch(self.roi_rect)
                
                # Update canvas
                self.canvas.draw_idle()
        
        def on_release(event):
            # Only respond to left mouse button release
            if (self.is_dragging and event.button == 1 and event.inaxes == self.ax_residual and 
                self.roi_start is not None and event.xdata is not None):
                
                self.roi_end = event.xdata
                self.is_dragging = False
                
                # Ensure start < end
                if self.roi_start > self.roi_end:
                    self.roi_start, self.roi_end = self.roi_end, self.roi_start
                
                # Validate ROI size (must be at least 10 cm⁻¹ wide)
                roi_width = abs(self.roi_end - self.roi_start)
                if roi_width < 10:
                    print(f"ROI too small ({roi_width:.1f} cm⁻¹). Minimum width is 10 cm⁻¹")
                    self._clear_roi()
                    return
                
                print(f"ROI selected: {self.roi_start:.1f} - {self.roi_end:.1f} cm⁻¹ (width: {roi_width:.1f} cm⁻¹)")
                
                # Update the plot to show persistent ROI
                self._update_roi_display()
            elif self.is_dragging:
                # Release outside residual plot or invalid data - cancel selection
                self.is_dragging = False
                self._clear_roi()
        
        # Connect events
        self.canvas.mpl_connect('button_press_event', on_press)
        self.canvas.mpl_connect('motion_notify_event', on_motion)
        self.canvas.mpl_connect('button_release_event', on_release)
    
    def _clear_roi(self):
        """Clear the ROI selection."""
        if self.roi_rect:
            try:
                self.roi_rect.remove()
            except:
                pass
            self.roi_rect = None
        
        self.roi_start = None
        self.roi_end = None
        self.is_dragging = False
        self.canvas.draw_idle()
        print("ROI selection cleared")
    
    def _update_roi_display(self):
        """Update the ROI display to show the persistent selection."""
        if self.roi_start is not None and self.roi_end is not None:
            # Remove old rectangle
            if self.roi_rect:
                try:
                    self.roi_rect.remove()
                except:
                    pass
            
            # Create persistent ROI rectangle
            y_min, y_max = self.ax_residual.get_ylim()
            self.roi_rect = patches.Rectangle(
                (self.roi_start, y_min),
                self.roi_end - self.roi_start,
                y_max - y_min,
                alpha=0.25, facecolor='yellow', edgecolor='orange', linewidth=2,
                linestyle='--'
            )
            self.ax_residual.add_patch(self.roi_rect)
            self.canvas.draw()
    
    def _start_manual_selection(self):
        """Start with manual mineral selection instead of automatic."""
        messagebox.showinfo("Mixed Mineral Analysis", 
            "Welcome to Peak Intensity Matching Analysis!\n\n"
            "1. First, select your starting mineral manually\n"
            "2. We'll match peak intensities using NNLS\n"
            "3. Then search residual regions for additional components\n"
            "4. Iteratively refine the fit\n\n"
            "Click 'Add Mineral' to begin.")
    
    def _add_mineral_manual(self):
        """Manual mineral selection with search assistance."""
        # Create mineral selection window
        selection_window = tk.Toplevel(self.analysis_window)
        selection_window.title("Select Mineral Component")
        selection_window.geometry("800x600")
        
        # Main frame
        main_frame = ttk.Frame(selection_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Search frame
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(search_frame, text="Search minerals:").pack(side=tk.LEFT)
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=search_var, width=20)
        search_entry.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Button(search_frame, text="Auto-Search", 
                  command=lambda: self._simple_auto_search(selection_window)).pack(side=tk.LEFT, padx=5)
        
        # Two-panel layout
        panels_frame = ttk.Frame(main_frame)
        panels_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left: mineral list
        list_frame = ttk.LabelFrame(panels_frame, text="Available Minerals", padding=5)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        mineral_listbox = tk.Listbox(list_frame, height=20)
        list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=mineral_listbox.yview)
        mineral_listbox.configure(yscrollcommand=list_scrollbar.set)
        
        mineral_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right: preview
        preview_frame = ttk.LabelFrame(panels_frame, text="Preview", padding=5)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Preview plot
        preview_fig = Figure(figsize=(6, 8), dpi=80)
        preview_ax = preview_fig.add_subplot(111)
        preview_canvas = FigureCanvasTkAgg(preview_fig, preview_frame)
        preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Populate mineral list
        all_minerals = []
        for name in self.parent_app.raman.database.keys():
            display_name = self._get_display_name(name)
            all_minerals.append((name, display_name))
        all_minerals.sort(key=lambda x: x[1])
        
        def update_mineral_list(filter_text=""):
            mineral_listbox.delete(0, tk.END)
            for name, display_name in all_minerals:
                if filter_text.lower() in display_name.lower():
                    mineral_listbox.insert(tk.END, display_name)
        
        def on_search_change(*args):
            update_mineral_list(search_var.get())
        
        def on_mineral_select(event):
            """Preview selected mineral."""
            selection = mineral_listbox.curselection()
            if selection:
                selected_display = mineral_listbox.get(selection[0])
                # Find actual name
                selected_name = None
                for name, display in all_minerals:
                    if display == selected_display:
                        selected_name = name
                        break
                
                if selected_name:
                    self._show_mineral_preview(selected_name, preview_ax, preview_canvas)
        
        def select_mineral():
            """Select the chosen mineral."""
            selection = mineral_listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a mineral.")
                return
                
            selected_display = mineral_listbox.get(selection[0])
            selected_name = None
            for name, display in all_minerals:
                if display == selected_display:
                    selected_name = name
                    break
            
            if selected_name:
                selection_window.destroy()
                self._add_component_with_nnls(selected_name)
        
        search_var.trace('w', on_search_change)
        mineral_listbox.bind('<<ListboxSelect>>', on_mineral_select)
        mineral_listbox.bind('<Double-Button-1>', lambda e: select_mineral())
        
        update_mineral_list()
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Select Mineral", command=select_mineral).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=selection_window.destroy).pack(side=tk.LEFT, padx=5)
        
        search_entry.focus()
    
    def _simple_auto_search(self, parent_window):
        """Simple auto-search for manual add dialog."""
        try:
            # Quick correlation-based search with current residual
            matches = []
            
            for mineral_name, mineral_data in self.parent_app.raman.database.items():
                # Skip already added minerals
                if any(comp['name'] == mineral_name for comp in self.components):
                    continue
                    
                # Get database spectrum
                db_wavenumbers = mineral_data['wavenumbers']
                db_intensities = mineral_data['intensities']
                
                # Interpolate to match current wavenumbers
                db_spectrum = np.interp(self.current_wavenumbers, db_wavenumbers, db_intensities)
                
                # Calculate correlation with current residual
                correlation = np.corrcoef(self.current_residual, db_spectrum)[0, 1]
                
                if not np.isnan(correlation) and correlation > 0.1:
                    matches.append((mineral_name, abs(correlation)))
            
            # Sort and show results
            matches.sort(key=lambda x: x[1], reverse=True)
            
            if matches:
                result_text = "Top suggestions based on residual:\n\n"
                for i, (name, score) in enumerate(matches[:5]):
                    result_text += f"{i+1}. {self._get_display_name(name)} (Score: {score:.3f})\n"
                
                messagebox.showinfo("Auto-Search Results", result_text)
            else:
                messagebox.showinfo("No Suggestions", "No good matches found.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error in simple auto-search: {str(e)}")
    
    def _manual_add_known_mineral(self):
        """Manual addition of a known mineral without search."""
        # Create simple selection window
        selection_window = tk.Toplevel(self.analysis_window)
        selection_window.title("Manual Add Known Mineral")
        selection_window.geometry("600x500")
        
        # Main frame
        main_frame = ttk.Frame(selection_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Instructions
        instructions = ttk.Label(main_frame, 
            text="Select a mineral you know is present in your sample.\n"
                 "This bypasses search and adds the mineral directly for NNLS fitting.",
            font=("TkDefaultFont", 10), justify=tk.CENTER)
        instructions.pack(pady=(0, 10))
        
        # Search frame
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(search_frame, text="Filter minerals:").pack(side=tk.LEFT)
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=search_var, width=30)
        search_entry.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
        
        # Mineral list
        list_frame = ttk.LabelFrame(main_frame, text="Available Minerals", padding=5)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        mineral_listbox = tk.Listbox(list_frame, height=15)
        list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=mineral_listbox.yview)
        mineral_listbox.configure(yscrollcommand=list_scrollbar.set)
        
        mineral_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate mineral list
        all_minerals = []
        for name in self.parent_app.raman.database.keys():
            display_name = self._get_display_name(name)
            all_minerals.append((name, display_name))
        all_minerals.sort(key=lambda x: x[1])
        
        def update_mineral_list(filter_text=""):
            mineral_listbox.delete(0, tk.END)
            for name, display_name in all_minerals:
                if filter_text.lower() in display_name.lower():
                    # Check if already added
                    already_added = any(comp['name'] == name for comp in self.components)
                    status = " (Already added)" if already_added else ""
                    mineral_listbox.insert(tk.END, f"{display_name}{status}")
        
        def on_search_change(*args):
            update_mineral_list(search_var.get())
        
        def add_selected_mineral():
            selection = mineral_listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a mineral.")
                return
                
            selected_display = mineral_listbox.get(selection[0])
            
            # Remove status suffix if present
            if " (Already added)" in selected_display:
                messagebox.showwarning("Already Added", "This mineral is already in the analysis.")
                return
                
            selected_display = selected_display.replace(" (Already added)", "")
            
            # Find actual name
            selected_name = None
            for name, display in all_minerals:
                if display == selected_display:
                    selected_name = name
                    break
            
            if selected_name:
                selection_window.destroy()
                print(f"Manually adding known mineral: {self._get_display_name(selected_name)}")
                self._add_component_with_nnls(selected_name)
        
        search_var.trace('w', on_search_change)
        mineral_listbox.bind('<Double-Button-1>', lambda e: add_selected_mineral())
        
        update_mineral_list()
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Add Selected Mineral", 
                  command=add_selected_mineral).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", 
                  command=selection_window.destroy).pack(side=tk.LEFT, padx=5)
        
        search_entry.focus()
    
    def _auto_suggest_next_component(self):
        """Auto-suggest next component based on positive residuals."""
        try:
            print(f"\n=== AUTO-SUGGEST NEXT COMPONENT ===")
            
            # Check if we have residuals to analyze
            if self.current_residual is None:
                messagebox.showwarning("No Residuals", "No residual spectrum available for analysis.")
                return
            
            # Find positive residuals (peaks above baseline that aren't explained)
            positive_residuals = np.where(self.current_residual > 0)[0]
            
            if len(positive_residuals) < 10:
                messagebox.showinfo("No Significant Residuals", 
                    "No significant positive residuals found. Current model may be complete.")
                return
            
            # Calculate weighted residual spectrum (emphasize positive peaks)
            weighted_residual = np.copy(self.current_residual)
            weighted_residual[weighted_residual < 0] = 0  # Zero out negative residuals
            
            # Boost significant positive peaks
            residual_max = np.max(weighted_residual)
            significant_threshold = residual_max * 0.1
            significant_peaks = weighted_residual > significant_threshold
            weighted_residual[significant_peaks] *= 2.0  # Boost significant peaks
            
            print(f"Analyzing {len(positive_residuals)} positive residual points")
            print(f"Max residual intensity: {residual_max:.1f}")
            print(f"Significant peaks above {significant_threshold:.1f}")
            
            # Search database using the weighted positive residuals
            matches = []
            
            for mineral_name, mineral_data in self.parent_app.raman.database.items():
                try:
                    # Skip already added minerals
                    if any(comp['name'] == mineral_name for comp in self.components):
                        continue
                    
                    # Get database spectrum data
                    db_wavenumbers = mineral_data['wavenumbers']
                    db_intensities = mineral_data['intensities']
                    
                    # Interpolate database spectrum to match current wavenumbers
                    db_spectrum = np.interp(self.current_wavenumbers, db_wavenumbers, db_intensities)
                    
                    # Normalize database spectrum
                    if np.max(db_spectrum) > 0:
                        db_spectrum = db_spectrum / np.max(db_spectrum) * residual_max
                    
                    # Calculate correlation with positive residuals only
                    positive_residual_values = weighted_residual[positive_residuals]
                    positive_db_values = db_spectrum[positive_residuals]
                    
                    if len(positive_residual_values) > 5 and np.std(positive_db_values) > 0:
                        correlation = np.corrcoef(positive_residual_values, positive_db_values)[0, 1]
                        
                        # Handle NaN correlations
                        if np.isnan(correlation):
                            correlation = 0.0
                        
                        # Store good correlations
                        if correlation > 0.2:  # Threshold for positive residual matching
                            matches.append((mineral_name, abs(correlation)))
                            
                except Exception as e:
                    print(f"Error processing {mineral_name}: {e}")
                    continue
            
            # Sort by correlation score
            matches.sort(key=lambda x: x[1], reverse=True)
            
            print(f"Found {len(matches) if matches else 0} potential matches for positive residuals")
            
            if matches:
                self._show_auto_suggest_results(matches[:10])  # Show top 10
            else:
                messagebox.showinfo("No Suggestions", 
                    "No good matches found for the positive residuals.\n"
                    "Try manual addition or ROI search for specific features.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error in auto-suggestion: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _show_auto_suggest_results(self, matches):
        """Show auto-suggested mineral matches."""
        suggestions_window = tk.Toplevel(self.analysis_window)
        suggestions_window.title("Auto-Suggested Next Components")
        suggestions_window.geometry("600x500")
        
        # Main frame
        main_frame = ttk.Frame(suggestions_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Instructions
        instructions = ttk.Label(main_frame, 
            text="These minerals best match the positive residuals (unexplained peaks) in your spectrum.\n"
                 "Select one to add to your analysis.",
            font=("TkDefaultFont", 10), justify=tk.CENTER)
        instructions.pack(pady=(0, 10))
        
        # Results list
        list_frame = ttk.LabelFrame(main_frame, text="Suggested Minerals", padding=5)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        listbox = tk.Listbox(list_frame, height=15)
        listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        for mineral_name, score in matches:
            display_name = self._get_display_name(mineral_name)
            listbox.insert(tk.END, f"{display_name} (Match: {score:.3f})")
        
        def add_suggested():
            selection = listbox.curselection()
            if selection:
                selected_mineral = matches[selection[0]][0]
                suggestions_window.destroy()
                print(f"Adding auto-suggested mineral: {self._get_display_name(selected_mineral)}")
                self._add_component_with_nnls(selected_mineral)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Add Selected", command=add_suggested).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=suggestions_window.destroy).pack(side=tk.LEFT, padx=5)
    
    def _show_mineral_preview(self, mineral_name, ax, canvas):
        """Show preview of mineral spectrum vs current residual."""
        try:
            mineral_data = self.parent_app.raman.database[mineral_name]
            mineral_intensities = mineral_data['intensities']
            mineral_wavenumbers = mineral_data['wavenumbers']
            
            # Interpolate to current wavenumbers
            interpolated = np.interp(self.current_wavenumbers, mineral_wavenumbers, mineral_intensities)
            
            ax.clear()
            ax.plot(self.current_wavenumbers, self.current_residual, 'b-', label='Current Residual', linewidth=1.5)
            
            # Normalize for display
            if np.max(interpolated) > 0:
                normalized = interpolated * (np.max(self.current_residual) / np.max(interpolated))
                ax.plot(self.current_wavenumbers, normalized, 'r-', label=f'{self._get_display_name(mineral_name)} (normalized)', linewidth=1.5, alpha=0.7)
            
            ax.set_xlabel('Wavenumber (cm⁻¹)')
            ax.set_ylabel('Intensity')
            ax.set_title('Preview Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            canvas.draw()
            
        except Exception as e:
            print(f"Error in preview: {e}")
    
    def _add_component_with_nnls(self, mineral_name):
        """Add component using NNLS fitting and peak intensity matching."""
        try:
            print(f"\n=== ADDING COMPONENT WITH NNLS: {mineral_name} ===")
            
            # Get mineral data
            mineral_data = self.parent_app.raman.database[mineral_name]
            mineral_intensities = mineral_data['intensities']
            mineral_wavenumbers = mineral_data['wavenumbers']
            
            # Interpolate to match current wavenumbers
            interpolated_spectrum = np.interp(
                self.current_wavenumbers, 
                mineral_wavenumbers, 
                mineral_intensities
            )
            
            # Add to components list
            component = {
                'name': mineral_name,
                'spectrum': interpolated_spectrum
            }
            self.components.append(component)
            
            # Perform NNLS fitting with all components
            self._perform_nnls_fitting()
            
            # Update displays
            self._update_components_display()
            self._update_plots()
            self._update_fit_quality()
            
            print(f"Component {mineral_name} added successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error adding component: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _perform_nnls_fitting(self):
        """Perform Non-Negative Least Squares fitting for all components."""
        try:
            from scipy.optimize import nnls
            
            if not self.components:
                return
            
            print(f"Performing NNLS fitting with {len(self.components)} components...")
            
            # Create design matrix
            A = np.column_stack([comp['spectrum'] for comp in self.components])
            b = self.original_spectrum
            
            # Perform NNLS
            weights, residual_norm = nnls(A, b)
            self.component_weights = weights
            
            # Update residual
            fitted_spectrum = np.dot(A, weights)
            self.current_residual = self.original_spectrum - fitted_spectrum
            
            # Calculate R²
            ss_res = np.sum(self.current_residual ** 2)
            ss_tot = np.sum((self.original_spectrum - np.mean(self.original_spectrum)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            print(f"NNLS Results:")
            for i, comp in enumerate(self.components):
                print(f"  {self._get_display_name(comp['name'])}: weight={weights[i]:.3f}")
            print(f"  Overall R²: {r_squared:.3f}")
            print(f"  Residual norm: {residual_norm:.1f}")
            
        except ImportError:
            messagebox.showerror("Error", "scipy is required for NNLS fitting. Please install scipy.")
        except Exception as e:
            print(f"Error in NNLS fitting: {e}")
            import traceback
            traceback.print_exc()
    
    def _search_roi(self):
        """Search for minerals in the selected ROI."""
        if self.roi_start is None or self.roi_end is None:
            messagebox.showwarning("No ROI", "Please select a region of interest first by clicking and dragging on the residual plot.")
            return
        
        try:
            # Create a mask for the ROI region
            roi_mask = (self.current_wavenumbers >= self.roi_start) & (self.current_wavenumbers <= self.roi_end)
            roi_points = np.sum(roi_mask)
            
            if roi_points < 5:
                messagebox.showwarning("ROI Too Small", f"ROI contains only {roi_points} data points. Please select a larger region.")
                return
            
            print(f"\n=== ROI SEARCH ===")
            print(f"ROI Region: {self.roi_start:.1f} - {self.roi_end:.1f} cm⁻¹")
            print(f"ROI contains {roi_points} data points out of {len(self.current_wavenumbers)} total")
            
            # Extract ROI portion
            roi_wavenumbers = self.current_wavenumbers[roi_mask]
            roi_residual = self.current_residual[roi_mask]
            
            print(f"ROI intensity range: {np.min(roi_residual):.1f} to {np.max(roi_residual):.1f}")
            print(f"ROI mean intensity: {np.mean(roi_residual):.1f}")
            
            # Check if ROI has significant signal
            roi_std = np.std(roi_residual)
            if roi_std < np.std(self.current_residual) * 0.1:
                response = messagebox.askyesno("Low Signal ROI", 
                    f"The selected ROI has very low signal variation (std={roi_std:.1f}).\n"
                    "This might not yield good search results.\n\n"
                    "Continue with search anyway?")
                if not response:
                    return
            
            # Perform ROI-specific search
            print(f"Performing ROI-specific database search...")
            matches = self._perform_roi_database_search(roi_wavenumbers, roi_residual)
            
            print(f"Found {len(matches) if matches else 0} potential matches")
            
            if matches:
                # Filter out already added components
                new_matches = []
                for mineral_name, score in matches:
                    already_added = any(comp['name'] == mineral_name for comp in self.components)
                    if not already_added:
                        new_matches.append((mineral_name, score))
                        print(f"  {self._get_display_name(mineral_name)}: {score:.3f}")
                    else:
                        print(f"  {self._get_display_name(mineral_name)}: {score:.3f} (already added)")
                
                if new_matches:
                    self._show_roi_matches(new_matches)
                else:
                    messagebox.showinfo("No New Matches", 
                        f"Found {len(matches)} matches in ROI, but all are already in the analysis.")
            else:
                messagebox.showinfo("No Matches", 
                    f"No matches found in ROI {self.roi_start:.1f} - {self.roi_end:.1f} cm⁻¹.\n"
                    f"Try selecting a region with stronger peaks or larger area.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error searching ROI: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _perform_roi_database_search(self, roi_wavenumbers, roi_spectrum):
        """Perform database search specifically for ROI region."""
        try:
            matches = []
            
            # Iterate through all minerals in database
            for mineral_name, mineral_data in self.parent_app.raman.database.items():
                try:
                    # Get database spectrum data
                    db_wavenumbers = mineral_data['wavenumbers']
                    db_intensities = mineral_data['intensities']
                    
                    # Interpolate database spectrum to match ROI wavenumber range
                    db_roi_spectrum = np.interp(roi_wavenumbers, db_wavenumbers, db_intensities)
                    
                    # Calculate correlation coefficient (simple and robust)
                    correlation = np.corrcoef(roi_spectrum, db_roi_spectrum)[0, 1]
                    
                    # Handle NaN correlations (constant spectra)
                    if np.isnan(correlation):
                        correlation = 0.0
                    
                    # Store positive correlations (similarity)
                    if correlation > 0.1:  # Minimum threshold
                        matches.append((mineral_name, abs(correlation)))
                        
                except Exception as e:
                    print(f"Error processing {mineral_name}: {e}")
                    continue
            
            # Sort by correlation score (highest first)
            matches.sort(key=lambda x: x[1], reverse=True)
            
            # Return top 15 matches
            return matches[:15]
            
        except Exception as e:
            print(f"Error in ROI database search: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _show_roi_matches(self, matches):
        """Show matches found in ROI."""
        roi_window = tk.Toplevel(self.analysis_window)
        roi_window.title(f"ROI Search Results ({self.roi_start:.1f} - {self.roi_end:.1f} cm⁻¹)")
        roi_window.geometry("500x400")
        
        ttk.Label(roi_window, text=f"Minerals found in ROI {self.roi_start:.1f} - {self.roi_end:.1f} cm⁻¹:", 
                 font=("TkDefaultFont", 10, "bold")).pack(pady=10)
        
        listbox = tk.Listbox(roi_window, height=15)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        for mineral_name, score in matches:
            display_name = self._get_display_name(mineral_name)
            # Check if already added
            already_added = any(comp['name'] == mineral_name for comp in self.components)
            status = " (Already added)" if already_added else ""
            listbox.insert(tk.END, f"{display_name} (Score: {score:.3f}){status}")
        
        def add_roi_match():
            selection = listbox.curselection()
            if selection:
                selected_mineral = matches[selection[0]][0]
                
                # Check if already added
                if any(comp['name'] == selected_mineral for comp in self.components):
                    messagebox.showwarning("Already Added", "This mineral is already in the analysis.")
                    return
                
                roi_window.destroy()
                self._add_component_with_nnls(selected_mineral)
        
        button_frame = ttk.Frame(roi_window)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Add Selected", command=add_roi_match).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=roi_window.destroy).pack(side=tk.LEFT, padx=5)
    
    def _refine_nnls(self):
        """Refine the NNLS fitting with enhanced diagnostics and options."""
        if not self.components:
            messagebox.showwarning("No Components", "Please add some components first.")
            return
        
        # Create refinement options window
        refine_window = tk.Toplevel(self.analysis_window)
        refine_window.title("NNLS Refinement Options")
        refine_window.geometry("500x500")  # Made taller to show all options
        
        main_frame = ttk.Frame(refine_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Instructions
        ttk.Label(main_frame, 
                 text="Choose refinement method. Area-Normalized and Weighted NNLS give the best\n"
                      "balance of realistic proportions and good fit quality!",
                 font=("TkDefaultFont", 10), justify=tk.CENTER).pack(pady=(0, 15))
        
        # Current results display
        current_frame = ttk.LabelFrame(main_frame, text="Current Results", padding=10)
        current_frame.pack(fill=tk.X, pady=(0, 15))
        
        result_text = tk.Text(current_frame, height=6, wrap=tk.WORD)
        result_text.pack(fill=tk.X)
        
        # Show current weights
        if len(self.component_weights) > 0:
            total_weight = sum(self.component_weights)
            summary = "Current NNLS weights:\n"
            for i, comp in enumerate(self.components):
                if i < len(self.component_weights):
                    weight = self.component_weights[i]
                    percentage = (weight / total_weight * 100) if total_weight > 0 else 0
                    summary += f"  {self._get_display_name(comp['name'])}: {weight:.3f} ({percentage:.1f}%)\n"
            
            # Calculate fit quality
            combined_model = np.zeros_like(self.original_spectrum)
            for i, component in enumerate(self.components):
                if i < len(self.component_weights):
                    combined_model += component['spectrum'] * self.component_weights[i]
            
            ss_res = np.sum((self.original_spectrum - combined_model) ** 2)
            ss_tot = np.sum((self.original_spectrum - np.mean(self.original_spectrum)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            summary += f"\nOverall R²: {r_squared:.3f}"
            result_text.insert(tk.END, summary)
        
        result_text.config(state=tk.DISABLED)
        
        # Refinement options
        options_frame = ttk.LabelFrame(main_frame, text="Refinement Options", padding=10)
        options_frame.pack(fill=tk.X, pady=(0, 15))
        
        def run_standard_nnls():
            refine_window.destroy()
            self._perform_nnls_fitting()
            self._update_displays()
            messagebox.showinfo("Refinement Complete", "Standard NNLS refinement completed.")
        
        def run_normalized_nnls():
            refine_window.destroy()
            self._perform_normalized_nnls()
            self._update_displays()
            messagebox.showinfo("Refinement Complete", "Normalized NNLS refinement completed.")
        
        def run_constrained_nnls():
            refine_window.destroy()
            self._show_constraint_dialog()
        
        def show_diagnostics():
            refine_window.destroy()
            self._show_fitting_diagnostics()
        
        def run_area_normalized_nnls():
            refine_window.destroy()
            self._perform_area_normalized_nnls()
            self._update_displays()
            messagebox.showinfo("Refinement Complete", "Area-Normalized NNLS refinement completed.")
        
        def run_weighted_nnls():
            refine_window.destroy()
            self._perform_weighted_nnls()
            self._update_displays()
            messagebox.showinfo("Refinement Complete", "Weighted NNLS refinement completed.")
        
        def run_scattering_aware_nnls():
            refine_window.destroy()
            self._show_scattering_strength_dialog()
        
        # BEST METHODS FIRST (reordered)
        ttk.Button(options_frame, text="🌟 Area-Normalized NNLS (RECOMMENDED)", 
                  command=run_area_normalized_nnls).pack(fill=tk.X, pady=2)
        ttk.Button(options_frame, text="🌟 Weighted NNLS (RECOMMENDED)", 
                  command=run_weighted_nnls).pack(fill=tk.X, pady=2)
        
        # Separator
        ttk.Separator(options_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # Advanced option for scattering strength correction
        ttk.Button(options_frame, text="⚙️ Scattering Strength Aware NNLS", 
                  command=run_scattering_aware_nnls).pack(fill=tk.X, pady=2)
        
        # Standard methods
        ttk.Button(options_frame, text="Standard NNLS", 
                  command=run_standard_nnls).pack(fill=tk.X, pady=2)
        ttk.Button(options_frame, text="Normalized NNLS (Equal Scaling)", 
                  command=run_normalized_nnls).pack(fill=tk.X, pady=2)
        ttk.Button(options_frame, text="Constrained NNLS (Set Limits)", 
                  command=run_constrained_nnls).pack(fill=tk.X, pady=2)
        ttk.Button(options_frame, text="Show Detailed Diagnostics", 
                  command=show_diagnostics).pack(fill=tk.X, pady=2)
        
        # Close button
        ttk.Button(main_frame, text="Cancel", command=refine_window.destroy).pack(pady=10)
    
    def _perform_normalized_nnls(self):
        """Perform NNLS with normalized spectra to handle intensity scale differences."""
        try:
            from scipy.optimize import nnls
            
            if not self.components:
                return
            
            print(f"\n=== NORMALIZED NNLS FITTING ===")
            print(f"Normalizing all spectra to same intensity scale...")
            
            # Normalize all component spectra to same scale
            normalized_components = []
            for comp in self.components:
                spectrum = comp['spectrum'].copy()
                max_intensity = np.max(spectrum)
                if max_intensity > 0:
                    spectrum = spectrum / max_intensity  # Normalize to max = 1
                normalized_components.append(spectrum)
            
            # Normalize original spectrum
            original_max = np.max(self.original_spectrum)
            normalized_original = self.original_spectrum / original_max if original_max > 0 else self.original_spectrum
            
            # Create design matrix with normalized spectra
            A = np.column_stack(normalized_components)
            b = normalized_original
            
            # Perform NNLS
            weights, residual_norm = nnls(A, b)
            
            # Scale weights back to account for normalization
            scaled_weights = []
            for i, comp in enumerate(self.components):
                original_max = np.max(comp['spectrum'])
                if original_max > 0:
                    scaled_weight = weights[i] * original_max
                else:
                    scaled_weight = weights[i]
                scaled_weights.append(scaled_weight)
            
            self.component_weights = np.array(scaled_weights)
            
            # PROPORTION-PRESERVING RESCALE STEP
            # Calculate current model
            fitted_spectrum = np.zeros_like(self.original_spectrum)
            for i, comp in enumerate(self.components):
                fitted_spectrum += comp['spectrum'] * self.component_weights[i]
            
            # Find optimal global scaling factor to match original intensity
            if np.sum(fitted_spectrum ** 2) > 0:
                optimal_scale = np.dot(self.original_spectrum, fitted_spectrum) / np.sum(fitted_spectrum ** 2)
                print(f"Applying proportion-preserving rescale factor: {optimal_scale:.3f}")
                
                # Apply scaling to all weights (preserves proportions)
                self.component_weights *= optimal_scale
                
                # Recalculate with scaled weights
                fitted_spectrum = np.zeros_like(self.original_spectrum)
                for i, comp in enumerate(self.components):
                    fitted_spectrum += comp['spectrum'] * self.component_weights[i]
            
            # Update residual with properly scaled model
            self.current_residual = self.original_spectrum - fitted_spectrum
            
            # Calculate R² with properly scaled model
            ss_res = np.sum(self.current_residual ** 2)
            ss_tot = np.sum((self.original_spectrum - np.mean(self.original_spectrum)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            print(f"Normalized NNLS Results:")
            for i, comp in enumerate(self.components):
                print(f"  {self._get_display_name(comp['name'])}: weight={self.component_weights[i]:.3f}")
            print(f"  Overall R²: {r_squared:.3f}")
            
        except ImportError:
            messagebox.showerror("Error", "scipy is required for NNLS fitting.")
        except Exception as e:
            print(f"Error in normalized NNLS fitting: {e}")
            import traceback
            traceback.print_exc()
    
    def _perform_area_normalized_nnls(self):
        """Perform NNLS with area-normalized spectra for better scaling."""
        try:
            from scipy.optimize import nnls
            
            if not self.components:
                return
            
            print(f"\n=== AREA-NORMALIZED NNLS FITTING ===")
            print(f"Normalizing all spectra by area under curve...")
            
            # Normalize all component spectra by area
            normalized_components = []
            original_areas = []
            
            for comp in self.components:
                spectrum = comp['spectrum'].copy()
                # Calculate area under curve (simple trapezoid integration)
                area = np.trapz(np.abs(spectrum), self.current_wavenumbers)
                original_areas.append(area)
                
                if area > 0:
                    spectrum = spectrum / area  # Normalize by area
                normalized_components.append(spectrum)
            
            # Normalize original spectrum by area
            original_area = np.trapz(np.abs(self.original_spectrum), self.current_wavenumbers)
            normalized_original = self.original_spectrum / original_area if original_area > 0 else self.original_spectrum
            
            # Create design matrix with normalized spectra
            A = np.column_stack(normalized_components)
            b = normalized_original
            
            # Perform NNLS
            weights, residual_norm = nnls(A, b)
            
            # Scale weights back based on area ratios
            scaled_weights = []
            for i, (comp, orig_area) in enumerate(zip(self.components, original_areas)):
                if orig_area > 0:
                    scaled_weight = weights[i] * orig_area / original_area * original_area
                else:
                    scaled_weight = weights[i]
                scaled_weights.append(scaled_weight)
            
            self.component_weights = np.array(scaled_weights)
            
            # PROPORTION-PRESERVING RESCALE STEP
            # Calculate current model
            fitted_spectrum = np.zeros_like(self.original_spectrum)
            for i, comp in enumerate(self.components):
                fitted_spectrum += comp['spectrum'] * self.component_weights[i]
            
            # Find optimal global scaling factor to match original intensity
            if np.sum(fitted_spectrum ** 2) > 0:
                optimal_scale = np.dot(self.original_spectrum, fitted_spectrum) / np.sum(fitted_spectrum ** 2)
                print(f"Applying proportion-preserving rescale factor: {optimal_scale:.3f}")
                
                # Apply scaling to all weights (preserves proportions)
                self.component_weights *= optimal_scale
                
                # Recalculate with scaled weights
                fitted_spectrum = np.zeros_like(self.original_spectrum)
                for i, comp in enumerate(self.components):
                    fitted_spectrum += comp['spectrum'] * self.component_weights[i]
            
            # Update residual with properly scaled model
            self.current_residual = self.original_spectrum - fitted_spectrum
            
            # Calculate R² with properly scaled model
            ss_res = np.sum(self.current_residual ** 2)
            ss_tot = np.sum((self.original_spectrum - np.mean(self.original_spectrum)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            print(f"Area-Normalized NNLS Results (after rescaling):")
            for i, comp in enumerate(self.components):
                print(f"  {self._get_display_name(comp['name'])}: weight={self.component_weights[i]:.3f}")
            print(f"  Overall R²: {r_squared:.3f}")
            print(f"  Proportions preserved with better intensity matching!")
            
        except ImportError:
            messagebox.showerror("Error", "scipy is required for NNLS fitting.")
        except Exception as e:
            print(f"Error in area-normalized NNLS fitting: {e}")
            import traceback
            traceback.print_exc()
    
    def _perform_weighted_nnls(self):
        """Perform weighted NNLS that balances fit quality with realistic proportions."""
        try:
            from scipy.optimize import nnls, minimize
            
            if not self.components:
                return
            
            print(f"\n=== WEIGHTED NNLS FITTING ===")
            print(f"Balancing fit quality with realistic proportions...")
            
            # Start with standard NNLS to get baseline
            A = np.column_stack([comp['spectrum'] for comp in self.components])
            b = self.original_spectrum
            
            standard_weights, _ = nnls(A, b)
            standard_fit = np.dot(A, standard_weights)
            standard_r2 = 1 - np.sum((b - standard_fit)**2) / np.sum((b - np.mean(b))**2)
            
            print(f"Standard NNLS R²: {standard_r2:.3f}")
            print(f"Standard weights: {[f'{w:.3f}' for w in standard_weights]}")
            
            # Try to balance by giving priority to components based on their relative contribution
            # Use weighted least squares approach
            
            # Calculate relative peak intensities as proxy for expected abundance
            peak_strengths = []
            for comp in self.components:
                # Find maximum correlation peaks with original
                correlation = np.corrcoef(comp['spectrum'], self.original_spectrum)[0, 1]
                peak_strength = np.max(comp['spectrum']) * max(0, correlation)
                peak_strengths.append(peak_strength)
            
            total_strength = sum(peak_strengths)
            expected_ratios = [s / total_strength for s in peak_strengths] if total_strength > 0 else [1.0/len(self.components)] * len(self.components)
            
            print(f"Expected ratios from peak analysis: {[f'{r:.3f}' for r in expected_ratios]}")
            
            # Define objective function that balances fit quality and realistic proportions
            def objective(weights):
                if np.any(weights < 0):
                    return 1e10  # Penalty for negative weights
                
                # Fit quality term
                fit = np.dot(A, weights)
                fit_error = np.sum((b - fit)**2)
                
                # Proportion realism term
                total_weight = np.sum(weights)
                if total_weight > 0:
                    actual_ratios = weights / total_weight
                    proportion_error = np.sum((actual_ratios - expected_ratios)**2)
                else:
                    proportion_error = 1e10
                
                # Combined objective (balance fit quality with proportion realism)
                alpha = 0.7  # Weight for fit quality vs proportion realism
                return alpha * fit_error / np.sum(b**2) + (1-alpha) * proportion_error * 10
            
            # Initial guess: blend of standard NNLS and expected ratios
            initial_guess = 0.7 * standard_weights + 0.3 * np.array(expected_ratios) * np.sum(standard_weights)
            
            # Bounds: non-negative weights
            bounds = [(0, None) for _ in self.components]
            
            # Optimize
            result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                self.component_weights = result.x
                
                # PROPORTION-PRESERVING RESCALE STEP
                # Calculate current model
                fitted_spectrum = np.dot(A, self.component_weights)
                
                # Find optimal global scaling factor to match original intensity
                if np.sum(fitted_spectrum ** 2) > 0:
                    optimal_scale = np.dot(self.original_spectrum, fitted_spectrum) / np.sum(fitted_spectrum ** 2)
                    print(f"Applying proportion-preserving rescale factor: {optimal_scale:.3f}")
                    
                    # Apply scaling to all weights (preserves proportions)
                    self.component_weights *= optimal_scale
                    
                    # Recalculate with scaled weights
                    fitted_spectrum = np.dot(A, self.component_weights)
                
                # Update residual with properly scaled model
                self.current_residual = self.original_spectrum - fitted_spectrum
                
                # Calculate final R² with properly scaled model
                ss_res = np.sum(self.current_residual ** 2)
                ss_tot = np.sum((self.original_spectrum - np.mean(self.original_spectrum)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                print(f"Weighted NNLS Results (after rescaling):")
                for i, comp in enumerate(self.components):
                    print(f"  {self._get_display_name(comp['name'])}: weight={self.component_weights[i]:.3f}")
                print(f"  Overall R²: {r_squared:.3f}")
                print(f"  Optimization successful with proportions preserved!")
                
            else:
                print(f"Optimization failed, falling back to standard NNLS")
                self.component_weights = standard_weights
                self.current_residual = self.original_spectrum - standard_fit
                
        except ImportError:
            messagebox.showerror("Error", "scipy is required for weighted NNLS fitting.")
        except Exception as e:
            print(f"Error in weighted NNLS fitting: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to standard NNLS
            self._perform_nnls_fitting()
    
    def _show_scattering_strength_dialog(self):
        """Show dialog for inputting relative Raman scattering strengths."""
        scattering_window = tk.Toplevel(self.analysis_window)
        scattering_window.title("Raman Scattering Strength Correction")
        scattering_window.geometry("700x600")
        
        main_frame = ttk.Frame(scattering_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Warning about limitations
        warning_frame = ttk.LabelFrame(main_frame, text="⚠️ Important Limitation", padding=10)
        warning_frame.pack(fill=tk.X, pady=(0, 15))
        
        warning_text = tk.Text(warning_frame, height=4, wrap=tk.WORD)
        warning_text.pack(fill=tk.X)
        warning_text.insert(tk.END, 
            "FUNDAMENTAL CHALLENGE: Different minerals have vastly different Raman scattering "
            "cross-sections. A 'weak scatterer' at 80% abundance might show less intensity than a "
            "'strong scatterer' at 5% abundance. Without calibration standards, quantitative analysis "
            "has inherent limitations. Use results as relative indicators, not absolute concentrations.")
        warning_text.config(state=tk.DISABLED)
        
        # Instructions
        ttk.Label(main_frame, 
                 text="If you know relative Raman scattering strengths for your minerals,\n"
                      "enter them below (1.0 = baseline, 0.1 = weak scatterer, 10.0 = strong scatterer):",
                 font=("TkDefaultFont", 10), justify=tk.CENTER).pack(pady=(0, 15))
        
        # Scattering strength entries
        strength_frame = ttk.LabelFrame(main_frame, text="Relative Scattering Strengths", padding=10)
        strength_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Headers
        headers_frame = ttk.Frame(strength_frame)
        headers_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(headers_frame, text="Mineral", width=25).pack(side=tk.LEFT)
        ttk.Label(headers_frame, text="Scattering Strength", width=15).pack(side=tk.LEFT, padx=10)
        ttk.Label(headers_frame, text="Notes", width=25).pack(side=tk.LEFT, padx=10)
        
        strength_entries = []
        
        # Common scattering strength estimates (rough guidelines)
        scattering_estimates = {
            'quartz': 2.0,
            'calcite': 3.0,
            'gypsum': 2.5,
            'sulfur': 5.0,
            'corundum': 1.5,
            'zorite': 1.0,  # Default baseline
            'forsterite': 0.8,
            'feldspar': 0.5,
            'magnetite': 0.3
        }
        
        for comp in self.components:
            row_frame = ttk.Frame(strength_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            mineral_name = comp['name'].lower()
            # Try to find a reasonable estimate
            estimated_strength = 1.0
            for key, value in scattering_estimates.items():
                if key in mineral_name:
                    estimated_strength = value
                    break
            
            ttk.Label(row_frame, text=self._get_display_name(comp['name']), width=25).pack(side=tk.LEFT)
            
            strength_var = tk.StringVar(value=f"{estimated_strength:.1f}")
            strength_entry = ttk.Entry(row_frame, textvariable=strength_var, width=12)
            strength_entry.pack(side=tk.LEFT, padx=10)
            
            # Note about the estimate
            note = "estimated" if estimated_strength != 1.0 else "baseline"
            ttk.Label(row_frame, text=f"({note})", width=25).pack(side=tk.LEFT, padx=10)
            
            strength_entries.append(strength_var)
        
        # Information section
        info_frame = ttk.LabelFrame(main_frame, text="Scattering Strength Guidelines", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 15))
        
        info_text = tk.Text(info_frame, height=8, wrap=tk.WORD)
        info_text.pack(fill=tk.X)
        info_text.insert(tk.END,
            "Typical relative strengths (very approximate):\n"
            "• Strong scatterers (3-5x): Sulfur, Calcite, Gypsum, Diamond\n"
            "• Moderate scatterers (1-2x): Quartz, Corundum, many silicates\n"
            "• Weak scatterers (0.3-0.8x): Feldspars, some oxides, metals\n"
            "• Very weak (0.1-0.3x): Some sulfides, native metals\n\n"
            "⚠️ These are rough estimates! Real values depend on crystal structure, "
            "orientation, laser wavelength, and measurement conditions.\n\n"
            "💡 FUTURE ENHANCEMENT: Database metadata for known scattering cross-sections "
            "should be implemented to provide automatic, literature-based corrections.")
        info_text.config(state=tk.DISABLED)
        
        def apply_scattering_correction():
            try:
                # Get scattering strengths
                strengths = []
                for strength_var in strength_entries:
                    strength = float(strength_var.get())
                    if strength <= 0:
                        strength = 1.0
                    strengths.append(strength)
                
                scattering_window.destroy()
                self._perform_scattering_aware_nnls(strengths)
                self._update_displays()
                messagebox.showinfo("Scattering Correction Applied", 
                    "NNLS performed with scattering strength correction.\n\n"
                    "Remember: Results are still approximate due to the fundamental "
                    "challenges of quantitative Raman spectroscopy.")
                
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numbers for scattering strengths.")
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Apply Correction", command=apply_scattering_correction).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=scattering_window.destroy).pack(side=tk.LEFT, padx=5)
    
    def _perform_scattering_aware_nnls(self, scattering_strengths):
        """Perform NNLS with Raman scattering strength correction."""
        try:
            from scipy.optimize import nnls
            
            if not self.components or len(scattering_strengths) != len(self.components):
                return
            
            print(f"\n=== SCATTERING STRENGTH AWARE NNLS ===")
            print(f"Applying relative scattering strength corrections...")
            
            # First, perform standard NNLS with original spectra
            A = np.column_stack([comp['spectrum'] for comp in self.components])
            b = self.original_spectrum
            
            # Perform NNLS to get "apparent" weights (influenced by scattering)
            apparent_weights, residual_norm = nnls(A, b)
            
            print(f"Apparent weights (before scattering correction):")
            for i, comp in enumerate(self.components):
                print(f"  {self._get_display_name(comp['name'])}: {apparent_weights[i]:.3f} (strength: {scattering_strengths[i]:.2f})")
            
            # Correct the weights for scattering strength
            # If scattering strength is high (strong scatterer), reduce the apparent weight
            # If scattering strength is low (weak scatterer), increase the apparent weight
            corrected_weights = []
            for i, (apparent_weight, strength) in enumerate(zip(apparent_weights, scattering_strengths)):
                # True concentration = apparent_weight / scattering_strength
                # Strong scatterers (strength > 1) will have reduced true concentration
                # Weak scatterers (strength < 1) will have increased true concentration
                corrected_weight = apparent_weight / strength
                corrected_weights.append(corrected_weight)
            
            self.component_weights = np.array(corrected_weights)
            
            # Calculate fitted spectrum using original spectra and corrected weights
            fitted_spectrum = np.zeros_like(self.original_spectrum)
            for i, comp in enumerate(self.components):
                fitted_spectrum += comp['spectrum'] * apparent_weights[i]  # Use apparent weights for display fit
            
            self.current_residual = self.original_spectrum - fitted_spectrum
            
            # Calculate better quality metrics for spectroscopic data
            # Correlation coefficient (more meaningful for spectral data)
            combined_model = np.zeros_like(self.original_spectrum)
            for i, component in enumerate(self.components):
                if i < len(self.component_weights):
                    combined_model += component['spectrum'] * self.component_weights[i]
            
            # Correlation coefficient (more meaningful for spectral data)
            correlation = np.corrcoef(self.original_spectrum, combined_model)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            # Relative RMSE (normalized by signal range)
            signal_range = np.max(self.original_spectrum) - np.min(self.original_spectrum)
            rmse = np.sqrt(np.mean((self.original_spectrum - combined_model) ** 2))
            relative_rmse = (rmse / signal_range * 100) if signal_range > 0 else 100
            
            # Mean Absolute Error
            mae = np.mean(np.abs(self.original_spectrum - combined_model))
            
            total_weight = sum(self.component_weights)
            
            quality_text = f"NNLS FIT QUALITY\n{'='*20}\n\n"
            quality_text += f"Correlation: {correlation:.3f}\n"
            quality_text += f"Relative RMSE: {relative_rmse:.1f}%\n"
            quality_text += f"Absolute RMSE: {rmse:.1f}\n"
            quality_text += f"MAE: {mae:.1f}\n"
            quality_text += f"Total Weight: {total_weight:.3f}\n\n"
            
            if correlation > 0.95:
                quality_text += "✅ Excellent correlation!"
            elif correlation > 0.85:
                quality_text += "✓ Good correlation"
            elif correlation > 0.7:
                quality_text += "⚠ Moderate correlation"
            else:
                quality_text += "❌ Poor correlation - try different components"
            
            self.fit_quality_text.insert(tk.END, quality_text)
            
            # Print console output
            print(f"\nScattering-Corrected Results:")
            total_corrected = sum(self.component_weights)
            for i, comp in enumerate(self.components):
                percentage = (self.component_weights[i] / total_corrected * 100) if total_corrected > 0 else 0
                print(f"  {self._get_display_name(comp['name'])}: {percentage:.1f}% (true abundance estimate)")
            
            print(f"\nFit Quality Metrics:")
            print(f"  Correlation: {correlation:.3f} (1.0 = perfect)")
            print(f"  Relative RMSE: {relative_rmse:.1f}% (lower is better)")
            print(f"  Percentages reflect estimated true abundance after scattering correction")
            
        except ImportError:
            messagebox.showerror("Error", "scipy is required for NNLS fitting.")
        except Exception as e:
            print(f"Error in scattering-aware NNLS fitting: {e}")
            import traceback
            traceback.print_exc()
    
    def _show_constraint_dialog(self):
        """Show dialog for setting weight constraints."""
        constraint_window = tk.Toplevel(self.analysis_window)
        constraint_window.title("Set Weight Constraints")
        constraint_window.geometry("600x400")
        
        main_frame = ttk.Frame(constraint_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, 
                 text="Set minimum and maximum weight constraints for each component.\n"
                      "Use this when you know approximate proportions.",
                 font=("TkDefaultFont", 10), justify=tk.CENTER).pack(pady=(0, 15))
        
        # Constraint entries
        constraint_frame = ttk.LabelFrame(main_frame, text="Weight Constraints", padding=10)
        constraint_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Headers
        headers_frame = ttk.Frame(constraint_frame)
        headers_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(headers_frame, text="Mineral", width=20).pack(side=tk.LEFT)
        ttk.Label(headers_frame, text="Min Weight", width=12).pack(side=tk.LEFT, padx=10)
        ttk.Label(headers_frame, text="Max Weight", width=12).pack(side=tk.LEFT, padx=10)
        ttk.Label(headers_frame, text="Expected %", width=12).pack(side=tk.LEFT, padx=10)
        
        constraint_entries = []
        for comp in self.components:
            row_frame = ttk.Frame(constraint_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(row_frame, text=self._get_display_name(comp['name']), width=20).pack(side=tk.LEFT)
            
            min_var = tk.StringVar(value="0.0")
            min_entry = ttk.Entry(row_frame, textvariable=min_var, width=10)
            min_entry.pack(side=tk.LEFT, padx=10)
            
            max_var = tk.StringVar(value="10.0")
            max_entry = ttk.Entry(row_frame, textvariable=max_var, width=10)
            max_entry.pack(side=tk.LEFT, padx=10)
            
            expected_var = tk.StringVar(value="")
            expected_entry = ttk.Entry(row_frame, textvariable=expected_var, width=10)
            expected_entry.pack(side=tk.LEFT, padx=10)
            
            constraint_entries.append((min_var, max_var, expected_var))
        
        def apply_constraints():
            # This would require a more sophisticated constrained optimization
            # For now, show a message about the limitation
            messagebox.showinfo("Feature Coming Soon", 
                "Constrained NNLS is being developed. For now, try:\n\n"
                "1. Normalized NNLS for scale issues\n"
                "2. Remove components that seem wrong\n"
                "3. Add components in order of expected abundance")
            constraint_window.destroy()
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Apply Constraints", command=apply_constraints).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=constraint_window.destroy).pack(side=tk.LEFT, padx=5)
    
    def _show_fitting_diagnostics(self):
        """Show detailed fitting diagnostics."""
        diag_window = tk.Toplevel(self.analysis_window)
        diag_window.title("Fitting Diagnostics")
        diag_window.geometry("800x600")
        
        main_frame = ttk.Frame(diag_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create diagnostic text
        diag_text = tk.Text(main_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=diag_text.yview)
        diag_text.configure(yscrollcommand=scrollbar.set)
        
        diag_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Generate diagnostics
        diagnostics = self._generate_fitting_diagnostics()
        diag_text.insert(tk.END, diagnostics)
        diag_text.config(state=tk.DISABLED)
        
        ttk.Button(main_frame, text="Close", command=diag_window.destroy).pack(pady=10)
    
    def _generate_fitting_diagnostics(self):
        """Generate detailed fitting diagnostics."""
        diagnostics = "=== DETAILED FITTING DIAGNOSTICS ===\n\n"
        
        if not self.components or len(self.component_weights) == 0:
            return diagnostics + "No components fitted yet.\n"
        
        # Overall statistics
        combined_model = np.zeros_like(self.original_spectrum)
        for i, component in enumerate(self.components):
            if i < len(self.component_weights):
                combined_model += component['spectrum'] * self.component_weights[i]
        
        ss_res = np.sum((self.original_spectrum - combined_model) ** 2)
        ss_tot = np.sum((self.original_spectrum - np.mean(self.original_spectrum)) ** 2)
        
        if ss_tot > 0:
            r_squared = 1 - (ss_res / ss_tot)
            # Ensure R² is not negative
            r_squared = max(0.0, r_squared)
        else:
            r_squared = 0.0
        
        rmse = np.sqrt(np.mean((self.original_spectrum - combined_model) ** 2))
        mae = np.mean(np.abs(self.original_spectrum - combined_model))
        
        total_weight = sum(self.component_weights)
        
        quality_text = f"NNLS FIT QUALITY\n{'='*20}\n\n"
        quality_text += f"Overall R²: {r_squared:.3f}\n"
        quality_text += f"RMSE: {rmse:.1f}\n"
        quality_text += f"MAE: {mae:.1f}\n"
        quality_text += f"Total Weight: {total_weight:.3f}\n\n"
        
        if r_squared > 0.8:
            quality_text += "✅ Excellent fit!"
        elif r_squared > 0.6:
            quality_text += "✓ Good fit"
        elif r_squared > 0.4:
            quality_text += "⚠ Moderate fit"
        else:
            quality_text += "❌ Poor fit - try different components"
        
        return diagnostics + quality_text
    
    def _update_displays(self):
        """Update all display elements."""
        self._update_components_display()
        self._update_plots()
        self._update_fit_quality()
    
    def _update_components_display(self):
        """Update the components treeview."""
        # Clear existing items
        for item in self.components_tree.get_children():
            self.components_tree.delete(item)
        
        # Add components with weights
        total_weight = sum(self.component_weights) if len(self.component_weights) > 0 else 1.0
        
        # Calculate overall model for individual R² calculations
        combined_model = np.zeros_like(self.original_spectrum)
        if len(self.component_weights) > 0:
            for i, component in enumerate(self.components):
                if i < len(self.component_weights):
                    combined_model += component['spectrum'] * self.component_weights[i]
        
        for i, component in enumerate(self.components):
            display_name = self._get_display_name(component['name'])
            weight = self.component_weights[i] if i < len(self.component_weights) else 0.0
            contribution = (weight / total_weight * 100) if total_weight > 0 else 0.0
            
            # Calculate individual R² - how well this component alone explains the original spectrum
            if len(self.component_weights) > 0 and weight > 0:
                # Individual component contribution
                individual_fit = component['spectrum'] * weight
                
                # Calculate R² for this component against original spectrum
                ss_res_individual = np.sum((self.original_spectrum - individual_fit) ** 2)
                ss_tot_original = np.sum((self.original_spectrum - np.mean(self.original_spectrum)) ** 2)
                
                if ss_tot_original > 0:
                    individual_r2 = 1 - (ss_res_individual / ss_tot_original)
                    # Ensure R² is not negative (means worse than mean)
                    individual_r2 = max(0.0, individual_r2)
                else:
                    individual_r2 = 0.0
            else:
                individual_r2 = 0.0
            
            self.components_tree.insert("", "end", values=(
                display_name,
                f"{weight:.3f}",
                f"{contribution:.1f}%",
                f"{individual_r2:.3f}"
            ))
    
    def _update_plots(self):
        """Update all plots with current analysis state."""
        # Clear axes
        self.ax_main.clear()
        self.ax_residual.clear()
        self.ax_components.clear()
        self.ax_peaks.clear()
        
        # Main plot: Original vs Model
        self.ax_main.plot(self.current_wavenumbers, self.original_spectrum, 
                         'b-', label='Original', linewidth=2)
        
        if self.components and len(self.component_weights) > 0:
            # Calculate combined model
            combined_model = np.zeros_like(self.original_spectrum)
            for i, component in enumerate(self.components):
                if i < len(self.component_weights):
                    combined_model += component['spectrum'] * self.component_weights[i]
            
            self.ax_main.plot(self.current_wavenumbers, combined_model, 
                             'r--', label='NNLS Model', linewidth=2)
            
            # Calculate correlation coefficient (more reliable than R² for spectral data)
            correlation = np.corrcoef(self.original_spectrum, combined_model)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            self.ax_main.set_title(f'Original vs NNLS Model (Correlation = {correlation:.3f})')
        else:
            self.ax_main.set_title('Original Spectrum')
            
        self.ax_main.set_ylabel('Intensity')
        self.ax_main.legend()
        self.ax_main.grid(True, alpha=0.3)
        
        # Residual plot with ROI highlight
        self.ax_residual.plot(self.current_wavenumbers, self.current_residual, 
                             'g-', label='Residual', linewidth=1.5)
        self.ax_residual.axhline(y=0, color='k', linestyle=':', alpha=0.5)
        
        self.ax_residual.set_ylabel('Intensity')
        self.ax_residual.set_title('Residual - Click and drag to select ROI for search')
        self.ax_residual.grid(True, alpha=0.3)
        
        # Restore ROI rectangle if it exists
        if self.roi_start is not None and self.roi_end is not None:
            y_min, y_max = self.ax_residual.get_ylim()
            self.roi_rect = patches.Rectangle(
                (self.roi_start, y_min),
                self.roi_end - self.roi_start,
                y_max - y_min,
                alpha=0.25, facecolor='yellow', edgecolor='orange', linewidth=2,
                linestyle='--', label='Selected ROI'
            )
            self.ax_residual.add_patch(self.roi_rect)
            self.ax_residual.legend()  # Update legend to include ROI
        else:
            self.ax_residual.legend()
        
        # Components plot
        colors = ['red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, component in enumerate(self.components):
            if i < len(self.component_weights):
                color = colors[i % len(colors)]
                weighted_spectrum = component['spectrum'] * self.component_weights[i]
                display_name = self._get_display_name(component['name'])
                
                self.ax_components.plot(self.current_wavenumbers, weighted_spectrum, 
                                       color=color, label=f"{display_name} ({self.component_weights[i]:.3f}x)",
                                       linewidth=1.5)
        
        self.ax_components.set_ylabel('Intensity')
        self.ax_components.set_title('Individual NNLS Weighted Components')
        if self.components:
            self.ax_components.legend()
        self.ax_components.grid(True, alpha=0.3)
        
        # Peak analysis plot
        self._plot_peak_analysis()
        
        self.canvas.draw()
    
    def _plot_peak_analysis(self):
        """Plot peak intensity analysis for verification."""
        try:
            from scipy.signal import find_peaks
            
            # Find peaks in original spectrum
            orig_peaks, orig_props = find_peaks(
                self.original_spectrum, 
                height=np.max(self.original_spectrum) * 0.1,
                prominence=np.max(self.original_spectrum) * 0.05
            )
            
            if len(orig_peaks) > 0:
                self.ax_peaks.plot(self.current_wavenumbers, self.original_spectrum, 'b-', label='Original', linewidth=1.5)
                self.ax_peaks.plot(self.current_wavenumbers[orig_peaks], self.original_spectrum[orig_peaks], 
                                  'bo', markersize=6, label='Original Peaks')
                
                # Plot fitted model if available
                if self.components and len(self.component_weights) > 0:
                    combined_model = np.zeros_like(self.original_spectrum)
                    for i, component in enumerate(self.components):
                        if i < len(self.component_weights):
                            combined_model += component['spectrum'] * self.component_weights[i]
                    
                    self.ax_peaks.plot(self.current_wavenumbers, combined_model, 'r--', label='NNLS Model', linewidth=2)
                    
                    # Find peaks in model
                    model_peaks, _ = find_peaks(
                        combined_model,
                        height=np.max(combined_model) * 0.1,
                        prominence=np.max(combined_model) * 0.05
                    )
                    
                    if len(model_peaks) > 0:
                        self.ax_peaks.plot(self.current_wavenumbers[model_peaks], combined_model[model_peaks], 
                                          'rs', markersize=6, label='Model Peaks')
                
                self.ax_peaks.set_xlabel('Wavenumber (cm⁻¹)')
                self.ax_peaks.set_ylabel('Intensity')
                self.ax_peaks.set_title('Peak Intensity Matching Verification')
                self.ax_peaks.legend()
                self.ax_peaks.grid(True, alpha=0.3)
                
        except Exception as e:
            print(f"Error in peak analysis plot: {e}")
    
    def _update_fit_quality(self):
        """Update fit quality display."""
        self.fit_quality_text.delete(1.0, tk.END)
        
        if not self.components or len(self.component_weights) == 0:
            self.fit_quality_text.insert(tk.END, "No components fitted yet.")
            return
        
        # Calculate better fit quality metrics for spectroscopic data
        combined_model = np.zeros_like(self.original_spectrum)
        for i, component in enumerate(self.components):
            if i < len(self.component_weights):
                combined_model += component['spectrum'] * self.component_weights[i]
        
        # Correlation coefficient (more meaningful for spectral data)
        correlation = np.corrcoef(self.original_spectrum, combined_model)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # Relative RMSE (normalized by signal range)
        signal_range = np.max(self.original_spectrum) - np.min(self.original_spectrum)
        rmse = np.sqrt(np.mean((self.original_spectrum - combined_model) ** 2))
        relative_rmse = (rmse / signal_range * 100) if signal_range > 0 else 100
        
        # Mean Absolute Error
        mae = np.mean(np.abs(self.original_spectrum - combined_model))
        
        total_weight = sum(self.component_weights)
        
        quality_text = f"NNLS FIT QUALITY\n{'='*20}\n\n"
        quality_text += f"Correlation: {correlation:.3f}\n"
        quality_text += f"Relative RMSE: {relative_rmse:.1f}%\n"
        quality_text += f"Absolute RMSE: {rmse:.1f}\n"
        quality_text += f"MAE: {mae:.1f}\n"
        quality_text += f"Total Weight: {total_weight:.3f}\n\n"
        
        if correlation > 0.95:
            quality_text += "✅ Excellent correlation!"
        elif correlation > 0.85:
            quality_text += "✓ Good correlation"
        elif correlation > 0.7:
            quality_text += "⚠ Moderate correlation"
        else:
            quality_text += "❌ Poor correlation - try different components"
        
        self.fit_quality_text.insert(tk.END, quality_text)
    
    def _get_display_name(self, mineral_name):
        """Get display name for a mineral."""
        if hasattr(self.parent_app, 'get_mineral_display_name'):
            return self.parent_app.get_mineral_display_name(mineral_name)
        return mineral_name

    def _remove_component(self):
        """Remove selected component."""
        selection = self.components_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a component to remove.")
            return
        
        item = selection[0]
        index = self.components_tree.index(item)
        
        if 0 <= index < len(self.components):
            removed = self.components.pop(index)
            
            # Re-fit with remaining components
            if self.components:
                self._perform_nnls_fitting()
            else:
                self.component_weights = []
                self.current_residual = self.original_spectrum.copy()
            
            self._update_displays()
            
            print(f"Removed component: {self._get_display_name(removed['name'])}")
    
    def _reorder_components(self):
        """Reorder components to improve fitting."""
        if len(self.components) < 2:
            messagebox.showinfo("Cannot Reorder", "Need at least 2 components to reorder.")
            return
        
        reorder_window = tk.Toplevel(self.analysis_window)
        reorder_window.title("Reorder Components")
        reorder_window.geometry("400x500")
        
        main_frame = ttk.Frame(reorder_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, 
                 text="Drag components to reorder. NNLS fitting order can affect results.\n"
                      "Generally, put more abundant components first.",
                 font=("TkDefaultFont", 10), justify=tk.CENTER).pack(pady=(0, 15))
        
        # Current order
        current_frame = ttk.LabelFrame(main_frame, text="Current Order", padding=10)
        current_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        listbox = tk.Listbox(current_frame, height=10)
        listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        for i, comp in enumerate(self.components):
            weight = self.component_weights[i] if i < len(self.component_weights) else 0.0
            listbox.insert(tk.END, f"{self._get_display_name(comp['name'])} (weight: {weight:.3f})")
        
        # Reorder buttons
        button_frame = ttk.Frame(current_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        def move_up():
            selection = listbox.curselection()
            if selection and selection[0] > 0:
                idx = selection[0]
                # Swap in components list
                self.components[idx], self.components[idx-1] = self.components[idx-1], self.components[idx]
                # Update display
                listbox.delete(0, tk.END)
                for i, comp in enumerate(self.components):
                    weight = self.component_weights[i] if i < len(self.component_weights) else 0.0
                    listbox.insert(tk.END, f"{self._get_display_name(comp['name'])} (weight: {weight:.3f})")
                listbox.selection_set(idx-1)
        
        def move_down():
            selection = listbox.curselection()
            if selection and selection[0] < len(self.components) - 1:
                idx = selection[0]
                # Swap in components list
                self.components[idx], self.components[idx+1] = self.components[idx+1], self.components[idx]
                # Update display
                listbox.delete(0, tk.END)
                for i, comp in enumerate(self.components):
                    weight = self.component_weights[i] if i < len(self.component_weights) else 0.0
                    listbox.insert(tk.END, f"{self._get_display_name(comp['name'])} (weight: {weight:.3f})")
                listbox.selection_set(idx+1)
        
        def apply_reorder():
            reorder_window.destroy()
            # Re-fit with new order
            self._perform_nnls_fitting()
            self._update_displays()
            messagebox.showinfo("Reorder Complete", "Components reordered and NNLS refitted.")
        
        ttk.Button(button_frame, text="Move Up", command=move_up).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Move Down", command=move_down).pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X)
        
        ttk.Button(action_frame, text="Apply New Order", command=apply_reorder).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Cancel", command=reorder_window.destroy).pack(side=tk.LEFT, padx=5)
    
    def _reset_analysis(self):
        """Reset the entire analysis."""
        result = messagebox.askyesno("Reset Analysis", 
            "This will remove all components and start over.\nAre you sure?")
        
        if result:
            self.components = []
            self.component_weights = []
            self.current_residual = self.original_spectrum.copy()
            self._clear_roi()
            self._update_displays()
            
            messagebox.showinfo("Analysis Reset", 
                "Analysis reset. You can now start adding components fresh.")


# Legacy compatibility functions
def main(parent_app):
    """Legacy function for backward compatibility."""
    analyzer = EnhancedMixedMineralAnalysis(parent_app)
    analyzer.launch_analysis()

# Old interface compatibility  
class EnhancedMixedMineralAnalysis_Legacy:
    """Legacy interface for old mixed mineral analysis calls."""
    
    def __init__(self, parent_app, window, selected_minerals, mineral_weights, 
                 selected_minerals_listbox, update_fit, current_wavenumbers):
        self.parent_app = parent_app
        
    def add_mineral_enhanced(self):
        """Redirect to new intelligent analysis."""
        analyzer = EnhancedMixedMineralAnalysis(self.parent_app)
        analyzer.launch_analysis() 