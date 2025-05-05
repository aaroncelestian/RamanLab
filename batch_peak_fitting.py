#!/usr/bin/env python3
# Batch Peak Fitting Module for ClaritySpectra
"""
Module for batch processing of Raman spectra with peak fitting.
Allows sequential refinement of peak positions, shapes, and backgrounds
across multiple spectra, with visualization of trends.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve
import os

class BatchPeakFittingWindow:
    """Window for batch processing of Raman spectra with peak fitting."""
    
    def __init__(self, parent, raman):
        """
        Initialize the batch peak fitting window.
        
        Parameters:
        -----------
        parent : tk.Tk or tk.Toplevel
            Parent window
        raman : RamanSpectra
            Reference to the RamanSpectra instance
        """
        self.window = tk.Toplevel(parent)
        self.window.title("Batch Peak Fitting")
        self.window.geometry("1400x800")
        self.window.minsize(1200, 700)
        
        # Store references
        self.parent = parent
        self.raman = raman
        
        # Initialize data storage
        self.spectra_files = []
        self.current_spectrum_index = 0
        self.batch_results = []
        self.reference_peaks = None
        self.reference_background = None
        
        # Variables for fitted peaks
        self.peaks = []
        self.fit_params = []
        self.fit_result = None
        self.background = None
        self.current_model = tk.StringVar(value="Gaussian")
        self.residuals = None
        self.show_fitted_peaks = tk.BooleanVar(value=True)
        self.show_individual_peaks = tk.BooleanVar(value=True)
        
        # Create GUI
        self.create_gui()
        
        # Set up window close event
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_gui(self):
        """Create the GUI elements."""
        # Main container
        main_container = ttk.Frame(self.window, padding=10)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # --- LEFT PANE: Replace with Notebook Tabs ---
        self.left_panel_width = 350
        self.left_panel_container = ttk.Frame(main_container, width=self.left_panel_width)
        self.left_panel_container.pack(side=tk.LEFT, fill=tk.Y)
        self.left_panel_container.pack_propagate(False)

        # Create notebook for tabs
        self.left_notebook = ttk.Notebook(self.left_panel_container)
        self.left_notebook.pack(fill=tk.BOTH, expand=True)

        # --- Tab 1: File Selection ---
        file_tab = ttk.Frame(self.left_notebook)
        self.left_notebook.add(file_tab, text="File Selection")

        file_frame = ttk.LabelFrame(file_tab, text="File Selection", padding=10)
        file_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        self.file_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=scrollbar.set)
        button_frame = ttk.Frame(file_frame)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Add Files", command=self.add_files).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(button_frame, text="Remove Selected", command=self.remove_selected_files).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # --- Tab 2: Peak Controls ---
        peak_tab = ttk.Frame(self.left_notebook)
        self.left_notebook.add(peak_tab, text="Peaks")

        controls_frame = ttk.LabelFrame(peak_tab, text="Peak Fitting Controls", padding=10)
        controls_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Background controls
        bg_frame = ttk.LabelFrame(controls_frame, text="Background", padding=5)
        bg_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bg_frame, text="λ (smoothness):").pack(anchor=tk.W)
        self.var_lambda = tk.StringVar(value="1e5")
        ttk.Entry(bg_frame, textvariable=self.var_lambda).pack(fill=tk.X, pady=2)
        ttk.Label(bg_frame, text="p (asymmetry):").pack(anchor=tk.W)
        self.var_p = tk.StringVar(value="0.01")
        ttk.Entry(bg_frame, textvariable=self.var_p).pack(fill=tk.X, pady=2)
        ttk.Button(bg_frame, text="Subtract Background", command=self.subtract_background).pack(fill=tk.X, pady=2)

        # Manual peak controls
        manual_frame = ttk.LabelFrame(controls_frame, text="Manual Peak Control", padding=5)
        manual_frame.pack(fill=tk.X, pady=2)
        
        # Add/Delete buttons
        button_frame = ttk.Frame(manual_frame)
        button_frame.pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Click to Add Peak", command=self.enable_peak_addition).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(button_frame, text="Delete Peak", command=self.show_peak_deletion_dialog).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Peak detection controls
        peak_frame = ttk.LabelFrame(controls_frame, text="Peak Detection", padding=5)
        peak_frame.pack(fill=tk.X, pady=2)
        ttk.Label(peak_frame, text="Height:").pack(anchor=tk.W)
        self.var_height = tk.StringVar(value="Auto")
        ttk.Entry(peak_frame, textvariable=self.var_height).pack(fill=tk.X, pady=2)
        ttk.Label(peak_frame, text="Distance:").pack(anchor=tk.W)
        self.var_distance = tk.StringVar(value="Auto")
        ttk.Entry(peak_frame, textvariable=self.var_distance).pack(fill=tk.X, pady=2)
        ttk.Label(peak_frame, text="Prominence:").pack(anchor=tk.W)
        self.var_prominence = tk.StringVar(value="Auto")
        ttk.Entry(peak_frame, textvariable=self.var_prominence).pack(fill=tk.X, pady=2)
        
        # Create a frame for the buttons
        peak_button_frame = ttk.Frame(peak_frame)
        peak_button_frame.pack(fill=tk.X, pady=2)
        ttk.Button(peak_button_frame, text="Find Peaks", command=self.find_peaks).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(peak_button_frame, text="Clear Peaks", command=self.clear_peaks).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Model selection
        model_frame = ttk.LabelFrame(controls_frame, text="Peak Model", padding=5)
        model_frame.pack(fill=tk.X, pady=2)
        self.current_model = tk.StringVar(value="Gaussian")
        model_combo = ttk.Combobox(model_frame, textvariable=self.current_model, 
                                 values=["Gaussian", "Lorentzian", "Pseudo-Voigt", "Asymmetric Voigt"])
        model_combo.pack(fill=tk.X, pady=2)
        ttk.Button(model_frame, text="Fit Peaks", command=self.fit_peaks).pack(fill=tk.X, pady=2)

        # In Peak Controls tab, add Fit Ranges entry
        fit_range_frame = ttk.LabelFrame(controls_frame, text="Fit Ranges (cm⁻¹)", padding=5)
        fit_range_frame.pack(fill=tk.X, pady=2)
        self.var_fit_ranges = tk.StringVar(value="")
        fit_range_entry = ttk.Entry(fit_range_frame, textvariable=self.var_fit_ranges)
        fit_range_entry.pack(fill=tk.X, pady=2)
        fit_range_entry.insert(0, "400-600,900-1100")  # Example default
        fit_range_entry_tooltip = ttk.Label(fit_range_frame, text="e.g. 400-600,900-1100", font=("", 8, "italic"), foreground="gray")
        fit_range_entry_tooltip.pack(anchor=tk.W)
        # Add Update ROI button
        ttk.Button(fit_range_frame, text="Update ROI", command=self.update_roi_regions).pack(fill=tk.X, pady=2)

        # --- Tab 3: Batch Processing ---
        batch_tab = ttk.Frame(self.left_notebook)
        self.left_notebook.add(batch_tab, text="Batch")

        batch_frame = ttk.LabelFrame(batch_tab, text="Batch Processing", padding=10)
        batch_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        ttk.Button(batch_frame, text="Set as Reference", command=self.set_reference).pack(fill=tk.X, pady=2)
        batch_buttons_row = ttk.Frame(batch_frame)
        batch_buttons_row.pack(fill=tk.X, pady=2)
        ttk.Button(batch_buttons_row, text="Apply to All", command=self.apply_to_all).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(batch_buttons_row, text="Stop", command=self.stop_batch).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(batch_frame, text="Export Results", command=self.export_results).pack(fill=tk.X, pady=2)

        # --- Tab 4: Peak Visibility ---
        visibility_tab = ttk.Frame(self.left_notebook)
        self.left_notebook.add(visibility_tab, text="Plots")

        visibility_frame = ttk.LabelFrame(visibility_tab, text="Peak Visibility Controls", padding=10)
        visibility_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create a frame for peak visibility checkboxes
        self.peak_visibility_frame = ttk.Frame(visibility_frame)
        self.peak_visibility_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialize peak visibility variables
        self.peak_visibility_vars = []
        
        # Add buttons to show/hide all peaks
        button_frame = ttk.Frame(visibility_frame)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Show All", command=self.show_all_peaks).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(button_frame, text="Hide All", command=self.hide_all_peaks).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # --- RIGHT PANEL: Visualization ---
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.viz_notebook = ttk.Notebook(right_panel)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True)

        # Current spectrum tab
        current_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(current_frame, text="Current Spectrum")
        self.fig_current, (self.ax1_current, self.ax2_current) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})
        self.canvas_current = FigureCanvasTkAgg(self.fig_current, master=current_frame)
        self.canvas_current.draw()
        self.canvas_current.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Add toolbar for current spectrum
        toolbar_frame = ttk.Frame(current_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar_current = NavigationToolbar2Tk(self.canvas_current, toolbar_frame)
        self.toolbar_current.update()

        # --- Navigation controls at the bottom of the plot ---
        nav_frame = ttk.Frame(current_frame)
        nav_frame.pack(fill=tk.X, pady=5, side=tk.BOTTOM)
        ttk.Button(nav_frame, text="<< First", command=lambda: self.navigate_spectrum(0)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(nav_frame, text="< Previous", command=lambda: self.navigate_spectrum(-1)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(nav_frame, text="Next >", command=lambda: self.navigate_spectrum(1)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(nav_frame, text="Last >>", command=lambda: self.navigate_spectrum(len(self.spectra_files)-1)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        # Current spectrum info label
        self.current_spectrum_label = ttk.Label(current_frame, text="No spectrum loaded")
        self.current_spectrum_label.pack(fill=tk.X, pady=2, side=tk.BOTTOM)

        # Trends tab
        trends_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(trends_frame, text="Plots")
        self.fig_trends, self.ax_trends = plt.subplots(figsize=(8, 6))
        self.canvas_trends = FigureCanvasTkAgg(self.fig_trends, master=trends_frame)
        self.canvas_trends.draw()
        self.canvas_trends.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_frame = ttk.Frame(trends_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar_trends = NavigationToolbar2Tk(self.canvas_trends, toolbar_frame)
        self.toolbar_trends.update()
    
    def add_files(self):
        """Add files to the batch processing list."""
        files = filedialog.askopenfilenames(
            title="Select Raman Spectra Files",
            filetypes=[("Text Files", "*.txt"), ("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if files:
            was_empty = len(self.spectra_files) == 0
            for file in files:
                if file not in self.spectra_files:
                    self.spectra_files.append(file)
                    self.file_listbox.insert(tk.END, os.path.basename(file))
            # Automatically load and plot the first file if this is the first time files are added
            if was_empty and self.spectra_files:
                self.load_spectrum(0)
    
    def remove_selected_files(self):
        """Remove selected files from the batch processing list."""
        selected_indices = self.file_listbox.curselection()
        for index in sorted(selected_indices, reverse=True):
            self.file_listbox.delete(index)
            del self.spectra_files[index]
        
        # Update current spectrum if needed
        if self.current_spectrum_index >= len(self.spectra_files):
            self.current_spectrum_index = len(self.spectra_files) - 1
            if self.current_spectrum_index >= 0:
                self.load_spectrum(self.current_spectrum_index)
            else:
                self.clear_plot()
    
    def navigate_spectrum(self, direction):
        """Navigate to the next or previous spectrum."""
        if not self.spectra_files:
            return
        
        if direction == 0:  # First
            new_index = 0
        elif direction == -1:  # Previous
            new_index = max(0, self.current_spectrum_index - 1)
        elif direction == 1:  # Next
            new_index = min(len(self.spectra_files) - 1, self.current_spectrum_index + 1)
        else:  # Last
            new_index = len(self.spectra_files) - 1
        
        if new_index != self.current_spectrum_index:
            self.load_spectrum(new_index)
    
    def load_spectrum(self, index):
        """Load a spectrum from the list."""
        if not self.spectra_files or index < 0 or index >= len(self.spectra_files):
            return
            
        try:
            # Load the spectrum data
            file_path = self.spectra_files[index]
            data = np.loadtxt(file_path)
            self.wavenumbers = data[:, 0]
            self.spectra = data[:, 1]
            self.original_spectra = np.copy(self.spectra)
            
            # Update current spectrum index and label
            self.current_spectrum_index = index
            self.current_spectrum_label.config(text=f"Current: {os.path.basename(file_path)}")
            
            # Initialize peak fitting variables
            self.peaks = []
            self.fit_params = []
            self.fit_result = None
            self.background = None
            self.residuals = None
            
            # Update the plot
            self.update_plot()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load spectrum: {str(e)}")
    
    def set_reference(self):
        """Set the current spectrum as the reference for batch processing."""
        if not hasattr(self, 'peaks') or not self.peaks:
            messagebox.showwarning("No Peaks", "Please detect and fit peaks first.")
            return
            
        self.reference_peaks = self.peaks.copy()
        self.reference_background = self.background.copy() if self.background is not None else None
        messagebox.showinfo("Reference Set", "Current spectrum set as reference for batch processing.")
    
    def apply_to_all(self):
        """Apply the current peak fitting parameters to all spectra."""
        if not self.reference_peaks:
            messagebox.showwarning("No Reference", "Please set a reference spectrum first.")
            return
        
        self._stop_batch = False  # Reset stop flag
        # Clear previous batch results
        self.batch_results = []
        # Store current index
        current_index = self.current_spectrum_index
        
        # Process all spectra
        for i in range(len(self.spectra_files)):
            if self._stop_batch:
                messagebox.showinfo("Stopped", "Batch processing was stopped by the user.")
                break
            self.load_spectrum(i)
            
            # Apply reference peaks
            self.peaks = self.reference_peaks.copy()
            
            # Subtract background using current spectrum's parameters
            try:
                # Get parameters
                lambda_val = float(self.var_lambda.get())
                p_val = float(self.var_p.get())
                
                # Calculate background
                L = len(self.spectra)
                D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
                w = np.ones(L)
                
                for j in range(10):
                    W = sparse.spdiags(w, 0, L, L)
                    Z = W + lambda_val * D.dot(D.transpose())
                    z = spsolve(Z, w*self.spectra)
                    w = p_val * (self.spectra > z) + (1-p_val) * (self.spectra < z)
                
                self.background = z
                self.spectra = self.original_spectra - self.background
            except Exception as e:
                messagebox.showerror("Error", f"Failed to subtract background for spectrum {i+1}: {str(e)}")
                continue
            
            # Fit peaks
            self.fit_peaks()
            
            # Store results
            self.batch_results.append({
                'file': self.spectra_files[i],
                'peaks': self.peaks.copy(),
                'fit_params': np.copy(self.fit_params) if self.fit_params is not None else None,
                'background': np.copy(self.background) if self.background is not None else None
            })
            
            # Update progress
            self.current_spectrum_label.config(text=f"Processing: {i+1}/{len(self.spectra_files)}")
            self.window.update()
        
        # Update peak visibility controls and trends plot
        self.update_peak_visibility_controls()
        self.update_trends_plot()
        
        if not self._stop_batch:
            messagebox.showinfo("Complete", "Batch processing completed.")
    
    def update_peak_visibility_controls(self):
        """Update the peak visibility checkboxes based on the current number of peaks."""
        # Clear existing checkboxes
        for widget in self.peak_visibility_frame.winfo_children():
            widget.destroy()
        self.peak_visibility_vars = []
        
        if not self.batch_results or len(self.batch_results) == 0:
            return
            
        # Get number of peaks
        model_type = self.current_model.get()
        if model_type == "Gaussian" or model_type == "Lorentzian":
            params_per_peak = 3
        elif model_type == "Pseudo-Voigt":
            params_per_peak = 4
        elif model_type == "Asymmetric Voigt":
            params_per_peak = 5
        else:
            params_per_peak = 3
            
        try:
            n_peaks = len(self.batch_results[0]['fit_params']) // params_per_peak
        except (KeyError, TypeError, AttributeError):
            return
        
        # Create checkboxes for each peak
        for i in range(n_peaks):
            var = tk.BooleanVar(value=True)  # Default to visible
            self.peak_visibility_vars.append(var)
            cb = ttk.Checkbutton(
                self.peak_visibility_frame,
                text=f"Peak {i+1}",
                variable=var,
                command=self.update_trends_plot
            )
            cb.pack(anchor=tk.W, pady=2)
    
    def show_all_peaks(self):
        """Show all peaks in the trends plot."""
        for var in self.peak_visibility_vars:
            var.set(True)
        self.update_trends_plot()
    
    def hide_all_peaks(self):
        """Hide all peaks in the trends plot."""
        for var in self.peak_visibility_vars:
            var.set(False)
        self.update_trends_plot()
    
    def update_trends_plot(self):
        """Update the trends plot with batch processing results."""
        if not self.batch_results:
            return
            
        # Clear the figure
        self.fig_trends.clear()
        
        # Get model type to determine number of parameters
        model_type = self.current_model.get()
        if model_type == "Gaussian" or model_type == "Lorentzian":
            params_per_peak = 3
            n_subplots = 3  # position, amplitude, width
        elif model_type == "Pseudo-Voigt":
            params_per_peak = 4
            n_subplots = 4  # position, amplitude, width, eta
        elif model_type == "Asymmetric Voigt":
            params_per_peak = 5
            n_subplots = 5  # position, amplitude, left width, right width, eta
        else:
            params_per_peak = 3
            n_subplots = 3
        
        # Create a 2x2 grid of subplots for up to 4 parameters, filling the space
        gs = self.fig_trends.add_gridspec(2, 2)

        # Assign subplots to fill the grid
        ax_pos = self.fig_trends.add_subplot(gs[0, 0])  # Peak positions
        ax_amp = self.fig_trends.add_subplot(gs[0, 1])  # Peak amplitudes
        ax_wid = self.fig_trends.add_subplot(gs[1, 0])  # Combined Widths (Left & Right)
        ax_eta = self.fig_trends.add_subplot(gs[1, 1])  # Eta parameter

        # Add a note about confidence intervals
        self.fig_trends.text(0.02, 0.02, 'Shaded regions show 95% confidence intervals',
                           fontsize=8, style='italic', alpha=0.7)
        
        # Extract data for plotting
        x = range(len(self.batch_results))
        peak_positions = []
        peak_amplitudes = []
        peak_widths = []
        peak_etas = []
        peak_widths_left = []
        peak_widths_right = []
        
        # Store confidence intervals
        pos_ci = []
        amp_ci = []
        wid_ci = []
        eta_ci = []
        wid_left_ci = []
        wid_right_ci = []
        
        for result in self.batch_results:
            if result.get('fit_params') is not None:
                # Extract parameters for each peak
                n_peaks = len(result['fit_params']) // params_per_peak
                for i in range(n_peaks):
                    start_idx = i * params_per_peak
                    amp = result['fit_params'][start_idx]
                    cen = result['fit_params'][start_idx+1]
                    
                    # Calculate 95% confidence intervals if covariance matrix is available
                    ci_factor = 1.96  # 95% confidence interval
                    
                    if model_type == "Asymmetric Voigt":
                        wid_left = result['fit_params'][start_idx+2]
                        wid_right = result['fit_params'][start_idx+3]
                        eta = result['fit_params'][start_idx+4]
                        
                        # Calculate confidence intervals if available
                        if result.get('fit_covariance') is not None:
                            amp_err = ci_factor * np.sqrt(result['fit_covariance'][start_idx, start_idx])
                            cen_err = ci_factor * np.sqrt(result['fit_covariance'][start_idx+1, start_idx+1])
                            wid_left_err = ci_factor * np.sqrt(result['fit_covariance'][start_idx+2, start_idx+2])
                            wid_right_err = ci_factor * np.sqrt(result['fit_covariance'][start_idx+3, start_idx+3])
                            eta_err = ci_factor * np.sqrt(result['fit_covariance'][start_idx+4, start_idx+4])
                        else:
                            # Default to small error if no covariance matrix
                            amp_err = cen_err = wid_left_err = wid_right_err = eta_err = 0.1
                        
                        peak_widths_left.append((x[len(peak_widths_left)//n_peaks], wid_left, i))
                        peak_widths_right.append((x[len(peak_widths_right)//n_peaks], wid_right, i))
                        peak_etas.append((x[len(peak_etas)//n_peaks], eta, i))
                        
                        wid_left_ci.append((x[len(wid_left_ci)//n_peaks], wid_left_err, i))
                        wid_right_ci.append((x[len(wid_right_ci)//n_peaks], wid_right_err, i))
                        eta_ci.append((x[len(eta_ci)//n_peaks], eta_err, i))
                    else:
                        wid = result['fit_params'][start_idx+2]
                        
                        # Calculate confidence intervals if available
                        if result.get('fit_covariance') is not None:
                            amp_err = ci_factor * np.sqrt(result['fit_covariance'][start_idx, start_idx])
                            cen_err = ci_factor * np.sqrt(result['fit_covariance'][start_idx+1, start_idx+1])
                            wid_err = ci_factor * np.sqrt(result['fit_covariance'][start_idx+2, start_idx+2])
                        else:
                            # Default to small error if no covariance matrix
                            amp_err = cen_err = wid_err = 0.1
                        
                        peak_widths.append((x[len(peak_widths)//n_peaks], wid, i))
                        wid_ci.append((x[len(wid_ci)//n_peaks], wid_err, i))
                        
                        if model_type == "Pseudo-Voigt":
                            eta = result['fit_params'][start_idx+3]
                            if result.get('fit_covariance') is not None:
                                eta_err = ci_factor * np.sqrt(result['fit_covariance'][start_idx+3, start_idx+3])
                            else:
                                eta_err = 0.1
                            peak_etas.append((x[len(peak_etas)//n_peaks], eta, i))
                            eta_ci.append((x[len(eta_ci)//n_peaks], eta_err, i))
                    
                    peak_positions.append((x[len(peak_positions)//n_peaks], cen, i))
                    peak_amplitudes.append((x[len(peak_amplitudes)//n_peaks], amp, i))
                    
                    pos_ci.append((x[len(pos_ci)//n_peaks], cen_err, i))
                    amp_ci.append((x[len(amp_ci)//n_peaks], amp_err, i))
        
        # Plot trends for each parameter
        if peak_positions:
            colors = plt.cm.tab10(np.linspace(0, 1, n_peaks))
            # Plot peak positions
            has_visible_peaks = False
            for i in range(n_peaks):
                if i < len(self.peak_visibility_vars) and self.peak_visibility_vars[i].get():
                    x_pos = [x for x, y, p in peak_positions if p == i]
                    y_pos = [y for x, y, p in peak_positions if p == i]
                    y_err = [y for x, y, p in pos_ci if p == i]
                    ax_pos.fill_between(x_pos, [y - err for y, err in zip(y_pos, y_err)], [y + err for y, err in zip(y_pos, y_err)], color=colors[i], alpha=0.2)
                    ax_pos.plot(x_pos, y_pos, 'o', color=colors[i], label=f'Peak {i+1}')
                    has_visible_peaks = True
            ax_pos.set_ylabel('Position (cm⁻¹)')
            ax_pos.grid(True, linestyle=':', color='gray', alpha=0.6)
            if has_visible_peaks:
                ax_pos.legend(loc='upper right', fontsize=8, frameon=True)

            # Plot peak amplitudes
            has_visible_peaks = False
            for i in range(n_peaks):
                if i < len(self.peak_visibility_vars) and self.peak_visibility_vars[i].get():
                    x_amp = [x for x, y, p in peak_amplitudes if p == i]
                    y_amp = [y for x, y, p in peak_amplitudes if p == i]
                    y_err = [y for x, y, p in amp_ci if p == i]
                    ax_amp.fill_between(x_amp, [y - err for y, err in zip(y_amp, y_err)], [y + err for y, err in zip(y_amp, y_err)], color=colors[i], alpha=0.2)
                    ax_amp.plot(x_amp, y_amp, 'o', color=colors[i], label=f'Peak {i+1}')
                    has_visible_peaks = True
            ax_amp.set_ylabel('Amplitude')
            ax_amp.grid(True, linestyle=':', color='gray', alpha=0.6)
            if has_visible_peaks:
                ax_amp.legend(loc='upper right', fontsize=8, frameon=True)

            # Plot widths (combined Left & Right for Asymmetric Voigt)
            if model_type == "Asymmetric Voigt":
                has_visible_peaks = False
                for i in range(n_peaks):
                    if i < len(self.peak_visibility_vars) and self.peak_visibility_vars[i].get():
                        # Left width
                        x_wid_left = [x for x, y, p in peak_widths_left if p == i]
                        y_wid_left = [y for x, y, p in peak_widths_left if p == i]
                        y_err_left = [y for x, y, p in wid_left_ci if p == i]
                        ax_wid.fill_between(x_wid_left, [y - err for y, err in zip(y_wid_left, y_err_left)], [y + err for y, err in zip(y_wid_left, y_err_left)], color=colors[i], alpha=0.15)
                        ax_wid.plot(x_wid_left, y_wid_left, 'o', color=colors[i], label=f'Peak {i+1} Left')
                        # Right width
                        x_wid_right = [x for x, y, p in peak_widths_right if p == i]
                        y_wid_right = [y for x, y, p in peak_widths_right if p == i]
                        y_err_right = [y for x, y, p in wid_right_ci if p == i]
                        ax_wid.fill_between(x_wid_right, [y - err for y, err in zip(y_wid_right, y_err_right)], [y + err for y, err in zip(y_wid_right, y_err_right)], color=colors[i], alpha=0.15)
                        ax_wid.plot(x_wid_right, y_wid_right, '^', color=colors[i], label=f'Peak {i+1} Right')
                        has_visible_peaks = True
                ax_wid.set_ylabel('Width (cm⁻¹)')
                ax_wid.set_xlabel('Spectrum Number')
                ax_wid.grid(True, linestyle=':', color='gray', alpha=0.6)
                if has_visible_peaks:
                    ax_wid.legend(loc='upper right', fontsize=8, frameon=True, ncol=2)
                # Eta
                has_visible_peaks = False
                for i in range(n_peaks):
                    if i < len(self.peak_visibility_vars) and self.peak_visibility_vars[i].get():
                        x_eta = [x for x, y, p in peak_etas if p == i]
                        y_eta = [y for x, y, p in peak_etas if p == i]
                        y_err = [y for x, y, p in eta_ci if p == i]
                        ax_eta.fill_between(x_eta, [y - err for y, err in zip(y_eta, y_err)], [y + err for y, err in zip(y_eta, y_err)], color=colors[i], alpha=0.2)
                        ax_eta.plot(x_eta, y_eta, 'o', color=colors[i], label=f'Peak {i+1}')
                        has_visible_peaks = True
                ax_eta.set_ylabel('Eta')
                ax_eta.set_xlabel('Spectrum Number')
                ax_eta.grid(True, linestyle=':', color='gray', alpha=0.6)
                if has_visible_peaks:
                    ax_eta.legend(loc='upper right', fontsize=8, frameon=True)
            else:
                # Width (other models)
                has_visible_peaks = False
                for i in range(n_peaks):
                    if i < len(self.peak_visibility_vars) and self.peak_visibility_vars[i].get():
                        x_wid = [x for x, y, p in peak_widths if p == i]
                        y_wid = [y for x, y, p in peak_widths if p == i]
                        y_err = [y for x, y, p in wid_ci if p == i]
                        ax_wid.fill_between(x_wid, [y - err for y, err in zip(y_wid, y_err)], [y + err for y, err in zip(y_wid, y_err)], color=colors[i], alpha=0.2)
                        ax_wid.plot(x_wid, y_wid, 'o', color=colors[i], label=f'Peak {i+1}')
                        has_visible_peaks = True
                ax_wid.set_ylabel('Width (cm⁻¹)')
                ax_wid.set_xlabel('Spectrum Number')
                ax_wid.grid(True, linestyle=':', color='gray', alpha=0.6)
                if has_visible_peaks:
                    ax_wid.legend(loc='upper right', fontsize=8, frameon=True)
                # Eta for Pseudo-Voigt
                if model_type == "Pseudo-Voigt":
                    has_visible_peaks = False
                    for i in range(n_peaks):
                        if i < len(self.peak_visibility_vars) and self.peak_visibility_vars[i].get():
                            x_eta = [x for x, y, p in peak_etas if p == i]
                            y_eta = [y for x, y, p in peak_etas if p == i]
                            y_err = [y for x, y, p in eta_ci if p == i]
                            ax_eta.fill_between(x_eta, [y - err for y, err in zip(y_eta, y_err)], [y + err for y, err in zip(y_eta, y_err)], color=colors[i], alpha=0.2)
                            ax_eta.plot(x_eta, y_eta, 'o', color=colors[i], label=f'Peak {i+1}')
                            has_visible_peaks = True
                    ax_eta.set_ylabel('Eta')
                    ax_eta.set_xlabel('Spectrum Number')
                    ax_eta.grid(True, linestyle=':', color='gray', alpha=0.6)
                    if has_visible_peaks:
                        ax_eta.legend(loc='upper right', fontsize=8, frameon=True)

        self.fig_trends.tight_layout(rect=[0, 0.03, 1, 0.97])
        self.fig_trends.subplots_adjust(wspace=0.2)
        self.canvas_trends.draw()
    
    def export_results(self):
        """Export the batch processing results to a CSV file."""
        if not self.batch_results:
            messagebox.showwarning("No Results", "No batch processing results to export.")
            return
            
        try:
            # Ask for file location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                title="Save Batch Results"
            )
            
            if not file_path:
                return
                
            # Prepare data for export
            data = []
            for result in self.batch_results:
                row = {'File': os.path.basename(result['file'])}
                
                if result['fit_params'] is not None:
                    # Get model type
                    model_type = self.current_model.get()
                    if model_type == "Gaussian" or model_type == "Lorentzian":
                        params_per_peak = 3
                    elif model_type == "Pseudo-Voigt":
                        params_per_peak = 4
                    elif model_type == "Asymmetric Voigt":
                        params_per_peak = 5
                    else:
                        params_per_peak = 3
                    
                    # Extract parameters for each peak
                    n_peaks = len(result['fit_params']) // params_per_peak
                    for i in range(n_peaks):
                        start_idx = i * params_per_peak
                        row[f'Peak_{i+1}_Position'] = result['fit_params'][start_idx+1]
                        row[f'Peak_{i+1}_Amplitude'] = result['fit_params'][start_idx]
                        row[f'Peak_{i+1}_Width'] = result['fit_params'][start_idx+2]
                        
                        if model_type == "Pseudo-Voigt":
                            row[f'Peak_{i+1}_Eta'] = result['fit_params'][start_idx+3]
                        elif model_type == "Asymmetric Voigt":
                            row[f'Peak_{i+1}_Width_Left'] = result['fit_params'][start_idx+2]
                            row[f'Peak_{i+1}_Width_Right'] = result['fit_params'][start_idx+3]
                            row[f'Peak_{i+1}_Eta'] = result['fit_params'][start_idx+4]
                
                data.append(row)
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            
            messagebox.showinfo("Export Complete", f"Results saved to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {str(e)}")
    
    def on_closing(self):
        """Handle window closing event."""
        self.window.destroy()
    
    def subtract_background(self):
        """Subtract background from the current spectrum using asymmetric least squares."""
        try:
            # Get parameters
            lambda_val = float(self.var_lambda.get())
            p_val = float(self.var_p.get())
            
            # Calculate background
            L = len(self.spectra)
            D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
            w = np.ones(L)
            
            for i in range(10):
                W = sparse.spdiags(w, 0, L, L)
                Z = W + lambda_val * D.dot(D.transpose())
                z = spsolve(Z, w*self.spectra)
                w = p_val * (self.spectra > z) + (1-p_val) * (self.spectra < z)
            
            self.background = z
            self.spectra = self.original_spectra - self.background
            
            # Update plot
            self.update_plot()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to subtract background: {str(e)}")
    
    def find_peaks(self):
        """Find peaks in the current spectrum."""
        try:
            # Get parameters
            height = self.var_height.get()
            distance = self.var_distance.get()
            prominence = self.var_prominence.get()
            
            # Convert parameters to float if not "Auto"
            height = float(height) if height != "Auto" else None
            distance = float(distance) if distance != "Auto" else None
            prominence = float(prominence) if prominence != "Auto" else None
            
            # Find peaks
            from scipy.signal import find_peaks
            peak_indices, properties = find_peaks(
                self.spectra, 
                height=height, 
                distance=distance,
                prominence=prominence
            )
            
            # Store peak positions and intensities
            self.peaks = []
            for idx in peak_indices:
                self.peaks.append({
                    'position': float(self.wavenumbers[int(idx)]),
                    'intensity': float(self.spectra[int(idx)]),
                    'index': int(idx)
                })
            
            # Update plot
            self.update_plot()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to find peaks: {str(e)}")
    
    def gaussian(self, x, a, x0, sigma):
        """Gaussian peak function."""
        return a * np.exp(-(x-x0)**2/(2*sigma**2))
    
    def lorentzian(self, x, a, x0, gamma):
        """Lorentzian peak function."""
        return a * gamma**2 / ((x-x0)**2 + gamma**2)
    
    def pseudo_voigt(self, x, a, x0, sigma, eta):
        """Pseudo-Voigt peak function."""
        return eta * self.lorentzian(x, a, x0, sigma) + (1-eta) * self.gaussian(x, a, x0, sigma)
    
    def asymmetric_voigt(self, x, amp, cen, wid_left, wid_right, eta=0.5):
        """
        Asymmetric Voigt function combining Gaussian and Lorentzian components
        with different widths on each side of the peak.
        
        Parameters:
        -----------
        x : array-like
            Wavenumber values
        amp : float
            Peak amplitude
        cen : float
            Peak center position
        wid_left : float
            Width parameter for the left side of the peak
        wid_right : float
            Width parameter for the right side of the peak
        eta : float, optional
            Mixing parameter between Gaussian and Lorentzian (0-1)
            Default is 0.5 (equal mix)
        
        Returns:
        --------
        array-like
            The asymmetric Voigt function evaluated at x
        """
        # Create separate arrays for left and right sides
        left_mask = x <= cen
        right_mask = ~left_mask
        
        # Initialize result array
        result = np.zeros_like(x)
        
        # Calculate Gaussian and Lorentzian components for left side
        g_left = np.exp(-((x[left_mask] - cen) / wid_left) ** 2)
        l_left = 1 / (1 + ((x[left_mask] - cen) / wid_left) ** 2)
        
        # Calculate Gaussian and Lorentzian components for right side
        g_right = np.exp(-((x[right_mask] - cen) / wid_right) ** 2)
        l_right = 1 / (1 + ((x[right_mask] - cen) / wid_right) ** 2)
        
        # Combine components using eta parameter
        result[left_mask] = amp * (eta * g_left + (1 - eta) * l_left)
        result[right_mask] = amp * (eta * g_right + (1 - eta) * l_right)
        
        return result
    
    def fit_peaks(self):
        """Fit peaks to the current spectrum using the selected model."""
        try:
            if not self.peaks:
                messagebox.showwarning("No Peaks", "Please detect peaks first.")
                return
            
            # Get model type
            model_type = self.current_model.get()
            
            # Select model function
            if model_type == "Gaussian":
                model_func = self.gaussian
                params_per_peak = 3
            elif model_type == "Lorentzian":
                model_func = self.lorentzian
                params_per_peak = 3
            elif model_type == "Pseudo-Voigt":
                model_func = self.pseudo_voigt
                params_per_peak = 4
            elif model_type == "Asymmetric Voigt":
                model_func = self.asymmetric_voigt
                params_per_peak = 5
            else:
                model_func = self.gaussian
                params_per_peak = 3
            
            # HARD CAP for Asymmetric Voigt
            if model_type == "Asymmetric Voigt" and len(self.peaks) > 8:
                messagebox.showerror("Too Many Peaks", "Asymmetric Voigt fitting is limited to 8 peaks for stability. Please reduce the number of detected peaks (e.g., by increasing the prominence or height threshold).")
                return

            # Parse fit ranges
            fit_ranges_str = self.var_fit_ranges.get().strip()
            mask = np.zeros_like(self.wavenumbers, dtype=bool)
            if fit_ranges_str:
                try:
                    for part in fit_ranges_str.split(','):
                        if '-' in part:
                            min_w, max_w = map(float, part.split('-'))
                            mask |= (self.wavenumbers >= min_w) & (self.wavenumbers <= max_w)
                    x_fit = self.wavenumbers[mask]
                    y_fit = self.spectra[mask]
                except Exception as e:
                    messagebox.showerror("Fit Range Error", f"Could not parse fit ranges: {fit_ranges_str}\nError: {e}")
                    return
            else:
                x_fit = self.wavenumbers
                y_fit = self.spectra
            
            # Prepare initial parameters
            initial_params = []
            bounds_lower = []
            bounds_upper = []
            
            for peak in self.peaks:
                # Initial amplitude
                initial_params.append(peak['intensity'])
                bounds_lower.append(0)
                bounds_upper.append(np.inf)
                
                # Initial position
                initial_params.append(peak['position'])
                bounds_lower.append(self.wavenumbers[0])
                bounds_upper.append(self.wavenumbers[-1])
                
                if model_type == "Gaussian" or model_type == "Lorentzian":
                    # Initial width
                    initial_params.append(10)  # Default width
                    bounds_lower.append(0.1)
                    bounds_upper.append(100)
                elif model_type == "Pseudo-Voigt":
                    # Initial width and eta
                    initial_params.append(10)  # Default width
                    initial_params.append(0.5)  # Default eta
                    bounds_lower.extend([0.1, 0])
                    bounds_upper.extend([100, 1])
                elif model_type == "Asymmetric Voigt":
                    # Initial left width, right width, eta
                    initial_params.append(10)  # Default left width (sigma_l)
                    initial_params.append(10)  # Default right width (sigma_r)
                    initial_params.append(0.5)  # Default eta
                    bounds_lower.extend([0.1, 0.1, 0])
                    bounds_upper.extend([100, 100, 1])

            # Check initial parameter count for Asymmetric Voigt
            if model_type == "Asymmetric Voigt" and len(initial_params) % 5 != 0:
                messagebox.showerror("Parameter Error", f"Initial parameter list for Asymmetric Voigt is not a multiple of 5 (got {len(initial_params)}). This indicates a bug or mismatch. Please clear peaks and try again.")
                return
            
            # Define combined model function
            def combined_model(x, *params):
                result = np.zeros_like(x)
                for i in range(0, len(params), params_per_peak):
                    peak_params = params[i:i+params_per_peak]
                    if len(peak_params) != params_per_peak:
                        print(f"Skipping peak at i={i}, expected {params_per_peak} params, got {len(peak_params)}: {peak_params}")
                        continue
                    try:
                        result += model_func(x, *peak_params)
                    except Exception as e:
                        print(f"Error in model_func at i={i} with params {peak_params}: {e}")
                        continue
                return result
            
            # Perform fit
            try:
                popt, pcov = curve_fit(combined_model, x_fit, y_fit,
                                     p0=initial_params,
                                     bounds=(bounds_lower, bounds_upper))
                # Robust check for Asymmetric Voigt
                if model_type == "Asymmetric Voigt" and len(popt) % 5 != 0:
                    messagebox.showerror("Fit Error", f"Fit failed: number of fit parameters ({len(popt)}) is not a multiple of 5. Try reducing the number of peaks or adjusting initial guesses.")
                    return
            except Exception as e:
                messagebox.showerror("Fit Error", f"Failed to fit peaks: {str(e)}")
                return
            
            # Store results
            self.fit_params = popt
            self.fit_result = combined_model(self.wavenumbers, *popt)
            self.residuals = self.spectra - self.fit_result
            
            # Update plot
            self.update_plot()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fit peaks: {str(e)}")
    
    def update_plot(self):
        """Update the current spectrum plot."""
        if not hasattr(self, 'wavenumbers') or not hasattr(self, 'spectra'):
            return
            
        # Clear plots
        self.ax1_current.clear()
        self.ax2_current.clear()
        
        # Set title to current filename
        if self.current_spectrum_index >= 0 and self.current_spectrum_index < len(self.spectra_files):
            filename = os.path.basename(self.spectra_files[self.current_spectrum_index])
            self.ax1_current.set_title(f"Current Spectrum: {filename}")
        
        # --- Highlight fit regions ---
        fit_ranges_str = getattr(self, 'var_fit_ranges', None)
        if fit_ranges_str is not None:
            fit_ranges_str = fit_ranges_str.get().strip()
            if fit_ranges_str:
                for part in fit_ranges_str.split(','):
                    if '-' in part:
                        try:
                            min_w, max_w = map(float, part.split('-'))
                            self.ax1_current.axvspan(min_w, max_w, color='lightgrey', alpha=0.3, zorder=0)
                        except Exception:
                            pass
        
        # Plot spectrum
        self.ax1_current.plot(self.wavenumbers, self.spectra, 'k-', label='Spectrum')
        
        # Plot background if available
        if self.background is not None:
            self.ax1_current.plot(self.wavenumbers, self.background, 'b--', label='Background')
        
        # Plot fitted peaks if available
        if self.fit_result is not None:
            print(f"fit_params: {self.fit_params}")
            model_type = self.current_model.get()
            if model_type == "Gaussian" or model_type == "Lorentzian":
                params_per_peak = 3
            elif model_type == "Pseudo-Voigt":
                params_per_peak = 4
            elif model_type == "Asymmetric Voigt":
                params_per_peak = 5
            else:
                params_per_peak = 3
            print(f"params_per_peak: {params_per_peak}, len(fit_params): {len(self.fit_params)}")
            if model_type == "Asymmetric Voigt" and len(self.fit_params) % 5 != 0:
                print(f"Warning: fit_params length {len(self.fit_params)} is not a multiple of 5. Skipping plotting.")
                return
            
            self.ax1_current.plot(self.wavenumbers, self.fit_result, 'r-', label='Fit')
            
            # Plot individual peaks if requested
            if self.show_individual_peaks.get():
                if model_type == "Gaussian":
                    peak = self.gaussian(self.wavenumbers, *self.fit_params[0:3])
                elif model_type == "Lorentzian":
                    peak = self.lorentzian(self.wavenumbers, *self.fit_params[0:3])
                elif model_type == "Pseudo-Voigt":
                    peak = self.pseudo_voigt(self.wavenumbers, *self.fit_params[0:4])
                elif model_type == "Asymmetric Voigt":
                    params = self.fit_params[0:5]
                    if len(params) == 5:
                        # Debug print
                        print(f"Asymmetric Voigt params: {params}")
                        peak = self.asymmetric_voigt(self.wavenumbers, *params)
                    else:
                        print(f"Skipping Asymmetric Voigt peak at i={0}, params={params}")
                        return
                else:
                    peak = self.gaussian(self.wavenumbers, *self.fit_params[0:3])
                
                self.ax1_current.plot(self.wavenumbers, peak, 'g--', alpha=0.5)
        
        # Plot peak positions if available
        if self.peaks:
            peak_positions = [peak['position'] for peak in self.peaks]
            peak_intensities = [peak['intensity'] for peak in self.peaks]
            self.ax1_current.plot(peak_positions, peak_intensities, 'ro', label='Peaks')
        
        # Plot residuals if available
        if self.residuals is not None:
            self.ax2_current.plot(self.wavenumbers, self.residuals, 'k-')
            self.ax2_current.axhline(y=0, color='r', linestyle='--')
        
        # Set labels and grid
        self.ax1_current.set_ylabel('Intensity')
        self.ax2_current.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax2_current.set_ylabel('Residuals')
        self.ax1_current.grid(True, linestyle=':', color='gray', alpha=0.6)
        self.ax2_current.grid(True, linestyle=':', color='gray', alpha=0.6)
        
        # Add legend
        self.ax1_current.legend()
        
        # Draw canvas
        self.canvas_current.draw()
    
    def clear_plot(self):
        """Clear the current spectrum plot."""
        self.ax1_current.clear()
        self.ax2_current.clear()
        self.ax1_current.grid(True, linestyle=':', color='gray', alpha=0.6)
        self.ax2_current.grid(True, linestyle=':', color='gray', alpha=0.6)
        self.ax1_current.set_ylabel('Intensity')
        self.ax2_current.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax2_current.set_ylabel('Residuals')
        self.canvas_current.draw()
    
    def clear_peaks(self):
        """Clear detected peaks and fitted peak profiles for the current spectrum."""
        self.peaks = []
        self.fit_params = []
        self.fit_result = None
        self.residuals = None
        self.update_plot()

    def stop_batch(self):
        """Set a flag to stop the batch process."""
        self._stop_batch = True
    
    def on_closing(self):
        """Handle window closing event."""
        self.window.destroy()
    
    def update_roi_regions(self):
        """Refresh the plot to show the current fit regions as shaded areas."""
        self.update_plot()

    def enable_peak_addition(self):
        """Enable peak addition mode by connecting click event."""
        self.canvas_current.mpl_connect('button_press_event', self.add_peak_on_click)
        messagebox.showinfo("Add Peak", "Click on the plot to add a peak. Right-click to cancel.")
    
    def add_peak_on_click(self, event):
        """Add a peak at the clicked position."""
        if event.button != 1:  # Right click cancels
            self.canvas_current.mpl_disconnect(self.canvas_current.mpl_connect('button_press_event', self.add_peak_on_click))
            return
            
        if event.inaxes != self.ax1_current:
            return
            
        try:
            # Get clicked position
            position = event.xdata
            amplitude = event.ydata
            
            # Find the closest wavenumber index
            idx = np.argmin(np.abs(self.wavenumbers - position))
            
            # Create new peak
            new_peak = {
                'position': position,
                'intensity': amplitude,
                'index': idx
            }
            
            # Add to peaks list
            self.peaks.append(new_peak)
            
            # Update plot
            self.update_plot()
            
            # Disconnect click event
            self.canvas_current.mpl_disconnect(self.canvas_current.mpl_connect('button_press_event', self.add_peak_on_click))
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add peak: {str(e)}")
            self.canvas_current.mpl_disconnect(self.canvas_current.mpl_connect('button_press_event', self.add_peak_on_click))
    
    def show_peak_deletion_dialog(self):
        """Show a dialog to select peaks for deletion."""
        if not self.peaks:
            messagebox.showinfo("No Peaks", "No peaks to delete.")
            return
            
        # Create dialog window
        dialog = tk.Toplevel(self.window)
        dialog.title("Delete Peaks")
        dialog.geometry("300x400")
        
        # Create listbox for peak selection
        list_frame = ttk.Frame(dialog, padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        peak_list = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set)
        peak_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=peak_list.yview)
        
        # Add peaks to listbox
        for i, peak in enumerate(self.peaks):
            peak_list.insert(tk.END, f"Peak {i+1}: {peak['position']:.2f} cm⁻¹")
        
        # Add buttons
        button_frame = ttk.Frame(dialog, padding=10)
        button_frame.pack(fill=tk.X)
        
        def delete_selected():
            # Get selected indices in reverse order to avoid index shifting
            selected_indices = sorted(peak_list.curselection(), reverse=True)
            
            if not selected_indices:
                messagebox.showinfo("No Selection", "Please select peaks to delete.")
                return
                
            # Confirm deletion
            if messagebox.askyesno("Confirm Delete", 
                                 f"Delete {len(selected_indices)} selected peak(s)?"):
                # Remove selected peaks
                for idx in selected_indices:
                    del self.peaks[idx]
                
                # Clear fit results
                self.fit_params = []
                self.fit_result = None
                self.residuals = None
                
                # Update plot
                self.update_plot()
                
                # Close dialog
                dialog.destroy()
        
        ttk.Button(button_frame, text="Delete Selected", command=delete_selected).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2) 