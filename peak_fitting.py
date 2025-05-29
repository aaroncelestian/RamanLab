#!/usr/bin/env python3
# Peak Fitting Module for ClaritySpectra
"""
Module for peak fitting of Raman spectra with various peak models
including Gaussian, Lorentzian, and Pseudo-Voigt.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve

class PeakFittingWindow:
    """Window for fitting peaks to Raman spectra."""
    
    def __init__(self, parent, raman, wavenumbers, spectra):
        """
        Initialize the peak fitting window.
        
        Parameters:
        -----------
        parent : tk.Tk or tk.Toplevel
            Parent window
        raman : RamanSpectra
            Reference to the RamanSpectra instance
        wavenumbers : numpy.ndarray
            X-axis data (wavenumbers)
        spectra : numpy.ndarray
            Y-axis data (intensity)
        """
        self.window = tk.Toplevel(parent)
        self.window.title("Peak Fitting")
        self.window.geometry("1300x700")
        self.window.minsize(1100, 600)
        
        # Store references
        self.parent = parent
        self.raman = raman
        self.wavenumbers = wavenumbers
        self.original_spectra = spectra
        
        # Create a copy of the data to work with
        self.spectra = np.copy(spectra)
        
        # Variables for fitted peaks
        self.peaks = []
        self.fit_params = []
        self.fit_result = None
        self.background = None
        self.current_model = tk.StringVar(value="Gaussian")
        self.residuals = None
        self.show_individual_peaks = tk.BooleanVar(value=True)
        
        # Flag for manual peak mode
        self.manual_peak_mode = False
        
        # Create GUI
        self.create_gui()
        
        # Initial plot
        self.update_plot()
        
        # Set up window close event
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_gui(self):
        """Create the GUI elements."""
        # Main frame
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left panel for controls with tabs
        controls_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10, width=450)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        controls_frame.pack_propagate(False)  # Prevent shrinking
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(controls_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create frames for each tab
        self.background_frame = ttk.Frame(self.notebook, padding=10)
        self.peak_detection_frame = ttk.Frame(self.notebook, padding=10)
        self.peak_fitting_frame = ttk.Frame(self.notebook, padding=10)
        
        # Add tabs to notebook
        self.notebook.add(self.background_frame, text="Background")
        self.notebook.add(self.peak_detection_frame, text="Peak Detection")
        self.notebook.add(self.peak_fitting_frame, text="Peak Fitting")
        
        # Set initial tab to Background
        self.notebook.select(self.background_frame)
        
        # Create right panel for visualization
        viz_frame = ttk.LabelFrame(main_frame, text="Spectrum and Fit", padding=10)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # -- Create visualization elements --
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6), 
                                                      gridspec_kw={'height_ratios': [3, 1]})
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(viz_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # -- Create Background Tab Controls --
        self.create_background_tab()
        
        # -- Create Peak Detection Tab Controls --
        self.create_peak_detection_tab()
        
        # -- Create Peak Fitting Tab Controls --
        self.create_peak_fitting_tab()
        
    def create_background_tab(self):
        """Create the background subtraction tab controls."""
        # Background subtraction frame
        bg_frame = ttk.LabelFrame(self.background_frame, text="ALS Background", padding=10)
        bg_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Compact help text
        help_text = tk.Text(bg_frame, height=2, width=42, wrap=tk.WORD)
        help_text.pack(fill=tk.X, pady=(0, 5))
        help_text.insert(tk.END, "ALS fits a smooth baseline following spectrum's lower envelope. λ=rigidity, p=asymmetry.")
        help_text.config(state=tk.DISABLED)
        
        # Control grid layout for more compact arrangement
        control_grid = ttk.Frame(bg_frame)
        control_grid.pack(fill=tk.X, pady=2)
        
        # Lambda row - use grid instead of pack
        ttk.Label(control_grid, text="λ (smoothness):").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        self.var_lambda = tk.StringVar(value="1e5")
        lambda_entry = ttk.Entry(control_grid, textvariable=self.var_lambda, width=10)
        lambda_entry.grid(row=0, column=1, padx=2, pady=2)
        
        lambda_buttons = ttk.Frame(control_grid)
        lambda_buttons.grid(row=0, column=2, padx=2, pady=2, sticky="w")
        ttk.Button(lambda_buttons, text="1e3", width=4, 
                  command=lambda: self.var_lambda.set("1e3")).pack(side=tk.LEFT, padx=1)
        ttk.Button(lambda_buttons, text="1e5", width=4, 
                  command=lambda: self.var_lambda.set("1e5")).pack(side=tk.LEFT, padx=1)
        ttk.Button(lambda_buttons, text="1e7", width=4, 
                  command=lambda: self.var_lambda.set("1e7")).pack(side=tk.LEFT, padx=1)
        
        # P row - use grid to align with lambda row
        ttk.Label(control_grid, text="p (asymmetry):").grid(row=1, column=0, sticky="w", padx=2, pady=2)
        self.var_p = tk.StringVar(value="0.01")
        p_entry = ttk.Entry(control_grid, textvariable=self.var_p, width=10)
        p_entry.grid(row=1, column=1, padx=2, pady=2)
        
        p_buttons = ttk.Frame(control_grid)
        p_buttons.grid(row=1, column=2, padx=2, pady=2, sticky="w")
        ttk.Button(p_buttons, text="0.001", width=4, 
                  command=lambda: self.var_p.set("0.001")).pack(side=tk.LEFT, padx=1)
        ttk.Button(p_buttons, text="0.01", width=4, 
                  command=lambda: self.var_p.set("0.01")).pack(side=tk.LEFT, padx=1)
        ttk.Button(p_buttons, text="0.1", width=4, 
                  command=lambda: self.var_p.set("0.1")).pack(side=tk.LEFT, padx=1)
        
        # Iterations row
        ttk.Label(control_grid, text="Iterations:").grid(row=2, column=0, sticky="w", padx=2, pady=2)
        self.var_niter = tk.StringVar(value="10")
        ttk.Entry(control_grid, textvariable=self.var_niter, width=10).grid(row=2, column=1, padx=2, pady=2)
        
        # Action buttons in a more compact layout
        button_frame = ttk.Frame(bg_frame)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Subtract", 
                  command=self.subtract_background).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(button_frame, text="Preview", 
                  command=self.preview_background).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        
        # Utility buttons in a more compact layout
        utility_frame = ttk.Frame(bg_frame)
        utility_frame.pack(fill=tk.X, pady=2)
        ttk.Button(utility_frame, text="Compare Parameters", 
                 command=self.compare_background_parameters).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(utility_frame, text="Interactive Tuning", 
                 command=self.interactive_background_tuning).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
    
    def create_peak_detection_tab(self):
        """Create the peak detection tab controls."""
        # Peak detection frame
        peak_frame = ttk.LabelFrame(self.peak_detection_frame, text="Peak Detection", padding=10)
        peak_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Help text for peak detection
        peak_help = tk.Text(peak_frame, height=2, width=42, wrap=tk.WORD)
        peak_help.pack(fill=tk.X, pady=(0, 5))
        peak_help.insert(tk.END, "Auto height sets threshold at 5% above background. Distance is minimum separation between peaks.")
        peak_help.config(state=tk.DISABLED)
        
        # Use grid for more compact layout
        peak_grid = ttk.Frame(peak_frame)
        peak_grid.pack(fill=tk.X)
        
        ttk.Label(peak_grid, text="Height:").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        self.var_height = tk.StringVar(value="Auto")
        ttk.Entry(peak_grid, textvariable=self.var_height, width=10).grid(row=0, column=1, sticky="w", padx=2, pady=2)
        
        ttk.Label(peak_grid, text="Distance:").grid(row=1, column=0, sticky="w", padx=2, pady=2)
        self.var_distance = tk.StringVar(value="Auto")
        ttk.Entry(peak_grid, textvariable=self.var_distance, width=10).grid(row=1, column=1, sticky="w", padx=2, pady=2)
        
        ttk.Label(peak_grid, text="Prominence:").grid(row=2, column=0, sticky="w", padx=2, pady=2)
        self.var_prominence = tk.StringVar(value="Auto")
        ttk.Entry(peak_grid, textvariable=self.var_prominence, width=10).grid(row=2, column=1, sticky="w", padx=2, pady=2)
        
        # Button layout with Find and Manual
        peak_buttons = ttk.Frame(peak_frame)
        peak_buttons.pack(fill=tk.X, pady=5)
        ttk.Button(peak_buttons, text="Find Peaks", 
                  command=self.find_peaks).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        
        # Store reference to the manual peak button for styling
        self.manual_peak_button = ttk.Button(peak_buttons, text="Add Peaks Manually", 
                  command=self.enable_manual_peak_adding)
        self.manual_peak_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        
        # Add clear and delete peaks buttons in a frame
        peak_management_frame = ttk.Frame(peak_frame)
        peak_management_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(peak_management_frame, text="Clear Peaks", 
                  command=self.clear_peaks).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(peak_management_frame, text="Delete Peak", 
                  command=self.delete_selected_peak).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
    
    def create_peak_fitting_tab(self):
        """Create the peak fitting tab controls."""
        # Display options
        display_frame = ttk.LabelFrame(self.peak_fitting_frame, text="Display Options", padding=10)
        display_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Checkbutton(
            display_frame,
            text="Show Individual Peaks",
            variable=self.show_individual_peaks,
            command=self.update_plot
        ).pack(anchor="w", pady=2)
        
        # Peak fitting frame
        fit_frame = ttk.LabelFrame(self.peak_fitting_frame, text="Peak Fitting", padding=10)
        fit_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model selection
        model_frame = ttk.Frame(fit_frame)
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT, padx=(0, 5))
        self.current_model = tk.StringVar(value="Gaussian")
        model_combo = ttk.Combobox(model_frame, textvariable=self.current_model, 
                                  values=["Gaussian", "Lorentzian", "Pseudo-Voigt", "Asymmetric Voigt"], 
                                  width=15, state="readonly")
        model_combo.pack(side=tk.LEFT)
        model_combo.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        
        # Fit button
        ttk.Button(fit_frame, text="Fit Peaks", command=self.fit_peaks).pack(pady=10)
        
        # Model info frame with compact info
        model_info_frame = ttk.LabelFrame(self.peak_fitting_frame, text="Model Info", padding=10)
        model_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        model_info_text = tk.Text(model_info_frame, height=4, width=45, wrap=tk.WORD)
        model_info_text.pack(fill=tk.X)
        model_info_text.insert(tk.END, 
            "Gaussian: Symmetric, best for instrumental broadening\n"
            "Lorentzian: Broader tails, natural line broadening\n"
            "Pseudo-Voigt: Combination, most flexible for real spectra\n"
            "Asymmetric Voigt: Handles asymmetric peaks with different widths"
        )
        model_info_text.config(state=tk.DISABLED)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.peak_fitting_frame, text="Fit Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.results_text = tk.Text(results_frame, height=6, width=45, wrap=tk.WORD)
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, 
                                        command=self.results_text.yview)
        self.results_text.config(yscrollcommand=results_scrollbar.set)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Export/Close buttons
        button_frame = ttk.Frame(self.peak_fitting_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Export Results", 
                  command=self.export_results).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(button_frame, text="Close", 
                  command=self.window.destroy).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        
    def subtract_background(self):
        """Subtract the background from the spectrum."""
        try:
            lam = float(self.var_lambda.get())
            p = float(self.var_p.get())
            niter = int(self.var_niter.get())
            
            # Use the baseline_als method from the RamanSpectra class
            self.background = self.baseline_als(self.original_spectra, lam=lam, p=p, niter=niter)
            self.spectra = self.original_spectra - self.background
            
            self.update_plot()
            # Update the plot title to show that background was subtracted
            self.ax1.set_title(f'Raman Spectrum - Background Subtracted (λ={lam:.1e}, p={p:.4f})')
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to subtract background: {str(e)}", parent=self.window)
    
    def preview_background(self):
        """Preview the background without subtracting it."""
        try:
            lam = float(self.var_lambda.get())
            p = float(self.var_p.get())
            niter = int(self.var_niter.get())
            
            # Calculate the background
            self.background = self.baseline_als(self.original_spectra, lam=lam, p=p, niter=niter)
            
            # Just update the plot without altering the spectrum
            self.update_plot()
            
            # Show parameter values in the title
            self.ax1.set_title(f'Raman Spectrum with ALS Background (λ={lam}, p={p}, iter={niter})')
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to preview background: {str(e)}", parent=self.window)
    
    def compare_background_parameters(self):
        """Compare different background parameters in a separate window."""
        try:
            # Create a new window
            compare_window = tk.Toplevel(self.window)
            compare_window.title("Background Parameter Comparison")
            compare_window.geometry("1000x750")  # Reduced width and height
            
            # Create a canvas with scrollbar for the entire window
            main_canvas = tk.Canvas(compare_window)
            scrollbar = ttk.Scrollbar(compare_window, orient=tk.VERTICAL, command=main_canvas.yview)
            
            # Configure the canvas with the scrollbar
            main_canvas.configure(yscrollcommand=scrollbar.set)
            main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Create a main frame inside the canvas to hold everything
            main_frame = ttk.Frame(main_canvas, padding=5)
            
            # Create a window in the canvas to display the main_frame
            canvas_window = main_canvas.create_window((0, 0), window=main_frame, anchor="nw")
            
            # Configure the canvas to resize with the frame
            def on_frame_configure(event):
                main_canvas.configure(scrollregion=main_canvas.bbox("all"))
                main_canvas.itemconfig(canvas_window, width=main_canvas.winfo_width())
            
            main_frame.bind("<Configure>", on_frame_configure)
            main_canvas.bind("<Configure>", lambda e: main_canvas.itemconfig(canvas_window, width=e.width))
            
            # Upper frame for plots
            plot_frame = ttk.Frame(main_frame)
            plot_frame.pack(fill=tk.BOTH, padx=10, pady=10)
            
            # Create a figure with subplots for different parameter combinations
            # Make plots smaller by reducing figsize even more
            fig, axs = plt.subplots(2, 3, figsize=(8, 5), sharex=True, sharey=True)
            fig.suptitle("Comparison of Asymmetric Least Squares Background Parameters", fontsize=12)
            
            # Flatten the axes array for easier iteration
            axs = axs.flatten()
            
            # Define parameter combinations to compare
            # Format: (lambda, p, niter, title)
            param_sets = [
                (1e3, 0.001, 10, "λ=1e3, p=0.001 (rigid, less peak influence)"),
                (1e3, 0.1, 10, "λ=1e3, p=0.1 (rigid, more peak influence)"),
                (1e5, 0.001, 10, "λ=1e5, p=0.001 (medium, less peak influence)"),
                (1e5, 0.1, 10, "λ=1e5, p=0.1 (medium, more peak influence)"),
                (1e7, 0.001, 10, "λ=1e7, p=0.001 (smooth, less peak influence)"),
                (1e7, 0.1, 10, "λ=1e7, p=0.1 (smooth, more peak influence)")
            ]
            
            # Create a variable to store the selected parameter set
            selected_param = tk.IntVar(value=-1)  # -1 means no selection
            
            # Store background data for each parameter set
            bg_data = []
            
            # Plot each parameter set
            for i, (lam, p, niter, title) in enumerate(param_sets):
                if i < len(axs):
                    ax = axs[i]
                    
                    # Calculate background for this parameter set
                    bg = self.baseline_als(self.original_spectra, lam=lam, p=p, niter=niter)
                    bg_data.append(bg)  # Store for later use
                    
                    # Plot original data
                    ax.plot(self.wavenumbers, self.original_spectra, 'k-', alpha=0.7, label='Original')
                    
                    # Plot background
                    ax.plot(self.wavenumbers, bg, 'r-', alpha=0.9, label='Background')
                    
                    # Plot corrected spectrum
                    ax.plot(self.wavenumbers, self.original_spectra - bg, 'b-', alpha=0.5, label='Corrected')
                    
                    # Set title and label with smaller fonts
                    ax.set_title(f"Set {i+1}: {title}", fontsize=7)
                    ax.set_xlabel('Wavenumber (cm⁻¹)' if i >= 3 else '', fontsize=7)
                    ax.set_ylabel('Intensity (a.u.)' if i % 3 == 0 else '', fontsize=7)
                    ax.tick_params(labelsize=6)  # Make tick labels smaller
                    
                    # Only add legend to the first plot
                    if i == 0:
                        ax.legend(loc='upper right', fontsize=6)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave room for suptitle
            
            # Create a canvas to display the figure
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=5)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, plot_frame)
            toolbar.update()
            
            # Add selection frame with radio buttons under each plot
            selection_frame = ttk.LabelFrame(main_frame, text="Select Your Preferred Background", padding=10)
            selection_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
            
            # Create a grid for the radio buttons
            radio_grid = ttk.Frame(selection_frame)
            radio_grid.pack(fill=tk.X, pady=5)
            
            # Create radio buttons for each parameter set
            for i, (lam, p, niter, title) in enumerate(param_sets):
                radio_frame = ttk.Frame(radio_grid)
                radio_frame.grid(row=i//3, column=i%3, padx=10, pady=5, sticky="w")
                
                rb = ttk.Radiobutton(
                    radio_frame, 
                    text=f"Set {i+1}: λ={lam:.0e}, p={p:.3f}", 
                    variable=selected_param, 
                    value=i
                )
                rb.pack(side=tk.LEFT)
                
                # Add preview button for each set
                ttk.Button(
                    radio_frame,
                    text="Preview",
                    command=lambda idx=i: preview_selected(idx)
                ).pack(side=tk.LEFT, padx=5)
            
            # Preview function to display a larger version of the selected background
            def preview_selected(index):
                preview_window = tk.Toplevel(compare_window)
                preview_window.title(f"Preview of Parameter Set {index+1}")
                preview_window.geometry("800x600")
                
                # Get parameters
                lam, p, niter, title = param_sets[index]
                bg = bg_data[index]
                
                # Create figure for preview
                preview_fig, preview_ax = plt.subplots(figsize=(8, 6))
                preview_ax.plot(self.wavenumbers, self.original_spectra, 'k-', alpha=0.7, label='Original')
                preview_ax.plot(self.wavenumbers, bg, 'r-', alpha=0.9, label='Background')
                preview_ax.plot(self.wavenumbers, self.original_spectra - bg, 'b-', alpha=0.7, label='Corrected')
                
                preview_ax.set_title(f"Parameter Set {index+1}: {title}")
                preview_ax.set_xlabel('Wavenumber (cm⁻¹)')
                preview_ax.set_ylabel('Intensity (a.u.)')
                preview_ax.legend(loc='best')
                
                # Create canvas
                preview_canvas = FigureCanvasTkAgg(preview_fig, master=preview_window)
                preview_canvas.draw()
                preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                # Add toolbar
                preview_toolbar = NavigationToolbar2Tk(preview_canvas, preview_window)
                preview_toolbar.update()
                
                # Buttons frame
                buttons_frame = ttk.Frame(preview_window)
                buttons_frame.pack(fill=tk.X, pady=10, padx=10)
                
                # Add apply button
                ttk.Button(
                    buttons_frame,
                    text="Use These Parameters",
                    command=lambda: apply_param_set(index, preview_window)
                ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
                
                # Add close button
                ttk.Button(
                    buttons_frame,
                    text="Close Preview",
                    command=preview_window.destroy
                ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            # Add a visual hint panel to highlight the differences
            hint_frame = ttk.LabelFrame(main_frame, text="Visual Comparison Guide", padding=10)
            hint_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
            
            # Create a grid layout for the hints
            hint_grid = ttk.Frame(hint_frame)
            hint_grid.pack(fill=tk.X)
            
            # Add hints for different lambda values
            ttk.Label(hint_grid, text="λ (smoothness):", font=("", 10, "bold")).grid(row=0, column=0, sticky="w", padx=5, pady=2)
            ttk.Label(hint_grid, text="1e3 - More flexible, follows data closely").grid(row=0, column=1, sticky="w", padx=5, pady=2)
            ttk.Label(hint_grid, text="1e5 - Balanced smoothness").grid(row=0, column=2, sticky="w", padx=5, pady=2)
            ttk.Label(hint_grid, text="1e7 - Very rigid, may underfit").grid(row=0, column=3, sticky="w", padx=5, pady=2)
            
            # Add hints for different p values
            ttk.Label(hint_grid, text="p (asymmetry):", font=("", 10, "bold")).grid(row=1, column=0, sticky="w", padx=5, pady=2)
            ttk.Label(hint_grid, text="0.001 - Less influenced by peaks").grid(row=1, column=1, sticky="w", padx=5, pady=2)
            ttk.Label(hint_grid, text="0.01 - Moderate peak influence").grid(row=1, column=2, sticky="w", padx=5, pady=2)
            ttk.Label(hint_grid, text="0.1 - More influenced by peaks").grid(row=1, column=3, sticky="w", padx=5, pady=2)
            
            # Add button frame
            button_frame = ttk.Frame(main_frame, padding=10)
            button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
            
            # Function to apply selected parameters
            def apply_param_set(param_set, window_to_close=None):
                lam, p, niter, _ = param_sets[param_set]
                self.var_lambda.set(str(lam))
                self.var_p.set(str(p))
                self.var_niter.set(str(niter))
                
                # Close the optional window if provided
                if window_to_close:
                    window_to_close.destroy()
                    
                compare_window.destroy()
                self.preview_background()
            
            # Button to apply the selected parameters - more prominent design
            apply_button = ttk.Button(
                button_frame, 
                text="Apply Selected Background", 
                command=lambda: apply_selected_param(),
                style="Apply.TButton"  # Custom style for prominence
            )
            apply_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            # Create custom style for the apply button
            style = ttk.Style()
            style.configure("Apply.TButton", font=("", 10, "bold"))
            
            # Function to apply the selected parameters
            def apply_selected_param():
                selected = selected_param.get()
                if selected >= 0:
                    apply_param_set(selected)
                else:
                    messagebox.showinfo("No Selection", "Please select a parameter set first.", parent=self.window)
            
            # Add a close button
            ttk.Button(
                button_frame, 
                text="Close Without Applying", 
                command=compare_window.destroy
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compare background parameters: {str(e)}", parent=self.window)
    
    def interactive_background_tuning(self):
        """Create an interactive window with sliders to adjust background parameters."""
        try:
            # Create a new window
            tuning_window = tk.Toplevel(self.window)
            tuning_window.title("Interactive Background Parameter Tuning")
            tuning_window.geometry("1000x800")  # Increased height from 700 to 800
            
            # Create a canvas with scrollbar for the entire window
            main_canvas = tk.Canvas(tuning_window)
            scrollbar = ttk.Scrollbar(tuning_window, orient=tk.VERTICAL, command=main_canvas.yview)
            
            # Configure the canvas with the scrollbar
            main_canvas.configure(yscrollcommand=scrollbar.set)
            main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Create a main frame inside the canvas to hold everything
            main_frame = ttk.Frame(main_canvas, padding=5)
            
            # Create a window in the canvas to display the main_frame
            canvas_window = main_canvas.create_window((0, 0), window=main_frame, anchor="nw")
            
            # Configure the canvas to resize with the frame
            def on_frame_configure(event):
                main_canvas.configure(scrollregion=main_canvas.bbox("all"))
                main_canvas.itemconfig(canvas_window, width=main_canvas.winfo_width())
            
            main_frame.bind("<Configure>", on_frame_configure)
            main_canvas.bind("<Configure>", lambda e: main_canvas.itemconfig(canvas_window, width=e.width))
            
            # Upper frame for plot
            plot_frame = ttk.Frame(main_frame, padding=10)
            plot_frame.pack(fill=tk.BOTH, expand=True)
            
            # Create a figure for the plot
            fig, ax = plt.subplots(figsize=(8, 5))
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, plot_frame)
            toolbar.update()
            
            # Lower frame for controls
            control_frame = ttk.LabelFrame(main_frame, text="Parameter Controls", padding=10)
            control_frame.pack(fill=tk.X, padx=10, pady=10)
            
            # Lambda slider (logarithmic scale)
            lambda_frame = ttk.Frame(control_frame)
            lambda_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(lambda_frame, text="λ (smoothness):").pack(side=tk.LEFT, padx=(0, 10))
            
            # Lambda value displays
            lambda_val = tk.StringVar(value="1e5")
            lambda_display = ttk.Entry(lambda_frame, textvariable=lambda_val, width=10)
            lambda_display.pack(side=tk.RIGHT)
            
            # Lambda slider (using logarithmic scale internally, displayed as power of 10)
            lambda_slider_var = tk.DoubleVar(value=5.0)  # 10^5 = 1e5
            lambda_slider = ttk.Scale(
                control_frame, 
                from_=2.0, 
                to=8.0, 
                orient=tk.HORIZONTAL, 
                variable=lambda_slider_var,
                length=400
            )
            lambda_slider.pack(fill=tk.X, pady=(0, 10))
            
            # P slider (logarithmic scale)
            p_frame = ttk.Frame(control_frame)
            p_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(p_frame, text="p (asymmetry):").pack(side=tk.LEFT, padx=(0, 10))
            
            # P value displays
            p_val = tk.StringVar(value="0.01")
            p_display = ttk.Entry(p_frame, textvariable=p_val, width=10)
            p_display.pack(side=tk.RIGHT)
            
            # P slider (using logarithmic scale)
            p_slider_var = tk.DoubleVar(value=-2.0)  # 10^-2 = 0.01
            p_slider = ttk.Scale(
                control_frame, 
                from_=-4.0, 
                to=-0.5, 
                orient=tk.HORIZONTAL, 
                variable=p_slider_var,
                length=400
            )
            p_slider.pack(fill=tk.X, pady=(0, 10))
            
            # Iteration slider
            niter_frame = ttk.Frame(control_frame)
            niter_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(niter_frame, text="Iterations:").pack(side=tk.LEFT, padx=(0, 10))
            
            # Iteration value display
            niter_val = tk.StringVar(value="10")
            niter_display = ttk.Entry(niter_frame, textvariable=niter_val, width=10)
            niter_display.pack(side=tk.RIGHT)
            
            # Iteration slider
            niter_slider_var = tk.IntVar(value=10)
            niter_slider = ttk.Scale(
                control_frame, 
                from_=1, 
                to=50, 
                orient=tk.HORIZONTAL, 
                variable=niter_slider_var,
                length=400
            )
            niter_slider.pack(fill=tk.X, pady=(0, 10))
            
            # Performance optimization: Cache data and plot objects
            # Store background trace and corrected trace as line objects
            lines = {}
            
            # Initial plot setup - do this once and then just update the data
            lines['original'], = ax.plot(self.wavenumbers, self.original_spectra, 'k-', alpha=0.7, label='Original')
            lines['background'], = ax.plot(self.wavenumbers, np.zeros_like(self.wavenumbers), 'r-', alpha=0.8, label='Background')
            lines['corrected'], = ax.plot(self.wavenumbers, np.zeros_like(self.wavenumbers), 'b-', label='Corrected')
            
            # Set labels and title - do this once
            ax.set_xlabel('Wavenumber (cm⁻¹)')
            ax.set_ylabel('Intensity (a.u.)')
            ax.set_title('ALS Background')
            ax.legend(loc='upper right')
            
            # Set initial y-limits to avoid auto-scaling on every update
            ax.set_ylim(np.min(self.original_spectra) * 0.9, np.max(self.original_spectra) * 1.1)
            
            # Throttling mechanism for slider events
            update_pending = False
            last_update_time = 0
            MIN_UPDATE_INTERVAL = 200  # milliseconds
            
            # Function to update the plot
            def update_plot(throttled=False):
                nonlocal update_pending, last_update_time
                
                current_time = int(tuning_window.winfo_toplevel().winfo_id())  # Use as millisecond counter
                
                # If this is a throttled update or too soon after the last update, schedule for later
                if not throttled and current_time - last_update_time < MIN_UPDATE_INTERVAL:
                    if not update_pending:
                        update_pending = True
                        tuning_window.after(MIN_UPDATE_INTERVAL, lambda: update_plot(True))
                    return
                
                # Reset flags
                update_pending = False
                last_update_time = current_time
                
                # Get slider values and convert to actual parameters
                lam = 10 ** lambda_slider_var.get()
                p = 10 ** p_slider_var.get()
                niter = niter_slider_var.get()
                
                # Update entry displays without triggering callbacks
                lambda_val.set(f"{lam:.1e}")
                p_val.set(f"{p:.4f}")
                niter_val.set(str(niter))
                
                # Calculate background
                background = self.baseline_als(self.original_spectra, lam=lam, p=p, niter=niter)
                corrected = self.original_spectra - background
                
                # Update data without recreating the plot objects
                lines['background'].set_ydata(background)
                lines['corrected'].set_ydata(corrected)
                
                # Update title with current parameters
                ax.set_title(f'ALS Background (λ={lam:.1e}, p={p:.4f}, iter={niter})')
                
                # Draw only what needs to be updated
                canvas.draw_idle()
            
            # Bind sliders to update function with throttling
            def on_slider_change(*args):
                update_plot(False)
            
            # Bind slider release events for immediate update
            lambda_slider.bind("<ButtonRelease-1>", on_slider_change)
            p_slider.bind("<ButtonRelease-1>", on_slider_change)
            niter_slider.bind("<ButtonRelease-1>", on_slider_change)
            
            # Also bind slider motion, but this will be throttled
            lambda_slider.bind("<B1-Motion>", on_slider_change)
            p_slider.bind("<B1-Motion>", on_slider_change)
            niter_slider.bind("<B1-Motion>", on_slider_change)
            
            # Function to apply parameters to main window
            def apply_parameters():
                lam = 10 ** lambda_slider_var.get()
                p = 10 ** p_slider_var.get()
                niter = niter_slider_var.get()
                
                self.var_lambda.set(f"{lam:.1e}")
                self.var_p.set(f"{p:.4f}")
                self.var_niter.set(str(niter))
                
                tuning_window.destroy()
                self.preview_background()
            
            # Function to handle numeric entry changes
            def on_lambda_entry_change(*args):
                try:
                    value = float(lambda_val.get().replace('e', 'E'))
                    log_value = np.log10(value)
                    if 2.0 <= log_value <= 8.0:  # Ensure within slider range
                        lambda_slider_var.set(log_value)
                        update_plot()
                except (ValueError, TypeError):
                    pass  # Invalid input, ignore
            
            def on_p_entry_change(*args):
                try:
                    value = float(p_val.get().replace('e', 'E'))
                    log_value = np.log10(value)
                    if -4.0 <= log_value <= -0.5:  # Ensure within slider range
                        p_slider_var.set(log_value)
                        update_plot()
                except (ValueError, TypeError):
                    pass  # Invalid input, ignore
            
            def on_niter_entry_change(*args):
                try:
                    value = int(niter_val.get())
                    if 1 <= value <= 50:  # Ensure within slider range
                        niter_slider_var.set(value)
                        update_plot()
                except (ValueError, TypeError):
                    pass  # Invalid input, ignore
            
            # Bind entry fields to update function
            lambda_val.trace_add("write", on_lambda_entry_change)
            p_val.trace_add("write", on_p_entry_change)
            niter_val.trace_add("write", on_niter_entry_change)
            
            # Add buttons
            button_frame = ttk.Frame(main_frame, padding=10)
            button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
            
            ttk.Button(
                button_frame, 
                text="Apply Parameters", 
                command=apply_parameters
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            ttk.Button(
                button_frame, 
                text="Reset to Default", 
                command=lambda: [
                    lambda_slider_var.set(5.0),
                    p_slider_var.set(-2.0),
                    niter_slider_var.set(10),
                    update_plot(True)
                ]
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            ttk.Button(
                button_frame, 
                text="Close", 
                command=tuning_window.destroy
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            # Status bar for user feedback
            status_var = tk.StringVar(value="Ready. Move sliders to adjust parameters.")
            status_bar = ttk.Label(main_frame, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
            status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
            
            # Initial plot
            update_plot(True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create interactive tuning window: {str(e)}", parent=self.window)
    
    def baseline_als(self, y, lam=1e5, p=0.01, niter=10):
        """
        Asymmetric Least Squares Smoothing.
        Baseline correction algorithm that is useful for spectra with broad peaks.
        
        Parameters:
        -----------
        y : array_like
            Input data (intensity values)
        lam : float, optional
            Smoothness parameter. Higher values make the baseline more rigid.
        p : float, optional
            Asymmetry parameter. Values between 0.001 and 0.1 are good for baseline fitting.
        niter : int, optional
            Number of iterations to perform.
            
        Returns:
        --------
        array_like
            Fitted baseline.
        """
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        D = lam * D.dot(D.transpose())
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
        
        for i in range(niter):
            W.setdiag(w)
            Z = W + D
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1-p) * (y <= z)
            
        return z
    
    def find_peaks(self):
        """Find peaks in the spectrum."""
        try:
            # Initialize peaks list if it doesn't exist
            if self.peaks is None:
                self.peaks = []
                
            # Get parameters
            height_str = self.var_height.get().strip()
            distance_str = self.var_distance.get().strip()
            prominence_str = self.var_prominence.get().strip()
            
            # Determine spectrum for peak finding
            spectrum_to_use = self.spectra
            
            # Set auto height to 5% above background if background is available
            if height_str == "Auto":
                if self.background is not None:
                    # Calculate 5% of the intensity range above background
                    intensity_range = np.max(spectrum_to_use) - np.min(spectrum_to_use)
                    height = np.min(spectrum_to_use) + (0.05 * intensity_range)
                else:
                    # Without background, use 5% of max intensity
                    height = 0.05 * np.max(spectrum_to_use)
            else:
                try:
                    # Handle numeric input
                    height = float(height_str)
                except ValueError:
                    # Default if input is invalid
                    height = 0.05 * np.max(spectrum_to_use)
            
            # Handle other parameters
            distance = None if distance_str == "Auto" else int(distance_str)
            prominence = None if prominence_str == "Auto" else float(prominence_str)
            
            # Find peaks using scipy
            from scipy.signal import find_peaks
            peak_indices, peak_props = find_peaks(
                spectrum_to_use, 
                height=height,
                distance=distance,
                prominence=prominence
            )
            
            # Store peak data
            self.peaks = []
            for i in peak_indices:
                self.peaks.append({
                    'position': self.wavenumbers[i],
                    'intensity': spectrum_to_use[i],
                    'index': i
                })
                
            # Update the plot with newly found peaks
            self.update_plot()
            
            # Update title to show the number of peaks found
            self.ax1.set_title(f'Raman Spectrum - {len(self.peaks)} Peaks Found')
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to find peaks: {str(e)}", parent=self.window)
    
    def gaussian(self, x, amp, cen, wid):
        """Gaussian peak function."""
        return amp * np.exp(-(x - cen)**2 / (2 * wid**2))
    
    def lorentzian(self, x, amp, cen, wid):
        """Lorentzian peak function."""
        return amp * wid**2 / ((x - cen)**2 + wid**2)
    
    def pseudo_voigt(self, x, amp, cen, wid, eta=0.5):
        """Pseudo-Voigt peak function (linear combination of Gaussian and Lorentzian)."""
        return amp * (eta * self.gaussian(x, 1, cen, wid) + 
                    (1-eta) * self.lorentzian(x, 1, cen, wid))
    
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

    def multi_peak_model(self, x, *params):
        """
        Multi-peak model combining multiple peak functions.
        
        Parameters:
        -----------
        x : array-like
            Wavenumber values
        *params : float
            Peak parameters in the format [amp1, cen1, wid1, ...]
            
        Returns:
        --------
        array-like
            The sum of all peak functions evaluated at x
        """
        if not self.peaks:
            return np.zeros_like(x)
            
        model = self.current_model.get()
        n_peaks = len(self.peaks)
        
        if model == "Gaussian":
            n_params = 3
            peak_func = self.gaussian
        elif model == "Lorentzian":
            n_params = 3
            peak_func = self.lorentzian
        elif model == "Pseudo-Voigt":
            n_params = 4
            peak_func = self.pseudo_voigt
        elif model == "Asymmetric Voigt":
            n_params = 5
            peak_func = self.asymmetric_voigt
        else:
            raise ValueError(f"Unknown peak model: {model}")
            
        # Ensure we have the right number of parameters
        if len(params) != n_peaks * n_params:
            raise ValueError(f"Expected {n_peaks * n_params} parameters, got {len(params)}")
            
        # Calculate the sum of all peaks
        result = np.zeros_like(x)
        for i in range(n_peaks):
            start_idx = i * n_params
            end_idx = start_idx + n_params
            if end_idx > len(params):
                break
            result += peak_func(x, *params[start_idx:end_idx])
                
        return result

    def fit_peaks(self):
        """Fit the selected peaks to the spectrum."""
        if not self.peaks:
            messagebox.showwarning("No Peaks", "Please add peaks before fitting.", parent=self.window)
            return
            
        # Get the current model
        model = self.current_model.get()
        
        # Prepare initial parameters
        initial_params = []
        bounds_lower = []
        bounds_upper = []
        
        # Get the range of the spectrum for setting bounds
        x_min, x_max = np.min(self.wavenumbers), np.max(self.wavenumbers)
        y_min, y_max = np.min(self.spectra), np.max(self.spectra)
        
        for peak in self.peaks:
            if model == "Gaussian" or model == "Lorentzian":
                # [amplitude, center, width]
                initial_params.extend([peak['intensity'], peak['position'], 10.0])
                bounds_lower.extend([0, max(x_min, peak['position'] - 20), 1])
                bounds_upper.extend([y_max * 2, min(x_max, peak['position'] + 20), 100])
            elif model == "Pseudo-Voigt":
                # [amplitude, center, width, eta]
                initial_params.extend([peak['intensity'], peak['position'], 10.0, 0.5])
                bounds_lower.extend([0, max(x_min, peak['position'] - 20), 1, 0])
                bounds_upper.extend([y_max * 2, min(x_max, peak['position'] + 20), 100, 1])
            elif model == "Asymmetric Voigt":
                # [amplitude, center, width_left, width_right, eta]
                initial_params.extend([peak['intensity'], peak['position'], 10.0, 10.0, 0.5])
                bounds_lower.extend([0, max(x_min, peak['position'] - 20), 1, 1, 0])
                bounds_upper.extend([y_max * 2, min(x_max, peak['position'] + 20), 100, 100, 1])
        
        try:
            # Perform the fit
            popt, pcov = curve_fit(
                self.multi_peak_model,
                self.wavenumbers,
                self.spectra,
                p0=initial_params,
                bounds=(bounds_lower, bounds_upper),
                maxfev=10000
            )
            
            # Store the results
            self.fit_params = popt
            self.fit_result = self.multi_peak_model(self.wavenumbers, *popt)
            self.residuals = self.spectra - self.fit_result
            
            # Calculate individual peak R² values
            self.peak_r_squared = self.calculate_peak_r_squared(popt, model)
            
            # Display the results
            self.display_fit_results(popt, pcov)
            
            # Update the plot
            self.update_plot()
            
        except Exception as e:
            messagebox.showerror("Fit Error", f"Error during peak fitting: {str(e)}", parent=self.window)
            
    def display_fit_results(self, params, covariance):
        """Display the fit results in a new window."""
        # Clear the text box
        self.results_text.delete(1.0, tk.END)
        
        # Get model type
        model_type = self.current_model.get()
        if model_type == "Gaussian" or model_type == "Lorentzian":
            params_per_peak = 3
        elif model_type == "Pseudo-Voigt":
            params_per_peak = 4
        elif model_type == "Asymmetric Voigt":
            params_per_peak = 5
        else:
            params_per_peak = 3  # Default case
        
        # Calculate number of peaks
        n_peaks = len(params) // params_per_peak
        
        # Calculate parameter errors (standard deviations)
        errors = np.sqrt(np.diag(covariance))
        
        # Calculate goodness of fit (R-squared)
        ss_tot = np.sum((self.spectra - np.mean(self.spectra))**2)
        ss_res = np.sum(self.residuals**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Header for the results
        self.results_text.insert(tk.END, f"Model: {model_type}\n")
        self.results_text.insert(tk.END, f"Number of peaks: {n_peaks}\n")
        self.results_text.insert(tk.END, f"R-squared: {r_squared:.4f}\n\n")
        self.results_text.insert(tk.END, "Peak Parameters:\n")
        
        # Display parameters for each peak
        for i in range(n_peaks):
            self.results_text.insert(tk.END, f"Peak {i+1}:\n")
            
            start_idx = i * params_per_peak
            end_idx = start_idx + params_per_peak
            
            # Common parameters for all models
            amp = params[start_idx]
            amp_err = errors[start_idx]
            cen = params[start_idx+1]
            cen_err = errors[start_idx+1]
            
            self.results_text.insert(tk.END, f"  Position: {cen:.2f} ± {cen_err:.2f} cm⁻¹\n")
            self.results_text.insert(tk.END, f"  Amplitude: {amp:.2f} ± {amp_err:.2f}\n")
            
            # Add individual peak R² if available
            if hasattr(self, 'peak_r_squared') and i < len(self.peak_r_squared):
                self.results_text.insert(tk.END, f"  R²: {self.peak_r_squared[i]:.4f}\n")
            
            if model_type == "Gaussian" or model_type == "Lorentzian":
                wid = params[start_idx+2]
                wid_err = errors[start_idx+2]
                self.results_text.insert(tk.END, f"  Width: {wid:.2f} ± {wid_err:.2f} cm⁻¹\n")
            
            elif model_type == "Pseudo-Voigt":
                wid = params[start_idx+2]
                wid_err = errors[start_idx+2]
                eta = params[start_idx+3]
                eta_err = errors[start_idx+3]
                self.results_text.insert(tk.END, f"  Width: {wid:.2f} ± {wid_err:.2f} cm⁻¹\n")
                self.results_text.insert(tk.END, f"  Eta (mix ratio): {eta:.2f} ± {eta_err:.2f}\n")
            
            elif model_type == "Asymmetric Voigt":
                wid_left = params[start_idx+2]
                wid_left_err = errors[start_idx+2]
                wid_right = params[start_idx+3]
                wid_right_err = errors[start_idx+3]
                eta = params[start_idx+4]
                eta_err = errors[start_idx+4]
                
                # Calculate asymmetry and its error
                asymmetry = wid_right / wid_left
                # Error propagation for asymmetry
                asymmetry_err = asymmetry * np.sqrt((wid_right_err/wid_right)**2 + (wid_left_err/wid_left)**2)
                
                self.results_text.insert(tk.END, f"  Left Width: {wid_left:.2f} ± {wid_left_err:.2f} cm⁻¹\n")
                self.results_text.insert(tk.END, f"  Right Width: {wid_right:.2f} ± {wid_right_err:.2f} cm⁻¹\n")
                self.results_text.insert(tk.END, f"  Asymmetry (right/left): {asymmetry:.2f} ± {asymmetry_err:.2f}\n")
                self.results_text.insert(tk.END, f"  Eta (mix ratio): {eta:.2f} ± {eta_err:.2f}\n")
            
            self.results_text.insert(tk.END, "\n")
    
    def update_plot(self):
        """Update the plot with current data and fit."""
        # Clear the axes
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot original data
        self.ax1.plot(self.wavenumbers, self.original_spectra, 'k-', alpha=0.5)
        
        # Plot background if available
        if self.background is not None:
            self.ax1.plot(self.wavenumbers, self.background, 'r--', alpha=0.5)
        
        # Plot processed data
        self.ax1.plot(self.wavenumbers, self.spectra, 'b-')
        
        # Plot peaks
        if self.peaks:
            peak_positions = [peak['position'] for peak in self.peaks]
            peak_intensities = [peak['intensity'] for peak in self.peaks]
            self.ax1.plot(peak_positions, peak_intensities, 'ro')
        
        # Plot fit if available
        if self.fit_result is not None:
            # Always plot the combined fit
            self.ax1.plot(self.wavenumbers, self.fit_result, 'g-')
            
            # Plot individual peaks if checkbox is checked
            if self.show_individual_peaks.get():
                model_type = self.current_model.get()
                if model_type == "Gaussian" or model_type == "Lorentzian":
                    params_per_peak = 3
                elif model_type == "Pseudo-Voigt":
                    params_per_peak = 4
                elif model_type == "Asymmetric Voigt":
                    params_per_peak = 5
                else:
                    params_per_peak = 3  # Default case
                n_peaks = len(self.fit_params) // params_per_peak
                
                # Create a color cycle for individual peaks
                colors = plt.cm.viridis(np.linspace(0, 1, n_peaks))
                
                for i in range(n_peaks):
                    start_idx = i * params_per_peak
                    
                    if model_type == "Gaussian":
                        amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                        peak = self.gaussian(self.wavenumbers, amp, cen, wid)
                        self.ax1.plot(self.wavenumbers, peak, '--', alpha=0.7, color=colors[i])
                        # Add text annotation for peak position
                        self.ax1.annotate(f'{cen:.1f}', 
                                         xy=(cen, amp), 
                                         xytext=(0, 10),
                                         textcoords='offset points',
                                         ha='center',
                                         fontsize=8,
                                         rotation=65,
                                         bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
                    
                    elif model_type == "Lorentzian":
                        amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                        peak = self.lorentzian(self.wavenumbers, amp, cen, wid)
                        self.ax1.plot(self.wavenumbers, peak, '--', alpha=0.7, color=colors[i])
                        # Add text annotation for peak position
                        self.ax1.annotate(f'{cen:.1f}', 
                                         xy=(cen, amp), 
                                         xytext=(0, 10),
                                         textcoords='offset points',
                                         ha='center',
                                         fontsize=8,
                                         rotation=65,
                                         bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
                    
                    elif model_type == "Pseudo-Voigt":
                        amp, cen, wid, eta = self.fit_params[start_idx:start_idx+4]
                        peak = self.pseudo_voigt(self.wavenumbers, amp, cen, wid, eta)
                        self.ax1.plot(self.wavenumbers, peak, '--', alpha=0.7, color=colors[i])
                        # Add text annotation for peak position
                        self.ax1.annotate(f'{cen:.1f}', 
                                         xy=(cen, amp), 
                                         xytext=(0, 10),
                                         textcoords='offset points',
                                         ha='center',
                                         fontsize=8,
                                         rotation=65,
                                         bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
                        
                    elif model_type == "Asymmetric Voigt":
                        amp, cen, wid_left, wid_right, eta = self.fit_params[start_idx:start_idx+5]
                        peak = self.asymmetric_voigt(self.wavenumbers, amp, cen, wid_left, wid_right, eta)
                        self.ax1.plot(self.wavenumbers, peak, '--', alpha=0.7, color=colors[i])
                        # Add text annotation for peak position
                        self.ax1.annotate(f'{cen:.1f}', 
                                         xy=(cen, amp), 
                                         xytext=(0, 10),
                                         textcoords='offset points',
                                         ha='center',
                                         fontsize=8,
                                         rotation=65,
                                         bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
                
                # Plot residuals
                self.ax2.plot(self.wavenumbers, self.residuals, 'r-')
                # Add zero line
                self.ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
                
                # Add colored fill for residuals
                self.ax2.fill_between(self.wavenumbers, self.residuals, 0, 
                                    where=(self.residuals > 0), 
                                    color='red', alpha=0.3, interpolate=True)
                self.ax2.fill_between(self.wavenumbers, self.residuals, 0, 
                                    where=(self.residuals < 0), 
                                    color='blue', alpha=0.3, interpolate=True)
        
        # Configure axes
        self.ax1.set_title('Raman Spectrum and Peak Fit')
        self.ax1.set_ylabel('Intensity (a.u.)')
        
        self.ax2.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax2.set_ylabel('Residuals')
        
        # Update the figure
        self.fig.tight_layout()
        self.canvas.draw()
    
    def export_results(self):
        """Export the fitting results to a file and optionally to the mineral database."""
        # Check if fit_params exists and has data in a NumPy-safe way
        has_fit_results = hasattr(self, 'fit_params') and self.fit_params is not None
        if not has_fit_results or (isinstance(self.fit_params, np.ndarray) and self.fit_params.size == 0):
            messagebox.showwarning("Warning", "No fit results to export.", parent=self.window)
            return
        
        # Ask for export options
        export_window = tk.Toplevel(self.window)
        export_window.title("Export Fit Results")
        export_window.geometry("400x300")
        
        ttk.Label(export_window, text="Export Options:", font=("TkDefaultFont", 12, "bold")).pack(pady=(10, 5))
        
        # File export frame
        file_frame = ttk.LabelFrame(export_window, text="File Export", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # File format selection
        format_frame = ttk.Frame(file_frame)
        format_frame.pack(fill=tk.X, pady=5)
        ttk.Label(format_frame, text="Format:").pack(side=tk.LEFT, padx=(0, 5))
        
        format_var = tk.StringVar(value="JSON")
        formats = ["JSON", "CSV", "TXT"]
        format_combo = ttk.Combobox(format_frame, textvariable=format_var, values=formats, width=10, state="readonly")
        format_combo.pack(side=tk.LEFT)
        
        # Export to file button
        ttk.Button(file_frame, text="Export to File", 
                  command=lambda: self._export_to_file(format_var.get())).pack(pady=5)
        
        # Database export frame
        db_frame = ttk.LabelFrame(export_window, text="Mineral Database Export", padding=10)
        db_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(db_frame, text="Export peak data to the mineral database:").pack(pady=(0, 5))
        
        # Export to database button
        ttk.Button(db_frame, text="Export to Mineral Database", 
                  command=self._export_to_mineral_database).pack(pady=5)
                  
        # Close button
        ttk.Button(export_window, text="Close", command=export_window.destroy).pack(pady=10)
        
    def _export_to_file(self, format_type):
        """Export the fitting results to a file in the specified format."""
        # Create export data dictionary
        export_data = {
            "wavenumbers": self.wavenumbers.tolist(),
            "original_spectrum": self.original_spectra.tolist(),
            "fitted_spectrum": self.fit_result.tolist() if self.fit_result is not None else None,
            "background": self.background.tolist() if self.background is not None else None,
            "residuals": self.residuals.tolist() if self.residuals is not None else None,
            "peak_model": self.current_model.get(),
            "peaks": []
        }
        
        # Get model type and parameters per peak
        model = self.current_model.get()
        if model == "Gaussian" or model == "Lorentzian":
            params_per_peak = 3
        elif model == "Pseudo-Voigt":
            params_per_peak = 4
        elif model == "Asymmetric Voigt":
            params_per_peak = 5
        else:
            params_per_peak = 3  # Default case
        
        # Calculate number of peaks
        n_peaks = len(self.fit_params) // params_per_peak
        
        # Add peak data for each peak
        for i in range(n_peaks):
            peak_data = {}
            start_idx = i * params_per_peak
            
            if model == "Gaussian":
                amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                peak_data = {
                    "index": i + 1,
                    "amplitude": float(amp),
                    "center": float(cen),
                    "width": float(wid),
                    "area": float(amp * wid * np.sqrt(2 * np.pi)),
                    "model": "Gaussian"
                }
            elif model == "Lorentzian":
                amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                peak_data = {
                    "index": i + 1,
                    "amplitude": float(amp),
                    "center": float(cen),
                    "width": float(wid),
                    "area": float(amp * wid * np.pi),
                    "model": "Lorentzian"
                }
            elif model == "Pseudo-Voigt":
                amp, cen, wid, eta = self.fit_params[start_idx:start_idx+4]
                # Area is a combination of Gaussian and Lorentzian areas
                g_area = amp * wid * np.sqrt(2 * np.pi) * (1 - eta)
                l_area = amp * wid * np.pi * eta
                peak_data = {
                    "index": i + 1,
                    "amplitude": float(amp),
                    "center": float(cen),
                    "width": float(wid),
                    "eta": float(eta),
                    "area": float(g_area + l_area),
                    "model": "Pseudo-Voigt"
                }
            elif model == "Asymmetric Voigt":
                amp, cen, wid_l, wid_r, eta = self.fit_params[start_idx:start_idx+5]
                # Approximate area
                avg_width = (wid_l + wid_r) / 2
                g_area = amp * avg_width * np.sqrt(2 * np.pi) * (1 - eta)
                l_area = amp * avg_width * np.pi * eta
                peak_data = {
                    "index": i + 1,
                    "amplitude": float(amp),
                    "center": float(cen),
                    "width_left": float(wid_l),
                    "width_right": float(wid_r),
                    "eta": float(eta),
                    "area": float(g_area + l_area),
                    "model": "Asymmetric Voigt"
                }
            
            # Add individual peak R² if available
            if hasattr(self, 'peak_r_squared') and i < len(self.peak_r_squared):
                peak_data["r_squared"] = float(self.peak_r_squared[i])
            
            export_data["peaks"].append(peak_data)
        
        # Ask for save location
        file_types = [("JSON Files", "*.json")] if format_type == "JSON" else \
                     [("CSV Files", "*.csv")] if format_type == "CSV" else \
                     [("Text Files", "*.txt")]
        default_ext = ".json" if format_type == "JSON" else ".csv" if format_type == "CSV" else ".txt"
        
        file_path = filedialog.asksaveasfilename(
            parent=self.window,
            title="Save Fit Results",
            filetypes=file_types,
            defaultextension=default_ext
        )
        
        if not file_path:
            return
        
        try:
            if format_type == "JSON":
                with open(file_path, 'w') as f:
                    import json
                    json.dump(export_data, f, indent=2)
            elif format_type == "CSV":
                import csv
                import os
                
                # Split file path to get base name and directory
                base_dir = os.path.dirname(file_path)
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                
                # Create two files: one for peak parameters and one for curve data
                peak_params_file = os.path.join(base_dir, f"{base_name}_peak_params.csv")
                curve_data_file = os.path.join(base_dir, f"{base_name}_curve_data.csv")
                
                # Write peak parameters
                with open(peak_params_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Write header
                    writer.writerow(["Peak", "Center", "Amplitude", "Width", "Area", "R_Squared", "Model"])
                    # Write peak data
                    for peak in export_data["peaks"]:
                        writer.writerow([
                            peak["index"],
                            peak["center"],
                            peak["amplitude"],
                            peak.get("width", peak.get("width_left", 0)),
                            peak["area"],
                            peak.get("r_squared", "N/A"),
                            peak["model"]
                        ])
                
                # Write curve data
                with open(curve_data_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Prepare column headers
                    headers = ["Wavenumber", "Original_Spectrum"]
                    
                    # Add columns based on available data
                    if export_data["background"] is not None:
                        headers.append("Background")
                    
                    if export_data["background"] is not None:
                        headers.append("Background_Subtracted")
                    
                    if export_data["fitted_spectrum"] is not None:
                        headers.append("Fitted_Spectrum")
                    
                    if export_data["residuals"] is not None:
                        headers.append("Residuals")
                    
                    # Calculate individual peak profiles if we have fit params
                    individual_peaks = []
                    if len(self.fit_params) > 0:
                        # Get model type and parameters per peak
                        model = self.current_model.get()
                        if model == "Gaussian" or model == "Lorentzian":
                            params_per_peak = 3
                        elif model == "Pseudo-Voigt":
                            params_per_peak = 4
                        elif model == "Asymmetric Voigt":
                            params_per_peak = 5
                        else:
                            params_per_peak = 3
                        
                        # Calculate number of peaks
                        n_peaks = len(self.fit_params) // params_per_peak
                        
                        # Add header for each peak
                        for i in range(n_peaks):
                            peak_num = i + 1
                            headers.append(f"Peak_{peak_num}")
                            
                            # Calculate individual peak profile
                            x = np.array(self.wavenumbers)
                            start_idx = i * params_per_peak
                            
                            if model == "Gaussian":
                                amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                                peak = self.gaussian(x, amp, cen, wid)
                            elif model == "Lorentzian":
                                amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                                peak = self.lorentzian(x, amp, cen, wid)
                            elif model == "Pseudo-Voigt":
                                amp, cen, wid, eta = self.fit_params[start_idx:start_idx+4]
                                peak = self.pseudo_voigt(x, amp, cen, wid, eta)
                            elif model == "Asymmetric Voigt":
                                amp, cen, wid_left, wid_right, eta = self.fit_params[start_idx:start_idx+5]
                                peak = self.asymmetric_voigt(x, amp, cen, wid_left, wid_right, eta)
                            
                            individual_peaks.append(peak.tolist())
                    
                    # Write header
                    writer.writerow(headers)
                    
                    # Prepare data rows
                    for i in range(len(export_data["wavenumbers"])):
                        row = [export_data["wavenumbers"][i], export_data["original_spectrum"][i]]
                        
                        if export_data["background"] is not None:
                            row.append(export_data["background"][i])
                        
                        if export_data["background"] is not None:
                            # Calculate background subtracted data from original spectrum and background
                            row.append(export_data["original_spectrum"][i] - export_data["background"][i])
                        
                        if export_data["fitted_spectrum"] is not None:
                            row.append(export_data["fitted_spectrum"][i])
                        
                        if export_data["residuals"] is not None:
                            row.append(export_data["residuals"][i])
                        
                        # Add individual peak values
                        for peak in individual_peaks:
                            row.append(peak[i])
                        
                        writer.writerow(row)
                
                # Update file_path to reference base file for the success message
                file_path = os.path.join(base_dir, base_name)
            else:  # TXT
                with open(file_path, 'w') as f:
                    f.write(f"Peak Fitting Results - {self.current_model.get()} Model\n")
                    f.write("-" * 80 + "\n")
                    f.write("Peak\tCenter\tAmplitude\tWidth\tArea\tR²\n")
                    for peak in export_data["peaks"]:
                        width = peak.get("width", f"{peak.get('width_left', 0)}/{peak.get('width_right', 0)}")
                        r_squared = peak.get("r_squared", "N/A")
                        f.write(f"{peak['index']}\t{peak['center']:.2f}\t{peak['amplitude']:.3f}\t{width}\t{peak['area']:.2f}\t{r_squared}\n")
            
            if format_type == "CSV":
                messagebox.showinfo("Export Successful", 
                                   f"Results exported to:\n"
                                   f"- {peak_params_file} (peak parameters)\n"
                                   f"- {curve_data_file} (curve data)", parent=self.window)
            else:
                messagebox.showinfo("Export Successful", f"Results exported to {file_path}", parent=self.window)
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to export results: {e}", parent=self.window)
            
    def _export_to_mineral_database(self):
        """Export the fitting results to the mineral database."""
        try:
            # Check if the mineral database module is available
            import importlib.util
            spec = importlib.util.find_spec("import_peaks_to_database")
            
            if spec is None:
                # If the import_peaks_to_database.py is not available, try mineral_database directly
                spec = importlib.util.find_spec("mineral_database")
                
            if spec is None:
                messagebox.showwarning(
                    "Module Not Found", 
                    "The mineral database module was not found.\n\n"
                    "Please ensure mineral_database.py is in your project directory.",
                    parent=self.window
                )
                return
                
            # If import_peaks_to_database.py is available, use it
            if importlib.util.find_spec("import_peaks_to_database"):
                import import_peaks_to_database
                tool = import_peaks_to_database.PeakImportTool(self.window)
                # Let the import tool handle the rest
            else:
                # Otherwise use mineral_database directly
                import mineral_database
                # Create a new GUI instance
                db_gui = mineral_database.MineralDatabaseGUI(self.window)
                
                # Get model type and parameters per peak
                model = self.current_model.get()
                if model == "Gaussian" or model == "Lorentzian":
                    params_per_peak = 3
                elif model == "Pseudo-Voigt":
                    params_per_peak = 4
                elif model == "Asymmetric Voigt":
                    params_per_peak = 5
                else:
                    params_per_peak = 3  # Default case
                
                # Calculate number of peaks
                n_peaks = len(self.fit_params) // params_per_peak
                
                # Get peak data
                peak_data = []
                for i in range(n_peaks):
                    start_idx = i * params_per_peak
                    
                    if model in ["Gaussian", "Lorentzian"]:
                        amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                        peak_data.append({
                            'position': cen,
                            'amplitude': amp,
                            'width': wid
                        })
                    elif model == "Pseudo-Voigt":
                        amp, cen, wid, eta = self.fit_params[start_idx:start_idx+4]
                        peak_data.append({
                            'position': cen,
                            'amplitude': amp,
                            'width': wid
                        })
                    elif model == "Asymmetric Voigt":
                        amp, cen, wid_l, wid_r, eta = self.fit_params[start_idx:start_idx+5]
                        peak_data.append({
                            'position': cen,
                            'amplitude': amp,
                            'width': (wid_l + wid_r) / 2  # Average width
                        })
                
                # Show message about importing
                messagebox.showinfo(
                    "Manual Import Required", 
                    "Please use the database GUI to manually import the peak data:\n\n"
                    "1. Add a new mineral or select an existing one\n"
                    "2. Add each peak manually with position and symmetry",
                    parent=self.window
                )
                
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to export to mineral database: {e}", parent=self.window)
    
    def enable_manual_peak_adding(self):
        """Enable/disable manual peak addition mode where user can click on the plot to add peaks."""
        try:
            # Toggle the manual peak mode
            self.manual_peak_mode = not self.manual_peak_mode
            
            if self.manual_peak_mode:
                # Enable manual peak mode
                # Ensure peaks list is initialized
                if self.peaks is None:
                    self.peaks = []
                    
                # Change cursor to indicate interactive mode
                self.canvas.get_tk_widget().config(cursor="crosshair")
                
                # Update the plot title to show instructions
                self.ax1.set_title("MANUAL MODE: Click on peaks to add them • Click the red button or press ESC to finish")
                self.canvas.draw()
                
                # Store original toolbar state
                self.original_toolbar = getattr(self.toolbar, '_active', None)
                
                # Disable toolbar to prevent interaction conflicts
                if hasattr(self.toolbar, 'mode'):
                    self.toolbar.mode = ''
                self.toolbar._active = None
                
                # Connect the click event handler
                self.click_cid = self.canvas.mpl_connect('button_press_event', self.on_plot_click)
                
                # Connect key press event to exit peak selection mode with ESC
                self.key_cid = self.canvas.mpl_connect('key_press_event', self.on_key_press)
                
                # Update button appearance to show active state
                self.manual_peak_button.configure(
                    text="🛑 Stop Adding Peaks",
                    style="Active.TButton"
                )
                
                # Create a custom style for the active button if it doesn't exist
                style = ttk.Style()
                style.configure("Active.TButton", 
                               foreground="white", 
                               background="red",
                               focuscolor="none")
                # For better cross-platform compatibility
                style.map("Active.TButton",
                         background=[('active', 'darkred'),
                                   ('pressed', 'darkred')])
                
            else:
                # Disable manual peak mode
                self.disable_manual_peak_adding()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to toggle manual peak selection: {str(e)}", parent=self.window)
            # Ensure we're not in a broken state
            self.manual_peak_mode = False
            self.canvas.get_tk_widget().config(cursor="arrow")
    
    def on_plot_click(self, event):
        """Handle click on the plot to add a peak."""
        if not hasattr(self, 'manual_peak_mode') or not self.manual_peak_mode:
            return
            
        # Check if click is within the main plot area
        if event.inaxes == self.ax1:
            x, y = event.xdata, event.ydata
            
            # Initialize peaks list if needed
            if self.peaks is None:
                self.peaks = []
            
            # Find the closest index in the wavenumbers array
            idx = np.abs(self.wavenumbers - x).argmin()
            
            # Check if a peak already exists very close to this position
            duplicate_threshold = 5.0  # cm⁻¹
            for existing_peak in self.peaks:
                if abs(existing_peak['position'] - self.wavenumbers[idx]) < duplicate_threshold:
                    # Peak already exists nearby, show message and return
                    messagebox.showinfo("Peak Exists", 
                                      f"A peak already exists near {self.wavenumbers[idx]:.1f} cm⁻¹.\n"
                                      f"Existing peak at {existing_peak['position']:.1f} cm⁻¹", 
                                      parent=self.window)
                    return
            
            # Add the peak
            self.peaks.append({
                'position': self.wavenumbers[idx],
                'intensity': self.spectra[idx],
                'index': idx
            })
            
            # Update the plot
            self.update_plot()
            
            # Update title with current count and instructions
            self.ax1.set_title(f'MANUAL MODE: {len(self.peaks)} peaks added • Click red button or press ESC to finish')
            self.canvas.draw()
            
            # Provide audio/visual feedback (brief highlight)
            self.highlight_new_peak(self.wavenumbers[idx], self.spectra[idx])
    
    def highlight_new_peak(self, x, y):
        """Briefly highlight a newly added peak."""
        try:
            # Add a temporary highlight circle
            highlight = self.ax1.plot(x, y, 'yo', markersize=15, alpha=0.7)[0]
            self.canvas.draw()
            
            # Remove the highlight after a short delay
            self.window.after(500, lambda: self.remove_highlight(highlight))
        except:
            pass  # If highlighting fails, just continue
    
    def remove_highlight(self, highlight):
        """Remove the temporary highlight."""
        try:
            highlight.remove()
            self.canvas.draw()
        except:
            pass  # If removal fails, just continue
    
    def on_key_press(self, event):
        """Handle key press to exit manual peak selection mode."""
        if event.key == 'escape':
            self.disable_manual_peak_adding()
    
    def disable_manual_peak_adding(self):
        """Disable manual peak addition mode."""
        # Restore cursor
        self.canvas.get_tk_widget().config(cursor="arrow")
        
        # Disconnect event handlers
        if hasattr(self, 'click_cid'):
            self.canvas.mpl_disconnect(self.click_cid)
        if hasattr(self, 'key_cid'):
            self.canvas.mpl_disconnect(self.key_cid)
        
        # Restore toolbar
        if hasattr(self, 'original_toolbar'):
            self.toolbar._active = self.original_toolbar
        
        # Reset flag
        self.manual_peak_mode = False
        
        # Update title to show number of peaks
        if self.peaks:
            self.ax1.set_title(f'Raman Spectrum - {len(self.peaks)} Peaks Added')
        else:
            self.ax1.set_title('Raman Spectrum')
        
        # Reset button appearance to normal state
        self.manual_peak_button.configure(
            text="Add Peaks Manually",
            style="TButton"  # Reset to default style
        )
        
        self.canvas.draw()
    
    def clear_peaks(self):
        """Clear all peaks and fitting results."""
        # Clear peaks list
        self.peaks = []
        
        # Clear fitting results
        self.fit_params = []
        self.fit_result = None
        self.residuals = None
        
        # Clear the results text
        if hasattr(self, 'results_text'):
            self.results_text.delete(1.0, tk.END)
        
        # Update the plot
        self.update_plot()
        
        # Update title
        self.ax1.set_title('Raman Spectrum - All Peaks and Fits Cleared')
        self.canvas.draw()
    
    def delete_selected_peak(self):
        """Delete the currently selected peak."""
        if not self.peaks:
            messagebox.showinfo("No Peaks", "No peaks to delete.", parent=self.window)
            return
            
        # Create a new window for peak selection
        delete_window = tk.Toplevel(self.window)
        delete_window.title("Delete Peak")
        delete_window.geometry("300x400")
        
        # Create a listbox to display peaks
        listbox = tk.Listbox(delete_window, selectmode=tk.MULTIPLE)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sort peaks by position
        sorted_peaks = sorted(self.peaks, key=lambda x: x['position'])
        
        # Add peaks to the listbox
        for i, peak in enumerate(sorted_peaks):
            listbox.insert(tk.END, f"Peak {i+1}: {peak['position']:.1f} cm⁻¹")
        
        # Store the original peak colors
        original_colors = {}
        for line in self.ax1.lines:
            if line.get_label() == 'Peak':
                original_colors[line] = line.get_color()
        
        # Function to highlight selected peaks
        def highlight_peaks(event=None):
            # Reset all peak colors to original
            for line in self.ax1.lines:
                if line.get_label() == 'Peak':
                    line.set_color(original_colors[line])
            
            # Get selected indices
            selected_indices = listbox.curselection()
            
            # Highlight selected peaks in grey
            for idx in selected_indices:
                peak = sorted_peaks[idx]
                for line in self.ax1.lines:
                    if line.get_label() == 'Peak':
                        xdata = line.get_xdata()
                        if abs(xdata[0] - peak['position']) < 0.1:  # Match peak position
                            line.set_color('grey')
            
            self.canvas.draw()
        
        # Bind selection change to highlight function
        listbox.bind('<<ListboxSelect>>', highlight_peaks)
        
        # Function to handle peak deletion
        def delete_peaks():
            selected = listbox.curselection()
            if not selected:
                messagebox.showwarning("No Selection", "Please select at least one peak to delete.", parent=self.window)
                return
            
            # Get the positions of peaks to delete
            positions_to_delete = [sorted_peaks[idx]['position'] for idx in selected]
            
            # Remove peaks with matching positions
            self.peaks = [peak for peak in self.peaks if peak['position'] not in positions_to_delete]
            
            # Update the plot
            self.update_plot()
            
            # Close the window
            delete_window.destroy()
            
            # Show confirmation
            messagebox.showinfo("Success", f"{len(selected)} peak(s) deleted successfully.", parent=self.window)
        
        # Add delete button
        delete_button = ttk.Button(delete_window, text="Delete Selected Peaks", command=delete_peaks)
        delete_button.pack(pady=10)
        
        # Add close button
        close_button = ttk.Button(delete_window, text="Cancel", command=delete_window.destroy)
        close_button.pack(pady=5)
        
        # Initial highlight of any pre-selected peaks
        highlight_peaks()
    
    def on_closing(self):
        """Handle window closing event."""
        # Clean up event handlers if manual peak mode is active
        if hasattr(self, 'manual_peak_mode') and self.manual_peak_mode:
            self.disable_manual_peak_adding()
        
        # Close the window
        self.window.destroy() 

    def calculate_peak_r_squared(self, params, model_type):
        """Calculate R-squared value for individual peaks."""
        peak_r_squared = []
        
        # Check if peaks or parameters are empty
        if not self.peaks or len(params) == 0:
            return peak_r_squared
        
        # Determine number of parameters per peak based on model type
        if model_type == "Gaussian" or model_type == "Lorentzian":
            params_per_peak = 3
        elif model_type == "Pseudo-Voigt":
            params_per_peak = 4
        elif model_type == "Asymmetric Voigt":
            params_per_peak = 5
        else:
            params_per_peak = 3
            
        n_peaks = len(params) // params_per_peak
        
        # Continue with calculation for each peak
        for peak_idx in range(n_peaks):
            try:
                # Get start index for this peak's parameters
                i = peak_idx * params_per_peak
                
                # Extract peak parameters
                if model_type == "Gaussian":
                    peak_y = self.gaussian(self.wavenumbers, *params[i:i+3])
                    center = params[i+1]
                    width = params[i+2]
                    # FWHM for Gaussian
                    fwhm = 2.355 * width
                elif model_type == "Lorentzian":
                    peak_y = self.lorentzian(self.wavenumbers, *params[i:i+3])
                    center = params[i+1]
                    width = params[i+2]
                    # FWHM for Lorentzian
                    fwhm = 2 * width
                elif model_type == "Pseudo-Voigt":
                    peak_y = self.pseudo_voigt(self.wavenumbers, *params[i:i+4])
                    center = params[i+1]
                    width = params[i+2]
                    # Approximate FWHM for Pseudo-Voigt
                    fwhm = 2 * width
                elif model_type == "Asymmetric Voigt":
                    peak_y = self.asymmetric_voigt(self.wavenumbers, *params[i:i+5])
                    center = params[i+1]
                    width_left = params[i+2]
                    width_right = params[i+3]
                    # Average FWHM for asymmetric peak
                    fwhm = width_left + width_right
                else:
                    peak_y = self.gaussian(self.wavenumbers, *params[i:i+3])
                    center = params[i+1]
                    width = params[i+2]
                    fwhm = 2.355 * width
                
                # Create a mask for a region around the peak (±2*FWHM)
                region_width = 2 * fwhm
                min_bound = center - region_width
                max_bound = center + region_width
                
                # Create mask indices for the peak region
                mask_indices = np.where(
                    np.logical_and(
                        np.greater_equal(self.wavenumbers, min_bound),
                        np.less_equal(self.wavenumbers, max_bound)
                    )
                )[0]
                
                # If mask is too narrow, widen it to ensure enough points
                if len(mask_indices) < 10:
                    region_width = 3 * fwhm
                    min_bound = center - region_width
                    max_bound = center + region_width
                    mask_indices = np.where(
                        np.logical_and(
                            np.greater_equal(self.wavenumbers, min_bound),
                            np.less_equal(self.wavenumbers, max_bound)
                        )
                    )[0]
                
                # Calculate contribution of this peak to the total fit
                if len(mask_indices) > 3:  # Need at least a few points for meaningful R²
                    # Get data, total fit, and individual peak in the region
                    y_true_region = self.spectra[mask_indices]
                    y_fit_region = self.fit_result[mask_indices]
                    peak_y_region = peak_y[mask_indices]
                    
                    # Calculate peak's contribution to the fit in this region
                    peak_contribution = np.sum(peak_y_region) / np.sum(y_fit_region) if np.sum(y_fit_region) > 0 else 0
                    
                    # Only consider regions where this peak is significant
                    if peak_contribution > 0.2:  # At least 20% contribution
                        # Calculate R² based on how well the total fit matches data in this region
                        ss_res = np.sum((y_true_region - y_fit_region) ** 2)
                        ss_tot = np.sum((y_true_region - np.mean(y_true_region)) ** 2)
                        
                        if ss_tot > 0:
                            r2 = 1 - (ss_res / ss_tot)
                        else:
                            r2 = 0.0
                    else:
                        # Peak has minimal contribution here
                        r2 = 0.0
                else:
                    r2 = 0.0
                
                peak_r_squared.append(r2)
            except Exception as e:
                # If any error occurs, add a default value
                print(f"Error calculating R² for peak {peak_idx}: {e}")
                peak_r_squared.append(0.0)
        
        return peak_r_squared