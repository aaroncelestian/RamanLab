#!/usr/bin/env python3
"""
Multi-Spectrum Manager for RamanLab

@version: 2.6.3
@author: Aaron Celestian, Ph.D.
@copyright: © 2025 Aaron Celestian. All rights reserved
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import to_hex
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser
import json
from pathlib import Path
from raman_spectra import RamanSpectra


class MultiSpectrumManager:
    """
    A comprehensive multi-spectrum visualization and manipulation tool.
    Serves as a data playground for comparing and analyzing multiple Raman spectra.
    """
    
    def __init__(self, parent=None):
        """Initialize the Multi-Spectrum Manager window."""
        self.parent = parent
        self.loaded_spectra = {}  # Dictionary to store spectrum data
        self.spectrum_settings = {}  # Dictionary to store per-spectrum settings
        self.global_settings = {
            'normalize': True,
            'show_legend': True,
            'grid': True,
            'line_width': 1.5,
            'x_label': 'Wavenumber (cm⁻¹)',
            'y_label': 'Intensity (a.u.)',
            'title': 'Multi-Spectrum Analysis'
        }
        
        self.create_window()
        self.setup_ui()
        
    def create_window(self):
        """Create the main window."""
        self.window = tk.Toplevel(self.parent) if self.parent else tk.Tk()
        self.window.title("Multi-Spectrum Manager - Data Playground")
        self.window.geometry("1600x1000")
        self.window.minsize(1200, 800)
        
        # Set icon if available
        try:
            if self.parent and hasattr(self.parent, 'iconbitmap'):
                self.window.iconbitmap(self.parent.iconbitmap())
        except:
            pass
            
    def setup_ui(self):
        """Set up the user interface."""
        # Create main container
        main_container = ttk.Frame(self.window)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create paned window for resizable sections
        main_paned = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls (30% width)
        self.left_panel = ttk.Frame(main_paned)
        main_paned.add(self.left_panel, weight=1)
        
        # Right panel - Visualization (70% width)
        self.right_panel = ttk.Frame(main_paned)
        main_paned.add(self.right_panel, weight=2)
        
        # Setup left panel
        self.setup_control_panel()
        
        # Setup right panel
        self.setup_visualization_panel()
        
    def setup_control_panel(self):
        """Set up the control panel with tabbed interface for better organization."""
        # First, create the loaded spectra section that will be visible on both tabs
        self.setup_loaded_spectra_section()
        
        # Create notebook for tabs
        self.control_notebook = ttk.Notebook(self.left_panel)
        self.control_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.file_tab = ttk.Frame(self.control_notebook)
        self.spectrum_tab = ttk.Frame(self.control_notebook)
        
        # Add tabs to notebook
        self.control_notebook.add(self.file_tab, text="File Operations")
        self.control_notebook.add(self.spectrum_tab, text="Spectrum Controls")
        
        # Setup tab contents
        self.setup_file_operations_tab()
        self.setup_spectrum_controls_tab()
        
    def setup_loaded_spectra_section(self):
        """Set up the loaded spectra section that appears on both tabs."""
        # Loaded Spectra section (always visible)
        self.loaded_spectra_frame = ttk.LabelFrame(self.left_panel, text="Loaded Spectra", padding=10)
        self.loaded_spectra_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Spectrum listbox with scrollbar - fixed height
        listbox_frame = ttk.Frame(self.loaded_spectra_frame)
        listbox_frame.pack(fill=tk.X)
        
        self.spectrum_listbox = tk.Listbox(listbox_frame, selectmode=tk.SINGLE, height=6)
        self.spectrum_listbox.pack(fill=tk.X, side=tk.LEFT, expand=True)
        self.spectrum_listbox.bind('<<ListboxSelect>>', self.on_spectrum_select)
        
        list_scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, 
                                      command=self.spectrum_listbox.yview)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.spectrum_listbox.configure(yscrollcommand=list_scrollbar.set)
        
        # Quick action buttons
        quick_actions_frame = ttk.Frame(self.loaded_spectra_frame)
        quick_actions_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(quick_actions_frame, text="Remove Selected", 
                  command=self.remove_selected_spectrum).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(quick_actions_frame, text="Clear All", 
                  command=self.clear_all_spectra).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
    def setup_file_operations_tab(self):
        """Set up the file operations tab."""
        # File Operations
        file_frame = ttk.LabelFrame(self.file_tab, text="Load & Save", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Load Multiple Files", 
                  command=self.load_multiple_files).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Add Single File", 
                  command=self.add_single_file).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Save Session", 
                  command=self.save_session).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Load Session", 
                  command=self.load_session).pack(fill=tk.X, pady=2)
        
        # Spectrum Management
        mgmt_frame = ttk.LabelFrame(self.file_tab, text="Spectrum Management", padding=10)
        mgmt_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(mgmt_frame, text="Duplicate Selected", 
                  command=self.duplicate_spectrum).pack(fill=tk.X, pady=2)
        
        # Export Options
        export_frame = ttk.LabelFrame(self.file_tab, text="Export", padding=10)
        export_frame.pack(fill=tk.X, pady=(0, 10))
        
        export_buttons_frame = ttk.Frame(export_frame)
        export_buttons_frame.pack(fill=tk.X)
        ttk.Button(export_buttons_frame, text="Save Plot", 
                  command=self.save_plot).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(export_buttons_frame, text="Export Data", 
                  command=self.export_data).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
    def setup_spectrum_controls_tab(self):
        """Set up the spectrum controls tab."""
        # Global Settings
        global_frame = ttk.LabelFrame(self.spectrum_tab, text="Global Settings", padding=10)
        global_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Normalization
        self.normalize_var = tk.BooleanVar(value=self.global_settings['normalize'])
        ttk.Checkbutton(global_frame, text="Normalize Spectra", 
                       variable=self.normalize_var, 
                       command=self.update_plot).pack(anchor=tk.W)
        
        # Show Legend
        self.legend_var = tk.BooleanVar(value=self.global_settings['show_legend'])
        ttk.Checkbutton(global_frame, text="Show Legend", 
                       variable=self.legend_var, 
                       command=self.update_plot).pack(anchor=tk.W)
        
        # Grid
        self.grid_var = tk.BooleanVar(value=self.global_settings['grid'])
        ttk.Checkbutton(global_frame, text="Show Grid", 
                       variable=self.grid_var, 
                       command=self.update_plot).pack(anchor=tk.W)
        
        # Line Width
        ttk.Label(global_frame, text="Line Width:").pack(anchor=tk.W, pady=(10, 0))
        self.line_width_var = tk.DoubleVar(value=self.global_settings['line_width'])
        line_width_scale = ttk.Scale(global_frame, from_=0.5, to=5.0, 
                                    variable=self.line_width_var, 
                                    orient=tk.HORIZONTAL, 
                                    command=lambda x: self.update_plot())
        line_width_scale.pack(fill=tk.X, pady=(0, 10))
        
        # Individual Spectrum Controls (will be populated when spectrum is selected)
        self.individual_frame = ttk.LabelFrame(self.spectrum_tab, text="Selected Spectrum Controls", padding=10)
        self.individual_frame.pack(fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.no_selection_label = ttk.Label(self.individual_frame, 
                                           text="Select a spectrum from the list above\nto see individual controls")
        self.no_selection_label.pack(pady=20)
        
    def setup_visualization_panel(self):
        """Set up the visualization panel with matplotlib."""
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_xlabel(self.global_settings['x_label'])
        self.ax.set_ylabel(self.global_settings['y_label'])
        self.ax.set_title(self.global_settings['title'])
        self.ax.grid(self.global_settings['grid'], alpha=0.3)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(self.right_panel)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
    def load_multiple_files(self):
        """Load multiple spectrum files."""
        file_paths = filedialog.askopenfilenames(
            title="Select Multiple Spectrum Files",
            filetypes=[
                ("Text Files", "*.txt"),
                ("CSV Files", "*.csv"),
                ("All Files", "*.*"),
            ]
        )
        
        if not file_paths:
            return
            
        self.load_files(file_paths)
        
    def add_single_file(self):
        """Add a single spectrum file."""
        file_path = filedialog.askopenfilename(
            title="Select Spectrum File",
            filetypes=[
                ("Text Files", "*.txt"),
                ("CSV Files", "*.csv"),
                ("All Files", "*.*"),
            ]
        )
        
        if file_path:
            self.load_files([file_path])
            
    def load_files(self, file_paths):
        """Load spectrum files and add them to the manager."""
        success_count = 0
        error_files = []
        
        for file_path in file_paths:
            try:
                # Create temporary RamanSpectra instance
                temp_raman = RamanSpectra()
                success = temp_raman.import_spectrum(file_path)
                
                if success and temp_raman.current_spectra is not None:
                    # Generate unique name
                    filename = os.path.basename(file_path)
                    display_name = os.path.splitext(filename)[0]
                    
                    counter = 1
                    original_name = display_name
                    while display_name in self.loaded_spectra:
                        display_name = f"{original_name}_{counter}"
                        counter += 1
                    
                    # Store spectrum data
                    self.loaded_spectra[display_name] = {
                        'wavenumbers': temp_raman.current_wavenumbers.copy(),
                        'intensities': temp_raman.current_spectra.copy(),
                        'original_intensities': temp_raman.current_spectra.copy(),
                        'metadata': temp_raman.metadata.copy() if temp_raman.metadata else {},
                        'file_path': file_path
                    }
                    
                    # Initialize spectrum settings
                    self.spectrum_settings[display_name] = {
                        'visible': True,
                        'color': self.get_next_color(),
                        'transparency': 0.0,
                        'x_shift': 0.0,
                        'y_shift': 0.0,
                        'y_scale': 1.0,
                        'line_style': '-',
                        'marker': 'None'
                    }
                    
                    # Add to listbox
                    self.spectrum_listbox.insert(tk.END, display_name)
                    success_count += 1
                    
                else:
                    error_files.append(os.path.basename(file_path))
                    
            except Exception as e:
                error_files.append(f"{os.path.basename(file_path)} ({str(e)})")
        
        # Update display
        if success_count > 0:
            self.update_plot()
            self.window.title(f"Multi-Spectrum Manager - {len(self.loaded_spectra)} spectra loaded")
        
        if error_files:
            error_msg = f"Failed to load {len(error_files)} files:\n" + "\n".join(error_files[:5])
            if len(error_files) > 5:
                error_msg += f"\n... and {len(error_files) - 5} more"
            messagebox.showwarning("Loading Errors", error_msg)
            
    def get_next_color(self):
        """Get the next color in the cycle for new spectra."""
        colors = plt.cm.tab10.colors
        used_colors = [settings['color'] for settings in self.spectrum_settings.values()]
        
        for color in colors:
            color_hex = to_hex(color)
            if color_hex not in used_colors:
                return color_hex
                
        # If all colors used, return a random one
        return to_hex(colors[len(self.loaded_spectra) % len(colors)])
        
    def on_spectrum_select(self, event=None):
        """Handle spectrum selection in the listbox."""
        selection = self.spectrum_listbox.curselection()
        if not selection:
            self.clear_individual_controls()
            return
            
        spectrum_name = self.spectrum_listbox.get(selection[0])
        self.setup_individual_controls(spectrum_name)
        
    def clear_individual_controls(self):
        """Clear individual spectrum controls."""
        for widget in self.individual_frame.winfo_children():
            widget.destroy()
            
        self.no_selection_label = ttk.Label(self.individual_frame, 
                                           text="Select a spectrum from the list above\nto see individual controls")
        self.no_selection_label.pack(pady=20)
        
    def setup_individual_controls(self, spectrum_name):
        """Set up controls for individual spectrum manipulation."""
        # Clear existing controls
        for widget in self.individual_frame.winfo_children():
            widget.destroy()
            
        if spectrum_name not in self.spectrum_settings:
            return
            
        settings = self.spectrum_settings[spectrum_name]
        
        # Title
        title_label = ttk.Label(self.individual_frame, text=f"Controls: {spectrum_name}", 
                               font=('TkDefaultFont', 10, 'bold'))
        title_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Create a frame to hold the canvas and scrollbar
        scroll_container = ttk.Frame(self.individual_frame)
        scroll_container.pack(fill=tk.BOTH, expand=True, padx=(0, 5), pady=5)  # Add padding to prevent scrollbar from being hidden
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(scroll_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Visibility
        visible_var = tk.BooleanVar(value=settings['visible'])
        visible_check = ttk.Checkbutton(scrollable_frame, text="Visible", 
                                       variable=visible_var,
                                       command=lambda: self.update_spectrum_setting(spectrum_name, 'visible', visible_var.get()))
        visible_check.pack(anchor=tk.W, pady=(0, 10))
        
        # Color Selection
        color_frame = ttk.LabelFrame(scrollable_frame, text="Color", padding=5)
        color_frame.pack(fill=tk.X, pady=(0, 10), padx=2)  # Add small horizontal padding
        
        color_display_frame = ttk.Frame(color_frame)
        color_display_frame.pack(fill=tk.X)
        ttk.Label(color_display_frame, text="Current Color:").pack(side=tk.LEFT)
        
        color_button = tk.Button(color_display_frame, width=4, height=1, 
                                bg=settings['color'], 
                                command=lambda: self.choose_color(spectrum_name))
        color_button.pack(side=tk.LEFT, padx=(10, 0))
        
        # Line Style
        line_frame = ttk.LabelFrame(scrollable_frame, text="Line Style", padding=5)
        line_frame.pack(fill=tk.X, pady=(0, 10), padx=2)  # Add small horizontal padding
        
        line_style_var = tk.StringVar(value=settings['line_style'])
        line_style_combo = ttk.Combobox(line_frame, textvariable=line_style_var,
                                       values=['-', '--', '-.', ':'],
                                       state="readonly", width=15)
        line_style_combo.pack(anchor=tk.W, fill=tk.X, expand=True)  # Make combo fill width
        line_style_combo.bind('<<ComboboxSelected>>', 
                             lambda e: self.update_spectrum_setting(spectrum_name, 'line_style', line_style_var.get()))
        
        # Transparency
        transparency_frame = ttk.LabelFrame(scrollable_frame, text="Transparency", padding=5)
        transparency_frame.pack(fill=tk.X, pady=(0, 10), padx=2)  # Add small horizontal padding
        
        transparency_var = tk.DoubleVar(value=settings['transparency'])
        transparency_label = ttk.Label(transparency_frame, text=f"Value: {settings['transparency']:.2f}")
        transparency_label.pack(anchor=tk.W)
        
        transparency_scale = ttk.Scale(transparency_frame, from_=0.0, to=0.9, 
                                      variable=transparency_var, orient=tk.HORIZONTAL,
                                      command=lambda x: self.update_transparency(spectrum_name, transparency_var, transparency_label))
        transparency_scale.pack(fill=tk.X, padx=2)  # Add padding and ensure full width
        
        # X-axis Offset
        x_offset_frame = ttk.LabelFrame(scrollable_frame, text="X-axis Offset (cm⁻¹)", padding=5)
        x_offset_frame.pack(fill=tk.X, pady=(0, 10), padx=2)  # Add small horizontal padding
        
        x_shift_var = tk.DoubleVar(value=settings['x_shift'])
        x_shift_label = ttk.Label(x_offset_frame, text=f"Value: {settings['x_shift']:.1f} cm⁻¹")
        x_shift_label.pack(anchor=tk.W)
        
        x_shift_scale = ttk.Scale(x_offset_frame, from_=-500, to=500, 
                                 variable=x_shift_var, orient=tk.HORIZONTAL,
                                 command=lambda x: self.update_x_shift(spectrum_name, x_shift_var, x_shift_label))
        x_shift_scale.pack(fill=tk.X, padx=2)  # Add padding and ensure full width
        
        # Y-axis Offset
        y_offset_frame = ttk.LabelFrame(scrollable_frame, text="Y-axis Offset", padding=5)
        y_offset_frame.pack(fill=tk.X, pady=(0, 10), padx=2)  # Add small horizontal padding
        
        y_shift_var = tk.DoubleVar(value=settings['y_shift'])
        y_shift_label = ttk.Label(y_offset_frame, text=f"Value: {settings['y_shift']:.2f}")
        y_shift_label.pack(anchor=tk.W)
        
        y_shift_scale = ttk.Scale(y_offset_frame, from_=-2.0, to=2.0, 
                                 variable=y_shift_var, orient=tk.HORIZONTAL,
                                 command=lambda x: self.update_y_shift(spectrum_name, y_shift_var, y_shift_label))
        y_shift_scale.pack(fill=tk.X, padx=2)  # Add padding and ensure full width
        
        # Y-axis Scale
        y_scale_frame = ttk.LabelFrame(scrollable_frame, text="Y-axis Scale", padding=5)
        y_scale_frame.pack(fill=tk.X, pady=(0, 10), padx=2)  # Add small horizontal padding
        
        y_scale_var = tk.DoubleVar(value=settings['y_scale'])
        y_scale_label = ttk.Label(y_scale_frame, text=f"Value: {settings['y_scale']:.2f}")
        y_scale_label.pack(anchor=tk.W)
        
        y_scale_scale = ttk.Scale(y_scale_frame, from_=0.1, to=5.0, 
                                 variable=y_scale_var, orient=tk.HORIZONTAL,
                                 command=lambda x: self.update_y_scale(spectrum_name, y_scale_var, y_scale_label))
        y_scale_scale.pack(fill=tk.X, padx=2)  # Add padding and ensure full width
        
        # Reset button
        reset_frame = ttk.Frame(scrollable_frame)
        reset_frame.pack(fill=tk.X, pady=(20, 10))
        
        ttk.Button(reset_frame, text="Reset All Transformations", 
                  command=lambda: self.reset_spectrum_transformations(spectrum_name)).pack(fill=tk.X)
        
        # Pack the canvas and scrollbar with proper spacing
        canvas.pack(side="left", fill="both", expand=True, padx=(0, 2))  # Small gap before scrollbar
        scrollbar.pack(side="right", fill="y", padx=(0, 2))  # Small padding from edge
        
        # Configure canvas scrolling region after packing
        canvas.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
        
        # Bind mousewheel events for scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        # Bind mouse enter/leave events to enable/disable scrolling
        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)
        
        # Also bind to the frame for better interaction
        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        scrollable_frame.bind('<Configure>', _on_frame_configure)
        
    def choose_color(self, spectrum_name):
        """Open color chooser for spectrum."""
        current_color = self.spectrum_settings[spectrum_name]['color']
        color = colorchooser.askcolor(color=current_color, title=f"Choose color for {spectrum_name}")
        
        if color[1]:  # If user didn't cancel
            self.update_spectrum_setting(spectrum_name, 'color', color[1])
            # Update the color button
            self.setup_individual_controls(spectrum_name)
            
    def update_spectrum_setting(self, spectrum_name, setting, value):
        """Update a setting for a specific spectrum."""
        if spectrum_name in self.spectrum_settings:
            self.spectrum_settings[spectrum_name][setting] = value
            self.update_plot()
            
    def reset_spectrum_transformations(self, spectrum_name):
        """Reset all transformations for a spectrum."""
        if spectrum_name in self.spectrum_settings:
            self.spectrum_settings[spectrum_name].update({
                'x_shift': 0.0,
                'y_shift': 0.0,
                'y_scale': 1.0,
                'transparency': 0.0
            })
            self.setup_individual_controls(spectrum_name)
            self.update_plot()
            
    def update_plot(self):
        """Update the main plot with all visible spectra."""
        self.ax.clear()
        
        # Apply global settings
        self.ax.set_xlabel(self.global_settings['x_label'])
        self.ax.set_ylabel(self.global_settings['y_label'])
        self.ax.set_title(self.global_settings['title'])
        
        # Fix grid toggle - properly handle the boolean value
        grid_enabled = self.grid_var.get()
        if grid_enabled:
            self.ax.grid(True, alpha=0.3)
        else:
            self.ax.grid(False)
        
        normalize = self.normalize_var.get()
        line_width = self.line_width_var.get()
        
        # Plot each visible spectrum
        for spectrum_name, data in self.loaded_spectra.items():
            if spectrum_name not in self.spectrum_settings:
                continue
                
            settings = self.spectrum_settings[spectrum_name]
            if not settings['visible']:
                continue
                
            # Get data
            wavenumbers = data['wavenumbers'] + settings['x_shift']
            intensities = data['intensities'].copy()
            
            # Apply y-scale first
            intensities = intensities * settings['y_scale']
            
            # Normalize if requested (BEFORE applying y-shift)
            if normalize and np.max(intensities) > 0:
                intensities = intensities / np.max(intensities)
            
            # Apply y-shift AFTER normalization
            intensities = intensities + settings['y_shift']
                
            # Plot
            alpha = 1.0 - settings['transparency']
            self.ax.plot(wavenumbers, intensities,
                        color=settings['color'],
                        alpha=alpha,
                        linewidth=line_width,
                        linestyle=settings['line_style'],
                        label=spectrum_name)
        
        # Show legend if requested
        if self.legend_var.get() and self.loaded_spectra:
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
        plt.tight_layout()
        self.canvas.draw()
        
    def remove_selected_spectrum(self):
        """Remove the selected spectrum."""
        selection = self.spectrum_listbox.curselection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select a spectrum to remove.")
            return
            
        spectrum_name = self.spectrum_listbox.get(selection[0])
        
        # Remove from data structures
        if spectrum_name in self.loaded_spectra:
            del self.loaded_spectra[spectrum_name]
        if spectrum_name in self.spectrum_settings:
            del self.spectrum_settings[spectrum_name]
            
        # Remove from listbox
        self.spectrum_listbox.delete(selection[0])
        
        # Clear individual controls
        self.clear_individual_controls()
        
        # Update plot
        self.update_plot()
        
        # Update title
        self.window.title(f"Multi-Spectrum Manager - {len(self.loaded_spectra)} spectra loaded")
        
    def duplicate_spectrum(self):
        """Duplicate the selected spectrum."""
        selection = self.spectrum_listbox.curselection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select a spectrum to duplicate.")
            return
            
        spectrum_name = self.spectrum_listbox.get(selection[0])
        
        if spectrum_name not in self.loaded_spectra:
            return
            
        # Create new name
        base_name = f"{spectrum_name}_copy"
        new_name = base_name
        counter = 1
        while new_name in self.loaded_spectra:
            new_name = f"{base_name}_{counter}"
            counter += 1
            
        # Duplicate data
        original_data = self.loaded_spectra[spectrum_name]
        self.loaded_spectra[new_name] = {
            'wavenumbers': original_data['wavenumbers'].copy(),
            'intensities': original_data['intensities'].copy(),
            'original_intensities': original_data['original_intensities'].copy(),
            'metadata': original_data['metadata'].copy(),
            'file_path': original_data['file_path']
        }
        
        # Duplicate settings with new color
        original_settings = self.spectrum_settings[spectrum_name]
        self.spectrum_settings[new_name] = original_settings.copy()
        self.spectrum_settings[new_name]['color'] = self.get_next_color()
        
        # Add to listbox
        self.spectrum_listbox.insert(tk.END, new_name)
        
        # Update plot
        self.update_plot()
        
    def clear_all_spectra(self):
        """Clear all loaded spectra."""
        if not self.loaded_spectra:
            return
            
        result = messagebox.askyesno("Confirm Clear", 
                                   f"Are you sure you want to remove all {len(self.loaded_spectra)} spectra?")
        if result:
            self.loaded_spectra.clear()
            self.spectrum_settings.clear()
            self.spectrum_listbox.delete(0, tk.END)
            self.clear_individual_controls()
            self.update_plot()
            self.window.title("Multi-Spectrum Manager - Data Playground")
            
    def save_session(self):
        """Save the current session to a file."""
        if not self.loaded_spectra:
            messagebox.showinfo("No Data", "No spectra loaded to save.")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Session",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            session_data = {
                'loaded_spectra': {},
                'spectrum_settings': self.spectrum_settings,
                'global_settings': {
                    'normalize': self.normalize_var.get(),
                    'show_legend': self.legend_var.get(),
                    'grid': self.grid_var.get(),
                    'line_width': self.line_width_var.get()
                }
            }
            
            # Convert numpy arrays to lists for JSON serialization
            for name, data in self.loaded_spectra.items():
                session_data['loaded_spectra'][name] = {
                    'wavenumbers': data['wavenumbers'].tolist(),
                    'intensities': data['intensities'].tolist(),
                    'original_intensities': data['original_intensities'].tolist(),
                    'metadata': data['metadata'],
                    'file_path': data['file_path']
                }
                
            with open(file_path, 'w') as f:
                json.dump(session_data, f, indent=2)
                
            messagebox.showinfo("Success", f"Session saved to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save session: {str(e)}")
            
    def load_session(self):
        """Load a session from a file."""
        file_path = filedialog.askopenfilename(
            title="Load Session",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'r') as f:
                session_data = json.load(f)
                
            # Clear current data
            self.loaded_spectra.clear()
            self.spectrum_settings.clear()
            self.spectrum_listbox.delete(0, tk.END)
            
            # Load spectra data
            for name, data in session_data['loaded_spectra'].items():
                self.loaded_spectra[name] = {
                    'wavenumbers': np.array(data['wavenumbers']),
                    'intensities': np.array(data['intensities']),
                    'original_intensities': np.array(data['original_intensities']),
                    'metadata': data['metadata'],
                    'file_path': data['file_path']
                }
                self.spectrum_listbox.insert(tk.END, name)
                
            # Load settings
            self.spectrum_settings = session_data['spectrum_settings']
            
            # Load global settings
            global_settings = session_data.get('global_settings', {})
            self.normalize_var.set(global_settings.get('normalize', True))
            self.legend_var.set(global_settings.get('show_legend', True))
            self.grid_var.set(global_settings.get('grid', True))
            self.line_width_var.set(global_settings.get('line_width', 1.5))
            
            # Update display
            self.clear_individual_controls()
            self.update_plot()
            self.window.title(f"Multi-Spectrum Manager - {len(self.loaded_spectra)} spectra loaded")
            
            messagebox.showinfo("Success", f"Session loaded from {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load session: {str(e)}")
            
    def save_plot(self):
        """Save the current plot to a file."""
        file_path = filedialog.asksaveasfilename(
            title="Save Plot",
            defaultextension=".png",
            filetypes=[
                ("PNG Files", "*.png"),
                ("PDF Files", "*.pdf"),
                ("SVG Files", "*.svg"),
                ("EPS Files", "*.eps"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot: {str(e)}")
                
    def export_data(self):
        """Export all visible spectrum data to a CSV file."""
        if not self.loaded_spectra:
            messagebox.showinfo("No Data", "No spectra loaded to export.")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Export Data",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            import pandas as pd
            
            # Collect all visible spectra data
            export_data = {}
            
            for spectrum_name, data in self.loaded_spectra.items():
                if spectrum_name not in self.spectrum_settings:
                    continue
                    
                settings = self.spectrum_settings[spectrum_name]
                if not settings['visible']:
                    continue
                    
                # Apply transformations
                wavenumbers = data['wavenumbers'] + settings['x_shift']
                intensities = data['intensities'].copy()
                
                # Apply y-scale first
                intensities = intensities * settings['y_scale']
                
                # Normalize if requested (BEFORE applying y-shift)
                if self.normalize_var.get() and np.max(intensities) > 0:
                    intensities = intensities / np.max(intensities)
                
                # Apply y-shift AFTER normalization
                intensities = intensities + settings['y_shift']
                
                export_data[f"{spectrum_name}_wavenumber"] = wavenumbers
                export_data[f"{spectrum_name}_intensity"] = intensities
                
            if export_data:
                df = pd.DataFrame(export_data)
                df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Data exported to {file_path}")
            else:
                messagebox.showinfo("No Data", "No visible spectra to export.")
                
        except ImportError:
            messagebox.showerror("Error", "pandas is required for data export. Please install it.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export data: {str(e)}")

    def update_transparency(self, spectrum_name, var, label):
        """Update transparency setting and label."""
        value = var.get()
        self.update_spectrum_setting(spectrum_name, 'transparency', value)
        label.config(text=f"Value: {value:.2f}")
        
    def update_x_shift(self, spectrum_name, var, label):
        """Update x-axis shift setting and label."""
        value = var.get()
        self.update_spectrum_setting(spectrum_name, 'x_shift', value)
        label.config(text=f"Value: {value:.1f} cm⁻¹")
        
    def update_y_shift(self, spectrum_name, var, label):
        """Update y-axis shift setting and label."""
        value = var.get()
        self.update_spectrum_setting(spectrum_name, 'y_shift', value)
        label.config(text=f"Value: {value:.2f}")
        
    def update_y_scale(self, spectrum_name, var, label):
        """Update y-axis scale setting and label."""
        value = var.get()
        self.update_spectrum_setting(spectrum_name, 'y_scale', value)
        label.config(text=f"Value: {value:.2f}")


def main():
    """Main function for standalone testing."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    app = MultiSpectrumManager()
    app.window.mainloop()


if __name__ == "__main__":
    main() 