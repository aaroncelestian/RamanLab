#!/usr/bin/env python3
"""
RamanLab Qt6 - Multi-Spectrum Manager Window
Separate window for comprehensive multi-spectrum analysis
"""

import os
import numpy as np
from pathlib import Path

# Matplotlib imports
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from ui.matplotlib_config import CompactNavigationToolbar as NavigationToolbar
import matplotlib.pyplot as plt

# Qt6 imports
from PySide6.QtWidgets import (
    QDialog, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QLineEdit, QTextEdit, QSlider, QCheckBox, QComboBox,
    QGroupBox, QSplitter, QFileDialog, QMessageBox, QProgressBar,
    QListWidget, QListWidgetItem, QInputDialog, QColorDialog, QSpinBox,
    QDoubleSpinBox, QScrollArea
)
from PySide6.QtCore import Qt, QStandardPaths
from PySide6.QtGui import QFont, QColor


class MultiSpectrumManagerQt6(QMainWindow):
    """Qt6 Multi-Spectrum Manager Window for RamanLab."""
    
    def __init__(self, parent=None, raman_db=None):
        """
        Initialize the multi-spectrum manager.
        
        Parameters:
        -----------
        parent : QWidget, optional
            Parent widget
        raman_db : RamanSpectraQt6, optional
            Database instance
        """
        super().__init__(parent)
        self.parent_app = parent
        self.raman_db = raman_db
        
        # Initialize data storage
        self.loaded_spectra = {}
        self.spectrum_settings = {}  # Store individual spectrum settings
        self.global_settings = {
            'normalize': True,
            'show_legend': True,
            'grid': True,
            'line_width': 1.5,
            'waterfall_mode': False,
            'waterfall_spacing': 1.0,
            'colormap': 'tab10'
        }
        
        # Set window properties
        self.setWindowTitle("RamanLab Qt6 - Multi-Spectrum Manager")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # Create the UI
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - Controls (30% width)
        self.create_left_panel(splitter)
        
        # Right panel - Visualization (70% width)
        self.create_right_panel(splitter)
        
        # Set splitter proportions
        splitter.setSizes([400, 1000])
    
    def create_left_panel(self, parent):
        """Create the left control panel."""
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Loaded Spectra section - improved sizing
        spectra_group = QGroupBox("Loaded Spectra")
        spectra_layout = QVBoxLayout(spectra_group)
        
        self.multi_spectrum_list = QListWidget()
        # Remove fixed height, let it expand with available space
        self.multi_spectrum_list.setMinimumHeight(120)
        self.multi_spectrum_list.currentItemChanged.connect(self.on_multi_spectrum_select)
        spectra_layout.addWidget(self.multi_spectrum_list)
        
        # Quick action buttons
        quick_buttons = QHBoxLayout()
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_selected_multi_spectrum)
        quick_buttons.addWidget(remove_btn)
        
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self.clear_all_multi_spectra)
        quick_buttons.addWidget(clear_all_btn)
        
        spectra_layout.addLayout(quick_buttons)
        left_layout.addWidget(spectra_group)
        
        # Control tabs
        control_tabs = QTabWidget()
        left_layout.addWidget(control_tabs)
        
        # File Operations Tab
        self.create_file_operations_tab(control_tabs)
        
        # Spectrum Controls Tab
        self.create_spectrum_controls_tab(control_tabs)
        
        parent.addWidget(left_panel)
    
    def create_file_operations_tab(self, parent_tabs):
        """Create the file operations tab."""
        file_tab = QWidget()
        file_layout = QVBoxLayout(file_tab)
        
        # Load & Save operations
        load_group = QGroupBox("Load & Save")
        load_layout = QVBoxLayout(load_group)
        
        load_multiple_btn = QPushButton("Load Multiple Files")
        load_multiple_btn.clicked.connect(self.load_multiple_files)
        load_layout.addWidget(load_multiple_btn)
        
        add_single_btn = QPushButton("Add Single File")
        add_single_btn.clicked.connect(self.add_single_file)
        load_layout.addWidget(add_single_btn)
        
        # Enhanced database search section
        db_search_group = QGroupBox("Database Search")
        db_search_layout = QVBoxLayout(db_search_group)
        
        # Search field
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.db_search_field = QLineEdit()
        self.db_search_field.setPlaceholderText("Enter mineral name, formula, or classification...")
        self.db_search_field.textChanged.connect(self.filter_database_results)
        search_layout.addWidget(self.db_search_field)
        db_search_layout.addLayout(search_layout)
        
        # Search results list
        self.db_results_list = QListWidget()
        self.db_results_list.setMaximumHeight(120)
        self.db_results_list.itemDoubleClicked.connect(self.add_selected_from_database)
        db_search_layout.addWidget(self.db_results_list)
        
        # Add button
        add_from_db_btn = QPushButton("Add Selected from Database")
        add_from_db_btn.clicked.connect(self.add_selected_from_database)
        db_search_layout.addWidget(add_from_db_btn)
        
        load_layout.addWidget(db_search_group)
        
        load_current_btn = QPushButton("Add Current Spectrum")
        load_current_btn.clicked.connect(self.add_current_spectrum)
        load_layout.addWidget(load_current_btn)
        
        file_layout.addWidget(load_group)
        
        # Export operations
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        
        export_buttons = QHBoxLayout()
        save_plot_btn = QPushButton("Save Plot")
        save_plot_btn.clicked.connect(self.save_multi_plot)
        export_buttons.addWidget(save_plot_btn)
        
        export_data_btn = QPushButton("Export Data")
        export_data_btn.clicked.connect(self.export_multi_data)
        export_buttons.addWidget(export_data_btn)
        
        export_layout.addLayout(export_buttons)
        file_layout.addWidget(export_group)
        
        # Initialize database search
        self.populate_database_list()
        
        file_layout.addStretch()
        parent_tabs.addTab(file_tab, "File Operations")
    
    def create_spectrum_controls_tab(self, parent_tabs):
        """Create the spectrum controls tab."""
        spectrum_tab = QWidget()
        spectrum_layout = QVBoxLayout(spectrum_tab)
        
        # Global Settings
        global_group = QGroupBox("Global Settings")
        global_layout = QVBoxLayout(global_group)
        
        # Basic display options
        display_row1 = QHBoxLayout()
        self.normalize_checkbox = QCheckBox("Normalize Spectra")
        self.normalize_checkbox.setChecked(self.global_settings['normalize'])
        self.normalize_checkbox.toggled.connect(self.update_multi_plot)
        display_row1.addWidget(self.normalize_checkbox)
        
        self.legend_checkbox = QCheckBox("Show Legend")
        self.legend_checkbox.setChecked(self.global_settings['show_legend'])
        self.legend_checkbox.toggled.connect(self.update_multi_plot)
        display_row1.addWidget(self.legend_checkbox)
        
        self.grid_checkbox = QCheckBox("Show Grid")
        self.grid_checkbox.setChecked(self.global_settings['grid'])
        self.grid_checkbox.toggled.connect(self.update_multi_plot)
        display_row1.addWidget(self.grid_checkbox)
        global_layout.addLayout(display_row1)
        
        # Waterfall mode
        waterfall_row = QHBoxLayout()
        self.waterfall_checkbox = QCheckBox("Waterfall Plot")
        self.waterfall_checkbox.setChecked(self.global_settings['waterfall_mode'])
        self.waterfall_checkbox.toggled.connect(self.update_multi_plot)
        waterfall_row.addWidget(self.waterfall_checkbox)
        
        waterfall_row.addWidget(QLabel("Spacing:"))
        self.waterfall_spacing_slider = QSlider(Qt.Horizontal)
        self.waterfall_spacing_slider.setRange(1, 50)  # 0.1 to 5.0 (multiplied by 10)
        self.waterfall_spacing_slider.setValue(int(self.global_settings['waterfall_spacing'] * 10))
        self.waterfall_spacing_slider.valueChanged.connect(self.update_multi_plot)
        waterfall_row.addWidget(self.waterfall_spacing_slider)
        
        self.waterfall_spacing_label = QLabel(f"{self.global_settings['waterfall_spacing']:.1f}")
        waterfall_row.addWidget(self.waterfall_spacing_label)
        global_layout.addLayout(waterfall_row)
        
        # Line width control
        line_width_layout = QHBoxLayout()
        line_width_layout.addWidget(QLabel("Line Width:"))
        self.line_width_slider = QSlider(Qt.Horizontal)
        self.line_width_slider.setRange(5, 50)  # 0.5 to 5.0 (multiplied by 10)
        self.line_width_slider.setValue(int(self.global_settings['line_width'] * 10))
        self.line_width_slider.valueChanged.connect(self.update_multi_plot)
        line_width_layout.addWidget(self.line_width_slider)
        
        self.line_width_label = QLabel(f"{self.global_settings['line_width']:.1f}")
        line_width_layout.addWidget(self.line_width_label)
        global_layout.addLayout(line_width_layout)
        
        # Colormap theme selection
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(QLabel("Colormap Theme:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            'tab10', 'tab20', 'Set1', 'Set2', 'Set3', 'Dark2', 'Paired',
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Blues', 'Reds', 'Greens', 'Oranges', 'Purples',
            'coolwarm', 'RdYlBu', 'spectral', 'rainbow'
        ])
        self.colormap_combo.setCurrentText(self.global_settings['colormap'])
        self.colormap_combo.currentTextChanged.connect(self.update_multi_plot)
        colormap_layout.addWidget(self.colormap_combo)
        
        # Reset colors button
        reset_colors_btn = QPushButton("Reset Colors")
        reset_colors_btn.clicked.connect(self.reset_all_colors)
        colormap_layout.addWidget(reset_colors_btn)
        global_layout.addLayout(colormap_layout)
        
        spectrum_layout.addWidget(global_group)
        
        # Individual Spectrum Controls - make scrollable
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.individual_widget = QWidget()
        self.individual_layout = QVBoxLayout(self.individual_widget)
        
        self.individual_group = QGroupBox("Selected Spectrum Controls")
        self.individual_controls_layout = QVBoxLayout(self.individual_group)
        
        self.no_selection_label = QLabel("Select a spectrum from the list above\nto see individual controls")
        self.no_selection_label.setAlignment(Qt.AlignCenter)
        self.individual_controls_layout.addWidget(self.no_selection_label)
        
        self.individual_layout.addWidget(self.individual_group)
        self.individual_layout.addStretch()
        
        scroll_area.setWidget(self.individual_widget)
        spectrum_layout.addWidget(scroll_area)
        
        parent_tabs.addTab(spectrum_tab, "Spectrum Controls")
    
    def create_right_panel(self, parent):
        """Create the right visualization panel."""
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Create matplotlib figure for multi-spectrum plot
        self.multi_figure = Figure(figsize=(12, 8))
        self.multi_canvas = FigureCanvas(self.multi_figure)
        self.multi_toolbar = NavigationToolbar(self.multi_canvas, right_panel)
        
        right_layout.addWidget(self.multi_toolbar)
        right_layout.addWidget(self.multi_canvas)
        
        # Create the plot
        self.multi_ax = self.multi_figure.add_subplot(111)
        self.multi_ax.set_xlabel("Wavenumber (cm⁻¹)")
        self.multi_ax.set_ylabel("Intensity (a.u.)")
        self.multi_ax.set_title("Multi-Spectrum Analysis")
        self.multi_ax.grid(True, alpha=0.3)
        
        # Initial message
        self.multi_ax.text(0.5, 0.5, 
                          'Multi-Spectrum Manager\n\nLoad spectra using the controls on the left\n'
                          'to start your comparative analysis',
                          ha='center', va='center', fontsize=14, fontweight='bold',
                          transform=self.multi_ax.transAxes)
        
        parent.addWidget(right_panel)
    
    def get_spectrum_setting(self, spectrum_name, setting, default):
        """Get a setting for a specific spectrum."""
        if spectrum_name not in self.spectrum_settings:
            self.spectrum_settings[spectrum_name] = {}
        return self.spectrum_settings[spectrum_name].get(setting, default)
    
    def set_spectrum_setting(self, spectrum_name, setting, value):
        """Set a setting for a specific spectrum."""
        if spectrum_name not in self.spectrum_settings:
            self.spectrum_settings[spectrum_name] = {}
        self.spectrum_settings[spectrum_name][setting] = value
    
    def get_colormap_colors(self, n_colors):
        """Get colors from the selected colormap."""
        colormap_name = self.colormap_combo.currentText()
        try:
            if colormap_name.startswith('tab') or colormap_name in ['Set1', 'Set2', 'Set3', 'Dark2', 'Paired']:
                # Qualitative colormaps
                cmap = plt.get_cmap(colormap_name)
                colors = [cmap(i) for i in range(min(n_colors, cmap.N))]
            else:
                # Continuous colormaps
                cmap = plt.get_cmap(colormap_name)
                colors = [cmap(i / max(1, n_colors - 1)) for i in range(n_colors)]
            return colors
        except:
            # Fallback to default colors
            default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            return default_colors[:n_colors]
    
    def update_multi_plot(self):
        """Update multi-spectrum plot based on global settings."""
        if not self.loaded_spectra:
            self.multi_ax.clear()
            self.multi_ax.set_xlabel("Wavenumber (cm⁻¹)")
            self.multi_ax.set_ylabel("Intensity (a.u.)")
            self.multi_ax.set_title("Multi-Spectrum Analysis")
            self.multi_ax.text(0.5, 0.5, 
                              'Multi-Spectrum Manager\n\nLoad spectra using the controls on the left\n'
                              'to start your comparative analysis',
                              ha='center', va='center', fontsize=14, fontweight='bold',
                              transform=self.multi_ax.transAxes)
            self.multi_canvas.draw()
            return
            
        self.multi_ax.clear()
        
        # Update labels
        line_width = self.line_width_slider.value() / 10.0
        self.line_width_label.setText(f"{line_width:.1f}")
        
        waterfall_spacing = self.waterfall_spacing_slider.value() / 10.0
        self.waterfall_spacing_label.setText(f"{waterfall_spacing:.1f}")
        
        # Update global settings
        self.global_settings['normalize'] = self.normalize_checkbox.isChecked()
        self.global_settings['show_legend'] = self.legend_checkbox.isChecked()
        self.global_settings['grid'] = self.grid_checkbox.isChecked()
        self.global_settings['line_width'] = line_width
        self.global_settings['waterfall_mode'] = self.waterfall_checkbox.isChecked()
        self.global_settings['waterfall_spacing'] = waterfall_spacing
        self.global_settings['colormap'] = self.colormap_combo.currentText()
        
        # Get colors from colormap
        colors = self.get_colormap_colors(len(self.loaded_spectra))
        
        # Plot all spectra
        y_offset = 0
        for i, (spectrum_name, spectrum_data) in enumerate(self.loaded_spectra.items()):
            wavenumbers, intensities = spectrum_data
            
            # Get individual spectrum settings
            custom_color = self.get_spectrum_setting(spectrum_name, 'color', None)
            y_offset_individual = self.get_spectrum_setting(spectrum_name, 'y_offset', 0.0)
            x_offset_individual = self.get_spectrum_setting(spectrum_name, 'x_offset', 0.0)
            
            # Apply offsets
            plot_wavenumbers = wavenumbers + x_offset_individual
            plot_intensities = intensities.copy()
            
            # Normalize if requested
            if self.global_settings['normalize'] and np.max(plot_intensities) > 0:
                plot_intensities = plot_intensities / np.max(plot_intensities)
            
            # Apply y-offset (individual + waterfall)
            if self.global_settings['waterfall_mode']:
                plot_intensities += y_offset + y_offset_individual
                y_offset += waterfall_spacing
            else:
                plot_intensities += y_offset_individual
            
            # Use custom color if set, otherwise use colormap
            color = custom_color if custom_color else colors[i % len(colors)]
            
            self.multi_ax.plot(plot_wavenumbers, plot_intensities, 
                             color=color, 
                             linewidth=self.global_settings['line_width'],
                             label=spectrum_name)
        
        # Apply settings
        if self.global_settings['show_legend'] and self.loaded_spectra:
            self.multi_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Properly handle grid visibility
        if self.global_settings['grid']:
            self.multi_ax.grid(True, alpha=0.3)
        else:
            self.multi_ax.grid(False)
        
        self.multi_ax.set_xlabel("Wavenumber (cm⁻¹)")
        ylabel = "Intensity (a.u.)"
        if self.global_settings['waterfall_mode']:
            ylabel += " (Offset)"
        self.multi_ax.set_ylabel(ylabel)
        self.multi_ax.set_title("Multi-Spectrum Analysis")
        
        self.multi_figure.tight_layout()
        self.multi_canvas.draw()
    
    def on_multi_spectrum_select(self, current, previous):
        """Handle selection of a multi-spectrum."""
        if current:
            spectrum_name = current.text()
            self.update_individual_controls(spectrum_name)
    
    def update_individual_controls(self, spectrum_name):
        """Update individual spectrum controls."""
        # Clear existing controls
        for i in reversed(range(self.individual_controls_layout.count())):
            child = self.individual_controls_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        if spectrum_name not in self.loaded_spectra:
            self.no_selection_label = QLabel("Select a spectrum from the list above\nto see individual controls")
            self.no_selection_label.setAlignment(Qt.AlignCenter)
            self.individual_controls_layout.addWidget(self.no_selection_label)
            return
        
        # Title
        title_label = QLabel(f"Controls: {spectrum_name}")
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.individual_controls_layout.addWidget(title_label)
        
        # Color selection
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Color:"))
        
        color_btn = QPushButton("Choose Color")
        current_color = self.get_spectrum_setting(spectrum_name, 'color', None)
        if current_color:
            # Convert matplotlib color to QColor for display
            if isinstance(current_color, str):
                color_btn.setStyleSheet(f"background-color: {current_color};")
            else:
                # Handle RGB tuple
                rgb = [int(c * 255) for c in current_color[:3]]
                color_btn.setStyleSheet(f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]});")
        
        color_btn.clicked.connect(lambda: self.choose_spectrum_color(spectrum_name, color_btn))
        color_layout.addWidget(color_btn)
        
        reset_color_btn = QPushButton("Reset")
        reset_color_btn.clicked.connect(lambda: self.reset_spectrum_color(spectrum_name, color_btn))
        color_layout.addWidget(reset_color_btn)
        self.individual_controls_layout.addWidget(QWidget())  # Spacer
        color_widget = QWidget()
        color_widget.setLayout(color_layout)
        self.individual_controls_layout.addWidget(color_widget)
        
        # Y-offset control
        y_offset_layout = QHBoxLayout()
        y_offset_layout.addWidget(QLabel("Y-Offset:"))
        
        y_offset_slider = QSlider(Qt.Horizontal)
        y_offset_slider.setRange(-100, 100)  # -10.0 to 10.0
        current_y_offset = self.get_spectrum_setting(spectrum_name, 'y_offset', 0.0)
        y_offset_slider.setValue(int(current_y_offset * 10))
        
        y_offset_label = QLabel(f"{current_y_offset:.1f}")
        
        def update_y_offset(value):
            new_offset = value / 10.0
            y_offset_label.setText(f"{new_offset:.1f}")
            self.set_spectrum_setting(spectrum_name, 'y_offset', new_offset)
            self.update_multi_plot()
        
        y_offset_slider.valueChanged.connect(update_y_offset)
        y_offset_layout.addWidget(y_offset_slider)
        y_offset_layout.addWidget(y_offset_label)
        
        y_offset_widget = QWidget()
        y_offset_widget.setLayout(y_offset_layout)
        self.individual_controls_layout.addWidget(y_offset_widget)
        
        # X-offset control
        x_offset_layout = QHBoxLayout()
        x_offset_layout.addWidget(QLabel("X-Offset:"))
        
        x_offset_slider = QSlider(Qt.Horizontal)
        x_offset_slider.setRange(-1000, 1000)  # -100.0 to 100.0 cm⁻¹
        current_x_offset = self.get_spectrum_setting(spectrum_name, 'x_offset', 0.0)
        x_offset_slider.setValue(int(current_x_offset * 10))
        
        x_offset_label = QLabel(f"{current_x_offset:.1f}")
        
        def update_x_offset(value):
            new_offset = value / 10.0
            x_offset_label.setText(f"{new_offset:.1f}")
            self.set_spectrum_setting(spectrum_name, 'x_offset', new_offset)
            self.update_multi_plot()
        
        x_offset_slider.valueChanged.connect(update_x_offset)
        x_offset_layout.addWidget(x_offset_slider)
        x_offset_layout.addWidget(x_offset_label)
        
        x_offset_widget = QWidget()
        x_offset_widget.setLayout(x_offset_layout)
        self.individual_controls_layout.addWidget(x_offset_widget)
        
        # Reset offsets button
        reset_offsets_btn = QPushButton("Reset All Offsets")
        reset_offsets_btn.clicked.connect(lambda: self.reset_spectrum_offsets(spectrum_name))
        self.individual_controls_layout.addWidget(reset_offsets_btn)
        
        # Add stretch to push everything to top
        self.individual_controls_layout.addStretch()
    
    def choose_spectrum_color(self, spectrum_name, color_btn):
        """Open color picker for spectrum."""
        color = QColorDialog.getColor()
        if color.isValid():
            # Convert QColor to matplotlib color
            rgb = (color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0)
            self.set_spectrum_setting(spectrum_name, 'color', rgb)
            
            # Update button appearance
            color_btn.setStyleSheet(f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});")
            
            # Update plot
            self.update_multi_plot()
    
    def reset_spectrum_color(self, spectrum_name, color_btn):
        """Reset spectrum color to colormap default."""
        self.set_spectrum_setting(spectrum_name, 'color', None)
        color_btn.setStyleSheet("")  # Reset button style
        self.update_multi_plot()
    
    def reset_spectrum_offsets(self, spectrum_name):
        """Reset all offsets for a spectrum."""
        self.set_spectrum_setting(spectrum_name, 'y_offset', 0.0)
        self.set_spectrum_setting(spectrum_name, 'x_offset', 0.0)
        self.update_individual_controls(spectrum_name)  # Refresh controls
        self.update_multi_plot()
    
    def reset_all_colors(self):
        """Reset all spectrum colors to colormap defaults."""
        for spectrum_name in self.loaded_spectra.keys():
            self.set_spectrum_setting(spectrum_name, 'color', None)
        
        # Refresh individual controls if a spectrum is selected
        current_item = self.multi_spectrum_list.currentItem()
        if current_item:
            self.update_individual_controls(current_item.text())
        
        self.update_multi_plot()
    
    def populate_database_list(self):
        """Populate the database results list with all available spectra."""
        self.db_results_list.clear()
        
        if not self.raman_db or not hasattr(self.raman_db, 'database') or not self.raman_db.database:
            return
        
        for spectrum_name in self.raman_db.database.keys():
            self.db_results_list.addItem(spectrum_name)
    
    def filter_database_results(self):
        """Filter database results based on search text."""
        search_text = self.db_search_field.text().lower()
        self.db_results_list.clear()
        
        if not self.raman_db or not hasattr(self.raman_db, 'database') or not self.raman_db.database:
            return
        
        for spectrum_name, spectrum_data in self.raman_db.database.items():
            # Search in spectrum name
            if search_text in spectrum_name.lower():
                self.db_results_list.addItem(spectrum_name)
                continue
            
            # Search in metadata
            metadata = spectrum_data.get('metadata', {})
            for key, value in metadata.items():
                if isinstance(value, str) and search_text in value.lower():
                    self.db_results_list.addItem(spectrum_name)
                    break
    
    def add_selected_from_database(self):
        """Add selected spectrum from database search results."""
        current_item = self.db_results_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "No Selection", "Please select a spectrum from the search results.")
            return
        
        spectrum_name = current_item.text()
        
        if not self.raman_db or spectrum_name not in self.raman_db.database:
            QMessageBox.warning(self, "Spectrum Not Found", f"Spectrum '{spectrum_name}' not found in database.")
            return
        
        spectrum_data = self.raman_db.database[spectrum_name]
        wavenumbers = np.array(spectrum_data['wavenumbers'])
        intensities = np.array(spectrum_data['intensities'])
        
        # Generate unique name for multi-spectrum manager
        display_name = spectrum_name
        counter = 1
        while display_name in self.loaded_spectra:
            display_name = f"{spectrum_name}_{counter}"
            counter += 1
        
        self.loaded_spectra[display_name] = (wavenumbers, intensities)
        self.multi_spectrum_list.addItem(display_name)
        self.update_multi_plot()
        self.setWindowTitle(f"Multi-Spectrum Manager - {len(self.loaded_spectra)} spectra loaded")
    
    def remove_spectrum_by_name(self, spectrum_name):
        """Remove a specific spectrum by name."""
        if spectrum_name in self.loaded_spectra:
            del self.loaded_spectra[spectrum_name]
            
            # Remove settings
            if spectrum_name in self.spectrum_settings:
                del self.spectrum_settings[spectrum_name]
            
            # Remove from list widget
            for i in range(self.multi_spectrum_list.count()):
                if self.multi_spectrum_list.item(i).text() == spectrum_name:
                    self.multi_spectrum_list.takeItem(i)
                    break
            
            self.update_multi_plot()
            self.update_individual_controls("")  # Clear controls
    
    def remove_selected_multi_spectrum(self):
        """Remove selected multi-spectrum."""
        current_item = self.multi_spectrum_list.currentItem()
        if current_item:
            spectrum_name = current_item.text()
            self.remove_spectrum_by_name(spectrum_name)
        else:
            QMessageBox.information(self, "No Selection", "Please select a spectrum to remove.")
    
    def clear_all_multi_spectra(self):
        """Clear all multi-spectra."""
        if not self.loaded_spectra:
            return
            
        reply = QMessageBox.question(
            self,
            "Confirm Clear",
            f"Are you sure you want to remove all {len(self.loaded_spectra)} spectra?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.loaded_spectra.clear()
            self.spectrum_settings.clear()
            self.multi_spectrum_list.clear()
            self.update_multi_plot()
            self.update_individual_controls("")
    
    def load_spectrum_file_for_multi(self, file_path):
        """Load a spectrum file into the multi-spectrum manager."""
        try:
            # Load the file data
            data = np.loadtxt(file_path)
            if data.ndim == 2 and data.shape[1] >= 2:
                wavenumbers = data[:, 0]
                intensities = data[:, 1]
                
                # Generate unique name
                filename = Path(file_path).stem
                display_name = filename
                counter = 1
                while display_name in self.loaded_spectra:
                    display_name = f"{filename}_{counter}"
                    counter += 1
                
                # Store spectrum
                self.loaded_spectra[display_name] = (wavenumbers, intensities)
                
                # Add to list
                self.multi_spectrum_list.addItem(display_name)
                
                # Update plot
                self.update_multi_plot()
                
                self.setWindowTitle(f"Multi-Spectrum Manager - {len(self.loaded_spectra)} spectra loaded")
                
            else:
                QMessageBox.warning(self, "Invalid Format", f"File must contain at least two columns: {Path(file_path).name}")
                
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load {Path(file_path).name}:\n{str(e)}")
    
    def load_multiple_files(self):
        """Load multiple spectrum files."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Multiple Raman Spectra",
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
            "Text files (*.txt *.csv *.dat);;All files (*.*)"
        )
        
        if file_paths:
            success_count = 0
            for file_path in file_paths:
                try:
                    self.load_spectrum_file_for_multi(file_path)
                    success_count += 1
                except:
                    pass  # Error already handled in load_spectrum_file_for_multi
            
            if success_count > 0:
                QMessageBox.information(self, "Success", f"Successfully loaded {success_count} of {len(file_paths)} files.")
    
    def add_single_file(self):
        """Add a single spectrum file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Raman Spectrum",
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
            "Text files (*.txt *.csv *.dat);;All files (*.*)"
        )
        
        if file_path:
            self.load_spectrum_file_for_multi(file_path)
    
    def add_from_database(self):
        """Add a spectrum from the database (legacy method for backward compatibility)."""
        self.add_selected_from_database()
    
    def add_current_spectrum(self):
        """Add the current spectrum from the main application."""
        if not self.parent_app:
            QMessageBox.warning(self, "No Main App", "No connection to main application.")
            return
            
        if (not hasattr(self.parent_app, 'current_wavenumbers') or 
            self.parent_app.current_wavenumbers is None or 
            self.parent_app.current_intensities is None):
            QMessageBox.warning(self, "No Data", "No spectrum loaded in main application.")
            return
        
        # Generate unique name
        spectrum_name = f"Current_Spectrum_{len(self.loaded_spectra) + 1}"
        counter = 1
        base_name = spectrum_name
        while spectrum_name in self.loaded_spectra:
            spectrum_name = f"{base_name}_{counter}"
            counter += 1
        
        self.loaded_spectra[spectrum_name] = (
            self.parent_app.current_wavenumbers.copy(), 
            self.parent_app.processed_intensities.copy()
        )
        self.multi_spectrum_list.addItem(spectrum_name)
        self.update_multi_plot()
        self.setWindowTitle(f"Multi-Spectrum Manager - {len(self.loaded_spectra)} spectra loaded")
    
    def save_multi_plot(self):
        """Save the multi-spectrum plot."""
        if not self.loaded_spectra:
            QMessageBox.warning(self, "No Data", "No spectra to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Multi-Spectrum Plot",
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
            "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg);;All files (*.*)"
        )
        
        if file_path:
            try:
                self.multi_figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Plot saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save plot:\n{str(e)}")
    
    def export_multi_data(self):
        """Export multi-spectrum data to a file."""
        if not self.loaded_spectra:
            QMessageBox.warning(self, "No Data", "No spectra to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Multi-Spectrum Data",
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
            "Text files (*.txt);;CSV files (*.csv);;All files (*.*)"
        )
        
        if file_path:
            try:
                # Create header
                header = "Wavenumber"
                for name in self.loaded_spectra.keys():
                    header += f"\t{name}"
                
                # Export data (simplified version)
                with open(file_path, 'w') as f:
                    f.write(header + "\n")
                    f.write("# Multi-spectrum data export\n")
                    f.write("# Individual spectrum data follows...\n")
                    
                    for name, (wavenumbers, intensities) in self.loaded_spectra.items():
                        f.write(f"\n# {name}\n")
                        for wn, intensity in zip(wavenumbers, intensities):
                            f.write(f"{wn:.2f}\t{intensity:.6e}\n")
                
                QMessageBox.information(self, "Success", f"Data exported successfully!")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export data:\n{str(e)}") 