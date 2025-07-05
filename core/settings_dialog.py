#!/usr/bin/env python3
"""
RamanLab Settings Dialog
Provides a user interface for configuring application settings
"""

import os
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QLabel, 
    QPushButton, QLineEdit, QFileDialog, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QMessageBox,
    QDialogButtonBox, QFrame, QTextEdit, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from .config_manager import get_config_manager


class SettingsDialog(QDialog):
    """Settings dialog for configuring RamanLab preferences."""
    
    # Signal emitted when settings are changed
    settings_changed = Signal()
    
    def __init__(self, parent=None):
        """Initialize the settings dialog."""
        super().__init__(parent)
        self.config = get_config_manager()
        self.setWindowTitle("RamanLab Settings")
        self.setMinimumSize(600, 500)
        self.resize(750, 600)
        
        self.setup_ui()
        self.load_current_settings()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_general_tab()
        self.create_paths_tab()
        self.create_appearance_tab()
        self.create_analysis_tab()
        self.create_advanced_tab()
        
        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults
        )
        button_box.accepted.connect(self.accept_settings)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.restore_defaults)
        layout.addWidget(button_box)
    
    def create_general_tab(self):
        """Create the general settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Auto-save settings
        auto_save_group = QGroupBox("Auto-Save Settings")
        auto_save_layout = QFormLayout(auto_save_group)
        
        self.auto_save_enabled = QCheckBox("Enable auto-save")
        auto_save_layout.addRow("Auto-Save:", self.auto_save_enabled)
        
        self.auto_save_interval = QSpinBox()
        self.auto_save_interval.setRange(60, 3600)  # 1 minute to 1 hour
        self.auto_save_interval.setSuffix(" seconds")
        auto_save_layout.addRow("Auto-Save Interval:", self.auto_save_interval)
        
        layout.addWidget(auto_save_group)
        
        # Recent files settings
        recent_files_group = QGroupBox("Recent Files")
        recent_files_layout = QFormLayout(recent_files_group)
        
        self.max_recent_files = QSpinBox()
        self.max_recent_files.setRange(5, 50)
        recent_files_layout.addRow("Maximum Recent Files:", self.max_recent_files)
        
        clear_recent_btn = QPushButton("Clear Recent Files")
        clear_recent_btn.clicked.connect(self.clear_recent_files)
        recent_files_layout.addRow("", clear_recent_btn)
        
        layout.addWidget(recent_files_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "General")
    
    def create_paths_tab(self):
        """Create the paths settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Projects folder
        projects_group = QGroupBox("Projects Folder")
        projects_layout = QVBoxLayout(projects_group)
        
        # Description
        desc_label = QLabel(
            "Choose where RamanLab will store your projects, sessions, and analysis results. "
            "This folder will contain subfolders for auto-saves, exports, and other data."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; font-size: 11px; margin-bottom: 10px;")
        projects_layout.addWidget(desc_label)
        
        # Current path display and browse button
        path_layout = QHBoxLayout()
        self.projects_path_edit = QLineEdit()
        self.projects_path_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_projects_folder)
        
        path_layout.addWidget(QLabel("Projects Folder:"))
        path_layout.addWidget(self.projects_path_edit)
        path_layout.addWidget(browse_btn)
        projects_layout.addLayout(path_layout)
        
        # Open folder button
        open_folder_btn = QPushButton("📁 Open Projects Folder")
        open_folder_btn.clicked.connect(self.open_projects_folder)
        projects_layout.addWidget(open_folder_btn)
        
        layout.addWidget(projects_group)
        
        # Database paths
        database_group = QGroupBox("Database Paths")
        database_layout = QFormLayout(database_group)
        
        self.main_database_edit = QLineEdit()
        browse_main_db_btn = QPushButton("Browse...")
        browse_main_db_btn.clicked.connect(lambda: self.browse_database_file("main_database"))
        
        db_layout = QHBoxLayout()
        db_layout.addWidget(self.main_database_edit)
        db_layout.addWidget(browse_main_db_btn)
        database_layout.addRow("Main Database:", db_layout)
        
        self.mineral_modes_edit = QLineEdit()
        browse_modes_btn = QPushButton("Browse...")
        browse_modes_btn.clicked.connect(lambda: self.browse_database_file("mineral_modes"))
        
        modes_layout = QHBoxLayout()
        modes_layout.addWidget(self.mineral_modes_edit)
        modes_layout.addWidget(browse_modes_btn)
        database_layout.addRow("Mineral Modes:", modes_layout)
        
        layout.addWidget(database_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Paths")
    
    def create_appearance_tab(self):
        """Create the appearance settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Plot settings
        plot_group = QGroupBox("Plot Settings")
        plot_layout = QFormLayout(plot_group)
        
        self.plot_dpi = QSpinBox()
        self.plot_dpi.setRange(50, 300)
        plot_layout.addRow("Plot DPI:", self.plot_dpi)
        
        self.line_width = QDoubleSpinBox()
        self.line_width.setRange(0.5, 5.0)
        self.line_width.setSingleStep(0.1)
        plot_layout.addRow("Line Width:", self.line_width)
        
        self.grid_alpha = QDoubleSpinBox()
        self.grid_alpha.setRange(0.0, 1.0)
        self.grid_alpha.setSingleStep(0.1)
        plot_layout.addRow("Grid Transparency:", self.grid_alpha)
        
        layout.addWidget(plot_group)
        
        # Theme settings (placeholder for future implementation)
        theme_group = QGroupBox("Theme (Future Feature)")
        theme_layout = QFormLayout(theme_group)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Default", "Dark", "Light"])
        self.theme_combo.setEnabled(False)  # Disabled for now
        theme_layout.addRow("Theme:", self.theme_combo)
        
        layout.addWidget(theme_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Appearance")
    
    def create_analysis_tab(self):
        """Create the analysis settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Peak detection settings
        peak_group = QGroupBox("Peak Detection Defaults")
        peak_layout = QFormLayout(peak_group)
        
        self.peak_height = QDoubleSpinBox()
        self.peak_height.setRange(0.01, 1.0)
        self.peak_height.setSingleStep(0.01)
        self.peak_height.setDecimals(3)
        peak_layout.addRow("Minimum Height:", self.peak_height)
        
        self.peak_prominence = QDoubleSpinBox()
        self.peak_prominence.setRange(0.01, 1.0)
        self.peak_prominence.setSingleStep(0.01)
        self.peak_prominence.setDecimals(3)
        peak_layout.addRow("Minimum Prominence:", self.peak_prominence)
        
        self.peak_distance = QSpinBox()
        self.peak_distance.setRange(1, 100)
        peak_layout.addRow("Minimum Distance:", self.peak_distance)
        
        layout.addWidget(peak_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Analysis")
    
    def create_advanced_tab(self):
        """Create the advanced settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Configuration info
        info_group = QGroupBox("Configuration Information")
        info_layout = QVBoxLayout(info_group)
        
        config_path_label = QLabel(f"Config File: {self.config.config_file}")
        config_path_label.setWordWrap(True)
        config_path_label.setStyleSheet("font-family: monospace; font-size: 10px;")
        info_layout.addWidget(config_path_label)
        
        # Configuration preview
        self.config_preview = QTextEdit()
        self.config_preview.setReadOnly(True)
        self.config_preview.setMaximumHeight(200)
        self.config_preview.setStyleSheet("font-family: monospace; font-size: 10px;")
        info_layout.addWidget(QLabel("Current Configuration:"))
        info_layout.addWidget(self.config_preview)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh Preview")
        refresh_btn.clicked.connect(self.refresh_config_preview)
        info_layout.addWidget(refresh_btn)
        
        layout.addWidget(info_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Advanced")
    
    def load_current_settings(self):
        """Load current settings into the dialog."""
        # General settings
        self.auto_save_enabled.setChecked(self.config.get("auto_save_enabled", True))
        self.auto_save_interval.setValue(self.config.get("auto_save_interval", 300))
        self.max_recent_files.setValue(self.config.get("max_recent_files", 10))
        
        # Paths
        self.projects_path_edit.setText(self.config.get("projects_folder"))
        self.main_database_edit.setText(self.config.get("database_paths.main_database", ""))
        self.mineral_modes_edit.setText(self.config.get("database_paths.mineral_modes", ""))
        
        # Appearance
        self.plot_dpi.setValue(self.config.get("plot_settings.dpi", 100))
        self.line_width.setValue(self.config.get("plot_settings.line_width", 1.0))
        self.grid_alpha.setValue(self.config.get("plot_settings.grid_alpha", 0.3))
        
        # Analysis
        self.peak_height.setValue(self.config.get("peak_detection.height", 0.1))
        self.peak_prominence.setValue(self.config.get("peak_detection.prominence", 0.05))
        self.peak_distance.setValue(self.config.get("peak_detection.distance", 10))
        
        # Refresh config preview
        self.refresh_config_preview()
    
    def browse_projects_folder(self):
        """Browse for projects folder."""
        current_path = self.projects_path_edit.text()
        folder = QFileDialog.getExistingDirectory(
            self, 
            "Select Projects Folder",
            current_path if current_path else str(Path.home())
        )
        if folder:
            self.projects_path_edit.setText(folder)
    
    def browse_database_file(self, db_type):
        """Browse for database file."""
        if db_type == "main_database":
            edit_widget = self.main_database_edit
            title = "Select Main Database File"
        else:
            edit_widget = self.mineral_modes_edit
            title = "Select Mineral Modes Database File"
        
        current_path = edit_widget.text()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            current_path if current_path else "",
            "Pickle files (*.pkl);;All files (*.*)"
        )
        if file_path:
            edit_widget.setText(file_path)
    
    def open_projects_folder(self):
        """Open the projects folder in the file explorer."""
        import subprocess
        import platform
        
        folder_path = Path(self.projects_path_edit.text())
        folder_path.mkdir(parents=True, exist_ok=True)
        
        try:
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(folder_path)])
            elif platform.system() == "Windows":
                subprocess.run(["explorer", str(folder_path)])
            else:  # Linux
                subprocess.run(["xdg-open", str(folder_path)])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open folder: {e}")
    
    def clear_recent_files(self):
        """Clear the recent files list."""
        reply = QMessageBox.question(
            self, 
            "Clear Recent Files", 
            "This will clear all recent files from the list. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.config.set("recent_files", [])
            QMessageBox.information(self, "Success", "Recent files list cleared.")
    
    def refresh_config_preview(self):
        """Refresh the configuration preview."""
        import json
        try:
            config_text = json.dumps(self.config._config, indent=2)
            self.config_preview.setPlainText(config_text)
        except Exception as e:
            self.config_preview.setPlainText(f"Error displaying config: {e}")
    
    def accept_settings(self):
        """Accept and save the settings."""
        try:
            # Save all settings
            self.config.set("auto_save_enabled", self.auto_save_enabled.isChecked())
            self.config.set("auto_save_interval", self.auto_save_interval.value())
            self.config.set("max_recent_files", self.max_recent_files.value())
            
            # Paths
            self.config.set_projects_folder(self.projects_path_edit.text())
            self.config.set("database_paths.main_database", self.main_database_edit.text())
            self.config.set("database_paths.mineral_modes", self.mineral_modes_edit.text())
            
            # Appearance
            self.config.set("plot_settings.dpi", self.plot_dpi.value())
            self.config.set("plot_settings.line_width", self.line_width.value())
            self.config.set("plot_settings.grid_alpha", self.grid_alpha.value())
            
            # Analysis
            self.config.set("peak_detection.height", self.peak_height.value())
            self.config.set("peak_detection.prominence", self.peak_prominence.value())
            self.config.set("peak_detection.distance", self.peak_distance.value())
            
            # Emit signal that settings changed
            self.settings_changed.emit()
            
            QMessageBox.information(self, "Success", "Settings saved successfully.")
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {e}")
    
    def restore_defaults(self):
        """Restore default settings."""
        reply = QMessageBox.question(
            self, 
            "Restore Defaults", 
            "This will reset all settings to their default values. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.config.reset_to_defaults()
            self.load_current_settings()
            QMessageBox.information(self, "Success", "Settings restored to defaults.") 