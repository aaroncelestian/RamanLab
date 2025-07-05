"""
File Selection Tab Component
Handles file loading, navigation, and display for the Batch Peak Fitting interface
Extracted from the original batch_peak_fitting_qt6.py monolithic file
"""

import os
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QListWidget, 
    QPushButton, QLabel, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt

from ..base_tab import BaseTab

# Unified button style for consistent UI
BUTTON_STYLE = """
    QPushButton {
        background-color: #f8f9fa;
        color: #495057;
        border: 1px solid #dee2e6;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 11px;
        min-width: 50px;
    }
    QPushButton:hover {
        background-color: #e9ecef;
        border: 1px solid #adb5bd;
        color: #212529;
    }
    QPushButton:pressed {
        background-color: #dee2e6;
        border: 1px solid #6c757d;
    }
    QPushButton:disabled {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        color: #6c757d;
        opacity: 0.6;
    }
"""

PRIMARY_BUTTON_STYLE = """
    QPushButton {
        background-color: #0d6efd;
        color: white;
        border: 1px solid #0a58ca;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 11px;
        min-width: 50px;
    }
    QPushButton:hover {
        background-color: #0b5ed7;
        border: 1px solid #09408e;
    }
    QPushButton:pressed {
        background-color: #0a58ca;
        border: 1px solid #08356d;
    }
"""

DANGER_BUTTON_STYLE = """
    QPushButton {
        background-color: #dc3545;
        color: white;
        border: 1px solid #bb2d3b;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 11px;
        min-width: 50px;
    }
    QPushButton:hover {
        background-color: #c82333;
        border: 1px solid #a02834;
    }
    QPushButton:pressed {
        background-color: #bb2d3b;
        border: 1px solid #8d2130;
    }
"""


class FileTab(BaseTab):
    """
    File selection and navigation component.
    Focused solely on file management functionality.
    """
    
    def __init__(self, parent=None):
        # Initialize widgets that will be created
        self.file_list_widget = None
        self.current_file_label = None
        
        super().__init__(parent)
        self.tab_name = "File"
    
    def setup_ui(self):
        """Create the file selection UI"""
        # File selection group
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)
        
        # File list
        self.file_list_widget = QListWidget()
        file_layout.addWidget(self.file_list_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        add_btn = QPushButton("Add")
        add_btn.setToolTip("Add spectrum files for batch processing")
        add_btn.setStyleSheet(PRIMARY_BUTTON_STYLE)
        button_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("Remove")
        remove_btn.setToolTip("Remove selected files from the list")
        remove_btn.setStyleSheet(DANGER_BUTTON_STYLE)
        button_layout.addWidget(remove_btn)
        
        file_layout.addLayout(button_layout)
        
        # Navigation controls
        nav_group = QGroupBox("Navigation")
        nav_layout = QHBoxLayout(nav_group)
        
        first_btn = QPushButton("◀◀")
        first_btn.setToolTip("Go to first spectrum")
        nav_layout.addWidget(first_btn)
        
        prev_btn = QPushButton("◀")
        prev_btn.setToolTip("Go to previous spectrum")
        nav_layout.addWidget(prev_btn)
        
        next_btn = QPushButton("▶")
        next_btn.setToolTip("Go to next spectrum")
        nav_layout.addWidget(next_btn)
        
        last_btn = QPushButton("▶▶")
        last_btn.setToolTip("Go to last spectrum")
        nav_layout.addWidget(last_btn)
        
        # Apply consistent style to navigation buttons
        for btn in [first_btn, prev_btn, next_btn, last_btn]:
            btn.setStyleSheet(BUTTON_STYLE)
        
        # Status label
        self.current_file_label = QLabel("No files loaded")
        self.current_file_label.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                padding: 5px;
                border-radius: 4px;
                font-family: monospace;
                color: #495057;
            }
        """)
        
        # Add to main layout
        self.main_layout.addWidget(file_group)
        self.main_layout.addWidget(nav_group)
        self.main_layout.addWidget(self.current_file_label)
        self.main_layout.addStretch()
        
        # Store button references for connection
        self.add_btn = add_btn
        self.remove_btn = remove_btn
        self.first_btn = first_btn
        self.prev_btn = prev_btn
        self.next_btn = next_btn
        self.last_btn = last_btn
    
    def connect_signals(self):
        """Connect internal signals"""
        # File list interactions
        self.file_list_widget.itemDoubleClicked.connect(self._on_file_double_click)
        
        # Button connections
        self.add_btn.clicked.connect(self._add_files)
        self.remove_btn.clicked.connect(self._remove_selected_files)
        
        # Navigation connections
        self.first_btn.clicked.connect(lambda: self._navigate_spectrum(0))
        self.prev_btn.clicked.connect(lambda: self._navigate_spectrum(-1))
        self.next_btn.clicked.connect(lambda: self._navigate_spectrum(1))
        self.last_btn.clicked.connect(lambda: self._navigate_spectrum(-2))
    
    def connect_core_signals(self):
        """Connect to core component signals"""
        if self.data_processor:
            self.data_processor.current_spectrum_changed.connect(self._update_current_spectrum_display)
            self.data_processor.spectra_list_changed.connect(self._update_file_list_display)
    
    def _add_files(self):
        """Add files to the batch processing list"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Spectrum files (*.txt *.csv *.spc *.asp)")
        
        if file_dialog.exec():
            files = file_dialog.selectedFiles()
            self.emit_action("add_files", {"files": files})
            self.emit_status(f"Added {len(files)} files")
    
    def _remove_selected_files(self):
        """Remove selected files from the list"""
        selected_items = self.file_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select files to remove.")
            return
        
        file_paths = [item.text() for item in selected_items]
        self.emit_action("remove_files", {"files": file_paths})
        self.emit_status(f"Removed {len(file_paths)} files")
    
    def _navigate_spectrum(self, direction):
        """Navigate through spectra"""
        nav_actions = {
            0: "navigate_first",
            -1: "navigate_previous", 
            1: "navigate_next",
            -2: "navigate_last"
        }
        
        action = nav_actions.get(direction)
        if action:
            self.emit_action(action, {})
    
    def _on_file_double_click(self, item):
        """Handle file double-click"""
        file_path = item.text()
        self.emit_action("load_file", {"file_path": file_path})
        self.emit_status(f"Loading: {os.path.basename(file_path)}")
    
    def _update_current_spectrum_display(self, index):
        """Update current spectrum display"""
        if hasattr(self, 'data_processor') and self.data_processor and 0 <= index < len(self.data_processor.spectra_files):
            file_path = self.data_processor.spectra_files[index]
            filename = os.path.basename(file_path)
            self.current_file_label.setText(f"Current: {filename}")
        else:
            self.current_file_label.setText("No files loaded")
    
    def _update_file_list_display(self, file_list):
        """Update file list display"""
        self.file_list_widget.clear()
        for file_path in file_list:
            filename = os.path.basename(file_path)
            self.file_list_widget.addItem(filename)
        
        # Update navigation button states
        has_files = len(file_list) > 0
        for btn in [self.first_btn, self.prev_btn, self.next_btn, self.last_btn]:
            btn.setEnabled(has_files)
    
    def get_tab_data(self):
        """Get current tab state"""
        base_data = super().get_tab_data()
        
        file_list = []
        for i in range(self.file_list_widget.count()):
            file_list.append(self.file_list_widget.item(i).text())
        
        base_data.update({
            'file_count': len(file_list),
            'file_list': file_list,
            'current_file': self.current_file_label.text()
        })
        
        return base_data
    
    def reset_to_defaults(self):
        """Reset tab to default state"""
        self.file_list_widget.clear()
        self.current_file_label.setText("No files loaded")
        
        # Disable navigation buttons
        for btn in [self.first_btn, self.prev_btn, self.next_btn, self.last_btn]:
            btn.setEnabled(False)
        
        self.emit_status("Tab reset to defaults") 