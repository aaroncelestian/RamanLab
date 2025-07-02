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
        
        add_btn = QPushButton("Add Files")
        add_btn.setToolTip("Add spectrum files for batch processing")
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        button_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("Remove Selected")
        remove_btn.setToolTip("Remove selected files from the list")
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #D32F2F;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C62828;
            }
            QPushButton:pressed {
                background-color: #B71C1C;
            }
        """)
        button_layout.addWidget(remove_btn)
        
        file_layout.addLayout(button_layout)
        
        # Navigation controls
        nav_group = QGroupBox("Navigation")
        nav_layout = QHBoxLayout(nav_group)
        
        first_btn = QPushButton("First")
        first_btn.setToolTip("Go to first spectrum")
        nav_layout.addWidget(first_btn)
        
        prev_btn = QPushButton("Previous")
        prev_btn.setToolTip("Go to previous spectrum")
        nav_layout.addWidget(prev_btn)
        
        next_btn = QPushButton("Next")
        next_btn.setToolTip("Go to next spectrum")
        nav_layout.addWidget(next_btn)
        
        last_btn = QPushButton("Last")
        last_btn.setToolTip("Go to last spectrum")
        nav_layout.addWidget(last_btn)
        
        # Style navigation buttons
        nav_button_style = """
            QPushButton {
                background-color: #424242;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #616161;
            }
            QPushButton:pressed {
                background-color: #212121;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
                color: #757575;
            }
        """
        
        for btn in [first_btn, prev_btn, next_btn, last_btn]:
            btn.setStyleSheet(nav_button_style)
        
        # Status label
        self.current_file_label = QLabel("No files loaded")
        self.current_file_label.setStyleSheet("""
            QLabel {
                background-color: #F5F5F5;
                border: 1px solid #E0E0E0;
                padding: 5px;
                border-radius: 3px;
                font-family: monospace;
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
            self.data_processor.spectra_list_changed.connect(self._update_file_list)
            self.data_processor.current_spectrum_changed.connect(self._update_current_selection)
            self.data_processor.spectrum_loaded.connect(self._update_status)
    
    def _add_files(self):
        """Add spectrum files for batch processing"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Spectrum Files", "",
            "Text files (*.txt);;CSV files (*.csv);;All files (*.*)"
        )
        
        if file_paths:
            if self.data_processor:
                files_added = self.data_processor.add_files(file_paths)
                self.emit_status(f"Added {files_added} files")
                self.emit_action("files_added", {"count": files_added, "paths": file_paths})
            else:
                self.emit_status("Error: No data processor available")
        else:
            self.emit_status("No files selected")
    
    def _remove_selected_files(self):
        """Remove selected files from the list"""
        selected_items = self.file_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select files to remove.")
            return
        
        # Get indices of selected items
        indices = []
        for item in selected_items:
            row = self.file_list_widget.row(item)
            indices.append(row)
        
        if self.data_processor:
            files_removed = self.data_processor.remove_files(indices)
            self.emit_status(f"Removed {files_removed} files")
            self.emit_action("files_removed", {"count": files_removed, "indices": indices})
        else:
            self.emit_status("Error: No data processor available")
    
    def _on_file_double_click(self, item):
        """Handle double-click on file list item"""
        row = self.file_list_widget.row(item)
        if self.data_processor:
            success = self.data_processor.load_spectrum(row)
            if success:
                self.emit_status(f"Loaded spectrum {row + 1}")
                self.emit_action("spectrum_selected", {"index": row})
            else:
                self.emit_status(f"Failed to load spectrum {row + 1}")
        else:
            self.emit_status("Error: No data processor available")
    
    def _navigate_spectrum(self, direction):
        """Navigate through spectra"""
        if self.data_processor:
            success = self.data_processor.navigate_spectrum(direction)
            
            direction_names = {
                0: "first",
                -2: "last", 
                -1: "previous",
                1: "next"
            }
            
            if success:
                self.emit_status(f"Navigated to {direction_names.get(direction, 'unknown')} spectrum")
                self.emit_action("spectrum_navigated", {"direction": direction})
            else:
                self.emit_status("Navigation failed")
        else:
            self.emit_status("Error: No data processor available")
    
    def _update_file_list(self, file_list):
        """Update the file list display"""
        self.file_list_widget.clear()
        
        for file_path in file_list:
            filename = os.path.basename(file_path)
            self.file_list_widget.addItem(filename)
        
        # Update navigation button states
        self._update_navigation_buttons(len(file_list))
        
        self.emit_status(f"File list updated: {len(file_list)} files")
    
    def _update_current_selection(self, index):
        """Update current file selection"""
        if 0 <= index < self.file_list_widget.count():
            self.file_list_widget.setCurrentRow(index)
        
        self._update_navigation_buttons(self.file_list_widget.count(), index)
    
    def _update_status(self, spectrum_data):
        """Update the status label"""
        if self.data_processor:
            status = self.data_processor.get_file_status()
            self.current_file_label.setText(status)
        else:
            self.current_file_label.setText("No data processor")
    
    def _update_navigation_buttons(self, total_files, current_index=None):
        """Update navigation button enabled states"""
        has_files = total_files > 0
        
        # Get current index if not provided
        if current_index is None and self.data_processor:
            current_index = self.data_processor.current_spectrum_index
        
        # Enable/disable buttons based on state
        self.first_btn.setEnabled(has_files and current_index > 0)
        self.prev_btn.setEnabled(has_files and current_index > 0)
        self.next_btn.setEnabled(has_files and current_index < total_files - 1)
        self.last_btn.setEnabled(has_files and current_index < total_files - 1)
    
    def update_from_data_processor(self, data=None):
        """Update tab when data processor state changes"""
        if self.data_processor:
            file_list = self.data_processor.get_file_list()
            self._update_file_list(file_list)
            self._update_current_selection(self.data_processor.current_spectrum_index)
            self._update_status(None)  # Will get status from data processor
    
    def get_tab_data(self):
        """Get current tab state"""
        base_data = super().get_tab_data()
        
        # Add file-specific data
        base_data.update({
            'selected_files': [
                self.file_list_widget.item(i).text() 
                for i in range(self.file_list_widget.count())
            ] if self.file_list_widget else [],
            'current_selection': self.file_list_widget.currentRow() if self.file_list_widget else -1
        })
        
        return base_data
    
    def reset_to_defaults(self):
        """Reset tab to default state"""
        if self.file_list_widget:
            self.file_list_widget.clear()
        
        if self.current_file_label:
            self.current_file_label.setText("No files loaded")
        
        self._update_navigation_buttons(0)
        self.emit_status("Tab reset to defaults")
    
    def validate_input(self):
        """Validate current input state"""
        if not self.file_list_widget:
            return False, "File list widget not initialized"
        
        if self.file_list_widget.count() == 0:
            return False, "No files loaded"
        
        return True, "" 