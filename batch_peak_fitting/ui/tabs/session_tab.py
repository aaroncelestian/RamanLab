"""
Session Tab Component
Handles session save/load and application state management
Simplified version of the original session functionality
"""

import os
from datetime import datetime
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, 
    QLabel, QListWidget, QTextEdit, QMessageBox
)
from PySide6.QtCore import Qt

from ..base_tab import BaseTab

# Unified button styles for consistent UI
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

SUCCESS_BUTTON_STYLE = """
    QPushButton {
        background-color: #198754;
        color: white;
        border: 1px solid #146c43;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 11px;
        min-width: 50px;
    }
    QPushButton:hover {
        background-color: #157347;
        border: 1px solid #0f5132;
    }
    QPushButton:pressed {
        background-color: #146c43;
        border: 1px solid #0c3a22;
    }
"""

WARNING_BUTTON_STYLE = """
    QPushButton {
        background-color: #fd7e14;
        color: white;
        border: 1px solid #e76500;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 11px;
        min-width: 50px;
    }
    QPushButton:hover {
        background-color: #e76500;
        border: 1px solid #bf5700;
    }
    QPushButton:pressed {
        background-color: #d65d00;
        border: 1px solid #a04800;
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

INFO_BUTTON_STYLE = """
    QPushButton {
        background-color: #0dcaf0;
        color: #055160;
        border: 1px solid #0aa2c0;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 11px;
        min-width: 50px;
    }
    QPushButton:hover {
        background-color: #31d2f2;
        border: 1px solid #0994a8;
    }
    QPushButton:pressed {
        background-color: #0aa2c0;
        border: 1px solid #087990;
    }
"""

PURPLE_BUTTON_STYLE = """
    QPushButton {
        background-color: #6f42c1;
        color: white;
        border: 1px solid #59359a;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 11px;
        min-width: 50px;
    }
    QPushButton:hover {
        background-color: #59359a;
        border: 1px solid #4c2d83;
    }
    QPushButton:pressed {
        background-color: #4c2d83;
        border: 1px solid #3e246c;
    }
"""

GRAY_BUTTON_STYLE = """
    QPushButton {
        background-color: #6c757d;
        color: white;
        border: 1px solid #5c636a;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 11px;
        min-width: 50px;
    }
    QPushButton:hover {
        background-color: #5c636a;
        border: 1px solid #495057;
    }
    QPushButton:pressed {
        background-color: #495057;
        border: 1px solid #343a40;
    }
"""


class SessionTab(BaseTab):
    """
    Session management component for saving and loading application state.
    Simplified version focusing on core session operations.
    """
    
    def __init__(self, parent=None):
        # Initialize widgets that will be created
        self.session_list = None
        self.session_info = None
        self.auto_save_enabled = True
        
        super().__init__(parent)
        self.tab_name = "Session"
    
    def setup_ui(self):
        """Create the session management UI"""
        
        # Current session group
        current_group = QGroupBox("Current Session")
        current_layout = QVBoxLayout(current_group)
        
        # Session info
        self.current_session_label = QLabel("No active session")
        self.current_session_label.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                padding: 8px;
                border-radius: 4px;
                font-family: monospace;
                font-weight: 500;
                color: #495057;
            }
        """)
        current_layout.addWidget(self.current_session_label)
        
        # Current session controls
        current_controls_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save")
        save_btn.setStyleSheet(SUCCESS_BUTTON_STYLE)
        current_controls_layout.addWidget(save_btn)
        
        save_as_btn = QPushButton("Save As")
        save_as_btn.setStyleSheet(SUCCESS_BUTTON_STYLE)
        current_controls_layout.addWidget(save_as_btn)
        
        new_session_btn = QPushButton("New")
        new_session_btn.setStyleSheet(PRIMARY_BUTTON_STYLE)
        current_controls_layout.addWidget(new_session_btn)
        
        current_layout.addLayout(current_controls_layout)
        
        # Saved sessions group
        saved_group = QGroupBox("Saved Sessions")
        saved_layout = QVBoxLayout(saved_group)
        
        # Sessions list
        self.session_list = QListWidget()
        self.session_list.setMaximumHeight(150)
        saved_layout.addWidget(self.session_list)
        
        # Session controls
        session_controls_layout = QHBoxLayout()
        
        load_btn = QPushButton("Load")
        load_btn.setStyleSheet(PRIMARY_BUTTON_STYLE)
        session_controls_layout.addWidget(load_btn)
        
        delete_btn = QPushButton("Delete")
        delete_btn.setStyleSheet(DANGER_BUTTON_STYLE)
        session_controls_layout.addWidget(delete_btn)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setStyleSheet(BUTTON_STYLE)
        session_controls_layout.addWidget(refresh_btn)
        
        saved_layout.addLayout(session_controls_layout)
        
        # Session info display
        info_group = QGroupBox("Session Information")
        info_layout = QVBoxLayout(info_group)
        
        self.session_info = QTextEdit()
        self.session_info.setMaximumHeight(120)
        self.session_info.setReadOnly(True)
        self.session_info.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-family: monospace;
                font-size: 10px;
                color: #495057;
            }
        """)
        info_layout.addWidget(self.session_info)
        
        # Quick actions group
        quick_group = QGroupBox("Quick Actions")
        quick_layout = QHBoxLayout(quick_group)
        
        export_settings_btn = QPushButton("Export")
        export_settings_btn.setStyleSheet(PURPLE_BUTTON_STYLE)
        quick_layout.addWidget(export_settings_btn)
        
        import_settings_btn = QPushButton("Import")
        import_settings_btn.setStyleSheet(WARNING_BUTTON_STYLE)
        quick_layout.addWidget(import_settings_btn)
        
        open_folder_btn = QPushButton("Open Folder")
        open_folder_btn.setStyleSheet(INFO_BUTTON_STYLE)
        quick_layout.addWidget(open_folder_btn)
        
        # Add to main layout
        self.main_layout.addWidget(current_group)
        self.main_layout.addWidget(saved_group)
        self.main_layout.addWidget(info_group)
        self.main_layout.addWidget(quick_group)
        
        # Store button references
        self.save_btn = save_btn
        self.save_as_btn = save_as_btn
        self.new_session_btn = new_session_btn
        self.load_btn = load_btn
        self.delete_btn = delete_btn
        self.refresh_btn = refresh_btn
        self.export_settings_btn = export_settings_btn
        self.import_settings_btn = import_settings_btn
        self.open_folder_btn = open_folder_btn
        
        # Initialize display
        self._initialize_session_info()
        self._refresh_session_list()
    
    def connect_signals(self):
        """Connect internal signals"""
        # Current session controls
        self.save_btn.clicked.connect(self._save_session)
        self.save_as_btn.clicked.connect(self._save_session_as)
        self.new_session_btn.clicked.connect(self._new_session)
        
        # Saved sessions controls
        self.load_btn.clicked.connect(self._load_selected_session)
        self.delete_btn.clicked.connect(self._delete_selected_session)
        self.refresh_btn.clicked.connect(self._refresh_session_list)
        
        # Quick actions
        self.export_settings_btn.clicked.connect(self._export_settings)
        self.import_settings_btn.clicked.connect(self._import_settings)
        self.open_folder_btn.clicked.connect(self._open_session_folder)
        
        # Session list selection
        self.session_list.itemSelectionChanged.connect(self._on_session_selection_changed)
    
    def connect_core_signals(self):
        """Connect to core component signals"""
        # This would connect to state management signals when available
        pass
    
    def _save_session(self):
        """Save current session"""
        session_data = self._collect_session_data()
        self.emit_action("save_session", {"data": session_data, "prompt_name": False})
        self.emit_status("Session saved")
        self._refresh_session_list()
    
    def _save_session_as(self):
        """Save session with new name"""
        session_data = self._collect_session_data()
        self.emit_action("save_session", {"data": session_data, "prompt_name": True})
        self.emit_status("Save session as... requested")
    
    def _new_session(self):
        """Start new session"""
        reply = QMessageBox.question(
            self, "New Session",
            "Are you sure you want to start a new session? Unsaved changes will be lost.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.emit_action("new_session", {})
            self.emit_status("New session started")
            self.current_session_label.setText("New session (unsaved)")
    
    def _load_selected_session(self):
        """Load selected session"""
        selected_items = self.session_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a session to load.")
            return
        
        session_name = selected_items[0].text()
        self.emit_action("load_session", {"session_name": session_name})
        self.emit_status(f"Loading session: {session_name}")
        self.current_session_label.setText(f"Session: {session_name}")
    
    def _delete_selected_session(self):
        """Delete selected session"""
        selected_items = self.session_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a session to delete.")
            return
        
        session_name = selected_items[0].text()
        
        reply = QMessageBox.question(
            self, "Delete Session",
            f"Are you sure you want to delete session '{session_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.emit_action("delete_session", {"session_name": session_name})
            self.emit_status(f"Session deleted: {session_name}")
            self._refresh_session_list()
    
    def _refresh_session_list(self):
        """Refresh the session list"""
        self.session_list.clear()
        
        # This would normally get the list from a session manager
        # For now, add some placeholder sessions
        placeholder_sessions = [
            "batch_analysis_2024-06-29_10-15",
            "peak_fitting_session_2024-06-28_14-30",
            "background_test_2024-06-27_16-45"
        ]
        
        for session in placeholder_sessions:
            self.session_list.addItem(session)
        
        self.emit_action("refresh_session_list", {})
    
    def _export_settings(self):
        """Export current settings"""
        settings_data = self._collect_settings_data()
        self.emit_action("export_settings", {"data": settings_data})
        self.emit_status("Settings export initiated")
    
    def _import_settings(self):
        """Import settings from file"""
        self.emit_action("import_settings", {})
        self.emit_status("Settings import requested")
    
    def _open_session_folder(self):
        """Open session folder in file manager"""
        self.emit_action("open_session_folder", {})
        self.emit_status("Opening session folder")
    
    def _on_session_selection_changed(self):
        """Handle session selection change"""
        selected_items = self.session_list.selectedItems()
        if selected_items:
            session_name = selected_items[0].text()
            self._update_session_info(session_name)
        else:
            self._initialize_session_info()
    
    def _collect_session_data(self):
        """Collect current session data from all tabs"""
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0',
            'tabs': {}
        }
        
        # This would collect data from all tabs
        # For now, return basic structure with current tab data
        session_data['tabs'][self.tab_name] = self.get_tab_data()
        
        return session_data
    
    def _collect_settings_data(self):
        """Collect current settings for export"""
        return {
            'ui_settings': {
                'auto_save_enabled': self.auto_save_enabled
            },
            'export_timestamp': datetime.now().isoformat()
        }
    
    def _update_session_info(self, session_name):
        """Update session info display"""
        self.session_info.clear()
        self.session_info.append(f"Session: {session_name}")
        self.session_info.append("")
        
        # This would normally load actual session metadata
        # For now, show placeholder info
        self.session_info.append("Session Information:")
        self.session_info.append(f"  Created: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        self.session_info.append(f"  Modified: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        self.session_info.append("  Files: 0")
        self.session_info.append("  Processing results: 0")
        self.session_info.append("")
        self.session_info.append("Double-click to load this session")
    
    def _initialize_session_info(self):
        """Initialize session info display"""
        self.session_info.clear()
        self.session_info.append("Session Management")
        self.session_info.append("")
        self.session_info.append("Features:")
        self.session_info.append("• Save/load complete application state")
        self.session_info.append("• Automatic session backup")
        self.session_info.append("• Export/import settings")
        self.session_info.append("• Session history tracking")
        self.session_info.append("")
        self.session_info.append("Select a session above to view details")
    
    def update_session_status(self, status_message):
        """Update current session status"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.current_session_label.setText(f"{status_message} ({timestamp})")
    
    def set_current_session(self, session_name):
        """Set the current session name"""
        if session_name:
            self.current_session_label.setText(f"Session: {session_name}")
        else:
            self.current_session_label.setText("No active session")
    
    def add_session_to_list(self, session_name):
        """Add a new session to the list"""
        self.session_list.addItem(session_name)
    
    def remove_session_from_list(self, session_name):
        """Remove a session from the list"""
        for i in range(self.session_list.count()):
            if self.session_list.item(i).text() == session_name:
                self.session_list.takeItem(i)
                break
    
    def get_tab_data(self):
        """Get current tab state"""
        base_data = super().get_tab_data()
        
        base_data.update({
            'auto_save_enabled': self.auto_save_enabled,
            'current_session': self.current_session_label.text() if self.current_session_label else "No active session",
            'available_sessions': [
                self.session_list.item(i).text() 
                for i in range(self.session_list.count())
            ] if self.session_list else []
        })
        
        return base_data
    
    def reset_to_defaults(self):
        """Reset tab to default state"""
        self.auto_save_enabled = True
        self.current_session_label.setText("No active session")
        self._initialize_session_info()
        self._refresh_session_list()
        self.emit_status("Tab reset to defaults") 