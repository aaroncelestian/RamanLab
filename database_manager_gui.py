#!/usr/bin/env python3
"""
RamanLab Database Manager GUI
============================

GUI tool for managing RamanLab database files with the following features:
1. Database file selection and validation
2. Entry counting and integrity checks
3. Path correction and file moving
4. Database browser compatibility testing
5. Backup and repair functionality
"""

import os
import sys
import shutil
import pickle
import subprocess
from pathlib import Path
from datetime import datetime

# Qt6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
    QLabel, QPushButton, QFileDialog, QTextEdit, QGroupBox, QFormLayout,
    QProgressBar, QCheckBox, QMessageBox, QTabWidget, QListWidget,
    QListWidgetItem, QSplitter, QFrame, QComboBox, QSpinBox
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QIcon, QPixmap

# Import matplotlib config as per user rules
try:
    from polarization_ui.matplotlib_config import configure_compact_ui
    configure_compact_ui()
except ImportError:
    pass

class DatabaseValidator(QThread):
    """Thread for validating database files without blocking the UI."""
    
    progress_updated = Signal(int)
    status_updated = Signal(str)
    validation_complete = Signal(dict)
    
    def __init__(self, database_files):
        super().__init__()
        self.database_files = database_files
        self.results = {}
    
    def run(self):
        """Run database validation in separate thread."""
        total_files = len(self.database_files)
        
        for i, (db_name, db_path) in enumerate(self.database_files.items()):
            self.status_updated.emit(f"Validating {db_name}...")
            
            result = self.validate_database_file(db_path)
            self.results[db_name] = result
            
            progress = int((i + 1) / total_files * 100)
            self.progress_updated.emit(progress)
        
        self.validation_complete.emit(self.results)
    
    def validate_database_file(self, file_path):
        """Validate a single database file."""
        result = {
            'exists': False,
            'readable': False,
            'entry_count': 0,
            'file_size': 0,
            'format_valid': False,
            'sample_entries': [],
            'error': None
        }
        
        try:
            if not os.path.exists(file_path):
                result['error'] = 'File does not exist'
                return result
            
            result['exists'] = True
            result['file_size'] = os.path.getsize(file_path)
            
            # Handle different file types
            if file_path.endswith('.sqlite') or file_path.endswith('.db'):
                # SQLite database validation
                import sqlite3
                conn = sqlite3.connect(file_path)
                cursor = conn.cursor()
                
                # Check if it has expected tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                if 'spectra' in tables:
                    cursor.execute("SELECT COUNT(*) FROM spectra")
                    result['entry_count'] = cursor.fetchone()[0]
                    
                    # Get sample entries
                    cursor.execute("SELECT name FROM spectra LIMIT 3")
                    result['sample_entries'] = [row[0] for row in cursor.fetchall()]
                    
                    result['readable'] = True
                    result['format_valid'] = True
                else:
                    result['error'] = 'SQLite database missing expected "spectra" table'
                
                conn.close()
                
            else:
                # PKL file validation
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                result['readable'] = True
                result['format_valid'] = True
                
                if isinstance(data, dict):
                    result['entry_count'] = len(data)
                    # Get sample entries (first 3)
                    sample_keys = list(data.keys())[:3]
                    result['sample_entries'] = sample_keys
                else:
                    result['error'] = 'Database is not in expected dictionary format'
                
        except Exception as e:
            result['error'] = str(e)
        
        return result


class DatabaseManagerGUI(QMainWindow):
    """Main GUI window for database management."""
    
    def __init__(self):
        super().__init__()
        
        # Database file definitions
        self.database_files = {
            'RamanLab_Database_20250602.sqlite': None,
            'mineral_modes.pkl': None
        }
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.validation_results = {}
        
        # Initialize UI
        self.init_ui()
        self.auto_detect_databases()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("RamanLab Database Manager")
        self.setMinimumSize(900, 700)
        self.resize(1100, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel("🗄️ RamanLab Database Manager")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2E7D32; padding: 10px; background: #E8F5E8; border-radius: 5px; margin-bottom: 10px;")
        main_layout.addWidget(title_label)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_database_tab()
        self.create_validation_tab()
        self.create_repair_tab()
        self.create_settings_tab()
        
        # Status bar
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)
    
    def create_database_tab(self):
        """Create the database management tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Database file selection group
        db_group = QGroupBox("Database Files")
        db_layout = QFormLayout(db_group)
        
        self.db_widgets = {}
        for db_name in self.database_files.keys():
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            
            # File path label
            path_label = QLabel("Not selected")
            path_label.setStyleSheet("border: 1px solid #ccc; padding: 5px; background: #f9f9f9;")
            
            # Browse button
            browse_btn = QPushButton("Browse...")
            browse_btn.clicked.connect(lambda checked, name=db_name: self.browse_database_file(name))
            
            # Status indicator
            status_label = QLabel("❓")
            status_label.setFixedWidth(30)
            status_label.setAlignment(Qt.AlignCenter)
            
            row_layout.addWidget(path_label, 1)
            row_layout.addWidget(browse_btn)
            row_layout.addWidget(status_label)
            
            self.db_widgets[db_name] = {
                'path_label': path_label,
                'status_label': status_label,
                'browse_btn': browse_btn
            }
            
            db_layout.addRow(f"{db_name}:", row_widget)
        
        layout.addWidget(db_group)
        
        # Quick actions
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QHBoxLayout(actions_group)
        
        validate_btn = QPushButton("🔍 Validate All Databases")
        validate_btn.clicked.connect(self.validate_all_databases)
        
        auto_detect_btn = QPushButton("🔎 Auto-Detect Databases")
        auto_detect_btn.clicked.connect(self.auto_detect_databases)
        
        fix_paths_btn = QPushButton("🔧 Fix Database Paths")
        fix_paths_btn.clicked.connect(self.fix_database_paths)
        
        actions_layout.addWidget(validate_btn)
        actions_layout.addWidget(auto_detect_btn)
        actions_layout.addWidget(fix_paths_btn)
        
        layout.addWidget(actions_group)
        
        # Results display
        self.results_text = QTextEdit()
        self.results_text.setMinimumHeight(200)
        layout.addWidget(self.results_text)
        
        self.tab_widget.addTab(tab, "Database Files")
    
    def create_validation_tab(self):
        """Create the database validation tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Validation controls
        controls_group = QGroupBox("Validation Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        self.validate_btn = QPushButton("🔍 Run Full Validation")
        self.validate_btn.clicked.connect(self.run_full_validation)
        
        self.test_browser_btn = QPushButton("🧪 Test Database Browser")
        self.test_browser_btn.clicked.connect(self.test_database_browser)
        
        controls_layout.addWidget(self.validate_btn)
        controls_layout.addWidget(self.test_browser_btn)
        
        layout.addWidget(controls_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Validation results
        self.validation_text = QTextEdit()
        self.validation_text.setMinimumHeight(400)
        layout.addWidget(self.validation_text)
        
        self.tab_widget.addTab(tab, "Validation")
    
    def create_repair_tab(self):
        """Create the database repair tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Repair options
        repair_group = QGroupBox("Repair Options")
        repair_layout = QVBoxLayout(repair_group)
        
        self.backup_checkbox = QCheckBox("Create backup before repair")
        self.backup_checkbox.setChecked(True)
        repair_layout.addWidget(self.backup_checkbox)
        
        self.auto_move_checkbox = QCheckBox("Automatically move databases to correct location")
        self.auto_move_checkbox.setChecked(False)
        repair_layout.addWidget(self.auto_move_checkbox)
        
        # Repair buttons
        buttons_layout = QHBoxLayout()
        
        backup_btn = QPushButton("💾 Create Backup")
        backup_btn.clicked.connect(self.create_backup)
        
        move_btn = QPushButton("📁 Move Databases")
        move_btn.clicked.connect(self.move_databases)
        
        restore_btn = QPushButton("🔄 Restore Backup")
        restore_btn.clicked.connect(self.restore_backup)
        
        buttons_layout.addWidget(backup_btn)
        buttons_layout.addWidget(move_btn)
        buttons_layout.addWidget(restore_btn)
        
        repair_layout.addLayout(buttons_layout)
        layout.addWidget(repair_group)
        
        # Repair log
        self.repair_log = QTextEdit()
        self.repair_log.setMinimumHeight(300)
        layout.addWidget(self.repair_log)
        
        self.tab_widget.addTab(tab, "Repair")
    
    def create_settings_tab(self):
        """Create the settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Settings group
        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout(settings_group)
        
        # RamanLab directory
        dir_widget = QWidget()
        dir_layout = QHBoxLayout(dir_widget)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        
        self.ramanlab_dir_label = QLabel(self.script_dir)
        self.ramanlab_dir_label.setStyleSheet("border: 1px solid #ccc; padding: 5px; background: #f9f9f9;")
        
        browse_dir_btn = QPushButton("Browse...")
        browse_dir_btn.clicked.connect(self.browse_ramanlab_directory)
        
        dir_layout.addWidget(self.ramanlab_dir_label, 1)
        dir_layout.addWidget(browse_dir_btn)
        
        settings_layout.addRow("RamanLab Directory:", dir_widget)
        
        # Validation depth
        self.validation_depth = QSpinBox()
        self.validation_depth.setRange(1, 10)
        self.validation_depth.setValue(3)
        settings_layout.addRow("Sample Entry Count:", self.validation_depth)
        
        layout.addWidget(settings_group)
        
        # Help text
        help_text = QTextEdit()
        help_text.setHtml("""
        <h3>Database Manager Help</h3>
        <p><b>Database Files Tab:</b> Select and validate your database files</p>
        <p><b>Validation Tab:</b> Run comprehensive validation checks</p>
        <p><b>Repair Tab:</b> Fix issues and manage backups</p>
        <p><b>Settings Tab:</b> Configure application settings</p>
        
        <h4>Expected Database Files:</h4>
        <ul>
            <li><b>RamanLab_Database_20250602.sqlite</b> - Main SQLite Raman spectra database</li>
            <li><b>mineral_modes.pkl</b> - Mineral vibrational modes database (PKL format)</li>
        </ul>
        
        <h4>Common Issues:</h4>
        <ul>
            <li>Database files in wrong location</li>
            <li>Corrupted database files</li>
            <li>Permission issues</li>
            <li>Path reference problems</li>
        </ul>
        """)
        help_text.setMaximumHeight(200)
        layout.addWidget(help_text)
        
        self.tab_widget.addTab(tab, "Settings & Help")
    
    def browse_database_file(self, db_name):
        """Browse for a database file."""
        if db_name.endswith('.sqlite'):
            filter_string = "SQLite Database (*.sqlite);;Database Files (*.db);;All Files (*)"
        else:
            filter_string = "Pickle Files (*.pkl);;All Files (*)"
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            f"Select {db_name}", 
            self.script_dir,
            filter_string
        )
        
        if file_path:
            self.database_files[db_name] = file_path
            self.db_widgets[db_name]['path_label'].setText(file_path)
            self.db_widgets[db_name]['status_label'].setText("❓")
            self.log(f"Selected {db_name}: {file_path}")
    
    def auto_detect_databases(self):
        """Automatically detect database files in the current directory."""
        detected = []
        
        for db_name in self.database_files.keys():
            # Check current directory first
            current_path = os.path.join(self.script_dir, db_name)
            if os.path.exists(current_path):
                self.database_files[db_name] = current_path
                self.db_widgets[db_name]['path_label'].setText(current_path)
                self.db_widgets[db_name]['status_label'].setText("✅")
                detected.append(db_name)
        
        if detected:
            self.log(f"Auto-detected databases: {', '.join(detected)}")
        else:
            self.log("No databases auto-detected in current directory")
    
    def validate_all_databases(self):
        """Validate all selected database files."""
        selected_files = {name: path for name, path in self.database_files.items() if path}
        
        if not selected_files:
            QMessageBox.warning(self, "Warning", "No database files selected!")
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start validation in separate thread
        self.validator = DatabaseValidator(selected_files)
        self.validator.progress_updated.connect(self.progress_bar.setValue)
        self.validator.status_updated.connect(self.status_label.setText)
        self.validator.validation_complete.connect(self.on_validation_complete)
        self.validator.start()
    
    def on_validation_complete(self, results):
        """Handle validation completion."""
        self.progress_bar.setVisible(False)
        self.validation_results = results
        
        # Update status indicators
        for db_name, result in results.items():
            if result['readable'] and result['format_valid']:
                self.db_widgets[db_name]['status_label'].setText("✅")
            else:
                self.db_widgets[db_name]['status_label'].setText("❌")
        
        # Display results
        self.display_validation_results(results)
    
    def display_validation_results(self, results):
        """Display validation results in the text area."""
        output = []
        output.append("="*60)
        output.append("DATABASE VALIDATION RESULTS")
        output.append("="*60)
        output.append(f"Validation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("")
        
        for db_name, result in results.items():
            output.append(f"📁 {db_name}")
            output.append("-" * 40)
            
            if result['exists']:
                output.append(f"✅ File exists: {result['exists']}")
                output.append(f"📊 File size: {result['file_size']} bytes")
                
                if result['readable']:
                    output.append(f"✅ Readable: {result['readable']}")
                    output.append(f"✅ Format valid: {result['format_valid']}")
                    output.append(f"📈 Entry count: {result['entry_count']}")
                    
                    if result['sample_entries']:
                        output.append(f"📋 Sample entries: {', '.join(result['sample_entries'])}")
                else:
                    output.append(f"❌ Error: {result['error']}")
            else:
                output.append(f"❌ File not found")
                if result['error']:
                    output.append(f"❌ Error: {result['error']}")
            
            output.append("")
        
        self.results_text.setText("\n".join(output))
        self.validation_text.setText("\n".join(output))
    
    def run_full_validation(self):
        """Run comprehensive validation including database browser test."""
        self.validate_all_databases()
        
        # Additional validation can be added here
        QTimer.singleShot(2000, self.test_database_browser)  # Test browser after validation
    
    def test_database_browser(self):
        """Test if the database browser can read the databases."""
        self.log("Testing database browser compatibility...")
        
        try:
            # Try to import and test the database browser
            import importlib.util
            
            browser_path = os.path.join(self.script_dir, "database_browser_qt6.py")
            if not os.path.exists(browser_path):
                self.log("❌ Database browser file not found!")
                return
            
            # Test if we can import required modules
            try:
                import pickle
                import sqlite3
                
                test_results = []
                for db_name, db_path in self.database_files.items():
                    if db_path and os.path.exists(db_path):
                        try:
                            if db_path.endswith('.sqlite') or db_path.endswith('.db'):
                                # Test SQLite database
                                conn = sqlite3.connect(db_path)
                                cursor = conn.cursor()
                                cursor.execute("SELECT COUNT(*) FROM spectra")
                                count = cursor.fetchone()[0]
                                conn.close()
                                test_results.append(f"✅ {db_name}: SQLite database readable ({count} spectra)")
                            else:
                                # Test PKL database
                                with open(db_path, 'rb') as f:
                                    data = pickle.load(f)
                                test_results.append(f"✅ {db_name}: PKL database readable ({len(data)} entries)")
                        except Exception as e:
                            test_results.append(f"❌ {db_name}: Cannot read - {str(e)}")
                
                self.log("Database Browser Test Results:")
                for result in test_results:
                    self.log(result)
                    
            except ImportError as e:
                self.log(f"❌ Missing dependencies: {e}")
                
        except Exception as e:
            self.log(f"❌ Browser test failed: {e}")
    
    def fix_database_paths(self):
        """Fix database path issues in Python files."""
        self.log("Fixing database paths in Python files...")
        
        import re
        
        # Files to fix and their patterns (from original script)
        files_to_fix = {
            'mineral_database.py': [
                {
                    'pattern': r'self\.database_path = database_path or os\.path\.join\(os\.path\.dirname\(__file__\), "mineral_modes\.pkl"\)',
                    'replacement': 'self.database_path = database_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "mineral_modes.pkl")'
                }
            ],
            'raman_spectra_qt6.py': [
                {
                    'pattern': r'primary_db_path = self\.db_directory / "raman_database\.pkl"',
                    'replacement': 'primary_db_path = self.db_directory / "RamanLab_Database_20250602.sqlite"'
                }
            ],
            'raman_polarization_analyzer.py': [
                {
                    'pattern': r'database_path = os\.path\.join\(os\.path\.dirname\(__file__\), "mineral_modes\.pkl"\)',
                    'replacement': 'database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mineral_modes.pkl")'
                },
                {
                    'pattern': r'db_path = "mineral_modes\.pkl"',
                    'replacement': 'db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mineral_modes.pkl")'
                }
            ]
        }
        
        fixed_files = []
        for filename, patterns in files_to_fix.items():
            filepath = os.path.join(self.script_dir, filename)
            
            if not os.path.exists(filepath):
                self.log(f"⚠️ File not found: {filename}")
                continue
            
            # Read the file
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                modified = False
                
                # Apply each pattern fix
                for pattern_info in patterns:
                    pattern = pattern_info['pattern']
                    replacement = pattern_info['replacement']
                    
                    if re.search(pattern, content):
                        content = re.sub(pattern, replacement, content)
                        modified = True
                        self.log(f"✅ Fixed pattern in {filename}")
                
                # Write the fixed content back
                if modified:
                    # Create backup first
                    backup_path = filepath + '.backup'
                    shutil.copy2(filepath, backup_path)
                    self.log(f"Created backup: {backup_path}")
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    fixed_files.append(filename)
                    self.log(f"✅ Updated {filename}")
                else:
                    self.log(f"ℹ️ No changes needed for {filename}")
                    
            except Exception as e:
                self.log(f"❌ Error fixing {filename}: {e}")
        
        if fixed_files:
            self.log(f"Successfully fixed {len(fixed_files)} files: {', '.join(fixed_files)}")
        else:
            self.log("No files required fixing")
        
        self.log("Path fixing completed!")
    
    def create_backup(self):
        """Create backup of database files."""
        if not any(self.database_files.values()):
            QMessageBox.warning(self, "Warning", "No database files to backup!")
            return
        
        backup_dir = os.path.join(self.script_dir, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(backup_dir, exist_ok=True)
        
        backed_up = []
        for db_name, db_path in self.database_files.items():
            if db_path and os.path.exists(db_path):
                backup_path = os.path.join(backup_dir, db_name)
                shutil.copy2(db_path, backup_path)
                backed_up.append(db_name)
        
        self.log(f"Backup created: {backup_dir}")
        self.log(f"Backed up files: {', '.join(backed_up)}")
    
    def move_databases(self):
        """Move database files to the correct location."""
        target_dir = self.script_dir
        moved = []
        
        for db_name, db_path in self.database_files.items():
            if db_path and os.path.exists(db_path):
                target_path = os.path.join(target_dir, db_name)
                if db_path != target_path:
                    if self.backup_checkbox.isChecked():
                        shutil.copy2(db_path, f"{db_path}.backup")
                    
                    shutil.move(db_path, target_path)
                    self.database_files[db_name] = target_path
                    self.db_widgets[db_name]['path_label'].setText(target_path)
                    moved.append(db_name)
        
        if moved:
            self.log(f"Moved databases: {', '.join(moved)}")
        else:
            self.log("No databases needed moving")
    
    def restore_backup(self):
        """Restore from backup."""
        backup_dir = QFileDialog.getExistingDirectory(
            self, "Select Backup Directory", self.script_dir
        )
        
        if backup_dir:
            restored = []
            for db_name in self.database_files.keys():
                backup_file = os.path.join(backup_dir, db_name)
                if os.path.exists(backup_file):
                    target_path = os.path.join(self.script_dir, db_name)
                    shutil.copy2(backup_file, target_path)
                    self.database_files[db_name] = target_path
                    self.db_widgets[db_name]['path_label'].setText(target_path)
                    restored.append(db_name)
            
            if restored:
                self.log(f"Restored from backup: {', '.join(restored)}")
            else:
                self.log("No database files found in backup directory")
    
    def browse_ramanlab_directory(self):
        """Browse for RamanLab directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select RamanLab Directory", self.script_dir
        )
        
        if directory:
            self.script_dir = directory
            self.ramanlab_dir_label.setText(directory)
            self.auto_detect_databases()
    
    def log(self, message):
        """Log a message to all text areas."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        # Add to results text
        self.results_text.append(log_message)
        
        # Add to repair log
        self.repair_log.append(log_message)
        
        # Update status
        self.status_label.setText(message)


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("RamanLab Database Manager")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("RamanLab")
    
    # Create and show main window
    window = DatabaseManagerGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 