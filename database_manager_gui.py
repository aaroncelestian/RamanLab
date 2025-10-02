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
    from core.matplotlib_config import configure_compact_ui
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
            
            # Handle PKL files only (SQLite support removed)
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
            'RamanLab_Database_20250602.pkl': None,  # This is actually a pickle file
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
        
        # Explicitly set window flags to ensure minimize/maximize/close buttons on Windows
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        
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
        
        self.inspect_btn = QPushButton("🔍 Inspect File Details")
        self.inspect_btn.clicked.connect(self.inspect_file_details)
        
        controls_layout.addWidget(self.validate_btn)
        controls_layout.addWidget(self.test_browser_btn)
        controls_layout.addWidget(self.inspect_btn)
        
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
        
        fix_extensions_btn = QPushButton("🔧 Fix File Extensions")
        fix_extensions_btn.clicked.connect(self.fix_file_extensions)
        
        buttons_layout.addWidget(backup_btn)
        buttons_layout.addWidget(move_btn)
        buttons_layout.addWidget(restore_btn)
        buttons_layout.addWidget(fix_extensions_btn)
        
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
            <li><b>RamanLab_Database_20250602.pkl</b> - Main Raman spectra database (PKL format)</li>
            <li><b>mineral_modes.pkl</b> - Mineral vibrational modes database (PKL format)</li>
        </ul>
        
        <h4>File Extension Issues:</h4>
        <ul>
            <li>If you have files that won't validate, they might be pickle files with wrong extensions</li>
            <li>Use "🔧 Fix File Extensions" in the Repair tab to automatically correct misnamed files</li>
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
        # Only support PKL files now
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
            
            # Immediately validate the selected file
            validator = DatabaseValidator({db_name: file_path})
            result = validator.validate_database_file(file_path)
            
            if result['readable'] and result['format_valid']:
                self.db_widgets[db_name]['status_label'].setText("✅")
                self.log(f"✅ Selected {db_name}: {file_path} ({result['entry_count']} entries)")
            else:
                self.db_widgets[db_name]['status_label'].setText("❌")
                self.log(f"❌ Selected {db_name}: {file_path} - Validation failed: {result.get('error', 'Unknown error')}")
    
    def auto_detect_databases(self):
        """Automatically detect database files in the current directory."""
        detected = []
        validated = []
        
        for db_name in self.database_files.keys():
            # Check current directory first
            current_path = os.path.join(self.script_dir, db_name)
            if os.path.exists(current_path):
                self.database_files[db_name] = current_path
                self.db_widgets[db_name]['path_label'].setText(current_path)
                
                # Validate the detected file
                validator = DatabaseValidator({db_name: current_path})
                result = validator.validate_database_file(current_path)
                
                if result['readable'] and result['format_valid']:
                    self.db_widgets[db_name]['status_label'].setText("✅")
                    validated.append(f"{db_name} ({result['entry_count']} entries)")
                else:
                    self.db_widgets[db_name]['status_label'].setText("❌")
                    self.log(f"⚠️ {db_name} found but validation failed: {result.get('error', 'Unknown error')}")
                
                detected.append(db_name)
        
        if detected:
            self.log(f"Auto-detected files: {', '.join(detected)}")
            if validated:
                self.log(f"Successfully validated: {', '.join(validated)}")
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
            
            # Test PKL files only (SQLite support removed)
            test_results = []
            for db_name, db_path in self.database_files.items():
                if db_path and os.path.exists(db_path):
                    try:
                        # Test PKL database only
                        with open(db_path, 'rb') as f:
                            data = pickle.load(f)
                        test_results.append(f"✅ {db_name}: PKL database readable ({len(data)} entries)")
                    except Exception as e:
                        test_results.append(f"❌ {db_name}: Cannot read - {str(e)}")
            
            self.log("Database Browser Test Results:")
            for result in test_results:
                self.log(result)
                
        except Exception as e:
            self.log(f"❌ Browser test failed: {e}")
    
    def inspect_file_details(self):
        """Inspect detailed information about the database files."""
        self.log("🔍 Inspecting file details...")
        
        for db_name, db_path in self.database_files.items():
            if db_path and os.path.exists(db_path):
                self.log(f"\n📁 Inspecting {db_name}:")
                self.log(f"   Path: {db_path}")
                self.log(f"   Size: {os.path.getsize(db_path)} bytes")
                
                try:
                    # Read first 32 bytes to check file signature
                    with open(db_path, 'rb') as f:
                        header = f.read(32)
                    
                    self.log(f"   Header (first 16 bytes): {header[:16]}")
                    self.log(f"   Header (hex): {header[:16].hex()}")
                    
                    # Only check PKL files (SQLite support removed)
                    if db_path.endswith('.pkl'):
                        # Check if it's a valid pickle file
                        try:
                            with open(db_path, 'rb') as f:
                                data = pickle.load(f)
                            self.log(f"   ✅ Valid pickle file")
                            self.log(f"   Type: {type(data)}")
                            if hasattr(data, '__len__'):
                                self.log(f"   Length: {len(data)}")
                        except Exception as e:
                            self.log(f"   ❌ Pickle loading error: {e}")
                    else:
                        self.log(f"   ⚠️ Unsupported file type (only .pkl files supported)")
                
                except Exception as e:
                    self.log(f"   ❌ File inspection error: {e}")
            else:
                self.log(f"⚠️ {db_name}: File not selected or does not exist")
    
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
                    'replacement': 'primary_db_path = self.db_directory / "RamanLab_Database_20250602.pkl"'
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
    
    def fix_file_extensions(self):
        """Fix files that have wrong extensions (PKL files only now)."""
        self.log("🔧 Checking for files with incorrect extensions...")
        
        fixed_files = []
        
        for db_name, db_path in self.database_files.items():
            if db_path and os.path.exists(db_path):
                try:
                    # Read file header
                    with open(db_path, 'rb') as f:
                        header = f.read(16)
                    
                    # Check for misnamed pickle files (SQLite support removed)
                    if not db_name.endswith('.pkl') and header.startswith(b'\x80'):
                        # This is a pickle file with wrong extension
                        new_name = os.path.splitext(db_name)[0] + '.pkl'
                        new_path = os.path.join(os.path.dirname(db_path), new_name)
                        
                        # Create backup if requested
                        if self.backup_checkbox.isChecked():
                            backup_path = db_path + '.backup'
                            shutil.copy2(db_path, backup_path)
                            self.log(f"Created backup: {backup_path}")
                        
                        # Rename the file
                        os.rename(db_path, new_path)
                        
                        # Update the database manager to look for the correct file
                        self.database_files[new_name] = new_path
                        del self.database_files[db_name]
                        
                        # Update the GUI
                        if new_name in self.db_widgets:
                            self.db_widgets[new_name]['path_label'].setText(new_path)
                        
                        fixed_files.append(f"{db_name} → {new_name}")
                        self.log(f"✅ Renamed {db_name} to {new_name}")
                
                except Exception as e:
                    self.log(f"❌ Error checking {db_name}: {e}")
        
        if fixed_files:
            self.log(f"🎉 Fixed {len(fixed_files)} files with incorrect extensions:")
            for fix in fixed_files:
                self.log(f"   {fix}")
            self.log("Re-running auto-detection with corrected filenames...")
            self.auto_detect_databases()
        else:
            self.log("ℹ️ No files found with incorrect extensions")
    
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