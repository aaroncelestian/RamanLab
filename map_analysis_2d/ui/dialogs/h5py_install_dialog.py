"""
Custom dialog for h5py installation with diagnostic and auto-install options.
"""

import sys
import subprocess
import logging
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTextEdit, QMessageBox, QProgressDialog
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

logger = logging.getLogger(__name__)


class ScriptRunnerThread(QThread):
    """Thread to run Python scripts without blocking UI."""
    
    output_ready = Signal(str, bool)  # (output_text, success)
    
    def __init__(self, script_name, python_executable):
        super().__init__()
        self.script_name = script_name
        self.python_executable = python_executable
        
    def run(self):
        """Run the script and capture output."""
        try:
            # Get the RamanLab root directory (3 levels up from this file)
            script_dir = Path(__file__).parent.parent.parent.parent
            script_path = script_dir / self.script_name
            
            if not script_path.exists():
                self.output_ready.emit(
                    f"ERROR: Script not found at {script_path}\n\n"
                    f"Please ensure {self.script_name} is in the RamanLab root directory.",
                    False
                )
                return
            
            # Run the script
            result = subprocess.run(
                [self.python_executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Combine stdout and stderr
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += "\n--- ERRORS ---\n" + result.stderr
            
            success = result.returncode == 0
            self.output_ready.emit(output, success)
            
        except subprocess.TimeoutExpired:
            self.output_ready.emit(
                "ERROR: Script execution timed out after 5 minutes.\n\n"
                "This may indicate a problem with the installation process.",
                False
            )
        except Exception as e:
            self.output_ready.emit(
                f"ERROR: Failed to run script:\n{str(e)}",
                False
            )


class H5pyInstallDialog(QDialog):
    """Custom dialog for h5py installation assistance."""
    
    def __init__(self, python_executable, error_details, parent=None):
        super().__init__(parent)
        self.python_executable = python_executable
        self.error_details = error_details
        self.runner_thread = None
        
        self.setWindowTitle("h5py Installation Required")
        self.setMinimumSize(700, 500)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("⚠️ h5py Library Required")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Error info
        info_text = QLabel(
            "The h5py library is required to import HDF5/MAPX files, "
            "but it could not be imported in your current Python environment."
        )
        info_text.setWordWrap(True)
        layout.addWidget(info_text)
        
        # Python executable info
        python_label = QLabel(f"<b>Python executable:</b><br>{self.python_executable}")
        python_label.setWordWrap(True)
        python_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(python_label)
        
        # Error details
        error_label = QLabel(f"<b>Import error:</b><br>{self.error_details}")
        error_label.setWordWrap(True)
        error_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(error_label)
        
        # Output text area
        output_label = QLabel("<b>Diagnostic/Installation Output:</b>")
        layout.addWidget(output_label)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Courier", 9))
        self.output_text.setPlainText("Click a button below to run diagnostics or install h5py...")
        layout.addWidget(self.output_text)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        # Diagnostic button
        self.diagnostic_btn = QPushButton("🔍 Run Diagnostic")
        self.diagnostic_btn.setToolTip("Check h5py installation status and get specific recommendations")
        self.diagnostic_btn.clicked.connect(self.run_diagnostic)
        self.diagnostic_btn.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
            QPushButton:disabled {
                background-color: #9CA3AF;
            }
        """)
        button_layout.addWidget(self.diagnostic_btn)
        
        # Auto-install button
        self.install_btn = QPushButton("⚡ Auto-Install h5py")
        self.install_btn.setToolTip("Automatically install h5py into this Python environment")
        self.install_btn.clicked.connect(self.run_installer)
        self.install_btn.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:disabled {
                background-color: #9CA3AF;
            }
        """)
        button_layout.addWidget(self.install_btn)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #6B7280;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4B5563;
            }
        """)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        # Help text
        help_text = QLabel(
            "<i>💡 Tip: Run diagnostic first to see what's needed, "
            "then use auto-install to fix the issue automatically.</i>"
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        
    def run_diagnostic(self):
        """Run the check_h5py.py diagnostic script."""
        self.output_text.setPlainText("Running diagnostic script...\n")
        self.diagnostic_btn.setEnabled(False)
        self.install_btn.setEnabled(False)
        
        # Create and start runner thread
        self.runner_thread = ScriptRunnerThread("check_h5py.py", self.python_executable)
        self.runner_thread.output_ready.connect(self._on_diagnostic_complete)
        self.runner_thread.start()
        
    def run_installer(self):
        """Run the install_h5py.py auto-installer script."""
        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Confirm Installation",
            f"This will install h5py into:\n\n{self.python_executable}\n\n"
            "The installation may take a few minutes.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply != QMessageBox.Yes:
            return
        
        self.output_text.setPlainText("Running auto-installer...\n\n")
        self.diagnostic_btn.setEnabled(False)
        self.install_btn.setEnabled(False)
        
        # Create and start runner thread
        self.runner_thread = ScriptRunnerThread("install_h5py.py", self.python_executable)
        self.runner_thread.output_ready.connect(self._on_install_complete)
        self.runner_thread.start()
        
    def _on_diagnostic_complete(self, output, success):
        """Handle diagnostic completion."""
        self.output_text.setPlainText(output)
        self.diagnostic_btn.setEnabled(True)
        self.install_btn.setEnabled(True)
        
        if success:
            # Check if h5py is working
            if "✅ SUCCESS: h5py imported successfully" in output:
                QMessageBox.information(
                    self,
                    "h5py Available",
                    "Good news! h5py is already installed and working.\n\n"
                    "You can close this dialog and try importing your HDF5 file again."
                )
            else:
                QMessageBox.information(
                    self,
                    "Diagnostic Complete",
                    "Diagnostic complete. Review the output above for specific recommendations.\n\n"
                    "You can now click 'Auto-Install h5py' to fix the issue automatically."
                )
        else:
            QMessageBox.warning(
                self,
                "Diagnostic Issues",
                "The diagnostic script encountered some issues.\n\n"
                "Review the output above for details."
            )
        
    def _on_install_complete(self, output, success):
        """Handle installation completion."""
        self.output_text.setPlainText(output)
        self.diagnostic_btn.setEnabled(True)
        self.install_btn.setEnabled(True)
        
        if success:
            # Check if installation was successful
            if "🎉 h5py has been successfully installed" in output or "✅ SUCCESS" in output:
                QMessageBox.information(
                    self,
                    "Installation Successful",
                    "h5py has been successfully installed!\n\n"
                    "Please:\n"
                    "1. Close this dialog\n"
                    "2. Restart RamanLab (important!)\n"
                    "3. Try importing your HDF5 file again"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Installation Completed",
                    "The installation script completed, but verification may have failed.\n\n"
                    "Review the output above. You may need to:\n"
                    "1. Restart your computer (Windows DLL issues)\n"
                    "2. Install Visual C++ Redistributable (Windows)\n"
                    "3. Check WINDOWS_SETUP.md for troubleshooting"
                )
        else:
            QMessageBox.critical(
                self,
                "Installation Failed",
                "The installation script encountered errors.\n\n"
                "Review the output above for details.\n\n"
                "You may need to install h5py manually. See WINDOWS_SETUP.md for help."
            )
