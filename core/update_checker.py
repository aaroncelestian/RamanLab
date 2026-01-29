#!/usr/bin/env python3
"""
RamanLab Update Checker
Checks for updates from GitHub repository and provides update options.
"""

import sys
import os
import subprocess
import webbrowser
from pathlib import Path
from packaging import version as packaging_version

try:
    import requests
    import pyperclip
    UPDATE_CHECKER_AVAILABLE = True
except ImportError:
    UPDATE_CHECKER_AVAILABLE = False

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                              QPushButton, QTextBrowser, QProgressDialog,
                              QMessageBox, QApplication)
from PySide6.QtCore import Qt, QThread, QObject, Signal, QTimer, QProcess
from PySide6.QtGui import QFont, QPixmap

from version import __version__


class UpdateCheckWorker(QObject):
    """Worker thread for checking updates from GitHub."""
    
    update_available = Signal(dict)  # Emits update info if available
    no_update = Signal()
    error_occurred = Signal(str)
    
    def __init__(self, current_version):
        super().__init__()
        self.current_version = current_version
        self.github_api_url = "https://api.github.com/repos/aaroncelestian/RamanLab"
        self.github_repo_url = "https://github.com/aaroncelestian/RamanLab"
        
    def check_for_updates(self):
        """Check for updates from GitHub."""
        try:
            # Try to get latest release
            releases_url = f"{self.github_api_url}/releases/latest"
            response = requests.get(releases_url, timeout=10)
            
            if response.status_code == 200:
                release_data = response.json()
                latest_version = release_data['tag_name'].lstrip('v')
                
                # Compare versions
                if self._is_newer_version(latest_version, self.current_version):
                    update_info = {
                        'version': latest_version,
                        'name': release_data.get('name', 'Latest Release'),
                        'body': release_data.get('body', 'No release notes available.'),
                        'published_at': release_data.get('published_at', ''),
                        'html_url': release_data.get('html_url', f"{self.github_repo_url}/releases/latest")
                    }
                    self.update_available.emit(update_info)
                else:
                    self.no_update.emit()
                    
            else:
                # Fallback to commits if no releases
                self._check_commits()
                
        except requests.RequestException as e:
            self.error_occurred.emit(f"Network error: {str(e)}")
        except Exception as e:
            self.error_occurred.emit(f"Unexpected error: {str(e)}")
    
    def _check_commits(self):
        """Fallback method to check commits if no releases available."""
        try:
            commits_url = f"{self.github_api_url}/commits"
            response = requests.get(commits_url, timeout=10)
            
            if response.status_code == 200:
                commits_data = response.json()
                if commits_data:
                    latest_commit = commits_data[0]
                    update_info = {
                        'version': 'Latest Commit',
                        'name': 'Development Version',
                        'body': f"Latest commit: {latest_commit['commit']['message'][:200]}...",
                        'published_at': latest_commit['commit']['committer']['date'],
                        'html_url': f"{self.github_repo_url}/commits"
                    }
                    self.update_available.emit(update_info)
                else:
                    self.no_update.emit()
            else:
                self.error_occurred.emit("Could not fetch update information from GitHub.")
                
        except Exception as e:
            self.error_occurred.emit(f"Error checking commits: {str(e)}")
    
    def _is_newer_version(self, latest, current):
        """Compare version strings using packaging library."""
        try:
            return packaging_version.parse(latest) > packaging_version.parse(current)
        except Exception:
            # Fallback to string comparison if version parsing fails
            return latest != current


class UpdateDialog(QDialog):
    """Dialog for displaying update information and options."""
    
    def __init__(self, update_info, parent=None):
        super().__init__(parent)
        self.update_info = update_info
        self.setWindowTitle("RamanLab Update Available")
        self.setFixedSize(600, 500)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the update dialog UI."""
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        
        # Update icon (if available)
        try:
            icon_label = QLabel()
            icon_label.setFixedSize(64, 64)
            icon_label.setStyleSheet("background-color: #4CAF50; border-radius: 32px; color: white; font-size: 24px; font-weight: bold;")
            icon_label.setText("🔄")
            icon_label.setAlignment(Qt.AlignCenter)
            header_layout.addWidget(icon_label)
        except:
            pass
        
        # Update text
        update_text = QVBoxLayout()
        title = QLabel("Update Available!")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        update_text.addWidget(title)
        
        version_text = QLabel(f"Version {self.update_info['version']} is now available")
        version_text.setFont(QFont("Arial", 12))
        update_text.addWidget(version_text)
        
        current_text = QLabel(f"You have version {__version__}")
        current_text.setFont(QFont("Arial", 10))
        current_text.setStyleSheet("color: #666666;")
        update_text.addWidget(current_text)
        
        header_layout.addLayout(update_text)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Release notes
        notes_label = QLabel("Release Notes:")
        notes_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(notes_label)
        
        self.notes_browser = QTextBrowser()
        self.notes_browser.setPlainText(self.update_info['body'])
        self.notes_browser.setMaximumHeight(200)
        layout.addWidget(self.notes_browser)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Download button
        self.download_btn = QPushButton("📥 Download Latest Version")
        self.download_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.download_btn.clicked.connect(self.download_update)
        button_layout.addWidget(self.download_btn)
        
        # Auto-update button (if git repo)
        if self._is_git_repo():
            self.auto_update_btn = QPushButton("🔄 Auto-Update (Git)")
            self.auto_update_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
            """)
            self.auto_update_btn.clicked.connect(self.auto_update)
            button_layout.addWidget(self.auto_update_btn)
        
        # Copy git command button
        self.copy_git_btn = QPushButton("📋 Copy Git Command")
        self.copy_git_btn.clicked.connect(self.copy_git_command)
        button_layout.addWidget(self.copy_git_btn)
        
        # Later button
        later_btn = QPushButton("Later")
        later_btn.clicked.connect(self.reject)
        button_layout.addWidget(later_btn)
        
        layout.addLayout(button_layout)
        
    def download_update(self):
        """Open the GitHub release page for manual download."""
        try:
            webbrowser.open(self.update_info['html_url'])
            QMessageBox.information(self, "Download Started", 
                                  "The GitHub release page has been opened in your browser. "
                                  "Download the latest version and follow the installation instructions.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open browser: {str(e)}")
        
        self.accept()
        
    def auto_update(self):
        """Perform automatic git update."""
        if not self._is_git_repo():
            QMessageBox.warning(self, "Not a Git Repository", 
                              "This directory is not a git repository. Please use the download option instead.")
            return
            
        # Confirm the update
        reply = QMessageBox.question(self, "Confirm Auto-Update",
                                   "This will run 'git pull' to update RamanLab. "
                                   "Any local changes may be overwritten. Continue?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self._perform_git_update()
    
    def _perform_git_update(self):
        """Execute git pull with progress dialog."""
        # Get the application directory (where the main script is located)
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Create a more detailed progress dialog
        progress = QProgressDialog("Updating RamanLab...\nThis may take a moment.", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)  # Show immediately
        progress.setValue(0)
        progress.setMaximum(0)  # Indeterminate progress
        
        # Use QProcess for better output handling
        process = QProcess(self)
        process.setWorkingDirectory(app_dir)
        
        # Connect process signals
        def handle_finished(exit_code, exit_status):
            progress.close()
            if exit_code == 0 and exit_status == QProcess.NormalExit:
                output = process.readAllStandardOutput().data().decode('utf-8', errors='replace')
                
                # Ask user if they want to update dependencies
                reply = QMessageBox.question(
                    self,
                    "Update Dependencies?",
                    "RamanLab code has been updated successfully!\n\n"
                    "Would you like to update Python dependencies now?\n"
                    "This will install any new or updated packages from requirements_qt6.txt.\n\n"
                    "Note: This may take a few minutes.\n\n"
                    f"Git output:\n{output[:200]}...",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    self._update_dependencies(app_dir)
                else:
                    QMessageBox.information(
                        self,
                        "Update Complete",
                        "Code update successful!\n\n"
                        "You can update dependencies later by running:\n"
                        "  python update_dependencies.py --all\n\n"
                        "or:\n"
                        "  pip install -r requirements_qt6.txt\n\n"
                        "Please restart the application to use the new version."
                    )
                    self.accept()
            else:
                error_output = process.readAllStandardError().data().decode('utf-8', errors='replace')
                QMessageBox.warning(
                    self, 
                    "Update Failed",
                    f"Git pull failed with exit code {exit_code}:\n{error_output}\n\n"
                    "You may need to resolve conflicts manually or use the download option."
                )
        
        process.finished.connect(handle_finished)
        
        # Start the git pull process
        try:
            # First try with the standard git command
            process.start('git', ['pull', '--rebase'])
            process.waitForStarted()
            
            # Set a timeout (5 minutes)
            if not process.waitForFinished(300000):  # 5 minutes timeout
                process.kill()
                raise Exception("Git pull operation timed out after 5 minutes")
                
        except Exception as e:
            progress.close()
            QMessageBox.warning(
                self,
                "Update Error",
                f"An error occurred during update: {str(e)}\n\n"
                "Please try updating manually using the 'git pull' command in the RamanLab directory:\n"
                f"{app_dir}"
            )
    
    def _update_dependencies(self, app_dir):
        """Update Python dependencies from requirements_qt6.txt."""
        requirements_file = os.path.join(app_dir, 'requirements_qt6.txt')
        
        # Check if requirements file exists
        if not os.path.exists(requirements_file):
            QMessageBox.warning(
                self,
                "Requirements File Not Found",
                f"Could not find requirements_qt6.txt in:\n{app_dir}\n\n"
                "Please update dependencies manually."
            )
            self.accept()
            return
        
        # Create progress dialog for dependency update
        progress = QProgressDialog(
            "Updating Python dependencies...\n"
            "This may take several minutes.\n\n"
            "Installing packages from requirements_qt6.txt...",
            "Cancel", 0, 0, self
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.setMaximum(0)  # Indeterminate progress
        
        # Create process for pip install
        process = QProcess(self)
        process.setWorkingDirectory(app_dir)
        
        def handle_pip_finished(exit_code, exit_status):
            progress.close()
            if exit_code == 0 and exit_status == QProcess.NormalExit:
                output = process.readAllStandardOutput().data().decode('utf-8', errors='replace')
                QMessageBox.information(
                    self,
                    "Dependencies Updated",
                    "Python dependencies have been updated successfully!\n\n"
                    "Please restart RamanLab to use the new version with updated packages.\n\n"
                    "Update complete!"
                )
                self.accept()
            else:
                error_output = process.readAllStandardError().data().decode('utf-8', errors='replace')
                QMessageBox.warning(
                    self,
                    "Dependency Update Failed",
                    f"Failed to update some dependencies.\n\n"
                    f"Error output:\n{error_output[:500]}...\n\n"
                    "You can try updating manually with:\n"
                    "  python update_dependencies.py --all\n\n"
                    "or:\n"
                    "  pip install -r requirements_qt6.txt"
                )
                self.accept()
        
        process.finished.connect(handle_pip_finished)
        
        # Start pip install process
        try:
            # Use sys.executable to ensure we use the same Python interpreter
            python_exe = sys.executable
            process.start(python_exe, ['-m', 'pip', 'install', '--upgrade', '-r', 'requirements_qt6.txt'])
            process.waitForStarted()
            
            # Set a longer timeout for pip install (15 minutes)
            if not process.waitForFinished(900000):  # 15 minutes timeout
                process.kill()
                progress.close()
                QMessageBox.warning(
                    self,
                    "Update Timeout",
                    "Dependency update timed out after 15 minutes.\n\n"
                    "You can try updating manually with:\n"
                    "  python update_dependencies.py --all"
                )
                self.accept()
        
        except Exception as e:
            progress.close()
            QMessageBox.warning(
                self,
                "Update Error",
                f"An error occurred during dependency update:\n{str(e)}\n\n"
                "Please try updating manually with:\n"
                "  python update_dependencies.py --all\n\n"
                "or:\n"
                "  pip install -r requirements_qt6.txt"
            )
            self.accept()
    
    def copy_git_command(self):
        """Copy git pull command to clipboard."""
        try:
            if UPDATE_CHECKER_AVAILABLE:
                pyperclip.copy("git pull")
                QMessageBox.information(self, "Copied to Clipboard",
                                      "The command 'git pull' has been copied to your clipboard.\n\n"
                                      "Open a terminal in the RamanLab directory and paste the command to update.")
            else:
                QMessageBox.information(self, "Manual Update Command",
                                      "To update manually, run this command in the RamanLab directory:\n\n"
                                      "git pull\n\n"
                                      "(Command could not be copied - pyperclip not available)")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not copy to clipboard: {str(e)}")
    
    def _is_git_repo(self):
        """Check if current directory is a git repository."""
        return os.path.exists('.git') or subprocess.run(['git', 'rev-parse', '--git-dir'], 
                                                       capture_output=True).returncode == 0


class UpdateChecker(QObject):
    """Main update checker class."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_version = __version__
        self.worker_thread = None
        
    def check_for_updates(self, parent_widget=None, show_no_update=True):
        """Check for updates asynchronously."""
        if not UPDATE_CHECKER_AVAILABLE:
            QMessageBox.warning(parent_widget, "Update Checker Unavailable",
                              "The update checker requires additional dependencies:\n\n"
                              "pip install requests packaging pyperclip\n\n"
                              "You can still check for updates manually at:\n"
                              "https://github.com/aaroncelestian/RamanLab")
            return
        
        # Create and start worker thread
        self.worker_thread = QThread()
        self.worker = UpdateCheckWorker(self.current_version)
        self.worker.moveToThread(self.worker_thread)
        
        # Connect signals with Qt.QueuedConnection to ensure thread safety
        self.worker.update_available.connect(
            lambda info: self.show_update_dialog(info, parent_widget), 
            Qt.QueuedConnection
        )
        if show_no_update:
            self.worker.no_update.connect(
                lambda: self.show_no_update_message(parent_widget),
                Qt.QueuedConnection
            )
        self.worker.error_occurred.connect(
            lambda error: self.show_error_message(error, parent_widget),
            Qt.QueuedConnection
        )
        
        self.worker_thread.started.connect(self.worker.check_for_updates)
        self.worker_thread.start()
        
    def show_update_dialog(self, update_info, parent=None):
        """Show the update dialog."""
        dialog = UpdateDialog(update_info, parent)
        dialog.exec()
        self.cleanup_thread()
        
    def show_no_update_message(self, parent=None):
        """Show message when no updates are available."""
        QMessageBox.information(parent, "No Updates Available",
                              f"You are running the latest version of RamanLab ({self.current_version}).")
        self.cleanup_thread()
        
    def show_error_message(self, error, parent=None):
        """Show error message."""
        QMessageBox.warning(parent, "Update Check Failed",
                          f"Could not check for updates:\n{error}\n\n"
                          "You can check manually at:\n"
                          "https://github.com/aaroncelestian/RamanLab")
        self.cleanup_thread()
        
    def cleanup_thread(self):
        """Clean up the worker thread."""
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None


def check_for_updates(parent=None, show_no_update=True):
    """Convenience function to check for updates."""
    checker = UpdateChecker()
    checker.check_for_updates(parent, show_no_update)
    return checker 