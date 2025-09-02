#!/usr/bin/env python3
"""
Simple, Thread-Safe Update Checker for RamanLab
No background threads to avoid Qt threading issues.
"""

import webbrowser
import platform
import subprocess
import os
from packaging import version as packaging_version

try:
    import requests
    import pyperclip
    SIMPLE_UPDATE_CHECKER_AVAILABLE = True
except ImportError:
    SIMPLE_UPDATE_CHECKER_AVAILABLE = False

from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextBrowser, QMessageBox, QProgressDialog
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from version import __version__


class SimpleUpdateDialog(QDialog):
    """Simple update dialog without threading."""
    
    def __init__(self, update_info, parent=None):
        super().__init__(parent)
        self.update_info = update_info
        self.setWindowTitle("RamanLab Update Available")
        self.setFixedSize(600, 400)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the update dialog UI."""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🔄 Update Available!")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Version info
        version_text = QLabel(f"Version {self.update_info['version']} is now available\n(You have version {__version__})")
        version_text.setFont(QFont("Arial", 12))
        version_text.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_text)
        
        # Release notes
        notes_label = QLabel("Release Notes:")
        notes_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(notes_label)
        
        notes_browser = QTextBrowser()
        notes_browser.setPlainText(self.update_info['body'])
        notes_browser.setMaximumHeight(150)
        layout.addWidget(notes_browser)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Auto-update button (most convenient option)
        auto_update_btn = QPushButton("🔄 Auto Update")
        auto_update_btn.clicked.connect(self.auto_update)
        auto_update_btn.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        button_layout.addWidget(auto_update_btn)
        
        # Download button
        download_btn = QPushButton("📥 Download from GitHub")
        download_btn.clicked.connect(self.download_update)
        button_layout.addWidget(download_btn)
        
        # Copy git command button
        copy_btn = QPushButton("📋 Copy 'git pull'")
        copy_btn.clicked.connect(self.copy_git_command)
        button_layout.addWidget(copy_btn)
        
        # Later button
        later_btn = QPushButton("Later")
        later_btn.clicked.connect(self.reject)
        button_layout.addWidget(later_btn)
        
        layout.addLayout(button_layout)
        
    def download_update(self):
        """Open GitHub release page."""
        try:
            webbrowser.open(self.update_info['html_url'])
            QMessageBox.information(self, "Download Started", 
                                  "GitHub release page opened in your browser!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open browser: {str(e)}")
        self.accept()
        
    def copy_git_command(self):
        """Copy git pull command to clipboard."""
        try:
            if SIMPLE_UPDATE_CHECKER_AVAILABLE:
                pyperclip.copy("git pull")
                QMessageBox.information(self, "Copied!", 
                                      "Command 'git pull' copied to clipboard.\n\n"
                                      "Paste it in your terminal in the RamanLab directory.")
            else:
                QMessageBox.information(self, "Manual Command", 
                                      "Run this in your RamanLab directory:\n\ngit pull")
        except Exception as e:
            QMessageBox.information(self, "Manual Command", 
                                  "Run this in your RamanLab directory:\n\ngit pull")
    
    def auto_update(self):
        """Automatically perform git pull operation."""
        try:
            # Check if we're in a git repository
            if not os.path.exists('.git'):
                QMessageBox.warning(self, "Not a Git Repository", 
                                  "This directory is not a git repository.\n\n"
                                  "Auto-update only works if you cloned RamanLab using git.\n\n"
                                  "Please use 'Download from GitHub' instead or clone the repository using:\n"
                                  "git clone https://github.com/aaroncelestian/RamanLab.git")
                return
            
            # Show progress dialog
            progress_dialog = QProgressDialog("Updating RamanLab...", "Cancel", 0, 0, self)
            progress_dialog.setWindowTitle("Auto Update")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.show()
            progress_dialog.repaint()
            
            # Check if user cancelled
            if progress_dialog.wasCanceled():
                return
            
            # Check for uncommitted changes
            progress_dialog.setLabelText("Checking for local changes...")
            progress_dialog.repaint()
            
            try:
                # Check git status
                status_result = subprocess.run(['git', 'status', '--porcelain'], 
                                             capture_output=True, text=True, timeout=10)
                
                if progress_dialog.wasCanceled():
                    return
                    
                if status_result.returncode == 0 and status_result.stdout.strip():
                    # There are uncommitted changes
                    progress_dialog.close()
                    reply = QMessageBox.question(self, "Uncommitted Changes", 
                                               "You have uncommitted changes in your RamanLab directory.\n\n"
                                               "These changes might be lost during the update.\n\n"
                                               "Do you want to continue with the update anyway?",
                                               QMessageBox.Yes | QMessageBox.No,
                                               QMessageBox.No)
                    if reply != QMessageBox.Yes:
                        return
                    
                    # Restart progress dialog
                    progress_dialog = QProgressDialog("Updating RamanLab...", "Cancel", 0, 0, self)
                    progress_dialog.setWindowTitle("Auto Update")
                    progress_dialog.setWindowModality(Qt.WindowModal)
                    progress_dialog.setMinimumDuration(0)
                    progress_dialog.show()
                    progress_dialog.repaint()
                    
            except subprocess.TimeoutExpired:
                progress_dialog.close()
                QMessageBox.warning(self, "Update Failed", 
                                  "Git status check timed out.\n\n"
                                  "Please try the manual update method.")
                return
            except Exception as e:
                progress_dialog.close()
                QMessageBox.warning(self, "Update Failed", 
                                  f"Could not check git status: {str(e)}\n\n"
                                  "Please try the manual update method.")
                return
            
            # Perform git pull
            progress_dialog.setLabelText("Pulling latest changes from GitHub...")
            progress_dialog.repaint()
            
            if progress_dialog.wasCanceled():
                return
            
            try:
                # Run git pull
                pull_result = subprocess.run(['git', 'pull'], 
                                           capture_output=True, text=True, timeout=30)
                
                progress_dialog.close()
                
                if pull_result.returncode == 0:
                    # Success - check if there were actually changes
                    output = pull_result.stdout.strip()
                    if "Already up to date" in output or "Already up-to-date" in output:
                        QMessageBox.information(self, "Already Up to Date", 
                                              f"RamanLab is already up to date!\n\n"
                                              f"Current version: {__version__}\n\n"
                                              "No restart needed.")
                    else:
                        # Actual update occurred
                        reply = QMessageBox.question(self, "Update Successful! 🎉", 
                                                   f"RamanLab has been updated successfully!\n\n"
                                                   f"Updated to version {self.update_info['version']}\n\n"
                                                   "Changes:\n"
                                                   f"{output}\n\n"
                                                   "RamanLab must be restarted to use the new version.\n\n"
                                                   "Would you like to close RamanLab now so you can restart it?",
                                                   QMessageBox.Yes | QMessageBox.No,
                                                   QMessageBox.Yes)
                        if reply == QMessageBox.Yes:
                            # Close the application
                            import sys
                            if self.parent():
                                self.parent().close()
                            sys.exit(0)
                    self.accept()
                else:
                    # Git pull failed
                    error_msg = pull_result.stderr or pull_result.stdout or "Unknown error"
                    
                    if "merge conflict" in error_msg.lower():
                        QMessageBox.warning(self, "Merge Conflict", 
                                          "Update failed due to merge conflicts.\n\n"
                                          "This happens when you have local changes that conflict with the update.\n\n"
                                          "Please resolve manually:\n"
                                          "1. Open terminal in RamanLab directory\n"
                                          "2. Run: git status\n"
                                          "3. Resolve conflicts or reset with: git reset --hard origin/main\n"
                                          "4. Run: git pull")
                    elif "authentication" in error_msg.lower() or "permission" in error_msg.lower():
                        QMessageBox.warning(self, "Authentication Error", 
                                          "Update failed due to authentication issues.\n\n"
                                          "This might happen if you need to login to GitHub.\n\n"
                                          "Please try the manual update method or check your GitHub credentials.")
                    else:
                        QMessageBox.warning(self, "Update Failed", 
                                          f"Git pull failed:\n\n{error_msg}\n\n"
                                          "Please try the manual update method.")
                        
            except subprocess.TimeoutExpired:
                progress_dialog.close()
                QMessageBox.warning(self, "Update Failed", 
                                  "Git pull timed out.\n\n"
                                  "This might happen with slow internet connections.\n\n"
                                  "Please try the manual update method.")
            except Exception as e:
                progress_dialog.close()
                QMessageBox.warning(self, "Update Failed", 
                                  f"Could not run git pull: {str(e)}\n\n"
                                  "Please try the manual update method.")
                
        except Exception as e:
            QMessageBox.warning(self, "Update Failed", 
                              f"Unexpected error during auto-update: {str(e)}\n\n"
                              "Please try the manual update method.")


def simple_check_for_updates(parent=None, show_no_update=True):
    """
    Simple, synchronous update check without background threads.
    This runs in the main thread to avoid Qt threading issues.
    Windows-compatible version with proper exception handling.
    """
    if not SIMPLE_UPDATE_CHECKER_AVAILABLE:
        QMessageBox.information(
            parent,
            "Update Checker Unavailable",
            "Update checker requires:\n\n"
            "pip install requests packaging pyperclip\n\n"
            "Check manually at:\n"
            "https://github.com/aaroncelestian/RamanLab"
        )
        return
    
    # Windows-specific: Use QProgressDialog instead of QMessageBox for better control
    is_windows = platform.system().lower() == "windows"
    checking_dialog = None
    
    try:
        if is_windows:
            # Windows: Use QProgressDialog with cancel button for better UX
            checking_dialog = QProgressDialog("Checking for updates...", "Cancel", 0, 0, parent)
            checking_dialog.setWindowTitle("Update Check")
            checking_dialog.setWindowModality(Qt.WindowModal)
            checking_dialog.setMinimumDuration(500)  # Show after 500ms
            checking_dialog.setValue(0)
            checking_dialog.show()
            checking_dialog.repaint()
            
            # Check if user cancelled
            if checking_dialog.wasCanceled():
                return
        else:
            # Non-Windows: Use simpler approach
            checking_dialog = QMessageBox(parent)
            checking_dialog.setWindowTitle("Checking for Updates")
            checking_dialog.setText("Checking for updates...")
            checking_dialog.setStandardButtons(QMessageBox.Cancel)
            checking_dialog.show()
            checking_dialog.repaint()
        
        # Make the network request in main thread (blocking but safer)
        github_api_url = "https://api.github.com/repos/aaroncelestian/RamanLab"
        releases_url = f"{github_api_url}/releases/latest"
        
        # First check if the repository exists
        try:
            repo_response = requests.get(github_api_url, timeout=10)
            if is_windows and checking_dialog and checking_dialog.wasCanceled():
                return
                
            if repo_response.status_code == 404:
                if checking_dialog:
                    checking_dialog.close()
                QMessageBox.information(
                    parent,
                    "Repository Not Found",
                    f"The repository 'aaroncelestian/RamanLab' was not found on GitHub.\n\n"
                    f"You are running version {__version__}.\n\n"
                    "This might be a local development version.\n"
                    "Please check the correct repository URL or contact the developer."
                )
                return
        except Exception:
            pass  # Continue with release check anyway
        
        # Check for cancellation again (Windows)
        if is_windows and checking_dialog and checking_dialog.wasCanceled():
            return
            
        response = requests.get(releases_url, timeout=10)
        
        # Always close the checking dialog before proceeding
        if checking_dialog:
            checking_dialog.close()
            checking_dialog = None
        
        if response.status_code == 200:
            release_data = response.json()
            latest_version = release_data['tag_name'].lstrip('v')
            
            # Compare versions
            try:
                if packaging_version.parse(latest_version) > packaging_version.parse(__version__):
                    # Update available
                    update_info = {
                        'version': latest_version,
                        'name': release_data.get('name', 'Latest Release'),
                        'body': release_data.get('body', 'No release notes available.'),
                        'html_url': release_data.get('html_url', f"https://github.com/aaroncelestian/RamanLab/releases/latest")
                    }
                    
                    # Show update dialog
                    dialog = SimpleUpdateDialog(update_info, parent)
                    dialog.exec()
                    return
                    
            except Exception as e:
                # Version comparison failed, but we can still show info
                if latest_version != __version__:
                    update_info = {
                        'version': latest_version,
                        'name': release_data.get('name', 'Latest Release'),
                        'body': release_data.get('body', 'No release notes available.'),
                        'html_url': release_data.get('html_url', f"https://github.com/aaroncelestian/RamanLab/releases/latest")
                    }
                    dialog = SimpleUpdateDialog(update_info, parent)
                    dialog.exec()
                    return
            
            # No update available
            if show_no_update:
                QMessageBox.information(
                    parent,
                    "No Updates Available",
                    f"You have the latest version ({__version__})"
                )
        elif response.status_code == 404:
            # No releases found - repository exists but no releases
            if show_no_update:
                QMessageBox.information(
                    parent,
                    "No Releases Available",
                    f"The repository exists but has no releases yet.\n\n"
                    f"You are running version {__version__}.\n\n"
                    "Check for the latest code at:\n"
                    "https://github.com/aaroncelestian/RamanLab\n\n"
                    "You can update manually using 'git pull' if you cloned the repository."
                )
        else:
            # Other API request failure
            QMessageBox.warning(
                parent,
                "Update Check Failed",
                f"Could not check for updates (HTTP {response.status_code})\n\n"
                "Possible reasons:\n"
                "• Repository doesn't exist at the specified path\n"
                "• Network connectivity issues\n"
                "• GitHub API rate limiting\n\n"
                "Check manually at:\n"
                "https://github.com/aaroncelestian/RamanLab"
            )
            
    except requests.RequestException as e:
        if checking_dialog:
            checking_dialog.close()
            checking_dialog = None
        QMessageBox.warning(
            parent,
            "Network Error",
            f"Could not connect to GitHub:\n{str(e)}\n\n"
            "Check your internet connection or visit:\n"
            "https://github.com/aaroncelestian/RamanLab"
        )
    except Exception as e:
        if checking_dialog:
            checking_dialog.close()
            checking_dialog = None
        QMessageBox.warning(
            parent,
            "Update Check Error",
            f"Unexpected error:\n{str(e)}\n\n"
            "Check manually at:\n"
            "https://github.com/aaroncelestian/RamanLab"
        )
    finally:
        # Ensure dialog is always closed (Windows safety)
        if checking_dialog:
            try:
                checking_dialog.close()
            except:
                pass 