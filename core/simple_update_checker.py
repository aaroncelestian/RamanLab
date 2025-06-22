#!/usr/bin/env python3
"""
Simple, Thread-Safe Update Checker for RamanLab
No background threads to avoid Qt threading issues.
"""

import webbrowser
from packaging import version as packaging_version

try:
    import requests
    import pyperclip
    SIMPLE_UPDATE_CHECKER_AVAILABLE = True
except ImportError:
    SIMPLE_UPDATE_CHECKER_AVAILABLE = False

from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextBrowser, QMessageBox
from PySide6.QtCore import Qt
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


def simple_check_for_updates(parent=None, show_no_update=True):
    """
    Simple, synchronous update check without background threads.
    This runs in the main thread to avoid Qt threading issues.
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
    
    try:
        # Show a simple message that we're checking
        checking_msg = QMessageBox(parent)
        checking_msg.setWindowTitle("Checking for Updates")
        checking_msg.setText("Checking for updates...")
        checking_msg.setStandardButtons(QMessageBox.NoButton)
        checking_msg.show()
        checking_msg.repaint()  # Force immediate display
        
        # Make the network request in main thread (blocking but safer)
        github_api_url = "https://api.github.com/repos/aaroncelestian/RamanLab"
        releases_url = f"{github_api_url}/releases/latest"
        
        # First check if the repository exists
        try:
            repo_response = requests.get(github_api_url, timeout=10)
            if repo_response.status_code == 404:
                checking_msg.close()
                QMessageBox.information(
                    parent,
                    "Repository Not Found",
                    f"The repository 'aaroncelestian/RamanLab' was not found on GitHub.\n\n"
                    f"You are running version {__version__}.\n\n"
                    "This might be a local development version.\n"
                    "Please check the correct repository URL or contact the developer."
                )
                return
        except:
            pass  # Continue with release check anyway
        
        response = requests.get(releases_url, timeout=10)
        checking_msg.close()  # Close the checking message
        
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
        checking_msg.close()
        QMessageBox.warning(
            parent,
            "Network Error",
            f"Could not connect to GitHub:\n{str(e)}\n\n"
            "Check your internet connection or visit:\n"
            "https://github.com/aaroncelestian/RamanLab"
        )
    except Exception as e:
        checking_msg.close()
        QMessageBox.warning(
            parent,
            "Update Check Error",
            f"Unexpected error:\n{str(e)}\n\n"
            "Check manually at:\n"
            "https://github.com/aaroncelestian/RamanLab"
        ) 