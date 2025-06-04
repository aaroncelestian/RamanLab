#!/usr/bin/env python3
"""
RamanLab Update Checker
Checks for updates from the GitHub repository and provides update options.

@author: AaronCelestian
@version: 2.6.3
RamanLab
"""

import requests
import json
import tkinter as tk
from tkinter import messagebox, ttk
import webbrowser
import subprocess
import sys
import os
from pathlib import Path
import threading
from packaging import version
import re


class UpdateChecker:
    """
    Handles checking for updates and providing update options to users.
    """
    
    def __init__(self, current_version="2.6.3"):
        self.current_version = current_version
        self.github_api_url = "https://api.github.com/repos/aaroncelestian/RamanLab"
        self.github_repo_url = "https://github.com/aaroncelestian/RamanLab"
        self.latest_release = None
        
    def check_for_updates(self, show_no_updates=True):
        """
        Check for updates from GitHub repository.
        
        Args:
            show_no_updates (bool): Whether to show a message when no updates are available
            
        Returns:
            dict: Update information or None if no updates
        """
        try:
            # Check latest release
            response = requests.get(f"{self.github_api_url}/releases/latest", timeout=10)
            
            if response.status_code == 200:
                release_data = response.json()
                latest_version = release_data.get('tag_name', '').lstrip('v')
                
                if self._is_newer_version(latest_version, self.current_version):
                    self.latest_release = release_data
                    return {
                        'update_available': True,
                        'latest_version': latest_version,
                        'current_version': self.current_version,
                        'release_notes': release_data.get('body', ''),
                        'download_url': release_data.get('html_url', ''),
                        'published_at': release_data.get('published_at', ''),
                        'assets': release_data.get('assets', [])
                    }
                else:
                    if show_no_updates:
                        messagebox.showinfo(
                            "No Updates Available",
                            f"You are running the latest version of RamanLab ({self.current_version})."
                        )
                    return {'update_available': False}
                    
            else:
                # Fallback: check commits if no releases
                return self._check_commits()
                
        except requests.RequestException as e:
            messagebox.showerror(
                "Update Check Failed",
                f"Could not check for updates:\n{str(e)}\n\nPlease check your internet connection."
            )
            return None
        except Exception as e:
            messagebox.showerror(
                "Update Check Error",
                f"An error occurred while checking for updates:\n{str(e)}"
            )
            return None
    
    def _check_commits(self):
        """
        Fallback method to check for new commits if no releases are available.
        """
        try:
            response = requests.get(f"{self.github_api_url}/commits", timeout=10)
            if response.status_code == 200:
                commits = response.json()
                if commits:
                    latest_commit = commits[0]
                    return {
                        'update_available': True,
                        'latest_version': 'Latest Commit',
                        'current_version': self.current_version,
                        'release_notes': f"Latest commit: {latest_commit.get('commit', {}).get('message', 'No message')}",
                        'download_url': self.github_repo_url,
                        'published_at': latest_commit.get('commit', {}).get('author', {}).get('date', ''),
                        'commit_sha': latest_commit.get('sha', '')[:7]
                    }
        except:
            pass
        return {'update_available': False}
    
    def _is_newer_version(self, latest, current):
        """
        Compare version numbers to determine if an update is available.
        """
        try:
            # Try semantic version comparison first
            return version.parse(latest) > version.parse(current)
        except:
            # If version parsing fails, check if the latest tag looks like a version number
            # If it doesn't contain digits or version-like patterns, assume no update
            if not re.search(r'\d+\.\d+', latest):
                # Latest tag doesn't look like a version number, assume no update
                return False
            # If it does look like a version, fall back to string comparison
            return latest != current
    
    def show_update_dialog(self, update_info):
        """
        Show a dialog with update information and options.
        """
        if not update_info or not update_info.get('update_available'):
            return
            
        dialog = tk.Toplevel()
        dialog.title("RamanLab Update Available")
        dialog.geometry("600x500")
        dialog.resizable(True, True)
        
        # Make dialog modal
        dialog.transient()
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (600 // 2)
        y = (dialog.winfo_screenheight() // 2) - (500 // 2)
        dialog.geometry(f"600x500+{x}+{y}")
        
        # Main frame
        main_frame = ttk.Frame(dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="🚀 Update Available!", 
            font=('TkDefaultFont', 16, 'bold')
        )
        title_label.pack(pady=(0, 10))
        
        # Version info
        version_frame = ttk.LabelFrame(main_frame, text="Version Information", padding=10)
        version_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(version_frame, text=f"Current Version: {update_info['current_version']}").pack(anchor=tk.W)
        ttk.Label(
            version_frame, 
            text=f"Latest Version: {update_info['latest_version']}", 
            font=('TkDefaultFont', 10, 'bold')
        ).pack(anchor=tk.W)
        
        if update_info.get('published_at'):
            ttk.Label(version_frame, text=f"Released: {update_info['published_at'][:10]}").pack(anchor=tk.W)
        
        # Release notes
        notes_frame = ttk.LabelFrame(main_frame, text="What's New", padding=10)
        notes_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Scrollable text widget for release notes
        text_frame = ttk.Frame(notes_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        notes_text = tk.Text(text_frame, wrap=tk.WORD, height=10)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=notes_text.yview)
        notes_text.configure(yscrollcommand=scrollbar.set)
        
        notes_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Insert release notes
        release_notes = update_info.get('release_notes', 'No release notes available.')
        notes_text.insert(tk.END, release_notes)
        notes_text.config(state=tk.DISABLED)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Update options
        ttk.Button(
            button_frame,
            text="📥 Download Latest Version",
            command=lambda: self._open_download_page(update_info['download_url'])
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="🔄 Auto-Update (Git)",
            command=lambda: self._auto_update_git(dialog)
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="📋 Copy Git Command",
            command=lambda: self._copy_git_command()
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="❌ Close",
            command=dialog.destroy
        ).pack(side=tk.RIGHT)
        
        # Instructions
        instructions = ttk.Label(
            main_frame,
            text="💡 Choose an update method above. Git users can use auto-update or manual commands.",
            font=('TkDefaultFont', 9),
            foreground='gray'
        )
        instructions.pack(pady=(10, 0))
    
    def _open_download_page(self, url):
        """Open the download page in the default web browser."""
        webbrowser.open(url)
    
    def _auto_update_git(self, parent_dialog):
        """
        Attempt to auto-update using git if the repository is a git clone.
        """
        def update_thread():
            try:
                # Check if we're in a git repository
                result = subprocess.run(['git', 'status'], 
                                      capture_output=True, text=True, cwd=os.getcwd())
                
                if result.returncode != 0:
                    messagebox.showerror(
                        "Git Not Available",
                        "This doesn't appear to be a git repository.\n"
                        "Please use the download option or clone the repository manually."
                    )
                    return
                
                # Show progress dialog
                progress_dialog = tk.Toplevel(parent_dialog)
                progress_dialog.title("Updating...")
                progress_dialog.geometry("400x150")
                progress_dialog.transient(parent_dialog)
                progress_dialog.grab_set()
                
                # Center progress dialog
                progress_dialog.update_idletasks()
                x = (progress_dialog.winfo_screenwidth() // 2) - (200)
                y = (progress_dialog.winfo_screenheight() // 2) - (75)
                progress_dialog.geometry(f"400x150+{x}+{y}")
                
                progress_frame = ttk.Frame(progress_dialog, padding=20)
                progress_frame.pack(fill=tk.BOTH, expand=True)
                
                status_label = ttk.Label(progress_frame, text="Updating RamanLab...")
                status_label.pack(pady=(0, 10))
                
                progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
                progress_bar.pack(fill=tk.X, pady=(0, 10))
                progress_bar.start()
                
                # Perform git pull
                status_label.config(text="Fetching latest changes...")
                progress_dialog.update()
                
                result = subprocess.run(['git', 'pull'], 
                                      capture_output=True, text=True, cwd=os.getcwd())
                
                progress_bar.stop()
                progress_dialog.destroy()
                
                if result.returncode == 0:
                    messagebox.showinfo(
                        "Update Successful",
                        "RamanLab has been updated successfully!\n"
                        "Please restart the application to use the new version."
                    )
                    parent_dialog.destroy()
                else:
                    messagebox.showerror(
                        "Update Failed",
                        f"Git pull failed:\n{result.stderr}\n\n"
                        "You may need to resolve conflicts manually or use the download option."
                    )
                    
            except FileNotFoundError:
                messagebox.showerror(
                    "Git Not Found",
                    "Git is not installed or not in your PATH.\n"
                    "Please install Git or use the download option."
                )
            except Exception as e:
                messagebox.showerror(
                    "Update Error",
                    f"An error occurred during update:\n{str(e)}"
                )
        
        # Run update in separate thread to avoid blocking UI
        threading.Thread(target=update_thread, daemon=True).start()
    
    def _copy_git_command(self):
        """Copy git pull command to clipboard."""
        try:
            import pyperclip
            pyperclip.copy("git pull")
            messagebox.showinfo(
                "Command Copied",
                "Git pull command copied to clipboard!\n\n"
                "Run 'git pull' in your RamanLab directory to update."
            )
        except ImportError:
            # Fallback if pyperclip not available
            messagebox.showinfo(
                "Git Command",
                "To update manually, run this command in your RamanLab directory:\n\n"
                "git pull\n\n"
                "Then restart the application."
            )
    
    def check_for_updates_async(self, callback=None, show_no_updates=True):
        """
        Check for updates asynchronously to avoid blocking the UI.
        
        Args:
            callback: Function to call with update results
            show_no_updates: Whether to show message when no updates available
        """
        def check_thread():
            update_info = self.check_for_updates(show_no_updates)
            if callback:
                callback(update_info)
            elif update_info and update_info.get('update_available'):
                # Show update dialog in main thread
                self.show_update_dialog(update_info)
        
        threading.Thread(target=check_thread, daemon=True).start()


def create_update_menu_item(parent_menu, current_version="2.6.3"):
    """
    Create an update menu item for the main application.
    
    Args:
        parent_menu: The menu to add the update item to
        current_version: Current version of the application
    """
    checker = UpdateChecker(current_version)
    
    def check_updates():
        checker.check_for_updates_async(
            callback=lambda info: checker.show_update_dialog(info) if info and info.get('update_available') else None
        )
    
    parent_menu.add_command(label="Check for Updates", command=check_updates)
    return checker


def main():
    """Standalone update checker for testing."""
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    checker = UpdateChecker()
    update_info = checker.check_for_updates()
    
    if update_info and update_info.get('update_available'):
        root.deiconify()  # Show window for dialog
        checker.show_update_dialog(update_info)
        root.mainloop()
    else:
        root.destroy()


if __name__ == "__main__":
    main() 