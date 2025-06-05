#!/usr/bin/env python3
"""
Cross-Platform Utilities for RamanLab Qt6
This module demonstrates how Qt6 eliminates the need for platform-specific code.
"""

from pathlib import Path
from typing import Optional, List

from PySide6.QtCore import QStandardPaths, QUrl
from PySide6.QtWidgets import QFileDialog, QMessageBox
from PySide6.QtGui import QDesktopServices


class CrossPlatformFileUtils:
    """
    Utility class providing cross-platform file operations using Qt6.
    
    This replaces all the platform-specific code in the original application.
    """
    
    @staticmethod
    def get_documents_directory() -> str:
        """
        Get the user's Documents directory on any platform.
        
        OLD WAY (platform-specific):
        if platform.system() == "Windows":
            home_dir = os.path.expanduser("~")
            initial_dir = os.path.join(home_dir, "Documents")
            if not os.path.exists(initial_dir):
                initial_dir = home_dir
        else:
            initial_dir = os.path.expanduser("~")
            
        NEW WAY (Qt6):
        Just one line!
        """
        return QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
    
    @staticmethod
    def get_desktop_directory() -> str:
        """Get the user's Desktop directory on any platform."""
        return QStandardPaths.writableLocation(QStandardPaths.DesktopLocation)
    
    @staticmethod
    def get_home_directory() -> str:
        """Get the user's home directory on any platform."""
        return QStandardPaths.writableLocation(QStandardPaths.HomeLocation)
    
    @staticmethod
    def get_application_data_directory() -> str:
        """Get the appropriate directory for application data."""
        return QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    
    @staticmethod
    def open_folder_in_file_manager(folder_path: str) -> bool:
        """
        Open a folder in the system's file manager.
        
        OLD WAY (your current code):
        if platform.system() == "Windows":
            os.startfile(nmf_results_dir)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", nmf_results_dir])
        else:  # Linux
            subprocess.run(["xdg-open", nmf_results_dir])
            
        NEW WAY (Qt6):
        Just one line that works everywhere!
        """
        return QDesktopServices.openUrl(QUrl.fromLocalFile(folder_path))
    
    @staticmethod
    def open_file_in_default_application(file_path: str) -> bool:
        """Open a file in the default application (like double-clicking)."""
        return QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))
    
    @staticmethod
    def show_file_in_folder(file_path: str) -> bool:
        """
        Show a specific file in the folder (like "Show in Finder" on macOS).
        Note: This shows the file selected in the folder.
        """
        # For showing the file selected, we open the parent folder
        folder_path = str(Path(file_path).parent)
        return QDesktopServices.openUrl(QUrl.fromLocalFile(folder_path))
    
    @staticmethod
    def select_file_to_open(parent=None, title="Select File", 
                           initial_dir: Optional[str] = None,
                           file_filter="All files (*.*)") -> Optional[str]:
        """
        Cross-platform file selection dialog.
        
        Returns the selected file path or None if cancelled.
        """
        if initial_dir is None:
            initial_dir = CrossPlatformFileUtils.get_documents_directory()
        
        file_path, _ = QFileDialog.getOpenFileName(
            parent, title, initial_dir, file_filter
        )
        
        return file_path if file_path else None
    
    @staticmethod
    def select_files_to_open(parent=None, title="Select Files",
                            initial_dir: Optional[str] = None,
                            file_filter="All files (*.*)") -> List[str]:
        """
        Cross-platform multiple file selection dialog.
        
        Returns a list of selected file paths (empty list if cancelled).
        """
        if initial_dir is None:
            initial_dir = CrossPlatformFileUtils.get_documents_directory()
        
        file_paths, _ = QFileDialog.getOpenFileNames(
            parent, title, initial_dir, file_filter
        )
        
        return file_paths
    
    @staticmethod
    def select_folder(parent=None, title="Select Folder",
                     initial_dir: Optional[str] = None) -> Optional[str]:
        """
        Cross-platform folder selection dialog.
        
        Returns the selected folder path or None if cancelled.
        """
        if initial_dir is None:
            initial_dir = CrossPlatformFileUtils.get_documents_directory()
        
        folder_path = QFileDialog.getExistingDirectory(
            parent, title, initial_dir
        )
        
        return folder_path if folder_path else None
    
    @staticmethod
    def select_file_to_save(parent=None, title="Save File",
                           initial_dir: Optional[str] = None,
                           file_filter="All files (*.*)") -> Optional[str]:
        """
        Cross-platform file save dialog.
        
        Returns the selected file path or None if cancelled.
        """
        if initial_dir is None:
            initial_dir = CrossPlatformFileUtils.get_documents_directory()
        
        file_path, _ = QFileDialog.getSaveFileName(
            parent, title, initial_dir, file_filter
        )
        
        return file_path if file_path else None
    
    @staticmethod
    def ensure_directory_exists(directory_path: str) -> bool:
        """
        Ensure a directory exists, creating it if necessary.
        
        Returns True if directory exists or was created successfully.
        """
        try:
            Path(directory_path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"Failed to create directory {directory_path}: {e}")
            return False
    
    @staticmethod
    def show_error_message(parent, title: str, message: str):
        """Show a cross-platform error message."""
        QMessageBox.critical(parent, title, message)
    
    @staticmethod
    def show_info_message(parent, title: str, message: str):
        """Show a cross-platform information message."""
        QMessageBox.information(parent, title, message)
    
    @staticmethod
    def ask_yes_no_question(parent, title: str, question: str) -> bool:
        """
        Show a cross-platform yes/no question dialog.
        
        Returns True for Yes, False for No.
        """
        reply = QMessageBox.question(
            parent, title, question,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No  # Default to No for safety
        )
        return reply == QMessageBox.Yes


# Example usage and migration examples
def demonstrate_migration_examples():
    """
    Examples showing how to replace platform-specific code with Qt6 equivalents.
    """
    
    print("=== Cross-Platform File Operations Demo ===")
    
    # Example 1: Getting standard directories
    print("\n1. Standard Directories:")
    print(f"Documents: {CrossPlatformFileUtils.get_documents_directory()}")
    print(f"Desktop: {CrossPlatformFileUtils.get_desktop_directory()}")
    print(f"Home: {CrossPlatformFileUtils.get_home_directory()}")
    print(f"App Data: {CrossPlatformFileUtils.get_application_data_directory()}")
    
    # Example 2: Opening folders (replaces your platform-specific code)
    print("\n2. Opening Folders:")
    documents_dir = CrossPlatformFileUtils.get_documents_directory()
    print(f"Opening Documents folder: {documents_dir}")
    # CrossPlatformFileUtils.open_folder_in_file_manager(documents_dir)
    
    print("\n=== Migration Examples ===")
    
    # Show before/after code examples
    print("""
    BEFORE (platform-specific, your current code):
    =============================================
    import platform
    import subprocess
    import os
    
    if platform.system() == "Windows":
        os.startfile(nmf_results_dir)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", nmf_results_dir])
    else:  # Linux
        subprocess.run(["xdg-open", nmf_results_dir])
    
    AFTER (Qt6, cross-platform):
    ============================
    from cross_platform_utils import CrossPlatformFileUtils
    
    CrossPlatformFileUtils.open_folder_in_file_manager(nmf_results_dir)
    
    That's it! One line that works everywhere!
    """)


if __name__ == "__main__":
    # Run the demonstration
    demonstrate_migration_examples() 