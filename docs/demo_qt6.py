#!/usr/bin/env python3
"""
Simple RamanLab Qt6 Demo - Proof of Concept
"""

import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                               QWidget, QPushButton, QLabel, QMessageBox)
from PySide6.QtCore import QStandardPaths, QUrl
from PySide6.QtGui import QDesktopServices

class RamanLabQt6Demo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RamanLab Qt6 - Cross-Platform Demo")
        self.resize(500, 300)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Welcome message
        welcome = QLabel("🎉 Welcome to RamanLab Qt6!")
        welcome.setStyleSheet("font-size: 18px; font-weight: bold; color: blue;")
        layout.addWidget(welcome)
        
        info = QLabel("This demonstrates cross-platform file operations without platform-specific code!")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Demo buttons
        docs_btn = QPushButton("Show Documents Directory")
        docs_btn.clicked.connect(self.show_docs_dir)
        layout.addWidget(docs_btn)
        
        open_btn = QPushButton("Open Documents Folder")
        open_btn.clicked.connect(self.open_docs_folder)
        layout.addWidget(open_btn)
        
        paths_btn = QPushButton("Show All Standard Paths")
        paths_btn.clicked.connect(self.show_all_paths)
        layout.addWidget(paths_btn)
        
        # Exit button
        exit_btn = QPushButton("Exit")
        exit_btn.clicked.connect(self.close)
        layout.addWidget(exit_btn)
    
    def show_docs_dir(self):
        docs = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
        QMessageBox.information(self, "Documents Directory", 
                               f"Documents: {docs}\n\nNo platform checks needed! 🎯")
    
    def open_docs_folder(self):
        docs = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
        success = QDesktopServices.openUrl(QUrl.fromLocalFile(docs))
        if success:
            QMessageBox.information(self, "Success!", 
                                   "Folder opened! This replaces all your:\n"
                                   "• platform.system() checks\n"
                                   "• subprocess.run() calls\n"
                                   "• os.startfile() code\n\n"
                                   "One line works everywhere! 🚀")
    
    def show_all_paths(self):
        paths = [
            f"Documents: {QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)}",
            f"Desktop: {QStandardPaths.writableLocation(QStandardPaths.DesktopLocation)}",
            f"Home: {QStandardPaths.writableLocation(QStandardPaths.HomeLocation)}",
            f"Downloads: {QStandardPaths.writableLocation(QStandardPaths.DownloadLocation)}",
        ]
        QMessageBox.information(self, "Standard Paths", "\n".join(paths))

def main():
    app = QApplication(sys.argv)
    window = RamanLabQt6Demo()
    window.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main()) 