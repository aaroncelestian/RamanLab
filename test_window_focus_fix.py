#!/usr/bin/env python3
"""
Test script to demonstrate the window focus fix.
This script shows the before/after behavior of file dialogs.
"""

import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, 
    QFileDialog, QMessageBox, QLabel, QHBoxLayout, QTextEdit
)
from PySide6.QtCore import Qt

class WindowFocusTestApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RamanLab Window Focus Fix Test")
        self.setGeometry(100, 100, 600, 400)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("Window Focus Fix Test")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel(
            "This test demonstrates the window focus fix for file dialogs.\n"
            "After clicking either button and using the file dialog,\n"
            "observe whether this window stays in focus or goes to the background."
        )
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet("margin: 10px; color: #666;")
        layout.addWidget(instructions)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        # Test button WITHOUT focus fix
        self.test_without_fix_btn = QPushButton("Test WITHOUT Focus Fix")
        self.test_without_fix_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.test_without_fix_btn.clicked.connect(self.test_without_fix)
        button_layout.addWidget(self.test_without_fix_btn)
        
        # Test button WITH focus fix
        self.test_with_fix_btn = QPushButton("Test WITH Focus Fix")
        self.test_with_fix_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.test_with_fix_btn.clicked.connect(self.test_with_fix)
        button_layout.addWidget(self.test_with_fix_btn)
        
        layout.addLayout(button_layout)
        
        # Results area
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)
        self.results_text.setPlainText("Click a button above to test the window focus behavior...")
        layout.addWidget(self.results_text)
        
        # Status
        self.status_label = QLabel("Ready to test")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("margin: 10px; font-style: italic;")
        layout.addWidget(self.status_label)
        
        # Test counter
        self.test_count = 0
    
    def test_without_fix(self):
        """Test file dialog WITHOUT focus restoration."""
        self.test_count += 1
        self.status_label.setText("Testing WITHOUT focus fix...")
        
        # Show file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Test Save Dialog (WITHOUT Focus Fix)", 
            f"test_file_{self.test_count}.txt",
            "Text files (*.txt);;All files (*.*)"
        )
        
        if file_path:
            # Show success message
            QMessageBox.information(
                self, 
                "File Dialog Result", 
                f"Selected file: {file_path}\n\n"
                "❌ NO focus restoration applied.\n"
                "Notice if this window went to the background."
            )
            
            # Log result
            self.results_text.append(f"Test {self.test_count}: WITHOUT fix - Selected: {file_path}")
            self.status_label.setText("Test completed WITHOUT focus fix")
        else:
            self.results_text.append(f"Test {self.test_count}: WITHOUT fix - Cancelled")
            self.status_label.setText("Test cancelled")
    
    def test_with_fix(self):
        """Test file dialog WITH focus restoration."""
        self.test_count += 1
        self.status_label.setText("Testing WITH focus fix...")
        
        # Show file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Test Save Dialog (WITH Focus Fix)", 
            f"test_file_fixed_{self.test_count}.txt",
            "Text files (*.txt);;All files (*.*)"
        )
        
        if file_path:
            # Show success message
            QMessageBox.information(
                self, 
                "File Dialog Result", 
                f"Selected file: {file_path}\n\n"
                "✅ Focus restoration will be applied.\n"
                "This window should stay in focus!"
            )
            
            # Apply the focus fix
            try:
                from core.window_focus_manager import restore_window_focus_after_dialog
                restore_window_focus_after_dialog(self)
                self.results_text.append(f"Test {self.test_count}: WITH fix - Selected: {file_path} ✅")
                self.status_label.setText("Test completed WITH focus fix ✅")
            except ImportError:
                # Fallback if focus manager not available
                self.raise_()
                self.activateWindow()
                self.results_text.append(f"Test {self.test_count}: WITH fix (fallback) - Selected: {file_path} ⚠️")
                self.status_label.setText("Test completed WITH focus fix (fallback) ⚠️")
        else:
            self.results_text.append(f"Test {self.test_count}: WITH fix - Cancelled")
            self.status_label.setText("Test cancelled")


def main():
    """Main function to run the test application."""
    app = QApplication(sys.argv)
    
    # Create and show the test window
    test_window = WindowFocusTestApp()
    test_window.show()
    
    # Bring window to front initially
    test_window.raise_()
    test_window.activateWindow()
    
    print("🧪 Window Focus Fix Test Application Started")
    print("📋 Instructions:")
    print("   1. Click 'Test WITHOUT Focus Fix' and use the file dialog")
    print("   2. Observe if the main window goes to the background")
    print("   3. Click 'Test WITH Focus Fix' and use the file dialog")
    print("   4. Observe if the main window stays in focus")
    print("   5. Compare the behavior between the two tests")
    print()
    print("✅ The fix should keep the main window in focus after file dialogs")
    print("❌ Without the fix, the window may go to the background")
    
    # Run the application
    return app.exec()


if __name__ == "__main__":
    sys.exit(main()) 