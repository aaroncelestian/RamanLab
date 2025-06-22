#!/usr/bin/env python3
"""
Test script for the RamanLab Update Checker
This script tests the update checker functionality.
"""

import sys
import os
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
    from PySide6.QtCore import Qt
    from core.update_checker import check_for_updates, UPDATE_CHECKER_AVAILABLE, UpdateChecker
    from version import __version__
    
    print(f"✓ All imports successful")
    print(f"✓ Current RamanLab version: {__version__}")
    print(f"✓ Update checker available: {UPDATE_CHECKER_AVAILABLE}")
    
    if not UPDATE_CHECKER_AVAILABLE:
        print("❌ Update checker dependencies not available!")
        print("   Install with: pip install requests packaging pyperclip")
        sys.exit(1)
    
    class TestWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Update Checker Test")
            self.setGeometry(200, 200, 400, 200)
            
            # Create central widget
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            layout = QVBoxLayout(central_widget)
            
            # Test buttons
            manual_check_btn = QPushButton("Manual Update Check")
            manual_check_btn.clicked.connect(self.manual_check)
            layout.addWidget(manual_check_btn)
            
            silent_check_btn = QPushButton("Silent Update Check")
            silent_check_btn.clicked.connect(self.silent_check)
            layout.addWidget(silent_check_btn)
            
            test_checker_btn = QPushButton("Test UpdateChecker Class")
            test_checker_btn.clicked.connect(self.test_checker_class)
            layout.addWidget(test_checker_btn)
            
        def manual_check(self):
            """Test manual update check with dialog."""
            print("Testing manual update check...")
            check_for_updates(parent=self, show_no_update=True)
            
        def silent_check(self):
            """Test silent update check."""
            print("Testing silent update check...")
            check_for_updates(parent=self, show_no_update=False)
            
        def test_checker_class(self):
            """Test UpdateChecker class directly."""
            print("Testing UpdateChecker class...")
            checker = UpdateChecker(parent=self)
            checker.check_for_updates(parent_widget=self, show_no_update=True)
    
    def main():
        print("Starting Update Checker Test...")
        print("="*50)
        
        app = QApplication(sys.argv)
        window = TestWindow()
        window.show()
        
        print("✓ Test window created")
        print("✓ Click buttons to test different update check methods")
        print("✓ Check console for debug output")
        
        return app.exec()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMissing dependencies. Install with:")
    print("pip install PySide6 requests packaging pyperclip")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)

if __name__ == "__main__":
    sys.exit(main()) 