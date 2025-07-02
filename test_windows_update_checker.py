#!/usr/bin/env python3
"""
Windows-Specific Update Checker Test
Tests the fixed update checker that resolves the hanging dialog issue on Windows.
"""

import sys
import platform
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QTextEdit
    from PySide6.QtCore import Qt, QTimer
    from core.simple_update_checker import simple_check_for_updates, SIMPLE_UPDATE_CHECKER_AVAILABLE
    from version import __version__
    
    print(f"✓ All imports successful")
    print(f"✓ Platform: {platform.system()} {platform.release()}")
    print(f"✓ Current RamanLab version: {__version__}")
    print(f"✓ Update checker available: {SIMPLE_UPDATE_CHECKER_AVAILABLE}")
    
    if not SIMPLE_UPDATE_CHECKER_AVAILABLE:
        print("❌ Update checker dependencies not available!")
        print("   Install with: pip install requests packaging pyperclip")
        sys.exit(1)
    
    class WindowsTestWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Windows Update Checker Test")
            self.setGeometry(200, 200, 600, 400)
            
            # Create central widget
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            layout = QVBoxLayout(central_widget)
            
            # Platform info
            platform_label = QLabel(f"Platform: {platform.system()} {platform.release()}")
            platform_label.setStyleSheet("font-weight: bold; color: blue;")
            layout.addWidget(platform_label)
            
            # Instructions
            instructions = QLabel(
                "This test verifies the Windows update checker fix.\n"
                "The update checker should not hang or get stuck.\n"
                "You should be able to cancel the check if needed."
            )
            instructions.setWordWrap(True)
            layout.addWidget(instructions)
            
            # Test buttons
            test_update_btn = QPushButton("🔍 Test Update Check (Windows Safe)")
            test_update_btn.clicked.connect(self.test_windows_update_check)
            test_update_btn.setStyleSheet("""
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
            layout.addWidget(test_update_btn)
            
            test_silent_btn = QPushButton("🤫 Test Silent Check")
            test_silent_btn.clicked.connect(self.test_silent_check)
            layout.addWidget(test_silent_btn)
            
            # Test multiple rapid checks
            stress_test_btn = QPushButton("⚡ Stress Test (Multiple Rapid Checks)")
            stress_test_btn.clicked.connect(self.stress_test_checks)
            layout.addWidget(stress_test_btn)
            
            # Results area
            self.results_text = QTextEdit()
            self.results_text.setMaximumHeight(150)
            self.results_text.setPlainText("Test results will appear here...\n")
            layout.addWidget(self.results_text)
            
            # Status
            if platform.system().lower() == "windows":
                status_msg = "✅ Running on Windows - using Windows-specific fixes"
                status_color = "green"
            else:
                status_msg = "ℹ️ Not on Windows - using standard implementation"
                status_color = "blue"
            
            status_label = QLabel(status_msg)
            status_label.setStyleSheet(f"color: {status_color}; font-weight: bold;")
            layout.addWidget(status_label)
            
        def log_result(self, message):
            """Log a test result."""
            self.results_text.append(f"[{platform.system()}] {message}")
            self.results_text.repaint()
            
        def test_windows_update_check(self):
            """Test the Windows-safe update check."""
            self.log_result("🔍 Starting Windows-safe update check...")
            
            try:
                simple_check_for_updates(parent=self, show_no_update=True)
                self.log_result("✅ Update check completed successfully")
            except Exception as e:
                self.log_result(f"❌ Update check failed: {str(e)}")
                
        def test_silent_check(self):
            """Test silent update check."""
            self.log_result("🤫 Starting silent update check...")
            
            try:
                simple_check_for_updates(parent=self, show_no_update=False)
                self.log_result("✅ Silent check completed successfully")
            except Exception as e:
                self.log_result(f"❌ Silent check failed: {str(e)}")
                
        def stress_test_checks(self):
            """Stress test with multiple rapid checks."""
            self.log_result("⚡ Starting stress test (3 rapid checks)...")
            
            def run_check(check_num):
                try:
                    self.log_result(f"Running check #{check_num}...")
                    simple_check_for_updates(parent=self, show_no_update=False)
                    self.log_result(f"✅ Check #{check_num} completed")
                except Exception as e:
                    self.log_result(f"❌ Check #{check_num} failed: {str(e)}")
            
            # Schedule multiple checks with delays
            QTimer.singleShot(100, lambda: run_check(1))
            QTimer.singleShot(2000, lambda: run_check(2))
            QTimer.singleShot(4000, lambda: run_check(3))
            QTimer.singleShot(6000, lambda: self.log_result("🏁 Stress test completed"))
    
    def main():
        print("Starting Windows Update Checker Test...")
        print("="*50)
        
        app = QApplication(sys.argv)
        window = WindowsTestWindow()
        window.show()
        
        print("✓ Test window created")
        print("✓ Test the update checker functionality")
        print("✓ Verify it doesn't hang on Windows")
        print("✓ Check console and window for results")
        
        if platform.system().lower() == "windows":
            print("\n🎯 WINDOWS DETECTED:")
            print("   • Using QProgressDialog with Cancel button")
            print("   • Enhanced exception handling")
            print("   • Better modal dialog management")
        else:
            print(f"\n📱 NON-WINDOWS PLATFORM ({platform.system()}):")
            print("   • Using standard QMessageBox approach")
            print("   • Still includes improved exception handling")
        
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