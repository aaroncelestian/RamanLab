"""
Progress Dialog with Real-time Output Display
"""

from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QProgressBar, QLabel
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
import sys
from io import StringIO


class ProgressDialog(QDialog):
    """Dialog showing real-time progress with terminal output."""
    
    def __init__(self, parent=None, title="Processing..."):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(700, 400)
        
        layout = QVBoxLayout(self)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(self.status_label)
        
        # Progress bar (indeterminate for UMAP)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        layout.addWidget(self.progress_bar)
        
        # Output text area
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                border: 1px solid #3e3e3e;
            }
        """)
        layout.addWidget(self.output_text)
        
        # Capture stdout/stderr
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        self.output_buffer = StringIO()
        
        # Timer to update output periodically
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_output)
        self.update_timer.start(100)  # Update every 100ms
        
    def start_capture(self):
        """Start capturing stdout/stderr."""
        sys.stdout = self.output_buffer
        sys.stderr = self.output_buffer
        
    def stop_capture(self):
        """Stop capturing and restore stdout/stderr."""
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.update_timer.stop()
        # Final update
        self.update_output()
        
    def update_output(self):
        """Update the output text area with new content."""
        output = self.output_buffer.getvalue()
        if output:
            self.output_text.setPlainText(output)
            # Auto-scroll to bottom
            scrollbar = self.output_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            
    def set_status(self, status):
        """Update the status label."""
        self.status_label.setText(status)
        
    def closeEvent(self, event):
        """Clean up on close."""
        self.stop_capture()
        event.accept()
