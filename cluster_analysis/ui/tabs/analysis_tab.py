"""
Analysis Tab for RamanLab Cluster Analysis

This module contains the analysis results tab UI and functionality.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QPushButton, 
                              QTextEdit)


class AnalysisTab(QWidget):
    """Analysis results tab for displaying clustering analysis results."""
    
    def __init__(self, parent_window):
        """Initialize the analysis tab."""
        super().__init__()
        self.parent_window = parent_window
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the analysis tab UI."""
        layout = QVBoxLayout(self)
        
        # Create results display
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)
        
        # Results text area
        self.analysis_results_text = QTextEdit()
        self.analysis_results_text.setReadOnly(True)
        results_layout.addWidget(self.analysis_results_text)
        
        layout.addWidget(results_group)
        
        # Export button
        export_results_btn = QPushButton("Export Analysis Results")
        export_results_btn.clicked.connect(self.parent_window.export_analysis_results)
        layout.addWidget(export_results_btn)
    
    def get_analysis_results_text(self):
        """Get the analysis results text widget."""
        return self.analysis_results_text
    
    def set_analysis_results(self, text):
        """Set the analysis results text."""
        self.analysis_results_text.setPlainText(text)
    
    def append_analysis_results(self, text):
        """Append text to the analysis results."""
        self.analysis_results_text.append(text)
    
    def clear_analysis_results(self):
        """Clear the analysis results."""
        self.analysis_results_text.clear()
