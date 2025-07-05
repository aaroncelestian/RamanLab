"""
Batch Tab Component
Handles batch processing operations for multiple spectra
Simplified version of the original batch processing functionality
"""

from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, 
    QLabel, QProgressBar, QTextEdit, QCheckBox
)
from PySide6.QtCore import Qt

from ..base_tab import BaseTab

# Unified button styles for consistent UI
BUTTON_STYLE = """
    QPushButton {
        background-color: #f8f9fa;
        color: #495057;
        border: 1px solid #dee2e6;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 11px;
        min-width: 50px;
    }
    QPushButton:hover {
        background-color: #e9ecef;
        border: 1px solid #adb5bd;
        color: #212529;
    }
    QPushButton:pressed {
        background-color: #dee2e6;
        border: 1px solid #6c757d;
    }
    QPushButton:disabled {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        color: #6c757d;
        opacity: 0.6;
    }
"""

PRIMARY_BUTTON_STYLE = """
    QPushButton {
        background-color: #0d6efd;
        color: white;
        border: 1px solid #0a58ca;
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 12px;
        min-width: 60px;
    }
    QPushButton:hover {
        background-color: #0b5ed7;
        border: 1px solid #09408e;
    }
    QPushButton:pressed {
        background-color: #0a58ca;
        border: 1px solid #08356d;
    }
    QPushButton:disabled {
        background-color: #6c757d;
        border: 1px solid #5c636a;
        opacity: 0.6;
    }
"""

SUCCESS_BUTTON_STYLE = """
    QPushButton {
        background-color: #198754;
        color: white;
        border: 1px solid #146c43;
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 12px;
        min-width: 60px;
    }
    QPushButton:hover {
        background-color: #157347;
        border: 1px solid #0f5132;
    }
    QPushButton:pressed {
        background-color: #146c43;
        border: 1px solid #0c3a22;
    }
    QPushButton:disabled {
        background-color: #6c757d;
        border: 1px solid #5c636a;
        opacity: 0.6;
    }
"""

DANGER_BUTTON_STYLE = """
    QPushButton {
        background-color: #dc3545;
        color: white;
        border: 1px solid #bb2d3b;
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 12px;
        min-width: 60px;
    }
    QPushButton:hover {
        background-color: #c82333;
        border: 1px solid #a02834;
    }
    QPushButton:pressed {
        background-color: #bb2d3b;
        border: 1px solid #8d2130;
    }
    QPushButton:disabled {
        background-color: #6c757d;
        border: 1px solid #5c636a;
        opacity: 0.6;
    }
"""

INFO_BUTTON_STYLE = """
    QPushButton {
        background-color: #0dcaf0;
        color: #055160;
        border: 1px solid #0aa2c0;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 11px;
        min-width: 50px;
    }
    QPushButton:hover {
        background-color: #31d2f2;
        border: 1px solid #0994a8;
    }
    QPushButton:pressed {
        background-color: #0aa2c0;
        border: 1px solid #087990;
    }
"""

PURPLE_BUTTON_STYLE = """
    QPushButton {
        background-color: #6f42c1;
        color: white;
        border: 1px solid #59359a;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 11px;
        min-width: 50px;
    }
    QPushButton:hover {
        background-color: #59359a;
        border: 1px solid #4c2d83;
    }
    QPushButton:pressed {
        background-color: #4c2d83;
        border: 1px solid #3e246c;
    }
"""


class BatchTab(BaseTab):
    """
    Batch processing component for analyzing multiple spectra.
    Simplified version focusing on core batch operations.
    """
    
    def __init__(self, parent=None):
        # Initialize widgets that will be created
        self.progress_bar = None
        self.status_text = None
        self.batch_button = None
        self.stop_button = None
        self.auto_bg_checkbox = None
        self.auto_peaks_checkbox = None
        
        super().__init__(parent)
        self.tab_name = "Batch"
    
    def setup_ui(self):
        """Create the batch processing UI"""
        
        # Batch controls group
        controls_group = QGroupBox("Batch Processing Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Processing options
        options_layout = QVBoxLayout()
        
        self.auto_bg_checkbox = QCheckBox("Auto-apply background subtraction")
        self.auto_bg_checkbox.setChecked(True)
        options_layout.addWidget(self.auto_bg_checkbox)
        
        self.auto_peaks_checkbox = QCheckBox("Auto-detect peaks (if no reference set)")
        self.auto_peaks_checkbox.setChecked(True)
        options_layout.addWidget(self.auto_peaks_checkbox)
        
        save_results_checkbox = QCheckBox("Save individual results")
        save_results_checkbox.setChecked(True)
        options_layout.addWidget(save_results_checkbox)
        
        controls_layout.addLayout(options_layout)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        self.batch_button = QPushButton("Start")
        self.batch_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        buttons_layout.addWidget(self.batch_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet(DANGER_BUTTON_STYLE)
        buttons_layout.addWidget(self.stop_button)
        
        controls_layout.addLayout(buttons_layout)
        
        # Progress group
        progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                text-align: center;
                height: 25px;
                background-color: #f8f9fa;
            }
            QProgressBar::chunk {
                background-color: #198754;
                border-radius: 3px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        # Progress label
        self.progress_label = QLabel("Ready to process")
        self.progress_label.setStyleSheet("""
            QLabel {
                font-weight: 500;
                color: #495057;
                padding: 5px;
            }
        """)
        progress_layout.addWidget(self.progress_label)
        
        # Status and results group
        status_group = QGroupBox("Processing Status & Results")
        status_layout = QVBoxLayout(status_group)
        
        # Status text
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(200)
        self.status_text.setReadOnly(True)
        self.status_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-family: monospace;
                font-size: 10px;
                color: #495057;
            }
        """)
        status_layout.addWidget(self.status_text)
        
        # Results summary
        results_layout = QHBoxLayout()
        
        self.files_processed_label = QLabel("Files processed: 0")
        results_layout.addWidget(self.files_processed_label)
        
        self.success_count_label = QLabel("Successful: 0")
        self.success_count_label.setStyleSheet("color: #198754; font-weight: 500;")
        results_layout.addWidget(self.success_count_label)
        
        self.failed_count_label = QLabel("Failed: 0")
        self.failed_count_label.setStyleSheet("color: #dc3545; font-weight: 500;")
        results_layout.addWidget(self.failed_count_label)
        
        status_layout.addLayout(results_layout)
        
        # Export controls
        export_group = QGroupBox("Export Results")
        export_layout = QHBoxLayout(export_group)
        
        export_csv_btn = QPushButton("CSV")
        export_csv_btn.setStyleSheet(INFO_BUTTON_STYLE)
        export_layout.addWidget(export_csv_btn)
        
        export_comprehensive_btn = QPushButton("Full")
        export_comprehensive_btn.setStyleSheet(PURPLE_BUTTON_STYLE)
        export_layout.addWidget(export_comprehensive_btn)
        
        # Add to main layout
        self.main_layout.addWidget(controls_group)
        self.main_layout.addWidget(progress_group)
        self.main_layout.addWidget(status_group)
        self.main_layout.addWidget(export_group)
        
        # Store button references
        self.save_results_checkbox = save_results_checkbox
        self.export_csv_btn = export_csv_btn
        self.export_comprehensive_btn = export_comprehensive_btn
        
        # Initialize status
        self.status_text.append("Batch processing ready")
        self.status_text.append("Configure settings and click 'Start'")
    
    def connect_signals(self):
        """Connect internal signals"""
        self.batch_button.clicked.connect(self._start_batch_processing)
        self.stop_button.clicked.connect(self._stop_batch_processing)
        self.export_csv_btn.clicked.connect(self._export_csv)
        self.export_comprehensive_btn.clicked.connect(self._export_comprehensive)
    
    def connect_core_signals(self):
        """Connect to core component signals"""
        if self.data_processor:
            self.data_processor.spectra_list_changed.connect(self._update_file_count)
    
    def _start_batch_processing(self):
        """Start batch processing of all loaded files"""
        # Get processing options
        options = {
            'auto_background': self.auto_bg_checkbox.isChecked(),
            'auto_peaks': self.auto_peaks_checkbox.isChecked(),
            'save_results': self.save_results_checkbox.isChecked()
        }
        
        # Disable start button, enable stop button
        self.batch_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Clear previous status
        self.status_text.clear()
        self.status_text.append("Starting batch processing...")
        
        # Reset progress
        self.progress_bar.setValue(0)
        self.progress_label.setText("Processing...")
        
        # Reset counters
        self._reset_counters()
        
        # Emit action to start processing
        self.emit_action("start_batch_processing", options)
        self.emit_status("Batch processing started")
    
    def _stop_batch_processing(self):
        """Stop batch processing"""
        # Emit action to stop processing
        self.emit_action("stop_batch_processing", {})
        
        # Update UI
        self.batch_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_label.setText("Processing stopped")
        
        self.status_text.append("Batch processing stopped by user")
        self.emit_status("Batch processing stopped")
    
    def _export_csv(self):
        """Export results to CSV format"""
        self.emit_action("export_results", {"format": "csv"})
        self.emit_status("CSV export initiated")
    
    def _export_comprehensive(self):
        """Export comprehensive results"""
        self.emit_action("export_results", {"format": "comprehensive"})
        self.emit_status("Comprehensive export initiated")
    
    def _update_file_count(self, file_list):
        """Update file count display"""
        file_count = len(file_list)
        self.emit_status(f"Loaded {file_count} files for batch processing")
    
    def _update_progress(self, current, total):
        """Update progress bar and label"""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)
            self.progress_label.setText(f"Processing {current}/{total} files ({progress}%)")
    
    def _on_batch_completed(self, results):
        """Handle batch processing completion"""
        # Re-enable controls
        self.batch_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        # Update progress
        self.progress_bar.setValue(100)
        
        # Update status
        total = results.get('total', 0)
        successful = results.get('successful', 0)
        failed = results.get('failed', 0)
        
        self.progress_label.setText(f"Completed: {total} files processed")
        self.status_text.append("")
        self.status_text.append(f"Batch processing completed!")
        self.status_text.append(f"Total files: {total}")
        self.status_text.append(f"Successful: {successful}")
        self.status_text.append(f"Failed: {failed}")
        
        # Update counters
        self._update_counters(total, successful, failed)
        
        self.emit_status(f"Batch processing completed: {successful}/{total} files successful")
    
    def _on_file_processed(self, file_info):
        """Handle individual file processing completion"""
        filename = file_info.get('filename', 'Unknown')
        success = file_info.get('success', False)
        error = file_info.get('error', '')
        
        status_icon = "✓" if success else "✗"
        
        if success:
            self.status_text.append(f"{status_icon} {filename}")
        else:
            self.status_text.append(f"{status_icon} {filename} - Error: {error}")
        
        # Auto-scroll to bottom
        self.status_text.verticalScrollBar().setValue(
            self.status_text.verticalScrollBar().maximum()
        )
    
    def _reset_counters(self):
        """Reset processing counters"""
        self.files_processed_label.setText("Files processed: 0")
        self.success_count_label.setText("Successful: 0")
        self.failed_count_label.setText("Failed: 0")
    
    def _update_counters(self, total, successful, failed):
        """Update processing counter labels"""
        self.files_processed_label.setText(f"Files processed: {total}")
        self.success_count_label.setText(f"Successful: {successful}")
        self.failed_count_label.setText(f"Failed: {failed}")
    
    def get_tab_data(self):
        """Get current tab state"""
        base_data = super().get_tab_data()
        
        base_data.update({
            'auto_background': self.auto_bg_checkbox.isChecked(),
            'auto_peaks': self.auto_peaks_checkbox.isChecked(),
            'save_results': self.save_results_checkbox.isChecked(),
            'progress': self.progress_bar.value(),
            'is_processing': not self.batch_button.isEnabled()
        })
        
        return base_data
    
    def reset_to_defaults(self):
        """Reset tab to default state"""
        self.auto_bg_checkbox.setChecked(True)
        self.auto_peaks_checkbox.setChecked(True)
        self.save_results_checkbox.setChecked(True)
        
        self.progress_bar.setValue(0)
        self.progress_label.setText("Ready to process")
        
        self.status_text.clear()
        self.status_text.append("Batch processing ready")
        self.status_text.append("Configure settings and click 'Start'")
        
        self.batch_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        self._reset_counters()
        
        self.emit_status("Tab reset to defaults")
    
    # Public methods called by main controller
    def reset_batch_counters(self):
        """Reset batch processing counters (public interface)"""
        self._reset_counters()
    
    def update_batch_progress(self, current, total, current_file):
        """Update batch progress (public interface)"""
        self._update_progress(current, total)
        self.status_text.append(f"Processing {current}/{total}: {current_file}")
        
        # Auto-scroll to bottom
        self.status_text.verticalScrollBar().setValue(
            self.status_text.verticalScrollBar().maximum()
        )
    
    def update_batch_result(self, file_number, filename, success, message):
        """Update batch result for individual file (public interface)"""
        file_info = {
            'filename': filename,
            'success': success,
            'error': message if not success else ''
        }
        self._on_file_processed(file_info)
    
    def finish_batch_processing(self, total, successful, failed):
        """Finish batch processing (public interface)"""
        results = {
            'total': total,
            'successful': successful,
            'failed': failed
        }
        self._on_batch_completed(results) 