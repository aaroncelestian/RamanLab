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
        
        self.batch_button = QPushButton("Start Batch Processing")
        self.batch_button.setStyleSheet("""
            QPushButton {
                background-color: #388E3C;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2E7D32;
            }
            QPushButton:pressed {
                background-color: #1B5E20;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
                color: #757575;
            }
        """)
        buttons_layout.addWidget(self.batch_button)
        
        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #D32F2F;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #C62828;
            }
            QPushButton:pressed {
                background-color: #B71C1C;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
                color: #757575;
            }
        """)
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
                border: 1px solid #BDBDBD;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 4px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        # Progress label
        self.progress_label = QLabel("Ready to process")
        self.progress_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #424242;
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
                background-color: #FAFAFA;
                border: 1px solid #E0E0E0;
                border-radius: 3px;
                font-family: monospace;
                font-size: 10px;
            }
        """)
        status_layout.addWidget(self.status_text)
        
        # Results summary
        results_layout = QHBoxLayout()
        
        self.files_processed_label = QLabel("Files processed: 0")
        results_layout.addWidget(self.files_processed_label)
        
        self.success_count_label = QLabel("Successful: 0")
        self.success_count_label.setStyleSheet("color: #2E7D32; font-weight: bold;")
        results_layout.addWidget(self.success_count_label)
        
        self.failed_count_label = QLabel("Failed: 0")
        self.failed_count_label.setStyleSheet("color: #D32F2F; font-weight: bold;")
        results_layout.addWidget(self.failed_count_label)
        
        status_layout.addLayout(results_layout)
        
        # Export controls
        export_group = QGroupBox("Export Results")
        export_layout = QHBoxLayout(export_group)
        
        export_csv_btn = QPushButton("Export to CSV")
        export_csv_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
        """)
        export_layout.addWidget(export_csv_btn)
        
        export_comprehensive_btn = QPushButton("Export Comprehensive")
        export_comprehensive_btn.setStyleSheet("""
            QPushButton {
                background-color: #7B1FA2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #6A1B9A;
            }
        """)
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
        self.status_text.append("Configure settings and click 'Start Batch Processing'")
    
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
        
        # Validate that we have files to process
        if self.data_processor:
            file_list = self.data_processor.get_file_list()
            if not file_list:
                self.status_text.append("❌ No files loaded for processing")
                return
            
            # Update UI state
            self.batch_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.progress_bar.setValue(0)
            self.progress_label.setText(f"Processing {len(file_list)} files...")
            
            # Clear status
            self.status_text.clear()
            self.status_text.append(f"🚀 Starting batch processing of {len(file_list)} files")
            self.status_text.append(f"Options: BG={options['auto_background']}, Peaks={options['auto_peaks']}")
            
            # Emit action for main controller to handle
            self.emit_action("start_batch_processing", options)
            self.emit_status("Batch processing started")
        else:
            self.status_text.append("❌ No data processor available")
    
    def _stop_batch_processing(self):
        """Stop batch processing"""
        self.batch_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_label.setText("Processing stopped")
        
        self.status_text.append("⏹️ Batch processing stopped by user")
        self.emit_action("stop_batch_processing", {})
        self.emit_status("Batch processing stopped")
    
    def _export_csv(self):
        """Export results to CSV"""
        self.emit_action("export_results", {"format": "csv"})
        self.emit_status("CSV export initiated")
    
    def _export_comprehensive(self):
        """Export comprehensive results"""
        self.emit_action("export_results", {"format": "comprehensive"})
        self.emit_status("Comprehensive export initiated")
    
    def _update_file_count(self, file_list):
        """Update file count display"""
        count = len(file_list)
        self.progress_label.setText(f"Ready to process {count} files")
        
        # Enable/disable batch button based on file availability
        self.batch_button.setEnabled(count > 0)
    
    def update_batch_progress(self, current, total, current_file=""):
        """Update batch processing progress"""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)
            self.progress_label.setText(f"Processing {current}/{total}: {current_file}")
            
            # Update status
            if current_file:
                self.status_text.append(f"📊 Processing file {current}/{total}: {current_file}")
    
    def update_batch_result(self, file_index, file_name, success, message=""):
        """Update results for a single file"""
        if success:
            self.status_text.append(f"✅ {file_index}: {file_name} - Success")
            self._increment_success_count()
        else:
            self.status_text.append(f"❌ {file_index}: {file_name} - Failed: {message}")
            self._increment_failed_count()
        
        self._increment_processed_count()
        
        # Auto-scroll to bottom
        self.status_text.verticalScrollBar().setValue(
            self.status_text.verticalScrollBar().maximum()
        )
    
    def finish_batch_processing(self, total_processed, successful, failed):
        """Complete batch processing"""
        self.batch_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)
        self.progress_label.setText(f"Processing complete: {successful} successful, {failed} failed")
        
        self.status_text.append("")
        self.status_text.append("🎉 Batch processing completed!")
        self.status_text.append(f"📈 Summary: {total_processed} processed, {successful} successful, {failed} failed")
        
        self.emit_status(f"Batch complete: {successful}/{total_processed} successful")
    
    def _increment_processed_count(self):
        """Increment processed files count"""
        current = self._extract_count(self.files_processed_label.text())
        self.files_processed_label.setText(f"Files processed: {current + 1}")
    
    def _increment_success_count(self):
        """Increment successful files count"""
        current = self._extract_count(self.success_count_label.text())
        self.success_count_label.setText(f"Successful: {current + 1}")
    
    def _increment_failed_count(self):
        """Increment failed files count"""
        current = self._extract_count(self.failed_count_label.text())
        self.failed_count_label.setText(f"Failed: {current + 1}")
    
    def _extract_count(self, text):
        """Extract count from label text"""
        try:
            return int(text.split(": ")[1])
        except:
            return 0
    
    def reset_batch_counters(self):
        """Reset all batch counters"""
        self.files_processed_label.setText("Files processed: 0")
        self.success_count_label.setText("Successful: 0")
        self.failed_count_label.setText("Failed: 0")
        self.progress_bar.setValue(0)
        self.progress_label.setText("Ready to process")
        self.status_text.clear()
        self.status_text.append("Batch processing ready")
    
    def get_tab_data(self):
        """Get current tab state"""
        base_data = super().get_tab_data()
        
        base_data.update({
            'auto_background': self.auto_bg_checkbox.isChecked() if self.auto_bg_checkbox else True,
            'auto_peaks': self.auto_peaks_checkbox.isChecked() if self.auto_peaks_checkbox else True,
            'save_results': self.save_results_checkbox.isChecked() if self.save_results_checkbox else True,
            'processed_count': self._extract_count(self.files_processed_label.text()) if self.files_processed_label else 0,
            'success_count': self._extract_count(self.success_count_label.text()) if self.success_count_label else 0,
            'failed_count': self._extract_count(self.failed_count_label.text()) if self.failed_count_label else 0
        })
        
        return base_data
    
    def reset_to_defaults(self):
        """Reset tab to default state"""
        if self.auto_bg_checkbox:
            self.auto_bg_checkbox.setChecked(True)
        if self.auto_peaks_checkbox:
            self.auto_peaks_checkbox.setChecked(True)
        if self.save_results_checkbox:
            self.save_results_checkbox.setChecked(True)
        
        self.reset_batch_counters()
        self.emit_status("Tab reset to defaults") 