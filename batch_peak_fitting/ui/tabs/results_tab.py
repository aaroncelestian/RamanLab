"""
Results Tab Component
Displays analysis results, statistics, and provides export functionality
Simplified version of the original results functionality
"""

from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QTextEdit,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QWidget
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

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
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 11px;
        min-width: 50px;
    }
    QPushButton:hover {
        background-color: #0b5ed7;
        border: 1px solid #09408e;
    }
    QPushButton:pressed {
        background-color: #0a58ca;
        border: 1px solid #08356d;
    }
"""

SUCCESS_BUTTON_STYLE = """
    QPushButton {
        background-color: #198754;
        color: white;
        border: 1px solid #146c43;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 11px;
        min-width: 50px;
    }
    QPushButton:hover {
        background-color: #157347;
        border: 1px solid #0f5132;
    }
    QPushButton:pressed {
        background-color: #146c43;
        border: 1px solid #0c3a22;
    }
"""

WARNING_BUTTON_STYLE = """
    QPushButton {
        background-color: #fd7e14;
        color: white;
        border: 1px solid #e76500;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 11px;
        min-width: 50px;
    }
    QPushButton:hover {
        background-color: #e76500;
        border: 1px solid #bf5700;
    }
    QPushButton:pressed {
        background-color: #d65d00;
        border: 1px solid #a04800;
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


class ResultsTab(BaseTab):
    """
    Results display and management component.
    Shows analysis results, statistics, and provides export functionality.
    """
    
    def __init__(self, parent=None):
        # Initialize widgets that will be created
        self.results_table = None
        self.summary_text = None
        self.stats_display = None
        
        super().__init__(parent)
        self.tab_name = "Results"
    
    def setup_ui(self):
        """Create the results display UI"""
        
        # Create sub-tabs for different result views
        results_tabs = QTabWidget()
        self.main_layout.addWidget(results_tabs)
        
        # Summary tab
        summary_tab = self._create_summary_tab()
        results_tabs.addTab(summary_tab, "Summary")
        
        # Detailed results tab
        detailed_tab = self._create_detailed_tab()
        results_tabs.addTab(detailed_tab, "Detailed Results")
        
        # Statistics tab
        stats_tab = self._create_statistics_tab()
        results_tabs.addTab(stats_tab, "Statistics")
    
    def _create_summary_tab(self):
        """Create summary results view"""
        container = QVBoxLayout()
        
        # Summary statistics group
        summary_group = QGroupBox("Processing Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        # Summary labels
        self.files_total_label = QLabel("Total files: 0")
        self.files_processed_label = QLabel("Processed: 0")
        self.files_successful_label = QLabel("Successful: 0")
        self.files_failed_label = QLabel("Failed: 0")
        
        # Style summary labels
        label_style = """
            QLabel {
                font-size: 12px;
                font-weight: 500;
                padding: 3px;
                margin: 2px;
                color: #495057;
            }
        """
        
        for label in [self.files_total_label, self.files_processed_label, 
                     self.files_successful_label, self.files_failed_label]:
            label.setStyleSheet(label_style)
        
        self.files_successful_label.setStyleSheet(label_style + "color: #198754;")
        self.files_failed_label.setStyleSheet(label_style + "color: #dc3545;")
        
        summary_layout.addWidget(self.files_total_label)
        summary_layout.addWidget(self.files_processed_label)
        summary_layout.addWidget(self.files_successful_label)
        summary_layout.addWidget(self.files_failed_label)
        
        # Quick actions
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QHBoxLayout(actions_group)
        
        view_all_btn = QPushButton("View All")
        view_all_btn.setStyleSheet(PRIMARY_BUTTON_STYLE)
        actions_layout.addWidget(view_all_btn)
        
        export_summary_btn = QPushButton("Export")
        export_summary_btn.setStyleSheet(SUCCESS_BUTTON_STYLE)
        actions_layout.addWidget(export_summary_btn)
        
        clear_results_btn = QPushButton("Clear")
        clear_results_btn.setStyleSheet(WARNING_BUTTON_STYLE)
        actions_layout.addWidget(clear_results_btn)
        
        # Recent results summary
        recent_group = QGroupBox("Recent Results Summary")
        recent_layout = QVBoxLayout(recent_group)
        
        self.summary_text = QTextEdit()
        self.summary_text.setMaximumHeight(200)
        self.summary_text.setReadOnly(True)
        self.summary_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-family: monospace;
                font-size: 10px;
                color: #495057;
            }
        """)
        recent_layout.addWidget(self.summary_text)
        
        container.addWidget(summary_group)
        container.addWidget(actions_group)
        container.addWidget(recent_group)
        
        # Store button references
        self.view_all_btn = view_all_btn
        self.export_summary_btn = export_summary_btn
        self.clear_results_btn = clear_results_btn
        
        # Initialize summary
        self.summary_text.append("No results available")
        self.summary_text.append("Run batch processing to see results here")
        
        # Create widget to contain the layout
        widget = QWidget()
        widget.setLayout(container)
        return widget
    
    def _create_detailed_tab(self):
        """Create detailed results table view"""
        container = QVBoxLayout()
        
        # Results table
        table_group = QGroupBox("Detailed Results")
        table_layout = QVBoxLayout(table_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "File", "Status", "Peaks Found", "R²", "Processing Time", "Notes"
        ])
        
        # Configure table
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # File column stretches
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        table_layout.addWidget(self.results_table)
        
        # Table controls
        table_controls_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("All")
        select_all_btn.clicked.connect(self.results_table.selectAll)
        select_all_btn.setStyleSheet(BUTTON_STYLE)
        table_controls_layout.addWidget(select_all_btn)
        
        clear_selection_btn = QPushButton("None")
        clear_selection_btn.clicked.connect(self.results_table.clearSelection)
        clear_selection_btn.setStyleSheet(BUTTON_STYLE)
        table_controls_layout.addWidget(clear_selection_btn)
        
        export_selected_btn = QPushButton("Export")
        export_selected_btn.setStyleSheet(PURPLE_BUTTON_STYLE)
        table_controls_layout.addWidget(export_selected_btn)
        
        table_layout.addLayout(table_controls_layout)
        
        container.addWidget(table_group)
        
        # Store button references
        self.select_all_btn = select_all_btn
        self.clear_selection_btn = clear_selection_btn
        self.export_selected_btn = export_selected_btn
        
        # Create widget to contain the layout
        widget = QWidget()
        widget.setLayout(container)
        return widget
    
    def _create_statistics_tab(self):
        """Create statistics overview"""
        container = QVBoxLayout()
        
        # Statistics display
        stats_group = QGroupBox("Analysis Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_display = QTextEdit()
        self.stats_display.setReadOnly(True)
        self.stats_display.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-family: monospace;
                font-size: 11px;
                color: #495057;
            }
        """)
        stats_layout.addWidget(self.stats_display)
        
        # Statistics controls
        stats_controls_layout = QHBoxLayout()
        
        refresh_stats_btn = QPushButton("Refresh")
        refresh_stats_btn.setStyleSheet(PRIMARY_BUTTON_STYLE)
        stats_controls_layout.addWidget(refresh_stats_btn)
        
        export_stats_btn = QPushButton("Export")
        export_stats_btn.setStyleSheet(SUCCESS_BUTTON_STYLE)
        stats_controls_layout.addWidget(export_stats_btn)
        
        stats_layout.addLayout(stats_controls_layout)
        
        container.addWidget(stats_group)
        
        # Store button references
        self.refresh_stats_btn = refresh_stats_btn
        self.export_stats_btn = export_stats_btn
        
        # Initialize statistics display
        self._initialize_stats_display()
        
        # Create widget to contain the layout
        widget = QWidget()
        widget.setLayout(container)
        return widget
    
    def connect_signals(self):
        """Connect internal signals"""
        # Summary actions
        self.view_all_btn.clicked.connect(self._view_all_results)
        self.export_summary_btn.clicked.connect(self._export_summary)
        self.clear_results_btn.clicked.connect(self._clear_results)
        
        # Table actions
        self.export_selected_btn.clicked.connect(self._export_selected)
        
        # Statistics actions
        self.refresh_stats_btn.clicked.connect(self._refresh_statistics)
        self.export_stats_btn.clicked.connect(self._export_statistics)
    
    def connect_core_signals(self):
        """Connect to core component signals"""
        if self.data_processor:
            # Connect to available signals
            if hasattr(self.data_processor, 'file_operation_completed'):
                self.data_processor.file_operation_completed.connect(
                    lambda op, success, msg: self._on_file_operation(op, success, msg)
                )
            
        if self.peak_fitter:
            # Connect to available signals
            if hasattr(self.peak_fitter, 'peaks_fitted'):
                self.peak_fitter.peaks_fitted.connect(self._on_peaks_fitted)
            if hasattr(self.peak_fitter, 'fitting_completed'):
                self.peak_fitter.fitting_completed.connect(self._on_fitting_completed)
    
    def _on_file_operation(self, operation, success, message):
        """Handle file operation completion"""
        if operation == "batch_processing":
            # This could be a batch processing result
            self.emit_status(f"Batch operation: {message}")
    
    def _on_peaks_fitted(self, results):
        """Handle peak fitting results"""
        # Add the fitting result to the table
        self._add_result_to_table(results)
        
    def _on_fitting_completed(self, results):
        """Handle fitting completion (alias for peaks fitted)"""
        self._on_peaks_fitted(results)

    def _view_all_results(self):
        """View all results in detail"""
        self.emit_action("view_all_results", {})
        self.emit_status("Detailed results view requested")
    
    def _export_summary(self):
        """Export summary results"""
        self.emit_action("export_results", {"format": "summary"})
        self.emit_status("Summary export initiated")
    
    def _clear_results(self):
        """Clear all results"""
        self.results_table.setRowCount(0)
        self.summary_text.clear()
        self.summary_text.append("Results cleared")
        self._update_summary_labels(0, 0, 0, 0)
        self._initialize_stats_display()
        
        self.emit_action("clear_results", {})
        self.emit_status("Results cleared")
    
    def _export_selected(self):
        """Export selected results"""
        selected_rows = set()
        for item in self.results_table.selectedItems():
            selected_rows.add(item.row())
        
        if not selected_rows:
            self.emit_status("No results selected for export")
            return
        
        self.emit_action("export_results", {
            "format": "selected",
            "selected_rows": list(selected_rows)
        })
        self.emit_status(f"Export initiated for {len(selected_rows)} selected results")
    
    def _refresh_statistics(self):
        """Refresh statistics display"""
        self.emit_action("refresh_statistics", {})
        self.emit_status("Statistics refresh requested")
    
    def _export_statistics(self):
        """Export statistics"""
        self.emit_action("export_results", {"format": "statistics"})
        self.emit_status("Statistics export initiated")
    
    def _add_result_to_table(self, result_data):
        """Add a single result to the results table"""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        # Extract data with defaults
        file_name = result_data.get('file_name', 'Unknown')
        status = "Success" if result_data.get('success', False) else "Failed"
        peaks_count = len(result_data.get('peak_positions', []))
        r_squared = result_data.get('r_squared', 0.0)
        processing_time = result_data.get('processing_time', 0.0)
        notes = result_data.get('notes', '')
        
        # Create table items
        self.results_table.setItem(row, 0, QTableWidgetItem(file_name))
        
        status_item = QTableWidgetItem(status)
        if status == "Success":
            status_item.setBackground(QColor(209, 237, 255))  # Light blue
        else:
            status_item.setBackground(QColor(248, 215, 218))  # Light red
        self.results_table.setItem(row, 1, status_item)
        
        self.results_table.setItem(row, 2, QTableWidgetItem(str(peaks_count)))
        self.results_table.setItem(row, 3, QTableWidgetItem(f"{r_squared:.4f}"))
        self.results_table.setItem(row, 4, QTableWidgetItem(f"{processing_time:.2f}s"))
        self.results_table.setItem(row, 5, QTableWidgetItem(notes))
        
        # Update summary
        self._update_summary_from_table()
    
    def _update_summary_from_table(self):
        """Update summary statistics from table data"""
        total_rows = self.results_table.rowCount()
        successful = 0
        failed = 0
        
        for row in range(total_rows):
            status_item = self.results_table.item(row, 1)
            if status_item and status_item.text() == "Success":
                successful += 1
            else:
                failed += 1
        
        self._update_summary_labels(total_rows, total_rows, successful, failed)
    
    def _update_summary_labels(self, total, processed, successful, failed):
        """Update summary labels with current counts"""
        self.files_total_label.setText(f"Total files: {total}")
        self.files_processed_label.setText(f"Processed: {processed}")
        self.files_successful_label.setText(f"Successful: {successful}")
        self.files_failed_label.setText(f"Failed: {failed}")
    
    def _initialize_stats_display(self):
        """Initialize statistics display with default content"""
        self.stats_display.clear()
        self.stats_display.append("Statistical Analysis")
        self.stats_display.append("=" * 40)
        self.stats_display.append("")
        self.stats_display.append("No data available for analysis")
        self.stats_display.append("")
        self.stats_display.append("Process some spectra to see:")
        self.stats_display.append("• Peak count distribution")
        self.stats_display.append("• Fitting quality metrics")
        self.stats_display.append("• Processing time statistics")
        self.stats_display.append("• Success rate analysis")
    
    def update_from_data_processor(self, data=None):
        """Update results when data processor state changes"""
        # This could refresh the display based on current data processor state
        pass
    
    def get_tab_data(self):
        """Get current tab state"""
        base_data = super().get_tab_data()
        
        # Add results-specific data
        results_data = []
        for row in range(self.results_table.rowCount()):
            row_data = {}
            for col in range(self.results_table.columnCount()):
                item = self.results_table.item(row, col)
                header = self.results_table.horizontalHeaderItem(col).text()
                row_data[header.lower().replace(' ', '_')] = item.text() if item else ""
            results_data.append(row_data)
        
        base_data.update({
            'results_count': self.results_table.rowCount(),
            'results_data': results_data
        })
        
        return base_data
    
    def reset_to_defaults(self):
        """Reset tab to default state"""
        self._clear_results()
        self.emit_status("Tab reset to defaults") 