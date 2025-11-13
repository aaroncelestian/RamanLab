"""
Refinement Tab for RamanLab Cluster Analysis

This module contains the cluster refinement tab UI and functionality.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                              QFrame, QLabel, QComboBox, QSpinBox, QPushButton,
                              QTextEdit)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from core.matplotlib_config import CompactNavigationToolbar as NavigationToolbar


class RefinementTab(QWidget):
    """Cluster refinement tab for interactive cluster manipulation."""
    
    def __init__(self, parent_window):
        """Initialize the refinement tab."""
        super().__init__()
        self.parent_window = parent_window
        self.selected_points = set()
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the refinement tab UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)  # Reduce overall spacing
        
        # Create controls frame with tighter layout
        controls_group = QGroupBox("Refinement Controls")
        controls_layout = QVBoxLayout(controls_group)
        controls_layout.setSpacing(6)  # Tighter spacing within group
        
        # Visualization method control
        viz_frame = QFrame()
        viz_layout = QHBoxLayout(viz_frame)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        viz_layout.setSpacing(8)
        
        viz_layout.addWidget(QLabel("Visualization:"))
        self.refinement_viz_combo = QComboBox()
        self.refinement_viz_combo.addItems(['PCA'])
        # Add UMAP if available
        try:
            import umap
            self.refinement_viz_combo.addItem('UMAP')
        except ImportError:
            pass
        self.refinement_viz_combo.setCurrentText('PCA')
        self.refinement_viz_combo.currentTextChanged.connect(self.parent_window.update_refinement_plot)
        viz_layout.addWidget(self.refinement_viz_combo)
        
        viz_layout.addStretch()
        
        controls_layout.addWidget(viz_frame)
        
        # Mode control row
        mode_frame = QFrame()
        mode_layout = QHBoxLayout(mode_frame)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(8)
        
        self.refinement_mode_btn = QPushButton("Start Refinement Mode")
        self.refinement_mode_btn.clicked.connect(self.parent_window.toggle_refinement_mode)
        mode_layout.addWidget(self.refinement_mode_btn)
        
        self.undo_btn = QPushButton("Undo Last Action")
        self.undo_btn.clicked.connect(self.parent_window.undo_last_action)
        self.undo_btn.setEnabled(False)
        mode_layout.addWidget(self.undo_btn)
        
        controls_layout.addWidget(mode_frame)
        
        # Split controls group
        split_group = QGroupBox("Split Cluster")
        split_group.setMaximumHeight(80)  # Constrain height
        split_layout = QHBoxLayout(split_group)
        split_layout.setSpacing(6)
        
        split_layout.addWidget(QLabel("Method:"))
        self.split_method_combo = QComboBox()
        self.split_method_combo.addItems(['kmeans', 'hierarchical'])
        self.split_method_combo.setCurrentText('kmeans')
        self.split_method_combo.setMaximumWidth(100)
        split_layout.addWidget(self.split_method_combo)
        
        split_layout.addWidget(QLabel("Subclusters:"))
        self.n_subclusters_spinbox = QSpinBox()
        self.n_subclusters_spinbox.setRange(2, 10)
        self.n_subclusters_spinbox.setValue(2)
        self.n_subclusters_spinbox.setMaximumWidth(60)
        split_layout.addWidget(self.n_subclusters_spinbox)
        
        split_layout.addStretch()
        
        self.split_btn = QPushButton("Split Selected Cluster")
        self.split_btn.clicked.connect(self.parent_window.split_selected_cluster)
        self.split_btn.setEnabled(False)
        split_layout.addWidget(self.split_btn)
        
        controls_layout.addWidget(split_group)
        
        # Cluster operations row
        operations_frame = QFrame()
        operations_layout = QHBoxLayout(operations_frame)
        operations_layout.setContentsMargins(0, 0, 0, 0)
        operations_layout.setSpacing(8)
        
        self.merge_btn = QPushButton("Merge Selected Clusters")
        self.merge_btn.clicked.connect(self.parent_window.merge_selected_clusters)
        self.merge_btn.setEnabled(False)
        operations_layout.addWidget(self.merge_btn)
        
        self.reset_selection_btn = QPushButton("Reset Selection")
        self.reset_selection_btn.clicked.connect(self.parent_window.reset_selection)
        self.reset_selection_btn.setEnabled(False)
        operations_layout.addWidget(self.reset_selection_btn)
        
        operations_layout.addStretch()
        
        controls_layout.addWidget(operations_frame)
        
        # Apply/Cancel row
        action_frame = QFrame()
        action_layout = QHBoxLayout(action_frame)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)
        
        self.apply_refinement_btn = QPushButton("Apply Refinement")
        self.apply_refinement_btn.clicked.connect(self.parent_window.apply_refinement)
        self.apply_refinement_btn.setEnabled(False)
        self.apply_refinement_btn.setStyleSheet("""
            QPushButton {
                background-color: #5CB85C;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #449D44;
            }
            QPushButton:disabled {
                background-color: #CCC;
                color: #888;
            }
        """)
        action_layout.addWidget(self.apply_refinement_btn)
        
        self.cancel_refinement_btn = QPushButton("Cancel Refinement")
        self.cancel_refinement_btn.clicked.connect(self.parent_window.cancel_refinement)
        self.cancel_refinement_btn.setEnabled(False)
        self.cancel_refinement_btn.setStyleSheet("""
            QPushButton {
                background-color: #D9534F;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C9302C;
            }
            QPushButton:disabled {
                background-color: #CCC;
                color: #888;
            }
        """)
        action_layout.addWidget(self.cancel_refinement_btn)
        
        action_layout.addStretch()
        
        controls_layout.addWidget(action_frame)
        
        layout.addWidget(controls_group)
        
        # Selection status with better styling
        status_frame = QFrame()
        status_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; }")
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(8, 4, 8, 4)
        
        status_icon = QLabel("ⓘ")
        status_icon.setStyleSheet("color: #007bff; font-weight: bold; font-size: 12px;")
        status_layout.addWidget(status_icon)
        
        self.selection_status = QLabel("No clusters selected")
        self.selection_status.setStyleSheet("color: #495057; font-size: 11px;")
        status_layout.addWidget(self.selection_status)
        
        status_layout.addStretch()
        
        layout.addWidget(status_frame)
        
        # Create refinement visualization with minimal spacing
        self.refinement_fig = Figure(figsize=(10, 6))
        self.refinement_ax = self.refinement_fig.add_subplot(111)
        self.refinement_canvas = FigureCanvas(self.refinement_fig)
        layout.addWidget(self.refinement_canvas)
        
        # Add toolbar
        self.refinement_toolbar = NavigationToolbar(self.refinement_canvas, self)
        layout.addWidget(self.refinement_toolbar)
        
        # Initialize refinement state
        self.update_refinement_controls()
    
    def get_refinement_controls(self):
        """Get all refinement control widgets."""
        return {
            'viz_combo': self.refinement_viz_combo,
            'mode_btn': self.refinement_mode_btn,
            'undo_btn': self.undo_btn,
            'split_method_combo': self.split_method_combo,
            'n_subclusters_spinbox': self.n_subclusters_spinbox,
            'split_btn': self.split_btn,
            'merge_btn': self.merge_btn,
            'reset_selection_btn': self.reset_selection_btn,
            'apply_refinement_btn': self.apply_refinement_btn,
            'cancel_refinement_btn': self.cancel_refinement_btn,
            'selection_status': self.selection_status,
            'figure': self.refinement_fig,
            'axis': self.refinement_ax,
            'canvas': self.refinement_canvas,
            'toolbar': self.refinement_toolbar,
            'selected_points': self.selected_points
        }
    
    def update_refinement_controls(self):
        """Update the state of refinement controls based on selection."""
        has_selection = len(self.selected_points) > 0
        self.split_btn.setEnabled(has_selection and len(self.selected_points) == 1)
        self.merge_btn.setEnabled(has_selection and len(self.selected_points) >= 2)
        self.reset_selection_btn.setEnabled(has_selection)
        
        # Update selection status
        if not has_selection:
            self.selection_status.setText("No clusters selected")
        elif len(self.selected_points) == 1:
            self.selection_status.setText("1 cluster selected (can split)")
        else:
            self.selection_status.setText(f"{len(self.selected_points)} clusters selected (can merge)")
    
    def set_refinement_mode(self, enabled):
        """Enable or disable refinement mode."""
        if enabled:
            self.refinement_mode_btn.setText("Stop Refinement Mode")
            self.refinement_mode_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f0ad4e;
                    color: white;
                    border: none;
                    padding: 6px 12px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #ec971f;
                }
            """)
        else:
            self.refinement_mode_btn.setText("Start Refinement Mode")
            self.refinement_mode_btn.setStyleSheet("")
            self.selected_points.clear()
            self.update_refinement_controls()
    
    def enable_refinement_actions(self, enabled):
        """Enable or disable refinement action buttons."""
        self.apply_refinement_btn.setEnabled(enabled)
        self.cancel_refinement_btn.setEnabled(enabled)
