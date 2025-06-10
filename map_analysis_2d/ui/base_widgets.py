"""
Base UI widgets and utilities for the map analysis application.

This module contains common UI components and utilities used throughout
the user interface.
"""

import logging
from typing import Optional, Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QLabel, QCheckBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QLineEdit, QScrollArea, QProgressBar
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

logger = logging.getLogger(__name__)


class SafeWidgetMixin:
    """Mixin class that provides safe widget access methods."""
    
    def safe_checkbox_access(self, checkbox_attr: str, default: bool = False) -> bool:
        """Safely access a checkbox widget, handling cases where it may be deleted."""
        try:
            if hasattr(self, checkbox_attr) and getattr(self, checkbox_attr) is not None:
                checkbox = getattr(self, checkbox_attr)
                return checkbox.isChecked()
            else:
                return default
        except RuntimeError:
            # Widget has been deleted
            return default

    def safe_widget_access(self, widget_attr: str, method_name: str, default: Any = None) -> Any:
        """Safely access any widget method, handling cases where the widget may be deleted."""
        try:
            if hasattr(self, widget_attr) and getattr(self, widget_attr) is not None:
                widget = getattr(self, widget_attr)
                if hasattr(widget, method_name):
                    method = getattr(widget, method_name)
                    if callable(method):
                        return method()
                    else:
                        return method
                else:
                    return default
            else:
                return default
        except RuntimeError:
            # Widget has been deleted
            return default


class ParameterGroupBox(QGroupBox, SafeWidgetMixin):
    """
    A group box widget for organizing analysis parameters.
    
    Provides convenient methods for adding different types of parameter controls.
    """
    
    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.layout = QGridLayout(self)
        self.layout.setSpacing(12)  # More generous spacing for better usability
        self.layout.setContentsMargins(15, 15, 15, 15)  # More generous margins
        self.layout.setColumnStretch(0, 0)  # Don't stretch label column
        self.layout.setColumnStretch(1, 1)  # Stretch control column
        self.row_count = 0
        
    def add_double_spinbox(self, label: str, min_val: float, max_val: float, 
                          value: float, step: float = 1.0, width: int = 120) -> QDoubleSpinBox:
        """Add a double spinbox parameter."""
        label_widget = QLabel(f"{label}:")
        label_widget.setMaximumWidth(90)  # Constrain label width
        label_widget.setWordWrap(True)  # Allow text wrapping
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(value)
        spinbox.setSingleStep(step)
        spinbox.setMaximumWidth(width)  # Increased default width
        spinbox.setMinimumWidth(80)  # Increased minimum width for better usability
        spinbox.setMinimumHeight(30)  # Add minimum height for better touch targets
        
        self.layout.addWidget(label_widget, self.row_count, 0, Qt.AlignmentFlag.AlignTop)
        self.layout.addWidget(spinbox, self.row_count, 1, Qt.AlignmentFlag.AlignLeft)
        self.row_count += 1
        
        return spinbox
    
    def add_spinbox(self, label: str, min_val: int, max_val: int, 
                   value: int, width: int = 120) -> QSpinBox:
        """Add an integer spinbox parameter."""
        label_widget = QLabel(f"{label}:")
        label_widget.setMaximumWidth(90)  # Constrain label width
        label_widget.setWordWrap(True)  # Allow text wrapping
        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(value)
        spinbox.setMaximumWidth(width)  # Increased default width
        spinbox.setMinimumWidth(80)  # Increased minimum width for better usability
        spinbox.setMinimumHeight(30)  # Add minimum height for better touch targets
        
        self.layout.addWidget(label_widget, self.row_count, 0, Qt.AlignmentFlag.AlignTop)
        self.layout.addWidget(spinbox, self.row_count, 1, Qt.AlignmentFlag.AlignLeft)
        self.row_count += 1
        
        return spinbox
    
    def add_checkbox(self, label: str, checked: bool = False) -> QCheckBox:
        """Add a checkbox parameter."""
        checkbox = QCheckBox(label)
        checkbox.setChecked(checked)
        # Note: QCheckBox doesn't have setWordWrap, text wrapping is handled automatically
        
        self.layout.addWidget(checkbox, self.row_count, 0, 1, 2)
        self.row_count += 1
        
        return checkbox
    
    def add_combobox(self, label: str, items: list, current_index: int = 0, width: int = 120) -> QComboBox:
        """Add a combobox parameter."""
        label_widget = QLabel(f"{label}:")
        label_widget.setMaximumWidth(90)  # Constrain label width 
        label_widget.setWordWrap(True)  # Allow text wrapping
        combobox = QComboBox()
        combobox.addItems(items)
        combobox.setCurrentIndex(current_index)
        combobox.setMaximumWidth(width)  # Control combobox width
        combobox.setMinimumWidth(80)  # Minimum width for readability
        
        self.layout.addWidget(label_widget, self.row_count, 0, Qt.AlignmentFlag.AlignTop)
        self.layout.addWidget(combobox, self.row_count, 1, Qt.AlignmentFlag.AlignLeft)
        self.row_count += 1
        
        return combobox


class ButtonGroup(QWidget):
    """Widget for organizing related buttons in rows with consistent flat rounded styling."""
    
    BUTTON_STYLE = """
        QPushButton {
            background-color: #64748b;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 6px;
            font-weight: 500;
            font-size: 10px;
            min-height: 18px;
        }
        QPushButton:hover {
            background-color: #475569;
        }
        QPushButton:pressed {
            background-color: #334155;
        }
        QPushButton:disabled {
            background-color: #94a3b8;
            color: #e2e8f0;
        }
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(4)
        
    def add_button_row(self, button_configs: list) -> list:
        """
        Add a row of buttons with consistent flat rounded styling.
        
        Args:
            button_configs: List of (text, callback) tuples
            
        Returns:
            List of created QPushButton widgets
        """
        row_layout = QHBoxLayout()
        buttons = []
        
        for text, callback in button_configs:
            button = QPushButton(text)
            button.setStyleSheet(self.BUTTON_STYLE)
            if callback:
                button.clicked.connect(callback)
            buttons.append(button)
            row_layout.addWidget(button)
        
        self.main_layout.addLayout(row_layout)
        return buttons


class ScrollableControlPanel(QScrollArea, SafeWidgetMixin):
    """
    Scrollable control panel that can dynamically update its content.
    
    This is used for the left control panel that changes based on the selected tab.
    """
    
    def __init__(self, parent=None, max_width: int = 500):
        super().__init__(parent)
        
        # Configure scroll area
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Allow resizing by setting maximum and minimum widths more flexibly
        self.setMaximumWidth(max_width)
        self.setMinimumWidth(200)  # Reasonable minimum width
        # Set preferred width but allow resizing
        self.resize(280, self.height())
        
        # Create main widget and layout
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setSpacing(6)
        self.main_layout.setContentsMargins(4, 4, 4, 4)
        
        # Set as scroll area content
        self.setWidget(self.main_widget)
        
        # Initialize sections
        self.sections = {}
        
    def add_section(self, name: str, widget: QWidget, permanent: bool = False):
        """
        Add a section to the control panel.
        
        Args:
            name: Section identifier
            widget: Widget to add
            permanent: If True, section won't be removed by clear_dynamic_sections
        """
        self.sections[name] = {'widget': widget, 'permanent': permanent}
        self.main_layout.addWidget(widget)
        
    def clear_dynamic_sections(self):
        """Remove all non-permanent sections."""
        for name, section in list(self.sections.items()):
            if not section['permanent']:
                widget = section['widget']
                self.main_layout.removeWidget(widget)
                widget.deleteLater()
                del self.sections[name]
                
    def add_stretch(self):
        """Add stretch to push content to top."""
        self.main_layout.addStretch()


class TitleLabel(QLabel):
    """Styled title label for sections."""
    
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)


class StandardButton(QPushButton):
    """Button with standard flat rounded styling for the application."""
    
    STANDARD_STYLE = """
        QPushButton {
            background-color: #0D9488;
            color: white;
            border: none;
            padding: 10px 16px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 11px;
            min-height: 20px;
        }
        QPushButton:hover {
            background-color: #0F766E;
        }
        QPushButton:pressed {
            background-color: #115E59;
        }
        QPushButton:disabled {
            background-color: #94a3b8;
            color: #e2e8f0;
        }
    """
    
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(self.STANDARD_STYLE)


class PrimaryButton(QPushButton):
    """Primary action button with flat rounded styling."""
    
    PRIMARY_STYLE = """
        QPushButton {
            background-color: #3b82f6;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 12px;
            min-height: 24px;
        }
        QPushButton:hover {
            background-color: #2563eb;
        }
        QPushButton:pressed {
            background-color: #1d4ed8;
        }
        QPushButton:disabled {
            background-color: #94a3b8;
            color: #e2e8f0;
        }
    """
    
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(self.PRIMARY_STYLE)


class SecondaryButton(QPushButton):
    """Secondary action button with flat rounded styling."""
    
    SECONDARY_STYLE = """
        QPushButton {
            background-color: #6b7280;
            color: white;
            border: none;
            padding: 10px 16px;
            border-radius: 6px;
            font-weight: 500;
            font-size: 11px;
            min-height: 20px;
        }
        QPushButton:hover {
            background-color: #4b5563;
        }
        QPushButton:pressed {
            background-color: #374151;
        }
        QPushButton:disabled {
            background-color: #94a3b8;
            color: #e2e8f0;
        }
    """
    
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(self.SECONDARY_STYLE)


class DangerButton(QPushButton):
    """Danger/destructive action button with flat rounded styling."""
    
    DANGER_STYLE = """
        QPushButton {
            background-color: #ef4444;
            color: white;
            border: none;
            padding: 10px 16px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 11px;
            min-height: 20px;
        }
        QPushButton:hover {
            background-color: #dc2626;
        }
        QPushButton:pressed {
            background-color: #b91c1c;
        }
        QPushButton:disabled {
            background-color: #94a3b8;
            color: #e2e8f0;
        }
    """
    
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(self.DANGER_STYLE)


class SuccessButton(QPushButton):
    """Success/positive action button with flat rounded styling."""
    
    SUCCESS_STYLE = """
        QPushButton {
            background-color: #10b981;
            color: white;
            border: none;
            padding: 10px 16px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 11px;
            min-height: 20px;
        }
        QPushButton:hover {
            background-color: #059669;
        }
        QPushButton:pressed {
            background-color: #047857;
        }
        QPushButton:disabled {
            background-color: #94a3b8;
            color: #e2e8f0;
        }
    """
    
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(self.SUCCESS_STYLE)


class WarningButton(QPushButton):
    """Warning/action button with flat rounded styling."""
    
    WARNING_STYLE = """
        QPushButton {
            background-color: #f59e0b;
            color: white;
            border: none;
            padding: 10px 16px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 11px;
            min-height: 20px;
        }
        QPushButton:hover {
            background-color: #d97706;
        }
        QPushButton:pressed {
            background-color: #b45309;
        }
        QPushButton:disabled {
            background-color: #94a3b8;
            color: #e2e8f0;
        }
    """
    
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(self.WARNING_STYLE)


class InfoButton(QPushButton):
    """Info/information button with flat rounded styling."""
    
    INFO_STYLE = """
        QPushButton {
            background-color: #3b82f6;
            color: white;
            border: none;
            padding: 10px 16px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 11px;
            min-height: 20px;
        }
        QPushButton:hover {
            background-color: #2563eb;
        }
        QPushButton:pressed {
            background-color: #1d4ed8;
        }
        QPushButton:disabled {
            background-color: #94a3b8;
            color: #e2e8f0;
        }
    """
    
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(self.INFO_STYLE)


def apply_icon_button_style(button):
    """Apply consistent flat rounded styling to icon buttons."""
    button.setStyleSheet("""
        QPushButton {
            background-color: #64748b;
            color: white;
            border: none;
            padding: 6px;
            border-radius: 6px;
            font-weight: 500;
            font-size: 12px;
            min-height: 18px;
            min-width: 30px;
        }
        QPushButton:hover {
            background-color: #475569;
        }
        QPushButton:pressed {
            background-color: #334155;
        }
        QPushButton:disabled {
            background-color: #94a3b8;
            color: #e2e8f0;
        }
    """)


class ProgressStatusWidget(QWidget):
    """Widget that combines progress bar with status display."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        
        # Status label
        self.status_label = QLabel("Ready")
        
        self.layout.addWidget(self.status_label)
        self.layout.addStretch()
        self.layout.addWidget(self.progress_bar)
        
    def show_progress(self, message: str = None):
        """Show progress bar with optional message."""
        if message:
            self.status_label.setText(message)
        self.progress_bar.setVisible(True)
        
    def hide_progress(self):
        """Hide progress bar and reset status."""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ready")
        
    def update_progress(self, value: int, message: str = None):
        """Update progress value and optionally message."""
        self.progress_bar.setValue(value)
        if message:
            self.status_label.setText(message) 