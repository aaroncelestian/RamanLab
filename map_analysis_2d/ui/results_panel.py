"""Results panel embedded in the peak fitting control panel sidebar."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGroupBox, QLabel, QVBoxLayout, QWidget

from .overall_stats_widget import OverallStatsWidget


class PixelDetailsWidget(QWidget):
    """Per-pixel detail display — populated in Phase 3."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._placeholder = QLabel("Click a map pixel to see details.")
        self._placeholder.setEnabled(False)
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._placeholder)

    def clear(self):
        pass


class ResultsPanel(QGroupBox):
    """Permanent sidebar panel showing overall stats and per-pixel details."""

    def __init__(self, parent=None):
        super().__init__("Results Summary", parent)
        layout = QVBoxLayout(self)

        self.overall_stats = OverallStatsWidget()
        self.pixel_details = PixelDetailsWidget()

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)

        layout.addWidget(self.overall_stats)
        layout.addWidget(separator)
        layout.addWidget(self.pixel_details)

    def clear(self):
        self.overall_stats.clear_stats()
        self.pixel_details.clear()
