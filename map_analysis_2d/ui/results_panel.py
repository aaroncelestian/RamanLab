"""Results panel embedded in the peak fitting control panel sidebar."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGroupBox, QLabel, QVBoxLayout

from .overall_stats_widget import OverallStatsWidget


class PixelDetailsWidget(QGroupBox):
    """Per-pixel detail display — populated in Phase 3."""

    PLACEHOLDER_TEXT = "Click a map pixel to see details"

    def __init__(self, parent=None):
        super().__init__("Pixel details", parent)
        layout = QVBoxLayout(self)
        self._placeholder_label = QLabel(self.PLACEHOLDER_TEXT)
        self._placeholder_label.setWordWrap(True)
        self._placeholder_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._placeholder_label.setStyleSheet("color: #777;")
        layout.addWidget(self._placeholder_label)

    def clear(self) -> None:
        self._placeholder_label.setText(self.PLACEHOLDER_TEXT)


class ResultsPanel(QGroupBox):
    """Permanent sidebar panel showing overall stats and per-pixel details."""

    def __init__(self, parent=None):
        super().__init__("Results", parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.overall_stats = OverallStatsWidget(self)
        self.pixel_details = PixelDetailsWidget(self)

        layout.addWidget(self.overall_stats)
        layout.addWidget(self.pixel_details)

    def clear(self) -> None:
        self.overall_stats.clear_stats()
        self.pixel_details.clear()
