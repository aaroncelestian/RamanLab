"""Results panel widgets for map peak fitting.

Phase 1: provides placeholder UI only.
"""

from PySide6.QtWidgets import QGroupBox, QLabel, QVBoxLayout
from PySide6.QtCore import Qt


class OverallStatsWidget(QGroupBox):
    """Shows overall fitting stats (placeholder for now)."""

    PLACEHOLDER_TEXT = "Run fitting to see results"

    def __init__(self, parent=None):
        super().__init__("Overall stats", parent)

        layout = QVBoxLayout(self)
        self._placeholder_label = QLabel(self.PLACEHOLDER_TEXT)
        self._placeholder_label.setWordWrap(True)
        self._placeholder_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._placeholder_label.setStyleSheet("color: #777;")
        layout.addWidget(self._placeholder_label)

    def clear(self) -> None:
        self._placeholder_label.setText(self.PLACEHOLDER_TEXT)


class PixelDetailsWidget(QGroupBox):
    """Shows details for selected pixel (placeholder for now)."""

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
    """Base results panel embedded in the left sidebar."""

    def __init__(self, parent=None):
        super().__init__("Results", parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.overall_stats_widget = OverallStatsWidget(self)
        self.pixel_details_widget = PixelDetailsWidget(self)
        layout.addWidget(self.overall_stats_widget)
        layout.addWidget(self.pixel_details_widget)

    def clear(self) -> None:
        """Reset both sub-sections to placeholder state."""
        self.overall_stats_widget.clear()
        self.pixel_details_widget.clear()
