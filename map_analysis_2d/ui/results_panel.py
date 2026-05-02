"""Results panel embedded in the peak fitting control panel sidebar."""

from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QVBoxLayout

from .overall_stats_widget import OverallStatsWidget
from .pixel_details_widget import PixelDetailsWidget


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
