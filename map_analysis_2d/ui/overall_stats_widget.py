"""Widget for displaying overall peak fitting statistics."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..core.statistics import OverallStatistics, compute_overall_statistics


class OverallStatsWidget(QWidget):
    """Display per-peak and grand total integrated areas after fitting."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._stats: Optional[OverallStatistics] = None
        self._placeholder_text = "Run map peak fitting to see overall statistics."

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(4)

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setSpacing(2)
        self._main_layout.addLayout(self._rows_layout)

        self._grand_total_row = QWidget()
        self._grand_total_layout = QHBoxLayout(self._grand_total_row)
        self._grand_total_layout.setContentsMargins(0, 0, 0, 0)
        self._grand_total_layout.setSpacing(6)

        self._grand_total_label = QLabel()
        self._copy_button = QPushButton("Copy")
        self._copy_button.setEnabled(False)
        self._copy_button.clicked.connect(self._copy_grand_total)
        self._grand_total_layout.addWidget(self._grand_total_label)
        self._grand_total_layout.addStretch(1)
        self._grand_total_layout.addWidget(self._copy_button)

        self._main_layout.addWidget(self._grand_total_row)
        self.clear_stats()

    def _clear_rows(self) -> None:
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def clear_stats(self):
        self._clear_rows()
        self._stats = None
        self._grand_total_label.setText(self._placeholder_text)
        self._copy_button.setEnabled(False)

    @Slot(dict)
    def update_from_fitting_results(self, fitting_results: dict):
        stats = compute_overall_statistics(fitting_results)
        if stats is None:
            self.clear_stats()
            return

        self._stats = stats
        self._clear_rows()

        for index, total in enumerate(stats.per_peak_total_areas, start=1):
            label = QLabel(f"Peak {index} total area: {self._format_number(total)}")
            self._rows_layout.addWidget(label)

        self._grand_total_label.setText(f"Grand total area: {self._format_number(stats.grand_total_area)}")
        self._copy_button.setEnabled(True)

    def _copy_grand_total(self):
        if self._stats is None:
            return
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(f"{self._stats.grand_total_area:.2f}")

    @staticmethod
    def _format_number(value: float) -> str:
        return f"{value:,.2f}"
