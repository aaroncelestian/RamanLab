"""UI widget for displaying per-pixel peak fitting details."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class PixelDetailsWidget(QWidget):
    """Display location + per-peak parameters for a selected pixel."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Selected Pixel")
        group_layout = QVBoxLayout(group)

        header_layout = QHBoxLayout()
        self.position_label = QLabel("Click the peak fitting map")
        self.position_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header_layout.addWidget(self.position_label, 1)

        self.r_squared_label = QLabel("")
        self.r_squared_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header_layout.addWidget(self.r_squared_label)
        group_layout.addLayout(header_layout)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        group_layout.addWidget(self.status_label)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Peak", "Area", "Center", "Width"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.horizontalHeader().setStretchLastSection(True)
        group_layout.addWidget(self.table)

        layout.addWidget(group)

        self.clear()

    def clear(self):
        """Reset to placeholder state."""
        self.position_label.setText("Click the peak fitting map")
        self.r_squared_label.setText("")
        self.status_label.setText("")
        self.table.setRowCount(0)

    def show_fit_failed(self, position_text: str):
        self.position_label.setText(position_text)
        self.r_squared_label.setText("")
        self.status_label.setText("Fit failed for this pixel")
        self.table.setRowCount(0)

    def show_results(
        self,
        *,
        position_text: str,
        r_squared: Optional[float],
        peak_rows: Sequence[Tuple[str, float, float, float]],
    ):
        self.position_label.setText(position_text)
        self.status_label.setText("")
        if r_squared is None:
            self.r_squared_label.setText("")
        else:
            self.r_squared_label.setText(f"R²: {r_squared:.3f}")

        self.table.setRowCount(len(peak_rows))
        for row_idx, (peak_name, area, center, width) in enumerate(peak_rows):
            values = [peak_name, f"{area:.2f}", f"{center:.2f}", f"{width:.2f}"]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row_idx, col_idx, item)

