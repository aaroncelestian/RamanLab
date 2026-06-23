#!/usr/bin/env python3
"""
RamanLab Crash Reporter
=======================
Catches unhandled exceptions, writes a timestamped crash log to
  ~/Documents/RamanLab_Qt6/crash_logs/
and shows a user-friendly dialog with the log path.

Usage (in main_qt6.py, before app.exec()):
    from core.crash_reporter import install_crash_reporter
    install_crash_reporter(app)
"""

import sys
import os
import traceback
import platform
from datetime import datetime
from pathlib import Path


def _get_log_dir() -> Path:
    """Return (and create if needed) the crash log directory."""
    try:
        from PySide6.QtCore import QStandardPaths
        docs = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
    except Exception:
        docs = str(Path.home() / "Documents")
    log_dir = Path(docs) / "RamanLab_Qt6" / "crash_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _build_report(exc_type, exc_value, exc_tb) -> str:
    """Assemble the full crash report text."""
    try:
        from version import __version_string__
    except Exception:
        __version_string__ = "unknown"

    tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
    tb_text = "".join(tb_lines)

    lines = [
        "=" * 70,
        "RamanLab Crash Report",
        "=" * 70,
        f"Timestamp : {datetime.now().isoformat()}",
        f"Version   : {__version_string__}",
        f"Platform  : {platform.platform()}",
        f"Python    : {sys.version}",
        "=" * 70,
        "Traceback:",
        tb_text,
        "=" * 70,
    ]
    return "\n".join(lines)


def _write_log(report: str) -> Path:
    """Write the report to a file and return its path."""
    log_dir = _get_log_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"crash_{stamp}.log"
    log_path.write_text(report, encoding="utf-8")
    return log_path


def _show_crash_dialog(exc_value, log_path: Path):
    """Display a Qt crash dialog.  Safe to call even if the app is half-broken."""
    try:
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QLabel,
            QPushButton, QTextEdit, QApplication
        )
        from PySide6.QtGui import QFont
        from PySide6.QtCore import Qt

        # Ensure a QApplication exists
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        dialog = QDialog()
        dialog.setWindowTitle("RamanLab — Unexpected Error")
        dialog.setMinimumWidth(580)
        layout = QVBoxLayout(dialog)

        title = QLabel("⚠️  RamanLab encountered an unexpected error")
        title.setFont(QFont("Arial", 13, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        summary = QLabel(
            f"<b>Error:</b> {type(exc_value).__name__}: {exc_value}<br><br>"
            f"A crash log has been saved to:<br>"
            f"<code>{log_path}</code><br><br>"
            "Please include this file when reporting the bug."
        )
        summary.setWordWrap(True)
        summary.setTextFormat(Qt.RichText)
        layout.addWidget(summary)

        detail_box = QTextEdit()
        detail_box.setReadOnly(True)
        detail_box.setFont(QFont("Courier", 9))
        detail_box.setPlainText(log_path.read_text(encoding="utf-8"))
        detail_box.setMaximumHeight(200)
        layout.addWidget(detail_box)

        btn_row = QHBoxLayout()

        open_btn = QPushButton("📁 Open Log Folder")
        open_btn.clicked.connect(lambda: _open_folder(log_path.parent))
        btn_row.addWidget(open_btn)

        close_btn = QPushButton("Close")
        close_btn.setDefault(True)
        close_btn.clicked.connect(dialog.accept)
        btn_row.addWidget(close_btn)

        layout.addLayout(btn_row)
        dialog.exec()

    except Exception:
        # Absolute last resort — plain stderr
        print(f"\nCRASH: {exc_value}\nLog: {log_path}\n", file=sys.stderr)


def _open_folder(path: Path):
    """Open a folder in the OS file manager."""
    try:
        if platform.system() == "Darwin":
            import subprocess
            subprocess.Popen(["open", str(path)])
        elif platform.system() == "Windows":
            os.startfile(str(path))
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


def _excepthook(exc_type, exc_value, exc_tb):
    """Replacement for sys.excepthook."""
    # Let KeyboardInterrupt pass through normally
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return

    # Print to stderr as usual so the terminal still shows the error
    traceback.print_exception(exc_type, exc_value, exc_tb)

    try:
        report = _build_report(exc_type, exc_value, exc_tb)
        log_path = _write_log(report)
        _show_crash_dialog(exc_value, log_path)
    except Exception as reporter_error:
        print(f"[CrashReporter] Failed to write report: {reporter_error}", file=sys.stderr)


def install_crash_reporter(_app=None):
    """
    Install the global exception hook.  Call once at startup, after QApplication
    is created.  The _app argument is accepted for convenience but not required.
    """
    sys.excepthook = _excepthook
