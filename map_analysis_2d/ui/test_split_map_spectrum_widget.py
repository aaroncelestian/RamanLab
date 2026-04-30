import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from plotting_widgets import SplitMapSpectrumWidget


class SplitMapSpectrumWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_show_spectrum_panel_preserves_user_resized_splitter(self):
        widget = SplitMapSpectrumWidget()
        widget.resize(800, 600)

        widget.show_spectrum_panel(True)
        widget.splitter.setSizes([300, 300])
        user_sizes = widget.splitter.sizes()

        widget.show_spectrum_panel(True)

        self.assertEqual(widget.splitter.sizes(), user_sizes)


if __name__ == "__main__":
    unittest.main()
