import unittest

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

from plot_view_state import capture_axis_view, restore_axis_view


class PlotViewStateTests(unittest.TestCase):
    def test_capture_axis_view_ignores_empty_axes(self):
        axis = Figure().add_subplot(111)

        self.assertIsNone(capture_axis_view(axis))

    def test_restore_axis_view_preserves_limits_after_redraw(self):
        axis = Figure().add_subplot(111)
        axis.plot([100, 200, 300], [1, 3, 2])
        axis.set_xlim(125, 225)
        axis.set_ylim(1.5, 3.5)

        state = capture_axis_view(axis)
        axis.clear()
        axis.plot([100, 200, 300], [10, 30, 20])

        restore_axis_view(axis, state)

        self.assertEqual(axis.get_xlim(), (125, 225))
        self.assertEqual(axis.get_ylim(), (1.5, 3.5))


if __name__ == "__main__":
    unittest.main()
