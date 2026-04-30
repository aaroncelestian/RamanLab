import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(module_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyButton:
    def __init__(self):
        self.enabled = True

    def setEnabled(self, enabled):
        self.enabled = enabled


class DummyControlPanel:
    def __init__(self):
        self.export_batch_btn = DummyButton()


class DummyWindow:
    def __init__(self):
        self.peak_fitting_results = {"old": "results"}


class PeakFittingReviewFixTests(unittest.TestCase):
    def test_invalidating_results_clears_stale_results_and_disables_export(self):
        state_module = load_module(
            ROOT / "map_analysis_2d" / "core" / "peak_fitting_state.py",
            "peak_fitting_state_under_test",
        )
        window = DummyWindow()
        control_panel = DummyControlPanel()

        state_module.invalidate_peak_fitting_results(window, control_panel)

        self.assertIsNone(window.peak_fitting_results)
        self.assertFalse(control_panel.export_batch_btn.enabled)

    def test_math_models_does_not_use_invalid_parent_relative_import(self):
        source = (ROOT / "map_analysis_2d" / "core" / "math_models.py").read_text()

        self.assertNotIn("from ...core.peak_fitting import PeakFitter", source)


if __name__ == "__main__":
    unittest.main()
