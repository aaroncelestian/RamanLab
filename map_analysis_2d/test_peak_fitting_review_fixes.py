import importlib
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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

    def test_merging_config_preserves_existing_visualization_selection(self):
        state_module = load_module(
            ROOT / "map_analysis_2d" / "core" / "peak_fitting_state.py",
            "peak_fitting_state_under_test",
        )
        existing_config = {
            "region": (400.0, 600.0),
            "shapes": ["Lorentzian"],
            "initial_params": [10.0, 500.0, 5.0],
            "bounds": ([0.0, 400.0, 1.0], [100.0, 600.0, 20.0]),
            "visualize_key": "P2_Wid",
            "visualize_param": "Peak 2 Width",
        }
        panel_config = {
            "region": (450.0, 700.0),
            "shapes": ["Gaussian"],
            "initial_params": [20.0, 550.0, 8.0],
            "bounds": ([0.0, 450.0, 1.0], [200.0, 700.0, 30.0]),
            "visualize_key": "R-Squared",
            "visualize_param": "R-Squared (Fit Quality)",
        }

        merged = state_module.merge_peak_fitting_configuration(existing_config, panel_config)

        self.assertEqual(merged["region"], panel_config["region"])
        self.assertEqual(merged["shapes"], panel_config["shapes"])
        self.assertEqual(merged["visualize_key"], "P2_Wid")
        self.assertEqual(merged["visualize_param"], "Peak 2 Width")

    def test_validate_peak_fitting_window_rejects_out_of_range_region(self):
        state_module = load_module(
            ROOT / "map_analysis_2d" / "core" / "peak_fitting_state.py",
            "peak_fitting_state_under_test",
        )

        error = state_module.validate_peak_fitting_window(
            [400.0, 500.0, 600.0],
            (300.0, 500.0),
            parameter_count=3,
        )

        self.assertIn("outside", error)

    def test_validate_peak_fitting_window_rejects_too_few_points(self):
        state_module = load_module(
            ROOT / "map_analysis_2d" / "core" / "peak_fitting_state.py",
            "peak_fitting_state_under_test",
        )

        error = state_module.validate_peak_fitting_window(
            [400.0, 500.0, 600.0],
            (450.0, 550.0),
            parameter_count=3,
        )

        self.assertIn("fewer points", error)

    def test_validate_peak_fitting_window_accepts_sufficient_in_range_points(self):
        state_module = load_module(
            ROOT / "map_analysis_2d" / "core" / "peak_fitting_state.py",
            "peak_fitting_state_under_test",
        )

        error = state_module.validate_peak_fitting_window(
            [400.0, 500.0, 600.0],
            (400.0, 600.0),
            parameter_count=3,
        )

        self.assertIsNone(error)

    def test_validate_peak_fitting_window_ignores_non_numeric_wavenumbers(self):
        state_module = load_module(
            ROOT / "map_analysis_2d" / "core" / "peak_fitting_state.py",
            "peak_fitting_state_under_test",
        )

        error = state_module.validate_peak_fitting_window(
            [400.0, "bad", None, 500.0, 600.0],
            (400.0, 600.0),
            parameter_count=3,
        )

        self.assertIsNone(error)

    def test_math_models_import_exposes_peak_fitter(self):
        try:
            module = importlib.import_module("map_analysis_2d.core.math_models")
        except ImportError as exc:
            self.fail(f"Could not import math_models: {exc}")

        self.assertTrue(hasattr(module, "PeakFitter"))

    def test_get_param_names_rejects_unknown_shape(self):
        module = importlib.import_module("map_analysis_2d.core.math_models")

        with self.assertRaisesRegex(ValueError, "Unsupported peak shape at index 2: Not-A-Peak"):
            module.get_param_names(["Lorentzian", "Not-A-Peak"])


if __name__ == "__main__":
    unittest.main()
