import importlib.util
import sys
import types
import unittest
from pathlib import Path


def load_worker_module():
    qtcore = types.ModuleType("PySide6.QtCore")

    class DummyQThread:
        pass

    class DummySignal:
        def __init__(self, *args, **kwargs):
            pass

    qtcore.QThread = DummyQThread
    qtcore.Signal = DummySignal

    pyside = types.ModuleType("PySide6")
    pyside.QtCore = qtcore

    scipy = types.ModuleType("scipy")
    scipy_optimize = types.ModuleType("scipy.optimize")

    class DummyOptimizeWarning(Warning):
        pass

    scipy_optimize.OptimizeWarning = DummyOptimizeWarning
    scipy.optimize = scipy_optimize

    original_modules = {
        name: sys.modules.get(name)
        for name in ("PySide6", "PySide6.QtCore", "scipy", "scipy.optimize")
    }
    sys.modules["PySide6"] = pyside
    sys.modules["PySide6.QtCore"] = qtcore
    sys.modules["scipy"] = scipy
    sys.modules["scipy.optimize"] = scipy_optimize

    try:
        module_path = Path(__file__).with_name("peak_fitting_worker.py")
        spec = importlib.util.spec_from_file_location("peak_fitting_worker_under_test", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class FitWarningMergeTests(unittest.TestCase):
    def test_append_fit_warning_preserves_existing_warning(self):
        worker_module = load_worker_module()
        warnings_by_position = {(1, 2): "optimizer reached maxfev"}

        worker_module.append_fit_warning(
            warnings_by_position,
            (1, 2),
            "parameter uncertainty could not be estimated",
        )

        self.assertEqual(
            warnings_by_position[(1, 2)],
            "optimizer reached maxfev; parameter uncertainty could not be estimated",
        )

    def test_append_fit_warning_stores_first_warning(self):
        worker_module = load_worker_module()
        warnings_by_position = {}

        worker_module.append_fit_warning(warnings_by_position, (3, 4), "first warning")

        self.assertEqual(warnings_by_position[(3, 4)], "first warning")


if __name__ == "__main__":
    unittest.main()
