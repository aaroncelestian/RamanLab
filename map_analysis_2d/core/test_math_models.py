import numpy as np
import pytest

from map_analysis_2d.core.math_models import create_multi_peak_model, get_param_names


def test_multi_peak_superposition_is_exact():
    f2 = create_multi_peak_model(["Lorentzian", "Lorentzian"])
    f1 = create_multi_peak_model(["Lorentzian"])
    x = np.linspace(400, 800, 200)
    p1 = (100.0, 500.0, 10.0)
    p2 = (80.0, 700.0, 8.0)

    assert np.allclose(f2(x, *p1, *p2), f1(x, *p1) + f1(x, *p2))


@pytest.mark.parametrize("shapes", [
    ["Lorentzian"],
    ["Gaussian"],
    ["Pseudo-Voigt"],
    ["Lorentzian", "Gaussian"],
])
def test_param_names_length_matches_model(shapes):
    names = get_param_names(shapes)
    model = create_multi_peak_model(shapes)
    dummy_params = [1.0] * len(names)
    x = np.array([500.0])

    model(x, *dummy_params)


def test_param_names_rejects_unknown_shape_with_index():
    with pytest.raises(ValueError, match="Unsupported peak shape at index 2: Not-A-Peak"):
        get_param_names(["Lorentzian", "Not-A-Peak"])


def test_lorentzian_at_center_equals_amplitude():
    f = create_multi_peak_model(["Lorentzian"])
    y = f(np.array([500.0]), 75.0, 500.0, 15.0)

    assert abs(y[0] - 75.0) < 1e-9


def test_gaussian_at_center_equals_amplitude():
    f = create_multi_peak_model(["Gaussian"])
    y = f(np.array([500.0]), 75.0, 500.0, 15.0)

    assert abs(y[0] - 75.0) < 1e-9
