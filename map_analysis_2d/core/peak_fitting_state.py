"""State helpers for map peak fitting workflows."""

import math


def invalidate_peak_fitting_results(owner, control_panel=None):
    """Clear stale fit results and disable export until a new run completes."""
    owner.peak_fitting_results = None

    if control_panel is not None and hasattr(control_panel, "results_panel"):
        control_panel.results_panel.clear()

    if control_panel is not None and hasattr(control_panel, "export_batch_btn"):
        control_panel.export_batch_btn.setEnabled(False)


def merge_peak_fitting_configuration(existing_config, panel_config):
    """Merge fresh panel parameters while preserving the selected map visualization."""
    merged_config = dict(panel_config or {})

    if existing_config:
        for key in ("visualize_key", "visualize_param"):
            if existing_config.get(key) is not None:
                merged_config[key] = existing_config[key]

    return merged_config


def validate_peak_fitting_window(wavenumbers, region, parameter_count):
    """Return an error when a fit window cannot support the configured model."""
    finite_wavenumbers = []
    for value in wavenumbers:
        if value is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_value):
            finite_wavenumbers.append(numeric_value)

    if not finite_wavenumbers:
        return "No valid spectrum wavenumber axis is available for peak fitting."

    min_wn, max_wn = region
    spectrum_min = min(finite_wavenumbers)
    spectrum_max = max(finite_wavenumbers)

    if min_wn < spectrum_min or max_wn > spectrum_max:
        return (
            f"Fitting window ({min_wn:.1f}-{max_wn:.1f} cm⁻¹) is outside the spectrum "
            f"range ({spectrum_min:.1f}-{spectrum_max:.1f} cm⁻¹)."
        )

    point_count = sum(min_wn <= value <= max_wn for value in finite_wavenumbers)
    if point_count < parameter_count:
        return (
            f"The selected fitting window contains fewer points ({point_count}) than "
            f"fit parameters ({parameter_count}). Choose a wider window or fewer peaks."
        )

    return None
