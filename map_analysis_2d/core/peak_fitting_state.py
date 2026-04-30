"""State helpers for map peak fitting workflows."""


def invalidate_peak_fitting_results(owner, control_panel=None):
    """Clear stale fit results and disable export until a new run completes."""
    owner.peak_fitting_results = None

    if control_panel is not None and hasattr(control_panel, "export_batch_btn"):
        control_panel.export_batch_btn.setEnabled(False)
