"""Worker thread for map peak fitting to keep the UI responsive."""

import logging
import warnings

from PySide6.QtCore import QThread, Signal
from scipy.optimize import OptimizeWarning

logger = logging.getLogger(__name__)


def combine_fit_warning(existing_warning, new_warning: str) -> str:
    """Append a fit warning without discarding earlier diagnostics."""
    if existing_warning:
        return f"{existing_warning}; {new_warning}"
    return new_warning


def append_fit_warning(fit_warnings: dict, pos_key, warning_message: str):
    fit_warnings[pos_key] = combine_fit_warning(fit_warnings.get(pos_key), warning_message)


class PeakFittingWorker(QThread):
    """Run map peak fitting in a background worker thread."""

    progress_updated = Signal(int, int, str)
    fitting_complete = Signal(dict)
    fitting_failed = Signal(str)

    def __init__(self, spectra, use_processed: bool, config: dict):
        super().__init__()
        self.spectra = spectra
        self.use_processed = use_processed
        self.config = dict(config)
        self._is_running = True

    def stop(self):
        """Request that the worker stop after the current spectrum."""
        self._is_running = False

    def run(self):
        """Execute peak fitting across the supplied spectra."""
        try:
            import numpy as np
            from scipy.optimize import curve_fit

            from map_analysis_2d.core.math_models import (
                DEFAULT_CURVE_FIT_MAXFEV,
                create_multi_peak_model,
                get_param_names,
            )

            model_func = create_multi_peak_model(self.config['shapes'])
            param_names = get_param_names(self.config['shapes'])
            bounds = self.config['bounds']
            min_wn, max_wn = self.config['region']

            total = len(self.spectra)
            results = {
                'n_peaks': len(self.config['shapes']),
                'peak_shapes': list(self.config['shapes']),
                'param_names': param_names,
                'map_parameters': {name: {} for name in param_names},
                'r_squared': {},
                'fit_errors': {},
                'fit_warnings': {},
                'config': dict(self.config),
            }

            def store_failed_fit(pos_key, message):
                results['fit_errors'][pos_key] = message
                for name in param_names:
                    results['map_parameters'][name][pos_key] = np.nan
                results['r_squared'][pos_key] = np.nan

            for index, spectrum in enumerate(self.spectra, start=1):
                if not self._is_running:
                    return

                wavenumbers = spectrum.wavenumbers
                intensities = (
                    spectrum.processed_intensities
                    if self.use_processed and spectrum.processed_intensities is not None
                    else spectrum.intensities
                )

                mask = (wavenumbers >= min_wn) & (wavenumbers <= max_wn)
                x_fit = wavenumbers[mask]
                y_fit = intensities[mask]
                pos_key = (spectrum.x_pos, spectrum.y_pos)

                if not np.all(np.isfinite(x_fit)) or not np.all(np.isfinite(y_fit)):
                    message = "Selected fitting region contains NaN or infinite values."
                    logger.debug("Skipping position %s: %s", pos_key, message)
                    store_failed_fit(pos_key, message)
                elif len(x_fit) >= len(self.config['initial_params']):
                    try:
                        with warnings.catch_warnings(record=True) as caught_warnings:
                            warnings.simplefilter("always", OptimizeWarning)
                            warnings.simplefilter("always", RuntimeWarning)
                            popt, pcov = curve_fit(
                                model_func,
                                x_fit,
                                y_fit,
                                p0=self.config['initial_params'],
                                bounds=bounds,
                                maxfev=DEFAULT_CURVE_FIT_MAXFEV,
                            )

                        warning_messages = [
                            str(warning.message)
                            for warning in caught_warnings
                            if issubclass(warning.category, (OptimizeWarning, RuntimeWarning))
                        ]
                        if warning_messages:
                            append_fit_warning(results['fit_warnings'], pos_key, "; ".join(warning_messages))

                        covariance_diag = np.diag(pcov)
                        if not np.all(np.isfinite(covariance_diag)):
                            append_fit_warning(
                                results['fit_warnings'],
                                pos_key,
                                "Fit converged, but parameter uncertainty could not be estimated reliably.",
                            )

                        ss_res = np.sum((y_fit - model_func(x_fit, *popt)) ** 2)
                        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                        for param_index, name in enumerate(param_names):
                            results['map_parameters'][name][pos_key] = popt[param_index]
                        results['r_squared'][pos_key] = r_squared
                    except Exception as exc:
                        logger.debug("Fit failed for position %s: %s", pos_key, exc)
                        store_failed_fit(pos_key, str(exc))
                else:
                    store_failed_fit(
                        pos_key,
                        "Selected fitting region contains fewer points than the number of fit parameters.",
                    )

                if self._is_running:
                    self.progress_updated.emit(index, total, f"Fitting spectrum {index:,} of {total:,}...")

            if self._is_running:
                self.fitting_complete.emit(results)

        except Exception as exc:
            import traceback

            self.fitting_failed.emit(f"{exc}\n\n{traceback.format_exc()}")
