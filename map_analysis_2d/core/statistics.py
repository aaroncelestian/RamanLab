"""Statistics helpers for map peak fitting results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .math_models import compute_integrated_intensity


PosKey = Tuple[float, float]


@dataclass(frozen=True)
class OverallStatistics:
    per_peak_total_areas: List[float]
    grand_total_area: float


def _iter_positions(*param_maps: Mapping[PosKey, float]) -> Iterable[PosKey]:
    positions = set()
    for mapping in param_maps:
        positions.update(mapping.keys())
    return positions


def compute_overall_statistics(fitting_results: dict) -> Optional[OverallStatistics]:
    """Compute per-peak and grand total integrated areas.

    The worker stores per-position peak parameters in ``fitting_results['map_parameters']``.
    For each position where all peak parameters are finite and there is no recorded
    fit error, the analytical integrated intensity for each peak is computed using
    :func:`~map_analysis_2d.core.math_models.compute_integrated_intensity`.
    Totals are summed across the map.
    """

    if not fitting_results:
        return None

    shapes: Sequence[str] = fitting_results.get("peak_shapes") or fitting_results.get("config", {}).get("shapes")
    if not shapes:
        return None

    map_params: Dict[str, Dict[PosKey, float]] = fitting_results.get("map_parameters") or {}
    fit_errors: Mapping[PosKey, str] = fitting_results.get("fit_errors") or {}

    peak_amp_dicts = [map_params.get(f"P{i}_Amp", {}) for i in range(1, len(shapes) + 1)]
    peak_wid_dicts = [map_params.get(f"P{i}_Wid", {}) for i in range(1, len(shapes) + 1)]
    peak_eta_dicts = [map_params.get(f"P{i}_Eta", {}) for i in range(1, len(shapes) + 1)]

    positions = _iter_positions(*peak_amp_dicts, *peak_wid_dicts)
    if not positions:
        return None

    per_peak_totals = np.zeros(len(shapes), dtype=float)
    grand_total = 0.0
    any_valid = False

    for pos_key in positions:
        if fit_errors.get(pos_key):
            continue

        per_pos_values: List[float] = []
        valid = True
        for shape, amp_dict, wid_dict, eta_dict in zip(shapes, peak_amp_dicts, peak_wid_dicts, peak_eta_dicts, strict=True):
            amp = amp_dict.get(pos_key, np.nan)
            wid = wid_dict.get(pos_key, np.nan)
            eta = eta_dict.get(pos_key, 0.5)
            if not (np.isfinite(amp) and np.isfinite(wid)):
                valid = False
                break
            per_pos_values.append(float(compute_integrated_intensity(amp, wid, shape, eta)))

        if not valid or not all(np.isfinite(value) for value in per_pos_values):
            continue

        any_valid = True
        for idx, value in enumerate(per_pos_values):
            per_peak_totals[idx] += value
        grand_total += float(sum(per_pos_values))

    if not any_valid:
        return None

    return OverallStatistics(
        per_peak_total_areas=[float(v) for v in per_peak_totals],
        grand_total_area=float(grand_total),
    )

