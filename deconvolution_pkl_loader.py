#!/usr/bin/env python3
"""
Load fitted peak data exported from RamanLab Spectral Deconvolution (batch PKL).

Supports the pandas export format (summary_df, peaks_df, spectra_dict, metadata)
and the legacy list-of-file-results format.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


INTERNAL_STANDARDS = {
    "silicon": {
        "label": "Silicon",
        "wmin": 515.0,
        "wmax": 525.0,
        "nominal": 520.7,
    },
    "corundum": {
        "label": "Corundum (417 cm⁻¹)",
        "wmin": 405.0,
        "wmax": 430.0,
        "nominal": 417.0,
    },
}


def load_deconvolution_pkl(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and normalize a deconvolution/batch-results PKL file."""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    normalized = normalize_deconvolution_data(data)
    normalized["source_path"] = str(Path(file_path).resolve())
    normalized["source_name"] = Path(file_path).name
    return normalized


def normalize_deconvolution_data(data: Any) -> Dict[str, Any]:
    """Convert supported PKL payloads to a common dict structure."""
    if isinstance(data, dict) and "peaks_df" in data:
        peaks_df = data.get("peaks_df", pd.DataFrame())
        summary_df = data.get("summary_df", pd.DataFrame())
        if not isinstance(peaks_df, pd.DataFrame):
            peaks_df = pd.DataFrame(peaks_df)
        if not isinstance(summary_df, pd.DataFrame):
            summary_df = pd.DataFrame(summary_df)
        metadata = dict(data.get("metadata") or {})
        return {
            "summary_df": summary_df,
            "peaks_df": peaks_df,
            "spectra_dict": data.get("spectra_dict") or {},
            "metadata": metadata,
            "format": "pandas",
        }

    if isinstance(data, list):
        peaks_df, summary_df = _legacy_list_to_dataframes(data)
        return {
            "summary_df": summary_df,
            "peaks_df": peaks_df,
            "spectra_dict": {},
            "metadata": {
                "export_method": "legacy_batch_list",
                "total_files": len(data),
                "total_regions": int(len(summary_df)),
                "total_peaks": int(len(peaks_df)),
            },
            "format": "legacy",
        }

    raise ValueError(
        f"Unsupported PKL format ({type(data).__name__}). "
        "Expected deconvolution batch export with peaks_df or legacy file list."
    )


def _legacy_list_to_dataframes(file_results: List[dict]) -> tuple:
    """Build peaks_df and summary_df from legacy batch_results list."""
    peak_rows: List[dict] = []
    summary_rows: List[dict] = []
    spectrum_index = 0

    for file_result in file_results:
        filename = file_result.get("filename", f"Spectrum_{spectrum_index}")
        filepath = file_result.get("filepath", "")

        for region_result in file_result.get("regions", []):
            peaks = region_result.get("peaks")
            fit_params = region_result.get("fit_params")
            total_r2 = region_result.get("total_r2")
            region_start = region_result.get("start")
            region_end = region_result.get("end")

            summary_rows.append({
                "file_index": len(summary_rows),
                "filename": filename,
                "filepath": filepath,
                "region_start": region_start,
                "region_end": region_end,
                "n_peaks": len(peaks) if peaks is not None else 0,
                "total_r2": total_r2,
                "scan_index": spectrum_index,
            })

            if peaks is not None and fit_params is not None:
                peak_params = region_result.get("peak_params", {})
                model = peak_params.get("model", "gaussian") if isinstance(peak_params, dict) else "gaussian"
                params_per_peak = 5 if str(model).lower() == "asymmetric voigt" else (
                    4 if str(model).lower() in ("pseudo-voigt", "voigt") else 3
                )

                for i in range(len(peaks)):
                    base = i * params_per_peak
                    if base + 2 >= len(fit_params):
                        continue
                    amplitude = fit_params[base]
                    center = fit_params[base + 1]
                    width = fit_params[base + 2]
                    fwhm = width * 2 * np.sqrt(2 * np.log(2))
                    area = amplitude * width * np.sqrt(2 * np.pi)

                    peak_rows.append({
                        "file_index": len(summary_rows) - 1,
                        "filename": filename,
                        "region_start": region_start,
                        "region_end": region_end,
                        "peak_number": i + 1,
                        "peak_center": center,
                        "amplitude": amplitude,
                        "width": width,
                        "fwhm": fwhm,
                        "area": area,
                        "total_r2": total_r2 if i == 0 else None,
                        "scan_index": spectrum_index,
                    })

            spectrum_index += 1

    return pd.DataFrame(peak_rows), pd.DataFrame(summary_rows)


def get_available_peaks(peaks_df: pd.DataFrame) -> List[int]:
    """Return sorted peak numbers present in the export."""
    if peaks_df is None or peaks_df.empty or "peak_number" not in peaks_df.columns:
        return []
    return sorted(int(v) for v in peaks_df["peak_number"].dropna().unique())


def peak_number_mean_wavenumber(peaks_df: pd.DataFrame, peak_number: int) -> Optional[float]:
    """Average fitted center for a peak number across the series."""
    if peaks_df is None or peaks_df.empty:
        return None
    subset = peaks_df[peaks_df["peak_number"] == peak_number]
    if subset.empty or "peak_center" not in subset.columns:
        return None
    return float(subset["peak_center"].mean())


def _prepare_peaks_df(peaks_df: pd.DataFrame) -> pd.DataFrame:
    """Sort peaks by scan order and ensure scan_index exists."""
    df = peaks_df.copy()
    if "scan_index" not in df.columns:
        if "file_index" in df.columns:
            df["scan_index"] = df["file_index"]
        else:
            df = df.reset_index(drop=True)
            df["scan_index"] = np.arange(len(df))
    return df.sort_values("scan_index").reset_index(drop=True)


def _pick_peak_in_range(group: pd.DataFrame, wmin: float, wmax: float, nominal: float) -> Optional[pd.Series]:
    """Choose the peak row in a wavenumber window (closest to nominal if several match)."""
    if group.empty or "peak_center" not in group.columns:
        return None
    in_range = group[(group["peak_center"] >= wmin) & (group["peak_center"] <= wmax)]
    if in_range.empty:
        return None
    idx = (in_range["peak_center"] - nominal).abs().idxmin()
    return in_range.loc[idx]


def extract_standard_peak_evolution(
    data: Dict[str, Any],
    standard: str,
    position_axis: str = "scan_index",
) -> Dict[str, Any]:
    """
    Extract internal-standard peak evolution by wavenumber window (Si or Corundum).

    standard: 'silicon' or 'corundum'
    """
    if standard not in INTERNAL_STANDARDS:
        raise ValueError(f"Unknown internal standard: {standard}")

    peaks_df = _prepare_peaks_df(data.get("peaks_df", pd.DataFrame()))
    if peaks_df.empty:
        raise ValueError("No peak data found in the PKL file.")

    spec = INTERNAL_STANDARDS[standard]
    rows: List[dict] = []
    for scan_index, group in peaks_df.groupby("scan_index", sort=True):
        row = _pick_peak_in_range(group, spec["wmin"], spec["wmax"], spec["nominal"])
        if row is None:
            continue
        rows.append(row.to_dict())

    if not rows:
        raise ValueError(
            f"No {spec['label']} peaks found between {spec['wmin']:.0f} and {spec['wmax']:.0f} cm⁻¹."
        )

    selected = pd.DataFrame(rows).sort_values("scan_index").reset_index(drop=True)
    n = len(selected)
    filenames = selected["filename"].astype(str).tolist() if "filename" in selected.columns else [f"point_{i}" for i in range(n)]

    if position_axis == "parsed_filename":
        x_values = np.array([_parse_scan_position(fn, i) for i, fn in enumerate(filenames)], dtype=float)
        x_label = "Parsed position"
    else:
        x_values = selected["scan_index"].astype(float).values
        x_label = "Scan index"

    positions = selected["peak_center"].astype(float).values
    return {
        "standard": standard,
        "standard_label": spec["label"],
        "x_values": x_values,
        "x_label": x_label,
        "scan_index": selected["scan_index"].astype(float).values,
        "scan_indices": np.arange(n, dtype=float),
        "filenames": filenames,
        "positions": positions,
        "n_points": n,
    }


def apply_internal_standard_correction(
    evolution: Dict[str, Any],
    standard_evolution: Dict[str, Any],
    reference_last_n: int,
) -> Dict[str, Any]:
    """
    Correct peak positions for instrumental drift using an internal standard.

    ν_corrected(t) = ν(t) + (ν_std,ref − ν_std(t)), where ν_std,ref is the mean of
    the last N standard peaks (crystallized end state).
    """
    target_scan = evolution.get("scan_index")
    if target_scan is None:
        target_scan = evolution.get("scan_indices", np.arange(evolution["n_points"], dtype=float))

    std_df = pd.DataFrame({
        "scan_index": standard_evolution.get("scan_index", standard_evolution.get("scan_indices")),
        "std_pos": standard_evolution["positions"],
    })

    merged = pd.DataFrame({"scan_index": target_scan}).merge(std_df, on="scan_index", how="left")
    if merged["std_pos"].isna().any():
        missing = int(merged["std_pos"].isna().sum())
        raise ValueError(
            f"Internal standard missing for {missing} scan(s). "
            "Ensure Si/Corundum is fit in every spectrum."
        )

    std_positions = merged["std_pos"].astype(float).values
    ref_std = reference_wavenumber_last_n(std_positions, reference_last_n)
    correction = ref_std - std_positions
    corrected = np.asarray(evolution["positions"], dtype=float) + correction

    corrected_evo = dict(evolution)
    corrected_evo["positions_raw"] = np.asarray(evolution["positions"], dtype=float).copy()
    corrected_evo["positions"] = corrected
    corrected_evo["standard_correction"] = correction
    corrected_evo["standard_reference"] = ref_std
    corrected_evo["standard_label"] = standard_evolution.get("standard_label", standard_evolution.get("standard"))
    return corrected_evo


def reference_wavenumber_last_n(positions: np.ndarray, last_n: int) -> float:
    """Mean wavenumber of the last N points (crystallized end-state reference)."""
    positions = np.asarray(positions, dtype=float)
    if positions.size == 0:
        raise ValueError("No positions available for reference.")
    n = max(1, min(int(last_n), len(positions)))
    return float(np.mean(positions[-n:]))


def _parse_scan_position(filename: str, index: int) -> float:
    """Best-effort numeric position from filename, else scan index."""
    name = Path(filename).stem
    patterns = [
        r"[xX][_\-]?(\d+(?:\.\d+)?)",
        r"[yY][_\-]?(\d+(?:\.\d+)?)",
        r"pos(?:ition)?[_\-]?(\d+(?:\.\d+)?)",
        r"scan[_\-]?(\d+(?:\.\d+)?)",
        r"line[_\-]?(\d+(?:\.\d+)?)",
        r"point[_\-]?(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*mm",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return float(match.group(1))
    return float(index)


def extract_peak_evolution(
    data: Dict[str, Any],
    peak_number: int = 1,
    position_axis: str = "scan_index",
) -> Dict[str, Any]:
    """
    Extract peak parameter evolution for one fitted peak across a line scan / series.

    position_axis: 'scan_index' | 'parsed_filename' | 'file_order'
    """
    peaks_df = data.get("peaks_df", pd.DataFrame())
    if peaks_df is None or peaks_df.empty:
        raise ValueError("No peak data found in the PKL file.")

    if "peak_number" not in peaks_df.columns:
        raise ValueError("peaks_df is missing 'peak_number' column.")

    selected = peaks_df[peaks_df["peak_number"] == peak_number].copy()
    if selected.empty:
        available = get_available_peaks(peaks_df)
        raise ValueError(
            f"Peak {peak_number} not found. Available peaks: {available or 'none'}"
        )

    if "scan_index" not in selected.columns:
        if "file_index" in selected.columns:
            selected["scan_index"] = selected["file_index"]
        else:
            selected = selected.reset_index(drop=True)
            selected["scan_index"] = np.arange(len(selected))

    selected = selected.sort_values("scan_index").reset_index(drop=True)

    n = len(selected)
    scan_index_values = selected["scan_index"].astype(float).values
    filenames = selected["filename"].astype(str).tolist() if "filename" in selected.columns else [f"point_{i}" for i in range(n)]

    if position_axis == "parsed_filename":
        x_values = np.array([_parse_scan_position(fn, i) for i, fn in enumerate(filenames)], dtype=float)
        x_label = "Parsed position"
    elif position_axis == "file_order":
        x_values = np.arange(n, dtype=float)
        x_label = "File order"
    else:
        x_values = scan_index_values.copy()
        x_label = "Scan index"

    positions = selected["peak_center"].astype(float).values
    amplitudes = selected["amplitude"].astype(float).values if "amplitude" in selected.columns else np.full(n, np.nan)
    fwhms = selected["fwhm"].astype(float).values if "fwhm" in selected.columns else np.full(n, np.nan)

    if "FWHM_Left" in selected.columns and "FWHM_Right" in selected.columns:
        fwhm_left = selected["FWHM_Left"].astype(float).values
        fwhm_right = selected["FWHM_Right"].astype(float).values
    else:
        fwhm_left = fwhms.copy()
        fwhm_right = fwhms.copy()

    r2 = (
        selected["total_r2"].astype(float).ffill().values
        if "total_r2" in selected.columns
        else np.full(n, np.nan)
    )

    return {
        "peak_number": peak_number,
        "x_values": x_values,
        "x_label": x_label,
        "scan_index": scan_index_values,
        "scan_indices": np.arange(n, dtype=float),
        "filenames": filenames,
        "positions": positions,
        "amplitudes": amplitudes,
        "fwhms": fwhms,
        "fwhm_left": fwhm_left,
        "fwhm_right": fwhm_right,
        "r2_values": r2,
        "n_points": n,
    }


def compute_strain_evolution(
    evolution: Dict[str, Any],
    reference_wavenumber: Optional[float] = None,
    gruneisen: float = 1.0,
    last_n_reference: int = 0,
) -> Dict[str, Any]:
    """
    Estimate strain proxy from peak shift along a scan/time series.

    Uses ε ≈ Δν / (ν_ref · γ). When last_n_reference > 0, ν_ref is the mean of the
    last N peak positions (crystallized end-state reference).
    """
    positions = np.asarray(evolution["positions"], dtype=float)
    if positions.size == 0:
        raise ValueError("No peak positions to analyze.")

    if last_n_reference and last_n_reference > 0:
        ref_wn = reference_wavenumber_last_n(positions, last_n_reference)
        reference_mode = f"last_{last_n_reference}_points"
    elif reference_wavenumber is not None:
        ref_wn = float(reference_wavenumber)
        reference_mode = "manual"
    else:
        ref_wn = float(positions[0])
        reference_mode = "first_point"

    if ref_wn == 0:
        raise ValueError("Reference wavenumber cannot be zero.")
    if gruneisen == 0:
        raise ValueError("Grüneisen parameter cannot be zero.")

    delta_nu = positions - ref_wn
    strain = delta_nu / (ref_wn * gruneisen)
    fractional_shift = delta_nu / ref_wn

    return {
        "reference_wavenumber": ref_wn,
        "reference_mode": reference_mode,
        "reference_last_n": int(last_n_reference) if last_n_reference else 0,
        "gruneisen": gruneisen,
        "delta_nu": delta_nu,
        "fractional_shift": fractional_shift,
        "strain": strain,
        "max_abs_strain": float(np.nanmax(np.abs(strain))),
        "total_shift": float(positions[-1] - positions[0]) if len(positions) > 1 else 0.0,
        "mean_shift": float(np.nanmean(delta_nu)),
    }
