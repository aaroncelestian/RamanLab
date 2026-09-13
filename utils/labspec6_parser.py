"""
LabSpec6 Binary File Parser (.l6s)

Parser for HORIBA LabSpec 6 binary spectrum files (.l6s).

File Format Notes
~~~~~~~~~~~~~~~~~
The .l6s format serializes a live LabSpec 6 session as a sentinel-linked
block graph. Each block node is preceded by the 4-byte sentinel
0x09 0x10 0x00 0x00 and a 32-bit RAM address baked in at acquisition
time (not usable as a file offset). Data arrays are located by 4-char
ASCII tag rather than by pointer:

  'tam\\x00'  block — float32 array of spectral intensity (detector counts).
               Header layout after tag: pad(4) + RAM_ptr(4) + n_spectra(1) +
               align(2) -> data[N x float32]. Skip = 15 bytes from tag start.

  '\\xe3tam'  variant used by some LabSpec 6 exports (including newer single
               spectra). The intensity array begins 16 bytes after the
               '\\xe3' marker. When multiple '\\xe3tam' blocks exist, the
               intensity block is chosen heuristically (finite values that
               do not look like a Raman-shift axis).

  counted     fallback used when neither 'tam\\x00' nor '\\xe3tam' is present:
               locate the precursor record
                   f6 7f 00 00 | sub_id | n_points
               and read float32[n] immediately afterward (or after an
               optional sentinel + '\\xe3tam' header).

  'temx'     block (second occurrence, adjacent to the '1/cm' unit label)
               — float32 array of the Raman-shift axis (cm^-1), identified
               by a monotonically increasing scan with Raman-plausible step
               size (0.3-20 cm^-1/pixel).

  'film'     block — mixed-encoding sample name: ASCII prefix 'Spectrum'
               + null byte, then the filename as UTF-16LE + double-null.

Byte order: all multi-byte values little-endian.
"""

from __future__ import annotations

import struct
import warnings
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path


# Constants
_MAGIC           = b"LabSpec6"
_AXIS_MIN_CM1    = 40.0    # cm^-1  lowest plausible Raman shift stored
_AXIS_MAX_CM1    = 5000.0  # cm^-1  highest plausible Raman shift stored
_AXIS_MIN_STEP   = 0.3     # cm^-1 / pixel
_AXIS_MAX_STEP   = 20.0    # cm^-1 / pixel
_AXIS_MIN_POINTS = 50      # minimum run length to accept as a valid axis
_TAM_DATA_SKIP   = 15      # bytes from start of 'tam\x00' tag to first data word
_ETAM_MARKER     = b"\xe3tam"
_ETAM_DATA_SKIP  = 16      # bytes from start of '\xe3tam' marker to first data word


class LabSpec6Parser:
    """
    Parser for Horiba LabSpec 6 binary spectrum files (.l6s format).
    
    These files contain single Raman spectra with wavenumber and intensity data.
    """
    
    def __init__(self):
        """Initialize the LabSpec6 parser."""
        self.supported_extensions = ['.l6s']
        
    def load_spectrum(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """
        Load a LabSpec6 .l6s spectrum file.
        
        Args:
            file_path: Path to the .l6s file
            
        Returns:
            Tuple of (wavenumbers, intensities, metadata)
            Returns (None, None, {'error': message}) if loading fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return None, None, {'error': f'File not found: {file_path}'}
        
        try:
            raw = file_path.read_bytes()
            
            if raw[:8] != _MAGIC:
                return None, None, {'error': 'Not a LabSpec 6 file'}
            
            wavenumbers = self._read_axis(raw)
            if wavenumbers is None:
                return None, None, {'error': 'Could not extract wavenumber axis'}
            
            intensities = self._read_intensity(raw, len(wavenumbers))
            if intensities is None:
                return None, None, {'error': 'Could not extract intensity data'}
            
            sample_name = self._read_sample_name(raw)
            
            metadata = {
                'filename': file_path.name,
                'filepath': str(file_path),
                'file_size': file_path.stat().st_size,
                'format': 'LabSpec6',
                'sample_name': sample_name,
                'data_points': len(wavenumbers),
                'wavenumber_range': (float(wavenumbers[0]), float(wavenumbers[-1])),
                'wavenumber_step': float(np.mean(np.diff(wavenumbers))),
                'intensity_range': (float(np.min(intensities)), float(np.max(intensities)))
            }
            
            return wavenumbers, intensities, metadata
            
        except Exception as e:
            return None, None, {'error': f'Error parsing file: {str(e)}'}
    
    def _read_axis(self, raw: bytes) -> Optional[np.ndarray]:
        """
        Locate the Raman-shift axis via the '1/cm' unit label and return it.
        """
        try:
            unit_pos = raw.index(b"1/cm")
        except ValueError:
            return None
        
        wavenumber = self._scan_monotonic_floats(raw, start=unit_pos + 4)
        
        if len(wavenumber) < _AXIS_MIN_POINTS:
            return None
        
        return wavenumber
    
    def _scan_monotonic_floats(self, raw: bytes, start: int) -> np.ndarray:
        """
        Scan forward from start, collecting a monotonically increasing run
        of float32 values within the plausible Raman-shift range.
        """
        values = []
        pos = start
        
        while pos + 4 <= len(raw):
            v = struct.unpack_from("<f", raw, pos)[0]
            
            if not (_AXIS_MIN_CM1 < v < _AXIS_MAX_CM1):
                if values:
                    break
                pos += 4
                continue
            
            if values:
                step = v - values[-1]
                if not (_AXIS_MIN_STEP < step < _AXIS_MAX_STEP):
                    break
            
            values.append(v)
            pos += 4
        
        return np.array(values, dtype=np.float32)
    
    def _read_intensity(self, raw: bytes, n: int) -> Optional[np.ndarray]:
        """
        Locate spectral intensity data and return n float32 values.

        Tries, in order:
          1. classic 'tam\\x00' block
          2. newer '\\xe3tam' block(s)
          3. inline array after the point-count field (some exports omit the
             tam/e3tam wrapper and store floats immediately)
        """
        intensity = self._read_intensity_tam(raw, n)
        if intensity is not None:
            return intensity

        intensity = self._read_intensity_etam(raw, n)
        if intensity is not None:
            return intensity

        intensity = self._read_intensity_counted(raw, n)
        if intensity is not None:
            return intensity

        return None

    def _read_intensity_tam(self, raw: bytes, n: int) -> Optional[np.ndarray]:
        """
        Classic 'tam\\x00' intensity block.

        Block header layout (offsets relative to start of 'tam' tag):
            [0:4]   tag        = b'tam\\x00'
            [4:8]   uint32     pad / local flags  (= 0x00000000)
            [8:12]  uint32     RAM pointer        (ignore)
            [12]    uint8      n_spectra          (= 0x01 for single acquisition)
            [13:15] 2 bytes    alignment padding
            [15:]   N x float32  intensity values
        """
        try:
            tag_pos = raw.index(b"tam\x00")
        except ValueError:
            return None

        data_offset = tag_pos + _TAM_DATA_SKIP
        intensity = self._extract_float32_array(raw, data_offset, n)
        if intensity is None:
            return None

        return self._finalize_intensity(intensity)

    def _read_intensity_etam(self, raw: bytes, n: int) -> Optional[np.ndarray]:
        """
        Newer '\\xe3tam' intensity blocks.

        Data begins at marker_offset + 16. Some files contain multiple
        '\\xe3tam' blocks (e.g. intensity plus a stored axis copy); pick
        the best intensity-like candidate.
        """
        candidates: List[np.ndarray] = []
        start = 0
        while True:
            tag_pos = raw.find(_ETAM_MARKER, start)
            if tag_pos < 0:
                break

            data_offset = tag_pos + _ETAM_DATA_SKIP
            intensity = self._extract_float32_array(raw, data_offset, n)
            if (
                intensity is not None
                and not self._looks_like_wavenumber_axis(intensity)
                and self._looks_like_intensity(intensity)
            ):
                candidates.append(intensity)

            start = tag_pos + 1

        if not candidates:
            return None

        best = max(candidates, key=lambda arr: float(np.ptp(arr)))
        return self._finalize_intensity(best)

    def _read_intensity_counted(self, raw: bytes, n: int) -> Optional[np.ndarray]:
        """
        Fallback for exports that omit the tam/e3tam wrapper.

        LabSpec often stores a precursor record:
            f6 7f 00 00 | sub_id | n_points | [optional sentinel + e3tam] | float32[n]

        When the e3tam header is missing, intensity floats begin immediately
        after the point-count field.
        """
        if n <= 0:
            return None

        n_bytes = struct.pack("<I", n)
        prefix = b"\xf6\x7f\x00\x00"
        candidates: List[np.ndarray] = []
        start = 0

        while True:
            pos = raw.find(prefix, start)
            if pos < 0:
                break

            if raw[pos + 8 : pos + 12] != n_bytes:
                start = pos + 1
                continue

            data_offset = pos + 12
            # Optional block graph sentinel + e3tam header before the array
            if raw[data_offset : data_offset + 4] == b"\x09\x10\x00\x00":
                etam = raw.find(_ETAM_MARKER, data_offset, data_offset + 64)
                if etam >= 0:
                    data_offset = etam + _ETAM_DATA_SKIP

            intensity = self._extract_float32_array(raw, data_offset, n)
            if (
                intensity is not None
                and not self._looks_like_wavenumber_axis(intensity)
                and self._looks_like_intensity(intensity)
            ):
                candidates.append(intensity)

            start = pos + 1

        if not candidates:
            return None

        best = max(candidates, key=lambda arr: float(np.ptp(arr)))
        return self._finalize_intensity(best)

    def _extract_float32_array(self, raw: bytes, data_offset: int, n: int) -> Optional[np.ndarray]:
        """Read n little-endian float32 values starting at data_offset."""
        if n <= 0 or data_offset + n * 4 > len(raw):
            return None

        intensity = np.frombuffer(
            raw[data_offset : data_offset + n * 4], dtype="<f4"
        ).copy()

        if int(np.sum(~np.isfinite(intensity))) > 0:
            return None

        return intensity

    def _looks_like_wavenumber_axis(self, values: np.ndarray) -> bool:
        """Return True if values look like a Raman-shift axis, not intensities."""
        if len(values) < _AXIS_MIN_POINTS:
            return False

        if not (_AXIS_MIN_CM1 < float(values[0]) < _AXIS_MAX_CM1):
            return False
        if not (_AXIS_MIN_CM1 < float(values[-1]) < _AXIS_MAX_CM1):
            return False

        diffs = np.diff(values.astype(np.float64))
        if diffs.size == 0:
            return False

        return bool(
            np.all(diffs > _AXIS_MIN_STEP) and np.all(diffs < _AXIS_MAX_STEP)
        )

    def _looks_like_intensity(self, values: np.ndarray) -> bool:
        """Heuristic filter for plausible detector-count arrays."""
        if values.size == 0:
            return False

        peak_to_peak = float(np.ptp(values))
        max_abs = float(np.max(np.abs(values)))

        # Reject flat / denormal garbage and absurd magnitudes
        if peak_to_peak < 1.0:
            return False
        if max_abs > 1e8:
            return False

        return True

    def _finalize_intensity(self, intensity: np.ndarray) -> np.ndarray:
        """Apply intensity post-checks and return the array."""
        n_neg = int(np.sum(intensity < 0))
        if n_neg > 0:
            warnings.warn(
                f"{n_neg} negative intensity values detected -- "
                "dark-current or baseline correction may be required.",
                UserWarning,
                stacklevel=5,
            )

        return intensity

    def _read_sample_name(self, raw: bytes) -> str:
        """
        Extract the sample name from the 'film' block.
        
        The block payload begins with an ASCII prefix 'Spectrum' followed by
        a null byte, then the actual filename as UTF-16LE terminated by 0x0000.
        """
        try:
            film_pos = raw.index(b"film")
        except ValueError:
            return ""
        
        pos = film_pos + 16
        
        while pos < len(raw) and raw[pos] != 0:
            pos += 1
        pos += 1
        
        name_bytes = bytearray()
        while pos + 1 < len(raw):
            word = raw[pos : pos + 2]
            if word == b"\x00\x00":
                break
            name_bytes += word
            pos += 2
        
        return name_bytes.decode("utf-16-le", errors="replace")
    
    def is_supported_file(self, file_path: str) -> bool:
        """
        Check if a file is supported by this parser.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if supported, False otherwise
        """
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_extensions


def load_labspec6_spectrum(file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Convenience function to load a LabSpec6 .l6s spectrum file.
    
    Args:
        file_path: Path to the .l6s file
        
    Returns:
        Tuple of (wavenumbers, intensities, metadata)
    """
    parser = LabSpec6Parser()
    return parser.load_spectrum(file_path)


# Export main classes and functions
__all__ = ['LabSpec6Parser', 'load_labspec6_spectrum']
