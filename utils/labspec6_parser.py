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
from typing import Tuple, Optional, Dict, Any
from pathlib import Path


# Constants
_MAGIC           = b"LabSpec6"
_AXIS_MIN_CM1    = 40.0    # cm^-1  lowest plausible Raman shift stored
_AXIS_MAX_CM1    = 5000.0  # cm^-1  highest plausible Raman shift stored
_AXIS_MIN_STEP   = 0.3     # cm^-1 / pixel
_AXIS_MAX_STEP   = 20.0    # cm^-1 / pixel
_AXIS_MIN_POINTS = 50      # minimum run length to accept as a valid axis
_TAM_DATA_SKIP   = 15      # bytes from start of 'tam\x00' tag to first data word


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
        Locate the 'tam' data block and return n float32 intensity values.
        
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
        
        if data_offset + n * 4 > len(raw):
            return None
        
        intensity = np.frombuffer(
            raw[data_offset : data_offset + n * 4], dtype="<f4"
        ).copy()
        
        n_nonfinite = int(np.sum(~np.isfinite(intensity)))
        if n_nonfinite > 0:
            return None
        
        n_neg = int(np.sum(intensity < 0))
        if n_neg > 0:
            warnings.warn(
                f"{n_neg} negative intensity values detected -- "
                "dark-current or baseline correction may be required.",
                UserWarning,
                stacklevel=4,
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
