"""
LabSpec6 Map File Parser (.l6m)

Parser for HORIBA LabSpec 6 binary 2D map files (.l6m).

File Format Notes
~~~~~~~~~~~~~~~~~
The .l6m format is structurally identical to .l6s at the block level: a
sentinel-linked graph with 4-char ASCII tags and 32-bit RAM addresses (baked
in at acquisition time, not usable as file offsets). The distinguishing
feature is the 'SpIm' (Spectral Image) type tag in the header.

Data blocks are located by an '\xe3tam' marker (a single-byte 0xe3
prefix followed by the 'tam\x00' tag). Each such block carries a 4-byte
sub-ID field at bytes [5:9] from the '\xe3' marker (little-endian,
pattern '7f 00 00 <sub_id>'). Sub-ID assignments:

    sub_id  hex   block content
    ------  ----  -------------
      17    0x11  spectral intensity cube  (float32, N_spectra × N_pts)
      23    0x17  Raman-shift axis         (float32, N_pts)
      26    0x1a  X spatial coordinates    (float32, N_x, µm)
      29    0x1d  Y spatial coordinates    (float32, N_y, µm)

All data arrays begin at offset +16 from the '\xe3' marker.

Map dimensions:
  * N_pts  = length of the wavenumber axis (sub_id 0x17)
  * N_x    = length of the X coordinate array (sub_id 0x1a)
  * N_y    = length of the Y coordinate array (sub_id 0x1d)
  * N_spectra = N_x × N_y
  * Spectral cube is stored row-major: spectra[N_y, N_x, N_pts]

The 'film' block encodes the sample name as pure UTF-16LE (no ASCII prefix),
null-terminated, beginning 16 bytes after the 'film' tag.

Byte order: all multi-byte values little-endian.
"""

from __future__ import annotations

import struct
import warnings
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path


# Constants
_MAGIC         = b"LabSpec6"
_ETAM_MARKER   = b"\xe3tam"
_DATA_SKIP     = 16   # bytes from \xe3 marker to first float32

# Sub-IDs (4th byte of the 4-byte flag field at marker+5)
_SID_INTENSITY  = 0x11   # 17 — spectral data cube
_SID_WAVENUMBER = 0x17   # 23 — Raman shift axis (cm^-1)
_SID_X          = 0x1a   # 26 — X stage positions (µm)
_SID_Y          = 0x1d   # 29 — Y stage positions (µm)

# Plausible Raman-shift bounds for axis validation
_WN_MIN, _WN_MAX   = 40.0, 5000.0
_WN_STEP_MIN       = 0.3
_WN_STEP_MAX       = 20.0

# Plausible stage-coordinate bounds (µm)
_STAGE_MIN, _STAGE_MAX = -300_000.0, 300_000.0


class LabSpec6MapParser:
    """
    Parser for Horiba LabSpec 6 binary map files (.l6m format).
    
    These files contain 2D Raman maps with spatial coordinates and spectral data.
    """
    
    def __init__(self):
        """Initialize the LabSpec6 map parser."""
        self.supported_extensions = ['.l6m']
    
    def load_map(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], 
                                                  Optional[np.ndarray], Optional[np.ndarray], 
                                                  Dict[str, Any]]:
        """
        Load a LabSpec6 .l6m map file.
        
        Args:
            file_path: Path to the .l6m file
            
        Returns:
            Tuple of (wavenumbers, x_coords, y_coords, cube, metadata)
            cube has shape (n_y, n_x, n_pts)
            Returns (None, None, None, None, {'error': message}) if loading fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return None, None, None, None, {'error': f'File not found: {file_path}'}
        
        try:
            raw = file_path.read_bytes()
            
            if raw[:8] != _MAGIC:
                return None, None, None, None, {'error': 'Not a LabSpec 6 file'}
            
            if b"SpIm" not in raw:
                return None, None, None, None, {'error': 'Not a LabSpec 6 map file (missing SpIm marker)'}
            
            blocks = self._locate_etam_blocks(raw)
            
            wavenumbers = self._read_etam_floats(raw, blocks, sub_id=_SID_WAVENUMBER)
            if wavenumbers is None:
                return None, None, None, None, {'error': 'Could not extract wavenumber axis'}
            
            x_coords = self._read_etam_floats(raw, blocks, sub_id=_SID_X)
            if x_coords is None:
                return None, None, None, None, {'error': 'Could not extract X coordinates'}
            
            y_coords = self._read_etam_floats(raw, blocks, sub_id=_SID_Y)
            if y_coords is None:
                return None, None, None, None, {'error': 'Could not extract Y coordinates'}
            
            n_pts = len(wavenumbers)
            n_x = len(x_coords)
            n_y = len(y_coords)
            
            cube = self._read_intensity_cube(raw, blocks, n_x=n_x, n_y=n_y, n_pts=n_pts)
            if cube is None:
                return None, None, None, None, {'error': 'Could not extract intensity cube'}
            
            sample_name = self._read_sample_name(raw)
            
            metadata = {
                'filename': file_path.name,
                'filepath': str(file_path),
                'file_size': file_path.stat().st_size,
                'format': 'LabSpec6 Map',
                'sample_name': sample_name,
                'n_x': n_x,
                'n_y': n_y,
                'n_pts': n_pts,
                'n_spectra': n_x * n_y,
                'wavenumber_range': (float(wavenumbers[0]), float(wavenumbers[-1])),
                'wavenumber_step': float(np.mean(np.diff(wavenumbers))),
                'x_range': (float(x_coords[0]), float(x_coords[-1])),
                'x_step': float(np.mean(np.diff(x_coords))),
                'y_range': (float(y_coords[0]), float(y_coords[-1])),
                'y_step': float(np.mean(np.diff(y_coords))),
                'intensity_range': (float(np.min(cube)), float(np.max(cube)))
            }
            
            return wavenumbers, x_coords, y_coords, cube, metadata
            
        except Exception as e:
            return None, None, None, None, {'error': f'Error parsing file: {str(e)}'}
    
    def _locate_etam_blocks(self, raw: bytes) -> dict:
        """
        Return a dict mapping sub_id -> list of (file_offset, data_offset) tuples.
        
        Each '\xe3tam' marker carries a sub-ID byte at offset +8 from the
        marker start. The float32 data begins at marker_offset + _DATA_SKIP.
        """
        import re
        blocks = {}
        for m in re.finditer(re.escape(_ETAM_MARKER), raw):
            p = m.start()
            if p + 9 > len(raw):
                continue
            flag_bytes = raw[p + 5 : p + 9]
            sub_id = flag_bytes[3]
            data_offset = p + _DATA_SKIP
            blocks.setdefault(sub_id, []).append((p, data_offset))
        return blocks
    
    def _read_etam_floats(self, raw: bytes, blocks: dict, sub_id: int) -> Optional[np.ndarray]:
        """
        Read the float32 array from the first matching sub_id block.
        """
        if sub_id not in blocks:
            return None
        
        _, data_offset = blocks[sub_id][0]
        
        if sub_id == _SID_WAVENUMBER:
            return self._scan_monotonic(raw, data_offset,
                                       vmin=_WN_MIN, vmax=_WN_MAX,
                                       step_min=_WN_STEP_MIN, step_max=_WN_STEP_MAX,
                                       min_points=50)
        
        if sub_id in (_SID_X, _SID_Y):
            return self._scan_monotonic(raw, data_offset,
                                       vmin=_STAGE_MIN, vmax=_STAGE_MAX,
                                       step_min=1e-6, step_max=1e5,
                                       min_points=2)
        
        return None
    
    def _scan_monotonic(self, raw: bytes, start: int, vmin: float, vmax: float,
                       step_min: float, step_max: float, min_points: int) -> Optional[np.ndarray]:
        """
        Scan forward from start, collecting a monotonically changing float32 run.
        """
        values = []
        pos = start
        direction = None  # +1 ascending, -1 descending
        
        while pos + 4 <= len(raw):
            v = struct.unpack_from("<f", raw, pos)[0]
            
            if not (vmin <= v <= vmax) and not (vmin >= v >= vmax):
                if values:
                    break
                pos += 4
                continue
            
            if values:
                step = v - values[-1]
                if direction is None:
                    if abs(step) < 1e-9:
                        pos += 4
                        continue
                    direction = 1.0 if step > 0 else -1.0
                signed_step = step * direction
                if not (step_min <= signed_step <= step_max):
                    break
            
            values.append(v)
            pos += 4
        
        arr = np.array(values, dtype=np.float32)
        if len(arr) < min_points:
            return None
        return arr
    
    def _read_intensity_cube(self, raw: bytes, blocks: dict, n_x: int, n_y: int, 
                            n_pts: int) -> Optional[np.ndarray]:
        """
        Read and reshape the spectral intensity cube.
        
        Returns:
            cube: ndarray, shape (n_y, n_x, n_pts), dtype float32
        """
        if _SID_INTENSITY not in blocks:
            return None
        
        _, data_offset = blocks[_SID_INTENSITY][0]
        
        n_spectra = n_x * n_y
        n_bytes = n_spectra * n_pts * 4
        
        if data_offset + n_bytes > len(raw):
            return None
        
        flat = np.frombuffer(raw[data_offset : data_offset + n_bytes],
                            dtype="<f4").copy()
        cube = flat.reshape(n_y, n_x, n_pts)
        
        n_nonfinite = int(np.sum(~np.isfinite(cube)))
        if n_nonfinite > 0:
            warnings.warn(
                f"{n_nonfinite} non-finite values in intensity cube. "
                "Offset misalignment or file corruption suspected.",
                UserWarning,
                stacklevel=4,
            )
        
        n_neg = int(np.sum(cube < 0))
        if n_neg > 0:
            warnings.warn(
                f"{n_neg} negative intensity values -- "
                "dark-current or baseline correction may be required.",
                UserWarning,
                stacklevel=4,
            )
        
        return cube
    
    def _read_sample_name(self, raw: bytes) -> str:
        """
        Extract sample name from the 'film' block (pure UTF-16LE in .l6m).
        """
        try:
            film_pos = raw.index(b"film")
        except ValueError:
            return ""
        
        pos = film_pos + 16
        
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
        """
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_extensions


def load_labspec6_map(file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], 
                                                 Optional[np.ndarray], Optional[np.ndarray], 
                                                 Dict[str, Any]]:
    """
    Convenience function to load a LabSpec6 .l6m map file.
    
    Args:
        file_path: Path to the .l6m file
        
    Returns:
        Tuple of (wavenumbers, x_coords, y_coords, cube, metadata)
        cube has shape (n_y, n_x, n_pts) where cube[i_y, i_x, i_wavenumber]
    """
    parser = LabSpec6MapParser()
    return parser.load_map(file_path)


# Export main classes and functions
__all__ = ['LabSpec6MapParser', 'load_labspec6_map']
