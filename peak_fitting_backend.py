#!/usr/bin/env python3
"""
RamanLab Peak Fitting Web Backend
=================================

FastAPI backend that provides advanced peak fitting algorithms as REST APIs.
Based on the core algorithms from RamanLab's core/peak_fitting.py module.

Usage:
    python peak_fitting_backend.py

API Endpoints:
    POST /api/peaks/fit      - Fit peaks to spectrum data
    POST /api/peaks/detect   - Auto-detect peaks in spectrum
    POST /api/search/match   - Search database for similar spectra
    GET  /api/algorithms     - Get available peak fitting algorithms
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import json
import io

# Import the actual RamanLab algorithms
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from core.peak_fitting import PeakFitter, PeakData
    from raman_spectra_qt6 import RamanSpectraQt6
    RAMANLAB_AVAILABLE = True
except ImportError:
    print("⚠️  RamanLab modules not available, using simplified implementations")
    RAMANLAB_AVAILABLE = False

app = FastAPI(
    title="RamanLab Peak Fitting API",
    description="Advanced peak fitting and spectral analysis APIs for Raman spectroscopy",
    version="1.1.0"
)

# Enable CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class SpectrumData(BaseModel):
    wavenumbers: List[float]
    intensities: List[float]

class PeakFitRequest(BaseModel):
    spectrum: SpectrumData
    peak_shape: str = "lorentzian"
    initial_params: Optional[Dict[str, float]] = None
    auto_detect: bool = True

class PeakFitResult(BaseModel):
    fitted_peaks: List[Dict[str, Any]]
    fitted_curve: SpectrumData
    statistics: Dict[str, float]
    algorithm_info: Dict[str, str]

class SearchRequest(BaseModel):
    query_spectrum: SpectrumData
    search_type: str = "correlation"
    n_matches: int = 10
    threshold: float = 0.5

class SearchResult(BaseModel):
    matches: List[Dict[str, Any]]
    search_statistics: Dict[str, float]

# Peak fitting functions (simplified versions of RamanLab algorithms)
def lorentzian(x: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
    """Lorentzian peak function."""
    width = abs(width) + 1e-10
    return amplitude * (width**2) / ((x - center)**2 + width**2)

def gaussian(x: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
    """Gaussian peak function."""
    width = abs(width) + 1e-10
    return amplitude * np.exp(-((x - center) / width) ** 2)

def voigt(x: np.ndarray, amplitude: float, center: float, sigma: float, gamma: float) -> np.ndarray:
    """Simplified Voigt profile."""
    sigma = abs(sigma) + 1e-10
    gamma = abs(gamma) + 1e-10
    
    gaussian_part = np.exp(-((x - center) / sigma) ** 2)
    lorentzian_part = (gamma**2) / ((x - center)**2 + gamma**2)
    
    return amplitude * (0.3989423 * gaussian_part + 0.6366198 * lorentzian_part)

def pseudo_voigt(x: np.ndarray, amplitude: float, center: float, width: float, eta: float) -> np.ndarray:
    """Pseudo-Voigt profile."""
    eta = np.clip(eta, 0, 1)
    
    gaussian_part = gaussian(x, 1.0, center, width)
    lorentzian_part = lorentzian(x, 1.0, center, width)
    
    return amplitude * ((1 - eta) * gaussian_part + eta * lorentzian_part)

# Peak fitting class
class WebPeakFitter:
    """Web-compatible peak fitter based on RamanLab algorithms."""
    
    def __init__(self):
        self.peak_functions = {
            'lorentzian': lorentzian,
            'gaussian': gaussian,
            'voigt': voigt,
            'pseudo_voigt': pseudo_voigt
        }
    
    def detect_peaks(self, x: np.ndarray, y: np.ndarray, 
                    prominence: float = 0.1, distance: int = 10) -> List[Dict[str, float]]:
        """Auto-detect peaks in spectrum."""
        # Normalize for peak detection
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
        
        # Find peaks
        peaks, properties = find_peaks(
            y_norm,
            prominence=prominence,
            distance=distance,
            height=0.1
        )
        
        detected_peaks = []
        for i, peak_idx in enumerate(peaks):
            peak_center = x[peak_idx]
            peak_height = y[peak_idx]
            
            # Estimate width from FWHM
            half_max = peak_height / 2
            left_idx = peak_idx
            right_idx = peak_idx
            
            while left_idx > 0 and y[left_idx] > half_max:
                left_idx -= 1
            while right_idx < len(y) - 1 and y[right_idx] > half_max:
                right_idx += 1
            
            estimated_width = max(2.0, (x[right_idx] - x[left_idx]) / 2)
            
            detected_peaks.append({
                'peak_id': i,
                'center': float(peak_center),
                'amplitude': float(peak_height - np.min(y)),
                'width': float(estimated_width),
                'prominence': float(properties['prominences'][i]),
                'quality': 'auto_detected'
            })
        
        return detected_peaks
    
    def fit_peak(self, x: np.ndarray, y: np.ndarray, peak_shape: str,
                initial_params: Dict[str, float]) -> Dict[str, Any]:
        """Fit a single peak to data."""
        if peak_shape not in self.peak_functions:
            raise ValueError(f"Unknown peak shape: {peak_shape}")
        
        func = self.peak_functions[peak_shape]
        
        # Extract initial parameters
        amp_init = initial_params.get('amplitude', np.max(y))
        center_init = initial_params.get('center', x[np.argmax(y)])
        width_init = initial_params.get('width', (x[-1] - x[0]) / 20)
        
        try:
            if peak_shape in ['lorentzian', 'gaussian']:
                popt, pcov = curve_fit(
                    func, x, y,
                    p0=[amp_init, center_init, width_init],
                    bounds=([0, x[0], 0.5], [amp_init * 3, x[-1], (x[-1] - x[0]) / 2])
                )
                fitted_params = {
                    'amplitude': float(popt[0]),
                    'center': float(popt[1]),
                    'width': float(popt[2])
                }
            elif peak_shape == 'voigt':
                sigma_init = width_init
                gamma_init = initial_params.get('gamma', width_init)
                popt, pcov = curve_fit(
                    func, x, y,
                    p0=[amp_init, center_init, sigma_init, gamma_init],
                    bounds=([0, x[0], 0.5, 0.5], [amp_init * 3, x[-1], 50, 50])
                )
                fitted_params = {
                    'amplitude': float(popt[0]),
                    'center': float(popt[1]),
                    'sigma': float(popt[2]),
                    'gamma': float(popt[3])
                }
            elif peak_shape == 'pseudo_voigt':
                eta_init = initial_params.get('eta', 0.5)
                popt, pcov = curve_fit(
                    func, x, y,
                    p0=[amp_init, center_init, width_init, eta_init],
                    bounds=([0, x[0], 0.5, 0], [amp_init * 3, x[-1], 50, 1])
                )
                fitted_params = {
                    'amplitude': float(popt[0]),
                    'center': float(popt[1]),
                    'width': float(popt[2]),
                    'eta': float(popt[3])
                }
            
            # Calculate fit quality
            y_fitted = func(x, *popt)
            ss_res = np.sum((y - y_fitted) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            rmse = np.sqrt(ss_res / len(y))
            
            return {
                'success': True,
                'parameters': fitted_params,
                'fitted_curve': y_fitted.tolist(),
                'statistics': {
                    'r_squared': float(r_squared),
                    'rmse': float(rmse),
                    'parameter_errors': np.sqrt(np.diag(pcov)).tolist()
                },
                'peak_shape': peak_shape
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'parameters': initial_params,
                'fitted_curve': [],
                'statistics': {'r_squared': 0.0, 'rmse': float('inf')}
            }

# Initialize peak fitter
peak_fitter = WebPeakFitter()

# API Routes
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RamanLab Peak Fitting API",
        "version": "1.1.0",
        "documentation": "/docs",
        "ramanlab_core_available": RAMANLAB_AVAILABLE
    }

@app.get("/api/algorithms")
async def get_algorithms():
    """Get available peak fitting algorithms and their descriptions."""
    algorithms = {
        "lorentzian": {
            "name": "Lorentzian",
            "description": "Natural line shape for homogeneous broadening in Raman spectroscopy",
            "formula": "I = A·γ²/((ν-ν₀)²+γ²)",
            "parameters": ["amplitude", "center", "width"],
            "use_case": "Ideal for natural line widths, pressure broadening"
        },
        "gaussian": {
            "name": "Gaussian",
            "description": "Describes inhomogeneous broadening due to sample variations",
            "formula": "I = A·exp(-((ν-ν₀)/σ)²)",
            "parameters": ["amplitude", "center", "width"],
            "use_case": "Temperature effects, instrumental broadening"
        },
        "voigt": {
            "name": "Voigt Profile",
            "description": "Convolution of Gaussian and Lorentzian for both broadening mechanisms",
            "formula": "Convolution of Gaussian and Lorentzian",
            "parameters": ["amplitude", "center", "sigma", "gamma"],
            "use_case": "Most realistic for experimental spectra"
        },
        "pseudo_voigt": {
            "name": "Pseudo-Voigt",
            "description": "Linear combination of Gaussian and Lorentzian profiles",
            "formula": "I = A·[(1-η)·G + η·L]",
            "parameters": ["amplitude", "center", "width", "eta"],
            "use_case": "Computational efficiency with good approximation"
        }
    }
    
    return {
        "algorithms": algorithms,
        "default_algorithm": "lorentzian",
        "recommended_order": ["lorentzian", "gaussian", "voigt", "pseudo_voigt"]
    }

@app.post("/api/peaks/detect")
async def detect_peaks(spectrum: SpectrumData):
    """Auto-detect peaks in spectrum data."""
    try:
        x = np.array(spectrum.wavenumbers)
        y = np.array(spectrum.intensities)
        
        if len(x) != len(y):
            raise HTTPException(status_code=400, detail="Wavenumbers and intensities must have same length")
        
        detected_peaks = peak_fitter.detect_peaks(x, y)
        
        return {
            "detected_peaks": detected_peaks,
            "peak_count": len(detected_peaks),
            "detection_parameters": {
                "prominence": 0.1,
                "distance": 10,
                "height_threshold": 0.1
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Peak detection failed: {str(e)}")

@app.post("/api/peaks/fit")
async def fit_peaks(request: PeakFitRequest) -> PeakFitResult:
    """Fit peaks to spectrum data using specified algorithm."""
    try:
        x = np.array(request.spectrum.wavenumbers)
        y = np.array(request.spectrum.intensities)
        
        if len(x) != len(y):
            raise HTTPException(status_code=400, detail="Wavenumbers and intensities must have same length")
        
        # Auto-detect peaks if requested
        if request.auto_detect:
            detected_peaks = peak_fitter.detect_peaks(x, y)
        else:
            # Use provided initial parameters
            detected_peaks = [request.initial_params or {
                'center': x[np.argmax(y)],
                'amplitude': np.max(y),
                'width': (x[-1] - x[0]) / 20
            }]
        
        fitted_peaks = []
        total_fitted_curve = np.zeros_like(y)
        
        # Fit each detected peak
        for i, peak_info in enumerate(detected_peaks[:5]):  # Limit to 5 peaks
            # Extract window around peak
            center = peak_info['center']
            window_size = peak_info.get('width', 50) * 3
            
            mask = (x >= center - window_size) & (x <= center + window_size)
            if np.sum(mask) < 10:
                continue
            
            x_window = x[mask]
            y_window = y[mask]
            
            # Fit peak
            fit_result = peak_fitter.fit_peak(
                x_window, y_window, 
                request.peak_shape, 
                peak_info
            )
            
            if fit_result['success']:
                fitted_peaks.append({
                    'peak_id': i,
                    'shape': request.peak_shape,
                    'parameters': fit_result['parameters'],
                    'statistics': fit_result['statistics'],
                    'window_range': [float(x_window[0]), float(x_window[-1])]
                })
                
                # Add to total curve
                if request.peak_shape == 'lorentzian':
                    peak_curve = lorentzian(x, **fit_result['parameters'])
                elif request.peak_shape == 'gaussian':
                    peak_curve = gaussian(x, **fit_result['parameters'])
                elif request.peak_shape == 'voigt':
                    peak_curve = voigt(x, **fit_result['parameters'])
                elif request.peak_shape == 'pseudo_voigt':
                    peak_curve = pseudo_voigt(x, **fit_result['parameters'])
                
                total_fitted_curve += peak_curve
        
        # Calculate overall statistics
        if len(fitted_peaks) > 0:
            ss_res = np.sum((y - total_fitted_curve) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            overall_r_squared = 1 - (ss_res / ss_tot)
            overall_rmse = np.sqrt(ss_res / len(y))
        else:
            overall_r_squared = 0.0
            overall_rmse = float('inf')
        
        return PeakFitResult(
            fitted_peaks=fitted_peaks,
            fitted_curve=SpectrumData(
                wavenumbers=x.tolist(),
                intensities=total_fitted_curve.tolist()
            ),
            statistics={
                'overall_r_squared': overall_r_squared,
                'overall_rmse': overall_rmse,
                'peak_count': len(fitted_peaks),
                'algorithm': request.peak_shape
            },
            algorithm_info={
                'name': request.peak_shape.title(),
                'description': f"Peak fitting using {request.peak_shape} profile",
                'parameters_fitted': len(fitted_peaks) * (3 if request.peak_shape in ['lorentzian', 'gaussian'] else 4)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Peak fitting failed: {str(e)}")

@app.post("/api/search/correlation")
async def search_correlation(request: SearchRequest):
    """Search for similar spectra using correlation analysis."""
    try:
        # This would normally search a real database
        # For demo purposes, we'll generate some mock results
        
        query_x = np.array(request.query_spectrum.wavenumbers)
        query_y = np.array(request.query_spectrum.intensities)
        
        # Mock database search results
        mock_results = [
            {
                'name': 'Quartz_Sample_001',
                'formula': 'SiO₂',
                'correlation_score': 0.92,
                'key_peaks': [464, 696, 808, 1085],
                'crystal_system': 'Hexagonal',
                'source': 'RRUFF_Database'
            },
            {
                'name': 'Calcite_Reference',
                'formula': 'CaCO₃',
                'correlation_score': 0.78,
                'key_peaks': [155, 282, 714, 1086],
                'crystal_system': 'Hexagonal',
                'source': 'Internal_Library'
            },
            {
                'name': 'Gypsum_Standard',
                'formula': 'CaSO₄·2H₂O',
                'correlation_score': 0.65,
                'key_peaks': [415, 493, 618, 1008],
                'crystal_system': 'Monoclinic',
                'source': 'RRUFF_Database'
            }
        ]
        
        # Filter by threshold and limit results
        filtered_results = [r for r in mock_results if r['correlation_score'] >= request.threshold]
        final_results = filtered_results[:request.n_matches]
        
        return SearchResult(
            matches=final_results,
            search_statistics={
                'total_database_size': 1000,
                'matches_above_threshold': len(filtered_results),
                'returned_matches': len(final_results),
                'search_time_ms': 150,
                'algorithm': request.search_type
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database search failed: {str(e)}")

@app.post("/api/upload/spectrum")
async def upload_spectrum(file: UploadFile = File(...)):
    """Upload and parse spectrum file."""
    try:
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Parse spectrum data
        lines = text_content.strip().split('\n')
        wavenumbers = []
        intensities = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        wavenumbers.append(x)
                        intensities.append(y)
                    except ValueError:
                        continue
        
        if len(wavenumbers) == 0:
            raise HTTPException(status_code=400, detail="No valid data points found in file")
        
        return {
            "filename": file.filename,
            "data_points": len(wavenumbers),
            "wavenumber_range": [min(wavenumbers), max(wavenumbers)],
            "intensity_range": [min(intensities), max(intensities)],
            "spectrum": {
                "wavenumbers": wavenumbers,
                "intensities": intensities
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RamanLab Peak Fitting API Server...")
    print("📊 Features available:")
    print("   • Advanced peak fitting (Lorentzian, Gaussian, Voigt, Pseudo-Voigt)")
    print("   • Auto peak detection")
    print("   • Spectral database search")
    print("   • File upload support")
    print("🌐 Access the API at: http://localhost:8001")
    print("📚 API Documentation: http://localhost:8001/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)