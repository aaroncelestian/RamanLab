"""
Quantitative Calibration System for Raman Spectroscopy

This module provides proper quantitative analysis with calibration standards,
response curves, and actual concentration values - addressing the fundamental
limitation of existing hybrid analysis methods that work only in arbitrary units.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import curve_fit
import pickle
import os

logger = logging.getLogger(__name__)


@dataclass
class CalibrationStandard:
    """Represents a calibration standard with known concentrations."""
    name: str
    concentration: float  # Known concentration (e.g., mg/kg, %)
    concentration_unit: str  # Unit of concentration
    material_type: str  # Type of material (e.g., "polypropylene", "polystyrene")
    spectrum_data: Optional[np.ndarray] = None  # Reference spectrum
    template_coefficient: Optional[float] = None  # Template fitting result
    nmf_intensity: Optional[float] = None  # NMF component intensity
    notes: str = ""


@dataclass
class CalibrationCurve:
    """Represents a calibration curve for quantitative analysis."""
    material_type: str
    method: str  # "template", "nmf", or "hybrid"
    concentrations: List[float]  # Known concentrations
    responses: List[float]  # Measured responses (template coeffs, NMF intensities, etc.)
    curve_params: Dict[str, float]  # Fitted curve parameters
    r_squared: float  # Quality of fit
    concentration_unit: str
    detection_limit: Optional[float] = None  # Method detection limit
    quantification_limit: Optional[float] = None  # Method quantification limit


class QuantitativeCalibrationManager:
    """Manages calibration standards and quantitative analysis."""
    
    def __init__(self):
        self.standards: List[CalibrationStandard] = []
        self.calibration_curves: Dict[str, CalibrationCurve] = {}  # keyed by method_material
        self.matrix_corrections: Dict[str, float] = {}  # Matrix effect corrections
        
    def add_standard(self, standard: CalibrationStandard) -> bool:
        """Add a calibration standard to the database."""
        try:
            # Validate standard
            if standard.concentration < 0:
                raise ValueError("Concentration cannot be negative")
            
            # Check for duplicates
            for existing in self.standards:
                if (existing.name == standard.name or 
                    (existing.material_type == standard.material_type and 
                     abs(existing.concentration - standard.concentration) < 1e-6)):
                    logger.warning(f"Similar standard already exists: {existing.name}")
            
            self.standards.append(standard)
            logger.info(f"Added calibration standard: {standard.name} ({standard.concentration} {standard.concentration_unit})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding calibration standard: {e}")
            return False
    
    def remove_standard(self, standard_name: str) -> bool:
        """Remove a calibration standard."""
        for i, standard in enumerate(self.standards):
            if standard.name == standard_name:
                del self.standards[i]
                logger.info(f"Removed calibration standard: {standard_name}")
                return True
        return False
    
    def get_standards_by_material(self, material_type: str) -> List[CalibrationStandard]:
        """Get all standards for a specific material type."""
        return [s for s in self.standards if s.material_type.lower() == material_type.lower()]
    
    def build_calibration_curve(self, material_type: str, method: str = "template", 
                               curve_type: str = "linear") -> Optional[CalibrationCurve]:
        """
        Build calibration curve from standards.
        
        Args:
            material_type: Type of material (e.g., "polypropylene")
            method: Analysis method ("template", "nmf", or "hybrid")
            curve_type: Type of curve ("linear", "quadratic", "exponential")
        """
        try:
            # Get standards for this material
            material_standards = self.get_standards_by_material(material_type)
            if len(material_standards) < 3:
                raise ValueError(f"Need at least 3 standards for {material_type}, found {len(material_standards)}")
            
            # Extract data based on method
            concentrations = []
            responses = []
            
            for standard in material_standards:
                if method == "template" and standard.template_coefficient is not None:
                    concentrations.append(standard.concentration)
                    responses.append(standard.template_coefficient)
                elif method == "nmf" and standard.nmf_intensity is not None:
                    concentrations.append(standard.concentration)
                    responses.append(standard.nmf_intensity)
                elif method == "hybrid" and standard.template_coefficient is not None and standard.nmf_intensity is not None:
                    # Use geometric mean of template and NMF responses
                    hybrid_response = np.sqrt(standard.template_coefficient * standard.nmf_intensity)
                    concentrations.append(standard.concentration)
                    responses.append(hybrid_response)
            
            if len(concentrations) < 3:
                raise ValueError(f"Insufficient data for {method} method: {len(concentrations)} points")
            
            concentrations = np.array(concentrations)
            responses = np.array(responses)
            
            # Fit calibration curve
            if curve_type == "linear":
                # y = mx + b
                slope, intercept, r_value, p_value, std_err = stats.linregress(concentrations, responses)
                curve_params = {"slope": slope, "intercept": intercept, "std_err": std_err}
                r_squared = r_value**2
                
            elif curve_type == "quadratic":
                # y = ax² + bx + c
                coeffs = np.polyfit(concentrations, responses, 2)
                curve_params = {"a": coeffs[0], "b": coeffs[1], "c": coeffs[2]}
                # Calculate R²
                y_pred = np.polyval(coeffs, concentrations)
                r_squared = 1 - np.sum((responses - y_pred)**2) / np.sum((responses - np.mean(responses))**2)
                
            elif curve_type == "exponential":
                # y = a * exp(b*x) + c
                def exp_func(x, a, b, c):
                    return a * np.exp(b * x) + c
                
                # Initial guess
                p0 = [np.max(responses), 0.1, np.min(responses)]
                popt, pcov = curve_fit(exp_func, concentrations, responses, p0=p0)
                curve_params = {"a": popt[0], "b": popt[1], "c": popt[2]}
                
                # Calculate R²
                y_pred = exp_func(concentrations, *popt)
                r_squared = 1 - np.sum((responses - y_pred)**2) / np.sum((responses - np.mean(responses))**2)
            
            else:
                raise ValueError(f"Unknown curve type: {curve_type}")
            
            # Calculate detection limits (3σ and 10σ criteria)
            if curve_type == "linear":
                # Calculate residuals standard deviation
                y_pred = slope * concentrations + intercept
                residuals_std = np.std(responses - y_pred)
                
                # Detection limit (3σ)
                detection_limit = 3 * residuals_std / slope if slope > 0 else None
                # Quantification limit (10σ)
                quantification_limit = 10 * residuals_std / slope if slope > 0 else None
            else:
                # For non-linear curves, use approximate method
                detection_limit = None
                quantification_limit = None
            
            # Create calibration curve
            curve = CalibrationCurve(
                material_type=material_type,
                method=method,
                concentrations=concentrations.tolist(),
                responses=responses.tolist(),
                curve_params=curve_params,
                r_squared=r_squared,
                concentration_unit=material_standards[0].concentration_unit,
                detection_limit=detection_limit,
                quantification_limit=quantification_limit
            )
            
            # Store curve
            curve_key = f"{method}_{material_type}"
            self.calibration_curves[curve_key] = curve
            
            logger.info(f"Built calibration curve for {material_type} using {method} method (R² = {r_squared:.4f})")
            return curve
            
        except Exception as e:
            logger.error(f"Error building calibration curve: {e}")
            return None
    
    def predict_concentration(self, response_value: float, material_type: str, 
                            method: str = "template") -> Optional[Dict[str, Any]]:
        """
        Predict concentration from response value using calibration curve.
        
        Args:
            response_value: Measured response (template coeff, NMF intensity, etc.)
            material_type: Type of material
            method: Analysis method used
            
        Returns:
            Dictionary with predicted concentration and uncertainty
        """
        try:
            curve_key = f"{method}_{material_type}"
            if curve_key not in self.calibration_curves:
                raise ValueError(f"No calibration curve available for {curve_key}")
            
            curve = self.calibration_curves[curve_key]
            
            # Predict concentration based on curve type
            if "slope" in curve.curve_params:  # Linear
                slope = curve.curve_params["slope"]
                intercept = curve.curve_params["intercept"]
                std_err = curve.curve_params.get("std_err", 0)
                
                concentration = (response_value - intercept) / slope
                
                # Calculate uncertainty (95% confidence interval)
                uncertainty = 1.96 * std_err / slope if slope != 0 else float('inf')
                
            elif "a" in curve.curve_params and "b" in curve.curve_params:  # Quadratic
                a, b, c = curve.curve_params["a"], curve.curve_params["b"], curve.curve_params["c"]
                # Solve quadratic equation: ax² + bx + (c - response_value) = 0
                discriminant = b**2 - 4*a*(c - response_value)
                if discriminant < 0:
                    raise ValueError("No real solution for quadratic equation")
                
                conc1 = (-b + np.sqrt(discriminant)) / (2*a)
                conc2 = (-b - np.sqrt(discriminant)) / (2*a)
                
                # Choose positive solution
                concentration = conc1 if conc1 >= 0 else conc2
                uncertainty = None  # More complex for quadratic
                
            else:
                raise ValueError("Unknown curve parameters")
            
            # Check against detection/quantification limits
            status = "quantified"
            if curve.quantification_limit and concentration < curve.quantification_limit:
                if curve.detection_limit and concentration < curve.detection_limit:
                    status = "not_detected"
                else:
                    status = "detected_not_quantified"
            
            result = {
                "concentration": concentration,
                "uncertainty": uncertainty,
                "unit": curve.concentration_unit,
                "status": status,
                "r_squared": curve.r_squared,
                "detection_limit": curve.detection_limit,
                "quantification_limit": curve.quantification_limit,
                "method": method
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting concentration: {e}")
            return None
    
    def apply_matrix_correction(self, concentration: float, sample_matrix: str, 
                              standard_matrix: str = "pure") -> float:
        """Apply matrix effect correction."""
        correction_key = f"{sample_matrix}_to_{standard_matrix}"
        correction_factor = self.matrix_corrections.get(correction_key, 1.0)
        return concentration * correction_factor
    
    def set_matrix_correction(self, sample_matrix: str, standard_matrix: str, 
                            correction_factor: float):
        """Set matrix effect correction factor."""
        correction_key = f"{sample_matrix}_to_{standard_matrix}"
        self.matrix_corrections[correction_key] = correction_factor
        logger.info(f"Set matrix correction {correction_key} = {correction_factor}")
    
    def validate_calibration(self, material_type: str, method: str = "template") -> Dict[str, Any]:
        """Validate calibration curve using cross-validation."""
        try:
            curve_key = f"{method}_{material_type}"
            if curve_key not in self.calibration_curves:
                raise ValueError(f"No calibration curve for {curve_key}")
            
            curve = self.calibration_curves[curve_key]
            concentrations = np.array(curve.concentrations)
            responses = np.array(curve.responses)
            
            # Leave-one-out cross-validation
            predictions = []
            actuals = []
            
            for i in range(len(concentrations)):
                # Remove one point
                train_conc = np.delete(concentrations, i)
                train_resp = np.delete(responses, i)
                test_conc = concentrations[i]
                test_resp = responses[i]
                
                # Fit curve without this point
                if len(train_conc) >= 2:
                    slope, intercept, _, _, _ = stats.linregress(train_conc, train_resp)
                    predicted_conc = (test_resp - intercept) / slope
                    
                    predictions.append(predicted_conc)
                    actuals.append(test_conc)
            
            if predictions:
                # Calculate validation metrics
                predictions = np.array(predictions)
                actuals = np.array(actuals)
                
                mse = np.mean((predictions - actuals)**2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(predictions - actuals))
                cv_r_squared = 1 - np.sum((actuals - predictions)**2) / np.sum((actuals - np.mean(actuals))**2)
                
                validation_results = {
                    "cross_validation_r_squared": cv_r_squared,
                    "rmse": rmse,
                    "mae": mae,
                    "n_points": len(actuals),
                    "bias": np.mean(predictions - actuals)
                }
                
                logger.info(f"Calibration validation for {curve_key}: CV R² = {cv_r_squared:.4f}")
                return validation_results
            
        except Exception as e:
            logger.error(f"Error validating calibration: {e}")
        
        return {}
    
    def save_calibration_data(self, filepath: str) -> bool:
        """Save calibration data to file."""
        try:
            data = {
                "standards": self.standards,
                "calibration_curves": self.calibration_curves,
                "matrix_corrections": self.matrix_corrections
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved calibration data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving calibration data: {e}")
            return False
    
    def load_calibration_data(self, filepath: str) -> bool:
        """Load calibration data from file."""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Calibration file not found: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.standards = data.get("standards", [])
            self.calibration_curves = data.get("calibration_curves", {})
            self.matrix_corrections = data.get("matrix_corrections", {})
            
            logger.info(f"Loaded calibration data from {filepath}")
            logger.info(f"Standards: {len(self.standards)}, Curves: {len(self.calibration_curves)}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading calibration data: {e}")
            return False
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of current calibration status."""
        materials = set(s.material_type for s in self.standards)
        
        summary = {
            "total_standards": len(self.standards),
            "materials": list(materials),
            "calibration_curves": list(self.calibration_curves.keys()),
            "matrix_corrections": len(self.matrix_corrections)
        }
        
        # Add per-material details
        for material in materials:
            material_standards = self.get_standards_by_material(material)
            summary[f"{material}_standards"] = len(material_standards)
            
            # Check if we have enough standards for calibration
            template_ready = sum(1 for s in material_standards if s.template_coefficient is not None) >= 3
            nmf_ready = sum(1 for s in material_standards if s.nmf_intensity is not None) >= 3
            
            summary[f"{material}_template_ready"] = template_ready
            summary[f"{material}_nmf_ready"] = nmf_ready
        
        return summary 