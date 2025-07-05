"""
Raman Geothermometry Calculator
Author: Based on published calibrations for metamorphic temperature determination
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

class GeothermometerMethod(Enum):
    """Enumeration of available geothermometry methods"""
    BEYSSAC_2002 = "Beyssac et al. (2002)"
    AOYA_2010_514 = "Aoya et al. (2010) - 514.5nm"
    AOYA_2010_532 = "Aoya et al. (2010) - 532nm"
    RAHL_2005 = "Rahl et al. (2005)"
    KOUKETSU_2014_D1 = "Kouketsu et al. (2014) - D1-FWHM"
    KOUKETSU_2014_D2 = "Kouketsu et al. (2014) - D2-FWHM"
    RANTITSCH_2004 = "Rantitsch et al. (2004)"

@dataclass
class GeothermometerInfo:
    """Information about each geothermometer method"""
    name: str
    description: str
    temp_range: str
    error: str
    required_params: List[str]
    best_for: str
    limitations: str

class RamanGeothermometry:
    """
    Class for calculating metamorphic temperatures from Raman spectral parameters
    """
    
    def __init__(self):
        self.methods_info = self._initialize_methods_info()
    
    def _initialize_methods_info(self) -> Dict[GeothermometerMethod, GeothermometerInfo]:
        """Initialize information about each geothermometry method"""
        return {
            GeothermometerMethod.BEYSSAC_2002: GeothermometerInfo(
                name="Beyssac et al. (2002)",
                description="T(°C) = -445 × R2 + 641",
                temp_range="330-650°C",
                error="±50°C",
                required_params=["R2"],
                best_for="Regional metamorphism, most widely used and tested",
                limitations="Not suitable for T < 330°C, requires good spectral quality"
            ),
            GeothermometerMethod.AOYA_2010_514: GeothermometerInfo(
                name="Aoya et al. (2010) - 514.5nm laser",
                description="T(°C) = 221 × R2² - 637.1 × R2 + 672.3",
                temp_range="340-655°C",
                error="±30°C",
                required_params=["R2"],
                best_for="Contact metamorphism, 514.5nm laser systems",
                limitations="Laser wavelength specific, similar range to Beyssac"
            ),
            GeothermometerMethod.AOYA_2010_532: GeothermometerInfo(
                name="Aoya et al. (2010) - 532nm laser",
                description="T(°C) = 91.4 × R2² - 556.3 × R2 + 676.3",
                temp_range="340-655°C",
                error="±30°C",
                required_params=["R2"],
                best_for="Contact metamorphism, 532nm laser systems",
                limitations="Laser wavelength specific, similar range to Beyssac"
            ),
            GeothermometerMethod.RAHL_2005: GeothermometerInfo(
                name="Rahl et al. (2005)",
                description="T(°C) = 737.3 + 320.9×R1 - 1067×R2 - 80.638×R1²",
                temp_range="100-700°C",
                error="±50°C",
                required_params=["R1", "R2"],
                best_for="Widest temperature range, low-grade metamorphism",
                limitations="Requires both R1 and R2, more complex calculation"
            ),
            GeothermometerMethod.KOUKETSU_2014_D1: GeothermometerInfo(
                name="Kouketsu et al. (2014) - D1-FWHM",
                description="T(°C) = -2.15 × D1-FWHM + 478",
                temp_range="150-400°C",
                error="±30°C",
                required_params=["D1_FWHM"],
                best_for="Low-grade metamorphism, diagenesis-metagenesis",
                limitations="Limited to low temperatures, requires D1 band fitting"
            ),
            GeothermometerMethod.KOUKETSU_2014_D2: GeothermometerInfo(
                name="Kouketsu et al. (2014) - D2-FWHM",
                description="T(°C) = -6.78 × D2-FWHM + 535",
                temp_range="150-400°C",
                error="±50°C",
                required_params=["D2_FWHM"],
                best_for="Low-grade metamorphism, when D2 band is resolvable",
                limitations="Difficult to separate D2 from G-band at low temperatures"
            ),
            GeothermometerMethod.RANTITSCH_2004: GeothermometerInfo(
                name="Rantitsch et al. (2004)",
                description="T(°C) = -457 × R2 + 648",
                temp_range="350-550°C",
                error="±53°C",
                required_params=["R2"],
                best_for="Isolated organic matter, medium-grade metamorphism",
                limitations="Limited temperature range, higher uncertainty"
            )
        }
    
    def get_method_info(self, method: GeothermometerMethod) -> GeothermometerInfo:
        """Get information about a specific method"""
        return self.methods_info[method]
    
    def get_all_methods(self) -> List[str]:
        """Get list of all available method names"""
        return [method.value for method in GeothermometerMethod]
    
    def get_tooltip_text(self, method: GeothermometerMethod) -> str:
        """Generate tooltip text for a method"""
        info = self.methods_info[method]
        tooltip = f"""
{info.name}
Equation: {info.description}
Temperature range: {info.temp_range}
Error: {info.error}

Best for: {info.best_for}
Limitations: {info.limitations}
Required parameters: {', '.join(info.required_params)}
        """.strip()
        return tooltip
    
    def calculate_temperature(self, method: GeothermometerMethod, **kwargs) -> Tuple[Optional[float], str]:
        """
        Calculate temperature using specified method
        
        Args:
            method: GeothermometerMethod enum
            **kwargs: Required parameters for the method
            
        Returns:
            Tuple of (temperature in °C, status message)
        """
        try:
            if method == GeothermometerMethod.BEYSSAC_2002:
                return self._beyssac_2002(**kwargs)
            elif method == GeothermometerMethod.AOYA_2010_514:
                return self._aoya_2010_514(**kwargs)
            elif method == GeothermometerMethod.AOYA_2010_532:
                return self._aoya_2010_532(**kwargs)
            elif method == GeothermometerMethod.RAHL_2005:
                return self._rahl_2005(**kwargs)
            elif method == GeothermometerMethod.KOUKETSU_2014_D1:
                return self._kouketsu_2014_d1(**kwargs)
            elif method == GeothermometerMethod.KOUKETSU_2014_D2:
                return self._kouketsu_2014_d2(**kwargs)
            elif method == GeothermometerMethod.RANTITSCH_2004:
                return self._rantitsch_2004(**kwargs)
            else:
                return None, "Unknown method"
        except Exception as e:
            return None, f"Calculation error: {str(e)}"
    
    def _beyssac_2002(self, R2: float) -> Tuple[float, str]:
        """Beyssac et al. (2002) method"""
        if not (0 <= R2 <= 1):
            return None, "R2 must be between 0 and 1"
        
        temp = -445 * R2 + 641
        
        if temp < 330 or temp > 650:
            status = f"Warning: Temperature ({temp:.1f}°C) outside calibrated range (330-650°C)"
        else:
            status = "Success"
        
        return temp, status
    
    def _aoya_2010_514(self, R2: float) -> Tuple[float, str]:
        """Aoya et al. (2010) method for 514.5nm laser"""
        if not (0 <= R2 <= 1):
            return None, "R2 must be between 0 and 1"
        
        temp = 221 * R2**2 - 637.1 * R2 + 672.3
        
        if temp < 340 or temp > 655:
            status = f"Warning: Temperature ({temp:.1f}°C) outside calibrated range (340-655°C)"
        else:
            status = "Success"
        
        return temp, status
    
    def _aoya_2010_532(self, R2: float) -> Tuple[float, str]:
        """Aoya et al. (2010) method for 532nm laser"""
        if not (0 <= R2 <= 1):
            return None, "R2 must be between 0 and 1"
        
        temp = 91.4 * R2**2 - 556.3 * R2 + 676.3
        
        if temp < 340 or temp > 655:
            status = f"Warning: Temperature ({temp:.1f}°C) outside calibrated range (340-655°C)"
        else:
            status = "Success"
        
        return temp, status
    
    def _rahl_2005(self, R1: float, R2: float) -> Tuple[float, str]:
        """Rahl et al. (2005) method"""
        if not (0 <= R2 <= 1):
            return None, "R2 must be between 0 and 1"
        if R1 < 0:
            return None, "R1 must be positive"
        
        temp = 737.3 + 320.9 * R1 - 1067 * R2 - 80.638 * R1**2
        
        if temp < 100 or temp > 700:
            status = f"Warning: Temperature ({temp:.1f}°C) outside calibrated range (100-700°C)"
        else:
            status = "Success"
        
        return temp, status
    
    def _kouketsu_2014_d1(self, D1_FWHM: float) -> Tuple[float, str]:
        """Kouketsu et al. (2014) D1-FWHM method"""
        if D1_FWHM <= 0:
            return None, "D1-FWHM must be positive"
        
        temp = -2.15 * D1_FWHM + 478
        
        if temp < 150 or temp > 400:
            status = f"Warning: Temperature ({temp:.1f}°C) outside calibrated range (150-400°C)"
        else:
            status = "Success"
        
        return temp, status
    
    def _kouketsu_2014_d2(self, D2_FWHM: float) -> Tuple[float, str]:
        """Kouketsu et al. (2014) D2-FWHM method"""
        if D2_FWHM <= 0:
            return None, "D2-FWHM must be positive"
        
        temp = -6.78 * D2_FWHM + 535
        
        if temp < 150 or temp > 400:
            status = f"Warning: Temperature ({temp:.1f}°C) outside calibrated range (150-400°C)"
        else:
            status = "Success"
        
        return temp, status
    
    def _rantitsch_2004(self, R2: float) -> Tuple[float, str]:
        """Rantitsch et al. (2004) method"""
        if not (0 <= R2 <= 1):
            return None, "R2 must be between 0 and 1"
        
        temp = -457 * R2 + 648
        
        if temp < 350 or temp > 550:
            status = f"Warning: Temperature ({temp:.1f}°C) outside calibrated range (350-550°C)"
        else:
            status = "Success"
        
        return temp, status
    
    def batch_calculate(self, method: GeothermometerMethod, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Batch calculate temperatures for multiple measurements
        
        Args:
            method: GeothermometerMethod enum
            data_dict: Dictionary containing parameter arrays
            
        Returns:
            Dictionary with 'temperatures' and 'status' arrays
        """
        required_params = self.methods_info[method].required_params
        
        # Check if all required parameters are present
        for param in required_params:
            if param not in data_dict:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Get the length of arrays (assume all have same length)
        n_points = len(data_dict[required_params[0]])
        
        temperatures = np.full(n_points, np.nan)
        status_messages = np.full(n_points, "", dtype=object)
        
        for i in range(n_points):
            # Extract parameters for this measurement
            params = {}
            for param in required_params:
                params[param] = data_dict[param][i]
            
            # Calculate temperature
            temp, status = self.calculate_temperature(method, **params)
            temperatures[i] = temp if temp is not None else np.nan
            status_messages[i] = status
        
        return {
            'temperatures': temperatures,
            'status': status_messages
        }

# Example usage functions for testing
def example_single_calculation():
    """Example of single temperature calculation"""
    calc = RamanGeothermometry()
    
    # Example using Beyssac method
    temp, status = calc.calculate_temperature(
        GeothermometerMethod.BEYSSAC_2002, 
        R2=0.5
    )
    print(f"Beyssac (2002): {temp:.1f}°C - {status}")
    
    # Example using Rahl method
    temp, status = calc.calculate_temperature(
        GeothermometerMethod.RAHL_2005,
        R1=1.2,
        R2=0.4
    )
    print(f"Rahl (2005): {temp:.1f}°C - {status}")

def example_batch_calculation():
    """Example of batch temperature calculation"""
    calc = RamanGeothermometry()
    
    # Simulate batch data
    data = {
        'R2': np.array([0.3, 0.4, 0.5, 0.6, 0.7]),
        'R1': np.array([1.0, 1.2, 1.5, 1.8, 2.0])
    }
    
    # Batch calculation using Rahl method
    results = calc.batch_calculate(GeothermometerMethod.RAHL_2005, data)
    
    print("Batch calculation results:")
    for i, (temp, status) in enumerate(zip(results['temperatures'], results['status'])):
        print(f"Point {i+1}: {temp:.1f}°C - {status}")

if __name__ == "__main__":
    # Run examples
    print("Single calculation example:")
    example_single_calculation()
    
    print("\nBatch calculation example:")
    example_batch_calculation()
    
    # Show available methods and tooltips
    calc = RamanGeothermometry()
    print("\nAvailable methods and tooltips:")
    for method in GeothermometerMethod:
        print(f"\n{method.value}:")
        print(calc.get_tooltip_text(method))
