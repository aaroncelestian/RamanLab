# RamanLab Jupyter Console Integration Update
## New Interactive Console Features

### What's New? 🎉

RamanLab now includes **interactive Jupyter console integration** that allows you to:

- 🐍 **Interactive Python Console**: Run Python commands directly within RamanLab
- 📊 **Live Data Access**: Access your loaded spectra and analysis results interactively
- 🔬 **Advanced Analysis**: Run custom analysis scripts without leaving RamanLab
- 📈 **Dynamic Plotting**: Create custom plots and visualizations on-the-fly
- 🧪 **Experiment Interactively**: Test analysis parameters and methods in real-time

### New Dependencies Required

To use the new Jupyter console features, you need these additional packages:

```bash
pip install qtconsole>=5.4.0 jupyter-client>=7.0.0 ipykernel>=6.0.0
```

### Quick Update Guide

#### Option 1: Automatic Update (Recommended)
```bash
# Run the new update script
python update_dependencies.py --jupyter
```

#### Option 2: Manual Update
```bash
# Install new Jupyter packages
pip install qtconsole jupyter-client ipykernel

# Update all packages
pip install --upgrade -r requirements_qt6.txt
```

#### Option 3: Full Update
```bash
# Update everything
python update_dependencies.py --all
```

### Using the New Features

#### Launch Jupyter Console
```bash
python launch_jupyter_console.py
```

#### Access from RamanLab
- Look for the **"Interactive Console"** button in the main interface
- Open the **Advanced** tab for console integration
- Use **Tools → Interactive Console** from the menu

### Features Available in Console

```python
# Access loaded spectra
spectra = get_current_spectra()

# Run custom analysis
import numpy as np
import matplotlib.pyplot as plt

# Create custom plots
plt.plot(spectra.wavenumber, spectra.intensity)
plt.show()

# Access RamanLab functions
from core.peak_fitting import fit_peaks
results = fit_peaks(spectra, method='voigt')
```

### Troubleshooting

#### If Installation Fails
```bash
# Check dependencies
python check_dependencies.py

# Try updating pip first
pip install --upgrade pip

# Install one by one
pip install qtconsole
pip install jupyter-client
pip install ipykernel
```

#### If Console Doesn't Launch
```bash
# Test Jupyter installation
python -c "import qtconsole; print('✅ qtconsole OK')"
python -c "import jupyter_client; print('✅ jupyter-client OK')"
python -c "import ipykernel; print('✅ ipykernel OK')"
```

### Virtual Environment Recommendation

For best results, use a virtual environment:

```bash
# Create virtual environment
python -m venv ramanlab_env

# Activate (Windows)
ramanlab_env\Scripts\activate

# Activate (macOS/Linux)
source ramanlab_env/bin/activate

# Install dependencies
pip install -r requirements_qt6.txt
```

### Check Installation Status

```bash
# Check all dependencies
python check_dependencies.py

# Check specific packages
python update_dependencies.py --status
```

### What if I Don't Want Jupyter Features?

**No problem!** These are **optional** features:

- RamanLab works perfectly without Jupyter packages
- All existing functionality remains unchanged
- You can skip the Jupyter packages and still use RamanLab
- The console features will simply be unavailable

### Compatibility

- **Python**: 3.8+ (3.9+ recommended)
- **Operating Systems**: Windows, macOS, Linux
- **Qt Version**: PySide6 (recommended) or PyQt6
- **Jupyter**: qtconsole 5.4.0+, jupyter-client 7.0.0+, ipykernel 6.0.0+

### Benefits of the New Console

1. **Interactive Analysis**: Test analysis methods without restarting RamanLab
2. **Custom Scripts**: Run your own analysis scripts directly
3. **Data Exploration**: Explore loaded data interactively
4. **Rapid Prototyping**: Develop new analysis methods quickly
5. **Educational**: Great for learning Raman spectroscopy and Python
6. **Advanced Users**: Full Python power within RamanLab

### Examples

#### Basic Usage
```python
# Get current spectrum
spectrum = get_current_spectrum()
print(f"Spectrum has {len(spectrum.wavenumber)} points")

# Basic plot
plt.figure(figsize=(10, 6))
plt.plot(spectrum.wavenumber, spectrum.intensity)
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Intensity')
plt.title('Current Spectrum')
plt.show()
```

#### Advanced Analysis
```python
# Custom peak detection
from scipy.signal import find_peaks
peaks, _ = find_peaks(spectrum.intensity, height=0.1)
peak_positions = spectrum.wavenumber[peaks]
print(f"Found peaks at: {peak_positions}")

# Custom baseline correction
from scipy.signal import savgol_filter
baseline = savgol_filter(spectrum.intensity, 51, 3)
corrected = spectrum.intensity - baseline
```

### Support

- **Documentation**: Check `docs/` directory for detailed guides
- **Issues**: Report problems on the GitHub repository
- **Dependencies**: Run `check_dependencies.py` for diagnostics

### Next Steps

1. **Update your dependencies** using the methods above
2. **Test the installation** with `python check_dependencies.py`
3. **Try the console** with `python launch_jupyter_console.py`
4. **Explore the features** in RamanLab's Advanced tab

Happy analyzing! 🔬📊 