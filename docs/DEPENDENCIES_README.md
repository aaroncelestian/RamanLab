# RamanLab Dependencies Guide - Complete Trilogy Edition

## Overview

This guide covers all dependencies and installation instructions for the complete RamanLab crystal orientation optimization trilogy:

- **🚀 Stage 1**: Enhanced Individual Peak Optimization
- **🧠 Stage 2**: Probabilistic Bayesian Framework  
- **🌟 Stage 3**: Advanced Multi-Objective Bayesian Optimization

## Quick Start

### 1. Check Current Status
```bash
python check_dependencies.py
```

### 2. Install Dependencies
```bash
# Option A: Use requirements.txt
pip install -r requirements.txt

# Option B: Use installation script
python install_trilogy_dependencies.py --complete

# Option C: Stage-specific installation
python install_trilogy_dependencies.py --advanced  # Stages 2 & 3
```

### 3. Verify Installation
```bash
python install_trilogy_dependencies.py --check
```

## Dependency Categories

### Core Dependencies (Required for All Stages)
```
numpy>=1.16.0              # Numerical computations
matplotlib>=3.0.0           # Plotting and visualization  
scipy>=1.2.0                # Scientific computing
pandas>=0.25.0              # Data manipulation
seaborn>=0.11.0             # Statistical visualization
pillow>=8.0.0               # Image processing (imported as PIL)
mplcursors>=0.5.0           # Interactive matplotlib cursors
reportlab>=3.5.0            # PDF export functionality
openpyxl>=3.0.0             # Excel file support
fastdtw>=0.3.4              # Fast Dynamic Time Warping
```

### Advanced Dependencies (Stages 2 & 3)
```
scikit-learn>=0.21.0        # Machine learning, Gaussian Processes, ensemble methods
emcee>=3.0.0                # MCMC sampling for Bayesian analysis
```

### Optional Dependencies (Enhanced Features)
```
tensorflow>=2.12.0          # Deep learning framework
keras>=2.12.0               # High-level neural networks API
pymatgen>=2022.0.0          # Materials analysis and crystallography
pyinstaller>=5.0.0          # Creating standalone executables
```

### GUI Framework
```
tkinter                     # GUI framework (bundled with Python)
```

## Stage-Specific Requirements

### 🚀 Stage 1: Enhanced Individual Peak Optimization
**Dependencies**: Core packages only
- ✅ **Fully functional** with core dependencies
- **Features**: Multi-start global optimization, uncertainty-weighted objectives, character-based assignment
- **Performance**: 30-60 seconds, ±1-5° accuracy

### 🧠 Stage 2: Probabilistic Bayesian Framework  
**Dependencies**: Core packages + `emcee` + `scikit-learn`
- ⚠️ **Reduced functionality** without `emcee` (no MCMC sampling)
- ⚠️ **Reduced functionality** without `scikit-learn` (no clustering)
- **Features**: Bayesian parameter estimation, probabilistic peak assignment, model comparison
- **Performance**: 1-3 minutes, ±0.5-3° accuracy

### 🌟 Stage 3: Advanced Multi-Objective Bayesian Optimization
**Dependencies**: Core packages + `scikit-learn` + `emcee`
- ❌ **Critical**: Requires `scikit-learn` for Gaussian Processes
- ⚠️ **Reduced functionality** without `emcee` (no MCMC sampling)
- **Features**: Gaussian Process surrogate models, multi-objective optimization, ensemble methods
- **Performance**: 2-5 minutes, ±0.5-2° accuracy

## Installation Methods

### Method 1: Requirements File (Recommended)
```bash
# Install all dependencies
pip install -r requirements.txt

# For virtual environment (recommended)
python -m venv clarityspectra_env
source clarityspectra_env/bin/activate  # Windows: clarityspectra_env\Scripts\activate
pip install -r requirements.txt
```

### Method 2: Installation Script
```bash
# Interactive installation
python install_trilogy_dependencies.py

# Core dependencies only (Stage 1)
python install_trilogy_dependencies.py --core

# Advanced dependencies (Stages 2 & 3)
python install_trilogy_dependencies.py --advanced

# Complete installation (all features)
python install_trilogy_dependencies.py --complete

# Check current status
python install_trilogy_dependencies.py --check
```

### Method 3: Manual Installation
```bash
# Core functionality
pip install numpy matplotlib scipy pandas seaborn pillow mplcursors reportlab openpyxl fastdtw

# Advanced functionality (Stages 2 & 3)
pip install scikit-learn emcee

# Optional features
pip install tensorflow keras pymatgen pyinstaller
```

## Platform-Specific Instructions

### Windows
```bash
# Python should include tkinter
pip install -r requirements.txt
```

### macOS
```bash
# Install tkinter if needed
brew install python-tk

# Install dependencies
pip install -r requirements.txt
```

### Linux (Debian/Ubuntu)
```bash
# Install tkinter
sudo apt-get install python3-tk

# Install dependencies
pip install -r requirements.txt
```

### Linux (Fedora/RHEL)
```bash
# Install tkinter
sudo dnf install python3-tkinter

# Install dependencies
pip install -r requirements.txt
```

## Troubleshooting

### Common Issues

#### 1. tkinter Not Found
```
ImportError: No module named 'tkinter'
```
**Solution**: Install tkinter according to your platform (see above)

#### 2. scikit-learn Import Error
```
ImportError: No module named 'sklearn'
```
**Solution**: 
```bash
pip install scikit-learn
```

#### 3. emcee Not Available
```
ImportError: No module named 'emcee'
```
**Solution**:
```bash
pip install emcee
```

#### 4. Pillow Import Error
```
ImportError: No module named 'PIL'
```
**Solution**:
```bash
pip install pillow
```

#### 5. Version Conflicts
```
VersionConflict: package requires version X but you have Y
```
**Solution**:
```bash
pip install --upgrade package_name
```

### Performance Issues

#### Slow Installation
- Use binary packages: `pip install --only-binary=all package_name`
- Update pip: `pip install --upgrade pip`
- Use conda for scientific packages: `conda install numpy scipy matplotlib`

#### Runtime Performance
- Install optimized BLAS libraries
- Use conda-forge packages for better optimization
- Consider using virtual environments to avoid conflicts

## Verification

### Dependency Checker
```bash
python check_dependencies.py
```

**Expected Output for Complete Installation**:
```
📊 Stage Availability Summary:
   🚀 Stage 1 (Enhanced): ✅ AVAILABLE
   🧠 Stage 2 (Probabilistic): ✅ AVAILABLE  
   🌟 Stage 3 (Advanced): ✅ AVAILABLE

🎉 Complete trilogy functionality available!
```

### Feature Availability
```bash
python install_trilogy_dependencies.py --check
```

**Expected Features**:
- ✅ Core Optimization
- ✅ PDF Export
- ✅ MCMC Sampling (Stages 2 & 3)
- ✅ Gaussian Processes (Stage 3)
- ✅ Deep Learning (optional)
- ✅ Advanced Crystallography (optional)
- ✅ Executable Creation (optional)

## Virtual Environment Setup (Recommended)

### Create Environment
```bash
# Create virtual environment
python -m venv clarityspectra_env

# Activate environment
# Windows:
clarityspectra_env\Scripts\activate
# macOS/Linux:
source clarityspectra_env/bin/activate
```

### Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python check_dependencies.py
```

### Deactivate Environment
```bash
deactivate
```

## Development Setup

### For Contributors
```bash
# Clone repository
git clone <repository_url>
cd RamanLab

# Create development environment
python -m venv dev_env
source dev_env/bin/activate  # Windows: dev_env\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8  # Additional dev tools

# Run tests
python test_stage1.py
python test_stage2.py  
python test_stage3.py
```

## Minimum System Requirements

### Python Version
- **Minimum**: Python 3.6
- **Recommended**: Python 3.8+
- **Optimal**: Python 3.9+

### Memory
- **Minimum**: 4 GB RAM
- **Recommended**: 8 GB RAM
- **Optimal**: 16 GB RAM (for large datasets)

### Storage
- **Minimum**: 2 GB free space
- **Recommended**: 5 GB free space (including databases)

### CPU
- **Minimum**: Dual-core processor
- **Recommended**: Quad-core processor
- **Optimal**: 8+ cores (for parallel processing)

## Data Files

### Required Files
```
✅ raman_database.pkl (250 MB) - Core mineral database
✅ RRUFF_Export_with_Hey_Classification.csv (3.1 MB) - RRUFF data
✅ mineral_modes.pkl (6.5 MB) - Mineral mode database
```

### File Locations
- Files should be in the same directory as the main application
- Will be created automatically on first run if missing
- RRUFF file may need manual download from https://rruff.info/

## Support

### Getting Help
1. **Check Dependencies**: Run `python check_dependencies.py`
2. **Read Error Messages**: Most errors include installation instructions
3. **Check Platform**: Ensure platform-specific packages are installed
4. **Update Packages**: Try `pip install --upgrade package_name`
5. **Virtual Environment**: Use isolated environment to avoid conflicts

### Reporting Issues
When reporting dependency issues, include:
- Python version (`python --version`)
- Platform (`python -c "import platform; print(platform.platform())"`)
- Dependency checker output (`python check_dependencies.py`)
- Full error message and traceback

---

**RamanLab Complete Trilogy** - The most sophisticated crystal orientation optimization system available for Raman polarization analysis. 