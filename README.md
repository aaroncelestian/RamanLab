# RamanLab - Advanced Raman Spectrum Analysis Tool

RamanLab is a comprehensive cross-platform desktop application for analyzing and identifying Raman spectra. Built with Qt6, it provides a modern, user-friendly interface for importing, processing, and matching Raman spectra against databases of known materials with advanced machine learning capabilities.

**Current Version:** 1.0.0  
**Release Date:** 2025-01-26  
**Framework:** Qt6 (PySide6/PyQt6)  
**Platform Support:** Windows, macOS, Linux

## 🚀 Quick Start

Get RamanLab running in 5 minutes:

```bash
# 1. Create and activate conda environment
conda create -n ramanlab python=3.9
conda activate ramanlab

# 2. Clone repository
git clone https://github.com/aaroncelestian/RamanLab.git
cd RamanLab

# 3. Install dependencies
pip install -r requirements_qt6.txt

# 4. Check dependencies (recommended)
python check_dependencies.py

# 5. Launch RamanLab
python launch_ramanlab.py

# 6. Install desktop shortcut (optional)
python install_desktop_icon.py
```

**# 6. Download the database and metadata correlation file that I made from:**
- Find the pre-compiled Raman datases here: 10.6084/m9.figshare.28928345
- https://figshare.com/s/220f7de3c17172dbaae6 (Compiled RRUFF and SLOPP/SLOPP-E)
- https://figshare.com/s/b55e31e89743d246895b (Compiled database of Raman calculated mdoes from WURM.info)
- Place these files in the RamanLab directory.

*Note: Always use `python` (not `python3`) when in the conda environment.*

---

## ✨ Features

### 🔬 Core Spectrum Analysis
- **Import and Visualize:** Support for CSV, TXT, and other common Raman data formats
- **Background Subtraction:** Manual and automated background modeling with interactive refinement
- **Peak Detection:** Algorithmic peak identification with user override capabilities
- **Interactive Plotting:** Advanced zooming, panning, and spectral comparison tools
- **Publication Quality:** Export high-resolution plots and comprehensive analysis reports

### 🗄️ Advanced Database Management
- **Comprehensive Storage:** Organize Raman spectra with rich metadata and classifications
- **Batch Operations:** Import multiple spectra with automated processing
- **Advanced Search:** Multi-criteria filtering with Hey classification and element limiting
- **Database Browser:** Dedicated interface (`launch_raman_database_browser.py`) with:
  - Interactive spectrum visualization with peak analysis
  - Comprehensive metadata viewing and editing
  - Export capabilities for individual spectra
  - Database statistics and classification summaries

### 🎯 Intelligent Matching & Identification
- **Multiple Algorithms:**
  - Correlation-based matching
  - Peak-based matching (position and intensity)
  - Hybrid correlation + peak analysis
  - Machine learning-based matching with dynamic time warping
- **Advanced Filtering:** Peak regions, Hey index classification, elemental composition
- **Confidence Scoring:** Statistical assessment of match quality
- **Mixed Mineral Analysis:** Unique capability to identify multiple minerals in complex spectra
- **Molecular Vibrational Groups:** Experimental heatmap analysis with chemical scoring

### 🔧 Professional Peak Fitting
- **Advanced Background Models:**
  - Manual interactive modeling
  - AI-suggested models with user selection
  - Real-time background refinement
- **Peak Detection:** Smart algorithmic identification with manual override
- **Peak Models:**
  - Gaussian, Lorentzian, Pseudo-Voigt, Asymmetric Voigt
  - No region definition required
  - Optional spectral smoothing
- **Professional Reports:** Publication-quality graphics and comprehensive fitting statistics
- **Database Integration:** Export fitted peaks directly to mineral Raman modes database

### ⚡ Batch Processing & Analysis
- **High-Volume Processing:** Handle unlimited spectra (system memory dependent)
- **Automated Background:** Intelligent background refinement across datasets
- **Selective Region Fitting:** User-defined regions for optimized processing speed
- **Statistical Analysis:** 95% confidence intervals (superior to traditional error bars)
- **Comprehensive Reporting:** Batch analysis summaries with statistical visualizations

### 🗺️ 2D Raman Map Analysis
- **Map Import:** Directory-based import for 2D Raman mapping datasets
- **Heatmap Generation:** 
  - Integrated intensity maps
  - Template coefficient visualization
  - Component distribution analysis
- **Template Analysis:**
  - Multiple fitting methods (NNLS, LSQ)
  - Percentage contribution calculations
  - Interactive template visibility controls
  - Export template analysis results
- **Data Quality:** Automated cosmic ray filtering
- **Machine Learning:** PCA, NMF, and Random Forest classification
- **Publication Export:** High-quality visualizations and analysis summaries

### 📊 Advanced Group Analysis
- **Hierarchical Clustering:** Sophisticated grouping of Raman spectra collections
- **Flexible Import:**
  - Folder-based import
  - Database import with advanced filtering
- **Visualization Methods:**
  - PCA (Principal Component Analysis)
  - t-SNE (t-Distributed Stochastic Neighbor Embedding)
  - UMAP (Uniform Manifold Approximation and Projection)
- **Interactive Exploration:**
  - Dendrogram visualization
  - Heatmap analysis
  - Scatter plots with cluster highlighting
  - Individual cluster examination
- **Cluster Refinement:**
  - K-means and Spectral Clustering splitting
  - Interactive cluster merging
  - Undo functionality for cluster editing
- **Professional Export:** Statistics, representative spectra, and publication graphics

### 🔍 Raman Polarization Analysis
- **Polarized Spectra:** Import and analyze orientation-dependent Raman data
- **Interactive Peak Fitting:** Specialized tools for polarized spectral analysis
- **Crystal Orientation Analysis:**
  - Calculate orientation-dependent intensities
  - Optimize crystal orientation to match experimental data
  - Support for all crystal symmetries and Raman tensor characters
- **CIF Integration:**
  - Parse crystallographic information files
  - Extract symmetry and atomic positions
  - Calculate Raman tensors from crystal structure
  - Optional pymatgen integration for advanced analysis
- **3D Tensor Visualization:**
  - Interactive 3D tensor ellipsoids
  - Principal axes exploration
  - Crystal shape visualization by point group
  - Customizable display settings
  - Publication-quality 3D exports

### 🧠 Machine Learning & AI
- **Classification Models:** Random Forest and other ML algorithms for automated identification
- **Dimensionality Reduction:** PCA, NMF, t-SNE, UMAP for data exploration
- **Template Matching:** AI-enhanced template fitting with confidence scoring
- **Class Flip Detection:** Intelligent detection of misclassified spectra
- **Model Management:** Save, load, and retrain classification models

### 🏗️ Advanced Architecture
- **Modern Qt6 Framework:** Cross-platform compatibility with native performance
- **Modular Design:** Extensible architecture for custom analysis modules
- **Multi-Threading:** Background processing for responsive user interface
- **Memory Management:** Efficient handling of large datasets
- **Session Management:** Save and restore complete analysis sessions
- **Crash Recovery:** Automatic session backup and recovery

---

## 🛠️ Installation

### Recommended: Anaconda + Modern IDE

For the best experience, we recommend using [Anaconda](https://www.anaconda.com/products/distribution) with a modern Python IDE:

**Steps:**

1. **Install Anaconda:**
   - Download from [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

2. **Create Environment:**
   ```bash
   conda create -n ramanlab python=3.9
   conda activate ramanlab
   ```

3. **Clone Repository:**
   ```bash
   git clone https://github.com/aaroncelestian/RamanLab.git
   cd RamanLab
   ```

4. **Install Dependencies:**
   ```bash
   pip install -r requirements_qt6.txt
   ```

5. **Verify Installation:**
   ```bash
   python check_dependencies.py
   ```

6. **Download Database Files:**
   - Main Database: [10.6084/m9.figshare.28928345](https://figshare.com/s/220f7de3c17172dbaae6)
   - Calculated Modes: [WURM Database](https://figshare.com/s/b55e31e89743d246895b)
   - Place files in the RamanLab directory

7. **Launch Application:**
   ```bash
   python launch_ramanlab.py
   ```

### Alternative Installation Methods

#### Quick Setup (pip)
```bash
git clone https://github.com/aaroncelestian/RamanLab.git
cd RamanLab
pip install -r requirements_qt6.txt
python launch_ramanlab.py
```

#### Development Setup
```bash
# Clone and setup virtual environment
git clone https://github.com/aaroncelestian/RamanLab.git
cd RamanLab
python -m venv ramanlab_env
source ramanlab_env/bin/activate  # Windows: ramanlab_env\Scripts\activate
pip install -r requirements_qt6.txt
python check_dependencies.py
```

---

## 🖥️ Desktop Integration

### Automatic Desktop Icon Installation

```bash
python install_desktop_icon.py
```

Creates platform-specific shortcuts:
- **Windows:** Desktop shortcut with custom icon
- **macOS:** Application bundle in ~/Applications  
- **Linux:** Desktop entry and applications menu integration

### Uninstall Desktop Integration
```bash
python install_desktop_icon.py --uninstall
```

---

## 📋 System Requirements

### Minimum Requirements
- **Python:** 3.8+ (3.9+ recommended)
- **RAM:** 4GB (8GB+ for large datasets)
- **Storage:** 2GB free space
- **OS:** Windows 10+, macOS 10.14+, Linux (modern distributions)

### Recommended Configuration
- **Python:** 3.9 or 3.10
- **RAM:** 8GB+ for complex analyses
- **Storage:** 5GB+ (including databases)
- **GPU:** Optional, beneficial for machine learning tasks

---

## 📦 Dependencies

### Core Requirements (Always Required)
```
PySide6>=6.5.0              # Modern Qt6 GUI framework
numpy>=1.21.0               # Numerical computations
matplotlib>=3.5.0           # Plotting and visualization
scipy>=1.7.0                # Scientific computing
pandas>=1.3.0               # Data manipulation
```

### Analysis & Visualization
```
seaborn>=0.11.0             # Statistical visualization
scikit-learn>=1.0.0         # Machine learning
fastdtw>=0.3.4              # Dynamic time warping
tqdm>=4.60.0                # Progress indicators
```

### Data Processing & Export
```
openpyxl>=3.0.0             # Excel file support
pillow>=8.0.0               # Image processing
dask>=2021.0.0              # Parallel computing
psutil>=5.8.0               # System utilities
```

### Optional Advanced Features
```
pymatgen>=2022.0.0          # Advanced crystallography
reportlab>=3.5.0            # PDF report generation
tensorflow>=2.12.0          # Deep learning (optional)
umap-learn                  # UMAP visualization
pyinstaller>=5.0.0          # Standalone executables
```

---

## 🚀 Usage Guide

### Basic Workflow

1. **Import Spectra:** Use File menu or drag-and-drop
2. **Process Data:** Apply background subtraction and peak detection
3. **Database Search:** Match against reference database
4. **Analysis:** Perform peak fitting or advanced analysis
5. **Export:** Save results as reports, images, or data files

### Advanced Features

#### 2D Map Analysis
```bash
# Launch standalone map analyzer
python map_analysis_2d/main.py
```

#### Polarization Analysis
```bash
# Launch polarization analyzer
python launch_polarization_analyzer.py
```

#### Group Analysis
```bash
# Launch cluster analysis
python raman_cluster_analysis_qt6.py
```

#### Database Management
```bash
# Browse mineral database
python launch_raman_database_browser.py
```

---

## 📁 Project Structure

```
RamanLab/
├── core/                    # Core analysis modules
│   ├── database.py         # Database management
│   ├── spectrum.py         # Spectrum processing
│   ├── peak_fitting.py     # Peak fitting algorithms
│   └── state_management/   # Session management
├── map_analysis_2d/        # 2D mapping analysis
├── polarization_ui/        # Polarization analysis
├── ml_raman_map/          # Machine learning modules
├── battery_strain_analysis/ # Specialized strain tools
├── docs/                   # Comprehensive documentation
├── main_qt6.py            # Main application entry
├── launch_ramanlab.py     # Application launcher
├── check_dependencies.py  # Dependency checker
├── requirements_qt6.txt   # Dependencies list
└── version.py             # Version information
```

---

## 🔧 Troubleshooting

### Common Issues

1. **Qt6 Import Errors:**
   ```bash
   pip install --upgrade PySide6
   ```

2. **Missing Dependencies:**
   ```bash
   python check_dependencies.py
   pip install -r requirements_qt6.txt
   ```

3. **Database File Missing:**
   - Download from provided links
   - Place in main RamanLab directory

4. **Memory Issues with Large Datasets:**
   - Increase system memory
   - Use batch processing for large analyses
   - Enable data streaming for map analysis

### Getting Help

- Check `docs/` directory for detailed guides
- Run `python check_dependencies.py` for diagnostics
- Review error logs in application console

---

## 🤝 Contributing

We welcome contributions! Please see our development documentation in the `docs/` directory for guidelines on:
- Code style and standards
- Testing procedures
- Documentation requirements
- Feature request process

---

## 📜 License

MIT License - see LICENSE file for details

---

## 👨‍🔬 Author

**Aaron Celestian, Ph.D.**  
Curator of Mineral Sciences  
Natural History Museum of Los Angeles County

---

## 🙏 Acknowledgments

- **RRUFF Database** (www.rruff.info) - Reference mineral spectra
- **SLOPP/SLOPP-E** - Plastic spectral libraries
- **WURM Database** - DFT calculations and theoretical Raman spectra  
- **Scientific Community** - Raman spectroscopy research and development
- **Qt Project** - Modern cross-platform GUI framework
- **Python Scientific Stack** - NumPy, SciPy, Matplotlib, and scikit-learn communities

---

*For the latest updates and detailed documentation, visit our [documentation directory](./docs/) or check the individual module README files.*
