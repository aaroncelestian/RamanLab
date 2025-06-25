# RamanLab - Advanced Raman Spectrum Analysis Suite

**Author:** Aaron J. Celestian, Ph.D.  
**Curator of Mineral Sciences**  
**Natural History Museum of Los Angeles County**

RamanLab is a comprehensive, cross-platform desktop application for analyzing and identifying Raman spectra. Built with PySide6, it provides a modern, professional interface for importing, processing, and matching Raman spectra against databases of known materials with advanced machine learning capabilities and specialized analysis tools.

**Current Version:** 1.0.2  
**Release Date:** 2025-06-22  
**Framework:** Qt6 (PySide6)  
**Platform Support:** Windows, macOS, Linux  
**Python Requirements:** 3.8+ (3.9+ recommended)

---

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

**Download Essential Databases:**
- **Pre-compiled Raman Databases**
 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15717961.svg)](https://doi.org/10.5281/zenodo.15717961)

- Place these files in the RamanLab directory.

*Note: Always use `python` (not `python3`) when in the conda environment.*

---

## ✨ Core Features

### 🔬 **Comprehensive Spectrum Analysis**
- **Multi-Format Import:** CSV, TXT, DAT, and other common Raman data formats
- **Advanced Background Subtraction:** Manual and automated modeling with interactive refinement
- **Smart Peak Detection:** Algorithmic identification with user override capabilities
- **Interactive Visualization:** Advanced zooming, panning, and spectral comparison tools
- **Publication Quality Exports:** High-resolution plots and comprehensive analysis reports

### 🗄️ **Advanced Database Management**
- **Comprehensive Storage:** Organize Raman spectra with rich metadata and classifications
- **Batch Operations:** Import multiple spectra with automated processing
- **Advanced Search:** Multi-criteria filtering with Hey-Celestian classification and element limiting
- **Dedicated Database Browser:** (`launch_raman_database_browser.py`) featuring:
  - Interactive spectrum visualization with peak analysis
  - Comprehensive metadata viewing and editing
  - Export capabilities for individual spectra
  - Database statistics and classification summaries

### 🎯 **Intelligent Matching & Identification**
- **Multiple Algorithms:**
  - Correlation-based matching
  - Peak-based matching (position and intensity)
  - Hybrid correlation + peak analysis
  - Machine learning-based matching with dynamic time warping
- **Advanced Filtering:** Peak regions, Hey-Celestian classification, elemental composition
- **Confidence Scoring:** Statistical assessment of match quality
- **Mixed Mineral Analysis:** Unique capability to identify multiple minerals in complex spectra
- **Molecular Vibrational Groups:** Experimental heatmap analysis with chemical scoring

---

## 🏆 **New Hey-Celestian Classification System**

### **Vibrational Mode-Based Mineral Classification**

*In development by Aaron J. Celestian & RamanLab Development Team*

The **Hey-Celestian Classification System** is a spectroscopy based classification system for mineral specifically designed for Raman spectroscopy. Unlike traditional systems that organize minerals by chemical composition, this system organizes them by their **dominant vibrational signatures**.

#### **15 Main Vibrational Groups:**
1. **Framework Modes - Tetrahedral Networks** (Quartz, Feldspar, Zeolites)
2. **Framework Modes - Octahedral Networks** (Rutile, Anatase, Spinel)
3. **Characteristic Vibrational Mode - Carbonate Groups** (Calcite, Aragonite)
4. **Characteristic Vibrational Mode - Sulfate Groups** (Gypsum, Anhydrite, Barite)
5. **Characteristic Vibrational Mode - Phosphate Groups** (Apatite, Vivianite)
6. **Chain Modes - Single Chain Silicates** (Pyroxenes, Wollastonite)
7. **Chain Modes - Double Chain Silicates** (Amphiboles, Actinolite)
8. **Ring Modes - Cyclosilicates** (Tourmaline, Beryl, Cordierite)
9. **Layer Modes - Sheet Silicates** (Micas, Clays, Talc, Serpentine)
10. **Layer Modes - Non-Silicate Layers** (Graphite, Molybdenite, Brucite)
11. **Simple Oxides** (Hematite, Magnetite, Corundum)
12. **Complex Oxides** (Spinels, Chromites, Garnets, Perovskites)
13. **Hydroxides** (Goethite, Lepidocrocite, Diaspore)
14. **Organic Groups** (Abelsonite, Organic minerals)
15. **Mixed Modes** (Epidote, Vesuvianite, Complex structures)

These 15 groups will have branching subgroups of a hierarchical, multi-dimensional classification scheme that maintains the elegant simplicity of vibrational mode organization while incorporating the chemical, structural, and environmental complexity that characterizes real mineral systems. This will involve a tree-like structure where the 15 primary vibrational groups serve as main branches, with increasingly specific sub-classifications based on composition, structure, and environmental factors.
1. Works with solid solutions
2. Polytipic variations
3. Polymorph catagories
4. Crystal orientation effects

You will notice that a lot of groundwork is already implemented in RamanLab to build out this new classifation system.  Once the system is implemented, new quantitaive enhancements could be realized, such as, mixing ratios, crystallinity indices, stress/strain, weathering/alteration pathways, hydration effects, phase diagram integration, metastability, and aid in additive learning systems for future AI predictive modeling.

#### **Advantages:**
- **Predictive Analysis:** Expected peak positions and vibrational characteristics
- **Enhanced Database Searching:** Filter by vibrational mode families
- **Educational Value:** Connect structural features to spectral signatures
- **Analysis Strategy Guidance:** Optimal regions and expected interferences

---

## 🔧 **Peak Fitting & Batch Processing**

### **Advanced Peak Fitting:**
- **Smart Background Models:** AI-suggested models with user selection
- **Multiple Peak Models:** Gaussian, Lorentzian, Pseudo-Voigt, Asymmetric Voigt
- **No Region Definition Required:** Automatic peak detection across full spectrum
- **Reports:** Publication-quality graphics with comprehensive fitting statistics

### **High-Volume Batch Processing:**
- **Unlimited Spectra:** Handle large datasets (system memory dependent)
- **Automated Background Refinement:** Intelligent background modeling across datasets
- **Statistical Analysis:** 95% confidence intervals (superior to traditional error bars)
- **Selective Region Fitting:** User-defined regions for optimized processing speed

---

## 🗺️ **2D Raman Map Analysis**

### **Comprehensive Mapping Capabilities:**
- **Directory-Based Import:** Seamless 2D Raman mapping dataset handling
- **Multiple Visualization Methods:**
  - Integrated intensity heatmaps
  - Template coefficient visualization
  - Component distribution analysis
- **Template Analysis:**
  - Multiple fitting methods (NNLS, LSQ)
  - Percentage contribution calculations
  - Interactive template visibility controls
  - Export template analysis results
- **Data Quality Control:** Automated cosmic ray filtering
- **Machine Learning Integration:** PCA, NMF, and Random Forest classification

---

## ⚡ **Battery & Materials Analysis**

### **LiMn2O4 Battery Strain Analysis System:**
*Specialized module for battery materials research*

- **Chemical Strain Analysis:** Track H/Li exchange effects in battery materials
- **Jahn-Teller Distortion Monitoring:** Quantify Mn³⁺ formation and structural distortions
- **Time Series Processing:** Handle time-resolved Raman spectroscopy data
- **Phase Transition Detection:** Identify structural phase changes during cycling
- **Comprehensive Visualization:** 3D strain tensor plots and evolution tracking

#### **Spinel-Specific Features:**
- **Mode Definitions:** A1g, Eg, T2g breathing and framework modes
- **Composition Tracking:** Li content and chemical disorder analysis
- **Electrochemical Synchronization:** Correlate with battery state-of-charge
- **Degradation Analysis:** Monitor cycling-induced structural changes

#### More specialized modules to come
---

## 📊 **Advanced Group & Cluster Analysis**

### **Hierarchical Clustering:**
- **Flexible Import:** Folder-based or database import with advanced filtering
- **Multiple Visualization Methods:**
  - PCA (Principal Component Analysis)
  - UMAP (Uniform Manifold Approximation and Projection)
- **Interactive Exploration:**
  - Dendrogram visualization
  - Heatmap analysis
  - Scatter plots with cluster highlighting
- **Cluster Refinement:** K-means and Spectral Clustering with undo functionality

---

## 🔍 **Comprehensive Raman Polarization Analysis**

### **Crystal Orientation & Tensor Analysis:**
- **Polarized Spectra Analysis:** Import and analyze orientation-dependent data
- **Crystal Orientation Optimization:** Calculate and optimize orientation to match experimental data
- **All Crystal Symmetries:** Support for complete range of crystal systems
- **CIF Integration:**
  - Parse crystallographic information files
  - Extract symmetry and atomic positions
  - Calculate Raman tensors from crystal structure
  - Optional pymatgen integration

### **3D Tensor Visualization:**
- **Interactive 3D Ellipsoids:** Real-time tensor visualization
- **Principal Axes Exploration:** Understand tensor orientation
- **Crystal Shape Visualization:** Point group-specific shapes
- **Publication-Quality 3D Exports:** Professional graphics for publications

---

## 🧠 **Machine Learning & AI Integration**

### **Classification & Analysis:**
- **Random Forest Algorithms:** Automated mineral identification
- **Dimensionality Reduction:** PCA, NMF, t-SNE, UMAP for data exploration
- **Template Matching:** AI-enhanced template fitting with confidence scoring
- **Class Flip Detection:** Intelligent detection of misclassified spectra
- **Model Management:** Save, load, and retrain classification models

### **Advanced Features:**
- **Dynamic Time Warping:** Handle spectral shifting and distortions
- **Ensemble Methods:** Combine multiple algorithms for robust identification
- **Uncertainty Quantification:** Statistical confidence in ML predictions

---

## 🏗️ **Modern Architecture & Session Management**

### **Comprehensive State Management:**
*session save/restore system in the near future*

- **Complete Session Recovery:** Save and restore entire application state
- **Window Layout Preservation:** Remember panel positions, zoom levels, and preferences
- **Data State Management:** Loaded spectra, analysis results, template libraries
- **Auto-Save & Crash Recovery:** Automatic session backup with crash recovery
- **Session Sharing:** Export and share complete analysis sessions

### **Technical Excellence:**
- **Modern Qt6 Framework:** Cross-platform native performance
- **Modular Architecture:** Extensible design for custom analysis modules
- **Multi-Threading:** Background processing for responsive interface
- **Efficient Memory Management:** Handle large datasets without performance loss
- **Professional Updates:** Built-in update checker with version management

---

## 📦 **Installation & Setup**

### **Recommended: Anaconda Setup**

```bash
# 1. Install Anaconda from https://www.anaconda.com/products/distribution

# 2. Create Environment
conda create -n ramanlab python=3.9
conda activate ramanlab

# 3. Clone Repository
git clone https://github.com/aaroncelestian/RamanLab.git
cd RamanLab

# 4. Install Dependencies
pip install -r requirements_qt6.txt

# 5. Verify Installation
python check_dependencies.py

# 6. Launch Application
python launch_ramanlab.py
```

### **Desktop Integration**

```bash
# Install desktop shortcuts and application integration
python install_desktop_icon.py

# Creates platform-specific integration:
# Windows: Desktop shortcut with custom icon
# macOS: Application bundle in ~/Applications
# Linux: Desktop entry and applications menu integration
```

---

## 🖥️ **Application Suite**

### **Core Applications:**

```bash
# Main RamanLab Application
python launch_ramanlab.py

# Database Browser & Management
python launch_raman_database_browser.py

# 2D Map Analysis (Standalone)
python map_analysis_2d/main.py

# Polarization Analyzer
python launch_orientation_optimizer.py

# Cluster Analysis
python raman_cluster_analysis_qt6.py

# Peak Fitting (Standalone)
python peak_fitting_qt6.py

# Batch Peak Fitting
python batch_peak_fitting_qt6.py

# Line Scan Analysis
python launch_line_scan_splitter.py
```

---

## 📋 **System Requirements**

### **Minimum Requirements:**
- **Python:** 3.8+ (3.9+ recommended)
- **RAM:** 4GB (8GB+ for large datasets)
- **Storage:** 2GB free space (5GB+ including databases)
- **OS:** Windows 10+, macOS 10.14+, Linux (modern distributions)

### **Recommended Configuration:**
- **Python:** 3.9 or 3.10
- **RAM:** 16GB+ for complex analyses and large maps
- **Storage:** 10GB+ (including databases and results)
- **GPU:** Optional, beneficial for machine learning tasks

---

## 📁 **Project Structure**

```
RamanLab/
├── main_qt6.py                    # Main application entry point
├── launch_ramanlab.py             # Primary application launcher
├── core/                          # Core analysis modules
│   ├── database.py               # Database management
│   ├── spectrum.py               # Spectrum processing
│   ├── peak_fitting.py           # Peak fitting algorithms
│   └── state_management/         # Session management system
├── map_analysis_2d/              # 2D mapping analysis suite
├── polarization_ui/              # Polarization analysis tools
├── battery_strain_analysis/      # Battery materials analysis
├── Hey_class/                    # Hey-Celestian classification system
├── ml_raman_map/                 # Machine learning modules
├── database_browser_qt6.py       # Database browser application
├── peak_fitting_qt6.py           # Standalone peak fitting
├── batch_peak_fitting_qt6.py     # Batch processing tools
├── raman_cluster_analysis_qt6.py # Cluster analysis application
├── docs/                         # Comprehensive documentation
├── requirements_qt6.txt          # Dependencies list
├── check_dependencies.py         # Dependency verification
└── version.py                    # Version information
```

---

## 🔧 **Troubleshooting**

### **Common Issues & Solutions:**

1. **Qt6 Import Errors:**
   ```bash
   pip install --upgrade PySide6
   ```

2. **Missing Dependencies:**
   ```bash
   python check_dependencies.py
   pip install -r requirements_qt6.txt
   ```

3. **Database Files Missing:**
   - Download from provided figshare links
   - Place in main RamanLab directory
   - Verify with database browser application

4. **Memory Issues:**
   - Increase system memory for large datasets
   - Use batch processing for extensive analyses
   - Enable data streaming for map analysis

5. **Platform-Specific Issues:**
   - Check `docs/` directory for platform guides
   - Review cross-platform utilities in `cross_platform_utils.py`

---

## 🚀 **Usage Examples**

### **Basic Workflow:**
1. **Import Spectra:** File menu or drag-and-drop
2. **Process Data:** Background subtraction and peak detection
3. **Database Search:** Match against reference database using Hey-Celestian classification
4. **Advanced Analysis:** Peak fitting, strain analysis, or polarization analysis
5. **Export Results:** Save as reports, images, or complete sessions

### **Advanced Workflows:**

#### **Material Identification Pipeline:**
```bash
# 1. Launch main application
python launch_ramanlab.py

# 2. Import spectrum and apply Hey-Celestian classification
# 3. Use predicted vibrational modes for targeted analysis
# 4. Validate with database matching
# 5. Export comprehensive identification report
```

#### **Battery Research Workflow:**
```bash
# 1. Time-series strain analysis
cd battery_strain_analysis
python demo_limn2o4_analysis.py

# 2. Results include:
#    - Strain tensor evolution
#    - Composition tracking
#    - Phase transition detection
#    - 3D visualization
```

---

## 🤝 **Contributing**

RamanLab welcomes contributions from the scientific community. Areas of particular interest:

- **New Classification Systems:** Extend Hey-Celestian or develop specialized systems
- **Analysis Algorithms:** Novel peak fitting, background subtraction, or identification methods
- **Database Expansion:** Additional reference spectra and metadata
- **Specialized Modules:** Industry or research-specific analysis tools
- **Documentation:** User guides, tutorials, and scientific validation

---

## 📚 **Documentation**

Comprehensive documentation available in the `docs/` directory:

- **User Manual:** Complete application guide
- **API Documentation:** Developer reference
- **Scientific Methods:** Algorithmic descriptions and validation
- **Installation Guides:** Platform-specific setup instructions
- **Tutorial Collection:** Step-by-step analysis examples

---

## 📜 **License**

MIT License - see LICENSE file for details

---

## 🙏 **Acknowledgments**

### **Scientific Databases:**
- **RRUFF Database** (www.rruff.info) - Comprehensive mineral reference spectra
- **SLOPP/SLOPP-E** - Plastic and polymer spectral libraries
- **WURM Database** - DFT calculations and theoretical Raman spectra

### **Development Framework:**
- **Qt Project** - Modern cross-platform GUI framework (PySide6)
- **Python Scientific Stack** - NumPy, SciPy, Matplotlib, scikit-learn, and pandas
- **Scientific Community** - Raman spectroscopy research and methodology development

### **Research Collaborations:**
- **Natural History Museum of Los Angeles County** - Institutional support
- **International Raman Community** - Feedback, validation, and collaborative development

---

## 👨‍🔬 **About the Author**

**Aaron J. Celestian, Ph.D.**  
Curator of Mineral Sciences  
Natural History Museum of Los Angeles County  

Dr. Celestian specializes in mineral physics and crystallography with extensive experience in vibrational spectroscopy applications. His research focuses on the relationship between crystal structure and physical properties, making him uniquely qualified to develop the revolutionary Hey-Celestian classification system that bridges traditional mineralogy with modern spectroscopic analysis.

---

*For the latest updates, detailed documentation, and scientific publications related to RamanLab, visit our [documentation directory](./docs/) or check the individual module README files.*

**RamanLab - Advancing Raman Spectroscopy Through Innovation**
