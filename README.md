# RamanLab - Raman Spectrum Analysis Tool

RamanLab is a comprehensive desktop application for analyzing and identifying Raman spectra. It provides a user-friendly interface for importing, processing, and matching Raman spectra against a database of known materials.

**Current Version:** 1.0.0

## Quick Start

Get RamanLab running in 5 minutes:

```bash
# 1. Create and activate conda environment
conda create -n ramanlab python=3.9
conda activate ramanlab

# 2. Clone repository
git clone https://github.com/aaroncelestian/RamanLab.git
cd RamanLab

# 3. Install dependencies
conda install -c conda-forge PySide6 scipy matplotlib pandas numpy
conda install -c conda-forge chardet tqdm scikit-learn dask seaborn openpyxl fastdtw emcee reportlab

# 4. Launch RamanLab
python launch_ramanlab.py

# 5. Install desktop shortcut (optional)
python install_desktop_icon.py
```

*Note: Always use `python` (not `python3`) when in the conda environment.*

---

## Features

- **Spectrum Analysis**
  - Import and visualize Raman spectra
  - Background subtraction
  - Peak detection and analysis
  - Interactive plotting and zooming
  - Spectral comparison
  - **NEW: Multi-Spectrum Manager Integration**
    - Direct access from File tab for multi-spectrum analysis
    - Comprehensive data playground for comparing multiple spectra
    - Advanced visualization and manipulation tools
    - Session saving and loading capabilities

- **Database Management**
  - Store and organize Raman spectra
  - Edit metadata and classifications
  - Batch import capabilities
  - View and search database contents
  - **NEW: Advanced Raman Database Browser**
    - Dedicated interface for browsing the complete Raman database
    - Advanced search and filtering capabilities
    - Interactive spectrum visualization with peak analysis
    - Metadata viewing and editing with comprehensive field support
    - Export capabilities for individual spectra
    - Database statistics and Hey classification summaries
    - Standalone mode available via `launch_raman_database_browser.py`
  - **Mineral Raman Modes Database**
    - Store peak positions, symmetry characters, and relative intensities
    - Import modes from peak fitting results
    - Build a customizable reference database for minerals
    - Visualize mineral Raman modes
    - **NEW: Enhanced database with improved visualization and search capabilities**

- **Search and Matching**
  - Multiple search algorithms:
    - Correlation-based matching
    - Peak-based matching
    - Correlation + Peak
    - Machine learning-based matching (dynamic time warping)
  - Advanced filtering options (peak regions, Hey index, element limiting)
  - Confidence scoring for matches
  - Experimental: Heatmap of best fit to molecular vibrational groups with chemical scoring
  - Mixed Mineral Analysis - unique way of find mulitple minerals in your Raman data

- **Peak Fitting**
  - Advanced background modeling: 
    - Manual modeling
    - Software suggested models that are user selectable
    - Interactive background modeling
  - Algorithmically identify peaks, or add user selected/deleted
  - Peak modeling:
    - Gaussian, Lorentzian, Pseudo-Voigt, and Asymmetric Voigt
    - No need to define regions for fitting, no need for spectral smoothing (although it is an option)
    - Report generation and export of publication quality graphics
  - Export fitted peaks to the mineral Raman modes database
  - **NEW: Advanced Analysis Tab**
    - Dedicated tab for advanced analysis tools
    - Professional-styled buttons with hover effects
    - Quick access to:
      - Peak Fitting Window
      - Batch Peak Fitting
      - 2D Map Analysis  
      - Raman Group Analysis
      - Hey-Celestian Frequency Analysis
      - Raman Polarization Analysis
      - Specialized strain analysis tools (Stress/Strain and Chemical Strain)
    - Integrated error handling and module availability checking

- **Batch Processing**
  - Import any number of spectra that your system can handle
  - Same background and peak shapes as in Peak Fitting
  - AUTOMATIC background refinement
  - User selective region fitting to speed fitting times
  - Graphics analysis with 95% confidence intervals plotted (better than error bars)

- **2D Raman Map Analysis**
  - Import and visualize 2D Raman mapping data
  - Create heatmaps for integrated intensity, template coefficients, and others
  - Advanced template analysis for component identification:
    - Multiple template fitting methods (NNLS, LSQ)
    - Template coefficient visualization
    - Percentage contribution calculation for visible components
    - Interactive template visibility controls
    - Export template analysis results
  - Cosmic ray filtering for improved data quality
  - Machine learning analysis with PCA, NMF, and Random Forest classification
  - Export analysis results and publication-quality visualizations
  - Class distribution visualization and sample identification

- **Raman Group Analysis**
  - Hierarchical clustering of Raman spectra collections
  - Multiple data import options:
    - Import from folder
    - Import from database with filtering options
  - Advanced visualization methods:
    - PCA (Principal Component Analysis)
    - t-SNE (t-Distributed Stochastic Neighbor Embedding)
    - UMAP (Uniform Manifold Approximation and Projection) when available
  - Interactive data exploration:
    - Dendrogram visualization of hierarchical clustering
    - Heatmap visualization of spectral features
    - Scatter plots with cluster highlighting
    - Selection and examination of specific clusters
  - Cluster refinement capabilities:
    - Split clusters using K-means or Spectral Clustering
    - Merge selected clusters
    - Interactive cluster editing with undo functionality
  - Analysis and export:
    - Cluster statistics and representative spectra
    - Publication-quality visualization export
    - Data export for further analysis
  - **NEW: Enhanced group analysis with improved visualization and comparison tools**

- **NEW: Raman Polarization Analysis**
  - Import and analyze polarized Raman spectra
  - Interactive peak fitting for polarized spectra
  - Advanced crystal orientation analysis:
    - Calculate orientation-dependent Raman intensities
    - Optimize crystal orientation to match experimental data
    - Support for all crystal symmetries and Raman tensor characters
  - CIF file import and analysis:
    - Parse crystallographic information files
    - Extract symmetry and atomic position data
    - Calculate Raman tensors from crystal structure
    - **Optional pymatgen integration for advanced crystallographic analysis**
  - **NEW: 3D Tensor Visualization**
    - Interactive 3D visualization of Raman tensors
    - Visualize tensor ellipsoids for different vibrational modes
    - Explore principal axes and orientation effects
    - View crystal shapes based on point group symmetry
    - Customizable visualization settings (scale, opacity, orientation)
    - Export publication-quality 3D visualizations

## Installation

### Recommended for Beginners: Anaconda + Spyder IDE

If you are new to Python or want the easiest way to get started, I recommend using the free [Anaconda](https://www.anaconda.com/products/distribution) Python distribution (this is the one I use). Anaconda includes Python and many scientific packages, and comes with the Spyder IDE, which is beginner-friendly and great for running this application.

**Steps:**

1. **Install Anaconda:**
   - Download and install Anaconda from [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution) (choose the version for your operating system).

2. **Open Anaconda Navigator:**
   - Launch "Anaconda Navigator" from your applications menu.

3. **Launch Spyder IDE:**
   - In Anaconda Navigator, click "Launch" under the Spyder IDE.

4. **Clone or Download the Repository:**
   - In Spyder, open the built-in terminal (or use your system terminal) and run:
     ```bash
     git clone https://github.com/aaroncelestian/RamanLab.git
     cd RamanLab
     ```
   - Alternatively, you can download the ZIP from GitHub and extract it.
   - Note, ```bash is just the terminal interface (Born Again SHell) in Spyder where you type in commands.

5. **Install Dependencies:**
   - In the Spyder terminal, run:
     ```bash
     pip install -r requirements.txt
     ```
   - (If you see errors, you may need to run `conda install PACKAGENAME` for any missing packages.)

6. **Download the Database File:**
   - Download the database and metadata correlation file that I made from:
     - 10.6084/m9.figshare.28928345  https://figshare.com/s/220f7de3c17172dbaae6  (Compiled database of Raman reference spectra from RRUFF.info)
                                     https://figshare.com/s/b55e31e89743d246895b  (Compiled database of Raman calculated mdoes from WURM.info)
   - Place the file in the RamanLab directory.

7. **Open and Run the Application:**
   - In Spyder, go to File > Open, and select `main.py` from the RamanLab folder.
   - Click the green "Run" button (or press F5) to start the application, and it should launch.
   - If things go wrong with the program, you may need to reset the kernel (in one of the menus). This forces python to reset and you can re-run main.py

---

### Advanced/Alternative Installation (pip/manual)

1. Clone the repository: (bash is your terminal window)
```bash
git clone https://github.com/aaroncelestian/RamanLab.git
cd RamanLab
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the database and metadata correlation file:
  - 10.6084/m9.figshare.28928345  https://figshare.com/s/220f7de3c17172dbaae6  (Compiled database of Raman reference spectra from RRUFF.info)
                                  https://figshare.com/s/b55e31e89743d246895b  (Compiled database of Raman calculated mdoes from WURM.info)
4. Run the application:
```bash
python3 main.py
```

## Manual Installation Method

1. Download all the files from this repository into a local drive on your computer. It should not matter where.
2. Run the dependency checker from a terminal window (or IDE if you have one):
   ```bash
   python3 check_dependencies.py
   ```
3. Make sure all the check boxes are checked on the output. If they are, then you are good to go! If not, then you need to install the missing ones (except for the raman_database.pkl, that can be added later) before proceeding.
4. After all dependencies are installed, you are good to go:
   ```bash
   python3 main.py
   ```

---

## Desktop Icon Installation

For easier access, you can install a desktop shortcut to launch RamanLab directly from your desktop or applications menu:

### Automatic Installation (Recommended)

1. **Install Desktop Icon:**
   ```bash
   python3 install_desktop_icon.py
   ```
   This will create platform-appropriate shortcuts:
   - **Windows:** Desktop shortcut (.lnk) with icon
   - **macOS:** Application bundle in ~/Applications
   - **Linux:** Desktop entry and applications menu shortcut

2. **Launch RamanLab:**
   - Double-click the desktop icon or find "RamanLab" in your applications menu
   - The launcher will automatically check dependencies before starting

### Uninstall Desktop Icon

To remove the desktop shortcuts:
```bash
python3 install_desktop_icon.py --uninstall
```

### Alternative Launcher

You can also use the simple launcher script:
```bash
python3 launch_ramanlab.py
```
This launcher checks dependencies and provides helpful error messages if anything is missing.

## Dependencies

- Python 3.6+
- Core requirements for the basics:
  - numpy >= 1.16.0
  - matplotlib >= 3.0.0
  - scipy >= 1.2.0
  - pandas >= 0.25.0
  - scikit-learn >= 0.21.0
  - seaborn >= 0.11.0
  - mplcursors >= 0.5.0
  - fastdtw >= 0.3.4

- Optional packages for the full experience:
  - reportlab >= 3.5.0 (for PDF export)
  - openpyxl >= 3.0.0 (for Excel support)
  - pillow >= 8.0.0 (for image processing)
  - tensorflow >= 2.12.0 (for deep learning)
  - keras >= 2.12.0 (for deep learning models)
  - umap-learn (for UMAP visualization in Raman Group Analysis)
  - pymatgen (for advanced crystallographic analysis)
  - pyinstaller >= 5.0.0 (for creating standalone executables)

## Usage

1. **Importing Spectra**
   - Use the "Import" button to load Raman spectra files
   - Supported formats: CSV, TXT, and other common data formats

2. **Processing Spectra**
   - Apply background subtraction
   - Detect and analyze peaks
   - Adjust visualization parameters

3. **Database Operations**
   - Add spectra to the database
   - Edit metadata and classifications
   - Search and filter database contents

4. **Matching and Analysis**
   - Choose search algorithm
   - Set matching parameters
   - View and export results
  
5. **Single and Batch Processing and Analysis**
   - Fit peaks
   - Auto subtract/refine background
   - View and export results

6. **2D Map Analysis**
   - Import Raman mapping data from directory
   - Create heatmaps for various spectral features
   - Apply template matching for component identification:
     - Load and manage template spectra
     - Fit templates to map data
     - View coefficient maps and percentages
     - Export analysis results
   - Filter cosmic rays automatically
   - Use machine learning for classification and clustering
   - Export analysis results and publication-quality images

7. **Raman Group Analysis**
   - Import multiple spectra from folders or database
   - Perform hierarchical clustering to group similar spectra
   - Visualize relationships using dendrograms, heatmaps, and scatter plots
   - Refine clusters with splitting and merging operations
   - Analyze cluster characteristics and export results

8. **Mineral Raman Modes Database**
   - Access the database with `python3 browse_mineral_database.py`
   - Add minerals and their Raman modes manually
   - Import peak data directly from peak fitting results
   - Import data from other pickle files
   - Visualize mineral Raman modes with simulated spectra

9. **NEW: Raman Polarization Analysis**
   - Access with `python3 raman_polarization_analyzer.py`
   - Import and analyze polarized Raman spectra
   - Perform peak fitting on polarized data
   - Import crystal structure from CIF files
   - Calculate and visualize Raman tensors
   - Determine optimal crystal orientation
   - Visualize Raman tensors in 3D

## Project Structure

All files should be in the same directory:
- `main.py` - Main application entry point
- `raman_analysis_app.py` - Main application GUI
- `raman_spectra.py` - Core spectrum processing functionality
- `check_dependencies.py` - Dependency verification
- `requirements.txt` - Project dependencies
- `peak_fitting.py` - Peak fitting
- `batch_peak_fitting.py` - Batch fitting
- `map_analysis_2d.py` - 2D Raman map analysis
- `raman_group_analysis.py` - Group analysis and clustering
- `ml_raman_map/` - Machine learning modules for map analysis
- `mineral_database.py` - Mineral Raman modes database management
- `import_peaks_to_database.py` - Tool for importing peak fitting results to database
- `browse_mineral_database.py` - Utility to browse the mineral database
- `mineral_modes.pkl` - Pickle file storing mineral Raman modes
- `raman_polarization_analyzer.py` - **NEW: Polarized Raman analysis and crystal orientation**
- `raman_tensor_3d_visualization.py` - **NEW: 3D visualization of Raman tensors**
- `tensor_character_demo.py` - **NEW: Demonstrations of tensor symmetry characters**
- `version.py` - Version information

## License

MIT

## Author

Aaron Celestian, Ph.D.
Curator of Mineral Sciences
Natural History Museum of Los Angeles County

## Acknowledgments

- RRUFF database for reference spectra (www.rruff.info)
- SLOPP and SLOPP-E for reference plastic spectra (https://rochmanlab.wordpress.com/spectral-libraries-for-microplastics-research/)
- WURM database for DFT tensors and calculated Raman.
- Scientific community for Raman spectroscopy resources
