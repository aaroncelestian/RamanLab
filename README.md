# ClaritySpectra - Raman Spectrum Analysis Tool

ClaritySpectra is a comprehensive desktop application for analyzing and identifying Raman spectra. It provides a user-friendly interface for importing, processing, and matching Raman spectra against a database of known materials.

## Features

- **Spectrum Analysis**
  - Import and visualize Raman spectra
  - Background subtraction
  - Peak detection and analysis
  - Interactive plotting and zooming
  - Spectral comparison

- **Database Management**
  - Store and organize Raman spectra
  - Edit metadata and classifications
  - Batch import capabilities
  - View and search database contents

- **Search and Matching**
	- Multiple search algorithms:
			- Correlation-based matching
			- Peak-based matching
   			- Correlation + Peak
			- Machine learning-based matching (dynamic time warping)
	- Advanced filtering options (peak regions, Hey index, element limiting)
	- Confidence scoring for matches
	- Experimental: Heatmap of best fit to molecular vibrational groups with chemical scoring

-  **Peak Fitting**
	- Advanced background modeling: 
			- Manual modeling
			- Software suggested models that are user selectable
			- Interactive background modeling
	-    Algorytmically identify peaks, or add user selected/deleted
	-    Peak modeling:
			- Gaussian, Lorentzian, Pseudo-Voigt, and Aysmmetric Voigt
			- No need to define regions for fitting, no need to spectral smoothing (although it is an option)
			- Report generation and export of pubilication quality graphics

- **Batch Processing**
	- Import any number of spectra that your system can handle
	- Same background and peak shapes as in Peak Fitting
	- AUTOMATIC background refinement
	- User selective region fitting to speed fitting times
	- Graphics analysis with 95% confidence intervals plotted (better than error bars)

- **2D Raman Map Analysis** (New!)
  - Import and visualize 2D Raman mapping data
  - Create heatmaps for integrated intensity, peak position, template coefficients
  - Advanced template analysis for component identification
  - Cosmic ray filtering for improved data quality
  - Machine learning analysis with PCA, NMF, and Random Forest classification
  - Export analysis results and publication-quality visualizations
  - Class distribution visualization and sample identification

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ClaritySpectra.git
cd ClaritySpectra
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the database and metadata correlation file:
  - 10.6084/m9.figshare.28928345  https://figshare.com/s/220f7de3c17172dbaae6 

6. Run the application:
```bash
python3 main.py
```
## Manual Installation Method

1. Download all the files from this reposity into a local drive on your computer.  It should not matter where.
2. Run the dependecy checker from a terminal window (or IDE if you have one)
   ```bash
   python3 check_dependencies.py
   ```
3. Make sure all the check boxes are checked on the output.  If it is, then you are good to go!  If not, then you need to install the missing ones (except for the raman_database.pkl, that can be added later) before proceeding.
4. After all dependecies are installed, you are good to go.
   ```bash
   python3 main.py
   ```

## Dependencies

- Python 3.6+
- Core requirements:
  - numpy >= 1.16.0
  - matplotlib >= 3.0.0
  - scipy >= 1.2.0
  - pandas >= 0.25.0
  - scikit-learn >= 0.21.0 (for ML search functionality)
  - dask (for large dataset processing)

- Optional packages:
  - reportlab >= 3.5.0 (for PDF export)

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

6. **2D Map Analysis** (New!)
   - Import Raman mapping data from directory
   - Create heatmaps for various spectral features
   - Apply template matching for component identification
   - Filter cosmic rays automatically
   - Use machine learning for classification and clustering
   - Export analysis results and publication-quality images

## Project Structure

All files should be in the same directory
- `main.py` - just run this from terminal: python3 main.py
- `raman_analysis_app.py` - Main application GUI
- `raman_spectra.py` - Core spectrum processing functionality
- `check_dependencies.py` - Dependency verification
- `requirements.txt` - Project dependencies
- `peak_fitting.py` - Peak fitting
- `batch_peak_fitting.py` - Batch fitting
- `map_analysis_2d.py` - 2D Raman map analysis
- `ml_raman_map/` - Machine learning modules for map analysis
- `raman_database.pkl` - Database of Raman spectra https://drive.google.com/drive/folders/1U1Wk9N82M9zt0PawAxlwHxPIHrpYpzkW?usp=drive_link
- `RRUFF_Export_with_Hey_Classifcation.csv` - this is needed for adding new entries to the database so everything is mapped propertly. 

## License

MIT

## Author

Aaron Celestian, Ph.D.
Curator of Mineal Sciences
Natural History Museum of Los Angeles County

## Acknowledgments

- RRUFF database for reference spectra
    www.rruff.info
- SLOPP and SLOPP-E for reference plastic spectra; 
    https://rochmanlab.wordpress.com/spectral-libraries-for-microplastics-research/
- Scientific community for Raman spectroscopy resources 
