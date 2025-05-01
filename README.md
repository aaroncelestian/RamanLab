# ClariSpectra - Raman Spectrum Analysis Tool

ClariSpectra is a comprehensive desktop application for analyzing and identifying Raman spectra. It provides a user-friendly interface for importing, processing, and matching Raman spectra against a database of known materials.

## Features

- **Spectrum Analysis**
  - Import and visualize Raman spectra
  - Background subtraction
  - Peak detection and analysis
  - Interactive plotting and zooming

- **Database Management**
  - Store and organize Raman spectra
  - Edit metadata and classifications
  - Batch import capabilities
  - View and search database contents

- **Search and Matching**
  - Multiple search algorithms:
    - Correlation-based matching
    - Peak-based matching
    - Machine learning-based matching
  - Advanced filtering options
  - Confidence scoring for matches  

- **Export and Reporting**
  - Generate detailed match reports
  - Export results to PDF, TXT, and CSV formats
  - Create correlation heatmaps
  - Save processed spectra

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ClariSpectra.git
cd ClariSpectra
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python raman_analysis_app.py
```

## Dependencies

- Python 3.6+
- Core requirements:
  - numpy >= 1.16.0
  - matplotlib >= 3.0.0
  - scipy >= 1.2.0
  - pandas >= 0.25.0

- Optional packages:
  - reportlab >= 3.5.0 (for PDF export)
  - scikit-learn >= 0.21.0 (for ML search functionality)
  - pyinstaller >= 5.0.0 (for packaging)

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

## Project Structure

- `raman_analysis_app.py` - Main application GUI
- `raman_spectra.py` - Core spectrum processing functionality
- `check_dependencies.py` - Dependency verification
- `requirements.txt` - Project dependencies
- `raman_database.pkl` - Database of Raman spectra

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT

## Author

Aaron Celestian, Ph.D.
Curator of Mineal Sciences
Natural History Museum of Los Angeles

## Acknowledgments

- RRUFF database for reference spectra
    www.rruff.info
- SLOPP and SLOPP-E for reference plastic spectra; 
    https://rochmanlab.wordpress.com/spectral-libraries-for-microplastics-research/
- Scientific community for Raman spectroscopy resources 