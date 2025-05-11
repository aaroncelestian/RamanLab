# Unknown Spectra Classification Guide

## Overview

The "Classify Unknowns & Visualize" feature allows you to classify unknown Raman spectra and visualize the results. This guide explains how to use this feature effectively.

## Prerequisites

1. You must have a trained model (the default name is `raman_plastic_classifier.joblib` in the main directory)
2. Your unknown spectra should be organized in a directory with `.txt` files (one spectrum per file)

## Directory Structure

For best results, organize your files as follows:

```
unknown_directory/
├── spectrum_Y010_X010.txt
├── spectrum_Y010_X020.txt
├── spectrum_Y020_X010.txt
├── ...
```

The `_Y###_X###` naming convention helps the system recognize the spatial coordinates of each spectrum, which is important for generating the 2D visualization map.

## File Format

Each spectrum file should contain tab-delimited data with:
- First column: Wavenumbers (cm⁻¹)
- Second column: Intensity values

Example:
```
200.0	105.3
201.0	106.2
202.0	107.5
...
```

## Steps to Classify Unknowns

1. In the 2D Map Analysis window, click on "Classify Unknowns & Visualize"
2. Select the directory containing your unknown spectra files
3. The system will:
   - Create a `results` subdirectory in your chosen directory
   - Process all spectra and classify them
   - Save results to `results/unknown_spectra_results.csv`
   - Generate a visualization of the results

## Output Format

The CSV output file contains:
- `Filename`: Original spectrum filename
- `x_pos`, `y_pos`: Spatial coordinates (extracted from filenames)
- `Prediction`: Class prediction (Class A or Class B)
- `Confidence`: Confidence score (0-1)

## Troubleshooting

If you encounter errors:

1. **"CSV missing required columns"**: The system couldn't find coordinate information. Make sure your filenames follow the `_Y###_X###.txt` format.

2. **"Directory does not exist"**: Double-check that you selected a valid directory.

3. **"Model file not found"**: Ensure you have trained and saved a model first.

4. **For other errors**: Check the output in the results text area for more details.

## Sample Data

A sample output file (`sample_output.csv`) is provided in the `ml_raman_map` directory to demonstrate the correct format.

## Quick Test Method

For the quickest way to test this functionality without generating test files or training a model:

1. Copy the `ml_raman_map/sample_results.csv` file to a new location like `./test_results/results/unknown_spectra_results.csv`

   ```
   mkdir -p ./test_results/results
   cp ml_raman_map/sample_results.csv ./test_results/results/unknown_spectra_results.csv
   ```

2. Open the 2D Map Analysis window and click "Classify Unknowns & Visualize"

3. Select the `test_results` directory when prompted

This will load the pre-made results and show you the visualization directly, skipping the classification process.

## Testing with Generated Data

To help you test the feature, we provide two helper scripts:

### 1. Creating Test Spectra Files

Run the following command to create a directory with test spectra:

```
python ml_raman_map/create_test_data.py ./test_unknowns --grid-size 5
```

This will create a directory called `test_unknowns` with synthetic Raman spectra arranged in a 5×5 grid pattern. You can then use this directory with the "Classify Unknowns & Visualize" feature.

### 2. Creating a Test Model

If you don't have a trained model yet, you can create a dummy model for testing:

```
python ml_raman_map/create_test_model.py
```

This creates a `raman_plastic_classifier.joblib` file in the current directory that will classify spectra randomly (since it's just for testing).

### Complete Test Workflow

To test the entire feature from start to finish:

1. Create a test model:
   ```
   python ml_raman_map/create_test_model.py
   ```

2. Create test spectra:
   ```
   python ml_raman_map/create_test_data.py ./test_unknowns
   ```

3. Open the 2D Map Analysis window and click "Classify Unknowns & Visualize"

4. Select the `test_unknowns` directory

5. View the results in the visualization

## Advanced Usage

If your files don't follow the naming convention, the system will attempt to display results as a simple bar chart showing the distribution of classifications rather than a 2D map. 