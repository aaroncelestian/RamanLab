# Machine Learning Implementation Guide

## Overview

The RamanLab application now includes a comprehensive Machine Learning (ML) analysis system supporting both **supervised classification** and **unsupervised clustering** of Raman spectroscopy data. This implementation provides professional-grade ML capabilities with an intuitive user interface.

## Features

### 🔍 **Supervised Classification**
- **Random Forest**: Robust ensemble method, excellent for Raman data
- **Support Vector Machine (SVM)**: Powerful for high-dimensional spectral data
- **Gradient Boosting**: Advanced boosting algorithm for complex patterns

### 🎯 **Unsupervised Clustering**
- **K-Means**: Fast and efficient for well-separated clusters
- **Gaussian Mixture Model (GMM)**: Probabilistic clustering with soft assignments
- **DBSCAN**: Density-based clustering, robust to noise and outliers
- **Hierarchical Clustering**: Tree-based clustering for hierarchical relationships

### 🚀 **Advanced Features**
- **Multi-class Training**: Support for unlimited number of classes
- **Feature Integration**: Seamless integration with PCA and NMF features
- **Model Persistence**: Save and load trained models
- **Cross-validation**: Robust performance evaluation
- **Map Integration**: Visualize ML results directly on Raman maps

## User Workflow

### Supervised Classification Workflow

#### 1. **Prepare Training Data**
- Organize your training spectra into separate folders by class
- Each folder should contain CSV files with wavenumber and intensity columns
- Recommended: 15-50 spectra per class for good training

```
Training_Data/
├── Class_A_Polymer/
│   ├── spectrum_001.csv
│   ├── spectrum_002.csv
│   └── ...
├── Class_B_Mineral/
│   ├── spectrum_001.csv
│   ├── spectrum_002.csv
│   └── ...
└── Class_C_Biological/
    ├── spectrum_001.csv
    ├── spectrum_002.csv
    └── ...
```

#### 2. **Load Training Data**
1. Switch to the **ML Analysis** tab
2. Select **"Supervised Classification"** in Analysis Type
3. Click **"Load Training Data Folders"**
4. Specify the number of classes
5. Name each class and select its folder
6. Wait for data loading completion

#### 3. **Configure Training Parameters**
- **Model Type**: Choose from Random Forest, SVM, or Gradient Boosting
- **N Estimators**: Number of trees (50-200 recommended)
- **Max Depth**: Tree depth (5-15 recommended)
- **Test Size**: Fraction for testing (0.2-0.3 recommended)
- **Feature Options**: Choose raw spectra, PCA, or NMF features

#### 4. **Train the Model**
1. Click **"Train Supervised Model"**
2. Monitor progress in the status bar
3. Review training results in the ML tab
4. Check accuracy and cross-validation scores

#### 5. **Apply to Map**
1. Load your Raman map data
2. Click **"Apply to Map"**
3. Switch to Map View to see classification results
4. Select "ML Classification" from the feature dropdown

### Unsupervised Clustering Workflow

#### 1. **Load Map Data**
- Load your Raman map data as usual
- Ensure data quality and preprocessing if needed

#### 2. **Configure Clustering Parameters**
1. Switch to the **ML Analysis** tab
2. Select **"Unsupervised Clustering"** in Analysis Type
3. Choose clustering method:
   - **K-Means**: Set number of clusters (2-10 typical)
   - **GMM**: Set number of components
   - **DBSCAN**: Configure eps (distance) and min_samples
   - **Hierarchical**: Set number of clusters

#### 3. **Run Clustering**
1. Optionally enable PCA or NMF feature transformation
2. Click **"Train Clustering Model"**
3. Review clustering results and metrics
4. Check silhouette score for quality assessment

#### 4. **Visualize Results**
- Clustering results automatically appear in Map View
- Select "ML Clusters" from the feature dropdown
- Different colors represent different clusters

## Technical Implementation

### Architecture

```
ML System Architecture:
├── MLTrainingDataManager
│   ├── Multi-class data loading
│   ├── Label encoding
│   └── Data validation
├── SupervisedMLAnalyzer
│   ├── Random Forest
│   ├── Support Vector Machine
│   ├── Gradient Boosting
│   └── Cross-validation
├── UnsupervisedAnalyzer
│   ├── K-Means clustering
│   ├── Gaussian Mixture Model
│   ├── DBSCAN clustering
│   └── Hierarchical clustering
└── Feature Integration
    ├── PCA transformation
    ├── NMF transformation
    └── Raw spectral features
```

### Data Flow

1. **Input**: Raman spectral data (wavenumbers + intensities)
2. **Preprocessing**: Cosmic ray removal, normalization (optional)
3. **Feature Extraction**: Raw spectra, PCA components, or NMF components
4. **Model Training**: Supervised or unsupervised learning
5. **Prediction**: Apply trained model to map data
6. **Visualization**: Color-coded maps showing results

### Model Performance Metrics

#### Supervised Learning
- **Accuracy**: Overall classification accuracy
- **Cross-validation**: 5-fold CV with mean and standard deviation
- **Confusion Matrix**: Detailed per-class performance
- **Classification Report**: Precision, recall, F1-score per class

#### Unsupervised Learning
- **Silhouette Score**: Cluster quality measure (-1 to 1, higher is better)
- **Number of Clusters**: Detected or specified clusters
- **Inertia**: Within-cluster sum of squares (K-Means)
- **Noise Points**: Number of outliers (DBSCAN)

## Best Practices

### Data Preparation
- **Quality Control**: Remove bad spectra before training
- **Cosmic Ray Removal**: Enable for better data quality
- **Balanced Classes**: Aim for similar numbers of spectra per class
- **Representative Sampling**: Ensure training data covers expected variations

### Model Selection
- **Random Forest**: Good default choice, robust and interpretable
- **SVM**: Best for high-dimensional data with clear boundaries
- **Gradient Boosting**: For complex, non-linear relationships
- **K-Means**: Fast clustering for well-separated groups
- **DBSCAN**: When you expect noise or variable cluster densities

### Feature Engineering
- **Raw Spectra**: Use when you have good signal-to-noise ratio
- **PCA Features**: Reduce dimensionality, remove noise
- **NMF Features**: Extract interpretable spectral components
- **Preprocessing**: Consider normalization for better results

### Validation
- **Cross-validation**: Always check CV scores for model reliability
- **Test Set**: Use separate test data for final validation
- **Visual Inspection**: Check map results for spatial consistency
- **Domain Knowledge**: Validate results against known material properties

## Troubleshooting

### Common Issues and Solutions

#### Low Classification Accuracy
- **Solution**: Increase training data size
- **Solution**: Try different algorithms (SVM often works better for spectral data)
- **Solution**: Use feature transformation (PCA/NMF)
- **Solution**: Check data quality and remove outliers

#### Poor Clustering Results
- **Solution**: Adjust clustering parameters (n_clusters, eps)
- **Solution**: Try different algorithms
- **Solution**: Use feature transformation to reduce noise
- **Solution**: Check silhouette score for optimal cluster number

#### Memory Issues with Large Datasets
- **Solution**: Use PCA feature transformation to reduce dimensionality
- **Solution**: Reduce batch size in NMF analysis
- **Solution**: Process data in smaller chunks

#### Model Training Fails
- **Solution**: Check training data format (CSV with wavenumbers + intensities)
- **Solution**: Ensure all spectra have the same wavenumber range
- **Solution**: Remove corrupted or empty files
- **Solution**: Check for sufficient data per class (minimum 5-10 spectra)

### Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "No training data loaded" | Training folders not loaded | Use "Load Training Data Folders" |
| "No valid spectra found" | Bad data format | Check CSV format and file integrity |
| "Training failed" | Algorithm parameters | Adjust parameters or try different algorithm |
| "Model not trained" | No model available | Train a model first |
| "Classification failed" | Model/data mismatch | Ensure compatible wavenumber ranges |

## Advanced Usage

### Custom Preprocessing
```python
def custom_preprocessor(wavenumbers, intensities):
    # Custom preprocessing pipeline
    # Example: normalization + smoothing
    from scipy.signal import savgol_filter
    
    # Normalize
    intensities = intensities / np.max(intensities)
    
    # Smooth
    intensities = savgol_filter(intensities, 11, 3)
    
    return intensities
```

### Batch Processing
- Use the ML system to classify multiple maps
- Save trained models for consistent analysis
- Apply the same model across different datasets

### Integration with Other Analysis
- Combine ML results with template fitting
- Use ML to identify regions for detailed PCA/NMF analysis
- Create multi-modal analysis workflows

## File Formats

### Training Data Format
```csv
# spectrum_001.csv
wavenumber,intensity
200.0,1234.5
201.0,1245.2
202.0,1256.8
...
```

### Model Files
- **Format**: `.pkl` (Python pickle format)
- **Content**: Complete model state including preprocessing parameters
- **Compatibility**: Models can be shared between users with same software version

## Performance Guidelines

### Recommended System Requirements
- **RAM**: 8GB minimum, 16GB recommended for large datasets
- **CPU**: Multi-core processor recommended for faster training
- **Storage**: SSD recommended for faster data loading

### Performance Optimization
- **Feature Selection**: Use PCA/NMF for large spectral ranges
- **Model Complexity**: Balance accuracy vs. training time
- **Data Size**: Consider sub-sampling very large datasets
- **Parallel Processing**: Take advantage of multi-core systems

## Future Enhancements

### Planned Features
- **Deep Learning**: Neural network models for complex patterns
- **Time Series**: Support for time-resolved Raman data
- **Multi-modal**: Integration with other spectroscopic techniques
- **Cloud Processing**: Remote training for very large datasets

### Custom Models
The architecture supports adding new ML algorithms:
- Implement new analyzer classes
- Add to UI control panels
- Integrate with existing workflow

## Support and Resources

### Documentation
- **API Reference**: See inline code documentation
- **Example Scripts**: Check `test_ml_comprehensive.py`
- **Video Tutorials**: [Coming soon]

### Community
- **GitHub Issues**: Report bugs and feature requests
- **User Forum**: Share analysis strategies
- **Workshops**: Hands-on training sessions

---

**The ML implementation provides professional-grade machine learning capabilities for Raman spectroscopy analysis, making advanced data science techniques accessible through an intuitive interface.** 