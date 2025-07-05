# 🔬 RamanLab Peak Fitting Web Demo

## Overview

This demo showcases how RamanLab's advanced peak fitting and spectral analysis capabilities can be transformed into a modern web application. The demo includes both a **frontend web interface** and an **optional backend API** that demonstrate the core algorithms in action.

## ✨ Features Demonstrated

### 🎯 **Peak Fitting Algorithms** (from `core/peak_fitting.py`)
- **Lorentzian**: Natural line shape for homogeneous broadening
- **Gaussian**: Inhomogeneous broadening effects  
- **Voigt**: Convolution of Gaussian and Lorentzian
- **Pseudo-Voigt**: Linear combination for computational efficiency

### 📊 **Interactive Capabilities**
- Real-time parameter adjustment with sliders
- Click-to-position peak centers on spectrum
- Auto peak detection and fitting
- Multi-peak fitting support
- Live statistics calculation (R², RMSE, FWHM)

### 📁 **Data Input Options**
- File upload support (.txt, .csv, .dat files)
- Sample spectrum generation
- Standard Raman data formats

### 🔍 **Advanced Features**
- Database search simulation
- Fit quality assessment
- Algorithm information and formulas
- Responsive design for all devices

## 🚀 Quick Start

### Option 1: Frontend Only (Immediate Demo)

1. **Open the web demo directly:**
   ```bash
   # Start a simple web server
   python3 -m http.server 8000
   ```

2. **Access the demo:**
   ```
   http://localhost:8000/peak_fitting_web_demo.html
   ```

3. **Try it out:**
   - Click "Generate Sample Spectrum" 
   - Adjust peak parameters with sliders
   - Try different peak shapes
   - Click "Auto Detect Peaks" for automatic fitting

### Option 2: Full Demo with Backend API

1. **Install dependencies:**
   ```bash
   pip install fastapi uvicorn numpy scipy pydantic
   ```

2. **Start the backend API:**
   ```bash
   python peak_fitting_backend.py
   ```

3. **Start the frontend:**
   ```bash
   # In another terminal
   python3 -m http.server 8000
   ```

4. **Access both:**
   - **Web Demo**: http://localhost:8000/peak_fitting_web_demo.html
   - **API Documentation**: http://localhost:8001/docs

## 🎮 How to Use the Demo

### Basic Peak Fitting
1. **Load Data**: Upload your spectrum file or generate a sample
2. **Select Peak Shape**: Choose from Lorentzian, Gaussian, Voigt, or Pseudo-Voigt
3. **Position Peak**: Click on the spectrum or use the center slider
4. **Adjust Parameters**: Use sliders for amplitude, width, and shape-specific parameters
5. **View Results**: Real-time fit statistics and visual feedback

### Advanced Features
- **Auto Detection**: Automatically find peaks in your spectrum
- **Multi-Peak Fitting**: Add multiple peaks and fit complex spectra
- **Auto Optimization**: Let the algorithm find the best fit parameters
- **Export**: Download fitted parameters and results

### File Format Support
The demo accepts standard Raman spectroscopy data formats:
```
# Wavenumber (cm⁻¹)    Intensity (counts)
200.0                  150.2
202.0                  145.8
204.0                  152.1
...
```

## 🔧 Technical Implementation

### Frontend Technology Stack
- **HTML5 + CSS3**: Modern responsive design
- **JavaScript ES6**: Real-time calculations and interactions
- **Chart.js**: High-performance scientific plotting
- **Mathematical Functions**: Direct ports from RamanLab's algorithms

### Backend API Stack (Optional)
- **FastAPI**: Modern Python web framework
- **NumPy + SciPy**: Scientific computing (same as RamanLab)
- **Pydantic**: Data validation and API documentation
- **CORS Enabled**: Cross-origin requests for web frontend

### Core Algorithms
The web demo uses the **exact same mathematical functions** as RamanLab:

```javascript
// Direct port from core/peak_fitting.py
function lorentzian(x, amplitude, center, width) {
    const gamma = Math.abs(width) + 1e-10;
    return amplitude * (gamma * gamma) / ((x - center) * (x - center) + gamma * gamma);
}
```

## 📈 Performance Comparison

| Feature | Desktop App | Web Demo | Backend API |
|---------|-------------|----------|-------------|
| **Peak Fitting Speed** | ⚡ Excellent | ⚡ Excellent | ⚡ Excellent |
| **File Upload** | ✅ Native | ✅ Web API | ✅ REST API |
| **Real-time Updates** | ✅ Qt Signals | ✅ JavaScript | ✅ WebSocket Ready |
| **Cross-platform** | ✅ Qt6 | ✅ Any Browser | ✅ Any OS |
| **Installation** | ❌ Required | ✅ None | ⚠️ Python Only |

## 🌐 Web App Advantages

### ✅ **Immediate Benefits**
- **Zero Installation**: Works on any device with a browser
- **Universal Access**: Tablets, phones, computers, any OS
- **Automatic Updates**: Always the latest version
- **Collaboration**: Easily share results and analyses
- **Cloud Integration**: Connect to cloud databases and storage

### 🚀 **Enhanced Possibilities**
- **Real-time Collaboration**: Multiple users on same analysis
- **Cloud Computing**: Handle larger datasets on remote servers
- **Mobile Spectroscopy**: Use with portable Raman devices
- **Integration**: Embed in lab information systems
- **Scalability**: Auto-scale for large research groups

## 🔬 Scientific Accuracy

The web demo maintains **100% scientific accuracy** with RamanLab:
- ✅ Same peak fitting algorithms
- ✅ Same mathematical formulations  
- ✅ Same optimization methods
- ✅ Same statistical calculations
- ✅ Compatible data formats

## 📊 API Endpoints

The backend provides RESTful APIs for integration:

```bash
# Peak fitting
POST /api/peaks/fit
{
  "spectrum": {"wavenumbers": [...], "intensities": [...]},
  "peak_shape": "lorentzian",
  "auto_detect": true
}

# Peak detection
POST /api/peaks/detect
{
  "wavenumbers": [...],
  "intensities": [...]
}

# Database search
POST /api/search/correlation
{
  "query_spectrum": {...},
  "n_matches": 10,
  "threshold": 0.5
}
```

## 🎯 Production Deployment

For a production web application:

### Frontend Hosting
- **Static Hosting**: Netlify, Vercel, GitHub Pages
- **CDN**: Global content delivery for fast access
- **Progressive Web App**: Offline capability

### Backend Deployment  
- **Cloud Platforms**: AWS, Google Cloud, Azure
- **Containerization**: Docker for easy deployment
- **Auto-scaling**: Handle variable workloads
- **Database Integration**: PostgreSQL, MongoDB for spectra

### Security & Performance
- **HTTPS**: Secure data transmission
- **Authentication**: User accounts and access control
- **Caching**: Fast repeat analyses
- **Compression**: Efficient data transfer

## 🤝 Conclusion

This demo proves that RamanLab's sophisticated peak fitting algorithms can be seamlessly adapted for web deployment while maintaining full scientific accuracy and performance. The modular design of RamanLab's core algorithms makes web transformation straightforward and highly effective.

**Next Steps**: This foundation could easily be extended to include the full RamanLab feature set including database search, multi-component analysis, and advanced spectral processing.

---

**Try the demo now**: `python3 -m http.server 8000` then visit http://localhost:8000/peak_fitting_web_demo.html 