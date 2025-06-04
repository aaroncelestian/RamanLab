# Advanced Cluster Analysis with RamanLab

This enhanced version of the Raman Cluster Analysis provides comprehensive tools for analyzing **any mineral system** with ion exchange, cation substitution, or structural changes. Originally designed for Y-for-Na ion exchange studies in hilairite, it now supports a wide range of mineral systems with customizable spectral regions and analysis parameters.

## 🎯 Universal Mineral System Support

### 📋 Built-in Preset Systems
**Choose from 10 predefined mineral systems with optimized spectral regions:**

1. **Hilairite (Y-for-Na Exchange)** - Original ion exchange analysis
2. **Zeolites (Cation Exchange)** - Multi-cation systems (Na/K/Ca/Mg)
3. **Feldspars (Al-Si Ordering)** - Tetrahedral site ordering/disordering
4. **Pyroxenes (Fe-Mg Substitution)** - M1/M2 site occupancy changes
5. **Clay Minerals (Interlayer Exchange)** - Hydration and layer charge effects
6. **Olivine (Fe-Mg Exchange)** - Forsterite-fayalite solid solutions
7. **Garnet (Cation Substitution)** - Complex multi-site substitutions
8. **Carbonates (Mg-Ca Exchange)** - Calcite-magnesite-dolomite series
9. **Spinels (Cation Ordering)** - Normal vs inverse spinel structures
10. **Custom Configuration** - Build your own system from scratch

### 🔧 Dynamic Region Management
**Fully customizable spectral regions for any mineral system:**

- **Add/Remove Regions**: Dynamic addition and deletion of spectral regions
- **Custom Names**: Define meaningful region names (e.g., "T-O-T Bending", "M1 Site Modes")
- **Flexible Ranges**: Set any wavenumber range for your specific chemistry
- **Chemical Context**: Add descriptions explaining the structural/chemical significance
- **Save/Load Configurations**: Export and import custom region setups
- **Preset Loading**: Instantly load optimized configurations for common systems

## 🔬 Enhanced Structural Analysis Features

### 🎛️ System Configuration Interface
```
┌─ System Presets & Configuration ─────────────────────────┐
│ Preset System: [Zeolites (Cation Exchange) ▼]          │
│ [Load Preset] [Save Configuration] [Load Configuration] │
│                                                          │
│ Zeolite cation exchange: K⁺, Na⁺, Ca²⁺, Mg²⁺          │
│ substitutions affecting framework and water coordination │
└──────────────────────────────────────────────────────────┘

┌─ Spectral Regions of Interest ───────────────────────────┐
│ [Add Region] [Remove Selected] [Clear All]              │
│                                                          │
│ ☑ Framework T-O-T    Range: 400 - 600 cm⁻¹             │
│   Tetrahedral framework vibrations                       │
│                                                          │
│ ☐ T-O Stretching     Range: 900 - 1200 cm⁻¹            │
│   Si-O and Al-O stretching modes                        │
│                                                          │
│ ☐ Water Modes        Range: 3200 - 3600 cm⁻¹           │
│   H₂O stretching vibrations                             │
└──────────────────────────────────────────────────────────┘
```

### 🔍 Example System Configurations

#### Hilairite (Y-for-Na Exchange)
- **Framework Vibrations** (200-600 cm⁻¹): Channel expansion/contraction
- **Si-O Stretching** (800-1200 cm⁻¹): Framework distortion effects  
- **Y-O Coordination** (300-500 cm⁻¹): New peaks from Y³⁺ incorporation
- **Na-O Coordination** (100-300 cm⁻¹): Decreasing peaks as Na⁺ is replaced

#### Zeolites (Cation Exchange)
- **Framework T-O-T** (400-600 cm⁻¹): Tetrahedral framework vibrations
- **T-O Stretching** (900-1200 cm⁻¹): Si-O and Al-O stretching modes
- **Cation-O Modes** (200-400 cm⁻¹): Cation-oxygen coordination
- **Water Modes** (3200-3600 cm⁻¹): H₂O stretching vibrations
- **OH Modes** (3600-3800 cm⁻¹): Hydroxyl group vibrations

#### Pyroxenes (Fe-Mg Substitution)
- **Si-O Stretching** (900-1100 cm⁻¹): Silicate chain stretching
- **M-O Stretching** (600-800 cm⁻¹): Metal-oxygen stretching
- **Chain Bending** (400-600 cm⁻¹): Silicate chain bending
- **M-Site Modes** (200-400 cm⁻¹): M1/M2 site vibrations
- **Fe-O Modes** (250-350 cm⁻¹): Iron-oxygen coordination
- **Mg-O Modes** (350-450 cm⁻¹): Magnesium-oxygen coordination

## 📊 Complete Analysis Capabilities

### 1. Ion Exchange/Substitution Progression Analysis
**Quantifies temporal ordering and reaction pathways**

- **UMAP Distance-Based Ordering**: Progressive structural changes
- **Progression Pathway Visualization**: Systematic reaction evolution
- **Distance Matrix Analysis**: Identify reaction intermediates

### 2. Kinetics Modeling
**Fits population data to kinetic models for rate constants**

- **Multiple Models**: Pseudo-first-order, Avrami, diffusion-controlled, multi-exponential
- **Rate Constants**: With uncertainties and confidence intervals
- **Mechanism Identification**: Through model comparison and parameter analysis

### 3. Universal Structural Characterization
**Customizable for any mineral system**

- **Peak Shift Tracking**: Monitor systematic wavenumber changes
- **Intensity Ratio Analysis**: Compositional indicators for any element pair
- **Differential Spectra**: Identify positive/negative changes between stages
- **Mean Cluster Spectra**: Representative spectra for each structural state

### 4. Quantitative Validation
**Publication-ready statistical validation**

- **Silhouette Analysis**: Target >0.7 for robust cluster assignments
- **Bootstrap Stability**: Tests clustering consistency with resampling
- **Boundary Analysis**: Identifies transition/intermediate samples

### 5. Advanced Statistics
**Comprehensive significance testing and feature identification**

- **Feature Importance**: Random Forest/LDA for most discriminating wavenumbers
- **Statistical Significance**: PERMANOVA, ANOSIM, Kruskal-Wallis testing
- **Discriminant Analysis**: Optimal cluster separation and chemical interpretation

## 🚀 Quick Start for Any Mineral System

### 1. Launch with Examples
```python
python example_advanced_cluster_analysis.py
```
**Includes synthetic data for multiple mineral systems:**
- Hilairite (Y-for-Na exchange)
- Zeolites (multi-cation exchange)
- Feldspars (Al-Si ordering)
- Pyroxenes (Fe-Mg substitution)
- Clay minerals (interlayer exchange)

### 2. Configure Your System
1. **Go to "Structural Analysis" tab**
2. **Select preset** from dropdown or choose "Custom Configuration"
3. **Modify regions** - Add/remove/edit spectral regions as needed
4. **Add descriptions** - Provide chemical context for each region
5. **Save configuration** - Export for future use

### 3. Customize Regions
```
Add Region → Name: "M1 Site Vibrations"
          → Range: 200-400 cm⁻¹
          → Description: "Octahedral M1 site metal-oxygen vibrations"
```

### 4. Run Complete Analysis
1. **Import your data** (folder, database, or main app)
2. **Run clustering** with optimized parameters
3. **Analyze progression** using UMAP or centroid distances
4. **Fit kinetic models** with time/composition data
5. **Validate clusters** with statistical tests
6. **Generate results** for publication

## 🎯 Applications

### Ideal for Any System With:
- **Ion exchange processes** (any cation combination)
- **Cation ordering/disordering** (temperature-dependent)
- **Phase transitions** (structural changes)
- **Solid-state reactions** (progressive changes)
- **Compositional gradients** (solid solutions)
- **Hydration/dehydration** (water content changes)
- **Pressure-induced changes** (coordination changes)

### Research Areas:
- **Mineralogy & Petrology**: Natural mineral variations
- **Materials Science**: Synthetic material optimization
- **Geochemistry**: Environmental mineral alterations
- **Crystallography**: Structure-property relationships
- **Catalysis**: Active site characterization

## 💾 Configuration Management

### Save Custom Configurations
```json
{
  "system_name": "Custom Pyroxene Analysis",
  "description": "Fe-Mg substitution in augite with pressure effects",
  "regions": [
    {
      "name": "Si-O Chain Stretching",
      "min_wn": "900",
      "max_wn": "1100", 
      "description": "Silicate chain stretching modes"
    },
    {
      "name": "M2 Site Coordination",
      "min_wn": "180",
      "max_wn": "280",
      "description": "M2 site metal-oxygen coordination"
    }
  ]
}
```

### Load and Share
- **Export configurations** as JSON files
- **Share with collaborators** for consistent analysis
- **Version control** your analysis parameters
- **Reproduce results** exactly

## 🤝 From Hilairite to Universal Tool

**This tool evolved from hilairite-specific analysis to universal mineral system support:**

✅ **Maintains all original capabilities** for hilairite Y-for-Na exchange  
✅ **Adds 9 more preset systems** with optimized spectral regions  
✅ **Enables unlimited customization** for any mineral system  
✅ **Preserves all advanced analysis features** (kinetics, validation, statistics)  
✅ **Provides examples** for multiple mineral types  

**Perfect for:**
- Researchers working with **any mineral system**
- Labs analyzing **multiple mineral types**
- Studies requiring **custom spectral regions**
- Publications needing **flexible, validated analysis**

## 📖 Citation

When using this enhanced tool, please cite:
- **RamanLab software** (original development)
- **Relevant algorithms** (scikit-learn, UMAP if used)
- **Your specific mineral system** preset or configuration
- **Statistical methods** used (PERMANOVA, silhouette analysis, etc.)

---

**🎉 Now supports ANY mineral system with ion exchange, cation substitution, or structural changes!** 

Configure once, analyze forever. From hilairite to spinels, zeolites to garnets - one tool for all your mineral cluster analysis needs. 