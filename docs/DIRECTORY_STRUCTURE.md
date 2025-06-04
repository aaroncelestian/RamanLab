# RamanLab Directory Structure

This document outlines the organization of the RamanLab project directory.

## Root Directory

```
RamanLab/
├── main_qt6.py                    # Main entry point
├── raman_analysis_app_qt6.py      # Main application class
├── requirements_qt6.txt           # Dependencies
├── version.py                     # Version info
├── VERSION.txt                    # Version details
└── README_Qt6.md                  # This file
```

## 🏗️ **CURRENT STATE**

```
RamanLab/
├── 📄 raman_polarization_analyzer.py              # ⚠️  ORIGINAL (11,378 lines)
├── 📄 raman_polarization_analyzer_qt6.py          # ✅ BASIC QT6 (1,200 lines)
├── 📄 raman_polarization_analyzer_modular_qt6.py  # ✅ MODULAR DEMO (576 lines)
├── 📁 core/                                        # ✅ CREATED
│   ├── 📄 __init__.py                             # ✅ DONE
│   ├── 📄 database.py                             # ✅ COMPLETE (533 lines)
│   └── 📄 peak_fitting.py                         # ✅ COMPLETE (477 lines)
├── 📄 PROJECT_TRACKING.md                         # ✅ CREATED
├── 📄 DAILY_TODO.md                              # ✅ CREATED
└── 📄 DIRECTORY_STRUCTURE.md                     # ✅ THIS FILE
```

---

## 🎯 **TARGET STRUCTURE** (Final Goal)

```
RamanLab/
├── 📄 main_qt6.py                                 # 🎯 MAIN APP (<500 lines)
├── 📄 setup.py                                    # 🎯 PACKAGE SETUP
├── 📄 requirements.txt                            # 🎯 DEPENDENCIES
├── 📄 README.md                                   # 🎯 USER GUIDE
├── 📄 CHANGELOG.md                                # 🎯 VERSION HISTORY
│
├── 📁 core/                                        # 🏭 BUSINESS LOGIC
│   ├── 📄 __init__.py                             # ✅ DONE
│   ├── 📄 database.py                             # ✅ COMPLETE (533 lines)
│   ├── 📄 peak_fitting.py                         # ✅ COMPLETE (477 lines)
│   ├── 📄 polarization.py                         # 🔄 IN PROGRESS (~150 lines)
│   ├── 📄 tensor_calc.py                          # ⭕ TODO (~50 lines)
│   ├── 📄 spectrum.py                             # ⭕ TODO (~110 lines)
│   └── 📄 file_io.py                              # ⭕ TODO (~160 lines)
│
├── 📁 ui/                                          # 🎨 USER INTERFACE
│   ├── 📄 __init__.py                             # ⭕ TODO
│   ├── 📄 main_window.py                          # ⭕ TODO (main UI container)
│   ├── 📄 spectrum_analysis.py                    # ⭕ TODO (~120 lines)
│   ├── 📄 peak_fitting.py                         # ⭕ TODO (~120 lines)
│   ├── 📄 polarization.py                         # ⭕ TODO (~100 lines)
│   ├── 📄 polarization_dialogs.py                 # ⭕ TODO (~380 lines)
│   ├── 📄 crystal_structure.py                    # ⭕ TODO (~75 lines)
│   ├── 📄 tensor_analysis.py                      # ⭕ TODO (~50 lines)
│   ├── 📄 orientation.py                          # ⭕ TODO (~40 lines)
│   ├── 📄 stress_strain.py                        # ⭕ TODO (~60 lines)
│   └── 📄 visualization_3d.py                     # ⭕ TODO (~80 lines)
│
├── 📁 parsers/                                     # 📖 FILE PARSERS
│   ├── 📄 __init__.py                             # ⭕ TODO
│   ├── 📄 cif_parser.py                           # ⭕ TODO (~290 lines)
│   ├── 📄 spectrum_parser.py                      # ⭕ TODO (~80 lines)
│   └── 📄 database_parser.py                      # ⭕ TODO (~60 lines)
│
├── 📁 analysis/                                    # 🔬 ANALYSIS ALGORITHMS
│   ├── 📄 __init__.py                             # ⭕ TODO
│   ├── 📄 symmetry.py                             # ⭕ TODO (~115 lines)
│   ├── 📄 crystal_structure.py                    # ⭕ TODO (~150 lines)
│   ├── 📄 stress_strain.py                        # ⭕ TODO (~80 lines)
│   └── 📄 orientation.py                          # ⭕ TODO (~70 lines)
│
├── 📁 utils/                                       # 🛠️ UTILITIES
│   ├── 📄 __init__.py                             # ⭕ TODO
│   ├── 📄 plotting.py                             # ⭕ TODO (~265 lines)
│   ├── 📄 math_utils.py                           # ⭕ TODO (~100 lines)
│   └── 📄 validation.py                           # ⭕ TODO (~50 lines)
│
├── 📁 tests/                                       # 🧪 TESTING
│   ├── 📄 __init__.py                             # ⭕ TODO
│   ├── 📄 test_database.py                        # ⭕ TODO
│   ├── 📄 test_peak_fitting.py                    # ⭕ TODO
│   ├── 📄 test_polarization.py                    # ⭕ TODO
│   ├── 📄 test_parsers.py                         # ⭕ TODO
│   ├── 📄 test_analysis.py                        # ⭕ TODO
│   └── 📄 test_integration.py                     # ⭕ TODO
│
├── 📁 examples/                                    # 📚 EXAMPLES
│   ├── 📄 basic_usage.py                          # ⭕ TODO
│   ├── 📄 custom_analysis.py                      # ⭕ TODO
│   ├── 📄 batch_processing.py                     # ⭕ TODO
│   └── 📄 plugin_example.py                       # ⭕ TODO
│
├── 📁 docs/                                        # 📖 DOCUMENTATION
│   ├── 📄 index.rst                               # ⭕ TODO
│   ├── 📄 api_reference.rst                       # ⭕ TODO
│   ├── 📄 user_guide.rst                          # ⭕ TODO
│   ├── 📄 developer_guide.rst                     # ⭕ TODO
│   └── 📄 migration_guide.rst                     # ⭕ TODO
│
└── 📁 resources/                                   # 📦 RESOURCES
    ├── 📄 mineral_database.pkl                    # ⭕ TODO
    ├── 📄 example_spectra/                        # ⭕ TODO
    └── 📄 icons/                                   # ⭕ TODO
```

---

## 📊 **PROGRESS BY PACKAGE**

### **📦 Package Completion Status**

| **Package** | **Files** | **Complete** | **Progress** | **Est. Lines** |
|-------------|-----------|--------------|--------------|----------------|
| `core/` | 5 | 2/5 | 40% | ~830 total |
| `ui/` | 10 | 0/10 | 0% | ~1,025 total |
| `parsers/` | 4 | 0/4 | 0% | ~430 total |
| `analysis/` | 5 | 0/5 | 0% | ~415 total |
| `utils/` | 4 | 0/4 | 0% | ~415 total |
| `tests/` | 7 | 0/7 | 0% | ~700 total |

**Overall**: 2/35 files complete (6% by file count, 35% by complexity)

---

## 🎯 **NEXT DIRECTORIES TO CREATE**

### **Immediate (Today)**
```bash
mkdir parsers
mkdir analysis  
mkdir utils
touch parsers/__init__.py
touch analysis/__init__.py
touch utils/__init__.py
```

### **This Week**
```bash
mkdir tests
mkdir examples
mkdir docs
mkdir resources
# Create all __init__.py files
# Create placeholder module files
```

---

## 📋 **FILE SIZE DISTRIBUTION**

### **By Module Size** (estimated)
- **🔴 Large (>150 lines)**: 8 files
  - `core/polarization.py` (~150)
  - `core/file_io.py` (~160)
  - `parsers/cif_parser.py` (~290)
  - `analysis/crystal_structure.py` (~150)
  - `ui/polarization_dialogs.py` (~380)
  - `utils/plotting.py` (~265)

- **🟡 Medium (50-150 lines)**: 15 files
- **🟢 Small (<50 lines)**: 12 files

### **Complexity Heatmap**
```
🔴🔴🔴 parsers/cif_parser.py (290 lines, very complex)
🔴🔴🔴 ui/polarization_dialogs.py (380 lines, very complex)
🔴🔴   utils/plotting.py (265 lines, complex)
🔴🔴   core/file_io.py (160 lines, complex)
🔴🔴   core/polarization.py (150 lines, complex)
🔴🔴   analysis/crystal_structure.py (150 lines, complex)
🟡     ui/spectrum_analysis.py (120 lines, medium)
🟡     ui/peak_fitting.py (120 lines, medium)
```

---

## 🚀 **MIGRATION STRATEGY**

### **Phase 1**: Core Foundation (Week 1)
1. ✅ Create basic structure
2. 🔄 Complete `core/` modules
3. ⭕ Create `parsers/` package

### **Phase 2**: Major Functions (Week 2)
1. Extract largest functions first
2. Focus on `parsers/cif_parser.py`
3. Begin `ui/polarization_dialogs.py`

### **Phase 3**: UI Modules (Week 3-4)
1. Extract all UI tab modules
2. Create dialog modules
3. Test UI functionality

### **Phase 4**: Polish (Week 5)
1. Testing and documentation
2. Examples and guides
3. Performance optimization

---

## 🔍 **DEPENDENCY MAP**

### **Core Dependencies**
- `core/database.py` → No dependencies
- `core/peak_fitting.py` → No dependencies  
- `core/polarization.py` → `core/database.py`
- `core/tensor_calc.py` → `core/database.py`
- `core/spectrum.py` → `core/peak_fitting.py`

### **UI Dependencies**
- `ui/main_window.py` → All core modules
- `ui/spectrum_analysis.py` → `core/spectrum.py`, `core/database.py`
- `ui/peak_fitting.py` → `core/peak_fitting.py`
- `ui/polarization.py` → `core/polarization.py`

### **Parser Dependencies**
- `parsers/cif_parser.py` → `core/database.py`
- `parsers/spectrum_parser.py` → `core/spectrum.py`

---

**📝 Note**: This structure follows Python packaging best practices and separates concerns clearly for maximum maintainability and testability. 