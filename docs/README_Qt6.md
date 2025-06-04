# RamanLab Qt6 Version

This is the Qt6 conversion project of RamanLab, designed to eliminate cross-platform compatibility issues and provide a more modern, professional interface.

## 🎯 **Goals of Qt6 Conversion**

### **Primary Objective: Solve Cross-Platform Issues**
- **Eliminate platform-specific code** (no more `if platform.system() == "Windows"`)
- **Consistent behavior** across macOS, Windows, and Linux
- **Native look and feel** on each platform
- **Better file operations** with Qt's cross-platform APIs

### **Secondary Benefits**
- Modern, professional appearance
- Better matplotlib integration
- Superior widgets and controls
- Future-proof technology stack
- Easier maintenance

## 🚀 **Getting Started**

### **1. Install Dependencies**

```bash
# Core requirements (recommended)
pip install PySide6 numpy matplotlib scipy pandas seaborn pillow mplcursors reportlab openpyxl fastdtw scikit-learn emcee

# Or install everything
pip install -r requirements_qt6.txt
```

### **2. Test the Basic App**

```bash
python main_qt6.py
```

This will launch the basic Qt6 version with:
- Spectrum import functionality
- Basic plotting with matplotlib
- Cross-platform file dialogs
- Peak detection
- Tab-based interface

## 🏗️ **Development Strategy**

### **Phase 1: Core Functionality (Current)**
- ✅ Basic Qt6 application structure
- ✅ Spectrum import with cross-platform file dialogs
- ✅ Matplotlib integration
- ✅ Basic spectrum display and peak detection
- 🔄 File save functionality
- 🔄 Background subtraction

### **Phase 2: Cross-Platform Features**
- 🔄 Replace all platform-specific file operations
- 🔄 Database management with Qt widgets
- 🔄 Search and matching functionality
- 🔄 Better progress indicators

### **Phase 3: Advanced Analysis**
- 🔄 Peak fitting window (Qt6 version)
- 🔄 Batch processing interface
- 🔄 2D map analysis
- 🔄 Cluster analysis

### **Phase 4: Polish and Optimization**
- 🔄 Themes and styling
- 🔄 Performance optimization
- 🔄 Advanced Qt features

## 📂 **Project Structure**

```
RamanLab_Qt6/
├── main_qt6.py                    # Main entry point
├── raman_analysis_app_qt6.py      # Main application class
├── requirements_qt6.txt           # Dependencies
├── version.py                     # Version info (copied from original)
├── VERSION.txt                    # Version details (copied from original)
└── README_Qt6.md                  # This file
```

## 🔄 **Migration Approach**

### **Component-by-Component Migration**
Instead of converting everything at once, we're migrating components progressively:

1. **Start with basic structure** (✅ Done)
2. **Add core functionality** one feature at a time
3. **Test thoroughly** on multiple platforms
4. **Copy over analysis algorithms** (these don't need GUI changes)
5. **Gradually replace tkinter-specific code**

### **Key Qt6 Improvements Over tkinter**

| Feature | tkinter (Original) | Qt6 (New) |
|---------|-------------------|-----------|
| **File Dialogs** | Platform-specific behavior | Consistent, native dialogs |
| **Opening Folders** | `os.startfile()` + platform checks | `QDesktopServices.openUrl()` |
| **Standard Paths** | Manual path handling | `QStandardPaths` |
| **Threading** | Basic threading | Advanced `QThread` with signals |
| **Matplotlib** | tkinter backend (limited) | Qt backend (superior) |
| **Styling** | Limited theme support | Rich styling, themes, CSS-like |
| **High-DPI** | Poor support | Excellent built-in support |

## 💡 **Key Qt6 Concepts**

### **Signals and Slots**
Replace tkinter's `command=` with Qt's signal-slot system:

```python
# tkinter
button.configure(command=self.on_click)

# Qt6
button.clicked.connect(self.on_click)
```

### **Layouts**
Replace tkinter's pack/grid with Qt's layout managers:

```python
# Create layout
layout = QVBoxLayout()
layout.addWidget(widget1)
layout.addWidget(widget2)

# Apply to container
container.setLayout(layout)
```

### **Cross-Platform File Operations**
Replace platform-specific code:

```python
# OLD (tkinter + platform checks)
if platform.system() == "Windows":
    os.startfile(path)
elif platform.system() == "Darwin":
    subprocess.run(["open", path])

# NEW (Qt6 - one line!)
QDesktopServices.openUrl(QUrl.fromLocalFile(path))
```

## 🔧 **Development Tips**

### **Testing on Multiple Platforms**
- Test file operations on different OSes
- Check high-DPI scaling
- Verify native look and feel

### **Common Migration Patterns**
1. **Widget replacement**: `ttk.Button()` → `QPushButton()`
2. **Layout conversion**: `.pack()` → layout managers
3. **Event handling**: `command=` → `.connect()`
4. **File operations**: Platform checks → Qt APIs

### **Debugging**
- Use Qt's debugging tools
- Check signal-slot connections
- Monitor layout issues with Qt's layout debugger

## 📈 **Progress Tracking**

### **Completed Features**
- [x] Basic application structure
- [x] Spectrum import and display
- [x] Cross-platform file dialogs
- [x] Basic peak detection
- [x] Matplotlib integration

### **In Progress**
- [ ] Background subtraction
- [ ] File save operations
- [ ] Database integration

### **Planned**
- [ ] Peak fitting window
- [ ] Search functionality
- [ ] Advanced analysis tools
- [ ] Styling and themes

## 🤝 **Contributing to the Conversion**

### **Priority Areas**
1. **File operations** - eliminate platform-specific code
2. **Database management** - convert tkinter widgets to Qt
3. **Analysis windows** - convert popup windows to Qt dialogs
4. **Cross-platform testing** - verify consistency

### **Testing**
Always test on multiple platforms when possible:
- macOS (your current platform)
- Windows (virtual machine or colleague)
- Linux (if accessible)

## 🔮 **Future Enhancements**

Once the basic conversion is complete, Qt6 enables:
- **Advanced themes** and dark mode
- **Better performance** with large datasets
- **Modern UI patterns** (ribbon interfaces, dockable panels)
- **Better internationalization** support
- **Professional packaging** with Qt installers

---

**Current Status**: Basic framework complete, ready for feature-by-feature migration.

**Next Steps**: 
1. Test the basic app with `python main_qt6.py`
2. Start implementing file save functionality
3. Begin converting your most-used features first 