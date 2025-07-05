# Configurable Projects Folder System

## Overview

RamanLab now features a comprehensive configuration system that allows users to specify where their projects, sessions, and analysis results are stored. No more hardcoded paths to the user's home directory!

## Features

### ✅ What's New

- **Configurable Projects Folder**: Choose any location for your RamanLab data
- **Settings Dialog**: User-friendly interface accessible via Settings → Preferences
- **Automatic Path Management**: All modules now use the configured paths
- **Persistent Settings**: Configuration is saved between sessions
- **Cross-Platform**: Works on Windows, macOS, and Linux

### 🔧 Configuration System Components

#### 1. Configuration Manager (`core/config_manager.py`)
- Handles loading, saving, and managing user preferences
- Stores configuration in JSON format
- Provides convenient methods for path management
- Automatic fallback to default values

#### 2. Settings Dialog (`core/settings_dialog.py`)
- Tabbed interface for different setting categories:
  - **General**: Auto-save settings, recent files
  - **Paths**: Projects folder, database paths
  - **Appearance**: Plot settings, themes
  - **Analysis**: Peak detection defaults
  - **Advanced**: Configuration preview and diagnostics

#### 3. Integration
- Main application automatically loads configuration
- Batch Peak Fitting uses configurable paths
- Universal State Manager respects user settings
- All new modules will use the configuration system

## Usage

### 🖥️ GUI Access

1. Launch RamanLab: `python main_qt6.py`
2. Go to **Settings → Preferences...**
3. Navigate to the **Paths** tab
4. Click **Browse...** to select your preferred projects folder
5. Configure other settings as needed
6. Click **OK** to save

### 📁 Default Folder Structure

```
<Your_Projects_Folder>/
├── auto_saves/          # Session auto-saves
├── exports/             # Exported data and results
├── analysis_results/    # Analysis outputs
└── custom_databases/    # User-created databases
```

### 🔧 Programmatic Access

```python
from core.config_manager import get_config_manager

# Get configuration instance
config = get_config_manager()

# Get paths
projects_folder = config.get_projects_folder()
auto_save_folder = config.get_auto_save_folder()
session_folder = config.get_session_folder()

# Set custom paths
config.set_projects_folder("/path/to/my/ramanlab/data")

# Access other settings
auto_save_enabled = config.get("auto_save_enabled")
plot_dpi = config.get("plot_settings.dpi")

# Set settings
config.set("auto_save_interval", 600)  # 10 minutes
```

## Configuration File

Settings are stored in a platform-appropriate location:

- **Windows**: `%APPDATA%/RamanLab/ramanlab_config.json`
- **macOS**: `~/Library/Application Support/RamanLab/ramanlab_config.json`
- **Linux**: `~/.local/share/RamanLab/ramanlab_config.json`

### Example Configuration

```json
{
  "projects_folder": "/Users/username/Documents/MyRamanLabData",
  "auto_save_enabled": true,
  "auto_save_interval": 300,
  "theme": "default",
  "recent_files": [
    "/path/to/recent/spectrum1.txt",
    "/path/to/recent/spectrum2.csv"
  ],
  "max_recent_files": 10,
  "database_paths": {
    "main_database": "RamanLab_Database_20250602.pkl",
    "mineral_modes": "mineral_modes.pkl"
  },
  "plot_settings": {
    "dpi": 100,
    "figure_size": [10, 6],
    "line_width": 1.0,
    "grid_alpha": 0.3
  },
  "peak_detection": {
    "height": 0.1,
    "prominence": 0.05,
    "distance": 10
  }
}
```

## Demo and Testing

### 🧪 Try the Demo

Run the configuration demo to see the system in action:

```bash
python demo_configuration_system.py
```

This will show:
- Current configuration settings
- How to change the projects folder
- Benefits of the new system
- Folder structure explanation

### 🔍 Module Integration Status

| Module | Status | Description |
|--------|--------|-------------|
| Main Application | ✅ Integrated | Settings menu added |
| Batch Peak Fitting | ✅ Integrated | Uses configurable session folder |
| Universal State Manager | ✅ Integrated | Respects projects folder setting |
| Configuration Manager | ✅ Complete | Core system implementation |
| Settings Dialog | ✅ Complete | Full GUI interface |

## Benefits

### 🎯 For Users

- **Flexibility**: Store data wherever you want
- **Organization**: Keep all RamanLab data in one chosen location
- **Backup**: Easy to backup your entire RamanLab workspace
- **Network Storage**: Can use network drives or cloud-synced folders
- **Multi-User**: Different users can have different project locations

### 🔧 For Developers

- **Consistency**: All modules use the same configuration system
- **Maintainability**: No more hardcoded paths throughout the codebase
- **Extensibility**: Easy to add new configuration options
- **Platform Independence**: Works across all operating systems

## Migration from Hardcoded Paths

The system automatically:
1. Creates the default projects folder if it doesn't exist
2. Maintains backward compatibility with existing installations
3. Uses sensible defaults for all settings
4. Gracefully handles missing configuration files

### Manual Migration (Optional)

If you have existing data in the old hardcoded location:

1. Open the old location: `~/RamanLab_Projects/`
2. Copy your data to your new chosen location
3. Update the configuration to point to the new location
4. Verify everything works correctly

## Troubleshooting

### Common Issues

**Settings dialog won't open**
- Ensure you're running from the RamanLab directory
- Check that `core/config_manager.py` and `core/settings_dialog.py` exist

**Configuration not saving**
- Check write permissions to the config directory
- Verify disk space is available

**Old paths still being used**
- Restart RamanLab after changing settings
- Check if any modules are using hardcoded paths (report as bug)

### Getting Help

- Run `python demo_configuration_system.py` for diagnostics
- Check the console output for configuration loading messages
- Settings → Preferences → Advanced shows current configuration

## Future Enhancements

Planned improvements:
- **Themes**: Dark/light theme support
- **Backup**: Built-in configuration backup/restore
- **Profiles**: Multiple configuration profiles
- **Cloud Sync**: Configuration synchronization across devices
- **Import/Export**: Share configuration between installations

---

*This configuration system provides the foundation for a more flexible and user-friendly RamanLab experience!* 