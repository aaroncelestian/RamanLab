#!/usr/bin/env python3
"""
RamanLab Configuration Manager
Handles user preferences and application settings
"""

import json
import os
from pathlib import Path
from PySide6.QtCore import QStandardPaths


class ConfigManager:
    """Manages user configuration and application settings."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.config_dir = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation))
        self.config_file = self.config_dir / "ramanlab_config.json"
        self._config = self._load_default_config()
        self.load_config()
    
    def _load_default_config(self):
        """Load default configuration values."""
        return {
            "projects_folder": str(Path.home() / "RamanLab_Projects"),
            "auto_save_enabled": True,
            "auto_save_interval": 300,  # seconds
            "theme": "default",
            "window_geometry": {},
            "recent_files": [],
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
    
    def load_config(self):
        """Load configuration from file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults to handle new config options
                self._config.update(loaded_config)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            print("Using default configuration")
    
    def save_config(self):
        """Save configuration to file."""
        try:
            # Ensure config directory exists
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def get(self, key, default=None):
        """Get a configuration value."""
        keys = key.split('.')
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key, value):
        """Set a configuration value."""
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        
        # Auto-save configuration
        self.save_config()
    
    def get_projects_folder(self):
        """Get the configured projects folder path."""
        path = Path(self.get("projects_folder"))
        # Ensure the folder exists
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def set_projects_folder(self, folder_path):
        """Set the projects folder path."""
        path = Path(folder_path)
        path.mkdir(parents=True, exist_ok=True)
        self.set("projects_folder", str(path))
    
    def get_auto_save_folder(self):
        """Get the auto-save folder path."""
        return self.get_projects_folder() / "auto_saves"
    
    def get_session_folder(self):
        """Get the session folder path (same as auto-save for now)."""
        return self.get_auto_save_folder()
    
    def add_recent_file(self, file_path):
        """Add a file to the recent files list."""
        recent = self.get("recent_files", [])
        file_path = str(file_path)
        
        # Remove if already exists
        if file_path in recent:
            recent.remove(file_path)
        
        # Add to beginning
        recent.insert(0, file_path)
        
        # Limit size
        max_files = self.get("max_recent_files", 10)
        recent = recent[:max_files]
        
        self.set("recent_files", recent)
    
    def get_recent_files(self):
        """Get the list of recent files."""
        return self.get("recent_files", [])
    
    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self._config = self._load_default_config()
        self.save_config()


# Global configuration instance
_config_manager = None

def get_config_manager():
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager 