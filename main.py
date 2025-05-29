#!/usr/bin/env python3
# Raman Spectrum Analysis Tool - Main Application Runner

import sys
import importlib

# Force reload any modules that might be cached
if 'raman_polarization_analyzer' in sys.modules:
    importlib.reload(sys.modules['raman_polarization_analyzer'])

import tkinter as tk
from tkinter import ttk
from raman_analysis_app import RamanAnalysisApp


def main():
    """
    Main function to run the Raman Spectrum Analysis Tool application.
    """
    # Create the root window
    root = tk.Tk()
    root.title("Raman Spectrum Analyzer")
    
    # Set window icon if available
    try:
        root.iconbitmap("raman_icon.ico")
    except:
        pass
    
    # Configure initial size and position
    root.geometry("1400x800")
    
    # Make the window resizable
    root.resizable(True, True)
    
    # Apply a theme for consistent styling
    try:
        style = ttk.Style(root)
        available_themes = style.theme_names()
        if 'clam' in available_themes:
             style.theme_use('clam')
        elif 'aqua' in available_themes:
             style.theme_use('aqua')
    except Exception as e:
        print(f"Could not set theme: {e}")
    
    # Create the application instance
    app = RamanAnalysisApp(root)
    
    # Create the menu bar (this ensures the help menu with version checking is available)
    app.create_menu_bar()
    
    # Start the main event loop
    root.mainloop()


if __name__ == "__main__":
    main()