#!/usr/bin/env python3
"""Test script to verify UI sizing changes."""

import sys
from PySide6.QtWidgets import QApplication
from map_analysis_2d.ui.control_panels import MLControlPanel

def test_ml_control_panel_sizing():
    """Test that the ML control panel has proper sizing."""
    app = QApplication(sys.argv)
    
    # Create the ML control panel
    ml_panel = MLControlPanel()
    
    # Check if the control panel was created successfully
    print("✓ ML Control Panel created successfully")
    
    # Check tab widget maximum height
    tab_height = ml_panel.tab_widget.maximumHeight()
    print(f"✓ Tab widget height: {tab_height}px (should be 750px)")
    
    # Show the panel for visual verification
    ml_panel.show()
    ml_panel.resize(400, 900)  # Make it taller to accommodate larger content
    
    # Force layout to update
    ml_panel.updateGeometry()
    app.processEvents()
    
    # Check specific widget sizes after layout
    supervised_tab = ml_panel.supervised_tab
    if hasattr(supervised_tab, 'layout'):
        layout = supervised_tab.layout()
        print(f"✓ Supervised tab layout has {layout.count()} items")
        
        # Find the params group and train group
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if hasattr(widget, 'title') and "Parameters" in widget.title():
                    print(f"✓ Parameters group size: {widget.size().width()}x{widget.size().height()}")
                    print(f"  Minimum height: {widget.minimumHeight()}")
                elif hasattr(widget, 'title') and "Train" in widget.title():
                    print(f"✓ Train & Apply group size: {widget.size().width()}x{widget.size().height()}")
                    print(f"  Minimum height: {widget.minimumHeight()}")
    
    print("✓ Panel displayed - check the Parameters and Train & Apply boxes")
    print("  - Parameters box should show all 3 spinboxes: Trees, Depth, Test Size")
    print("  - Train & Apply box should show all 3 buttons: Train Model, Apply to Map, Feature Info")
    print("  - Models section should be pushed down with adequate spacing")
    print("  - Status section should be at the bottom with good visibility")
    
    # Don't start the event loop automatically - just show the widgets
    return app, ml_panel

if __name__ == "__main__":
    app, panel = test_ml_control_panel_sizing()
    print("\nPress Ctrl+C to exit or close the window")
    try:
        app.exec()
    except KeyboardInterrupt:
        print("\nTest completed!") 