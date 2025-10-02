"""
Integration script for adding Interactive Mixture Analysis to RamanLab main interface.

This script demonstrates how to integrate the interactive mixture analysis
as a module launcher in the main RamanLab interface.
"""

import sys
import os
from pathlib import Path
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QFrame
from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QFont, QIcon

def create_interactive_mixture_analysis_launcher(parent_layout, workspace_root=None):
    """
    Create a launcher button for the Interactive Mixture Analysis.
    
    This function can be integrated into the main RamanLab interface
    to provide easy access to the interactive mixture analysis.
    
    Args:
        parent_layout: QLayout to add the launcher to
        workspace_root: Path to RamanLab workspace (optional)
    
    Returns:
        QPushButton: The created launcher button
    """
    
    # Create launcher frame
    launcher_frame = QFrame()
    launcher_frame.setFrameStyle(QFrame.Box)
    launcher_frame.setStyleSheet("""
        QFrame {
            border: 1px solid #cccccc;
            border-radius: 5px;
            background-color: #f9f9f9;
            padding: 5px;
        }
    """)
    
    # Layout for the launcher
    launcher_layout = QVBoxLayout(launcher_frame)
    
    # Title
    title_label = QLabel("🔬 Interactive Mixture Analysis")
    title_label.setFont(QFont("Arial", 12, QFont.Bold))
    title_label.setAlignment(Qt.AlignCenter)
    launcher_layout.addWidget(title_label)
    
    # Description
    desc_label = QLabel("Expert-guided iterative mixture analysis\nwith interactive peak selection")
    desc_label.setFont(QFont("Arial", 9))
    desc_label.setAlignment(Qt.AlignCenter)
    desc_label.setStyleSheet("color: #666666;")
    launcher_layout.addWidget(desc_label)
    
    # Launch button
    launch_btn = QPushButton("Launch Interactive Analysis")
    launch_btn.setFont(QFont("Arial", 10, QFont.Bold))
    launch_btn.setStyleSheet("""
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px;
            margin: 5px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QPushButton:pressed {
            background-color: #3d8b40;
        }
    """)
    
    # Connect button to launcher function
    def launch_interactive_analysis():
        """Launch the interactive mixture analysis."""
        try:
            # Get the current working directory
            if workspace_root:
                working_dir = str(workspace_root)
            else:
                working_dir = str(Path.cwd())
            
            # Launch the interactive analysis
            process = QProcess()
            script_path = Path(working_dir) / "launch_interactive_mixture_analysis.py"
            
            if script_path.exists():
                process.startDetached(sys.executable, [str(script_path)], working_dir)
                print("🚀 Launched Interactive Mixture Analysis")
            else:
                print(f"❌ Could not find launcher script: {script_path}")
                # Try alternative launch method
                try:
                    from launch_interactive_mixture_analysis import main
                    print("🔄 Launching directly...")
                    main()
                except ImportError as e:
                    print(f"❌ Could not launch interactive analysis: {e}")
                    
        except Exception as e:
            print(f"❌ Error launching interactive mixture analysis: {e}")
    
    launch_btn.clicked.connect(launch_interactive_analysis)
    launcher_layout.addWidget(launch_btn)
    
    # Features list
    features_label = QLabel("""
• Database search with top 10 matches
• Interactive peak selection by clicking
• Pseudo-Voigt peak fitting
• Iterative residual analysis
• Real-time fit quality monitoring
• Expert-guided component identification
    """.strip())
    features_label.setFont(QFont("Arial", 8))
    features_label.setStyleSheet("color: #444444; margin: 5px;")
    launcher_layout.addWidget(features_label)
    
    # Add to parent layout
    parent_layout.addWidget(launcher_frame)
    
    return launch_btn

def add_to_peak_fitting_qt6():
    """
    Example integration into peak_fitting_qt6.py interface.
    
    This shows how to add the interactive mixture analysis
    to the Advanced tab of the peak fitting interface.
    """
    
    integration_code = '''
    # Add this to the setup_advanced_tab() method in peak_fitting_qt6.py
    
    # Interactive Mixture Analysis section
    mixture_analysis_group = QGroupBox("Interactive Mixture Analysis")
    mixture_analysis_layout = QVBoxLayout(mixture_analysis_group)
    
    # Import the launcher function
    try:
        from integrate_interactive_mixture_analysis import create_interactive_mixture_analysis_launcher
        create_interactive_mixture_analysis_launcher(mixture_analysis_layout, self.workspace_root)
    except ImportError as e:
        error_label = QLabel(f"Interactive Mixture Analysis not available: {e}")
        error_label.setStyleSheet("color: red;")
        mixture_analysis_layout.addWidget(error_label)
    
    # Add to the advanced tab layout
    advanced_layout.addWidget(mixture_analysis_group)
    '''
    
    print("Integration code for peak_fitting_qt6.py:")
    print("=" * 60)
    print(integration_code)
    print("=" * 60)

def add_to_main_interface():
    """
    Example integration into main RamanLab interface.
    
    This shows how to add the interactive mixture analysis
    as a main module in raman_analysis_app_qt6.py or main_qt6.py.
    """
    
    integration_code = '''
    # Add this to the main interface setup in raman_analysis_app_qt6.py
    
    def setup_module_launchers(self):
        """Setup module launcher buttons."""
        
        # Create layout for module launchers
        modules_layout = QVBoxLayout()
        
        # Existing modules...
        # (spectrum analysis, polarization analysis, etc.)
        
        # Interactive Mixture Analysis
        try:
            from integrate_interactive_mixture_analysis import create_interactive_mixture_analysis_launcher
            create_interactive_mixture_analysis_launcher(modules_layout, self.workspace_root)
        except ImportError as e:
            print(f"Interactive Mixture Analysis not available: {e}")
        
        # Add modules layout to main interface
        self.main_layout.addLayout(modules_layout)
    '''
    
    print("Integration code for main interface:")
    print("=" * 60)
    print(integration_code)
    print("=" * 60)

def create_standalone_demo():
    """Create a standalone demo showing the launcher in action."""
    
    from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
    
    class DemoWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("RamanLab Module Integration Demo")
            self.setGeometry(100, 100, 400, 300)
            
            # Explicitly set window flags to ensure minimize/maximize/close buttons on Windows
            self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
            
            # Central widget
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            # Layout
            layout = QVBoxLayout(central_widget)
            
            # Title
            title = QLabel("RamanLab Module Integration Demo")
            title.setFont(QFont("Arial", 16, QFont.Bold))
            title.setAlignment(Qt.AlignCenter)
            layout.addWidget(title)
            
            # Add the interactive mixture analysis launcher
            create_interactive_mixture_analysis_launcher(layout)
            
            # Spacer
            layout.addStretch()
    
    return DemoWindow

if __name__ == "__main__":
    print("🔧 Interactive Mixture Analysis Integration Utility")
    print("=" * 60)
    
    # Show integration examples
    print("\n1. Integration code for peak_fitting_qt6.py:")
    add_to_peak_fitting_qt6()
    
    print("\n2. Integration code for main interface:")
    add_to_main_interface()
    
    print("\n3. Standalone demo:")
    print("   Run this script with --demo to see launcher demo")
    
    # Check for demo flag
    if "--demo" in sys.argv:
        from PySide6.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        
        # Create demo window
        demo_window_class = create_standalone_demo()
        demo_window = demo_window_class()
        demo_window.show()
        
        print("\n🎯 Demo window launched!")
        print("   Click the 'Launch Interactive Analysis' button to test")
        
        sys.exit(app.exec())
    
    else:
        print("\n✅ Integration utility ready!")
        print("   Add the integration code above to your RamanLab interface")
        print("   Or run with --demo to see a demonstration") 