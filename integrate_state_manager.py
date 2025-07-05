#!/usr/bin/env python3
"""
Integration Helper for Universal State Manager
Easily integrate state management into existing batch peak fitting module
"""

from core.universal_state_manager import (
    get_state_manager, register_module, save_module_state, 
    load_module_state, auto_save_module, StateSerializerInterface
)
from PySide6.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, QLineEdit, QInputDialog

def integrate_batch_peak_fitting(batch_peak_fitting_instance):
    """
    Integrate Universal State Manager with BatchPeakFittingQt6 instance
    This adds save/load capabilities and auto-save functionality
    """
    
    # Register the module with the state manager
    state_manager = get_state_manager()
    register_module('batch_peak_fitting', batch_peak_fitting_instance)
    
    # Add save/load methods to the instance
    batch_peak_fitting_instance.save_analysis_state = lambda notes="": save_module_state('batch_peak_fitting', notes)
    batch_peak_fitting_instance.load_analysis_state = lambda: load_module_state('batch_peak_fitting')
    batch_peak_fitting_instance.auto_save_state = lambda: auto_save_module('batch_peak_fitting')
    
    # Enhance existing methods with auto-save hooks
    _add_auto_save_hooks(batch_peak_fitting_instance)
    
    # Add menu items for save/load
    _add_state_menu_items(batch_peak_fitting_instance)
    
    print("✅ BatchPeakFittingQt6 successfully integrated with Universal State Manager")
    print("📁 Auto-save location: ~/RamanLab_Projects/auto_saves/")
    print("💾 Use Ctrl+S to quick save, Ctrl+O to load")
    
    return state_manager

def _add_auto_save_hooks(instance):
    """Add auto-save hooks to critical methods"""
    
    # Hook into manual fit saving
    if hasattr(instance, 'update_batch_results_with_manual_fit'):
        original_method = instance.update_batch_results_with_manual_fit
        
        def enhanced_method(*args, **kwargs):
            result = original_method(*args, **kwargs)
            # Auto-save after manual adjustments
            auto_save_module('batch_peak_fitting')
            return result
        
        instance.update_batch_results_with_manual_fit = enhanced_method
    
    # Hook into reference setting
    if hasattr(instance, 'set_reference'):
        original_method = instance.set_reference
        
        def enhanced_method(*args, **kwargs):
            result = original_method(*args, **kwargs)
            # Auto-save after setting reference
            auto_save_module('batch_peak_fitting')
            return result
        
        instance.set_reference = enhanced_method
    
    # Hook into batch completion
    if hasattr(instance, 'apply_to_all'):
        original_method = instance.apply_to_all
        
        def enhanced_method(*args, **kwargs):
            result = original_method(*args, **kwargs)
            # Auto-save after batch completion
            save_module_state('batch_peak_fitting', "Batch analysis completed")
            return result
        
        instance.apply_to_all = enhanced_method

def _add_state_menu_items(instance):
    """Add save/load menu items to the existing menu bar"""
    
    if hasattr(instance, 'menu_bar'):
        # Add File menu if it doesn't exist
        file_menu = None
        for action in instance.menu_bar.actions():
            if action.text() == "File":
                file_menu = action.menu()
                break
        
        if not file_menu:
            file_menu = instance.menu_bar.addMenu("File")
        
        # Add separator
        file_menu.addSeparator()
        
        # Add save state action
        save_action = file_menu.addAction("💾 Save Analysis State")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(lambda: _show_save_dialog(instance))
        
        # Add load state action
        load_action = file_menu.addAction("📂 Load Analysis State") 
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(lambda: _show_load_dialog(instance))
        
        # Add create project action
        project_action = file_menu.addAction("📁 Create New Project")
        project_action.triggered.connect(lambda: _show_project_dialog(instance))

def _show_save_dialog(instance):
    """Show save dialog with notes"""
    notes, ok = QInputDialog.getText(
        instance, 
        "Save Analysis State", 
        "Enter notes for this save (optional):",
        text="Manual save"
    )
    
    if ok:
        success = save_module_state('batch_peak_fitting', notes)
        if success:
            QMessageBox.information(
                instance, 
                "Save Successful", 
                f"Analysis state saved successfully!\n\nNotes: {notes}\n\n"
                f"Your complete analysis including:\n"
                f"• All spectrum files\n"
                f"• Batch results\n" 
                f"• Manual adjustments\n"
                f"• Reference settings\n"
                f"• UI preferences\n\n"
                f"...has been preserved and can be restored later."
            )
        else:
            QMessageBox.warning(instance, "Save Failed", "Failed to save analysis state.")

def _show_load_dialog(instance):
    """Show load confirmation dialog"""
    reply = QMessageBox.question(
        instance,
        "Load Analysis State",
        "This will restore your complete analysis state including:\n"
        "• All spectrum files and batch results\n"
        "• Manual adjustments and reference settings\n"
        "• UI preferences\n\n"
        "Any unsaved changes will be lost.\n\n"
        "Do you want to continue?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )
    
    if reply == QMessageBox.Yes:
        success = load_module_state('batch_peak_fitting')
        if success:
            QMessageBox.information(
                instance,
                "Load Successful", 
                "Analysis state restored successfully!\n\n"
                "Your complete analysis has been restored exactly\n"
                "as it was when last saved."
            )
        else:
            QMessageBox.warning(instance, "Load Failed", "Failed to load analysis state.")

def _show_project_dialog(instance):
    """Show create project dialog"""
    
    class ProjectDialog(QDialog):
        def __init__(self, parent):
            super().__init__(parent)
            self.setWindowTitle("Create New Analysis Project")
            self.setModal(True)
            self.resize(400, 200)
            
            layout = QVBoxLayout(self)
            
            # Project name
            layout.addWidget(QLabel("Project Name:"))
            self.name_edit = QLineEdit()
            self.name_edit.setPlaceholderText("e.g., Hilairite_Analysis_2025")
            layout.addWidget(self.name_edit)
            
            # Description
            layout.addWidget(QLabel("Description (optional):"))
            self.desc_edit = QTextEdit()
            self.desc_edit.setPlaceholderText("Describe your analysis project...")
            self.desc_edit.setMaximumHeight(80)
            layout.addWidget(self.desc_edit)
            
            # Buttons
            button_layout = QHBoxLayout()
            
            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(self.reject)
            button_layout.addWidget(cancel_btn)
            
            create_btn = QPushButton("Create Project")
            create_btn.clicked.connect(self.accept)
            create_btn.setDefault(True)
            button_layout.addWidget(create_btn)
            
            layout.addLayout(button_layout)
    
    dialog = ProjectDialog(instance)
    if dialog.exec() == QDialog.Accepted:
        project_name = dialog.name_edit.text().strip()
        description = dialog.desc_edit.toPlainText().strip()
        
        if not project_name:
            QMessageBox.warning(instance, "Invalid Input", "Please enter a project name.")
            return
        
        state_manager = get_state_manager()
        project_path = state_manager.create_project(project_name, description)
        
        QMessageBox.information(
            instance,
            "Project Created",
            f"Project '{project_name}' created successfully!\n\n"
            f"Location: {project_path}\n\n"
            f"All future saves will be organized under this project.\n"
            f"You can now save your analysis state with Ctrl+S."
        )

# Additional serializers for other modules (examples)
class PolarizationAnalysisSerializer(StateSerializerInterface):
    """Example serializer for polarization analysis module"""
    
    def serialize_state(self, module_instance):
        return {
            'module_type': 'polarization_analysis',
            'version': '1.0.0',
            'crystal_structure': getattr(module_instance, 'crystal_structure', {}),
            'orientation_data': getattr(module_instance, 'orientation_data', {}),
            'polarization_results': getattr(module_instance, 'polarization_results', []),
            'analysis_settings': getattr(module_instance, 'analysis_settings', {})
        }
    
    def deserialize_state(self, state_data, module_instance):
        try:
            module_instance.crystal_structure = state_data.get('crystal_structure', {})
            module_instance.orientation_data = state_data.get('orientation_data', {})
            module_instance.polarization_results = state_data.get('polarization_results', [])
            module_instance.analysis_settings = state_data.get('analysis_settings', {})
            return True
        except Exception as e:
            print(f"Error deserializing polarization analysis: {e}")
            return False
    
    def get_state_summary(self, state_data):
        results = state_data.get('polarization_results', [])
        crystal = state_data.get('crystal_structure', {}).get('name', 'Unknown')
        return f"Polarization Analysis\n├─ Results: {len(results)} orientations\n└─ Crystal: {crystal}"
    
    def validate_state(self, state_data):
        return state_data.get('module_type') == 'polarization_analysis'

def integrate_polarization_analysis(polarization_instance):
    """Integrate polarization analysis module"""
    state_manager = get_state_manager()
    serializer = PolarizationAnalysisSerializer()
    register_module('polarization_analysis', polarization_instance, serializer)
    
    # Add convenience methods
    polarization_instance.save_analysis_state = lambda notes="": save_module_state('polarization_analysis', notes)
    polarization_instance.load_analysis_state = lambda: load_module_state('polarization_analysis')
    
    print("✅ Polarization Analysis integrated with Universal State Manager")

class ClusterAnalysisSerializer(StateSerializerInterface):
    """Example serializer for cluster analysis module"""
    
    def serialize_state(self, module_instance):
        return {
            'module_type': 'cluster_analysis',
            'version': '1.0.0',
            'cluster_data': getattr(module_instance, 'cluster_data', {}),
            'pca_results': getattr(module_instance, 'pca_results', {}),
            'nmf_results': getattr(module_instance, 'nmf_results', {}),
            'analysis_parameters': getattr(module_instance, 'analysis_parameters', {})
        }
    
    def deserialize_state(self, state_data, module_instance):
        try:
            module_instance.cluster_data = state_data.get('cluster_data', {})
            module_instance.pca_results = state_data.get('pca_results', {})
            module_instance.nmf_results = state_data.get('nmf_results', {})
            module_instance.analysis_parameters = state_data.get('analysis_parameters', {})
            return True
        except Exception as e:
            print(f"Error deserializing cluster analysis: {e}")
            return False
    
    def get_state_summary(self, state_data):
        cluster_data = state_data.get('cluster_data', {})
        n_clusters = cluster_data.get('n_clusters', 0)
        method = cluster_data.get('method', 'Unknown')
        return f"Cluster Analysis\n├─ Clusters: {n_clusters}\n└─ Method: {method}"
    
    def validate_state(self, state_data):
        return state_data.get('module_type') == 'cluster_analysis'

def integrate_cluster_analysis(cluster_instance):
    """Integrate cluster analysis module"""
    state_manager = get_state_manager()
    serializer = ClusterAnalysisSerializer()
    register_module('cluster_analysis', cluster_instance, serializer)
    
    # Add convenience methods
    cluster_instance.save_analysis_state = lambda notes="": save_module_state('cluster_analysis', notes)
    cluster_instance.load_analysis_state = lambda: load_module_state('cluster_analysis')
    
    print("✅ Cluster Analysis integrated with Universal State Manager")

# Usage example for batch peak fitting
def example_integration():
    """Example of how to integrate with batch peak fitting"""
    
    # In your BatchPeakFittingQt6.__init__ method, add this line:
    # integrate_batch_peak_fitting(self)
    
    # That's it! The integration will:
    # 1. Register your module with the state manager
    # 2. Add auto-save hooks to critical methods
    # 3. Add save/load menu items with keyboard shortcuts
    # 4. Handle all serialization/deserialization automatically
    
    print("""
To integrate with your existing BatchPeakFittingQt6 class:

1. Import the integration function:
   from integrate_state_manager import integrate_batch_peak_fitting

2. Add one line to your __init__ method:
   integrate_batch_peak_fitting(self)

3. That's it! You now have:
   ✅ Auto-save after manual adjustments
   ✅ Auto-save after setting references  
   ✅ Auto-save after batch completion
   ✅ Ctrl+S to save with notes
   ✅ Ctrl+O to load previous state
   ✅ Project management
   ✅ Complete state preservation
   
The system will automatically save:
• All your spectrum files
• Batch fitting results  
• Manual adjustments
• Reference settings
• Background parameters
• UI preferences
• Peak fitting parameters

Your analysis state will persist across:
• App restarts
• System crashes
• Computer shutdowns
• Moving between computers (via project export)
""")

if __name__ == "__main__":
    example_integration() 