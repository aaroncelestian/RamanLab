#!/usr/bin/env python3
"""
Prototype implementation of RamanLab Application State Management

This is a working prototype that demonstrates the core concepts
and can be used as a foundation for the full implementation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatefulModule(ABC):
    """Abstract base class for modules that can save/restore state"""
    
    def __init__(self, module_id: str):
        self.module_id = module_id
        self.state_version = "1.0"
    
    @abstractmethod
    def save_state(self) -> Dict[str, Any]:
        """Save current state - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def restore_state(self, state_data: Dict[str, Any]) -> bool:
        """Restore from saved state - must be implemented by subclasses"""
        pass
    
    def validate_state(self, state_data: Dict[str, Any]) -> bool:
        """Validate state data - can be overridden by subclasses"""
        return isinstance(state_data, dict)
    
    def get_state_metadata(self) -> Dict[str, Any]:
        """Get metadata about current state"""
        return {
            "module_id": self.module_id,
            "version": self.state_version,
            "timestamp": datetime.now().isoformat()
        }

class ApplicationStateManager:
    """Central coordinator for application state management"""
    
    def __init__(self):
        self.registered_modules: Dict[str, StatefulModule] = {}
        self.session_dir = Path.home() / ".ramanlab" / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"State manager initialized: {self.session_dir}")
    
    def register_module(self, module_id: str, module: StatefulModule) -> bool:
        """Register a module for state management"""
        if module_id in self.registered_modules:
            logger.warning(f"Module {module_id} already registered")
            return False
        
        self.registered_modules[module_id] = module
        logger.info(f"Registered module: {module_id}")
        return True
    
    def save_session(self, session_name: str) -> bool:
        """Save current application state as named session"""
        try:
            logger.info(f"Saving session: {session_name}")
            
            # Collect state from all modules
            session_data = {
                "session_name": session_name,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
                "modules": {}
            }
            
            # Save each module's state
            for module_id, module in self.registered_modules.items():
                try:
                    module_state = module.save_state()
                    session_data["modules"][module_id] = {
                        "metadata": module.get_state_metadata(),
                        "state": module_state
                    }
                    logger.info(f"Saved state for module: {module_id}")
                except Exception as e:
                    logger.error(f"Failed to save {module_id}: {e}")
                    return False
            
            # Write session file
            session_file = self.session_dir / f"{session_name}.json"
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            logger.info(f"Session saved successfully: {session_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False
    
    def load_session(self, session_name: str) -> bool:
        """Load and restore named session"""
        try:
            logger.info(f"Loading session: {session_name}")
            
            # Load session file
            session_file = self.session_dir / f"{session_name}.json"
            if not session_file.exists():
                logger.error(f"Session file not found: {session_file}")
                return False
            
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Restore each module
            successful_modules = []
            failed_modules = []
            
            for module_id, module_data in session_data.get("modules", {}).items():
                if module_id not in self.registered_modules:
                    logger.warning(f"Module {module_id} not registered, skipping")
                    continue
                
                module = self.registered_modules[module_id]
                state_data = module_data.get("state", {})
                
                try:
                    if module.validate_state(state_data):
                        if module.restore_state(state_data):
                            successful_modules.append(module_id)
                            logger.info(f"Restored module: {module_id}")
                        else:
                            failed_modules.append(module_id)
                            logger.error(f"Failed to restore module: {module_id}")
                    else:
                        failed_modules.append(module_id)
                        logger.error(f"Invalid state for module: {module_id}")
                except Exception as e:
                    failed_modules.append(module_id)
                    logger.error(f"Error restoring {module_id}: {e}")
            
            success = len(successful_modules) > 0
            logger.info(f"Session restored: {len(successful_modules)} successful, {len(failed_modules)} failed")
            return success
            
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions"""
        sessions = []
        
        for session_file in self.session_dir.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                sessions.append({
                    "name": session_file.stem,
                    "timestamp": session_data.get("timestamp", "unknown"),
                    "modules": list(session_data.get("modules", {}).keys()),
                    "file_path": str(session_file)
                })
            except Exception as e:
                logger.error(f"Error reading session {session_file}: {e}")
        
        return sorted(sessions, key=lambda x: x["timestamp"], reverse=True)

# Example implementation for main analysis window
class MainAnalysisState(StatefulModule):
    """Example state handler for main analysis window"""
    
    def __init__(self):
        super().__init__("main_analysis")
        
        # Example state variables
        self.current_file = None
        self.zoom_level = 1.0
        self.processing_params = {
            "background_method": "als",
            "smoothing_window": 5
        }
        self.window_geometry = None
    
    def save_state(self) -> Dict[str, Any]:
        """Save main analysis window state"""
        return {
            "current_file": self.current_file,
            "zoom_level": self.zoom_level,
            "processing_params": self.processing_params.copy(),
            "window_geometry": self.window_geometry
        }
    
    def restore_state(self, state_data: Dict[str, Any]) -> bool:
        """Restore main analysis window state"""
        try:
            self.current_file = state_data.get("current_file")
            self.zoom_level = state_data.get("zoom_level", 1.0)
            self.processing_params = state_data.get("processing_params", {})
            self.window_geometry = state_data.get("window_geometry")
            
            logger.info(f"Restored main analysis state: file={self.current_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore main analysis state: {e}")
            return False

# Example implementation for map analysis
class MapAnalysisState(StatefulModule):
    """Example state handler for map analysis"""
    
    def __init__(self):
        super().__init__("map_analysis")
        
        # Example state variables
        self.map_file = None
        self.pca_components = 3
        self.templates = []
        self.analysis_results = {}
    
    def save_state(self) -> Dict[str, Any]:
        """Save map analysis state"""
        return {
            "map_file": self.map_file,
            "pca_components": self.pca_components,
            "templates": self.templates.copy(),
            "analysis_results": self.analysis_results.copy()
        }
    
    def restore_state(self, state_data: Dict[str, Any]) -> bool:
        """Restore map analysis state"""
        try:
            self.map_file = state_data.get("map_file")
            self.pca_components = state_data.get("pca_components", 3)
            self.templates = state_data.get("templates", [])
            self.analysis_results = state_data.get("analysis_results", {})
            
            logger.info(f"Restored map analysis state: map={self.map_file}, templates={len(self.templates)}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore map analysis state: {e}")
            return False

def demo_state_management():
    """Demonstrate the state management system"""
    print("\n🧪 RamanLab State Management Prototype Demo")
    print("=" * 50)
    
    # Create state manager
    state_manager = ApplicationStateManager()
    
    # Create and register modules
    main_state = MainAnalysisState()
    map_state = MapAnalysisState()
    
    state_manager.register_module("main_analysis", main_state)
    state_manager.register_module("map_analysis", map_state)
    
    # Simulate some application state
    print("\n📊 Simulating application usage...")
    main_state.current_file = "example_spectrum.txt"
    main_state.zoom_level = 1.5
    main_state.processing_params["background_method"] = "snip"
    
    map_state.map_file = "raman_map_data.dat"
    map_state.pca_components = 5
    map_state.templates = ["quartz", "feldspar", "mica"]
    map_state.analysis_results = {"pca_variance": [0.6, 0.2, 0.1, 0.05, 0.03]}
    
    print(f"  • Loaded spectrum: {main_state.current_file}")
    print(f"  • Zoom level: {main_state.zoom_level}")
    print(f"  • Map file: {map_state.map_file}")
    print(f"  • Templates: {len(map_state.templates)}")
    
    # Save session
    print("\n💾 Saving session...")
    if state_manager.save_session("demo_analysis"):
        print("  ✅ Session saved successfully!")
    else:
        print("  ❌ Session save failed!")
        return
    
    # Simulate clearing state (as if restarting application)
    print("\n🔄 Simulating application restart...")
    main_state.current_file = None
    main_state.zoom_level = 1.0
    map_state.map_file = None
    map_state.templates = []
    print("  • State cleared")
    
    # Restore session
    print("\n📥 Restoring session...")
    if state_manager.load_session("demo_analysis"):
        print("  ✅ Session restored successfully!")
        print(f"  • Restored spectrum: {main_state.current_file}")
        print(f"  • Restored zoom: {main_state.zoom_level}")
        print(f"  • Restored map: {map_state.map_file}")
        print(f"  • Restored templates: {len(map_state.templates)}")
    else:
        print("  ❌ Session restore failed!")
    
    # List sessions
    print("\n📋 Available sessions:")
    sessions = state_manager.list_sessions()
    for session in sessions:
        print(f"  • {session['name']} ({session['timestamp']}) - {len(session['modules'])} modules")
    
    print("\n✅ Demo completed successfully!")
    print(f"📁 Session files stored in: {state_manager.session_dir}")

if __name__ == "__main__":
    demo_state_management() 