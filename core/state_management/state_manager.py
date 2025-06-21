"""
Central application state manager for RamanLab.

This module coordinates state saving and restoration across all registered
modules and handles session management, auto-save, and crash recovery.
"""

from PySide6.QtCore import QSettings, QTimer, QObject, Signal
from PySide6.QtWidgets import QApplication, QMessageBox
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import shutil

from .stateful_module import StatefulModule
from .data_state import DataStateManager

class ApplicationStateManager(QObject):
    """
    Central coordinator for all application state management.
    
    This class manages the registration of stateful modules, coordinates
    state saving and restoration, and handles session lifecycle management.
    """
    
    # Signals for state management events
    state_save_started = Signal()
    state_save_completed = Signal(bool)  # success
    state_restore_started = Signal()
    state_restore_completed = Signal(bool)  # success
    auto_save_triggered = Signal()
    
    def __init__(self, app_instance):
        """
        Initialize the state manager.
        
        Args:
            app_instance: The main application instance
        """
        super().__init__()
        
        self.app = app_instance
        self.settings = QSettings("RamanLab", "RamanLab")
        self.registered_modules: Dict[str, StatefulModule] = {}
        
        # Setup directories
        self.session_dir = Path.home() / ".ramanlab" / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "auto_save").mkdir(exist_ok=True)
        (self.session_dir / "named_sessions").mkdir(exist_ok=True)
        (self.session_dir / "templates").mkdir(exist_ok=True)
        
        # Initialize data manager
        self.data_manager = DataStateManager(self.session_dir)
        
        # Session tracking
        self.current_session_id: Optional[str] = None
        self.session_modified = False
        
        # Auto-save configuration
        self.auto_save_timer = QTimer()
        self.auto_save_interval = 300000  # 5 minutes in milliseconds
        self.auto_save_enabled = True
        
        # Setup auto-save
        self.setup_auto_save()
        
        # Setup crash recovery check
        self.check_crash_recovery_on_startup()
        
        # Register for application exit
        if QApplication.instance():
            QApplication.instance().aboutToQuit.connect(self.on_application_exit)
        
        self.logger = logging.getLogger("state_manager")
    
    def register_module(self, module_id: str, module: StatefulModule) -> bool:
        """
        Register a module for state management.
        
        Args:
            module_id: Unique identifier for the module
            module: Module instance implementing StatefulModule
            
        Returns:
            True if registration was successful
        """
        try:
            if module_id in self.registered_modules:
                self.logger.warning(f"Module {module_id} is already registered")
                return False
            
            self.registered_modules[module_id] = module
            self.logger.info(f"Registered stateful module: {module_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register module {module_id}: {e}")
            return False
    
    def unregister_module(self, module_id: str) -> bool:
        """
        Unregister a module from state management.
        
        Args:
            module_id: Module identifier to unregister
            
        Returns:
            True if unregistration was successful
        """
        if module_id in self.registered_modules:
            del self.registered_modules[module_id]
            self.logger.info(f"Unregistered module: {module_id}")
            return True
        return False
    
    def save_application_state(self, session_name: Optional[str] = None) -> bool:
        """
        Save complete application state.
        
        Args:
            session_name: Optional name for the session. If None, saves as current session.
            
        Returns:
            True if save was successful
        """
        try:
            self.state_save_started.emit()
            
            # Create session metadata
            session_data = {
                "timestamp": datetime.now().isoformat(),
                "version": getattr(self.app, '__version__', 'unknown'),
                "ramanlab_version": getattr(self.app, 'version', 'unknown'),
                "modules": {},
                "global_settings": self.save_global_settings(),
                "session_id": session_name or self.current_session_id or "current"
            }
            
            # Save each module's state in dependency order
            save_order = self._calculate_save_order()
            successful_modules = []
            failed_modules = []
            
            for module_id in save_order:
                if module_id not in self.registered_modules:
                    continue
                    
                module = self.registered_modules[module_id]
                try:
                    self.logger.info(f"Saving state for module: {module_id}")
                    module_state = module.save_state()
                    
                    if module_state:
                        session_data["modules"][module_id] = {
                            "metadata": module.get_state_metadata(),
                            "state": module_state,
                            "version": module.state_version,
                            "timestamp": datetime.now().isoformat()
                        }
                        successful_modules.append(module_id)
                    else:
                        self.logger.warning(f"Module {module_id} returned empty state")
                        
                except Exception as e:
                    self.logger.error(f"Failed to save state for module {module_id}: {e}")
                    failed_modules.append(module_id)
                    # Continue with other modules
            
            # Save session data
            if session_name:
                success = self._save_named_session(session_name, session_data)
            else:
                success = self._save_current_session(session_data)
            
            if success:
                self.session_modified = False
                self.logger.info(f"State saved successfully. "
                               f"Modules: {len(successful_modules)} successful, "
                               f"{len(failed_modules)} failed")
            
            self.state_save_completed.emit(success)
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to save application state: {e}")
            self.state_save_completed.emit(False)
            return False
    
    def restore_application_state(self, session_name: Optional[str] = None) -> bool:
        """
        Restore complete application state.
        
        Args:
            session_name: Optional session name to restore. If None, restores current session.
            
        Returns:
            True if restoration was successful
        """
        try:
            self.state_restore_started.emit()
            
            # Load session data
            if session_name:
                session_data = self._load_named_session(session_name)
            else:
                session_data = self._load_current_session()
            
            if not session_data:
                self.logger.error("No session data found to restore")
                self.state_restore_completed.emit(False)
                return False
            
            # Validate session compatibility
            if not self._validate_session(session_data):
                self.logger.error("Session validation failed")
                self.state_restore_completed.emit(False)
                return False
            
            # Restore global settings first
            self._restore_global_settings(session_data.get("global_settings", {}))
            
            # Restore modules in dependency order
            restore_order = self._calculate_restore_order(session_data.get("modules", {}))
            successful_modules = []
            failed_modules = []
            
            # Call pre-restore hooks
            for module_id in restore_order:
                if module_id in self.registered_modules:
                    try:
                        self.registered_modules[module_id].pre_restore_hook()
                    except Exception as e:
                        self.logger.error(f"Pre-restore hook failed for {module_id}: {e}")
            
            # Restore each module
            for module_id in restore_order:
                if module_id not in self.registered_modules:
                    self.logger.warning(f"Module {module_id} not registered, skipping")
                    continue
                
                if module_id not in session_data["modules"]:
                    self.logger.warning(f"No state data for module {module_id}")
                    continue
                
                module = self.registered_modules[module_id]
                module_data = session_data["modules"][module_id]
                
                try:
                    self.logger.info(f"Restoring state for module: {module_id}")
                    
                    # Handle version migration if needed
                    state_data = module_data.get("state", {})
                    old_version = module_data.get("version", "1.0")
                    
                    if old_version != module.state_version:
                        self.logger.info(f"Migrating {module_id} from {old_version} to {module.state_version}")
                        state_data = module.migrate_state(state_data, old_version)
                    
                    # Validate state before restoration
                    if not module.validate_state(state_data):
                        self.logger.error(f"State validation failed for module {module_id}")
                        failed_modules.append(module_id)
                        continue
                    
                    # Restore the module
                    success = module.restore_state(state_data)
                    
                    if success:
                        successful_modules.append(module_id)
                    else:
                        failed_modules.append(module_id)
                        
                except Exception as e:
                    self.logger.error(f"Failed to restore state for module {module_id}: {e}")
                    failed_modules.append(module_id)
            
            # Call post-restore hooks
            for module_id in restore_order:
                if module_id in self.registered_modules:
                    try:
                        success = module_id in successful_modules
                        self.registered_modules[module_id].post_restore_hook(success)
                    except Exception as e:
                        self.logger.error(f"Post-restore hook failed for {module_id}: {e}")
            
            # Update session tracking
            self.current_session_id = session_data.get("session_id")
            self.session_modified = False
            
            overall_success = len(successful_modules) > 0
            self.logger.info(f"State restoration completed. "
                           f"Modules: {len(successful_modules)} successful, "
                           f"{len(failed_modules)} failed")
            
            self.state_restore_completed.emit(overall_success)
            return overall_success
            
        except Exception as e:
            self.logger.error(f"Failed to restore application state: {e}")
            self.state_restore_completed.emit(False)
            return False
    
    def setup_auto_save(self):
        """Setup automatic state saving."""
        self.auto_save_timer.timeout.connect(self.auto_save)
        if self.auto_save_enabled:
            self.auto_save_timer.start(self.auto_save_interval)
    
    def auto_save(self):
        """Perform automatic save for crash recovery."""
        try:
            if not self.session_modified:
                return  # No changes to save
            
            self.auto_save_triggered.emit()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            auto_save_file = self.session_dir / "auto_save" / f"auto_save_{timestamp}.json"
            
            # Save current state
            if self.save_application_state():
                # Copy to auto-save location
                current_session = self.session_dir / "current_session.json"
                if current_session.exists():
                    shutil.copy2(current_session, auto_save_file)
                    self.logger.info(f"Auto-save completed: {auto_save_file}")
            
            # Clean up old auto-saves
            self._cleanup_auto_saves()
            
        except Exception as e:
            self.logger.error(f"Auto-save failed: {e}")
    
    def _cleanup_auto_saves(self, keep_count: int = 10):
        """Clean up old auto-save files."""
        auto_save_dir = self.session_dir / "auto_save"
        auto_saves = sorted(auto_save_dir.glob("auto_save_*.json"))
        
        if len(auto_saves) > keep_count:
            for old_save in auto_saves[:-keep_count]:
                try:
                    old_save.unlink()
                except Exception as e:
                    self.logger.error(f"Failed to delete old auto-save {old_save}: {e}")
    
    def check_crash_recovery_on_startup(self):
        """Check if crash recovery is needed on startup."""
        try:
            # Look for auto-save files
            auto_save_dir = self.session_dir / "auto_save"
            auto_saves = list(auto_save_dir.glob("auto_save_*.json"))
            
            if not auto_saves:
                return
            
            # Check if last session was properly closed
            last_shutdown_file = self.session_dir / ".last_clean_shutdown"
            
            if not last_shutdown_file.exists():
                self._offer_crash_recovery()
                return
            
            # Compare timestamps
            latest_auto_save = max(auto_saves, key=lambda x: x.stat().st_mtime)
            last_shutdown_time = last_shutdown_file.stat().st_mtime
            
            if latest_auto_save.stat().st_mtime > last_shutdown_time:
                self._offer_crash_recovery()
                
        except Exception as e:
            self.logger.error(f"Crash recovery check failed: {e}")
    
    def _offer_crash_recovery(self):
        """Show crash recovery dialog."""
        try:
            reply = QMessageBox.question(
                None, "Crash Recovery",
                "It appears RamanLab didn't close properly last time. "
                "Would you like to restore your work from the auto-save?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Find latest auto-save
                auto_save_dir = self.session_dir / "auto_save"
                auto_saves = sorted(auto_save_dir.glob("auto_save_*.json"))
                
                if auto_saves:
                    latest_auto_save = auto_saves[-1]
                    # Copy to current session and restore
                    current_session = self.session_dir / "current_session.json"
                    shutil.copy2(latest_auto_save, current_session)
                    
                    # Restore after a short delay to let UI initialize
                    QTimer.singleShot(1000, lambda: self.restore_application_state())
                    
        except Exception as e:
            self.logger.error(f"Crash recovery offer failed: {e}")
    
    def on_application_exit(self):
        """Handle application exit."""
        try:
            # Save current state
            self.save_application_state()
            
            # Mark clean shutdown
            last_shutdown_file = self.session_dir / ".last_clean_shutdown"
            last_shutdown_file.touch()
            
        except Exception as e:
            self.logger.error(f"Exit save failed: {e}")
    
    def mark_session_modified(self):
        """Mark the current session as modified."""
        self.session_modified = True
    
    def has_previous_session(self) -> bool:
        """Check if there's a previous session to restore."""
        current_session = self.session_dir / "current_session.json"
        return current_session.exists()
    
    def save_global_settings(self) -> Dict[str, Any]:
        """Save global application settings."""
        # This can be expanded to include global preferences
        return {
            "auto_save_enabled": self.auto_save_enabled,
            "auto_save_interval": self.auto_save_interval
        }
    
    def _restore_global_settings(self, settings: Dict[str, Any]):
        """Restore global application settings."""
        self.auto_save_enabled = settings.get("auto_save_enabled", True)
        self.auto_save_interval = settings.get("auto_save_interval", 300000)
        
        # Restart auto-save timer with new interval
        if self.auto_save_enabled:
            self.auto_save_timer.start(self.auto_save_interval)
        else:
            self.auto_save_timer.stop()
    
    def _calculate_save_order(self) -> List[str]:
        """Calculate the order in which modules should be saved."""
        # For now, use registration order
        # Could be enhanced with dependency analysis
        return list(self.registered_modules.keys())
    
    def _calculate_restore_order(self, modules_data: Dict[str, Any]) -> List[str]:
        """Calculate the order in which modules should be restored."""
        # Simple dependency-based ordering
        # This could be enhanced with a proper topological sort
        available_modules = set(modules_data.keys()) & set(self.registered_modules.keys())
        ordered = []
        
        # Add modules with no dependencies first
        for module_id in available_modules:
            if module_id in self.registered_modules:
                deps = self.registered_modules[module_id].get_dependencies()
                if not any(dep in available_modules for dep in deps):
                    ordered.append(module_id)
        
        # Add remaining modules
        remaining = available_modules - set(ordered)
        ordered.extend(remaining)
        
        return ordered
    
    def _validate_session(self, session_data: Dict[str, Any]) -> bool:
        """Validate session data for consistency."""
        required_fields = ["timestamp", "version", "modules"]
        
        for field in required_fields:
            if field not in session_data:
                self.logger.error(f"Missing required field: {field}")
                return False
        
        return True
    
    def _save_current_session(self, session_data: Dict[str, Any]) -> bool:
        """Save session as current session."""
        try:
            current_session_file = self.session_dir / "current_session.json"
            with open(current_session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save current session: {e}")
            return False
    
    def _save_named_session(self, name: str, session_data: Dict[str, Any]) -> bool:
        """Save session with a specific name."""
        try:
            session_file = self.session_dir / "named_sessions" / f"{name}.json"
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save named session {name}: {e}")
            return False
    
    def _load_current_session(self) -> Optional[Dict[str, Any]]:
        """Load the current session."""
        try:
            current_session_file = self.session_dir / "current_session.json"
            if not current_session_file.exists():
                return None
                
            with open(current_session_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load current session: {e}")
            return None
    
    def _load_named_session(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a named session."""
        try:
            session_file = self.session_dir / "named_sessions" / f"{name}.json"
            if not session_file.exists():
                return None
                
            with open(session_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load named session {name}: {e}")
            return None 