"""
Abstract base class for stateful modules in RamanLab.

All modules that want to participate in state management must inherit
from StatefulModule and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

class StatefulModule(ABC):
    """
    Abstract base class for modules that support state saving/loading.
    
    All modules that want to participate in application state management
    must inherit from this class and implement the required abstract methods.
    """
    
    def __init__(self, module_id: str):
        """
        Initialize the stateful module.
        
        Args:
            module_id: Unique identifier for this module
        """
        self.module_id = module_id
        self.state_version = "1.0"  # For backwards compatibility
        self.logger = logging.getLogger(f"state.{module_id}")
    
    @abstractmethod
    def save_state(self) -> Dict[str, Any]:
        """
        Save the current state of the module.
        
        This method should capture all necessary information to restore
        the module to its current state, including:
        - Data objects (with references for large data)
        - UI state and preferences
        - Analysis parameters and results
        - Window geometry and layout
        
        Returns:
            Dict containing all state information. Should be JSON-serializable
            where possible, with references to external data files for large objects.
        """
        pass
    
    @abstractmethod
    def restore_state(self, state_data: Dict[str, Any]) -> bool:
        """
        Restore the module to the given state.
        
        This method should restore the module to the state described by
        the state_data dictionary. It should handle missing data gracefully
        and provide meaningful error messages.
        
        Args:
            state_data: State dictionary from save_state()
            
        Returns:
            True if restoration was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def validate_state(self, state_data: Dict[str, Any]) -> bool:
        """
        Validate that state data is compatible with current module version.
        
        This method should check:
        - Required fields are present
        - Data types are correct
        - Referenced files exist
        - Version compatibility
        
        Args:
            state_data: State dictionary to validate
            
        Returns:
            True if state is valid and can be restored, False otherwise
        """
        pass
    
    @abstractmethod
    def get_state_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the current state.
        
        This method should provide information about the module's current
        state without actually saving it. Useful for displaying session
        information and estimating save sizes.
        
        Returns:
            Dict with metadata like:
            - version: Module state version
            - data_size: Estimated size in bytes
            - has_data: Whether module has data to save
            - dependencies: Other modules this depends on
        """
        pass
    
    def get_state_size_estimate(self, state_data: Dict[str, Any]) -> int:
        """
        Estimate the size of state data in bytes.
        
        This method provides a rough estimate of how much storage space
        the state data will require. Used for progress bars and warnings.
        
        Args:
            state_data: State data to estimate size for
            
        Returns:
            Estimated size in bytes
        """
        try:
            import sys
            return sys.getsizeof(str(state_data))
        except Exception:
            return 0
    
    def migrate_state(self, old_state: Dict[str, Any], old_version: str) -> Dict[str, Any]:
        """
        Migrate state from older version to current version.
        
        Override this method to handle backwards compatibility when
        the state format changes between versions.
        
        Args:
            old_state: State data from older version
            old_version: Version string of the old state
            
        Returns:
            Migrated state data compatible with current version
        """
        self.logger.info(f"No migration needed from version {old_version}")
        return old_state
    
    def cleanup_state_files(self, state_data: Dict[str, Any]) -> None:
        """
        Clean up any external files associated with this state.
        
        This method is called when a session is deleted and should
        remove any external data files that are no longer needed.
        
        Args:
            state_data: State data containing file references
        """
        # Default implementation does nothing
        pass
    
    def get_dependencies(self) -> list[str]:
        """
        Get list of module IDs that this module depends on.
        
        Returns:
            List of module IDs that should be restored before this module
        """
        return []
    
    def pre_restore_hook(self) -> None:
        """
        Called before state restoration begins.
        
        Override this method to perform any necessary cleanup or
        preparation before the state is restored.
        """
        pass
    
    def post_restore_hook(self, success: bool) -> None:
        """
        Called after state restoration completes.
        
        Override this method to perform any necessary finalization
        after the state has been restored.
        
        Args:
            success: Whether the restoration was successful
        """
        pass 