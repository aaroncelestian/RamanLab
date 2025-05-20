"""
Database Helper Functions

This module provides helper functions for database operations and 
fixes common issues with pandas DataFrames and database operations.
"""

import pandas as pd


def safe_launch_database_browser(parent=None):
    """
    Safely launch the mineral database browser with proper DataFrame handling.
    
    This function handles the case where a DataFrame's truth value is checked,
    which would normally cause an "ambiguous truth value" error.
    
    Parameters:
    -----------
    parent : tkinter.Tk or tkinter.Toplevel, optional
        Parent window for embedding the database browser
        
    Returns:
    --------
    bool
        True if successful, False if an error occurred
    """
    try:
        from .mineral_database import MineralDatabaseGUI
        
        # Create the database GUI with the parent window
        db_gui = MineralDatabaseGUI(parent=parent)
        
        # Ensure is_standalone is a proper boolean
        if hasattr(db_gui, 'is_standalone'):
            # Check if it's a DataFrame (has .empty attribute)
            if hasattr(db_gui.is_standalone, 'empty'):
                db_gui.is_standalone = bool(not db_gui.is_standalone.empty)
            else:
                # Make sure it's a proper boolean
                db_gui.is_standalone = bool(db_gui.is_standalone)
        
        # If no parent, run in standalone mode
        if parent is None:
            db_gui.run()
        
        # Store reference to prevent garbage collection if embedded
        elif hasattr(parent, 'db_gui'):
            parent.db_gui = db_gui
        
        return True
    except Exception as e:
        import traceback
        print(f"Error launching mineral database browser: {str(e)}")
        traceback.print_exc()
        return False


def is_dataframe_empty(df):
    """
    Safely check if a DataFrame is empty.
    
    Parameters:
    -----------
    df : pandas.DataFrame or any
        DataFrame or other object to check
        
    Returns:
    --------
    bool
        True if the DataFrame is empty or not a DataFrame, False otherwise
    """
    if isinstance(df, pd.DataFrame):
        return df.empty
    # If it's a Series
    elif isinstance(df, pd.Series):
        return df.empty
    # Default for non-pandas objects
    return not bool(df)


def to_dataframe(data):
    """
    Safely convert data to a DataFrame if it isn't already.
    
    Parameters:
    -----------
    data : any
        Data to convert to a DataFrame
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame representation of the data
    """
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, pd.Series):
        return pd.DataFrame(data)
    elif isinstance(data, dict):
        return pd.DataFrame.from_dict(data, orient='index')
    elif isinstance(data, list):
        return pd.DataFrame(data)
    else:
        # Try to convert other types
        try:
            return pd.DataFrame([data])
        except:
            return pd.DataFrame() 