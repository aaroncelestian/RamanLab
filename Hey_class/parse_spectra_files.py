def update_hey_classification_from_spectra_files(self):
    """Update Hey Classification by parsing the original spectra files to extract mineral names."""
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    import os
    import re
    
    if not self.raman.database:
        messagebox.showinfo("Database", "The database is empty.")
        return
    
    # Ask for directory containing the original spectra files
    spectra_dir = filedialog.askdirectory(title="Select Directory with Original Spectrum Files")
    if not spectra_dir:
        return
    
    # Check if Hey Classification data is loaded
    if not hasattr(self.raman, 'hey_classification') or not self.raman.hey_classification:
        # Try to load Hey Classification data if not already loaded
        hey_csv_path = "RRUFF_Export_with_Hey_Classification.csv"
        if os.path.exists(hey_csv_path):
            self.raman.hey_classification = self.raman.load_hey_classification(hey_csv_path)
        
        # Check again if data was loaded
        if not hasattr(self.raman, 'hey_classification') or not self.raman.hey_classification:
            messagebox.showerror("Error", "Hey Classification data could not be loaded.")
            return
    
    # Create progress window
    progress_window = tk.Toplevel(self.root)
    progress_window.title("Updating Hey Classification")
    progress_window.geometry("600x400")
    
    # Create progress bar
    ttk.Label(progress_window, text="Updating Hey Classification by parsing spectra files...").pack(pady=10)
    progress = ttk.Progressbar(progress_window, length=400, mode="determinate")
    progress.pack(pady=10)
    
    # Create status label
    status_var = tk.StringVar(value="Starting update...")
    status_label = ttk.Label(progress_window, textvariable=status_var)
    status_label.pack(pady=5)
    
    # Create log text area
    log_frame = ttk.Frame(progress_window)
    log_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
    log_text = tk.Text(log_frame, height=15, width=60, wrap=tk.WORD)
    log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_text.yview)
    log_text.config(yscrollcommand=log_scrollbar.set)
    log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Configure tags for log text
    log_text.tag_configure("success", foreground="green")
    log_text.tag_configure("warning", foreground="orange")
    log_text.tag_configure("error", foreground="red")
    log_text.tag_configure("info", foreground="blue")
    
    # Get database items and update progress bar maximum
    db_items = list(self.raman.database.items())
    progress["maximum"] = len(db_items)
    
    # Statistics counters
    total = len(db_items)
    updated = 0
    already_had = 0
    not_found = 0
    file_not_found = 0
    
    # Create mapping from database entry name to spectra file
    file_mapping = {}
    
    # Function to find original file from database entry name
    def find_original_file(db_entry_name):
        # Try direct match first
        direct_match = os.path.join(spectra_dir, db_entry_name)
        if os.path.exists(direct_match):
            return direct_match
        
        # Try adding common extensions
        for ext in ['.txt', '.csv', '.dat', '.sp']:
            file_path = os.path.join(spectra_dir, db_entry_name + ext)
            if os.path.exists(file_path):
                return file_path
        
        # Extract RRUFF ID from database entry name, if present
        rruff_id = None
        if '__' in db_entry_name:
            parts = db_entry_name.split('__')
            for part in parts:
                if part.startswith('R') and part[1:].isdigit():
                    rruff_id = part
                    break
        
        # If we found an RRUFF ID, search for files containing it
        if rruff_id:
            for filename in os.listdir(spectra_dir):
                if rruff_id in filename:
                    return os.path.join(spectra_dir, filename)
        
        # Last resort: Try to match by searching for the prefix part
        if '__' in db_entry_name:
            prefix = db_entry_name.split('__')[0]
            for filename in os.listdir(spectra_dir):
                if filename.startswith(prefix):
                    return os.path.join(spectra_dir, filename)
        
        return None
    
    # Update function for progress
    def update_progress(current, name, status, log_message=None, tag="info"):
        progress["value"] = current
        percentage = int((current / total) * 100) if total > 0 else 0
        status_var.set(f"{status} - {percentage}% complete ({current}/{total})")
        
        if log_message:
            log_text.insert(tk.END, log_message + "\n", tag)
            log_text.see(tk.END)
        
        progress_window.update_idletasks()
    
    # Function to extract mineral name from spectra file
    def extract_mineral_name_from_file(file_path):
        try:
            with open(file_path, 'r', errors='replace') as f:
                # Read first 20 lines to look for metadata
                for i, line in enumerate(f):
                    if i >= 20:  # Only check first 20 lines
                        break
                    
                    line = line.strip()
                    
                    # Check for NAMES field
                    if line.startswith('##NAMES=') or line.startswith('#NAMES='):
                        return line.split('=', 1)[1].strip()
                    
                    # Check for NAME field
                    if line.startswith('##NAME=') or line.startswith('#NAME='):
                        return line.split('=', 1)[1].strip()
                    
                    # Check for MINERAL field
                    if line.startswith('##MINERAL=') or line.startswith('#MINERAL='):
                        return line.split('=', 1)[1].strip()
                    
                    # Check for colon format
                    if ':' in line and line.startswith('#'):
                        key, value = line[1:].split(':', 1)
                        key = key.strip().upper()
                        if key in ['NAME', 'MINERAL', 'MINERAL NAME', 'NAMES']:
                            return value.strip()
            
            # If we've reached here, check if the filename itself contains the mineral name
            filename = os.path.basename(file_path)
            # Look for patterns like "Actinolite_R123456"
            match = re.match(r'([A-Za-z]+)[_-]', filename)
            if match:
                return match.group(1)
            
            return None
        except Exception as e:
            update_progress(0, file_path, "Error", f"Error reading file {file_path}: {str(e)}", "error")
            return None
    
    # Process each database entry
    for i, (name, data) in enumerate(db_items):
        # Update progress display
        update_progress(i, name, f"Processing {name}")
        
        # Skip entries without metadata
        if 'metadata' not in data or not data['metadata']:
            data['metadata'] = {}  # Create metadata dict if it doesn't exist
        
        metadata = data['metadata']
        
        # Check if Hey Classification already exists
        if 'HEY CLASSIFICATION' in metadata and metadata['HEY CLASSIFICATION']:
            already_had += 1
            update_progress(i, name, f"Processing {name}", 
                          f"Skipped {name}: Already has Hey Classification '{metadata['HEY CLASSIFICATION']}'", "info")
            continue
        
        # Find the original spectra file
        spectra_file = find_original_file(name)
        if not spectra_file:
            file_not_found += 1
            update_progress(i, name, f"Processing {name}", 
                          f"Could not find original spectra file for {name}", "warning")
            continue
        
        # Extract mineral name from the file
        mineral_name = extract_mineral_name_from_file(spectra_file)
        if not mineral_name:
            update_progress(i, name, f"Processing {name}", 
                          f"Could not extract mineral name from file {os.path.basename(spectra_file)}", "warning")
            continue
        
        # Update the metadata with the extracted name
        metadata['NAME'] = mineral_name
        update_progress(i, name, f"Processing {name}", 
                      f"Extracted mineral name '{mineral_name}' from file {os.path.basename(spectra_file)}", "info")
        
        # Try to add Hey Classification using the mineral name
        hey_class = self.raman.get_hey_classification(mineral_name)
        
        if hey_class:
            # Update the metadata with Hey Classification
            metadata['HEY CLASSIFICATION'] = hey_class
            
            # Also derive chemical family if Hey Classification is found
            if ' - ' in hey_class:
                # Extract the first part as chemical family (before the hyphen)
                family = hey_class.split(' - ')[0].strip()
                metadata['CHEMICAL FAMILY'] = family
                update_progress(i, name, f"Processing {name}", 
                              f"Updated {name}: Added Hey Classification '{hey_class}' and Chemical Family '{family}'", "success")
            else:
                update_progress(i, name, f"Processing {name}", 
                              f"Updated {name}: Added Hey Classification '{hey_class}'", "success")
                              
            updated += 1
        else:
            # Try with cleaned name
            cleaned_name = self.raman._clean_mineral_name(mineral_name)
            if cleaned_name and cleaned_name != mineral_name:
                hey_class = self.raman.get_hey_classification(cleaned_name)
                
                if hey_class:
                    # Update the metadata with Hey Classification
                    metadata['HEY CLASSIFICATION'] = hey_class
                    
                    # Also derive chemical family
                    if ' - ' in hey_class:
                        family = hey_class.split(' - ')[0].strip()
                        metadata['CHEMICAL FAMILY'] = family
                        update_progress(i, name, f"Processing {name}", 
                                      f"Updated {name}: Added Hey Classification '{hey_class}' and Chemical Family '{family}' (using cleaned name)", "success")
                    else:
                        update_progress(i, name, f"Processing {name}", 
                                      f"Updated {name}: Added Hey Classification '{hey_class}' (using cleaned name)", "success")
                    
                    updated += 1
                    continue
            
            # Try with more aggressive cleaning
            very_clean_name = self.raman.extract_base_mineral_name(mineral_name)
            if very_clean_name and very_clean_name != mineral_name and very_clean_name != cleaned_name:
                hey_class = self.raman.get_hey_classification(very_clean_name)
                
                if hey_class:
                    # Update the metadata with Hey Classification
                    metadata['HEY CLASSIFICATION'] = hey_class
                    
                    # Also derive chemical family
                    if ' - ' in hey_class:
                        family = hey_class.split(' - ')[0].strip()
                        metadata['CHEMICAL FAMILY'] = family
                        update_progress(i, name, f"Processing {name}", 
                                      f"Updated {name}: Added Hey Classification '{hey_class}' and Chemical Family '{family}' (using aggressively cleaned name)", "success")
                    else:
                        update_progress(i, name, f"Processing {name}", 
                                      f"Updated {name}: Added Hey Classification '{hey_class}' (using aggressively cleaned name)", "success")
                    
                    updated += 1
                    continue
            
            # If we get here, we couldn't find a match
            not_found += 1
            update_progress(i, name, f"Processing {name}", 
                          f"Could not find Hey Classification for '{mineral_name}'", "warning")
    
    # Final progress update
    update_progress(total, "", "Complete", 
                  f"Update complete! Updated: {updated}, Already had: {already_had}, Not found: {not_found}, Files not found: {file_not_found}", "info")
    
    # Save the database
    saved = self.raman.save_database()
    if saved:
        log_text.insert(tk.END, f"Database saved successfully to {self.raman.db_path}\n", "success")
    else:
        log_text.insert(tk.END, "Warning: Could not save database\n", "error")
    log_text.see(tk.END)
    
    # Add close button
    ttk.Button(progress_window, text="Close", command=progress_window.destroy).pack(pady=10)
    
    # Update main GUI database statistics
    if hasattr(self, 'update_database_stats'):
        self.update_database_stats()
    
    # Update metadata filter options if the method exists
    if hasattr(self, 'update_metadata_filter_options'):
        self.update_metadata_filter_options()
    
    # Make progress window modal
    progress_window.transient(self.root)
    progress_window.grab_set()
    self.root.wait_window(progress_window)
