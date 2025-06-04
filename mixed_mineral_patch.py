#!/usr/bin/env python3
"""
Patch script to enhance the mixed mineral analysis functionality
Run this script to apply the enhanced mineral search with fallback options
"""

import re
import os
from pathlib import Path

def apply_mixed_mineral_enhancement():
    """Apply the enhanced mixed mineral analysis functionality."""
    
    # Path to the main application file
    app_file = Path("raman_analysis_app.py")
    
    if not app_file.exists():
        print("Error: raman_analysis_app.py not found in current directory")
        return False
    
    # Read the current file
    with open(app_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the problematic add_mineral function and replace it
    old_pattern = r'''def add_mineral\(\):
            
            matches = self\._perform_search\("correlation", 5, 0\.5\)
            if not matches:
                messagebox\.showwarning\("Warning", "No good matches found\."\)
                return'''
    
    new_function = '''def add_mineral():
            """Enhanced mineral addition with fallback options when no matches are found."""
            
            # First attempt: correlation search with standard threshold
            matches = self._perform_search("correlation", 5, 0.5)
            
            if not matches:
                # No matches found - provide user with options instead of just showing warning
                fallback_window = tk.Toplevel(window)
                fallback_window.title("No Matches Found - Choose Option")
                fallback_window.geometry("500x400")
                fallback_window.transient(window)
                fallback_window.grab_set()
                
                # Center the window
                fallback_window.update_idletasks()
                x = (fallback_window.winfo_screenwidth() // 2) - (fallback_window.winfo_width() // 2)
                y = (fallback_window.winfo_screenheight() // 2) - (fallback_window.winfo_height() // 2)
                fallback_window.geometry(f"+{x}+{y}")
                
                main_frame = ttk.Frame(fallback_window, padding=20)
                main_frame.pack(fill=tk.BOTH, expand=True)
                
                # Title and explanation
                title_label = ttk.Label(main_frame, text="No Matches Found", 
                                      font=("TkDefaultFont", 12, "bold"))
                title_label.pack(pady=(0, 10))
                
                explanation = ttk.Label(main_frame, 
                    text="No minerals matched your spectrum with the standard threshold (0.5).\\n"
                         "Choose one of the following options to continue:",
                    justify=tk.CENTER, wraplength=450)
                explanation.pack(pady=(0, 20))
                
                # Option buttons frame
                options_frame = ttk.Frame(main_frame)
                options_frame.pack(fill=tk.X, pady=10)
                
                def try_lower_threshold():
                    """Try correlation search with progressively lower thresholds."""
                    fallback_window.destroy()
                    
                    # Try progressively lower thresholds
                    thresholds = [0.3, 0.2, 0.1, 0.05]
                    matches_found = None
                    threshold_used = None
                    
                    for threshold in thresholds:
                        matches_found = self._perform_search("correlation", 10, threshold)
                        if matches_found:
                            threshold_used = threshold
                            break
                    
                    if matches_found:
                        show_enhanced_selection(matches_found, f"Matches with reduced threshold ({threshold_used})")
                    else:
                        messagebox.showinfo("Info", "No matches found even with very low thresholds. Try DTW search or browse database.")
                
                def try_dtw_search():
                    """Try DTW-based search which is more flexible."""
                    fallback_window.destroy()
                    
                    if not SKLEARN_AVAILABLE:
                        messagebox.showerror("Error", "DTW search requires scikit-learn. Please install it or try other options.")
                        return
                    
                    # DTW search with low threshold to get more results
                    matches_found = self._perform_search("ml", 20, 0.1)
                    
                    if matches_found:
                        show_enhanced_selection(matches_found, "DTW Search Results (Top 20)")
                    else:
                        messagebox.showinfo("Info", "No matches found with DTW search. Try browsing the database manually.")
                
                def browse_database():
                    """Open database browser for manual mineral selection."""
                    fallback_window.destroy()
                    show_database_browser()
                
                def cancel_operation():
                    """Cancel the add mineral operation."""
                    fallback_window.destroy()
                
                # Create option buttons
                ttk.Button(options_frame, text="1. Try Lower Threshold", 
                          command=try_lower_threshold,
                          width=25).pack(pady=5, fill=tk.X)
                
                ttk.Button(options_frame, text="2. Use DTW Search (More Flexible)", 
                          command=try_dtw_search,
                          width=25).pack(pady=5, fill=tk.X)
                
                ttk.Button(options_frame, text="3. Browse Database Manually", 
                          command=browse_database,
                          width=25).pack(pady=5, fill=tk.X)
                
                ttk.Button(options_frame, text="Cancel", 
                          command=cancel_operation,
                          width=25).pack(pady=(15, 5), fill=tk.X)
                
                # Add descriptions for each option
                desc_frame = ttk.Frame(main_frame)
                desc_frame.pack(fill=tk.X, pady=(10, 0))
                
                descriptions = ttk.Label(desc_frame,
                    text="Option 1: Automatically reduces similarity threshold to find more matches\\n"
                         "Option 2: Uses Dynamic Time Warping for better pattern matching\\n"
                         "Option 3: Browse and manually select from the entire mineral database",
                    justify=tk.LEFT, font=("TkDefaultFont", 9), foreground="gray")
                descriptions.pack()
                
                return  # Exit here, user will choose an option
            
            # If we have matches from the initial search, show them
            show_enhanced_selection(matches, "Correlation Search Results")
            
        def show_enhanced_selection(matches, title="Select Mineral"):
            """Show enhanced mineral selection window."""
            select_window = tk.Toplevel(window)
            select_window.title(title)
            select_window.geometry("500x400")
            select_window.transient(window)
            
            main_frame = ttk.Frame(select_window, padding=10)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Title
            title_label = ttk.Label(main_frame, text=title, font=("TkDefaultFont", 11, "bold"))
            title_label.pack(pady=(0, 10))
            
            # Match list frame
            match_frame = ttk.Frame(main_frame)
            match_frame.pack(fill=tk.BOTH, expand=True)
            
            match_listbox = tk.Listbox(match_frame, height=15)
            match_scrollbar = ttk.Scrollbar(match_frame, orient=tk.VERTICAL, command=match_listbox.yview)
            match_listbox.configure(yscrollcommand=match_scrollbar.set)
            
            match_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            match_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Populate matches
            for match_name, score in matches:
                display_name = self.get_mineral_display_name(match_name)
                match_listbox.insert(tk.END, f"{display_name} (Score: {score:.3f})")
            
            # Button frame
            btn_frame = ttk.Frame(main_frame)
            btn_frame.pack(fill=tk.X, pady=(10, 0))
            
            def on_select():
                selection = match_listbox.curselection()
                if not selection:
                    messagebox.showwarning("Warning", "Please select a mineral.")
                    return
                
                match_name = matches[selection[0]][0]
                if match_name in selected_minerals:
                    messagebox.showwarning("Warning", "This mineral is already selected.")
                    return
                
                add_selected_mineral_enhanced(match_name)
                select_window.destroy()
            
            ttk.Button(btn_frame, text="Select", command=on_select).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(btn_frame, text="Cancel", command=select_window.destroy).pack(side=tk.LEFT)
            
            # Double-click to select
            match_listbox.bind('<Double-Button-1>', lambda e: on_select())
        
        def show_database_browser():
            """Open database browser for manual mineral selection."""
            try:
                # Create a database browser window
                browser_window = tk.Toplevel(window)
                browser_window.title("Browse Mineral Database")
                browser_window.geometry("600x500")
                browser_window.transient(window)
                
                browser_frame = ttk.Frame(browser_window, padding=10)
                browser_frame.pack(fill=tk.BOTH, expand=True)
                
                # Search frame
                search_frame = ttk.Frame(browser_frame)
                search_frame.pack(fill=tk.X, pady=(0, 10))
                
                ttk.Label(search_frame, text="Search minerals:").pack(side=tk.LEFT)
                search_var = tk.StringVar()
                search_entry = ttk.Entry(search_frame, textvariable=search_var, width=30)
                search_entry.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
                
                # Mineral list
                list_frame = ttk.Frame(browser_frame)
                list_frame.pack(fill=tk.BOTH, expand=True)
                
                mineral_listbox = tk.Listbox(list_frame, height=20)
                scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=mineral_listbox.yview)
                mineral_listbox.configure(yscrollcommand=scrollbar.set)
                
                mineral_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                # Populate mineral list
                all_minerals = []
                for name in self.raman.database.keys():
                    display_name = self.get_mineral_display_name(name)
                    all_minerals.append((name, display_name))
                
                all_minerals.sort(key=lambda x: x[1])  # Sort by display name
                
                def update_mineral_list(filter_text=""):
                    mineral_listbox.delete(0, tk.END)
                    for name, display_name in all_minerals:
                        if filter_text.lower() in display_name.lower():
                            mineral_listbox.insert(tk.END, display_name)
                
                def on_search_change(*args):
                    update_mineral_list(search_var.get())
                
                search_var.trace('w', on_search_change)
                update_mineral_list()  # Initial population
                
                # Button frame
                btn_frame = ttk.Frame(browser_frame)
                btn_frame.pack(fill=tk.X, pady=(10, 0))
                
                def select_from_browser():
                    selection = mineral_listbox.curselection()
                    if not selection:
                        messagebox.showwarning("Warning", "Please select a mineral.")
                        return
                    
                    selected_display_name = mineral_listbox.get(selection[0])
                    
                    # Find the actual mineral name
                    selected_name = None
                    for name, display_name in all_minerals:
                        if display_name == selected_display_name:
                            selected_name = name
                            break
                    
                    if selected_name and selected_name not in selected_minerals:
                        add_selected_mineral_enhanced(selected_name)
                        browser_window.destroy()
                    elif selected_name in selected_minerals:
                        messagebox.showwarning("Warning", "This mineral is already selected.")
                    else:
                        messagebox.showerror("Error", "Could not find selected mineral.")
                
                ttk.Button(btn_frame, text="Select Mineral", command=select_from_browser).pack(side=tk.LEFT, padx=(0, 5))
                ttk.Button(btn_frame, text="Cancel", command=browser_window.destroy).pack(side=tk.LEFT)
                
                # Focus on search entry
                search_entry.focus()
                
                # Double-click to select
                mineral_listbox.bind('<Double-Button-1>', lambda e: select_from_browser())
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open database browser: {str(e)}")
        
        def add_selected_mineral_enhanced(match_name):
            """Add a selected mineral to the analysis."""
            try:
                match_data = self.raman.database[match_name]
                match_spectrum = match_data["intensities"]
                match_wavenumbers = match_data["wavenumbers"]
                match_spectrum_interp = np.interp(
                    current_wavenumbers, match_wavenumbers, match_spectrum
                )
                selected_minerals[match_name] = match_spectrum_interp
                mineral_weights[match_name] = 1.0
                display_name = self.get_mineral_display_name(match_name)
                selected_minerals_listbox.insert(tk.END, display_name)
                update_fit()
                messagebox.showinfo("Success", f"Added {display_name} to the analysis.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add mineral: {str(e)}")'''
    
    # Apply the replacement
    if old_pattern in content:
        content = re.sub(old_pattern, new_function, content, flags=re.MULTILINE | re.DOTALL)
        
        # Write the modified content back
        with open(app_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Successfully enhanced the mixed mineral analysis functionality!")
        print("\nEnhancements added:")
        print("1. Auto-reduce threshold search (0.3, 0.2, 0.1, 0.05)")
        print("2. DTW-based search for better pattern matching")
        print("3. Manual database browser with search functionality")
        print("4. User-friendly fallback options when no matches are found")
        return True
    else:
        print("❌ Could not find the target function to replace.")
        print("The file may have been modified or the pattern has changed.")
        return False

if __name__ == "__main__":
    print("Mixed Mineral Analysis Enhancement Patch")
    print("=" * 50)
    
    success = apply_mixed_mineral_enhancement()
    
    if success:
        print("\n🎉 Patch applied successfully!")
        print("You can now use the enhanced mixed mineral analysis with fallback options.")
    else:
        print("\n❌ Patch failed to apply.")
        print("Please check the file manually or contact support.") 