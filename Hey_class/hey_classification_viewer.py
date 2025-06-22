def view_hey_classification(self):
    """Create a window to browse Hey Classification hierarchy and associated minerals."""
    # Try to load Hey Classification data from CSV
    hey_csv_path = "RRUFF_Hey_Index.csv"
    if not os.path.exists(hey_csv_path):
        messagebox.showerror("Error", f"Hey Classification data file not found: {hey_csv_path}")
        return
    
    try:
        # Read the CSV file
        df = pd.read_csv(hey_csv_path)
        
        # Check if required columns exist
        if 'Hey Classification ID' not in df.columns or 'Hey Classification Name' not in df.columns:
            messagebox.showerror("Error", "Hey Classification columns not found in the data file")
            return
        
        # Create a new window
        hey_window = tk.Toplevel(self.root)
        hey_window.title("Hey Classification Browser")
        hey_window.geometry("800x600")
        
        # Main frame
        main_frame = ttk.Frame(hey_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a paned window for resizable sections
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left frame for classification tree
        left_frame = ttk.Frame(paned, padding=5)
        paned.add(left_frame, weight=1)
        
        # Right frame for mineral list
        right_frame = ttk.Frame(paned, padding=5)
        paned.add(right_frame, weight=2)
        
        # Create tree structure for Hey Classification
        ttk.Label(left_frame, text="Hey Classification Hierarchy").pack(anchor=tk.W, pady=(0, 5))
        
        # Create treeview
        tree = ttk.Treeview(left_frame)
        tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Add scrollbar to tree
        tree_scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=tree.yview)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=tree_scrollbar.set)
        
        # Create listbox for minerals
        ttk.Label(right_frame, text="Minerals in Selected Classification").pack(anchor=tk.W, pady=(0, 5))
        
        # Search frame for minerals
        search_frame = ttk.Frame(right_frame)
        search_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=(0, 5))
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Mineral list with scrollbar
        list_frame = ttk.Frame(right_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        mineral_listbox = tk.Listbox(list_frame, selectmode=tk.BROWSE)
        mineral_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=mineral_listbox.yview)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        mineral_listbox.configure(yscrollcommand=list_scrollbar.set)
        
        # Details section
        detail_frame = ttk.LabelFrame(right_frame, text="Mineral Details", padding=5)
        detail_frame.pack(fill=tk.X, pady=(5, 0))
        
        detail_text = tk.Text(detail_frame, height=8, wrap=tk.WORD)
        detail_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        detail_scrollbar = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=detail_text.yview)
        detail_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        detail_text.configure(yscrollcommand=detail_scrollbar.set)
        detail_text.config(state=tk.DISABLED)  # Read-only
        
        # Organize Hey Classification data by hierarchy
        hierarchy = {}
        minerals_by_class = {}
        
        # Function to get all the minerals for a specific Hey Classification
        def get_minerals_for_classification(classification):
            """Get all minerals for a specific Hey Classification."""
            if classification in minerals_by_class:
                return minerals_by_class[classification]
            
            minerals = df[df['Hey Classification Name'] == classification]['Mineral Name'].tolist()
            minerals_by_class[classification] = sorted(minerals)
            return minerals_by_class[classification]
        
        # Analyze Hey Classification structure
        for _, row in df.iterrows():
            classification = row['Hey Classification Name']
            mineral = row['Mineral Name']
            
            # Skip if classification is missing
            if pd.isna(classification):
                continue
            
            # Extract classification parts (assuming format like "X. Silicates - Tectosilicates")
            parts = classification.split(' - ')
            main_class = parts[0].strip()
            subclass = parts[1].strip() if len(parts) > 1 else ""
            
            # Build hierarchy
            if main_class not in hierarchy:
                hierarchy[main_class] = set()
            
            if subclass:
                hierarchy[main_class].add(subclass)
            
            # Build mineral lists by classification
            if classification not in minerals_by_class:
                minerals_by_class[classification] = []
            
            minerals_by_class[classification].append(mineral)
        
        # Sort minerals in each classification
        for classification in minerals_by_class:
            minerals_by_class[classification].sort()
        
        # Populate treeview with Hey Classification hierarchy
        # First, clear existing items
        for item in tree.get_children():
            tree.delete(item)
        
        # Add main classes as parent nodes
        for main_class in sorted(hierarchy.keys()):
            main_id = tree.insert('', 'end', text=main_class, values=(main_class, ''))
            
            # Add subclasses as child nodes
            subclasses = sorted(hierarchy[main_class]) if hierarchy[main_class] else []
            for subclass in subclasses:
                full_class = f"{main_class} - {subclass}"
                tree.insert(main_id, 'end', text=subclass, values=(full_class, ''))
        
        # Function to update mineral list when a classification is selected
        def on_classification_select(event):
            selected_items = tree.selection()
            if not selected_items:
                return
            
            # Get selected classification
            selected_item = selected_items[0]
            values = tree.item(selected_item, 'values')
            
            classification = values[0] if values else ""
            update_mineral_list(classification)
        
        # Function to update mineral list based on classification and search filter
        def update_mineral_list(classification=None, search_term=None):
            mineral_listbox.delete(0, tk.END)
            
            if not classification and not search_term:
                return
            
            # Get minerals for selected classification
            minerals = []
            if classification:
                minerals = get_minerals_for_classification(classification)
            else:
                # If no classification but searching, search all minerals
                minerals = sorted(df['Mineral Name'].unique())
            
            # Apply search filter if provided
            if search_term:
                search_lower = search_term.lower()
                minerals = [m for m in minerals if search_lower in m.lower()]
            
            # Update listbox
            for mineral in minerals:
                mineral_listbox.insert(tk.END, mineral)
        
        # Function to display mineral details when selected
        def on_mineral_select(event):
            selected_indices = mineral_listbox.curselection()
            if not selected_indices:
                return
            
            # Get selected mineral
            selected_index = selected_indices[0]
            mineral_name = mineral_listbox.get(selected_index)
            
            # Clear detail text
            detail_text.config(state=tk.NORMAL)
            detail_text.delete(1.0, tk.END)
            
            # Get mineral details from dataframe
            mineral_data = df[df['Mineral Name'] == mineral_name].iloc[0]
            
            # Add mineral details
            detail_text.insert(tk.END, f"Mineral: {mineral_name}\n")
            detail_text.insert(tk.END, f"Hey Classification: {mineral_data['Hey Classification Name']}\n")
            
            # Add other available details
            for col in ['RRUFF Chemistry (concise)', 'Chemistry Elements', 'Crystal Systems', 'Space Groups']:
                if col in mineral_data and not pd.isna(mineral_data[col]):
                    detail_text.insert(tk.END, f"{col}: {mineral_data[col]}\n")
            
            detail_text.config(state=tk.DISABLED)
        
        # Function to search minerals
        def on_search(event=None):
            search_term = search_var.get().strip()
            
            # Get selected classification
            selected_items = tree.selection()
            classification = None
            if selected_items:
                values = tree.item(selected_items[0], 'values')
                classification = values[0] if values else None
            
            update_mineral_list(classification, search_term)
        
        # Bind events
        tree.bind('<<TreeviewSelect>>', on_classification_select)
        mineral_listbox.bind('<<ListboxSelect>>', on_mineral_select)
        search_entry.bind('<Return>', on_search)
        
        # Add search button
        ttk.Button(search_frame, text="Search", command=on_search).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(search_frame, text="Clear", 
                 command=lambda: [search_var.set(""), on_search()]).pack(side=tk.LEFT, padx=(5, 0))
        
        # Button to add selected mineral to database
        def import_selected_mineral():
            selected_indices = mineral_listbox.curselection()
            if not selected_indices:
                messagebox.showinfo("Selection Required", "Please select a mineral first.")
                return
            
            # Get selected mineral
            selected_index = selected_indices[0]
            mineral_name = mineral_listbox.get(selected_index)
            
            # Close the Hey Classification viewer
            hey_window.destroy()
            
            # Set the mineral name in the database tab for adding
            self.var_name.set(mineral_name)
            
            # Switch to the database tab
            self.notebook.select(self.tab_database)
            
            # Show message to guide the user
            messagebox.showinfo("Import Mineral", 
                              f"The mineral name '{mineral_name}' has been selected. \n\n"
                              "Please import a spectrum file first, then click 'Add Current Spectrum' "
                              "to add it to the database with this mineral name and Hey Classification.")
        
        # Add button to bottom of right frame
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(button_frame, text="Use Selected Mineral", 
                 command=import_selected_mineral).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Close", 
                 command=hey_window.destroy).pack(side=tk.RIGHT)
        
        # Initialize with all main classifications expanded
        for item in tree.get_children():
            tree.item(item, open=True)
        
        # Make window modal
        hey_window.transient(self.root)
        hey_window.grab_set()
        self.root.wait_window(hey_window)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error loading Hey Classification data: {str(e)}")

# Add this button to the appropriate menu or tab
# For example, in create_database_tab method:
def create_database_tab(self):
    """Create content for the database operations tab."""
    # [Your existing code...]
    
    # Add button to view Hey Classification browser
    ttk.Button(manage_frame, text="Browse Hey Classification", 
             command=self.view_hey_classification).pack(fill=tk.X, pady=2)
    
    # [Rest of your existing code...]
