import os
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import requests
from bs4 import BeautifulSoup
import time
from hey_classification_analysis import (
    analyze_hey_classification_relationships,
    analyze_age_distribution,
    analyze_crystal_systems,
    analyze_space_groups,
    analyze_paragenetic_modes,
    analyze_element_distribution
)

from improved_hey_classification import ImprovedHeyClassifier
from improved_element_extraction import extract_elements_from_formula, update_chemistry_elements_column
from raman_vibrational_classifier import HeyCelestianClassifier, create_hey_celestian_classification_report

class HeyClassificationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Hey & Hey-Celestian Classification Tool")
        self.root.geometry("1200x800")  # Increased window size
        
        # Initialize variables
        self.input_file_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.df = None  # Store the loaded DataFrame
        self.reference_data = {}  # Store reference classifications
        
        # Show initial debug message
        messagebox.showinfo("Debug", "Application starting...")
        
        # Check for existing indexed files
        self.existing_indexed_file = self.find_existing_indexed_file()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Create tabs - Hey & Hey-Celestian classifier as primary!
        self.create_hey_celestian_tab()  # Primary tab - integrated classifier
        self.create_database_editor_tab()  # Database editor
        self.create_analysis_tab()  # Analysis tools
        
        # Store tab indices
        self.classifier_tab_index = 0  # Primary integrated classifier tab
        self.editor_tab_index = 1
        self.analysis_tab_index = 2
        
        # If we found an existing indexed file, suggest using it
        if self.existing_indexed_file:
            self.suggest_using_existing_file()
        else:
            # Initially disable other tabs (integrated classifier stays enabled as primary)
            self.notebook.tab(self.editor_tab_index, state="disabled")
            self.notebook.tab(self.analysis_tab_index, state="disabled")

    def find_existing_indexed_file(self):
        """Find any existing indexed files in the current directory"""
        current_dir = os.getcwd()
        debug_info = [f"Searching for indexed files in: {current_dir}"]
        
        # Look for files that start with INDEXED_ and end with .csv
        for file in os.listdir(current_dir):
            debug_info.append(f"Checking file: {file}")
            if file.upper().startswith("INDEXED_") and file.lower().endswith(".csv"):
                full_path = os.path.join(current_dir, file)
                debug_info.append(f"Found indexed file: {full_path}")
                messagebox.showinfo("Debug Info", "\n".join(debug_info))
                return full_path
        
        debug_info.append("No indexed files found")
        messagebox.showinfo("Debug Info", "\n".join(debug_info))
        return None

    def suggest_using_existing_file(self):
        """Suggest using the existing indexed file"""
        response = messagebox.askyesno(
            "Existing Indexed File Found",
            f"Found an existing indexed file: {os.path.basename(self.existing_indexed_file)}\n"
            f"Would you like to use this file for analysis?"
        )
        if response:
            # Set the input file for all tabs
            self.input_file_var.set(self.existing_indexed_file)
            self.celestian_file_var.set(self.existing_indexed_file)
            
            # Enable all tabs (integrated classifier is already enabled as primary)
            self.notebook.tab(self.editor_tab_index, state="normal")
            self.notebook.tab(self.analysis_tab_index, state="normal")
            
            # Update the status to show we're using an existing file
            self.status_var.set(f"Using existing indexed file: {os.path.basename(self.existing_indexed_file)}")
            
            # Load the database
            self.load_database()
        else:
            # If user declines, disable other tabs (integrated classifier stays enabled as primary)
            self.notebook.tab(self.editor_tab_index, state="disabled")
            self.notebook.tab(self.analysis_tab_index, state="disabled")
            self.status_var.set("Please use the integrated classifier to proceed")

    def create_analysis_tab(self):
        """Create the analysis tab with controls for running analysis"""
        analysis_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(analysis_frame, text="Analysis")
        
        # Help text
        help_text = "Select the RRUFF_Hey_Index.csv file that was created after running the Hey Classification update."
        ttk.Label(analysis_frame, text=help_text, wraplength=500).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # Input file selection
        ttk.Label(analysis_frame, text="Input File (RRUFF_Hey_Index.csv):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(analysis_frame, textvariable=self.input_file_var, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(analysis_frame, text="Browse", command=self.browse_input_file).grid(row=1, column=2)
        
        # Output directory selection
        ttk.Label(analysis_frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.output_dir_var = tk.StringVar()
        ttk.Entry(analysis_frame, textvariable=self.output_dir_var, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(analysis_frame, text="Browse", command=self.browse_output_dir).grid(row=2, column=2)
        
        # Run analysis button
        ttk.Button(analysis_frame, text="Run Analysis", command=self.run_analysis).grid(row=3, column=0, columnspan=3, pady=20)

    def create_database_editor_tab(self):
        """Create the database editor tab with navigation tree and metadata editing"""
        editor_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(editor_frame, text="Database Editor")
        
        # Create paned window for resizable sections
        paned = tk.PanedWindow(editor_frame, orient=tk.HORIZONTAL, sashwidth=5, sashrelief=tk.RAISED)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left pane for Hey groups tree
        left_frame = ttk.Frame(paned, padding=5)
        paned.add(left_frame)
        
        # Middle pane for minerals list
        middle_frame = ttk.Frame(paned, padding=5)
        paned.add(middle_frame)
        
        # Right pane for metadata editing
        right_frame = ttk.Frame(paned, padding=5)
        paned.add(right_frame)
        
        # Left pane: Hey groups tree
        ttk.Label(left_frame, text="Hey Classification Groups").pack(anchor=tk.W)
        
        # Configure Treeview style
        style = ttk.Style()
        style.configure("Treeview", foreground="black", background="white")
        style.configure("Treeview.Heading", foreground="black", background="lightgray")
        
        self.groups_tree = ttk.Treeview(left_frame, style="Treeview")
        self.groups_tree.pack(fill=tk.BOTH, expand=True)
        
        # Add columns
        self.groups_tree["columns"] = ("ID", "Name")
        self.groups_tree.column("#0", width=0, stretch=tk.NO)
        self.groups_tree.column("ID", anchor=tk.W, width=100)
        self.groups_tree.column("Name", anchor=tk.W, width=200)
        
        # Add headings
        self.groups_tree.heading("#0", text="", anchor=tk.W)
        self.groups_tree.heading("ID", text="ID", anchor=tk.W)
        self.groups_tree.heading("Name", text="Name", anchor=tk.W)
        
        # Add scrollbar to groups tree
        groups_scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.groups_tree.yview)
        groups_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.groups_tree.configure(yscrollcommand=groups_scrollbar.set)
        
        # Middle pane: Minerals list
        minerals_header = ttk.Frame(middle_frame)
        minerals_header.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(minerals_header, text="Minerals in Selected Group").pack(side=tk.LEFT)
        
        # Add validation button
        ttk.Button(minerals_header, text="Validate Classifications", 
                  command=self.validate_classifications).pack(side=tk.RIGHT, padx=5)
        
        # Add select all/none buttons
        select_frame = ttk.Frame(minerals_header)
        select_frame.pack(side=tk.RIGHT)
        ttk.Button(select_frame, text="Select All", command=self.select_all_minerals).pack(side=tk.LEFT, padx=2)
        ttk.Button(select_frame, text="Select None", command=self.deselect_all_minerals).pack(side=tk.LEFT, padx=2)
        
        # Configure Listbox style with multi-select
        self.minerals_list = tk.Listbox(middle_frame, bg="white", fg="black", selectmode=tk.EXTENDED)
        self.minerals_list.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbar to minerals list
        minerals_scrollbar = ttk.Scrollbar(middle_frame, orient=tk.VERTICAL, command=self.minerals_list.yview)
        minerals_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.minerals_list.configure(yscrollcommand=minerals_scrollbar.set)
        
        # Add move controls below minerals list
        move_frame = ttk.Frame(middle_frame)
        move_frame.pack(fill=tk.X, pady=5)
        
        # Create group selection dropdown
        self.target_group_var = tk.StringVar()
        self.target_group_dropdown = ttk.Combobox(move_frame, textvariable=self.target_group_var, state="readonly")
        self.target_group_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Create move button
        ttk.Button(move_frame, text="Move Selected to Group", command=self.move_minerals_to_group).pack(side=tk.RIGHT)
        
        # Right pane: Metadata editor
        ttk.Label(right_frame, text="Mineral Metadata").pack(anchor=tk.W)
        
        # Create a frame for the metadata editor with scrollbar
        metadata_frame = ttk.Frame(right_frame)
        metadata_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas and scrollbar for metadata
        canvas = tk.Canvas(metadata_frame)
        scrollbar = ttk.Scrollbar(metadata_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create frame inside canvas for metadata fields
        self.metadata_editor = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self.metadata_editor, anchor=tk.NW)
        
        # Configure canvas scrolling
        def configure_scroll(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        self.metadata_editor.bind("<Configure>", configure_scroll)
        
        # Save button
        save_frame = ttk.Frame(right_frame)
        save_frame.pack(fill=tk.X, pady=10)
        ttk.Button(save_frame, text="Save Changes", command=self.save_database_changes).pack(side=tk.RIGHT)
        
        # Bind events
        self.groups_tree.bind('<<TreeviewSelect>>', self.on_group_select)
        self.minerals_list.bind('<<ListboxSelect>>', self.on_mineral_select)
        
        # Initialize metadata fields
        self.metadata_fields = {}
        self.current_mineral = None
        self.current_group_id = None

    def create_hey_celestian_tab(self):
        """Create the integrated Hey & Hey-Celestian classification tab (primary)."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🌟 Hey & Hey-Celestian Classifier")
        
        # Main frame with padding
        main_frame = ttk.Frame(tab)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title and description
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill='x', pady=(0, 10))
        
        title_label = ttk.Label(title_frame, text="Integrated Hey & Hey-Celestian Classification System", 
                               font=('Arial', 14, 'bold'))
        title_label.pack()
        
        desc_label = ttk.Label(title_frame, 
                              text="Dual classification: Traditional Hey system + novel vibrational mode analysis for Raman spectroscopy",
                              font=('Arial', 10), foreground='gray')
        desc_label.pack()
        
        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="📁 Input File Selection", padding=10)
        input_frame.pack(fill='x', pady=(0, 10))
        
        # File selection
        file_frame = ttk.Frame(input_frame)
        file_frame.pack(fill='x', pady=(0, 5))
        
        ttk.Label(file_frame, text="CSV File:").pack(side='left')
        self.celestian_file_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.celestian_file_var, width=50)
        file_entry.pack(side='left', padx=(5, 5), fill='x', expand=True)
        
        ttk.Button(file_frame, text="Browse", 
                  command=self.browse_celestian_file).pack(side='right')
        
        # Options section
        options_frame = ttk.LabelFrame(main_frame, text="⚙️ Classification Options", padding=10)
        options_frame.pack(fill='x', pady=(0, 10))
        
        # Classification options
        class_opts_frame = ttk.Frame(options_frame)
        class_opts_frame.pack(fill='x', pady=(0, 5))
        
        self.enable_hey_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(class_opts_frame, text="Traditional Hey Classification", 
                       variable=self.enable_hey_var).pack(side='left', padx=(0, 20))
        
        self.enable_celestian_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(class_opts_frame, text="Hey-Celestian Vibrational Classification", 
                       variable=self.enable_celestian_var).pack(side='left')
        
        # Confidence threshold
        conf_frame = ttk.Frame(options_frame)
        conf_frame.pack(fill='x', pady=(5, 5))
        
        ttk.Label(conf_frame, text="Hey-Celestian Min. Confidence:").pack(side='left')
        self.celestian_confidence_var = tk.DoubleVar(value=0.5)
        conf_scale = ttk.Scale(conf_frame, from_=0.0, to=1.0, 
                              variable=self.celestian_confidence_var, orient='horizontal')
        conf_scale.pack(side='left', padx=(5, 5), fill='x', expand=True)
        
        self.celestian_conf_label = ttk.Label(conf_frame, text="0.50")
        self.celestian_conf_label.pack(side='right')
        
        # Update label when scale changes
        conf_scale.configure(command=lambda v: self.celestian_conf_label.configure(text=f"{float(v):.2f}"))
        
        # Output options
        output_frame = ttk.Frame(options_frame)
        output_frame.pack(fill='x', pady=(5, 0))
        
        self.celestian_create_backup_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(output_frame, text="Create backup of original file", 
                       variable=self.celestian_create_backup_var).pack(side='left')
        
        self.celestian_detailed_report_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(output_frame, text="Generate detailed analysis report", 
                       variable=self.celestian_detailed_report_var).pack(side='right')
        
        # Single mineral test section
        single_frame = ttk.LabelFrame(main_frame, text="🧪 Test Single Mineral", padding=10)
        single_frame.pack(fill='x', pady=(0, 10))
        
        # Input fields
        input_grid = ttk.Frame(single_frame)
        input_grid.pack(fill='x', pady=(0, 10))
        
        ttk.Label(input_grid, text="Chemical Formula:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.test_formula_var = tk.StringVar()
        formula_entry = ttk.Entry(input_grid, textvariable=self.test_formula_var, width=30)
        formula_entry.grid(row=0, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(input_grid, text="Elements:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.test_elements_var = tk.StringVar()
        elements_entry = ttk.Entry(input_grid, textvariable=self.test_elements_var, width=50)
        elements_entry.grid(row=1, column=1, padx=5, sticky=tk.W)
        
        # Test buttons
        test_buttons = ttk.Frame(single_frame)
        test_buttons.pack(fill='x', pady=(0, 10))
        
        ttk.Button(test_buttons, text="Extract Elements", 
                  command=self.extract_test_elements).pack(side=tk.LEFT, padx=5)
        ttk.Button(test_buttons, text="🔬 Test Both Classifications", 
                  command=self.test_dual_classification).pack(side=tk.LEFT, padx=5)
        
        # Results display
        self.test_result_var = tk.StringVar()
        result_label = ttk.Label(single_frame, textvariable=self.test_result_var, 
                               foreground="blue", font=("Arial", 10, "bold"))
        result_label.pack(pady=5, anchor=tk.W)
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(0, 10))
        
        # Information buttons
        info_frame = ttk.Frame(button_frame)
        info_frame.pack(fill='x', pady=(0, 5))
        
        ttk.Button(info_frame, text="📊 View Hey-Celestian Groups", 
                  command=self.show_celestian_groups).pack(side='left', padx=(0, 10))
        
        ttk.Button(info_frame, text="📖 View Hey Categories", 
                  command=self.show_hey_categories).pack(side='left')
        
        # Main processing buttons
        process_frame = ttk.Frame(button_frame)
        process_frame.pack(fill='x', pady=(5, 0))
        
        ttk.Button(process_frame, text="🚀 Classify All Minerals", 
                  command=self.run_dual_classification,
                  style='Accent.TButton').pack(side='left')
        
        ttk.Button(process_frame, text="📈 Generate Comparison Report", 
                  command=self.generate_dual_comparison).pack(side='left', padx=(10, 0))
        
        # Progress and output section
        output_section = ttk.LabelFrame(main_frame, text="📋 Progress & Results", padding=10)
        output_section.pack(fill='both', expand=True)
        
        # Progress bar
        self.celestian_progress = ttk.Progressbar(output_section, mode='indeterminate')
        self.celestian_progress.pack(fill='x', pady=(0, 5))
        
        # Output text with scrollbar
        text_frame = ttk.Frame(output_section)
        text_frame.pack(fill='both', expand=True)
        
        self.celestian_output_text = tk.Text(text_frame, height=15, wrap='word')
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=self.celestian_output_text.yview)
        self.celestian_output_text.configure(yscrollcommand=scrollbar.set)
        
        self.celestian_output_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Initialize classifiers
        self.celestian_classifier = None
        self.improved_classifier = None
        
        try:
            self.celestian_classifier = HeyCelestianClassifier()
            self.celestian_output_text.insert(tk.END, "✅ Hey-Celestian Classifier initialized successfully!\n")
            self.celestian_output_text.insert(tk.END, "🌟 15 vibrational groups available for classification\n")
        except Exception as e:
            self.celestian_output_text.insert(tk.END, f"❌ Error initializing Hey-Celestian classifier: {str(e)}\n")
        
        try:
            self.improved_classifier = ImprovedHeyClassifier()
            self.celestian_output_text.insert(tk.END, "✅ Improved Hey Classifier initialized successfully!\n")
            self.celestian_output_text.insert(tk.END, "📊 Traditional Hey classification with 62.5% accuracy ready\n\n")
        except Exception as e:
            self.celestian_output_text.insert(tk.END, f"❌ Error initializing Hey classifier: {str(e)}\n")

    def extract_test_elements(self):
        """Extract elements from the test chemical formula."""
        try:
            from improved_element_extraction import extract_elements_from_formula
            
            formula = self.test_formula_var.get().strip()
            if not formula:
                self.test_result_var.set("Please enter a chemical formula")
                return
            
            elements = extract_elements_from_formula(formula)
            elements_str = ', '.join(sorted(elements)) if elements else ""
            self.test_elements_var.set(elements_str)
            
            self.test_result_var.set(f"Extracted {len(elements)} elements: {elements_str}")
            
        except Exception as e:
            self.test_result_var.set(f"Error extracting elements: {str(e)}")

    def test_dual_classification(self):
        """Test both Hey and Hey-Celestian classification on a single mineral."""
        try:
            formula = self.test_formula_var.get().strip()
            elements_str = self.test_elements_var.get().strip()
            
            if not formula:
                self.test_result_var.set("Please enter a chemical formula")
                return
            
            if not elements_str:
                self.extract_test_elements()
                elements_str = self.test_elements_var.get().strip()
            
            results = []
            
            # Traditional Hey Classification
            if self.enable_hey_var.get() and self.improved_classifier:
                hey_result = self.improved_classifier.classify_mineral(formula, elements_str)
                results.append(f"Hey: {hey_result['id']} - {hey_result['name']}")
            
            # Hey-Celestian Classification
            if self.enable_celestian_var.get() and self.celestian_classifier:
                celestian_result = self.celestian_classifier.classify_mineral(formula, elements_str, "Test Mineral")
                results.append(f"Hey-Celestian: {celestian_result['id']} - {celestian_result['name']} (Conf: {celestian_result['confidence']:.2f})")
            
            if results:
                self.test_result_var.set(" | ".join(results))
            else:
                self.test_result_var.set("No classifiers enabled or available")
                
        except Exception as e:
            self.test_result_var.set(f"Error: {str(e)}")

    def show_hey_categories(self):
        """Show the traditional Hey classification categories."""
        try:
            from improved_hey_classification import HEY_CATEGORIES
            
            # Create popup window
            dialog = tk.Toplevel(self.root)
            dialog.title("Hey Classification Categories")
            dialog.geometry("800x600")
            
            # Create treeview
            tree_frame = ttk.Frame(dialog)
            tree_frame.pack(fill='both', expand=True, padx=10, pady=10)
            
            tree = ttk.Treeview(tree_frame, columns=('ID', 'Name'), show='headings')
            tree.heading('ID', text='ID')
            tree.heading('Name', text='Category Name')
            
            tree.column('ID', width=50)
            tree.column('Name', width=700)
            
            # Add scrollbar
            scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            
            tree.pack(side='left', fill='both', expand=True)
            scrollbar.pack(side='right', fill='y')
            
            # Populate tree
            for cat_id, cat_name in sorted(HEY_CATEGORIES.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
                tree.insert('', 'end', values=(cat_id, cat_name))
                
        except Exception as e:
            messagebox.showerror("Error", f"Error showing Hey categories: {str(e)}")

    def run_dual_classification(self):
        """Run both Hey and Hey-Celestian classification on the selected file."""
        if not self.enable_hey_var.get() and not self.enable_celestian_var.get():
            messagebox.showerror("Error", "Please enable at least one classification system")
            return
        
        input_file = self.celestian_file_var.get().strip()
        if not input_file:
            messagebox.showerror("Error", "Please select an input CSV file")
            return
        
        if not os.path.exists(input_file):
            messagebox.showerror("Error", f"Input file not found: {input_file}")
            return
        
        try:
            # Start progress
            self.celestian_progress.start()
            self.celestian_output_text.insert(tk.END, f"🚀 Starting dual classification...\n")
            self.celestian_output_text.insert(tk.END, f"📁 Input file: {input_file}\n")
            
            enabled_systems = []
            if self.enable_hey_var.get():
                enabled_systems.append("Traditional Hey")
            if self.enable_celestian_var.get():
                enabled_systems.append("Hey-Celestian")
            
            self.celestian_output_text.insert(tk.END, f"🔧 Enabled systems: {', '.join(enabled_systems)}\n\n")
            self.root.update()
            
            # Create backup if requested
            if self.celestian_create_backup_var.get():
                backup_file = f"{input_file}.dual_backup"
                import shutil
                shutil.copy2(input_file, backup_file)
                self.celestian_output_text.insert(tk.END, f"📁 Backup created: {backup_file}\n")
            
            # Read CSV
            df = pd.read_csv(input_file)
            total_rows = len(df)
            
            self.celestian_output_text.insert(tk.END, f"📊 Processing {total_rows} minerals...\n")
            self.root.update()
            
            # Prepare new columns
            if self.enable_hey_var.get() and 'Improved Hey ID' not in df.columns:
                df['Improved Hey ID'] = ''
                df['Improved Hey Name'] = ''
            
            if self.enable_celestian_var.get():
                if 'Hey-Celestian Group ID' not in df.columns:
                    df['Hey-Celestian Group ID'] = ''
                if 'Hey-Celestian Group Name' not in df.columns:
                    df['Hey-Celestian Group Name'] = ''
                if 'Hey-Celestian Confidence' not in df.columns:
                    df['Hey-Celestian Confidence'] = ''
                if 'Hey-Celestian Reasoning' not in df.columns:
                    df['Hey-Celestian Reasoning'] = ''
            
            # Process each mineral
            hey_processed = 0
            celestian_processed = 0
            
            for i, row in df.iterrows():
                mineral_name = row.get('Mineral Name', '')
                chemistry = row.get('RRUFF Chemistry (concise)', '')
                elements = row.get('Chemistry Elements', '')
                
                if chemistry:
                    # Traditional Hey Classification
                    if self.enable_hey_var.get() and self.improved_classifier:
                        hey_result = self.improved_classifier.classify_mineral(chemistry, elements)
                        df.at[i, 'Improved Hey ID'] = hey_result['id']
                        df.at[i, 'Improved Hey Name'] = hey_result['name']
                        hey_processed += 1
                    
                    # Hey-Celestian Classification
                    if self.enable_celestian_var.get() and self.celestian_classifier:
                        celestian_result = self.celestian_classifier.classify_mineral(chemistry, elements, mineral_name)
                        df.at[i, 'Hey-Celestian Group ID'] = celestian_result['id']
                        df.at[i, 'Hey-Celestian Group Name'] = celestian_result['name']
                        df.at[i, 'Hey-Celestian Confidence'] = celestian_result['confidence']
                        df.at[i, 'Hey-Celestian Reasoning'] = celestian_result['reasoning']
                        celestian_processed += 1
                
                if (i + 1) % 100 == 0:
                    self.celestian_output_text.insert(tk.END, f"⏳ Processed {i+1}/{total_rows} minerals...\n")
                    self.root.update()
            
            # Generate output filename
            input_dir = os.path.dirname(input_file)
            input_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(input_dir, f"{input_name}_Dual_Classification.csv")
            
            # Save results
            df.to_csv(output_file, index=False)
            
            # Stop progress
            self.celestian_progress.stop()
            
            # Show results
            self.celestian_output_text.insert(tk.END, f"\n✅ Dual classification complete!\n")
            self.celestian_output_text.insert(tk.END, f"📊 Total minerals: {total_rows}\n")
            if self.enable_hey_var.get():
                self.celestian_output_text.insert(tk.END, f"🔍 Hey classifications: {hey_processed}\n")
            if self.enable_celestian_var.get():
                self.celestian_output_text.insert(tk.END, f"🌟 Hey-Celestian classifications: {celestian_processed}\n")
            self.celestian_output_text.insert(tk.END, f"💾 Results saved to: {output_file}\n\n")
            
            self.celestian_output_text.see(tk.END)
            
            messagebox.showinfo("Success", f"Dual classification complete!\n\nProcessed {total_rows} minerals\nResults saved to:\n{output_file}")
            
        except Exception as e:
            self.celestian_progress.stop()
            self.celestian_output_text.insert(tk.END, f"❌ Error: {str(e)}\n")
            messagebox.showerror("Error", f"Error running dual classification: {str(e)}")

    def generate_dual_comparison(self):
        """Generate comparison report for both classification systems."""
        input_file = self.celestian_file_var.get().strip()
        if not input_file:
            messagebox.showerror("Error", "Please select an input CSV file")
            return
        
        if not os.path.exists(input_file):
            messagebox.showerror("Error", f"Input file not found: {input_file}")
            return
        
        try:
            # Generate output filename
            input_dir = os.path.dirname(input_file)
            input_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(input_dir, f"{input_name}_Dual_Comparison_Report.csv")
            
            self.celestian_output_text.insert(tk.END, f"📈 Generating dual comparison report...\n")
            self.root.update()
            
            # Read the data
            df = pd.read_csv(input_file)
            
            # Generate summary statistics
            stats = {
                'Total Minerals': len(df),
                'Traditional Hey Classifications': 0,
                'Hey-Celestian Classifications': 0,
                'High Confidence Hey-Celestian (>0.8)': 0,
                'Both Systems Available': 0
            }
            
            if 'Improved Hey ID' in df.columns:
                stats['Traditional Hey Classifications'] = len(df[df['Improved Hey ID'].notna() & (df['Improved Hey ID'] != '')])
            
            if 'Hey-Celestian Group ID' in df.columns:
                stats['Hey-Celestian Classifications'] = len(df[df['Hey-Celestian Group ID'].notna() & (df['Hey-Celestian Group ID'] != '')])
            
            if 'Hey-Celestian Confidence' in df.columns:
                confidence_col = pd.to_numeric(df['Hey-Celestian Confidence'], errors='coerce')
                stats['High Confidence Hey-Celestian (>0.8)'] = len(confidence_col[confidence_col > 0.8])
            
            # Count minerals with both classifications
            if ('Improved Hey ID' in df.columns and 'Hey-Celestian Group ID' in df.columns):
                both_available = df[
                    (df['Improved Hey ID'].notna() & (df['Improved Hey ID'] != '')) &
                    (df['Hey-Celestian Group ID'].notna() & (df['Hey-Celestian Group ID'] != ''))
                ]
                stats['Both Systems Available'] = len(both_available)
            
            # Save summary
            summary_df = pd.DataFrame([stats])
            summary_df.to_csv(output_file, index=False)
            
            self.celestian_output_text.insert(tk.END, f"✅ Dual comparison report generated!\n")
            self.celestian_output_text.insert(tk.END, f"📊 Summary Statistics:\n")
            for key, value in stats.items():
                self.celestian_output_text.insert(tk.END, f"   {key}: {value}\n")
            self.celestian_output_text.insert(tk.END, f"💾 Report saved to: {output_file}\n\n")
            
            self.celestian_output_text.see(tk.END)
            
            messagebox.showinfo("Success", f"Dual comparison report generated!\n\nReport saved to:\n{output_file}")
            
        except Exception as e:
            self.celestian_output_text.insert(tk.END, f"❌ Error: {str(e)}\n")
            messagebox.showerror("Error", f"Error generating comparison report: {str(e)}")

    def select_all_minerals(self):
        """Select all minerals in the current group"""
        self.minerals_list.selection_set(0, tk.END)

    def deselect_all_minerals(self):
        """Deselect all minerals"""
        self.minerals_list.selection_clear(0, tk.END)
        self.current_mineral = None
        self.update_metadata_editor(None)

    def move_minerals_to_group(self):
        """Move selected minerals to the target group"""
        selected_indices = self.minerals_list.curselection()
        if not selected_indices or not self.target_group_var.get():
            messagebox.showerror("Error", "Please select both minerals and a target group")
            return
            
        try:
            # Get target group ID from display value
            target_group_id = self.group_display_to_id[self.target_group_var.get()]
            
            # Get selected minerals
            selected_minerals = [self.minerals_list.get(i) for i in selected_indices]
            
            # Update the DataFrame
            for mineral in selected_minerals:
                self.df.loc[self.df['Mineral Name'] == mineral, 'Hey Classification ID'] = target_group_id
            
            # Update the minerals list
            self.update_minerals_list(self.current_group_id)
            
            # Clear the current mineral selection
            self.minerals_list.selection_clear(0, tk.END)
            self.current_mineral = None
            
            # Update metadata editor
            self.update_metadata_editor(None)
            
            messagebox.showinfo("Success", f"Moved {len(selected_minerals)} minerals to group {target_group_id}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error moving minerals: {str(e)}")

    def on_group_select(self, event):
        """Handle selection of a Hey group"""
        selected_item = self.groups_tree.selection()
        if not selected_item:
            return
        
        group_id = self.groups_tree.item(selected_item[0])['values'][0]
        self.current_group_id = group_id
        self.update_minerals_list(group_id)
        self.update_group_dropdown()

    def on_mineral_select(self, event):
        """Handle selection of a mineral"""
        selected_indices = self.minerals_list.curselection()
        if not selected_indices:
            return
        
        mineral_name = self.minerals_list.get(selected_indices[0])
        self.current_mineral = mineral_name
        self.update_metadata_editor(mineral_name)

    def update_minerals_list(self, group_id):
        """Update the minerals list based on selected group"""
        self.minerals_list.delete(0, tk.END)
        
        if self.df is None:
            return
        
        # Filter minerals by group
        group_minerals = self.df[self.df['Hey Classification ID'] == group_id]['Mineral Name'].tolist()
        for mineral in sorted(group_minerals):
            self.minerals_list.insert(tk.END, mineral)

    def update_metadata_editor(self, mineral_name):
        """Update the metadata editor with mineral data"""
        # Clear existing fields
        for widget in self.metadata_editor.winfo_children():
            widget.destroy()
        
        if self.df is None:
            return
        
        # Get mineral data
        mineral_data = self.df[self.df['Mineral Name'] == mineral_name].iloc[0]
        
        # Create fields for each column
        row = 0
        self.metadata_fields = {}
        
        for col in self.df.columns:
            ttk.Label(self.metadata_editor, text=col).grid(row=row, column=0, sticky=tk.W, pady=2)
            
            if col in ['Mineral Name', 'Hey Classification ID']:
                # These fields are read-only
                ttk.Label(self.metadata_editor, text=str(mineral_data[col])).grid(row=row, column=1, sticky=tk.W, pady=2)
            else:
                # Create entry field
                var = tk.StringVar(value=str(mineral_data[col]))
                entry = ttk.Entry(self.metadata_editor, textvariable=var)
                entry.grid(row=row, column=1, sticky=tk.W, pady=2)
                self.metadata_fields[col] = var
            
            row += 1

    def save_database_changes(self):
        """Save changes to the database"""
        if self.df is None or self.current_mineral is None:
            messagebox.showerror("Error", "No mineral selected or database not loaded")
            return
        
        try:
            # Update the DataFrame with changes
            for col, var in self.metadata_fields.items():
                self.df.loc[self.df['Mineral Name'] == self.current_mineral, col] = var.get()
            
            # Ask for save location
            output_file = filedialog.asksaveasfilename(
                title="Save Database",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )
            
            if output_file:
                # Save the updated DataFrame
                self.df.to_csv(output_file, index=False)
                messagebox.showinfo("Success", f"Database saved to {output_file}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error saving database: {str(e)}")

    def load_database(self):
        """Load the database and populate the Hey groups tree"""
        try:
            input_file = self.input_file_var.get()
            if not input_file:
                return  # Don't show error if no file selected yet
            
            # Load the database
            self.df = pd.read_csv(input_file)
            
            # Check if this is a processed file with Hey Classification columns
            if 'Hey Classification ID' not in self.df.columns or 'Hey Classification Name' not in self.df.columns:
                # This is a raw file - show helpful message
                self.status_var.set("Raw RRUFF file detected. Use the integrated classifier to process the data.")
                return
            
            # Clear existing tree
            for item in self.groups_tree.get_children():
                self.groups_tree.delete(item)
            
            # Get unique Hey groups and sort by ID
            hey_groups = self.df[['Hey Classification ID', 'Hey Classification Name']].drop_duplicates()
            hey_groups = hey_groups[hey_groups['Hey Classification ID'] != '0']  # Exclude unclassified
            hey_groups = hey_groups.sort_values('Hey Classification ID')  # Sort by ID
            
            # Add groups to tree
            for _, row in hey_groups.iterrows():
                self.groups_tree.insert('', 'end', values=(row['Hey Classification ID'], row['Hey Classification Name']))
            
            # Update the group dropdown
            self.update_group_dropdown()
            
            # Enable the editor tab
            self.notebook.tab(self.editor_tab_index, state="normal")
            
        except Exception as e:
            # Don't show error for raw files, just update status
            if "Hey Classification" in str(e):
                self.status_var.set("Raw RRUFF file detected. Use the integrated classifier to process the data.")
            else:
                messagebox.showerror("Error", f"Error loading database: {str(e)}")

    def browse_input_file(self):
        """Open file dialog to select input file"""
        filename = filedialog.askopenfilename(
            title="Select Input File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.input_file_var.set(filename)
            # Load the database when a file is selected
            self.load_database()

    def browse_output_dir(self):
        """Open directory dialog to select output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_var.set(directory)

    def run_analysis(self):
        """Run the selected analysis options"""
        try:
            input_file = self.input_file_var.get()
            output_dir = self.output_dir_var.get()
            
            if not input_file or not output_dir:
                messagebox.showerror("Error", "Please select both input file and output directory")
                return
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Run the main analysis function
            analyze_hey_classification_relationships(input_file, output_dir)
            
            messagebox.showinfo("Success", "Analysis completed successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error running analysis: {str(e)}")



    def update_group_dropdown(self):
        """Update the target group dropdown with available groups"""
        if self.df is None:
            return
            
        # Get unique Hey groups
        hey_groups = self.df[['Hey Classification ID', 'Hey Classification Name']].drop_duplicates()
        hey_groups = hey_groups[hey_groups['Hey Classification ID'] != '0']  # Exclude unclassified
        hey_groups = hey_groups.sort_values('Hey Classification ID')
        
        # Create display values for dropdown
        display_values = [f"{row['Hey Classification ID']} - {row['Hey Classification Name']}" 
                         for _, row in hey_groups.iterrows()]
        
        # Update dropdown values
        self.target_group_dropdown['values'] = display_values
        
        # Store the mapping of display values to IDs
        self.group_display_to_id = {}
        for _, row in hey_groups.iterrows():
            display = f"{row['Hey Classification ID']} - {row['Hey Classification Name']}"
            self.group_display_to_id[display] = row['Hey Classification ID']

    def validate_classifications(self):
        """Validate mineral classifications against reference data"""
        if not self.df is not None:
            messagebox.showerror("Error", "Please load a database first")
            return
            
        try:
            # Create progress window
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Validating Classifications")
            progress_window.geometry("400x150")
            
            # Add progress label
            ttk.Label(progress_window, text="Validating mineral classifications...").pack(pady=10)
            
            # Add progress bar
            progress = ttk.Progressbar(progress_window, length=300, mode='determinate')
            progress.pack(pady=10)
            
            # Add status label
            status_label = ttk.Label(progress_window, text="")
            status_label.pack(pady=10)
            
            # Get unique minerals
            minerals = self.df['Mineral Name'].unique()
            total_minerals = len(minerals)
            
            # Initialize results
            mismatches = []
            
            # Update progress
            progress['maximum'] = total_minerals
            
            # Validate each mineral
            for i, mineral in enumerate(minerals):
                # Update progress
                progress['value'] = i + 1
                status_label['text'] = f"Checking {mineral} ({i+1}/{total_minerals})"
                progress_window.update()
                
                # Get current classification
                current_class = self.df[self.df['Mineral Name'] == mineral]['Hey Classification ID'].iloc[0]
                
                # Get reference classification
                ref_class = self.get_reference_classification(mineral)
                
                if ref_class and ref_class != current_class:
                    mismatches.append({
                        'mineral': mineral,
                        'current': current_class,
                        'reference': ref_class
                    })
                
                # Small delay to prevent overwhelming the server
                time.sleep(0.1)
            
            # Close progress window
            progress_window.destroy()
            
            # Show results
            if mismatches:
                self.show_validation_results(mismatches)
            else:
                messagebox.showinfo("Validation Complete", "No classification mismatches found!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error validating classifications: {str(e)}")

    def get_reference_classification(self, mineral):
        """Get the reference classification for a mineral from Mindat"""
        try:
            # Check if we already have the reference data
            if mineral in self.reference_data:
                return self.reference_data[mineral]
            
            # Construct URL for Mindat
            url = f"https://www.mindat.org/min-{mineral.lower().replace(' ', '_')}.html"
            
            # Get the page
            response = requests.get(url)
            if response.status_code != 200:
                return None
            
            # Parse the page
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for Hey classification
            hey_section = soup.find('div', string=lambda x: x and 'Hey\'s CIM Ref.' in x)
            if hey_section:
                hey_text = hey_section.find_next('div').text.strip()
                # Extract the Hey classification ID
                hey_id = hey_text.split('.')[0]
                self.reference_data[mineral] = hey_id
                return hey_id
            
            return None
            
        except Exception as e:
            print(f"Error getting reference for {mineral}: {str(e)}")
            return None

    def show_validation_results(self, mismatches):
        """Show validation results in a new window"""
        # Create results window
        results_window = tk.Toplevel(self.root)
        results_window.title("Classification Validation Results")
        results_window.geometry("600x400")
        
        # Create treeview for results
        columns = ("Mineral", "Current Classification", "Reference Classification")
        tree = ttk.Treeview(results_window, columns=columns, show="headings")
        
        # Add headings
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_window, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Add results
        for mismatch in mismatches:
            tree.insert('', 'end', values=(
                mismatch['mineral'],
                mismatch['current'],
                mismatch['reference']
            ))
        
        tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add buttons
        button_frame = ttk.Frame(results_window)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Export Results", 
                  command=lambda: self.export_validation_results(mismatches)).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Close", 
                  command=results_window.destroy).pack(side=tk.RIGHT)

    def export_validation_results(self, mismatches):
        """Export validation results to CSV"""
        try:
            # Ask for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )
            
            if filename:
                # Create DataFrame
                df = pd.DataFrame(mismatches)
                
                # Save to CSV
                df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Results exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting results: {str(e)}")

    # Hey-Celestian Classification Methods
    def browse_celestian_file(self):
        """Browse for Hey-Celestian input file"""
        filename = filedialog.askopenfilename(
            title="Select CSV File for Hey-Celestian Classification",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.celestian_file_var.set(filename)

    def test_single_celestian_mineral(self):
        """Test single mineral with Hey-Celestian classifier"""
        if not self.celestian_classifier:
            messagebox.showerror("Error", "Hey-Celestian classifier not initialized")
            return
        
        # Create dialog for single mineral input
        dialog = tk.Toplevel(self.root)
        dialog.title("Test Single Mineral - Hey-Celestian")
        dialog.geometry("600x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Input fields
        ttk.Label(dialog, text="Mineral Name:").grid(row=0, column=0, sticky='w', padx=10, pady=5)
        name_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=name_var, width=40).grid(row=0, column=1, padx=10, pady=5)
        
        ttk.Label(dialog, text="Chemical Formula:").grid(row=1, column=0, sticky='w', padx=10, pady=5)
        formula_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=formula_var, width=40).grid(row=1, column=1, padx=10, pady=5)
        
        ttk.Label(dialog, text="Elements:").grid(row=2, column=0, sticky='w', padx=10, pady=5)
        elements_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=elements_var, width=40).grid(row=2, column=1, padx=10, pady=5)
        
        # Extract elements button
        def extract_elements():
            formula = formula_var.get().strip()
            if formula:
                try:
                    elements = extract_elements_from_formula(formula)
                    elements_var.set(', '.join(elements))
                except Exception as e:
                    messagebox.showerror("Error", f"Error extracting elements: {str(e)}")
        
        ttk.Button(dialog, text="Extract Elements", command=extract_elements).grid(row=2, column=2, padx=10, pady=5)
        
        # Results area
        ttk.Label(dialog, text="Classification Results:").grid(row=3, column=0, sticky='nw', padx=10, pady=5)
        
        results_frame = ttk.Frame(dialog)
        results_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=5, sticky='nsew')
        
        results_text = tk.Text(results_frame, height=15, wrap='word')
        results_scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=results_text.yview)
        results_text.configure(yscrollcommand=results_scrollbar.set)
        
        results_text.pack(side='left', fill='both', expand=True)
        results_scrollbar.pack(side='right', fill='y')
        
        # Classify button
        def classify_mineral():
            try:
                name = name_var.get().strip()
                formula = formula_var.get().strip()
                elements = elements_var.get().strip()
                
                if not formula:
                    messagebox.showerror("Error", "Please enter a chemical formula")
                    return
                
                result = self.celestian_classifier.classify_mineral(formula, elements, name)
                
                results_text.delete(1.0, tk.END)
                results_text.insert(tk.END, f"🌟 Hey-Celestian Classification Results\n")
                results_text.insert(tk.END, f"{'='*50}\n\n")
                results_text.insert(tk.END, f"Mineral: {name or 'Unknown'}\n")
                results_text.insert(tk.END, f"Formula: {formula}\n")
                results_text.insert(tk.END, f"Elements: {elements}\n\n")
                results_text.insert(tk.END, f"Classification ID: {result['id']}\n")
                results_text.insert(tk.END, f"Classification: {result['name']}\n")
                results_text.insert(tk.END, f"Confidence: {result['confidence']:.2f}\n")
                results_text.insert(tk.END, f"Reasoning: {result['reasoning']}\n\n")
                
                # Show group information
                group_info = self.celestian_classifier.get_classification_info(result['id'])
                if group_info:
                    results_text.insert(tk.END, f"📊 Group Information:\n")
                    results_text.insert(tk.END, f"Description: {group_info.get('description', 'N/A')}\n")
                    results_text.insert(tk.END, f"Typical Range: {group_info.get('typical_range', 'N/A')}\n")
                    results_text.insert(tk.END, f"Examples: {', '.join(group_info.get('examples', []))}\n")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error classifying mineral: {str(e)}")
        
        ttk.Button(dialog, text="🚀 Classify", command=classify_mineral).grid(row=5, column=1, pady=10)
        
        # Configure grid weights
        dialog.grid_rowconfigure(4, weight=1)
        dialog.grid_columnconfigure(1, weight=1)

    def show_celestian_groups(self):
        """Show all Hey-Celestian classification groups"""
        if not self.celestian_classifier:
            messagebox.showerror("Error", "Hey-Celestian classifier not initialized")
            return
        
        # Create dialog to show groups
        dialog = tk.Toplevel(self.root)
        dialog.title("Hey-Celestian Classification Groups")
        dialog.geometry("800x600")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Create treeview for groups
        tree_frame = ttk.Frame(dialog)
        tree_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        columns = ('ID', 'Name', 'Range', 'Examples')
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        tree.heading('ID', text='ID')
        tree.heading('Name', text='Classification Name')
        tree.heading('Range', text='Typical Range')
        tree.heading('Examples', text='Examples')
        
        tree.column('ID', width=50)
        tree.column('Name', width=300)
        tree.column('Range', width=200)
        tree.column('Examples', width=200)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Populate tree with groups
        for group_id, group_info in self.celestian_classifier.vibrational_groups.items():
            examples = ', '.join(group_info.get('examples', [])[:3])  # Show first 3 examples
            if len(group_info.get('examples', [])) > 3:
                examples += '...'
            
            tree.insert('', 'end', values=(
                group_id,
                group_info['name'],
                group_info.get('typical_range', 'N/A'),
                examples
            ))
        
        # Description area
        desc_frame = ttk.LabelFrame(dialog, text="Group Description", padding=10)
        desc_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        desc_text = tk.Text(desc_frame, height=4, wrap='word')
        desc_text.pack(fill='x')
        
        def on_group_select(event):
            selection = tree.selection()
            if selection:
                item = tree.item(selection[0])
                group_id = item['values'][0]
                group_info = self.celestian_classifier.vibrational_groups.get(str(group_id), {})
                
                desc_text.delete(1.0, tk.END)
                desc_text.insert(tk.END, f"Group {group_id}: {group_info.get('name', 'Unknown')}\n\n")
                desc_text.insert(tk.END, f"Description: {group_info.get('description', 'N/A')}\n")
                desc_text.insert(tk.END, f"Typical Range: {group_info.get('typical_range', 'N/A')}\n")
                desc_text.insert(tk.END, f"Examples: {', '.join(group_info.get('examples', []))}")
        
        tree.bind('<<TreeviewSelect>>', on_group_select)

    def run_celestian_classification(self):
        """Run Hey-Celestian classification on the selected file"""
        if not self.celestian_classifier:
            messagebox.showerror("Error", "Hey-Celestian classifier not initialized")
            return
        
        input_file = self.celestian_file_var.get().strip()
        if not input_file:
            messagebox.showerror("Error", "Please select an input CSV file")
            return
        
        if not os.path.exists(input_file):
            messagebox.showerror("Error", f"Input file not found: {input_file}")
            return
        
        try:
            # Start progress
            self.celestian_progress.start()
            self.celestian_output_text.insert(tk.END, f"🚀 Starting Hey-Celestian classification...\n")
            self.celestian_output_text.insert(tk.END, f"📁 Input file: {input_file}\n\n")
            self.root.update()
            
            # Create backup if requested
            if self.celestian_create_backup_var.get():
                backup_file = f"{input_file}.celestian_backup"
                import shutil
                shutil.copy2(input_file, backup_file)
                self.celestian_output_text.insert(tk.END, f"📁 Backup created: {backup_file}\n")
            
            # Read CSV
            df = pd.read_csv(input_file)
            total_rows = len(df)
            
            self.celestian_output_text.insert(tk.END, f"📊 Processing {total_rows} minerals...\n")
            self.root.update()
            
            # Apply Hey-Celestian classification
            celestian_ids = []
            celestian_names = []
            confidences = []
            reasonings = []
            
            for i, row in df.iterrows():
                mineral_name = row.get('Mineral Name', '')
                chemistry = row.get('RRUFF Chemistry (concise)', '')
                elements = row.get('Chemistry Elements', '')
                
                if chemistry:
                    result = self.celestian_classifier.classify_mineral(chemistry, elements, mineral_name)
                    celestian_ids.append(result['id'])
                    celestian_names.append(result['name'])
                    confidences.append(result['confidence'])
                    reasonings.append(result['reasoning'])
                else:
                    celestian_ids.append('0')
                    celestian_names.append('Unclassified')
                    confidences.append(0.0)
                    reasonings.append('No chemical formula provided')
                
                if (i + 1) % 100 == 0:
                    self.celestian_output_text.insert(tk.END, f"⏳ Processed {i+1}/{total_rows} minerals...\n")
                    self.root.update()
            
            # Add new columns
            df['Hey-Celestian Group ID'] = celestian_ids
            df['Hey-Celestian Group Name'] = celestian_names
            df['Hey-Celestian Confidence'] = confidences
            df['Hey-Celestian Reasoning'] = reasonings
            
            # Generate output filename
            input_dir = os.path.dirname(input_file)
            input_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(input_dir, f"{input_name}_Hey_Celestian.csv")
            
            # Save results
            df.to_csv(output_file, index=False)
            
            # Stop progress
            self.celestian_progress.stop()
            
            # Show results
            high_confidence = sum(1 for c in confidences if c > 0.8)
            group_distribution = {}
            for group_name in celestian_names:
                group_distribution[group_name] = group_distribution.get(group_name, 0) + 1
            
            self.celestian_output_text.insert(tk.END, f"\n✅ Hey-Celestian classification complete!\n")
            self.celestian_output_text.insert(tk.END, f"📊 Processed: {total_rows} minerals\n")
            self.celestian_output_text.insert(tk.END, f"🎯 High confidence (>0.8): {high_confidence} minerals\n")
            self.celestian_output_text.insert(tk.END, f"💾 Results saved to: {output_file}\n\n")
            
            self.celestian_output_text.see(tk.END)
            
            messagebox.showinfo("Success", f"Hey-Celestian classification complete!\n\nProcessed {total_rows} minerals\nHigh confidence: {high_confidence} minerals\n\nResults saved to:\n{output_file}")
            
        except Exception as e:
            self.celestian_progress.stop()
            self.celestian_output_text.insert(tk.END, f"❌ Error: {str(e)}\n")
            messagebox.showerror("Error", f"Error running Hey-Celestian classification: {str(e)}")

    def generate_celestian_comparison(self):
        """Generate comparison report between traditional Hey and Hey-Celestian classifications"""
        input_file = self.celestian_file_var.get().strip()
        if not input_file:
            messagebox.showerror("Error", "Please select an input CSV file")
            return
        
        if not os.path.exists(input_file):
            messagebox.showerror("Error", f"Input file not found: {input_file}")
            return
        
        try:
            # Generate output filename
            input_dir = os.path.dirname(input_file)
            input_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(input_dir, f"{input_name}_Comparison_Report.csv")
            
            self.celestian_output_text.insert(tk.END, f"📈 Generating comparison report...\n")
            self.root.update()
            
            # Use the create_hey_celestian_classification_report function
            results_df, summary = create_hey_celestian_classification_report(input_file, output_file)
            
            self.celestian_output_text.insert(tk.END, f"✅ Comparison report generated!\n")
            self.celestian_output_text.insert(tk.END, f"📊 Summary Statistics:\n")
            self.celestian_output_text.insert(tk.END, f"   Total Minerals: {summary['Total Minerals']}\n")
            self.celestian_output_text.insert(tk.END, f"   Hey-Celestian Groups Used: {summary['Hey-Celestian Groups Used']}\n")
            self.celestian_output_text.insert(tk.END, f"   Average Confidence: {summary['Average Confidence']:.3f}\n")
            self.celestian_output_text.insert(tk.END, f"   High Confidence (>0.8): {summary['High Confidence (>0.8)']}\n")
            self.celestian_output_text.insert(tk.END, f"💾 Report saved to: {output_file}\n\n")
            
            self.celestian_output_text.see(tk.END)
            
            messagebox.showinfo("Success", f"Comparison report generated!\n\nReport saved to:\n{output_file}")
            
        except Exception as e:
            self.celestian_output_text.insert(tk.END, f"❌ Error: {str(e)}\n")
            messagebox.showerror("Error", f"Error generating comparison report: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = HeyClassificationTool(root)
    root.mainloop() 