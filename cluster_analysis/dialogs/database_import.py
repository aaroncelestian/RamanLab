"""
Database Import Dialog for RamanLab Cluster Analysis

This module contains the dialog for importing spectra from the database
into the cluster analysis tool.
"""

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, 
                              QLabel, QLineEdit, QPushButton, QTableWidget, 
                              QTableWidgetItem, QComboBox, QCheckBox, QMessageBox,
                              QProgressDialog, QApplication)
from PySide6.QtCore import Qt, QThread, Signal
from joblib import Parallel, delayed
import multiprocessing
import time


def extract_spectrum_metadata(item_data):
    """Extract metadata from a single spectrum (for parallel processing)."""
    try:
        if isinstance(item_data, tuple):
            name, spectrum_data = item_data
        else:
            spectrum_data = item_data
            name = getattr(spectrum_data, 'name', 'Unknown')
        
        # Extract metadata
        metadata = spectrum_data.get('metadata', spectrum_data)
        
        mineral_name = (metadata.get('NAME') or metadata.get('name') or 
                       metadata.get('mineral_name') or name or 'Unknown')
        
        formula = (metadata.get('FORMULA') or metadata.get('Formula') or 
                  metadata.get('formula') or metadata.get('CHEMICAL_FORMULA') or 
                  metadata.get('Chemical_Formula') or metadata.get('chemical_formula') or 
                  metadata.get('IDEAL CHEMISTRY') or metadata.get('Ideal Chemistry') or 
                  metadata.get('ideal_chemistry') or metadata.get('IDEAL_CHEMISTRY') or 
                  metadata.get('composition') or spectrum_data.get('formula', ''))
        
        description = (metadata.get('DESCRIPTION') or metadata.get('Description') or 
                      metadata.get('description') or metadata.get('desc') or '')
        
        hey_class = (metadata.get('HEY CLASSIFICATION') or metadata.get('Hey Classification') or 
                    metadata.get('hey_classification') or metadata.get('HEY_CLASSIFICATION') or
                    metadata.get('classification') or metadata.get('mineral_class') or
                    metadata.get('MINERAL_CLASS') or '')
        
        chemical_family = (metadata.get('CHEMICAL FAMILY') or metadata.get('Chemical Family') or 
                          metadata.get('chemical_family') or metadata.get('CHEMICAL_FAMILY') or
                          metadata.get('family') or metadata.get('anion_group') or
                          metadata.get('ANION_GROUP') or '')
        
        wavenumbers = spectrum_data.get('wavenumbers', [])
        data_points = len(wavenumbers) if wavenumbers is not None else 0
        
        return {
            'name': name,
            'mineral': mineral_name,
            'formula': formula,
            'hey_classification': hey_class,
            'chemical_family': chemical_family,
            'description': description,
            'data_points': data_points,
            'spectrum_data': spectrum_data
        }
    except Exception as e:
        return None


class DatabaseImportDialog(QDialog):
    """Dialog for selecting spectra from the database for import into cluster analysis."""
    
    def __init__(self, raman_db, parent=None):
        """Initialize the database import dialog."""
        super().__init__(parent)
        self.raman_db = raman_db
        self.selected_spectra = {}
        self.all_spectra = []  # Store all spectra metadata
        self.filtered_spectra = []  # Store currently filtered spectra
        self.setup_ui()
        self.load_database_spectra()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Search controls
        search_group = QGroupBox("Search & Filter")
        search_group_layout = QVBoxLayout(search_group)
        
        # Text search
        text_search_layout = QHBoxLayout()
        text_search_layout.addWidget(QLabel("Text Search:"))
        self.search_entry = QLineEdit()
        self.search_entry.setPlaceholderText("Search by name, mineral, formula, description...")
        self.search_entry.textChanged.connect(self.filter_spectra)
        text_search_layout.addWidget(self.search_entry)
        
        clear_search_btn = QPushButton("Clear")
        clear_search_btn.clicked.connect(self.clear_search)
        text_search_layout.addWidget(clear_search_btn)
        
        search_group_layout.addLayout(text_search_layout)
        
        # Filter dropdowns
        filter_layout = QHBoxLayout()
        
        # Hey Classification filter
        filter_layout.addWidget(QLabel("Hey Classification:"))
        self.hey_filter = QComboBox()
        self.hey_filter.setEditable(True)
        self.hey_filter.addItem("")  # Empty option for "All"
        self.hey_filter.currentTextChanged.connect(self.filter_spectra)
        filter_layout.addWidget(self.hey_filter)
        
        # Chemical Family filter (anion types)
        filter_layout.addWidget(QLabel("Chemical Family:"))
        self.family_filter = QComboBox()
        self.family_filter.setEditable(True)
        self.family_filter.addItem("")  # Empty option for "All"
        self.family_filter.currentTextChanged.connect(self.filter_spectra)
        filter_layout.addWidget(self.family_filter)
        
        search_group_layout.addLayout(filter_layout)
        layout.addWidget(search_group)
        
        # Spectra table
        self.spectra_table = QTableWidget()
        self.spectra_table.setColumnCount(8)
        self.spectra_table.setHorizontalHeaderLabels([
            "Select", "Name", "Mineral", "Formula", "Hey Classification", "Chemical Family", "Data Points", "Description"
        ])
        
        # Connect itemChanged signal to update selection count
        self.spectra_table.itemChanged.connect(self.update_selection_count)
        
        # Set column widths
        header = self.spectra_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.resizeSection(0, 60)   # Select column
        header.resizeSection(1, 150)  # Name column
        header.resizeSection(2, 120)  # Mineral column
        header.resizeSection(3, 100)  # Formula column
        header.resizeSection(4, 150)  # Hey Classification column
        header.resizeSection(5, 120)  # Chemical Family column
        header.resizeSection(6, 80)   # Data Points column
        
        layout.addWidget(self.spectra_table)
        
        # Selection controls
        selection_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all)
        selection_layout.addWidget(select_all_btn)
        
        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(self.select_none)
        selection_layout.addWidget(select_none_btn)
        
        selection_layout.addStretch()
        
        self.selection_label = QLabel("0 spectra selected")
        selection_layout.addWidget(self.selection_label)
        
        layout.addLayout(selection_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        import_btn = QPushButton("Import Selected")
        import_btn.clicked.connect(self.accept_selection)
        import_btn.setDefault(True)
        button_layout.addWidget(import_btn)
        
        layout.addLayout(button_layout)
    
    def load_database_spectra(self):
        """Load spectra from the Raman database into the table."""
        try:
            overall_start = time.time()
            print(f"\n{'='*60}")
            print(f"DEBUG: Loading database spectra...")
            print(f"DEBUG: raman_db type = {type(self.raman_db)}")
            
            # Check database structure
            if hasattr(self.raman_db, 'database'):
                print(f"DEBUG: Found database attribute with {len(self.raman_db.database)} entries")
                spectra_source = self.raman_db.database
            elif hasattr(self.raman_db, '__len__'):
                print(f"DEBUG: Database is directly accessible with {len(self.raman_db)} entries")
                spectra_source = self.raman_db
            else:
                print(f"DEBUG: Unexpected database structure: {dir(self.raman_db)}")
                self.spectra_table.setRowCount(1)
                self.spectra_table.setItem(0, 1, QTableWidgetItem("Invalid database structure"))
                return
            
            if not spectra_source:
                print("DEBUG: No spectra found in database")
                self.spectra_table.setRowCount(1)
                self.spectra_table.setItem(0, 1, QTableWidgetItem("No spectra found in database"))
                return
            
            spectra_list = []
            
            # Handle different database structures
            if hasattr(spectra_source, 'items'):
                print("DEBUG: Using database.items() approach")
                iterator = spectra_source.items()
            elif hasattr(spectra_source, '__iter__'):
                print("DEBUG: Using direct iteration approach")
                iterator = spectra_source
            else:
                print("DEBUG: Cannot iterate over database")
                self.spectra_table.setRowCount(1)
                self.spectra_table.setItem(0, 1, QTableWidgetItem("Cannot access database entries"))
                return
            
            count = 0
            for item in iterator:
                count += 1
                if count > 5:  # Limit debug output for first few items
                    break
                    
                try:
                    if hasattr(spectra_source, 'items'):
                        name, spectrum_data = item
                    else:
                        spectrum_data = item
                        name = getattr(spectrum_data, 'name', f'Spectrum_{count}')
                    
                    print(f"DEBUG: Processing spectrum {count}: {name}")
                    print(f"DEBUG: Spectrum data type = {type(spectrum_data)}")
                    if hasattr(spectrum_data, 'keys'):
                        keys = list(spectrum_data.keys())
                        print(f"DEBUG: Spectrum data keys = {keys[:10]}...")  # Show first 10 keys
                    
                except Exception as e:
                    print(f"DEBUG: Error in initial processing: {str(e)}")
                    continue
            
            # Process ALL spectra with parallel processing for speed
            metadata_start = time.time()
            print("DEBUG: Processing all spectra for table...")
            
            total_spectra = len(spectra_source)
            
            # Show progress dialog
            progress = QProgressDialog("Loading database spectra...", None, 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(10)
            progress.setLabelText(f"Processing {total_spectra:,} spectra in parallel...")
            QApplication.processEvents()
            
            # Convert to list for parallel processing
            list_start = time.time()
            if hasattr(spectra_source, 'items'):
                items_list = list(spectra_source.items())
            else:
                items_list = list(spectra_source)
            list_time = time.time() - list_start
            print(f"⏱️  List conversion: {list_time:.2f}s")
            
            progress.setValue(20)
            progress.setLabelText(f"Extracting metadata using {multiprocessing.cpu_count()} cores...")
            QApplication.processEvents()
            
            # Use parallel processing for metadata extraction
            parallel_start = time.time()
            n_jobs = -1  # Use all available cores
            spectra_list = Parallel(n_jobs=n_jobs, backend='threading', verbose=0)(
                delayed(extract_spectrum_metadata)(item) for item in items_list
            )
            parallel_time = time.time() - parallel_start
            print(f"⏱️  Parallel metadata extraction: {parallel_time:.2f}s")
            
            # Filter out None results (failed extractions)
            spectra_list = [s for s in spectra_list if s is not None]
            
            progress.setValue(90)
            progress.setLabelText(f"Processed {len(spectra_list):,} spectra successfully")
            QApplication.processEvents()
            
            metadata_time = time.time() - metadata_start
            print(f"⏱️  TOTAL metadata processing: {metadata_time:.2f}s")
            print(f"DEBUG: Total processed {len(spectra_list)} spectra successfully")
            
            # Sort by name
            spectra_list.sort(key=lambda x: x['name'])
            
            # Store all spectra for filtering
            self.all_spectra = spectra_list
            
            # Collect unique values for filters (fast - no UI updates)
            progress.setValue(92)
            progress.setLabelText("Collecting filter options...")
            QApplication.processEvents()
            
            hey_classifications = set()
            chemical_families = set()
            for spectrum_info in spectra_list:
                if spectrum_info['hey_classification']:
                    hey_classifications.add(spectrum_info['hey_classification'])
                if spectrum_info['chemical_family']:
                    chemical_families.add(spectrum_info['chemical_family'])
            
            # Populate filter dropdowns
            self._populate_hey_filter(hey_classifications)
            self._populate_family_filter(chemical_families)
            
            # Use lazy loading - only populate first 100 rows for instant loading
            progress.setValue(94)
            progress.setLabelText(f"Preparing table (lazy loading {len(spectra_list):,} spectra)...")
            QApplication.processEvents()
            
            # CRITICAL: Only populate first 100 rows for instant loading
            MAX_INITIAL_ROWS = 100
            initial_rows = min(MAX_INITIAL_ROWS, len(spectra_list))
            
            print(f"🚀 LAZY LOADING: Populating only first {initial_rows} rows (out of {len(spectra_list):,})")
            
            # Use helper function to populate initial rows
            self._populate_table_rows(spectra_list[:initial_rows])
            
            # Add info message about lazy loading
            if len(spectra_list) > MAX_INITIAL_ROWS:
                print(f"ℹ️  Showing first {initial_rows} of {len(spectra_list):,} spectra. Use search/filter to find specific spectra.")
            
            progress.setValue(100)
            progress.close()
            
            overall_time = time.time() - overall_start
            print(f"\n{'='*60}")
            print(f"⏱️  OVERALL DIALOG LOAD TIME: {overall_time:.2f}s")
            print(f"{'='*60}\n")
            
        except Exception as e:
            QMessageBox.critical(self, "Database Error", f"Failed to load database spectra:\n{str(e)}")
    
    def _populate_table_rows(self, spectra_list):
        """Populate table with a list of spectra (helper for filtering and initial load)."""
        populate_start = time.time()
        
        # Disable updates during population
        self.spectra_table.setUpdatesEnabled(False)
        self.spectra_table.setSortingEnabled(False)
        self.spectra_table.setRowCount(len(spectra_list))
        
        for row, spectrum_info in enumerate(spectra_list):
            # Use checkable item instead of QCheckBox widget
            select_item = QTableWidgetItem()
            select_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            select_item.setCheckState(Qt.Unchecked)
            self.spectra_table.setItem(row, 0, select_item)
            
            # Spectrum information
            name_item = QTableWidgetItem(spectrum_info['name'])
            name_item.setData(Qt.UserRole, spectrum_info['spectrum_data'])
            self.spectra_table.setItem(row, 1, name_item)
            self.spectra_table.setItem(row, 2, QTableWidgetItem(spectrum_info['mineral']))
            self.spectra_table.setItem(row, 3, QTableWidgetItem(spectrum_info['formula']))
            self.spectra_table.setItem(row, 4, QTableWidgetItem(spectrum_info['hey_classification']))
            self.spectra_table.setItem(row, 5, QTableWidgetItem(spectrum_info['chemical_family']))
            self.spectra_table.setItem(row, 6, QTableWidgetItem(str(spectrum_info['data_points'])))
            self.spectra_table.setItem(row, 7, QTableWidgetItem(spectrum_info['description']))
        
        # Re-enable updates
        self.spectra_table.setSortingEnabled(True)
        self.spectra_table.setUpdatesEnabled(True)
        
        populate_time = time.time() - populate_start
        print(f"⏱️  Table repopulation: {populate_time:.2f}s ({len(spectra_list)} rows)")
        
        self.update_selection_count()
    
    def filter_spectra(self):
        """Filter the spectra table based on search text and dropdown filters."""
        filter_start = time.time()
        search_text = self.search_entry.text().lower()
        hey_filter = self.hey_filter.currentText().strip()
        family_filter = self.family_filter.currentText().strip()
        
        # Filter from the full spectra list
        filtered_list = []
        for spectrum_info in self.all_spectra:
            should_show = True
            
            # Text search filter
            if search_text:
                text_match = False
                # Check all text fields
                searchable_text = f"{spectrum_info['name']} {spectrum_info['mineral']} {spectrum_info['formula']} {spectrum_info['description']}".lower()
                if search_text in searchable_text:
                    text_match = True
                if not text_match:
                    should_show = False
            
            # Hey classification filter
            if should_show and hey_filter:
                should_show = self._matches_hey_classification(spectrum_info['hey_classification'], hey_filter)
            
            # Chemical family filter
            if should_show and family_filter:
                should_show = self._matches_chemical_family(spectrum_info['chemical_family'], family_filter)
            
            if should_show:
                filtered_list.append(spectrum_info)
        
        filter_time = time.time() - filter_start
        print(f"⏱️  Filtering: {filter_time:.2f}s ({len(filtered_list)} of {len(self.all_spectra)} spectra match)")
        
        # Store the full filtered list for "Select All" functionality
        self.filtered_spectra = filtered_list
        
        # Repopulate table with filtered results (limit to 3000 for performance)
        MAX_DISPLAY = 3000
        self._populate_table_rows(filtered_list[:MAX_DISPLAY])
        
        if len(filtered_list) > MAX_DISPLAY:
            print(f"ℹ️  Showing first {MAX_DISPLAY:,} of {len(filtered_list):,} filtered results. Refine search for more specific results.")
            QMessageBox.information(self, "Large Result Set", 
                f"Found {len(filtered_list):,} matching spectra.\n\n"
                f"Showing first {MAX_DISPLAY:,} for performance.\n\n"
                f"Note: 'Select All' will select ALL {len(filtered_list):,} matching spectra,\n"
                f"not just the displayed {MAX_DISPLAY:,}.")
    
    def _matches_hey_classification(self, classification_text, filter_text):
        """
        Strict Hey classification matching logic.
        
        Returns True if the classification matches the filter criteria.
        For pure classifications (e.g., "Borates"), only matches exact "Borates", 
        not "Borates with other anions".
        """
        if not classification_text or not filter_text:
            return False
        
        classification_lower = classification_text.lower().strip()
        filter_lower = filter_text.lower().strip()
        
        # Handle compound filter option (remove [Compound] prefix)
        if filter_lower.startswith('[compound] '):
            compound_filter = filter_lower[11:]  # Remove '[compound] ' prefix
            # For compound filters, allow exact match of the full compound string
            return classification_lower == compound_filter
        
        # Exact match first (highest priority)
        if classification_lower == filter_lower:
            return True
        
        # For strict filtering, only allow exact matches of individual components
        # Split classification on common delimiters
        class_parts = []
        for delimiter in [',', ' or ', ' and ', ';', '/', '|']:
            if delimiter in classification_text:
                class_parts = [part.strip() for part in classification_text.split(delimiter)]
                break
        
        if not class_parts:
            class_parts = [classification_text.strip()]
        
        # Check if any part is an EXACT match to the filter
        for part in class_parts:
            part_lower = part.lower().strip()
            if part_lower == filter_lower:
                return True
        
        return False
    
    def _matches_chemical_family(self, family_text, filter_text):
        """
        Strict chemical family matching logic.
        
        Returns True if the family matches the filter criteria.
        For pure families (e.g., "Borates"), only matches exact "Borates", 
        not "Borates with other anions".
        """
        if not family_text or not filter_text:
            return False
        
        family_lower = family_text.lower().strip()
        filter_lower = filter_text.lower().strip()
        
        # Handle compound filter option (remove [Compound] prefix)
        if filter_lower.startswith('[compound] '):
            compound_filter = filter_lower[11:]  # Remove '[compound] ' prefix
            # For compound filters, allow exact match of the full compound string
            return family_lower == compound_filter
        
        # Exact match first
        if family_lower == filter_lower:
            return True
        
        # For strict filtering, only allow exact matches of individual components
        # Split family on common delimiters
        family_parts = []
        for delimiter in [',', ' or ', ' and ', ';', '/', '|', ' - ']:
            if delimiter in family_text:
                family_parts = [part.strip() for part in family_text.split(delimiter)]
                break
        
        if not family_parts:
            family_parts = [family_text.strip()]
        
        # Check if any part is an EXACT match to the filter
        for part in family_parts:
            part_lower = part.lower().strip()
            if part_lower == filter_lower:
                return True
        
        return False
    
    def _populate_hey_filter(self, hey_classifications):
        """
        Populate Hey classification filter with organized options.
        
        Separates simple classifications from compound ones and provides
        individual classification options for better filtering.
        """
        simple_classes = set()
        compound_classes = set()
        all_individual_classes = set()
        
        for classification in hey_classifications:
            if not classification:
                continue
                
            # Check if it's a compound classification
            is_compound = any(delimiter in classification for delimiter in [',', ' or ', ' and ', ';', '/', '|'])
            
            if is_compound:
                compound_classes.add(classification)
                # Extract individual components
                for delimiter in [',', ' or ', ' and ', ';', '/', '|']:
                    if delimiter in classification:
                        parts = [part.strip() for part in classification.split(delimiter)]
                        for part in parts:
                            # Clean up the part (remove trailing descriptors)
                            clean_part = part.split(' with ')[0].split(' - ')[0].strip()
                            if clean_part and len(clean_part) > 2:  # Avoid very short fragments
                                all_individual_classes.add(clean_part)
                        break
            else:
                simple_classes.add(classification)
                all_individual_classes.add(classification)
        
        # Combine and sort options
        filter_options = sorted(all_individual_classes)
        
        # Add compound classifications at the end if they provide unique information
        for compound in sorted(compound_classes):
            if compound not in filter_options:
                filter_options.append(f"[Compound] {compound}")
        
        self.hey_filter.addItems(filter_options)
    
    def _populate_family_filter(self, chemical_families):
        """
        Populate chemical family filter with organized options.
        
        Similar to Hey classification but for chemical families.
        """
        simple_families = set()
        compound_families = set()
        all_individual_families = set()
        
        for family in chemical_families:
            if not family:
                continue
                
            # Check if it's a compound family
            is_compound = any(delimiter in family for delimiter in [',', ' or ', ' and ', ';', '/', '|', ' - '])
            
            if is_compound:
                compound_families.add(family)
                # Extract individual components
                for delimiter in [',', ' or ', ' and ', ';', '/', '|', ' - ']:
                    if delimiter in family:
                        parts = [part.strip() for part in family.split(delimiter)]
                        for part in parts:
                            # Clean up the part
                            clean_part = part.split(' with ')[0].split(' - ')[0].strip()
                            if clean_part and len(clean_part) > 2:
                                all_individual_families.add(clean_part)
                        break
            else:
                simple_families.add(family)
                all_individual_families.add(family)
        
        # Combine and sort options
        filter_options = sorted(all_individual_families)
        
        # Add compound families at the end
        for compound in sorted(compound_families):
            if compound not in filter_options:
                filter_options.append(f"[Compound] {compound}")
        
        self.family_filter.addItems(filter_options)
    
    def clear_search(self):
        """Clear all search filters and show all spectra."""
        self.search_entry.clear()
        self.hey_filter.setCurrentText("")
        self.family_filter.setCurrentText("")
    
    def select_all(self):
        """Select all filtered spectra (including those not displayed)."""
        # Check all visible rows in the table
        for row in range(self.spectra_table.rowCount()):
            if not self.spectra_table.isRowHidden(row):
                item = self.spectra_table.item(row, 0)
                if item:
                    item.setCheckState(Qt.Checked)
        
        # If there are more filtered results than displayed, inform user
        if len(self.filtered_spectra) > self.spectra_table.rowCount():
            extra = len(self.filtered_spectra) - self.spectra_table.rowCount()
            print(f"ℹ️  Selected all {self.spectra_table.rowCount()} displayed spectra + {extra} additional filtered spectra not shown")
    
    def select_none(self):
        """Deselect all spectra."""
        for row in range(self.spectra_table.rowCount()):
            item = self.spectra_table.item(row, 0)
            if item:
                item.setCheckState(Qt.Unchecked)
    
    def update_selection_count(self):
        """Update the selection count label."""
        count = 0
        for row in range(self.spectra_table.rowCount()):
            item = self.spectra_table.item(row, 0)
            if item and item.checkState() == Qt.Checked:
                count += 1
        
        self.selection_label.setText(f"{count} spectra selected")
    
    def accept_selection(self):
        """Accept the current selection and close the dialog."""
        self.selected_spectra = {}
        
        # Get selected from visible table rows
        for row in range(self.spectra_table.rowCount()):
            select_item = self.spectra_table.item(row, 0)
            if select_item and select_item.checkState() == Qt.Checked:
                name_item = self.spectra_table.item(row, 1)
                if name_item:
                    spectrum_name = name_item.text()
                    spectrum_data = name_item.data(Qt.UserRole)
                    if spectrum_data:
                        self.selected_spectra[spectrum_name] = spectrum_data
        
        # If "Select All" was used and there are more filtered results, include them all
        if (len(self.selected_spectra) == self.spectra_table.rowCount() and 
            len(self.filtered_spectra) > self.spectra_table.rowCount()):
            # User selected all visible rows - import ALL filtered spectra
            print(f"ℹ️  Importing all {len(self.filtered_spectra)} filtered spectra (not just the {self.spectra_table.rowCount()} displayed)")
            for spectrum_info in self.filtered_spectra:
                spectrum_name = spectrum_info['name']
                spectrum_data = spectrum_info['spectrum_data']
                if spectrum_data:
                    self.selected_spectra[spectrum_name] = spectrum_data
        
        if not self.selected_spectra:
            QMessageBox.warning(self, "No Selection", "Please select at least one spectrum to import.")
            return
        
        print(f"✅ Importing {len(self.selected_spectra)} spectra")
        self.accept()
    
    def get_selected_spectra(self):
        """Return the selected spectra data."""
        return self.selected_spectra
