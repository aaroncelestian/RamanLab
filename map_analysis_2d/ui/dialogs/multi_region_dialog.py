from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                                QLineEdit, QPushButton, QTableWidget, 
                                QTableWidgetItem, QHeaderView, QMessageBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class MultiRegionDialog(QDialog):
    def __init__(self, region_info, default_base_name, parent=None):
        super().__init__(parent)
        self.region_info = region_info
        self.default_base_name = default_base_name
        self.region_names = {}
        
        self.setWindowTitle("Multiple Regions Detected")
        self.setMinimumWidth(700)
        self.setMinimumHeight(400)
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        title_label = QLabel("Multiple Raman Map Regions Detected")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        info_label = QLabel(
            f"This H5 file contains {len(self.region_info)} separate map regions.\n"
            "Each region will be saved as a separate PKL file.\n"
            "Please provide a filename for each region below:"
        )
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)
        
        layout.addSpacing(10)
        
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Region", "Dimensions", "Spectra", "Output Filename"])
        self.table.setRowCount(len(self.region_info))
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        
        for row, (region_key, info) in enumerate(sorted(self.region_info.items())):
            region_item = QTableWidgetItem(region_key)
            region_item.setFlags(region_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, region_item)
            
            dims_text = f"{info['n_x']} × {info['n_y']}"
            dims_item = QTableWidgetItem(dims_text)
            dims_item.setFlags(dims_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 1, dims_item)
            
            spectra_text = f"{info['n_spectra']:,}"
            spectra_item = QTableWidgetItem(spectra_text)
            spectra_item.setFlags(spectra_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 2, spectra_item)
            
            default_name = f"{self.default_base_name}_{region_key}.pkl"
            filename_item = QTableWidgetItem(default_name)
            self.table.setItem(row, 3, filename_item)
        
        layout.addWidget(self.table)
        
        layout.addSpacing(10)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        auto_name_btn = QPushButton("Auto-Name All")
        auto_name_btn.clicked.connect(self.auto_name_all)
        button_layout.addWidget(auto_name_btn)
        
        button_layout.addSpacing(20)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept_names)
        button_layout.addWidget(ok_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def auto_name_all(self):
        for row in range(self.table.rowCount()):
            region_key = self.table.item(row, 0).text()
            default_name = f"{self.default_base_name}_{region_key}.pkl"
            self.table.item(row, 3).setText(default_name)
    
    def accept_names(self):
        self.region_names = {}
        
        for row in range(self.table.rowCount()):
            region_key = self.table.item(row, 0).text()
            filename = self.table.item(row, 3).text().strip()
            
            if not filename:
                QMessageBox.warning(
                    self, "Invalid Filename",
                    f"Please provide a filename for {region_key}"
                )
                return
            
            if not filename.endswith('.pkl'):
                filename += '.pkl'
            
            if filename in self.region_names.values():
                QMessageBox.warning(
                    self, "Duplicate Filename",
                    f"Filename '{filename}' is used for multiple regions.\n"
                    "Please use unique filenames."
                )
                return
            
            self.region_names[region_key] = filename
        
        self.accept()
    
    def get_region_names(self):
        return self.region_names
