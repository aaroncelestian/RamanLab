"""
Statistics Dialog for Microplastic Detection Results

Shows comprehensive statistics including:
- Total plastics by area with confidence intervals
- Breakdown by plastic type
- Example spectra with reference overlays
"""

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
                                QWidget, QLabel, QTextEdit, QTableWidget,
                                QTableWidgetItem, QHeaderView, QPushButton,
                                QGroupBox, QGridLayout)
from PySide6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from scipy import stats


class MicroplasticStatisticsDialog(QDialog):
    """Dialog showing comprehensive microplastic detection statistics."""
    
    def __init__(self, detection_results, map_data, detector, parent=None):
        super().__init__(parent)
        self.detection_results = detection_results
        self.map_data = map_data
        self.detector = detector
        self.parent_window = parent
        
        self.setWindowTitle("Microplastic Detection Statistics")
        self.setMinimumSize(1000, 700)
        
        self.setup_ui()
        self.calculate_statistics()
        
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Tab 1: Summary Statistics
        tabs.addTab(self.create_summary_tab(), "📊 Summary")
        
        # Tab 2: By Plastic Type
        tabs.addTab(self.create_by_type_tab(), "🔬 By Type")
        
        # Tab 3: Example Spectra
        tabs.addTab(self.create_examples_tab(), "📈 Example Spectra")
        
        layout.addWidget(tabs)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
    def create_summary_tab(self):
        """Create summary statistics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Summary text
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMinimumHeight(300)
        layout.addWidget(self.summary_text)
        
        # Statistics table
        table_group = QGroupBox("Detection Statistics")
        table_layout = QVBoxLayout(table_group)
        
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(5)
        self.summary_table.setHorizontalHeaderLabels([
            "Metric", "Value", "95% CI Lower", "95% CI Upper", "Notes"
        ])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_layout.addWidget(self.summary_table)
        
        layout.addWidget(table_group)
        
        return widget
    
    def create_by_type_tab(self):
        """Create by-type statistics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Table for each plastic type
        self.type_table = QTableWidget()
        self.type_table.setColumnCount(9)
        self.type_table.setHorizontalHeaderLabels([
            "Plastic Type", "Pixels", "Clusters", "Cluster Sizes", 
            "Mean Score", "Score Range", "Spatial Coherence", "Est. Particles", "Confidence"
        ])
        self.type_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.type_table)
        
        return widget
    
    def create_examples_tab(self):
        """Create example spectra tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create matplotlib figure for example spectra
        self.examples_figure = Figure(figsize=(10, 8), dpi=100)
        self.examples_canvas = FigureCanvas(self.examples_figure)
        layout.addWidget(self.examples_canvas)
        
        return widget
    
    def calculate_statistics(self):
        """Calculate all statistics and populate UI."""
        # Get threshold
        threshold = self.parent_window.threshold_slider.value() / 100.0 if hasattr(self.parent_window, 'threshold_slider') else 0.3
        
        # Filter out metadata
        score_maps = {k: v for k, v in self.detection_results.items() 
                     if not k.startswith('_')}
        
        # Calculate total detections
        all_detections = []
        for plastic_type, score_map in score_maps.items():
            detections = np.sum(score_map > threshold)
            all_detections.append(detections)
        
        total_detections = sum(all_detections)
        total_spectra = score_maps[list(score_maps.keys())[0]].size if score_maps else 0
        
        # Calculate 95% confidence intervals using bootstrap
        detection_rate = total_detections / total_spectra if total_spectra > 0 else 0
        
        # Bootstrap confidence interval for detection rate
        n_bootstrap = 1000
        bootstrap_rates = []
        for _ in range(n_bootstrap):
            # Resample with replacement
            sample_detections = np.random.choice(all_detections, size=len(all_detections), replace=True)
            bootstrap_rate = sum(sample_detections) / total_spectra if total_spectra > 0 else 0
            bootstrap_rates.append(bootstrap_rate)
        
        ci_lower = np.percentile(bootstrap_rates, 2.5)
        ci_upper = np.percentile(bootstrap_rates, 97.5)
        
        # Calculate cluster-based statistics for summary
        from scipy import ndimage
        total_clusters = 0
        high_conf_clusters = 0
        
        for plastic_type, score_map in score_maps.items():
            if score_map.ndim == 2:
                detection_mask = score_map > threshold
                _, num_clusters = ndimage.label(detection_mask)
                total_clusters += num_clusters
                
                # Count high-confidence clusters (mean score > 0.6)
                labeled, n = ndimage.label(detection_mask)
                for i in range(1, n + 1):
                    cluster_scores = score_map[labeled == i]
                    if np.mean(cluster_scores) > 0.6:
                        high_conf_clusters += 1
        
        # Populate summary text
        summary = f"""
<h2>Microplastic Detection Summary</h2>

<h3>Overall Statistics:</h3>
<ul>
<li><b>Total Spectra Analyzed:</b> {total_spectra:,}</li>
<li><b>Total Detection Pixels:</b> {total_detections:,}</li>
<li><b>Detection Rate:</b> {detection_rate*100:.2f}%</li>
<li><b>Detection Threshold:</b> {threshold:.2f}</li>
</ul>

<h3>Cluster Analysis:</h3>
<ul>
<li><b>Total Clusters Found:</b> {total_clusters}</li>
<li><b>High-Confidence Clusters:</b> {high_conf_clusters} (mean score > 0.6)</li>
<li><b>Estimated Particle Count:</b> {high_conf_clusters} (based on spatial coherence)</li>
</ul>

<h3>Quality Metrics Explained:</h3>
<table border="1" cellpadding="5" style="border-collapse: collapse;">
<tr><th>Metric</th><th>Description</th><th>Good Value</th></tr>
<tr><td><b>Clusters</b></td><td>Connected groups of detection pixels</td><td>Fewer, distinct clusters</td></tr>
<tr><td><b>Cluster Sizes</b></td><td>Range of pixels per cluster</td><td>Consistent with particle size</td></tr>
<tr><td><b>Mean Score</b></td><td>Average correlation with reference</td><td>> 0.55</td></tr>
<tr><td><b>Score Range</b></td><td>Min-Max scores in detections</td><td>Narrow range = consistent</td></tr>
<tr><td><b>Spatial Coherence</b></td><td>Cluster compactness (0-1)</td><td>> 0.5 = compact particle</td></tr>
<tr><td><b>Est. Particles</b></td><td>Clusters with high score + coherence</td><td>Best estimate of real count</td></tr>
</table>

<h3>Confidence Levels:</h3>
<ul>
<li>🟢 <b>HIGH:</b> Mean score > 0.65, coherence > 0.5, particles detected</li>
<li>🟡 <b>MEDIUM:</b> Mean score > 0.55, coherence > 0.3</li>
<li>🟠 <b>LOW:</b> Mean score > 0.45 or clusters present</li>
<li>🔴 <b>VERY LOW/NONE:</b> Below thresholds or no detections</li>
</ul>
"""
        self.summary_text.setHtml(summary)
        
        # Populate summary table
        self.summary_table.setRowCount(4)
        
        rows = [
            ("Total Spectra", f"{total_spectra:,}", "-", "-", "Number of spectra analyzed"),
            ("Total Detections", f"{total_detections:,}", "-", "-", "Spectra above threshold"),
            ("Detection Rate (%)", f"{detection_rate*100:.2f}", f"{ci_lower*100:.2f}", 
             f"{ci_upper*100:.2f}", "Percentage with microplastics"),
            ("Non-Plastic Rate (%)", f"{(1-detection_rate)*100:.2f}", f"{(1-ci_upper)*100:.2f}",
             f"{(1-ci_lower)*100:.2f}", "Percentage without microplastics")
        ]
        
        for row_idx, (metric, value, ci_low, ci_high, notes) in enumerate(rows):
            self.summary_table.setItem(row_idx, 0, QTableWidgetItem(metric))
            self.summary_table.setItem(row_idx, 1, QTableWidgetItem(value))
            self.summary_table.setItem(row_idx, 2, QTableWidgetItem(ci_low))
            self.summary_table.setItem(row_idx, 3, QTableWidgetItem(ci_high))
            self.summary_table.setItem(row_idx, 4, QTableWidgetItem(notes))
        
        # Populate by-type table
        self.populate_by_type_table(score_maps, threshold, total_spectra)
        
        # Generate example spectra
        self.generate_example_spectra(score_maps, threshold)
    
    def populate_by_type_table(self, score_maps, threshold, total_spectra):
        """Populate the by-type statistics table with cluster-based metrics."""
        from scipy import ndimage
        
        self.type_table.setRowCount(len(score_maps))
        
        # Map full names to codes for lookup
        name_to_code = {
            'Polyethylene': 'PE',
            'Polypropylene': 'PP',
            'Polystyrene': 'PS',
            'Polyethylene Terephthalate': 'PET',
            'Polyamide': 'PA',
            'Polycarbonate': 'PC',
            'Polyurethane': 'PU',
            'Acrylic': 'PMMA',
            'PVC': 'PVC'
        }
        
        for row_idx, (plastic_type, score_map) in enumerate(score_maps.items()):
            # Get plastic info - try both the key directly and mapped code
            plastic_info = self.detector.PLASTIC_SIGNATURES.get(plastic_type, {})
            if not plastic_info:
                code = name_to_code.get(plastic_type, plastic_type)
                plastic_info = self.detector.PLASTIC_SIGNATURES.get(code, {})
            plastic_name = plastic_info.get('name', plastic_type)
            
            # Ensure 2D for cluster analysis
            if score_map.ndim == 1:
                side = int(np.sqrt(len(score_map)))
                if side * side == len(score_map):
                    score_map_2d = score_map.reshape(side, side)
                else:
                    # Can't reshape, use basic stats
                    score_map_2d = None
            else:
                score_map_2d = score_map
            
            # Basic pixel count
            detection_mask = score_map > threshold
            pixel_count = np.sum(detection_mask)
            
            # Get scores above threshold
            scores_above = score_map[detection_mask]
            mean_score = np.mean(scores_above) if len(scores_above) > 0 else 0
            min_score = np.min(scores_above) if len(scores_above) > 0 else 0
            max_score = np.max(scores_above) if len(scores_above) > 0 else 0
            
            # Cluster analysis (if 2D available)
            if score_map_2d is not None and pixel_count > 0:
                detection_mask_2d = score_map_2d > threshold
                labeled_array, num_clusters = ndimage.label(detection_mask_2d)
                
                # Get cluster sizes
                cluster_sizes = []
                cluster_mean_scores = []
                for label_id in range(1, num_clusters + 1):
                    cluster_mask = labeled_array == label_id
                    cluster_size = np.sum(cluster_mask)
                    cluster_scores = score_map_2d[cluster_mask]
                    cluster_sizes.append(cluster_size)
                    cluster_mean_scores.append(np.mean(cluster_scores))
                
                if cluster_sizes:
                    min_cluster = min(cluster_sizes)
                    max_cluster = max(cluster_sizes)
                    median_cluster = int(np.median(cluster_sizes))
                    cluster_size_str = f"{min_cluster}-{max_cluster} (med: {median_cluster})"
                    
                    # Spatial coherence: ratio of cluster pixels to bounding box
                    # Higher = more compact/real, Lower = scattered/noise
                    coherence_scores = []
                    for label_id in range(1, num_clusters + 1):
                        cluster_mask = labeled_array == label_id
                        rows = np.any(cluster_mask, axis=1)
                        cols = np.any(cluster_mask, axis=0)
                        if np.any(rows) and np.any(cols):
                            rmin, rmax = np.where(rows)[0][[0, -1]]
                            cmin, cmax = np.where(cols)[0][[0, -1]]
                            bbox_area = (rmax - rmin + 1) * (cmax - cmin + 1)
                            cluster_area = np.sum(cluster_mask)
                            coherence_scores.append(cluster_area / bbox_area)
                    
                    spatial_coherence = np.mean(coherence_scores) if coherence_scores else 0
                    
                    # Estimate real particles: clusters with high coherence AND high mean score
                    est_particles = sum(1 for i, cs in enumerate(cluster_mean_scores) 
                                       if cs > 0.55 and (len(coherence_scores) <= i or coherence_scores[i] > 0.3))
                else:
                    cluster_size_str = "N/A"
                    spatial_coherence = 0
                    est_particles = 0
            else:
                num_clusters = 0
                cluster_size_str = "N/A"
                spatial_coherence = 0
                est_particles = 0
            
            # Score range string
            score_range_str = f"{min_score:.2f} - {max_score:.2f}" if pixel_count > 0 else "N/A"
            
            # Confidence based on multiple factors
            if pixel_count == 0:
                confidence = "🔴 NONE"
            elif mean_score > 0.65 and spatial_coherence > 0.5 and est_particles > 0:
                confidence = "🟢 HIGH"
            elif mean_score > 0.55 and spatial_coherence > 0.3:
                confidence = "🟡 MEDIUM"
            elif mean_score > 0.45 or num_clusters > 0:
                confidence = "🟠 LOW"
            else:
                confidence = "🔴 VERY LOW"
            
            # Populate row
            self.type_table.setItem(row_idx, 0, QTableWidgetItem(plastic_name))
            self.type_table.setItem(row_idx, 1, QTableWidgetItem(f"{pixel_count:,}"))
            self.type_table.setItem(row_idx, 2, QTableWidgetItem(f"{num_clusters}"))
            self.type_table.setItem(row_idx, 3, QTableWidgetItem(cluster_size_str))
            self.type_table.setItem(row_idx, 4, QTableWidgetItem(f"{mean_score:.3f}"))
            self.type_table.setItem(row_idx, 5, QTableWidgetItem(score_range_str))
            self.type_table.setItem(row_idx, 6, QTableWidgetItem(f"{spatial_coherence:.2f}"))
            self.type_table.setItem(row_idx, 7, QTableWidgetItem(f"{est_particles}"))
            self.type_table.setItem(row_idx, 8, QTableWidgetItem(confidence))
    
    def generate_example_spectra(self, score_maps, threshold):
        """Generate example spectra plots with reference overlays."""
        self.examples_figure.clear()
        
        # Find top 3 plastic types by detection count
        plastic_counts = [(ptype, np.sum(smap > threshold)) 
                         for ptype, smap in score_maps.items()]
        plastic_counts.sort(key=lambda x: x[1], reverse=True)
        top_plastics = plastic_counts[:3]
        
        if not top_plastics or not self.map_data:
            ax = self.examples_figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No example spectra available',
                   ha='center', va='center', fontsize=12, color='gray')
            self.examples_canvas.draw()
            return
        
        # Create subplots for top 3 plastics
        n_plots = len(top_plastics)
        for idx, (plastic_type, count) in enumerate(top_plastics):
            if count == 0:
                continue
                
            ax = self.examples_figure.add_subplot(n_plots, 1, idx + 1)
            
            # Find highest scoring detection for this plastic
            score_map = score_maps[plastic_type]
            max_idx = np.argmax(score_map)
            
            # Get spectrum at this position
            if hasattr(self.map_data, 'get_processed_data_matrix'):
                # Single file map
                intensity_matrix = self.map_data.get_processed_data_matrix()
                wavenumbers = self.map_data.target_wavenumbers
                spectrum = intensity_matrix[max_idx]
            else:
                # Multi-file map - get first spectrum as example
                first_spec = list(self.map_data.spectra.values())[0]
                wavenumbers = first_spec.wavenumbers
                spectrum = first_spec.intensities
            
            # Apply baseline correction
            from map_analysis_2d.analysis.microplastic_detector import MicroplasticDetector
            spectrum_corrected = MicroplasticDetector.baseline_als(spectrum)
            
            # Normalize
            spectrum_norm = spectrum_corrected - np.min(spectrum_corrected)
            if np.max(spectrum_norm) > 0:
                spectrum_norm = spectrum_norm / np.max(spectrum_norm)
            
            # Plot spectrum
            plastic_info = self.detector.PLASTIC_SIGNATURES.get(plastic_type, {})
            plastic_name = plastic_info.get('name', plastic_type)
            plastic_color = plastic_info.get('color', '#FF0000')
            
            ax.plot(wavenumbers, spectrum_norm, 'b-', linewidth=2, 
                   label='Detected Spectrum', alpha=0.8)
            
            # Overlay reference if available
            if plastic_type in self.detector.reference_spectra:
                ref_wn, ref_int = self.detector.reference_spectra[plastic_type]
                
                # Normalize reference
                ref_norm = ref_int - np.min(ref_int)
                if np.max(ref_norm) > 0:
                    ref_norm = ref_norm / np.max(ref_norm)
                
                # Interpolate to match wavenumber range
                ref_interp = np.interp(wavenumbers, ref_wn, ref_norm)
                
                ax.plot(wavenumbers, ref_interp * 0.9, color=plastic_color,
                       linewidth=2, linestyle='--', alpha=0.7,
                       label=f'{plastic_name} Reference')
            
            max_score = np.max(score_map)
            ax.set_title(f'{plastic_name} - Best Match (Score: {max_score:.3f})',
                        fontweight='bold', fontsize=10)
            ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=9)
            ax.set_ylabel('Normalized Intensity', fontsize=9)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
        
        self.examples_figure.tight_layout()
        self.examples_canvas.draw()
