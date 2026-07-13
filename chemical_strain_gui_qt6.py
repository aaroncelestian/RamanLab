#!/usr/bin/env python3
"""Qt6 GUI for general chemical strain analysis."""

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, QTextEdit, QMessageBox, QFormLayout,
    QTabWidget, QFileDialog,
)
from PySide6.QtCore import Qt

from core.matplotlib_config import apply_theme, CompactNavigationToolbar as NavigationToolbar
from chemical_strain_enhancement import ChemicalStrainAnalyzer, ChemicalRamanMode
from deconvolution_pkl_loader import (
    load_deconvolution_pkl,
    get_available_peaks,
    peak_number_mean_wavenumber,
    extract_peak_evolution,
    extract_standard_peak_evolution,
    apply_internal_standard_correction,
    compute_strain_evolution,
)


def _build_default_analyzer(crystal_system: str, composition_model: str) -> ChemicalStrainAnalyzer:
    """Create an analyzer with a generic two-mode example setup."""
    analyzer = ChemicalStrainAnalyzer(crystal_system, composition_model)
    analyzer.modes["A1g"] = ChemicalRamanMode(
        name="A1g",
        omega0_pure=595.0,
        gamma_components_pure=np.array([1.2, 1.2, 0.8, 0.0, 0.0, 0.0]),
        composition_sensitivity=np.array([0.3, 0.3, -0.2, 0.0, 0.0, 0.0]),
        jahn_teller_coupling=0.5,
        intensity_pure=100.0,
    )
    analyzer.modes["Eg"] = ChemicalRamanMode(
        name="Eg",
        omega0_pure=485.0,
        gamma_components_pure=np.array([0.8, 0.8, 1.5, 0.0, 0.0, 0.0]),
        composition_sensitivity=np.array([0.1, 0.1, 0.4, 0.0, 0.0, 0.0]),
        jahn_teller_coupling=0.2,
        intensity_pure=60.0,
    )
    return analyzer


class GeneralChemicalStrainWindow(QMainWindow):
    """General chemical strain analysis window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        apply_theme("compact")
        self.setWindowTitle("RamanLab - General Chemical Strain Analysis")
        self.setMinimumSize(960, 700)
        self.resize(1100, 780)

        self.wavenumbers = None
        self.intensities = None
        self.spectrum_name = "No spectrum loaded"
        self.analyzer = _build_default_analyzer("hexagonal", "vegard")
        self.deconv_data = None
        self.evolution = None
        self.strain_results = None
        self.standard_evolution = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        info = QLabel(
            "Analyze single-spectrum chemical strain, or import fitted line-scan / time-series "
            "data from Spectral Deconvolution batch PKL exports to track stress/strain evolution."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.tabs = QTabWidget()
        self._configure_tab_widget(self.tabs)
        self.tabs.addTab(self._build_single_spectrum_tab(), "Single Spectrum")
        self.tabs.addTab(self._build_evolution_tab(), "Evolving System")
        layout.addWidget(self.tabs, 1)

    @staticmethod
    def _configure_tab_widget(tab_widget):
        tab_widget.setDocumentMode(False)
        tab_widget.setUsesScrollButtons(True)
        tab_widget.setElideMode(Qt.ElideRight)
        tab_widget.setStyleSheet("")

    def _build_single_spectrum_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        controls = QGroupBox("Analysis Settings")
        form = QFormLayout(controls)

        self.crystal_combo = QComboBox()
        self.crystal_combo.addItems([
            "hexagonal", "trigonal", "cubic", "tetragonal",
            "orthorhombic", "monoclinic", "triclinic",
        ])
        form.addRow("Crystal system:", self.crystal_combo)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["vegard", "linear", "non_linear"])
        form.addRow("Composition model:", self.model_combo)

        self.composition_spin = QDoubleSpinBox()
        self.composition_spin.setRange(0.0, 1.0)
        self.composition_spin.setSingleStep(0.05)
        self.composition_spin.setValue(0.7)
        form.addRow("Composition:", self.composition_spin)

        self.disorder_spin = QDoubleSpinBox()
        self.disorder_spin.setRange(0.0, 1.0)
        self.disorder_spin.setSingleStep(0.05)
        self.disorder_spin.setValue(0.1)
        form.addRow("Chemical disorder:", self.disorder_spin)

        layout.addWidget(controls)

        self.spectrum_label = QLabel(f"Spectrum: {self.spectrum_name}")
        layout.addWidget(self.spectrum_label)

        plot_group = QGroupBox("Spectrum")
        plot_layout = QVBoxLayout(plot_group)
        self.single_figure = Figure(figsize=(8, 3.5))
        self.single_canvas = FigureCanvas(self.single_figure)
        self.single_toolbar = NavigationToolbar(self.single_canvas, plot_group)
        self.single_ax = self.single_figure.add_subplot(111)
        self.single_ax.set_xlabel("Wavenumber (cm⁻¹)")
        self.single_ax.set_ylabel("Intensity (a.u.)")
        self.single_ax.grid(True, alpha=0.3)
        plot_layout.addWidget(self.single_toolbar)
        plot_layout.addWidget(self.single_canvas)
        layout.addWidget(plot_group, 1)

        self.single_results_text = QTextEdit()
        self.single_results_text.setReadOnly(True)
        self.single_results_text.setMaximumHeight(140)
        self.single_results_text.setPlainText("Configure settings and run an analysis.")
        layout.addWidget(self.single_results_text)

        btn_row = QHBoxLayout()
        example_btn = QPushButton("Run Example Analysis")
        example_btn.clicked.connect(self.run_example_analysis)
        btn_row.addWidget(example_btn)

        fit_btn = QPushButton("Fit Loaded Spectrum")
        fit_btn.clicked.connect(self.fit_loaded_spectrum)
        btn_row.addWidget(fit_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._plot_single_empty()
        return tab

    def _build_evolution_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        import_group = QGroupBox("Import Deconvolution Data")
        import_layout = QVBoxLayout(import_group)

        import_row = QHBoxLayout()
        self.import_pkl_btn = QPushButton("Import PKL…")
        self.import_pkl_btn.clicked.connect(self.import_deconvolution_pkl)
        import_row.addWidget(self.import_pkl_btn)
        import_row.addStretch()
        import_layout.addLayout(import_row)

        self.pkl_info_label = QLabel("No PKL loaded. Export batch/line-scan results from Spectral Deconvolution.")
        self.pkl_info_label.setWordWrap(True)
        import_layout.addWidget(self.pkl_info_label)
        layout.addWidget(import_group)

        settings = QGroupBox("Evolution Analysis")
        settings_form = QFormLayout(settings)

        self.peak_combo = QComboBox()
        self.peak_combo.setEnabled(False)
        settings_form.addRow("Tracked peak:", self.peak_combo)

        self.axis_combo = QComboBox()
        self.axis_combo.addItems([
            "Scan index (file order)",
            "Parsed position from filename",
        ])
        settings_form.addRow("X axis:", self.axis_combo)

        self.internal_standard_combo = QComboBox()
        self.internal_standard_combo.addItem("None", "")
        self.internal_standard_combo.addItem("Silicon (~520 cm⁻¹)", "silicon")
        self.internal_standard_combo.addItem("Corundum (~417 cm⁻¹)", "corundum")
        self.internal_standard_combo.setCurrentIndex(1)
        self.internal_standard_combo.setToolTip(
            "Correct tracked peak positions for drift using ν_corr = ν + (ν_std,ref − ν_std(t)); "
            "ν_std,ref is the mean of the last N standard peaks."
        )
        settings_form.addRow("Internal standard:", self.internal_standard_combo)

        self.ref_last_n_spin = QSpinBox()
        self.ref_last_n_spin.setRange(1, 500)
        self.ref_last_n_spin.setValue(5)
        self.ref_last_n_spin.setToolTip(
            "Number of final scans averaged as the crystallized reference (oxalate strain) "
            "and as the standard peak reference for drift correction."
        )
        settings_form.addRow("Reference last N:", self.ref_last_n_spin)

        self.computed_ref_label = QLabel("Reference: —")
        self.computed_ref_label.setWordWrap(True)
        settings_form.addRow("", self.computed_ref_label)

        self.gruneisen_spin = QDoubleSpinBox()
        self.gruneisen_spin.setRange(0.01, 20.0)
        self.gruneisen_spin.setDecimals(2)
        self.gruneisen_spin.setValue(1.0)
        self.gruneisen_spin.setToolTip("Grüneisen parameter for ε ≈ Δν / (ν_ref · γ)")
        settings_form.addRow("Grüneisen γ:", self.gruneisen_spin)

        layout.addWidget(settings)

        plot_group = QGroupBox("Stress / Strain Evolution")
        plot_layout = QVBoxLayout(plot_group)
        self.evolution_figure = Figure(figsize=(8, 5))
        self.evolution_canvas = FigureCanvas(self.evolution_figure)
        self.evolution_toolbar = NavigationToolbar(self.evolution_canvas, plot_group)
        plot_layout.addWidget(self.evolution_toolbar)
        plot_layout.addWidget(self.evolution_canvas)
        layout.addWidget(plot_group, 1)

        self.evolution_results_text = QTextEdit()
        self.evolution_results_text.setReadOnly(True)
        self.evolution_results_text.setMaximumHeight(160)
        self.evolution_results_text.setPlainText(
            "Import a PKL file from Spectral Deconvolution batch export, choose a peak, "
            "then analyze evolution along the line scan or time series."
        )
        layout.addWidget(self.evolution_results_text)

        evo_btn_row = QHBoxLayout()
        analyze_btn = QPushButton("Analyze Evolution")
        analyze_btn.clicked.connect(self.analyze_evolution)
        evo_btn_row.addWidget(analyze_btn)
        evo_btn_row.addStretch()
        layout.addLayout(evo_btn_row)

        self._plot_evolution_empty()
        return tab

    def load_spectrum(self, wavenumbers, intensities, name="Current Spectrum"):
        """Load spectrum data from the main RamanLab app."""
        self.wavenumbers = np.asarray(wavenumbers, dtype=float)
        self.intensities = np.asarray(intensities, dtype=float)
        self.spectrum_name = name
        self.spectrum_label.setText(f"Spectrum: {self.spectrum_name} ({len(self.wavenumbers)} points)")
        self.single_ax.clear()
        self.single_ax.plot(self.wavenumbers, self.intensities, color="#2563eb", linewidth=1.2, label="Loaded")
        self.single_ax.set_xlabel("Wavenumber (cm⁻¹)")
        self.single_ax.set_ylabel("Intensity (a.u.)")
        self.single_ax.set_title("Loaded spectrum")
        self.single_ax.grid(True, alpha=0.3)
        self.single_ax.legend()
        self.single_canvas.draw()

    def import_deconvolution_pkl(self):
        """Load fitted peak data exported from Spectral Deconvolution."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Deconvolution PKL",
            "",
            "Pickle Files (*.pkl);;All Files (*)",
        )
        if not file_path:
            return

        try:
            self.deconv_data = load_deconvolution_pkl(file_path)
            peaks = get_available_peaks(self.deconv_data["peaks_df"])
            if not peaks:
                raise ValueError("The PKL file contains no fitted peaks.")

            meta = self.deconv_data.get("metadata", {})
            n_regions = meta.get("total_regions", len(self.deconv_data.get("summary_df", [])))
            n_peaks = meta.get("total_peaks", len(self.deconv_data["peaks_df"]))

            self.pkl_info_label.setText(
                f"Loaded: {Path(file_path).name}  ·  "
                f"{n_regions} spectra/regions  ·  {n_peaks} peak records  ·  "
                f"format: {self.deconv_data.get('format', 'unknown')}"
            )

            self.peak_combo.blockSignals(True)
            self.peak_combo.clear()
            peaks_df = self.deconv_data["peaks_df"]
            for peak_num in peaks:
                mean_wn = peak_number_mean_wavenumber(peaks_df, peak_num)
                label = f"Peak {peak_num}"
                if mean_wn is not None:
                    label += f" (~{mean_wn:.0f} cm⁻¹)"
                self.peak_combo.addItem(label, peak_num)
            self.peak_combo.setEnabled(True)
            self.peak_combo.blockSignals(False)

            self.tabs.setCurrentIndex(1)
            self.evolution_results_text.setPlainText(
                f"Imported {n_regions} spectra with fitted peaks.\n"
                "Select an oxalate peak (~1400–1600 cm⁻¹), keep Silicon as internal standard, "
                "set reference last N to crystallized points, then Analyze Evolution."
            )

        except Exception as exc:
            QMessageBox.critical(self, "Import Error", f"Failed to load PKL file:\n{exc}")

    def analyze_evolution(self):
        """Compute and plot peak shift / strain evolution from imported PKL data."""
        if self.deconv_data is None:
            QMessageBox.warning(self, "No Data", "Import a deconvolution PKL file first.")
            return

        peak_number = self.peak_combo.currentData()
        if peak_number is None:
            QMessageBox.warning(self, "No Peak", "Select a peak to track.")
            return

        try:
            axis_mode = "parsed_filename" if self.axis_combo.currentIndex() == 1 else "scan_index"
            last_n = self.ref_last_n_spin.value()
            standard_key = self.internal_standard_combo.currentData()

            evolution = extract_peak_evolution(
                self.deconv_data,
                peak_number=int(peak_number),
                position_axis=axis_mode,
            )
            self.standard_evolution = None

            if standard_key:
                self.standard_evolution = extract_standard_peak_evolution(
                    self.deconv_data,
                    standard=standard_key,
                    position_axis=axis_mode,
                )
                evolution = apply_internal_standard_correction(
                    evolution,
                    self.standard_evolution,
                    reference_last_n=last_n,
                )

            self.evolution = evolution
            self.strain_results = compute_strain_evolution(
                evolution,
                gruneisen=self.gruneisen_spin.value(),
                last_n_reference=last_n,
            )

            ref_text = (
                f"Oxalate reference (last {last_n}): "
                f"{self.strain_results['reference_wavenumber']:.2f} cm⁻¹"
            )
            if standard_key and "standard_reference" in evolution:
                ref_text += (
                    f"  ·  {evolution.get('standard_label', 'Standard')} ref (last {last_n}): "
                    f"{evolution['standard_reference']:.2f} cm⁻¹"
                )
            self.computed_ref_label.setText(ref_text)

            self._plot_evolution()
            self._show_evolution_summary()
        except Exception as exc:
            QMessageBox.critical(self, "Analysis Error", f"Evolution analysis failed:\n{exc}")

    def _show_evolution_summary(self):
        evo = self.evolution
        strain = self.strain_results
        lines = [
            "Evolution analysis summary",
            "=" * 40,
            f"Peak: {evo['peak_number']}",
            f"Points: {evo['n_points']}",
            f"Reference: last {strain['reference_last_n']} points (crystallized end state)",
            f"Reference wavenumber: {strain['reference_wavenumber']:.2f} cm⁻¹",
        ]
        if self.standard_evolution:
            lines.append(
                f"Internal standard: {evo.get('standard_label', 'yes')} "
                f"(ref {evo.get('standard_reference', float('nan')):.2f} cm⁻¹)"
            )
        lines.extend([
            f"Grüneisen γ: {strain['gruneisen']:.2f}",
            "",
            f"Δν range: {np.nanmin(strain['delta_nu']):+.2f} to {np.nanmax(strain['delta_nu']):+.2f} cm⁻¹",
            f"Max |strain| proxy: {strain['max_abs_strain']:.5f}",
            "",
            "Point   position   Δν      ε       amplitude   FWHM",
        ])
        raw = evo.get("positions_raw", evo["positions"])
        for i in range(evo["n_points"]):
            fname = Path(evo["filenames"][i]).name
            lines.append(
                f"  {i+1:3d}  {evo['positions'][i]:7.2f}  {strain['delta_nu'][i]:+7.2f}  "
                f"{strain['strain'][i]:+.5f}  {evo['amplitudes'][i]:9.1f}  {evo['fwhms'][i]:6.1f}  {fname}"
            )
        if "positions_raw" in evo:
            lines.append("")
            lines.append(f"Raw mean position (uncorrected): {np.nanmean(raw):.2f} cm⁻¹")
            lines.append(f"Corrected mean position: {np.nanmean(evo['positions']):.2f} cm⁻¹")

        self.evolution_results_text.setPlainText("\n".join(lines))

    def _plot_evolution(self):
        evo = self.evolution
        strain = self.strain_results
        self.evolution_figure.clear()

        axes = self.evolution_figure.subplots(2, 2, sharex=True)
        ax_pos, ax_strain, ax_amp, ax_fwhm = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
        x = evo["x_values"]

        pos_label = "Corrected position" if "positions_raw" in evo else "Peak position"
        ax_pos.plot(x, evo["positions"], "o-", color="#2563eb", linewidth=1.2, markersize=4, label=pos_label)
        if "positions_raw" in evo:
            ax_pos.plot(x, evo["positions_raw"], ".--", color="#93c5fd", linewidth=1, markersize=4, label="Raw")
        ax_pos.axhline(strain["reference_wavenumber"], color="#6b7280", linestyle="--", linewidth=1, label="Ref (last N)")
        ax_pos.set_ylabel("cm⁻¹")
        ax_pos.set_title(f"Peak {evo['peak_number']} position")
        ax_pos.grid(True, alpha=0.3)
        ax_pos.legend(loc="best", fontsize=7)

        ax_strain.plot(x, strain["strain"], "s-", color="#0D9488", linewidth=1.2, markersize=4)
        ax_strain.axhline(0.0, color="#6b7280", linestyle="--", linewidth=1)
        ax_strain.set_ylabel("Strain ε")
        ax_strain.set_title("Strain vs crystal reference")
        ax_strain.grid(True, alpha=0.3)

        ax_amp.plot(x, evo["amplitudes"], "^-", color="#7c3aed", linewidth=1.2, markersize=4)
        ax_amp.set_ylabel("Amplitude")
        ax_amp.set_xlabel(evo["x_label"])
        ax_amp.set_title("Peak amplitude")
        ax_amp.grid(True, alpha=0.3)

        ax_fwhm.plot(x, evo["fwhms"], "d-", color="#ea580c", linewidth=1.2, markersize=4)
        ax_fwhm.set_ylabel("FWHM (cm⁻¹)")
        ax_fwhm.set_xlabel(evo["x_label"])
        ax_fwhm.set_title("Peak width")
        ax_fwhm.grid(True, alpha=0.3)

        self.evolution_figure.tight_layout()
        self.evolution_canvas.draw()

    def _plot_evolution_empty(self):
        self.evolution_figure.clear()
        ax = self.evolution_figure.add_subplot(111)
        ax.text(
            0.5, 0.5, "Import a Spectral Deconvolution PKL to view evolution",
            transform=ax.transAxes, ha="center", va="center", alpha=0.6,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        self.evolution_canvas.draw()

    def _refresh_analyzer(self):
        self.analyzer = _build_default_analyzer(
            self.crystal_combo.currentText(),
            self.model_combo.currentText(),
        )
        self.analyzer.set_composition(self.composition_spin.value())
        self.analyzer.chemical_disorder = self.disorder_spin.value()

    def _format_results(self, result: dict) -> str:
        strain = result["strain_tensor"]
        lines = [
            "Chemical strain fit results",
            "=" * 40,
            f"Success: {result['success']}",
            f"R²: {result['r_squared']:.4f}",
            f"Composition: {result['composition']:.3f}",
            f"Jahn-Teller parameter: {result['jahn_teller_parameter']:.4f}",
            "",
            "Strain tensor (Voigt ε1–ε6):",
            f"  ε1={strain[0]:+.5f}  ε2={strain[1]:+.5f}  ε3={strain[2]:+.5f}",
            f"  ε4={strain[3]:+.5f}  ε5={strain[4]:+.5f}  ε6={strain[5]:+.5f}",
        ]
        return "\n".join(lines)

    def _plot_single_fit(self, frequencies, observed, model):
        self.single_ax.clear()
        self.single_ax.plot(frequencies, observed, color="#2563eb", linewidth=1.2, label="Observed")
        self.single_ax.plot(frequencies, model, color="#dc2626", linewidth=1.2, linestyle="--", label="Model")
        self.single_ax.set_xlabel("Wavenumber (cm⁻¹)")
        self.single_ax.set_ylabel("Intensity (a.u.)")
        self.single_ax.set_title("Chemical strain fit")
        self.single_ax.grid(True, alpha=0.3)
        self.single_ax.legend()
        self.single_canvas.draw()

    def _plot_single_empty(self):
        self.single_ax.clear()
        self.single_ax.text(
            0.5, 0.5, "Load a spectrum in RamanLab or run the example analysis",
            transform=self.single_ax.transAxes, ha="center", va="center", alpha=0.6,
        )
        self.single_ax.set_xticks([])
        self.single_ax.set_yticks([])
        self.single_canvas.draw()

    def run_example_analysis(self):
        """Run a synthetic demonstration fit."""
        try:
            self._refresh_analyzer()
            frequencies = np.linspace(420, 680, 600)
            true_strain = np.array([0.012, 0.012, -0.006, 0.0, 0.0, 0.0])
            observed = self.analyzer.multi_phase_spectrum(frequencies, true_strain)
            observed = self.analyzer._apply_disorder_broadening(observed, frequencies)
            observed += np.random.normal(0, 0.015, observed.shape)

            result = self.analyzer.fit_chemical_strain_tensor(
                frequencies,
                observed,
                composition=self.composition_spin.value(),
                fit_composition=True,
                fit_jahn_teller=True,
            )
            self.single_results_text.setPlainText(self._format_results(result))
            self._plot_single_fit(frequencies, observed, result["model_spectrum"])
        except Exception as exc:
            QMessageBox.critical(self, "Analysis Error", f"Example analysis failed:\n{exc}")

    def fit_loaded_spectrum(self):
        """Fit chemical strain model to the spectrum passed from the main app."""
        if self.wavenumbers is None or self.intensities is None:
            QMessageBox.warning(
                self,
                "No Spectrum",
                "Load a spectrum in the main RamanLab window first, then launch this tool again.",
            )
            return

        try:
            self._refresh_analyzer()
            y = self.intensities.astype(float)
            if y.max() > 0:
                y = y / y.max()

            result = self.analyzer.fit_chemical_strain_tensor(
                self.wavenumbers,
                y,
                composition=self.composition_spin.value(),
                fit_composition=True,
                fit_jahn_teller=True,
            )
            self.single_results_text.setPlainText(self._format_results(result))
            self._plot_single_fit(self.wavenumbers, y, result["model_spectrum"])
        except Exception as exc:
            QMessageBox.critical(self, "Analysis Error", f"Fit failed:\n{exc}")
