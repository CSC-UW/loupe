"""Subplot-controls modal/non-modal dialogs used by :class:`loupe.app.LoupeApp`.

Each dialog takes ``parent`` (the ``LoupeApp`` window) and reads/writes its
state through that reference — moving them out doesn't change the call
pattern, only the file they live in.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6 import QtCore, QtWidgets

from loupe._heatmap_utils import (
    ARRAY_COLORMAP_PRESETS,
    ARRAY_MIPMAP_TARGET_MIN_COLS,
    _colormap_cache_token,
    _colormap_display_name,
)


class YAxisControlsDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Y-Axis Controls")
        self.setModal(False)

        self.main_window = parent

        main_layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        main_layout.addLayout(form_layout)

        self.controls = []
        for idx, s in enumerate(self.main_window.series):
            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            auto_check = QtWidgets.QCheckBox("Auto")
            auto_check.setChecked(
                self.main_window.plots[idx].getViewBox().autoRangeEnabled()[1]
            )

            min_spin = QtWidgets.QDoubleSpinBox()
            max_spin = QtWidgets.QDoubleSpinBox()

            for spin in (min_spin, max_spin):
                spin.setDecimals(3)
                spin.setRange(-1e12, 1e12)
                spin.setEnabled(not auto_check.isChecked())

            current_y_range = self.main_window.plots[idx].getViewBox().viewRange()[1]
            min_spin.setValue(current_y_range[0])
            max_spin.setValue(current_y_range[1])

            auto_check.stateChanged.connect(
                lambda state, i=idx, ac=auto_check, mn=min_spin, mx=max_spin: self.apply_y_range(
                    i, ac, mn, mx
                )
            )
            min_spin.editingFinished.connect(
                lambda i=idx, ac=auto_check, mn=min_spin, mx=max_spin: self.apply_y_range(
                    i, ac, mn, mx
                )
            )
            max_spin.editingFinished.connect(
                lambda i=idx, ac=auto_check, mn=min_spin, mx=max_spin: self.apply_y_range(
                    i, ac, mn, mx
                )
            )

            row_layout.addStretch(1)
            row_layout.addWidget(auto_check)
            row_layout.addWidget(QtWidgets.QLabel("Min"))
            row_layout.addWidget(min_spin)
            row_layout.addWidget(QtWidgets.QLabel("Max"))
            row_layout.addWidget(max_spin)

            form_layout.addRow(s.name, row_widget)
            self.controls.append((auto_check, min_spin, max_spin))

    def apply_y_range(self, plot_index, auto_check, min_spin, max_spin):
        plot_item = self.main_window.plots[plot_index]
        if auto_check.isChecked():
            plot_item.enableAutoRange("y", True)
            min_spin.setEnabled(False)
            max_spin.setEnabled(False)
        else:
            plot_item.enableAutoRange("y", False)
            lo, hi = min_spin.value(), max_spin.value()
            if hi <= lo:
                hi = lo + 1e-6
            plot_item.setYRange(lo, hi, padding=0.05)
            min_spin.setEnabled(True)
            max_spin.setEnabled(True)


class DenseViewControlsDialog(QtWidgets.QDialog):
    """Non-modal dialog for adjusting dense view gain, spacing, and step."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Dense View Controls")
        self.setModal(False)
        self.main_window = parent

        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, 1)

        container = QtWidgets.QWidget()
        scroll.setWidget(container)
        main_layout = QtWidgets.QVBoxLayout(container)

        self.resize(420, 500)
        self._group_widgets: list[dict] = []

        for gi, group in enumerate(self.main_window.dense_groups):
            grp_box = QtWidgets.QGroupBox(group.name)
            grp_layout = QtWidgets.QFormLayout(grp_box)

            # Gain slider + spinbox
            gain_layout = QtWidgets.QHBoxLayout()
            gain_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            gain_slider.setRange(-200, 200)  # log10(gain) * 100
            gain_slider.setValue(int(math.log10(max(1e-6, group.gain)) * 100))
            gain_spin = QtWidgets.QDoubleSpinBox()
            gain_spin.setRange(0.01, 100.0)
            gain_spin.setDecimals(2)
            gain_spin.setValue(group.gain)
            gain_spin.setSingleStep(0.1)
            gain_layout.addWidget(gain_slider, 3)
            gain_layout.addWidget(gain_spin, 1)
            grp_layout.addRow("Gain:", gain_layout)

            # Step spinbox
            step_spin = QtWidgets.QSpinBox()
            step_spin.setRange(1, max(1, len(group.series)))
            step_spin.setValue(group.step)
            grp_layout.addRow("Step (nth trace):", step_spin)

            # Traces per page spinbox (0 = all)
            tpp_spin = QtWidgets.QSpinBox()
            tpp_spin.setRange(0, max(1, len(group.series)))
            tpp_spin.setSpecialValueText("All")
            tpp_spin.setValue(group.traces_per_page or 0)
            grp_layout.addRow("Traces per page:", tpp_spin)

            main_layout.addWidget(grp_box)

            widgets = {
                "gain_slider": gain_slider,
                "gain_spin": gain_spin,
                "step_spin": step_spin,
                "tpp_spin": tpp_spin,
            }
            self._group_widgets.append(widgets)

            # Connect signals
            gain_slider.valueChanged.connect(
                lambda val, g=gi, sp=gain_spin: self._on_gain_slider(g, val, sp)
            )
            gain_spin.valueChanged.connect(
                lambda val, g=gi, sl=gain_slider: self._on_gain_spin(g, val, sl)
            )
            step_spin.valueChanged.connect(
                lambda val, g=gi: self._on_step_changed(g, val)
            )

            tpp_spin.valueChanged.connect(
                lambda val, g=gi: self._on_tpp_changed(g, val)
            )

    def _on_gain_slider(self, gi: int, slider_val: int, spin: QtWidgets.QDoubleSpinBox):
        gain = 10 ** (slider_val / 100.0)
        spin.blockSignals(True)
        spin.setValue(gain)
        spin.blockSignals(False)
        self.main_window.dense_groups[gi].gain = gain
        self.main_window._refresh_dense_curves()

    def _on_gain_spin(self, gi: int, val: float, slider: QtWidgets.QSlider):
        val = max(0.01, val)
        slider.blockSignals(True)
        slider.setValue(int(math.log10(val) * 100))
        slider.blockSignals(False)
        self.main_window.dense_groups[gi].gain = val
        self.main_window._refresh_dense_curves()

    def _on_step_changed(self, gi: int, val: int):
        self.main_window.dense_groups[gi].step = val
        self.main_window._rebuild_dense_curves(gi)
        self.main_window._refresh_dense_curves()

    def _on_tpp_changed(self, gi: int, val: int):
        group = self.main_window.dense_groups[gi]
        group.traces_per_page = val if val > 0 else None
        # Reset Y-range to show the requested page size
        offsets = self.main_window._dense_offsets(gi)
        if len(offsets) > 0:
            plt = self.main_window.dense_plots[gi]
            margin = self.main_window._dense_offset_margin(offsets)
            tpp = group.traces_per_page
            if tpp is not None and tpp < len(offsets):
                page_max = float(offsets[min(tpp, len(offsets)) - 1])
                plt.setYRange(float(offsets[0]) - margin, page_max + margin, padding=0)
            else:
                plt.setYRange(
                    float(offsets.min()) - margin,
                    float(offsets.max()) + margin,
                    padding=0,
                )
        self.main_window._setup_dense_vscrollbar_for_group(gi)


class ColormapLevelsDialog(QtWidgets.QDialog):
    """Bulk vmin/vmax editor for heatmaps grouped by current colormap."""

    def __init__(self, parent: "HeatmapControlsDialog"):
        super().__init__(parent)
        self.setWindowTitle("Adjust vmin/vmax by colormap")
        self.setModal(False)
        self.controls_dialog = parent
        self._entries = self._current_colormap_entries()
        self._selected_tokens: frozenset[object] = frozenset()

        layout = QtWidgets.QVBoxLayout(self)
        instructions = QtWidgets.QLabel(
            "Select any combination of the colormaps currently in use. "
            "Level changes are applied immediately to every matching heatmap."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        self.colormap_list = QtWidgets.QListWidget()
        for entry_index, (_token, display_name, count) in enumerate(self._entries):
            suffix = "heatmap" if count == 1 else "heatmaps"
            item = QtWidgets.QListWidgetItem(
                f"{display_name} — {count} {suffix}", self.colormap_list
            )
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, entry_index)
        layout.addWidget(self.colormap_list, 1)

        selection_buttons = QtWidgets.QHBoxLayout()
        self.select_all_btn = QtWidgets.QPushButton("Select all")
        self.clear_btn = QtWidgets.QPushButton("Clear")
        selection_buttons.addWidget(self.select_all_btn)
        selection_buttons.addWidget(self.clear_btn)
        selection_buttons.addStretch(1)
        layout.addLayout(selection_buttons)

        levels = QtWidgets.QFormLayout()
        self.vmin_spin = self._make_level_spinbox()
        self.vmax_spin = self._make_level_spinbox()
        levels.addRow("vmin:", self.vmin_spin)
        levels.addRow("vmax:", self.vmax_spin)
        layout.addLayout(levels)

        self.status_label = QtWidgets.QLabel("Select at least one colormap.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.colormap_list.itemChanged.connect(self._on_selection_changed)
        self.select_all_btn.clicked.connect(self._select_all)
        self.clear_btn.clicked.connect(self._clear_selection)
        self.vmin_spin.valueChanged.connect(
            lambda value: self._apply_level("vmin", value)
        )
        self.vmax_spin.valueChanged.connect(
            lambda value: self._apply_level("vmax", value)
        )
        self._set_level_controls_enabled(False)
        self.resize(470, 390)

    @staticmethod
    def _make_level_spinbox() -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(-1e100, 1e100)
        spin.setSingleStep(0.01)
        spin.setKeyboardTracking(True)
        return spin

    def _current_colormap_entries(self) -> list[tuple[object, str, int]]:
        entries: list[list] = []
        by_token: dict[object, int] = {}
        for heatmap in self.controls_dialog.main_window.heatmap_series:
            token = _colormap_cache_token(heatmap.colormap)
            if token in by_token:
                entries[by_token[token]][2] += 1
                continue
            by_token[token] = len(entries)
            entries.append([token, _colormap_display_name(heatmap.colormap), 1])

        # Distinct custom colormap objects may share a display name. Make those
        # rows unambiguous without changing how matching itself works.
        name_totals: dict[str, int] = {}
        for _, name, _ in entries:
            name_totals[name] = name_totals.get(name, 0) + 1
        name_seen: dict[str, int] = {}
        result = []
        for token, name, count in entries:
            if name_totals[name] > 1:
                name_seen[name] = name_seen.get(name, 0) + 1
                name = f"{name} [{name_seen[name]}]"
            result.append((token, name, count))
        return result

    def _checked_tokens(self) -> frozenset[object]:
        tokens = []
        for row in range(self.colormap_list.count()):
            item = self.colormap_list.item(row)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                entry_index = int(item.data(QtCore.Qt.ItemDataRole.UserRole))
                tokens.append(self._entries[entry_index][0])
        return frozenset(tokens)

    def _set_level_controls_enabled(self, enabled: bool) -> None:
        self.vmin_spin.setEnabled(enabled)
        self.vmax_spin.setEnabled(enabled)

    def _matching_heatmaps(self):
        return [
            heatmap
            for heatmap in self.controls_dialog.main_window.heatmap_series
            if _colormap_cache_token(heatmap.colormap) in self._selected_tokens
        ]

    def _sync_levels_from_heatmaps(self) -> None:
        heatmaps = self._matching_heatmaps()
        if not heatmaps:
            return
        self.vmin_spin.blockSignals(True)
        self.vmax_spin.blockSignals(True)
        try:
            self.vmin_spin.setValue(float(heatmaps[0].vmin))
            self.vmax_spin.setValue(float(heatmaps[0].vmax))
        finally:
            self.vmin_spin.blockSignals(False)
            self.vmax_spin.blockSignals(False)

    def _update_status(self) -> None:
        heatmaps = self._matching_heatmaps()
        if not heatmaps:
            self.status_label.setText("Select at least one colormap.")
            return
        vmins = {float(heatmap.vmin) for heatmap in heatmaps}
        vmaxs = {float(heatmap.vmax) for heatmap in heatmaps}
        if len(vmins) > 1 or len(vmaxs) > 1:
            self.status_label.setText(
                f"{len(heatmaps)} heatmaps selected; their current levels are mixed. "
                "Editing either field applies that value to all selected heatmaps."
            )
        else:
            self.status_label.setText(
                f"{len(heatmaps)} heatmaps selected. Changes apply immediately."
            )

    def _on_selection_changed(self, _item=None) -> None:
        selected = self._checked_tokens()
        changed = selected != self._selected_tokens
        self._selected_tokens = selected
        self._set_level_controls_enabled(bool(selected))
        if selected and changed:
            self._sync_levels_from_heatmaps()
        self._update_status()

    def _set_all_checked(self, checked: bool) -> None:
        self.colormap_list.blockSignals(True)
        try:
            state = (
                QtCore.Qt.CheckState.Checked
                if checked
                else QtCore.Qt.CheckState.Unchecked
            )
            for row in range(self.colormap_list.count()):
                self.colormap_list.item(row).setCheckState(state)
        finally:
            self.colormap_list.blockSignals(False)
        self._on_selection_changed()

    def _select_all(self) -> None:
        self._set_all_checked(True)

    def _clear_selection(self) -> None:
        self._set_all_checked(False)

    def _apply_level(self, attribute: str, value: float) -> None:
        if not self._selected_tokens:
            return
        self.controls_dialog._apply_level_by_colormap(
            self._selected_tokens, attribute, float(value)
        )
        self._update_status()


class HeatmapControlsDialog(QtWidgets.QDialog):
    """Non-modal dialog for adjusting per-heatmap vmin/vmax, colormap, and decim method."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Heatmap Plot Controls")
        self.setModal(False)
        self.main_window = parent

        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, 1)

        container = QtWidgets.QWidget()
        scroll.setWidget(container)
        main_layout = QtWidgets.QVBoxLayout(container)

        self._group_widgets: list[dict] = []
        self._colormap_levels_dialog = None

        self.adjust_by_colormap_btn = QtWidgets.QPushButton(
            "Adjust vmin/vmax by colormap"
        )
        self.adjust_by_colormap_btn.clicked.connect(
            self._show_colormap_levels_dialog
        )
        main_layout.addWidget(self.adjust_by_colormap_btn)

        for ai, asx in enumerate(self.main_window.heatmap_series):
            grp_box = QtWidgets.QGroupBox(asx.name)
            grp_layout = QtWidgets.QFormLayout(grp_box)

            # Determine slider range from the data values (finite only).
            finite = asx.Y[np.isfinite(asx.Y)]
            if finite.size > 0:
                d_lo = float(np.min(finite))
                d_hi = float(np.max(finite))
                if d_hi <= d_lo:
                    d_hi = d_lo + 1.0
                pad = 0.1 * (d_hi - d_lo)
                slider_lo = d_lo - pad
                slider_hi = d_hi + pad
            else:
                slider_lo, slider_hi = 0.0, 1.0
            slider_span = max(slider_hi - slider_lo, 1e-9)

            # vmin slider + spinbox
            vmin_layout = QtWidgets.QHBoxLayout()
            vmin_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            vmin_slider.setRange(0, 10000)
            vmin_slider.setValue(
                int(((asx.vmin - slider_lo) / slider_span) * 10000)
            )
            vmin_spin = QtWidgets.QDoubleSpinBox()
            vmin_spin.setRange(slider_lo - 10 * slider_span, slider_hi + 10 * slider_span)
            vmin_spin.setDecimals(4)
            vmin_spin.setValue(asx.vmin)
            vmin_spin.setSingleStep(slider_span / 100.0)
            vmin_layout.addWidget(vmin_slider, 3)
            vmin_layout.addWidget(vmin_spin, 1)
            grp_layout.addRow("vmin:", vmin_layout)

            # vmax slider + spinbox
            vmax_layout = QtWidgets.QHBoxLayout()
            vmax_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            vmax_slider.setRange(0, 10000)
            vmax_slider.setValue(
                int(((asx.vmax - slider_lo) / slider_span) * 10000)
            )
            vmax_spin = QtWidgets.QDoubleSpinBox()
            vmax_spin.setRange(slider_lo - 10 * slider_span, slider_hi + 10 * slider_span)
            vmax_spin.setDecimals(4)
            vmax_spin.setValue(asx.vmax)
            vmax_spin.setSingleStep(slider_span / 100.0)
            vmax_layout.addWidget(vmax_slider, 3)
            vmax_layout.addWidget(vmax_spin, 1)
            grp_layout.addRow("vmax:", vmax_layout)

            # Reset to robust 1-99% percentile button
            reset_btn = QtWidgets.QPushButton("Reset to 1–99% percentile")
            grp_layout.addRow("", reset_btn)

            # Colormap dropdown
            cmap_combo = QtWidgets.QComboBox()
            cmap_combo.setEditable(True)
            for name in ARRAY_COLORMAP_PRESETS:
                cmap_combo.addItem(name)
            cmap_label = _colormap_display_name(asx.colormap)
            if cmap_label not in ARRAY_COLORMAP_PRESETS:
                cmap_combo.addItem(cmap_label)
            cmap_combo.setCurrentText(cmap_label)
            grp_layout.addRow("Colormap:", cmap_combo)

            # Decimation method radios
            decim_box = QtWidgets.QHBoxLayout()
            decim_peak = QtWidgets.QRadioButton("Peak (max-abs)")
            decim_mean = QtWidgets.QRadioButton("Mean")
            if asx.decim_method == "mean":
                decim_mean.setChecked(True)
            else:
                decim_peak.setChecked(True)
            decim_grp = QtWidgets.QButtonGroup(grp_box)
            decim_grp.addButton(decim_peak, 0)
            decim_grp.addButton(decim_mean, 1)
            decim_box.addWidget(decim_peak)
            decim_box.addWidget(decim_mean)
            decim_box.addStretch(1)
            grp_layout.addRow("Decimation:", decim_box)

            # Apply-to-all button
            apply_all_btn = QtWidgets.QPushButton("Apply to all heatmaps")
            grp_layout.addRow("", apply_all_btn)

            main_layout.addWidget(grp_box)

            widgets = {
                "vmin_slider": vmin_slider,
                "vmin_spin": vmin_spin,
                "vmax_slider": vmax_slider,
                "vmax_spin": vmax_spin,
                "cmap_combo": cmap_combo,
                "decim_peak": decim_peak,
                "decim_mean": decim_mean,
                "slider_lo": slider_lo,
                "slider_span": slider_span,
            }
            self._group_widgets.append(widgets)

            # Wire signals
            vmin_slider.valueChanged.connect(
                lambda val, ai=ai: self._on_vmin_slider(ai, val)
            )
            vmin_spin.valueChanged.connect(
                lambda val, ai=ai: self._on_vmin_spin(ai, val)
            )
            vmax_slider.valueChanged.connect(
                lambda val, ai=ai: self._on_vmax_slider(ai, val)
            )
            vmax_spin.valueChanged.connect(
                lambda val, ai=ai: self._on_vmax_spin(ai, val)
            )
            cmap_combo.currentTextChanged.connect(
                lambda name, ai=ai: self._on_cmap_changed(ai, name)
            )
            decim_peak.toggled.connect(
                lambda checked, ai=ai: (
                    self._on_decim_changed(ai, "peak") if checked else None
                )
            )
            decim_mean.toggled.connect(
                lambda checked, ai=ai: (
                    self._on_decim_changed(ai, "mean") if checked else None
                )
            )
            reset_btn.clicked.connect(lambda _=False, ai=ai: self._reset_levels(ai))
            apply_all_btn.clicked.connect(lambda _=False, ai=ai: self._apply_to_all(ai))

        outer.addStretch(0)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

    # ---- helpers --------------------------------------------------------

    def _slider_to_value(self, ai: int, slider_val: int) -> float:
        w = self._group_widgets[ai]
        return w["slider_lo"] + (slider_val / 10000.0) * w["slider_span"]

    def _value_to_slider(self, ai: int, value: float) -> int:
        w = self._group_widgets[ai]
        return int(((value - w["slider_lo"]) / w["slider_span"]) * 10000)

    def _invalidate_array_cache(self, ai: int) -> None:
        if ai < len(self.main_window._heatmap_cache_keys):
            self.main_window._heatmap_cache_keys[ai] = None

    def _refresh_one(self, ai: int) -> None:
        self._invalidate_array_cache(ai)
        self.main_window._refresh_heatmap_plots()

    def _sync_widgets_to_state(self, ai: int) -> None:
        """Update slider/spinbox/combo/radio without firing handlers."""
        if ai >= len(self._group_widgets):
            return
        w = self._group_widgets[ai]
        asx = self.main_window.heatmap_series[ai]
        for sl in (w["vmin_slider"], w["vmax_slider"]):
            sl.blockSignals(True)
        for sp in (w["vmin_spin"], w["vmax_spin"]):
            sp.blockSignals(True)
        w["cmap_combo"].blockSignals(True)
        w["decim_peak"].blockSignals(True)
        w["decim_mean"].blockSignals(True)
        try:
            w["vmin_slider"].setValue(self._value_to_slider(ai, asx.vmin))
            if not (
                w["vmin_spin"].minimum()
                <= asx.vmin
                <= w["vmin_spin"].maximum()
            ):
                w["vmin_spin"].setRange(
                    min(w["vmin_spin"].minimum(), asx.vmin),
                    max(w["vmin_spin"].maximum(), asx.vmin),
                )
            w["vmin_spin"].setValue(asx.vmin)
            w["vmax_slider"].setValue(self._value_to_slider(ai, asx.vmax))
            if not (
                w["vmax_spin"].minimum()
                <= asx.vmax
                <= w["vmax_spin"].maximum()
            ):
                w["vmax_spin"].setRange(
                    min(w["vmax_spin"].minimum(), asx.vmax),
                    max(w["vmax_spin"].maximum(), asx.vmax),
                )
            w["vmax_spin"].setValue(asx.vmax)
            cmap_label = _colormap_display_name(asx.colormap)
            if w["cmap_combo"].findText(cmap_label) < 0:
                w["cmap_combo"].addItem(cmap_label)
            w["cmap_combo"].setCurrentText(cmap_label)
            if asx.decim_method == "mean":
                w["decim_mean"].setChecked(True)
            else:
                w["decim_peak"].setChecked(True)
        finally:
            for sl in (w["vmin_slider"], w["vmax_slider"]):
                sl.blockSignals(False)
            for sp in (w["vmin_spin"], w["vmax_spin"]):
                sp.blockSignals(False)
            w["cmap_combo"].blockSignals(False)
            w["decim_peak"].blockSignals(False)
            w["decim_mean"].blockSignals(False)

    # ---- handlers -------------------------------------------------------

    def _on_vmin_slider(self, ai: int, slider_val: int) -> None:
        v = self._slider_to_value(ai, slider_val)
        asx = self.main_window.heatmap_series[ai]
        asx.vmin = float(v)
        spin = self._group_widgets[ai]["vmin_spin"]
        spin.blockSignals(True)
        spin.setValue(v)
        spin.blockSignals(False)
        self._refresh_one(ai)

    def _on_vmin_spin(self, ai: int, value: float) -> None:
        asx = self.main_window.heatmap_series[ai]
        asx.vmin = float(value)
        slider = self._group_widgets[ai]["vmin_slider"]
        slider.blockSignals(True)
        slider.setValue(self._value_to_slider(ai, value))
        slider.blockSignals(False)
        self._refresh_one(ai)

    def _on_vmax_slider(self, ai: int, slider_val: int) -> None:
        v = self._slider_to_value(ai, slider_val)
        asx = self.main_window.heatmap_series[ai]
        asx.vmax = float(v)
        spin = self._group_widgets[ai]["vmax_spin"]
        spin.blockSignals(True)
        spin.setValue(v)
        spin.blockSignals(False)
        self._refresh_one(ai)

    def _on_vmax_spin(self, ai: int, value: float) -> None:
        asx = self.main_window.heatmap_series[ai]
        asx.vmax = float(value)
        slider = self._group_widgets[ai]["vmax_slider"]
        slider.blockSignals(True)
        slider.setValue(self._value_to_slider(ai, value))
        slider.blockSignals(False)
        self._refresh_one(ai)

    def _on_cmap_changed(self, ai: int, name: str) -> None:
        if not name:
            return
        self.main_window.heatmap_series[ai].colormap = str(name)
        self._refresh_one(ai)

    def _on_decim_changed(self, ai: int, method: str) -> None:
        asx = self.main_window.heatmap_series[ai]
        if asx.decim_method == method:
            return
        asx.decim_method = method
        # Rebuild the mip-map with the new reduction method, if one exists.
        if asx.mipmap_levels is not None:
            from loupe.xr_loader import _build_mipmap

            asx.mipmap_levels = _build_mipmap(
                asx.Y, method, ARRAY_MIPMAP_TARGET_MIN_COLS
            )
        self._refresh_one(ai)

    def _reset_levels(self, ai: int) -> None:
        asx = self.main_window.heatmap_series[ai]
        finite = asx.Y[np.isfinite(asx.Y)]
        if finite.size == 0:
            return
        asx.vmin = float(np.percentile(finite, 1.0))
        asx.vmax = float(np.percentile(finite, 99.0))
        if asx.vmax <= asx.vmin:
            asx.vmax = asx.vmin + 1.0
        self._sync_widgets_to_state(ai)
        self._refresh_one(ai)

    def _apply_to_all(self, ai: int) -> None:
        src = self.main_window.heatmap_series[ai]
        for j, asx in enumerate(self.main_window.heatmap_series):
            if j == ai:
                continue
            asx.vmin = src.vmin
            asx.vmax = src.vmax
            asx.colormap = src.colormap
            asx.decim_method = src.decim_method
            self._sync_widgets_to_state(j)
            self._invalidate_array_cache(j)
        self.main_window._refresh_heatmap_plots()

    def _show_colormap_levels_dialog(self) -> None:
        if self._colormap_levels_dialog is not None:
            self._colormap_levels_dialog.close()
            self._colormap_levels_dialog.deleteLater()
        self._colormap_levels_dialog = ColormapLevelsDialog(self)
        self._colormap_levels_dialog.show()
        self._colormap_levels_dialog.raise_()
        self._colormap_levels_dialog.activateWindow()

    def _apply_level_by_colormap(
        self, tokens: frozenset[object], attribute: str, value: float
    ) -> None:
        """Apply one level to all heatmaps using any selected colormap."""
        changed_indices = []
        for ai, heatmap in enumerate(self.main_window.heatmap_series):
            if _colormap_cache_token(heatmap.colormap) not in tokens:
                continue
            setattr(heatmap, attribute, value)
            self._sync_widgets_to_state(ai)
            self._invalidate_array_cache(ai)
            changed_indices.append(ai)
        if changed_indices:
            self.main_window._refresh_heatmap_plots()
