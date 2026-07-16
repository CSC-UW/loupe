"""The Tuner dock — a PySide6 panel of live controls, one per :class:`Param`.

Kept separate from :mod:`loupe.tuner` (which is Qt-free) so importing the
primitives never pulls in PySide6.  Only built when a window with tunable
params is shown.

Each control follows the established loupe idiom (see
:class:`loupe.dialogs.DenseViewControlsDialog` /
:class:`loupe.dialogs.HeatmapControlsDialog`): a slider and spin box kept in
sync via ``blockSignals``, whose handler writes ``param.value`` and then asks
the main window to recompute (debounced).
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6 import QtCore, QtWidgets

from loupe.tuner import BoolParam, ChoiceParam, IntParam, Param

_SLIDER_TICKS = 1000


class TunerDock(QtWidgets.QDockWidget):
    """A dockable panel exposing one control per tunable :class:`Param`."""

    def __init__(self, main_window, params: list[Param]) -> None:
        super().__init__("Tuner", main_window)
        self.main_window = main_window
        self.params = list(params)
        self.setObjectName("loupe_tuner_dock")
        self.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        # Zero-arg callbacks that push each param's current value back into its
        # widgets (used by "Reset").
        self._sync_callbacks: list[Callable[[], None]] = []

        container = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(container)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        form_host = QtWidgets.QWidget()
        self.form = QtWidgets.QFormLayout(form_host)
        scroll.setWidget(form_host)
        outer.addWidget(scroll, 1)

        for i, p in enumerate(self.params):
            name = p.name if p.name else f"param {i}"
            self.form.addRow(name, self._build_control(p))

        btn_row = QtWidgets.QHBoxLayout()
        copy_btn = QtWidgets.QPushButton("Copy values")
        copy_btn.setToolTip(
            "Copy a {name: value, ...} dict of the current params to the "
            "clipboard, ready to paste back into your notebook."
        )
        copy_btn.clicked.connect(self._copy_values)
        reset_btn = QtWidgets.QPushButton("Reset")
        reset_btn.setToolTip("Reset every param to its default value.")
        reset_btn.clicked.connect(self._reset_all)
        btn_row.addWidget(copy_btn)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        self.setWidget(container)
        self.resize(340, min(640, 110 + 56 * max(1, len(self.params))))

    # ---- per-type control builders -------------------------------------

    def _build_control(self, p: Param) -> QtWidgets.QWidget:
        # Order matters: the typed subclasses must be checked before Param.
        if isinstance(p, BoolParam):
            return self._build_bool(p)
        if isinstance(p, ChoiceParam):
            return self._build_choice(p)
        if isinstance(p, IntParam):
            return self._build_int(p)
        return self._build_float(p)

    def _build_float(self, p: Param) -> QtWidgets.QWidget:
        has_range = p.min is not None and p.max is not None
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)

        span = (float(p.max) - float(p.min)) if has_range else 1.0
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(self._decimals_for(p, span))
        spin.setRange(
            float(p.min) if p.min is not None else -1e12,
            float(p.max) if p.max is not None else 1e12,
        )
        spin.setSingleStep(
            float(p.step) if p.step else (span / 100.0 if has_range else 0.01)
        )
        spin.setValue(float(p.value))

        slider = None
        if has_range:
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(0, _SLIDER_TICKS)
            slider.setValue(self._v2s(p.value, p.min, p.max))
            h.addWidget(slider, 3)
        h.addWidget(spin, 1)

        def on_slider(sv: int) -> None:
            p.value = self._s2v(sv, p.min, p.max)
            spin.blockSignals(True)
            spin.setValue(float(p.value))
            spin.blockSignals(False)
            self._notify(p)

        def on_spin(v: float) -> None:
            p.value = v
            if slider is not None:
                slider.blockSignals(True)
                slider.setValue(self._v2s(p.value, p.min, p.max))
                slider.blockSignals(False)
            self._notify(p)

        if slider is not None:
            slider.valueChanged.connect(on_slider)
        spin.valueChanged.connect(on_spin)

        def sync() -> None:
            spin.blockSignals(True)
            spin.setValue(float(p.value))
            spin.blockSignals(False)
            if slider is not None:
                slider.blockSignals(True)
                slider.setValue(self._v2s(p.value, p.min, p.max))
                slider.blockSignals(False)

        self._sync_callbacks.append(sync)
        return w

    def _build_int(self, p: Param) -> QtWidgets.QWidget:
        has_range = p.min is not None and p.max is not None
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)

        lo = int(p.min) if p.min is not None else -2_000_000_000
        hi = int(p.max) if p.max is not None else 2_000_000_000
        spin = QtWidgets.QSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(int(p.step) if p.step else 1)
        spin.setValue(int(p.value))

        slider = None
        if has_range and (hi - lo) <= 100_000:
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(lo, hi)
            slider.setValue(int(p.value))
            h.addWidget(slider, 3)
        h.addWidget(spin, 1)

        def on_slider(v: int) -> None:
            p.value = v
            spin.blockSignals(True)
            spin.setValue(int(p.value))
            spin.blockSignals(False)
            self._notify(p)

        def on_spin(v: int) -> None:
            p.value = v
            if slider is not None:
                slider.blockSignals(True)
                slider.setValue(int(p.value))
                slider.blockSignals(False)
            self._notify(p)

        if slider is not None:
            slider.valueChanged.connect(on_slider)
        spin.valueChanged.connect(on_spin)

        def sync() -> None:
            spin.blockSignals(True)
            spin.setValue(int(p.value))
            spin.blockSignals(False)
            if slider is not None:
                slider.blockSignals(True)
                slider.setValue(int(p.value))
                slider.blockSignals(False)

        self._sync_callbacks.append(sync)
        return w

    def _build_bool(self, p: Param) -> QtWidgets.QWidget:
        chk = QtWidgets.QCheckBox()
        chk.setChecked(bool(p.value))

        def on_toggled(checked: bool) -> None:
            p.value = checked
            self._notify(p)

        chk.toggled.connect(on_toggled)

        def sync() -> None:
            chk.blockSignals(True)
            chk.setChecked(bool(p.value))
            chk.blockSignals(False)

        self._sync_callbacks.append(sync)
        return chk

    def _build_choice(self, p: ChoiceParam) -> QtWidgets.QWidget:
        combo = QtWidgets.QComboBox()
        for c in p.choices:
            combo.addItem(str(c), c)
        combo.setCurrentIndex(p.choices.index(p.value))

        def on_changed(i: int) -> None:
            if i < 0:
                return
            p.value = combo.itemData(i)
            self._notify(p)

        combo.currentIndexChanged.connect(on_changed)

        def sync() -> None:
            combo.blockSignals(True)
            combo.setCurrentIndex(p.choices.index(p.value))
            combo.blockSignals(False)

        self._sync_callbacks.append(sync)
        return combo

    # ---- helpers --------------------------------------------------------

    @staticmethod
    def _s2v(slider_val: int, lo: float, hi: float) -> float:
        return float(lo) + (slider_val / _SLIDER_TICKS) * (float(hi) - float(lo))

    @staticmethod
    def _v2s(value: float, lo: float, hi: float) -> int:
        span = float(hi) - float(lo)
        if span <= 0:
            return 0
        frac = (float(value) - float(lo)) / span
        return int(round(max(0.0, min(1.0, frac)) * _SLIDER_TICKS))

    @staticmethod
    def _decimals_for(p: Param, span: float) -> int:
        if p.step:
            s = repr(float(p.step))
            if "." in s:
                return min(8, max(1, len(s.split(".")[1].rstrip("0"))))
            return 0
        if span <= 1:
            return 4
        if span <= 100:
            return 3
        return 2

    def _notify(self, p: Param) -> None:
        self.main_window._on_tuner_param_changed(p)

    def sync_from_params(self) -> None:
        """Refresh controls after values are changed outside this dock."""
        for sync in self._sync_callbacks:
            sync()

    def _copy_values(self) -> None:
        d: dict[str, Any] = {}
        for i, p in enumerate(self.params):
            d[p.name or f"param_{i}"] = p.value
        QtWidgets.QApplication.clipboard().setText(repr(d))
        if hasattr(self.main_window, "_update_status"):
            self.main_window._update_status(
                f"Tuner: copied {len(d)} param value(s) to clipboard."
            )

    def _reset_all(self) -> None:
        for p in self.params:
            p.reset()
        self.sync_from_params()
        for p in self.params:
            self.main_window._on_tuner_param_changed(p)
