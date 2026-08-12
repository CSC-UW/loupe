"""The "Open in loupe…" launcher dialog used by :mod:`loupe.varview`.

Given a variable (xarray Dataset/DataArray, polars DataFrame, or NumPy
array), presents a small config-builder form: the user picks the view
type (stacked traces / dense traces / heatmap / raster), sets the
relevant options (``order_by``, ``split_by``, ``hue``, colormap, …) with
dropdowns populated from the object's own coords/columns, and launches
:func:`loupe.view` on the result.

The dialog is **modeless** (``show()``, never ``exec()``) so it plays
nicely inside a kernel-integrated Qt event loop, and launched LoupeApp
windows are kept alive in a module-level registry.
"""

from __future__ import annotations

import sys
from typing import Any

from PySide6 import QtCore, QtWidgets

#: Keep-alive registries: Qt widgets owned only by Python must be
#: referenced somewhere or they are garbage-collected mid-display.
_OPEN_DIALOGS: list["LoupeLaunchDialog"] = []
_LAUNCHED_VIEWS: list[Any] = []

_NONE_LABEL = "(none)"

_CMAP_CHOICES = [
    "magma", "viridis", "plasma", "inferno", "cividis",
    "gray", "RdBu_r", "coolwarm", "turbo",
]

#: Stacked-subplots views beyond this many traces prompt for confirmation.
_STACKED_TRACE_WARN = 200


def _keep_alive(registry: list, obj: Any) -> None:
    registry.append(obj)
    try:
        obj.destroyed.connect(lambda *_: registry.remove(obj) if obj in registry else None)
    except Exception:
        pass


def open_launcher(
    name: str, obj: Any, parent: QtWidgets.QWidget | None = None
) -> "LoupeLaunchDialog":
    """Show the launcher dialog for *obj* (modeless) and return it."""
    dlg = LoupeLaunchDialog(name, obj, parent=parent)
    _keep_alive(_OPEN_DIALOGS, dlg)
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg


# ---------------------------------------------------------------------------
# Object adapters — normalize the supported types into "a DataArray source
# with pickable dims" or "a DataFrame source with pickable columns".
# ---------------------------------------------------------------------------


def _kind_of(obj: Any) -> str:
    pl = sys.modules.get("polars")
    xr = sys.modules.get("xarray")
    np = sys.modules.get("numpy")
    if pl is not None and isinstance(obj, pl.DataFrame):
        return "dataframe"
    if xr is not None and isinstance(obj, xr.Dataset):
        return "dataset"
    if xr is not None and isinstance(obj, xr.DataArray):
        return "dataarray"
    if np is not None and isinstance(obj, np.ndarray):
        return "ndarray"
    raise TypeError(
        f"Open in loupe supports xarray Dataset/DataArray, polars "
        f"DataFrame, and numpy ndarray — got {type(obj).__name__}."
    )


def _prepare_dataarray(da, time_dim: str, *, fs: float | None = None):
    """Return *da* renamed so *time_dim* is ``'time'``, with coords ensured.

    Loupe's converters require a dim literally named ``time`` **with a
    coordinate**, and a coordinate on every non-time dim. Missing coords
    become integer indices (or ``arange/fs`` seconds for time when *fs*
    is given).
    """
    import numpy as np

    if time_dim != "time":
        if "time" in da.dims:
            da = da.rename({"time": "time_"})
        da = da.rename({time_dim: "time"})
    if "time" not in da.coords:
        n = da.sizes["time"]
        t = np.arange(n, dtype=float)
        if fs is not None and fs > 0:
            t = t / fs
        da = da.assign_coords(time=t)
    for d in da.dims:
        if d != "time" and d not in da.coords:
            da = da.assign_coords({d: np.arange(da.sizes[d])})
    return da


def _ndarray_to_dataarray(arr, name: str, time_axis: int, fs: float):
    """Wrap a 1-D/2-D ndarray in a DataArray with a proper time axis."""
    import numpy as np
    import xarray as xr

    if arr.ndim == 1:
        da = xr.DataArray(arr, dims=["time"], name=name or "array")
    elif arr.ndim == 2:
        dims = ["row", "time"] if time_axis == 1 else ["time", "row"]
        da = xr.DataArray(arr, dims=dims, name=name or "array")
        if dims[0] == "time":
            da = da.transpose("row", "time")
        da = da.assign_coords(row=np.arange(da.sizes["row"]))
    else:
        raise ValueError(
            f"Only 1-D / 2-D arrays can be opened in loupe (got {arr.ndim}-D)."
        )
    n = da.sizes["time"]
    t = np.arange(n, dtype=float)
    if fs > 0:
        t = t / fs
    return da.assign_coords(time=t)


def _orderable_coords(da, *, exclude_time: bool = True) -> list[str]:
    """Coord names usable for ``order_by`` / ``hue``: 1-D on a non-time dim."""
    out = []
    for cname, coord in da.coords.items():
        if len(coord.dims) != 1:
            continue
        if exclude_time and coord.dims[0] == "time":
            continue
        out.append(str(cname))
    return out


def _splittable_names(da) -> list[str]:
    """Names valid for ``HeatmapConfig.split_by``: non-time coords or dims."""
    names = _orderable_coords(da)
    for d in da.dims:
        if d != "time" and str(d) not in names:
            names.append(str(d))
    return names


def _numeric_columns(df) -> list[str]:
    return [c for c, t in df.schema.items() if t.is_numeric()]


# ---------------------------------------------------------------------------
# Small form helpers
# ---------------------------------------------------------------------------


def _combo(items: list[str], *, none_option: bool = True,
           current: str | None = None) -> QtWidgets.QComboBox:
    box = QtWidgets.QComboBox()
    if none_option:
        box.addItem(_NONE_LABEL)
    box.addItems(items)
    if current is not None:
        idx = box.findText(current)
        if idx >= 0:
            box.setCurrentIndex(idx)
    return box


def _combo_value(box: QtWidgets.QComboBox) -> str | None:
    text = box.currentText()
    return None if text in ("", _NONE_LABEL) else text


def _float_or_none(edit: QtWidgets.QLineEdit) -> float | None:
    text = edit.text().strip()
    if not text:
        return None
    return float(text)  # ValueError propagates to the launch handler


# ---------------------------------------------------------------------------
# The dialog
# ---------------------------------------------------------------------------


class LoupeLaunchDialog(QtWidgets.QDialog):
    """Config-builder dialog: pick a view type, set options, launch loupe."""

    def __init__(
        self, name: str, obj: Any, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._name = name
        self._obj = obj
        self._kind = _kind_of(obj)

        self.setWindowTitle(f"Open in loupe — {name}")
        self.setMinimumWidth(420)

        outer = QtWidgets.QVBoxLayout(self)
        self._top_form = QtWidgets.QFormLayout()
        self._top_form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        outer.addLayout(self._top_form)

        # -- object-level pickers -------------------------------------------
        self._var_combo: QtWidgets.QComboBox | None = None
        self._time_combo: QtWidgets.QComboBox | None = None
        self._axis_combo: QtWidgets.QComboBox | None = None
        self._fs_spin: QtWidgets.QDoubleSpinBox | None = None

        if self._kind == "dataset":
            self._var_combo = _combo(
                [str(v) for v in obj.data_vars], none_option=False
            )
            self._var_combo.currentTextChanged.connect(self._rebuild_forms)
            self._top_form.addRow("Data variable:", self._var_combo)

        if self._kind in ("dataset", "dataarray"):
            self._time_combo = _combo([], none_option=False)
            self._time_combo.currentTextChanged.connect(self._rebuild_forms)
            self._top_form.addRow("Time dimension:", self._time_combo)

        if self._kind == "ndarray":
            if obj.ndim == 2:
                self._axis_combo = _combo(
                    [f"axis {i} (n={obj.shape[i]:,})" for i in range(2)],
                    none_option=False,
                    current=f"axis 1 (n={obj.shape[1]:,})",
                )
                self._axis_combo.currentIndexChanged.connect(
                    lambda _=None: self._rebuild_forms()
                )
                self._top_form.addRow("Time axis:", self._axis_combo)
            self._fs_spin = QtWidgets.QDoubleSpinBox()
            self._fs_spin.setRange(0.000001, 10_000_000.0)
            self._fs_spin.setValue(1.0)
            self._fs_spin.setDecimals(4)
            self._fs_spin.setSuffix(" Hz")
            self._top_form.addRow("Sample rate:", self._fs_spin)

        # -- view-type selector + per-type option forms ---------------------
        self._view_combo = _combo([], none_option=False)
        self._view_combo.currentIndexChanged.connect(
            lambda _=None: self._on_view_type_changed()
        )
        self._top_form.addRow("View as:", self._view_combo)

        self._stack = QtWidgets.QStackedWidget()
        outer.addWidget(self._stack)

        self._window_len = QtWidgets.QDoubleSpinBox()
        self._window_len.setRange(0.01, 1e7)
        self._window_len.setValue(10.0)
        self._window_len.setSuffix(" s")
        bottom_form = QtWidgets.QFormLayout()
        bottom_form.addRow("Initial window:", self._window_len)
        outer.addLayout(bottom_form)

        self._hint = QtWidgets.QLabel("")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color: gray;")
        outer.addWidget(self._hint)

        buttons = QtWidgets.QDialogButtonBox()
        self._launch_btn = buttons.addButton(
            "Launch", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole
        )
        buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._launch)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._forms: dict[str, QtWidgets.QWidget] = {}
        self._rebuild_forms()

    # ------------------------------------------------------------------
    # Form construction
    # ------------------------------------------------------------------

    def _current_dataarray(self):
        """The DataArray the forms describe (before time-dim preparation)."""
        if self._kind == "dataset":
            var = self._var_combo.currentText()
            return self._obj[var] if var else None
        if self._kind == "dataarray":
            return self._obj
        if self._kind == "ndarray":
            time_axis = (
                self._axis_combo.currentIndex()
                if self._axis_combo is not None
                else 0
            )
            fs = self._fs_spin.value() if self._fs_spin is not None else 1.0
            try:
                return _ndarray_to_dataarray(
                    self._obj, self._name, time_axis, fs
                )
            except Exception:
                return None
        return None

    def _selected_time_dim(self, da) -> str:
        if self._time_combo is not None and self._time_combo.currentText():
            return self._time_combo.currentText()
        return "time" if "time" in da.dims else str(list(da.dims)[-1])

    def _rebuild_forms(self, *_args) -> None:
        """(Re)populate the time-dim picker, view types, and option forms."""
        if self._kind == "dataframe":
            self._set_view_types(["Raster"])
            self._forms = {"Raster": self._build_raster_form(self._obj)}
            self._apply_forms()
            return

        da = self._current_dataarray()
        if da is None:
            self._set_view_types([])
            self._forms = {}
            self._apply_forms()
            return

        # Time-dim combo tracks the currently selected data variable.
        if self._time_combo is not None:
            current = self._time_combo.currentText()
            dims = [str(d) for d in da.dims]
            with QtCore.QSignalBlocker(self._time_combo):
                self._time_combo.clear()
                self._time_combo.addItems(dims)
                if current in dims:
                    self._time_combo.setCurrentText(current)
                elif "time" in dims:
                    self._time_combo.setCurrentText("time")
                else:
                    self._time_combo.setCurrentIndex(len(dims) - 1)

        time_dim = self._selected_time_dim(da)
        # Build forms against the *prepared* view of the array so coord
        # dropdowns reflect what loupe will actually see.
        try:
            prepared = _prepare_dataarray(da, time_dim)
        except Exception:
            prepared = da

        n_rows = 1
        for d in prepared.dims:
            if d != "time":
                n_rows *= prepared.sizes[d]

        types = ["Traces (stacked)", "Traces (dense)"]
        if prepared.ndim == 2 or (
            prepared.ndim > 2 and len(_splittable_names(prepared)) > 0
        ):
            types.append("Heatmap")
        self._set_view_types(types)

        self._forms = {
            "Traces (stacked)": self._build_stacked_form(prepared, n_rows),
            "Traces (dense)": self._build_dense_form(prepared),
        }
        if "Heatmap" in types:
            self._forms["Heatmap"] = self._build_heatmap_form(prepared)
        self._apply_forms()

    def _set_view_types(self, types: list[str]) -> None:
        current = self._view_combo.currentText()
        with QtCore.QSignalBlocker(self._view_combo):
            self._view_combo.clear()
            self._view_combo.addItems(types)
            if current in types:
                self._view_combo.setCurrentText(current)
        self._launch_btn.setEnabled(bool(types))

    def _apply_forms(self) -> None:
        while self._stack.count():
            w = self._stack.widget(0)
            self._stack.removeWidget(w)
            w.deleteLater()
        for label in [
            self._view_combo.itemText(i)
            for i in range(self._view_combo.count())
        ]:
            self._stack.addWidget(self._forms[label])
        self._on_view_type_changed()

    def _on_view_type_changed(self) -> None:
        idx = self._view_combo.currentIndex()
        if 0 <= idx < self._stack.count():
            self._stack.setCurrentIndex(idx)
        self._update_hint()

    def _update_hint(self) -> None:
        label = self._view_combo.currentText()
        form = self._forms.get(label)
        hint = getattr(form, "_hint_text", "") if form is not None else ""
        self._hint.setText(hint)

    # -- per-type forms -------------------------------------------------

    @staticmethod
    def _make_form() -> tuple[QtWidgets.QWidget, QtWidgets.QFormLayout]:
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)
        form.setContentsMargins(0, 4, 0, 4)
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        return w, form

    def _build_stacked_form(self, da, n_rows: int) -> QtWidgets.QWidget:
        w, form = self._make_form()
        coords = _orderable_coords(da)
        w._order_by = _combo(coords)
        w._descending = QtWidgets.QCheckBox()
        w._hue = _combo(coords)
        w._color = QtWidgets.QLineEdit()
        w._color.setPlaceholderText("#RRGGBB (optional)")
        form.addRow("Order by:", w._order_by)
        form.addRow("Descending:", w._descending)
        form.addRow("Hue:", w._hue)
        form.addRow("Color:", w._color)
        w._n_rows = n_rows
        w._hint_text = f"Will create {n_rows:,} stacked subplot(s)."
        return w

    def _build_dense_form(self, da) -> QtWidgets.QWidget:
        w, form = self._make_form()
        coords = _orderable_coords(da)
        w._order_by = _combo(coords)
        w._descending = QtWidgets.QCheckBox()
        w._hue = _combo(coords)
        w._gain = QtWidgets.QDoubleSpinBox()
        w._gain.setRange(1e-6, 1e6)
        w._gain.setValue(1.0)
        w._gain.setDecimals(4)
        w._step = QtWidgets.QSpinBox()
        w._step.setRange(1, 100_000)
        w._step.setValue(1)
        w._per_page = QtWidgets.QSpinBox()
        w._per_page.setRange(0, 1_000_000)
        w._per_page.setValue(0)
        w._per_page.setSpecialValueText("all")
        form.addRow("Order by:", w._order_by)
        form.addRow("Descending:", w._descending)
        form.addRow("Hue:", w._hue)
        form.addRow("Gain:", w._gain)
        form.addRow("Step (every Nth):", w._step)
        form.addRow("Traces per page:", w._per_page)
        w._hint_text = "All traces on one axis, EEG-style offsets."
        return w

    def _build_heatmap_form(self, da) -> QtWidgets.QWidget:
        w, form = self._make_form()
        w._split_by = _combo(_splittable_names(da))
        w._order_by = _combo(_orderable_coords(da))
        w._descending = QtWidgets.QCheckBox()
        w._cmap = QtWidgets.QComboBox()
        w._cmap.setEditable(True)
        w._cmap.addItems(_CMAP_CHOICES)
        w._vmin = QtWidgets.QLineEdit()
        w._vmin.setPlaceholderText("auto (1st pctl)")
        w._vmax = QtWidgets.QLineEdit()
        w._vmax.setPlaceholderText("auto (99th pctl)")
        w._decim = _combo(["peak", "mean"], none_option=False)
        form.addRow("Split by:", w._split_by)
        form.addRow("Order rows by:", w._order_by)
        form.addRow("Descending:", w._descending)
        form.addRow("Colormap:", w._cmap)
        form.addRow("vmin:", w._vmin)
        form.addRow("vmax:", w._vmax)
        form.addRow("Decimation:", w._decim)
        needs_split = da.ndim > 2
        w._hint_text = (
            "3-D+ array: pick a Split-by to reduce each subplot to 2-D."
            if needs_split
            else "One row per value of the non-time dimension."
        )
        return w

    def _build_raster_form(self, df) -> QtWidgets.QWidget:
        w, form = self._make_form()
        cols = list(df.columns)
        numeric = _numeric_columns(df)
        time_default = next(
            (c for c in numeric if c.lower() in ("time", "t", "t_sec", "peak_time")),
            numeric[0] if numeric else None,
        )
        w._time_col = _combo(numeric, none_option=False, current=time_default)
        w._order_by = _combo(cols, none_option=False)
        w._split_by = _combo(cols)
        w._alpha_by = _combo(numeric)
        w._hue = _combo(cols)
        form.addRow("Time column:", w._time_col)
        form.addRow("Order by (row id):", w._order_by)
        form.addRow("Split by:", w._split_by)
        form.addRow("Alpha by:", w._alpha_by)
        form.addRow("Hue:", w._hue)
        w._hint_text = "One tick per event row; rows grouped by Order-by."
        return w

    # ------------------------------------------------------------------
    # Config construction + launch
    # ------------------------------------------------------------------

    def build_config(self):
        """Build the loupe Config the current form state describes."""
        from loupe.configs import HeatmapConfig, RasterConfig, TraceConfig

        label = self._view_combo.currentText()
        form = self._forms[label]

        if label == "Raster":
            return RasterConfig(
                data=self._obj,
                time_col=_combo_value(form._time_col),
                order_by=_combo_value(form._order_by),
                split_by=_combo_value(form._split_by),
                alpha_by=_combo_value(form._alpha_by),
                hue=_combo_value(form._hue),
            )

        da = self._current_dataarray()
        if da is None:
            raise ValueError("No data variable selected.")
        time_dim = self._selected_time_dim(da)
        fs = self._fs_spin.value() if self._fs_spin is not None else None
        da = _prepare_dataarray(da, time_dim, fs=fs)
        if da.name is None:
            da = da.rename(self._name)

        if label == "Traces (stacked)":
            color = form._color.text().strip() or None
            return TraceConfig(
                data=da,
                mode="stacked-subplots",
                order_by=_combo_value(form._order_by),
                descending=form._descending.isChecked(),
                hue=_combo_value(form._hue),
                color=color,
            )
        if label == "Traces (dense)":
            per_page = form._per_page.value() or None
            return TraceConfig(
                data=da,
                mode="dense",
                order_by=_combo_value(form._order_by),
                descending=form._descending.isChecked(),
                hue=_combo_value(form._hue),
                gain=form._gain.value(),
                step=form._step.value(),
                traces_per_page=per_page,
            )
        if label == "Heatmap":
            return HeatmapConfig(
                data=da,
                split_by=_combo_value(form._split_by),
                order_by=_combo_value(form._order_by),
                descending=form._descending.isChecked(),
                cmap=form._cmap.currentText().strip() or "magma",
                vmin=_float_or_none(form._vmin),
                vmax=_float_or_none(form._vmax),
                decim_method=form._decim.currentText(),
            )
        raise ValueError(f"Unknown view type {label!r}")

    def window_len(self) -> float:
        return float(self._window_len.value())

    def _launch(self) -> None:
        label = self._view_combo.currentText()
        form = self._forms.get(label)

        # Guard: absurd stacked-subplot counts are almost always a mistake.
        n_rows = getattr(form, "_n_rows", 0)
        if label == "Traces (stacked)" and n_rows > _STACKED_TRACE_WARN:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Many subplots",
                f"This will create {n_rows:,} stacked subplots, which may "
                f"be very slow. Consider the dense or heatmap view.\n\n"
                f"Launch anyway?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        try:
            cfg = self.build_config()
        except Exception as exc:
            self._show_error(f"Invalid configuration:\n{exc}")
            return

        from loupe.view import view

        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            win = view(cfg, window_len=self.window_len())
        except Exception as exc:
            self._show_error(f"Loupe launch failed:\n{exc}")
            return
        finally:
            if app is not None:
                app.restoreOverrideCursor()

        _keep_alive(_LAUNCHED_VIEWS, win)
        self.accept()

    def _show_error(self, message: str) -> None:
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        box.setWindowTitle("Open in loupe")
        box.setText(message)
        box.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        box.show()


__all__ = ["LoupeLaunchDialog", "open_launcher"]
