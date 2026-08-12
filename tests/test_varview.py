import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import polars as pl
import pytest
import xarray as xr
from PySide6 import QtWidgets

from loupe.configs import HeatmapConfig, RasterConfig, TraceConfig
from loupe.loupeDF import DataFrameViewer
from loupe.varview import (
    DisplayPubTee,
    VarViewWindow,
    scan_namespace,
    varview,
)
from loupe.varview_launcher import (
    LoupeLaunchDialog,
    _ndarray_to_dataarray,
    _prepare_dataarray,
)


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _sample_ns():
    da = xr.DataArray(
        np.random.randn(4, 50),
        dims=["syn", "time"],
        coords={
            "syn": [1, 2, 3, 4],
            "time": np.arange(50) / 10.0,
            "depth": ("syn", [10.0, 20.0, 30.0, 40.0]),
        },
        name="dff",
    )
    return {
        "df": pl.DataFrame({"time": [0.1, 0.5], "syn_id": ["a", "b"]}),
        "da": da,
        "ds": da.to_dataset(name="dff"),
        "arr": np.arange(12.0).reshape(3, 4),
        "x": 42,
        "label": "hello world",
        "items": [1, 2, 3],
        "_hidden": 1,
        "In": ["cell"],
        "Out": {},
    }


# ---------------------------------------------------------------------------
# Namespace scanning
# ---------------------------------------------------------------------------


def test_scan_namespace_classifies_and_filters():
    import types

    ns = _sample_ns()
    ns["amodule"] = types.ModuleType("amodule")
    ns["afunc"] = lambda: None

    rows = {r.name: r for r in scan_namespace(ns)}

    assert "_hidden" not in rows
    assert "In" not in rows and "Out" not in rows
    assert "amodule" not in rows and "afunc" not in rows

    assert rows["df"].kind == "polars_df"
    assert "2 rows" in rows["df"].info
    assert rows["da"].kind == "xr_dataarray"
    assert "syn: 4" in rows["da"].info
    assert rows["ds"].kind == "xr_dataset"
    assert rows["arr"].kind == "ndarray"
    assert rows["arr"].size_bytes == 96
    assert rows["x"].kind == "scalar"
    assert rows["label"].kind == "str"
    assert rows["items"].kind == "sequence"


def test_scan_namespace_include_all_keeps_functions_and_modules():
    import types

    ns = {"amodule": types.ModuleType("amodule"), "afunc": lambda: None}
    rows = {r.name for r in scan_namespace(ns, include_all=True)}
    assert rows == {"amodule", "afunc"}


def test_scan_survives_hostile_objects():
    class Evil:
        def __repr__(self):
            raise RuntimeError("no repr for you")

        @property
        def shape(self):
            raise RuntimeError("nope")

    rows = {r.name: r for r in scan_namespace({"evil": Evil()})}
    assert rows["evil"].type_name == "Evil"


# ---------------------------------------------------------------------------
# Window behavior
# ---------------------------------------------------------------------------


def test_window_lists_and_inspects_dataframe(qapp):
    ns = _sample_ns()
    w = VarViewWindow(lambda: ns)
    w.refresh()

    names = [r.name for r in w.model.rows()]
    assert "df" in names and "da" in names

    assert w.select_variable("df")
    # Selecting a polars frame embeds the virtualized DataFrameViewer.
    embedded = w.detail.findChild(DataFrameViewer)
    assert embedded is not None
    assert embedded.dataframe.height == 2
    w.close()


def test_window_inspects_xarray_as_text(qapp):
    ns = _sample_ns()
    w = VarViewWindow(lambda: ns)
    w.refresh()
    assert w.select_variable("da")
    text_widget = w.detail.findChild(QtWidgets.QPlainTextEdit)
    assert text_widget is not None
    assert "xarray.DataArray" in text_widget.toPlainText()
    w.close()


def test_window_refresh_tracks_namespace_changes(qapp):
    ns = _sample_ns()
    w = VarViewWindow(lambda: ns)
    w.refresh()
    assert "newvar" not in [r.name for r in w.model.rows()]

    ns["newvar"] = 3.14
    w.refresh()
    assert "newvar" in [r.name for r in w.model.rows()]

    # Inspector follows a changed object (token change → rebuild).
    w.select_variable("df")
    ns["df"] = pl.DataFrame({"time": [1.0, 2.0, 3.0], "syn_id": ["a", "b", "c"]})
    w.refresh()
    embedded = w.detail.findChild(DataFrameViewer)
    assert embedded is not None and embedded.dataframe.height == 3

    # Inspector clears when the variable disappears.
    del ns["df"]
    w.refresh()
    assert w.detail.current_name is None
    w.close()


def test_selection_survives_refresh(qapp):
    ns = _sample_ns()
    w = VarViewWindow(lambda: ns)
    w.refresh()
    w.select_variable("da")
    ns["another"] = 1
    w.refresh()
    assert w.detail.current_name == "da"
    w.close()


def test_ndarray_detail_has_stats(qapp):
    ns = {"arr": np.array([[1.0, 2.0], [3.0, np.nan]])}
    w = VarViewWindow(lambda: ns)
    w.refresh()
    w.select_variable("arr")
    text = w.detail.findChild(QtWidgets.QPlainTextEdit).toPlainText()
    assert "shape:  (2, 2)" in text
    assert "min:    1" in text
    assert "non-finite: 1" in text
    w.close()


# ---------------------------------------------------------------------------
# Plots gallery + display tee
# ---------------------------------------------------------------------------


def _tiny_png() -> bytes:
    from PySide6 import QtCore, QtGui

    img = QtGui.QImage(8, 6, QtGui.QImage.Format.Format_RGB32)
    img.fill(QtGui.QColor("red"))
    buf = QtCore.QBuffer()
    buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    return bytes(buf.data())


def test_gallery_receives_images(qapp):
    ns = {}
    w = VarViewWindow(lambda: ns)
    assert w.gallery.count() == 0
    w.add_gallery_image(_tiny_png(), "[1] test")
    assert w.gallery.count() == 1
    w.gallery.clear()
    assert w.gallery.count() == 0
    w.close()


def test_display_pub_tee_captures_and_forwards(qapp):
    published = []
    captured = []

    def original(*args, **kwargs):
        published.append((args, kwargs))
        return "orig-result"

    tee = DisplayPubTee(original, lambda b, s: captured.append((b, s)), lambda: "[7]")

    png = _tiny_png()
    result = tee(data={"image/png": png, "text/plain": "fig"})
    assert result == "orig-result"
    assert len(published) == 1
    assert captured == [(png, "[7]")]

    # Base64 string payloads (the on-the-wire form) decode too.
    import base64

    tee(data={"image/png": base64.b64encode(png).decode()})
    assert len(captured) == 2 and captured[1][0] == png

    # Non-image publishes pass through untouched.
    tee(data={"text/plain": "hello"})
    assert len(captured) == 2 and len(published) == 3

    # A hostile payload never breaks the original publish.
    tee(data={"image/png": 123.456})
    assert len(published) == 4


# ---------------------------------------------------------------------------
# varview() entry point
# ---------------------------------------------------------------------------


def test_varview_singleton_outside_ipython(qapp):
    import loupe.varview as vv

    ns = {"a": 1}
    w1 = varview(ns=ns)
    w2 = varview(ns=ns)
    assert w1 is w2
    w1.close()
    assert vv._WINDOW is None

    w3 = varview(ns=ns)
    assert w3 is not w1
    w3.close()


def test_varview_requires_ns_outside_ipython():
    with pytest.raises(RuntimeError, match="globals"):
        varview()


# ---------------------------------------------------------------------------
# Launcher: DataArray preparation
# ---------------------------------------------------------------------------


def test_prepare_dataarray_renames_and_fills_coords():
    da = xr.DataArray(np.zeros((3, 10)), dims=["syn", "samples"])
    out = _prepare_dataarray(da, "samples", fs=10.0)
    assert "time" in out.dims
    assert list(out.coords["time"].values[:3]) == [0.0, 0.1, 0.2]
    assert list(out.coords["syn"].values) == [0, 1, 2]


def test_prepare_dataarray_noop_when_ready():
    da = xr.DataArray(
        np.zeros((2, 5)),
        dims=["syn", "time"],
        coords={"syn": [7, 8], "time": np.arange(5.0)},
    )
    out = _prepare_dataarray(da, "time")
    assert out.identical(da)


def test_ndarray_wrapping_1d_and_2d():
    out1 = _ndarray_to_dataarray(np.arange(6.0), "sig", 0, 2.0)
    assert out1.dims == ("time",)
    assert out1.coords["time"].values[-1] == pytest.approx(2.5)

    arr = np.arange(12.0).reshape(3, 4)
    out2 = _ndarray_to_dataarray(arr, "sig", 1, 1.0)
    assert out2.dims == ("row", "time")
    assert out2.sizes == {"row": 3, "time": 4}

    # time on axis 0 → transposed so row comes first
    out3 = _ndarray_to_dataarray(arr, "sig", 0, 1.0)
    assert out3.dims == ("row", "time")
    assert out3.sizes == {"row": 4, "time": 3}


# ---------------------------------------------------------------------------
# Launcher: dialog → configs
# ---------------------------------------------------------------------------


def test_launch_dialog_dataarray_heatmap_config(qapp):
    ns = _sample_ns()
    dlg = LoupeLaunchDialog("da", ns["da"])

    types = [dlg._view_combo.itemText(i) for i in range(dlg._view_combo.count())]
    assert types == ["Traces (stacked)", "Traces (dense)", "Heatmap"]

    dlg._view_combo.setCurrentText("Heatmap")
    form = dlg._forms["Heatmap"]
    form._order_by.setCurrentText("depth")
    form._descending.setChecked(True)
    form._cmap.setCurrentText("viridis")
    form._vmin.setText("-1.5")

    cfg = dlg.build_config()
    assert isinstance(cfg, HeatmapConfig)
    assert cfg.order_by == "depth"
    assert cfg.descending is True
    assert cfg.cmap == "viridis"
    assert cfg.vmin == -1.5 and cfg.vmax is None
    assert cfg.data.name == "dff"
    dlg.close()


def test_launch_dialog_stacked_and_dense_configs(qapp):
    ns = _sample_ns()
    dlg = LoupeLaunchDialog("da", ns["da"])

    dlg._view_combo.setCurrentText("Traces (stacked)")
    form = dlg._forms["Traces (stacked)"]
    form._order_by.setCurrentText("depth")
    form._hue.setCurrentText("syn")
    cfg = dlg.build_config()
    assert isinstance(cfg, TraceConfig)
    assert cfg.mode == "stacked-subplots"
    assert cfg.order_by == "depth" and cfg.hue == "syn"

    dlg._view_combo.setCurrentText("Traces (dense)")
    form = dlg._forms["Traces (dense)"]
    form._gain.setValue(2.5)
    form._per_page.setValue(16)
    cfg = dlg.build_config()
    assert cfg.mode == "dense"
    assert cfg.gain == 2.5 and cfg.traces_per_page == 16
    dlg.close()


def test_launch_dialog_dataset_variable_picker(qapp):
    ns = _sample_ns()
    dlg = LoupeLaunchDialog("ds", ns["ds"])
    assert dlg._var_combo is not None
    assert dlg._var_combo.currentText() == "dff"
    cfg = dlg.build_config()
    assert isinstance(cfg, TraceConfig)
    assert cfg.data.name == "dff"
    dlg.close()


def test_launch_dialog_dataframe_raster_config(qapp):
    df = pl.DataFrame(
        {
            "time": [0.1, 0.5, 0.9],
            "syn_id": ["a", "b", "a"],
            "dmd": [1, 1, 2],
            "snr": [3.0, 4.0, 5.0],
        }
    )
    dlg = LoupeLaunchDialog("events", df)
    assert [dlg._view_combo.itemText(i) for i in range(dlg._view_combo.count())] == [
        "Raster"
    ]
    form = dlg._forms["Raster"]
    assert form._time_col.currentText() == "time"  # smart default
    form._order_by.setCurrentText("syn_id")
    form._split_by.setCurrentText("dmd")
    form._alpha_by.setCurrentText("snr")

    cfg = dlg.build_config()
    assert isinstance(cfg, RasterConfig)
    assert cfg.time_col == "time"
    assert cfg.order_by == "syn_id"
    assert cfg.split_by == "dmd"
    assert cfg.alpha_by == "snr"
    assert cfg.hue is None
    dlg.close()


def test_launch_dialog_ndarray_time_axis_and_fs(qapp):
    arr = np.random.randn(3, 100)
    dlg = LoupeLaunchDialog("arr", arr)
    assert dlg._axis_combo is not None
    dlg._fs_spin.setValue(200.0)

    dlg._view_combo.setCurrentText("Heatmap")
    cfg = dlg.build_config()
    assert isinstance(cfg, HeatmapConfig)
    assert cfg.data.dims == ("row", "time")
    assert cfg.data.sizes == {"row": 3, "time": 100}
    assert cfg.data.coords["time"].values[-1] == pytest.approx(99 / 200.0)
    dlg.close()


def test_launch_dialog_time_dim_selection_renames(qapp):
    da = xr.DataArray(
        np.zeros((5, 20)),
        dims=["cell", "samples"],
        coords={"cell": list("abcde")},
    )
    dlg = LoupeLaunchDialog("sig", da)
    dlg._time_combo.setCurrentText("samples")
    cfg = dlg.build_config()
    assert "time" in cfg.data.dims
    assert cfg.data.sizes["time"] == 20
    dlg.close()


def test_launch_dialog_rejects_unsupported_type(qapp):
    with pytest.raises(TypeError, match="Open in loupe supports"):
        LoupeLaunchDialog("x", object())


def test_launch_end_to_end_opens_loupe_app(qapp):
    """The full path: dialog → config → view() → live LoupeApp window."""
    import loupe.varview_launcher as launcher_mod
    from loupe.app import LoupeApp

    ns = _sample_ns()
    dlg = LoupeLaunchDialog("da", ns["da"])
    dlg._view_combo.setCurrentText("Heatmap")
    n_before = len(launcher_mod._LAUNCHED_VIEWS)

    dlg._launch()

    assert len(launcher_mod._LAUNCHED_VIEWS) == n_before + 1
    win = launcher_mod._LAUNCHED_VIEWS[-1]
    assert isinstance(win, LoupeApp)
    assert dlg.result() == QtWidgets.QDialog.DialogCode.Accepted
    win.close()
    dlg.close()
