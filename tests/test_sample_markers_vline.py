"""Tests for ``SampleMarkers(marker="vline")`` and per-config stacked markers.

A ``"vline"`` marker set draws a full-height vertical line at every flagged
sample instead of a symbol at the sample's value: one ``pg.PlotCurveItem`` with
``connect="pairs"`` per series (stacked) or per group (dense), spanning beyond
the visible y-range, excluded from auto-range and re-spanned when the y-range
changes. Stacked marker sets are anchored to the series block of the
``TraceConfig`` they belong to (``SampleMarkers.series_start``), so several
stacked configs may each carry their own markers in one window.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
import xarray as xr
from PySide6 import QtWidgets

import loupe.app as _loupe_app
from loupe import HeatmapConfig, SampleMarkers, TraceConfig, view

_STATE_DEFS = os.path.join(
    os.path.dirname(_loupe_app.__file__), "example_state_definitions.json"
)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pg.setConfigOptions(useOpenGL=False)
    yield app


def _da_1d(name: str, n: int = 500) -> xr.DataArray:
    t = np.linspace(0.0, 50.0, n)
    return xr.DataArray(np.sin(2 * np.pi * 0.3 * t), dims=("time",), coords={"time": t}, name=name)


def _mask_1d(da: xr.DataArray, idxs) -> xr.DataArray:
    arr = np.zeros(da.shape, dtype=bool)
    arr[list(idxs)] = True
    return xr.DataArray(arr, dims=da.dims, coords=da.coords)


def _dense_da(ids=("a", "b", "c"), depths=(2.0, 0.0, 1.0), n=500, name="lfp"):
    t = np.linspace(0.0, 50.0, n)
    rows = [np.sin(2 * np.pi * 0.3 * t) + 10.0 * k for k in range(len(ids))]
    return xr.DataArray(
        np.asarray(rows),
        dims=("channel", "time"),
        coords={"channel": list(ids), "depth": ("channel", list(depths)), "time": t},
        name=name,
    )


def _bool_like(da, true_cells):
    arr = np.zeros(da.shape, dtype=bool)
    chans = list(da.coords["channel"].values)
    for ch, i in true_cells:
        arr[chans.index(ch), i] = True
    return xr.DataArray(arr, dims=da.dims, coords=da.coords)


def _heat_da(n_rows=20, n=300, name="heat"):
    t = np.linspace(0.0, 30.0, n)
    Y = np.random.default_rng(0).normal(size=(n_rows, n))
    return xr.DataArray(Y, dims=("row", "time"), coords={"row": np.arange(n_rows), "time": t}, name=name)


# --------------------------------------------------------------- per-config


def test_two_stacked_carriers_each_get_their_own_markers():
    a, b = _da_1d("A"), _da_1d("B")
    w = view(
        [
            TraceConfig(a, sample_markers=[SampleMarkers("o", "#ff0000", _mask_1d(a, [50]))]),
            TraceConfig(b, sample_markers=[SampleMarkers("x", "lime", _mask_1d(b, [60]))]),
        ],
        state_definitions=_STATE_DEFS,
    )
    assert [m.series_start for m in w.sample_markers] == [0, 1]
    w._refresh_curves()
    t = a.time.values
    # series 0 draws set 0 only; series 1 draws set 1 only.
    x00, _ = w.sample_marker_scatters[0][0].getData()
    x01, _ = w.sample_marker_scatters[0][1].getData()
    x10, _ = w.sample_marker_scatters[1][0].getData()
    x11, _ = w.sample_marker_scatters[1][1].getData()
    np.testing.assert_allclose(x00, [t[50]])
    assert len(x01) == 0
    assert len(x10) == 0
    np.testing.assert_allclose(x11, [t[60]])


def test_stacked_carrier_with_plain_traceconfig_is_allowed():
    a, b = _da_1d("A"), _da_1d("B")
    w = view(
        [TraceConfig(a, sample_markers=[SampleMarkers("o", "#ff0000", _mask_1d(a, [50]))]), TraceConfig(b)],
        state_definitions=_STATE_DEFS,
    )
    w._refresh_curves()
    x1, _ = w.sample_marker_scatters[1][0].getData()
    assert len(x1) == 0


def test_stacked_carrier_still_rejects_heatmap():
    a = _da_1d("A")
    with pytest.raises(ValueError, match="cannot coexist"):
        view(
            [TraceConfig(a, sample_markers=[SampleMarkers("o", "#ff0000", _mask_1d(a, [50]))]), HeatmapConfig(_heat_da())],
            state_definitions=_STATE_DEFS,
        )


# --------------------------------------------------------------------- vline


def _vline_app(idxs=(50, 70), **marker_kwargs):
    a = _da_1d("A")
    w = view(
        TraceConfig(a, sample_markers=[SampleMarkers("vline", "#00ff00", _mask_1d(a, idxs), **marker_kwargs)]),
        state_definitions=_STATE_DEFS,
    )
    w._refresh_curves()
    return w, a


def test_vline_marker_draws_full_height_pairs():
    w, a = _vline_app()
    item = w.sample_marker_scatters[0][0]
    assert isinstance(item, pg.PlotCurveItem)
    assert item.opts["connect"] == "pairs"
    x, y = item.getData()
    t = a.time.values
    np.testing.assert_allclose(x, np.repeat(t[[50, 70]], 2))
    y0, y1 = w.plots[0].getViewBox().viewRange()[1]
    assert y[0] < y0 and y[1] > y1
    np.testing.assert_allclose(y[2:], y[:2])


def test_vline_marker_respans_when_y_range_changes():
    w, _ = _vline_app()
    item = w.sample_marker_scatters[0][0]
    vb = w.plots[0].getViewBox()
    vb.setYRange(-500.0, 500.0, padding=0)
    _, y = item.getData()
    assert y.min() < -500.0 and y.max() > 500.0


def test_vline_marker_does_not_inflate_autorange():
    w, _ = _vline_app()
    vb = w.plots[0].getViewBox()
    vb.enableAutoRange("y", True)
    vb.updateAutoRange()
    y0, y1 = vb.viewRange()[1]
    assert (y1 - y0) < 5.0  # a unit sine, not the ±span padding of the lines


def test_vline_defaults_and_live_restyle():
    w, _ = _vline_app()
    marker = w.sample_markers[0]
    assert marker.size == 1.0 and marker.alpha == 200
    marker.color = (0, 0, 255)
    marker.size = 3.0
    w._apply_sample_marker_style(0)
    pen = w.sample_marker_scatters[0][0].opts["pen"]
    assert pen.color().blue() == 255 and pen.color().red() == 0
    assert pen.widthF() == pytest.approx(3.0)


def test_vline_marker_empty_window_clears_lines():
    w, _ = _vline_app(idxs=(400,))  # t≈40 s, outside the initial 0–10 s window
    x, _ = w.sample_marker_scatters[0][0].getData()
    assert len(x) == 0


def test_vline_marker_in_dense_mode():
    da = _dense_da()
    w = view(
        TraceConfig(
            da, mode="dense", order_by="depth",
            sample_markers=[SampleMarkers("vline", "#ff00ff", _bool_like(da, [("a", 50), ("b", 60)]))],
        ),
        state_definitions=_STATE_DEFS,
    )
    w._refresh_dense_curves()
    item = w.dense_marker_scatters[0][0]
    assert isinstance(item, pg.PlotCurveItem)
    x, y = item.getData()
    t = da.time.values
    assert sorted(set(np.round(x, 9))) == sorted(set(np.round(t[[50, 60]], 9)))
    assert len(x) == 4 and len(y) == 4
    y0, y1 = item.getViewBox().viewRange()[1]
    assert y.min() < y0 and y.max() > y1
