"""Tests for ``TraceConfig.sample_markers`` in **dense** mode.

Dense markers reuse the same converter (``convert_event_arrays_aligned_with``)
and ``bool_array`` contract as stacked mode, but render as one aggregated
``pg.ScatterPlotItem`` per marker set per group, drawn at each trace's
*displayed* y (``(value - mean) * gain + offset``).

Covers the pure alignment converter, end-to-end ``view()`` wiring (one scatter
per marker set created and populated on refresh, at the right display
coordinates), gain tracking, step/hidden exclusion, and the relaxed validation
(dense markers are unrestricted; stacked guards still fire). A real (offscreen)
LoupeApp is built because the scatter creation lives in
``LoupeApp._create_all_plots``.
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
from loupe.xr_loader import (
    convert_event_arrays_aligned_with,
    convert_xarray_inputs_with_order,
)

_STATE_DEFS = os.path.join(
    os.path.dirname(_loupe_app.__file__), "example_state_definitions.json"
)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pg.setConfigOptions(useOpenGL=False)
    yield app


def _dense_da(ids=("a", "b", "c"), depths=(2.0, 0.0, 1.0), n=500, name="lfp"):
    """2-D (channel, time). Each channel's signal is its index plus a sine, so a
    channel is identifiable from its mean. ``depth`` drives a non-trivial sort."""
    t = np.linspace(0.0, 50.0, n)
    rows = [np.sin(2 * np.pi * 0.3 * t) + 10.0 * k for k in range(len(ids))]
    return xr.DataArray(
        np.asarray(rows),
        dims=("channel", "time"),
        coords={"channel": list(ids), "depth": ("channel", list(depths)), "time": t},
        name=name,
    )


def _bool_like(da, true_cells):
    """Bool DataArray matching *da*'s dims/coords; True at (channel, sample)."""
    arr = np.zeros(da.shape, dtype=bool)
    chans = list(da.coords["channel"].values)
    for ch, i in true_cells:
        arr[chans.index(ch), i] = True
    return xr.DataArray(arr, dims=da.dims, coords=da.coords)


def _heat_da(n_rows=20, n=300, name="heat"):
    t = np.linspace(0.0, 30.0, n)
    Y = np.random.default_rng(0).normal(size=(n_rows, n))
    return xr.DataArray(
        Y, dims=("row", "time"),
        coords={"row": np.arange(n_rows), "time": t}, name=name,
    )


# --------------------------------------------------------------------------- #
# pure converter
# --------------------------------------------------------------------------- #
def test_converter_aligns_markers_to_dense_series_under_ordering():
    da = _dense_da()  # depth a=2, b=0, c=1 -> ascending sort => b, c, a
    # one True on channel "a", sample 50
    bools = _bool_like(da, [("a", 50)])
    out = convert_event_arrays_aligned_with(
        da, [bools], order_by="depth", descending=False
    )
    assert len(out) == 1  # one marker set
    per_series = out[0]

    # The data converter produces series in the SAME sorted order; find which
    # series index corresponds to channel "a" via its trace label.
    _, _, trace_labels, _ = convert_xarray_inputs_with_order(
        da, order_by="depth", descending=False
    )
    a_idx = list(trace_labels).index("a")

    # Exactly the "a" series carries the True at sample 50; all others empty.
    for si, mask in enumerate(per_series):
        if si == a_idx:
            assert mask[50] and mask.sum() == 1
        else:
            assert not mask.any()


# --------------------------------------------------------------------------- #
# end-to-end view() wiring
# --------------------------------------------------------------------------- #
def test_dense_marker_scatter_created_per_set():
    da = _dense_da()
    w = view(
        TraceConfig(
            da, mode="dense", order_by="depth",
            sample_markers=[
                SampleMarkers(marker="o", color="#ff0000", bool_array=_bool_like(da, [("a", 50)])),
                SampleMarkers(marker="x", color="lime", bool_array=_bool_like(da, [("b", 60)])),
            ],
        ),
        state_definitions=_STATE_DEFS,
    )
    assert len(w.dense_groups) == 1
    assert len(w.dense_groups[0].sample_markers) == 2
    # one ScatterPlotItem per marker set, drawn above the curves (z=10)
    assert len(w.dense_marker_scatters) == 1
    scatters = w.dense_marker_scatters[0]
    assert len(scatters) == 2
    for sc in scatters:
        assert isinstance(sc, pg.ScatterPlotItem)
        assert sc.zValue() == 10


def test_dense_marker_refresh_places_point_at_displayed_y():
    da = _dense_da()
    idx = 50
    w = view(
        TraceConfig(
            da, mode="dense", order_by="depth",
            sample_markers=[
                SampleMarkers(marker="o", color="#ff0000", bool_array=_bool_like(da, [("a", idx)])),
            ],
        ),
        state_definitions=_STATE_DEFS,
    )
    group = w.dense_groups[0]
    # Locate the series for channel "a" via its trace label, plus its offset.
    means = w._dense_means[0]
    a_si = list(group.trace_labels).index("a")
    offsets = w._dense_offsets(0)
    a_offset = float(offsets[list(w._dense_visible_indices(0)).index(a_si)])

    # Window the whole series so the marker is in view, then refresh.
    w.window_start = float(da.coords["time"].values[0])
    w.window_len = float(da.coords["time"].values[-1] - da.coords["time"].values[0])
    w._refresh_dense_curves()

    x, y = w.dense_marker_scatters[0][0].getData()
    assert len(x) == 1
    t_vals = da.coords["time"].values
    raw = float(group.series[a_si].y[idx])
    expected_y = (raw - means[a_si]) * group.gain + a_offset
    assert x[0] == pytest.approx(t_vals[idx])
    assert y[0] == pytest.approx(expected_y)


def test_dense_marker_tracks_gain():
    da = _dense_da()
    idx = 50
    w = view(
        TraceConfig(
            da, mode="dense", order_by="depth",
            sample_markers=[
                SampleMarkers(marker="o", color="#ff0000", bool_array=_bool_like(da, [("a", idx)])),
            ],
        ),
        state_definitions=_STATE_DEFS,
    )
    w.window_start = float(da.coords["time"].values[0])
    w.window_len = float(da.coords["time"].values[-1] - da.coords["time"].values[0])
    w._refresh_dense_curves()
    _, y0 = w.dense_marker_scatters[0][0].getData()

    a_si = list(w.dense_groups[0].trace_labels).index("a")
    offsets = w._dense_offsets(0)
    a_offset = float(offsets[list(w._dense_visible_indices(0)).index(a_si)])

    w._adjust_dense_gain(2.0)  # calls _refresh_dense_curves internally
    _, y1 = w.dense_marker_scatters[0][0].getData()

    # displaced about the offset, so distance-from-offset doubles
    assert (y1[0] - a_offset) == pytest.approx(2.0 * (y0[0] - a_offset))


def test_dense_markers_exclude_stepped_and_hidden_traces():
    # No order_by -> itertools order a, b, c; offsets = arange.
    da = _dense_da(depths=(0.0, 0.0, 0.0))
    idx = 40
    w = view(
        TraceConfig(
            da, mode="dense", step=2,  # visible series indices: 0, 2 (skip 1)
            sample_markers=[
                SampleMarkers(
                    marker="o", color="#ff0000",
                    bool_array=_bool_like(da, [("a", idx), ("b", idx)]),
                ),
            ],
        ),
        state_definitions=_STATE_DEFS,
    )
    # a -> series 0 (visible), b -> series 1 (skipped by step=2)
    assert list(w._dense_visible_indices(0)) == [0, 2]
    w.window_start = float(da.coords["time"].values[0])
    w.window_len = float(da.coords["time"].values[-1] - da.coords["time"].values[0])
    w._refresh_dense_curves()
    x, _ = w.dense_marker_scatters[0][0].getData()
    assert len(x) == 1  # only channel "a"; "b" is excluded by step

    # Hiding "a" too leaves no markers.
    w.dense_groups[0].hidden_traces.add(0)
    w._refresh_dense_curves()
    x2, _ = w.dense_marker_scatters[0][0].getData()
    assert len(x2) == 0


# --------------------------------------------------------------------------- #
# validation (relaxed dense scope; stacked guards preserved)
# --------------------------------------------------------------------------- #
def test_two_dense_marker_carriers_allowed():
    a = _dense_da(name="A")
    b = _dense_da(name="B")
    w = view(
        [
            TraceConfig(a, mode="dense", order_by="depth",
                        sample_markers=[SampleMarkers("o", "#ff0000", _bool_like(a, [("a", 10)]))]),
            TraceConfig(b, mode="dense", order_by="depth",
                        sample_markers=[SampleMarkers("x", "lime", _bool_like(b, [("b", 10)]))]),
        ],
        state_definitions=_STATE_DEFS,
    )
    assert len(w.dense_groups) == 2
    assert all(g.sample_markers for g in w.dense_groups)


def test_dense_markers_coexist_with_heatmap_and_stacked():
    da = _dense_da()
    stk = _dense_da(name="stk")
    w = view(
        [
            TraceConfig(da, mode="dense", order_by="depth",
                        sample_markers=[SampleMarkers("o", "#ff0000", _bool_like(da, [("a", 10)]))]),
            TraceConfig(stk),  # plain stacked, no markers
            HeatmapConfig(_heat_da()),
        ],
        state_definitions=_STATE_DEFS,
    )
    assert len(w.dense_groups) == 1
    assert w.dense_groups[0].sample_markers


def test_two_stacked_marker_carriers_each_own_their_series_block():
    # Several stacked carriers per window are allowed; each marker set is
    # anchored to its own TraceConfig's contiguous series block.
    a = _dense_da(name="A")
    b = _dense_da(name="B")
    w = view(
        [
            TraceConfig(a, sample_markers=[SampleMarkers("o", "#ff0000", _bool_like(a, [("a", 10)]))]),
            TraceConfig(b, sample_markers=[SampleMarkers("x", "lime", _bool_like(b, [("b", 10)]))]),
        ],
        state_definitions=_STATE_DEFS,
    )
    assert [m.series_start for m in w.sample_markers] == [0, 3]
    assert [len(m.bool_per_series) for m in w.sample_markers] == [3, 3]


def test_stacked_marker_carrier_coexists_with_other_traceconfig():
    a = _dense_da(name="A")
    b = _dense_da(name="B")
    w = view(
        [
            TraceConfig(a, sample_markers=[SampleMarkers("o", "#ff0000", _bool_like(a, [("a", 10)]))]),
            TraceConfig(b, mode="dense"),  # a dense sibling is fine now
        ],
        state_definitions=_STATE_DEFS,
    )
    assert len(w.sample_markers) == 1
    assert len(w.dense_groups) == 1
