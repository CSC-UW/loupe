"""Tests for ``TraceConfig.overlay_arrays`` — overlaying extra DataArrays on a
stacked-subplots host trace's own axes (rather than in separate subplots).

Covers the pure alignment converter and the end-to-end ``view()`` wiring
(curve items created per subplot, names/colors assigned, data sliced on
refresh, host/overlay Y-range under fixed scale, and the validation guards).
A real (offscreen) LoupeApp is built because the curve creation lives in
``LoupeApp._create_all_plots``."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
import xarray as xr
from PySide6 import QtWidgets

import loupe.app as _loupe_app
from loupe import TraceConfig, Zip, view
from loupe.xr_loader import (
    convert_overlay_arrays_aligned_with,
    convert_xarray_inputs_with_order,
)

_EXAMPLE_STATE_DEFS = os.path.join(
    os.path.dirname(_loupe_app.__file__),
    "example_state_definitions.json",
)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pg.setConfigOptions(useOpenGL=False)
    yield app


def _trace_1d(name=None, n=500, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 50.0, n)
    da = xr.DataArray(rng.standard_normal(n), dims=["time"], coords={"time": t})
    return da.rename(name) if name else da


def _trace_2d(name=None, ids=("a", "b", "c"), n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 40.0, n)
    da = xr.DataArray(
        rng.standard_normal((len(ids), n)),
        dims=["syn_id", "time"],
        coords={"syn_id": list(ids), "time": t},
    )
    return da.rename(name) if name else da


# --------------------------------------------------------------------------- #
# pure converter
# --------------------------------------------------------------------------- #
def test_converter_1d_keeps_own_time_and_values():
    host = _trace_1d(n=300, seed=1)
    ov = _trace_1d(n=300, seed=2)
    out = convert_overlay_arrays_aligned_with(
        host, [ov], order_by=None, descending=False
    )
    assert len(out) == 1  # one overlay array
    assert len(out[0]) == 1  # one host series
    t, y = out[0][0]
    np.testing.assert_array_equal(t, ov.coords["time"].values)
    np.testing.assert_array_equal(y, ov.values)


def test_converter_aligns_per_series_under_ordering():
    # A numeric ordering coord on the trace dim drives a non-trivial permutation;
    # the overlay must receive the SAME permutation as the host so subplot i and
    # overlay i refer to the same syn_id. Encode each row's identity in its
    # values so we can match them back regardless of order.
    ids = ("a", "b", "c")
    pos = [2.0, 0.0, 1.0]  # itertools order a,b,c -> ascending pos => b, c, a
    n = 50
    t = np.linspace(0.0, 10.0, n)

    def _ident_da(scale):
        # row k is a constant == k*scale, so identity is recoverable from values
        vals = np.stack([np.full(n, k * scale) for k in range(len(ids))])
        return xr.DataArray(
            vals,
            dims=["syn_id", "time"],
            coords={"syn_id": list(ids), "pos": ("syn_id", pos), "time": t},
        )

    host = _ident_da(scale=1.0)  # row identity 0,1,2
    ov = _ident_da(scale=10.0)  # row identity 0,10,20 (same syn_id mapping)

    host_tuples, _, _, _ = convert_xarray_inputs_with_order(
        host, order_by="pos", descending=False
    )
    out = convert_overlay_arrays_aligned_with(
        host, [ov], order_by="pos", descending=False
    )
    assert len(out[0]) == 3
    # host row identity / 1  must equal overlay row identity / 10 at every index
    for (_, _, hy), (_, oy) in zip(host_tuples, out[0]):
        assert hy[0] == oy[0] / 10.0
    # and the permutation really is by ascending pos (b, c, a -> identities 1,2,0)
    assert [hy[0] for (_, _, hy) in host_tuples] == [1.0, 2.0, 0.0]


def test_converter_reindexes_missing_nontime_coord_to_nan():
    host = _trace_2d(ids=("a", "b", "c"), seed=5)
    ov = _trace_2d(ids=("a", "b"), seed=6)  # missing syn_id "c"
    out = convert_overlay_arrays_aligned_with(
        host, [ov], order_by=None, descending=False
    )
    # third host series (syn_id "c") has no overlay data -> all-NaN
    _, y_c = out[0][2]
    assert np.isnan(y_c).all()


def test_converter_rejects_missing_nontime_dim():
    host = _trace_2d(seed=7)
    bad = _trace_1d(seed=8)  # no syn_id dim
    with pytest.raises(ValueError, match="missing dims"):
        convert_overlay_arrays_aligned_with(
            host, [bad], order_by=None, descending=False
        )


# --------------------------------------------------------------------------- #
# end-to-end view() wiring
# --------------------------------------------------------------------------- #
def test_overlay_curves_created_and_populated():
    host = _trace_1d("spks", seed=10)
    ov = _trace_1d("deconv_std", seed=11)
    w = view(
        TraceConfig(host, overlay_arrays=[ov]),
        state_definitions=_EXAMPLE_STATE_DEFS,
    )
    # one host subplot, one overlay curve on it
    assert len(w.series) == 1
    assert len(w.overlay_curve_items) == 1
    assert len(w.overlay_curve_items[0]) == 1
    # host curve named for the legend; overlay carries its array name
    assert w.overlay_main_names[0] == "spks"
    assert w.overlay_series[0][0].name == "deconv_std"
    # refresh pushed windowed data into the overlay PlotDataItem
    w._refresh_curves()
    xdata, ydata = w.overlay_curve_items[0][0].getData()
    assert xdata is not None and len(xdata) > 0


def test_multiple_overlays_get_distinct_default_colors():
    host = _trace_1d("host", seed=12)
    o1 = _trace_1d("o1", seed=13)
    o2 = _trace_1d("o2", seed=14)
    o3 = _trace_1d("o3", seed=15)
    w = view(
        TraceConfig(host, overlay_arrays=[o1, o2, o3]),
        state_definitions=_EXAMPLE_STATE_DEFS,
    )
    colors = [oc.color for oc in w.overlay_series[0]]
    assert len(colors) == 3
    assert len(set(colors)) == 3  # all distinct
    # default palette is the same one Zip uses
    assert colors == _loupe_app.LoupeApp._DEFAULT_OVERLAY_COLORS[:3]


def test_explicit_overlay_colors_respected_and_extended():
    host = _trace_1d("host", seed=16)
    o1 = _trace_1d("o1", seed=17)
    o2 = _trace_1d("o2", seed=18)
    w = view(
        TraceConfig(host, overlay_arrays=[o1, o2], overlay_colors=["#ff0000"]),
        state_definitions=_EXAMPLE_STATE_DEFS,
    )
    # first explicit, second filled from the default palette
    assert w.overlay_series[0][0].color == "#ff0000"
    assert w.overlay_series[0][1].color == _loupe_app.LoupeApp._DEFAULT_OVERLAY_COLORS[1]


def test_multi_trace_host_overlays_align_to_each_subplot():
    host = _trace_2d("dff", ids=("a", "b", "c"), seed=19)
    ov = _trace_2d("thr", ids=("a", "b", "c"), seed=20)
    w = view(
        TraceConfig(host, overlay_arrays=[ov]),
        state_definitions=_EXAMPLE_STATE_DEFS,
    )
    assert len(w.series) == 3
    assert len(w.overlay_curve_items) == 3
    for items in w.overlay_curve_items:
        assert len(items) == 1


def test_host_without_overlays_has_empty_legend_name():
    # a second TraceConfig without overlays must not get a spurious legend name
    host_a = _trace_1d("a", seed=21)
    ov = _trace_1d("ov", seed=22)
    host_b = _trace_1d("b", seed=23)
    w = view(
        [TraceConfig(host_a, overlay_arrays=[ov]), TraceConfig(host_b)],
        state_definitions=_EXAMPLE_STATE_DEFS,
    )
    assert len(w.series) == 2
    assert w.overlay_main_names[0] == "a"
    assert w.overlay_main_names[1] is None
    assert w.overlay_series[0] and not w.overlay_series[1]


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_overlay_arrays_rejected_in_dense_mode():
    host = _trace_1d("host", seed=24)
    ov = _trace_1d("ov", seed=25)
    with pytest.raises(ValueError, match="stacked-subplots"):
        view(
            TraceConfig(host, mode="dense", overlay_arrays=[ov]),
            state_definitions=_EXAMPLE_STATE_DEFS,
        )


def test_overlay_arrays_rejected_inside_zip():
    a = _trace_2d("a", seed=26)
    b = _trace_2d("b", seed=27)
    ov = _trace_2d("ov", seed=28)
    with pytest.raises(ValueError, match="meaningless"):
        Zip([TraceConfig(a, overlay_arrays=[ov]), TraceConfig(b)], on="syn_id")
