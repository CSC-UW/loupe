"""Tests for two aesthetic features:

1. Every subplot shares one left-axis width so the y-spines line up — across
   trace, raster, AND heatmap plots — and the width stays put across resizes
   (auto-fit at startup, then locked).
2. ``TraceConfig.add_bottom_spine`` draws a minimal bottom-boundary line on a
   trace subplot, pinned to the bottom of the y-range, skipped on the
   bottom-most subplot, and consuming no extra registry slots when disabled.

Both behaviours live inside ``LoupeApp`` plot construction, so we build real
(offscreen) windows and introspect them.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import polars as pl
import pyqtgraph as pg
import pytest
import xarray as xr
from PySide6 import QtWidgets

import loupe.app as _loupe_app
from loupe import HeatmapConfig, RasterConfig, TraceConfig, view

_STATE_DEFS = os.path.join(
    os.path.dirname(_loupe_app.__file__), "example_state_definitions.json"
)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pg.setConfigOptions(useOpenGL=False)
    yield app


def _trace_da(syn_ids, n=300, name="sig", scale=1.0):
    t = np.linspace(0.0, 30.0, n)
    rows = [np.sin(2 * np.pi * 0.5 * t) * scale + k for k in range(len(syn_ids))]
    return xr.DataArray(
        np.asarray(rows),
        dims=("syn_id", "time"),
        coords={"syn_id": list(syn_ids), "time": t},
        name=name,
    )


def _heat_da(n_rows=40, n=300, name="heat"):
    t = np.linspace(0.0, 30.0, n)
    Y = np.random.default_rng(0).normal(size=(n_rows, n))
    return xr.DataArray(
        Y, dims=("row", "time"),
        coords={"row": np.arange(n_rows), "time": t}, name=name,
    )


def _raster_df(n=80, seed=1):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "time": np.sort(rng.uniform(0.0, 30.0, n)),
        "source_id": rng.integers(0, 12, n).astype(np.int64),
    })


def _settle(qapp, w):
    w.resize(1400, 900)
    w.show()
    for _ in range(40):
        qapp.processEvents()
    w._align_left_axes()
    qapp.processEvents()


def _all_left_widths(w):
    all_plots = (
        list(w.plots)
        + list(w.dense_plots)
        + list(w.raster_plots)
        + list(w.heatmap_plots)
    )
    return [int(p.getAxis("left").width()) for p in all_plots]


# --------------------------------------------------------------------------
# Feature 1 — uniform left-axis width (auto-fit, then lock)
# --------------------------------------------------------------------------


def test_left_axis_width_uniform_across_all_plot_types(_qapp):
    """ts + heatmap + raster all share one locked left-axis width."""
    w = view(
        [
            TraceConfig(_trace_da([0, 1], scale=100.0, name="eeg")),
            HeatmapConfig(_heat_da(name="A1")),
            RasterConfig(_raster_df(), time_col="time", order_by="source_id"),
        ],
        state_definitions=_STATE_DEFS,
    )
    _settle(_qapp, w)
    widths = _all_left_widths(w)
    assert w._left_axis_width is not None, "width never locked"
    assert len(set(widths)) == 1, f"left-axis widths diverge: {widths}"
    assert widths[0] == int(w._left_axis_width)
    # heatmap is the last of the three plot groups and must be aligned too
    assert len(w.heatmap_plots) == 1
    w.close()


def test_left_axis_width_stable_across_resize(_qapp):
    w = view(
        [
            TraceConfig(_trace_da([0, 1], scale=100.0, name="eeg")),
            HeatmapConfig(_heat_da(name="A1")),
        ],
        state_definitions=_STATE_DEFS,
    )
    _settle(_qapp, w)
    before = _all_left_widths(w)
    w.resize(900, 700)
    for _ in range(20):
        _qapp.processEvents()
    after = _all_left_widths(w)
    assert len(set(after)) == 1, after
    # locked: same value before and after resize (no jitter)
    assert after[0] == before[0], (before, after)
    w.close()


# --------------------------------------------------------------------------
# Feature 2 — add_bottom_spine
# --------------------------------------------------------------------------


def test_bottom_spine_created_only_for_flagged_non_last(_qapp):
    w = view(
        [
            TraceConfig(_trace_da([0, 1], scale=100.0, name="eeg"), add_bottom_spine=True),
            TraceConfig(_trace_da([0], scale=1.5, name="soma")),  # no spine
            HeatmapConfig(_heat_da(name="A1")),  # makes the ts traces non-last
        ],
        state_definitions=_STATE_DEFS,
    )
    _settle(_qapp, w)
    assert w.series_bottom_spine == [True, True, False]
    assert len(w.plot_bottom_spines) == len(w.series) == 3
    assert isinstance(w.plot_bottom_spines[0], pg.InfiniteLine)
    assert isinstance(w.plot_bottom_spines[1], pg.InfiniteLine)
    assert w.plot_bottom_spines[2] is None
    w.close()


def test_bottom_spine_pinned_to_y_min(_qapp):
    w = view(
        [
            TraceConfig(_trace_da([0], scale=100.0, name="eeg"), add_bottom_spine=True),
            HeatmapConfig(_heat_da(name="A1")),
        ],
        state_definitions=_STATE_DEFS,
    )
    _settle(_qapp, w)
    sp = w.plot_bottom_spines[0]
    assert isinstance(sp, pg.InfiniteLine)
    assert sp.angle == 0  # horizontal
    y0, y1 = w.plots[0].getViewBox().viewRange()[1]
    pos = sp.value()
    # within ~1px of the bottom edge, never below it
    assert y0 - 1e-6 <= pos <= y0 + 0.05 * (y1 - y0) + 1e-6, (pos, y0, y1)
    w.close()


def test_bottom_spine_skipped_on_bottom_most_subplot(_qapp):
    # The flagged trace is the only/last subplot, so the line must be skipped
    # (the real time axis already marks the bottom there).
    w = view(
        TraceConfig(_trace_da([0], name="only"), add_bottom_spine=True),
        state_definitions=_STATE_DEFS,
    )
    _settle(_qapp, w)
    assert w.series_bottom_spine == [True]
    assert w.plot_bottom_spines == [None]
    w.close()


def test_no_spines_when_flag_absent(_qapp):
    w = view(
        [
            TraceConfig(_trace_da([0, 1], scale=100.0, name="eeg")),
            HeatmapConfig(_heat_da(name="A1")),
        ],
        state_definitions=_STATE_DEFS,
    )
    _settle(_qapp, w)
    assert all(x is None for x in w.plot_bottom_spines)
    w.close()
