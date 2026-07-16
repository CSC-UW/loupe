"""Ctrl+Shift+wheel Y-scaling and Ctrl+drag trace selection.

The gesture plumbing lives on ``SelectableViewBox`` (signals) and
``LoupeApp`` (handlers); these tests drive the handlers directly (as the
signals would) on a real offscreen stacked-subplots window.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
import xarray as xr
from PySide6 import QtCore, QtWidgets

import loupe.app as _loupe_app
from loupe import TraceConfig, view
from loupe.series import Series

_EXAMPLE_STATE_DEFS = os.path.join(
    os.path.dirname(_loupe_app.__file__),
    "example_state_definitions.json",
)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pg.setConfigOptions(useOpenGL=False)
    yield app


def _stack(n_traces=3, n=1500):
    rng = np.random.default_rng(0)
    ids = list(range(10, 10 + n_traces))
    t = np.arange(n) / 1000.0
    return xr.DataArray(
        rng.standard_normal((n_traces, n)).astype("float32"),
        dims=["syn_id", "time"],
        coords={"syn_id": ids, "time": t},
        name="ls",
    )


def _built_window(n_traces=3):
    w = view(TraceConfig(data=_stack(n_traces)), state_definitions=_EXAMPLE_STATE_DEFS)
    w.resize(1000, 700)
    w.show()
    QtWidgets.QApplication.processEvents()
    return w


def _yspan(plt):
    lo, hi = plt.getViewBox().viewRange()[1]
    return hi - lo


def test_ctrl_shift_wheel_scales_hovered_plot_proportionally():
    w = _built_window()
    try:
        p = w.plots[0]
        w.hovered_plot = p
        h0 = _yspan(p)
        sib0 = _yspan(w.plots[1])  # a sibling's own baseline
        w._on_wheel_y_scale(120)  # one notch up -> magnify (shrink range ~15%)
        assert _yspan(p) == pytest.approx(0.85 * h0, rel=1e-3)
        w._on_wheel_y_scale(-120)  # symmetric restore
        assert _yspan(p) == pytest.approx(h0, rel=1e-3)
        # proportional to speed: 2x delta == the single-notch factor squared
        base = _yspan(p)
        w._on_wheel_y_scale(240)
        assert _yspan(p) == pytest.approx(0.85 * 0.85 * base, rel=1e-3)
        # untouched sibling stays put
        assert _yspan(w.plots[1]) == pytest.approx(sib0, rel=1e-3)
    finally:
        w.close()


def test_ctrl_drag_span_selects_and_scales_together():
    w = _built_window(n_traces=3)
    try:
        y_first = w.plots[0].getViewBox().sceneBoundingRect().center().y()
        y_last = w.plots[2].getViewBox().sceneBoundingRect().center().y()
        w._on_trace_select_start(y_first)
        w._on_trace_select_update(y_last)  # drag across the whole stack
        w._on_trace_select_finish()
        assert w._trace_selection == set(w.plots)

        before = [_yspan(p) for p in w.plots]
        w.hovered_plot = w.plots[1]  # hovering any selected plot scales them all
        w._on_wheel_y_scale(120)
        for p, b in zip(w.plots, before):
            assert _yspan(p) == pytest.approx(0.85 * b, rel=1e-3)
    finally:
        w.close()


def test_ctrl_click_toggles_single_and_escape_clears():
    w = _built_window()
    try:
        yc = w.plots[1].getViewBox().sceneBoundingRect().center().y()
        # click selects one
        w._on_trace_select_start(yc)
        w._on_trace_select_finish()
        assert w._trace_selection == {w.plots[1]}
        # click the same sole selection again -> toggles off
        w._on_trace_select_start(yc)
        w._on_trace_select_finish()
        assert w._trace_selection == set()
        # Esc clears a multi-selection
        w._set_trace_selection({w.plots[0], w.plots[2]})
        ev = QtGuiKey(QtCore.Qt.Key.Key_Escape)
        w.keyPressEvent(ev)
        assert w._trace_selection == set()
    finally:
        w.close()


def test_selection_reset_on_rebuild():
    w = _built_window()
    try:
        w._set_trace_selection(set(w.plots))
        assert w._trace_selection
        t = np.arange(500) / 1000.0
        w.set_series([Series("a", t, np.zeros(500)), Series("b", t, np.ones(500))])
        QtWidgets.QApplication.processEvents()
        assert w._trace_selection == set()
        assert len(w.plots) == 2
    finally:
        w.close()


class QtGuiKey:
    """Minimal stand-in for a QKeyEvent (only key()/text() are read)."""

    def __init__(self, key):
        self._key = key

    def key(self):
        return self._key

    def text(self):
        return ""
