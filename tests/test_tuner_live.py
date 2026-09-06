"""End-to-end Tuner tests — binding capture in :func:`loupe.view` and the live
recompute path in :class:`loupe.app.LoupeApp`.

Real (offscreen) windows are built because the binding capture, the
``_apply_binding`` / ``_flush_tuner`` update path, the Y-refit, and the dock
auto-show all live in / around ``LoupeApp``. The debounce timer is bypassed by
calling ``_flush_tuner`` directly (it is the timer's slot).
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
import xarray as xr
from PySide6 import QtWidgets

import loupe.app as _loupe_app
from loupe import Param, SampleMarkers, TraceConfig, tunable, view

_EXAMPLE_STATE_DEFS = os.path.join(
    os.path.dirname(_loupe_app.__file__),
    "example_state_definitions.json",
)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pg.setConfigOptions(useOpenGL=False)
    yield app


def _ls(ids=(10, 11, 12), n=1500, seed=0):
    """A synthetic ``(syn_id, time)`` trace with a seconds time coord."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 1000.0
    da = xr.DataArray(
        rng.standard_normal((len(ids), n)).astype("float32"),
        dims=["syn_id", "time"],
        coords={"syn_id": list(ids), "time": t},
        name="ls",
    )
    return da


def _scale(da, k=1.0):
    """Pure, shape-preserving stand-in for a tunable transform."""
    return da * float(k)


def _marker_after(da, t0=0.0):
    """Boolean marker mask for the portion of a trace at/after ``t0``."""
    return xr.DataArray(
        np.broadcast_to(da.time.values >= float(t0), da.shape),
        dims=da.dims,
        coords=da.coords,
    )


def _view(cfg):
    return view(cfg, state_definitions=_EXAMPLE_STATE_DEFS)


# --------------------------------------------------------------------------- #
# binding capture
# --------------------------------------------------------------------------- #
def test_overlay_tunable_captures_binding_and_param():
    ls = _ls()
    tau = Param(0.1, 0.01, 1.0, name="tau")
    w = _view(TraceConfig(data=ls, overlay_arrays=[tunable(_scale, ls, k=tau)]))
    try:
        assert len(w._tuner_bindings) == 1
        b = w._tuner_bindings[0]
        assert b.kind == "trace_overlay" and b.overlay_k == 0
        assert w._tuner_params == [tau]
    finally:
        w.close()


def test_host_tunable_captures_trace_stacked_binding():
    ls = _ls()
    k = Param(1.0, 0.1, 4.0, name="k")
    w = _view(TraceConfig(data=tunable(_scale, ls, k=k)))
    try:
        assert len(w._tuner_bindings) == 1
        assert w._tuner_bindings[0].kind == "trace_stacked"
        assert w._tuner_params == [k]
    finally:
        w.close()


def test_no_tunable_means_no_bindings():
    ls = _ls()
    w = _view(TraceConfig(data=ls, overlay_arrays=[ls * 2.0]))
    try:
        assert w._tuner_bindings == []
        assert w._tuner_params == []
    finally:
        w.close()


def test_tunable_sample_marker_captures_binding_and_recomputes():
    ls = _ls(ids=(10,), n=200)
    t0 = Param(0.05, 0.0, 0.19, name="marker_t0")
    w = _view(TraceConfig(
        data=ls,
        sample_markers=[
            SampleMarkers(
                marker="o",
                color="#ff00aa",
                bool_array=tunable(_marker_after, ls, t0=t0),
            )
        ],
    ))
    try:
        assert len(w._tuner_bindings) == 1
        binding = w._tuner_bindings[0]
        assert binding.kind == "trace_marker"
        assert binding.marker_k == 0
        assert w._tuner_params == [t0]
        before = np.array(w.sample_markers[0].bool_per_series[0], copy=True)

        t0.value = 0.15
        w._on_tuner_param_changed(t0)
        w._flush_tuner()

        after = w.sample_markers[0].bool_per_series[0]
        assert after.sum() < before.sum()
        x, _ = w.sample_marker_scatters[0][0].getData()
        assert x is not None and len(x) == int(after.sum())
    finally:
        w.close()


# --------------------------------------------------------------------------- #
# live recompute
# --------------------------------------------------------------------------- #
def test_overlay_recompute_rescales_curve():
    ls = _ls()
    k = Param(1.0, 0.1, 8.0, name="k")
    w = _view(TraceConfig(data=ls, overlay_arrays=[tunable(_scale, ls, k=k)]))
    try:
        before = np.array(w.overlay_series[0][0].y, copy=True)
        k.value = 3.0
        w._on_tuner_param_changed(k)
        assert w._tuner_refresh_timer.isActive()  # debounce armed
        w._flush_tuner()
        after = w.overlay_series[0][0].y
        np.testing.assert_allclose(after, before * 3.0, rtol=1e-4)
        # the pyqtgraph item received data without error
        xd, yd = w.overlay_curve_items[0][0].getData()
        assert yd is not None and len(yd) > 0
    finally:
        w.close()


def test_host_recompute_rescales_each_series():
    ls = _ls()
    k = Param(1.0, 0.1, 4.0, name="k")
    w = _view(TraceConfig(data=tunable(_scale, ls, k=k)))
    try:
        before = [np.array(s.y, copy=True) for s in w.series]
        k.value = 2.0
        w._on_tuner_param_changed(k)
        w._flush_tuner()
        for s, b in zip(w.series, before):
            np.testing.assert_allclose(s.y, b * 2.0, rtol=1e-4)
    finally:
        w.close()


def test_only_dirty_bindings_recompute():
    # Two independent params; moving one must not recompute the other's overlay.
    ls = _ls()
    k1 = Param(1.0, 0.1, 8.0, name="k1")
    k2 = Param(1.0, 0.1, 8.0, name="k2")
    w = _view(TraceConfig(
        data=ls,
        overlay_arrays=[tunable(_scale, ls, k=k1), tunable(_scale, ls, k=k2)],
    ))
    try:
        ov1_before = np.array(w.overlay_series[0][0].y, copy=True)
        ov2_before = np.array(w.overlay_series[0][1].y, copy=True)
        k1.value = 5.0
        w._on_tuner_param_changed(k1)
        w._flush_tuner()
        np.testing.assert_allclose(w.overlay_series[0][0].y, ov1_before * 5.0, rtol=1e-4)
        np.testing.assert_allclose(w.overlay_series[0][1].y, ov2_before, rtol=1e-4)
    finally:
        w.close()


def test_binding_slice_offsets_correct_with_multiple_configs():
    # A non-tunable config precedes a tunable one; the binding must address the
    # SECOND config's contiguous host-series block, not index 0.
    a = _ls(ids=(1, 2), seed=1)            # series 0, 1 (no overlay)
    b = _ls(ids=(10, 11, 12), seed=2)      # series 2, 3, 4 (tunable overlay)
    k = Param(1.0, 0.1, 8.0, name="k")
    w = _view([
        TraceConfig(data=a),
        TraceConfig(data=b, overlay_arrays=[tunable(_scale, b, k=k)]),
    ])
    try:
        assert len(w.series) == 5
        binding = w._tuner_bindings[0]
        assert binding.kind == "trace_overlay"
        assert binding.overlay_host_slice == slice(2, 5)
        # only the second config's subplots carry an overlay
        assert all(len(w.overlay_series[i]) == 0 for i in (0, 1))
        assert all(len(w.overlay_series[i]) == 1 for i in (2, 3, 4))
        before = np.array(w.overlay_series[2][0].y, copy=True)
        k.value = 4.0
        w._on_tuner_param_changed(k)
        w._flush_tuner()
        np.testing.assert_allclose(w.overlay_series[2][0].y, before * 4.0, rtol=1e-4)
    finally:
        w.close()


def test_bare_lambda_global_param_discovered_and_tuned():
    ls = _ls()
    tau = Param(0.2, 0.01, 1.0, name="tau_g")
    w = _view(TraceConfig(data=ls, overlay_arrays=[lambda: _scale(ls, k=tau.value)]))
    try:
        assert w._tuner_params == [tau]
        before = np.array(w.overlay_series[0][0].y, copy=True)
        tau.value = 0.4
        w._on_tuner_param_changed(tau)
        w._flush_tuner()
        np.testing.assert_allclose(w.overlay_series[0][0].y, before * 2.0, rtol=1e-4)
    finally:
        w.close()


def test_view_preserved_by_default_on_amplitude_growth():
    ls = _ls()
    k = Param(1.0, 0.1, 10.0, name="k")  # overlay = host * k
    w = _view(TraceConfig(data=ls, overlay_arrays=[tunable(_scale, ls, k=k)]))
    try:
        assert w.fixed_scale and w.tuner_preserve_view  # defaults
        x0, y0 = (tuple(r) for r in w.plots[0].getViewBox().viewRange())
        k.value = 8.0
        w._on_tuner_param_changed(k)
        w._flush_tuner()
        x1, y1 = (tuple(r) for r in w.plots[0].getViewBox().viewRange())
        # Default: tuning must NOT move the view, even as the overlay grows 8x.
        assert x1 == pytest.approx(x0) and y1 == pytest.approx(y0)
    finally:
        w.close()


def test_yrange_refits_on_amplitude_growth_when_preserve_view_off():
    ls = _ls()
    k = Param(1.0, 0.1, 10.0, name="k")  # overlay = host * k
    w = view(
        TraceConfig(data=ls, overlay_arrays=[tunable(_scale, ls, k=k)]),
        state_definitions=_EXAMPLE_STATE_DEFS,
        tuner_preserve_view=False,
    )
    try:
        assert w.fixed_scale and not w.tuner_preserve_view
        lo0, hi0 = w.plots[0].getViewBox().viewRange()[1]
        k.value = 8.0
        w._on_tuner_param_changed(k)
        w._flush_tuner()
        lo1, hi1 = w.plots[0].getViewBox().viewRange()[1]
        # Legacy behaviour: the 8x overlay widens the fixed Y-range substantially.
        assert (hi1 - lo1) > 2.0 * (hi0 - lo0)
    finally:
        w.close()


def test_overlay_symbol_renders_points_and_survives_recompute():
    ls = _ls(ids=(10,), n=400)
    k = Param(1.0, 0.1, 8.0, name="k")
    w = _view(TraceConfig(
        data=ls,
        overlay_arrays=[tunable(_scale, ls, k=k)],
        overlay_symbols=["o"],
        overlay_symbol_sizes=[9.0],
    ))
    try:
        item = w.overlay_curve_items[0][0]
        # The overlay is a point cloud: symbol set, no connecting line.
        assert item.opts.get("symbol") == "o"
        assert item.opts.get("pen") is None
        before = np.array(w.overlay_series[0][0].y, copy=True)
        k.value = 3.0
        w._on_tuner_param_changed(k)
        w._flush_tuner()
        np.testing.assert_allclose(
            w.overlay_series[0][0].y, before * 3.0, rtol=1e-4
        )
        # symbol styling is preserved across the live recompute
        assert item.opts.get("symbol") == "o"
    finally:
        w.close()


# --------------------------------------------------------------------------- #
# structural-change guard
# --------------------------------------------------------------------------- #
def test_trace_count_change_is_skipped_not_crashing():
    ls = _ls(ids=(10, 11, 12))

    def _select(da, n=3):
        return da.isel(syn_id=slice(0, int(n)))

    from loupe import IntParam

    n = IntParam(3, 1, 3, name="n")
    w = _view(TraceConfig(data=tunable(_select, ls, n=n)))
    try:
        assert len(w.series) == 3
        before = [np.array(s.y, copy=True) for s in w.series]
        n.value = 2  # would drop a trace -> structural, must be skipped
        w._on_tuner_param_changed(n)
        w._flush_tuner()  # must not raise
        # series left untouched (skip, not a partial/garbled update)
        assert len(w.series) == 3
        for s, b in zip(w.series, before):
            np.testing.assert_allclose(s.y, b, rtol=1e-6)
    finally:
        w.close()


# --------------------------------------------------------------------------- #
# panel / menu wiring
# --------------------------------------------------------------------------- #
def test_dock_auto_shown_and_action_checked_when_params():
    ls = _ls()
    k = Param(1.0, 0.1, 4.0, name="k")
    w = _view(TraceConfig(data=ls, overlay_arrays=[tunable(_scale, ls, k=k)]))
    try:
        assert w._tuner_dock is not None
        assert w._tuner_action is not None
        assert w._tuner_action.isEnabled()
        assert w._tuner_action.isChecked()
        # one control row per param
        assert len(w._tuner_dock._sync_callbacks) == 1
    finally:
        w.close()


def test_action_disabled_and_no_dock_without_params():
    ls = _ls()
    w = _view(TraceConfig(data=ls))
    try:
        assert w._tuner_dock is None
        assert w._tuner_action is not None
        assert not w._tuner_action.isEnabled()
    finally:
        w.close()


def test_toggle_action_hides_and_shows_dock():
    ls = _ls()
    k = Param(1.0, 0.1, 4.0, name="k")
    w = _view(TraceConfig(data=ls, overlay_arrays=[tunable(_scale, ls, k=k)]))
    try:
        assert w._tuner_dock.isVisibleTo(w)
        w._toggle_tuner_panel(False)
        assert not w._tuner_dock.isVisibleTo(w)
        w._toggle_tuner_panel(True)
        assert w._tuner_dock.isVisibleTo(w)
    finally:
        w.close()


def test_reset_restores_defaults_and_recomputes():
    ls = _ls()
    k = Param(1.0, 0.1, 8.0, name="k")
    w = _view(TraceConfig(data=ls, overlay_arrays=[tunable(_scale, ls, k=k)]))
    try:
        before = np.array(w.overlay_series[0][0].y, copy=True)
        k.value = 4.0
        w._on_tuner_param_changed(k)
        w._flush_tuner()
        w._tuner_dock._reset_all()
        assert k.value == 1.0
        w._flush_tuner()
        np.testing.assert_allclose(w.overlay_series[0][0].y, before, rtol=1e-4)
    finally:
        w.close()


# --------------------------------------------------------------------------- #
# tunable interval labels
# --------------------------------------------------------------------------- #
def _on_labels(n=2, thr=0.0):
    """Pure stand-in for an interval detector: ``n`` ON periods above ``thr``."""
    import polars as pl

    starts = [0.1 + 0.3 * i for i in range(int(n))]
    return pl.DataFrame(
        {
            "start_s": starts,
            "end_s": [s + 0.1 for s in starts],
            "label": ["ON"] * len(starts),
        }
    )


def test_tunable_interval_labels_capture_binding_and_recompute():
    ls = _ls()
    n = Param(2, 0, 4, step=1, name="n")
    w = view(
        TraceConfig(data=ls),
        interval_labels=tunable(_on_labels, n=n),
        label_colors={"ON": "#F5A623"},
        state_definitions=_EXAMPLE_STATE_DEFS,
    )
    try:
        assert [b.kind for b in w._tuner_bindings] == ["interval_labels"]
        assert w._tuner_params == [n]
        assert len(w.interval_label_set) == 2
        assert [r.label for r in w.interval_label_set] == ["ON", "ON"]

        n.value = 4
        w._on_tuner_param_changed(n)
        w._flush_tuner()
        assert len(w.interval_label_set) == 4
        assert w._interval_label_starts.tolist() == pytest.approx([0.1, 0.4, 0.7, 1.0])

        n.value = 0
        w._on_tuner_param_changed(n)
        w._flush_tuner()
        assert len(w.interval_label_set) == 0
    finally:
        w.close()


def test_tunable_interval_labels_reject_writeback():
    ls = _ls()
    n = Param(2, 0, 4, step=1, name="n")
    with pytest.raises(ValueError, match="writeback"):
        view(
            TraceConfig(data=ls),
            interval_labels=tunable(_on_labels, n=n),
            interval_labels_writeback=True,
            state_definitions=_EXAMPLE_STATE_DEFS,
        )
