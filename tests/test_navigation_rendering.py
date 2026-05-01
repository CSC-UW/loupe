import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtWidgets

import loupe.app as loupe_app
from loupe.app import (
    MATRIX_ALPHA_LEVEL_COUNT,
    LoupeApp,
    MatrixSeries,
    Series,
)
from loupe.state_config import load_state_config


def _test_state_config():
    pkg_dir = os.path.dirname(loupe_app.__file__)
    return load_state_config(
        path=os.path.join(pkg_dir, "example_state_definitions.json"),
        package_default=False,
    )


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture()
def loupe_factory(monkeypatch, qapp):
    original_set_config = pg.setConfigOptions

    def _safe_set_config(**kwargs):
        kwargs["useOpenGL"] = False
        return original_set_config(**kwargs)

    monkeypatch.setattr(pg, "setConfigOptions", _safe_set_config)
    windows: list[LoupeApp] = []

    def _make_window(
        *,
        n_series: int = 3,
        include_matrix: bool = False,
    ) -> LoupeApp:
        series = [
            Series(
                name=f"trace_{i}",
                t=np.linspace(0.0, 60.0, 400, dtype=float),
                y=np.sin(np.linspace(0.0, 8.0, 400, dtype=float) + i),
            )
            for i in range(n_series)
        ]
        matrix_series_list = None
        if include_matrix:
            timestamps = np.linspace(0.0, 60.0, 240, dtype=float)
            yvals = (np.arange(240) % 6).astype(int)
            alphas = np.linspace(0.1, 1.0, 240, dtype=float)
            matrix_series_list = [
                MatrixSeries(
                    name="events",
                    timestamps=timestamps,
                    yvals=yvals,
                    alphas=alphas,
                    color=(255, 255, 255),
                    n_rows=6,
                )
            ]

        window = LoupeApp(
            xr_series=series,
            fixed_scale=True,
            matrix_series_list=matrix_series_list,
            state_config=_test_state_config(),
        )
        qapp.processEvents()
        windows.append(window)
        return window

    yield _make_window

    for window in windows:
        for worker in (
            window._video_worker,
            window._video2_worker,
            window._video3_worker,
        ):
            QtCore.QMetaObject.invokeMethod(worker, "stop", QtCore.Qt.QueuedConnection)
        for thread in (
            window._video_thread,
            window._video2_thread,
            window._video3_thread,
        ):
            thread.quit()
            if not thread.wait(1000):
                thread.terminate()
                thread.wait(1000)
        window.close()
        qapp.processEvents()


def _reference_segment_for_window(t, y, t0, t1, max_pts=4000):
    if t1 <= t0:
        return np.empty(0), np.empty(0)

    i0 = max(0, np.searchsorted(t, t0) - 1)
    i1 = min(len(t), np.searchsorted(t, t1) + 1)
    ts = t[i0:i1]
    ys = y[i0:i1]
    n = len(ts)
    if n <= 2 or n <= max_pts:
        return ts, ys

    bins = max(1, max_pts // 2)
    edges = np.linspace(t0, t1, bins + 1)
    bi = np.clip(np.digitize(ts, edges) - 1, 0, bins - 1)

    order = np.argsort(bi, kind="mergesort")
    bi_s = bi[order]
    ts_s = ts[order]
    ys_s = ys[order]
    starts = np.searchsorted(bi_s, np.arange(bins), "left")
    ends = np.searchsorted(bi_s, np.arange(bins), "right")

    out_t = np.empty(2 * bins, dtype=float)
    out_y = np.empty(2 * bins, dtype=float)
    k = 0
    for b in range(bins):
        s, e = starts[b], ends[b]
        if s == e:
            tt = 0.5 * (edges[b] + edges[b + 1])
            out_t[k] = tt
            out_y[k] = np.nan
            k += 1
            out_t[k] = tt
            out_y[k] = np.nan
            k += 1
        else:
            yb = ys_s[s:e]
            tb = ts_s[s:e]
            ymin = float(np.nanmin(yb))
            ymax = float(np.nanmax(yb))
            tmid = float(tb[len(tb) // 2])
            out_t[k] = tmid
            out_y[k] = ymin
            k += 1
            out_t[k] = tmid
            out_y[k] = ymax
            k += 1

    return out_t[:k], out_y[:k]


def test_segment_for_window_matches_reference():
    t = np.linspace(0.0, 120.0, 50_000, dtype=float)
    y = np.sin(t * 3.0) + 0.1 * np.cos(t * 13.0)

    for t0, t1, max_pts in [
        (0.0, 10.0, 4_000),
        (5.0, 55.0, 4_000),
        (40.0, 120.0, 1_500),
    ]:
        got_t, got_y = loupe_app.segment_for_window(t, y, t0, t1, max_pts=max_pts)
        ref_t, ref_y = _reference_segment_for_window(t, y, t0, t1, max_pts=max_pts)

        assert np.array_equal(got_t, ref_t)
        assert np.array_equal(got_y, ref_y, equal_nan=True)


def test_refresh_curves_skips_hidden_traces(loupe_factory, monkeypatch, qapp):
    window = loupe_factory(n_series=4, include_matrix=False)
    window.trace_visible = [True, False, True, False]
    window._apply_trace_visibility()
    qapp.processEvents()

    updated_indices = []
    for idx, curve in enumerate(window.curves):
        orig_setData = curve.setData

        def make_tracker(i, orig):
            def tracking_setData(*args, **kwargs):
                updated_indices.append(i)
                return orig(*args, **kwargs)
            return tracking_setData

        monkeypatch.setattr(curve, "setData", make_tracker(idx, orig_setData))

    window._refresh_curves()
    qapp.processEvents()

    assert sorted(updated_indices) == [0, 2]


def test_matrix_refresh_reuses_items_and_skips_hidden(loupe_factory, monkeypatch, qapp):
    window = loupe_factory(n_series=1, include_matrix=True)
    assert len(window._matrix_line_items) == 1
    assert len(window._matrix_line_items[0]) == MATRIX_ALPHA_LEVEL_COUNT

    initial_line_items = list(window._matrix_line_items[0])
    window.window_start = 20.0
    window._refresh_matrix_plots()
    qapp.processEvents()

    assert window._matrix_line_items[0] == initial_line_items

    calls = []
    original = window._matrix_segment_for_window

    def counting_segment(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(window, "_matrix_segment_for_window", counting_segment)
    window.matrix_visible = [False]
    window._refresh_matrix_plots()
    qapp.processEvents()

    assert calls == []


def test_request_video_frame_suppresses_duplicate_nearest_indices(
    loupe_factory, monkeypatch
):
    window = loupe_factory(n_series=1, include_matrix=False)
    frame_times = np.array([0.0, 1.0, 2.0], dtype=float)
    calls: list[int] = []

    with monkeypatch.context() as m:
        original_invoke = QtCore.QMetaObject.invokeMethod

        def fake_invoke(target, member, connection_type, *args):
            if target is window._video_worker and member == "requestFrame":
                calls.append(window._video_requested_frame_idx)
                return True
            return original_invoke(target, member, connection_type, *args)

        m.setattr(QtCore.QMetaObject, "invokeMethod", fake_invoke)

        window._video_requested_frame_idx = None
        window._request_video_frame(
            frame_times=frame_times,
            worker=window._video_worker,
            requested_attr="_video_requested_frame_idx",
            t=0.01,
        )
        window._request_video_frame(
            frame_times=frame_times,
            worker=window._video_worker,
            requested_attr="_video_requested_frame_idx",
            t=0.02,
        )
        window._request_video_frame(
            frame_times=frame_times,
            worker=window._video_worker,
            requested_attr="_video_requested_frame_idx",
            t=1.01,
        )

    assert calls == [0, 1]


def test_window_label_visuals_only_materialize_current_window(loupe_factory, qapp):
    window = loupe_factory(n_series=2, include_matrix=True)
    window.window_len = 10.0
    window.window_start = 0.0
    wake_id = window.label_set.add(0.0, 5.0, "Wake")
    nrem_id = window.label_set.add(10.0, 15.0, "NREM")
    rem_id = window.label_set.add(40.0, 50.0, "REM")
    window._finalize_label_change(force_rebuild=True, refresh_summary=False)
    qapp.processEvents()

    assert set(window._hypnogram_label_visuals) == {wake_id, nrem_id, rem_id}
    assert set(window._label_visuals) == {wake_id}


def test_paging_swaps_window_label_visuals_without_rebuilding_hypnogram(
    loupe_factory, qapp
):
    window = loupe_factory(n_series=1, include_matrix=False)
    window.window_len = 10.0
    window.window_start = 0.0
    wake_id = window.label_set.add(0.0, 5.0, "Wake")
    nrem_id = window.label_set.add(12.0, 18.0, "NREM")
    window.label_set.add(24.0, 30.0, "REM")
    window._finalize_label_change(force_rebuild=True, refresh_summary=False)
    qapp.processEvents()

    previous_hypnogram_visuals = dict(window._hypnogram_label_visuals)

    assert set(window._label_visuals) == {wake_id}

    window._page(1)
    qapp.processEvents()

    assert set(window._label_visuals) == {nrem_id}
    assert set(window._hypnogram_label_visuals) == set(previous_hypnogram_visuals)
    for key, old_region in previous_hypnogram_visuals.items():
        assert window._hypnogram_label_visuals[key] is old_region


def test_visibility_changes_rebuild_window_label_visuals(loupe_factory, qapp):
    window = loupe_factory(n_series=1, include_matrix=True)
    window.window_len = 10.0
    window.window_start = 0.0
    wake_id = window.label_set.add(0.0, 5.0, "Wake")
    window._finalize_label_change(force_rebuild=True, refresh_summary=False)
    qapp.processEvents()

    bundle = window._label_visuals[wake_id]
    assert len(bundle.plot_regions) == 1
    assert len(bundle.matrix_regions) == 1

    window.trace_visible = [False]
    window._apply_trace_visibility()
    qapp.processEvents()

    bundle = window._label_visuals[wake_id]
    assert len(bundle.plot_regions) == 0
    assert len(bundle.matrix_regions) == 1

    window.matrix_visible = [False]
    window._apply_trace_visibility()
    qapp.processEvents()

    assert window._label_visuals == {}
    assert wake_id in window._hypnogram_label_visuals

    window.trace_visible = [True]
    window.matrix_visible = [True]
    window._apply_trace_visibility()
    qapp.processEvents()

    bundle = window._label_visuals[wake_id]
    assert len(bundle.plot_regions) == 1
    assert len(bundle.matrix_regions) == 1
