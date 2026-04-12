import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtWidgets

from loupe.app import LoupeApp, Series


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture()
def loupe_window(monkeypatch, qapp):
    original_set_config = pg.setConfigOptions

    def _safe_set_config(**kwargs):
        kwargs["useOpenGL"] = False
        return original_set_config(**kwargs)

    monkeypatch.setattr(pg, "setConfigOptions", _safe_set_config)

    series = [
        Series(
            name=f"trace_{i}",
            t=np.linspace(0.0, 60.0, 300, dtype=float),
            y=np.sin(np.linspace(0.0, 6.0, 300, dtype=float) + i),
        )
        for i in range(3)
    ]
    window = LoupeApp(xr_series=series, fixed_scale=True)
    qapp.processEvents()

    yield window

    for worker in (window._video_worker, window._video2_worker, window._video3_worker):
        QtCore.QMetaObject.invokeMethod(worker, "stop", QtCore.Qt.QueuedConnection)
    for thread in (window._video_thread, window._video2_thread, window._video3_thread):
        thread.quit()
        if not thread.wait(1000):
            thread.terminate()
            thread.wait(1000)
    window.close()
    qapp.processEvents()


def test_incremental_label_sync_preserves_unchanged_visuals(loupe_window, qapp):
    loupe_window.window_len = 40.0
    loupe_window.window_start = 0.0
    loupe_window._apply_x_range()
    qapp.processEvents()

    loupe_window.labels = [
        {"start": 0.0, "end": 10.0, "label": "NREM"},
        {"start": 10.0, "end": 20.0, "label": "REM"},
        {"start": 30.0, "end": 40.0, "label": "Wake"},
    ]
    loupe_window._finalize_label_change(force_rebuild=True, refresh_summary=False)
    qapp.processEvents()

    unchanged_key = (30.0, 40.0, "Wake")
    previous_visuals = dict(loupe_window._label_visuals)
    previous_hypnogram_visuals = dict(loupe_window._hypnogram_label_visuals)

    loupe_window._add_new_label(10.0, 20.0, "NREM")
    qapp.processEvents()

    assert (0.0, 10.0, "NREM") not in loupe_window._label_visuals
    assert (10.0, 20.0, "REM") not in loupe_window._label_visuals
    assert (0.0, 20.0, "NREM") in loupe_window._label_visuals
    assert loupe_window._label_visuals[unchanged_key] is previous_visuals[unchanged_key]
    assert (
        loupe_window._hypnogram_label_visuals[unchanged_key]
        is previous_hypnogram_visuals[unchanged_key]
    )


def test_plot_rebuild_recreates_label_visuals(loupe_window, qapp):
    loupe_window.window_len = 40.0
    loupe_window.window_start = 0.0
    loupe_window._apply_x_range()
    qapp.processEvents()

    loupe_window.labels = [
        {"start": 5.0, "end": 15.0, "label": "Wake"},
        {"start": 20.0, "end": 30.0, "label": "NREM"},
    ]
    loupe_window._finalize_label_change(force_rebuild=True, refresh_summary=False)
    qapp.processEvents()

    previous_visuals = dict(loupe_window._label_visuals)
    previous_hypnogram_visuals = dict(loupe_window._hypnogram_label_visuals)

    loupe_window._rebuild_all_plots()
    qapp.processEvents()

    assert set(loupe_window._label_visuals) == set(previous_visuals)
    for key, old_bundle in previous_visuals.items():
        assert loupe_window._label_visuals[key] is not old_bundle

    assert set(loupe_window._hypnogram_label_visuals) == set(previous_hypnogram_visuals)
    for key, old_region in previous_hypnogram_visuals.items():
        assert loupe_window._hypnogram_label_visuals[key] is not old_region
