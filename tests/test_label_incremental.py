import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtWidgets

from loupe.app import LoupeApp, Series
from loupe.state_config import load_state_config


def _test_state_config():
    pkg_dir = os.path.dirname(__import__("loupe").app.__file__)
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
    window = LoupeApp(
        xr_series=series, fixed_scale=True, state_config=_test_state_config()
    )
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

    loupe_window.label_set.add(0.0, 10.0, "NREM")
    loupe_window.label_set.add(10.0, 20.0, "REM")
    wake_id = loupe_window.label_set.add(30.0, 40.0, "Wake")
    loupe_window._finalize_label_change(force_rebuild=True, refresh_summary=False)
    qapp.processEvents()

    previous_visuals = dict(loupe_window._label_visuals)
    previous_hypnogram_visuals = dict(loupe_window._hypnogram_label_visuals)

    loupe_window._add_new_label(10.0, 20.0, "NREM")
    qapp.processEvents()

    # The (30, 40, Wake) row_id should still index the same visual bundle
    # because that row was untouched by the (10, 20) edit.
    assert wake_id in loupe_window._label_visuals
    assert loupe_window._label_visuals[wake_id] is previous_visuals[wake_id]
    assert (
        loupe_window._hypnogram_label_visuals[wake_id]
        is previous_hypnogram_visuals[wake_id]
    )

    # And the merged (0, 20) NREM label appears as a single row that overlaps
    # both edited intervals.
    rows = list(loupe_window.label_set)
    merged = [r for r in rows if r.label == "NREM"]
    assert len(merged) == 1
    assert merged[0].start == 0.0
    assert merged[0].end == 20.0


def test_plot_rebuild_recreates_label_visuals(loupe_window, qapp):
    loupe_window.window_len = 40.0
    loupe_window.window_start = 0.0
    loupe_window._apply_x_range()
    qapp.processEvents()

    loupe_window.label_set.add(5.0, 15.0, "Wake")
    loupe_window.label_set.add(20.0, 30.0, "NREM")
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
