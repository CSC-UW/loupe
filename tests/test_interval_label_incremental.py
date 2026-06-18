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

    for slot in window.video_slots:
        QtCore.QMetaObject.invokeMethod(
            slot.worker, "stop", QtCore.Qt.QueuedConnection
        )
    for slot in window.video_slots:
        slot.thread.quit()
        if not slot.thread.wait(1000):
            slot.thread.terminate()
            slot.thread.wait(1000)
    window.close()
    qapp.processEvents()


def test_incremental_label_sync_preserves_unchanged_visuals(loupe_window, qapp):
    loupe_window.window_len = 40.0
    loupe_window.window_start = 0.0
    loupe_window._apply_x_range()
    qapp.processEvents()

    loupe_window.interval_label_set.add(0.0, 10.0, "NREM")
    loupe_window.interval_label_set.add(10.0, 20.0, "REM")
    wake_id = loupe_window.interval_label_set.add(30.0, 40.0, "Wake")
    loupe_window._finalize_interval_label_change(force_rebuild=True, refresh_summary=False)
    qapp.processEvents()

    previous_visuals = dict(loupe_window._interval_label_visuals)
    previous_hypnogram_visuals = dict(loupe_window._hypnogram_interval_label_visuals)

    loupe_window._add_new_interval_label(10.0, 20.0, "NREM")
    qapp.processEvents()

    # The (30, 40, Wake) row_id should still index the same visual bundle
    # because that row was untouched by the (10, 20) edit.
    assert wake_id in loupe_window._interval_label_visuals
    assert loupe_window._interval_label_visuals[wake_id] is previous_visuals[wake_id]
    assert (
        loupe_window._hypnogram_interval_label_visuals[wake_id]
        is previous_hypnogram_visuals[wake_id]
    )

    # And the merged (0, 20) NREM label appears as a single row that overlaps
    # both edited intervals.
    rows = list(loupe_window.interval_label_set)
    merged = [r for r in rows if r.label == "NREM"]
    assert len(merged) == 1
    assert merged[0].start == 0.0
    assert merged[0].end == 20.0

    # The surviving NREM row kept its row_id through merge_adjacent but its end
    # extended 10 -> 20; its in-place region must follow, not stay at [0, 10].
    merged_id = merged[0].row_id
    merged_bundle = loupe_window._interval_label_visuals[merged_id]
    assert (merged_bundle.start, merged_bundle.end) == (0.0, 20.0)
    for _i, reg in merged_bundle.plot_regions:
        assert reg.getRegion() == pytest.approx((0.0, 20.0))
    assert (
        loupe_window._hypnogram_interval_label_visuals[merged_id].getRegion()
        == pytest.approx((0.0, 20.0))
    )


def test_partial_overwrite_right_repositions_surviving_visual(loupe_window, qapp):
    """Reproduce the doubled-overlay bug: a partial overwrite that ends inside
    an existing epoch keeps the surviving left tail's row_id, so its region
    must be repositioned to the shrunk span instead of covering the old one."""
    loupe_window.window_len = 40.0
    loupe_window.window_start = 0.0
    loupe_window._apply_x_range()
    qapp.processEvents()

    wake_id = loupe_window.interval_label_set.add(0.0, 20.0, "Wake")
    loupe_window._finalize_interval_label_change(
        force_rebuild=True, refresh_summary=False
    )
    qapp.processEvents()

    wake_bundle = loupe_window._interval_label_visuals[wake_id]
    for _i, reg in wake_bundle.plot_regions:
        assert reg.getRegion() == pytest.approx((0.0, 20.0))

    # Overwrite [10, 30] with NREM. Wake's right half is overwritten; Wake
    # survives as [0, 10] keeping its row_id.
    loupe_window._add_new_interval_label(10.0, 30.0, "NREM")
    qapp.processEvents()

    data = {r.label: (r.start, r.end) for r in loupe_window.interval_label_set}
    assert data["Wake"] == pytest.approx((0.0, 10.0))
    assert data["NREM"] == pytest.approx((10.0, 30.0))

    # Same bundle object (stable row_id), now repositioned to [0, 10].
    assert wake_id in loupe_window._interval_label_visuals
    wake_bundle = loupe_window._interval_label_visuals[wake_id]
    assert (wake_bundle.start, wake_bundle.end) == (0.0, 10.0)
    for _i, reg in wake_bundle.plot_regions:
        assert reg.getRegion() == pytest.approx((0.0, 10.0))
    assert (
        loupe_window._hypnogram_interval_label_visuals[wake_id].getRegion()
        == pytest.approx((0.0, 10.0))
    )

    # The NREM overlay sits exactly on [10, 30] — no Wake region still covers
    # [10, 20] (the impossible "two states" appearance).
    nrem_bundles = [
        b for b in loupe_window._interval_label_visuals.values() if b.label == "NREM"
    ]
    assert len(nrem_bundles) == 1
    for _i, reg in nrem_bundles[0].plot_regions:
        assert reg.getRegion() == pytest.approx((10.0, 30.0))


def test_partial_overwrite_left_repositions_surviving_visual(loupe_window, qapp):
    """Mirror case: a partial overwrite that starts inside an existing epoch
    keeps the surviving right tail's row_id, whose start must move forward."""
    loupe_window.window_len = 40.0
    loupe_window.window_start = 0.0
    loupe_window._apply_x_range()
    qapp.processEvents()

    wake_id = loupe_window.interval_label_set.add(10.0, 30.0, "Wake")
    loupe_window._finalize_interval_label_change(
        force_rebuild=True, refresh_summary=False
    )
    qapp.processEvents()

    # Overwrite [0, 20] with NREM. Wake's left half is overwritten; Wake
    # survives as [20, 30] keeping its row_id (right-only tail).
    loupe_window._add_new_interval_label(0.0, 20.0, "NREM")
    qapp.processEvents()

    data = {r.label: (r.start, r.end) for r in loupe_window.interval_label_set}
    assert data["Wake"] == pytest.approx((20.0, 30.0))
    assert data["NREM"] == pytest.approx((0.0, 20.0))

    wake_bundle = loupe_window._interval_label_visuals[wake_id]
    assert (wake_bundle.start, wake_bundle.end) == (20.0, 30.0)
    for _i, reg in wake_bundle.plot_regions:
        assert reg.getRegion() == pytest.approx((20.0, 30.0))
    assert (
        loupe_window._hypnogram_interval_label_visuals[wake_id].getRegion()
        == pytest.approx((20.0, 30.0))
    )


def test_plot_rebuild_recreates_interval_label_visuals(loupe_window, qapp):
    loupe_window.window_len = 40.0
    loupe_window.window_start = 0.0
    loupe_window._apply_x_range()
    qapp.processEvents()

    loupe_window.interval_label_set.add(5.0, 15.0, "Wake")
    loupe_window.interval_label_set.add(20.0, 30.0, "NREM")
    loupe_window._finalize_interval_label_change(force_rebuild=True, refresh_summary=False)
    qapp.processEvents()

    previous_visuals = dict(loupe_window._interval_label_visuals)
    previous_hypnogram_visuals = dict(loupe_window._hypnogram_interval_label_visuals)

    loupe_window._rebuild_all_plots()
    qapp.processEvents()

    assert set(loupe_window._interval_label_visuals) == set(previous_visuals)
    for key, old_bundle in previous_visuals.items():
        assert loupe_window._interval_label_visuals[key] is not old_bundle

    assert set(loupe_window._hypnogram_interval_label_visuals) == set(previous_hypnogram_visuals)
    for key, old_region in previous_hypnogram_visuals.items():
        assert loupe_window._hypnogram_interval_label_visuals[key] is not old_region
