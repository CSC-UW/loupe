"""Tests for the pinned label strip.

Two layers:

* ``LabelBandRenderer`` diff logic — exercised with stub plot/region objects so
  it needs no Qt event loop (the bulk of the coverage).
* A thin integration test that builds a real (offscreen) ``LoupeApp`` and checks
  the strip is wired into the label-sync and x-range paths, mirroring the
  fixture style in ``test_interval_label_incremental.py``.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtWidgets

from loupe.app import LoupeApp, Series
from loupe.label_strip import LabelBandRenderer
from loupe.state_config import load_state_config


# --------------------------------------------------------------------------
# Unit layer: LabelBandRenderer diff, with stubs (no Qt)
# --------------------------------------------------------------------------


class _StubRegion:
    def __init__(self, a, b, color):
        self.a = float(a)
        self.b = float(b)
        self.color = color
        self.z = None

    def setRegion(self, span):
        self.a, self.b = float(span[0]), float(span[1])

    def setZValue(self, z):
        self.z = z


class _StubPlot:
    def __init__(self):
        self.items = []
        self.add_calls = 0
        self.remove_calls = 0

    def addItem(self, item):
        self.items.append(item)
        self.add_calls += 1

    def removeItem(self, item):
        if item in self.items:
            self.items.remove(item)
        self.remove_calls += 1


def _row(start, end, label):
    return SimpleNamespace(start=start, end=end, label=label)


def _make_renderer(entries, colors=None):
    """Return (renderer, plot, state) where state['entries'] is mutable so a
    test can change what the next ``sync()`` sees."""
    colors = colors if colors is not None else {}
    plot = _StubPlot()
    state = {"entries": list(entries)}

    def entries_provider():
        return state["entries"]

    def color_provider(name):
        return colors.get(name, (1, 2, 3, 4))

    def region_factory(a, b, color):
        return _StubRegion(a, b, color)

    def set_region_color(region, color):
        region.color = color

    renderer = LabelBandRenderer(
        plot,
        entries_provider=entries_provider,
        color_provider=color_provider,
        region_factory=region_factory,
        set_region_color=set_region_color,
    )
    return renderer, plot, state


def test_sync_adds_one_band_per_entry():
    renderer, plot, _ = _make_renderer(
        [(1, _row(0.0, 10.0, "A")), (2, _row(10.0, 20.0, "B"))]
    )
    renderer.sync()
    assert plot.add_calls == 2
    assert set(renderer._visuals) == {1, 2}
    assert renderer._drawn[1] == (0.0, 10.0, "A")
    assert renderer._drawn[2] == (10.0, 20.0, "B")


def test_sync_removes_entries_that_left_the_window():
    renderer, plot, state = _make_renderer(
        [(1, _row(0.0, 10.0, "A")), (2, _row(10.0, 20.0, "B"))]
    )
    renderer.sync()
    region_one = renderer._visuals[1]

    state["entries"] = [(1, _row(0.0, 10.0, "A"))]
    renderer.sync()

    assert set(renderer._visuals) == {1}
    assert renderer._visuals[1] is region_one  # survivor untouched
    assert 2 not in renderer._drawn
    assert plot.remove_calls == 1


def test_sync_repositions_changed_span_in_place():
    renderer, _, state = _make_renderer([(1, _row(0.0, 10.0, "A"))])
    renderer.sync()
    region = renderer._visuals[1]
    add_calls_before = _plot_of(renderer).add_calls

    state["entries"] = [(1, _row(0.0, 20.0, "A"))]
    renderer.sync()

    assert renderer._visuals[1] is region  # same object, repositioned
    assert (region.a, region.b) == (0.0, 20.0)
    assert renderer._drawn[1] == (0.0, 20.0, "A")
    assert _plot_of(renderer).add_calls == add_calls_before  # no re-add


def test_sync_recolors_on_rename_in_place():
    renderer, _, state = _make_renderer(
        [(1, _row(0.0, 10.0, "A"))], colors={"A": (10, 0, 0, 50), "B": (0, 20, 0, 60)}
    )
    renderer.sync()
    region = renderer._visuals[1]
    assert region.color == (10, 0, 0, 50)

    state["entries"] = [(1, _row(0.0, 10.0, "B"))]
    renderer.sync()

    assert renderer._visuals[1] is region
    assert region.color == (0, 20, 0, 60)
    assert renderer._drawn[1] == (0.0, 10.0, "B")


def test_sync_unchanged_row_is_a_noop():
    renderer, plot, state = _make_renderer([(1, _row(0.0, 10.0, "A"))])
    renderer.sync()
    region = renderer._visuals[1]

    renderer.sync()  # identical entries
    assert renderer._visuals[1] is region
    assert plot.add_calls == 1
    assert plot.remove_calls == 0


def test_force_rebuild_clears_and_recreates():
    renderer, plot, _ = _make_renderer([(1, _row(0.0, 10.0, "A"))])
    renderer.sync()
    old_region = renderer._visuals[1]

    renderer.sync(force_rebuild=True)  # same entries, but torn down first
    assert plot.remove_calls == 1
    assert renderer._visuals[1] is not old_region
    assert renderer._drawn[1] == (0.0, 10.0, "A")


def test_refresh_colors_reapplies_current_palette():
    colors = {"A": (10, 0, 0, 50)}
    renderer, _, _ = _make_renderer([(1, _row(0.0, 10.0, "A"))], colors=colors)
    renderer.sync()
    region = renderer._visuals[1]

    colors["A"] = (0, 0, 90, 99)  # e.g. alpha multiplier changed
    renderer.refresh_colors()
    assert region.color == (0, 0, 90, 99)


def test_clear_removes_everything():
    renderer, plot, _ = _make_renderer(
        [(1, _row(0.0, 10.0, "A")), (2, _row(10.0, 20.0, "B"))]
    )
    renderer.sync()
    renderer.clear()
    assert renderer._visuals == {}
    assert renderer._drawn == {}
    assert plot.items == []


def _plot_of(renderer):
    return renderer._plot


# --------------------------------------------------------------------------
# Integration layer: real (offscreen) LoupeApp wiring
# --------------------------------------------------------------------------


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


def test_label_strip_is_built_and_visible_by_default(loupe_window):
    assert isinstance(loupe_window.label_strip_widget, pg.PlotWidget)
    assert isinstance(loupe_window.label_strip_renderer, LabelBandRenderer)
    # The top-level window is never show()n in these offscreen tests, so
    # isVisible() is always False; isHidden() reflects the explicit setVisible.
    assert not loupe_window.label_strip_widget.isHidden()
    assert loupe_window.label_strip_visible is True


def test_label_strip_tracks_label_edits_and_window(loupe_window, qapp):
    loupe_window.window_len = 40.0
    loupe_window.window_start = 0.0
    loupe_window._apply_x_range()
    qapp.processEvents()

    nrem_id = loupe_window.interval_label_set.add(0.0, 10.0, "NREM")
    rem_id = loupe_window.interval_label_set.add(20.0, 30.0, "REM")
    loupe_window._finalize_interval_label_change(
        force_rebuild=True, refresh_summary=False
    )
    qapp.processEvents()

    # Both labels fall inside the [0, 40] window → both drawn on the strip.
    assert set(loupe_window.label_strip_renderer._visuals) == {nrem_id, rem_id}

    # Narrow the window so only the first label overlaps; the strip follows.
    loupe_window.window_len = 15.0
    loupe_window.window_start = 0.0
    loupe_window._apply_x_range()
    qapp.processEvents()
    assert set(loupe_window.label_strip_renderer._visuals) == {nrem_id}


def test_toggle_label_strip_visibility(loupe_window):
    assert not loupe_window.label_strip_widget.isHidden()
    loupe_window._toggle_label_strip_visibility()
    assert loupe_window.label_strip_widget.isHidden()
    assert loupe_window.label_strip_visible is False
    loupe_window._toggle_label_strip_visibility()
    assert not loupe_window.label_strip_widget.isHidden()
    assert loupe_window.label_strip_visible is True


def test_overlays_disabled_keeps_strip(loupe_window, qapp):
    # Default-on overlays draw across the subplots; turning them off must leave
    # the strip as the label display.
    loupe_window.interval_label_overlays_enabled = False
    loupe_window.window_len = 40.0
    loupe_window.window_start = 0.0
    loupe_window._apply_x_range()
    qapp.processEvents()

    nrem_id = loupe_window.interval_label_set.add(0.0, 10.0, "NREM")
    loupe_window._finalize_interval_label_change(
        force_rebuild=True, refresh_summary=False
    )
    qapp.processEvents()

    assert loupe_window._interval_label_visuals == {}  # no trace/dense overlays
    assert nrem_id in loupe_window.label_strip_renderer._visuals  # strip still on


def test_label_strip_edges_align_with_subplots(loupe_window, qapp):
    # The strip is a separate full-width widget; its left edge must sit on the
    # subplots' y-spine gutter and its right edge must not overhang the area the
    # subplots reserve for scrollbars. Verified by comparing global x of the
    # window edges mapped through a subplot vs. the strip.
    loupe_window.resize(1000, 600)
    loupe_window.show()
    for _ in range(4):
        loupe_window._align_left_axes()
        qapp.processEvents()

    ref = loupe_window._first_realized_plot()
    if ref is None:
        pytest.skip("offscreen layout not realized")
    x_left = float(loupe_window.window_start)
    x_right = float(loupe_window.window_start + loupe_window.window_len)
    pl = loupe_window._data_x_to_global_x(ref, x_left)
    pr = loupe_window._data_x_to_global_x(ref, x_right)
    sl = loupe_window._data_x_to_global_x(loupe_window.label_strip_plot, x_left)
    sr = loupe_window._data_x_to_global_x(loupe_window.label_strip_plot, x_right)
    if None in (pl, pr, sl, sr):
        pytest.skip("offscreen coordinate mapping unavailable")
    assert abs(sl - pl) <= 2  # left edge flush with the y-spine gutter
    assert abs(sr - pr) <= 2  # right edge flush (no scrollbar-column overhang)


def test_toggle_interval_label_overlays(loupe_window, qapp):
    loupe_window.window_len = 40.0
    loupe_window.window_start = 0.0
    loupe_window._apply_x_range()
    nrem_id = loupe_window.interval_label_set.add(0.0, 10.0, "NREM")
    loupe_window._finalize_interval_label_change(
        force_rebuild=True, refresh_summary=False
    )
    qapp.processEvents()
    assert nrem_id in loupe_window._interval_label_visuals  # overlays on by default

    loupe_window._toggle_interval_label_overlays()
    qapp.processEvents()
    assert loupe_window.interval_label_overlays_enabled is False
    assert loupe_window._interval_label_visuals == {}
    assert nrem_id in loupe_window.label_strip_renderer._visuals  # strip unaffected

    loupe_window._toggle_interval_label_overlays()
    qapp.processEvents()
    assert loupe_window.interval_label_overlays_enabled is True
    assert nrem_id in loupe_window._interval_label_visuals  # redrawn
