"""Viewport-fit sizing for large stacks of split heatmaps."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
import xarray as xr
from PySide6 import QtWidgets

from loupe import HeatmapConfig, TraceConfig, view


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pg.setConfigOptions(useOpenGL=False)
    yield app


def _notebook_style_configs() -> list:
    """One trace plus two heatmaps split into 12 subplots apiece."""
    time = np.linspace(0.0, 30.0, 120)
    groups = np.repeat([f"dend-{i}" for i in range(12)], 4)
    heat = xr.DataArray(
        np.random.default_rng(0).normal(size=(len(groups), len(time))),
        dims=("row", "time"),
        coords={
            "row": np.arange(len(groups)),
            "time": time,
            "dend-ID": ("row", groups),
        },
        name="heat",
    )
    trace = xr.DataArray(
        np.sin(time),
        dims=("time",),
        coords={"time": time},
        name="trace",
    )
    return [
        TraceConfig(trace),
        HeatmapConfig(heat, split_by="dend-ID"),
        HeatmapConfig(heat, split_by="dend-ID"),
    ]


def _heterogeneous_configs() -> tuple[list, list[int]]:
    """Notebook-like duplicated heatmaps spanning 1 through 64 rows."""
    time = np.linspace(0.0, 30.0, 120)
    counts = [64, 38, 10, 5, 3, 1]
    groups = np.concatenate(
        [np.repeat(f"dend-{i}", count) for i, count in enumerate(counts)]
    )
    heat = xr.DataArray(
        np.random.default_rng(1).normal(size=(len(groups), len(time))),
        dims=("row", "time"),
        coords={
            "row": np.arange(len(groups)),
            "time": time,
            "dend-ID": ("row", groups),
        },
        name="heterogeneous",
    )
    trace = xr.DataArray(
        np.cos(time),
        dims=("time",),
        coords={"time": time},
        name="trace",
    )
    return [
        TraceConfig(trace),
        HeatmapConfig(heat, split_by="dend-ID"),
        HeatmapConfig(heat, split_by="dend-ID"),
    ], counts


def _settle(qapp, window, *, width: int = 1400, height: int = 900) -> None:
    # Loupe enables OpenGL for interactive use; switch it back off before the
    # offscreen backend paints anything.
    pg.setConfigOptions(useOpenGL=False)
    window.resize(width, height)
    window.show()
    for _ in range(30):
        qapp.processEvents()
    # Exercise the same operations as the deferred resize handler without
    # depending on wall-clock timer delivery in the offscreen test backend.
    window._update_plot_area_height()
    window._apply_custom_plot_heights()
    for _ in range(10):
        qapp.processEvents()


def _close(qapp, window) -> None:
    pg.setConfigOptions(useOpenGL=False)
    window.close()
    qapp.processEvents()


def test_view_option_fits_notebook_style_heatmap_stack_without_scrolling(_qapp):
    window = view(
        _notebook_style_configs(),
        compact_heatmaps_to_fit=True,
    )
    _settle(_qapp, window)

    scrollbar = window.plot_scroll_area.verticalScrollBar()
    assert window.compact_heatmaps_to_fit is True
    assert window.action_compact_heatmaps_to_fit.isChecked()
    assert window.capture_view_config().display["compact_heatmaps_to_fit"] is True
    assert len(window.heatmap_plots) == 24
    assert scrollbar.maximum() == 0
    assert window.plot_area.minimumHeight() == 0
    assert window.plot_area.height() <= window.plot_scroll_area.viewport().height()
    assert all(plot.geometry().height() > 0 for plot in window.heatmap_plots)

    # A resize recomputes the fit against the new visible viewport.
    _settle(_qapp, window, width=1100, height=650)
    assert scrollbar.maximum() == 0
    assert window.plot_area.height() <= window.plot_scroll_area.viewport().height()

    _close(_qapp, window)


def test_runtime_toggle_switches_between_natural_and_viewport_fit_heights(_qapp):
    window = view(
        _notebook_style_configs(),
        compact_heatmaps_to_fit=False,
    )
    _settle(_qapp, window)

    scrollbar = window.plot_scroll_area.verticalScrollBar()
    assert window.compact_heatmaps_to_fit is False
    assert not window.action_compact_heatmaps_to_fit.isChecked()
    assert scrollbar.maximum() > 0

    window.action_compact_heatmaps_to_fit.setChecked(True)
    _settle(_qapp, window)
    assert window.compact_heatmaps_to_fit is True
    assert scrollbar.maximum() == 0

    window.action_compact_heatmaps_to_fit.setChecked(False)
    _settle(_qapp, window)
    assert window.compact_heatmaps_to_fit is False
    assert scrollbar.maximum() > 0

    _close(_qapp, window)


def test_proportional_sizing_extends_through_one_row_heatmaps(_qapp):
    configs, counts = _heterogeneous_configs()
    window = view(configs, compact_heatmaps_to_fit=True)
    _settle(_qapp, window)

    rendered_counts = [int(series.Y.shape[0]) for series in window.heatmap_series]
    assert rendered_counts == counts * 2

    content_heights = [
        float(plot.getViewBox().geometry().height())
        for plot in window.heatmap_plots
    ]
    pixels_per_row = np.asarray(content_heights) / np.asarray(rendered_counts)

    # The old Qt layout floor rendered every <=10-row heatmap at the same
    # ~34 px. They must now remain monotonic and share nearly the same scale,
    # including the bottom plot whose separate time axis adds fixed chrome.
    for offset in (0, len(counts)):
        heights = content_heights[offset : offset + len(counts)]
        assert heights[2] > heights[3] > heights[4] > heights[5] > 0
    assert float(pixels_per_row.max() - pixels_per_row.min()) < 0.75

    # Compact plots drop unreadable y-axis text without removing the aligned
    # axis gutter/spine. Larger heatmaps keep both their name and tick labels.
    for i, plot in enumerate(window.heatmap_plots):
        left = plot.getAxis("left")
        if rendered_counts[i] <= 10:
            assert left.style["showValues"] is False
            assert not left.label.isVisible()
        else:
            assert left.style["showValues"] is True
            assert left.label.isVisible()

    _close(_qapp, window)


def test_natural_proportional_mode_also_has_no_small_heatmap_floor(_qapp):
    configs, counts = _heterogeneous_configs()
    window = view(configs, compact_heatmaps_to_fit=False)
    _settle(_qapp, window)

    first_set = [
        float(plot.getViewBox().geometry().height())
        for plot in window.heatmap_plots[: len(counts)]
    ]
    assert first_set[2] > first_set[3] > first_set[4] > first_set[5] > 0
    assert first_set[2] / counts[2] == pytest.approx(
        first_set[5] / counts[5], rel=0.2
    )

    _close(_qapp, window)
