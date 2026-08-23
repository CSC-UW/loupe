"""Viewport-fit sizing for large stacks of split rasters (``compact_rasters_to_fit``)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import polars as pl
import pyqtgraph as pg
import pytest
import xarray as xr
from PySide6 import QtWidgets

from loupe import RasterConfig, TraceConfig, view


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pg.setConfigOptions(useOpenGL=False)
    yield app


def _raster_df(n_groups: int = 12, rows_per: int = 6, n_ev: int = 40) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    time, dend, syn = [], [], []
    for g in range(n_groups):
        for r in range(rows_per):
            ts = np.sort(rng.uniform(0.0, 30.0, n_ev))
            time.extend(ts.tolist())
            dend.extend([f"dend-{g}"] * n_ev)
            syn.extend([g * rows_per + r] * n_ev)
    return pl.DataFrame({"time": time, "dend": dend, "syn": syn})


def _notebook_style_configs() -> list:
    """One trace plus two rasters split into 12 subplots apiece."""
    t = np.linspace(0.0, 30.0, 120)
    trace = xr.DataArray(np.sin(t), dims=("time",), coords={"time": t}, name="trace")
    df = _raster_df()
    return [
        TraceConfig(trace),
        RasterConfig(df, time_col="time", order_by="syn", split_by="dend"),
        RasterConfig(df, time_col="time", order_by="syn", split_by="dend"),
    ]


def _settle(qapp, window, *, width: int = 1400, height: int = 900) -> None:
    pg.setConfigOptions(useOpenGL=False)
    window.resize(width, height)
    window.show()
    for _ in range(30):
        qapp.processEvents()
    window._update_plot_area_height()
    window._apply_custom_plot_heights()
    for _ in range(10):
        qapp.processEvents()


def _close(qapp, window) -> None:
    pg.setConfigOptions(useOpenGL=False)
    window.close()
    qapp.processEvents()


def test_view_option_fits_raster_stack_without_scrolling(_qapp):
    window = view(_notebook_style_configs(), compact_rasters_to_fit=True)
    _settle(_qapp, window)

    scrollbar = window.plot_scroll_area.verticalScrollBar()
    assert window.compact_rasters_to_fit is True
    assert window.action_compact_rasters_to_fit.isChecked()
    assert window.capture_view_config().display["compact_rasters_to_fit"] is True
    assert len(window.raster_plots) == 24
    assert scrollbar.maximum() == 0
    assert window.plot_area.minimumHeight() == 0
    assert window.plot_area.height() <= window.plot_scroll_area.viewport().height()
    assert all(plot.geometry().height() > 0 for plot in window.raster_plots)

    # A resize recomputes the fit against the new visible viewport.
    _settle(_qapp, window, width=1100, height=650)
    assert scrollbar.maximum() == 0
    assert window.plot_area.height() <= window.plot_scroll_area.viewport().height()

    _close(_qapp, window)


def test_runtime_toggle_switches_between_natural_and_viewport_fit_heights(_qapp):
    window = view(_notebook_style_configs(), compact_rasters_to_fit=False)
    _settle(_qapp, window)

    scrollbar = window.plot_scroll_area.verticalScrollBar()
    assert window.compact_rasters_to_fit is False
    assert not window.action_compact_rasters_to_fit.isChecked()
    assert scrollbar.maximum() > 0

    window.action_compact_rasters_to_fit.setChecked(True)
    _settle(_qapp, window)
    assert window.compact_rasters_to_fit is True
    assert scrollbar.maximum() == 0

    window.action_compact_rasters_to_fit.setChecked(False)
    _settle(_qapp, window)
    assert window.compact_rasters_to_fit is False
    assert scrollbar.maximum() > 0

    _close(_qapp, window)


def test_view_config_round_trips_compact_rasters_flag(_qapp):
    window = view(_notebook_style_configs(), compact_rasters_to_fit=True)
    _settle(_qapp, window)
    cfg = window.capture_view_config()
    _close(_qapp, window)

    other = view(_notebook_style_configs(), compact_rasters_to_fit=False)
    _settle(_qapp, other)
    assert other.compact_rasters_to_fit is False
    other.apply_view_config(cfg)
    _settle(_qapp, other)
    assert other.compact_rasters_to_fit is True
    assert other.action_compact_rasters_to_fit.isChecked()
    assert other.plot_scroll_area.verticalScrollBar().maximum() == 0
    _close(_qapp, other)
