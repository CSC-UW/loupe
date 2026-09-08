"""The stacked plot area is one GL surface, so its height is bounded by the GPU.

``LoupeApp._update_plot_area_height`` requests ``n * trace_height_px`` CSS px
for ``n`` stacked subplots. The plot area's OpenGL viewport is a single
framebuffer sized in *device* pixels, so past ``GL_MAX_TEXTURE_SIZE /
devicePixelRatio`` the framebuffer fails and the view silently blanks. The
height must be clamped to that ceiling and the user told. Offscreen Qt has no
OpenGL, so the ceiling is injected through ``max_surface_px`` here.
"""

import os
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
import xarray as xr
from PySide6 import QtWidgets

import loupe.app as _loupe_app
from loupe import HeatmapConfig, TraceConfig, view
from loupe.app import MAX_SURFACE_PX_FALLBACK

_EXAMPLE_STATE_DEFS = os.path.join(
    os.path.dirname(_loupe_app.__file__), "example_state_definitions.json"
)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pg.setConfigOptions(useOpenGL=False)
    yield app


def _stack(n_traces: int, n: int = 300) -> xr.DataArray:
    rng = np.random.default_rng(0)
    return xr.DataArray(
        rng.standard_normal((n_traces, n)).astype("float32"),
        dims=["ch", "time"],
        coords={"ch": list(range(n_traces)), "time": np.arange(n) / 100.0},
        name="lfp",
    )


def _heatmap() -> HeatmapConfig:
    time = np.linspace(0.0, 3.0, 300)
    groups = np.repeat([f"g-{i}" for i in range(4)], 3)
    heat = xr.DataArray(
        np.random.default_rng(1).normal(size=(len(groups), len(time))),
        dims=("row", "time"),
        coords={"row": np.arange(len(groups)), "time": time, "grp": ("row", groups)},
        name="heat",
    )
    return HeatmapConfig(heat, split_by="grp")


def _settle(qapp, w, *, width: int = 1400, height: int = 900) -> None:
    pg.setConfigOptions(useOpenGL=False)
    w.resize(width, height)
    w.show()
    for _ in range(30):
        qapp.processEvents()
    w._update_plot_area_height()
    w._apply_custom_plot_heights()
    for _ in range(10):
        qapp.processEvents()


def _close(qapp, w) -> None:
    w.close()
    qapp.processEvents()


def _build(n_traces: int, *, max_surface_px, dpr=None, **kw):
    w = view(TraceConfig(data=_stack(n_traces)), state_definitions=_EXAMPLE_STATE_DEFS, **kw)
    w.max_surface_px = max_surface_px
    if dpr is not None:
        w._device_pixel_ratio = lambda: float(dpr)
    # Construction already ran the clamp against the fallback ceiling (there is
    # no GL context before show). Start the once-per-count notification guard
    # fresh so each test observes the injected ceiling's own notification.
    w._height_clamp_warned.clear()
    return w


def test_construction_clamps_with_the_fallback_and_warns_once(_qapp):
    # Before the first show there is no GL context, so the fallback applies.
    with pytest.warns(UserWarning, match="200") as rec:
        w = view(TraceConfig(data=_stack(200)), state_definitions=_EXAMPLE_STATE_DEFS)
    assert len([r for r in rec if "200" in str(r.message)]) == 1
    assert w.plot_area.minimumHeight() == int(MAX_SURFACE_PX_FALLBACK / w._device_pixel_ratio())
    _close(_qapp, w)


def test_under_the_cap_is_unchanged(_qapp):
    w = _build(20, max_surface_px=1_000_000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _settle(_qapp, w)
    assert w.plot_area.minimumHeight() == 20 * w.trace_height_px
    assert not [c for c in caught if "surface" in str(c.message)]
    _close(_qapp, w)


def test_over_the_cap_is_clamped_to_the_surface_limit(_qapp):
    w = _build(200, max_surface_px=4096, dpr=1.0)
    with pytest.warns(UserWarning, match="200"):
        _settle(_qapp, w)
    assert w.plot_area.minimumHeight() == 4096
    _close(_qapp, w)


@pytest.mark.parametrize("n_traces", [10, 40, 90])
def test_device_height_never_exceeds_the_limit_and_nothing_is_clipped(_qapp, n_traces):
    w = _build(n_traces, max_surface_px=4096, dpr=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _settle(_qapp, w)
    assert w.plot_area.minimumHeight() * 1.0 <= 4096
    plots = [w.plots[idx] for kind, idx in w._get_visible_subplot_order() if kind == "ts"]
    assert len(plots) == n_traces
    assert all(p.geometry().height() > 0 for p in plots)
    assert max(p.geometry().bottom() for p in plots) <= w.plot_area.height() + 1
    _close(_qapp, w)


def test_device_pixel_ratio_divides_the_ceiling(_qapp):
    w = _build(200, max_surface_px=16384, dpr=2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _settle(_qapp, w)
    assert w.plot_area.minimumHeight() == 8192
    _close(_qapp, w)


def test_user_is_notified_once_per_trace_count(_qapp):
    w = _build(200, max_surface_px=4096, dpr=1.0)
    with pytest.warns(UserWarning, match="200") as rec:
        _settle(_qapp, w)
    assert len([r for r in rec if "200" in str(r.message)]) == 1
    assert "GPU" in w.status.currentMessage()
    w._update_status()  # the routine refresh keeps the note
    assert "GPU" in w.status.currentMessage()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w._update_plot_area_height()
        w._update_plot_area_height()
        w.max_surface_px = 8192  # a different ceiling, same trace count
        w._update_plot_area_height()
    assert caught == []
    assert w.plot_area.minimumHeight() == 8192  # the status bar / clamp still track the cap
    assert "8192" in w.status.currentMessage()

    w.max_surface_px = 10**7  # the clamp stops binding: note gone, height natural
    w._update_plot_area_height()
    assert "GPU" not in w.status.currentMessage()
    assert w.plot_area.minimumHeight() == 200 * w.trace_height_px
    _close(_qapp, w)


def test_fallback_when_no_gl_limit_can_be_measured(_qapp):
    w = _build(200, max_surface_px=None)
    assert w._gl_max_surface_px() is None  # offscreen: no QOpenGLWidget viewport
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _settle(_qapp, w)
    expected = int(MAX_SURFACE_PX_FALLBACK / w._device_pixel_ratio())
    assert w.plot_area.minimumHeight() == expected
    assert w.plot_area.minimumHeight() < 200 * w.trace_height_px
    _close(_qapp, w)


def test_compact_mode_early_return_takes_precedence(_qapp):
    w = view(
        [TraceConfig(data=_stack(80)), _heatmap()],
        compact_heatmaps_to_fit=True,
        state_definitions=_EXAMPLE_STATE_DEFS,
    )
    w.max_surface_px = 4096
    w._device_pixel_ratio = lambda: 1.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _settle(_qapp, w)
    assert w.plot_area.minimumHeight() == 0
    assert not [c for c in caught if "surface" in str(c.message)]
    _close(_qapp, w)
