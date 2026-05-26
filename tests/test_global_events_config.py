"""Tests for the GlobalEventsConfig overlay system: dataclass defaults,
view() validation, default style cycling, per-class rendering across panes,
and live restyle via _apply_global_event_class_style."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import polars as pl
import pyqtgraph as pg
import pytest
import xarray as xr
from PySide6 import QtCore, QtWidgets

import loupe.app as _loupe_app
from loupe import (
    GlobalEventsConfig,
    RasterConfig,
    TraceConfig,
    view,
)
from loupe.app import (
    _GLOBAL_EVENT_COLOR_CYCLE,
    _GLOBAL_EVENT_LINE_STYLE_TO_QT,
    _GLOBAL_EVENT_STYLE_ORDER,
    _GLOBAL_EVENT_Z,
)

_EXAMPLE_STATE_DEFS = os.path.join(
    os.path.dirname(_loupe_app.__file__),
    "example_state_definitions.json",
)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pg.setConfigOptions(useOpenGL=False)
    yield app


def _trace_da(n: int = 200, name: str = "signal") -> xr.DataArray:
    t = np.linspace(0.0, 30.0, n)
    return xr.DataArray(
        np.sin(2 * np.pi * 0.5 * t)[None, :],
        dims=("syn_id", "time"),
        coords={"syn_id": [0], "time": t},
        name=name,
    )


def _raster_df(n: int = 12, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "time": np.sort(rng.uniform(0.0, 30.0, n)),
        "source_id": rng.integers(0, 2, n).astype(np.int64),
    })


def _events_df(times, kinds=None) -> pl.DataFrame:
    cols = {"time": list(times)}
    if kinds is not None:
        cols["kind"] = list(kinds)
    return pl.DataFrame(cols)


# ---------------- Dataclass defaults ----------------


def test_global_events_config_defaults():
    df = _events_df([1.0, 2.0, 3.0])
    cfg = GlobalEventsConfig(df)
    assert cfg.event_times_column == "time"
    assert cfg.style_events_on is None
    assert cfg.style_kwargs is None


def test_global_events_config_accepts_explicit_fields():
    df = _events_df([1.0, 2.0], kinds=["a", "b"])
    cfg = GlobalEventsConfig(
        df,
        event_times_column="time",
        style_events_on="kind",
        style_kwargs={"a": {"line_color": "#ff0000"}},
    )
    assert cfg.style_events_on == "kind"
    assert cfg.style_kwargs == {"a": {"line_color": "#ff0000"}}


# ---------------- view() validation ----------------


def test_view_rejects_non_global_events_config():
    with pytest.raises(TypeError, match="GlobalEventsConfig"):
        view(
            TraceConfig(_trace_da()),
            global_events="not a config",
            state_definitions=_EXAMPLE_STATE_DEFS,
        )


def test_view_rejects_missing_event_times_column():
    df = _events_df([1.0, 2.0])
    with pytest.raises(ValueError, match="event_times_column"):
        view(
            TraceConfig(_trace_da()),
            global_events=GlobalEventsConfig(df, event_times_column="bad"),
            state_definitions=_EXAMPLE_STATE_DEFS,
        )


def test_view_rejects_missing_style_events_on_column():
    df = _events_df([1.0, 2.0])
    with pytest.raises(ValueError, match="style_events_on"):
        view(
            TraceConfig(_trace_da()),
            global_events=GlobalEventsConfig(df, style_events_on="bad"),
            state_definitions=_EXAMPLE_STATE_DEFS,
        )


def test_view_warns_on_style_kwargs_without_style_events_on():
    df = _events_df([1.0])
    with pytest.warns(UserWarning, match="ignored when style_events_on is None"):
        w = view(
            TraceConfig(_trace_da()),
            global_events=GlobalEventsConfig(
                df,
                style_kwargs={"a": {"line_color": "#ff0000"}},
            ),
            state_definitions=_EXAMPLE_STATE_DEFS,
        )
    w.close()


def test_view_warns_on_stray_style_kwargs_key():
    df = _events_df([1.0, 2.0], kinds=["a", "b"])
    with pytest.warns(UserWarning, match="not present"):
        w = view(
            TraceConfig(_trace_da()),
            global_events=GlobalEventsConfig(
                df,
                style_events_on="kind",
                style_kwargs={"nonexistent": {"line_color": "#ff0000"}},
            ),
            state_definitions=_EXAMPLE_STATE_DEFS,
        )
    w.close()


# ---------------- Default style cycling ----------------


def test_default_styles_cycle_styles_first_then_colors():
    # 7 classes → 5 styles × first color, then 2 more cycling to second color.
    kinds = [f"k{i}" for i in range(7)]
    df = _events_df(list(range(1, 8)), kinds=kinds)
    w = view(
        TraceConfig(_trace_da()),
        global_events=GlobalEventsConfig(df, style_events_on="kind"),
        state_definitions=_EXAMPLE_STATE_DEFS,
    )
    resolved = w._resolved_event_styles
    sorted_keys = sorted(resolved.keys(), key=repr)
    # Classes 0..4 share color 0, get styles solid/dashed/dotted/dashdot/dashdotdot
    for i in range(5):
        s = resolved[sorted_keys[i]]
        assert s["line_color"] == _GLOBAL_EVENT_COLOR_CYCLE[0]
        assert s["line_style"] == _GLOBAL_EVENT_STYLE_ORDER[i]
    # Class 5 cycles to color 1, style 0 (solid)
    assert resolved[sorted_keys[5]]["line_color"] == _GLOBAL_EVENT_COLOR_CYCLE[1]
    assert resolved[sorted_keys[5]]["line_style"] == _GLOBAL_EVENT_STYLE_ORDER[0]
    # Class 6 cycles to color 1, style 1 (dashed)
    assert resolved[sorted_keys[6]]["line_color"] == _GLOBAL_EVENT_COLOR_CYCLE[1]
    assert resolved[sorted_keys[6]]["line_style"] == _GLOBAL_EVENT_STYLE_ORDER[1]
    w.close()


def test_single_class_uses_sentinel_none_key():
    df = _events_df([1.0, 2.0])
    w = view(
        TraceConfig(_trace_da()),
        global_events=GlobalEventsConfig(df),
        state_definitions=_EXAMPLE_STATE_DEFS,
    )
    assert list(w._resolved_event_styles.keys()) == [None]
    style = w._resolved_event_styles[None]
    assert style["line_style"] == "solid"
    assert style["line_color"] == (230, 230, 230)
    assert style["line_alpha"] == 200
    w.close()


def test_user_style_kwargs_override_defaults():
    df = _events_df([1.0, 2.0], kinds=["a", "b"])
    w = view(
        TraceConfig(_trace_da()),
        global_events=GlobalEventsConfig(
            df,
            style_events_on="kind",
            style_kwargs={
                "a": {"line_color": "#ff0000", "line_width": 3.0},
            },
        ),
        state_definitions=_EXAMPLE_STATE_DEFS,
    )
    s_a = w._resolved_event_styles["a"]
    s_b = w._resolved_event_styles["b"]
    # Overridden fields
    assert s_a["line_color"] == (255, 0, 0)
    assert s_a["line_width"] == 3.0
    # Non-overridden fields keep defaults
    assert s_a["line_style"] == "solid"
    assert s_a["line_alpha"] == 200
    # Class b is untouched
    assert s_b["line_color"] == _GLOBAL_EVENT_COLOR_CYCLE[0]
    assert s_b["line_style"] == "dashed"
    w.close()


def test_invalid_line_style_raises():
    df = _events_df([1.0], kinds=["a"])
    with pytest.raises(ValueError, match="line_style"):
        view(
            TraceConfig(_trace_da()),
            global_events=GlobalEventsConfig(
                df,
                style_events_on="kind",
                style_kwargs={"a": {"line_style": "bogus"}},
            ),
            state_definitions=_EXAMPLE_STATE_DEFS,
        )


def test_unknown_style_kwargs_keys_warn_and_are_dropped():
    df = _events_df([1.0], kinds=["a"])
    with pytest.warns(UserWarning, match="unknown keys"):
        w = view(
            TraceConfig(_trace_da()),
            global_events=GlobalEventsConfig(
                df,
                style_events_on="kind",
                style_kwargs={"a": {"wrong_key": 1, "line_width": 2.0}},
            ),
            state_definitions=_EXAMPLE_STATE_DEFS,
        )
    s = w._resolved_event_styles["a"]
    assert "wrong_key" not in s
    assert s["line_width"] == 2.0
    w.close()


# ---------------- Per-pane line rendering ----------------


def test_lines_rendered_across_trace_and_raster_panes():
    # 2 trace panes (syn_id=2) + 1 raster pane = 3 visible panes.
    t = np.linspace(0.0, 30.0, 200)
    da = xr.DataArray(
        np.zeros((2, len(t))),
        dims=("syn_id", "time"),
        coords={"syn_id": [0, 1], "time": t},
        name="signal",
    )
    events = _events_df([5.0, 12.5, 20.0], kinds=["a", "b", "a"])
    w = view(
        [TraceConfig(da), RasterConfig(_raster_df(), time_col="time", order_by="source_id")],
        global_events=GlobalEventsConfig(events, style_events_on="kind"),
        state_definitions=_EXAMPLE_STATE_DEFS,
    )

    n_panes = len(w._global_event_panes())
    assert n_panes == 3  # 2 trace + 1 matrix

    buckets = w._global_event_lines_by_class
    assert set(buckets.keys()) == {"a", "b"}
    assert len(buckets["a"]) == 2 * n_panes  # 2 events of kind "a"
    assert len(buckets["b"]) == 1 * n_panes  # 1 event of kind "b"

    # Every line is an InfiniteLine at the top z-layer.
    for bucket in buckets.values():
        for ln in bucket:
            assert isinstance(ln, pg.InfiniteLine)
            assert ln.zValue() == _GLOBAL_EVENT_Z

    w.close()


def test_lines_use_resolved_pen_style_and_color():
    events = _events_df([5.0, 12.5], kinds=["a", "b"])
    w = view(
        TraceConfig(_trace_da()),
        global_events=GlobalEventsConfig(
            events,
            style_events_on="kind",
            style_kwargs={
                "a": {"line_color": "#ff8800", "line_width": 4.0, "line_alpha": 150},
            },
        ),
        state_definitions=_EXAMPLE_STATE_DEFS,
    )
    a_lines = w._global_event_lines_by_class["a"]
    assert a_lines, "expected at least one line for class 'a'"
    pen = a_lines[0].pen
    color = pen.color()
    assert (color.red(), color.green(), color.blue()) == (255, 136, 0)
    assert color.alpha() == 150
    # Width may be returned as float; allow either.
    assert float(pen.widthF()) == pytest.approx(4.0)
    # Style is solid by default (overridden style had no line_style key).
    assert pen.style() == _GLOBAL_EVENT_LINE_STYLE_TO_QT["solid"]
    w.close()


def test_apply_global_event_class_style_relays_color_change():
    events = _events_df([5.0], kinds=["a"])
    w = view(
        TraceConfig(_trace_da()),
        global_events=GlobalEventsConfig(events, style_events_on="kind"),
        state_definitions=_EXAMPLE_STATE_DEFS,
    )
    w._resolved_event_styles["a"]["line_color"] = (12, 34, 56)
    w._resolved_event_styles["a"]["line_alpha"] = 99
    w._apply_global_event_class_style("a")
    for ln in w._global_event_lines_by_class["a"]:
        c = ln.pen.color()
        assert (c.red(), c.green(), c.blue(), c.alpha()) == (12, 34, 56, 99)
    w.close()


def test_no_global_events_means_no_lines_and_no_menu_action():
    w = view(
        TraceConfig(_trace_da()),
        state_definitions=_EXAMPLE_STATE_DEFS,
    )
    assert w._global_event_lines_by_class == {}
    assert w._resolved_event_styles == {}
    # The "Style Global Events…" action should NOT have been added.
    view_menu = next(
        m for m in w.menuBar().findChildren(QtWidgets.QMenu)
        if m.title().replace("&", "") == "View"
    )
    titles = [a.text() for a in view_menu.actions()]
    assert not any("Style Global Events" in t for t in titles)
    w.close()


def test_menu_action_present_when_global_events_set():
    df = _events_df([1.0, 2.0])
    w = view(
        TraceConfig(_trace_da()),
        global_events=GlobalEventsConfig(df),
        state_definitions=_EXAMPLE_STATE_DEFS,
    )
    view_menu = next(
        m for m in w.menuBar().findChildren(QtWidgets.QMenu)
        if m.title().replace("&", "") == "View"
    )
    titles = [a.text() for a in view_menu.actions()]
    assert any("Style Global Events" in t for t in titles)
    w.close()
