"""View-Config schema, semantic matching, and live-window round trips."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import polars as pl
import pyqtgraph as pg
import pytest
import xarray as xr
from PySide6 import QtGui, QtWidgets

import loupe.app as _loupe_app
from loupe import (
    GlobalEventsConfig,
    HeatmapConfig,
    Param,
    RasterConfig,
    SampleMarkers,
    TraceConfig,
    ViewConfig,
    ViewConfigError,
    Zip,
    tunable,
    view,
)

_STATE_DEFS = os.path.join(
    os.path.dirname(_loupe_app.__file__), "example_state_definitions.json"
)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pg.setConfigOptions(useOpenGL=False)
    yield app


def _trace(name: str, channel: str, scale: float = 1.0) -> xr.DataArray:
    t = np.linspace(0.0, 20.0, 201)
    return xr.DataArray(
        (scale * np.sin(t))[None, :],
        dims=("channel", "time"),
        coords={"channel": [channel], "time": t},
        name=name,
    )


def _dense() -> xr.DataArray:
    t = np.linspace(0.0, 20.0, 201)
    return xr.DataArray(
        np.asarray([np.sin(t) + i for i in range(4)]),
        dims=("channel", "time"),
        coords={
            "channel": ["a", "b", "c", "d"],
            "time": t,
            "region": ("channel", ["CA1", "CA1", "DG", "DG"]),
        },
        name="dense",
    )


def _heat() -> xr.DataArray:
    t = np.linspace(0.0, 20.0, 201)
    return xr.DataArray(
        np.arange(6 * len(t), dtype=float).reshape(6, len(t)),
        dims=("row", "time"),
        coords={"row": np.arange(6), "time": t},
        name="heat",
    )


def _raster() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "unit": [0, 1, 0, 1],
        }
    )


def _events() -> GlobalEventsConfig:
    return GlobalEventsConfig(
        pl.DataFrame(
            {
                "time": [2.0, 6.0],
                "kind": ["stim", "reward"],
            }
        ),
        style_events_on="kind",
    )


def _mixed_configs(order=("alpha", "beta", "dense", "raster", "heat")):
    alpha = _trace("alpha", "A")
    beta = _trace("beta", "B", scale=2.0)
    choices = {
        "alpha": TraceConfig(
            alpha,
            array_name=True,
            overlay_arrays=[alpha * 2.0],
            overlay_colors=["#00ff00"],
            view_id="alpha-source",
        ),
        "beta": TraceConfig(
            beta,
            array_name=True,
            view_id="beta-source",
        ),
        "dense": TraceConfig(
            _dense(),
            mode="dense",
            hue="region",
            array_name=True,
            view_id="dense-source",
        ),
        "raster": RasterConfig(
            _raster(),
            time_col="time",
            order_by="unit",
            array_name="events",
            view_id="raster-source",
        ),
        "heat": HeatmapConfig(
            _heat(),
            array_name=True,
            shade_nans="#3E1715",
            view_id="heat-source",
        ),
    }
    return [choices[key] for key in order]


def _source_index(window, kind: str, source_id: str) -> int:
    for i, record in enumerate(window._view_plot_identities[kind]):
        if record.get("source_id") == source_id:
            return i
    raise AssertionError((kind, source_id, window._view_plot_identities))


def _close(window) -> None:
    pg.setConfigOptions(useOpenGL=False)
    window.close()
    QtWidgets.QApplication.processEvents()


def test_schema_round_trip_and_validation(tmp_path):
    config = ViewConfig(
        metadata={"subject": "mouse-1"},
        display={"window_len": 12.0},
    )
    path = config.save(tmp_path / "subject-view")
    assert path.name == "subject-view.loupe-view.json"
    assert ViewConfig.load(path).to_dict() == config.to_dict()

    raw = json.loads(path.read_text())
    assert raw["format"] == "loupe-view-config"
    assert raw["schema_version"] == 1
    assert "session" not in raw
    assert "tuner" not in raw

    raw["unknown"] = True
    with pytest.raises(ViewConfigError, match="Unknown top-level"):
        ViewConfig.from_dict(raw)

    raw.pop("unknown")
    raw["schema_version"] = 999
    with pytest.raises(ViewConfigError, match="newer"):
        ViewConfig.from_dict(raw)

    with pytest.raises(ViewConfigError, match="non-finite"):
        ViewConfig(display={"window_len": float("nan")})


def test_mixed_view_round_trip_matches_semantically_after_reordering(tmp_path):
    source = view(
        _mixed_configs(),
        global_events=_events(),
        state_definitions=_STATE_DEFS,
    )
    alpha_i = _source_index(source, "ts", "alpha-source")
    beta_i = _source_index(source, "ts", "beta-source")

    source.window_len = 7.5
    source.smooth_scroll_fraction = 0.23
    source.playback_speed = 1.75
    source.interval_label_alpha_multiplier = 0.42
    source.interval_label_overlays_enabled = False
    source.label_strip_visible = False
    source.plot_height_factors[alpha_i] = 1.6
    source.trace_visible[beta_i] = False
    source.series_colors[alpha_i] = (10, 20, 30, 240)
    source.series_line_widths[alpha_i] = 2.5
    source.overlay_series[alpha_i][0].color = (40, 50, 60, 255)
    source.overlay_series[alpha_i][0].width = 3.0
    source.overlay_series[alpha_i][0].symbol = "x"
    source.overlay_series[alpha_i][0].symbol_size = 11.0
    source.dense_groups[0].gain = 2.75
    source.dense_groups[0].step = 2
    source.dense_groups[0].traces_per_page = 2
    source.dense_height_factors[0] = 1.3
    source.raster_series[0].color = (70, 80, 90)
    source.raster_height_factors[0] = 1.4
    source.heatmap_series[0].vmin = 12.0
    source.heatmap_series[0].vmax = 900.0
    source.heatmap_series[0].colormap = "plasma"
    source.heatmap_series[0].decim_method = "mean"
    source.heatmap_series[0].shade_nans = (62, 23, 21, 0.85)
    source.heatmap_height_factors[0] = 1.5
    source._resolved_event_styles["stim"].update(
        {
            "line_color": (101, 102, 103),
            "line_style": "dashed",
            "line_width": 3.5,
            "line_alpha": 123,
        }
    )
    source.plots[alpha_i].enableAutoRange("y", False)
    source.plots[alpha_i].setYRange(-4.0, 6.0, padding=0)
    source.subplot_order = [
        ("heatmap", 0),
        ("ts", beta_i),
        ("raster", 0),
        ("dense", 0),
        ("ts", alpha_i),
    ]

    path = source.save_view_config(tmp_path / "mixed.loupe-view.json")
    saved = ViewConfig.load(path)
    assert saved.session is None
    assert saved.tuner is None
    _close(source)

    target = view(
        _mixed_configs(("heat", "beta", "raster", "alpha", "dense")),
        global_events=_events(),
        view_config=path,
        view_config_strict=True,
        state_definitions=_STATE_DEFS,
    )
    alpha_i = _source_index(target, "ts", "alpha-source")
    beta_i = _source_index(target, "ts", "beta-source")

    file_actions = {
        action.text().replace("&", "") for action in target.findChildren(QtGui.QAction)
    }
    assert "Load View-Config…" in file_actions
    assert "Save View-Config As…" in file_actions

    assert target.window_len == pytest.approx(7.5)
    assert target.smooth_scroll_fraction == pytest.approx(0.23)
    assert target.playback_speed == pytest.approx(1.75)
    assert target.interval_label_alpha_multiplier == pytest.approx(0.42)
    assert not target.interval_label_overlays_enabled
    assert not target.label_strip_visible
    assert target.plot_height_factors[alpha_i] == pytest.approx(1.6)
    assert not target.trace_visible[beta_i]
    assert target.series_colors[alpha_i] == (10, 20, 30, 240)
    assert target.series_line_widths[alpha_i] == pytest.approx(2.5)
    overlay = target.overlay_series[alpha_i][0]
    assert overlay.color == (40, 50, 60, 255)
    assert overlay.width == pytest.approx(3.0)
    assert overlay.symbol == "x"
    assert overlay.symbol_size == pytest.approx(11.0)
    assert target.dense_groups[0].gain == pytest.approx(2.75)
    assert target.dense_groups[0].step == 2
    assert target.dense_groups[0].traces_per_page == 2
    assert target.raster_series[0].color == (70, 80, 90)
    assert target.heatmap_series[0].vmin == pytest.approx(12.0)
    assert target.heatmap_series[0].vmax == pytest.approx(900.0)
    assert target.heatmap_series[0].colormap == "plasma"
    assert target.heatmap_series[0].decim_method == "mean"
    assert target.heatmap_series[0].shade_nans == (62, 23, 21, 0.85)
    assert target._resolved_event_styles["stim"]["line_color"] == (101, 102, 103)
    assert target._resolved_event_styles["stim"]["line_style"] == "dashed"
    y_range = target.plots[alpha_i].getViewBox().viewRange()[1]
    assert y_range == pytest.approx([-4.0, 6.0])
    assert target.subplot_order == [
        ("heatmap", 0),
        ("ts", beta_i),
        ("raster", 0),
        ("dense", 0),
        ("ts", alpha_i),
    ]
    _close(target)


def test_explicit_id_falls_back_to_local_plot_when_display_name_changes():
    source = view(
        TraceConfig(
            _trace("recording-one", "old-channel"),
            array_name=True,
            view_id="stable-signal",
        ),
        state_definitions=_STATE_DEFS,
    )
    source.plot_height_factors[0] = 1.8
    config = source.capture_view_config()
    _close(source)

    target = view(
        TraceConfig(
            _trace("recording-two", "new-channel"),
            array_name=True,
            view_id="stable-signal",
        ),
        state_definitions=_STATE_DEFS,
    )
    report = target.apply_view_config(config, strict=True)
    assert target.plot_height_factors[0] == pytest.approx(1.8)
    assert len(report.fallback_matches) == 1
    assert not report.unmatched_saved
    assert not report.unmatched_current
    _close(target)


def test_strict_mismatch_is_rejected_before_mutation():
    source = view(
        [
            TraceConfig(_trace("one", "A"), array_name=True, view_id="one"),
            TraceConfig(_trace("two", "B"), array_name=True, view_id="two"),
        ],
        state_definitions=_STATE_DEFS,
    )
    source.window_len = 17.0
    config = source.capture_view_config()
    _close(source)

    target = view(
        TraceConfig(_trace("one", "A"), array_name=True, view_id="one"),
        window_len=5.0,
        state_definitions=_STATE_DEFS,
    )
    with pytest.raises(ViewConfigError, match="Strict View-Config") as exc_info:
        target.apply_view_config(config, strict=True)
    assert target.window_len == pytest.approx(5.0)
    assert exc_info.value.report.unmatched_saved == ["ts:two: B"]

    report = target.apply_view_config(config)
    assert target.window_len == pytest.approx(17.0)
    assert report.unmatched_saved == ["ts:two: B"]
    assert not report.is_exact
    _close(target)


def test_marker_ids_survive_marker_reordering():
    data = _dense()
    mask_a = xr.zeros_like(data, dtype=bool)
    mask_b = xr.zeros_like(data, dtype=bool)
    mask_a.values[0, 20] = True
    mask_b.values[1, 30] = True
    first = SampleMarkers("o", "red", mask_a, view_id="peaks")
    second = SampleMarkers("x", "blue", mask_b, view_id="troughs")
    source = view(
        TraceConfig(
            data,
            mode="dense",
            sample_markers=[first, second],
            view_id="dense",
        ),
        state_definitions=_STATE_DEFS,
    )
    peaks, troughs = source.dense_groups[0].sample_markers
    peaks.color, peaks.size, peaks.alpha = (1, 2, 3, 255), 13.0, 77
    troughs.color, troughs.size, troughs.alpha = (4, 5, 6, 255), 14.0, 88
    config = source.capture_view_config()
    _close(source)

    target = view(
        TraceConfig(
            data,
            mode="dense",
            sample_markers=[second, first],
            view_id="dense",
        ),
        state_definitions=_STATE_DEFS,
    )
    report = target.apply_view_config(config, strict=True)
    assert report.is_exact
    by_id = {m.view_id: m for m in target.dense_groups[0].sample_markers}
    assert (by_id["peaks"].color, by_id["peaks"].size, by_id["peaks"].alpha) == (
        (1, 2, 3, 255),
        13.0,
        77,
    )
    assert (
        by_id["troughs"].color,
        by_id["troughs"].size,
        by_id["troughs"].alpha,
    ) == ((4, 5, 6, 255), 14.0, 88)
    _close(target)


def test_zip_curve_ids_survive_source_reordering():
    t = np.linspace(0.0, 10.0, 101)
    first = xr.DataArray(
        np.sin(t)[None, :],
        dims=("synapse", "time"),
        coords={"synapse": [1], "time": t},
        name="first",
    )
    second = xr.DataArray(
        np.cos(t)[None, :],
        dims=("synapse", "time"),
        coords={"synapse": [1], "time": t},
        name="second",
    )
    source = view(
        Zip(
            [
                TraceConfig(first, view_id="first-source"),
                TraceConfig(second, view_id="second-source"),
            ],
            on="synapse",
            colors=[(11, 12, 13, 255), (21, 22, 23, 255)],
            view_id="zip-plots",
        ),
        state_definitions=_STATE_DEFS,
    )
    config = source.capture_view_config()
    _close(source)

    target = view(
        Zip(
            [
                TraceConfig(second, view_id="second-source"),
                TraceConfig(first, view_id="first-source"),
            ],
            on="synapse",
            colors=["white", "white"],
            view_id="zip-plots",
        ),
        state_definitions=_STATE_DEFS,
    )
    report = target.apply_view_config(config, strict=True)
    assert report.is_exact
    assert target.overlay_colors[0] == (21, 22, 23, 255)
    assert target.overlay_colors[1] == (11, 12, 13, 255)
    _close(target)


def test_session_and_tuner_are_opt_in_and_apply():
    scale = Param(1.0, 0.5, 3.0, name="scale")
    raw = _trace("tuned", "A")

    def scaled(values, amount):
        return values * amount

    window = view(
        TraceConfig(
            raw,
            array_name=True,
            overlay_arrays=[tunable(scaled, raw, scale)],
            view_id="tuned",
        ),
        state_definitions=_STATE_DEFS,
    )
    scale.value = 2.25
    window.window_start = 6.0
    window.cursor_time = 7.0
    default = window.capture_view_config()
    assert default.session is None
    assert default.tuner is None

    config = window.capture_view_config(include_session=True, include_tuner=True)
    assert config.session is not None
    assert config.tuner[0]["value"] == pytest.approx(2.25)
    scale.value = 1.0
    window.window_start = 0.0
    window.cursor_time = 0.0
    report = window.apply_view_config(config, strict=True)
    assert report.is_exact
    assert scale.value == pytest.approx(2.25)
    assert window.window_start == pytest.approx(6.0)
    assert window.cursor_time == pytest.approx(7.0)
    _close(window)


def test_duplicate_or_blank_semantic_ids_are_rejected():
    data = _trace("signal", "A")
    with pytest.raises(ValueError, match="Duplicate plot Config view_id"):
        view(
            [TraceConfig(data, view_id="same"), TraceConfig(data, view_id="same")],
            state_definitions=_STATE_DEFS,
        )
    with pytest.raises(ValueError, match="SampleMarkers.view_id"):
        view(
            TraceConfig(
                data,
                sample_markers=[
                    SampleMarkers(
                        "o", "red", xr.zeros_like(data, dtype=bool), view_id=""
                    )
                ],
            ),
            state_definitions=_STATE_DEFS,
        )


def test_partial_config_without_order_keys_preserves_interleaved_layout(_qapp):
    """A heights-only View-Config (no "order" per record) must not reorder the
    subplots; explicit "order" keys still do."""
    import numpy as np
    import xarray as xr

    from loupe import HeatmapConfig, TraceConfig, view
    from loupe.view_config import ViewConfig

    t = np.linspace(0.0, 10.0, 200)

    def _tr(name):
        return TraceConfig(
            xr.DataArray(np.sin(t), dims=("time",), coords={"time": t}, name=name)
        )

    def _hm(name):
        return HeatmapConfig(
            xr.DataArray(
                np.random.default_rng(0).normal(size=(4, len(t))),
                dims=("row", "time"),
                coords={"row": np.arange(4), "time": t},
                name=name,
            )
        )

    interleaved = [("ts", 0), ("heatmap", 0), ("ts", 1), ("heatmap", 1)]
    w = view([_tr("a"), _hm("h1"), _tr("b"), _hm("h2")])
    try:
        assert w.subplot_order == interleaved

        cfg = w.capture_view_config().to_dict()
        for p in cfg["plots"]:
            p.pop("order")
            p["height"] = 0.5
        w.apply_view_config(ViewConfig.from_dict(cfg))
        assert w.subplot_order == interleaved  # heights-only: layout untouched

        cfg2 = w.capture_view_config().to_dict()
        # capture inventories plots type-segregated (ts a, ts b, h1, h2);
        # reversing those positions puts h2 first, then h1, b, a.
        for i, p in enumerate(cfg2["plots"]):
            p["order"] = len(cfg2["plots"]) - 1 - i  # reverse explicitly
        w.apply_view_config(ViewConfig.from_dict(cfg2))
        assert w.subplot_order == [("heatmap", 1), ("heatmap", 0), ("ts", 1), ("ts", 0)]
    finally:
        w.close()
