"""Tests for `view(RasterConfig(...))` interaction between `hue` and the
single-color `color` / per-group `palette` fields. The precedence rule
(hue wins, with a warning if `color` is also supplied; `color` wins over
`palette` and warns) lives inside `view()`, so we have to construct a real
LoupeApp to exercise it."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import polars as pl
import pyqtgraph as pg
import pytest
from PySide6 import QtWidgets

import loupe.app as _loupe_app
from loupe import RasterConfig, view

_EXAMPLE_STATE_DEFS = os.path.join(
    os.path.dirname(_loupe_app.__file__),
    "example_state_definitions.json",
)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # Avoid sporadic OpenGL flakiness on headless CI.
    pg.setConfigOptions(useOpenGL=False)
    yield app


def _events_df(n: int = 60, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "time": np.sort(rng.uniform(0, 30, n)),
        "source_id": rng.integers(0, 4, n),
        "cell_type": rng.choice(["pyr", "pv"], n),
    })


def test_hue_precedence_over_color():
    df = _events_df()
    with pytest.warns(UserWarning, match="hue takes precedence"):
        w = view(
            RasterConfig(
                df,
                time_col="time",
                order_by="source_id",
                hue="cell_type",
                color="#ff0000",
            ),
            state_definitions=_EXAMPLE_STATE_DEFS,
        )
    ms = w.raster_series[0]
    # Per-event coloring populated; the legacy single-color override was
    # NOT applied (otherwise ms.color would be red AND category_colors None).
    assert ms.category_index is not None
    assert ms.category_colors is not None
    w.close()


def test_hue_uses_palette_mapping():
    df = _events_df()
    w = view(
        RasterConfig(
            df,
            time_col="time",
            order_by="source_id",
            hue="cell_type",
            palette={"pyr": (1, 2, 3), "pv": (10, 20, 30)},
        ),
        state_definitions=_EXAMPLE_STATE_DEFS,
    )
    ms = w.raster_series[0]
    assert ms.category_index is not None
    # category_colors order matches sorted uniques: ["pv", "pyr"]
    assert ms.category_colors[0] == (10, 20, 30)
    assert ms.category_colors[1] == (1, 2, 3)
    w.close()


def test_color_alone_still_applied_when_hue_absent():
    df = _events_df()
    # No warning expected; legacy single-color path should still work.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        w = view(
            RasterConfig(df, time_col="time", order_by="source_id", color="#a020f0"),
            state_definitions=_EXAMPLE_STATE_DEFS,
        )
    ms = w.raster_series[0]
    assert ms.color == (160, 32, 240)
    assert ms.category_index is None
    w.close()


def test_color_warns_over_palette():
    df = _events_df()
    with pytest.warns(UserWarning, match="color takes precedence over palette"):
        w = view(
            RasterConfig(df, time_col="time", order_by="source_id",
                         color="#a020f0", palette={0: (1, 2, 3)}),
            state_definitions=_EXAMPLE_STATE_DEFS,
        )
    ms = w.raster_series[0]
    assert ms.color == (160, 32, 240)
    w.close()
