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


# ---------------------------------------------------------------------------
# horizontal_separators
# ---------------------------------------------------------------------------


def _rows_df(rows, per_row: int = 5, grp: str | None = None, seed: int = 0):
    """Build an events DataFrame in which every value in *rows* is a present
    raster row (so ``unique(source_id) == sorted(set(rows))``)."""
    rng = np.random.default_rng(seed)
    src = np.repeat(list(rows), per_row)
    cols = {
        "time": np.sort(rng.uniform(0, 30, len(src))),
        "source_id": src.astype(np.int64),
    }
    if grp is not None:
        cols["grp"] = np.array([grp] * len(src))
    return pl.DataFrame(cols)


def test_separators_disabled_is_byte_identical():
    # Without the arg, nothing about the legacy layout changes.
    w = view(
        RasterConfig(_rows_df(range(10)), time_col="time", order_by="source_id"),
        state_definitions=_EXAMPLE_STATE_DEFS,
    )
    ms = w.raster_series[0]
    assert np.issubdtype(ms.yvals.dtype, np.integer)
    assert ms.separator_lines is None
    assert ms.y_extent is None
    assert ms.separator_color is None and ms.separator_width is None
    # Registry is populated (one entry per subplot) but holds no lines.
    assert len(w.raster_separator_lines) == len(w.raster_series)
    assert w.raster_separator_lines[0] == []
    w.close()


def test_separators_shift_and_lines():
    # 10 rows (source_id 0..9), one separator at value 5, default gap 0.6.
    w = view(
        RasterConfig(
            _rows_df(range(10)),
            time_col="time",
            order_by="source_id",
            horizontal_separators=[5],
        ),
        state_definitions=_EXAMPLE_STATE_DEFS,
    )
    ms = w.raster_series[0]
    assert np.issubdtype(ms.yvals.dtype, np.floating)
    assert ms.separator_lines == pytest.approx([5.3])  # 5 + 0.6/2
    assert ms.y_extent == pytest.approx(10.6)          # 10 rows + one 0.6 gap
    assert float(ms.yvals.max()) == pytest.approx(9.6)  # top row shifted by 0.6
    # No event row-center lands inside the empty gap band [5.0, 5.6].
    centers = ms.yvals + 0.5
    assert not np.any((centers > 5.0) & (centers < 5.6))
    # The rendered line handles match the recorded positions.
    handles = w.raster_separator_lines[0]
    assert len(handles) == 1
    assert isinstance(handles[0], pg.InfiniteLine)
    w.close()


def test_separators_respect_split_by():
    # Group "A" straddles value 5; group "B" (rows 0..2) does not.
    df = pl.concat([_rows_df(range(10), grp="A"), _rows_df(range(3), grp="B")])
    w = view(
        RasterConfig(
            df,
            time_col="time",
            order_by="source_id",
            split_by="grp",
            horizontal_separators=[5],
        ),
        state_definitions=_EXAMPLE_STATE_DEFS,
    )
    by_rows = {ms.n_rows: ms for ms in w.raster_series}
    assert by_rows[10].separator_lines == pytest.approx([5.3])  # straddling group
    assert by_rows[3].separator_lines is None                   # non-straddling group
    assert np.issubdtype(by_rows[3].yvals.dtype, np.integer)
    w.close()


def test_separators_out_of_range_noop():
    # Below-min (-1), above-max (100), and equal-to-min (0) all resolve to
    # boundary 0 or n_rows, which are dropped -> behaves exactly as disabled.
    w = view(
        RasterConfig(
            _rows_df(range(4)),
            time_col="time",
            order_by="source_id",
            horizontal_separators=[-1, 100, 0],
        ),
        state_definitions=_EXAMPLE_STATE_DEFS,
    )
    ms = w.raster_series[0]
    assert np.issubdtype(ms.yvals.dtype, np.integer)
    assert ms.separator_lines is None and ms.y_extent is None
    assert w.raster_separator_lines[0] == []
    w.close()


def test_separator_params_override():
    w = view(
        RasterConfig(
            _rows_df(range(10)),
            time_col="time",
            order_by="source_id",
            horizontal_separators=[5],
            separator_params={"gap": 1.0, "color": "#ff0000", "width": 2.5},
        ),
        state_definitions=_EXAMPLE_STATE_DEFS,
    )
    ms = w.raster_series[0]
    assert ms.separator_lines == pytest.approx([5.5])  # 5 + 1.0/2
    assert ms.y_extent == pytest.approx(11.0)          # 10 rows + one 1.0 gap
    assert ms.separator_color == (255, 0, 0)
    assert ms.separator_width == 2.5
    assert len(w.raster_separator_lines[0]) == len(ms.separator_lines)
    w.close()


def test_separator_params_unknown_key_warns():
    with pytest.warns(UserWarning, match="unknown key"):
        w = view(
            RasterConfig(
                _rows_df(range(6)),
                time_col="time",
                order_by="source_id",
                horizontal_separators=[3],
                separator_params={"bogus": 1},
            ),
            state_definitions=_EXAMPLE_STATE_DEFS,
        )
    w.close()
