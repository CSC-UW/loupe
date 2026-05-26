"""Tests for the `hue` / `palette` features of `dataframe_to_raster_series`.
These exercise the categorical-coloring data plumbing without touching the
renderer."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import warnings

import numpy as np
import polars as pl
import pytest

from loupe.app import RASTER_MAX_CATEGORIES, RASTER_NA_COLOR
from loupe.df_loader import _DEFAULT_COLORS, dataframe_to_raster_series

# Default required-column kwargs for the basic fixture DataFrame.
_REQUIRED = dict(time_col="time", order_by="source_id")


def _basic_df(n: int = 30, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "time": np.sort(rng.uniform(0, 60, n)),
        "source_id": rng.integers(0, 5, n),
        "cat": rng.choice(["a", "b", "c"], n),
        "snr": rng.uniform(0.1, 5.0, n),
    })


def test_hue_basic():
    df = _basic_df()
    out = dataframe_to_raster_series(
        df,
        **_REQUIRED,
        hue="cat",
        palette={
            "a": (255, 0, 0),
            "b": "#00ff00",
            "c": (0, 0, 255),
        },
    )
    assert len(out) == 1
    ms = out[0]
    assert ms.category_index is not None
    assert ms.category_colors is not None
    # Sorted-value order: a, b, c
    assert ms.category_colors == [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    assert ms.category_index.shape == ms.timestamps.shape
    # Parallel after time-sort: spot-check by recovering categories
    assert ms.category_index.dtype == np.int16


def test_hue_missing_palette_no_warning():
    df = _basic_df()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Any warning becomes a test failure.
        out = dataframe_to_raster_series(df, **_REQUIRED, hue="cat")
    ms = out[0]
    # 3 sorted unique values → first 3 entries from the default palette.
    assert ms.category_colors == list(_DEFAULT_COLORS[:3])


def test_hue_partial_palette_warns():
    df = _basic_df()
    with pytest.warns(UserWarning, match="missing entries"):
        out = dataframe_to_raster_series(
            df,
            **_REQUIRED,
            hue="cat",
            palette={"a": (123, 45, 67)},
        )
    ms = out[0]
    # 'a' uses configured color; 'b' and 'c' fall back to default palette[0], [1].
    assert ms.category_colors[0] == (123, 45, 67)
    assert ms.category_colors[1] == _DEFAULT_COLORS[0]
    assert ms.category_colors[2] == _DEFAULT_COLORS[1]


def test_hue_with_alpha_by_preserves_parallel_order():
    rng = np.random.default_rng(1)
    n = 50
    df = pl.DataFrame({
        # Intentionally unsorted: forces argsort to actually reorder all 4
        # parallel arrays. If category_index were not reindexed alongside
        # timestamps, the per-event mapping would be wrong.
        "time": rng.uniform(0, 100, n),
        "source_id": rng.integers(0, 4, n),
        "cat": ["a"] * 25 + ["b"] * 25,
        "snr": rng.uniform(0.1, 1.0, n),
    })
    # Tag each row with its original-order category so we can verify the
    # post-sort category_index matches.
    df = df.with_columns(pl.arange(0, n).alias("orig_idx"))

    out = dataframe_to_raster_series(
        df, **_REQUIRED, hue="cat", alpha_by="snr",
    )
    ms = out[0]

    # Reconstruct: the renderer sees timestamps sorted; for each emitted event
    # we should be able to look up the original cat and confirm category_index.
    order = np.argsort(df["time"].to_numpy())
    expected_cats_sorted = np.array(df["cat"].to_numpy()[order])
    # 'a' < 'b' alphabetically, so a→0, b→1
    expected_idx = np.where(expected_cats_sorted == "a", 0, 1).astype(np.int16)
    assert np.array_equal(ms.category_index, expected_idx)

    # Alphas should be normalized to alpha_range (default 0.3..1.0).
    assert ms.alphas.min() >= 0.3 - 1e-9
    assert ms.alphas.max() <= 1.0 + 1e-9
    assert ms.alphas.shape == ms.timestamps.shape


def test_hue_with_split_by_shared_palette():
    # Two DMD groups; cat=c appears in both but cat=a only in group 1 and
    # cat=b only in group 2. The shared palette must still emit identical
    # category_colors across the two RasterSeries and stable indices.
    df = pl.DataFrame({
        "time":      [0.1, 0.2, 0.3, 1.1, 1.2, 1.3],
        "source_id": [0,   1,   0,   0,   1,   0],
        "dmd":       [1,   1,   1,   2,   2,   2],
        "cat":       ["a", "c", "a", "b", "c", "c"],
    })
    out = dataframe_to_raster_series(
        df,
        **_REQUIRED,
        split_by="dmd",
        hue="cat",
        palette={"a": (255, 0, 0), "b": (0, 255, 0), "c": (0, 0, 255)},
    )
    assert len(out) == 2
    ms1, ms2 = out
    # Identical palette across groups.
    assert ms1.category_colors == ms2.category_colors == [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
    ]
    # 'c' is index 2 in both groups.
    # ms1 rows are (a, c, a) -> indices (0, 2, 0)
    # ms2 rows are (b, c, c) -> indices (1, 2, 2)
    assert list(ms1.category_index) == [0, 2, 0]
    assert list(ms2.category_index) == [1, 2, 2]


def test_hue_with_nulls_appends_na_category():
    df = pl.DataFrame({
        "time":      [0.1, 0.2, 0.3, 0.4],
        "source_id": [0,   0,   1,   1],
        "cat":       ["a", None, "b", None],
    })
    with pytest.warns(UserWarning, match="null values"):
        out = dataframe_to_raster_series(df, **_REQUIRED, hue="cat")
    ms = out[0]
    # 2 real categories (a, b) + 1 NA category appended at the end.
    assert len(ms.category_colors) == 3
    assert ms.category_colors[-1] == RASTER_NA_COLOR
    # Null events get category_index == 2 (NA position).
    assert list(ms.category_index) == [0, 2, 1, 2]


def test_hue_too_many_categories_raises():
    n = RASTER_MAX_CATEGORIES + 1
    df = pl.DataFrame({
        "time": np.linspace(0.0, 1.0, n),
        "source_id": np.arange(n),
        "cat": [f"cat_{i}" for i in range(n)],
    })
    with pytest.raises(ValueError, match="RASTER_MAX_CATEGORIES"):
        dataframe_to_raster_series(df, **_REQUIRED, hue="cat")


def test_hue_missing_column_raises():
    df = _basic_df()
    with pytest.raises(ValueError, match="missing required column"):
        dataframe_to_raster_series(df, **_REQUIRED, hue="not_a_real_column")


def test_hue_hex_strings_in_palette():
    df = _basic_df()
    out = dataframe_to_raster_series(
        df,
        **_REQUIRED,
        hue="cat",
        palette={"a": "#ff8000", "b": "#00ff80", "c": "#8000ff"},
    )
    ms = out[0]
    assert ms.category_colors == [(255, 128, 0), (0, 255, 128), (128, 0, 255)]


def test_hue_none_preserves_legacy_behavior():
    df = _basic_df()
    out = dataframe_to_raster_series(df, **_REQUIRED)  # no hue
    ms = out[0]
    assert ms.category_index is None
    assert ms.category_colors is None
    assert ms.color == (255, 255, 255)  # white default
