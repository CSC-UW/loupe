"""``RasterConfig.rows`` / ``dataframe_to_raster_series(rows=...)`` row pinning."""

import numpy as np
import polars as pl

from loupe.df_loader import dataframe_to_raster_series


def _df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "syn": [10, 10, 12, 12, 13, 99],
            "grp": ["a", "a", "a", "b", "b", "b"],
        }
    )


def test_default_rows_are_the_observed_sorted_values():
    (s,) = dataframe_to_raster_series(_df(), time_col="time", order_by="syn")
    assert s.n_rows == 4
    assert s.row_keys.tolist() == [10, 12, 13, 99]
    assert s.yvals.tolist() == [0, 0, 1, 1, 2, 3]


def test_explicit_rows_pin_layout_keep_empty_rows_and_drop_unlisted():
    rows = [13, 11, 12, 10]  # deliberate non-sorted order; 11 has no events
    (s,) = dataframe_to_raster_series(
        _df(), time_col="time", order_by="syn", rows=rows
    )
    assert s.n_rows == 4
    assert s.row_keys.tolist() == rows
    # syn 99 (not in rows) dropped; remaining events mapped by position in rows
    assert len(s.timestamps) == 5
    assert s.yvals.tolist() == [3, 3, 2, 2, 0]
    assert np.all(np.diff(s.timestamps) > 0)


def test_explicit_rows_apply_per_split_group():
    rows = [10, 12, 13]
    series = dataframe_to_raster_series(
        _df(), time_col="time", order_by="syn", split_by="grp", rows=rows
    )
    assert [s.name for s in series] == ["a", "b"]
    assert all(s.n_rows == 3 for s in series)
    assert all(s.row_keys.tolist() == rows for s in series)
    a, b = series
    assert a.yvals.tolist() == [0, 0, 1]
    assert b.yvals.tolist() == [1, 2]  # 99 dropped


def test_empty_frame_with_rows_yields_empty_pinned_series():
    empty = _df().clear()
    (s,) = dataframe_to_raster_series(
        empty, time_col="time", order_by="syn", rows=[10, 11, 12]
    )
    assert s.n_rows == 3 and s.row_keys.tolist() == [10, 11, 12]
    assert len(s.timestamps) == 0 and len(s.yvals) == 0 and len(s.alphas) == 0
    # without rows (or with split_by) an empty frame still yields nothing
    assert dataframe_to_raster_series(empty, time_col="time", order_by="syn") == []
    assert (
        dataframe_to_raster_series(
            empty, time_col="time", order_by="syn", split_by="grp", rows=[10]
        )
        == []
    )
