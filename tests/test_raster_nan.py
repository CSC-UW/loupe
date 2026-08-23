"""``RasterConfig.nan_spans`` / ``shade_nans`` -> shaded path items behind ticks."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import polars as pl
import pyqtgraph as pg
import pytest
from PySide6 import QtWidgets

from loupe import Param, RasterConfig, tunable, view


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pg.setConfigOptions(useOpenGL=False)
    yield app


def _df() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    syn = np.repeat([0, 1, 2, 3], 30)
    return pl.DataFrame(
        {"time": rng.uniform(0, 30, syn.size), "syn": syn, "z": rng.uniform(3, 9, syn.size)}
    ).sort("time")


def test_full_height_spans_become_one_path_item():
    w = view(
        RasterConfig(
            _df(), time_col="time", order_by="syn",
            nan_spans=[(1.0, 2.0), (10.0, 12.5)], shade_nans=("#C0C0C0", 0.45),
        )
    )
    try:
        assert w.raster_series[0].nan_spans == [(1.0, 2.0), (10.0, 12.5)]
        assert w.raster_series[0].nan_shade == (192, 192, 192, 0.45)
        assert len(w.raster_nan_items) == 1
        item = w.raster_nan_items[0]
        assert isinstance(item, QtWidgets.QGraphicsPathItem)
        rect = item.path().boundingRect()
        assert rect.left() == pytest.approx(1.0) and rect.right() == pytest.approx(12.5)
        assert rect.top() == pytest.approx(-0.5) and rect.bottom() >= 4.0
        assert item.zValue() < -10
        assert item.brush().color().alpha() == round(0.45 * 255)
    finally:
        w.close()


def test_per_row_spans_use_row_keys():
    w = view(
        RasterConfig(
            _df(), time_col="time", order_by="syn", rows=[0, 1, 2, 3],
            nan_spans={0: [(0.0, 1.0)], 3: [(2.0, 3.0)], 99: [(5.0, 6.0)]},
            shade_nans="#FFFF00",
        )
    )
    try:
        rect = w.raster_nan_items[0].path().boundingRect()
        # row 0 -> y in [0, 1]; row 3 -> y in [3, 4]; key 99 ignored (x <= 3)
        assert rect.top() == pytest.approx(0.0) and rect.bottom() == pytest.approx(4.0)
        assert rect.left() == pytest.approx(0.0) and rect.right() == pytest.approx(3.0)
        assert w.raster_series[0].nan_shade[3] == pytest.approx(0.7)
    finally:
        w.close()


def test_no_spans_or_no_shade_means_no_item():
    w = view([
        RasterConfig(_df(), time_col="time", order_by="syn"),
        RasterConfig(_df(), time_col="time", order_by="syn", nan_spans=[(1, 2)]),
        RasterConfig(_df(), time_col="time", order_by="syn", shade_nans="#C0C0C0"),
    ])
    try:
        assert w.raster_nan_items == [None, None, None]
    finally:
        w.close()


def test_invalid_shade_value_rejected_at_construction():
    with pytest.raises(ValueError):
        RasterConfig(_df(), time_col="time", order_by="syn", shade_nans=123)


def test_spans_survive_live_retuning():
    zt = Param(3.0, 3.0, 15.0, name="z")

    def above(df, z=3.0):
        return df.filter(pl.col("z") >= z)

    w = view(
        RasterConfig(
            tunable(above, _df(), z=zt), time_col="time", order_by="syn",
            rows=[0, 1, 2, 3], nan_spans=[(1.0, 2.0)], shade_nans="#C0C0C0",
        )
    )
    try:
        zt.value = 6.0
        w._on_tuner_param_changed(zt)
        w._flush_tuner()
        s = w.raster_series[0]
        assert s.nan_spans == [(1.0, 2.0)] and s.nan_shade is not None
        assert 0 < len(s.timestamps) < 120
    finally:
        w.close()
