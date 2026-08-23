"""Live Tuner support for ``RasterConfig.data`` (binding capture + recompute).

Mirrors ``test_tuner_live.py``: real offscreen windows, debounce bypassed by
calling ``_flush_tuner`` directly.
"""

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


def _catalog() -> pl.DataFrame:
    """Events for 4 synapses; synapse 3 only has weak (z < 4) events."""
    rng = np.random.default_rng(0)
    syn = np.repeat([0, 1, 2, 3], 50)
    z = rng.uniform(3.0, 12.0, size=syn.size)
    z[syn == 3] = rng.uniform(3.0, 3.9, size=(syn == 3).sum())
    return pl.DataFrame(
        {
            "time": rng.uniform(0.0, 30.0, size=syn.size),
            "syn": syn,
            "z": z,
            "grp": np.where(syn < 2, "a", "b"),
        }
    ).sort("time")


def _filter(df: pl.DataFrame, z: float = 3.0) -> pl.DataFrame:
    return df.filter(pl.col("z") >= float(z))


def test_tunable_raster_captures_binding_and_recomputes_with_pinned_rows():
    zt = Param(3.0, 3.0, 15.0, name="z")
    cfg = RasterConfig(
        tunable(_filter, _catalog(), z=zt),
        time_col="time",
        order_by="syn",
        rows=[0, 1, 2, 3],
    )
    w = view(cfg)
    try:
        assert len(w._tuner_bindings) == 1
        b = w._tuner_bindings[0]
        assert b.kind == "raster" and b.series_slice == slice(0, 1)
        assert w._tuner_params == [zt]
        s0 = w.raster_series[0]
        assert s0.n_rows == 4 and len(s0.timestamps) == 200

        zt.value = 4.0  # synapse 3 loses every event; its row must stay

        w._on_tuner_param_changed(zt)
        w._flush_tuner()
        s1 = w.raster_series[0]
        assert s1.n_rows == 4 and s1.row_keys.tolist() == [0, 1, 2, 3]
        assert 0 < len(s1.timestamps) < 200
        assert set(np.unique(s1.yvals).tolist()) == {0, 1, 2}
        assert np.all(np.diff(s1.timestamps) >= 0)

        zt.value = 100.0  # nothing survives -> empty raster, geometry intact

        w._on_tuner_param_changed(zt)
        w._flush_tuner()
        s2 = w.raster_series[0]
        assert s2.n_rows == 4 and len(s2.timestamps) == 0

        zt.value = 3.0

        w._on_tuner_param_changed(zt)
        w._flush_tuner()
        assert len(w.raster_series[0].timestamps) == 200
    finally:
        w.close()


def test_rows_are_pinned_from_initial_render_without_explicit_rows():
    zt = Param(3.0, 3.0, 15.0, name="z")
    w = view(
        RasterConfig(tunable(_filter, _catalog(), z=zt), time_col="time", order_by="syn")
    )
    try:
        assert w.raster_series[0].n_rows == 4
        zt.value = 4.0
        w._on_tuner_param_changed(zt)
        w._flush_tuner()
        s = w.raster_series[0]
        # synapse 3 vanished from the data but keeps its row; others keep index
        assert s.n_rows == 4 and s.row_keys.tolist() == [0, 1, 2, 3]
        assert set(np.unique(s.yvals).tolist()) == {0, 1, 2}
    finally:
        w.close()


def test_split_groups_matched_by_name_and_vanished_group_renders_empty():
    zt = Param(3.0, 3.0, 15.0, name="z")
    cfg = RasterConfig(
        tunable(_filter, _catalog(), z=zt),
        time_col="time",
        order_by="syn",
        split_by="grp",
    )
    w = view(cfg)
    try:
        assert [s.name for s in w.raster_series] == ["a", "b"]
        assert w._tuner_bindings[0].series_slice == slice(0, 2)
        zt.value = 4.0
        w._on_tuner_param_changed(zt)
        w._flush_tuner()
        a, b = w.raster_series
        assert a.n_rows == 2 and b.n_rows == 2  # pinned from initial render
        assert len(b.timestamps) > 0 and set(np.unique(b.yvals).tolist()) == {0}
        # raise until group 'b' (syn 2 max z < 12) disappears entirely
        zt.value = 12.0
        w._on_tuner_param_changed(zt)
        w._flush_tuner()
        a, b = w.raster_series
        assert [s.name for s in w.raster_series] == ["a", "b"]
        assert len(b.timestamps) == 0 and b.n_rows == 2
    finally:
        w.close()


def test_shared_param_drives_multiple_raster_configs():
    zt = Param(3.0, 3.0, 15.0, name="z")
    cat = _catalog()
    w = view(
        [
            RasterConfig(tunable(_filter, cat, z=zt), time_col="time", order_by="syn"),
            RasterConfig(tunable(_filter, cat, z=zt), time_col="time", order_by="syn"),
        ]
    )
    try:
        assert [b.series_slice for b in w._tuner_bindings] == [slice(0, 1), slice(1, 2)]
        assert w._tuner_params == [zt]
        assert all(len(s.timestamps) == 200 for s in w.raster_series)
        zt.value = 100.0
        w._on_tuner_param_changed(zt)
        w._flush_tuner()
        assert all(len(s.timestamps) == 0 for s in w.raster_series)
    finally:
        w.close()


def test_initially_empty_raster_with_rows_exists_and_fills_when_tuned_down():
    zt = Param(13.0, 3.0, 15.0, name="z")  # above every event's z at launch
    cfgs = [
        RasterConfig(
            tunable(_filter, _catalog(), z=zt),
            time_col="time",
            order_by="syn",
            rows=[0, 1, 2, 3],
            array_name="empty-at-start",
        ),
        RasterConfig(_catalog(), time_col="time", order_by="syn", array_name="plain"),
    ]
    w = view(cfgs)
    try:
        assert [s.name for s in w.raster_series] == ["empty-at-start", "plain"]
        assert len(w.raster_plots) == 2
        s = w.raster_series[0]
        assert s.n_rows == 4 and len(s.timestamps) == 0
        assert w._tuner_bindings[0].series_slice == slice(0, 1)
        zt.value = 3.0
        w._on_tuner_param_changed(zt)
        w._flush_tuner()
        s = w.raster_series[0]
        assert len(s.timestamps) == 200 and s.n_rows == 4
        assert len(w.raster_series[1].timestamps) == 200  # untouched neighbour
    finally:
        w.close()
