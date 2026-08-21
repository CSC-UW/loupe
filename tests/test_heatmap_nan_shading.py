"""Validation and renderer coverage for ``HeatmapConfig.shade_nans``."""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from loupe import HeatmapConfig
from loupe.app import LoupeApp
from loupe.xr_loader import dataarray_to_heatmaps


def _heat() -> xr.DataArray:
    return xr.DataArray(
        np.asarray([
            [np.nan, 1.0, np.nan, 2.0],
            [0.0, np.nan, 3.0, np.nan],
        ]),
        dims=("row", "time"),
        coords={"row": [0, 1], "time": [0.0, 1.0, 2.0, 3.0]},
    )


class _ImageItem:
    def __init__(self) -> None:
        self.image = None
        self.rect = None

    def setImage(self, image, *, autoLevels):
        assert autoLevels is False
        self.image = np.array(image, copy=True)

    def setRect(self, rect) -> None:
        self.rect = rect

    def clear(self) -> None:
        self.image = None


class _ViewBox:
    def width(self) -> float:
        return 100.0


class _Plot:
    def getViewBox(self) -> _ViewBox:
        return _ViewBox()


def _render(shade_nans=False):
    series = dataarray_to_heatmaps(
        _heat(),
        vmin=0.0,
        vmax=3.0,
        shade_nans=shade_nans,
    )[0]
    image_item = _ImageItem()
    app = SimpleNamespace(
        heatmap_series=[series],
        heatmap_visible=[True],
        heatmap_image_items=[image_item],
        heatmap_plots=[_Plot()],
        _heatmap_cache_keys=[None],
        window_start=0.0,
        window_len=3.0,
    )
    for name in (
        "_is_heatmap_plot_visible",
        "_decimate_along_time",
        "_slice_array_at_window",
        "_refresh_heatmap_plots",
    ):
        setattr(app, name, MethodType(getattr(LoupeApp, name), app))

    lut = np.column_stack([
        np.arange(256, dtype=np.uint8),
        np.zeros(256, dtype=np.uint8),
        np.zeros(256, dtype=np.uint8),
        np.full(256, 255, dtype=np.uint8),
    ])
    app._get_array_lut = lambda _cmap: lut
    app._refresh_heatmap_plots()
    return series, image_item.image


def test_shade_nans_default_is_disabled_and_rendering_is_unchanged():
    config = HeatmapConfig(_heat())
    assert config.shade_nans is False

    series, image = _render()
    assert series.shade_nans is None
    assert np.all(image[np.isnan(_heat().values)] == (0, 0, 0, 255))


def test_hex_color_uses_default_alpha_and_only_recolors_nans():
    _, baseline = _render()
    series, shaded = _render("#3E1715")
    nan_mask = np.isnan(_heat().values)

    assert series.shade_nans == (62, 23, 21, 0.7)
    assert np.all(shaded[nan_mask] == (62, 23, 21, 179))
    assert np.array_equal(shaded[~nan_mask], baseline[~nan_mask])


def test_tuple_color_uses_explicit_alpha():
    series, shaded = _render(("#3E1715", 0.85))
    assert series.shade_nans == (62, 23, 21, 0.85)
    assert np.all(shaded[np.isnan(_heat().values)] == (62, 23, 21, 217))


@pytest.mark.parametrize(
    "value, match",
    [
        (True, "must be False"),
        ("3E1715", "#RRGGBB"),
        ("#nothex", "#RRGGBB"),
        (("#3E1715", -0.01), "between 0 and 1"),
        (("#3E1715", 1.01), "between 0 and 1"),
        (("#3E1715", float("nan")), "between 0 and 1"),
        (("#3E1715", True), "number between 0 and 1"),
        (("#3E1715",), "must be False"),
    ],
)
def test_invalid_shade_nans_values_fail_at_config_construction(value, match):
    with pytest.raises(ValueError, match=match):
        HeatmapConfig(_heat(), shade_nans=value)


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_alpha_range_is_inclusive(alpha):
    config = HeatmapConfig(_heat(), shade_nans=("#000000", alpha))
    assert config.shade_nans == ("#000000", alpha)
