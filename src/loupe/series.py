"""Runtime data containers used inside :class:`loupe.app.LoupeApp`.

These are distinct from the public-API configuration dataclasses in
:mod:`loupe.configs` — they hold the resolved/converted data that the
viewer actually renders, after :func:`loupe.view` has processed each
:class:`TraceConfig` / :class:`HeatmapConfig` / :class:`RasterConfig`.

Kept Qt-free (the graphics-item references in
:class:`IntervalLabelVisualBundle` are stored as ``Any``-like attributes
populated at runtime) so this module is cheap to import.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg


@dataclass
class Series:
    name: str
    t: np.ndarray  # seconds, monotonic
    y: np.ndarray


@dataclass
class SampleMarkers:
    """Marker symbols stamped onto specific samples of stacked-subplot traces.

    ``bool_per_series`` has one 1-D bool array per :class:`Series`, in the
    same order as :attr:`LoupeApp.series`. ``True`` at sample i means a
    marker is drawn at ``(s.t[i], s.y[i])`` on series s.
    """

    marker: str
    color: str | tuple
    bool_per_series: list[np.ndarray]
    size: float = 8.0
    alpha: int = 255  # 0..255; controls fill alpha for 'o', stroke alpha otherwise


@dataclass
class OverlayCurve:
    """One overlay trace drawn on a stacked-subplots host series' subplot.

    Built from :attr:`loupe.configs.TraceConfig.overlay_arrays`. There is one
    per (overlay array, host series); :attr:`LoupeApp.overlay_series` holds a
    ``list[OverlayCurve]`` per host series, indexed the same as
    :attr:`LoupeApp.series`. ``t`` / ``y`` are 1-D and need not share the host
    series' time axis — each is sliced independently to the current window on
    scroll.
    """

    name: str
    color: "tuple | str"
    t: np.ndarray
    y: np.ndarray


@dataclass
class RasterSeries:
    """Holds data for a raster subplot."""

    name: str
    timestamps: np.ndarray  # 1D array of event times (seconds)
    yvals: np.ndarray  # 1D array of row positions; integer 0..n_rows-1, or float
    # once horizontal separators shift rows apart (see separator_lines).
    alphas: np.ndarray  # 1D array of alpha values (0.0 to 1.0)
    color: tuple  # (R, G, B) fallback color used when category_index is None
    n_rows: int  # logical number of unique rows (NOT max(yvals)+1 when separators exist)
    # Per-event categorical coloring. Both fields must be set together or both
    # left as None; None preserves the legacy single-color fast path.
    category_index: np.ndarray | None = None  # (N,) int16, parallel to timestamps
    category_colors: list[tuple[int, int, int]] | None = None  # one RGB per category index
    # Horizontal separators (opt-in via RasterConfig.horizontal_separators). All
    # left as None preserves the legacy no-gap layout exactly.
    separator_lines: list[float] | None = None  # y-positions of horizontal separator lines
    y_extent: float | None = None  # total vertical extent incl. gaps; None means n_rows
    separator_color: "tuple | str | None" = None  # resolved line color; None -> app default
    separator_width: float | None = None  # line width in px; None -> app default


@dataclass
class HeatmapSeries:
    """Holds data for a heatmap (imshow-style) subplot.

    Represents a single 2-D buffer ``Y`` of shape ``(n_rows, n_time)`` with a
    shared time axis ``t``.  Each row corresponds to one entry of the row dim
    after ``order_by`` ordering; each column corresponds to one time sample.
    """

    name: str
    t: np.ndarray  # 1-D, shape (n_time,), seconds, monotonic
    Y: np.ndarray  # 2-D, shape (n_rows, n_time), float32, NaN→-inf at load
    row_labels: np.ndarray | None  # values of order_by per row (post-sort), or None
    row_dim_name: str  # name of the row dimension (for tooltips / control board)
    colormap: "str" = "magma"  # matplotlib colormap name or Colormap instance
    vmin: float = 0.0
    vmax: float = 1.0
    decim_method: str = "peak"  # "peak" | "mean"
    mipmap_levels: list[np.ndarray] | None = None  # built lazily for big arrays


# A row_id from IntervalLabelSet uniquely identifies a label across edits/merges.
IntervalLabelKey = int


@dataclass
class IntervalLabelVisualBundle:
    """Graphics items used to display a single labelled interval in plot scenes.

    ``start`` / ``end`` / ``label`` record the geometry and state name the
    region items are currently drawn at. The incremental visual sync keys
    bundles by the (merge-stable) ``row_id``, so when an edit shrinks or moves
    a surviving row in place (e.g. a partial overwrite splits an existing
    epoch but keeps its ``row_id``), these fields let the sync detect that the
    already-drawn region is stale and reposition/recolor it instead of leaving
    it covering its old span.
    """

    plot_regions: list[tuple[int, pg.LinearRegionItem]]
    raster_regions: list[tuple[int, pg.LinearRegionItem]]
    dense_regions: list[tuple[int, pg.LinearRegionItem]]
    hypnogram_region: pg.LinearRegionItem | None
    heatmap_regions: list[tuple[int, pg.LinearRegionItem]] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0
    label: str = ""


@dataclass
class DenseGroup:
    """A group of traces rendered on a single dense plot (EEG-style)."""

    name: str
    series: list[Series]
    trace_labels: list[str]
    order_values: np.ndarray | None = None
    color_values: np.ndarray | None = None
    palette: "dict | list | None" = None
    descending: bool = False
    gain: float = 1.0
    step: int = 1
    traces_per_page: int | None = None
    hidden_traces: set[int] = field(default_factory=set)
