"""Public-API configuration dataclasses for :func:`loupe.view`.

Users construct one of these per data source and pass them into
:func:`loupe.view`, which dispatches each Config to its converter and
hands the result to :class:`loupe.app.LoupeApp`.

Re-exported from :mod:`loupe`, so the canonical import is
``from loupe import TraceConfig, HeatmapConfig, RasterConfig, ...``.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import polars as pl
    import xarray as xr
    from matplotlib.colors import Colormap

    from loupe.tuner import Tunable


@dataclass
class SampleMarkers:
    """Marker symbols stamped onto specific samples of a :class:`TraceConfig`.

    Pass one or more in :attr:`TraceConfig.sample_markers` to overlay spike-,
    seizure-, or other sample-aligned markers on the trace they annotate.
    Supported in both ``mode="stacked-subplots"`` and ``mode="dense"``. In
    dense mode markers are drawn at the *displayed* y (``(value − mean) × gain
    + offset``), so they track the traces as gain changes; each marker set is a
    single color (no hue tinting). Stacked mode allows only one marker-carrying
    ``TraceConfig`` per window, while dense markers are unrestricted (multiple
    carriers, free coexistence with heatmaps / rasters / stacked traces).

    Parameters
    ----------
    marker : str
        Marker symbol.  ``'o'`` renders a semi-transparent filled circle;
        ``'x'`` renders a solid X.  Other pyqtgraph symbols (``'+'``,
        ``'s'``, ``'t'``, …) fall through with a default outlined style.
    color : str or RGB(A) tuple
        Marker color — named string, hex (``"#RRGGBB"``), or
        ``(R, G, B[, A])`` tuple.
    bool_array : xr.DataArray or Tunable or Callable
        Boolean DataArray whose dims/shape match the parent
        :class:`TraceConfig`'s ``data`` (for an ``N``-trace × ``S``-sample
        config, an ``(N, S)`` array with the same dims).  ``True`` at sample
        *i* on trace *j* draws a marker at that trace's value at that
        timepoint.  Aligned to the traces by coordinate label and the same
        ``order_by`` / ``descending`` ordering as the data. In stacked-subplot
        mode this can instead be a :func:`loupe.tunable` wrapper (or a bare
        zero-argument callable returning that Boolean DataArray), so the
        markers move live with the Tuner. Dense-mode marker masks are static.
    size : float, optional
        Marker size in points.  ``None`` (default) picks 8.0 for ``'o'``
        and 9.0 for other markers.
    alpha : int, optional
        Marker alpha in ``0..255``.  ``None`` (default) picks 110 for
        ``'o'`` (semi-transparent fill) and 255 for other markers
        (solid stroke).
    view_id : str or None
        Optional stable identity used when a View-Config is replayed against
        an analogous recording. Recommended when several marker sets could
        otherwise have the same symbol.
    """

    marker: str
    color: "str | tuple"
    bool_array: "xr.DataArray | Tunable | Callable"
    size: float | None = None
    alpha: int | None = None
    view_id: str | None = None


@dataclass
class TraceConfig:
    """Per-DataArray display configuration for line-plot views in :func:`view`.

    For 2-D heatmap views, use :class:`HeatmapConfig` instead.

    Parameters
    ----------
    data : xr.DataArray or Tunable
        The DataArray to display.  May instead be a :func:`loupe.tunable`
        wrapper (or a bare zero-arg callable returning a DataArray) whose
        scalar :class:`loupe.Param` arguments become live sliders in the
        Tuner panel — the trace then recomputes as you tune.
    mode : str
        ``"stacked-subplots"`` (default) for one subplot per trace, or
        ``"dense"`` for EEG-style offset traces on a single axis.
    order_by : str or None
        Coordinate name to control trace ordering and spacing.
    descending : bool
        Reverse the ordering given by *order_by*.
    gain : float
        Initial amplitude gain multiplier (dense mode only).
    step : int
        Show every *step*-th trace (dense mode only).
    traces_per_page : int or None
        How many traces to show at once in dense mode. ``None`` = all.
        Use Alt+scroll to page through the rest.
    hue : str or None
        Coordinate name whose categorical values determine per-trace color.
        Overridden by *color* when both are set.
    palette : dict, list, or None
        Per-*hue*-value color mapping.  ``dict {value: color}`` keys each
        unique *hue* value to a color; a list of colors is assigned in
        sorted *hue*-value order.  ``None`` falls back to a default palette.
        Ignored when *hue* is unset.
    color : str, RGB(A) tuple, or None
        Single color applied to every trace produced by this DataArray,
        e.g. ``"#a020f0"`` or ``(160, 32, 240)``.  Overrides ``hue`` /
        ``palette`` when both are set.
    line_width : float
        Line width (in pixels) of this config's own trace(s).  Default
        ``1.0``.  Stacked-subplots mode only.
    overlay_line_widths : list[float] or None
        One line width (pixels) per entry in *overlay_arrays*.  ``None``
        (default) draws every overlay at width ``1.0``; a short list is
        extended with ``1.0``.  Ignored when *overlay_arrays* is unset.
        Stacked-subplots mode only.
    array_name : bool or str
        Controls the array-level component of each trace name.  ``False``
        (default) prepends nothing; multi-trace DataArrays render as just
        the coord value (e.g. ``"CA1-SR"``).  ``True`` uses
        ``data.name`` as the prefix and raises ``ValueError`` if it's
        unset.  A string is used as the prefix verbatim (e.g.
        ``array_name="LFP"`` → ``"LFP: CA1-SR"``).
    sample_markers : list[SampleMarkers] or None
        Optional sample-aligned markers drawn on top of the traces produced by
        this DataArray.  Supported in both ``"stacked-subplots"`` and
        ``"dense"`` mode.  In stacked mode at most one :class:`TraceConfig` per
        window may carry markers and no :class:`HeatmapConfig` /
        :class:`RasterConfig` / :class:`Zip` may appear alongside; dense markers
        have no such restriction (multiple carriers, free coexistence) and are
        drawn at the displayed y so they track the gain. In a stacked view,
        each marker's ``bool_array`` may be a live tunable result. See
        :class:`SampleMarkers`.
    overlay_arrays : list[xr.DataArray] or None
        Extra DataArrays to draw *on the same axes* as this TraceConfig's own
        trace(s), rather than in their own subplots.  Each overlay array must
        share ``data``'s dimensions (same non-time dims; the time axis may
        differ and is sliced independently).  When ``data`` produces several
        traces (a non-time dim), overlay trace *i* is drawn onto subplot *i*,
        so the overlays follow the same ``order_by`` / ``descending`` ordering
        as the host.  Each overlay gets a distinct color (see
        *overlay_colors*) and a legend entry from its ``.name``.
        Stacked-subplots mode only.  Any entry may be a :func:`loupe.tunable`
        wrapper (or bare zero-arg callable) instead of a concrete DataArray, so
        the overlay recomputes live as you drag the matching slider in the
        Tuner panel — this is the canonical tuning target (e.g. a matched
        filter over a raw trace).
    overlay_colors : list or None
        One color per entry in *overlay_arrays* — hex strings (``"#ff0000"``)
        or RGB(A) tuples.  ``None`` (default) cycles a built-in distinct
        palette.  A short list is extended from the palette.  Ignored when
        *overlay_arrays* is unset.
    overlay_symbols : list or None
        One entry per *overlay_arrays* item selecting how that overlay is drawn.
        ``None`` (the default, and the per-entry default) draws a connected
        line; a pyqtgraph symbol string (``"o"``, ``"t"``, ``"x"``, ``"s"``, …)
        draws *unconnected point markers* at the overlay's finite samples
        instead — ideal for stamping landmark points (a trough, a peak) onto a
        trace. A short list is padded with ``None`` (line). Stacked-subplots
        mode only; ignored when *overlay_arrays* is unset.
    overlay_symbol_sizes : list or None
        One marker size (points) per *overlay_arrays* entry, used only where
        *overlay_symbols* names a symbol. ``None`` (default) uses ``8.0``.
    add_bottom_spine : bool
        When ``True`` (stacked-subplots mode only), draw a minimal horizontal
        line at the bottom of each subplot produced by this DataArray, marking
        the subplot boundary.  Purely a visual guide: it has no ticks or labels
        and adds no vertical space, so the tight stacking is unchanged.  The
        line is skipped on the bottom-most subplot, which already shows the full
        time axis.
    view_id : str or None
        Optional stable identity for semantic View-Config matching. Give a
        logical signal the same ID in analogous recordings even if its runtime
        subplot name or input-list position changes.
    """

    data: "xr.DataArray | Tunable | Callable"
    mode: str = "stacked-subplots"
    order_by: str | None = None
    descending: bool = False
    gain: float = 1.0
    step: int = 1
    traces_per_page: int | None = None
    hue: str | None = None
    palette: "dict | list | None" = None
    color: "str | tuple | None" = None
    array_name: bool | str = False
    sample_markers: "list[SampleMarkers] | None" = None
    overlay_arrays: "list[xr.DataArray | Tunable | Callable] | None" = None
    overlay_colors: "list | None" = None
    line_width: float = 1.0
    overlay_line_widths: "list | None" = None
    overlay_symbols: "list | None" = None
    overlay_symbol_sizes: "list | None" = None
    add_bottom_spine: bool = False
    view_id: str | None = None

    @classmethod
    def from_path(
        cls,
        path: str,
        *,
        group: str | None = None,
        variable: str = "data",
        filter_dict: dict | None = None,
        **trace_kwargs,
    ) -> "TraceConfig":
        """Load a DataArray from a zarr / netCDF store and wrap it in a
        :class:`TraceConfig`.

        Parameters
        ----------
        path : str
            Path to a ``.zarr`` directory or a netCDF file.
        group : str or None
            Group within the store (e.g. ``'dmd_2'``).
        variable : str
            Variable name inside the dataset (default ``'data'``).
        filter_dict : dict or None
            Dimension slicing to apply before loading, e.g.
            ``{"syn_id": slice(3, 6), "time": slice(0, 1800)}``.
        **trace_kwargs
            Forwarded to :class:`TraceConfig` (``mode``, ``order_by``, …).

        Returns
        -------
        TraceConfig
        """
        from loupe.xr_loader import load_xarray_from_path

        da = load_xarray_from_path(
            path, group=group, variable=variable, filter_dict=filter_dict
        )
        return cls(data=da, **trace_kwargs)


@dataclass
class HeatmapConfig:
    """Per-DataArray display configuration for 2-D heatmap views in :func:`view`.

    For line-plot views, use :class:`TraceConfig` instead.

    Parameters
    ----------
    data : xr.DataArray
        The DataArray to display as a heatmap (imshow-style) over time.
    array_name : bool, str, or callable
        Controls subplot names.  ``False`` (default) names subplots as
        just ``"{split_by}={split_val}"`` (or an empty string when there's
        no split).  ``True`` uses ``data.name`` as the prefix and raises
        ``ValueError`` if it's unset.  A string is used as the prefix
        verbatim.  A callable ``(split_val, sub_da) -> str`` (or 1-arg
        ``(split_val) -> str``) is invoked per split group and its return
        value is the full subplot name.
    split_by : str or None
        Coordinate or dim name to split into one subplot per unique value
        (e.g. one heatmap per dendrite).
    order_by : str or None
        Coordinate name on the row dim controlling y-axis row order within
        each subplot.
    descending : bool
        Reverse the ordering given by *order_by*.
    cmap : str, Colormap, list, dict, or callable
        Matplotlib colormap name (e.g. ``"magma"``) or a Colormap instance
        (e.g. ``cmcrameri.cm.batlow``).  Also accepts:

        * a list — one entry per split group in iteration order, cycling;
        * a dict ``{split_val: cmap}`` — keyed by split value;
        * a callable ``(split_val, sub_da) -> str | Colormap`` — invoked
          per split group.  1-arg callables ``(split_val) -> ...`` are also
          accepted.
    vmin, vmax : float or None
        Color scale limits.  Default is robust 1–99 percentile per heatmap.
    decim_method : str
        Time-axis decimation when zoomed out. ``"peak"`` (max-absolute per
        bin, preserves transients) or ``"mean"``.
    shade_nans : bool, str, or tuple[str, float]
        ``False`` (default) preserves the existing blank appearance for NaN
        values. Pass a ``"#RRGGBB"`` hex color to shade NaNs at 0.7 alpha, or
        ``("#RRGGBB", alpha)`` to set alpha explicitly between 0 and 1.
    view_id : str or None
        Optional stable identity for semantic View-Config matching.
    """

    data: "xr.DataArray"
    array_name: "bool | str | Callable[..., str]" = False
    split_by: str | None = None
    order_by: str | None = None
    descending: bool = False
    cmap: "str | Colormap | list | dict | Callable[..., Any]" = "magma"
    vmin: float | None = None
    vmax: float | None = None
    decim_method: str = "peak"
    shade_nans: "bool | str | tuple[str, float]" = False
    view_id: str | None = None

    def __post_init__(self) -> None:
        from loupe._heatmap_utils import _normalize_nan_shade

        _normalize_nan_shade(self.shade_nans)


@dataclass
class RasterConfig:
    """Per-DataFrame display configuration for raster plots in :func:`view`.

    Parameters
    ----------
    data : pl.DataFrame
        Polars DataFrame of events.  Must contain *time_col* and *order_by*.
    time_col : str
        Column containing event timestamps in seconds.  Required.
    order_by : str
        Column whose values identify the raster row for each event (rows are
        sorted by unique values of this column).  Required.
    split_by : str, list[str] or None
        Column(s) used to split this DataFrame into separate raster
        subplots.  ``None`` (default) puts all events in a single subplot.
    alpha_by : str or None
        Column for per-event opacity, normalized to *alpha_range*.
    array_name : str or callable
        Array-level component of each subplot label.  ``""`` (default)
        leaves grouped subplots labeled by raw group values (e.g.
        ``"CA1-SR"``) and ungrouped subplots labeled with an empty
        string.  A non-empty string is used as a prefix verbatim
        (e.g. ``array_name="units"`` → ``"units: CA1-SR"``).  Multi-column
        groups join values with ``"-"`` (e.g. ``"imec0-CA1-SR"``).
        A callable ``(group_val, sub_df) -> str`` (or 1-arg
        ``(group_val) -> str``) is invoked per group and its return value
        is the full subplot name.
    hue : str or None
        Column whose values determine per-event color.  When set, each
        event in the raster is colored according to its value in this
        column.  Takes precedence over *color* (which is ignored, with a
        warning, if also set).  Pair with *palette* to control the mapping.
    palette : dict, list, tuple or None
        Per-*hue*-value (or per-group, when *hue* is unset) color mapping:
        dict ``{value: (R,G,B)}`` or ``{value: "#RRGGBB"}``, a list of
        colors assigned in sorted-value order, or a single tuple applied to
        all values.  ``None`` (default) falls back to white (no *hue*) or a
        default palette cycle (with *hue*).  When *hue* and *split_by* are
        both set the mapping is shared across every subplot so the same
        column value always renders as the same color.
    color : str, RGB(A) tuple, or None
        Single color applied to every group produced by this DataFrame
        (e.g. ``"#a020f0"`` or ``(160, 32, 240)``).  Takes precedence over
        *palette* when both are set.  Ignored when *hue* is set.
    alpha_range : tuple[float, float]
        ``(min_alpha, max_alpha)`` for normalizing *alpha_by* values.
    horizontal_separators : list or None
        Values in *order_by* space at which to draw a thin horizontal
        separator line plus a small vertical gap, purely as a visual border
        (e.g. to delimit units recorded on different probes that share one
        raster).  Each value ``v`` draws a separator just *below* the row
        whose *order_by* value is ``v`` (rows with ``order_by >= v`` form the
        block above the line).  Values below all rows, above all rows, or
        landing on an existing boundary are silently ignored.  Composes with
        *split_by*: each subplot resolves the values against its own rows, so
        a value only produces a separator in subplots whose rows straddle it.
        ``None`` (default) draws no separators and leaves the layout
        byte-identical to before.
    separator_params : dict or None
        Optional styling for the separators.  Recognized keys: ``"gap"``
        (vertical gap height in row-units, default ``0.6``), ``"color"``
        (hex string or RGB(A) tuple, default gray ``(120, 120, 120)``), and
        ``"width"`` (line width in pixels, default ``1.0``).  Unknown keys
        emit a warning.  Ignored unless *horizontal_separators* is set.
    rows : sequence or None
        Explicit, ordered set of *order_by* values to render as raster rows
        (row ``i`` is ``rows[i]``).  Rows with no events are still drawn, and
        events whose *order_by* value is not listed are dropped.  Use this to
        keep the row layout fixed while *data* is live-tuned (e.g. an event
        catalog filtered by a :class:`loupe.Param` threshold).  ``None``
        (default) derives rows from the values present in *data*.
    nan_spans : list[tuple[float, float]] or dict or None
        Time spans where the *source signal* behind the events was NaN (a raster
        cannot know this from the events alone). Either a list of
        ``(t_start, t_end)`` spans shaded across the full subplot height, or
        ``{order_by_value: [(t_start, t_end), ...]}`` for per-row shading (keys
        resolved against the rendered rows). Only drawn when *shade_nans* is
        set. Default ``None``.
    shade_nans : bool, str, or tuple[str, float]
        ``False`` (default) draws nothing. Pass a ``"#RRGGBB"`` hex color to
        shade *nan_spans* at 0.7 alpha, or ``("#RRGGBB", alpha)`` to set alpha
        explicitly between 0 and 1. Same semantics as ``HeatmapConfig``.
    view_id : str or None
        Optional stable identity for semantic View-Config matching.

    Notes
    -----
    *data* may be a :func:`loupe.tunable` (or a bare zero-arg callable)
    returning a DataFrame; its :class:`loupe.Param` arguments become live
    Tuner sliders and the raster re-renders when they move.  Subplot groups
    are matched by name between re-evaluations (a group with no remaining
    events renders empty), and rows stay pinned to the initial render (or
    to *rows* when given).
    """

    data: "pl.DataFrame"
    time_col: str
    order_by: str
    split_by: "str | list[str] | None" = None
    alpha_by: str | None = None
    array_name: "str | Callable[..., str]" = ""
    hue: str | None = None
    palette: "dict | list | tuple | None" = None
    color: "str | tuple | None" = None
    alpha_range: tuple[float, float] = (0.3, 1.0)
    horizontal_separators: "list | None" = None
    separator_params: "dict | None" = None
    rows: "Sequence | None" = None
    nan_spans: "list | dict | None" = None
    shade_nans: "bool | str | tuple[str, float]" = False
    view_id: str | None = None

    def __post_init__(self) -> None:
        from loupe._heatmap_utils import _normalize_nan_shade

        _normalize_nan_shade(self.shade_nans)

    @classmethod
    def from_parquet(
        cls,
        path: "str | list[str]",
        *,
        time_col: str,
        order_by: str,
        **raster_kwargs,
    ) -> "RasterConfig":
        """Load one or more parquet files into a DataFrame and wrap it in a
        :class:`RasterConfig`.

        Parameters
        ----------
        path : str or list[str]
            Path(s) to parquet file(s).  Multiple paths are concatenated.
        time_col : str
            Time column name.  Files using ``"t_sec"`` are auto-renamed to
            *time_col* for backward compatibility.
        order_by : str
            Column whose values identify the raster row for each event.
        **raster_kwargs
            Forwarded to :class:`RasterConfig` (``split_by``, ``alpha_by``, …).

        Returns
        -------
        RasterConfig
        """
        from loupe.df_loader import load_dataframe_from_parquet

        df = load_dataframe_from_parquet(path, time_col=time_col)
        return cls(data=df, time_col=time_col, order_by=order_by, **raster_kwargs)


@dataclass
class VideoConfig:
    """Per-video display configuration for :func:`view`.

    Parameters
    ----------
    video_path : str or list[str]
        Path to a video file readable by OpenCV (.mp4, .avi, .mov, .mkv),
        OR a list of such paths to be displayed as one continuous video
        (concatenated in list order).  When a list is given,
        *frame_times_path* must also be a list of the same length.
    frame_times_path : str or list[str]
        Path to a 1-D ``.npy`` file of per-frame timestamps in seconds, or
        a list of such paths matching *video_path*.  Arrays are
        concatenated as-is — the caller is responsible for putting them on
        a single shared time axis (typically TDT-block-relative seconds).
    name : str or None
        Display label for this video — used as the placeholder text on the
        empty frame and as the entry name in the Show / Frame Step Target
        menus.  Defaults to ``"Video {i+1}"`` based on list position.
    stretch : int or None
        Initial vertical layout weight relative to other videos.  Defaults
        to ``3`` for the first slot and ``2`` for the rest, matching the
        previous hard-coded layout.
    frame_times_correction : float
        Scalar (seconds) added to every frame time after loading.  Applied
        uniformly whether *frame_times_path* is a single file or a list —
        the offset is added once to the (possibly concatenated) array.
        Useful as a quick alignment shim against the trace cursor without
        rewriting the underlying ``.npy`` files.  Defaults to ``0.0``.
    view_id : str or None
        Optional stable identity for matching saved visibility, layout weight,
        and frame-step selection across analogous recordings.
    """

    video_path: "str | list[str]"
    frame_times_path: "str | list[str]"
    name: str | None = None
    stretch: int | None = None
    frame_times_correction: float = 0.0
    view_id: str | None = None


@dataclass
class Zip:
    """Co-plot traces sharing a coordinate value across multiple DataArrays.

    ``Zip([TraceConfig(F), TraceConfig(dFF), TraceConfig(denoised)], on="syn_id")``
    produces one subplot per unique ``syn_id`` value; each subplot holds the
    F, dF/F, and denoised trace for that synapse.  Semantics match Python's
    :func:`zip` along the named dim.

    Parameters
    ----------
    traces : list[TraceConfig]
        Two or more :class:`TraceConfig` instances whose ``data`` shares
        the dim named by *on*.  Only ``color`` and ``view_id`` on each
        TraceConfig apply within a Zip; other fields must remain at their
        defaults (Zip dictates the per-subplot layout).
    on : str
        Coordinate dim to zip on (e.g. ``"syn_id"``).
    colors : list, optional
        One color per wrapped TraceConfig.  If omitted, a default palette
        is used.
    array_name : bool or str
        Optional subplot-name prefix.  ``False`` (default) leaves subplot
        labels as just the overlay dim value.  A string is used as the
        prefix verbatim.  ``True`` is rejected because a Zip wraps
        multiple DataArrays and has no single source name.
    view_id : str or None
        Optional stable identity for semantic View-Config matching.
    """

    traces: list
    on: str
    colors: list | None = None
    array_name: bool | str = False
    view_id: str | None = None

    def __post_init__(self):
        if len(self.traces) < 2:
            raise ValueError("Zip requires at least 2 TraceConfigs")
        if self.array_name is True:
            raise ValueError(
                "Zip(array_name=True) is not supported: a Zip wraps "
                "multiple DataArrays and has no single source name. "
                "Pass an explicit string or leave array_name=False."
            )
        meaningless = {
            "mode": "stacked-subplots",
            "order_by": None,
            "descending": False,
            "gain": 1.0,
            "step": 1,
            "traces_per_page": None,
            "hue": None,
            "palette": None,
            "array_name": False,
            "sample_markers": None,
            "overlay_arrays": None,
            "overlay_colors": None,
        }
        source_view_ids: set[str] = set()
        for i, t in enumerate(self.traces):
            if not isinstance(t, TraceConfig):
                raise TypeError(
                    f"Zip.traces[{i}] must be a TraceConfig, got "
                    f"{type(t).__name__}."
                )
            if t.view_id is not None:
                if not isinstance(t.view_id, str) or not t.view_id.strip():
                    raise ValueError(
                        f"Zip.traces[{i}].view_id must be a non-empty string."
                    )
                if t.view_id in source_view_ids:
                    raise ValueError(
                        f"Duplicate Zip trace view_id={t.view_id!r}."
                    )
                source_view_ids.add(t.view_id)
            for fname, fdefault in meaningless.items():
                actual = getattr(t, fname)
                if actual != fdefault:
                    raise ValueError(
                        f"Zip.traces[{i}].{fname}={actual!r} is meaningless "
                        f"within a Zip; only `color` applies."
                    )


@dataclass
class GlobalEventsConfig:
    """Vertical event-marker lines drawn across every plot pane in :func:`view`.

    Pass via the ``global_events=`` kwarg on :func:`view` to overlay
    time-locked event markers (stimulus onsets, behavioral events, manually
    noted transitions, etc.) on top of every trace / dense / heatmap / raster
    pane.  Drawn as the topmost layer so they remain visible on top of
    label shading.  Pen color, line style, width, and alpha are editable
    live via the "Style Global Events…" entry in the View menu.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame with one row per event.  Must contain *event_times_column*.
        Additional columns may be referenced via *style_events_on*.
    event_times_column : str
        Column with event times in seconds (default ``"time"``).
    style_events_on : str or None
        Column whose values group events into styled classes.  ``None``
        (default) renders every event with a single style.
    style_kwargs : dict or None
        Per-class style overrides keyed by unique values of
        *style_events_on*.  Each value is a dict with any of:
        ``line_color``, ``line_style``, ``line_width``, ``line_alpha``.
        Unspecified classes fall back to an auto-generated palette that
        cycles through distinct line styles first, then adds colors.
        Ignored (with a warning) when *style_events_on* is ``None``.

        ``line_style`` values: ``"solid"``, ``"dashed"``, ``"dotted"``,
        ``"dashdot"``, ``"dashdotdot"``.

        ``line_color`` accepts an ``(R, G, B)`` tuple or a ``"#RRGGBB"``
        hex string.  ``line_alpha`` is an integer in ``0..255``.
    """

    data: "pl.DataFrame"
    event_times_column: str = "time"
    style_events_on: str | None = None
    style_kwargs: dict | None = None
