"""Loupe: Multi-trace data viewer for neuroscience."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loupe.labels import LabelSchema, LabelSet
from loupe.state_config import StateConfig, load_state_config

if TYPE_CHECKING:
    import polars as pl
    import xarray as xr
    from matplotlib.colors import Colormap

    from loupe.app import LoupeApp

__all__ = [
    "EventLayer",
    "HeatmapConfig",
    "LabelSchema",
    "LabelSet",
    "RasterConfig",
    "StateConfig",
    "TraceConfig",
    "VideoConfig",
    "Zip",
    "view",
]


@dataclass
class EventLayer:
    """Point markers drawn on top of a stacked-subplots :class:`TraceConfig`.

    Pass one or more in :attr:`TraceConfig.event_layers` to overlay spike-,
    seizure-, or other point-event markers on the trace they annotate.

    Parameters
    ----------
    marker : str
        Marker symbol.  ``'o'`` renders a semi-transparent filled circle;
        ``'x'`` renders a solid X.  Other pyqtgraph symbols (``'+'``,
        ``'s'``, ``'t'``, …) fall through with a default outlined style.
    color : str or RGB(A) tuple
        Marker color — named string, hex (``"#RRGGBB"``), or
        ``(R, G, B[, A])`` tuple.
    bool_array : xr.DataArray
        Boolean DataArray whose dims/shape match the parent
        :class:`TraceConfig`'s ``data``.  ``True`` at sample *i* on trace
        *j* draws a marker at ``y = trace_value`` at that timepoint.
    size : float, optional
        Marker size in points.  ``None`` (default) picks 8.0 for ``'o'``
        and 9.0 for other markers.
    alpha : int, optional
        Marker alpha in ``0..255``.  ``None`` (default) picks 110 for
        ``'o'`` (semi-transparent fill) and 255 for other markers
        (solid stroke).
    """

    marker: str
    color: "str | tuple"
    bool_array: "xr.DataArray"
    size: float | None = None
    alpha: int | None = None


@dataclass
class TraceConfig:
    """Per-DataArray display configuration for line-plot views in :func:`view`.

    For 2-D heatmap views, use :class:`HeatmapConfig` instead.

    Parameters
    ----------
    data : xr.DataArray
        The DataArray to display.
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
    color_by : str or None
        Coordinate name whose categorical values determine per-trace color.
    color : str, RGB(A) tuple, or None
        Single color applied to every trace produced by this DataArray,
        e.g. ``"#a020f0"`` or ``(160, 32, 240)``.  Overrides ``color_by``
        when both are set.
    label : str or None
        Display name override.  Used as the trace name (or, for multi-trace
        DataArrays, as the name prefix replacing ``data.name``).
    event_layers : list[EventLayer] or None
        Optional point-event markers drawn on top of the traces produced by
        this DataArray.  Stacked-subplots mode only; at most one
        :class:`TraceConfig` per window may carry event layers, and no
        :class:`HeatmapConfig` / :class:`RasterConfig` / :class:`Zip` may
        appear alongside.
    """

    data: xr.DataArray
    mode: str = "stacked-subplots"
    order_by: str | None = None
    descending: bool = False
    gain: float = 1.0
    step: int = 1
    traces_per_page: int | None = None
    color_by: str | None = None
    color: "str | tuple | None" = None
    label: str | None = None
    event_layers: "list[EventLayer] | None" = None

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
    label : str, callable, or None
        Display name override.  Either a string (used verbatim per subplot,
        no ``"split_on=val"`` suffix) or a callable
        ``(split_val, sub_da) -> str`` invoked once per split group.
        1-arg callables ``(split_val) -> str`` are also accepted.
    split_on : str or None
        Coordinate or dim name to split into one subplot per unique value
        (e.g. one heatmap per dendrite).
    sort_on : str or None
        Coordinate name on the row dim controlling y-axis row order within
        each subplot.
    colormap : str, Colormap, list, dict, or callable
        Matplotlib colormap name (e.g. ``"magma"``) or a Colormap instance
        (e.g. ``cmcrameri.cm.batlow``).  Also accepts:

        * a list — one entry per split group in iteration order, cycling;
        * a dict ``{split_val: cmap}`` — keyed by split value;
        * a callable ``(split_val, sub_da) -> str | Colormap`` — invoked
          per split group.  1-arg callables ``(split_val) -> ...`` are also
          accepted.
    vmin, vmax : float or None
        Color scale limits.  Default is robust 1–99 percentile per array.
    decim_method : str
        Time-axis decimation when zoomed out. ``"peak"`` (max-absolute per
        bin, preserves transients) or ``"mean"``.
    """

    data: xr.DataArray
    label: "str | Callable[..., str] | None" = None
    split_on: str | None = None
    sort_on: str | None = None
    colormap: "str | Colormap | list | dict | Callable[..., Any]" = "magma"
    vmin: float | None = None
    vmax: float | None = None
    decim_method: str = "peak"


@dataclass
class RasterConfig:
    """Per-DataFrame display configuration for raster plots in :func:`view`.

    Parameters
    ----------
    data : pl.DataFrame
        Polars DataFrame of events.  Must contain *time_col* and *y_col*.
    time_col : str
        Column containing event timestamps in seconds (default ``"time"``).
    y_col : str
        Column whose values identify the matrix row for each event
        (default ``"source_id"``).
    group_col : str, list[str] or None
        Column(s) used to split this DataFrame into separate raster
        subplots.  ``None`` (default) puts all events in a single subplot.
    alpha_col : str or None
        Column for per-event opacity, normalized to *alpha_range*.
    name : str
        Base name for the raster subplots (default ``"events"``).  Group
        values are appended when *group_col* is set.
    color : str, RGB(A) tuple, or None
        Single color applied to every group produced by this DataFrame
        (e.g. ``"#a020f0"`` or ``(160, 32, 240)``).  Takes precedence over
        *colors* when both are set.  Ignored when *color_on* is set.
    colors : dict, list, tuple or None
        Per-group palette: dict ``{group_value: (R,G,B)}``, list of tuples
        assigned in sorted group order, or a single tuple applied to all
        groups.  ``None`` (default) means white.  Ignored when *color_on*
        is set.
    alpha_range : tuple[float, float]
        ``(min_alpha, max_alpha)`` for normalizing *alpha_col* values.
    color_on : str or None
        Column whose values determine per-event color.  When set, each
        event in the raster is colored according to its value in this
        column.  Takes precedence over *color* and *colors* (both are
        ignored, with a warning if either is also set).
    color_on_config : dict or None
        Optional ``{column_value: color}`` mapping used by *color_on*.
        Each value may be an ``(R, G, B)`` tuple or a ``"#RRGGBB"`` hex
        string.  Unique values not listed fall back to a default palette
        cycle (with a warning).  ``None`` assigns the entire palette from
        the default cycle (no warning).  When *group_col* is also set the
        mapping is shared across every subplot so the same column value
        always renders as the same color.
    """

    data: "pl.DataFrame"
    time_col: str = "time"
    y_col: str = "source_id"
    group_col: "str | list[str] | None" = None
    alpha_col: str | None = None
    name: str = "events"
    color: "str | tuple | None" = None
    colors: "dict | list | tuple | None" = None
    alpha_range: tuple[float, float] = (0.3, 1.0)
    color_on: str | None = None
    color_on_config: dict | None = None

    @classmethod
    def from_parquet(
        cls,
        path: "str | list[str]",
        *,
        time_col: str = "time",
        **raster_kwargs,
    ) -> "RasterConfig":
        """Load one or more parquet files into a DataFrame and wrap it in a
        :class:`RasterConfig`.

        Parameters
        ----------
        path : str or list[str]
            Path(s) to parquet file(s).  Multiple paths are concatenated.
        time_col : str
            Time column name (default ``"time"``).  Files using ``"t_sec"``
            are auto-renamed for backward compatibility.
        **raster_kwargs
            Forwarded to :class:`RasterConfig` (``y_col``, ``group_col``,
            ``alpha_col``, …).

        Returns
        -------
        RasterConfig
        """
        from loupe.df_loader import load_dataframe_from_parquet

        df = load_dataframe_from_parquet(path, time_col=time_col)
        return cls(data=df, time_col=time_col, **raster_kwargs)


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
    """

    video_path: "str | list[str]"
    frame_times_path: "str | list[str]"
    name: str | None = None
    stretch: int | None = None
    frame_times_correction: float = 0.0


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
        the dim named by *on*.  Only ``color`` and ``label`` on each
        TraceConfig apply within a Zip; other fields must remain at their
        defaults (Zip dictates the per-subplot layout).
    on : str
        Coordinate dim to zip on (e.g. ``"syn_id"``).
    colors : list, optional
        One color per wrapped TraceConfig.  If omitted, a default palette
        is used.
    label : str, optional
        Optional subplot-name prefix.
    """

    traces: list
    on: str
    colors: list | None = None
    label: str | None = None

    def __post_init__(self):
        if len(self.traces) < 2:
            raise ValueError("Zip requires at least 2 TraceConfigs")
        meaningless = {
            "mode": "stacked-subplots",
            "order_by": None,
            "descending": False,
            "gain": 1.0,
            "step": 1,
            "traces_per_page": None,
            "color_by": None,
            "event_layers": None,
        }
        for i, t in enumerate(self.traces):
            if not isinstance(t, TraceConfig):
                raise TypeError(
                    f"Zip.traces[{i}] must be a TraceConfig, got "
                    f"{type(t).__name__}."
                )
            for fname, fdefault in meaningless.items():
                actual = getattr(t, fname)
                if actual != fdefault:
                    raise ValueError(
                        f"Zip.traces[{i}].{fname}={actual!r} is meaningless "
                        f"within a Zip; only `color` and `label` apply."
                    )


def _parse_raster_color(c: "str | tuple") -> tuple[int, int, int]:
    """Normalize a hex string or RGB(A) tuple to an ``(r, g, b)`` 3-tuple.

    Raster rendering (``MatrixSeries.color``) expects exactly 3 channels —
    the alpha is supplied separately per event via ``alpha_col``.
    """
    if isinstance(c, str):
        s = c.strip().lstrip("#")
        if len(s) in (6, 8):
            try:
                return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
            except ValueError:
                pass
        raise ValueError(f"Cannot parse raster color: {c!r}")
    if isinstance(c, (tuple, list)) and len(c) >= 3:
        return (int(c[0]), int(c[1]), int(c[2]))
    raise ValueError(f"Cannot parse raster color: {c!r}")


def view(
    data: "TraceConfig | HeatmapConfig | RasterConfig | Zip | list | None" = None,
    *,
    window_len: float = 10.0,
    # Video sources
    videos: "VideoConfig | list[VideoConfig] | None" = None,
    # Label loading
    labels: "pl.DataFrame | str | Path | None" = None,
    label_schema: LabelSchema | None = None,
    labels_writeback: bool = False,
    # State definitions
    state_definitions: str | Path | None = None,
    keymap: dict | None = None,
    label_colors: dict | None = None,
    label_alpha: float | None = None,
    **kwargs,
) -> LoupeApp:
    """Launch the Loupe viewer.

    Parameters
    ----------
    data : Config or list of Configs, optional
        One or more of :class:`TraceConfig`, :class:`HeatmapConfig`,
        :class:`RasterConfig`, or :class:`Zip`.  A bare scalar Config is
        accepted as shorthand for a one-element list.  List position
        determines top-to-bottom subplot order.

        Bare ``xr.DataArray`` and ``pl.DataFrame`` inputs are **not**
        accepted — wrap them explicitly in ``TraceConfig(da)`` or
        ``RasterConfig(df)``.
    window_len : float
        Initial time window in seconds (default ``10.0``).
    videos : VideoConfig or list[VideoConfig], optional
        Synchronized video sources for the right panel.  A single
        :class:`VideoConfig` is accepted as shorthand for a one-element
        list.  Both ``video_path`` and ``frame_times_path`` may be lists
        of equal length, in which case the files are loaded as one
        continuous (concatenated) video — see :class:`VideoConfig`.
    labels : pl.DataFrame, str, or Path, optional
        Initial labels.  Either a polars DataFrame (requires
        ``label_schema``) or a path to a ``.csv``, ``.htsv``,
        ``.parquet``, or Visbrain ``.txt`` file.
    label_schema : LabelSchema, optional
        Required when ``labels`` is a DataFrame or an ``.htsv``/
        ``.parquet`` file.
    labels_writeback : bool
        If True, the GUI's "Save Labels (overwrite source)" action will
        overwrite the file passed in ``labels``.  Default False.
    state_definitions : str or Path, optional
        Path to a JSON file with ``"keymap"`` and ``"label_colors"``.
    keymap : dict, optional
        Programmatic state hotkeys; overrides ``state_definitions``.
    label_colors : dict, optional
        Programmatic ``state -> color`` map; overrides
        ``state_definitions``.
    label_alpha : float, optional
        Initial label-overlay alpha multiplier in ``[0.0, 1.0]``.
    **kwargs
        Forwarded to :class:`LoupeApp` (``fixed_scale``,
        ``low_profile_x``, etc.).

    Returns
    -------
    LoupeApp
        The viewer window.  In Jupyter (with ``%gui qt6``) the window stays
        alive after the call returns.  In a script the call blocks until
        the window is closed.

    Examples
    --------
    In-memory traces::

        view(TraceConfig(da))

    From a zarr store::

        view(TraceConfig.from_path("data.zarr", group="dmd_2",
                                   filter_dict={"syn_id": slice(3, 6)}))

    DataFrame raster::

        view(RasterConfig(ev, y_col="source_id", group_col="dmd",
                          alpha_col="snr_denoised"))

    Mixed layout (list order = top-to-bottom)::

        view([
            TraceConfig(traces),
            RasterConfig(events, group_col="dmd"),
            HeatmapConfig(dff, split_on="dend-ID", sort_on="pos"),
        ])

    Zip (one subplot per shared coord value)::

        view(Zip(
            [TraceConfig(F), TraceConfig(dFF), TraceConfig(denoised)],
            on="syn_id", colors=["#ff0000", "#00ff00", "#0000ff"],
        ))
    """
    from PySide6 import QtWidgets

    from loupe.app import (
        ArraySeries,
        DenseGroup,
        EventLayer as _RenderedEventLayer,
        LoupeApp,
        Series,
    )
    from loupe.df_loader import dataframe_to_matrix_series
    from loupe.xr_loader import (
        convert_event_arrays_aligned_with,
        convert_xarray_inputs_overlay,
        convert_xarray_inputs_with_order,
        dataarray_to_arrays,
    )

    # ---- normalize data into a list of Configs ----------------------------
    _allowed = (TraceConfig, HeatmapConfig, RasterConfig, Zip)
    if data is None:
        data_list: list = []
    elif isinstance(data, _allowed):
        data_list = [data]
    elif isinstance(data, list):
        data_list = list(data)
    else:
        raise TypeError(
            f"view() data= must be a Config or list of Configs (TraceConfig, "
            f"HeatmapConfig, RasterConfig, Zip), got {type(data).__name__}. "
            f"Wrap bare DataArrays in TraceConfig(da) and bare DataFrames "
            f"in RasterConfig(df)."
        )
    for i, item in enumerate(data_list):
        if not isinstance(item, _allowed):
            raise TypeError(
                f"view() data[{i}] must be TraceConfig / HeatmapConfig / "
                f"RasterConfig / Zip, got {type(item).__name__}. Wrap bare "
                f"DataArrays in TraceConfig(da) and bare DataFrames in "
                f"RasterConfig(df)."
            )

    # ---- cross-Config validation ------------------------------------------
    n_zip = sum(1 for x in data_list if isinstance(x, Zip))
    if n_zip > 1:
        raise ValueError("Only one Zip is supported per window in this release.")
    if n_zip == 1:
        if any(isinstance(x, (TraceConfig, HeatmapConfig)) for x in data_list):
            raise ValueError(
                "Zip cannot coexist with TraceConfig or HeatmapConfig in the "
                "same window. Move the traces into the Zip, or remove the Zip."
            )

    event_carrier = next(
        (
            x for x in data_list
            if isinstance(x, TraceConfig) and x.event_layers is not None
        ),
        None,
    )
    if event_carrier is not None:
        if event_carrier.mode != "stacked-subplots":
            raise ValueError(
                "TraceConfig.event_layers require mode='stacked-subplots'."
            )
        others = [x for x in data_list if x is not event_carrier]
        if any(
            isinstance(x, TraceConfig) and x.event_layers is not None
            for x in others
        ):
            raise ValueError(
                "Only one TraceConfig may carry event_layers per window."
            )
        if any(isinstance(x, (HeatmapConfig, RasterConfig, Zip)) for x in others):
            raise ValueError(
                "TraceConfig.event_layers cannot coexist with HeatmapConfig / "
                "RasterConfig / Zip in the same window."
            )
        if any(isinstance(x, TraceConfig) for x in others):
            raise ValueError(
                "TraceConfig.event_layers require a single TraceConfig in the "
                "data list."
            )

    # ---- dispatch each Config to its converter ----------------------------
    xr_series: list[Series] = []
    stacked_colors_acc: list = []
    any_stacked_color = False
    dense_list: list[DenseGroup] = []
    array_list: list[ArraySeries] = []
    matrix_list = []
    overlay_groups = None
    overlay_colors: list | None = None
    order_acc: list[tuple[str, int]] = []
    event_layers_rendered: list | None = None

    # Prefix is used to namespace trace names when multiple xarray Configs
    # contribute traces; same logic as before, but scoped to xr-source Configs.
    xr_configs = [c for c in data_list if isinstance(c, (TraceConfig, HeatmapConfig))]
    use_prefix = len(xr_configs) > 1
    all_named = all(getattr(c.data, "name", None) for c in xr_configs)

    def _prefix_for(cfg, i):
        if isinstance(cfg, TraceConfig) and isinstance(cfg.label, str):
            return cfg.label
        if isinstance(cfg, HeatmapConfig) and isinstance(cfg.label, str):
            return cfg.label
        if use_prefix:
            return str(cfg.data.name) if all_named else f"arr{i}"
        return ""

    for i, item in enumerate(data_list):
        if isinstance(item, Zip):
            das = [t.data for t in item.traces]
            overlay_groups = convert_xarray_inputs_overlay(das, item.on)
            overlay_colors = item.colors
        elif isinstance(item, RasterConfig):
            new_ms = dataframe_to_matrix_series(
                item.data,
                time_col=item.time_col,
                y_col=item.y_col,
                group_col=item.group_col,
                alpha_col=item.alpha_col,
                name=item.name,
                colors=item.colors,
                alpha_range=item.alpha_range,
                color_on=item.color_on,
                color_on_config=item.color_on_config,
            )
            if item.color_on is not None:
                if item.color is not None or item.colors is not None:
                    import warnings
                    warnings.warn(
                        "RasterConfig: color_on takes precedence; "
                        "color/colors are ignored.",
                        stacklevel=2,
                    )
            elif item.color is not None:
                resolved = _parse_raster_color(item.color)
                for ms in new_ms:
                    ms.color = resolved
            base = len(matrix_list)
            matrix_list.extend(new_ms)
            for j in range(len(new_ms)):
                order_acc.append(("matrix", base + j))
        elif isinstance(item, HeatmapConfig):
            prefix = _prefix_for(item, i)
            new_arrays = dataarray_to_arrays(
                item.data,
                split_on=item.split_on,
                sort_on=item.sort_on,
                colormap=item.colormap,
                vmin=item.vmin,
                vmax=item.vmax,
                decim_method=item.decim_method,
                name_prefix=prefix,
                label=item.label,
            )
            base = len(array_list)
            array_list.extend(new_arrays)
            for j in range(len(new_arrays)):
                order_acc.append(("array", base + j))
        else:  # TraceConfig
            cfg = item
            prefix = _prefix_for(cfg, i)
            if cfg.mode == "dense":
                tuples, order_vals, trace_labels, color_vals = convert_xarray_inputs_with_order(
                    cfg.data,
                    order_by=cfg.order_by,
                    descending=cfg.descending,
                    name_prefix=prefix,
                    color_by=cfg.color_by,
                )
                series_objs = [Series(n, t, y) for n, t, y in tuples]
                group_name = (
                    cfg.label if isinstance(cfg.label, str) else None
                ) or prefix or cfg.data.name or f"dense_{i}"
                dense_idx = len(dense_list)
                dense_list.append(DenseGroup(
                    name=str(group_name),
                    series=series_objs,
                    trace_labels=trace_labels,
                    order_values=order_vals,
                    color_values=color_vals,
                    descending=cfg.descending,
                    gain=cfg.gain,
                    step=cfg.step,
                    traces_per_page=cfg.traces_per_page,
                ))
                order_acc.append(("dense", dense_idx))
            elif cfg.mode == "stacked-subplots":
                tuples, _, _, _ = convert_xarray_inputs_with_order(
                    cfg.data,
                    order_by=cfg.order_by,
                    descending=cfg.descending,
                    name_prefix=prefix,
                    color_by=cfg.color_by,
                )
                new_series = [Series(n, t, y) for n, t, y in tuples]
                base = len(xr_series)
                xr_series.extend(new_series)
                for _s in new_series:
                    stacked_colors_acc.append(cfg.color)
                    if cfg.color is not None:
                        any_stacked_color = True
                    order_acc.append(("ts", base))
                    base += 1
                if cfg.event_layers is not None:
                    bool_per_layer = convert_event_arrays_aligned_with(
                        cfg.data,
                        [layer.bool_array for layer in cfg.event_layers],
                        order_by=cfg.order_by,
                        descending=cfg.descending,
                    )
                    event_layers_rendered = []
                    for layer, bps in zip(cfg.event_layers, bool_per_layer):
                        size = layer.size
                        if size is None:
                            size = 8.0 if layer.marker == "o" else 9.0
                        alpha = layer.alpha
                        if alpha is None:
                            alpha = 110 if layer.marker == "o" else 255
                        event_layers_rendered.append(
                            _RenderedEventLayer(
                                marker=layer.marker,
                                color=layer.color,
                                bool_per_series=bps,
                                size=size,
                                alpha=alpha,
                            )
                        )
            else:
                raise ValueError(
                    f"Unknown TraceConfig.mode={cfg.mode!r} "
                    f"(expected 'stacked-subplots' or 'dense')."
                )

    # Compute subplot_order, forwarded only if it deviates from the default
    # (ts → dense → matrix → array) — matches the default ordering inside
    # LoupeApp so callers without mixed input never carry a no-op list.
    default_order = (
        [("ts", k) for k in range(len(xr_series))]
        + [("dense", k) for k in range(len(dense_list))]
        + [("matrix", k) for k in range(len(matrix_list))]
        + [("array", k) for k in range(len(array_list))]
    )
    config_subplot_order = order_acc if order_acc != default_order else None

    xr_series_out = xr_series or None
    stacked_colors = stacked_colors_acc if any_stacked_color else None
    dense_groups = dense_list or None
    array_series = array_list or None
    matrix_series_list = matrix_list or None

    # ---- Qt event loop ----------------------------------------------------
    app = QtWidgets.QApplication.instance()
    created_app = False
    if app is None:
        _warn_if_ipython_without_qt()
        app = QtWidgets.QApplication([])
        created_app = True

    # Resolve the state config (keymap + label colors) up front so any
    # config error surfaces before we build the GUI.
    state_config = load_state_config(
        path=state_definitions,
        keymap=keymap,
        label_colors=label_colors,
    )

    # Build the initial LabelSet, if any.
    label_set: LabelSet | None = None
    if labels is not None:
        try:
            import polars as pl_runtime
        except ImportError:  # pragma: no cover - polars is a hard dep
            pl_runtime = None
        if pl_runtime is not None and isinstance(labels, pl_runtime.DataFrame):
            if label_schema is None:
                raise ValueError(
                    "label_schema= is required when labels is a polars DataFrame."
                )
            label_set = LabelSet.from_dataframe(
                labels,
                label_schema,
                writeback_allowed=labels_writeback,
            )
        else:
            label_set = LabelSet.from_path(
                labels,
                schema=label_schema,
                writeback_allowed=labels_writeback,
            )

    if stacked_colors is not None and "colors" not in kwargs:
        kwargs["colors"] = stacked_colors
    if config_subplot_order is not None and "subplot_order" not in kwargs:
        kwargs["subplot_order"] = config_subplot_order

    if videos is None:
        video_configs = []
    elif isinstance(videos, VideoConfig):
        video_configs = [videos]
    else:
        video_configs = list(videos)
        for v in video_configs:
            if not isinstance(v, VideoConfig):
                raise TypeError(
                    f"videos must contain VideoConfig instances, got {type(v).__name__}"
                )

    w = LoupeApp(
        xr_series=xr_series_out,
        matrix_series_list=matrix_series_list,
        overlay_groups=overlay_groups,
        overlay_colors=overlay_colors,
        dense_groups=dense_groups,
        array_series=array_series,
        window_len=window_len,
        event_layers=event_layers_rendered,
        state_config=state_config,
        label_set=label_set,
        label_alpha=label_alpha,
        video_configs=video_configs,
        **kwargs,
    )
    w.show()

    if created_app:
        import sys
        sys.exit(app.exec())
    else:
        return w


def _warn_if_ipython_without_qt() -> None:
    """Print a hint if we're inside IPython but no Qt loop is running."""
    try:
        ip = get_ipython()  # type: ignore[name-defined]  # noqa: F821
        loop = getattr(ip, "active_eventloop", None)
        if loop not in ("qt", "qt5", "qt6"):
            import warnings
            warnings.warn(
                "No Qt event loop detected. Run '%gui qt6' before calling "
                "view() for interactive use in Jupyter/IPython.",
                stacklevel=3,
            )
    except NameError:
        pass  # Not in IPython
