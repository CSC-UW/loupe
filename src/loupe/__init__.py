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
    "LabelSchema",
    "LabelSet",
    "RasterConfig",
    "StateConfig",
    "TraceConfig",
    "view",
]


@dataclass
class TraceConfig:
    """Per-DataArray display configuration for :func:`view`.

    Parameters
    ----------
    data : xr.DataArray
        The DataArray to display.
    mode : str
        ``"stacked-subplots"`` (default) for one subplot per trace,
        ``"dense"`` for EEG-style offset traces on a single axis, or
        ``"array"`` for a 2-D heatmap (imshow-style) over time.
    order_by : str or None
        Coordinate name to control trace ordering and spacing
        (stacked / dense modes).
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
        (line modes) Single color applied to every trace produced by this
        DataArray, e.g. ``"#a020f0"`` or ``(160, 32, 240)``.  Overrides
        ``color_by`` when both are set.  Ignored in array mode.
    label : str, callable, or None
        Display name override.

        * In stacked-subplots / dense mode: a string used as the trace name
          (or, for multi-trace DataArrays, as the name prefix replacing
          ``data.name``).
        * In array mode: either a string (used verbatim per subplot, no
          ``"split_on=val"`` suffix) or a callable
          ``(split_val, sub_da) -> str`` invoked once per split group.
          1-arg callables ``(split_val) -> str`` are also accepted.
    split_on : str or None
        (array mode) Coordinate or dim name to split into one subplot per
        unique value (e.g. one heatmap per dendrite).
    sort_on : str or None
        (array mode) Coordinate name on the row dim controlling y-axis row
        order within each subplot.
    colormap : str, Colormap, list, dict, or callable
        (array mode) Matplotlib colormap name (e.g. ``"magma"``) or a
        Colormap instance (e.g. ``cmcrameri.cm.batlow``).  Also accepts:

        * a list — one entry per split group in iteration order, cycling;
        * a dict ``{split_val: cmap}`` — keyed by split value;
        * a callable ``(split_val, sub_da) -> str | Colormap`` — invoked
          per split group.  1-arg callables ``(split_val) -> ...`` are also
          accepted.
    vmin, vmax : float or None
        (array mode) Color scale limits.  Default is robust 1–99 percentile
        per array.
    decim_method : str
        (array mode) Time-axis decimation when zoomed out. ``"peak"`` (max-
        absolute per bin, preserves transients) or ``"mean"``.
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
    label: "str | Callable[..., str] | None" = None
    # Array-mode parameters
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
    data=None,
    *,
    path: str | list[str] | None = None,
    group: str | list[str] | None = None,
    variable: str = "data",
    filter_dict: dict | None = None,
    # DataFrame / matrix parameters
    matrix_df=None,
    matrix_parquet: str | list[str] | None = None,
    y_col: str = "source_id",
    group_col: str | list[str] | None = None,
    alpha_col: str | None = None,
    matrix_name: str = "events",
    matrix_colors=None,
    alpha_range: tuple[float, float] = (0.3, 1.0),
    overlay: str | None = None,
    overlay_colors: list | None = None,
    # Dense / display mode convenience parameters
    dense: bool = False,
    gain: float = 1.0,
    order_by: str | None = None,
    descending: bool = False,
    step: int = 1,
    traces_per_page: int | None = None,
    color_by: str | None = None,
    # Array (heatmap) mode convenience parameters
    array: bool = False,
    split_on: str | None = None,
    sort_on: str | None = None,
    colormap: "str | Colormap | list[str | Colormap]" = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    decim_method: str = "peak",
    window_len: float = 10.0,
    # Bool-event marker overlays (stacked-subplots mode only)
    bool_event_arrays: list | None = None,
    event_markers: list[str] | None = None,
    event_marker_colors: list | None = None,
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
    """Launch the Loupe viewer with xarray and/or DataFrame data.

    Parameters
    ----------
    data : xr.DataArray or list[xr.DataArray], optional
        In-memory DataArray(s) to display as time-series traces.  Each must
        have a ``'time'`` dimension.
    path : str or list[str], optional
        Path(s) to zarr or netCDF stores.  Mutually exclusive with *data*.
    group : str or list[str], optional
        Group(s) within the store(s) (e.g. ``'dmd_2'``).  A single string
        is applied to every path; a list must match *path* length.
    variable : str
        Variable name in the dataset (default ``'data'``).
    filter_dict : dict, optional
        Dimension slicing applied to every loaded DataArray, e.g.
        ``{"syn_id": slice(3, 6), "time": slice(0, 1800)}``.
    matrix_df : pl.DataFrame, RasterConfig, or list thereof, optional
        In-memory Polars DataFrame(s) (or :class:`RasterConfig` objects) to
        display as raster plots.  Mutually exclusive with *matrix_parquet*.
        Pass :class:`RasterConfig` to override per-source settings (color,
        grouping, etc.); bare DataFrames inherit the top-level matrix
        kwargs (``y_col``, ``group_col``, ``alpha_col``, ``matrix_name``,
        ``matrix_colors``, ``alpha_range``) as defaults.
    matrix_parquet : str or list[str], optional
        Path(s) to parquet file(s) to load as raster plots.  Mutually
        exclusive with *matrix_df*.
    y_col : str
        Default DataFrame column for matrix row assignment
        (default ``"source_id"``).  Ignored for items wrapped in
        :class:`RasterConfig`.
    group_col : str or list[str] or None
        Default DataFrame column(s) to split into separate raster subplots.
        Ignored for items wrapped in :class:`RasterConfig`.
    alpha_col : str or None
        Default DataFrame column for per-event opacity.  Ignored for items
        wrapped in :class:`RasterConfig`.
    matrix_name : str
        Default base name for raster subplots (default ``"events"``).
        Ignored for items wrapped in :class:`RasterConfig`.
    matrix_colors : dict, list, tuple or None
        Default color specification per group (see
        :func:`df_loader.dataframe_to_matrix_series`).  ``None`` means white.
        Ignored for items wrapped in :class:`RasterConfig`.
    alpha_range : tuple[float, float]
        Default ``(min_alpha, max_alpha)`` for normalizing *alpha_col*.
        Ignored for items wrapped in :class:`RasterConfig`.
    overlay : str or None
        Dimension name to overlay on (e.g. ``'syn_id'``).  When set, traces
        from different DataArrays that share the same coordinate value on this
        dimension are plotted on the same subplot.  Requires *data* to be a
        list of at least 2 DataArrays.
    overlay_colors : list or None
        Optional list of colors (one per input DataArray) for overlay mode.
        Each element can be a hex string (``'#RRGGBB'``) or an RGB(A) tuple.
        If not specified, a default palette is used.
    dense : bool
        If True, display all DataArrays in dense (EEG-style) mode by default.
        Ignored for items wrapped in :class:`TraceConfig`.
    gain : float
        Default amplitude gain multiplier (applied to all DataArrays not
        wrapped in :class:`TraceConfig`).
    order_by : str or None
        Default coordinate name for trace ordering.
    descending : bool
        Reverse the ordering given by *order_by*.
    step : int
        Default trace step (dense mode only, 1 = show all).
    traces_per_page : int or None
        How many traces to show at once in dense mode (None = all).
        Use Option+scroll to page through the rest.
    color_by : str or None
        Coordinate name whose categorical values determine per-trace color.
        Uses a colorblind-friendly palette visible against the black background.
    bool_event_arrays : list[xr.DataArray] or None
        Optional list of boolean DataArrays whose dims/shape match *data*.
        ``True`` at sample i indicates a point event on that trace at that
        timepoint; markers are drawn at ``y = trace_value`` so they sit on
        the line.  Stacked-subplots mode only.
    event_markers : list[str] or None
        One marker symbol per entry in *bool_event_arrays*.  ``'o'`` renders
        a semi-transparent filled circle; ``'x'`` renders a solid X. Other
        pyqtgraph symbols (``'+'``, ``'s'``, ``'t'``, …) fall through with
        a default outlined style.
    event_marker_colors : list or None
        One color per entry in *bool_event_arrays* — named string, hex, or
        ``(r, g, b[, a])`` tuple.
    labels : pl.DataFrame, str, or Path, optional
        Optional initial labels. Either a polars DataFrame (in which case
        ``label_schema`` is required) or a path to a ``.csv``, ``.htsv``,
        ``.parquet``, or Visbrain ``.txt`` file. For ``.htsv`` and ``.parquet``,
        ``label_schema`` is also required. CSV defaults to the legacy
        ``start_s/end_s/label/note`` schema; ``.txt`` to Visbrain.
    label_schema : LabelSchema, optional
        Describes how the user's columns map to start/end/duration/label/note
        and which extra columns to display in the GUI.
    labels_writeback : bool
        If True, the GUI's "Save Labels (overwrite source)" action will
        overwrite the file passed in ``labels``. Default False; the source
        file is never overwritten without this opt-in.
    state_definitions : str or Path, optional
        Path to a JSON file with ``"keymap"`` and ``"label_colors"`` keys.
        See ``example_state_definitions.json`` for the schema.
    keymap : dict, optional
        Programmatic state hotkeys. Accepts either forward
        (``{"w": "Wake", "1": "NREM"}``) or inverse
        (``{"Wake": ["w", "W"]}``) form. Multiple hotkeys per state are
        supported. Overrides any keys also defined in the file.
    label_colors : dict, optional
        Programmatic ``state -> color`` mapping. Color values may be RGBA
        tuples, ``[R, G, B[, A]]`` lists, or hex strings (``"#RRGGBBAA"``).
        Overrides any colors also defined in the file.
    label_alpha : float, optional
        Initial label-overlay alpha multiplier in ``[0.0, 1.0]``. Equivalent
        to setting the View → "Adjust Label Alpha…" slider at launch.
        Defaults to ``1.0`` (use each state's alpha as defined in
        ``label_colors``).
    **kwargs
        Forwarded to :class:`LoupeApp` (``video_path``,
        ``frame_times_path``, ``fixed_scale``, ``low_profile_x``, etc.).

    Returns
    -------
    LoupeApp
        The viewer window.  In Jupyter (with ``%gui qt6``) the window stays
        alive after the call returns.  In a script the call blocks until the
        window is closed.

    Examples
    --------
    xarray time-series::

        w = view(path="data.zarr", group="dmd_2",
                 filter_dict={"syn_id": slice(3, 6), "time": slice(0, 1800)})

    DataFrame raster plot::

        import polars as pl
        ev = pl.read_parquet("glut_events.parquet")
        w = view(matrix_df=ev, y_col="source_id", group_col="dmd",
                 alpha_col="snr_denoised")

    Per-source settings via :class:`RasterConfig`::

        from loupe import view, RasterConfig
        w = view(matrix_df=RasterConfig(
            ev, y_col="source_id", group_col="dmd", color="#ff00ff",
        ))

    Combined::

        w = view(path="traces.zarr", group="dmd_2",
                 matrix_df=ev, y_col="source_id", group_col="dmd",
                 alpha_col="snr_denoised")
    """
    from PySide6 import QtWidgets

    from loupe.app import ArraySeries, DenseGroup, LoupeApp, Series
    from loupe.xr_loader import (
        convert_xarray_inputs_overlay,
        convert_xarray_inputs_with_order,
        dataarray_to_arrays,
        load_xarray_from_path,
    )

    if dense and array:
        raise ValueError("Cannot pass dense=True and array=True together.")

    if data is not None and path is not None:
        raise ValueError("Provide either 'data' or 'path', not both.")
    if matrix_df is not None and matrix_parquet is not None:
        raise ValueError("Provide either 'matrix_df' or 'matrix_parquet', not both.")

    # ---- validate bool_event_arrays / event_markers / event_marker_colors -
    # All three must be provided together; only stacked-subplots inputs
    # accept event-marker overlays in v1.
    _ev_inputs = (bool_event_arrays, event_markers, event_marker_colors)
    use_event_markers = any(x is not None for x in _ev_inputs)
    if use_event_markers:
        if not all(x is not None for x in _ev_inputs):
            raise ValueError(
                "bool_event_arrays, event_markers, and event_marker_colors "
                "must all be provided together."
            )
        if not (
            len(bool_event_arrays)
            == len(event_markers)
            == len(event_marker_colors)
        ):
            raise ValueError(
                "bool_event_arrays, event_markers, and event_marker_colors "
                "must all be the same length."
            )
        if len(bool_event_arrays) == 0:
            raise ValueError("bool_event_arrays cannot be empty.")
        if dense or array or overlay is not None:
            raise ValueError(
                "Event markers are only supported in the default "
                "stacked-subplots mode (not dense, array, or overlay)."
            )
        if matrix_df is not None or matrix_parquet is not None:
            raise ValueError(
                "Event markers are not supported alongside matrix/raster "
                "DataFrame inputs."
            )
        if path is not None:
            raise ValueError(
                "Event markers require an in-memory `data=` DataArray "
                "(not `path=`)."
            )
        if data is None:
            raise ValueError(
                "Event markers require an in-memory `data=` DataArray."
            )
        if isinstance(data, list):
            raise ValueError(
                "Event markers only support a single DataArray input "
                "(not a list of DataArrays)."
            )

    # ---- resolve path(s) to in-memory DataArrays --------------------------
    if path is not None:
        paths = [path] if isinstance(path, str) else list(path)
        # Normalise group to a list matching paths
        if group is None:
            groups = [None] * len(paths)
        elif isinstance(group, str):
            groups = [group] * len(paths)
        else:
            groups = list(group)
            if len(groups) != len(paths):
                raise ValueError(
                    f"group list length ({len(groups)}) must match "
                    f"path list length ({len(paths)})"
                )

        data = [
            load_xarray_from_path(p, group=g, variable=variable,
                                  filter_dict=filter_dict)
            for p, g in zip(paths, groups)
        ]

    # ---- normalise data to list[TraceConfig] if needed --------------------
    configs: list[TraceConfig] | None = None
    if data is not None and overlay is None:
        if not isinstance(data, list):
            data = [data]
        configs = []
        if array:
            default_mode = "array"
        elif dense:
            default_mode = "dense"
        else:
            default_mode = "stacked-subplots"
        for item in data:
            if isinstance(item, TraceConfig):
                configs.append(item)
            else:
                configs.append(TraceConfig(
                    data=item,
                    mode=default_mode,
                    order_by=order_by,
                    descending=descending,
                    gain=gain,
                    step=step,
                    traces_per_page=traces_per_page,
                    color_by=color_by,
                    split_on=split_on,
                    sort_on=sort_on,
                    colormap=colormap,
                    vmin=vmin,
                    vmax=vmax,
                    decim_method=decim_method,
                ))

    # ---- build event_layers (validated above) ----------------------------
    event_layers = None
    if use_event_markers:
        if configs is None or len(configs) != 1:
            raise ValueError(
                "Event markers only support a single DataArray input."
            )
        cfg = configs[0]
        if cfg.mode != "stacked-subplots":
            raise ValueError(
                "Event markers are only supported in stacked-subplots mode."
            )
        from loupe.app import EventLayer
        from loupe.xr_loader import convert_event_arrays_aligned_with

        bool_per_layer = convert_event_arrays_aligned_with(
            cfg.data,
            list(bool_event_arrays),
            order_by=cfg.order_by,
            descending=cfg.descending,
        )
        # Per-marker defaults preserve the v1 spec: 'o' = semi-transparent
        # circle (alpha 110, size 8), 'x' / others = solid stroke (alpha 255,
        # size 9). User can override live via View → Adjust Event Marker
        # Properties.
        event_layers = []
        for m, c, bps in zip(event_markers, event_marker_colors, bool_per_layer):
            if m == "o":
                size, alpha = 8.0, 110
            else:
                size, alpha = 9.0, 255
            event_layers.append(
                EventLayer(
                    marker=m, color=c, bool_per_series=bps,
                    size=size, alpha=alpha,
                )
            )

    # ---- convert DataArray(s) → Series / DenseGroups / ArraySeries -------
    xr_series: list[Series] | None = None
    stacked_colors: list | None = None  # one entry per Series in xr_series; None = default
    overlay_groups = None
    dense_groups: list[DenseGroup] | None = None
    array_series: list[ArraySeries] | None = None
    # subplot_order: list of ("ts"|"dense"|"array", idx) entries describing
    # the visual layout top-to-bottom. Built in the same order configs are
    # processed so that interleaving lines and arrays in the input list
    # produces an interleaved on-screen layout.
    config_subplot_order: list[tuple[str, int]] | None = None
    if data is not None:
        if overlay is not None:
            if not isinstance(data, list):
                data = [data]
            overlay_groups = convert_xarray_inputs_overlay(data, overlay)
        elif configs is not None:
            stacked_series: list[Series] = []
            stacked_colors_acc: list = []
            any_color = False
            dense_list: list[DenseGroup] = []
            array_list: list[ArraySeries] = []
            order_acc: list[tuple[str, int]] = []
            use_prefix = len(configs) > 1
            all_named = all(
                getattr(c.data, "name", None) for c in configs
            )
            for i, cfg in enumerate(configs):
                # In stacked/dense mode, an explicit cfg.label overrides the
                # auto-derived prefix (which is what becomes the trace name
                # for 1-D DataArrays).
                if cfg.mode != "array" and isinstance(cfg.label, str):
                    prefix = cfg.label
                elif use_prefix:
                    prefix = str(cfg.data.name) if all_named else f"arr{i}"
                else:
                    prefix = ""
                if cfg.mode == "array":
                    new_arrays = dataarray_to_arrays(
                        cfg.data,
                        split_on=cfg.split_on,
                        sort_on=cfg.sort_on,
                        colormap=cfg.colormap,
                        vmin=cfg.vmin,
                        vmax=cfg.vmax,
                        decim_method=cfg.decim_method,
                        name_prefix=prefix,
                        label=cfg.label,
                    )
                    base = len(array_list)
                    array_list.extend(new_arrays)
                    for j in range(len(new_arrays)):
                        order_acc.append(("array", base + j))
                elif cfg.mode == "dense":
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
                else:
                    tuples, _, _, _ = convert_xarray_inputs_with_order(
                        cfg.data,
                        order_by=cfg.order_by,
                        descending=cfg.descending,
                        name_prefix=prefix,
                        color_by=cfg.color_by,
                    )
                    new_series = [Series(n, t, y) for n, t, y in tuples]
                    base = len(stacked_series)
                    stacked_series.extend(new_series)
                    # One color slot per produced Series (broadcast cfg.color)
                    for j, _s in enumerate(new_series):
                        stacked_colors_acc.append(cfg.color)
                        if cfg.color is not None:
                            any_color = True
                        order_acc.append(("ts", base + j))
            if stacked_series:
                xr_series = stacked_series
                if any_color:
                    stacked_colors = stacked_colors_acc
            if dense_list:
                dense_groups = dense_list
            if array_list:
                array_series = array_list
            # Only forward an order if it actually deviates from the default
            # (ts → dense → array). Avoids carrying around a no-op list.
            default_order = (
                [("ts", k) for k in range(len(stacked_series))]
                + [("dense", k) for k in range(len(dense_list))]
                + [("array", k) for k in range(len(array_list))]
            )
            if order_acc != default_order:
                config_subplot_order = order_acc

    # ---- resolve DataFrame(s) to MatrixSeries -----------------------------
    matrix_series_list = None
    if matrix_parquet is not None:
        from loupe.df_loader import load_dataframe_from_parquet

        matrix_df = load_dataframe_from_parquet(matrix_parquet)

    if matrix_df is not None:
        from loupe.df_loader import dataframe_to_matrix_series

        if not isinstance(matrix_df, list):
            matrix_df = [matrix_df]
        # Normalise to list[RasterConfig]: bare DataFrames inherit the
        # top-level matrix kwargs as defaults; explicit RasterConfigs pass
        # through unchanged.
        raster_configs: list[RasterConfig] = []
        n_inputs = len(matrix_df)
        for i, item in enumerate(matrix_df):
            default_name = matrix_name if n_inputs == 1 else f"{matrix_name}_{i}"
            if isinstance(item, RasterConfig):
                raster_configs.append(item)
            else:
                raster_configs.append(RasterConfig(
                    data=item,
                    y_col=y_col,
                    group_col=group_col,
                    alpha_col=alpha_col,
                    name=default_name,
                    colors=matrix_colors,
                    alpha_range=alpha_range,
                ))
        all_ms = []
        for cfg in raster_configs:
            new_ms = dataframe_to_matrix_series(
                cfg.data,
                time_col=cfg.time_col,
                y_col=cfg.y_col,
                group_col=cfg.group_col,
                alpha_col=cfg.alpha_col,
                name=cfg.name,
                colors=cfg.colors,
                alpha_range=cfg.alpha_range,
                color_on=cfg.color_on,
                color_on_config=cfg.color_on_config,
            )
            if cfg.color_on is not None:
                if cfg.color is not None or cfg.colors is not None:
                    import warnings
                    warnings.warn(
                        "RasterConfig: color_on takes precedence; "
                        "color/colors are ignored.",
                        stacklevel=2,
                    )
            elif cfg.color is not None:
                resolved_color = _parse_raster_color(cfg.color)
                for ms in new_ms:
                    ms.color = resolved_color
            all_ms.extend(new_ms)
        if all_ms:
            matrix_series_list = all_ms

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

    # Per-line colors derived from TraceConfig.color, unless the caller
    # passed an explicit colors= kwarg (which wins).
    if stacked_colors is not None and "colors" not in kwargs:
        kwargs["colors"] = stacked_colors
    # Initial layout order derived from TraceConfig list order; caller-supplied
    # subplot_order (via kwargs) wins.
    if config_subplot_order is not None and "subplot_order" not in kwargs:
        kwargs["subplot_order"] = config_subplot_order

    w = LoupeApp(
        xr_series=xr_series,
        matrix_series_list=matrix_series_list,
        overlay_groups=overlay_groups,
        overlay_colors=overlay_colors,
        dense_groups=dense_groups,
        array_series=array_series,
        window_len=window_len,
        event_layers=event_layers,
        state_config=state_config,
        label_set=label_set,
        label_alpha=label_alpha,
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
