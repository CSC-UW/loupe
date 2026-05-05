"""Loupe: Multi-trace data viewer for neuroscience."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import xarray as xr
    from matplotlib.colors import Colormap

    from loupe.app import LoupeApp

__all__ = ["view", "TraceConfig"]


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
    matrix_df : pl.DataFrame or list[pl.DataFrame], optional
        In-memory Polars DataFrame(s) to display as matrix/raster plots.
        Mutually exclusive with *matrix_parquet*.
    matrix_parquet : str or list[str], optional
        Path(s) to parquet file(s) to load as matrix/raster plots.
        Mutually exclusive with *matrix_df*.
    y_col : str
        DataFrame column for matrix row assignment (default ``"source_id"``).
    group_col : str or list[str] or None
        DataFrame column(s) to split into separate matrix subplots.
    alpha_col : str or None
        DataFrame column for per-event opacity.
    matrix_name : str
        Base name for the matrix subplots (default ``"events"``).
    matrix_colors : dict, list, tuple or None
        Color specification per group (see :func:`df_loader.dataframe_to_matrix_series`).
    alpha_range : tuple[float, float]
        ``(min_alpha, max_alpha)`` for normalizing *alpha_col*.
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

    Combined::

        w = view(path="traces.zarr", group="dmd_2",
                 matrix_df=ev, y_col="source_id", group_col="dmd",
                 alpha_col="snr_denoised")
    """
    from PySide6 import QtWidgets

    from loupe.app import ArraySeries, DenseGroup, LoupeApp, Series
    from loupe.xr_loader import (
        convert_xarray_inputs,
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
                    tuples, order_vals, labels, color_vals = convert_xarray_inputs_with_order(
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
                        trace_labels=labels,
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
        all_ms = []
        for i, mdf in enumerate(matrix_df):
            prefix = matrix_name if len(matrix_df) == 1 else f"{matrix_name}_{i}"
            all_ms.extend(
                dataframe_to_matrix_series(
                    mdf,
                    time_col="time",
                    y_col=y_col,
                    group_col=group_col,
                    alpha_col=alpha_col,
                    name=prefix,
                    colors=matrix_colors,
                    alpha_range=alpha_range,
                )
            )
        if all_ms:
            matrix_series_list = all_ms

    # ---- Qt event loop ----------------------------------------------------
    app = QtWidgets.QApplication.instance()
    created_app = False
    if app is None:
        _warn_if_ipython_without_qt()
        app = QtWidgets.QApplication([])
        created_app = True

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
