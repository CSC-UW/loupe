"""Loupe: Multi-trace data viewer for neuroscience."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loupe.labels import LabelSchema, LabelSet
from loupe.state_config import StateConfig, load_state_config

if TYPE_CHECKING:
    import polars as pl
    import xarray as xr

    from loupe.app import LoupeApp

__all__ = ["LabelSchema", "LabelSet", "StateConfig", "TraceConfig", "view"]


@dataclass
class TraceConfig:
    """Per-DataArray display configuration for :func:`view`.

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
        Initial amplitude gain multiplier.
    step : int
        Show every *step*-th trace (dense mode only).
    traces_per_page : int or None
        How many traces to show at once in dense mode. ``None`` = all.
        Use Alt+scroll to page through the rest.
    color_by : str or None
        Coordinate name whose categorical values determine per-trace color.
    label : str or None
        Optional display name for this group.
    """

    data: xr.DataArray
    mode: str = "stacked-subplots"
    order_by: str | None = None
    descending: bool = False
    gain: float = 1.0
    step: int = 1
    traces_per_page: int | None = None
    color_by: str | None = None
    label: str | None = None


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
    window_len: float = 10.0,
    # Label loading
    labels: "pl.DataFrame | str | Path | None" = None,
    label_schema: LabelSchema | None = None,
    labels_writeback: bool = False,
    # State definitions
    state_definitions: str | Path | None = None,
    keymap: dict | None = None,
    label_colors: dict | None = None,
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

    from loupe.app import DenseGroup, LoupeApp, Series
    from loupe.xr_loader import (
        convert_xarray_inputs_overlay,
        convert_xarray_inputs_with_order,
        load_xarray_from_path,
    )

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
        for item in data:
            if isinstance(item, TraceConfig):
                configs.append(item)
            else:
                configs.append(TraceConfig(
                    data=item,
                    mode="dense" if dense else "stacked-subplots",
                    order_by=order_by,
                    descending=descending,
                    gain=gain,
                    step=step,
                    traces_per_page=traces_per_page,
                    color_by=color_by,
                ))

    # ---- convert DataArray(s) → Series / DenseGroups ----------------------
    xr_series: list[Series] | None = None
    overlay_groups = None
    dense_groups: list[DenseGroup] | None = None
    if data is not None:
        if overlay is not None:
            if not isinstance(data, list):
                data = [data]
            overlay_groups = convert_xarray_inputs_overlay(data, overlay)
        elif configs is not None:
            stacked_series: list[Series] = []
            dense_list: list[DenseGroup] = []
            use_prefix = len(configs) > 1
            all_named = all(
                getattr(c.data, "name", None) for c in configs
            )
            for i, cfg in enumerate(configs):
                if use_prefix:
                    prefix = str(cfg.data.name) if all_named else f"arr{i}"
                else:
                    prefix = ""
                if cfg.mode == "dense":
                    tuples, order_vals, trace_labels, color_vals = convert_xarray_inputs_with_order(
                        cfg.data,
                        order_by=cfg.order_by,
                        descending=cfg.descending,
                        name_prefix=prefix,
                        color_by=cfg.color_by,
                    )
                    series_objs = [Series(n, t, y) for n, t, y in tuples]
                    group_name = cfg.label or prefix or cfg.data.name or f"dense_{i}"
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
                else:
                    tuples, _, _, _ = convert_xarray_inputs_with_order(
                        cfg.data,
                        order_by=cfg.order_by,
                        descending=cfg.descending,
                        name_prefix=prefix,
                        color_by=cfg.color_by,
                    )
                    stacked_series.extend(
                        Series(n, t, y) for n, t, y in tuples
                    )
            if stacked_series:
                xr_series = stacked_series
            if dense_list:
                dense_groups = dense_list

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

    w = LoupeApp(
        xr_series=xr_series,
        matrix_series_list=matrix_series_list,
        overlay_groups=overlay_groups,
        overlay_colors=overlay_colors,
        dense_groups=dense_groups,
        window_len=window_len,
        state_config=state_config,
        label_set=label_set,
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
