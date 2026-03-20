"""Loupe: Multi-trace data viewer for neuroscience."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loupe.app import LoupeApp

__all__ = ["view"]


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

    from loupe.app import LoupeApp, Series
    from loupe.xr_loader import (
        convert_xarray_inputs,
        convert_xarray_inputs_overlay,
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

    # ---- convert DataArray(s) → Series ------------------------------------
    xr_series: list[Series] | None = None
    overlay_groups = None
    if data is not None:
        if overlay is not None:
            if not isinstance(data, list):
                data = [data]
            overlay_groups = convert_xarray_inputs_overlay(data, overlay)
        else:
            tuples = convert_xarray_inputs(data)
            xr_series = [Series(name, t, y) for name, t, y in tuples]

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

    w = LoupeApp(
        xr_series=xr_series,
        matrix_series_list=matrix_series_list,
        overlay_groups=overlay_groups,
        overlay_colors=overlay_colors,
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
