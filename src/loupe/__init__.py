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
    **kwargs,
) -> LoupeApp:
    """Launch the Loupe viewer with xarray data.

    Parameters
    ----------
    data : xr.DataArray or list[xr.DataArray], optional
        In-memory DataArray(s) to display. Each must have a ``'time'``
        dimension.  All other dimension combinations become separate traces.
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
    Jupyter notebook::

        %gui qt6
        import xarray as xr
        from loupe import view

        ds = xr.open_zarr("data.zarr", group="dmd_2")
        da = ds["data"].sel(syn_id=slice(3, 6), time=slice(0, 1800)).load()
        w = view(da)

    Path-based::

        w = view(path="data.zarr", group="dmd_2",
                 filter_dict={"syn_id": slice(3, 6), "time": slice(0, 1800)})
    """
    from PySide6 import QtWidgets

    from loupe.app import LoupeApp, Series
    from loupe.xr_loader import (
        convert_xarray_inputs,
        load_xarray_from_path,
    )

    if data is not None and path is not None:
        raise ValueError("Provide either 'data' or 'path', not both.")

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
    if data is not None:
        tuples = convert_xarray_inputs(data)
        xr_series = [Series(name, t, y) for name, t, y in tuples]

    # ---- Qt event loop ----------------------------------------------------
    app = QtWidgets.QApplication.instance()
    created_app = False
    if app is None:
        _warn_if_ipython_without_qt()
        app = QtWidgets.QApplication([])
        created_app = True

    w = LoupeApp(xr_series=xr_series, **kwargs)
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
