"""Convert xarray DataArrays to (name, t, y) tuples for the Loupe viewer.

This module has no Qt dependencies. All xarray imports are lazy so the rest of
the application works without xarray installed.
"""

from __future__ import annotations

import itertools
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr


# ---------------------------------------------------------------------------
# Dimension-name abbreviation
# ---------------------------------------------------------------------------

_ABBREV_MAP = {
    "channel": "ch",
    "channels": "ch",
    "syn_id": "syn",
    "synapse": "syn",
    "neuron": "n",
    "trial": "tr",
    "frequency": "freq",
    "wavelength": "wl",
    "electrode": "el",
    "region": "reg",
    "condition": "cond",
    "repetition": "rep",
    "stimulus": "stim",
}


def _abbrev_dim(dim_name: str) -> str:
    """Return a short abbreviation for a dimension name."""
    low = dim_name.lower()
    if low in _ABBREV_MAP:
        return _ABBREV_MAP[low]
    # Generic: first 3 characters
    return low[:3] if len(low) > 4 else low


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def dataarray_to_series(
    da: xr.DataArray,
    name_prefix: str = "",
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Flatten a DataArray into a list of ``(name, t_array, y_array)`` tuples.

    Parameters
    ----------
    da : xr.DataArray
        Must have a ``'time'`` dimension with coordinates.
    name_prefix : str, optional
        Prefix prepended to each trace name (e.g. the DataArray's name).

    Returns
    -------
    list[tuple[str, np.ndarray, np.ndarray]]
        Each element is ``(trace_name, time_1d, values_1d)``.
    """
    if "time" not in da.dims:
        raise ValueError(
            f"DataArray must have a 'time' dimension. Found dims: {da.dims}"
        )

    # Extract time as float64 seconds
    time_raw = da.coords["time"].values
    if np.issubdtype(time_raw.dtype, np.datetime64):
        # Convert datetime64 to float seconds from first timestamp
        t0 = time_raw[0]
        time_vals = (time_raw - t0).astype("timedelta64[ns]").astype(float) / 1e9
    else:
        time_vals = time_raw.astype(float)

    non_time_dims = [d for d in da.dims if d != "time"]

    results: list[tuple[str, np.ndarray, np.ndarray]] = []

    if not non_time_dims:
        # Simple 1D case
        name = name_prefix or da.name or "trace"
        results.append((str(name), time_vals.copy(), da.values.astype(float)))
    else:
        dim_coords = [da.coords[d].values for d in non_time_dims]
        abbrevs = [_abbrev_dim(d) for d in non_time_dims]

        for combo in itertools.product(*dim_coords):
            sel_dict = dict(zip(non_time_dims, combo))
            y = da.sel(sel_dict).values.astype(float)

            label = "-".join(f"{ab}{v}" for ab, v in zip(abbrevs, combo))
            if name_prefix:
                name = f"{name_prefix}: {label}"
            elif da.name:
                name = f"{da.name}: {label}"
            else:
                name = label
            results.append((name, time_vals.copy(), y))

    return results


# ---------------------------------------------------------------------------
# Path-based loading
# ---------------------------------------------------------------------------


def load_xarray_from_path(
    path: str,
    group: str | None = None,
    variable: str = "data",
    filter_dict: dict | None = None,
) -> xr.DataArray:
    """Open a zarr or netCDF store and return a (filtered, loaded) DataArray.

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

    Returns
    -------
    xr.DataArray
        Loaded into memory.
    """
    import xarray as xr  # lazy

    if path.endswith(".zarr") or os.path.isdir(path):
        ds = xr.open_zarr(path, group=group)
    else:
        ds = xr.open_dataset(path, group=group)

    da = ds[variable]

    if filter_dict:
        da = da.sel(**filter_dict)

    return da.load()


# ---------------------------------------------------------------------------
# Multi-array helper
# ---------------------------------------------------------------------------


def convert_xarray_inputs(
    data: xr.DataArray | list[xr.DataArray],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Accept one or more DataArrays and return combined series tuples.

    Parameters
    ----------
    data : DataArray or list[DataArray]

    Returns
    -------
    list[tuple[str, np.ndarray, np.ndarray]]
    """
    if not isinstance(data, list):
        data = [data]

    # Decide prefixes
    all_named = all(da.name for da in data)
    use_prefix = len(data) > 1

    results: list[tuple[str, np.ndarray, np.ndarray]] = []
    for i, da in enumerate(data):
        if use_prefix:
            prefix = str(da.name) if all_named else f"arr{i}"
        else:
            prefix = ""  # single array: let dataarray_to_series use da.name
        results.extend(dataarray_to_series(da, name_prefix=prefix))

    return results
