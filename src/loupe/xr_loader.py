"""Convert xarray DataArrays to (name, t, y) tuples for the Loupe viewer.

This module has no Qt dependencies. All xarray imports are lazy so the rest of
the application works without xarray installed.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr


# ---------------------------------------------------------------------------
# Overlay data structures
# ---------------------------------------------------------------------------


@dataclass
class OverlayTrace:
    """A single trace within an overlay group, from one source DataArray."""

    name: str  # DataArray name (used for legend)
    t: np.ndarray
    y: np.ndarray
    source_idx: int  # index of the source DataArray (for color assignment)


@dataclass
class OverlayGroup:
    """A group of traces sharing the same overlay dimension value."""

    label: str  # shared dimension value label, e.g. "syn1"
    traces: list[OverlayTrace] = field(default_factory=list)


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


# ---------------------------------------------------------------------------
# Overlay conversion
# ---------------------------------------------------------------------------


def _extract_time_vals(da: xr.DataArray) -> np.ndarray:
    """Extract time coordinate as float64 seconds."""
    time_raw = da.coords["time"].values
    if np.issubdtype(time_raw.dtype, np.datetime64):
        t0 = time_raw[0]
        return (time_raw - t0).astype("timedelta64[ns]").astype(float) / 1e9
    return time_raw.astype(float)


def convert_xarray_inputs_overlay(
    data: list[xr.DataArray],
    overlay_dim: str,
) -> list[OverlayGroup]:
    """Group traces from multiple DataArrays by a shared dimension.

    Parameters
    ----------
    data : list[xr.DataArray]
        Two or more DataArrays, each with ``'time'`` and ``overlay_dim``.
    overlay_dim : str
        Dimension to overlay on (e.g. ``'syn_id'``). Traces sharing the same
        coordinate value on this dimension are grouped into a single subplot.

    Returns
    -------
    list[OverlayGroup]
        One group per unique value (or combination, if extra dims exist).
    """
    if len(data) < 2:
        raise ValueError("overlay requires at least 2 DataArrays")

    for i, da in enumerate(data):
        if "time" not in da.dims:
            raise ValueError(
                f"DataArray {i} must have a 'time' dimension. Found: {da.dims}"
            )
        if overlay_dim not in da.dims:
            raise ValueError(
                f"DataArray {i} does not have overlay dimension '{overlay_dim}'. "
                f"Found: {da.dims}"
            )

    # Determine source names
    all_named = all(da.name for da in data)
    source_names = [
        str(da.name) if all_named else f"arr{i}" for i, da in enumerate(data)
    ]

    # Collect the union of overlay_dim values across all arrays
    overlay_vals_set: dict[object, None] = {}  # ordered set via dict
    for da in data:
        for v in da.coords[overlay_dim].values:
            overlay_vals_set[v] = None
    overlay_vals = list(overlay_vals_set.keys())

    # Determine extra non-time, non-overlay dims (iterate over these)
    extra_dims = [d for d in data[0].dims if d not in ("time", overlay_dim)]

    overlay_abbrev = _abbrev_dim(overlay_dim)
    extra_abbrevs = [_abbrev_dim(d) for d in extra_dims]

    # Build groups
    groups: list[OverlayGroup] = []

    if not extra_dims:
        # Simple case: just overlay_dim
        for val in overlay_vals:
            label = f"{overlay_abbrev}{val}"
            group = OverlayGroup(label=label)
            for src_idx, da in enumerate(data):
                if val not in da.coords[overlay_dim].values:
                    continue
                t = _extract_time_vals(da)
                y = da.sel({overlay_dim: val}).values.astype(float)
                group.traces.append(
                    OverlayTrace(
                        name=source_names[src_idx],
                        t=t.copy(),
                        y=y,
                        source_idx=src_idx,
                    )
                )
            if group.traces:
                groups.append(group)
    else:
        # Extra dims: create one group per (overlay_val, extra_combo)
        extra_coords = [data[0].coords[d].values for d in extra_dims]
        for val in overlay_vals:
            for combo in itertools.product(*extra_coords):
                overlay_label = f"{overlay_abbrev}{val}"
                extra_label = "-".join(
                    f"{ab}{v}" for ab, v in zip(extra_abbrevs, combo)
                )
                label = f"{overlay_label}-{extra_label}"

                group = OverlayGroup(label=label)
                sel_extra = dict(zip(extra_dims, combo))

                for src_idx, da in enumerate(data):
                    if val not in da.coords[overlay_dim].values:
                        continue
                    if not all(d in da.dims for d in extra_dims):
                        continue
                    t = _extract_time_vals(da)
                    sel_dict = {overlay_dim: val, **sel_extra}
                    y = da.sel(sel_dict).values.astype(float)
                    group.traces.append(
                        OverlayTrace(
                            name=source_names[src_idx],
                            t=t.copy(),
                            y=y,
                            source_idx=src_idx,
                        )
                    )
                if group.traces:
                    groups.append(group)

    return groups
