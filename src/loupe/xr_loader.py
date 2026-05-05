"""Convert xarray DataArrays to (name, t, y) tuples for the Loupe viewer.

This module has no Qt dependencies. All xarray imports are lazy so the rest of
the application works without xarray installed.
"""

from __future__ import annotations

import inspect
import itertools
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    import xarray as xr
    from matplotlib.colors import Colormap


def _call_with_optional_subda(fn: Callable, split_val: Any, sub_da: Any) -> Any:
    """Invoke *fn* with either ``(split_val,)`` or ``(split_val, sub_da)``.

    Picks the arity by inspecting *fn*'s required positional parameters,
    so users can write either ``lambda v: ...`` or ``lambda v, sub: ...``.
    Falls back to single-argument call if the signature can't be read
    (e.g. C-implemented builtins).
    """
    try:
        sig = inspect.signature(fn)
        positional = [
            p for p in sig.parameters.values()
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            )
        ]
        # Use 2-arg call if the signature can accept >=2 positional args.
        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in positional):
            return fn(split_val, sub_da)
        if len(positional) >= 2:
            return fn(split_val, sub_da)
    except (TypeError, ValueError):
        pass
    return fn(split_val)


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


def _coord_values_per_trace(
    da: xr.DataArray,
    coord_name: str,
    non_time_dims: list[str],
) -> np.ndarray | None:
    """Extract one value of *coord_name* per trace, in ``itertools.product`` order.

    Returns *None* if the coordinate cannot be aligned to traces.
    """
    if coord_name not in da.coords:
        return None
    coord_vals = da.coords[coord_name].values
    if len(non_time_dims) == 1:
        return coord_vals
    if len(non_time_dims) > 1:
        target_dim = None
        for d in non_time_dims:
            if da.coords[coord_name].dims == (d,):
                target_dim = d
                break
        if target_dim is None:
            return None
        dim_coords = [da.coords[d].values for d in non_time_dims]
        dim_sizes = [len(c) for c in dim_coords]
        target_idx = non_time_dims.index(target_dim)
        target_vals = dim_coords[target_idx]
        reps_before = 1
        for j in range(target_idx):
            reps_before *= dim_sizes[j]
        reps_after = 1
        for j in range(target_idx + 1, len(dim_sizes)):
            reps_after *= dim_sizes[j]
        return np.tile(np.repeat(target_vals, reps_after), reps_before)
    return None


def convert_xarray_inputs_with_order(
    da: xr.DataArray,
    order_by: str | None = None,
    descending: bool = False,
    name_prefix: str = "",
    color_by: str | None = None,
) -> tuple[
    list[tuple[str, np.ndarray, np.ndarray]],
    np.ndarray | None,
    list[str],
    np.ndarray | None,
]:
    """Convert a DataArray to series tuples with optional ordering metadata.

    Parameters
    ----------
    da : xr.DataArray
        Must have a ``'time'`` dimension.
    order_by : str or None
        Coordinate name to use for trace ordering.  If *None* and there is
        exactly one non-time dimension, that dimension's coordinate values
        are used automatically.
    descending : bool
        Reverse the ordering.
    name_prefix : str
        Prefix prepended to each trace name.
    color_by : str or None
        Coordinate name whose (categorical) values determine per-trace
        color.  Extracted in the same order as the returned tuples.

    Returns
    -------
    series_tuples : list[tuple[str, np.ndarray, np.ndarray]]
    order_values : np.ndarray or None
        One value per trace, for ordering / spacing.
    trace_labels : list[str]
        Display label for each trace.
    color_values : np.ndarray or None
        One categorical value per trace, for coloring.
    """
    tuples = dataarray_to_series(da, name_prefix=name_prefix)
    labels = [t[0] for t in tuples]

    non_time_dims = [d for d in da.dims if d != "time"]

    order_values: np.ndarray | None = None

    if order_by is not None and order_by in da.coords:
        raw = _coord_values_per_trace(da, order_by, non_time_dims)
        if raw is not None:
            try:
                order_values = raw.astype(float)
            except (ValueError, TypeError):
                order_values = None
    elif order_by is None and len(non_time_dims) == 1:
        # Auto: use the single non-time dimension's coordinate values
        dim_name = non_time_dims[0]
        coord_vals = da.coords[dim_name].values
        try:
            order_values = coord_vals.astype(float)
        except (ValueError, TypeError):
            order_values = np.arange(len(coord_vals), dtype=float)

    color_values: np.ndarray | None = None
    if color_by is not None:
        color_values = _coord_values_per_trace(da, color_by, non_time_dims)

    # Apply ordering
    if order_values is not None and len(order_values) == len(tuples):
        sort_idx = np.argsort(order_values)
        if descending:
            sort_idx = sort_idx[::-1]
        tuples = [tuples[i] for i in sort_idx]
        labels = [labels[i] for i in sort_idx]
        order_values = order_values[sort_idx]
        if color_values is not None and len(color_values) == len(sort_idx):
            color_values = color_values[sort_idx]

    return tuples, order_values, labels, color_values


# ---------------------------------------------------------------------------
# Array (heatmap) conversion
# ---------------------------------------------------------------------------


def dataarray_to_arrays(
    da: xr.DataArray,
    *,
    split_on: str | None = None,
    sort_on: str | None = None,
    colormap: "str | Colormap | list | dict | Callable[..., Any]" = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    decim_method: str = "peak",
    name_prefix: str = "",
    label: "str | Callable[..., str] | None" = None,
):
    """Convert a DataArray into one or more :class:`ArraySeries` heatmaps.

    Parameters
    ----------
    da : xr.DataArray
        Must have a ``'time'`` dimension. After optional grouping by
        ``split_on`` it must have exactly one remaining non-time dim
        (the row dim).
    split_on : str or None
        Coordinate or dim name to group by — produces one ``ArraySeries``
        per unique value.  If *None*, the input must already have exactly
        one non-time dim.
    sort_on : str or None
        Coordinate name on the row dim controlling y-axis row order
        (sorted ascending).  If *None*, the row dim's coordinate values
        determine order (or simple integer order).
    colormap : str, Colormap, list, dict, or callable
        Per-group colormap selection. Accepts:

        * A matplotlib colormap name (string, e.g. ``"magma"``).
        * A ``matplotlib.colors.Colormap`` instance (e.g. ``cmcrameri.cm.batlow``).
        * A list — one entry per split group in iteration order, cycling.
        * A dict ``{split_val: cmap}`` — keyed by split value (KeyError on miss).
        * A callable ``(split_val, sub_da) -> str | Colormap`` — invoked per
          split group.  1-arg callables ``(split_val) -> ...`` also accepted.
    vmin, vmax : float or None
        Color scale limits.  Default is robust 1–99 percentile per group.
    decim_method : str
        ``"peak"`` (max-absolute per bin, default) or ``"mean"``.
    name_prefix : str
        Optional prefix prepended to every subplot name (default-naming
        path only; ignored when ``label`` is supplied).
    label : str, callable, or None
        Subplot-name override.  ``None`` (default) uses the auto-format
        ``"{name_prefix}: {split_on}={split_val}"`` (or just the prefix
        when there's no split).  A string is used verbatim per subplot.
        A callable ``(split_val, sub_da) -> str`` (or 1-arg
        ``(split_val) -> str``) is invoked per split group.

    Returns
    -------
    list[ArraySeries]
    """
    # Local imports keep the loader Qt-free at module-load time.
    import xarray as xr  # lazy

    from loupe.app import (
        ARRAY_MIPMAP_TARGET_MIN_COLS,
        ARRAY_MIPMAP_THRESHOLD,
        ArraySeries,
    )

    if "time" not in da.dims:
        raise ValueError(
            f"DataArray must have a 'time' dimension. Found dims: {da.dims}"
        )

    # Time as float64 seconds (mirror dataarray_to_series behaviour).
    time_raw = da.coords["time"].values
    if np.issubdtype(time_raw.dtype, np.datetime64):
        t0 = time_raw[0]
        time_vals = (time_raw - t0).astype("timedelta64[ns]").astype(float) / 1e9
    else:
        time_vals = time_raw.astype(float)

    # Resolve groups: list of (split_value or None, sub_da).
    if split_on is None:
        groups: list[tuple[object, xr.DataArray]] = [(None, da)]
    else:
        if split_on not in da.coords and split_on not in da.dims:
            raise ValueError(
                f"split_on='{split_on}' not found in DataArray coords or dims. "
                f"Available coords: {list(da.coords)}; dims: {da.dims}"
            )
        groups = [(val, sub_da) for val, sub_da in da.groupby(split_on)]

    # Resolve colormaps per group. Each entry may be a string name or a
    # matplotlib.colors.Colormap instance — both are accepted downstream.
    # Callables and dicts are resolved per-group inside the loop below; for
    # those, ``cmaps`` stays as a placeholder default fallback list.
    cmap_callable = callable(colormap) and not isinstance(colormap, type)
    cmap_dict = isinstance(colormap, dict)
    if cmap_callable or cmap_dict:
        cmaps = ["magma"]  # fallback; the resolver below takes precedence
    elif isinstance(colormap, (list, tuple)):
        cmaps = list(colormap)
        if not cmaps:
            cmaps = ["magma"]
    else:
        cmaps = [colormap]

    da_name = str(da.name) if da.name else ""
    array_series_list = []

    for gi, (split_val, sub_da) in enumerate(groups):
        non_time_dims = [d for d in sub_da.dims if d != "time"]
        if len(non_time_dims) != 1:
            extras = ", ".join(non_time_dims) or "(none)"
            raise ValueError(
                "Array mode requires exactly one non-time dimension per "
                f"subplot after split. Found dims: {sub_da.dims}. "
                f"Non-time dims after split: {extras}. Use split_on= to group "
                "or pre-select extra dims with .sel()."
            )
        row_dim = non_time_dims[0]

        # Row order via sort_on (or row dim's own coord).
        if sort_on is not None and sort_on in sub_da.coords:
            sort_vals = np.asarray(sub_da.coords[sort_on].values)
            if sort_vals.shape == (sub_da.sizes[row_dim],):
                try:
                    sort_idx = np.argsort(sort_vals.astype(float))
                except (ValueError, TypeError):
                    sort_idx = np.argsort(sort_vals)
                sub_da = sub_da.isel({row_dim: sort_idx})
                row_labels = sub_da.coords[sort_on].values
            else:
                # sort_on is not 1-D over the row dim — leave order as-is
                row_labels = None
        else:
            try:
                row_labels = sub_da.coords[row_dim].values
            except KeyError:
                row_labels = None

        # Materialize as float32 (rows, time).
        Y = sub_da.transpose(row_dim, "time").values.astype(np.float32, copy=False)
        # Replace NaN with -inf sentinel so np.max decimation is correct
        # without nan-aware overhead. (See plan: Performance Layer 3.)
        if np.any(np.isnan(Y)):
            Y = np.where(np.isnan(Y), -np.inf, Y).astype(np.float32, copy=False)

        # Default vmin/vmax via robust percentile of finite values.
        if vmin is None or vmax is None:
            finite = Y[np.isfinite(Y)]
            if finite.size > 0:
                lo = float(np.percentile(finite, 1.0))
                hi = float(np.percentile(finite, 99.0))
                if hi <= lo:
                    hi = lo + 1.0
            else:
                lo, hi = 0.0, 1.0
            this_vmin = lo if vmin is None else float(vmin)
            this_vmax = hi if vmax is None else float(vmax)
        else:
            this_vmin = float(vmin)
            this_vmax = float(vmax)

        # Pick a colormap for this group: callable / dict take precedence,
        # else cycle through the list.
        if cmap_callable:
            cmap_value = _call_with_optional_subda(colormap, split_val, sub_da)
        elif cmap_dict:
            cmap_value = colormap[split_val]
        else:
            cmap_value = cmaps[gi % len(cmaps)]

        # Subplot name: explicit ``label`` wins; otherwise auto-format
        # ``"{prefix}: {split_on}={split_val}"`` (or just prefix when no split).
        if callable(label):
            name = str(_call_with_optional_subda(label, split_val, sub_da))
        elif isinstance(label, str):
            name = label
        elif split_val is None:
            name = name_prefix or da_name or "array"
        else:
            base = name_prefix or da_name or "data"
            name = f"{base}: {split_on}={split_val}"

        # Build mip-map for big arrays (cheap perf insurance).
        mipmap = None
        if Y.size >= ARRAY_MIPMAP_THRESHOLD:
            mipmap = _build_mipmap(Y, decim_method, ARRAY_MIPMAP_TARGET_MIN_COLS)

        array_series_list.append(ArraySeries(
            name=name,
            t=time_vals.astype(float, copy=True),
            Y=Y,
            row_labels=np.asarray(row_labels) if row_labels is not None else None,
            row_dim_name=row_dim,
            colormap=cmap_value,
            vmin=this_vmin,
            vmax=this_vmax,
            decim_method=decim_method,
            mipmap_levels=mipmap,
        ))

    return array_series_list


def _build_mipmap(
    Y: np.ndarray,
    decim_method: str,
    target_min_cols: int,
) -> list[np.ndarray]:
    """Build a power-of-2 column-decimated pyramid for an array buffer.

    Level 0 is *Y* itself.  Each subsequent level halves the column count
    by reducing pairs with ``np.max`` (peak) or ``np.mean``.  Stops when
    column count is at or below ``target_min_cols``.
    """
    levels: list[np.ndarray] = [Y]
    use_peak = decim_method != "mean"
    while levels[-1].shape[1] > target_min_cols:
        prev = levels[-1]
        n = (prev.shape[1] // 2) * 2
        if n < 2:
            break
        reshaped = prev[:, :n].reshape(prev.shape[0], -1, 2)
        if use_peak:
            level = reshaped.max(axis=2)
        else:
            level = reshaped.mean(axis=2)
        levels.append(level.astype(np.float32, copy=False))
    return levels


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
