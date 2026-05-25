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

    label: str  # shared dimension value label, e.g. "1" or "prefix: 1"
    traces: list[OverlayTrace] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def dataarray_to_series(
    da: xr.DataArray,
    name_prefix: str = "",
    reporter=None,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Flatten a DataArray into a list of ``(name, t_array, y_array)`` tuples.

    Parameters
    ----------
    da : xr.DataArray
        Must have a ``'time'`` dimension with coordinates.
    name_prefix : str, optional
        Caller-resolved array-name string.  When non-empty, multi-trace
        outputs are named ``f"{name_prefix}: {coord_value}"`` and 1-D
        outputs take *name_prefix* verbatim.  When empty (the default),
        multi-trace outputs use just the coord value and 1-D outputs have
        an empty name.

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
        results.append((name_prefix, time_vals.copy(), da.values.astype(float)))
    else:
        dim_coords = [da.coords[d].values for d in non_time_dims]
        total = 1
        for c in dim_coords:
            total *= len(c)

        for idx, combo in enumerate(itertools.product(*dim_coords)):
            sel_dict = dict(zip(non_time_dims, combo))
            y = da.sel(sel_dict).values.astype(float)

            suffix = "-".join(str(v) for v in combo)
            name = f"{name_prefix}: {suffix}" if name_prefix else suffix
            results.append((name, time_vals.copy(), y))
            if reporter is not None:
                reporter.item(idx, total, detail=name)

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

    results: list[tuple[str, np.ndarray, np.ndarray]] = []
    for da in data:
        results.extend(dataarray_to_series(da))

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


def _trace_order_values(
    da: xr.DataArray,
    order_by: str | None,
) -> np.ndarray | None:
    """Compute one ordering value per trace, in itertools.product order.

    Returns *None* when no ordering can be derived (e.g. the order_by
    coordinate isn't aligned to a single non-time dim).
    """
    non_time_dims = [d for d in da.dims if d != "time"]

    if order_by is not None and order_by in da.coords:
        raw = _coord_values_per_trace(da, order_by, non_time_dims)
        if raw is None:
            return None
        try:
            return raw.astype(float)
        except (ValueError, TypeError):
            return None
    if order_by is None and len(non_time_dims) == 1:
        dim_name = non_time_dims[0]
        coord_vals = da.coords[dim_name].values
        try:
            return coord_vals.astype(float)
        except (ValueError, TypeError):
            return np.arange(len(coord_vals), dtype=float)
    return None


def _compute_trace_sort_index(
    da: xr.DataArray,
    *,
    order_by: str | None,
    descending: bool,
    n_traces: int,
) -> np.ndarray | None:
    """Permutation that turns itertools.product order into the user's order_by order.

    Returns *None* when no reordering applies. Uses ``kind='stable'`` so two
    DataArrays with identical ordering values produce byte-identical permutations.
    """
    order_values = _trace_order_values(da, order_by)
    if order_values is None or len(order_values) != n_traces:
        return None
    sort_idx = np.argsort(order_values, kind="stable")
    if descending:
        sort_idx = sort_idx[::-1]
    return sort_idx


def convert_xarray_inputs_with_order(
    da: xr.DataArray,
    order_by: str | None = None,
    descending: bool = False,
    name_prefix: str = "",
    color_by: str | None = None,
    reporter=None,
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
    tuples = dataarray_to_series(da, name_prefix=name_prefix, reporter=reporter)
    labels = [t[0] for t in tuples]

    non_time_dims = [d for d in da.dims if d != "time"]

    order_values = _trace_order_values(da, order_by)

    color_values: np.ndarray | None = None
    if color_by is not None:
        color_values = _coord_values_per_trace(da, color_by, non_time_dims)

    sort_idx = _compute_trace_sort_index(
        da, order_by=order_by, descending=descending, n_traces=len(tuples)
    )
    if sort_idx is not None:
        tuples = [tuples[i] for i in sort_idx]
        labels = [labels[i] for i in sort_idx]
        if order_values is not None:
            order_values = order_values[sort_idx]
        if color_values is not None and len(color_values) == len(sort_idx):
            color_values = color_values[sort_idx]

    return tuples, order_values, labels, color_values


def convert_event_arrays_aligned_with(
    da: xr.DataArray,
    bool_arrays: list[xr.DataArray],
    *,
    order_by: str | None,
    descending: bool,
) -> list[list[np.ndarray]]:
    """Flatten each bool DataArray into a list of per-series 1-D bool arrays
    aligned 1:1 with the series produced by
    ``convert_xarray_inputs_with_order(da, order_by=..., descending=...)``.

    Each input bool array is reindexed onto *da*'s coords (``fill_value=False``)
    so that dim/coord mismatches surface as silent False rather than crashes.
    The same flattening (``itertools.product`` over non-time dims) and stable
    sort permutation are applied as for the data array, guaranteeing the
    returned per-series bool arrays line up with the rendered traces.

    Parameters
    ----------
    da : xr.DataArray
        Reference DataArray. Must have a ``'time'`` dimension.
    bool_arrays : list[xr.DataArray]
        Boolean DataArrays. Each must share dims with *da*.
    order_by, descending
        Same values used for *da* in ``convert_xarray_inputs_with_order``.

    Returns
    -------
    list[list[np.ndarray]]
        Shape ``[n_layers][n_series]``. Each inner array is 1-D ``bool`` of
        length ``len(da.coords['time'])``.
    """
    import xarray as xr  # lazy

    if "time" not in da.dims:
        raise ValueError(
            f"DataArray must have a 'time' dimension. Found dims: {da.dims}"
        )

    n_time = int(da.sizes["time"])
    sort_idx = _compute_trace_sort_index(
        da,
        order_by=order_by,
        descending=descending,
        n_traces=int(np.prod([
            da.sizes[d] for d in da.dims if d != "time"
        ]) or 1),
    )

    out: list[list[np.ndarray]] = []
    for layer_i, arr in enumerate(bool_arrays):
        if not isinstance(arr, xr.DataArray):
            raise TypeError(
                f"bool_event_arrays[{layer_i}] must be an xr.DataArray, "
                f"got {type(arr).__name__}"
            )
        missing = [d for d in da.dims if d not in arr.dims]
        if missing:
            raise ValueError(
                f"bool_event_arrays[{layer_i}] is missing dims {missing}; "
                f"expected dims compatible with {da.dims}, got {arr.dims}"
            )
        aligned = arr.reindex_like(da, fill_value=False).astype(bool)
        tuples = dataarray_to_series(aligned)
        per_series = [np.asarray(y, dtype=bool) for _, _, y in tuples]
        for si, b in enumerate(per_series):
            if b.shape != (n_time,):
                raise ValueError(
                    f"bool_event_arrays[{layer_i}] series {si} has shape "
                    f"{b.shape}, expected ({n_time},)"
                )
        if sort_idx is not None and len(sort_idx) == len(per_series):
            per_series = [per_series[i] for i in sort_idx]
        out.append(per_series)
    return out


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
    array_name: "bool | str | Callable[..., str]" = False,
    reporter=None,
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
    array_name : bool, str, or callable
        Controls subplot names.  ``False`` (default) names subplots with
        just ``split_val`` (or an empty string when there's no split).
        ``True`` uses ``data.name`` as the prefix (``ValueError`` if
        unset).  A string is used as the prefix verbatim.  A callable
        ``(split_val, sub_da) -> str`` (or 1-arg ``(split_val) -> str``)
        is invoked per split group and its return value is the full
        subplot name.

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
    # Note: matplotlib Colormap instances are themselves callable (they map
    # values to RGBA), so we must exclude them from the resolver branch —
    # otherwise we would call ``cmap(split_val)`` and get a numeric error.
    try:
        from matplotlib.colors import Colormap as _MplColormap
    except ImportError:
        _MplColormap = ()  # isinstance check becomes a no-op
    cmap_callable = (
        callable(colormap)
        and not isinstance(colormap, type)
        and not isinstance(colormap, _MplColormap)
    )
    cmap_dict = isinstance(colormap, dict)
    if cmap_callable or cmap_dict:
        cmaps = ["magma"]  # fallback; the resolver below takes precedence
    elif isinstance(colormap, (list, tuple)):
        cmaps = list(colormap)
        if not cmaps:
            cmaps = ["magma"]
    else:
        cmaps = [colormap]

    if array_name is True:
        if not da.name:
            raise ValueError(
                "array_name=True requires data.name; set the DataArray's "
                "name or pass an explicit string."
            )
        name_prefix = str(da.name)
    elif array_name is False:
        name_prefix = ""
    elif callable(array_name):
        name_prefix = None  # resolved per-group below
    else:
        name_prefix = str(array_name)

    array_series_list = []

    n_groups = len(groups)
    for gi, (split_val, sub_da) in enumerate(groups):
        if reporter is not None:
            reporter.item(
                gi, n_groups, detail=str(split_val) if split_val is not None else ""
            )
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

        # Subplot name: callable array_name wins; otherwise combine the
        # resolved prefix with the split value.
        if callable(array_name):
            name = str(_call_with_optional_subda(array_name, split_val, sub_da))
        elif split_val is None:
            name = name_prefix
        elif name_prefix:
            name = f"{name_prefix}: {split_on}={split_val}"
        else:
            name = f"{split_on}={split_val}"

        # Build mip-map for big arrays (cheap perf insurance).
        mipmap = None
        if Y.size >= ARRAY_MIPMAP_THRESHOLD:
            if reporter is not None:
                reporter.phase(f"Building mipmap for {name}")
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
    name_prefix: str = "",
    reporter=None,
) -> list[OverlayGroup]:
    """Group traces from multiple DataArrays by a shared dimension.

    Parameters
    ----------
    data : list[xr.DataArray]
        Two or more DataArrays, each with ``'time'`` and ``overlay_dim``.
    overlay_dim : str
        Dimension to overlay on (e.g. ``'syn_id'``). Traces sharing the same
        coordinate value on this dimension are grouped into a single subplot.
    name_prefix : str, optional
        Prefix prepended to each subplot label as ``f"{name_prefix}: {val}"``.
        Empty string (default) leaves the label as the raw value.

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

    # Source names default to the DataArray's own .name (used for legend); a
    # missing .name renders as an empty string rather than a synthesized id.
    source_names = [str(da.name) if da.name else "" for da in data]

    # Collect the union of overlay_dim values across all arrays
    overlay_vals_set: dict[object, None] = {}  # ordered set via dict
    for da in data:
        for v in da.coords[overlay_dim].values:
            overlay_vals_set[v] = None
    overlay_vals = list(overlay_vals_set.keys())

    # Determine extra non-time, non-overlay dims (iterate over these)
    extra_dims = [d for d in data[0].dims if d not in ("time", overlay_dim)]

    def _with_prefix(label: str) -> str:
        return f"{name_prefix}: {label}" if name_prefix else label

    # Build groups
    groups: list[OverlayGroup] = []

    if not extra_dims:
        # Simple case: just overlay_dim
        n_vals = len(overlay_vals)
        for vi, val in enumerate(overlay_vals):
            if reporter is not None:
                reporter.item(vi, n_vals, detail=str(val))
            group = OverlayGroup(label=_with_prefix(str(val)))
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
        total_extra = 1
        for c in extra_coords:
            total_extra *= len(c)
        n_total = len(overlay_vals) * max(1, total_extra)
        idx = 0
        for val in overlay_vals:
            for combo in itertools.product(*extra_coords):
                extra_label = "-".join(str(v) for v in combo)
                if reporter is not None:
                    reporter.item(idx, n_total, detail=f"{val}-{extra_label}")
                idx += 1
                group = OverlayGroup(
                    label=_with_prefix(f"{val}-{extra_label}")
                )
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
