"""Convert Polars DataFrames to RasterSeries for the Loupe viewer.

This module has no Qt dependencies.  Polars imports are lazy so the rest of
the application works without polars installed.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    import polars as pl

# ---------------------------------------------------------------------------
# Default color palette for groups (when no explicit palette given)
# ---------------------------------------------------------------------------

_DEFAULT_COLORS: list[tuple[int, int, int]] = [
    (255, 13, 215),   # magenta/pink  (DMD1 convention)
    (66, 245, 81),    # green         (DMD2 convention)
    (255, 165, 0),    # orange
    (0, 191, 255),    # deep sky blue
    (255, 255, 0),    # yellow
    (148, 103, 189),  # purple
    (255, 127, 80),   # coral
    (0, 255, 255),    # cyan
]


def _call_with_optional_subdf(fn: Callable, group_val, sub_df) -> str:
    """Call *fn* with ``(group_val, sub_df)`` or ``(group_val,)``.

    Mirrors xr_loader's ``_call_with_optional_subda`` for the DataFrame path:
    callable ``array_name`` may accept either signature.
    """
    try:
        return fn(group_val, sub_df)
    except TypeError:
        return fn(group_val)


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def dataframe_to_raster_series(
    df: pl.DataFrame,
    *,
    time_col: str,
    order_by: str,
    split_by: str | list[str] | None = None,
    alpha_by: str | None = None,
    array_name: "str | Callable[..., str]" = "",
    palette: (
        dict[object, tuple[int, int, int]]
        | list[tuple[int, int, int]]
        | tuple[int, int, int]
        | None
    ) = None,
    alpha_range: tuple[float, float] = (0.3, 1.0),
    hue: str | None = None,
    reporter=None,
) -> list:
    """Convert a Polars DataFrame into one or more RasterSeries for raster display.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain *time_col* and *order_by* columns at minimum.
    time_col : str
        Column containing event timestamps in seconds.
    order_by : str
        Column whose values identify the raster row for each event (e.g.
        ``"source_id"``).  Unique values are mapped to contiguous 0-based
        integer indices within each group.
    split_by : str or list[str] or None
        Column(s) used to split the DataFrame into separate RasterSeries
        subplots.  Each unique combination of *split_by* values becomes one
        subplot.  ``None`` means all events share a single subplot.
    alpha_by : str or None
        Column for per-event opacity.  Values are normalized to
        *alpha_range*.  ``None`` gives every event alpha = 1.0.
    array_name : str or callable
        Array-level component of each subplot label.  ``""`` (default)
        leaves grouped subplots labeled by raw group values (e.g.
        ``"CA1-SR"``) and ungrouped subplots labeled with an empty
        string.  A non-empty string is used as a prefix verbatim
        (e.g. ``array_name="units"`` → ``"units: CA1-SR"``).  Multi-column
        groups join values with ``"-"`` (e.g. ``"imec0-CA1-SR"``).
        A callable ``(group_val, sub_df) -> str`` (or 1-arg
        ``(group_val) -> str``) is invoked per group and its return value
        is used as the full subplot name.
    palette : dict, list, tuple or None
        Color specification.  When *hue* is set, *palette* maps each
        *hue*-column value to a color (dict ``{value: color}``, list of
        colors assigned in sorted-value order, or single tuple).  When
        *hue* is unset, *palette* maps *split_by* group values to per-group
        colors (same accepted shapes).  ``None`` defaults to white (no hue)
        or the default palette cycle (with hue).
    alpha_range : tuple[float, float]
        ``(min_alpha, max_alpha)`` range for normalizing *alpha_by* values.
    hue : str or None
        Column whose values determine per-event color.  When set, each event
        is colored according to its value in this column rather than by the
        per-group *palette*.

    Returns
    -------
    list[RasterSeries]
        One per group (or one total if no grouping).
    """
    from loupe.app import (  # lazy to stay Qt-free at import time
        RASTER_MAX_CATEGORIES,
        RASTER_NA_COLOR,
        RasterSeries,
    )
    # Lazy import for the color parser to avoid a circular import at module load.
    from loupe import _parse_raster_color

    # ---- validate columns ---------------------------------------------------
    missing = []
    for col in (time_col, order_by):
        if col not in df.columns:
            missing.append(col)
    if split_by is not None:
        gcols = [split_by] if isinstance(split_by, str) else list(split_by)
        for gc in gcols:
            if gc not in df.columns:
                missing.append(gc)
    else:
        gcols = []
    if alpha_by is not None and alpha_by not in df.columns:
        missing.append(alpha_by)
    if hue is not None and hue not in df.columns:
        missing.append(hue)
    if missing:
        raise ValueError(
            f"DataFrame is missing required column(s): {missing}.  "
            f"Available: {df.columns}"
        )

    if df.height == 0:
        return []

    # ---- shared categorical palette (built once across the whole DataFrame) -
    # When hue is set we precompute a single value -> category-index map
    # and the matching RGB palette, then apply it inside each group's loop so
    # the same column value always renders as the same color across subplots.
    value_to_idx: dict[object, int] | None = None
    shared_category_colors: list[tuple[int, int, int]] | None = None
    na_idx: int | None = None
    if hue is not None:
        import polars as pl  # lazy

        non_null = df.filter(pl.col(hue).is_not_null())
        try:
            global_uniques = sorted(non_null[hue].unique().to_list())
        except TypeError as exc:
            raise ValueError(
                f"hue column {hue!r} contains values that cannot be "
                f"sorted ({exc}); use a column with a single, comparable dtype."
            ) from exc

        if len(global_uniques) > RASTER_MAX_CATEGORIES:
            raise ValueError(
                f"hue column {hue!r} has {len(global_uniques)} unique "
                f"values, exceeding RASTER_MAX_CATEGORIES={RASTER_MAX_CATEGORIES}. "
                f"For high-cardinality columns prefer a colormap-style binning."
            )

        resolved: dict[object, tuple[int, int, int]] = {}
        unmapped: list[object] = []
        if isinstance(palette, dict):
            for v in global_uniques:
                if v in palette:
                    resolved[v] = _parse_raster_color(palette[v])
                else:
                    unmapped.append(v)
            if unmapped:
                warnings.warn(
                    f"palette is missing entries for {unmapped!r}; "
                    f"falling back to default palette.",
                    stacklevel=2,
                )
        elif isinstance(palette, (list, tuple)) and not (
            len(palette) == 3 and all(isinstance(x, (int, float)) for x in palette)
        ):
            # List of colors — assigned in sorted-uniques order, cycling.
            pal_list = list(palette)
            for i, v in enumerate(global_uniques):
                resolved[v] = _parse_raster_color(pal_list[i % len(pal_list)])
        elif palette is not None:
            # Single tuple — apply to all values (collapses every hue value
            # to the same color, which is a degenerate but legal case).
            single = _parse_raster_color(palette)
            for v in global_uniques:
                resolved[v] = single
        else:
            unmapped = list(global_uniques)

        # Anything still unmapped (dict miss, or None palette) gets default cycle.
        for i, v in enumerate(unmapped):
            resolved[v] = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]

        value_to_idx = {v: i for i, v in enumerate(global_uniques)}
        shared_category_colors = [resolved[v] for v in global_uniques]
        if df[hue].null_count() > 0:
            na_idx = len(global_uniques)
            shared_category_colors = shared_category_colors + [RASTER_NA_COLOR]
            warnings.warn(
                f"hue column {hue!r} contains "
                f"{df[hue].null_count()} null values; rendered as gray.",
                stacklevel=2,
            )

    # ---- split into groups --------------------------------------------------
    if gcols:
        groups: list[tuple[object, ...]] = (
            df.select(gcols).unique().sort(gcols).rows()
        )
    else:
        groups = [()]  # single synthetic group

    # ---- resolve color helper -----------------------------------------------
    # Default tick color is white, matching traces.  Per-group palette colors
    # require explicit opt-in via `palette=...` — group separation comes from
    # being in different subplots, not from auto-cycling a palette.
    def _color_for(group_key: tuple, idx: int) -> tuple[int, int, int]:
        key = group_key[0] if len(group_key) == 1 else group_key
        if palette is None:
            return (255, 255, 255)
        if isinstance(palette, dict):
            if key in palette:
                return palette[key]
            if group_key in palette:
                return palette[group_key]
            return _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)]
        if isinstance(palette, list):
            return palette[idx % len(palette)] if palette else _DEFAULT_COLORS[0]
        # single tuple
        return palette  # type: ignore[return-value]

    # ---- build RasterSeries per group ---------------------------------------
    result: list = []
    n_groups = len(groups)
    for idx, gkey in enumerate(groups):
        if reporter is not None:
            label = "-".join(str(gv) for gv in gkey) if gkey else ""
            reporter.item(idx, n_groups, detail=label)
        # filter to group
        if gcols:
            import polars as pl  # lazy

            mask = pl.lit(True)
            for gc, gv in zip(gcols, gkey):
                mask = mask & (pl.col(gc) == gv)
            gdf = df.filter(mask)
        else:
            gdf = df

        # timestamps
        timestamps = gdf[time_col].to_numpy().astype(np.float64)

        # y-values: map unique sorted values to contiguous 0-based ints
        unique_y = np.sort(gdf[order_by].unique().to_numpy())
        y_map = {v: i for i, v in enumerate(unique_y)}
        raw_y = gdf[order_by].to_numpy()
        yvals = np.array([y_map[v] for v in raw_y], dtype=np.intp)
        n_rows = len(unique_y)

        # alphas
        if alpha_by is not None:
            raw_alpha = gdf[alpha_by].to_numpy().astype(np.float64)
            a_min, a_max = np.nanmin(raw_alpha), np.nanmax(raw_alpha)
            if a_min == a_max:
                alphas = np.full(len(raw_alpha), alpha_range[1])
            else:
                normed = (raw_alpha - a_min) / (a_max - a_min)
                alphas = alpha_range[0] + normed * (alpha_range[1] - alpha_range[0])
            alphas = np.clip(alphas, 0.0, 1.0)
        else:
            alphas = np.ones(len(timestamps), dtype=np.float64)

        # per-event category indices (only when hue is set)
        if value_to_idx is not None:
            raw_cat = gdf[hue].to_list()  # list preserves None/null
            cat_idx = np.empty(len(raw_cat), dtype=np.int16)
            for i, v in enumerate(raw_cat):
                if v is None:
                    cat_idx[i] = na_idx if na_idx is not None else 0
                else:
                    cat_idx[i] = value_to_idx.get(v, 0)
        else:
            cat_idx = None

        # sort by time (required by _raster_segment_for_window). All per-event
        # arrays must be reordered together — including the category index.
        order = np.argsort(timestamps)
        timestamps = timestamps[order]
        yvals = yvals[order]
        alphas = alphas[order]
        if cat_idx is not None:
            cat_idx = cat_idx[order]

        if callable(array_name):
            group_val = gkey[0] if len(gkey) == 1 else (gkey if gkey else None)
            series_name = str(_call_with_optional_subdf(array_name, group_val, gdf))
        elif gcols:
            suffix = "-".join(str(gv) for gv in gkey)
            series_name = f"{array_name}: {suffix}" if array_name else suffix
        else:
            series_name = array_name

        color = _color_for(gkey, idx)

        result.append(
            RasterSeries(
                name=series_name,
                timestamps=timestamps,
                yvals=yvals,
                alphas=alphas,
                color=color,
                n_rows=n_rows,
                category_index=cat_idx,
                category_colors=shared_category_colors if cat_idx is not None else None,
            )
        )

    return result


# ---------------------------------------------------------------------------
# Parquet loading helper
# ---------------------------------------------------------------------------


def load_dataframe_from_parquet(
    path: str | list[str],
    time_col: str = "time",
) -> "pl.DataFrame":
    """Load one or more parquet files into a single Polars DataFrame.

    Parameters
    ----------
    path : str or list[str]
        Path(s) to parquet file(s).  Multiple paths are concatenated.
    time_col : str
        Expected time column name.  If the file uses ``"t_sec"`` instead,
        it is automatically renamed for backward compatibility.

    Returns
    -------
    pl.DataFrame
    """
    import polars as pl  # lazy

    paths = [path] if isinstance(path, str) else list(path)
    frames: list[pl.DataFrame] = []
    for p in paths:
        frames.append(pl.read_parquet(p))
    df = pl.concat(frames) if len(frames) > 1 else frames[0]

    # Backward-compat: older files may use "t_sec"
    if "t_sec" in df.columns and time_col not in df.columns:
        df = df.rename({"t_sec": time_col})

    return df
