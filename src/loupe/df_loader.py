"""Convert Polars DataFrames to RasterSeries for the Loupe viewer.

This module has no Qt dependencies.  Polars imports are lazy so the rest of
the application works without polars installed.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import polars as pl

# ---------------------------------------------------------------------------
# Default color palette for groups (when no explicit colors given)
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


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def dataframe_to_raster_series(
    df: pl.DataFrame,
    *,
    time_col: str = "time",
    y_col: str = "source_id",
    group_col: str | list[str] | None = None,
    alpha_col: str | None = None,
    name: str = "events",
    colors: (
        dict[object, tuple[int, int, int]]
        | list[tuple[int, int, int]]
        | tuple[int, int, int]
        | None
    ) = None,
    alpha_range: tuple[float, float] = (0.3, 1.0),
    color_on: str | None = None,
    color_on_config: dict | None = None,
) -> list:
    """Convert a Polars DataFrame into one or more RasterSeries for raster display.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain *time_col* and *y_col* columns at minimum.
    time_col : str
        Column containing event timestamps in seconds.
    y_col : str
        Column whose values identify the raster row for each event (e.g.
        ``"source_id"``).  Unique values are mapped to contiguous 0-based
        integer indices within each group.
    group_col : str or list[str] or None
        Column(s) used to split the DataFrame into separate RasterSeries
        subplots.  Each unique combination of group_col values becomes one
        subplot.  ``None`` means all events share a single subplot.
    alpha_col : str or None
        Column for per-event opacity.  Values are normalized to
        *alpha_range*.  ``None`` gives every event alpha = 1.0.
    name : str
        Base name for the RasterSeries.  Group values are appended when
        *group_col* is provided (e.g. ``"events: dmd=1"``).
    colors : dict, list, tuple or None
        Color specification per group:

        * dict mapping group value -> ``(R, G, B)``
        * list of ``(R, G, B)`` tuples (assigned in sorted group order)
        * single ``(R, G, B)`` tuple (applied to all groups)
        * ``None``: use default palette

        Ignored when *color_on* is set.
    alpha_range : tuple[float, float]
        ``(min_alpha, max_alpha)`` range for normalizing *alpha_col* values.
    color_on : str or None
        Column whose values determine per-event color.  When set, each event
        is colored according to its value in this column rather than by the
        per-group *colors* setting.
    color_on_config : dict or None
        Optional ``{column_value: color}`` mapping driving the *color_on*
        palette.  Values may be ``(R, G, B)`` tuples or ``"#RRGGBB"`` hex
        strings.  Unique values not listed here cycle through the default
        palette (with a warning).  ``None`` assigns the entire palette from
        the default cycle (no warning).  When *group_col* is also set, this
        palette is shared across every subplot so the same value always
        renders as the same color.

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
    for col in (time_col, y_col):
        if col not in df.columns:
            missing.append(col)
    if group_col is not None:
        gcols = [group_col] if isinstance(group_col, str) else list(group_col)
        for gc in gcols:
            if gc not in df.columns:
                missing.append(gc)
    else:
        gcols = []
    if alpha_col is not None and alpha_col not in df.columns:
        missing.append(alpha_col)
    if color_on is not None and color_on not in df.columns:
        missing.append(color_on)
    if missing:
        raise ValueError(
            f"DataFrame is missing required column(s): {missing}.  "
            f"Available: {df.columns}"
        )

    if df.height == 0:
        return []

    # ---- shared categorical palette (built once across the whole DataFrame) -
    # When color_on is set we precompute a single value -> category-index map
    # and the matching RGB palette, then apply it inside each group's loop so
    # the same column value always renders as the same color across subplots.
    value_to_idx: dict[object, int] | None = None
    shared_category_colors: list[tuple[int, int, int]] | None = None
    na_idx: int | None = None
    if color_on is not None:
        import polars as pl  # lazy

        non_null = df.filter(pl.col(color_on).is_not_null())
        try:
            global_uniques = sorted(non_null[color_on].unique().to_list())
        except TypeError as exc:
            raise ValueError(
                f"color_on column {color_on!r} contains values that cannot be "
                f"sorted ({exc}); use a column with a single, comparable dtype."
            ) from exc

        if len(global_uniques) > RASTER_MAX_CATEGORIES:
            raise ValueError(
                f"color_on column {color_on!r} has {len(global_uniques)} unique "
                f"values, exceeding RASTER_MAX_CATEGORIES={RASTER_MAX_CATEGORIES}. "
                f"For high-cardinality columns prefer a colormap-style binning."
            )

        palette: dict[object, tuple[int, int, int]] = {}
        unmapped: list[object] = []
        if color_on_config is not None:
            for v in global_uniques:
                if v in color_on_config:
                    palette[v] = _parse_raster_color(color_on_config[v])
                else:
                    unmapped.append(v)
        else:
            unmapped = list(global_uniques)
        for i, v in enumerate(unmapped):
            palette[v] = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
        if color_on_config is not None and unmapped:
            warnings.warn(
                f"color_on_config is missing entries for {unmapped!r}; "
                f"falling back to default palette.",
                stacklevel=2,
            )

        value_to_idx = {v: i for i, v in enumerate(global_uniques)}
        shared_category_colors = [palette[v] for v in global_uniques]
        if df[color_on].null_count() > 0:
            na_idx = len(global_uniques)
            shared_category_colors = shared_category_colors + [RASTER_NA_COLOR]
            warnings.warn(
                f"color_on column {color_on!r} contains "
                f"{df[color_on].null_count()} null values; rendered as gray.",
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
    # require explicit opt-in via `colors=...` — group separation comes from
    # being in different subplots, not from auto-cycling a palette.
    def _color_for(group_key: tuple, idx: int) -> tuple[int, int, int]:
        key = group_key[0] if len(group_key) == 1 else group_key
        if colors is None:
            return (255, 255, 255)
        if isinstance(colors, dict):
            if key in colors:
                return colors[key]
            if group_key in colors:
                return colors[group_key]
            return _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)]
        if isinstance(colors, list):
            return colors[idx % len(colors)] if colors else _DEFAULT_COLORS[0]
        # single tuple
        return colors  # type: ignore[return-value]

    # ---- build RasterSeries per group ---------------------------------------
    result: list = []
    for idx, gkey in enumerate(groups):
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
        unique_y = np.sort(gdf[y_col].unique().to_numpy())
        y_map = {v: i for i, v in enumerate(unique_y)}
        raw_y = gdf[y_col].to_numpy()
        yvals = np.array([y_map[v] for v in raw_y], dtype=np.intp)
        n_rows = len(unique_y)

        # alphas
        if alpha_col is not None:
            raw_alpha = gdf[alpha_col].to_numpy().astype(np.float64)
            a_min, a_max = np.nanmin(raw_alpha), np.nanmax(raw_alpha)
            if a_min == a_max:
                alphas = np.full(len(raw_alpha), alpha_range[1])
            else:
                normed = (raw_alpha - a_min) / (a_max - a_min)
                alphas = alpha_range[0] + normed * (alpha_range[1] - alpha_range[0])
            alphas = np.clip(alphas, 0.0, 1.0)
        else:
            alphas = np.ones(len(timestamps), dtype=np.float64)

        # per-event category indices (only when color_on is set)
        if value_to_idx is not None:
            raw_cat = gdf[color_on].to_list()  # list preserves None/null
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

        # name
        if gcols:
            label_parts = [f"{gc}={gv}" for gc, gv in zip(gcols, gkey)]
            series_name = f"{name}: {', '.join(label_parts)}"
        else:
            series_name = name

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
