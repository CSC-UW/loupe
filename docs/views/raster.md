# Raster (`RasterConfig`)

Displays discrete events as vertical lines in a raster format — e.g. neural spike rasters, behavioral events, stimulus onsets. Backed by a polars DataFrame; one row per event, one row of the raster per unique value of `order_by`.

Defined in `src/loupe/__init__.py:225-316`.

## Full parameter reference

| Param | Default | Purpose |
| --- | --- | --- |
| `data` | _required_ | `pl.DataFrame` of events. Must contain `time_col` and `order_by`. May also be a `loupe.tunable(...)` (or bare zero-arg callable) returning such a DataFrame — see *Live tuning* below. |
| `time_col` | _required_ | Column with event timestamps in seconds. |
| `order_by` | _required_ | Column whose values identify the raster row for each event. Rows are sorted by the unique values of this column (unless `rows` is given). |
| `split_by` | `None` | Column(s) to split into separate raster subplots. `None` = single subplot. |
| `alpha_by` | `None` | Column for per-event opacity, normalized to `alpha_range`. |
| `hue` | `None` | Column whose values determine per-event color. Takes precedence over `color`. |
| `palette` | `None` | Per-`hue`-value (or per-group, when `hue` is unset) color mapping: dict `{value: (R,G,B)}` / `{value: "#RRGGBB"}`, a list assigned in sorted-value order, or a single tuple applied to all values. When `hue` and `split_by` are both set, the mapping is shared across subplots so the same value always renders as the same color. |
| `color` | `None` | Single color override (`"#RRGGBB"` or `(R,G,B)`). Takes precedence over `palette`. Ignored when `hue` is set (warns). |
| `alpha_range` | `(0.3, 1.0)` | `(min, max)` alpha bounds when `alpha_by` is set. |
| `array_name` | `""` | Subplot label prefix. `""` (default) leaves grouped subplots labeled by raw group values. A non-empty string is used verbatim. A callable `(group_val, sub_df) -> str` returns the full subplot name. Multi-column groups join with `"-"` (e.g. `"imec0-CA1-SR"`). |
| `horizontal_separators` | `None` | Values in `order_by` space at which to draw a thin horizontal line plus a small vertical gap — a purely visual border (e.g. to delimit units from different probes in one raster). Each value `v` draws the line just below the row whose `order_by` value is `v`. Resolved per subplot under `split_by`; out-of-range / on-boundary values are ignored. |
| `separator_params` | `None` | Optional styling dict for the separators: `"gap"` (row-units, default `0.6`), `"color"` (hex / RGB(A), default gray `(120,120,120)`), `"width"` (px, default `1.0`). Unknown keys warn. Ignored unless `horizontal_separators` is set. |
| `rows` | `None` | Explicit, ordered list of `order_by` values to use as raster rows (row `i` is `rows[i]`). Rows with no events are still drawn (empty) and events whose `order_by` value is not listed are dropped. Applies per subplot under `split_by`. An empty unsplit DataFrame with `rows` still yields an (empty) subplot with the full row layout. Use it to keep one row per unit when `data` is live-tuned. |

## Loader

```python
RasterConfig.from_parquet(
    path,                       # str or list[str] (concatenated)
    *,
    time_col,
    order_by,
    **raster_kwargs,
)
```

Files using legacy `"t_sec"` are auto-renamed to `time_col` for backward compatibility. Backed by `loupe.df_loader.load_dataframe_from_parquet`.

## Usage

```python
import polars as pl
from loupe import view, RasterConfig

ev = pl.read_parquet("spikes.parquet")
view(RasterConfig(
    ev,
    time_col="time",
    order_by="source_id",
    split_by="dmd",
    alpha_by="snr_denoised",
    palette={"dmd0": "#ff8888", "dmd1": "#88ccff"},
))
```

## Horizontal separators

Pass `horizontal_separators` to insert a small vertical gap plus a thin horizontal line at one or more y-axis positions, so a single raster subplot can be visually divided into blocks — for example, units recorded on different probes within one brain region. The values are in `order_by` space: a value `v` puts the line just *below* the row whose `order_by` value is `v` (rows with `order_by >= v` form the block above the line).

```python
view(RasterConfig(
    ev,
    time_col="time",
    order_by="unit_id",
    split_by="region",                 # one brain region per subplot…
    horizontal_separators=[64, 128],   # …each split into per-probe blocks
    separator_params={"gap": 0.6, "color": "#888888", "width": 1.0},
))
```

The feature is purely cosmetic — it shifts row *positions* apart to open the gap, but does not change the data, the row ordering, or `n_rows`. It composes with `split_by`: each subplot resolves the values against its own rows, so a value only produces a separator in subplots whose rows straddle it. Values below all rows, above all rows, or landing exactly on an existing block boundary are silently ignored. Leaving `horizontal_separators` unset keeps the layout byte-identical to before.

## Live tuning (Tuner)

`data` may be a `loupe.tunable(...)` whose `Param` arguments become sliders in the
Tuner dock; moving a slider re-evaluates the DataFrame and re-renders the raster
in place — the idiomatic use is an event catalog filtered by a live threshold:

```python
from loupe import Param, RasterConfig, tunable, view

z = Param(5.0, 3.0, 15.0, step=0.25, name="z ≥")

def above(df, z=5.0):
    return df.filter(pl.col("z_shot") >= z)

view(RasterConfig(
    tunable(above, catalog, z=z),
    time_col="time",
    order_by="syn_id",
    rows=all_syn_ids,              # one row per synapse, whatever survives
))
```

The row layout is pinned: rows come from `rows` when given, otherwise from the
initial render, so tightening the threshold empties rows instead of renumbering
them (events for `order_by` values that were absent initially are dropped).
Subplot groups under `split_by` are matched by name; a group whose events all
vanish renders empty. Sharing one `Param` across several `RasterConfig`s gives a
single slider that drives all of them.

## Runtime controls

| Action | Binding |
| --- | --- |
| Toggle Proportional Raster Plots | `Ctrl+Shift+R` |
| Subplot height / visibility / order | `Ctrl+H` |

Brightness, event height, and event thickness are adjusted via View → Adjust Raster Brightness… / Raster Event Height… / Raster Event Thickness… (no dedicated keybinding).

"Proportional Raster Plots" (on by default) sizes raster subplots by row count, so a 1000-row raster gets proportionally more vertical space than a 50-row one.

## Hot-path entry points

- Conversion: `loupe.df_loader.dataframe_to_raster_series` (`src/loupe/__init__.py:681`).
- Color parsing: `loupe._parse_raster_color` (`src/loupe/__init__.py:427`).
- Render: `RasterSeries` in `loupe.app`.

## Notes

- `hue` + `color`: setting both warns and uses `hue` (it's the more specific signal).
- `color` + `palette`: setting both warns and uses `color`.
- Rows are integer-indexed internally — `order_by` values can be any type, but the unique-value sort determines row order (or `rows`, when given).
