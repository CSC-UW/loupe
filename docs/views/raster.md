# Raster (`RasterConfig`)

Displays discrete events as vertical lines in a raster format — e.g. neural spike rasters, behavioral events, stimulus onsets. Backed by a polars DataFrame; one row per event, one row of the raster per unique value of `order_by`.

Defined in `src/loupe/__init__.py:225-316`.

## Full parameter reference

| Param | Default | Purpose |
| --- | --- | --- |
| `data` | _required_ | `pl.DataFrame` of events. Must contain `time_col` and `order_by`. |
| `time_col` | _required_ | Column with event timestamps in seconds. |
| `order_by` | _required_ | Column whose values identify the raster row for each event. Rows are sorted by the unique values of this column. |
| `split_by` | `None` | Column(s) to split into separate raster subplots. `None` = single subplot. |
| `alpha_by` | `None` | Column for per-event opacity, normalized to `alpha_range`. |
| `hue` | `None` | Column whose values determine per-event color. Takes precedence over `color`. |
| `palette` | `None` | Per-`hue`-value (or per-group, when `hue` is unset) color mapping: dict `{value: (R,G,B)}` / `{value: "#RRGGBB"}`, a list assigned in sorted-value order, or a single tuple applied to all values. When `hue` and `split_by` are both set, the mapping is shared across subplots so the same value always renders as the same color. |
| `color` | `None` | Single color override (`"#RRGGBB"` or `(R,G,B)`). Takes precedence over `palette`. Ignored when `hue` is set (warns). |
| `alpha_range` | `(0.3, 1.0)` | `(min, max)` alpha bounds when `alpha_by` is set. |
| `array_name` | `""` | Subplot label prefix. `""` (default) leaves grouped subplots labeled by raw group values. A non-empty string is used verbatim. A callable `(group_val, sub_df) -> str` returns the full subplot name. Multi-column groups join with `"-"` (e.g. `"imec0-CA1-SR"`). |

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
- Rows are integer-indexed internally — `order_by` values can be any type, but the unique-value sort determines row order.
