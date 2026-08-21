# Heatmap (`HeatmapConfig`)

Renders an `xr.DataArray` as a 2-D heatmap (imshow-style) with time on the x-axis and one row per entry of a non-time dimension. Designed for inspecting many traces at once at fine-grained detail — e.g. `dF[syn_id, time]` shown as a heatmap with synapses on the y-axis — while keeping all of Loupe's synchronized cursor / labeling / video / hypnogram infrastructure.

Defined in `src/loupe/__init__.py:172-222`.

## Full parameter reference

| Param | Default | Purpose |
| --- | --- | --- |
| `data` | _required_ | `xr.DataArray` with `'time'` plus exactly one non-time dim per subplot (after split). |
| `split_by` | `None` | Coordinate or dim name to split into one subplot per unique value (e.g. `"dend-ID"` → one heatmap per dendrite). Backed by `xr.DataArray.groupby`. |
| `order_by` | `None` | Coordinate name on the row dim controlling y-axis row order. |
| `descending` | `False` | Reverse the `order_by` order. |
| `cmap` | `"magma"` | Matplotlib colormap name or `Colormap` instance. Also accepts a list (cycled per split group), a dict `{split_val: cmap}`, or a callable `(split_val, sub_da) -> str | Colormap`. |
| `vmin` | `None` | Color scale lower bound. Default: 1st percentile per heatmap. |
| `vmax` | `None` | Color scale upper bound. Default: 99th percentile per heatmap. |
| `decim_method` | `"peak"` | Time-axis decimation when zoomed out: `"peak"` (max-absolute per bin, preserves transients) or `"mean"`. |
| `shade_nans` | `False` | Preserve blank NaNs when false. Pass `"#RRGGBB"` to shade NaNs at 0.7 alpha, or `(hex_color, alpha)` with alpha in 0–1. |
| `array_name` | `False` | `False` → subplot named just `"{split_by}={split_val}"`. `True` → prefix with `data.name`. A string → verbatim prefix. A callable `(split_val, sub_da) -> str` → full subplot name. |

Each subplot must have exactly one non-time dim remaining after `split_by` — otherwise a clear error is raised.

## Usage

```python
from loupe import view, HeatmapConfig

# Per-dendrite heatmap, rows ordered by anatomical position:
view(HeatmapConfig(
    dnv,
    split_by="dend-ID",
    order_by="pos",
    cmap=["magma", "viridis", "plasma", "inferno"],
    shade_nans=("#3E1715", 0.85),
))

# Single array (no split):
view(HeatmapConfig(dF_one_dend, order_by="pos"))

# Force a large split heatmap stack to fit without vertical scrolling:
view(
    HeatmapConfig(dnv, split_by="dend-ID", order_by="pos"),
    compact_heatmaps_to_fit=True,
)
```

`compact_heatmaps_to_fit=True` uniformly reduces heatmap heights until the full
visible subplot stack fits in the plot viewport. Other subplot types retain
their requested heights whenever possible, and Loupe recomputes the fit after
window resizes. The same setting is available at runtime under **View → Compact
Heatmap Plots to Fit Screen** and is included in saved View-Configs. Heatmap
data-area heights stay proportional to row count down through one-row arrays;
fixed plot chrome such as the shared bottom time axis is budgeted separately.
When a compacted heatmap's data area is under 40 px tall, Loupe hides its
y-axis name and tick labels automatically and restores them if the plot later
grows. The aligned y-axis gutter and spine remain in place.

## Runtime controls

| Action | Binding |
| --- | --- |
| Open Heatmap Plot Controls dialog | `Ctrl+Shift+H` |
| Subplot height / visibility / order | `Ctrl+H` |
| Toggle viewport-fit heatmap sizing | View → Compact Heatmap Plots to Fit Screen |

The Heatmap Plot Controls dialog provides per-subplot live adjustment of:
- `vmin` / `vmax` (slider + spinbox; "Reset to 1–99% percentile" button)
- Colormap (dropdown of presets, freely editable)
- Decimation method (peak / mean)
- "Apply to all arrays" copies the current row's settings to every other heatmap.

## Performance

Heatmap plots use a layered strategy to stay responsive even with multiple plots loaded:

1. Cursor moves, selection drags, label additions, and Y-zoom skip the heatmap refresh entirely.
2. Each plot caches its last-rendered `(window, view-width, vmin, vmax, cmap, decim_method, shade_nans)` and short-circuits if unchanged.
3. NaN values are sentinel-replaced at load time so refresh uses fast `np.max` / `np.mean` (no nan-aware overhead).
4. Manual NumPy LUT mapping → uint8 RGBA upload bypasses pyqtgraph's per-pixel level math.
5. Arrays exceeding 5 M elements get a power-of-2 mip-map at load time (~2× memory), so pan latency stays O(viewbox-width) regardless of recording length.

## Hot-path entry points

- Conversion: `loupe.xr_loader.dataarray_to_heatmaps` (`src/loupe/__init__.py:725`).
- Render class: `HeatmapSeries` in `loupe.app`.
