# Dense traces (`TraceConfig`, `mode="dense"`)

Many traces (potentially hundreds) on a single pair of axes, EEG-style. Each trace is mean-subtracted, multiplied by `gain`, and vertically offset by its `order_by` value. Best for high-channel-count recordings — neuropixel LFPs, multi-site EEG, dense imaging arrays.

Defined in `src/loupe/__init__.py:69-169` (shared dataclass with stacked mode).

## Full parameter reference

| Param | Default | Purpose |
| --- | --- | --- |
| `data` | _required_ | `xr.DataArray` with a `'time'` dimension. |
| `mode` | _required_ | Must be `"dense"`. |
| `order_by` | `None` | Coordinate name controlling trace order and vertical spacing (e.g. `"y"` for electrode depth). When unset and there is exactly one non-time dim, that dim's coordinate is used automatically. |
| `descending` | `False` | Reverse the trace order. |
| `gain` | `1.0` | Initial amplitude multiplier. Adjustable at runtime via `Alt+wheel` or `Ctrl+G`. |
| `step` | `1` | Show every _n_-th trace. |
| `traces_per_page` | `None` | How many traces to show at once. `None` = all. When set, a vertical scrollbar appears. |
| `hue` | `None` | Coordinate name for categorical coloring (e.g. anatomical region). |
| `palette` | `None` | `dict {hue_value: color}` or list. |
| `color` | `None` | Single color overrides `hue` / `palette`. |
| `array_name` | `False` | Prefix for the dense group's name. See [stacked-traces.md](stacked-traces.md). |
| `sample_markers` | `None` | Optional list of `SampleMarkers` overlays, aligned by `order_by` and drawn at each trace's displayed y. See "Sample markers" below. |

## Sample markers

Dense mode supports the same sample-aligned marker overlays as stacked mode (spikes, events, …) via the `sample_markers` field:

```python
from loupe import TraceConfig, SampleMarkers, view

view(TraceConfig(
    lfp, mode="dense", order_by="y",
    sample_markers=[
        SampleMarkers(marker="o", color="#ff0000", bool_array=spike_mask),
    ],
))
```

Each `bool_array` must be a boolean `xr.DataArray` with **the same dims and shape as the dense `TraceConfig`'s `data`** — for a dense view of `N` traces × `S` samples that is an `(N, S)` array whose dims match `data` (e.g. `("channel", "time")`). `True` at `[trace j, sample i]` draws a marker on trace _j_ at `(time[i], displayed_y)`, where displayed y is the dense transform `(value − mean) × gain + offset` — so markers track the traces as you adjust gain (`Alt+wheel`) or page vertically. It is **never** a 1-D `(S,)` mask; pass one full `(N, S)` array per `SampleMarkers` set.

If `data` has multiple non-time dims (e.g. `(probe, channel, time)`), the bool array must carry the same dims; both are flattened to the `N` traces with the identical sort permutation, so marker trace _k_ lines up with data trace _k_. Alignment is by coordinate label (`reindex_like(data, fill_value=False)`): coords must match `data`'s, a missing dim raises, and non-overlapping coords are silently filled `False`.

Each marker set is drawn as a single aggregated `pg.ScatterPlotItem` per group (one scene item / draw call regardless of trace count), so markers add negligible cost to the dense scroll path. Markers are one color per set (no hue tinting). Restyle live via **View → Adjust Sample Marker Properties…**

`marker="vline"` is supported here too: the set becomes one aggregated `pg.PlotCurveItem` (`connect="pairs"`) drawing a full-height vertical line at every flagged sample of every visible trace, spanning beyond the visible y-range and excluded from auto-range (`size` = line width in px; defaults width 1.0, alpha 200). See [stacked-traces.md](stacked-traces.md#vertical-line-markers-markervline).

Unlike stacked markers, dense markers are unrestricted: multiple dense `TraceConfig`s may each carry markers, and they coexist freely with `HeatmapConfig` / `RasterConfig` / stacked traces in the same window.

## Multiple dense `TraceConfig`s

Multiple dense `TraceConfig`s coexist with synchronized X-axes; each picks its own gain, step, and traces_per_page independently. Mix freely with stacked-subplots:

```python
view([
    TraceConfig(lfp, mode="dense", order_by="y", descending=True, traces_per_page=16),
    TraceConfig(emg, mode="stacked-subplots"),
])
```

## Runtime controls

| Action | Binding |
| --- | --- |
| Adjust trace gain | `Alt+wheel` |
| Smooth vertical scroll (~3 traces / notch) | `Shift+Alt+wheel` |
| Open Dense View Controls (gain, step, traces_per_page) | `Ctrl+G` |
| Subplot height / visibility / order | `Ctrl+H` |

## Hot-path entry points

- Data conversion: `loupe.xr_loader.convert_xarray_inputs_with_order` (`src/loupe/__init__.py:745`).
- Rendering: `LoupeApp._refresh_dense_curves` (curves and marker scatters share the same per-window loop).
- Per-group state: the `DenseGroup` dataclass in `loupe.app`.
- Sample-marker overlay: `loupe.xr_loader.convert_event_arrays_aligned_with`; rendered as one aggregated `pg.ScatterPlotItem` per marker set in `LoupeApp.dense_marker_scatters`.

## Notes & gotchas

- Mean subtraction is performed once on the full series at load time; `gain` is a multiplier applied to the centered values.
- When `hue` is set, the legend currently has no built-in viewer; that's an open TODO item.
- When `traces_per_page` is set on multiple dense plots, `Shift+Alt+wheel` pages whichever plot is under the cursor.
