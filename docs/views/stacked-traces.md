# Stacked traces (`TraceConfig`, `mode="stacked-subplots"`)

One subplot per trace, all X-linked. The default `TraceConfig` mode. Best for a small-to-moderate number of traces (single-digit to a few dozen) where each trace deserves its own axis and Y-range.

Defined in `src/loupe/__init__.py:69-169`.

## Full parameter reference

| Param | Default | Purpose |
| --- | --- | --- |
| `data` | _required_ | `xr.DataArray` with a `'time'` dimension. All other dims are flattened into individual traces. |
| `mode` | `"stacked-subplots"` | Set this to `"dense"` for the EEG-style layout — see [dense-traces.md](dense-traces.md). |
| `order_by` | `None` | Coordinate name controlling top-to-bottom subplot order. |
| `descending` | `False` | Reverse the `order_by` order. |
| `hue` | `None` | Coordinate name whose categorical values determine per-trace color. Overridden by `color` when both set. |
| `palette` | `None` | `dict {hue_value: color}` or list assigned in sorted hue-value order. Ignored when `hue` is unset. |
| `color` | `None` | Single color applied to every trace from this DataArray. Overrides `hue` / `palette`. |
| `array_name` | `False` | `False` prepends nothing; `True` uses `data.name` (raises if unset); a string is used verbatim as the prefix. |
| `sample_markers` | `None` | Optional list of `SampleMarkers` overlays. See "Sample markers" below. |
| `overlay_arrays` | `None` | Extra DataArrays drawn on this trace's own axes. See "Overlay arrays" below. |
| `overlay_colors` | `None` | Per-overlay colors (hex / RGB(A) tuples); `None` cycles a distinct palette. |
| `gain`, `step`, `traces_per_page` | — | Dense-mode only; ignored here. |

## Loader

```python
TraceConfig.from_path(
    path,                       # .zarr directory or netCDF file
    *,
    group=None,                 # group within the store
    variable="data",            # variable name inside the dataset
    filter_dict=None,           # {dim: slice(...)} applied before load
    **trace_kwargs,             # forwarded to TraceConfig
)
```

Backed by `loupe.xr_loader.load_xarray_from_path`.

## Sample markers

Stacked-subplots mode supports sample-aligned marker overlays (e.g. spikes, events) via the `sample_markers` field:

```python
from loupe import TraceConfig, SampleMarkers, view

view(TraceConfig(
    da,
    sample_markers=[
        SampleMarkers(marker="o", color="#ff0000", bool_array=spike_mask),
        SampleMarkers(marker="x", color="lime",    bool_array=event_mask),
    ],
))
```

`bool_array` must have the same dims/shape as `data`. `True` at sample _i_ on trace _j_ draws a marker at `(time[i], data[j, i])`. In a stacked view it may also be a `tunable(...)` result (or zero-argument callable) returning that Boolean DataArray, so marker locations update with the rest of the Tuner-bound layers. Dense-mode marker masks remain static.

Marker defaults:
- `'o'` → size 8.0, alpha 110 (semi-transparent filled circle)
- `'vline'` → width 1.0 px, alpha 200 (see below)
- any other symbol → size 9.0, alpha 255 (solid stroke)

### Vertical-line markers (`marker="vline"`)

`marker="vline"` draws a **full-height vertical line** through the subplot at every flagged sample instead of a symbol at the sample's value — the natural way to stamp event *times* (spikes, stimulus onsets) onto the trace they were detected on. `size` is the line width in pixels. Each series gets one `pg.PlotCurveItem` (`connect="pairs"`) that is sliced to the visible window like the trace itself, spans well beyond the visible y-range, is excluded from auto-range, and is re-spanned whenever the y-range changes, so the lines never affect or lag the y-scale.

```python
spike_mask = xr.zeros_like(dff, dtype=bool)
spike_mask[np.searchsorted(dff.time.values, spike_times)] = True
view([
    TraceConfig(dff, sample_markers=[SampleMarkers(marker="vline", color="#ff0000", bool_array=spike_mask)]),
    TraceConfig(other, sample_markers=[SampleMarkers(marker="vline", color="#0088ff", bool_array=other_mask)]),
])
```

Unlike `global_events=` (which draws across every pane), a `vline` marker set only appears on the traces of the `TraceConfig` it belongs to.

### Constraints

These constraints apply to **stacked-subplots** markers only. Dense mode also supports `sample_markers`, with none of these restrictions — see [dense-traces.md](dense-traces.md).

- Any number of stacked `TraceConfig`s may carry sample markers; each marker set annotates only the traces of the config it is attached to (runtime `SampleMarkers.series_start` anchors it to that config's series block), and plain or dense `TraceConfig`s may sit alongside.
- A stacked `TraceConfig` with sample markers cannot coexist with `HeatmapConfig`, `RasterConfig`, or `Zip` in the same window.

## Overlay arrays

Draw one or more extra DataArrays *on the same axes* as a trace, instead of in their own subplots — e.g. a denoised trace, a rolling-noise band, or a fit over the raw signal:

```python
from loupe import TraceConfig, view

# spks and deconv_std both have only a `time` dim
view([
    TraceConfig(dff),
    TraceConfig(spks, overlay_arrays=[deconv_std]),   # overlay only on this trace
])
```

Each overlay array must share the host's **non-time** dims (the time axis may differ — overlays are sliced to the window independently). When the host produces several traces (a non-time dim like `syn_id`), the overlays are flattened and sorted the same way, so overlay trace _i_ is drawn onto subplot _i_:

```python
# dff and thresholds are both (syn_id, time); each synapse subplot gets its threshold
view(TraceConfig(dff, order_by="pos", overlay_arrays=[thresholds]))
```

Colors: each overlay gets a distinct color from a built-in palette (the same one `Zip` uses); pass `overlay_colors=["#ff0000", (0,255,0)]` to control them (a short list is extended from the palette). The host curve takes a legend entry named after its DataArray's `.name` (falling back to the trace label), and each overlay is labelled by its own `.name`.

### Constraints

- `overlay_arrays` requires `mode="stacked-subplots"`.
- Under fixed-scale (`fixed_scale=True`), the per-subplot Y-range is computed from the host **and** its overlays so nothing is clipped; under auto-range the subplot fits all curves automatically.
- Unlike `Zip`, overlays compose freely with other `TraceConfig`s / `HeatmapConfig`s / rasters in the same window — they are a property of one trace, not a window mode.

## Runtime controls

| Action | Binding |
| --- | --- |
| Per-trace Y-axis autorange / min / max | `Ctrl+D` |
| Y-zoom in / out (hovered plot) | `Ctrl+1` / `Ctrl+2` |
| Subplot height / visibility / order | `Ctrl+H` (Subplot Control Board) |
| Resize focused subplot | `Ctrl+Shift+,` / `Ctrl+Shift+.` / `Ctrl+Shift+0` |

See [KEYBINDINGS.md](../../KEYBINDINGS.md) for the full list.

## Hot-path entry points

- Data conversion: `loupe.xr_loader.convert_xarray_inputs_with_order` (`src/loupe/__init__.py:770`).
- Rendering: `LoupeApp._refresh_curves`, `LoupeApp._apply_x_range`.
- Sample-marker overlay: `loupe.xr_loader.convert_event_arrays_aligned_with` (`src/loupe/__init__.py:788`).
- Overlay arrays: `loupe.xr_loader.convert_overlay_arrays_aligned_with`; rendered as extra `PlotDataItem`s in `LoupeApp.overlay_curve_items`, refreshed in `LoupeApp._refresh_curves`.
