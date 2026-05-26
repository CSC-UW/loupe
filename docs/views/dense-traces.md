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
| `sample_markers` | _unsupported_ | Sample markers require stacked-subplots mode. |

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
- Rendering: `LoupeApp._refresh_dense_curves`.
- Per-group state: the `DenseGroup` dataclass in `loupe.app`.

## Notes & gotchas

- Mean subtraction is performed once on the full series at load time; `gain` is a multiplier applied to the centered values.
- When `hue` is set, the legend currently has no built-in viewer; that's an open TODO item.
- When `traces_per_page` is set on multiple dense plots, `Shift+Alt+wheel` pages whichever plot is under the cursor.
