# Zip overlay (`Zip`)

Co-plots traces sharing a coordinate value across multiple DataArrays. For example, given F, dF/F, and denoised arrays each indexed by `syn_id`, `Zip` produces one subplot per `syn_id` value, with F / dF/F / denoised overlaid in each subplot. Semantics match Python's `zip()` along the named dim.

Defined in `src/loupe/__init__.py:358-424`.

## Full parameter reference

| Param | Default | Purpose |
| --- | --- | --- |
| `traces` | _required_ | List of 2+ `TraceConfig` instances whose `data` shares the dim named by `on`. |
| `on` | _required_ | Coordinate dim to zip on (e.g. `"syn_id"`). |
| `colors` | `None` | One color per wrapped `TraceConfig`. If omitted, a default palette is used. |
| `array_name` | `False` | Subplot-name prefix. `False` leaves labels as the overlay dim value. A string is used verbatim. `True` is rejected (a Zip wraps multiple DataArrays and has no single source name). |

## Per-TraceConfig restrictions inside a `Zip`

Only the `color` field of each wrapped `TraceConfig` applies — `Zip` dictates the per-subplot layout. The following fields must remain at their dataclass defaults; otherwise `__post_init__` raises `ValueError`:

`mode`, `order_by`, `descending`, `gain`, `step`, `traces_per_page`, `hue`, `palette`, `array_name`, `sample_markers`.

## Usage

```python
from loupe import view, TraceConfig, Zip

view(Zip(
    [TraceConfig(F), TraceConfig(dFF), TraceConfig(denoised)],
    on="syn_id",
    colors=["#ff0000", "#00ff00", "#0000ff"],
))
```

## Cross-Config constraints

- Only **one** `Zip` per window.
- A `Zip` cannot coexist with `TraceConfig` or `HeatmapConfig` in the same window. Move those traces into the `Zip`, or remove the `Zip`. `RasterConfig` and `VideoConfig` are fine alongside.

## Hot-path entry points

- Conversion: `loupe.xr_loader.convert_xarray_inputs_overlay` (`src/loupe/__init__.py:675`).
- Validation: `Zip.__post_init__` (`src/loupe/__init__.py:391`).
