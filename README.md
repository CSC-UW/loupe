## Loupe — Multi‑trace + Multi‑video data viewer

Loupe is a fast, Qt-based application for interactive time‑series review and labeling. It combines a high‑performance windowed renderer for multiple traces with any number of time‑synchronized videos, a global hypnogram overview, and an efficient click‑and‑drag labeling workflow. While its labeling system is well-suited for sleep scoring, Loupe is a general-purpose tool for inspecting any time‑series data at fine-grained detail.

This document is organized around the **plot types** Loupe exposes. Each plot type has its own Config class (`TraceConfig`, `HeatmapConfig`, `RasterConfig`, `Zip`, `VideoConfig`); pass one or more to `view()` to launch the viewer. A shared [Data inputs](#data-inputs) section covers cross-cutting input concerns (xarray, npy pairs, interval labels, state definitions).

For technical design and implementation details, see [DESIGN.md](DESIGN.md).

---

### Quick start

Requirements:
- Python 3.12+
- pip packages: PySide6, pyqtgraph, opencv‑python, numpy, xarray, zarr, polars

Install:
```bash
pip install loupe
# or with uv:
uv pip install loupe
```

```python
%gui qt6
import xarray as xr
from loupe import view, TraceConfig

ds = xr.open_zarr("data.zarr", group="dmd_2")
da = ds["data"].sel(syn_id=slice(3, 6), time=slice(0, 1800)).load()
view(TraceConfig(da))
```

Every data input to `view()` must be wrapped in a Config — bare `xr.DataArray` / `pl.DataFrame` are not accepted. List position determines top-to-bottom subplot order:

```python
view([
    TraceConfig(traces),
    RasterConfig(events, time_col="time", order_by="source_id", split_by="dmd"),
    HeatmapConfig(dff, split_by="dend-ID", order_by="pos"),
])
```

---

### Plot types

| View | Config | Best for |
| --- | --- | --- |
| [Stacked traces](#stacked-traces) | `TraceConfig(da)` | Small-to-moderate number of traces; per-trace Y range. |
| [Dense traces](#dense-traces) | `TraceConfig(da, mode="dense")` | Many traces (EEG, neuropixel LFP) on one axis. |
| [Heatmap](#heatmap) | `HeatmapConfig(da)` | Many traces shown as a 2-D imshow over time. |
| [Raster](#raster) | `RasterConfig(df, time_col=..., order_by=...)` | Discrete events (spikes, behavioral marks). |
| [Zip overlay](#zip-overlay) | `Zip([TraceConfig, ...], on=...)` | Multiple DataArrays co-plotted per shared coord. |
| [Video](#video) | `VideoConfig(path, frame_times_path)` | Time-synchronized video alongside any of the above. |

#### Stacked traces

One subplot per trace, all X-linked. Default mode of `TraceConfig`.

```python
from loupe import view, TraceConfig
view(TraceConfig(da))
```

Key parameters:

| Param | Default | Purpose |
| --- | --- | --- |
| `order_by` | `None` | Coordinate name controlling subplot order. |
| `descending` | `False` | Reverse the order. |
| `hue` / `palette` | `None` | Categorical coloring by a coordinate. |
| `color` | `None` | Single color override for all traces. |
| `array_name` | `False` | Prefix on trace names (see deep-dive doc). |
| `sample_markers` | `None` | List of `SampleMarkers` (spike/event overlays). |

Runtime: `Ctrl+D` for per-trace Y autorange/min/max; `Ctrl+1`/`Ctrl+2` to Y-zoom the hovered plot. Stacked mode supports `sample_markers` for sample-aligned overlays (single-config-per-window constraint).

See [docs/views/stacked-traces.md](docs/views/stacked-traces.md) for the full parameter reference, `from_path` loader, and `SampleMarkers` usage.

#### Dense traces

Many traces (potentially hundreds) on a single pair of axes, EEG-style. Each trace is mean-subtracted, scaled by `gain`, and offset vertically.

```python
view(TraceConfig(da, mode="dense", traces_per_page=16,
                 order_by="y", descending=True))
```

Key parameters:

| Param | Default | Purpose |
| --- | --- | --- |
| `order_by` | `None` | Coordinate controlling order + vertical spacing (e.g. `"y"` for depth). |
| `gain` | `1.0` | Amplitude multiplier (live-adjustable). |
| `step` | `1` | Show every _n_-th trace. |
| `traces_per_page` | `None` | Vertical pagination; shows a scrollbar when set. |
| `hue` / `palette` | `None` | Categorical coloring (e.g. anatomy). |

Runtime: `Alt+wheel` for gain, `Shift+Alt+wheel` for vertical scroll, `Ctrl+G` for the Dense View Controls dialog.

See [docs/views/dense-traces.md](docs/views/dense-traces.md) for the full parameter reference and mixed-layout patterns.

#### Heatmap

Renders an `xr.DataArray` as a 2-D heatmap (imshow-style) with time on the x-axis and one row per entry of a non-time dim — e.g. `dF[syn_id, time]` shown as a heatmap with synapses on the y-axis.

```python
from loupe import view, HeatmapConfig
view(HeatmapConfig(dnv, split_by="dend-ID", order_by="pos",
                   cmap=["magma", "viridis", "plasma", "inferno"]))
```

Key parameters:

| Param | Default | Purpose |
| --- | --- | --- |
| `split_by` | `None` | Coordinate/dim to groupby into separate subplots. |
| `order_by` | `None` | Row order within each heatmap. |
| `cmap` | `"magma"` | Colormap name, `Colormap`, list (per-split), dict, or callable. |
| `vmin` / `vmax` | `None` | Color scale limits (default: 1–99 percentile per heatmap). |
| `decim_method` | `"peak"` | `"peak"` preserves transients; `"mean"` smooths. |

Each subplot must have exactly one non-time dim remaining after the split — otherwise a clear error is raised.

Runtime: `Ctrl+Shift+H` opens the Heatmap Plot Controls dialog for per-subplot `vmin`/`vmax`, colormap, and decimation, with "Apply to all arrays".

Heatmaps include heavy performance optimizations (cache, mip-mapping for >5 M elements, NaN sentinel replacement). See [docs/views/heatmap.md](docs/views/heatmap.md) for the full parameter reference and performance notes.

#### Raster

Discrete events as vertical lines in a raster format — e.g. neural spike rasters, behavioral events. Backed by a polars DataFrame.

```python
import polars as pl
from loupe import view, RasterConfig

ev = pl.read_parquet("spikes.parquet")
view(RasterConfig(ev, time_col="time", order_by="source_id",
                  split_by="dmd", alpha_by="snr_denoised"))
```

Key parameters:

| Param | Default | Purpose |
| --- | --- | --- |
| `time_col` | _required_ | Event timestamp column. |
| `order_by` | _required_ | Column whose unique values index raster rows. |
| `split_by` | `None` | Column(s) to split into separate raster subplots. |
| `alpha_by` | `None` | Column for per-event opacity (normalized to `alpha_range`). |
| `hue` / `palette` | `None` | Per-event coloring. |
| `color` | `None` | Single color override. |

A `RasterConfig.from_parquet(path, time_col=..., order_by=...)` classmethod handles single-file or concatenated-file loading.

Runtime: `Ctrl+Shift+R` toggles Proportional Raster Plots (on by default; sizes raster subplots by row count). Brightness / event height / thickness via the View menu.

See [docs/views/raster.md](docs/views/raster.md) for the full parameter reference and color-precedence rules.

#### Zip overlay

Co-plots traces sharing a coordinate value across multiple DataArrays. Useful for inspecting derived signals together — e.g. F, dF/F, and denoised per synapse.

```python
from loupe import view, TraceConfig, Zip
view(Zip(
    [TraceConfig(F), TraceConfig(dFF), TraceConfig(denoised)],
    on="syn_id",
    colors=["#ff0000", "#00ff00", "#0000ff"],
))
```

Inside a `Zip`, only the `color` field on each wrapped `TraceConfig` applies; other fields must remain at their defaults. Only one `Zip` per window, and a `Zip` cannot coexist with `TraceConfig` or `HeatmapConfig`.

See [docs/views/zip.md](docs/views/zip.md) for the full constraints and parameter reference.

#### Video

Time-synchronized video frames in the right panel, locked to the trace cursor. Pass one or more `VideoConfig` items via the `videos=` kwarg on `view()`:

```python
from loupe import view, TraceConfig, VideoConfig
view(TraceConfig(da), videos=[
    VideoConfig("cam1.mp4",    "cam1_frame_times.npy",    name="side cam"),
    VideoConfig("cam2.mp4",    "cam2_frame_times.npy",    name="overhead"),
])
```

A bare `VideoConfig` is accepted as shorthand for a one-element list. Both `video_path` and `frame_times_path` may also be **lists of equal length**, in which case the files are loaded as one continuous (concatenated) video.

Runtime: `Left` / `Right` step the selected video one frame; `Space` toggles playback; `Ctrl+Shift+1`…`9` toggle the _N_-th video's visibility. View → Frame Step Target chooses which video the arrows control.

See [docs/views/video.md](docs/views/video.md) for the full parameter reference, including `frame_times_correction` and multi-file concat details.

---

### Data inputs

Inputs that aren't tied to a single plot type.

#### xarray DataArrays

Used by `TraceConfig`, `HeatmapConfig`, and `Zip`.

- Each DataArray must have a `'time'` dimension with coordinates.
- For `TraceConfig`, all other dimension combinations are flattened into individual traces. A DataArray with dims `(channel=2, syn_id=3, time=N)` produces 6 traces named `ch0-syn0`, `ch0-syn1`, etc.
- For `HeatmapConfig`, each subplot (after `split_by`) must have exactly one non-time dim remaining.
- Supports zarr and netCDF stores via `TraceConfig.from_path(path, group=..., variable=..., filter_dict=...)`.
- Multiple DataArrays can be viewed simultaneously; trace names are optionally prefixed by `array_name`.

#### Time-series npy pairs

- Each time series is provided as a pair: `<name>_t.npy` (1‑D float seconds, monotonic) and `<name>_y.npy` (1‑D float values).
- Pairs are matched by basename; row order follows first appearance.
- Optional per‑series colors accept `#RRGGBB[AA]`, `0xRRGGBB`, or `R,G,B[,A]`.
- These are loaded via the legacy GUI loader (File → Load Traces).

#### Global event markers

Vertical event-marker lines drawn across every plot pane (trace, dense, heatmap, raster) on top of all other layers — including label shading — for time-locked annotations like stimulus onsets, behavioral markers, or sleep-stage transitions. Pass a `GlobalEventsConfig` via the `global_events=` kwarg on `view()`:

```python
import polars as pl
from loupe import view, TraceConfig, GlobalEventsConfig

events = pl.DataFrame({
    "time": [5.0, 12.5, 20.0, 33.3, 47.1],
    "kind": ["stim", "lick", "stim", "blink", "lick"],
})

# Single style applied to every event
view(TraceConfig(da), global_events=GlobalEventsConfig(events))

# Per-class styling — each unique value of `kind` gets a distinct style
view(
    TraceConfig(da),
    global_events=GlobalEventsConfig(
        events,
        style_events_on="kind",
        style_kwargs={
            "stim":  {"line_color": "#ff8800", "line_width": 2.0},
            "lick":  {"line_color": (100, 200, 255), "line_style": "dashed"},
            "blink": {"line_alpha": 100},                  # keep cycle defaults
        },
    ),
)
```

`GlobalEventsConfig` fields:
- `data` — polars DataFrame with one row per event.
- `event_times_column` (default `"time"`) — column with event times in seconds.
- `style_events_on` (optional) — column whose unique values group events into styled classes. When unset, every event uses a single style.
- `style_kwargs` (optional) — `{class_value: {…style overrides…}}` mapping. Unspecified classes auto-cycle through distinct line styles (solid → dashed → dotted → dashdot → dashdotdot) on light gray, then cycle through a small color palette before any style repeats. Ignored (with a warning) when `style_events_on` is `None`.

Per-class style fields:
- `line_color` — RGB(A) tuple or `"#RRGGBB"` hex string.
- `line_style` — `"solid"`, `"dashed"`, `"dotted"`, `"dashdot"`, or `"dashdotdot"`.
- `line_width` — pen width in pixels (default `1.5`).
- `line_alpha` — `0–255` (default `200`).

Colors, line styles, widths, and alphas can be edited live from the menu bar at **View → Style Global Events…** — the menu entry only appears when `global_events=` was passed. Markers stay on top of label shading by design, so they remain visible even with full-opacity hypnogram regions.

#### Interval labels

Loupe loads and saves interval labels via a small registry of formats and an `IntervalLabelSchema` that tells it which user‑named columns mean start, end, duration, label, note, and which extras to display. Rows are half‑open intervals `[start, end)`.

Supported formats (all read; CSV / HTSV / Parquet also write):

| Extension | Read | Write | Notes |
|---|---|---|---|
| `.csv`     | ✓ | ✓ | Defaults to legacy schema `start_s,end_s,label,note` |
| `.htsv`    | ✓ | ✓ | Header‑bearing TSV; pass an explicit `IntervalLabelSchema` |
| `.parquet` | ✓ | ✓ | Pass an explicit `IntervalLabelSchema` |
| `.txt`     | ✓ | ✗ | Visbrain hypnograms; read‑only (lossy if written) |

Pass labels into `view()`:

```python
import polars as pl
from loupe import view, IntervalLabelSchema, TraceConfig

# Legacy CSV (no schema needed)
view(TraceConfig(da), interval_labels="labels.csv")

# HTSV with custom column names + extras shown in the GUI
schema = IntervalLabelSchema(
    start_col="start_time",
    end_col="end_time",
    duration_col="duration",   # optional; if both end_col and duration_col
                               # are given they must agree on every row
    label_col="state",
    extra_cols=("scorer", "confidence"),
)
view(TraceConfig(da), interval_labels="hypnogram.htsv", interval_label_schema=schema)

# Visbrain .txt (start of each bout = previous bout's end)
view(TraceConfig(da), interval_labels="hypnogram.txt")

# Existing in-memory polars DataFrame
df = pl.read_parquet("labels.parquet")
view(TraceConfig(da), interval_labels=df, interval_label_schema=schema)
```

`extra_cols` columns appear as additional cells in the labels summary table,
the Jump‑to‑Epochs dialog, and the `Ctrl+Shift+N` edit dialog. They round‑trip
on save preserving the user's original column names.

**Save safety.** File → Export Labels As… always opens a save dialog and
writes a copy. The original file is **never** overwritten unless the caller
explicitly opted in:

```python
view(TraceConfig(da), interval_labels="labels.htsv",
     interval_label_schema=schema, interval_labels_writeback=True)
```

When `interval_labels_writeback=True`, an extra File → Save Labels (overwrite source) action becomes available (`Ctrl+S`). Without it, the menu item is disabled.

#### State definitions

State hotkeys and per‑state label colors come from any combination of:

1. an explicit `state_definitions=<path>` kwarg on `view()`,
2. otherwise, a `state_definitions.json` file next to `loupe/app.py`
   (gitignored, user‑local — copy `example_state_definitions.json` to bootstrap),
3. plus any `keymap=` / `label_colors=` kwargs on `view()`, which override
   per‑state on top of the file.

If none of these supplies any definitions, `view()` raises `LoupeConfigError`
— there are no built‑in defaults. The bundled `example_state_definitions.json`
is the authoritative schema reference.

JSON shape:

```json
{
    "keymap":       { "w": "Wake",  "1": "NREM" },
    "label_colors": { "Wake": [0, 209, 40, 60], "NREM": "#291effA0" }
}
```

Multiple hotkeys per state are supported. The keymap can be written either
forward (`{key: state}`) or inverse (`{state: [keys]}`):

```json
{ "keymap": { "Wake": ["w", "W"], "NREM": ["1", "n"] } }
```

…or programmatically:

```python
view(
    TraceConfig(da),
    keymap={"Wake": ["w", "W"], "NREM": ["1", "n"]},
    label_colors={"Wake": "#00d128", "NREM": "#291effA0"},
)
```

Color values may be `[R, G, B]`, `[R, G, B, A]`, or a hex string
(`"#RRGGBB"` / `"#RRGGBBAA"`). Binding the same key to two different states
raises `LoupeConfigError` at load time.

---

### UI tour

Left side:
- Multi‑trace panel: stacked subplots (one per trace) and/or dense plots (many traces on one axis), all X‑linked.
- Dense plots include a vertical scrollbar showing position within the full trace set.
- Click‑and‑drag inside any plot creates a selection region across all traces.
- Each plot has a vertical cursor line synchronized across traces.

Right side:
- Videos panel: any number of time‑synchronized videos stacked vertically, plus a per‑window cursor slider underneath the top video. When no videos are loaded, a dark placeholder occupies the area.
- Hypnogram overview at the bottom: shows full‑recording label spans and a translucent region indicating the current window.

Top:
- Window length (seconds) spinner; global navigator slider for paging through time.

Status bar:
- Displays window start/time span and current cursor time (with label state at cursor).

---

### Keyboard & mouse shortcuts

See [KEYBINDINGS.md](KEYBINDINGS.md) for the complete list of keyboard
shortcuts and mouse/wheel interactions. The Help menu inside the app
("Shortcuts / Help") also prints the active state hotkeys, which are
configurable per project.

Import/Export labels:
- File → Load Labels… reads `.csv`, `.htsv`, `.parquet`, or Visbrain `.txt`.
  For `.htsv`/`.parquet`, pass an explicit `IntervalLabelSchema` via the
  `view()` kwargs (the load dialog cannot guess column names).
- File → Export Labels As… writes `.csv`, `.htsv`, or `.parquet`, preserving
  the user's original column names.
- File → Save Labels (overwrite source) — `Ctrl+S` — overwrites the original
  file. Available only when `view()` was called with `interval_labels_writeback=True`.

---

### Tips and recommended workflow

1. Set window length for your inspection resolution (e.g., 10–30 s).
2. Page `[ ]` or `Shift+wheel` to find regions of interest.
3. Click‑drag to select an epoch; press a label key. Repeat across the recording.
4. Use `0` to clear labels for re‑labeling specific regions.
5. Use the hypnogram to verify global dynamics; toggle `z` to zoom the overview.
6. Adjust Y scales per trace via `Ctrl+D`.
7. If reviewing behavior videos, step the selected video frame‑by‑frame with `Left`/`Right`. Use the frame step target menu to choose which video to step.
8. Add notes to epochs (`Ctrl+Shift+N`) to flag unclear or interesting cases for later review.
9. Use Jump to Epochs (`Ctrl+J`) to quickly navigate to epochs with specific states or notes.
10. Customize state hotkeys and colors either by copying `example_state_definitions.json` to `state_definitions.json` and editing it, or by passing `keymap=` / `label_colors=` / `state_definitions=` to `view()` from a script. Multiple hotkeys per state are supported.

---

### Troubleshooting

- No videos appear:
  - Ensure `opencv-python` is installed and the `video_path` / `frame_times_path` you passed to `VideoConfig` exist.
  - Verify `frame_times.npy` is 1‑D and aligned with the video frames.
- X grid lines missing (low profile mode):
  - The app retains vertical grid lines by keeping a minimal bottom axis per row with hidden tick text. If you manually change plot styles, keep axes alive to preserve grids.
- Labels don't export:
  - Ensure you have created at least one label. Export requires at least one interval.

---

### Extensibility

- Add new label hotkeys or colors by:
  - editing your local `state_definitions.json` (gitignored; copy
    `example_state_definitions.json` to bootstrap), or
  - passing `state_definitions=`, `keymap=`, or `label_colors=` to `view()`
    at runtime.
- Load labels in any supported format by passing `interval_labels=` plus a
  custom `IntervalLabelSchema` (see `src/loupe/interval_labels.py`). Extra
  columns appear in the labels table and round‑trip on save.
- The primary extension surfaces are:
  - `loupe.IntervalLabelSchema` — describes user column names.
  - `loupe.IntervalLabelSet` — DataFrame‑backed label store with `add`, `clear_range`,
    `merge_adjacent`, `update_cell`, `save_as`, `save_to_source`, etc.
  - `loupe.StateConfig` — keymap + label colors.
- Internal modular hot paths:
  - Label management: `IntervalLabelSet.add` / `clear_range` / `merge_adjacent`
    plus the GUI wrappers `_add_new_label`, `_clear_labels_in_range`,
    `_merge_adjacent_same_labels`, `_finalize_label_change`.
  - Rendering pipeline: `_apply_x_range`, `_refresh_curves`,
    `_refresh_dense_curves`, `_sync_label_visuals`.
  - Video plumbing: `VideoConfig` (public, `loupe.VideoConfig`), `VideoSlot`
    (internal, one per loaded video), `VideoWorker`, and the slot-loop helpers
    `_on_frame_ready(slot, ...)`, `_rescale_video_frame(slot)`,
    `_request_video_frame(slot, t)`.
- Per-plot-type deep-dives with full parameter references and dev notes:
  - [docs/views/stacked-traces.md](docs/views/stacked-traces.md)
  - [docs/views/dense-traces.md](docs/views/dense-traces.md)
  - [docs/views/heatmap.md](docs/views/heatmap.md)
  - [docs/views/raster.md](docs/views/raster.md)
  - [docs/views/zip.md](docs/views/zip.md)
  - [docs/views/video.md](docs/views/video.md)

---

### License and citation
If you publish results produced with the help of Loupe, please include an appropriate acknowledgment.

For questions or contributions, open an issue in the repository.
