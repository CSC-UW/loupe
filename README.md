## Loupe — Multi‑trace + Multi‑video data viewer

Loupe is a fast, Qt-based application for interactive time‑series review and labeling. It combines a high‑performance windowed renderer for multiple traces with any number of time‑synchronized videos, a global hypnogram overview, and an efficient click‑and‑drag labeling workflow. While its labeling system is well-suited for sleep scoring, Loupe is a general-purpose tool for inspecting any time‑series data at fine-grained detail.

This document explains:
- What the application does
- How to install and use it
- A complete tour of features and shortcuts
- Data format
- xarray integration for Jupyter notebooks

For technical design and implementation details, see [DESIGN.md](DESIGN.md).

---

### Quick start
Requirements:
- Python 3.12+
- pip packages: PySide6, pyqtgraph, opencv‑python, numpy, xarray, zarr

Install:
```bash
pip install loupe
# or with uv:
uv pip install loupe
```

#### Python / Jupyter (xarray)
```python
%gui qt6
import xarray as xr
from loupe import view, TraceConfig

# In-memory DataArray (stacked subplots, one per trace)
ds = xr.open_zarr("data.zarr", group="dmd_2")
da = ds["data"].sel(syn_id=slice(3, 6), time=slice(0, 1800)).load()
w = view(TraceConfig(da))

# Dense view — all traces on a single axis (EEG-style)
w = view(TraceConfig(da, mode="dense", traces_per_page=16,
                    order_by="y", descending=True))

# Set initial time window to 30 seconds
w = view(TraceConfig(da, mode="dense"), window_len=30)

# Mixed layout — list position determines top-to-bottom subplot order
w = view([
    TraceConfig(da1, mode="dense", gain=2.0, traces_per_page=20),
    TraceConfig(da2, mode="stacked-subplots"),
])

# Path-based loader (zarr/netCDF; filter applied before load)
w = view(TraceConfig.from_path(
    "data.zarr", group="dmd_2",
    filter_dict={"syn_id": slice(3, 6), "time": slice(0, 1800)},
))

# With time-synchronized videos — pass one or more VideoConfig items
from loupe import VideoConfig
w = view(TraceConfig(da), videos=[
    VideoConfig("cam1.mp4", "cam1_frame_times.npy", name="side cam"),
    VideoConfig("cam2.mp4", "cam2_frame_times.npy", name="overhead"),
    VideoConfig("thermal.mp4", "thermal_frame_times.npy", name="thermal"),
])
# A single VideoConfig is accepted as shorthand for a one-element list.
```

Every data input to `view()` must be wrapped in a Config:
`TraceConfig` for line traces, `HeatmapConfig` for 2-D heatmaps,
`RasterConfig` for DataFrame rasters, or `Zip` to co-plot traces sharing a
shared coordinate across multiple DataArrays. Bare DataArrays / DataFrames
are not accepted.

---

### Data format

#### Time series (npy)
- Each time series is provided as a pair: `<name>_t.npy` (1‑D float seconds, monotonic) and `<name>_y.npy` (1‑D float values).
- Pairs are matched by basename; row order follows first appearance.
- Optional per‑series colors accept `#RRGGBB[AA]`, `0xRRGGBB`, or `R,G,B[,A]`.

#### xarray DataArrays
- Each DataArray must have a `'time'` dimension with coordinates.
- All other dimension combinations are flattened into individual traces. For example, a DataArray with dims `(channel=2, syn_id=3, time=N)` produces 6 traces named `ch0-syn0`, `ch0-syn1`, etc.
- Supports zarr and netCDF stores via path-based loading with optional dimension filtering.
- Multiple DataArrays can be viewed simultaneously; traces are prefixed with the array name.
- Each DataArray can be displayed in **stacked-subplots** mode (one subplot per trace, the default) or **dense** mode (all traces on a single axis with vertical offsets). See "Dense view" below.

#### Videos
- From Python, pass one or more `VideoConfig` items via the `videos=` kwarg on `view()` (a bare `VideoConfig` is also accepted as shorthand for a one-element list). Each `VideoConfig` takes:
  - `video_path` — path to a file readable by OpenCV (`.mp4`, `.avi`, `.mov`, `.mkv`).
  - `frame_times_path` — path to a 1‑D `.npy` of per-frame timestamps in seconds, used to align frames with the trace cursor.
  - `name` (optional) — display label used for the empty-frame placeholder and the Show / Frame Step Target menu entries. Defaults to `"Video {i+1}"`.
  - `stretch` (optional) — initial vertical layout weight relative to other videos. Defaults to 3 for the first slot and 2 for the rest.
  - `frame_times_correction` (optional, default `0.0`) — float (seconds) added to every frame time after loading. Applied uniformly whether `frame_times_path` is a single file or a list; useful as a quick alignment shim against the trace cursor without rewriting the underlying `.npy` files.
- All loaded videos play together, locked to the trace cursor. Each runs in its own `VideoWorker` thread.

#### Interval labels
Loupe loads and saves interval labels via a small registry of formats and an
`IntervalLabelSchema` that tells it which user‑named columns mean start, end,
duration, label, note, and which extras to display. Rows are half‑open
intervals `[start, end)`.

Supported formats (all read; CSV / HTSV / Parquet also write):

| Extension | Read | Write | Notes |
|---|---|---|---|
| `.csv`     | ✓ | ✓ | Defaults to legacy schema `start_s,end_s,label,note` |
| `.htsv`    | ✓ | ✓ | Header‑bearing TSV; pass an explicit `IntervalLabelSchema` |
| `.parquet` | ✓ | ✓ | Pass an explicit `IntervalLabelSchema` |
| `.txt`     | ✓ | ✗ | Visbrain hypnograms; read‑only (lossy if written) |

Pass interval labels into `view()`:

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

`extra_cols` columns appear as additional cells in the interval-labels summary
table, the Jump‑to‑Epochs dialog, and the Ctrl+Shift+N edit dialog. They
round‑trip on save preserving the user’s original column names.

**Save safety.** File → Export Interval Labels As… always opens a save dialog
and writes a copy. The original file is **never** overwritten unless the caller
explicitly opted in:

```python
view(TraceConfig(da), interval_labels="labels.htsv", interval_label_schema=schema, interval_labels_writeback=True)
```

When `interval_labels_writeback=True`, an extra File → Save Interval Labels
(overwrite source) action becomes available (Ctrl+S). Without it, the menu item
is disabled.

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

#### Raster data
- Raster plots display discrete events as vertical lines in a raster format (e.g., neural spike rasters).
- Each raster subplot requires:
  - `raster_timestamps`: 1‑D array of event times (seconds, same timebase as time series)
  - `raster_yvals`: 1‑D array of row indices (integers 0 to N-1) specifying which row each event belongs to
  - `raster_alphas` (optional): 1‑D array of alpha values (0.0 to 1.0) for each event
  - `raster_colors`: hex color for each subplot (all events in a subplot share the same color)
- Events are rendered as vertical lines centered within their row, with configurable height and thickness.

#### Dense view
The dense view plots many traces (potentially hundreds) on a single pair of axes, like an EEG viewer. Each trace is mean-subtracted, scaled by a gain factor, and offset vertically.

`TraceConfig` parameters that control the dense view:
- `mode="dense"` — enable dense mode.
- `order_by` — coordinate name to control trace ordering and vertical spacing (e.g., `"y"` for electrode depth). If not specified and there is exactly one non-time dimension, its coordinate values are used automatically.
- `descending` — reverse the trace order (default `False`).
- `gain` — amplitude gain multiplier (default `1.0`). Also adjustable at runtime via Alt+scroll or the Dense View Controls dialog (Ctrl+G).
- `step` — show every *n*-th trace (default `1` = all).
- `traces_per_page` — how many traces to show at once (default `None` = all). A vertical scrollbar appears when set. Adjustable at runtime via the Dense View Controls dialog.

`view()`-level: `window_len` — initial time window duration in seconds (default `10.0`). Applies to all display modes.

When multiple DataArrays are loaded, each `TraceConfig` independently chooses dense or stacked-subplots:
```python
from loupe import view, TraceConfig
view([
    TraceConfig(lfp, mode="dense", order_by="y", descending=True, traces_per_page=16),
    TraceConfig(emg, mode="stacked-subplots"),
])
```
Both views share synchronized X (time) axes.

#### Heatmap view
The heatmap view renders an `xr.DataArray` as a 2-D heatmap (imshow-style) over time, with one row per entry of a non-time dimension. It is designed for inspecting many traces at once at fine-grained detail — e.g. `dF[syn_id, time]` shown as a heatmap with synapses on the y-axis and time on the x-axis — while keeping all the synchronized cursor / labeling / video / hypnogram infrastructure of Loupe.

`HeatmapConfig` parameters:
- `split_by` — coordinate or dim name to split into one subplot per unique value (e.g. `'dend-ID'` to get one heatmap per dendrite). Uses `xr.DataArray.groupby`, so works with both dim names and 1-D coords on a dim.
- `order_by` — coordinate name on the row dim controlling y-axis row order (sorted ascending).
- `descending` — reverse the row ordering given by `order_by` (default `False`).
- `cmap` — matplotlib colormap name. A list applies one entry per `split_by` group in order. Default `"magma"`.
- `vmin`, `vmax` — color scale limits. Default is robust 1–99 percentile per heatmap.
- `decim_method` — `"peak"` (max-absolute per bin, preserves transients; default) or `"mean"`.

Each subplot must have exactly one non-time dim remaining after the split — otherwise a clear error is raised.

```python
from loupe import view, HeatmapConfig
# Per-dendrite heatmap, rows ordered by anatomical position:
w = view(HeatmapConfig(dnv, split_by="dend-ID", order_by="pos",
                       cmap=["magma", "viridis", "plasma", "inferno"]))

# Single heatmap (no split):
w = view(HeatmapConfig(dF_one_dend, order_by="pos"))
```

The **Heatmap Plot Controls** dialog (View → Heatmap Plot Controls…, `Ctrl+Shift+H`) provides per-subplot live adjustment of:
- vmin / vmax (slider + spinbox; "Reset to 1–99% percentile" button)
- Colormap (dropdown of presets, freely editable)
- Decimation method (peak / mean)
- "Apply to all heatmaps" copies the current row's settings to every other heatmap plot.

**Performance.** Heatmap plots use a layered strategy to stay responsive even with multiple plots loaded:
1. Cursor moves, selection drags, label additions, and Y-zoom skip the heatmap refresh entirely.
2. Each plot caches its last-rendered `(window, view-width, vmin, vmax, cmap, decim_method)` and short-circuits if unchanged.
3. NaN values are sentinel-replaced at load time so refresh uses fast `np.max` / `np.mean` (no nan-aware overhead).
4. Manual NumPy LUT mapping → uint8 RGBA upload bypasses pyqtgraph's per-pixel level math.
5. Heatmaps exceeding 5 M elements get a power-of-2 mip-map at load time (~2× memory), so pan latency stays O(viewbox-width) regardless of recording length.

---

### UI tour
Left side:
- Multi‑trace panel: stacked subplots (one per trace) and/or dense plots (many traces on one axis), all X‑linked.
- Dense plots include a vertical scrollbar showing position within the full trace set.
- Click‑and‑drag inside any plot creates a selection region across all traces.
- Each plot has a vertical cursor line synchronized across traces.

Right side:
- Videos panel: any number of time‑synchronized videos stacked vertically (pass them via `view(..., videos=[VideoConfig(...), ...])`), plus a per‑window cursor slider underneath the top video.
- When no videos are loaded, a dark placeholder occupies the videos panel area.
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

Import/Export interval labels
- File → Load Interval Labels… reads `.csv`, `.htsv`, `.parquet`, or Visbrain
  `.txt`. For `.htsv`/`.parquet`, pass an explicit `IntervalLabelSchema` via
  the `view()` kwargs (the load dialog cannot guess column names).
- File → Export Interval Labels As… writes `.csv`, `.htsv`, or `.parquet`,
  preserving the user's original column names.
- File → Save Interval Labels (overwrite source) — Ctrl+S — overwrites the
  original file. Available only when `view()` was called with
  `interval_labels_writeback=True`.

---

### Tips and recommended workflow
1. Set window length for your inspection resolution (e.g., 10–30 s).
2. Page `[ ]` or Shift+wheel to find regions of interest.
3. Click‑drag to select an epoch; press a label key. Repeat across the recording.
4. Use `0` to clear labels for re‑labeling specific regions.
5. Use the hypnogram to verify global dynamics; toggle `z` to zoom the overview.
6. Adjust Y scales per trace via Ctrl+D.
7. If reviewing behavior videos, step the selected video frame‑by‑frame with Left/Right. Use the frame step target menu to choose which video to step.
8. Add notes to epochs (Ctrl+Shift+N) to flag unclear or interesting cases for later review.
9. Use Jump to Epochs (Ctrl+J) to quickly navigate to epochs with specific states or notes.
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
- Load interval labels in any supported format by passing `interval_labels=`
  plus a custom `IntervalLabelSchema` (see `loupe/interval_labels.py`). Extra
  columns appear in the interval-labels table and round‑trip on save.
- The primary extension surfaces are:
  - `loupe.IntervalLabelSchema` — describes user column names.
  - `loupe.IntervalLabelSet` — DataFrame‑backed interval-label store with
    `add`, `clear_range`, `merge_adjacent`, `update_cell`, `save_as`,
    `save_to_source`, etc.
  - `loupe.StateConfig` — keymap + label colors.
- Internal modular hot paths:
  - Interval-label management: `IntervalLabelSet.add`/`clear_range`/
    `merge_adjacent` plus the GUI wrappers `_add_new_interval_label`,
    `_clear_interval_labels_in_range`, `_merge_adjacent_same_interval_labels`,
    `_finalize_interval_label_change`.
  - Rendering pipeline: `_apply_x_range`, `_refresh_curves`,
    `_refresh_dense_curves`, `_sync_interval_label_visuals`.
  - Video plumbing: `VideoConfig` (public, `loupe.VideoConfig`), `VideoSlot`
    (internal, one per loaded video), `VideoWorker`, and the slot-loop helpers
    `_on_frame_ready(slot, ...)`, `_rescale_video_frame(slot)`,
    `_request_video_frame(slot, t)`.

---

### License and citation
If you publish results produced with the help of Loupe, please include an appropriate acknowledgment.

For questions or contributions, open an issue in the repository.
