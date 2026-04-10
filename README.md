## Loupe — Multi‑trace + Multi‑video data viewer

Loupe is a fast, Qt-based application for interactive time‑series review and labeling. It combines a high‑performance windowed renderer for multiple traces with one to three time‑synchronized videos, a global hypnogram overview, and an efficient click‑and‑drag labeling workflow. While its labeling system is well-suited for sleep scoring, Loupe is a general-purpose tool for inspecting any time‑series data at fine-grained detail.

This document explains:
- What the application does
- How to install and use it
- A complete tour of features and shortcuts
- Command‑line flags and data format
- xarray integration for Jupyter notebooks
- Technical design and implementation details

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
w = view(da)

# Dense view — all traces on a single axis (EEG-style)
w = view(da, dense=True, traces_per_page=16, order_by="y", descending=True)

# Set initial time window to 30 seconds
w = view(da, dense=True, window_len=30)

# Mixed mode — per-DataArray display configuration
w = view(data=[
    TraceConfig(da1, mode="dense", gain=2.0, traces_per_page=20),
    TraceConfig(da2, mode="stacked-subplots"),
])

# Path-based (loads and filters automatically)
w = view(path="data.zarr", group="dmd_2",
         filter_dict={"syn_id": slice(3, 6), "time": slice(0, 1800)})
```

#### Command line (npy files)
```bash
# Load all *_t.npy/*_y.npy pairs from a folder
python -m loupe.app --data_dir ./data

# Load explicit pairs and one video
python -m loupe.app \
  --data_files ./data/eeg_t.npy ./data/eeg_y.npy ./data/load_t.npy ./data/load_y.npy \
  --video ./data/video.mp4 --frame_times ./data/frame_times.npy

# Multiple videos
python -m loupe.app \
  --data_dir ./data \
  --video ./data/video.mp4 --frame_times ./data/frame_times.npy \
  --video2 ./data/video2.mp4 --frame_times2 ./data/frame_times2.npy \
  --video3 ./data/video3.mp4 --frame_times3 ./data/frame_times3.npy

# Matrix/raster plots (e.g., neural spike rasters)
python -m loupe.app \
  --data_dir ./data \
  --matrix_timestamps ./data/spikes1_timestamps.npy ./data/spikes2_timestamps.npy \
  --matrix_yvals ./data/spikes1_yvals.npy ./data/spikes2_yvals.npy \
  --alpha_vals ./data/spikes1_alphas.npy ./data/spikes2_alphas.npy \
  --matrix_colors "#FF5500" "#00AAFF"

# xarray data from the command line
python -m loupe.app \
  --xr_path ./data.zarr \
  --xr_group dmd_2 --xr_variable data \
  --xr_filter '{"syn_id": [3, 6], "time": [0, 1800]}'
```

---

### Data format

#### Time series (npy)
- Each time series is provided as a pair: `<name>_t.npy` (1‑D float seconds, monotonic) and `<name>_y.npy` (1‑D float values).
- You can:
  - Point the app to a directory with many pairs using `--data_dir`, or
  - Provide an ordered list of files using `--data_files` (any mix of `_t.npy` and `_y.npy` files). Pairs are matched by basename; row order follows first appearance in your list.
- Optional per‑series colors can be provided with `--colors`. Accepted formats: `#RRGGBB[AA]`, `0xRRGGBB`, or `R,G,B[,A]`.

#### xarray DataArrays
- Each DataArray must have a `'time'` dimension with coordinates.
- All other dimension combinations are flattened into individual traces. For example, a DataArray with dims `(channel=2, syn_id=3, time=N)` produces 6 traces named `ch0-syn0`, `ch0-syn1`, etc.
- Supports zarr and netCDF stores via path-based loading with optional dimension filtering.
- Multiple DataArrays can be viewed simultaneously; traces are prefixed with the array name.
- Each DataArray can be displayed in **stacked-subplots** mode (one subplot per trace, the default) or **dense** mode (all traces on a single axis with vertical offsets). See "Dense view" below.

#### Videos
- Provide `--video/--frame_times` for the first video, `--video2/--frame_times2` for the second, and `--video3/--frame_times3` for the third.
- Frame times are 1‑D numpy arrays of timestamps (seconds) matching the video's frames.
- A static image (`--image`) can be shown when only one video is present or for custom use.

#### Labels
- CSV import/export uses the header `start_s,end_s,label,note` with rows specifying half‑open intervals `[start_s, end_s)`.
- The `note` column is optional; old CSV files without notes are still supported.
- Notes can be added to any labeled epoch via Ctrl+Shift+N.

#### State definitions
- State hotkeys and label colors are loaded from `state_definitions.json` in the same directory as the main script.
- The file contains a `keymap` object (key → state name) and a `label_colors` object (state name → [R, G, B, A]).
- If the file is missing or invalid, built-in defaults are used.
- To customize states, edit `state_definitions.json` and restart the app.

#### Matrix/Raster data
- Matrix plots display discrete events as vertical lines in a raster format (e.g., neural spike rasters).
- Each matrix subplot requires:
  - `matrix_timestamps`: 1‑D array of event times (seconds, same timebase as time series)
  - `matrix_yvals`: 1‑D array of row indices (integers 0 to N-1) specifying which row each event belongs to
  - `alpha_vals` (optional): 1‑D array of alpha values (0.0 to 1.0) for each event
  - `matrix_colors`: hex color for each subplot (all events in a subplot share the same color)
- Events are rendered as vertical lines centered within their row, with configurable height and thickness.

#### Dense view
The dense view plots many traces (potentially hundreds) on a single pair of axes, like an EEG viewer. Each trace is mean-subtracted, scaled by a gain factor, and offset vertically.

Parameters (pass to `view()` or wrap in `TraceConfig`):
- `dense=True` — enable dense mode for all DataArrays (convenience shorthand).
- `mode="dense"` — enable dense mode via `TraceConfig`.
- `order_by` — coordinate name to control trace ordering and vertical spacing (e.g., `"y"` for electrode depth). If not specified and there is exactly one non-time dimension, its coordinate values are used automatically.
- `descending` — reverse the trace order (default `False`).
- `gain` — amplitude gain multiplier (default `1.0`). Also adjustable at runtime via Alt+scroll or the Dense View Controls dialog (Ctrl+G).
- `step` — show every *n*-th trace (default `1` = all).
- `traces_per_page` — how many traces to show at once (default `None` = all). A vertical scrollbar appears when set. Adjustable at runtime via the Dense View Controls dialog.
- `window_len` — initial time window duration in seconds (default `10.0`). Applies to all display modes, not just dense.

When multiple DataArrays are loaded, each can independently be dense or stacked-subplots by wrapping in `TraceConfig`:
```python
from loupe import view, TraceConfig
view(data=[
    TraceConfig(lfp, mode="dense", order_by="y", descending=True, traces_per_page=16),
    TraceConfig(emg, mode="stacked-subplots"),
])
```
Both views share synchronized X (time) axes.

---

### Command‑line flags

Time series:
- `--data_dir PATH` — load all `<name>_t.npy` / `<name>_y.npy` pairs from a directory.
- `--data_files FILE...` — explicit ordered list of `.npy` files for multiple series.
- `--colors COLOR...` — optional colors matching series order (see format above).

Video:
- `--video, --frame_times` — main video and frame times.
- `--video2, --frame_times2` — second video and frame times.
- `--video3, --frame_times3` — third video and frame times.
- `--image` — static image (shown when a 2nd/3rd video is not used).

Display:
- `--fixed_scale` — disable Y auto‑scaling; initial per‑trace Y limits are set from robust percentiles (1–99%) with padding.
- `--low_profile_x` — hide X axis labels/ticks for all but the bottom trace; vertical grid lines are preserved on hidden axes. This is now the default automatically whenever Loupe launches with 3 or more total subplots.

Matrix viewer:
- `--matrix_timestamps FILE...` — list of .npy files with event timestamps for each matrix subplot.
- `--matrix_yvals FILE...` — list of .npy files with row indices (0 to N-1) for each event.
- `--alpha_vals FILE...` — optional list of .npy files with alpha values (0-1) for each event.
- `--matrix_colors COLOR...` — list of hex colors (#RRGGBB) for each matrix subplot.

xarray:
- `--xr_path FILE...` — path(s) to zarr or netCDF stores.
- `--xr_group GROUP...` — group(s) within the store(s).
- `--xr_variable NAME` — variable name in the dataset (default: `data`).
- `--xr_filter JSON` — JSON filter dict for dimension slicing (e.g. `'{"syn_id": [3, 6]}'`).

---

### UI tour
Left side:
- Multi‑trace panel: stacked subplots (one per trace) and/or dense plots (many traces on one axis), all X‑linked.
- Dense plots include a vertical scrollbar showing position within the full trace set.
- Click‑and‑drag inside any plot creates a selection region across all traces.
- Each plot has a vertical cursor line synchronized across traces.

Right side:
- Videos panel: up to three time‑synchronized videos stacked vertically, plus a per‑window cursor slider underneath the top video.
- An optional static image can be shown if fewer than three videos are loaded.
- Hypnogram overview at the bottom: shows full‑recording label spans and a translucent region indicating the current window.

Top:
- Window length (seconds) spinner; global navigator slider for paging through time.

Status bar:
- Displays window start/time span and current cursor time (with label state at cursor).

---

### Keyboard & mouse cheatsheet

Navigation and windowing
- Mouse wheel: page left/right one full window.
- Shift + wheel: smooth scroll window (fraction of window length; configurable).
- Ctrl + wheel: cursor scrub within the current window (like dragging the cursor slider).
- `[` `]` or PageUp/PageDown: page window left/right.
- Window spinner: change window length; the app keeps the cursor anchored proportionally.

Dense view controls
- Alt + wheel: adjust trace gain (amplitude scaling) up/down.
- Shift + Alt + wheel: smooth vertical scroll through traces (~3 traces per notch).
- Vertical scrollbar (right edge): drag to scroll through traces; reflects current position.
- Ctrl+G: open Dense View Controls dialog (gain slider, step, traces per page).

Playback and frame stepping
- Space: toggle playback (loops within current window).
- View → Set Playback Speed…: choose 0.25× to 4× (default 1×).
- View → Frame Step Target → Video 1/2/3: pick which video to step.
- Left/Right arrow: step the selected video one frame back/forward (holding repeats).

Selection & labeling
- Click‑drag in any plot: create/update selection. Drag handles to extend or refine.
- While a selection is active, press a label key:
  - `w` Wake
  - `q` Quiet‑Wake
  - `b` Brief‑Arousal
  - `2` NREM‑light
  - `1` NREM
  - `r` REM
  - `t` Transition‑to‑REM
  - `a` Artifact
  - `u` unclear
  - `o` ON
  - `f` OFF
  - `s` spindle
- `0`: Clear any labels in the selected range (splits existing intervals as needed).
- Backspace (Edit → Delete last label): removes the most recently ending label.
- Labels that overlap or are directly adjacent and have the same state are merged automatically into a single epoch.

Epoch notes & navigation
- Ctrl+Shift+N: Add or edit a note for the epoch at cursor (or most recently labeled epoch if cursor is unlabeled).
- Ctrl+J: Open "Jump to Epochs" dialog to view all epochs in a table. Double-click to navigate.
  - Filter by state (dropdown) or by text in notes (search box).
  - Double-clicking an epoch centers the window on that epoch.

Zoom & axes
- Ctrl + 1 / Ctrl + 2: zoom Y‑axis in/out on the hovered plot.
- View → Y‑Axis Controls… (Ctrl+D): per‑trace autorange toggle and min/max input.
- `z`: toggle hypnogram zoom (zoom to window ± padding vs. full extent).
- `h`: toggle hypnogram visibility (frees vertical space for videos).

Subplot management
- Ctrl+H: open Subplot Control Board (height, visibility, order for all subplots — stacked, dense, and matrix).

Video controls
- View → Adjust Secondary Videos Size…: slider to reduce/enlarge Video 1's share so Video 2/3 gain space (live preview).
- View → Show Video 1/2/3 (checkable) or:
  - Ctrl+Shift+1 / Ctrl+Shift+2 / Ctrl+Shift+3 to toggle each video.
- Videos auto‑scale to their label sizes; resizing the splitter re‑scales the frames.

Matrix viewer controls
- View → Proportional Matrix Plots (Ctrl+Shift+M): toggle proportional sizing of matrix plots based on their row count. When enabled, a plot with 20 rows will be twice as tall as one with 10 rows.
- View → Increase Matrix Share (Ctrl+Shift+,): increase the vertical space allocated to matrix plots by ~5%. No upper bound—you can keep increasing as needed.
- View → Decrease Matrix Share (Ctrl+Shift+.): decrease the vertical space allocated to matrix plots by ~5%. No lower bound—you can keep decreasing as needed.
- View → Adjust Matrix Brightness…: slider to adjust the brightness/visibility of matrix event lines (0.2–3.0). Default is 1.0; higher values make events more visible.
- View → Matrix Event Height…: adjust the vertical extent of event lines (0.1–0.5, distance from row center). Default is 0.4 (lines span 80% of row height).
- View → Matrix Event Thickness…: adjust the pen width of event lines in pixels (1–10). Default is 2.
- Matrix plots show only min/max Y tick labels and have no horizontal grid lines for a clean raster appearance.

Subplot Control Board
- View → Subplot Control Board… (Ctrl+H): opens a comprehensive dialog to control all subplots (time series and matrix).
  - **Height sliders**: Adjust individual plot heights from 0.01× to 20.0× the default. When one plot is made taller, the others proportionally shrink. For very small plots (below 0.2×), axis labels are automatically hidden to save space.
  - **Hide checkbox**: Check "Hide" to hide a subplot entirely from the view. Hidden subplots disappear completely and remaining plots expand to fill the space.
  - **Drag to reorder**: Drag subplot rows up/down to change their display order. Matrix plots can be moved above time series plots, and vice versa.
  - **Reset Heights**: Restore all height factors to 1.0×.
  - **Show All**: Unhide all subplots.
  - **Reset Order**: Restore the default order (all time series first, then all matrix plots).

Import/Export labels
- File → Load Labels… reads CSV with header `start_s,end_s,label`.
- File → Export Labels… writes the same format (values formatted to 6 decimals).

---

### Tips and recommended workflow
1. Set window length for your inspection resolution (e.g., 10–30 s).
2. Page `[ ]` or Shift+wheel to find regions of interest.
3. Click‑drag to select an epoch; press a label key. Repeat across the recording.
4. Use `0` to clear labels for re‑labeling specific regions.
5. Use the hypnogram to verify global dynamics; toggle `z` to zoom the overview.
6. Adjust Y scales per trace via Ctrl+D (or use `--fixed_scale` at launch).
7. If reviewing behavior videos, step the selected video frame‑by‑frame with Left/Right. Use the frame step target menu to choose which video to step.
8. Add notes to epochs (Ctrl+Shift+N) to flag unclear or interesting cases for later review.
9. Use Jump to Epochs (Ctrl+J) to quickly navigate to epochs with specific states or notes.
10. Customize state hotkeys and colors by editing `state_definitions.json` in the loupe package directory.

---

### Technical design

Rendering and decimation
- **Stacked-subplots mode:** Each trace is rendered in its own `pyqtgraph.PlotItem`.
- **Dense mode:** All traces in a group share a single `PlotItem`. Each trace is a `PlotCurveItem` with the transform `y_display = (y - mean) * gain + offset`, where the offset comes from coordinate values or integer indices. The Y-range viewport controls which traces are visible; a `QScrollBar` mirrors this range. Mean subtraction is cached at load time; gain is applied at display time on decimated data, making gain changes cheap.
- Both modes use a custom windowed decimator:
  - `segment_for_window()` returns either raw samples (if under a threshold) or a peak‑preserving min/max per time bin (interleaved at bin centers) to preserve spikes/peaks.
  - Rendering budget per plot is adaptive to pixel width to ensure interactivity.
- A custom `SelectableViewBox` disables the stock pan/zoom behavior and emits:
  - Drag start/update/finish signals (for selection)
  - Wheel events split into three intents:
    - Paging (no modifier)
    - Smooth scrolling (Shift)
    - Cursor scrubbing (Ctrl)
- `HoverablePlotItem` augments plots with hover enter/leave to target Y‑zoom on the active plot.

Labeling model
- Labels are stored as a sorted list of dicts `{start, end, label}` (seconds).
- Adding a label:
  - Overlapping existing intervals are split so the new label overwrites the selected span only.
  - After insertion, adjacent/overlapping intervals with the same label are merged.
- Clearing (`0`) removes any overlapping parts by splitting and discarding overlaps.
- All label regions are drawn across every trace as translucent `LinearRegionItem`s.
- The hypnogram overview shows the same label spans collapsed to a single row with a translucent "current window" region.
- Notes are stored in a separate dict keyed by (start, end) tuple and exported/imported with labels.
- State definitions (hotkeys and colors) are loaded from `state_definitions.json` at startup.

Videos and threading
- Each video is handled by a `VideoWorker` in its own `QThread`, with a small LRU frame cache. Frames are requested by nearest frame index to the current cursor time.
- The main window's `_set_cursor_time()` requests frames from any loaded videos; scaling is applied to fit inside their `QLabel`s.
- Frame stepping uses the selected video's frame times to pick the nearest index and move to the previous/next index. This accommodates different frame rates across videos.

Layout and sizing
- Left plot spines (Y axes) are aligned by measuring axis widths and applying the maximum using `setWidth()`.
- `--low_profile_x` keeps vertical grid lines for upper plots while hiding axis labels/ticks so only the bottom plot shows time tick labels. If you do not pass the flag, Loupe now turns this on automatically when 3 or more total subplots are loaded at launch.
- The videos are grouped in a dedicated right‑panel container with its own vertical layout. Stretches are applied only to video rows so you can reallocate space between Video 1 vs Videos 2/3 without fighting other controls.
- Traces are placed in a `GraphicsLayoutWidget` wrapped in a `QScrollArea` (for stacked-subplot vertical paging). Dense plots add a `QScrollBar` to the right of the plot area for vertical trace navigation.
- Individual subplot heights, visibility, and order are controlled via the Subplot Control Board (Ctrl+H). Three plot types are supported: `"ts"` (stacked subplots), `"dense"`, and `"matrix"`. Each has a height factor (default 1.0×) that scales from 0.01× to 20.0×. For very small plots (below 0.2×), axis labels are hidden automatically.
- Subplot order can be customized by dragging rows in the Subplot Control Board. This allows placing dense, matrix, and stacked-subplot plots in any order.

Matrix viewer rendering
- Matrix/raster plots display discrete events as vertical line segments.
- Each event is drawn as a vertical line at its timestamp, spanning from `(row + 0.5 - height)` to `(row + 0.5 + height)` where height is the configurable event height.
- Alpha values from the data are multiplied by a brightness factor (default 1.0, adjustable 0.2–3.0) before rendering.
- For performance, events are grouped by quantized alpha levels (11 levels) and rendered as batched line segments using `PlotDataItem` with `connect='pairs'`.
- Only events within the current time window are rendered, using binary search on sorted timestamps.
- Downsampling is applied if too many events are visible (>10,000) to maintain responsiveness.
- Matrix plots are X‑linked with time series plots and share the same cursor, selection, and labeling system.
- Proportional sizing mode adjusts row heights based on matrix row counts; the matrix share boost adjusts the relative space between time series and matrix plots (no bounds, allowing extreme customization).
- Individual plot heights can be further customized via the Subplot Control Board, which interacts with matrix proportional sizing when enabled.

Performance notes
- OpenGL is enabled in pyqtgraph config when available; antialiasing is off for speed.
- The decimation budget is bounded per plot and scales with plot pixel width.
- Long‑duration datasets (hours) remain responsive due to windowed rendering.

---

### Troubleshooting
- No videos appear:
  - Ensure `opencv-python` is installed and the paths to `--video` and `--frame_times` exist.
  - Verify `frame_times.npy` is 1‑D and aligned with the video frames.
- X grid lines missing (low profile mode):
  - The app retains vertical grid lines by keeping a minimal bottom axis per row with hidden tick text. If you manually change plot styles, keep axes alive to preserve grids.
- Labels don't export:
  - Ensure you have created at least one label. Export requires at least one interval.

---

### Extensibility
- Add new label keys or colors by editing `state_definitions.json`.
- The labeling and rendering code paths are modular:
  - Label management: `_add_new_label`, `_clear_labels_in_range`, `_merge_adjacent_same_labels`, `_redraw_all_labels`, `_redraw_hypnogram_labels`.
  - Rendering pipeline: `_apply_x_range`, `_refresh_curves`, `_refresh_dense_curves`, `segment_for_window`.
  - Video plumbing: `VideoWorker`, `_on_frame_ready`/`_on_frame2_ready`/`_on_frame3_ready`.

---

### License and citation
If you publish results produced with the help of Loupe, please include an appropriate acknowledgment.

For questions or contributions, open an issue in the repository.
