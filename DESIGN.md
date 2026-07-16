## Technical design

Rendering and decimation
- **Stacked-subplots mode:** Each trace is rendered in its own `pyqtgraph.PlotItem`.
- **Dense mode:** All traces in a group share a single `PlotItem`. Each trace is a `PlotDataItem` with the transform `y_display = (y - mean) * gain + offset`, where the offset comes from coordinate values or integer indices. The Y-range viewport controls which traces are visible; a `QScrollBar` mirrors this range. Mean subtraction is cached at load time; the per-trace transform is applied at refresh time to the windowed raw slice (two NumPy ops) before handing it to pyqtgraph.
- Both modes rely on pyqtgraph's built-in peak-preserving decimation:
  - On every pan/zoom, `_refresh_curves` / `_refresh_dense_curves` slice each series to the visible window via `np.searchsorted` and call `setData` on the underlying `PlotDataItem`.
  - Each `PlotDataItem` is configured with `setDownsampling(auto=True, method="peak")` and `setClipToView(True)`, so pyqtgraph clips the slice to the viewbox and then performs a contiguous reshape-based min/max-per-bin reduction in C. The downsample factor is computed from the viewbox pixel width, so the rendering budget is adaptive to display size.
- A custom `SelectableViewBox` disables the stock pan/zoom behavior and emits:
  - Drag start/update/finish signals (for selection)
  - Wheel events split into three intents:
    - Paging (no modifier)
    - Smooth scrolling (Shift)
    - Cursor scrubbing (Ctrl)
- `HoverablePlotItem` augments plots with hover enter/leave to target Y‑zoom on the active plot.

Labeling model
- Labels live in a `LabelSet` wrapping a polars `DataFrame`. The DataFrame uses
  the user's column names (per `LabelSchema`) plus an internal `__loupe_row_id`
  column for stable cross‑edit identity. The row_id column is stripped on save.
- Adding a label:
  - Overlapping existing intervals are split so the new label overwrites the selected span only.
  - The two halves of a split inherit the original row's note and extras.
  - After insertion, adjacent/overlapping intervals with the same label are merged.
- Clearing (`0`) removes any overlapping parts by splitting and discarding overlaps.
- All label regions are drawn across every trace as translucent `LinearRegionItem`s.
- The hypnogram overview shows the same label spans collapsed to a single row with a translucent "current window" region.
- Notes (and any additional `extra_cols`) are stored as columns on the
  DataFrame, keyed by the row's internal row_id rather than its `(start, end)`
  pair. Endpoint edits don't lose metadata.
- File I/O dispatches by extension through a small reader/writer registry
  (`loupe/labels.py`). Visbrain `.txt` is read‑only because the format would
  silently drop notes and extras on write.
- Save‑to‑source is gated by `labels_writeback=True`; without the opt‑in,
  exports always go through a Save‑As dialog.
- State definitions (hotkeys and colors) come from one or more sources at
  startup (see "State definitions" in the README); there are no built‑in defaults.

Videos and threading
- Each video is handled by a `VideoWorker` in its own `QThread`, with a small LRU frame cache. Frames are requested by nearest frame index to the current cursor time.
- The main window's `_set_cursor_time()` requests frames from any loaded videos; scaling is applied to fit inside their `QLabel`s.
- Frame stepping uses the selected video's frame times to pick the nearest index and move to the previous/next index. This accommodates different frame rates across videos.

Layout and sizing
- Left plot spines (Y axes) are aligned by measuring axis widths and applying the maximum using `setWidth()`.
- Low‑profile X mode keeps vertical grid lines for upper plots while hiding axis labels/ticks so only the bottom plot shows time tick labels. Loupe turns this on automatically when 3 or more total subplots are loaded at launch.
- The videos are grouped in a dedicated right‑panel container with its own vertical layout. Each `VideoSlot` carries its own stretch (default 3 for the first slot, 2 for the rest), reallocated via View → Adjust Secondary Videos Size… without fighting other controls.
- Traces are placed in a `GraphicsLayoutWidget` wrapped in a `QScrollArea` (for stacked-subplot vertical paging). Dense plots add a `QScrollBar` to the right of the plot area for vertical trace navigation.
- Individual subplot heights, visibility, and order are controlled via the Subplot Control Board (Ctrl+H). Three plot types are supported: `"ts"` (stacked subplots), `"dense"`, and `"raster"`. Each has a height factor (default 1.0×) that scales from 0.01× to 20.0×. For very small plots (below 0.2×), axis labels are hidden automatically.
- Subplot order can be customized by dragging rows in the Subplot Control Board. This allows placing dense, raster, and stacked-subplot plots in any order.

View-Config persistence
- Data-input Configs and View-Configs are intentionally separate. `TraceConfig`
  and its peers construct runtime data registries; `ViewConfig` stores only the
  presentation state of those registries.
- `loupe.view_config` owns the versioned JSON domain model, validation, atomic
  I/O, portable plot references, and apply reports. It imports no Qt code, so a
  startup file is parsed before QApplication creation or video-thread startup.
- `loupe.view_config_runtime` is the adapter between the domain model and a
  live `LoupeApp`. It inventories the four plot registries, captures state, and
  applies matches in one batched layout/render refresh.
- Plot matching is semantic. An explicit Config `view_id` plus a generated
  local index is preferred; otherwise matching uses plot kind, displayed name,
  and duplicate occurrence. Saved list positions never directly select a
  current plot. Unmatched records remain unchanged and are returned in a
  `ViewConfigApplyReport`; strict mode rejects plot-inventory mismatches before
  mutating the window.
- The default file is a reusable presentation preset. Session position/window
  geometry and Tuner parameter values are separate opt-in sections. Files do
  not contain source arrays, label rows, video paths, or opaque Qt state.
- Files are written to a temporary sibling, flushed, and atomically replaced.
  `schema_version` gates future migrations; version 1 rejects incompatible
  older/newer schemas rather than guessing.

Raster viewer rendering
- Raster/raster plots display discrete events as vertical line segments.
- Each event is drawn as a vertical line at its timestamp, spanning from `(row + 0.5 - height)` to `(row + 0.5 + height)` where height is the configurable event height.
- Alpha values from the data are multiplied by a brightness factor (default 1.0, adjustable 0.2–3.0) before rendering.
- For performance, events are grouped by quantized alpha levels (11 levels) and rendered as batched line segments using `PlotDataItem` with `connect='pairs'`.
- Only events within the current time window are rendered, using binary search on sorted timestamps.
- Downsampling is applied if too many events are visible (>10,000) to maintain responsiveness.
- Raster plots are X‑linked with time series plots and share the same cursor, selection, and labeling system.
- Proportional sizing mode adjusts row heights based on raster row counts; the raster share boost adjusts the relative space between time series and raster plots (no bounds, allowing extreme customization).
- Individual plot heights can be further customized via the Subplot Control Board, which interacts with raster proportional sizing when enabled.

Performance notes
- OpenGL is enabled in pyqtgraph config when available; antialiasing is off for speed.
- Pyqtgraph's auto downsample factor scales with viewbox pixel width, so the decimation budget is bounded per plot and adapts to display size.
- Long‑duration datasets (hours) remain responsive due to windowed slicing combined with pyqtgraph's peak‑preserving downsampling.
