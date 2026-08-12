# VarView — loupe's kernel-resident live variable viewer

*Written 2026-08-12 as both user documentation and an agent handoff doc: goal,
architecture, design decisions, learnings, and gotchas are all here so a new
agent (or future me) can pick this up without re-deriving anything.*

---

## 1. Goal

A **persistent variable viewer beside the notebook**: a window that lists every
variable in the running kernel, live-updates after each cell, and gives a rich
view of whatever is selected — a real table for polars DataFrames, the text
repr for xarray objects, stats for arrays, a gallery of every plot the kernel
has displayed — plus an **"Open in loupe…"** button that pops a small
config-builder GUI (pick trace/heatmap/raster view, set `order_by` /
`split_by` / `hue` / colormap / etc. from dropdowns populated by the object
itself) and launches the full Loupe viewer on the object.

The key architectural requirement: the user's data is huge (multi-GB xarrays,
100k+-row frames on a 512 GB workstation), so the viewer must read objects
**directly from the kernel's memory, zero-copy** — no serialization, no
separate process. That decision drives everything below.

## 2. Architecture: a Qt window living inside the kernel

Three architectures were evaluated before building (see §7 for the rejected
ones). The chosen one:

```
┌────────────────────────────── kernel process ──────────────────────────────┐
│  IPython shell                                                             │
│   ├── %gui qt6  → ipykernel integrates the Qt event loop (runs when idle)  │
│   ├── post_run_cell event ──────────► VarViewWindow.refresh()              │
│   ├── display_pub.publish (wrapped) ─► PlotsGallery.add_image()            │
│   └── user_ns ◄───────────────────── read directly, zero-copy              │
│                                                                            │
│  VarViewWindow (QMainWindow, singleton in loupe.varview._WINDOW)           │
│   ├── left: filter box + VarTableModel/QTableView + "all objects" toggle   │
│   └── right: QTabWidget                                                    │
│        ├── Inspector: VarDetailPanel (header + per-type body)              │
│        │    ├── polars → embedded loupe.loupeDF.DataFrameViewer            │
│        │    ├── xarray → monospace text repr                               │
│        │    ├── ndarray → shape/dtype/stats + corner preview               │
│        │    ├── mpl Figure → rendered PNG in scroll area                   │
│        │    └── other → bounded safe repr                                  │
│        └── Plots: PlotsGallery (thumbnail grid + fit-to-window display)    │
│                                                                            │
│  "Open in loupe…" → LoupeLaunchDialog (varview_launcher.py, modeless)      │
│        builds TraceConfig/HeatmapConfig/RasterConfig → loupe.view(cfg)     │
└────────────────────────────────────────────────────────────────────────────┘
```

**Files**

| File | Contents |
|---|---|
| `src/loupe/varview.py` | scanning (`scan_namespace`, `VarInfo`), `VarTableModel`, detail widgets, `PlotsGallery`, `DisplayPubTee`, `VarViewWindow`, `varview()` entry point |
| `src/loupe/varview_launcher.py` | `LoupeLaunchDialog` (config-builder forms), `open_launcher()`, DataArray preparation helpers |
| `tests/test_varview.py` | 23 tests, offscreen (`QT_QPA_PLATFORM=offscreen`), incl. an end-to-end dialog→`view()`→`LoupeApp` test |
| `~/.ipython/profile_default/startup/00-display-formatters.py` | defines a global `varview()` stub in **every** kernel: enables `%gui qt6` if needed, then imports and calls `loupe.varview.varview()` (lazy import — kernel startup stays fast) |

## 3. Usage

In any kernel (Zed REPL, Jupyter, terminal IPython) on this machine, no
imports needed — the IPython startup file provides the stub:

```python
varview()
```

Or explicitly:

```python
%gui qt6
from loupe.varview import varview
varview()
```

- The window refreshes automatically after every cell; F5 / Refresh rescans
  manually. The filter box narrows by name; "all objects" includes
  modules/functions/classes.
- Selecting a variable builds its inspector. Selection and window geometry
  survive refreshes (geometry persists across sessions via `QSettings
  ("loupe", "varview")`).
- **Double-click** a launchable variable (DataFrame / DataArray / Dataset /
  ndarray) — or click **Open in loupe…** — to get the launcher dialog.
- The **Plots** tab accumulates every `image/png`/`image/jpeg` the kernel
  displays (matplotlib inline figures included), newest selected, capped at 80.
- Outside IPython: `varview(ns=globals())` works with manual refresh only.

## 4. Design decisions and why

- **Zero-copy access**: the window holds only a `ns_getter` closure returning
  `ip.user_ns`. Nothing is copied; the inspector reads the live object. This
  is why in-kernel beats an external-process viewer for this project.
- **Scan must be cheap and safe** — it runs after *every* cell:
  - type detection via `sys.modules.get("polars")` etc., so scanning never
    imports anything the user hasn't;
  - **never call `repr()` on unknown objects during a scan** (arbitrary reprs
    can be slow or raise); bounded reprs only in the inspector, guarded;
  - measured: 253 vars incl. ~1.3 GB of data → **0.31 ms** per refresh.
- **Inspector rebuild policy**: each `VarInfo` has a `token`
  (kind|type|info|size). On refresh the inspector rebuilds only if the
  selected variable's token changed or it vanished. In-place mutations that
  don't change shape/size are *not* detected — use the ⟳ button on the
  inspector header. (Deliberate: cheap and predictable beats deep hashing.)
- **Plot capture point**: wrapping `ip.display_pub.publish` (a
  `DisplayPubTee` instance replaces the bound method) catches *everything*
  that goes through display — inline matplotlib, `display(...)` of images —
  and cannot break user code (whole capture path inside `try/except`, then
  the original publish always runs). Payloads can be `bytes` or base64 `str`
  (the on-the-wire form); both are handled.
- **Embedding `DataFrameViewer`**: a `QMainWindow` embeds fine as a child
  widget after `setWindowFlags(Qt.Widget)` — we get loupeDF's virtualized
  table, column-summary dock, and toolbar for free. No new table code.
- **Modeless everywhere**: the launcher dialog uses `show()`, never
  `exec()`, and message boxes are non-blocking instances. A nested
  `exec()` inside a kernel-integrated Qt loop is a re-entrancy trap (cells
  could execute into the nested loop).
- **Keep-alive registries**: Qt widgets referenced only from Python must be
  held somewhere or they get garbage-collected mid-display. Module-level
  lists `_OPEN_DIALOGS` / `_LAUNCHED_VIEWS` (launcher) mirror the
  `_OPEN_WINDOWS` pattern in `loupeDF.py`; entries are pruned via the
  `destroyed` signal.
- **The IPython hooks detach on close** (`closeEvent` → unregister
  `post_run_cell`, restore `display_pub.publish` only if it is still our
  tee). The `post_run_cell` callback swallows all exceptions — a raising
  callback would print an error after every cell forever.
- **Launcher guard**: stacked-subplots views over 200 traces prompt for
  confirmation (dense/heatmap is almost always what's wanted there).

## 5. Loupe API facts the launcher depends on (hard-won, verify if loupe changes)

- `loupe.view()` requires Configs (`TraceConfig` / `HeatmapConfig` /
  `RasterConfig` / `Zip`); bare arrays are rejected. With an existing
  `QApplication` (the kernel case) it **returns** the `LoupeApp` without
  blocking; in a bare script it calls `sys.exit(app.exec())`.
- The xarray converters (`loupe/xr_loader.py`) require a dim **literally
  named `time`** *and* a `time` coordinate, **and a coordinate on every
  non-time dim** (`da.coords[d]` is accessed unconditionally). The launcher's
  `_prepare_dataarray()` therefore renames the chosen time dim and
  `assign_coords`-fills any missing coords with integer indices
  (or `arange/fs` for time). ndarrays get wrapped as `(row, time)`
  DataArrays with a sample-rate spinbox.
- `order_by` / `hue` must name **coords** (1-D, on a non-time dim);
  `HeatmapConfig.split_by` accepts coords *or* dims. The dialog's dropdowns
  are populated accordingly (`_orderable_coords`, `_splittable_names`).
- `RasterConfig` requires `time_col` and `order_by` (columns); `split_by`,
  `alpha_by`, `hue` optional. The dialog defaults `time_col` to the first
  numeric column named like time (`time`/`t`/`t_sec`/`peak_time`).

## 6. Environment facts / gotchas

- **Qt binding**: this venv has *both* PyQt5 (WISynaptic dep) and PySide6
  (loupe). IPython's `%gui qt`/`%gui qt6` binds **PySide6** — good, matches
  loupe. But if something imports PyQt5 first and claims the binding (e.g.
  WISynaptic's napari whisker labeler forces `QT_API=pyqt5`), `%gui qt6`
  errors with "already imported an Incompatible QT Binding". Don't mix the
  whisker labeler and varview in one kernel.
- `%gui qt6` does **not** change the matplotlib backend — inline plots keep
  rendering into Zed *and* get captured by the gallery tee. Both, no conflict.
- **Responsiveness**: the window runs on the kernel's Qt loop, which only
  pumps while the kernel is idle. During a long-running cell the window
  freezes (like every `%gui qt6` window, e.g. loupe itself). Not a bug.
- **Kernel restart** kills the window (it lives in the kernel process). The
  user accepted this; re-run `varview()`. Geometry is restored.
- **Offscreen testing**: everything tests headless with
  `QT_QPA_PLATFORM=offscreen`, *except* actual pyqtgraph painting — loupe
  sets `useOpenGL=True` and QOpenGLWidget doesn't work offscreen, so a
  launched LoupeApp paints blank in offscreen screenshots (stderr fills with
  `QPainter` noise). Verify launched windows **structurally** instead
  (`len(lapp.heatmap_plots)` etc.). On the real desktop OpenGL is fine.
- `deleteLater()` is deferred — when swapping inspector bodies, call
  `setParent(None)` first or `findChild` (and tests) will still find the
  old widget.
- Programmatic `setCurrentIndex` on a selection model **emits signals**;
  refresh-time reselection wraps it in `QSignalBlocker` to avoid rebuilding
  the inspector, while the public `select_variable()` deliberately doesn't.

## 7. Architectures considered and rejected (context for future work)

1. **External process attaching to the kernel over the Jupyter protocol** —
   verified possible (Zed writes standard connection files to
   `/run/user/<uid>/jupyter/kernel-zed-*.json`; a `jupyter_client` handshake
   from outside works). Rejected for v1: introspection requires executing
   code in the kernel (only possible when idle — so no responsiveness win),
   and every payload must serialize over ZMQ, which kills the zero-copy
   requirement for multi-GB arrays. It *is* the right base if
   kernel-restart-survival ever becomes a requirement; the UI here could be
   reused with a different `ns_getter`-like backend.
2. **In-kernel web server + browser tab** — same zero-copy benefits, but a
   whole HTML table/plot UI would have to be built from scratch, while Qt
   gets `DataFrameViewer` and loupe integration for free.
3. **Existing tools** (Spyder's variable explorer via spyder-kernels swap,
   jupyterlab-variableinspector, VS Code, dtale) — none combine polars +
   xarray + Zed + "open in my own viewer".

## 8. Testing

- `uv run pytest tests/test_varview.py` in the loupe repo (23 tests, ~1 s):
  scanning/classification (incl. hostile objects), window behavior
  (selection, refresh tracking, inspector rebuild-on-change/clear-on-delete),
  gallery + tee (bytes and base64, forwarding, hostile payloads), launcher
  forms → exact Config field assertions for all four view types, ndarray
  wrapping/time-axis/fs, time-dim renaming, and one end-to-end
  dialog→`view()`→`LoupeApp`.
- End-to-end kernel verification (how it was originally validated): start
  the `slap_mi_2_sleep` kernelspec via `jupyter_client` with
  `QT_QPA_PLATFORM=offscreen`, call the startup-stub `varview()`, run cells,
  assert model rows / gallery count / embedded viewer, `w.grab().save(...)`
  for screenshots.
- Full loupe suite: 229 passed after integration (no regressions).

## 9. Future-work ideas (none blocking)

- Context-menu on a variable row (copy name, `del` from namespace, open in
  launcher).
- A `Zip` builder in the launcher (co-plot N arrays `on` a shared dim).
- Optional deep-refresh toggle that also catches in-place mutations
  (content hashing behind a debounce).
- Reuse the same UI with a `jupyter_client` backend (architecture #1) if a
  viewer that survives kernel restarts is ever wanted.
- Gallery: persist captured plots to disk per session.
