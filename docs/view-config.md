# View-Config files

A View-Config is a versioned, data-free snapshot of a live Loupe window's
presentation. It solves two related workflows:

- Apply one carefully tuned visual layout to an analogous recording.
- Reopen the same recording later with the same view.

It is deliberately separate from `TraceConfig`, `HeatmapConfig`,
`RasterConfig`, `Zip`, and `VideoConfig`. Those objects supply data and create a
window. A View-Config describes how the already-created plots should look.

## GUI workflow

1. Load data and adjust the window.
2. Choose **File → Save View-Config As…**.
3. Leave the optional boxes clear for a reusable presentation preset.
4. Choose **File → Load View-Config…** in another compatible Loupe window.

The load dialog applies compatible records. If the current and saved windows
differ, Loupe shows a report listing fallback matches, saved plots it could not
match, current plots it left unchanged, and settings it skipped.

The save dialog has two opt-in sections:

- **Session:** current time, cursor, vertical scroll, window geometry, and
  fullscreen/maximized state. Use this to resume the same recording.
- **Tuner:** current values of named `Param` controls. Values are matched by
  parameter name, type, and duplicate occurrence.

## Python API

```python
from loupe import TraceConfig, HeatmapConfig, ViewConfig, view

w = view([
    TraceConfig(eeg, view_id="eeg"),
    HeatmapConfig(spec, view_id="spectrogram"),
])

config = w.capture_view_config()
w.save_view_config("subject.loupe-view.json")

# Accepts a path, ViewConfig instance, or decoded mapping.
report = w.apply_view_config("subject.loupe-view.json")
print(report.summary())
```

Load during construction:

```python
w = view(
    [
        TraceConfig(next_eeg, view_id="eeg"),
        HeatmapConfig(next_spec, view_id="spectrogram"),
    ],
    view_config="subject.loupe-view.json",
)
```

The complete methods are:

```python
w.capture_view_config(include_session=False, include_tuner=False) -> ViewConfig
w.save_view_config(path, include_session=False, include_tuner=False) -> Path
w.apply_view_config(path_or_config_or_mapping, strict=False) -> ViewConfigApplyReport
ViewConfig.load(path) -> ViewConfig
config.save(path) -> Path
config.to_dict() -> dict
ViewConfig.from_dict(mapping) -> ViewConfig
```

`save()` adds `.loupe-view.json` when the supplied path has no suffix. Writes
use an atomic sibling-file replacement, so an interrupted save does not leave a
partially written target.

## Semantic matching

Raw subplot list positions are fragile: splits can be reordered, a recording
can add a channel, and mixed plot types have independent runtime registries.
Loupe therefore uses this priority:

1. An explicit `view_id` on the source Config. If one Config expands into
   several plots, Loupe also records the generated plot's local index.
2. Otherwise, plot kind + displayed name + duplicate occurrence.

Use stable IDs for reusable subject or experiment presets:

```python
data = [
    TraceConfig(eeg, view_id="eeg"),
    TraceConfig(emg, view_id="emg"),
    RasterConfig(
        spikes,
        time_col="time",
        order_by="unit",
        view_id="spikes",
    ),
]
videos = [
    VideoConfig(side_path, side_times, name="Side", view_id="side-camera"),
]
```

`view_id` values for plot Configs must be non-empty and unique within a window.
The same applies to video IDs. `SampleMarkers` also accepts `view_id`; marker
IDs must be unique and are especially useful when several marker sets use the
same symbol. For a `Zip`, put one ID on the `Zip` to identify its generated
subplots and optional IDs on its inner `TraceConfig`s to identify the curves
whose colors should follow each source if the source order changes.

Normal mode is intentionally tolerant: matching settings are applied,
unmatched saved plots are reported, and unmatched current plots retain their
current state. `strict=True` rejects a saved/current plot-inventory mismatch
before applying any setting:

```python
w.apply_view_config(config, strict=True)
# or
view(data, view_config=config, view_config_strict=True)
```

## What version 1 stores

- Global display: time-window length, scroll/playback rates, panel split,
  proportional raster/heatmap layout, raster rendering controls, label alpha
  and visibility, hypnogram mode, and Tuner-panel visibility.
- Every plot: semantic reference, order, visibility, height factor, and manual
  or automatic Y-axis state.
- Stacked/Zip traces: line colors and widths; stacked overlay colors, widths,
  symbols, and symbol sizes.
- Dense plots: gain, trace step, traces per page, and resolved categorical
  palette.
- Heatmaps: `vmin`, `vmax`, colormap, decimation method, and NaN shading. Named colormaps are
  stored by name; an arbitrary Matplotlib colormap is stored as a portable RGBA
  lookup table.
- Rasters: base/category colors and separator style, plus the global brightness,
  event height, and event thickness.
- Sample markers and global event classes: colors, sizes/widths, symbols/styles,
  and alpha.
- Videos: visibility intent, relative layout weight, and frame-step target.
- Optional session and Tuner sections as described above.

It does **not** store data arrays, event tables, interval-label rows, source
paths, video paths, decoder state, screenshots, Python pickles, or opaque Qt
state. Consequently, loading a View-Config never loads or replaces data.

## File contract and compatibility

Files are UTF-8 JSON with these top-level fields:

```json
{
  "format": "loupe-view-config",
  "schema_version": 1,
  "metadata": {},
  "display": {},
  "plots": [],
  "sample_markers": [],
  "global_events": [],
  "videos": []
}
```

`session` and `tuner` are omitted unless requested. JSON values are validated;
NaN, infinity, non-string object keys, and non-JSON Python objects are rejected.
Loupe rejects an unsupported schema version with a clear error. Future format
changes should add an explicit migration in the Qt-free `loupe.view_config`
layer rather than silently interpreting an incompatible file.
