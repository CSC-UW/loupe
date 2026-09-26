# Acquisition extensions

Extensions open recordings directly from their native acquisition folders.
Loupe includes a TDT extension; acquisition SDKs are optional dependencies.
Existing `view(...)` workflows continue to work without any extension installed.

## TDT quick start

Install the extra in your consuming uv project:

```bash
uv add 'loupe[tdt]'
uv run loupe
```

For a local editable checkout, add that checkout with the `tdt` extra instead.
The equivalent module entry point is `python -m loupe`. On this workstation,
the existing shared environment already has the TDT SDK and this editable
checkout, so you can launch immediately with:

```bash
/data/slap_analysis/slap_mi_2_sleep/.venv/bin/python -m loupe
```

Choose **File → Extensions → TDT → Open block…**, then select a block folder.
You can also open its launcher directly:

```bash
python -m loupe --tdt /path/to/block
```

In a notebook, enable Qt before opening the launcher:

```python
%gui qt6
from loupe.extensions.tdt import open_block
launcher = open_block('/path/to/block')
```

### Choose what to display

- **Streams:** select stores, choose a dense plot or a line per channel, and
  enter `all` or an ordered channel list such as `1-4, 8, 6`. IDs are the
  recorded TDT channel numbers. Dense gain `0` chooses an amplitude scale from
  a small preview; raw data are never rescaled on disk.
- **Epocs:** choose global shading, a TTL trace, or both. Color and opacity
  are independent per store. TTL is the union of active intervals: 0 outside
  and 1 inside, with exact onset/offset transitions. Open-ended intervals end
  at the block boundary. Epoc shading remains independent of editable labels.
- **Snips:** choose channels for a spike raster. The y-axis labels are recorded
  channel IDs, in the requested order; sort codes do not split rows. Waveform
  payloads are never loaded.
- **Video:** local video files are discovered automatically. A filename with
  an unambiguous epoc name (for example `Cam1`) is enabled by default. Add a
  video from another folder if needed, choose the frame epoc and optionally
  set a time correction in seconds. Loupe uses its onset timestamps and
  verifies their count against the video. It does not synthesize timing from
  encoded FPS or silently trim mismatched frame arrays.
- **Annotations:** choose CSV, HTSV, Parquet or Visbrain text. CSV column
  mappings can be adjusted in the launcher; start/end or start/duration and a
  state/label column are supported. Times must be seconds from block start.
- **Layout:** use Move up/down to order stores and set height and color before
  loading. Channel order determines the order of per-channel plots. Loupe's
  normal plot-order and display controls remain available after loading.

Discovery and preparation run in a worker thread. Cancel closes the launcher
after the current read returns; it never terminates a disk-reading thread or
writes an incomplete configuration. Each loaded block opens a separate viewer.

### Save and reopen a block

With **Save block config** checked, loading writes `.loupe/tdt.json` inside the
block. On a later visit, click **Use existing block config**, then **Load block**.
After adjusting a viewer, choose **File → Extensions → Update saved block
config**. This saves:

- Store/channel choices, video selection and timing correction.
- Plot order, visibility, colors, heights, scales, dense gain and hidden channels.
- Window position/duration, video visibility/layout, and label display settings.
- Annotation edits, notes, label colors and state hotkeys.
- Epoc color and opacity, adjustable via **TDT epoc appearance…**.

The annotation snapshot is contained in the block config; the original imported
annotation file is not overwritten. Use Loupe's label export when you want a
standalone annotation file. Choosing another annotation file or Clear replaces
the snapshot. Editing store settings in the launcher starts a fresh plot layout;
the saved runtime layout otherwise takes precedence.

Config writes use an atomic replacement. A file-size/mtime inventory detects
changed acquisition data and prevents silently reusing a stale config. Fresh
settings remain available if a saved config is corrupt or incompatible. A
read-only block can be viewed with Save block config unchecked. Video timestamp
files are temporary and are removed when the viewer closes; no full stream
cache or duplicated recording is required.

### Data and timing contract

The extension uses the TDT SDK for TSQ/TBK metadata and reads selected stream
windows directly from memory-mapped SEV or TEV files. It allocates neither a
full-recording sample array nor a full-recording timestamp vector. Wide windows
use a peak-preserving envelope with bounded read buffers and a cached viewport.
Initial y-limits and dense centering use a bounded amplitude preview; the
viewer's scale controls can refine them.

For single-file-per-channel SEV and indexed TEV stores, recorded TSQ packet
timestamps anchor the sample clock. This avoids cumulative drift from rounded
sampling-rate metadata. Actual packet gaps appear as NaNs, with blank overview
bins rather than interpolation. Split SEV recordings use their SEV logs and
nominal sampling rates; missing segments without timing logs fail explicitly.
Unsupported SEV headers/sample formats produce an error instead of guessed data.

The current extension supports SEV versions 1–3 with float32, int32, int16,
int8 or float64 samples and conventional TEV streams. It does not decode packed
raw formats or create scalar-store plots. Scalar store names appear in the block
information. Changing a viewer's data sources with generic file-loading actions
cannot be saved as a reconstructable TDT config; reopen the block launcher for
store changes.

References: [TDT Python SDK documentation](https://www.tdt.com/docs/sdk/offline-data-analysis/offline-data-python/)
and the installed SDK's `read_block` / `read_sev` implementations. See
[validation results](tdt-validation.md) for the example-block checks.

## Add another extension

An external package registers a factory in the `loupe.extensions` entry-point
group. The factory returns an `Extension` descriptor. Keep factories cheap;
import acquisition SDKs inside the opener, not at package import time.

```toml
[project.entry-points."loupe.extensions"]
my_system = "my_loupe_extension:extension"
```

```python
from loupe.extensions import Extension

def extension():
    return Extension(
        id='my_system', name='My system',
        opener='my_loupe_extension.launcher:open_recording',
        action='Open recording…',
    )

# open_recording(*, parent=None, path=None) returns a shown Qt dialog/window.
```

IDs must be unique; `tdt` is reserved by the built-in extension. A failing
external extension is reported in the Extensions menu without preventing core
Loupe from opening. An extension may provide its own optional-dependency check
and installation instructions. The `Extension.dependency` / `extra` fields are
provided for extras distributed as part of Loupe.
