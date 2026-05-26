# Video (`VideoConfig`)

Time-synchronized video frames displayed in the right panel, stacked vertically, locked to the trace cursor. Multiple videos play together; each runs in its own `VideoWorker` thread.

Defined in `src/loupe/__init__.py:319-355`.

## Full parameter reference

| Param | Default | Purpose |
| --- | --- | --- |
| `video_path` | _required_ | Path to a file readable by OpenCV (`.mp4`, `.avi`, `.mov`, `.mkv`), **or** a list of such paths displayed as one continuous (concatenated) video. |
| `frame_times_path` | _required_ | Path to a 1-D `.npy` file of per-frame timestamps in seconds, or a list of such paths matching `video_path`. Arrays are concatenated as-is — the caller is responsible for ensuring a single shared time axis. |
| `name` | `None` | Display label used for the empty-frame placeholder and the View → Show / Frame Step Target menu entries. Defaults to `"Video {i+1}"`. |
| `stretch` | `None` | Initial vertical layout weight relative to other videos. Defaults to `3` for the first slot and `2` for the rest. |
| `frame_times_correction` | `0.0` | Scalar (seconds) added to every frame time after loading. Applied uniformly whether `frame_times_path` is a single file or a list. Useful as a quick alignment shim against the trace cursor without rewriting the underlying `.npy` files. |

## Usage

```python
from loupe import view, TraceConfig, VideoConfig

view(TraceConfig(da), videos=[
    VideoConfig("cam1.mp4",    "cam1_frame_times.npy",    name="side cam"),
    VideoConfig("cam2.mp4",    "cam2_frame_times.npy",    name="overhead"),
    VideoConfig("thermal.mp4", "thermal_frame_times.npy", name="thermal"),
])

# Multi-file concat — frame_times must also be a list of equal length:
view(TraceConfig(da), videos=VideoConfig(
    video_path=["session_part1.mp4", "session_part2.mp4"],
    frame_times_path=["part1_t.npy", "part2_t.npy"],
    name="merged",
    frame_times_correction=-0.04,
))
```

A bare `VideoConfig` is accepted as shorthand for a one-element list.

## Runtime controls

| Action | Binding |
| --- | --- |
| Step the selected video back one frame | `Left` (hold to repeat) |
| Step the selected video forward one frame | `Right` (hold to repeat) |
| Toggle visibility of the _N_-th video | `Ctrl+Shift+1` … `Ctrl+Shift+9` |
| Toggle playback | `Space` (loops within current window) |
| Set playback speed | View → Set Playback Speed… (0.25× – 4×) |
| Choose which video the arrows step | View → Frame Step Target |

A per-window cursor slider sits underneath the top video.

## Hot-path entry points

- Single-file open: `VideoWorker.open` in `loupe.app`.
- Multi-file concat: `VideoWorker.openConcat`.
- Slot loop: `LoupeApp._on_frame_ready(slot, ...)`, `_rescale_video_frame(slot)`, `_request_video_frame(slot, t)`.
- Public config: `loupe.VideoConfig`.
- Per-slot state: `VideoSlot` (internal).

## Notes

- When no videos are passed, the right panel shows a dark placeholder. The hypnogram (`h` to toggle) can be used to free vertical space; an open TODO covers repurposing the panel when no videos are loaded.
- `frame_times_correction` shifts the entire timestamp array uniformly — there is no per-segment offset for concatenated videos. If you need per-segment correction, pre-process the `.npy` files.
