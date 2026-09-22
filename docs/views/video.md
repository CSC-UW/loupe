# Video (`VideoConfig`)

Time-synchronized video frames displayed in the right panel, stacked vertically, locked to the trace cursor. Multiple videos play together; each runs in its own `VideoWorker` thread.

Defined in `src/loupe/configs.py`.

## Full parameter reference

| Param | Default | Purpose |
| --- | --- | --- |
| `video_path` | _required_ | Path to a file readable by OpenCV (`.mp4`, `.avi`, `.mov`, `.mkv`), **or** a list of such paths displayed as one continuous (concatenated) video. |
| `frame_times_path` | _required_ | Path to a 1-D `.npy` file of per-frame timestamps in seconds, or a list of such paths matching `video_path`. Arrays are concatenated as-is — the caller is responsible for ensuring a single shared time axis. |
| `name` | `None` | Display label used for the empty-frame placeholder and the View → Show / Frame Step Target menu entries. Defaults to `"Video {i+1}"`. |
| `stretch` | `None` | Initial vertical layout weight relative to other videos. Defaults to `3` for the first slot and `2` for the rest. |
| `frame_times_correction` | `0.0` | Scalar (seconds) added to every frame time after loading. Applied uniformly whether `frame_times_path` is a single file or a list. Useful as a quick alignment shim against the trace cursor without rewriting the underlying `.npy` files. |
| `max_frame_distance_s` | `None` | Maximum distance from the cursor to a frame timestamp. The default is 0.55 × the median frame interval. The frame clears outside this range, so gaps never hold an old image indefinitely. |
| `separate_window` | `False` | `True` groups videos in a separate window titled "Videos"; a string chooses a shared window title. Four videos with the same title form a 2 × 2 grid. |
| `view_id` | `None` | Stable identity for saved visibility and layout preferences. |

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

Use `separate_window="Microscope"` on each microscope video to display them
together in a resizable window while keeping traces in the main window. These
videos start visible, and their captions show the actual decoded frame index
and timestamp. Closing the separate window returns its videos to the main panel.

Each timestamp array must be nonempty, finite, strictly increasing, and match
the decoded frame count of its corresponding file. Concatenated arrays must
also increase across file boundaries. A container whose index lists a few more
frames than it holds (a fragmented MP4 closed mid-fragment, as the last file of
an e3Vision recording is) is accepted when its last timestamped frame decodes
and the next one does not: playback is capped at the timestamp count and the
status bar notes the excess. `frame_count_slack` (default 120 frames) bounds
how large that excess may be; any other disagreement is an error. Playback
follows these timestamps and the trace cursor's clock; the video's encoded FPS
does not set the data clock.
Supply timestamps already in the data's timebase (for example, ephys seconds)
and keep `frame_times_correction=0` when that conversion has already been done.

## Runtime controls

| Action | Binding |
| --- | --- |
| Step the selected video back one frame | `Left` (hold to repeat) |
| Step the selected video forward one frame | `Right` (hold to repeat) |
| Toggle visibility of the _N_-th video | `Ctrl+Shift+1` … `Ctrl+Shift+9` |
| Toggle playback | `Space` (loops within current window) |
| Set playback speed | View → Set Playback Speed… (0.25× – 4×) |
| Choose which video the arrows step | View → Frame Step Target |
| Move a video to/from a separate window | View → _Video name_ in Separate Window |

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
