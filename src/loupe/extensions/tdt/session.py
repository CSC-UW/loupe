"""Prepare selected plots, preserve session state, and render independent epocs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from io import StringIO
from pathlib import Path
import copy
import json
import os
import tempfile

import numpy as np
import polars as pl

from loupe.configs import VideoConfig
from loupe.file_series import amplitude_preview
from loupe.interval_labels import IntervalLabelSchema, IntervalLabelSet
from loupe.series import DenseGroup, RasterSeries, Series
from loupe.state_config import load_state_config, parse_color
from .reader import Block, check_cancel, fingerprint, parse_channels

CONFIG_VERSION = 1
PALETTE = ["#73B7E5", "#A8C97F", "#DAA1E5", "#E8AD66", "#86CFC4"]
MODES = {
    "stream": ["dense", "lines"],
    "epoc": ["shade", "ttl", "both"],
    "snips": ["raster"],
}


def config_path(block: Block | Path):
    return (
        (block.path if isinstance(block, Block) else Path(block))
        / ".loupe"
        / "tdt.json"
    )


def default_config(block: Block) -> dict:
    rows = []
    for i, (name, stream) in enumerate(sorted(block.streams.items())):
        rows.append(
            dict(
                kind="stream",
                store=name,
                enabled=stream.fs < 5000 and not stream.error,
                mode="dense" if len(stream.channels) > 2 else "lines",
                channels="all",
                color=PALETTE[i % len(PALETTE)],
                height=1.0,
                gain=0.0,
            )
        )
    for name in sorted(block.epocs):
        rows.append(
            dict(
                kind="epoc",
                store=name,
                enabled=not name.lower().startswith("cam"),
                mode="shade",
                channels="",
                color="#E8AD66",
                height=0.6,
                gain=0.0,
                alpha=0.22,
            )
        )
    for name in sorted(block.snips):
        rows.append(
            dict(
                kind="snips",
                store=name,
                enabled=False,
                mode="raster",
                channels="all",
                color="#DDDDDD",
                height=1.0,
                gain=0.0,
            )
        )
    videos = []
    for path in block.videos:
        matches = [n for n in block.epocs if n.lower() in path.stem.lower()]
        epoc = matches[0] if len(matches) == 1 else ""
        videos.append(
            dict(path=str(path), epoc=epoc, enabled=bool(epoc), correction=0.0)
        )
    return {
        "version": CONFIG_VERSION,
        "extension": "tdt",
        "fingerprint": block.fingerprint,
        "rows": rows,
        "videos": videos,
        "window_len": 10.0,
        "annotation_path": "",
        "annotation_schema": None,
    }


def validate_config(block: Block, config: dict):
    if (
        not isinstance(config, dict)
        or config.get("version") != CONFIG_VERSION
        or config.get("extension") != "tdt"
    ):
        raise ValueError("Unsupported TDT block configuration version")
    if config.get("fingerprint") != block.fingerprint:
        raise ValueError(
            "The block's data files changed since this configuration was saved. Use fresh block settings."
        )
    if not isinstance(config.get("rows"), list) or not isinstance(
        config.get("videos", []), list
    ):
        raise ValueError("Invalid plot or video configuration")
    window = config.get("window_len", 10)
    if not isinstance(window, (int, float)) or not np.isfinite(window) or window <= 0:
        raise ValueError("Window duration must be a positive finite number")
    seen = set()
    registries = {"stream": block.streams, "epoc": block.epocs, "snips": block.snips}
    for row in config["rows"]:
        kind, name = row.get("kind"), row.get("store")
        if (
            kind not in registries
            or name not in registries[kind]
            or (kind, name) in seen
        ):
            raise ValueError(
                f"Unknown or duplicate store in configuration: {kind}/{name}"
            )
        seen.add((kind, name))
        if row.get("mode") not in MODES[kind]:
            raise ValueError(f"Invalid display mode for {name}")
        parse_color(row["color"])
        for field_name in ("height", "gain", "alpha"):
            val = row.get(field_name, 1.0 if field_name == "height" else 0.0)
            if not isinstance(val, (int, float)) or not np.isfinite(val) or val < 0:
                raise ValueError(f"Invalid {field_name} for {name}")
        if row["height"] <= 0 or row.get("alpha", 0.22) > 1:
            raise ValueError(f"Invalid height or opacity for {name}")
        if row.get("enabled"):
            if kind == "stream":
                if block.streams[name].error:
                    raise ValueError(f"{name}: {block.streams[name].error}")
                parse_channels(row["channels"], block.streams[name].channels)
            if kind == "snips":
                parse_channels(
                    row["channels"],
                    [int(c) for c in np.unique(block.snips[name].channels)],
                )
    for video in config.get("videos", []):
        if video.get("enabled"):
            if video.get("epoc") not in block.epocs or not video.get("path"):
                raise ValueError("Each enabled video needs a path and frame epoc store")
            if not np.isfinite(video.get("correction", 0)):
                raise ValueError("Video timing correction must be finite")


def read_config(block: Block) -> dict:
    with config_path(block).open() as f:
        config = json.load(f)
    validate_config(block, config)
    return config


def atomic_json(path: Path, content: dict):
    """Never leave a partial config or replace a good config on serialization failure."""
    payload = json.dumps(content, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tdt-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(payload + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def merged_intervals(intervals: np.ndarray) -> np.ndarray:
    """Union touching/overlapping spans; keep large epoc overlays inexpensive."""
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return np.asarray(merged, dtype=float).reshape(-1, 2)


def ttl_series(name: str, intervals: np.ndarray, duration: float) -> Series:
    """Piecewise-constant union of intervals with exact vertical transitions."""
    times, values = [0.0], [0.0]
    for start, end in merged_intervals(intervals):
        times.extend((start, start, end, end))
        values.extend((0.0, 1.0, 1.0, 0.0))
    times.append(duration)
    values.append(0.0)
    return Series(f"{name} TTL", np.array(times), np.array(values))


def _resolve_path(block: Block, path: str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else block.path / value


@dataclass
class Prepared:
    block: Block
    config: dict
    kwargs: dict
    heights: list[tuple[str, int, float]]
    epocs: list[tuple[dict, np.ndarray]]
    # Temporary timestamp files support read-only blocks without hidden writes.
    temporary: object = None
    notes: list[str] = field(default_factory=list)


def prepare_session(
    block: Block, config: dict, *, cancel=None, progress=lambda message: None
) -> Prepared:
    """Worker-safe data preparation; construct Qt widgets only in launch_session."""
    from loupe.video import MultiFileVideoCapture, validate_frame_times

    config = copy.deepcopy(config)
    validate_config(block, config)
    if fingerprint(block.path) != block.fingerprint:
        raise ValueError("The recording changed after discovery; reopen the block")
    traces, dense, rasters, colors, order, heights, overlays = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for row in config["rows"]:
        check_cancel(cancel)
        if not row.get("enabled"):
            continue
        name, kind = row["store"], row["kind"]
        progress(f"Preparing {name}…")
        if kind == "stream":
            source = block.streams[name]
            channels = parse_channels(row["channels"], source.channels)
            series = []
            for channel in channels:
                check_cancel(cancel)
                axis, signal = source.signal(channel)
                signal.preview()
                series.append(Series(f"{name} · Ch {channel}", axis, signal))
            if row["mode"] == "dense":
                span = np.nanmedian(
                    [
                        np.nanpercentile(amplitude_preview(s.y), 99)
                        - np.nanpercentile(amplitude_preview(s.y), 1)
                        for s in series
                    ]
                )
                gain = row.get("gain", 0) or (0.8 / span if span > 0 else 1.0)
                index, plot_kind = len(dense), "dense"
                dense.append(
                    DenseGroup(
                        name,
                        series,
                        [f"Ch {c}" for c in channels],
                        gain=gain,
                        color_values=np.array([name] * len(series)),
                        palette={name: parse_color(row["color"])},
                    )
                )
                order.append((plot_kind, index))
                heights.append((plot_kind, index, row["height"]))
            else:
                for series_item in series:
                    order.append(("ts", len(traces)))
                    heights.append(("ts", len(traces), row["height"]))
                    traces.append(series_item)
                    colors.append(row["color"])
        elif kind == "epoc":
            intervals = block.epocs[name].intervals(block.duration)
            if row["mode"] in ("shade", "both"):
                overlays.append((row, merged_intervals(intervals)))
            if row["mode"] in ("ttl", "both"):
                order.append(("ts", len(traces)))
                heights.append(("ts", len(traces), row["height"]))
                traces.append(ttl_series(name, intervals, block.duration))
                colors.append(row["color"])
        else:
            source = block.snips[name]
            channels = parse_channels(
                row["channels"], [int(c) for c in np.unique(source.channels)]
            )
            selected = np.isin(source.channels, channels) & np.isfinite(
                source.timestamps
            )
            ts, ch = source.timestamps[selected], source.channels[selected]
            permutation = (
                np.argsort(ts, kind="stable")
                if np.any(np.diff(ts) < 0)
                else slice(None)
            )
            ts, ch = ts[permutation], ch[permutation]
            # Actual channel IDs are tick labels; dense row positions make
            # sparse/ordered selections equally spaced and stable.
            mapping = np.full(max(channels) + 1, -1, dtype=np.int32)
            mapping[channels] = np.arange(len(channels))
            index = len(rasters)
            rasters.append(
                RasterSeries(
                    name,
                    ts,
                    mapping[ch],
                    np.ones(len(ts), dtype=np.float32),
                    parse_color(row["color"])[:3],
                    len(channels),
                    row_keys=np.array(channels),
                )
            )
            order.append(("raster", index))
            heights.append(("raster", index, row["height"]))
    labels = IntervalLabelSet.empty()
    snapshot = config.get("annotation_snapshot")
    if snapshot is not None:
        schema = IntervalLabelSchema(**snapshot["schema"])
        frame = pl.read_json(StringIO(snapshot["data"]))
        labels = (
            IntervalLabelSet.from_dataframe(frame, schema)
            if frame.width
            else IntervalLabelSet.empty(schema)
        )
    elif config.get("annotation_path"):
        schema = config.get("annotation_schema")
        labels = IntervalLabelSet.from_path(
            _resolve_path(block, config["annotation_path"]),
            IntervalLabelSchema(**schema) if schema else None,
        )
    videos, notes = [], []
    temporary = None
    try:
        for i, video in enumerate(config.get("videos", [])):
            check_cancel(cancel)
            if not video.get("enabled"):
                continue
            progress(f"Checking video {i + 1} and recorded frame times…")
            path = _resolve_path(block, video["path"])
            times = validate_frame_times(block.epocs[video["epoc"]].onset)
            # Cam epoc values are often a 1-based frame counter. A skipped
            # counter cannot be repaired merely by trimming timestamps.
            values = block.epocs[video["epoc"]].values
            if (
                len(values) == len(times)
                and len(values) > 1
                and values[0] == 1
                and np.all(values == np.floor(values))
            ):
                if not np.array_equal(values, np.arange(1, len(values) + 1)):
                    raise ValueError(
                        f"{video['epoc']}: discontinuous frame counter; verify video synchronization"
                    )
            cap = MultiFileVideoCapture([str(path)])
            try:
                notes.extend(cap.apply_expected_frame_counts([len(times)], 120))
                if not cap.isOpened():
                    raise ValueError(f"Cannot open video {path}")
                ok, _ = cap.read()
                if not ok:
                    raise ValueError(f"Cannot decode video {path}")
            finally:
                cap.release()
            check_cancel(cancel)
            if temporary is None:
                temporary = tempfile.TemporaryDirectory(prefix="loupe-tdt-")
            times_path = Path(temporary.name) / f"frames-{i}.npy"
            np.save(times_path, times, allow_pickle=False)
            videos.append(
                VideoConfig(
                    str(path),
                    str(times_path),
                    name=video["epoc"],
                    view_id=f"tdt:video:{i}:{video['epoc']}",
                    separate_window=video.get("separate_window", False),
                    frame_times_correction=float(video.get("correction", 0)),
                )
            )
        check_cancel(cancel)
    except Exception:
        if temporary is not None:
            temporary.cleanup()
        raise
    if not order:
        # Epoc/video/annotation-only sessions still have the full block clock.
        traces.append(
            Series(
                "Block timeline",
                np.array([0.0, block.duration]),
                np.array([0.0, 0.0]),
            )
        )
        colors.append("#80808000")
        order.append(("ts", 0))
    state = config.get("state_definitions", {})
    kwargs = dict(
        xr_series=traces,
        dense_groups=dense,
        raster_series_list=rasters,
        colors=colors,
        subplot_order=order,
        window_len=float(config.get("window_len", 10)),
        interval_label_set=labels,
        video_configs=videos,
        state_config=load_state_config(
            keymap=state.get("keymap"), label_colors=state.get("label_colors")
        ),
    )
    return Prepared(block, config, kwargs, heights, overlays, temporary, notes)


class BlockSession:
    def __init__(self, prepared: Prepared):
        self.prepared = prepared
        self._overlay_items = {}
        self._connected_plots = set()
        self._prefix_ends = [
            np.maximum.accumulate(intervals[:, 1]) if len(intervals) else np.array([])
            for _, intervals in prepared.epocs
        ]

    def close(self):
        """Release file mappings and temporary timestamps when a viewer closes."""
        traces = list(self.prepared.kwargs.get("xr_series", []))
        for group in self.prepared.kwargs.get("dense_groups", []):
            traces.extend(group.series)
        for trace in traces:
            if hasattr(trace.y, "_maps"):
                trace.y._maps.clear()
        if self.prepared.temporary is not None:
            self.prepared.temporary.cleanup()
        self._overlay_items.clear()
        self._connected_plots.clear()

    def refresh_overlays(self, window):
        """One batched graphics item per epoc/plot, restricted to the viewport."""
        import pyqtgraph as pg
        from PySide6 import QtCore, QtGui, QtWidgets

        plots = (
            window.plots
            + window.dense_plots
            + window.raster_plots
            + window.heatmap_plots
        )
        active = set(plots)
        for key in list(self._overlay_items):
            if key[0] not in active:
                del self._overlay_items[key]
        self._connected_plots.intersection_update(active)
        left, right = window.window_start, window.window_start + window.window_len
        for ei, (row, spans) in enumerate(self.prepared.epocs):
            first = np.searchsorted(self._prefix_ends[ei], left, side="right")
            stop = np.searchsorted(spans[:, 0], right, side="left") if len(spans) else 0
            visible = spans[first:stop]
            for plot in plots:
                if plot not in self._connected_plots:
                    plot.getViewBox().sigYRangeChanged.connect(
                        lambda *args: self.refresh_overlays(window)
                    )
                    self._connected_plots.add(plot)
                key = (plot, ei)
                item = self._overlay_items.get(key)
                if item is None:
                    item = QtWidgets.QGraphicsPathItem()
                    item.setPen(pg.mkPen(None))
                    item.setZValue(-7)
                    plot.addItem(item, ignoreBounds=True)
                    self._overlay_items[key] = item
                color = pg.mkColor(row["color"])
                color.setAlphaF(row.get("alpha", 0.22))
                item.setBrush(pg.mkBrush(color))
                path = QtGui.QPainterPath()
                y0, y1 = plot.getViewBox().viewRange()[1]
                for start, end in visible:
                    start, end = max(left, start), min(right, end)
                    if end > start:
                        path.addRect(QtCore.QRectF(start, y0, end - start, y1 - y0))
                item.setPath(path)
        for i, raster in enumerate(window.raster_series):
            if raster.row_keys is not None:
                plot = window.raster_plots[i]
                plot.getAxis("left").setTicks(
                    [[(j, str(ch)) for j, ch in enumerate(raster.row_keys)]]
                )

    def save(self, window) -> Path:
        from loupe.view_config_runtime import capture_view_config

        for current, key in [
            (window.series, "xr_series"),
            (window.dense_groups, "dense_groups"),
            (window.raster_series, "raster_series_list"),
        ]:
            expected = self.prepared.kwargs[key]
            if len(current) != len(expected) or any(
                a is not b for a, b in zip(current, expected)
            ):
                raise ValueError(
                    "This viewer's data sources changed. Reopen the TDT block to save its block configuration."
                )
        config = copy.deepcopy(self.prepared.config)
        config["window_len"] = float(window.window_len)
        enabled_videos = [
            video for video in config.get("videos", []) if video.get("enabled")
        ]
        active_slots = [slot for slot in window.video_slots if slot.video_path]
        if len(active_slots) != len(enabled_videos) or window.heatmap_series:
            raise ValueError(
                "This viewer's data sources changed. Reopen the TDT block to save its block configuration."
            )
        for video, source, slot in zip(
            enabled_videos, self.prepared.kwargs["video_configs"], active_slots
        ):
            if slot.frame_times_path != source.frame_times_path:
                raise ValueError(
                    "Video frame-time sources changed. Choose the video and frame epoc in the block launcher before saving."
                )
            video.update(
                path=slot.video_path,
                correction=slot.frame_times_correction,
                separate_window=slot.window_group if slot.separate_window else False,
            )
        config["view_config"] = capture_view_config(
            window, include_session=True
        ).to_dict()
        labels = window.interval_label_set
        config["annotation_snapshot"] = {
            "schema": asdict(labels.schema),
            "data": labels.to_savable_df().write_json(),
        }
        config["state_definitions"] = {
            "keymap": window.state_config.keys_for_state,
            "label_colors": window.state_config.label_colors,
        }
        # Persist independently editable epoc color/opacity alongside view state.
        rows = {(row["kind"], row["store"]): row for row in config["rows"]}
        for row, _ in self.prepared.epocs:
            rows[(row["kind"], row["store"])].update(
                color=row["color"], alpha=row.get("alpha", 0.22)
            )
        path = config_path(self.prepared.block)
        atomic_json(path, config)
        self.prepared.config = config
        return path


def launch_session(prepared: Prepared, *, save: bool = True):
    """Build the viewer in the GUI thread and attach its TDT session controls."""
    from PySide6 import QtCore, QtWidgets
    from loupe.app import LoupeApp

    window = LoupeApp(**prepared.kwargs)
    window.setWindowTitle(f"Loupe — TDT · {prepared.block.path.name}")
    session = window.extension_session = BlockSession(prepared)
    for kind, index, height in prepared.heights:
        {
            "ts": window.plot_height_factors,
            "dense": window.dense_height_factors,
            "raster": window.raster_height_factors,
        }[kind][index] = height
    window._apply_custom_plot_heights()
    # Include epocs, video, and annotations all the way to the block end.
    window.t_global_min, window.t_global_max = 0.0, prepared.block.duration
    window.show()
    try:
        if prepared.config.get("view_config"):
            window.apply_view_config(prepared.config["view_config"], strict=True)
        window.t_global_min, window.t_global_max = 0.0, prepared.block.duration
        window._apply_x_range()
        window._update_nav_slider_from_window()
        window._update_hypnogram_extents()
    except Exception:
        window.close()
        raise

    def save_current():
        try:
            path = session.save(window)
            window._update_status(f"Saved block config: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                window, "Could not save block config", str(exc)
            )

    action = window.extensions_menu.addAction("Update saved block config")
    action.triggered.connect(save_current)
    window.update_block_config_action = action
    epoc_action = window.extensions_menu.addAction("TDT epoc appearance…")
    epoc_action.setEnabled(bool(prepared.epocs))

    def style_epocs():
        from .launcher import EpocStyleDialog

        dialog = EpocStyleDialog(window)
        dialog.exec()

    epoc_action.triggered.connect(style_epocs)
    if save:
        try:
            session.save(window)
        except OSError as exc:
            # Viewing read-only recordings remains useful, and the failure is
            # explicit; no success message is shown for an unsaved session.
            QtWidgets.QMessageBox.warning(
                window, "Block opened; configuration not saved", str(exc)
            )
    if prepared.notes or prepared.block.warnings:
        window._update_status("; ".join(prepared.block.warnings + prepared.notes))
    app = QtWidgets.QApplication.instance()
    if not hasattr(app, "_loupe_block_windows"):
        app._loupe_block_windows = []
    app._loupe_block_windows.append(window)
    window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
    window.destroyed.connect(
        lambda: (
            app._loupe_block_windows.remove(window)
            if window in app._loupe_block_windows
            else None
        )
    )
    return window
