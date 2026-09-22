"""Video-decode pipeline classes used by :class:`loupe.app.LoupeApp`.

- :class:`MultiFileVideoCapture` adapts a list of MP4s as one virtual
  capture, mapping global frame indices to the appropriate file.
- :class:`VideoWorker` is a ``QObject`` living on a decoder thread; it
  pulls frames out of OpenCV and emits ``frameReady`` to the UI thread.
- :class:`VideoSlot` bundles the per-slot runtime state owned by the app.

The slot controller methods (loading, frame requests, rescaling) remain on
``LoupeApp`` since they need access to the current time window, layout,
and cursor state — only the leaf classes move here.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

try:
    import cv2
except Exception:
    cv2 = None


def validate_frame_times(times: np.ndarray) -> np.ndarray:
    """Require one finite, strictly increasing timestamp per video frame."""
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not times.size:
        raise ValueError("Frame timestamps must be a nonempty 1-D array.")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("Frame timestamps must be finite and strictly increasing.")
    return times


def frame_time_tolerance(times: np.ndarray, max_distance_s: float | None) -> float:
    """Default to half a typical frame interval, allowing 10% timing jitter."""
    if max_distance_s is not None:
        value = float(max_distance_s)
        if not np.isfinite(value) or value < 0:
            raise ValueError("max_frame_distance_s must be finite and nonnegative.")
        return value
    return 0.55 * float(np.median(np.diff(times))) if len(times) > 1 else 0.0


def frame_index_at_time(times: np.ndarray, time: float, tolerance: float) -> int | None:
    """Find the nearest available frame without extending it through gaps."""
    if not len(times) or not np.isfinite(time):
        return None
    idx = int(np.searchsorted(times, time, side="left"))
    idx = min(idx, len(times) - 1)
    if idx > 0 and abs(times[idx - 1] - time) <= abs(times[idx] - time):
        idx -= 1
    return idx if abs(float(times[idx]) - time) <= tolerance + 1e-12 else None


# How many indexed-but-undecodable trailing frames a file may have before its
# frame count is treated as a genuine mismatch rather than a truncated tail.
# Fragmented MP4s closed mid-fragment (the final file of an e3Vision recording)
# list up to one fragment of frames that were never written; a fragment is
# 60 frames at 60 fps there, so two fragments leaves headroom.
DEFAULT_FRAME_COUNT_SLACK = 120


@dataclass
class FrameCountReport:
    """Per-file frame counts a decoder reports once a video is open.

    ``header`` is what each container's index claims; ``usable`` is the count
    playback actually maps through (capped at the timestamp count when a
    truncated tail was tolerated); ``notes`` explains any cap, one per file.
    """

    header: list[int]
    usable: list[int]
    notes: list[str]


def frame_decodes(cap, index: int) -> bool:
    """Whether seeking *cap* to *index* and decoding yields a frame."""
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(index)):
        return False
    ok, _ = cap.read()
    return bool(ok)


class VideoRelay(QtCore.QObject):
    """Deliver decoder signals to GUI-owned slots on the GUI thread.

    A functools.partial directly connected to a worker signal is a plain
    Python callable, so PySide may invoke it on the emitting decoder thread.
    This QObject supplies the explicit receiver context needed by Qt.
    """

    def __init__(self, app, slot):
        super().__init__(app)
        self.app = app
        self.slot = slot

    @QtCore.Slot(int, QtGui.QImage)
    def frame_ready(self, index, image):
        self.app._on_frame_ready(self.slot, index, image)

    @QtCore.Slot(bool, str)
    def opened(self, ok, message):
        self.app._on_video_opened(self.slot, ok, message)

    @QtCore.Slot(object)
    def frame_counts(self, counts):
        self.app._on_video_frame_counts(self.slot, counts)


class VideoWindow(QtWidgets.QDialog):
    """Resizable shared video window; closing it returns videos to Loupe."""

    returned = QtCore.Signal()
    resized = QtCore.Signal()

    def __init__(self, title: str, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 700)
        self.grid = QtWidgets.QGridLayout(self)
        self.panels: dict[int, QtWidgets.QWidget] = {}
        self.labels: dict[int, QtWidgets.QLabel] = {}
        self.captions: dict[int, QtWidgets.QLabel] = {}

    def add_video(self, index: int, name: str) -> QtWidgets.QLabel:
        panel = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(panel)
        label = QtWidgets.QLabel(f"No {name.lower()}")
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(120, 100)
        label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        label.setStyleSheet("color:#ddd;background-color:#222;border:1px solid #444;")
        caption = QtWidgets.QLabel(name)
        caption.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label, 1)
        layout.addWidget(caption)
        self.panels[index] = panel
        self.labels[index] = label
        self.captions[index] = caption
        # Decoders open asynchronously. Keep the grid in configuration order,
        # independent of which worker finishes first.
        for item in self.panels.values():
            self.grid.removeWidget(item)
        for position, slot_index in enumerate(sorted(self.panels)):
            self.grid.addWidget(self.panels[slot_index], position // 2, position % 2)
        return label

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()

    def closeEvent(self, event):
        self.returned.emit()
        super().closeEvent(event)


class MultiFileVideoCapture:
    """Adapter exposing a list of MP4s as a single cv2.VideoCapture-like object.

    Implements the subset of the cv2.VideoCapture interface used by
    VideoWorker: isOpened, set(CAP_PROP_POS_FRAMES, idx), read, release,
    get(prop). Frame indices are global across the concatenated sequence;
    seeking transparently switches between underlying captures. Reads past
    the usable frame count fail rather than falling through to frames the
    container indexes but never wrote.
    """

    def __init__(self, paths: list[str]):
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is not installed.")
        self._caps = [cv2.VideoCapture(p) for p in paths]
        # What each container's index claims. A fragmented MP4 closed
        # mid-fragment lists frames past the end of the file, so this is only
        # an upper bound on what decodes; frame_counts holds the usable counts.
        self.header_frame_counts = [
            int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in self._caps
        ]
        self._active_idx = 0
        self._next_frame_idx = 0
        self._set_frame_counts(self.header_frame_counts)

    def _set_frame_counts(self, counts: list[int]) -> None:
        """Define the per-file frame counts that global indices map through."""
        self.frame_counts = [int(c) for c in counts]
        # _cumulative[i] = total frames in files 0..i-1; _cumulative[-1] = total.
        self._cumulative = np.concatenate(
            ([0], np.cumsum(self.frame_counts))
        ).astype(np.int64)
        self._total_frames = int(self._cumulative[-1])

    def apply_expected_frame_counts(
        self, expected: list[int], slack: int | None
    ) -> list[str]:
        """Reconcile each file's indexed frame count with its timestamp count.

        A container may list more frames than there are timestamps only when
        the last timestamped frame decodes and the first frame past it does
        not: that is a truncated tail, and the file's usable count is capped at
        the timestamp count so global frame indices stay aligned with the
        concatenated timestamps. *slack* bounds the tolerated excess (``None``
        for no bound). Returns one note per capped file; raises ``ValueError``
        when the counts genuinely disagree. Leaves every capture at frame 0.
        """
        expected = [int(n) for n in expected]
        if len(expected) != len(self._caps):
            raise ValueError(
                f"{len(expected)} timestamp arrays for {len(self._caps)} video files."
            )
        mismatch = (
            f"Video frame counts {self.header_frame_counts} do not match "
            f"timestamp counts {expected}"
        )
        usable: list[int] = []
        notes: list[str] = []
        for i, (cap, n_header, n_expected) in enumerate(
            zip(self._caps, self.header_frame_counts, expected)
        ):
            label = f"file {i + 1}" if len(self._caps) > 1 else "video"
            if n_header == n_expected:
                usable.append(n_header)
                continue
            if n_header < n_expected:
                raise ValueError(
                    f"{mismatch}: {label} has {n_expected - n_header} "
                    "timestamp(s) past the end of the video."
                )
            excess = n_header - n_expected
            if slack is not None and excess > slack:
                raise ValueError(
                    f"{mismatch}: {label} lists {excess} more frames than "
                    f"timestamps, above the {slack}-frame allowance for a "
                    "truncated tail."
                )
            if n_expected < 1 or not frame_decodes(cap, n_expected - 1):
                raise ValueError(
                    f"{mismatch}: {label}'s last timestamped frame "
                    f"({n_expected - 1}) does not decode."
                )
            if frame_decodes(cap, n_expected):
                raise ValueError(
                    f"{mismatch}: {label} decodes frame {n_expected}, "
                    "which has no timestamp."
                )
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            usable.append(n_expected)
            notes.append(
                f"{label} lists {excess} frame(s) past its last timestamp that "
                f"do not decode (truncated tail); playback capped at {n_expected}"
            )
        self._set_frame_counts(usable)
        self._active_idx = 0
        self._next_frame_idx = 0
        return notes

    def isOpened(self) -> bool:
        return self._total_frames > 0 and all(c.isOpened() for c in self._caps)

    def set(self, prop, value) -> bool:
        if prop != cv2.CAP_PROP_POS_FRAMES:
            return bool(self._caps[self._active_idx].set(prop, value))
        if self._total_frames == 0:
            return False
        global_idx = max(0, min(int(value), self._total_frames - 1))
        # file_idx = smallest j such that _cumulative[j+1] > global_idx.
        file_idx = int(
            np.searchsorted(self._cumulative[1:], global_idx, side="right")
        )
        if file_idx >= len(self._caps):
            file_idx = len(self._caps) - 1
        local_idx = global_idx - int(self._cumulative[file_idx])
        self._active_idx = file_idx
        self._next_frame_idx = global_idx
        return bool(
            self._caps[file_idx].set(cv2.CAP_PROP_POS_FRAMES, int(local_idx))
        )

    def read(self):
        if not self._prepare_read():
            return False, None
        ok, frame = self._caps[self._active_idx].read()
        if ok:
            self._next_frame_idx += 1
        return ok, frame

    def _prepare_read(self) -> bool:
        if not self._caps or self._next_frame_idx >= self._total_frames:
            return False
        if self._next_frame_idx >= self._cumulative[self._active_idx + 1]:
            return self.set(cv2.CAP_PROP_POS_FRAMES, self._next_frame_idx)
        return True

    def grab(self) -> bool:
        if not self._prepare_read():
            return False
        ok = bool(self._caps[self._active_idx].grab())
        if ok:
            self._next_frame_idx += 1
        return ok

    def release(self) -> None:
        for c in self._caps:
            c.release()
        self._caps = []
        self._cumulative = np.array([0], dtype=np.int64)
        self._total_frames = 0
        self._active_idx = 0

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._total_frames)
        return self._caps[0].get(prop) if self._caps else 0.0


class VideoWorker(QtCore.QObject):
    frameReady = QtCore.Signal(int, QtGui.QImage)
    opened = QtCore.Signal(bool, str)
    frameCounts = QtCore.Signal(object)

    def __init__(self, cache_frames=120, frame_count_slack=DEFAULT_FRAME_COUNT_SLACK):
        super().__init__()
        self.cap = None
        self.cache = OrderedDict()
        self.cache_frames = int(cache_frames)
        self.frame_count_slack = (
            None if frame_count_slack is None else int(frame_count_slack)
        )
        self._requested_idx: int | None = None
        self._request_queued = False
        self._next_frame_idx = 0

    @QtCore.Slot(str)
    @QtCore.Slot(str, "QVariantList")
    def open(self, path, expected_frame_counts=None):
        self._open([path], expected_frame_counts)

    @QtCore.Slot("QStringList")
    @QtCore.Slot("QStringList", "QVariantList")
    def openConcat(self, paths, expected_frame_counts=None):
        self._open(list(paths), expected_frame_counts)

    def _open(self, paths: list[str], expected_frame_counts=None):
        """Open *paths* as one capture, emitting frameCounts then opened.

        *expected_frame_counts* (one per file, the timestamp array lengths)
        is reconciled against the containers' indexed counts; see
        :meth:`MultiFileVideoCapture.apply_expected_frame_counts`. Without it
        the indexed counts are taken at face value.
        """
        if cv2 is None:
            self.opened.emit(False, "OpenCV (cv2) not installed.")
            return
        try:
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            self.cache.clear()
            self._requested_idx = None
            self._request_queued = False
            self._next_frame_idx = 0
            cap = MultiFileVideoCapture(paths)
            if not cap.isOpened():
                cap.release()
                self.opened.emit(False, f"Failed to open: {paths}")
                return
            notes: list[str] = []
            if expected_frame_counts is not None:
                try:
                    notes = cap.apply_expected_frame_counts(
                        [int(n) for n in expected_frame_counts],
                        self.frame_count_slack,
                    )
                except Exception:
                    cap.release()
                    raise
            self.cap = cap
            self.frameCounts.emit(
                FrameCountReport(
                    header=list(cap.header_frame_counts),
                    usable=list(cap.frame_counts),
                    notes=notes,
                )
            )
            self.opened.emit(True, "")
        except Exception as e:
            self.opened.emit(False, str(e))

    @QtCore.Slot(int)
    def requestFrame(self, idx):
        if self.cap is None:
            return

        self._requested_idx = int(idx)
        if self._request_queued:
            return
        self._request_queued = True
        QtCore.QMetaObject.invokeMethod(
            self,
            "_processRequestedFrame",
            QtCore.Qt.QueuedConnection,
        )

    @QtCore.Slot()
    def _processRequestedFrame(self):
        if self.cap is None or self._requested_idx is None:
            self._request_queued = False
            return

        idx = int(self._requested_idx)
        self._requested_idx = None

        qimg = self.cache.get(idx)
        if qimg is None:
            # Sequential decode avoids re-decoding an entire GOP on every tick.
            # Small forward skips can also use cheap grabs; distant/reverse
            # seeks still go through the codec's frame index.
            positioned = True
            if not (0 <= self._next_frame_idx <= idx <= self._next_frame_idx + 8):
                positioned = bool(self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx))
                self._next_frame_idx = idx if positioned else -1
            while positioned and self._next_frame_idx < idx:
                positioned = bool(self.cap.grab())
                if not positioned:
                    break
                self._next_frame_idx += 1
            ok, frame = self.cap.read() if positioned else (False, None)
            if ok:
                self._next_frame_idx = idx + 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QtGui.QImage(
                    rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888
                ).copy()
                self.cache[idx] = qimg
                if len(self.cache) > self.cache_frames:
                    self.cache.popitem(last=False)
            else:
                # Decoder position is unknown after failure. Clear the UI via
                # the null image below and force a real seek on the next try.
                self._next_frame_idx = -1

        # Skip emitting stale frames when a newer request is already pending.
        if self._requested_idx is None:
            self.frameReady.emit(idx, qimg if qimg is not None else QtGui.QImage())

        if self._requested_idx is not None:
            QtCore.QMetaObject.invokeMethod(
                self,
                "_processRequestedFrame",
                QtCore.Qt.QueuedConnection,
            )
        else:
            self._request_queued = False

    @QtCore.Slot()
    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cache.clear()
        self._requested_idx = None
        self._request_queued = False
        self._next_frame_idx = 0


@dataclass
class VideoSlot:
    """Per-video runtime state for :class:`LoupeApp`.

    Bundles the worker/thread pair, frame times, last-rendered pixmap,
    UI label, and menu actions for one synchronized video source.
    ``video_path`` and ``frame_times_path`` may each be either a single
    path or a list of paths; when both are lists they are loaded as one
    continuous (concatenated) video.
    """

    index: int
    name: str
    stretch: int
    worker: VideoWorker
    thread: QtCore.QThread
    video_path: "str | list[str] | None" = None
    frame_times_path: "str | list[str] | None" = None
    label: QtWidgets.QLabel | None = None
    show_action: QtGui.QAction | None = None
    step_action: QtGui.QAction | None = None
    frame_times: np.ndarray | None = None
    frame_times_correction: float = 0.0
    is_open: bool = False
    last_pixmap: QtGui.QPixmap | None = None
    requested_frame_idx: int | None = None
    view_id: str | None = None
    # Visibility intent is independent of QWidget.isVisible(), which is false
    # while an ancestor is hidden and can be overwritten by an async open.
    desired_visible: bool = True
    max_frame_distance_s: float | None = None
    frame_time_tolerance: float | None = None
    frame_count_slack: int | None = DEFAULT_FRAME_COUNT_SLACK
    expected_frame_counts: list[int] | None = None
    # Usable per-file counts the decoder reported (capped at the timestamps
    # when a truncated tail was tolerated) and the raw container counts.
    video_frame_counts: list[int] | None = None
    header_frame_counts: list[int] | None = None
    displayed_frame_idx: int | None = None
    displayed_frame_time: float | None = None
    separate_window: bool = False
    window_group: str = "Videos"
    window: VideoWindow | None = None
    window_action: QtGui.QAction | None = None
    relay: VideoRelay | None = None
