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


class MultiFileVideoCapture:
    """Adapter exposing a list of MP4s as a single cv2.VideoCapture-like object.

    Implements the subset of the cv2.VideoCapture interface used by
    VideoWorker: isOpened, set(CAP_PROP_POS_FRAMES, idx), read, release,
    get(prop). Frame indices are global across the concatenated sequence;
    seeking transparently switches between underlying captures.
    """

    def __init__(self, paths: list[str]):
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is not installed.")
        self._caps = [cv2.VideoCapture(p) for p in paths]
        counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in self._caps]
        # _cumulative[i] = total frames in files 0..i-1; _cumulative[-1] = total.
        self._cumulative = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        self._total_frames = int(self._cumulative[-1])
        self._active_idx = 0

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
        return bool(
            self._caps[file_idx].set(cv2.CAP_PROP_POS_FRAMES, int(local_idx))
        )

    def read(self):
        if not self._caps:
            return False, None
        return self._caps[self._active_idx].read()

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

    def __init__(self, cache_frames=120):
        super().__init__()
        self.cap = None
        self.cache = OrderedDict()
        self.cache_frames = int(cache_frames)
        self._requested_idx: int | None = None
        self._request_queued = False

    @QtCore.Slot(str)
    def open(self, path):
        self._open([path])

    @QtCore.Slot("QStringList")
    def openConcat(self, paths):
        self._open(list(paths))

    def _open(self, paths: list[str]):
        if cv2 is None:
            self.opened.emit(False, "OpenCV (cv2) not installed.")
            return
        try:
            if self.cap is not None:
                self.cap.release()
            if len(paths) == 1:
                self.cap = cv2.VideoCapture(paths[0])
            else:
                self.cap = MultiFileVideoCapture(paths)
            self.cache.clear()
            self._requested_idx = None
            self._request_queued = False
            ok = bool(self.cap.isOpened())
            msg = "" if ok else f"Failed to open: {paths}"
            self.opened.emit(ok, msg)
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
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QtGui.QImage(
                    rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888
                ).copy()
                self.cache[idx] = qimg
                if len(self.cache) > self.cache_frames:
                    self.cache.popitem(last=False)

        # Skip emitting stale frames when a newer request is already pending.
        if qimg is not None and self._requested_idx is None:
            self.frameReady.emit(idx, qimg)

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
