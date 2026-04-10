#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loupe: multi-trace viewer with windowed rendering, video scrubbing,
page hotkeys, and cross-page draggable labeling.

pip install PySide6 pyqtgraph opencv-python numpy
"""

import os, sys, glob, csv, math, argparse, json
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

try:
    import cv2
except Exception:
    cv2 = None


# ---------------- State Definitions Loader ----------------


def load_state_definitions():
    """Load keymap and label_colors from state_definitions.json.

    Looks for the file in the same directory as this script.
    Falls back to defaults if file not found or invalid.
    """
    # Default definitions (fallback)
    default_keymap = {
        "w": "Wake",
        "q": "Quiet-Wake",
        "b": "Brief-Arousal",
        "2": "NREM-light",
        "1": "NREM",
        "r": "REM",
        "a": "Artifact",
        "t": "Transition-to-REM",
        "u": "unclear",
        "o": "ON",
        "f": "OFF",
        "s": "spindle",
    }
    default_label_colors = {
        "Wake": (0, 209, 40, 60),
        "Quiet-Wake": (79, 255, 168, 60),
        "Brief-Arousal": (188, 255, 45, 60),
        "NREM-light": (79, 247, 255, 60),
        "NREM": (41, 30, 255, 60),
        "Transition-to-REM": (255, 101, 224, 60),
        "REM": (255, 30, 145, 60),
        "Artifact": (255, 0, 0, 80),
        "unclear": (242, 255, 41, 50),
        "ITI": (79, 247, 255, 60),
        "Go": (0, 209, 40, 60),
        "NoGo": (255, 0, 0, 80),
        "Timeout": (255, 30, 145, 60),
        "ON": (0, 209, 40, 60),
        "OFF": (255, 0, 0, 80),
        "spindle": (79, 247, 255, 60),
    }

    # Try to load from JSON file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "state_definitions.json")

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            keymap = data.get("keymap", default_keymap)
            label_colors_raw = data.get("label_colors", {})

            # Convert JSON arrays to tuples for colors
            label_colors = {}
            for name, color in label_colors_raw.items():
                if isinstance(color, list) and len(color) >= 3:
                    label_colors[name] = tuple(
                        color[:4] if len(color) >= 4 else color + [255]
                    )
                else:
                    label_colors[name] = default_label_colors.get(
                        name, (150, 150, 150, 80)
                    )

            # Merge with defaults to ensure all keys exist
            for k, v in default_label_colors.items():
                if k not in label_colors:
                    label_colors[k] = v

            return keymap, label_colors
        except Exception as e:
            print(f"Warning: Could not load state_definitions.json: {e}")
            print("Using default state definitions.")

    return default_keymap, default_label_colors


# ---------------- Data containers ----------------


@dataclass
class Series:
    name: str
    t: np.ndarray  # seconds, monotonic
    y: np.ndarray


@dataclass
class MatrixSeries:
    """Holds data for a matrix/raster subplot."""

    name: str
    timestamps: np.ndarray  # 1D array of event times (seconds)
    yvals: np.ndarray  # 1D array of integer row indices (0 to N_rows-1)
    alphas: np.ndarray  # 1D array of alpha values (0.0 to 1.0)
    color: tuple  # (R, G, B) base color for events
    n_rows: int  # number of unique rows (max(yvals) + 1)


LabelKey = tuple[float, float, str]


@dataclass
class LabelVisualBundle:
    """Graphics items used to display a single labelled interval in plot scenes."""

    plot_regions: list[tuple[int, pg.LinearRegionItem]]
    matrix_regions: list[tuple[int, pg.LinearRegionItem]]
    dense_regions: list[tuple[int, pg.LinearRegionItem]]
    hypnogram_region: pg.LinearRegionItem | None


MATRIX_ALPHA_LEVEL_COUNT = 11


@dataclass
class DenseGroup:
    """A group of traces rendered on a single dense plot (EEG-style)."""

    name: str
    series: list[Series]
    trace_labels: list[str]
    order_values: np.ndarray | None = None
    descending: bool = False
    gain: float = 1.0
    step: int = 1
    traces_per_page: int | None = None
    hidden_traces: set[int] = field(default_factory=set)


# ---------------- Utilities ----------------


def nice_time_range(t_arrays):
    vals = [(np.nanmin(t), np.nanmax(t)) for t in t_arrays if t is not None and len(t)]
    return (
        (0.0, 1.0)
        if not vals
        else (float(min(v[0] for v in vals)), float(max(v[1] for v in vals)))
    )


def resolve_low_profile_x(
    low_profile_x: bool | None, total_subplots: int
) -> bool:
    """Resolve the effective low-profile X-axis mode.

    When the caller does not specify a preference, Loupe defaults to the
    low-profile layout once three or more total subplots are loaded.
    """
    if low_profile_x is not None:
        return bool(low_profile_x)
    return total_subplots >= 3


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def find_nearest_frame(frame_times, t):
    if frame_times is None or len(frame_times) == 0:
        return 0
    i = int(np.searchsorted(frame_times, t, "left"))
    if i <= 0:
        return 0
    if i >= len(frame_times):
        return len(frame_times) - 1
    return i - 1 if abs(t - frame_times[i - 1]) <= abs(frame_times[i] - t) else i


def next_pow_two(n: int) -> int:
    n = int(max(1, n))
    return 1 << (n - 1).bit_length()


# ---------------- Peak-preserving window decimator ----------------


def segment_for_window(t, y, t0, t1, max_pts=4000):
    """
    Return (tx, yx) for the [t0, t1] window.
    Uses peak-preserving bin min/max if the window contains too many samples.
    """
    if t1 <= t0:
        return np.empty(0), np.empty(0)

    # 1) slice to window (with 1-sample guard on each side)
    i0 = max(0, np.searchsorted(t, t0) - 1)
    i1 = min(len(t), np.searchsorted(t, t1) + 1)
    ts = t[i0:i1]
    ys = y[i0:i1]
    n = len(ts)
    if n <= 2:
        return ts, ys

    # 2) if already small, return as-is
    if n <= max_pts:
        return ts, ys

    # 3) bin across time into ~max_pts/2 bins; emit min/max per bin
    bins = max(1, max_pts // 2)
    # bin edges across [t0, t1]
    edges = np.linspace(t0, t1, bins + 1)
    # assign each timestamp to a bin index (0..bins-1)
    bi = np.clip(np.digitize(ts, edges) - 1, 0, bins - 1)

    # Timestamps are monotonic, so bin ids are already ordered.
    starts = np.searchsorted(bi, np.arange(bins), "left")
    ends = np.searchsorted(bi, np.arange(bins), "right")

    nonempty = starts < ends

    # Per-bin min/max via reduceat (vectorised, no Python loop).
    # np.fmin/fmax ignore NaN, matching the original np.nanmin/nanmax behaviour.
    ymins = np.fmin.reduceat(ys, starts)
    ymaxs = np.fmax.reduceat(ys, starts)

    # Midpoint times: middle sample index in each bin.
    mid_indices = np.clip((starts + ends) // 2, 0, n - 1)
    tmids = ts[mid_indices]

    # Empty bins → NaN values at edge midpoints.
    empty = ~nonempty
    if np.any(empty):
        ymins[empty] = np.nan
        ymaxs[empty] = np.nan
        tmids[empty] = 0.5 * (edges[:-1][empty] + edges[1:][empty])

    # Interleave min/max pairs at each time.
    out_t = np.repeat(tmids, 2)
    out_y = np.empty(2 * bins, dtype=float)
    out_y[0::2] = ymins
    out_y[1::2] = ymaxs

    return out_t, out_y


# ---------------- Custom UI Components ----------------


class SelectableViewBox(pg.ViewBox):
    sigDragStart = QtCore.Signal(float)
    sigDragUpdate = QtCore.Signal(float)
    sigDragFinish = QtCore.Signal(float)
    sigWheelScrolled = QtCore.Signal(int)
    sigWheelSmoothScrolled = QtCore.Signal(int)
    sigWheelCursorScrolled = QtCore.Signal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # *** FIX 1: Explicitly disable default mouse pan/zoom behavior ***
        # This allows our custom event handlers to take full control.
        self.setMouseEnabled(x=False, y=False)
        self._drag = False

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            x = float(self.mapSceneToView(ev.scenePos()).x())
            if ev.isStart():
                self._drag = True
                self.sigDragStart.emit(x)
                ev.accept()
                return
            elif ev.isFinish():
                if self._drag:
                    self.sigDragFinish.emit(x)
                self._drag = False
                ev.accept()
                return
            else:
                if self._drag:
                    self.sigDragUpdate.emit(x)
                ev.accept()
                return
        # Do not call super, to prevent default drag (pan) behavior

    def wheelEvent(self, ev, axis=None):
        dy = 0
        if hasattr(ev, "delta"):
            try:
                dy = ev.delta()
            except Exception:
                dy = 0
        else:
            try:
                ad = ev.angleDelta()
                dy = ad.y() if hasattr(ad, "y") else 0
            except Exception:
                dy = 0
        direction = 1 if dy > 0 else -1
        # Use Shift+wheel for smooth scrolling; otherwise page
        try:
            mods = QtWidgets.QApplication.keyboardModifiers()
        except Exception:
            mods = QtCore.Qt.KeyboardModifier.NoModifier
        # Ctrl: cursor scroll within window (like dragging cursor slider)
        if mods & QtCore.Qt.KeyboardModifier.ControlModifier:
            self.sigWheelCursorScrolled.emit(int(dy))
        # Shift: smooth scroll the window
        elif mods & QtCore.Qt.KeyboardModifier.ShiftModifier:
            self.sigWheelSmoothScrolled.emit(direction)
        else:
            self.sigWheelScrolled.emit(direction)
        ev.accept()


class DenseViewBox(SelectableViewBox):
    """ViewBox for dense plots — Alt+wheel gain, Shift+Alt+wheel vertical scroll."""

    sigWheelGainAdjust = QtCore.Signal(int)
    sigWheelVerticalSmooth = QtCore.Signal(int)

    def wheelEvent(self, ev, axis=None):
        try:
            mods = QtWidgets.QApplication.keyboardModifiers()
        except Exception:
            mods = QtCore.Qt.KeyboardModifier.NoModifier
        alt = bool(mods & QtCore.Qt.KeyboardModifier.AltModifier)
        shift = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
        if alt:
            dy = 0
            if hasattr(ev, "delta"):
                try:
                    dy = ev.delta()
                except Exception:
                    dy = 0
            else:
                try:
                    ad = ev.angleDelta()
                    dy = ad.y() if hasattr(ad, "y") else 0
                except Exception:
                    dy = 0
            direction = 1 if dy > 0 else -1
            if shift:
                self.sigWheelVerticalSmooth.emit(direction)
            else:
                self.sigWheelGainAdjust.emit(direction)
            ev.accept()
        else:
            super().wheelEvent(ev, axis)


# *** FIX 2: Create a PlotItem that signals when the mouse enters/leaves it ***
class HoverablePlotItem(pg.PlotItem):
    sigHovered = QtCore.Signal(
        object, bool
    )  # Emits self, True on enter, False on leave

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Required to receive hover events
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, ev):
        self.sigHovered.emit(self, True)
        super().hoverEnterEvent(ev)

    def hoverLeaveEvent(self, ev):
        self.sigHovered.emit(self, False)
        super().hoverLeaveEvent(ev)


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
        if cv2 is None:
            self.opened.emit(False, "OpenCV (cv2) not installed.")
            return
        try:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(path)
            self.cache.clear()
            self._requested_idx = None
            self._request_queued = False
            ok = bool(self.cap.isOpened())
            self.opened.emit(ok, "" if ok else f"Failed to open: {path}")
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


# ---------------- Label Summary Panel ----------------


class StateComboDelegate(QtWidgets.QStyledItemDelegate):
    """Dropdown delegate for the State column in the label summary table."""

    def __init__(self, states: list[str], parent=None):
        super().__init__(parent)
        self._states = list(states)

    def set_states(self, states: list[str]):
        self._states = list(states)

    def createEditor(self, parent, option, index):
        combo = QtWidgets.QComboBox(parent)
        for state in self._states:
            combo.addItem(state)
        return combo

    def setEditorData(self, editor, index):
        current = index.data()
        idx = editor.findText(current or "")
        if idx >= 0:
            editor.setCurrentIndex(idx)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText(), QtCore.Qt.ItemDataRole.EditRole)


class LabelSummaryBarWidget(QtWidgets.QWidget):
    """Horizontal stacked color bar showing per-state labeling fractions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(24)
        self.segments: list[tuple[float, tuple, str]] = []

    def set_data(self, segments: list[tuple[float, tuple, str]]):
        """Set bar segments: list of (fraction, rgba_tuple, label_text)."""
        self.segments = segments
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        try:
            w, h = self.width(), self.height()
            if not self.segments:
                painter.fillRect(0, 0, w, h, QtGui.QColor(40, 40, 40))
                return
            x = 0.0
            for frac, color, text in self.segments:
                seg_w = frac * w
                int_x = int(round(x))
                int_w = int(round(x + seg_w)) - int_x
                painter.fillRect(int_x, 0, int_w, h, QtGui.QColor(color[0], color[1], color[2]))
                if int_w > 50:
                    painter.setPen(QtGui.QColor(255, 255, 255))
                    font = painter.font()
                    font.setPointSize(8)
                    painter.setFont(font)
                    painter.drawText(
                        int_x, 0, int_w, h,
                        QtCore.Qt.AlignmentFlag.AlignCenter,
                        text,
                    )
                x += seg_w
        finally:
            painter.end()


class LabelSummaryWidget(QtWidgets.QWidget):
    """Inline panel showing an editable table of all scored labels with summary stats."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._refreshing = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # Table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Start", "End", "Duration", "State", "Note"])
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
        )
        self.table.setStyleSheet(
            "QTableWidget { background-color: #1a1a1a; color: #ddd; gridline-color: #444; }"
            "QHeaderView::section { background-color: #333; color: #ddd; border: 1px solid #444; padding: 2px; }"
        )
        # State column uses a dropdown delegate
        self._state_delegate = StateComboDelegate(
            sorted(main_window.label_colors.keys())
        )
        self.table.setItemDelegateForColumn(3, self._state_delegate)

        self.table.cellChanged.connect(self._on_cell_changed)
        layout.addWidget(self.table, 1)

        # Summary text
        self.summary_label = QtWidgets.QLabel("Fraction of total recording labelled: 0.0%")
        self.summary_label.setStyleSheet("color: #ccc; padding: 2px;")
        layout.addWidget(self.summary_label)

        bar_header = QtWidgets.QLabel("Labelling by state:")
        bar_header.setStyleSheet("color: #ccc; font-weight: bold; padding: 0px 2px;")
        layout.addWidget(bar_header)

        # Color bar
        self.bar_widget = LabelSummaryBarWidget()
        layout.addWidget(self.bar_widget)

    def refresh(self):
        """Repopulate table and summary from main_window.labels."""
        if self._refreshing:
            return
        self._refreshing = True
        try:
            self.table.blockSignals(True)
            self.table.setRowCount(0)
            for i, lab in enumerate(self.main_window.labels):
                self.table.insertRow(i)
                start, end = float(lab["start"]), float(lab["end"])
                duration = end - start
                key = (start, end)
                note = self.main_window.label_notes.get(key, "")

                self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(f"{start:.3f}"))
                self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{end:.3f}"))
                self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{duration:.3f}"))
                self.table.setItem(i, 3, QtWidgets.QTableWidgetItem(lab["label"]))
                self.table.setItem(i, 4, QtWidgets.QTableWidgetItem(note))

                # Store label index for mapping back
                self.table.item(i, 0).setData(QtCore.Qt.ItemDataRole.UserRole, i)

            self.table.resizeColumnsToContents()
            self.table.blockSignals(False)
            self._update_summary()
            # Update known states for dropdown delegate
            self._state_delegate.set_states(sorted(self.main_window.label_colors.keys()))
        finally:
            self._refreshing = False

    def _update_summary(self):
        """Recompute summary stats and update the bar widget."""
        total_recording = self.main_window.t_global_max - self.main_window.t_global_min
        if total_recording <= 0:
            self.summary_label.setText("No recording loaded")
            self.bar_widget.set_data([])
            return

        total_labelled = sum(
            float(lab["end"]) - float(lab["start"])
            for lab in self.main_window.labels
        )
        pct = (total_labelled / total_recording) * 100.0
        self.summary_label.setText(
            f"Fraction of total recording labelled: {pct:.1f}%"
        )

        # Per-state breakdown
        state_durations: dict[str, float] = {}
        for lab in self.main_window.labels:
            dur = float(lab["end"]) - float(lab["start"])
            state_durations[lab["label"]] = state_durations.get(lab["label"], 0.0) + dur

        segments = []
        for state, dur in sorted(state_durations.items(), key=lambda x: -x[1]):
            frac = dur / total_recording
            color = self.main_window.label_colors.get(state, (150, 150, 150, 80))
            bar_color = (color[0], color[1], color[2], 255)
            text = f"{state} {frac * 100:.0f}%"
            segments.append((frac, bar_color, text))
        self.bar_widget.set_data(segments)

    def _migrate_note_key(self, old_key: tuple, new_key: tuple):
        """Move a note entry from old (start, end) key to new key."""
        if old_key in self.main_window.label_notes:
            note = self.main_window.label_notes.pop(old_key)
            self.main_window.label_notes[new_key] = note

    def _on_cell_changed(self, row: int, col: int):
        """Handle inline cell edits — validate and propagate to labels."""
        if self._refreshing:
            return
        labels = self.main_window.labels
        if row < 0 or row >= len(labels):
            return

        lab = labels[row]
        old_start, old_end = float(lab["start"]), float(lab["end"])
        old_key = (old_start, old_end)
        item = self.table.item(row, col)
        if item is None:
            return
        new_text = item.text().strip()

        try:
            if col == 0:  # Start
                new_start = float(new_text)
                if new_start >= old_end:
                    raise ValueError("Start must be less than End")
                if new_start < 0:
                    raise ValueError("Start must be non-negative")
                self._migrate_note_key(old_key, (new_start, old_end))
                lab["start"] = new_start

            elif col == 1:  # End
                new_end = float(new_text)
                if new_end <= old_start:
                    raise ValueError("End must be greater than Start")
                self._migrate_note_key(old_key, (old_start, new_end))
                lab["end"] = new_end

            elif col == 2:  # Duration
                new_dur = float(new_text)
                if new_dur <= 0:
                    raise ValueError("Duration must be positive")
                new_end = old_start + new_dur
                self._migrate_note_key(old_key, (old_start, new_end))
                lab["end"] = new_end

            elif col == 3:  # State
                if not new_text:
                    raise ValueError("State cannot be empty")
                lab["label"] = new_text

            elif col == 4:  # Note
                new_key = (float(lab["start"]), float(lab["end"]))
                if new_text:
                    self.main_window.label_notes[new_key] = new_text
                elif new_key in self.main_window.label_notes:
                    del self.main_window.label_notes[new_key]
                # Note edits don't need visual redraw of label regions
                return

        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Invalid edit", str(e))
            self.refresh()
            return

        # Re-sort, merge, and sync visuals (which also triggers refresh)
        self.main_window._finalize_label_change()


class YAxisControlsDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Y-Axis Controls")
        self.setModal(False)

        self.main_window = parent

        main_layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        main_layout.addLayout(form_layout)

        self.controls = []
        for idx, s in enumerate(self.main_window.series):
            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            auto_check = QtWidgets.QCheckBox("Auto")
            auto_check.setChecked(
                self.main_window.plots[idx].getViewBox().autoRangeEnabled()[1]
            )

            min_spin = QtWidgets.QDoubleSpinBox()
            max_spin = QtWidgets.QDoubleSpinBox()

            for spin in (min_spin, max_spin):
                spin.setDecimals(3)
                spin.setRange(-1e12, 1e12)
                spin.setEnabled(not auto_check.isChecked())

            current_y_range = self.main_window.plots[idx].getViewBox().viewRange()[1]
            min_spin.setValue(current_y_range[0])
            max_spin.setValue(current_y_range[1])

            auto_check.stateChanged.connect(
                lambda state, i=idx, ac=auto_check, mn=min_spin, mx=max_spin: self.apply_y_range(
                    i, ac, mn, mx
                )
            )
            min_spin.editingFinished.connect(
                lambda i=idx, ac=auto_check, mn=min_spin, mx=max_spin: self.apply_y_range(
                    i, ac, mn, mx
                )
            )
            max_spin.editingFinished.connect(
                lambda i=idx, ac=auto_check, mn=min_spin, mx=max_spin: self.apply_y_range(
                    i, ac, mn, mx
                )
            )

            row_layout.addStretch(1)
            row_layout.addWidget(auto_check)
            row_layout.addWidget(QtWidgets.QLabel("Min"))
            row_layout.addWidget(min_spin)
            row_layout.addWidget(QtWidgets.QLabel("Max"))
            row_layout.addWidget(max_spin)

            form_layout.addRow(s.name, row_widget)
            self.controls.append((auto_check, min_spin, max_spin))

    def apply_y_range(self, plot_index, auto_check, min_spin, max_spin):
        plot_item = self.main_window.plots[plot_index]
        if auto_check.isChecked():
            plot_item.enableAutoRange("y", True)
            min_spin.setEnabled(False)
            max_spin.setEnabled(False)
        else:
            plot_item.enableAutoRange("y", False)
            lo, hi = min_spin.value(), max_spin.value()
            if hi <= lo:
                hi = lo + 1e-6
            plot_item.setYRange(lo, hi, padding=0.05)
            min_spin.setEnabled(True)
            max_spin.setEnabled(True)


class DenseViewControlsDialog(QtWidgets.QDialog):
    """Non-modal dialog for adjusting dense view gain, spacing, and step."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Dense View Controls")
        self.setModal(False)
        self.main_window = parent

        main_layout = QtWidgets.QVBoxLayout(self)
        self._group_widgets: list[dict] = []

        for gi, group in enumerate(self.main_window.dense_groups):
            grp_box = QtWidgets.QGroupBox(group.name)
            grp_layout = QtWidgets.QFormLayout(grp_box)

            # Gain slider + spinbox
            gain_layout = QtWidgets.QHBoxLayout()
            gain_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            gain_slider.setRange(-200, 200)  # log10(gain) * 100
            gain_slider.setValue(int(math.log10(max(1e-6, group.gain)) * 100))
            gain_spin = QtWidgets.QDoubleSpinBox()
            gain_spin.setRange(0.01, 100.0)
            gain_spin.setDecimals(2)
            gain_spin.setValue(group.gain)
            gain_spin.setSingleStep(0.1)
            gain_layout.addWidget(gain_slider, 3)
            gain_layout.addWidget(gain_spin, 1)
            grp_layout.addRow("Gain:", gain_layout)

            # Step spinbox
            step_spin = QtWidgets.QSpinBox()
            step_spin.setRange(1, max(1, len(group.series)))
            step_spin.setValue(group.step)
            grp_layout.addRow("Step (nth trace):", step_spin)

            # Traces per page spinbox (0 = all)
            tpp_spin = QtWidgets.QSpinBox()
            tpp_spin.setRange(0, max(1, len(group.series)))
            tpp_spin.setSpecialValueText("All")
            tpp_spin.setValue(group.traces_per_page or 0)
            grp_layout.addRow("Traces per page:", tpp_spin)

            main_layout.addWidget(grp_box)

            widgets = {
                "gain_slider": gain_slider,
                "gain_spin": gain_spin,
                "step_spin": step_spin,
                "tpp_spin": tpp_spin,
            }
            self._group_widgets.append(widgets)

            # Connect signals
            gain_slider.valueChanged.connect(
                lambda val, g=gi, sp=gain_spin: self._on_gain_slider(g, val, sp)
            )
            gain_spin.valueChanged.connect(
                lambda val, g=gi, sl=gain_slider: self._on_gain_spin(g, val, sl)
            )
            step_spin.valueChanged.connect(
                lambda val, g=gi: self._on_step_changed(g, val)
            )

            tpp_spin.valueChanged.connect(
                lambda val, g=gi: self._on_tpp_changed(g, val)
            )

    def _on_gain_slider(self, gi: int, slider_val: int, spin: QtWidgets.QDoubleSpinBox):
        gain = 10 ** (slider_val / 100.0)
        spin.blockSignals(True)
        spin.setValue(gain)
        spin.blockSignals(False)
        self.main_window.dense_groups[gi].gain = gain
        self.main_window._refresh_dense_curves()

    def _on_gain_spin(self, gi: int, val: float, slider: QtWidgets.QSlider):
        val = max(0.01, val)
        slider.blockSignals(True)
        slider.setValue(int(math.log10(val) * 100))
        slider.blockSignals(False)
        self.main_window.dense_groups[gi].gain = val
        self.main_window._refresh_dense_curves()

    def _on_step_changed(self, gi: int, val: int):
        self.main_window.dense_groups[gi].step = val
        self.main_window._rebuild_dense_curves(gi)
        self.main_window._refresh_dense_curves()

    def _on_tpp_changed(self, gi: int, val: int):
        group = self.main_window.dense_groups[gi]
        group.traces_per_page = val if val > 0 else None
        # Reset Y-range to show the requested page size
        offsets = self.main_window._dense_offsets(gi)
        if len(offsets) > 0:
            plt = self.main_window.dense_plots[gi]
            margin = self.main_window._dense_offset_margin(offsets)
            tpp = group.traces_per_page
            if tpp is not None and tpp < len(offsets):
                page_max = float(offsets[min(tpp, len(offsets)) - 1])
                plt.setYRange(float(offsets[0]) - margin, page_max + margin, padding=0)
            else:
                plt.setYRange(
                    float(offsets.min()) - margin,
                    float(offsets.max()) + margin,
                    padding=0,
                )



# ---------------- Main window ----------------


class LoupeApp(QtWidgets.QMainWindow):
    def __init__(
        self,
        data_dir=None,
        data_files=None,
        colors=None,
        video_path=None,
        frame_times_path=None,
        video2_path=None,
        frame_times2_path=None,
        video3_path=None,
        frame_times3_path=None,
        fixed_scale=True,
        low_profile_x: bool | None = None,
        window_len: float = 10.0,
        # Matrix viewer arguments
        matrix_timestamps=None,
        matrix_yvals=None,
        alpha_vals=None,
        matrix_colors=None,
        # xarray series (pre-converted list[Series])
        xr_series=None,
        # Pre-converted MatrixSeries from df_loader
        matrix_series_list=None,
        # Overlay mode
        overlay_groups=None,
        overlay_colors=None,
        # Dense mode
        dense_groups=None,
    ):
        super().__init__()
        self.setWindowTitle("Loupe — Multi-Trace + Video + Labeling")
        self.resize(1400, 900)

        pg.setConfigOptions(
            antialias=False, useOpenGL=True, background="k", foreground="w"
        )

        # Data & plots
        self.series: list[Series] = []
        self.t_global_min = 0.0
        self.t_global_max = 1.0
        self.plots: list[pg.PlotItem] = []
        self.curves: list[pg.PlotDataItem] = []
        self.plot_cur_lines: list[pg.InfiniteLine] = []
        self.plot_sel_regions: list[pg.LinearRegionItem] = []
        self.hovered_plot = None  # *** FIX 2: Track which plot is hovered ***

        # Overlay mode
        self.overlay_mode: bool = False
        self.overlay_groups: list = []  # list[OverlayGroup]
        self.overlay_colors: list[tuple] = []
        self._plot_to_curves: list[list[pg.PlotDataItem]] = []
        self._plot_to_series: list[list[int]] = []

        # Dense mode (EEG-style stacked traces on a single axis)
        self.dense_groups: list[DenseGroup] = []
        self.dense_plots: list[pg.PlotItem] = []
        self.dense_curves: list[list[pg.PlotCurveItem]] = []
        self.dense_cur_lines: list[pg.InfiniteLine] = []
        self.dense_sel_regions: list[pg.LinearRegionItem] = []
        self.dense_label_regions: list[list[pg.LinearRegionItem]] = []
        self.dense_height_factors: list[float] = []
        self.dense_visible: list[bool] = []
        self._dense_means: list[list[float]] = []

        # Matrix viewer data and plots
        self.matrix_series: list[MatrixSeries] = []
        self.matrix_plots: list[pg.PlotItem] = []
        self.matrix_items: list[pg.ScatterPlotItem | None] = []
        self.matrix_cur_lines: list[pg.InfiniteLine] = []
        self.matrix_sel_regions: list[pg.LinearRegionItem] = []
        self._matrix_line_items: list[list[pg.PlotDataItem]] = []
        self._matrix_pens: list[list[QtGui.QPen]] = []
        # Matrix rendering settings
        self.matrix_event_height = (
            0.1  # distance from center in each direction (0.1-0.5)
        )
        self.matrix_event_thickness = 1  # pen width in pixels
        self.scale_matrix_proportionally = False  # toggled via View menu
        self.matrix_share_boost = (
            0  # adjustment to matrix share (each unit = 5%, no bounds)
        )
        self.matrix_brightness = (
            1.0  # brightness multiplier for alpha values (0.2 to 3.0)
        )
        # Custom height factors for individual plot height control (1.0 = default)
        self.plot_height_factors: list[float] = []  # one per time series plot
        self.matrix_height_factors: list[float] = []  # one per matrix plot
        # Visibility flags for matrix plots (similar to trace_visible for time series)
        self.matrix_visible: list[bool] = []
        # Plot order: list of (type, index) tuples, e.g., [("ts", 0), ("ts", 1), ("matrix", 0)]
        # None means use default order (all ts first, then all matrix)
        self.subplot_order: list[tuple] | None = None

        # Rendering budget (per plot)
        self.max_pts_per_plot = 4000

        # Window/cursor & labels
        self.window_len = float(window_len)
        self.window_start = 0.0
        self.cursor_time = 0.0
        # Vertical paging for stacked-subplots view
        self.trace_height_px = 120  # pixels per stacked subplot

        # Load state definitions from external config file
        self.keymap, self.label_colors = load_state_definitions()

        # Labels and notes
        self.labels = []
        self.label_notes = {}  # Maps (start, end) tuple to note string
        self.label_history = []  # Track order of epoch creation for "most recent"
        self._label_visuals: dict[LabelKey, LabelVisualBundle] = {}
        self._hypnogram_label_visuals: dict[LabelKey, pg.LinearRegionItem] = {}
        self._label_keys_in_order: list[LabelKey] = []
        self._label_starts = np.empty(0, dtype=float)
        self._label_ends = np.empty(0, dtype=float)
        self._select_start = None
        self._select_end = None
        self._is_zoom_drag = False
        self.fixed_scale = bool(fixed_scale)
        self._low_profile_x_preference = low_profile_x
        self.low_profile_x = resolve_low_profile_x(
            self._low_profile_x_preference, total_subplots=0
        )

        # Video
        self.video_frame_times = None
        self._video_is_open = False
        self._video_thread = QtCore.QThread(self)
        self._video_worker = VideoWorker(cache_frames=120)
        self._video_worker.moveToThread(self._video_thread)
        self.last_video_pixmap = None
        self._video_requested_frame_idx: int | None = None

        # Optional second video
        self.video2_frame_times = None
        self._video2_is_open = False
        self._video2_thread = QtCore.QThread(self)
        self._video2_worker = VideoWorker(cache_frames=120)
        self._video2_worker.moveToThread(self._video2_thread)
        self.last_video2_pixmap = None
        self._video2_requested_frame_idx: int | None = None

        # Optional third video
        self.video3_frame_times = None
        self._video3_is_open = False
        self._video3_thread = QtCore.QThread(self)
        self._video3_worker = VideoWorker(cache_frames=120)
        self._video3_worker.moveToThread(self._video3_thread)
        self.last_video3_pixmap = None
        self._video3_requested_frame_idx: int | None = None

        # Playback
        self.is_playing = False
        self.playback_timer = QtCore.QTimer(self)
        self.playback_timer.timeout.connect(self._advance_playback_frame)
        self.playback_elapsed_timer = QtCore.QElapsedTimer()

        # Smooth scroll settings (fraction of window per wheel step)
        self.smooth_scroll_fraction = 0.10
        self._deferred_view_refresh_timer = QtCore.QTimer(self)
        self._deferred_view_refresh_timer.setSingleShot(True)
        self._deferred_view_refresh_timer.timeout.connect(
            self._flush_deferred_view_refresh
        )
        self._deferred_view_refresh_needs_nav_slider = False
        # Playback speed (1.0 = real time)
        self.playback_speed = 1.0
        # Which video to use for frame-by-frame stepping (1, 2, or 3)
        self.frame_step_source = 1

        # Hypnogram overview
        self.hypnogram_widget = None
        self.hypnogram_plot = None
        self.hypnogram_view_region = None
        self.hypnogram_zoomed = False
        self.hypnogram_zoom_padding = 30.0

        # Right panel layout references and video stretch defaults (used in _build_ui)
        self.right_layout = None
        self.videos_layout = None
        self.videos_widget = None
        self.video1_stretch = 3
        self.video2_stretch = 2
        self.video3_stretch = 2

        self._build_ui()

        self.y_axis_dialog = None

        self._video_worker.frameReady.connect(self._on_frame_ready)
        self._video_worker.opened.connect(self._on_video_opened)
        self._video2_worker.frameReady.connect(self._on_frame2_ready)
        self._video2_worker.opened.connect(self._on_video2_opened)
        self._video3_worker.frameReady.connect(self._on_frame3_ready)
        self._video3_worker.opened.connect(self._on_video3_opened)

        self._video_thread.start()
        self._video2_thread.start()
        self._video3_thread.start()
        # Store dense groups early (before set_series triggers plot creation)
        if dense_groups:
            self.dense_groups = dense_groups
            self.dense_height_factors = [1.0] * len(dense_groups)
            self.dense_visible = [True] * len(dense_groups)
            # Cache per-trace means for display transform
            self._dense_means = [
                [float(np.nanmean(s.y)) for s in g.series]
                for g in dense_groups
            ]

        # Prefer overlay groups, then xarray series, then explicit file list, then dir
        if overlay_groups:
            self.set_overlay_series(overlay_groups, colors=overlay_colors)
        elif xr_series or dense_groups:
            self.set_series(xr_series or [], colors=colors)
        elif data_files:
            # Allow flexible formats: list[str], comma-separated strings, or "[a,b]".
            def _normalize_file_list(df):
                if not df:
                    return []
                items = [df] if isinstance(df, str) else list(df)
                out = []
                for it in items:
                    s = (it or "").strip()
                    # Strip surrounding list brackets if present
                    if s.startswith("[") and s.endswith("]"):
                        s = s[1:-1]
                    parts = s.split(",") if "," in s else [s]
                    for p in parts:
                        q = p.strip().strip('"').strip("'")
                        # Remove a lingering trailing comma if passed as a token like "file.npy,"
                        if q.endswith(","):
                            q = q[:-1].rstrip()
                        if q:
                            out.append(q)
                return out

            def _normalize_list(raw_list):
                if not raw_list:
                    return []
                items = [raw_list] if isinstance(raw_list, str) else list(raw_list)
                out = []
                for it in items:
                    s = (it or "").strip()
                    if s.startswith("[") and s.endswith("]"):
                        s = s[1:-1]
                    parts = s.split(",") if "," in s else [s]
                    for p in parts:
                        q = p.strip().strip('"').strip("'")
                        if q.endswith(","):
                            q = q[:-1].rstrip()
                        if q:
                            out.append(q)
                return out

            paths = _normalize_file_list(data_files)
            color_list = _normalize_list(colors) if colors else None
            self._load_series_from_files(paths, colors=color_list)
        elif data_dir:
            self._load_series_from_dir(data_dir)
        if video_path and frame_times_path:
            self._load_video_data(video_path, frame_times_path)
        if video2_path and frame_times2_path:
            self._load_video2_data(video2_path, frame_times2_path)
        if video3_path and frame_times3_path:
            self._load_video3_data(video3_path, frame_times3_path)

        # Load matrix viewer data if provided
        if matrix_timestamps and matrix_yvals:
            self._load_matrix_data(
                matrix_timestamps, matrix_yvals, alpha_vals, matrix_colors
            )

        # Load pre-converted MatrixSeries (from df_loader)
        if matrix_series_list:
            self.matrix_series = matrix_series_list
            self._refresh_low_profile_x()
            self.matrix_height_factors = [1.0] * len(matrix_series_list)
            self.matrix_visible = [True] * len(matrix_series_list)
            self.subplot_order = None
            self._update_status(
                f"Loaded {len(matrix_series_list)} matrix series from DataFrame."
            )
            if self.series:
                self._rebuild_all_plots()
            else:
                self._update_time_range_from_matrix()
                self._create_matrix_only_plots()

    def _refresh_low_profile_x(self) -> None:
        """Update low-profile X mode from explicit preference or subplot count."""
        total_subplots = len(self.series) + len(self.matrix_series) + len(self.dense_groups)
        self.low_profile_x = resolve_low_profile_x(
            self._low_profile_x_preference, total_subplots
        )

    def eventFilter(self, obj, ev):
        try:
            if ev.type() == QtCore.QEvent.Type.Resize:
                if obj is self.video_label:
                    self._rescale_video_frame()
                elif obj is self.video2_label:
                    self._rescale_video2_frame()
        except Exception:
            pass
        return super().eventFilter(obj, ev)

    # ---------- UI ----------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QHBoxLayout()
        v.addLayout(top)
        top.addWidget(QtWidgets.QLabel("Window (s):"))
        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(0.1, 3600.0)
        self.window_spin.setDecimals(2)
        self.window_spin.setValue(self.window_len)
        self.window_spin.valueChanged.connect(self._on_window_len_changed)
        top.addWidget(self.window_spin)
        top.addSpacing(20)
        top.addWidget(QtWidgets.QLabel("Navigate:"))
        self.nav_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.nav_slider.setRange(0, 10000)
        self.nav_slider.valueChanged.connect(self._on_nav_slider_changed)
        top.addWidget(self.nav_slider, 1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        v.addWidget(splitter, 1)
        self.splitter = splitter

        # left plots (wrapped in scroll area for vertical paging)
        left = QtWidgets.QWidget()
        leftl = QtWidgets.QVBoxLayout(left)
        leftl.setContentsMargins(0, 0, 0, 0)
        self.plot_area = pg.GraphicsLayoutWidget()
        self.plot_scroll_area = QtWidgets.QScrollArea()
        self.plot_scroll_area.setWidgetResizable(True)
        self.plot_scroll_area.setWidget(self.plot_area)
        self.plot_scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        # Prevent scroll area from stealing PageUp/PageDown key events
        self.plot_scroll_area.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.plot_scroll_area.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.plot_scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        # Horizontal layout: scroll area + optional dense vertical scrollbar
        plot_hbox = QtWidgets.QHBoxLayout()
        plot_hbox.setContentsMargins(0, 0, 0, 0)
        plot_hbox.setSpacing(0)
        plot_hbox.addWidget(self.plot_scroll_area, 1)

        self.dense_vscrollbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Vertical)
        self.dense_vscrollbar.hide()
        self.dense_vscrollbar.valueChanged.connect(self._on_dense_vscrollbar_changed)
        plot_hbox.addWidget(self.dense_vscrollbar)

        leftl.addLayout(plot_hbox, 1)
        splitter.addWidget(left)

        # right side
        right = QtWidgets.QWidget()
        right.setMinimumWidth(150)
        rl = QtWidgets.QVBoxLayout(right)
        self.right_layout = rl
        # Group videos into a dedicated container so we can control relative sizes
        self.videos_widget = QtWidgets.QWidget()
        self.videos_layout = QtWidgets.QVBoxLayout(self.videos_widget)
        self.videos_layout.setContentsMargins(0, 0, 0, 0)
        self.videos_layout.setSpacing(4)

        self.video_label = QtWidgets.QLabel("No video")
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumHeight(240)
        self.video_label.setStyleSheet("background-color:#222;border:1px solid #444;")
        self.videos_layout.addWidget(self.video_label, self.video1_stretch)
        self.video_label.installEventFilter(self)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Cursor:"))
        self.window_cursor_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.window_cursor_slider.setRange(0, 10000)
        self.window_cursor_slider.valueChanged.connect(self._on_window_cursor_changed)
        row.addWidget(self.window_cursor_slider)

        roww = QtWidgets.QWidget()
        roww.setLayout(row)
        # Add videos container before cursor row
        rl.addWidget(self.videos_widget, 1)
        rl.addWidget(roww)

        # Second video label (replaces image if video2 is loaded)
        self.video2_label = QtWidgets.QLabel("No video 2")
        self.video2_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video2_label.setMinimumHeight(200)
        self.video2_label.setStyleSheet("background-color:#222;border:1px solid #444;")
        self.video2_label.hide()
        self.videos_layout.addWidget(self.video2_label, self.video2_stretch)
        self.video2_label.installEventFilter(self)

        # Label Summary panel (replaces former static image; hidden when 2+ videos)
        self.label_summary_panel = LabelSummaryWidget(main_window=self)
        rl.addWidget(self.label_summary_panel, 2)

        # Third video label (same size weighting as video2, stacked below)
        self.video3_label = QtWidgets.QLabel("No video 3")
        self.video3_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video3_label.setMinimumHeight(200)
        self.video3_label.setStyleSheet("background-color:#222;border:1px solid #444;")
        self.video3_label.hide()
        self.videos_layout.addWidget(self.video3_label, self.video3_stretch)
        self.video3_label.installEventFilter(self)

        # Hypnogram overview plot (full-recording labels with moving window box)
        self.hypnogram_widget = pg.PlotWidget()
        self.hypnogram_widget.setMinimumHeight(90)
        hp = self.hypnogram_widget.getPlotItem()
        hp.showGrid(x=False, y=False)
        hp.hideAxis("left")
        hp.setMenuEnabled(False)
        hp.setMouseEnabled(x=False, y=False)
        hp.enableAutoRange("y", False)
        hp.setYRange(0, 1)
        self.hypnogram_plot = hp
        rl.addWidget(self.hypnogram_widget, 1)

        # Region showing the current window on the hypnogram
        self.hypnogram_view_region = pg.LinearRegionItem(
            values=(self.window_start, self.window_start + self.window_len),
            brush=pg.mkBrush(255, 255, 255, 50),
            movable=False,
        )
        self.hypnogram_view_region.setZValue(20)
        self.hypnogram_plot.addItem(self.hypnogram_view_region)

        # Ensure rescale happens when the splitter is adjusted
        self.splitter.splitterMoved.connect(self._on_splitter_moved)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.status = self.statusBar()
        self._update_status()
        self._build_menu()
        # Apply initial video stretches
        self._apply_video_stretches()

    def _build_menu(self):
        mfile = self.menuBar().addMenu("&File")
        a = QtGui.QAction("Load &Time Series…", self)
        a.triggered.connect(self._on_load_time_series)
        mfile.addAction(a)
        b = QtGui.QAction("Load &Video && Frame Times…", self)
        b.triggered.connect(self._on_load_video)
        mfile.addAction(b)
        m = QtGui.QAction("Load &Matrix Data…", self)
        m.triggered.connect(self._on_load_matrix_data)
        mfile.addAction(m)
        mfile.addSeparator()

        c = QtGui.QAction("Load &Labels…", self)
        c.triggered.connect(self._on_load_labels)
        mfile.addAction(c)

        d = QtGui.QAction("&Export Labels…", self)
        d.triggered.connect(self._on_export_labels)
        mfile.addAction(d)
        mfile.addSeparator()

        q = QtGui.QAction("&Quit", self)
        q.triggered.connect(self.close)
        mfile.addAction(q)

        medit = self.menuBar().addMenu("&Edit")
        clr = QtGui.QAction("Clear current selection", self)
        clr.triggered.connect(self._clear_selection)
        medit.addAction(clr)
        dl = QtGui.QAction("Delete last label", self)
        dl.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Backspace))
        dl.triggered.connect(self._delete_last_label)
        medit.addAction(dl)
        medit.addSeparator()
        note_action = QtGui.QAction("Add/Edit Epoch Note...", self)
        note_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+N"))
        note_action.triggered.connect(self._edit_epoch_note)
        medit.addAction(note_action)

        mview = self.menuBar().addMenu("&View")
        y_axis_action = QtGui.QAction("Y-Axis Controls...", self)
        y_axis_action.setShortcut(QtGui.QKeySequence("Ctrl+D"))
        y_axis_action.triggered.connect(self._show_y_axis_dialog)
        mview.addAction(y_axis_action)

        dense_ctrl_action = QtGui.QAction("Dense View Controls...", self)
        dense_ctrl_action.setShortcut(QtGui.QKeySequence("Ctrl+G"))
        dense_ctrl_action.triggered.connect(self._show_dense_controls_dialog)
        mview.addAction(dense_ctrl_action)

        scroll_speed_action = QtGui.QAction("Adjust Smooth Scroll Speed...", self)
        scroll_speed_action.triggered.connect(self._adjust_scroll_speed)
        mview.addAction(scroll_speed_action)

        adjust_video_sizes_action = QtGui.QAction(
            "Adjust Secondary Videos Size...", self
        )
        adjust_video_sizes_action.triggered.connect(self._adjust_secondary_video_sizes)
        mview.addAction(adjust_video_sizes_action)

        # Show/Hide videos (with hotkeys)
        self.action_show_v1 = QtGui.QAction("Show Video 1", self)
        self.action_show_v1.setCheckable(True)
        self.action_show_v1.setChecked(True)
        self.action_show_v1.setShortcut(QtGui.QKeySequence("Ctrl+Shift+1"))
        self.action_show_v1.toggled.connect(lambda ch: self._set_video_visible(1, ch))
        mview.addAction(self.action_show_v1)

        self.action_show_v2 = QtGui.QAction("Show Video 2", self)
        self.action_show_v2.setCheckable(True)
        self.action_show_v2.setChecked(False)
        self.action_show_v2.setShortcut(QtGui.QKeySequence("Ctrl+Shift+2"))
        self.action_show_v2.toggled.connect(lambda ch: self._set_video_visible(2, ch))
        mview.addAction(self.action_show_v2)

        self.action_show_v3 = QtGui.QAction("Show Video 3", self)
        self.action_show_v3.setCheckable(True)
        self.action_show_v3.setChecked(False)
        self.action_show_v3.setShortcut(QtGui.QKeySequence("Ctrl+Shift+3"))
        self.action_show_v3.toggled.connect(lambda ch: self._set_video_visible(3, ch))
        mview.addAction(self.action_show_v3)

        # Frame-step target selector
        step_menu = mview.addMenu("Frame Step Target")
        self.step_action_group = QtGui.QActionGroup(self)
        self.step_action_group.setExclusive(True)
        self.step_target_v1 = QtGui.QAction("Video 1", self, checkable=True)
        self.step_target_v2 = QtGui.QAction("Video 2", self, checkable=True)
        self.step_target_v3 = QtGui.QAction("Video 3", self, checkable=True)
        self.step_action_group.addAction(self.step_target_v1)
        self.step_action_group.addAction(self.step_target_v2)
        self.step_action_group.addAction(self.step_target_v3)
        self.step_target_v1.setChecked(True)
        self.step_target_v1.triggered.connect(lambda: self._set_frame_step_source(1))
        self.step_target_v2.triggered.connect(lambda: self._set_frame_step_source(2))
        self.step_target_v3.triggered.connect(lambda: self._set_frame_step_source(3))
        step_menu.addAction(self.step_target_v1)
        step_menu.addAction(self.step_target_v2)
        step_menu.addAction(self.step_target_v3)

        # Playback speed
        playback_speed_action = QtGui.QAction("Set Playback Speed...", self)
        playback_speed_action.triggered.connect(self._adjust_playback_speed)
        mview.addAction(playback_speed_action)

        # Matrix viewer settings
        mview.addSeparator()
        self.action_proportional_matrix = QtGui.QAction(
            "Proportional Matrix Plots", self
        )
        self.action_proportional_matrix.setCheckable(True)
        self.action_proportional_matrix.setChecked(self.scale_matrix_proportionally)
        self.action_proportional_matrix.setShortcut(QtGui.QKeySequence("Ctrl+Shift+M"))
        self.action_proportional_matrix.toggled.connect(
            self._toggle_proportional_matrix
        )
        mview.addAction(self.action_proportional_matrix)

        increase_matrix_share_action = QtGui.QAction("Increase Matrix Share", self)
        increase_matrix_share_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+,"))
        increase_matrix_share_action.triggered.connect(self._increase_matrix_share)
        mview.addAction(increase_matrix_share_action)

        decrease_matrix_share_action = QtGui.QAction("Decrease Matrix Share", self)
        decrease_matrix_share_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+."))
        decrease_matrix_share_action.triggered.connect(self._decrease_matrix_share)
        mview.addAction(decrease_matrix_share_action)

        matrix_brightness_action = QtGui.QAction("Adjust Matrix Brightness...", self)
        matrix_brightness_action.triggered.connect(self._adjust_matrix_brightness)
        mview.addAction(matrix_brightness_action)

        mview.addSeparator()
        matrix_height_action = QtGui.QAction("Matrix Event Height...", self)
        matrix_height_action.triggered.connect(self._adjust_matrix_event_height)
        mview.addAction(matrix_height_action)

        matrix_thickness_action = QtGui.QAction("Matrix Event Thickness...", self)
        matrix_thickness_action.triggered.connect(self._adjust_matrix_event_thickness)
        mview.addAction(matrix_thickness_action)

        mview.addSeparator()
        subplot_control_action = QtGui.QAction("Subplot Control Board...", self)
        subplot_control_action.setShortcut(QtGui.QKeySequence("Ctrl+H"))
        subplot_control_action.triggered.connect(self._show_subplot_control_dialog)
        mview.addAction(subplot_control_action)

        mview.addSeparator()
        jump_epochs_action = QtGui.QAction("Jump to Epochs...", self)
        jump_epochs_action.setShortcut(QtGui.QKeySequence("Ctrl+J"))
        jump_epochs_action.triggered.connect(self._show_jump_to_epochs_dialog)
        mview.addAction(jump_epochs_action)

        mhelp = self.menuBar().addMenu("&Help")
        hh = QtGui.QAction("Shortcuts / Help", self)
        hh.triggered.connect(self._show_help)
        mhelp.addAction(hh)

    def _adjust_scroll_speed(self):
        try:
            val, ok = QtWidgets.QInputDialog.getDouble(
                self,
                "Adjust Smooth Scroll Speed",
                "Fraction of window per wheel step (0.001 - 1.0):",
                float(self.smooth_scroll_fraction),
                0.001,
                1.0,
                3,
            )
        except Exception:
            val, ok = (self.smooth_scroll_fraction, False)
        if ok:
            self.smooth_scroll_fraction = float(max(0.001, min(1.0, val)))

    def _adjust_playback_speed(self):
        try:
            val, ok = QtWidgets.QInputDialog.getDouble(
                self,
                "Set Playback Speed",
                "Playback speed (0.25x - 4.0x, step 0.25):",
                float(self.playback_speed),
                0.25,
                4.0,
                2,
            )
        except Exception:
            val, ok = (self.playback_speed, False)
        if ok:
            # Quantize to nearest 0.25
            q = round(float(val) / 0.25) * 0.25
            self.playback_speed = float(max(0.25, min(4.0, q)))

    def _adjust_matrix_event_height(self):
        """Adjust the height of matrix event lines (distance from center in each direction)."""
        try:
            val, ok = QtWidgets.QInputDialog.getDouble(
                self,
                "Matrix Event Height",
                "Event height (0.1 - 0.5, distance from row center):",
                float(self.matrix_event_height),
                0.1,
                0.5,
                2,
            )
        except Exception:
            val, ok = (self.matrix_event_height, False)
        if ok:
            self.matrix_event_height = float(max(0.1, min(0.5, val)))
            self._refresh_matrix_plots()

    def _adjust_matrix_event_thickness(self):
        """Adjust the pen width of matrix event lines (in pixels)."""
        try:
            val, ok = QtWidgets.QInputDialog.getInt(
                self,
                "Matrix Event Thickness",
                "Event line thickness (pixels, 1-10):",
                int(self.matrix_event_thickness),
                1,
                10,
                1,
            )
        except Exception:
            val, ok = (self.matrix_event_thickness, False)
        if ok:
            self.matrix_event_thickness = int(max(1, min(10, val)))
            self._refresh_matrix_pen_cache()
            self._refresh_matrix_plots()

    def _toggle_proportional_matrix(self, checked: bool):
        """Toggle proportional matrix plot sizing on/off."""
        self.scale_matrix_proportionally = checked
        self._apply_trace_visibility()  # Rebuilds layout with new sizing

    def _increase_matrix_share(self):
        """Increase the vertical space share for matrix plots by ~5%."""
        if not self.matrix_series:
            return
        self.matrix_share_boost += 1
        self._apply_trace_visibility()
        self._update_status(f"Matrix share boost: {self.matrix_share_boost * 5:+d}%")

    def _decrease_matrix_share(self):
        """Decrease the vertical space share for matrix plots by ~5%."""
        if not self.matrix_series:
            return
        self.matrix_share_boost -= 1
        self._apply_trace_visibility()
        self._update_status(f"Matrix share boost: {self.matrix_share_boost * 5:+d}%")

    def _adjust_matrix_brightness(self):
        """Show a dialog to adjust matrix event brightness."""
        if not self.matrix_series:
            QtWidgets.QMessageBox.information(
                self, "Matrix Brightness", "No matrix plots loaded."
            )
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Adjust Matrix Brightness")
        lay = QtWidgets.QVBoxLayout(dlg)

        label = QtWidgets.QLabel(
            "Adjust brightness multiplier for matrix events.\n"
            "1.0 = default, <1.0 = dimmer, >1.0 = brighter"
        )
        lay.addWidget(label)

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(20, 300)  # 0.2 to 3.0
        slider.setValue(int(self.matrix_brightness * 100))
        lay.addWidget(slider)

        val_label = QtWidgets.QLabel(f"{self.matrix_brightness:.2f}")
        lay.addWidget(val_label)

        def on_change(val):
            brightness = val / 100.0
            val_label.setText(f"{brightness:.2f}")
            self.matrix_brightness = brightness
            self._refresh_matrix_plots()

        slider.valueChanged.connect(on_change)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)

        dlg.exec()

    def _edit_epoch_note(self):
        """Add or edit a note for the current or most recently scored epoch."""
        if not self.labels:
            QtWidgets.QMessageBox.warning(
                self,
                "No Epochs",
                "No scored epochs exist. Please score an epoch first.",
            )
            return

        # Find the epoch at cursor position
        target_epoch = None
        for lab in self.labels:
            if lab["start"] <= self.cursor_time < lab["end"]:
                target_epoch = lab
                break

        # If cursor is not in an epoch, use most recently scored epoch
        if target_epoch is None:
            if self.label_history:
                # Find the most recent epoch that still exists
                for start, end in reversed(self.label_history):
                    for lab in self.labels:
                        if (
                            abs(lab["start"] - start) < 1e-6
                            and abs(lab["end"] - end) < 1e-6
                        ):
                            target_epoch = lab
                            break
                    if target_epoch:
                        break

        if target_epoch is None:
            # Just use the last label in the list as fallback
            target_epoch = self.labels[-1]

        # Get existing note if any
        key = (float(target_epoch["start"]), float(target_epoch["end"]))
        existing_note = self.label_notes.get(key, "")

        # Show dialog to edit note
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Epoch Note")
        dlg.setMinimumWidth(400)
        layout = QtWidgets.QVBoxLayout(dlg)

        info_label = QtWidgets.QLabel(
            f"Epoch: {target_epoch['label']}\n"
            f"Time: {target_epoch['start']:.3f}s - {target_epoch['end']:.3f}s"
        )
        layout.addWidget(info_label)

        text_edit = QtWidgets.QTextEdit()
        text_edit.setPlainText(existing_note)
        text_edit.setMinimumHeight(100)
        layout.addWidget(text_edit)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() == QtWidgets.QDialog.Accepted:
            new_note = text_edit.toPlainText().strip()
            if new_note:
                self.label_notes[key] = new_note
            elif key in self.label_notes:
                del self.label_notes[key]
            self._update_status()
            self._refresh_label_summary()

    def _show_jump_to_epochs_dialog(self):
        """Show a table of all epochs for navigation and filtering."""
        if not self.labels:
            QtWidgets.QMessageBox.information(
                self, "Jump to Epochs", "No scored epochs to display."
            )
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Jump to Epochs")
        dlg.setMinimumWidth(700)
        dlg.setMinimumHeight(500)
        layout = QtWidgets.QVBoxLayout(dlg)

        # Filter controls
        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.addWidget(QtWidgets.QLabel("Filter by State:"))
        state_filter = QtWidgets.QComboBox()
        state_filter.addItem("(All)")
        # Add unique states from labels
        unique_states = sorted(set(lab["label"] for lab in self.labels))
        for state in unique_states:
            state_filter.addItem(state)
        filter_layout.addWidget(state_filter)

        filter_layout.addWidget(QtWidgets.QLabel("Filter Notes:"))
        notes_filter = QtWidgets.QLineEdit()
        notes_filter.setPlaceholderText("Enter text to search in notes...")
        filter_layout.addWidget(notes_filter, 1)

        layout.addLayout(filter_layout)

        # Table widget
        table = QtWidgets.QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Start (s)", "End (s)", "State", "Notes"])
        table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(table, 1)

        def populate_table():
            """Populate or filter the table based on current filters."""
            state_val = state_filter.currentText()
            notes_val = notes_filter.text().strip().lower()

            table.setRowCount(0)
            row = 0
            for lab in self.labels:
                # State filter
                if state_val != "(All)" and lab["label"] != state_val:
                    continue

                # Get note for this epoch
                key = (float(lab["start"]), float(lab["end"]))
                note = self.label_notes.get(key, "")

                # Notes filter
                if notes_val and notes_val not in note.lower():
                    continue

                table.insertRow(row)
                table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{lab['start']:.3f}"))
                table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{lab['end']:.3f}"))
                table.setItem(row, 2, QtWidgets.QTableWidgetItem(lab["label"]))
                table.setItem(row, 3, QtWidgets.QTableWidgetItem(note))

                # Store epoch data in the first cell for later retrieval
                table.item(row, 0).setData(QtCore.Qt.ItemDataRole.UserRole, lab)
                row += 1

            table.resizeColumnsToContents()

        populate_table()

        # Connect filters to update table
        state_filter.currentTextChanged.connect(lambda: populate_table())
        notes_filter.textChanged.connect(lambda: populate_table())

        def on_double_click(row, col):
            """Jump to the selected epoch (keeps dialog open)."""
            item = table.item(row, 0)
            if item:
                lab = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if lab:
                    # Calculate center of epoch
                    center = (lab["start"] + lab["end"]) / 2.0
                    # Position window so center is in the middle
                    new_start = center - self.window_len / 2.0
                    new_start = clamp(
                        new_start,
                        self.t_global_min,
                        max(self.t_global_min, self.t_global_max - self.window_len),
                    )
                    self.window_start = new_start
                    self.cursor_time = center
                    self._apply_x_range()
                    self._update_nav_slider_from_window()
                    # Dialog stays open so user can continue navigating

        table.cellDoubleClicked.connect(on_double_click)

        # Close button
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dlg.reject)
        layout.addWidget(close_btn)

        dlg.exec()

    def _jump_to_epoch_by_offset(self, direction: int) -> None:
        """Jump to the next (+1) or previous (-1) labelled epoch relative to cursor.

        Centers the view on the epoch without changing window size,
        using the same logic as the Jump to Epochs dialog.
        """
        if not self.labels:
            return

        cursor = self.cursor_time

        if direction > 0:
            # Find first epoch whose center is strictly after cursor
            for lab in self.labels:
                center = (lab["start"] + lab["end"]) / 2.0
                if center > cursor:
                    break
            else:
                return  # no epoch found ahead
        else:
            # Find last epoch whose center is strictly before cursor
            found = None
            for lab in self.labels:
                center = (lab["start"] + lab["end"]) / 2.0
                if center < cursor:
                    found = lab
                else:
                    break
            if found is None:
                return
            lab = found
            center = (lab["start"] + lab["end"]) / 2.0

        # Center view on the epoch (same as Jump to Epochs dialog)
        new_start = center - self.window_len / 2.0
        new_start = clamp(
            new_start,
            self.t_global_min,
            max(self.t_global_min, self.t_global_max - self.window_len),
        )
        self.window_start = new_start
        self.cursor_time = center
        self._apply_x_range()
        self._update_nav_slider_from_window()

    def _show_subplot_control_dialog(self):
        """Show a comprehensive dialog to control subplot heights, visibility, and order."""
        n_ts_plots = (
            len(self.overlay_groups)
            if self.overlay_mode
            else len(self.series)
        )
        total_subplots = n_ts_plots + len(self.dense_groups) + len(self.matrix_series)
        if total_subplots == 0:
            QtWidgets.QMessageBox.information(
                self, "Subplot Control", "No subplots loaded."
            )
            return

        # Ensure all control lists are properly sized
        while len(self.plot_height_factors) < n_ts_plots:
            self.plot_height_factors.append(1.0)
        while len(self.matrix_height_factors) < len(self.matrix_series):
            self.matrix_height_factors.append(1.0)
        while len(self.dense_height_factors) < len(self.dense_groups):
            self.dense_height_factors.append(1.0)
        if not hasattr(self, "trace_visible") or len(self.trace_visible) != n_ts_plots:
            self.trace_visible = [True] * n_ts_plots
        while len(self.matrix_visible) < len(self.matrix_series):
            self.matrix_visible.append(True)
        while len(self.dense_visible) < len(self.dense_groups):
            self.dense_visible.append(True)

        # Initialize subplot order if not set
        if self.subplot_order is None:
            self.subplot_order = []
            for i in range(n_ts_plots):
                self.subplot_order.append(("ts", i))
            for i in range(len(self.dense_groups)):
                self.subplot_order.append(("dense", i))
            for i in range(len(self.matrix_series)):
                self.subplot_order.append(("matrix", i))

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Subplot Control Board")
        dlg.setMinimumWidth(550)
        dlg.setMinimumHeight(400)
        main_lay = QtWidgets.QVBoxLayout(dlg)

        info_label = QtWidgets.QLabel(
            "Control subplot heights, visibility, and order.\n"
            "Drag rows to reorder. Check 'Hide' to hide a subplot."
        )
        main_lay.addWidget(info_label)

        # Create a list widget that supports drag-and-drop reordering
        list_widget = QtWidgets.QListWidget()
        list_widget.setDragDropMode(
            QtWidgets.QAbstractItemView.DragDropMode.InternalMove
        )
        list_widget.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        list_widget.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        list_widget.setMinimumHeight(250)
        main_lay.addWidget(list_widget, 1)

        # Store references to widgets for each row
        row_widgets = []  # List of dicts with plot info and widget references

        def create_row_widget(plot_type, idx):
            """Create a widget for a single row in the list."""
            if plot_type == "ts":
                if self.overlay_mode:
                    name = self.overlay_groups[idx].label
                else:
                    name = self.series[idx].name
                factor = self.plot_height_factors[idx]
                visible = self.trace_visible[idx]
                display_name = f"[TS] {name}"
            elif plot_type == "dense":
                group = self.dense_groups[idx]
                name = group.name
                factor = self.dense_height_factors[idx]
                visible = self.dense_visible[idx]
                n = len(group.series)
                display_name = f"[Dense/{n}] {name}"
            else:
                name = self.matrix_series[idx].name
                factor = self.matrix_height_factors[idx]
                visible = self.matrix_visible[idx]
                display_name = f"[Matrix] {name}"

            widget = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(widget)
            layout.setContentsMargins(4, 2, 4, 2)

            # Drag handle indicator
            drag_label = QtWidgets.QLabel("≡")
            drag_label.setStyleSheet("color: gray; font-size: 14px;")
            drag_label.setFixedWidth(20)
            layout.addWidget(drag_label)

            # Name label
            name_label = QtWidgets.QLabel(display_name)
            name_label.setMinimumWidth(120)
            name_label.setMaximumWidth(180)
            layout.addWidget(name_label)

            # Height slider
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(1, 2000)  # 0.01x to 20.0x (very wide range)
            slider.setValue(int(factor * 100))
            slider.setMinimumWidth(150)
            layout.addWidget(slider)

            # Value label
            val_label = QtWidgets.QLabel(f"{factor:.2f}x")
            val_label.setMinimumWidth(50)
            layout.addWidget(val_label)

            # Hide checkbox
            hide_check = QtWidgets.QCheckBox("Hide")
            hide_check.setChecked(not visible)
            layout.addWidget(hide_check)

            # Connect slider
            def on_slider_change(val):
                new_factor = val / 100.0
                if plot_type == "ts":
                    self.plot_height_factors[idx] = new_factor
                elif plot_type == "dense":
                    self.dense_height_factors[idx] = new_factor
                else:
                    self.matrix_height_factors[idx] = new_factor
                val_label.setText(f"{new_factor:.2f}x")
                self._apply_trace_visibility()

            slider.valueChanged.connect(on_slider_change)

            # Connect hide checkbox
            def on_hide_changed(state):
                is_visible = state != QtCore.Qt.CheckState.Checked.value
                if plot_type == "ts":
                    self.trace_visible[idx] = is_visible
                elif plot_type == "dense":
                    self.dense_visible[idx] = is_visible
                else:
                    self.matrix_visible[idx] = is_visible
                self._apply_trace_visibility()

            hide_check.stateChanged.connect(on_hide_changed)

            return {
                "widget": widget,
                "type": plot_type,
                "idx": idx,
                "slider": slider,
                "val_label": val_label,
                "hide_check": hide_check,
            }

        # Populate the list widget based on current order
        for plot_type, idx in self.subplot_order:
            # Validate the entry
            valid = False
            if plot_type == "ts" and idx < len(self.series):
                valid = True
            elif plot_type == "dense" and idx < len(self.dense_groups):
                valid = True
            elif plot_type == "matrix" and idx < len(self.matrix_series):
                valid = True
            if valid:
                row_data = create_row_widget(plot_type, idx)
                row_widgets.append(row_data)
                item = QtWidgets.QListWidgetItem()
                item.setSizeHint(row_data["widget"].sizeHint())
                item.setData(QtCore.Qt.ItemDataRole.UserRole, (plot_type, idx))
                list_widget.addItem(item)
                list_widget.setItemWidget(item, row_data["widget"])

        def update_order_from_list():
            """Update subplot_order based on current list widget order."""
            new_order = []
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                data = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if data:
                    new_order.append(data)
            self.subplot_order = new_order
            self._apply_trace_visibility()

        # Connect model changes to update order
        list_widget.model().rowsMoved.connect(lambda: update_order_from_list())

        # Button row
        btn_row = QtWidgets.QHBoxLayout()

        reset_heights_btn = QtWidgets.QPushButton("Reset Heights")

        def reset_heights():
            for rw in row_widgets:
                rw["slider"].blockSignals(True)
                rw["slider"].setValue(100)
                rw["val_label"].setText("1.00x")
                if rw["type"] == "ts":
                    self.plot_height_factors[rw["idx"]] = 1.0
                elif rw["type"] == "dense":
                    self.dense_height_factors[rw["idx"]] = 1.0
                else:
                    self.matrix_height_factors[rw["idx"]] = 1.0
                rw["slider"].blockSignals(False)
            self._apply_trace_visibility()

        reset_heights_btn.clicked.connect(reset_heights)
        btn_row.addWidget(reset_heights_btn)

        show_all_btn = QtWidgets.QPushButton("Show All")

        def show_all():
            for rw in row_widgets:
                rw["hide_check"].blockSignals(True)
                rw["hide_check"].setChecked(False)
                if rw["type"] == "ts":
                    self.trace_visible[rw["idx"]] = True
                elif rw["type"] == "dense":
                    self.dense_visible[rw["idx"]] = True
                else:
                    self.matrix_visible[rw["idx"]] = True
                rw["hide_check"].blockSignals(False)
            self._apply_trace_visibility()

        show_all_btn.clicked.connect(show_all)
        btn_row.addWidget(show_all_btn)

        reset_order_btn = QtWidgets.QPushButton("Reset Order")

        def reset_order():
            # Rebuild the default order
            n_ts_reset = (
                len(self.overlay_groups)
                if self.overlay_mode
                else len(self.series)
            )
            self.subplot_order = []
            for i in range(n_ts_reset):
                self.subplot_order.append(("ts", i))
            for i in range(len(self.dense_groups)):
                self.subplot_order.append(("dense", i))
            for i in range(len(self.matrix_series)):
                self.subplot_order.append(("matrix", i))
            # Close and reopen dialog to refresh
            dlg.accept()
            QtCore.QTimer.singleShot(50, self._show_subplot_control_dialog)

        reset_order_btn.clicked.connect(reset_order)
        btn_row.addWidget(reset_order_btn)

        btn_row.addStretch()
        main_lay.addLayout(btn_row)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        btns.accepted.connect(dlg.accept)
        main_lay.addWidget(btns)

        dlg.exec()

    def _apply_custom_plot_heights(self):
        """Apply custom height factors to all visible plots based on subplot_order."""
        try:
            layout = self.plot_area.ci.layout

            # Base height for calculations - allow very small heights
            BASE_HEIGHT = 100
            MIN_HEIGHT = 1  # Very small minimum to allow extreme shrinking

            # Get the ordered list of visible plots
            visible_plots = self._get_visible_subplot_order()

            row = 0
            for plot_type, idx in visible_plots:
                if plot_type == "ts":
                    factor = (
                        self.plot_height_factors[idx]
                        if idx < len(self.plot_height_factors)
                        else 1.0
                    )
                    plt = self.plots[idx] if idx < len(self.plots) else None

                    preferred = max(MIN_HEIGHT, int(BASE_HEIGHT * factor))
                    stretch = max(
                        1, int(factor * 100)
                    )  # Scale stretch more aggressively

                    # Hide axis labels for very small plots (below 0.2x)
                    if plt:
                        self._configure_plot_for_height(plt, factor, is_matrix=False)

                elif plot_type == "dense":
                    factor = (
                        self.dense_height_factors[idx]
                        if idx < len(self.dense_height_factors)
                        else 1.0
                    )
                    plt = (
                        self.dense_plots[idx] if idx < len(self.dense_plots) else None
                    )
                    # Dense plots get more height by default
                    preferred = max(MIN_HEIGHT, int(BASE_HEIGHT * factor * 3))
                    stretch = max(1, int(factor * 300))

                    if plt:
                        self._configure_plot_for_height(plt, factor, is_matrix=False)

                else:  # matrix
                    factor = (
                        self.matrix_height_factors[idx]
                        if idx < len(self.matrix_height_factors)
                        else 1.0
                    )
                    plt = (
                        self.matrix_plots[idx] if idx < len(self.matrix_plots) else None
                    )
                    ms = (
                        self.matrix_series[idx]
                        if idx < len(self.matrix_series)
                        else None
                    )

                    if self.scale_matrix_proportionally and ms:
                        # Combine proportional scaling with custom factor
                        boost_factor = 1.0 + (self.matrix_share_boost * 0.05)
                        BASE_HEIGHT_PER_ROW = 12 * boost_factor
                        preferred = max(
                            MIN_HEIGHT, int(ms.n_rows * BASE_HEIGHT_PER_ROW * factor)
                        )
                        stretch = max(1, int(ms.n_rows * boost_factor * factor * 10))
                    else:
                        preferred = max(MIN_HEIGHT, int(BASE_HEIGHT * factor))
                        stretch = max(1, int(factor * 100))

                    # Hide axis labels for very small plots
                    if plt:
                        self._configure_plot_for_height(plt, factor, is_matrix=True)

                layout.setRowPreferredHeight(row, preferred)
                layout.setRowMinimumHeight(row, MIN_HEIGHT)
                layout.setRowStretchFactor(row, stretch)
                row += 1

        except Exception as e:
            import traceback

            traceback.print_exc()

    def _configure_plot_for_height(self, plt, factor, is_matrix=False):
        """Configure plot axis visibility based on height factor."""
        try:
            # For very small plots (below 0.2x), hide axis labels to save space
            if factor < 0.2:
                plt.getAxis("left").setStyle(showValues=False)
                plt.getAxis("left").setWidth(15)
                plt.setLabel("left", "")
            else:
                plt.getAxis("left").setStyle(showValues=True)
                plt.getAxis("left").setWidth(None)  # Auto width
                # Restore label if needed (we don't store original, so this is best effort)
        except Exception:
            pass

    def _get_visible_subplot_order(self):
        """Get the list of visible subplots in their current order."""
        # Ensure visibility lists are properly sized
        n_ts = (
            len(self.overlay_groups)
            if self.overlay_mode
            else len(self.series)
        )
        if not hasattr(self, "trace_visible") or len(self.trace_visible) != n_ts:
            self.trace_visible = [True] * n_ts
        while len(self.matrix_visible) < len(self.matrix_series):
            self.matrix_visible.append(True)

        while len(self.dense_visible) < len(self.dense_groups):
            self.dense_visible.append(True)

        # Use subplot_order if set, otherwise default order
        if self.subplot_order:
            order = self.subplot_order
        else:
            order = [("ts", i) for i in range(n_ts)]
            order += [("dense", i) for i in range(len(self.dense_groups))]
            order += [("matrix", i) for i in range(len(self.matrix_series))]

        # Filter to only visible plots
        visible = []
        for plot_type, idx in order:
            if plot_type == "ts":
                if idx < len(self.trace_visible) and self.trace_visible[idx]:
                    visible.append((plot_type, idx))
            elif plot_type == "dense":
                if idx < len(self.dense_visible) and self.dense_visible[idx]:
                    visible.append((plot_type, idx))
            else:  # matrix
                if idx < len(self.matrix_visible) and self.matrix_visible[idx]:
                    visible.append((plot_type, idx))
        return visible

    # ---------- Data ----------
    def _load_series_from_dir(self, folder):
        self._stop_playback_if_playing()
        pairs = []
        for tpath in glob.glob(os.path.join(folder, "*_t.npy")):
            name = os.path.basename(tpath)[:-6]
            ypath = os.path.join(folder, f"{name}_y.npy")
            if os.path.exists(ypath):
                pairs.append((name, tpath, ypath))
        if not pairs:
            QtWidgets.QMessageBox.warning(
                self, "No data", "No *_t.npy / *_y.npy pairs found."
            )
            return

        series = []
        for name, tpath, ypath in sorted(pairs):
            try:
                t = np.load(tpath).astype(float)
                y = np.load(ypath).astype(float)
                if t.ndim != 1 or y.ndim != 1 or len(t) != len(y):
                    raise ValueError("t and y must be 1-D & equal length")
                series.append(Series(name, t, y))
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Load error", f"{name}: {e}")
        if series:
            self.set_series(series)

    def _load_matrix_from_dir(self, folder):
        """Load matrix series from a directory with *_t.npy, *_y.npy, and optional *_a.npy files."""
        pairs = []
        for tpath in sorted(glob.glob(os.path.join(folder, "*_t.npy"))):
            base = os.path.basename(tpath)[:-6]  # strip "_t.npy"
            ypath = os.path.join(folder, f"{base}_y.npy")
            if not os.path.exists(ypath):
                continue
            apath = os.path.join(folder, f"{base}_a.npy")
            pairs.append((tpath, ypath, apath if os.path.exists(apath) else ""))

        if not pairs:
            QtWidgets.QMessageBox.warning(
                self,
                "No matrix data",
                "No *_t.npy / *_y.npy pairs found in the selected directory.",
            )
            return

        ts_paths = [p[0] for p in pairs]
        yv_paths = [p[1] for p in pairs]
        al_paths = [p[2] for p in pairs]
        self._load_matrix_data(ts_paths, yv_paths, al_paths, colors=None)

    def _on_load_matrix_data(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder with matrix *_t.npy and *_y.npy files"
        )
        if folder:
            self._load_matrix_from_dir(folder)

    def _load_series_from_files(self, files, colors=None):
        """Load series from an explicit ordered list of *_t.npy / *_y.npy files.

        The display order (top to bottom) follows the order in which distinct
        base names first appear in the provided list. A "base name" is the
        filename without the trailing "_t.npy" or "_y.npy".
        """
        self._stop_playback_if_playing()

        if not files:
            QtWidgets.QMessageBox.warning(self, "No data", "No files provided.")
            return

        # Build mapping base_name -> { 't': path or None, 'y': path or None }
        series_map = {}
        order = []  # first-seen order of base names
        first_index = {}  # base name -> first index in files list

        def base_for(path: str):
            fn = os.path.basename(path)
            if fn.endswith("_t.npy"):
                return fn[:-6], "t"
            if fn.endswith("_y.npy"):
                return fn[:-6], "y"
            return None, None

        for idx, p in enumerate(files):
            if not p:
                continue
            b, kind = base_for(p)
            if b is None:
                QtWidgets.QMessageBox.warning(
                    self, "Skip", f"Not a *_t.npy or *_y.npy file: {p}"
                )
                continue
            if b not in series_map:
                series_map[b] = {"t": None, "y": None}
                order.append(b)
                first_index[b] = idx
            series_map[b][kind] = p

        # Assemble in the order seen; require both t and y
        series = []
        series_colors = []
        for b in order:
            paths = series_map.get(b, {})
            tpath, ypath = paths.get("t"), paths.get("y")
            if not tpath or not ypath:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Missing pair",
                    f"Skipping '{b}': need both {b}_t.npy and {b}_y.npy",
                )
                continue
            if not (os.path.exists(tpath) and os.path.exists(ypath)):
                QtWidgets.QMessageBox.warning(
                    self, "File not found", f"Missing files for '{b}'."
                )
                continue
            try:
                t = np.load(tpath).astype(float)
                y = np.load(ypath).astype(float)
                if t.ndim != 1 or y.ndim != 1 or len(t) != len(y):
                    raise ValueError("t and y must be 1-D & equal length")
                series.append(Series(b, t, y))
                series_colors.append(None)  # placeholder
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Load error", f"{b}: {e}")

        if not series:
            QtWidgets.QMessageBox.warning(
                self, "No data", "No valid *_t.npy / *_y.npy pairs in list."
            )
            return

        # Map provided colors to series order
        def _parse_color(cs: str):
            s = (cs or "").strip()
            try:
                if s.startswith("#"):
                    s = s[1:]
                    if len(s) in (6, 8):
                        r = int(s[0:2], 16)
                        g = int(s[2:4], 16)
                        b = int(s[4:6], 16)
                        a = int(s[6:8], 16) if len(s) == 8 else 255
                        return (r, g, b, a)
                if s.lower().startswith("0x"):
                    v = int(s, 16)
                    r = (v >> 16) & 0xFF
                    g = (v >> 8) & 0xFF
                    b = v & 0xFF
                    return (r, g, b, 255)
                if "," in s:
                    parts = [int(x.strip()) for x in s.split(",") if x.strip()]
                    if len(parts) == 3:
                        return (parts[0], parts[1], parts[2], 255)
                    if len(parts) >= 4:
                        return (parts[0], parts[1], parts[2], parts[3])
            except Exception:
                pass
            return None

        mapped_colors = None
        if colors:
            # If colors count matches series count, map 1:1
            if len(colors) == len(series):
                mapped_colors = [
                    (_parse_color(c) or (255, 255, 255, 255)) for c in colors
                ]
            # If colors matches files count, use first occurrence index mapping
            elif len(colors) == len(files):
                mapped_colors = []
                for b in order:
                    ci = first_index.get(b, 0)
                    col = _parse_color(colors[ci]) or (255, 255, 255, 255)
                    mapped_colors.append(col)
            else:
                QtWidgets.QMessageBox.information(
                    self,
                    "Colors",
                    (
                        f"Ignoring --colors: count {len(colors)} doesn't match series ({len(series)}) "
                        f"or file count ({len(files)})."
                    ),
                )

        # Store colors aligned with series; default to white if not provided
        self.series_colors = mapped_colors or [(255, 255, 255, 255)] * len(series)
        self.set_series(series)

    def _on_load_time_series(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder with *_t.npy and *_y.npy"
        )
        if folder:
            self._load_series_from_dir(folder)

    def set_series(self, series_list, colors=None):
        self.series = series_list
        self._refresh_low_profile_x()

        # Assign per-trace colours (RGBA tuples or default white).
        if colors and len(colors) == len(series_list):
            parsed = []
            for c in colors:
                if isinstance(c, str):
                    c = c.strip().lstrip("#")
                    try:
                        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
                        a = int(c[6:8], 16) if len(c) == 8 else 255
                        parsed.append((r, g, b, a))
                    except Exception:
                        parsed.append((255, 255, 255, 255))
                elif isinstance(c, (tuple, list)) and len(c) >= 3:
                    parsed.append(tuple(c[:4]) if len(c) >= 4 else (*c[:3], 255))
                else:
                    parsed.append((255, 255, 255, 255))
            self.series_colors = parsed
        else:
            self.series_colors = [(255, 255, 255, 255)] * len(series_list)

        self._clear_all_label_visuals()
        self.plot_area.clear()
        self.plots.clear()
        self.curves.clear()
        self.plot_cur_lines.clear()
        self.plot_sel_regions.clear()
        self.dense_plots.clear()
        self.dense_curves.clear()
        self.dense_cur_lines.clear()
        self.dense_sel_regions.clear()
        self.dense_label_regions.clear()
        self.matrix_plots.clear()
        self.matrix_items.clear()
        self.matrix_cur_lines.clear()
        self.matrix_sel_regions.clear()
        self._matrix_line_items.clear()
        self._matrix_pens.clear()

        # Initialize height factors and visibility for time series plots
        self.plot_height_factors = [1.0] * len(series_list)
        self.trace_visible = [True] * len(series_list)
        # Reset subplot order when loading new data
        self.subplot_order = None

        # Calculate time range from time series, dense groups, and matrix data
        t_arrays = [s.t for s in self.series]
        for g in self.dense_groups:
            for s in g.series:
                t_arrays.append(s.t)
        for ms in self.matrix_series:
            if len(ms.timestamps) > 0:
                t_arrays.append(ms.timestamps)
        self.t_global_min, self.t_global_max = nice_time_range(t_arrays)
        self.window_start = self.t_global_min
        self.cursor_time = self.window_start

        # Create all plots (time series and matrix)
        self._create_all_plots()

        self._apply_x_range()
        self._update_nav_slider_from_window()
        n_dense = sum(len(g.series) for g in self.dense_groups)
        parts = []
        if self.series:
            parts.append(f"{len(self.series)} series")
        if n_dense:
            parts.append(f"{n_dense} dense traces in {len(self.dense_groups)} group(s)")
        self._update_status(f"Loaded {', '.join(parts) or '0 series'}.")
        self._update_hypnogram_extents()
        # Align left axes after layout settles
        QtCore.QTimer.singleShot(0, self._align_left_axes)
        QtCore.QTimer.singleShot(100, self._align_left_axes)
        # Initialize trace visibility state if needed
        if not hasattr(self, "trace_visible") or len(self.trace_visible) != len(
            self.series
        ):
            self.trace_visible = [True] * len(self.series)
        self._apply_trace_visibility()
        if self.labels:
            self._sync_label_visuals(force_rebuild=True, refresh_summary=False)

    # Default overlay color palette
    _DEFAULT_OVERLAY_COLORS = [
        (100, 200, 255, 255),  # light blue
        (255, 150, 50, 255),  # orange
        (100, 255, 100, 255),  # green
        (255, 100, 255, 255),  # magenta
        (255, 255, 100, 255),  # yellow
        (255, 100, 100, 255),  # red
        (100, 255, 255, 255),  # cyan
        (200, 150, 255, 255),  # lavender
    ]

    def set_overlay_series(self, overlay_groups, colors=None):
        """Load overlay groups into the viewer.

        Parameters
        ----------
        overlay_groups : list[OverlayGroup]
            Groups of traces to overlay on shared subplots.
        colors : list or None
            One color per source DataArray. Accepts hex strings or RGB(A) tuples.
        """
        self._stop_playback_if_playing()
        self.overlay_mode = True
        self.overlay_groups = overlay_groups

        # Determine number of source DataArrays
        n_sources = 0
        for g in overlay_groups:
            for tr in g.traces:
                n_sources = max(n_sources, tr.source_idx + 1)

        # Parse overlay colors
        if colors and len(colors) >= n_sources:
            parsed = []
            for c in colors:
                if isinstance(c, str):
                    c = c.strip().lstrip("#")
                    try:
                        r, g_, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
                        a = int(c[6:8], 16) if len(c) == 8 else 255
                        parsed.append((r, g_, b, a))
                    except Exception:
                        parsed.append((255, 255, 255, 255))
                elif isinstance(c, (tuple, list)) and len(c) >= 3:
                    parsed.append(tuple(c[:4]) if len(c) >= 4 else (*c[:3], 255))
                else:
                    parsed.append((255, 255, 255, 255))
            self.overlay_colors = parsed
        else:
            self.overlay_colors = list(self._DEFAULT_OVERLAY_COLORS[:n_sources])
            # Extend with white if more sources than palette entries
            while len(self.overlay_colors) < n_sources:
                self.overlay_colors.append((255, 255, 255, 255))

        # Flatten all traces into self.series for data storage
        self.series = []
        self._plot_to_series = []
        series_idx = 0
        for group in overlay_groups:
            indices = []
            for tr in group.traces:
                self.series.append(Series(tr.name, tr.t, tr.y))
                indices.append(series_idx)
                series_idx += 1
            self._plot_to_series.append(indices)

        self._refresh_low_profile_x()

        # Clear existing plots
        self._clear_all_label_visuals()
        self.plot_area.clear()
        self.plots.clear()
        self.curves.clear()
        self._plot_to_curves.clear()
        self.plot_cur_lines.clear()
        self.plot_sel_regions.clear()
        self.matrix_plots.clear()
        self.matrix_items.clear()
        self.matrix_cur_lines.clear()
        self.matrix_sel_regions.clear()
        self._matrix_line_items.clear()
        self._matrix_pens.clear()

        # Height factors and visibility are per-plot (per overlay group)
        self.plot_height_factors = [1.0] * len(overlay_groups)
        self.trace_visible = [True] * len(overlay_groups)
        self.subplot_order = None

        # Calculate time range
        t_arrays = [s.t for s in self.series]
        for ms in self.matrix_series:
            if len(ms.timestamps) > 0:
                t_arrays.append(ms.timestamps)
        self.t_global_min, self.t_global_max = nice_time_range(t_arrays)
        self.window_start = self.t_global_min
        self.cursor_time = self.window_start

        # Create all plots
        self._create_all_plots()

        self._apply_x_range()
        self._update_nav_slider_from_window()
        self._update_status(
            f"Loaded {len(overlay_groups)} overlay groups "
            f"({len(self.series)} total traces)."
        )
        self._update_hypnogram_extents()
        QtCore.QTimer.singleShot(0, self._align_left_axes)
        QtCore.QTimer.singleShot(100, self._align_left_axes)
        self._apply_trace_visibility()
        if self.labels:
            self._sync_label_visuals(force_rebuild=True, refresh_summary=False)

    def set_xarray(self, data, filter_dict=None):
        """Load xarray DataArray(s) into the viewer.

        Parameters
        ----------
        data : xr.DataArray or list[xr.DataArray] or str
            In-memory DataArray(s), or a path to a zarr/netCDF store.
        filter_dict : dict, optional
            Dimension slicing, e.g. ``{"syn_id": slice(3, 6)}``.
        """
        from loupe.xr_loader import (
            convert_xarray_inputs,
            load_xarray_from_path,
        )

        if isinstance(data, str):
            data = load_xarray_from_path(data, filter_dict=filter_dict)
        elif filter_dict is not None and not isinstance(data, list):
            data = data.sel(**filter_dict)

        tuples = convert_xarray_inputs(data)
        self.set_series([Series(n, t, y) for n, t, y in tuples])

    def set_matrix_df(
        self,
        data,
        *,
        time_col: str = "time",
        y_col: str = "source_id",
        group_col: str | list[str] | None = None,
        alpha_col: str | None = None,
        name: str = "events",
        colors=None,
        alpha_range: tuple[float, float] = (0.3, 1.0),
    ):
        """Load a Polars DataFrame as matrix/raster plots.

        Parameters
        ----------
        data : pl.DataFrame, list[pl.DataFrame], or str
            In-memory DataFrame(s), or a path to a parquet file.
        time_col : str
            Column with event timestamps (seconds).
        y_col : str
            Column for matrix row assignment.
        group_col : str or list[str] or None
            Column(s) to split into separate subplots.
        alpha_col : str or None
            Column for per-event opacity.
        name : str
            Base name for the matrix subplots.
        colors : dict, list, tuple or None
            Color specification per group.
        alpha_range : tuple[float, float]
            ``(min_alpha, max_alpha)`` for normalizing *alpha_col*.
        """
        from loupe.df_loader import (
            dataframe_to_matrix_series,
            load_dataframe_from_parquet,
        )

        if isinstance(data, str):
            data = load_dataframe_from_parquet(data, time_col=time_col)

        if not isinstance(data, list):
            data = [data]

        all_ms: list[MatrixSeries] = []
        for i, mdf in enumerate(data):
            prefix = name if len(data) == 1 else f"{name}_{i}"
            all_ms.extend(
                dataframe_to_matrix_series(
                    mdf,
                    time_col=time_col,
                    y_col=y_col,
                    group_col=group_col,
                    alpha_col=alpha_col,
                    name=prefix,
                    colors=colors,
                    alpha_range=alpha_range,
                )
            )

        if not all_ms:
            return

        self.matrix_series = all_ms
        self._refresh_low_profile_x()
        self.matrix_height_factors = [1.0] * len(all_ms)
        self.matrix_visible = [True] * len(all_ms)
        self.subplot_order = None
        if self.series:
            self._rebuild_all_plots()
        else:
            self._update_time_range_from_matrix()
            self._create_matrix_only_plots()
        self._update_status(
            f"Loaded {len(all_ms)} matrix series from DataFrame."
        )

    # ---------- Matrix Viewer ----------
    def _load_matrix_data(self, timestamps_paths, yvals_paths, alpha_paths, colors):
        """Load matrix/raster data from provided file paths."""

        def _normalize_list(raw_list):
            if not raw_list:
                return []
            items = [raw_list] if isinstance(raw_list, str) else list(raw_list)
            out = []
            for it in items:
                s = (it or "").strip()
                if s.startswith("[") and s.endswith("]"):
                    s = s[1:-1]
                parts = s.split(",") if "," in s else [s]
                for p in parts:
                    q = p.strip().strip('"').strip("'")
                    if q.endswith(","):
                        q = q[:-1].rstrip()
                    if q:
                        out.append(q)
            return out

        ts_paths = _normalize_list(timestamps_paths)
        yv_paths = _normalize_list(yvals_paths)
        al_paths = _normalize_list(alpha_paths) if alpha_paths else []
        color_list = _normalize_list(colors) if colors else []

        if len(ts_paths) != len(yv_paths):
            QtWidgets.QMessageBox.warning(
                self,
                "Matrix Data",
                f"matrix_timestamps ({len(ts_paths)}) and matrix_yvals ({len(yv_paths)}) must have same length.",
            )
            return

        def _parse_color(cs: str):
            s = (cs or "").strip()
            try:
                if s.startswith("#"):
                    s = s[1:]
                    if len(s) in (6, 8):
                        r = int(s[0:2], 16)
                        g = int(s[2:4], 16)
                        b = int(s[4:6], 16)
                        return (r, g, b)
                if s.lower().startswith("0x"):
                    v = int(s, 16)
                    r = (v >> 16) & 0xFF
                    g = (v >> 8) & 0xFF
                    b = v & 0xFF
                    return (r, g, b)
                if "," in s:
                    parts = [int(x.strip()) for x in s.split(",") if x.strip()]
                    if len(parts) >= 3:
                        return (parts[0], parts[1], parts[2])
            except Exception:
                pass
            return None

        matrix_series = []
        for i, (ts_path, yv_path) in enumerate(zip(ts_paths, yv_paths)):
            if not os.path.exists(ts_path):
                QtWidgets.QMessageBox.warning(
                    self, "Matrix Data", f"Timestamps file not found: {ts_path}"
                )
                continue
            if not os.path.exists(yv_path):
                QtWidgets.QMessageBox.warning(
                    self, "Matrix Data", f"Yvals file not found: {yv_path}"
                )
                continue

            try:
                timestamps = np.load(ts_path).astype(float).flatten()
                yvals = np.load(yv_path).astype(int).flatten()

                if len(timestamps) != len(yvals):
                    raise ValueError(
                        f"timestamps ({len(timestamps)}) and yvals ({len(yvals)}) must have same length"
                    )

                # Load alphas if provided
                if i < len(al_paths) and al_paths[i] and os.path.exists(al_paths[i]):
                    alphas = np.load(al_paths[i]).astype(float).flatten()
                    if len(alphas) != len(timestamps):
                        raise ValueError(
                            f"alphas ({len(alphas)}) must match timestamps ({len(timestamps)})"
                        )
                    alphas = np.clip(alphas, 0.0, 1.0)
                else:
                    alphas = np.ones(len(timestamps), dtype=float)

                # Parse color
                if i < len(color_list) and color_list[i]:
                    color = _parse_color(color_list[i]) or (255, 255, 255)
                else:
                    color = (255, 255, 255)

                # Determine number of rows
                n_rows = int(np.max(yvals)) + 1 if len(yvals) > 0 else 1

                # Sort by timestamps for efficient windowed rendering
                order = np.argsort(timestamps)
                timestamps = timestamps[order]
                yvals = yvals[order]
                alphas = alphas[order]

                _bn = os.path.basename(ts_path)
                for _suf in ("_timestamps.npy", "_t.npy", ".npy"):
                    if _bn.endswith(_suf):
                        _bn = _bn[: -len(_suf)]
                        break
                name = _bn
                matrix_series.append(
                    MatrixSeries(
                        name=name,
                        timestamps=timestamps,
                        yvals=yvals,
                        alphas=alphas,
                        color=color,
                        n_rows=n_rows,
                    )
                )
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Matrix Load Error", f"Error loading matrix {i}: {e}"
                )
                continue

        if matrix_series:
            self.matrix_series = matrix_series
            self._refresh_low_profile_x()
            # Initialize height factors and visibility for matrix plots
            self.matrix_height_factors = [1.0] * len(matrix_series)
            self.matrix_visible = [True] * len(matrix_series)
            # Reset subplot order to include new matrix plots
            self.subplot_order = None
            self._update_status(f"Loaded {len(matrix_series)} matrix series.")
            # Rebuild plots to include matrix series
            if self.series:
                self._rebuild_all_plots()
            else:
                # Update time range and create matrix plots if no time series
                self._update_time_range_from_matrix()
                self._create_matrix_only_plots()

    def _update_time_range_from_matrix(self):
        """Update global time range to include matrix timestamps."""
        if not self.matrix_series:
            return
        for ms in self.matrix_series:
            if len(ms.timestamps) > 0:
                t_min = float(np.min(ms.timestamps))
                t_max = float(np.max(ms.timestamps))
                self.t_global_min = min(self.t_global_min, t_min)
                self.t_global_max = max(self.t_global_max, t_max)

    def _create_matrix_only_plots(self):
        """Create matrix plots when there are no time series."""
        if not self.matrix_series:
            return

        self._refresh_low_profile_x()
        self.window_start = self.t_global_min
        self.cursor_time = self.window_start

        # Clear any existing plots
        self._clear_all_label_visuals()
        self.plot_area.clear()
        self.matrix_plots.clear()
        self.matrix_items.clear()
        self.matrix_cur_lines.clear()
        self.matrix_sel_regions.clear()
        self._matrix_line_items.clear()
        self._matrix_pens.clear()

        master_plot = None
        total_plots = len(self.matrix_series)

        for idx, ms in enumerate(self.matrix_series):
            vb = SelectableViewBox()
            vb.sigWheelScrolled.connect(self._page)
            vb.sigWheelSmoothScrolled.connect(self._on_smooth_scroll)
            vb.sigWheelCursorScrolled.connect(self._on_cursor_wheel)

            plt = HoverablePlotItem(viewBox=vb)
            plt.sigHovered.connect(self._on_plot_hovered)
            self.plot_area.addItem(plt, row=idx, col=0)

            plt.setLabel("left", ms.name)
            is_last = idx == total_plots - 1
            plt.setLabel("bottom", "Time", units="s" if is_last else None)

            plt.showGrid(x=True, y=False, alpha=0.15)
            plt.enableAutoRange("x", False)
            plt.enableAutoRange("y", False)
            plt.setYRange(-0.5, ms.n_rows - 0.5, padding=0.02)

            left_axis = plt.getAxis("left")
            left_axis.setTicks([[(0, "0"), (ms.n_rows - 1, str(ms.n_rows - 1))]])
            try:
                lf = QtGui.QFont()
                lf.setPointSize(9)
                left_axis.setStyle(tickFont=lf)
            except Exception:
                pass

            if self.low_profile_x and not is_last:
                try:
                    plt.setLabel("bottom", "")
                    bax = plt.getAxis("bottom")
                    bax.setStyle(showValues=True, tickLength=0)
                    bax.setTextPen(pg.mkPen(0, 0, 0, 0))
                    bax.setHeight(12)
                except Exception:
                    pass

            line_items, pens = self._create_matrix_render_items(plt, ms)

            cur_line = pg.InfiniteLine(
                angle=90, movable=False, pen=pg.mkPen((255, 255, 255, 120))
            )
            plt.addItem(cur_line)

            sel_region = pg.LinearRegionItem(
                values=(0, 0), brush=pg.mkBrush(100, 200, 255, 40), movable=True
            )
            sel_region.setZValue(-10)
            sel_region.hide()
            plt.addItem(sel_region)
            sel_region.sigRegionChanged.connect(self._on_active_region_dragged)

            self.matrix_plots.append(plt)
            self.matrix_items.append(None)
            self.matrix_cur_lines.append(cur_line)
            self.matrix_sel_regions.append(sel_region)
            self._matrix_line_items.append(line_items)
            self._matrix_pens.append(pens)

            if master_plot is None:
                master_plot = plt
            else:
                plt.setXLink(master_plot)

            vb.sigDragStart.connect(self._on_drag_start)
            vb.sigDragUpdate.connect(self._on_drag_update)
            vb.sigDragFinish.connect(self._on_drag_finish)

        self._apply_custom_plot_heights()
        self._apply_x_range()
        self._update_nav_slider_from_window()
        self._update_hypnogram_extents()
        if self.labels:
            self._sync_label_visuals(force_rebuild=True, refresh_summary=False)
        QtCore.QTimer.singleShot(0, self._align_left_axes)

    def _rebuild_all_plots(self):
        """Rebuild plots including both time series and matrix plots."""
        # Store current state
        old_window_start = self.window_start
        old_cursor = self.cursor_time

        # Clear and rebuild
        self._clear_all_label_visuals()
        self.plot_area.clear()
        self.plots.clear()
        self.curves.clear()
        self.plot_cur_lines.clear()
        self.plot_sel_regions.clear()
        self.dense_plots.clear()
        self.dense_curves.clear()
        self.dense_cur_lines.clear()
        self.dense_sel_regions.clear()
        self.dense_label_regions.clear()
        self.matrix_plots.clear()
        self.matrix_items.clear()
        self.matrix_cur_lines.clear()
        self.matrix_sel_regions.clear()
        self._matrix_line_items.clear()
        self._matrix_pens.clear()

        # Recalculate time range
        t_arrays = [s.t for s in self.series]
        for g in self.dense_groups:
            for s in g.series:
                t_arrays.append(s.t)
        for ms in self.matrix_series:
            if len(ms.timestamps) > 0:
                t_arrays.append(ms.timestamps)
        self.t_global_min, self.t_global_max = nice_time_range(t_arrays)
        self._refresh_low_profile_x()

        # Create all plots
        self._create_all_plots()

        # Restore state
        self.window_start = clamp(
            old_window_start,
            self.t_global_min,
            max(self.t_global_min, self.t_global_max - self.window_len),
        )
        self.cursor_time = old_cursor

        self._apply_x_range()
        self._update_nav_slider_from_window()
        self._sync_label_visuals(force_rebuild=True, refresh_summary=False)
        QtCore.QTimer.singleShot(0, self._align_left_axes)

    def _create_all_plots(self):
        """Create time series and matrix plots in the layout."""
        if self.overlay_mode:
            self._create_overlay_plots()
            return
        master_plot = None
        row_idx = 0
        total_plots = len(self.series) + len(self.dense_groups) + len(self.matrix_series)

        # Create time series plots first
        for idx, s in enumerate(self.series):
            vb = SelectableViewBox()
            vb.sigWheelScrolled.connect(self._page)
            vb.sigWheelSmoothScrolled.connect(self._on_smooth_scroll)
            vb.sigWheelCursorScrolled.connect(self._on_cursor_wheel)

            plt = HoverablePlotItem(viewBox=vb)
            plt.sigHovered.connect(self._on_plot_hovered)
            self.plot_area.addItem(plt, row=row_idx, col=0)

            plt.setLabel("left", s.name)
            is_last = row_idx == total_plots - 1
            plt.setLabel("bottom", "Time", units="s" if is_last else None)
            plt.showGrid(x=True, y=True, alpha=0.15)
            plt.addLegend(offset=(10, 10))
            plt.enableAutoRange("x", False)

            if self.low_profile_x and not is_last:
                try:
                    plt.setLabel("bottom", "")
                    bax = plt.getAxis("bottom")
                    bax.setStyle(showValues=True, tickLength=0)
                    bax.setTextPen(pg.mkPen(0, 0, 0, 0))
                    bax.setHeight(12)
                except Exception:
                    pass

            try:
                lf = QtGui.QFont()
                lf.setPointSize(9)
                plt.getAxis("left").setStyle(tickFont=lf)
            except Exception:
                pass

            if getattr(self, "series_colors", None) and idx < len(self.series_colors):
                pen_color = self.series_colors[idx]
            else:
                pen_color = (255, 255, 255)
            pen = pg.mkPen(pen_color, width=1)
            curve = pg.PlotDataItem([], [], pen=pen, antialias=False)
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)
            plt.addItem(curve)

            cur_line = pg.InfiniteLine(
                angle=90, movable=False, pen=pg.mkPen((255, 255, 255, 120))
            )
            plt.addItem(cur_line)

            sel_region = pg.LinearRegionItem(
                values=(0, 0), brush=pg.mkBrush(100, 200, 255, 40), movable=True
            )
            sel_region.setZValue(-10)
            sel_region.hide()
            plt.addItem(sel_region)
            sel_region.sigRegionChanged.connect(self._on_active_region_dragged)

            self.plots.append(plt)
            self.curves.append(curve)
            self.plot_cur_lines.append(cur_line)
            self.plot_sel_regions.append(sel_region)

            if master_plot is None:
                master_plot = plt
            else:
                plt.setXLink(master_plot)

            vb.sigDragStart.connect(self._on_drag_start)
            vb.sigDragUpdate.connect(self._on_drag_update)
            vb.sigDragFinish.connect(self._on_drag_finish)

            if self.fixed_scale:
                try:
                    y = np.asarray(s.y, dtype=float)
                    lo = float(np.nanpercentile(y, 1.0))
                    hi = float(np.nanpercentile(y, 99.0))
                    if not np.isfinite(lo) or not np.isfinite(hi):
                        raise ValueError("non-finite percentiles")
                    if hi <= lo:
                        hi = lo + 1.0
                    pad = 0.05 * (hi - lo)
                    plt.enableAutoRange("y", False)
                    plt.setYRange(lo - pad, hi + pad, padding=0)
                except Exception:
                    plt.enableAutoRange("y", False)
            else:
                plt.enableAutoRange("y", True)

            row_idx += 1

        # Create dense plots
        for gi in range(len(self.dense_groups)):
            plt = self._create_dense_plot(gi, master_plot=master_plot)
            self.plot_area.addItem(plt, row=row_idx, col=0)
            is_last = row_idx == total_plots - 1
            plt.setLabel("bottom", "Time", units="s" if is_last else None)
            if self.low_profile_x and not is_last:
                try:
                    plt.setLabel("bottom", "")
                    bax = plt.getAxis("bottom")
                    bax.setStyle(showValues=True, tickLength=0)
                    bax.setTextPen(pg.mkPen(0, 0, 0, 0))
                    bax.setHeight(12)
                except Exception:
                    pass
            if master_plot is None:
                master_plot = plt
            row_idx += 1

        # Create matrix plots
        for idx, ms in enumerate(self.matrix_series):
            vb = SelectableViewBox()
            vb.sigWheelScrolled.connect(self._page)
            vb.sigWheelSmoothScrolled.connect(self._on_smooth_scroll)
            vb.sigWheelCursorScrolled.connect(self._on_cursor_wheel)

            plt = HoverablePlotItem(viewBox=vb)
            plt.sigHovered.connect(self._on_plot_hovered)
            self.plot_area.addItem(plt, row=row_idx, col=0)

            plt.setLabel("left", ms.name)
            is_last = row_idx == total_plots - 1
            plt.setLabel("bottom", "Time", units="s" if is_last else None)

            # Matrix plots: no horizontal grid, minimal y-axis
            plt.showGrid(x=True, y=False, alpha=0.15)
            plt.enableAutoRange("x", False)
            plt.enableAutoRange("y", False)

            # Set Y range to show all rows
            plt.setYRange(-0.5, ms.n_rows - 0.5, padding=0.02)

            # Configure Y-axis: only show min and max tick values
            left_axis = plt.getAxis("left")
            left_axis.setTicks([[(0, "0"), (ms.n_rows - 1, str(ms.n_rows - 1))]])
            try:
                lf = QtGui.QFont()
                lf.setPointSize(9)
                left_axis.setStyle(tickFont=lf)
            except Exception:
                pass

            if self.low_profile_x and not is_last:
                try:
                    plt.setLabel("bottom", "")
                    bax = plt.getAxis("bottom")
                    bax.setStyle(showValues=True, tickLength=0)
                    bax.setTextPen(pg.mkPen(0, 0, 0, 0))
                    bax.setHeight(12)
                except Exception:
                    pass

            line_items, pens = self._create_matrix_render_items(plt, ms)

            cur_line = pg.InfiniteLine(
                angle=90, movable=False, pen=pg.mkPen((255, 255, 255, 120))
            )
            plt.addItem(cur_line)

            sel_region = pg.LinearRegionItem(
                values=(0, 0), brush=pg.mkBrush(100, 200, 255, 40), movable=True
            )
            sel_region.setZValue(-10)
            sel_region.hide()
            plt.addItem(sel_region)
            sel_region.sigRegionChanged.connect(self._on_active_region_dragged)

            self.matrix_plots.append(plt)
            self.matrix_items.append(None)
            self.matrix_cur_lines.append(cur_line)
            self.matrix_sel_regions.append(sel_region)
            self._matrix_line_items.append(line_items)
            self._matrix_pens.append(pens)

            if master_plot is None:
                master_plot = plt
            else:
                plt.setXLink(master_plot)

            vb.sigDragStart.connect(self._on_drag_start)
            vb.sigDragUpdate.connect(self._on_drag_update)
            vb.sigDragFinish.connect(self._on_drag_finish)

            row_idx += 1

        # Apply custom plot heights (includes matrix row heights logic)
        self._apply_custom_plot_heights()
        self._setup_dense_vscrollbar()

    def _create_overlay_plots(self):
        """Create plots for overlay mode: multiple curves per subplot."""
        master_plot = None
        row_idx = 0
        total_plots = len(self.overlay_groups) + len(self.matrix_series)
        self._plot_to_curves = []

        for grp_idx, group in enumerate(self.overlay_groups):
            vb = SelectableViewBox()
            vb.sigWheelScrolled.connect(self._page)
            vb.sigWheelSmoothScrolled.connect(self._on_smooth_scroll)
            vb.sigWheelCursorScrolled.connect(self._on_cursor_wheel)

            plt = HoverablePlotItem(viewBox=vb)
            plt.sigHovered.connect(self._on_plot_hovered)
            self.plot_area.addItem(plt, row=row_idx, col=0)

            plt.setLabel("left", group.label)
            is_last = row_idx == total_plots - 1
            plt.setLabel("bottom", "Time", units="s" if is_last else None)
            plt.showGrid(x=True, y=True, alpha=0.15)
            plt.addLegend(offset=(10, 10))
            plt.enableAutoRange("x", False)

            if self.low_profile_x and not is_last:
                try:
                    plt.setLabel("bottom", "")
                    bax = plt.getAxis("bottom")
                    bax.setStyle(showValues=True, tickLength=0)
                    bax.setTextPen(pg.mkPen(0, 0, 0, 0))
                    bax.setHeight(12)
                except Exception:
                    pass

            try:
                lf = QtGui.QFont()
                lf.setPointSize(9)
                plt.getAxis("left").setStyle(tickFont=lf)
            except Exception:
                pass

            # Create one curve per trace in this group
            group_curves = []
            for tr in group.traces:
                pen_color = self.overlay_colors[tr.source_idx]
                pen = pg.mkPen(pen_color, width=1)
                curve = pg.PlotDataItem([], [], pen=pen, name=tr.name, antialias=False)
                curve.setDownsampling(auto=True, method="peak")
                curve.setClipToView(True)
                plt.addItem(curve)
                group_curves.append(curve)
                # Also maintain flat curves list for compat
                self.curves.append(curve)
            self._plot_to_curves.append(group_curves)

            # Cursor line and selection region (one per subplot)
            cur_line = pg.InfiniteLine(
                angle=90, movable=False, pen=pg.mkPen((255, 255, 255, 120))
            )
            plt.addItem(cur_line)

            sel_region = pg.LinearRegionItem(
                values=(0, 0), brush=pg.mkBrush(100, 200, 255, 40), movable=True
            )
            sel_region.setZValue(-10)
            sel_region.hide()
            plt.addItem(sel_region)
            sel_region.sigRegionChanged.connect(self._on_active_region_dragged)

            self.plots.append(plt)
            self.plot_cur_lines.append(cur_line)
            self.plot_sel_regions.append(sel_region)

            if master_plot is None:
                master_plot = plt
            else:
                plt.setXLink(master_plot)

            vb.sigDragStart.connect(self._on_drag_start)
            vb.sigDragUpdate.connect(self._on_drag_update)
            vb.sigDragFinish.connect(self._on_drag_finish)

            # Fixed scale: compute Y range from ALL traces in group
            if self.fixed_scale:
                try:
                    all_y = np.concatenate(
                        [np.asarray(tr.y, dtype=float) for tr in group.traces]
                    )
                    lo = float(np.nanpercentile(all_y, 1.0))
                    hi = float(np.nanpercentile(all_y, 99.0))
                    if not np.isfinite(lo) or not np.isfinite(hi):
                        raise ValueError("non-finite percentiles")
                    if hi <= lo:
                        hi = lo + 1.0
                    pad = 0.05 * (hi - lo)
                    plt.enableAutoRange("y", False)
                    plt.setYRange(lo - pad, hi + pad, padding=0)
                except Exception:
                    plt.enableAutoRange("y", False)
            else:
                plt.enableAutoRange("y", True)

            row_idx += 1

        # Create matrix plots (same as _create_all_plots)
        for idx, ms in enumerate(self.matrix_series):
            vb = SelectableViewBox()
            vb.sigWheelScrolled.connect(self._page)
            vb.sigWheelSmoothScrolled.connect(self._on_smooth_scroll)
            vb.sigWheelCursorScrolled.connect(self._on_cursor_wheel)

            plt = HoverablePlotItem(viewBox=vb)
            plt.sigHovered.connect(self._on_plot_hovered)
            self.plot_area.addItem(plt, row=row_idx, col=0)

            plt.setLabel("left", ms.name)
            is_last = row_idx == total_plots - 1
            plt.setLabel("bottom", "Time", units="s" if is_last else None)
            plt.showGrid(x=True, y=True, alpha=0.15)
            plt.enableAutoRange("x", False)
            plt.enableAutoRange("y", False)

            unique_y = np.unique(ms.yvals)
            if len(unique_y) > 0:
                y_min = float(unique_y[0]) - 0.5
                y_max = float(unique_y[-1]) + 0.5
                plt.setYRange(y_min, y_max, padding=0)

            line_items, pens = self._create_matrix_render_items(plt, ms)

            cur_line = pg.InfiniteLine(
                angle=90, movable=False, pen=pg.mkPen((255, 255, 255, 120))
            )
            plt.addItem(cur_line)

            sel_region = pg.LinearRegionItem(
                values=(0, 0), brush=pg.mkBrush(100, 200, 255, 40), movable=True
            )
            sel_region.setZValue(-10)
            sel_region.hide()
            plt.addItem(sel_region)
            sel_region.sigRegionChanged.connect(self._on_active_region_dragged)

            self.matrix_plots.append(plt)
            self.matrix_items.append(None)
            self.matrix_cur_lines.append(cur_line)
            self.matrix_sel_regions.append(sel_region)
            self._matrix_line_items.append(line_items)
            self._matrix_pens.append(pens)

            if master_plot is None:
                master_plot = plt
            else:
                plt.setXLink(master_plot)

            vb.sigDragStart.connect(self._on_drag_start)
            vb.sigDragUpdate.connect(self._on_drag_update)
            vb.sigDragFinish.connect(self._on_drag_finish)

            row_idx += 1

        self._apply_custom_plot_heights()

    def _build_matrix_alpha_pens(self, ms: MatrixSeries) -> list[QtGui.QPen]:
        r, g, b = ms.color
        return [
            pg.mkPen(
                color=(
                    r,
                    g,
                    b,
                    int((alevel / (MATRIX_ALPHA_LEVEL_COUNT - 1)) * 255),
                ),
                width=self.matrix_event_thickness,
            )
            for alevel in range(MATRIX_ALPHA_LEVEL_COUNT)
        ]

    def _create_matrix_render_items(
        self, plt: pg.PlotItem, ms: MatrixSeries
    ) -> tuple[list[pg.PlotDataItem], list[QtGui.QPen]]:
        pens = self._build_matrix_alpha_pens(ms)
        line_items: list[pg.PlotDataItem] = []
        for pen in pens:
            line_item = pg.PlotDataItem(
                [], [], pen=pen, connect="pairs", antialias=False
            )
            plt.addItem(line_item)
            line_items.append(line_item)
        return line_items, pens

    def _refresh_matrix_pen_cache(self) -> None:
        for midx, ms in enumerate(self.matrix_series):
            if midx >= len(self._matrix_line_items):
                break
            pens = self._build_matrix_alpha_pens(ms)
            if midx < len(self._matrix_pens):
                self._matrix_pens[midx] = pens
            else:
                self._matrix_pens.append(pens)
            for line_item, pen in zip(self._matrix_line_items[midx], pens):
                line_item.setPen(pen)

    def _is_trace_plot_visible(self, plot_idx: int) -> bool:
        return (
            not hasattr(self, "trace_visible")
            or plot_idx >= len(self.trace_visible)
            or self.trace_visible[plot_idx]
        )

    def _is_matrix_plot_visible(self, plot_idx: int) -> bool:
        return plot_idx >= len(self.matrix_visible) or self.matrix_visible[plot_idx]

    def _matrix_segment_for_window(
        self, ms: MatrixSeries, t0: float, t1: float, max_events: int = 10000
    ):
        """
        Return event data for the [t0, t1] window, limited to max_events for performance.
        Returns (timestamps, yvals, alphas) arrays for events in the window.
        """
        if t1 <= t0 or len(ms.timestamps) == 0:
            return np.empty(0), np.empty(0), np.empty(0)

        # Binary search for window bounds (timestamps are sorted)
        i0 = np.searchsorted(ms.timestamps, t0, side="left")
        i1 = np.searchsorted(ms.timestamps, t1, side="right")

        if i0 >= i1:
            return np.empty(0), np.empty(0), np.empty(0)

        # Slice to window
        ts = ms.timestamps[i0:i1]
        ys = ms.yvals[i0:i1]
        als = ms.alphas[i0:i1]

        # Downsample if too many events (uniform sampling)
        if len(ts) > max_events:
            step = len(ts) // max_events
            ts = ts[::step]
            ys = ys[::step]
            als = als[::step]

        return ts, ys, als

    # ------------------------------------------------------------------ dense
    def _dense_visible_indices(self, group_idx: int) -> list[int]:
        """Return indices into group.series for visible traces."""
        group = self.dense_groups[group_idx]
        return [
            i
            for i in range(0, len(group.series), group.step)
            if i not in group.hidden_traces
        ]

    def _dense_offsets(self, group_idx: int) -> np.ndarray:
        """Compute Y-offsets for visible traces in a dense group."""
        group = self.dense_groups[group_idx]
        visible = self._dense_visible_indices(group_idx)
        if group.order_values is not None and len(group.order_values) == len(group.series):
            offsets = group.order_values[visible].astype(float)
        else:
            offsets = np.arange(len(visible), dtype=float)
        return offsets

    def _dense_offset_margin(self, offsets: np.ndarray) -> float:
        """Compute a reasonable margin for dense plot Y-range."""
        if len(offsets) < 2:
            return 1.0
        gaps = np.diff(offsets)
        return float(np.median(gaps)) * 0.5 if len(gaps) > 0 else 1.0

    def _create_dense_plot(self, group_idx: int, master_plot=None):
        """Create a single dense (EEG-style) PlotItem for a DenseGroup."""
        group = self.dense_groups[group_idx]
        visible = self._dense_visible_indices(group_idx)
        offsets = self._dense_offsets(group_idx)
        visible_labels = [group.trace_labels[i] for i in visible]

        vb = DenseViewBox()
        vb.sigWheelScrolled.connect(self._page)
        vb.sigWheelSmoothScrolled.connect(self._on_smooth_scroll)
        vb.sigWheelCursorScrolled.connect(self._on_cursor_wheel)
        vb.sigWheelGainAdjust.connect(
            lambda d: self._adjust_dense_gain(1.2 if d > 0 else 1 / 1.2)
        )
        vb.sigWheelVerticalSmooth.connect(self._on_dense_vertical_smooth)

        plt = HoverablePlotItem(viewBox=vb)
        plt.sigHovered.connect(self._on_plot_hovered)

        plt.setLabel("left", group.name)
        plt.showGrid(x=True, y=False, alpha=0.15)
        plt.enableAutoRange("x", False)
        plt.enableAutoRange("y", False)

        # Y-axis ticks: trace labels at offset positions
        left_axis = plt.getAxis("left")
        tick_list = [(float(o), lbl) for o, lbl in zip(offsets, visible_labels)]
        left_axis.setTicks([tick_list])
        try:
            lf = QtGui.QFont()
            lf.setPointSize(8)
            left_axis.setStyle(tickFont=lf)
        except Exception:
            pass

        # Create curves
        pen = pg.mkPen((200, 200, 200), width=1)
        curves: list[pg.PlotCurveItem] = []
        for _ in visible:
            curve = pg.PlotCurveItem(pen=pen, antialias=False)
            curve.setZValue(5)
            plt.addItem(curve)
            curves.append(curve)

        # Cursor line
        cur_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen((255, 255, 255, 120))
        )
        plt.addItem(cur_line)

        # Selection region for labeling
        sel_region = pg.LinearRegionItem(
            values=(0, 0), brush=pg.mkBrush(100, 200, 255, 40), movable=True
        )
        sel_region.setZValue(-10)
        sel_region.hide()
        plt.addItem(sel_region)
        sel_region.sigRegionChanged.connect(self._on_active_region_dragged)

        # Set Y-range: show traces_per_page traces, or all if not set
        if len(offsets) > 0:
            margin = self._dense_offset_margin(offsets)
            tpp = group.traces_per_page
            if tpp is not None and tpp < len(offsets):
                # Show the first tpp traces (lowest offsets)
                page_max = float(offsets[min(tpp, len(offsets)) - 1])
                plt.setYRange(
                    float(offsets[0]) - margin,
                    page_max + margin,
                    padding=0,
                )
            else:
                plt.setYRange(
                    float(offsets.min()) - margin,
                    float(offsets.max()) + margin,
                    padding=0,
                )

        # X-link
        if master_plot is not None:
            plt.setXLink(master_plot)

        # Connect drag signals for labeling
        vb.sigDragStart.connect(self._on_drag_start)
        vb.sigDragUpdate.connect(self._on_drag_update)
        vb.sigDragFinish.connect(self._on_drag_finish)

        self.dense_plots.append(plt)
        self.dense_curves.append(curves)
        self.dense_cur_lines.append(cur_line)
        self.dense_sel_regions.append(sel_region)
        self.dense_label_regions.append([])

        return plt

    def _rebuild_dense_curves(self, group_idx: int):
        """Rebuild curve items for a dense group (after step/visibility change)."""
        group = self.dense_groups[group_idx]
        plt = self.dense_plots[group_idx]

        # Remove old curves
        for curve in self.dense_curves[group_idx]:
            plt.removeItem(curve)

        visible = self._dense_visible_indices(group_idx)
        offsets = self._dense_offsets(group_idx)
        visible_labels = [group.trace_labels[i] for i in visible]

        pen = pg.mkPen((200, 200, 200), width=1)
        curves: list[pg.PlotCurveItem] = []
        for _ in visible:
            curve = pg.PlotCurveItem(pen=pen, antialias=False)
            curve.setZValue(5)
            plt.addItem(curve)
            curves.append(curve)
        self.dense_curves[group_idx] = curves

        # Update Y-axis ticks
        left_axis = plt.getAxis("left")
        tick_list = [(float(o), lbl) for o, lbl in zip(offsets, visible_labels)]
        left_axis.setTicks([tick_list])

        # Keep the same number of traces visible as before the rebuild
        if len(offsets) > 0:
            old_yrange = plt.getViewBox().viewRange()[1]
            old_span = old_yrange[1] - old_yrange[0]
            old_center = (old_yrange[0] + old_yrange[1]) / 2.0
            # Clamp center to valid offset range
            center = max(float(offsets.min()), min(float(offsets.max()), old_center))
            half = old_span / 2.0
            plt.setYRange(center - half, center + half, padding=0)

    def _refresh_dense_curves(self):
        """Update dense plot curves for the current window."""
        if not self.dense_groups:
            return
        t0, t1 = self.window_start, self.window_start + self.window_len
        max_pts = self._target_pts()
        for gi, group in enumerate(self.dense_groups):
            visible = self._dense_visible_indices(gi)
            offsets = self._dense_offsets(gi)
            curves = self.dense_curves[gi]
            means = self._dense_means[gi]
            for li, (si, offset) in enumerate(zip(visible, offsets)):
                s = group.series[si]
                tx, yx = segment_for_window(s.t, s.y, t0, t1, max_pts=max_pts)
                yx_display = (yx - means[si]) * group.gain + offset
                curves[li].setData(tx, yx_display, _callSync="off")

    def _vertical_page(self, direction: int):
        """Scroll the plot scroll area up/down by one page."""
        if not hasattr(self, "plot_scroll_area"):
            return
        sb = self.plot_scroll_area.verticalScrollBar()
        page = self.plot_scroll_area.viewport().height()
        sb.setValue(sb.value() + direction * page)

    def _update_plot_area_height(self):
        """Set plot_area minimum height so the scroll area shows a scrollbar when needed."""
        visible = self._get_visible_subplot_order()
        n = len(visible)
        if n == 0:
            return
        # Desired height = n * trace_height_px, but at least the scroll area height
        scroll_h = self.plot_scroll_area.viewport().height() if hasattr(self, "plot_scroll_area") else 600
        desired = n * self.trace_height_px
        if desired > scroll_h:
            self.plot_area.setMinimumHeight(desired)
        else:
            self.plot_area.setMinimumHeight(0)

    # ---- Dense vertical scrollbar ------------------------------------------
    def _setup_dense_vscrollbar(self):
        """Configure the dense vertical scrollbar based on the first dense group."""
        if not self.dense_groups or not self.dense_plots:
            self.dense_vscrollbar.hide()
            return
        gi = 0
        offsets = self._dense_offsets(gi)
        if len(offsets) < 2:
            self.dense_vscrollbar.hide()
            return
        group = self.dense_groups[gi]
        margin = self._dense_offset_margin(offsets)
        total_min = float(offsets.min()) - margin
        total_max = float(offsets.max()) + margin

        plt = self.dense_plots[gi]
        y_range = plt.getViewBox().viewRange()[1]
        visible_span = y_range[1] - y_range[0]

        scale = 100.0
        sb = self.dense_vscrollbar
        sb.blockSignals(True)
        sb.setMinimum(int(total_min * scale))
        sb.setMaximum(int((total_max - visible_span) * scale))
        sb.setPageStep(int(visible_span * scale))
        sb.setSingleStep(int(scale))
        # When descending, scrollbar top = high Y, so invert
        if group.descending:
            sb.setValue(sb.maximum() - int(y_range[0] * scale) + sb.minimum())
        else:
            sb.setValue(int(y_range[0] * scale))
        sb.blockSignals(False)
        sb.show()
        self._dense_vscroll_inverted = group.descending

    def _on_dense_vscrollbar_changed(self, value: int):
        """Handle dense vertical scrollbar value change."""
        if not self.dense_groups or not self.dense_plots:
            return
        gi = 0
        plt = self.dense_plots[gi]
        vb = plt.getViewBox()
        y_range = vb.viewRange()[1]
        visible_span = y_range[1] - y_range[0]
        sb = self.dense_vscrollbar
        if getattr(self, "_dense_vscroll_inverted", False):
            new_min = (sb.maximum() - value + sb.minimum()) / 100.0
        else:
            new_min = value / 100.0
        vb.setYRange(new_min, new_min + visible_span, padding=0)

    def _sync_dense_vscrollbar_from_yrange(self):
        """Update scrollbar position to match the dense plot's current Y-range."""
        if not self.dense_groups or not self.dense_plots:
            return
        if not self.dense_vscrollbar.isVisible():
            return
        gi = 0
        plt = self.dense_plots[gi]
        y_range = plt.getViewBox().viewRange()[1]
        sb = self.dense_vscrollbar
        sb.blockSignals(True)
        if getattr(self, "_dense_vscroll_inverted", False):
            sb.setValue(sb.maximum() - int(y_range[0] * 100.0) + sb.minimum())
        else:
            sb.setValue(int(y_range[0] * 100.0))
        sb.blockSignals(False)

    def _dense_vertical_page(self, direction: int):
        """Page a dense plot vertically. Uses hovered plot, or first dense group."""
        if not self.dense_groups or not self.dense_plots:
            return False
        # Find which dense plot to page: hovered, or default to first
        gi = 0
        if self.hovered_plot is not None:
            found = False
            for i, plt in enumerate(self.dense_plots):
                if plt is self.hovered_plot:
                    gi = i
                    found = True
                    break
            if not found and self.hovered_plot in self.plots:
                # Hovered plot is a stacked subplot — let caller handle it
                return False

        plt = self.dense_plots[gi]
        offsets = self._dense_offsets(gi)
        if len(offsets) < 2:
            return True
        vb = plt.getViewBox()
        y_range = vb.viewRange()[1]
        visible_span = y_range[1] - y_range[0]
        scroll_amount = visible_span * direction
        vb.setYRange(
            y_range[0] + scroll_amount,
            y_range[1] + scroll_amount,
            padding=0,
        )
        self._sync_dense_vscrollbar_from_yrange()
        return True

    def _on_dense_vertical_smooth(self, direction: int):
        """Shift+Alt+wheel: smooth vertical scroll on the first dense plot."""
        if not self.dense_groups or not self.dense_plots:
            return
        gi = 0
        plt = self.dense_plots[gi]
        offsets = self._dense_offsets(gi)
        if len(offsets) < 2:
            return
        vb = plt.getViewBox()
        y_range = vb.viewRange()[1]
        # Scroll by ~3 traces per notch
        gap = float(np.median(np.diff(offsets)))
        scroll_amount = gap * 3 * direction
        vb.setYRange(
            y_range[0] + scroll_amount,
            y_range[1] + scroll_amount,
            padding=0,
        )
        self._sync_dense_vscrollbar_from_yrange()

    def _adjust_dense_gain(self, factor: float):
        """Scale gain for all dense groups (or hovered group) by *factor*."""
        if not self.dense_groups:
            return
        for group in self.dense_groups:
            group.gain = max(0.001, group.gain * factor)
        self._refresh_dense_curves()
        self._update_status(f"Dense gain: {self.dense_groups[0].gain:.2f}x")

    def _refresh_matrix_plots(self):
        """Update matrix raster plots for current window."""
        if not self.matrix_series:
            return

        t0 = self.window_start
        t1 = self.window_start + self.window_len
        height = self.matrix_event_height

        for midx, (ms, plt) in enumerate(zip(self.matrix_series, self.matrix_plots)):
            if not self._is_matrix_plot_visible(midx):
                continue

            if midx >= len(self._matrix_line_items):
                continue

            ts, ys, als = self._matrix_segment_for_window(ms, t0, t1)
            line_items = self._matrix_line_items[midx]
            if not line_items:
                continue

            if len(ts) == 0:
                for line_item in line_items:
                    line_item.setData([], [], _callSync="off")
                continue

            # Calculate Y positions for each event
            y_centers = ys.astype(float) + 0.5
            y_bottoms = y_centers - height
            y_tops = y_centers + height

            # Apply brightness multiplier to alphas (clamped to 0-1)
            brightness = getattr(self, "matrix_brightness", 1.0)
            adjusted_als = np.clip(als * brightness, 0.0, 1.0)

            # Group by alpha levels (quantize to 11 levels 0-10) for efficiency
            alpha_levels = np.round(adjusted_als * (MATRIX_ALPHA_LEVEL_COUNT - 1)).astype(
                int
            )

            for alevel, line_item in enumerate(line_items):
                mask = alpha_levels == alevel
                if not np.any(mask):
                    line_item.setData([], [], _callSync="off")
                    continue

                indices = np.where(mask)[0]
                seg_x = np.repeat(ts[indices], 2)
                seg_y = np.empty(2 * len(indices))
                seg_y[0::2] = y_bottoms[indices]
                seg_y[1::2] = y_tops[indices]
                line_item.setData(seg_x, seg_y, _callSync="off")

    # ---------- Video & Static Image ----------
    def _load_video_data(self, vpath, ft_path):
        self._stop_playback_if_playing()
        self._video_is_open = False
        self.video_frame_times = None
        self._video_requested_frame_idx = None

        if not os.path.exists(vpath) or not os.path.exists(ft_path):
            QtWidgets.QMessageBox.warning(
                self, "File Not Found", "Video or frame times file does not exist."
            )
            return

        QtCore.QMetaObject.invokeMethod(
            self._video_worker,
            "open",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, vpath),
        )
        try:
            ft = np.load(ft_path).astype(float)
            if ft.ndim != 1:
                raise ValueError("frame_times.npy must be 1-D")
            self.video_frame_times = ft
            self._update_status(f"Loaded frame_times ({len(ft)} frames).")
            self._request_initial_frame()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Frame times error", str(e))
            self.video_frame_times = None

    def _on_load_video(self):
        if cv2 is None:
            QtWidgets.QMessageBox.warning(
                self, "Video", "OpenCV (cv2) is not installed."
            )
            return
        vpath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select video file",
            filter="Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)",
        )
        if not vpath:
            return

        ft_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select frame_times.npy"
        )
        if not ft_path:
            return

        self._load_video_data(vpath, ft_path)

    # ---------- Video2 ----------
    def _load_video2_data(self, vpath, ft_path):
        self._stop_playback_if_playing()
        self._video2_is_open = False
        self.video2_frame_times = None
        self._video2_requested_frame_idx = None

        if not os.path.exists(vpath) or not os.path.exists(ft_path):
            QtWidgets.QMessageBox.warning(
                self, "File Not Found", "Video2 or frame times file does not exist."
            )
            return

        QtCore.QMetaObject.invokeMethod(
            self._video2_worker,
            "open",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, vpath),
        )
        try:
            ft = np.load(ft_path).astype(float)
            if ft.ndim != 1:
                raise ValueError("frame_times.npy must be 1-D")
            self.video2_frame_times = ft
            self._update_status(f"Loaded frame_times2 ({len(ft)} frames).")
            self.label_summary_panel.hide()
            self.video2_label.show()
            self._request_initial_frame()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Frame times 2 error", str(e))
            self.video2_frame_times = None

    def _on_video2_opened(self, ok, msg):
        if not ok:
            self._video2_is_open = False
            QtWidgets.QMessageBox.warning(self, "Video2", msg or "Failed to open.")
        else:
            self._video2_is_open = True
            self._video2_requested_frame_idx = None
            self.label_summary_panel.hide()
            self.video2_label.show()
            self._request_initial_frame()

    # ---------- Video3 ----------
    def _load_video3_data(self, vpath, ft_path):
        self._stop_playback_if_playing()
        self._video3_is_open = False
        self.video3_frame_times = None
        self._video3_requested_frame_idx = None

        if not os.path.exists(vpath) or not os.path.exists(ft_path):
            QtWidgets.QMessageBox.warning(
                self, "File Not Found", "Video3 or frame times file does not exist."
            )
            return

        QtCore.QMetaObject.invokeMethod(
            self._video3_worker,
            "open",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, vpath),
        )
        try:
            ft = np.load(ft_path).astype(float)
            if ft.ndim != 1:
                raise ValueError("frame_times.npy must be 1-D")
            self.video3_frame_times = ft
            self._update_status(f"Loaded frame_times3 ({len(ft)} frames).")
            self.label_summary_panel.hide()
            self.video3_label.show()
            self._request_initial_frame()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Frame times 3 error", str(e))
            self.video3_frame_times = None

    def _on_video3_opened(self, ok, msg):
        if not ok:
            self._video3_is_open = False
            QtWidgets.QMessageBox.warning(self, "Video3", msg or "Failed to open.")
        else:
            self._video3_is_open = True
            self._video3_requested_frame_idx = None
            self.video3_label.show()
            self._request_initial_frame()

    def _on_video_opened(self, ok, msg):
        if not ok:
            self._video_is_open = False
            QtWidgets.QMessageBox.warning(self, "Video", msg or "Failed to open.")
        else:
            self._video_is_open = True
            self._video_requested_frame_idx = None
            self._request_initial_frame()

    def _request_initial_frame(self):
        if self._video_is_open and self.video_frame_times is not None:
            self._set_cursor_time(self.cursor_time, update_slider=True)
        if self._video2_is_open and self.video2_frame_times is not None:
            self._set_cursor_time(self.cursor_time, update_slider=False)
        if self._video3_is_open and self.video3_frame_times is not None:
            self._set_cursor_time(self.cursor_time, update_slider=False)

    def _request_video_frame(
        self,
        *,
        frame_times: np.ndarray | None,
        worker: VideoWorker,
        requested_attr: str,
        t: float,
    ) -> None:
        if frame_times is None or len(frame_times) == 0:
            return

        idx = find_nearest_frame(frame_times, t)
        if getattr(self, requested_attr) == idx:
            return

        setattr(self, requested_attr, idx)
        QtCore.QMetaObject.invokeMethod(
            worker,
            "requestFrame",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(int, int(idx)),
        )

    def _schedule_deferred_view_refresh(self, *, update_nav_slider: bool) -> None:
        self._deferred_view_refresh_needs_nav_slider = (
            self._deferred_view_refresh_needs_nav_slider or update_nav_slider
        )
        if not self._deferred_view_refresh_timer.isActive():
            self._deferred_view_refresh_timer.start(0)

    def _flush_deferred_view_refresh(self) -> None:
        update_nav_slider = self._deferred_view_refresh_needs_nav_slider
        self._deferred_view_refresh_needs_nav_slider = False
        self._apply_x_range_core()
        if update_nav_slider:
            self._update_nav_slider_from_window()

    def _on_frame_ready(self, idx, qimg):
        if idx != self._video_requested_frame_idx:
            return
        if qimg is None or qimg.isNull():
            return
        pix = QtGui.QPixmap.fromImage(qimg)
        if pix.isNull():
            return

        self.last_video_pixmap = pix
        self._rescale_video_frame()

    def _on_frame2_ready(self, idx, qimg):
        if idx != self._video2_requested_frame_idx:
            return
        if qimg is None or qimg.isNull():
            return
        pix = QtGui.QPixmap.fromImage(qimg)
        if pix.isNull():
            return
        self.last_video2_pixmap = pix
        self._rescale_video2_frame()

    def _on_frame3_ready(self, idx, qimg):
        if idx != self._video3_requested_frame_idx:
            return
        if qimg is None or qimg.isNull():
            return
        pix = QtGui.QPixmap.fromImage(qimg)
        if pix.isNull():
            return
        self.last_video3_pixmap = pix
        self._rescale_video3_frame()

    def _rescale_video_frame(self):
        if self.last_video_pixmap:
            scaled = self.last_video_pixmap.scaled(
                self.video_label.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self.video_label.setPixmap(scaled)

    def _rescale_video2_frame(self):
        if self.last_video2_pixmap:
            scaled = self.last_video2_pixmap.scaled(
                self.video2_label.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self.video2_label.setPixmap(scaled)

    def _rescale_video3_frame(self):
        if self.last_video3_pixmap:
            scaled = self.last_video3_pixmap.scaled(
                self.video3_label.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self.video3_label.setPixmap(scaled)

    def _on_splitter_moved(self, pos, index):
        self._rescale_video_frame()
        self._rescale_video2_frame()
        self._rescale_video3_frame()

    # ---------- Selection / labeling ----------
    def _show_y_axis_dialog(self):
        if not self.series:
            QtWidgets.QMessageBox.information(
                self, "Y-Axis Controls", "Load time series data first."
            )
            return
        if self.y_axis_dialog is not None:
            self.y_axis_dialog.deleteLater()

        self.y_axis_dialog = YAxisControlsDialog(self)
        self.y_axis_dialog.show()
        self.y_axis_dialog.raise_()
        self.y_axis_dialog.activateWindow()

    def _show_dense_controls_dialog(self):
        if not self.dense_groups:
            QtWidgets.QMessageBox.information(
                self, "Dense View Controls", "No dense trace groups loaded."
            )
            return
        if hasattr(self, "_dense_ctrl_dialog") and self._dense_ctrl_dialog is not None:
            self._dense_ctrl_dialog.deleteLater()
        self._dense_ctrl_dialog = DenseViewControlsDialog(self)
        self._dense_ctrl_dialog.show()
        self._dense_ctrl_dialog.raise_()
        self._dense_ctrl_dialog.activateWindow()

    def _on_plot_hovered(self, plot, is_hovered):
        if is_hovered:
            self.hovered_plot = plot
        else:
            if self.hovered_plot is plot:
                self.hovered_plot = None

    def _on_drag_start(self, x):
        self._stop_playback_if_playing()
        self._select_start = x
        self._select_end = x
        # Determine if this drag is a zoom gesture (Shift held)
        try:
            mods = QtWidgets.QApplication.keyboardModifiers()
        except Exception:
            mods = QtCore.Qt.KeyboardModifier.NoModifier
        self._is_zoom_drag = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
        self._show_active_selection()

    def _on_drag_update(self, x):
        self._select_end = x
        self._show_active_selection()

    def _on_drag_finish(self, x):
        self._select_end = x
        # If this was a Shift+drag, zoom to the selected time range
        if self._is_zoom_drag:
            a = float(min(self._select_start, self._select_end))
            b = float(max(self._select_start, self._select_end))
            if b > a:
                new_len = max(0.1, b - a)
                self.window_len = new_len
                self.window_start = clamp(
                    a,
                    self.t_global_min,
                    max(self.t_global_min, self.t_global_max - self.window_len),
                )
                # Sync UI without triggering change handler
                self.window_spin.blockSignals(True)
                self.window_spin.setValue(self.window_len)
                self.window_spin.blockSignals(False)
                self._apply_x_range()
                self._update_nav_slider_from_window()
            self._is_zoom_drag = False
            self._clear_selection()
            return
        self._show_active_selection(final=True)

    def _on_active_region_dragged(self):
        self._stop_playback_if_playing()
        # Get region from whichever selection region was dragged
        if self.plot_sel_regions:
            a, b = self.plot_sel_regions[0].getRegion()
            self._select_start, self._select_end = float(a), float(b)
        elif self.matrix_sel_regions:
            a, b = self.matrix_sel_regions[0].getRegion()
            self._select_start, self._select_end = float(a), float(b)

    def _show_active_selection(self, final=False):
        if self._select_start is None or self._select_end is None:
            for r in self.plot_sel_regions:
                r.hide()
            for r in self.dense_sel_regions:
                r.hide()
            for r in self.matrix_sel_regions:
                r.hide()
            return
        a = min(self._select_start, self._select_end)
        b = max(self._select_start, self._select_end)
        for r in self.plot_sel_regions:
            r.setRegion((a, b))
            r.show()
        for r in self.dense_sel_regions:
            r.setRegion((a, b))
            r.show()
        for r in self.matrix_sel_regions:
            r.setRegion((a, b))
            r.show()

    def _clear_selection(self):
        self._select_start = None
        self._select_end = None
        for r in self.plot_sel_regions:
            r.hide()
        for r in self.dense_sel_regions:
            r.hide()
        for r in self.matrix_sel_regions:
            r.hide()

    def _label_key(self, label_data: dict) -> LabelKey:
        return (
            float(label_data["start"]),
            float(label_data["end"]),
            str(label_data["label"]),
        )

    def _rebuild_label_index(self) -> None:
        n_labels = len(self.labels)
        self._label_keys_in_order = []
        self._label_starts = np.empty(n_labels, dtype=float)
        self._label_ends = np.empty(n_labels, dtype=float)

        for idx, lab in enumerate(self.labels):
            start = float(lab["start"])
            end = float(lab["end"])
            key = (start, end, str(lab["label"]))
            self._label_keys_in_order.append(key)
            self._label_starts[idx] = start
            self._label_ends[idx] = end

    def _visible_label_index_range(self) -> tuple[int, int]:
        if not self.labels or self.window_len <= 0:
            return (0, 0)

        t0 = float(self.window_start)
        t1 = float(self.window_start + self.window_len)
        start_idx = int(np.searchsorted(self._label_ends, t0, side="right"))
        end_idx = int(np.searchsorted(self._label_starts, t1, side="left"))
        if end_idx < start_idx:
            end_idx = start_idx
        return (start_idx, end_idx)

    def _visible_label_entries(self) -> list[tuple[LabelKey, dict]]:
        start_idx, end_idx = self._visible_label_index_range()
        return [
            (self._label_keys_in_order[idx], self.labels[idx])
            for idx in range(start_idx, end_idx)
        ]

    def _has_visible_window_label_targets(self) -> bool:
        return any(
            self._is_trace_plot_visible(idx) for idx in range(len(self.plots))
        ) or any(self._is_matrix_plot_visible(idx) for idx in range(len(self.matrix_plots)))

    def _remove_graphics_item(self, item) -> None:
        try:
            if item is not None and item.scene():
                item.scene().removeItem(item)
        except Exception:
            pass

    def _add_window_label_visual(self, label_data: dict) -> None:
        key = self._label_key(label_data)
        if key in self._label_visuals:
            return

        a, b, name = key
        color = self.label_colors.get(name, (150, 150, 150, 80))
        plot_regions: list[tuple[int, pg.LinearRegionItem]] = []
        matrix_regions: list[tuple[int, pg.LinearRegionItem]] = []
        dense_regions: list[tuple[int, pg.LinearRegionItem]] = []

        for i, plt in enumerate(self.plots):
            if not self._is_trace_plot_visible(i):
                continue
            reg = pg.LinearRegionItem(
                values=(a, b), brush=pg.mkBrush(*color), movable=False
            )
            reg.setZValue(-20)
            plt.addItem(reg)
            plot_regions.append((i, reg))

        for i, plt in enumerate(self.dense_plots):
            reg = pg.LinearRegionItem(
                values=(a, b), brush=pg.mkBrush(*color), movable=False
            )
            reg.setZValue(-20)
            plt.addItem(reg)
            dense_regions.append((i, reg))

        for i, plt in enumerate(self.matrix_plots):
            if not self._is_matrix_plot_visible(i):
                continue
            reg = pg.LinearRegionItem(
                values=(a, b), brush=pg.mkBrush(*color), movable=False
            )
            reg.setZValue(-20)
            plt.addItem(reg)
            matrix_regions.append((i, reg))

        if not plot_regions and not matrix_regions and not dense_regions:
            return

        self._label_visuals[key] = LabelVisualBundle(
            plot_regions=plot_regions,
            matrix_regions=matrix_regions,
            dense_regions=dense_regions,
            hypnogram_region=None,
        )

    def _remove_window_label_visual(self, key: LabelKey) -> None:
        bundle = self._label_visuals.pop(key, None)
        if bundle is None:
            return

        for _i, item in bundle.plot_regions:
            self._remove_graphics_item(item)

        for _i, item in bundle.dense_regions:
            self._remove_graphics_item(item)

        for _i, item in bundle.matrix_regions:
            self._remove_graphics_item(item)

    def _add_hypnogram_label_visual(self, label_data: dict) -> None:
        key = self._label_key(label_data)
        if key in self._hypnogram_label_visuals or self.hypnogram_plot is None:
            return

        a, b, name = key
        color = self.label_colors.get(name, (150, 150, 150, 80))
        region = pg.LinearRegionItem(
            values=(a, b), brush=pg.mkBrush(*color), movable=False
        )
        region.setZValue(-10)
        self.hypnogram_plot.addItem(region)
        self._hypnogram_label_visuals[key] = region

    def _remove_hypnogram_label_visual(self, key: LabelKey) -> None:
        region = self._hypnogram_label_visuals.pop(key, None)
        if region is None:
            return
        self._remove_graphics_item(region)

    def _clear_window_label_visuals(self) -> None:
        for key in list(self._label_visuals):
            self._remove_window_label_visual(key)

    def _clear_hypnogram_label_visuals(self) -> None:
        for key in list(self._hypnogram_label_visuals):
            self._remove_hypnogram_label_visual(key)

    def _clear_all_label_visuals(self) -> None:
        self._clear_window_label_visuals()
        self._clear_hypnogram_label_visuals()

    def _refresh_label_summary(self, force: bool = False) -> None:
        panel = getattr(self, "label_summary_panel", None)
        if panel is None:
            return
        if force or not panel.isHidden():
            panel.refresh()

    def _sync_hypnogram_label_visuals(self, *, force_rebuild: bool = False) -> None:
        if force_rebuild:
            self._clear_hypnogram_label_visuals()
        else:
            new_key_set = set(self._label_keys_in_order)
            for key in list(self._hypnogram_label_visuals):
                if key not in new_key_set:
                    self._remove_hypnogram_label_visual(key)

        for label_data in self.labels:
            if self._label_key(label_data) not in self._hypnogram_label_visuals:
                self._add_hypnogram_label_visual(label_data)

    def _sync_window_label_visuals(self, *, force_rebuild: bool = False) -> None:
        if force_rebuild or not self._has_visible_window_label_targets():
            self._clear_window_label_visuals()
            if not self._has_visible_window_label_targets():
                return

        visible_entries = self._visible_label_entries()
        new_key_set = {key for key, _ in visible_entries}
        for key in list(self._label_visuals):
            if key not in new_key_set:
                self._remove_window_label_visual(key)

        for key, label_data in visible_entries:
            if key not in self._label_visuals:
                self._add_window_label_visual(label_data)

    def _sync_label_visuals(
        self,
        *,
        force_rebuild: bool = False,
        refresh_summary: bool = True,
        force_rebuild_window: bool = False,
    ) -> None:
        self._sync_hypnogram_label_visuals(force_rebuild=force_rebuild)
        self._sync_window_label_visuals(
            force_rebuild=force_rebuild or force_rebuild_window
        )

        if refresh_summary:
            self._refresh_label_summary()

    def _finalize_label_change(
        self, *, force_rebuild: bool = False, refresh_summary: bool = True
    ) -> None:
        self.labels = sorted(self.labels, key=lambda x: float(x["start"]))
        self._merge_adjacent_same_labels()
        self._rebuild_label_index()
        self._sync_label_visuals(
            force_rebuild=force_rebuild, refresh_summary=refresh_summary
        )

    def _add_new_label(self, start, end, label):
        """Adds new label, overwriting/modifying existing ones in the range."""
        updated_labels = []

        for existing in self.labels:
            ex_start, ex_end = existing["start"], existing["end"]

            overlap_start = max(ex_start, start)
            overlap_end = min(ex_end, end)

            if overlap_start >= overlap_end:
                updated_labels.append(existing)
                continue

            if ex_start < start and ex_end > end:
                updated_labels.append(
                    {"start": ex_start, "end": start, "label": existing["label"]}
                )
                updated_labels.append(
                    {"start": end, "end": ex_end, "label": existing["label"]}
                )
            elif ex_start < start:
                updated_labels.append(
                    {"start": ex_start, "end": start, "label": existing["label"]}
                )
            elif ex_end > end:
                updated_labels.append(
                    {"start": end, "end": ex_end, "label": existing["label"]}
                )

        updated_labels.append({"start": start, "end": end, "label": label})
        self.labels = updated_labels
        self._finalize_label_change()

        # Track this label as most recently created (for note assignment)
        self.label_history.append((float(start), float(end)))

    def _clear_labels_in_range(self, start: float, end: float):
        """Remove any labels overlapping [start, end). Preserve non-overlapping parts.

        If an existing labeled epoch partially overlaps the range, it is split
        and only the overlapping part is removed.
        """
        if end <= start or not self.labels:
            return
        kept: list[dict] = []
        a = float(start)
        b = float(end)
        for existing in self.labels:
            ex_start = float(existing["start"])
            ex_end = float(existing["end"])
            lab = existing["label"]

            # No overlap
            if ex_end <= a or ex_start >= b:
                kept.append(existing)
                continue

            # Overlap exists; keep non-overlapping tails
            if ex_start < a:
                kept.append({"start": ex_start, "end": a, "label": lab})
            if ex_end > b:
                kept.append({"start": b, "end": ex_end, "label": lab})

        # Keep labels ordered; merging not required but harmless if adjacent remains
        self.labels = kept
        self._finalize_label_change()

    def _merge_adjacent_same_labels(self, adjacency_eps: float = 1e-9):
        if not self.labels:
            return
        merged = []
        for lab in sorted(self.labels, key=lambda x: x["start"]):
            if not merged:
                merged.append(
                    {
                        "start": float(lab["start"]),
                        "end": float(lab["end"]),
                        "label": lab["label"],
                    }
                )
                continue
            prev = merged[-1]
            if (
                lab["label"] == prev["label"]
                and float(lab["start"]) <= float(prev["end"]) + adjacency_eps
            ):
                prev["end"] = max(float(prev["end"]), float(lab["end"]))
            else:
                merged.append(
                    {
                        "start": float(lab["start"]),
                        "end": float(lab["end"]),
                        "label": lab["label"],
                    }
                )
        self.labels = merged

    def _redraw_all_labels(self):
        """Force a full rebuild of all visual label regions."""
        self._sync_label_visuals(force_rebuild=True)

    def _update_hypnogram_extents(self):
        if self.hypnogram_plot is None:
            return
        # Keep current zoom mode when extents change
        if not self.hypnogram_zoomed:
            self.hypnogram_plot.enableAutoRange("x", False)
            self.hypnogram_plot.setXRange(
                self.t_global_min, self.t_global_max, padding=0
            )
        else:
            self._update_hypnogram_xrange()
        # Ensure the view region reflects current window
        if self.hypnogram_view_region is not None:
            a = self.window_start
            b = self.window_start + self.window_len
            self.hypnogram_view_region.setRegion((a, b))

    def _delete_last_label(self):
        if not self.labels:
            return

        latest_end_time = -1
        latest_label_index = -1
        for i, lab in enumerate(self.labels):
            if lab["end"] > latest_end_time:
                latest_end_time = lab["end"]
                latest_label_index = i

        if latest_label_index != -1:
            last = self.labels.pop(latest_label_index)
            self._finalize_label_change()
            self._update_status(
                f"Deleted label: {last['label']} [{last['start']:.3f}, {last['end']:.3f}]"
            )

    def _zoom_active_plot_y(self, factor):
        """Zooms the Y-axis of the currently hovered plot."""
        if self.hovered_plot is None:
            return

        plot = self.hovered_plot
        plot.enableAutoRange("y", False)
        vb = plot.getViewBox()
        y_range = vb.viewRange()[1]
        center = (y_range[0] + y_range[1]) / 2.0
        height = (y_range[1] - y_range[0]) * factor
        vb.setYRange(center - height / 2.0, center + height / 2.0, padding=0)

    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        ktxt = ev.text().lower()
        key = ev.key()

        # Toggle hypnogram visibility with 'h'
        if ktxt == "h":
            if self.hypnogram_widget is not None:
                vis = self.hypnogram_widget.isVisible()
                self.hypnogram_widget.setVisible(not vis)
            return

        if ev.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
            if key == QtCore.Qt.Key.Key_1:
                self._zoom_active_plot_y(0.9)
                return
            if key == QtCore.Qt.Key.Key_2:
                self._zoom_active_plot_y(1.1)
                return
        # Ctrl+Shift+1/2/3 toggle video visibility
        if ev.modifiers() == (
            QtCore.Qt.KeyboardModifier.ControlModifier
            | QtCore.Qt.KeyboardModifier.ShiftModifier
        ):
            if key == QtCore.Qt.Key.Key_1:
                self._set_video_visible(1, not self.video_label.isVisible())
                if self.action_show_v1:
                    self.action_show_v1.setChecked(self.video_label.isVisible())
                return
            if key == QtCore.Qt.Key.Key_2:
                self._set_video_visible(2, not self.video2_label.isVisible())
                if self.action_show_v2:
                    self.action_show_v2.setChecked(self.video2_label.isVisible())
                return
            if key == QtCore.Qt.Key.Key_3:
                self._set_video_visible(3, not self.video3_label.isVisible())
                if self.action_show_v3:
                    self.action_show_v3.setChecked(self.video3_label.isVisible())
                return

        if key == QtCore.Qt.Key.Key_Space:
            self._toggle_playback()
            return

        # ]/[ = horizontal time page
        if key in (QtCore.Qt.Key.Key_BracketRight, QtCore.Qt.Key.Key_PageDown):
            self._page(+1)
            return
        if key in (QtCore.Qt.Key.Key_BracketLeft, QtCore.Qt.Key.Key_PageUp):
            self._page(-1)
            return
        # Left/Right arrow = frame step on selected video
        if key == QtCore.Qt.Key.Key_Right:
            self._step_frame(+1)
            return
        if key == QtCore.Qt.Key.Key_Left:
            self._step_frame(-1)
            return

        if (
            ktxt in self.keymap
            and self._select_start is not None
            and self._select_end is not None
        ):
            self._stop_playback_if_playing()
            label = self.keymap[ktxt]
            a = float(min(self._select_start, self._select_end))
            b = float(max(self._select_start, self._select_end))
            if b > a:
                self._add_new_label(a, b, label)
                self._update_status(f"Labeled {label}: [{a:.3f}, {b:.3f}]")
                self._clear_selection()
                return

        # Clear labels in selected region with '0'
        if (
            key == QtCore.Qt.Key.Key_0
            and self._select_start is not None
            and self._select_end is not None
        ):
            self._stop_playback_if_playing()
            a = float(min(self._select_start, self._select_end))
            b = float(max(self._select_start, self._select_end))
            if b > a:
                self._clear_labels_in_range(a, b)
                self._update_status(f"Cleared labels in [{a:.3f}, {b:.3f}]")
                self._clear_selection()
                return

        # Toggle hypnogram zoom
        if ktxt == "z":
            self._toggle_hypnogram_zoom()
            return

        # Next / previous epoch navigation
        if ktxt == "n":
            self._jump_to_epoch_by_offset(+1)
            return
        if ktxt == "b":
            self._jump_to_epoch_by_offset(-1)
            return

        super().keyPressEvent(ev)

    def _toggle_hypnogram_zoom(self):
        self.hypnogram_zoomed = not self.hypnogram_zoomed
        self._update_hypnogram_xrange()

    def _set_frame_step_source(self, which: int):
        self.frame_step_source = int(which)

    def _available_frame_times(self, which: int):
        if which == 1 and self.video_frame_times is not None:
            return self.video_frame_times
        if which == 2 and self.video2_frame_times is not None:
            return self.video2_frame_times
        if which == 3 and self.video3_frame_times is not None:
            return self.video3_frame_times
        return None

    def _fallback_frame_source(self) -> int:
        # Choose first available in order 1,2,3
        if self.video_frame_times is not None:
            return 1
        if self.video2_frame_times is not None:
            return 2
        if self.video3_frame_times is not None:
            return 3
        return 0

    def _step_frame(self, direction: int):
        src = self.frame_step_source
        ft = self._available_frame_times(src)
        if ft is None:
            src = self._fallback_frame_source()
            ft = self._available_frame_times(src)
            if ft is None:
                return
        idx = find_nearest_frame(ft, self.cursor_time)
        new_idx = int(np.clip(idx + (1 if direction >= 1 else -1), 0, len(ft) - 1))
        new_t = float(ft[new_idx])
        self._set_cursor_time(new_t, update_slider=True)

    def _update_hypnogram_xrange(self):
        if self.hypnogram_plot is None:
            return
        if not self.hypnogram_zoomed:
            # Show full extent
            self.hypnogram_plot.enableAutoRange("x", False)
            self.hypnogram_plot.setXRange(
                self.t_global_min, self.t_global_max, padding=0
            )
        else:
            # Zoom around current window with +/- padding
            pad = float(self.hypnogram_zoom_padding)
            a = max(self.t_global_min, self.window_start - pad)
            b = min(self.t_global_max, self.window_start + self.window_len + pad)
            if b <= a:
                b = min(self.t_global_max, a + 1.0)
            self.hypnogram_plot.enableAutoRange("x", False)
            self.hypnogram_plot.setXRange(a, b, padding=0)

    # ---------- Export / Import ----------

    def load_labels(self, path: str) -> None:
        """Load labels from a CSV file.

        Parameters
        ----------
        path : str
            Path to a CSV file with columns ``start_s``, ``end_s``,
            ``label``, and optionally ``note``.
        """
        loaded_labels = []
        loaded_notes = {}
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

            has_notes = len(header) >= 4 and header[3] == "note"
            if header[:3] != ["start_s", "end_s", "label"]:
                raise ValueError("CSV header does not match expected format.")

            for row in reader:
                if not row:
                    continue
                start = float(row[0])
                end = float(row[1])
                label = str(row[2])
                loaded_labels.append({"start": start, "end": end, "label": label})

                if has_notes and len(row) >= 4 and row[3].strip():
                    loaded_notes[(start, end)] = row[3]

        self.labels = loaded_labels
        self.label_notes = loaded_notes
        self.label_history = [(lab["start"], lab["end"]) for lab in self.labels]
        self._finalize_label_change()
        self._update_status(
            f"Loaded {len(self.labels)} labels from {os.path.basename(path)}"
        )

    def _on_load_labels(self):
        self._stop_playback_if_playing()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load labels from CSV", filter="CSV (*.csv)"
        )
        if not path:
            return
        try:
            self.load_labels(path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Load error", f"Failed to load or parse labels file:\n\n{e}"
            )

    def _on_export_labels(self):
        self._stop_playback_if_playing()
        if not self.labels:
            QtWidgets.QMessageBox.information(self, "Export", "No labels to export.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export labels to CSV", filter="CSV (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["start_s", "end_s", "label", "note"])
                for lab in self.labels:
                    key = (float(lab["start"]), float(lab["end"]))
                    note = self.label_notes.get(key, "")
                    writer.writerow(
                        [f"{lab['start']:.6f}", f"{lab['end']:.6f}", lab["label"], note]
                    )
            self._update_status(f"Exported labels to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export error", str(e))

    # ---------- Navigation / rendering ----------

    def _stop_playback_if_playing(self):
        """Stops playback if it is currently active."""
        if self.is_playing:
            self.is_playing = False
            self.playback_timer.stop()
            self._update_status("Playback stopped.")

    def _toggle_playback(self):
        """Toggles video playback on or off."""
        if self.is_playing:
            self._stop_playback_if_playing()
        else:
            if self.video_frame_times is None:
                self._update_status("No video loaded to play.")
                return
            self.is_playing = True
            self.playback_elapsed_timer.start()
            self.playback_timer.start(16)
            self._update_status("Playing...")

    def _advance_playback_frame(self):
        """Called by the QTimer to advance the cursor time."""
        if not self.is_playing:
            return

        dt_ms = self.playback_elapsed_timer.restart()
        dt_sec = (dt_ms / 1000.0) * float(self.playback_speed)

        t_start = self.window_start
        t_end = self.window_start + self.window_len
        if t_end <= t_start:
            return

        new_cursor_time = self.cursor_time + dt_sec

        if new_cursor_time >= t_end:
            new_cursor_time = t_start + (new_cursor_time - t_end)
            if new_cursor_time >= t_end:
                new_cursor_time = t_start

        self._set_cursor_time(new_cursor_time, update_slider=True)

    def _page(self, direction: int):
        self._stop_playback_if_playing()
        direction = 1 if direction >= 1 else -1
        total = self.t_global_max - self.t_global_min
        if total <= 0:
            return
        new_start = self.window_start + direction * self.window_len
        new_start = clamp(
            new_start,
            self.t_global_min,
            max(self.t_global_min, self.t_global_max - self.window_len),
        )
        rel = (
            0.0
            if self.window_len <= 0
            else (self.cursor_time - self.window_start) / self.window_len
        )
        self.window_start = new_start
        self.cursor_time = self.window_start + rel * self.window_len

        self._apply_x_range()
        self._update_nav_slider_from_window()

    def _on_smooth_scroll(self, direction: int):
        self._stop_playback_if_playing()
        direction = 1 if direction >= 1 else -1
        total = self.t_global_max - self.t_global_min
        if total <= 0:
            return
        delta = direction * float(self.smooth_scroll_fraction) * float(self.window_len)
        new_start = self.window_start + delta
        new_start = clamp(
            new_start,
            self.t_global_min,
            max(self.t_global_min, self.t_global_max - self.window_len),
        )
        rel = (
            0.0
            if self.window_len <= 0
            else (self.cursor_time - self.window_start) / self.window_len
        )
        self.window_start = new_start
        self.cursor_time = self.window_start + rel * self.window_len
        self._schedule_deferred_view_refresh(update_nav_slider=True)

    def _on_cursor_wheel(self, dy: int):
        # Adjust the in-window cursor position proportionally to wheel delta
        self._stop_playback_if_playing()
        wl = float(self.window_len)
        if wl <= 0:
            return
        # Typical mouse wheel notch is 120 units. Move ~2% of window per notch.
        step_per_notch = 0.02
        frac = (float(dy) / 120.0) * step_per_notch
        dt = frac * wl
        xr0 = self.window_start
        xr1 = self.window_start + wl
        new_t = clamp(self.cursor_time + dt, xr0, xr1)
        self._set_cursor_time(new_t, update_slider=True)

    def _on_window_len_changed(self, v):
        self._stop_playback_if_playing()
        self.window_len = float(v)
        self.window_start = clamp(
            self.window_start,
            self.t_global_min,
            max(self.t_global_min, self.t_global_max - self.window_len),
        )
        self._apply_x_range()
        self._update_nav_slider_from_window()

    def _on_nav_slider_changed(self, value):
        self._stop_playback_if_playing()
        if self.t_global_max <= self.t_global_min:
            return
        total = max(1e-9, self.t_global_max - self.t_global_min)
        span = max(1e-9, total - self.window_len)
        start = self.t_global_min + (value / 10000.0) * span
        self.window_start = clamp(
            start,
            self.t_global_min,
            max(self.t_global_min, self.t_global_max - self.window_len),
        )
        self._schedule_deferred_view_refresh(update_nav_slider=False)

    def _apply_x_range(self):
        if self._deferred_view_refresh_timer.isActive():
            self._deferred_view_refresh_timer.stop()
        self._deferred_view_refresh_needs_nav_slider = False
        self._apply_x_range_core()

    def _apply_x_range_core(self):
        xr = (self.window_start, self.window_start + self.window_len)
        for plt in self.plots:
            plt.enableAutoRange("x", False)
            plt.setXRange(*xr, padding=0.0)

        # Also apply to dense plots
        for plt in self.dense_plots:
            plt.enableAutoRange("x", False)
            plt.setXRange(*xr, padding=0.0)

        # Also apply to matrix plots
        for plt in self.matrix_plots:
            plt.enableAutoRange("x", False)
            plt.setXRange(*xr, padding=0.0)

        new_cursor_time = clamp(self.cursor_time, xr[0], xr[1])
        self._set_cursor_time(new_cursor_time, update_slider=True)

        self._refresh_curves()
        self._sync_window_label_visuals()

        # Update hypnogram view region to show current window
        if self.hypnogram_view_region is not None:
            self.hypnogram_view_region.setRegion(xr)
        # If zoomed, keep hypnogram centered on the current window +/- padding
        if self.hypnogram_zoomed:
            self._update_hypnogram_xrange()

    def _update_nav_slider_from_window(self):
        if self.t_global_max <= self.t_global_min:
            self.nav_slider.setValue(0)
            return
        total = max(1e-9, self.t_global_max - self.t_global_min)
        span = max(1e-9, total - self.window_len)
        frac = (
            0.0
            if span <= 0
            else clamp((self.window_start - self.t_global_min) / span, 0.0, 1.0)
        )
        self.nav_slider.blockSignals(True)
        self.nav_slider.setValue(int(round(frac * 10000)))
        self.nav_slider.blockSignals(False)

    def _update_cursor_lines(self):
        for ln in self.plot_cur_lines:
            ln.setPos(self.cursor_time)
        for ln in self.dense_cur_lines:
            ln.setPos(self.cursor_time)
        for ln in self.matrix_cur_lines:
            ln.setPos(self.cursor_time)

    def _set_cursor_time(self, t, update_slider=True):
        self.cursor_time = t

        self._update_cursor_lines()

        if update_slider:
            self._update_window_cursor_from_cursor_time()

        self._request_video_frame(
            frame_times=self.video_frame_times,
            worker=self._video_worker,
            requested_attr="_video_requested_frame_idx",
            t=self.cursor_time,
        )
        self._request_video_frame(
            frame_times=self.video2_frame_times,
            worker=self._video2_worker,
            requested_attr="_video2_requested_frame_idx",
            t=self.cursor_time,
        )
        self._request_video_frame(
            frame_times=self.video3_frame_times,
            worker=self._video3_worker,
            requested_attr="_video3_requested_frame_idx",
            t=self.cursor_time,
        )

        if not self.is_playing:
            self._update_status()

    def _on_window_cursor_changed(self, value):
        self._stop_playback_if_playing()
        frac = value / 10000.0
        t = self.window_start + frac * self.window_len
        self._set_cursor_time(t, update_slider=False)

    def _update_window_cursor_from_cursor_time(self):
        frac = (
            0.0
            if self.window_len <= 0
            else clamp(
                (self.cursor_time - self.window_start) / self.window_len, 0.0, 1.0
            )
        )
        self.window_cursor_slider.blockSignals(True)
        self.window_cursor_slider.setValue(int(round(frac * 10000)))
        self.window_cursor_slider.blockSignals(False)

    def _target_pts(self):
        all_plots = self.plots or self.dense_plots
        if not all_plots:
            return self.max_pts_per_plot
        target_plot = None
        for idx, plt in enumerate(self.plots):
            if self._is_trace_plot_visible(idx):
                target_plot = plt
                break
        if target_plot is None:
            target_plot = all_plots[0]
        vb = target_plot.getViewBox()
        px = max(300, int(vb.width()))
        return int(min(2 * px, self.max_pts_per_plot))

    def _refresh_curves(self):
        t0, t1 = self.window_start, self.window_start + self.window_len
        max_pts = self._target_pts()
        if self.overlay_mode and self._plot_to_series:
            for plot_idx, series_indices in enumerate(self._plot_to_series):
                if not self._is_trace_plot_visible(plot_idx):
                    continue
                for local_idx, si in enumerate(series_indices):
                    s = self.series[si]
                    curve = self._plot_to_curves[plot_idx][local_idx]
                    tx, yx = segment_for_window(s.t, s.y, t0, t1, max_pts=max_pts)
                    curve.setData(tx, yx, _callSync="off")
        else:
            for idx, (s, curve) in enumerate(zip(self.series, self.curves)):
                if not self._is_trace_plot_visible(idx):
                    continue
                tx, yx = segment_for_window(s.t, s.y, t0, t1, max_pts=max_pts)
                curve.setData(tx, yx, _callSync="off")

        # Also refresh dense and matrix plots
        self._refresh_dense_curves()
        self._refresh_matrix_plots()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._rescale_video_frame()
        QtCore.QTimer.singleShot(50, self._refresh_curves)
        QtCore.QTimer.singleShot(60, self._align_left_axes)

    def _align_left_axes(self):
        try:
            all_plots = list(self.plots) + list(self.dense_plots) + list(self.matrix_plots)
            if not all_plots:
                return
            widths = []
            for plt in all_plots:
                ax = plt.getAxis("left")
                widths.append(int(ax.width()))
            if not widths:
                return
            target = max(max(widths), 55)
            for plt in all_plots:
                ax = plt.getAxis("left")
                ax.setWidth(int(target))
        except Exception:
            pass

    # ---------- Video size allocation (right panel) ----------
    def _apply_video_stretches(self):
        if self.videos_layout is None:
            return
        # Apply stretch to video widgets; hidden widgets will not take space
        try:

            def set_stretch(widget, stretch):
                idx = self.videos_layout.indexOf(widget)
                if idx >= 0:
                    self.videos_layout.setStretch(idx, max(0, int(stretch)))

            set_stretch(self.video_label, self.video1_stretch)
            set_stretch(self.video2_label, self.video2_stretch)
            set_stretch(self.video3_label, self.video3_stretch)
        except Exception:
            pass
        # Trigger layout update to reflect new stretches
        if self.videos_widget is not None:
            self.videos_widget.updateGeometry()
            self.videos_widget.adjustSize()

    def _set_video_stretches(self, v1: int, v2: int, v3: int):
        self.video1_stretch = max(0, int(v1))
        self.video2_stretch = max(0, int(v2))
        self.video3_stretch = max(0, int(v3))
        self._apply_video_stretches()
        # Trigger a resize to update scaling
        QtCore.QTimer.singleShot(0, self._rescale_video_frame)
        QtCore.QTimer.singleShot(0, self._rescale_video2_frame)
        QtCore.QTimer.singleShot(0, self._rescale_video3_frame)

    def _adjust_secondary_video_sizes(self):
        # Determine how many secondary videos are visible
        vid2_present = self.video2_label.isVisible()
        vid3_present = self.video3_label.isVisible()
        if not vid2_present and not vid3_present:
            QtWidgets.QMessageBox.information(
                self,
                "Adjust Sizes",
                "Secondary videos are not loaded.",
            )
            return

        # Compute current total and primary share percentage
        total = max(
            1,
            self.video1_stretch
            + (self.video2_stretch if vid2_present else 0)
            + (self.video3_stretch if vid3_present else 0),
        )
        current_primary_pct = int(round(100.0 * self.video1_stretch / total))

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Adjust Secondary Videos Size")
        lay = QtWidgets.QVBoxLayout(dlg)
        label = QtWidgets.QLabel("Primary video (Video 1) share (% of video area):")
        lay.addWidget(label)
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(5, 95)
        slider.setValue(current_primary_pct)
        lay.addWidget(slider)

        pct_lbl = QtWidgets.QLabel(f"{current_primary_pct:d}%")
        lay.addWidget(pct_lbl)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        lay.addWidget(btns)

        def on_change(val):
            pct_lbl.setText(f"{val:d}%")
            # Live preview: recompute stretches based on percentage
            primary = int(val)
            remainder = 100 - primary
            v1 = primary
            if vid2_present and vid3_present:
                v2 = max(1, remainder // 2)
                v3 = max(1, remainder - v2)
            elif vid2_present:
                v2 = remainder
                v3 = 0
            else:
                v2 = 0
                v3 = remainder
            self._set_video_stretches(v1, v2, v3)

        slider.valueChanged.connect(on_change)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        # Initialize preview
        on_change(slider.value())
        dlg.exec()
        # If canceled, nothing to do; if accepted, stretches already applied

    # ---------- Trace visibility ----------
    def _apply_trace_visibility(self):
        # Rebuild the graphics layout based on visibility and subplot order
        try:
            # Remove all plots from layout first
            self.plot_area.clear()
            master_plot = None
            row = 0

            # Get visible plots in order
            visible_plots = self._get_visible_subplot_order()

            # Determine which is the last visible plot for axis labeling
            last_idx = len(visible_plots) - 1

            for i, (plot_type, idx) in enumerate(visible_plots):
                is_last = i == last_idx

                if plot_type == "ts":
                    if idx >= len(self.plots):
                        continue
                    plt = self.plots[idx]
                    plt.setVisible(True)
                    self.plot_area.addItem(plt, row=row, col=0)

                    # Update bottom axis visibility
                    if self.low_profile_x and not is_last:
                        try:
                            plt.setLabel("bottom", "")
                            bax = plt.getAxis("bottom")
                            bax.setStyle(showValues=True, tickLength=0)
                            bax.setTextPen(pg.mkPen(0, 0, 0, 0))
                            bax.setHeight(12)
                        except Exception:
                            pass
                    else:
                        try:
                            plt.setLabel(
                                "bottom", "Time", units="s" if is_last else None
                            )
                            bax = plt.getAxis("bottom")
                            bax.setStyle(showValues=True, tickLength=-5)
                            bax.setTextPen(pg.mkPen("w"))
                            bax.setHeight(None)
                        except Exception:
                            pass

                elif plot_type == "dense":
                    if idx >= len(self.dense_plots):
                        continue
                    plt = self.dense_plots[idx]
                    plt.setVisible(True)
                    self.plot_area.addItem(plt, row=row, col=0)

                    if self.low_profile_x and not is_last:
                        try:
                            plt.setLabel("bottom", "")
                            bax = plt.getAxis("bottom")
                            bax.setStyle(showValues=True, tickLength=0)
                            bax.setTextPen(pg.mkPen(0, 0, 0, 0))
                            bax.setHeight(12)
                        except Exception:
                            pass
                    else:
                        try:
                            plt.setLabel(
                                "bottom", "Time", units="s" if is_last else None
                            )
                            bax = plt.getAxis("bottom")
                            bax.setStyle(showValues=True, tickLength=-5)
                            bax.setTextPen(pg.mkPen("w"))
                            bax.setHeight(None)
                        except Exception:
                            pass

                else:  # matrix
                    if idx >= len(self.matrix_plots):
                        continue
                    plt = self.matrix_plots[idx]
                    plt.setVisible(True)
                    self.plot_area.addItem(plt, row=row, col=0)

                    # Update bottom axis visibility
                    if self.low_profile_x and not is_last:
                        try:
                            plt.setLabel("bottom", "")
                            bax = plt.getAxis("bottom")
                            bax.setStyle(showValues=True, tickLength=0)
                            bax.setTextPen(pg.mkPen(0, 0, 0, 0))
                            bax.setHeight(12)
                        except Exception:
                            pass
                    else:
                        try:
                            plt.setLabel(
                                "bottom", "Time", units="s" if is_last else None
                            )
                            bax = plt.getAxis("bottom")
                            bax.setStyle(showValues=True, tickLength=-5)
                            bax.setTextPen(pg.mkPen("w"))
                            bax.setHeight(None)
                        except Exception:
                            pass

                if master_plot is None:
                    master_plot = plt
                else:
                    plt.setXLink(master_plot)
                row += 1

            # Hide plots that are not visible
            for idx, plt in enumerate(self.plots):
                if (
                    not self.trace_visible[idx]
                    if idx < len(self.trace_visible)
                    else False
                ):
                    plt.setVisible(False)
            for idx, plt in enumerate(self.dense_plots):
                if (
                    not self.dense_visible[idx]
                    if idx < len(self.dense_visible)
                    else False
                ):
                    plt.setVisible(False)
            for idx, plt in enumerate(self.matrix_plots):
                if (
                    not self.matrix_visible[idx]
                    if idx < len(self.matrix_visible)
                    else False
                ):
                    plt.setVisible(False)

            # Apply custom plot heights
            self._apply_custom_plot_heights()
            self._update_plot_area_height()
            # Re-apply x-range to keep all linked
            self._apply_x_range()
            if self.labels:
                self._sync_label_visuals(
                    refresh_summary=False,
                    force_rebuild_window=True,
                )
            QtCore.QTimer.singleShot(0, self._align_left_axes)
        except Exception:
            import traceback

            traceback.print_exc()

    def _set_video_visible(self, which: int, visible: bool):
        lbl = None
        if which == 1:
            lbl = self.video_label
        elif which == 2:
            lbl = self.video2_label
        elif which == 3:
            lbl = self.video3_label
        if lbl is None:
            return
        lbl.setVisible(bool(visible))
        self._apply_video_stretches()
        # Rescale frames to current label sizes
        QtCore.QTimer.singleShot(0, self._rescale_video_frame)
        QtCore.QTimer.singleShot(0, self._rescale_video2_frame)
        QtCore.QTimer.singleShot(0, self._rescale_video3_frame)

    # ---------- Help/Status & cleanup ----------

    def _show_help(self):
        self._stop_playback_if_playing()
        QtWidgets.QMessageBox.information(
            self,
            "Help",
            (
                "<b>Hotkeys</b><br>"
                "<b>Spacebar:</b> Toggle window playback<br>"
                "<b>Ctrl+D:</b> Show/hide Y-Axis Controls<br>"
                "<b>Ctrl+1 / Ctrl+2:</b> Zoom Y-Axis In / Out (on hovered plot)<br>"
                "<b>Labels:</b> w=Wake, n=NREM, r=REM, a=Artifact, Backspace=delete last<br>"
                "<b>Paging:</b> [ ] or Scroll Wheel = previous/next page<br><br>"
                "Click-drag in any plot to create selection. Selection stays active across pages; "
                "drag its handles to extend, then press a label hotkey.<br>"
            ),
        )

    def _update_status(self, msg=None):
        info = []
        if self.series:
            info += [
                f"{len(self.series)} traces",
                f"t=[{self.t_global_min:.2f},{self.t_global_max:.2f}]s",
            ]
        info += [
            f"win={self.window_len:.2f}s @ {self.window_start:.2f}s",
            self._format_cursor_with_state(),
        ]
        if self.is_playing and not msg:
            msg = "Playing..."
        if msg:
            info.append("| " + msg)
        self.status.showMessage("  ".join(info))

    def _format_cursor_with_state(self):
        label_info = self._get_state_and_epoch_at_time(self.cursor_time)
        if label_info is None:
            return f"cursor={self.cursor_time:.3f}s, state='Unlabeled'"

        state_txt = label_info["label"]
        result = f"cursor={self.cursor_time:.3f}s, state='{state_txt}'"

        # Check for note on this epoch
        key = (float(label_info["start"]), float(label_info["end"]))
        note = self.label_notes.get(key, "")
        if note:
            # Truncate note for status bar (max 40 chars)
            if len(note) > 40:
                note = note[:37] + "..."
            result += f" | Note: {note}"

        return result

    def _get_state_and_epoch_at_time(self, t):
        """Get the full label dict at time t, or None if unlabeled."""
        if not self.labels:
            return None
        for lab in self.labels:
            if lab["start"] <= t < lab["end"]:
                return lab
        return None

    def _get_state_at_time(self, t):
        lab = self._get_state_and_epoch_at_time(t)
        return lab["label"] if lab else None

    def closeEvent(self, ev):
        try:
            self._stop_playback_if_playing()
            QtCore.QMetaObject.invokeMethod(
                self._video_worker, "stop", QtCore.Qt.QueuedConnection
            )
            QtCore.QMetaObject.invokeMethod(
                self._video2_worker, "stop", QtCore.Qt.QueuedConnection
            )
            QtCore.QMetaObject.invokeMethod(
                self._video3_worker, "stop", QtCore.Qt.QueuedConnection
            )
            self._video_thread.quit()
            self._video2_thread.quit()
            self._video3_thread.quit()
            if not self._video_thread.wait(1000):
                self._video_thread.terminate()
            if not self._video2_thread.wait(1000):
                self._video2_thread.terminate()
            if not self._video3_thread.wait(1000):
                self._video3_thread.terminate()
        except Exception as e:
            print(f"ERROR: Exception during closeEvent: {e}")
        super().closeEvent(ev)


# ---------------- Main ----------------


def main():
    parser = argparse.ArgumentParser(description="Loupe — Multi-Trace Viewer")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to directory with time series files (*_t.npy, *_y.npy)",
    )
    parser.add_argument(
        "--data_files",
        nargs="+",
        type=str,
        help=(
            "Ordered list of .npy files (mix of *_t.npy and *_y.npy). "
            "Pairs are matched by basename; display order follows first appearance."
        ),
    )
    parser.add_argument(
        "--colors",
        nargs="+",
        type=str,
        help=(
            "Optional colors matching series order. Accepts hex (#RRGGBB[AA]), 0xRRGGBB, or R,G,B[,A]."
        ),
    )
    parser.add_argument("--video", type=str, help="Path to video file (.mp4)")
    parser.add_argument(
        "--frame_times", type=str, help="Path to video frame times file (.npy)"
    )
    parser.add_argument("--video2", type=str, help="Optional second video file (.mp4)")
    parser.add_argument(
        "--frame_times2", type=str, help="Path to second video frame times (.npy)"
    )
    parser.add_argument("--video3", type=str, help="Optional third video file (.mp4)")
    parser.add_argument(
        "--frame_times3", type=str, help="Path to third video frame times (.npy)"
    )
    parser.add_argument(
        "--auto_scale",
        action="store_true",
        help=(
            "Enable Y auto-scaling (default is fixed scale from robust percentiles)."
        ),
    )
    parser.add_argument(
        "--low_profile_x",
        action="store_true",
        default=None,
        help=(
            "Hide X-axis labels and ticks for all but the bottom plot. "
            "When omitted, this is enabled automatically for 3+ total subplots."
        ),
    )
    # Matrix viewer arguments
    parser.add_argument(
        "--matrix_timestamps",
        nargs="+",
        type=str,
        help="List of paths to .npy files containing event timestamps for matrix/raster plots.",
    )
    parser.add_argument(
        "--matrix_yvals",
        nargs="+",
        type=str,
        help="List of paths to .npy files containing row indices (0 to N-1) for each event.",
    )
    parser.add_argument(
        "--alpha_vals",
        nargs="+",
        type=str,
        help="List of paths to .npy files containing alpha values (0-1) for each event.",
    )
    parser.add_argument(
        "--matrix_colors",
        nargs="+",
        type=str,
        help="List of hex colors (#RRGGBB) for each matrix subplot.",
    )
    # xarray arguments
    parser.add_argument(
        "--xr_path",
        nargs="+",
        type=str,
        help="Path(s) to xarray-compatible store(s) (zarr directory or netCDF file).",
    )
    parser.add_argument(
        "--xr_group",
        nargs="+",
        type=str,
        help="Group(s) within the zarr/netCDF store(s). One per --xr_path, or a single value for all.",
    )
    parser.add_argument(
        "--xr_variable",
        type=str,
        default="data",
        help="Variable name within the dataset (default: 'data').",
    )
    parser.add_argument(
        "--xr_filter",
        type=str,
        default=None,
        help=(
            'JSON filter dict for dimension slicing, e.g. '
            '\'{"syn_id": [3, 6], "time": [0, 1800]}\'. '
            'Values as [start, stop] are converted to slice objects.'
        ),
    )
    args = parser.parse_args()

    # Convert xarray args to Series if provided
    xr_series = None
    if args.xr_path:
        from loupe.xr_loader import dataarray_to_series, load_xarray_from_path

        filter_dict = None
        if args.xr_filter:
            raw = json.loads(args.xr_filter)
            filter_dict = {
                k: slice(*v) if isinstance(v, list) else v
                for k, v in raw.items()
            }

        # Match groups to paths
        paths = args.xr_path
        if args.xr_group is None:
            groups = [None] * len(paths)
        elif len(args.xr_group) == 1:
            groups = args.xr_group * len(paths)
        else:
            groups = args.xr_group

        all_tuples = []
        for p, g in zip(paths, groups):
            da = load_xarray_from_path(
                p, group=g, variable=args.xr_variable, filter_dict=filter_dict
            )
            prefix = os.path.splitext(os.path.basename(p))[0] if len(paths) > 1 else ""
            all_tuples.extend(dataarray_to_series(da, name_prefix=prefix))

        xr_series = [Series(name, t, y) for name, t, y in all_tuples]

    app = QtWidgets.QApplication(sys.argv)
    w = LoupeApp(
        data_dir=args.data_dir,
        data_files=args.data_files,
        colors=args.colors,
        video_path=args.video,
        frame_times_path=args.frame_times,
        video2_path=args.video2,
        frame_times2_path=args.frame_times2,
        video3_path=args.video3,
        frame_times3_path=args.frame_times3,
        fixed_scale=not args.auto_scale,
        low_profile_x=args.low_profile_x,
        # Matrix viewer arguments
        matrix_timestamps=args.matrix_timestamps,
        matrix_yvals=args.matrix_yvals,
        alpha_vals=args.alpha_vals,
        matrix_colors=args.matrix_colors,
        # xarray series
        xr_series=xr_series,
    )
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
