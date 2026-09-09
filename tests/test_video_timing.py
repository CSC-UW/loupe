"""Timestamp and decoded-frame integration checks for synchronized videos."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

import loupe.app as loupe_app
from loupe import VideoConfig
from loupe.app import LoupeApp, Series
from loupe.state_config import load_state_config
from loupe.video import (
    MultiFileVideoCapture,
    VideoWorker,
    VideoWindow,
    cv2,
    frame_index_at_time,
    frame_time_tolerance,
    validate_frame_times,
)


@pytest.mark.parametrize("values", [[], [[1, 2]], [1, np.nan], [1, np.inf], [1, 1], [2, 1]])
def test_reject_invalid_timestamps(values):
    with pytest.raises(ValueError):
        validate_frame_times(np.asarray(values))


def test_frame_selection_preserves_recording_gaps_and_outer_bounds():
    times = validate_frame_times(np.array([10, 10.1, 10.2, 20, 20.1]))
    tolerance = frame_time_tolerance(times, None)
    assert tolerance == pytest.approx(0.055)
    assert frame_index_at_time(times, 10.04, tolerance) == 0
    assert frame_index_at_time(times, 10.08, tolerance) == 1
    assert frame_index_at_time(times, 15, tolerance) is None
    assert frame_index_at_time(times, 9, tolerance) is None
    assert frame_index_at_time(times, 21, tolerance) is None
    assert frame_index_at_time(times, np.nan, tolerance) is None


@pytest.mark.parametrize("value", [-1, np.nan, np.inf])
def test_reject_invalid_distance(value):
    with pytest.raises(ValueError):
        frame_time_tolerance(np.array([0., 1.]), value)


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wait_for(qapp, condition, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if condition():
            return
        time.sleep(.002)
    assert condition(), "Qt video operation timed out"


@pytest.fixture
def movie(tmp_path):
    path = tmp_path / "indexed.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for index in range(12):
        writer.write(np.full((48, 64, 3), 10 + index * 15, dtype=np.uint8))
    writer.release()
    times = np.r_[10 + np.arange(6) / 10, 20 + np.arange(6) / 10]
    time_path = tmp_path / "times.npy"
    np.save(time_path, times)
    return path, time_path, times


def _frame_value(image):
    return image.pixelColor(10, 10).red()


def test_worker_decodes_sequential_skip_reverse_and_cached_frames(qapp, movie):
    path, _, _ = movie
    worker = VideoWorker(cache_frames=4)
    received = []
    worker.frameReady.connect(lambda index, image: received.append((index, _frame_value(image))))
    worker.open(str(path))
    capture = worker.cap
    seeks = []
    grabs = []

    class CaptureSpy:
        def __getattr__(self, name):
            return getattr(capture, name)

        def set(self, prop, value):
            seeks.append(value)
            return capture.set(prop, value)

        def grab(self):
            grabs.append(1)
            return capture.grab()

    worker.cap = CaptureSpy()
    # Cache hits must not disturb the decoder position. Forward skips use grab.
    for index in [0, 1, 2, 8, 1, 9, 10, 11, 0]:
        worker.requestFrame(index)
        _wait_for(qapp, lambda: received and received[-1][0] == index)
        assert received[-1][1] == pytest.approx(10 + index * 15, abs=3)
    assert seeks == [0]  # only the final reverse seek; playback never seeks
    assert len(grabs) == 5
    worker.stop()


def test_concat_sequential_read_crosses_files(movie):
    path, _, _ = movie
    cap = MultiFileVideoCapture([str(path), str(path)])
    assert cap.frame_counts == [12, 12]
    for index in range(24):
        ok, image = cap.read()
        assert ok
        assert image[10, 10, 0] == pytest.approx(10 + (index % 12) * 15, abs=3)
    assert not cap.read()[0]
    cap.release()


@pytest.mark.parametrize("operation,index", [("seek", 10), ("grab", 1), ("read", 0)])
def test_worker_clears_failed_decode_and_recovers_position(qapp, operation, index):
    class FailingCapture:
        failure = operation
        position = 0
        seeks = []

        def set(self, prop, value):
            self.seeks.append(value)
            if self.failure == "seek":
                return False
            self.position = int(value)
            return True

        def grab(self):
            if self.failure == "grab":
                return False
            self.position += 1
            return True

        def read(self):
            if self.failure == "read":
                return False, None
            image = np.full((16, 16, 3), self.position, dtype=np.uint8)
            self.position += 1
            return True, image

        def release(self):
            pass

    worker = VideoWorker()
    capture = FailingCapture()
    worker.cap = capture
    received = []
    worker.frameReady.connect(lambda idx, image: received.append((idx, image)))
    worker.requestFrame(index)
    _wait_for(qapp, lambda: bool(received))
    assert received[-1][0] == index
    assert received[-1][1].isNull()
    assert not worker._request_queued
    capture.failure = None
    worker.requestFrame(0)
    _wait_for(qapp, lambda: len(received) == 2)
    assert capture.seeks[-1] == 0
    assert received[-1][0] == 0
    assert _frame_value(received[-1][1]) == 0
    worker.stop()


@pytest.fixture
def window(qapp, movie, monkeypatch):
    original_options = pg.setConfigOptions
    monkeypatch.setattr(pg, "setConfigOptions", lambda **kw: original_options(**(kw | {"useOpenGL": False})))
    path, time_path, _ = movie
    config = load_state_config(
        path=os.path.join(os.path.dirname(loupe_app.__file__), "example_state_definitions.json"),
        package_default=False,
    )
    videos = [VideoConfig(str(path), str(time_path), name=f"DMD {i}", separate_window="Microscope") for i in range(4)]
    result = LoupeApp(
        xr_series=[Series(name="Signal", t=np.linspace(0, 30, 301), y=np.sin(np.linspace(0, 30, 301)))],
        video_configs=videos, fixed_scale=True, state_config=config,
    )
    result.show()
    _wait_for(qapp, lambda: all(slot.is_open for slot in result.video_slots))
    yield result
    result.close()
    qapp.processEvents()


def test_real_video_seeks_clear_gaps_and_share_window(qapp, window):
    assert len(window._video_windows) == 1
    detached = window._video_windows["Microscope"]
    assert detached.isVisible()
    assert len(detached.labels) == 4
    for t, expected in [(10.2, 2), (20.4, 10), (10.0, 0)]:
        window._set_cursor_time(t, update_slider=True)
        _wait_for(qapp, lambda: all(slot.displayed_frame_idx == expected for slot in window.video_slots))
        for slot in window.video_slots:
            assert slot.displayed_frame_time == pytest.approx(t)
            assert _frame_value(slot.last_pixmap.toImage()) == pytest.approx(10 + expected * 15, abs=3)
    window._set_cursor_time(15., update_slider=True)
    qapp.processEvents()
    for slot in window.video_slots:
        assert slot.displayed_frame_idx is None
        assert slot.last_pixmap is None
        assert "No frame at this time" in detached.labels[slot.index].text()
    # A delayed decode result from before the gap must not resurrect that frame.
    worker = window.video_slots[0].worker
    window._on_frame_ready(window.video_slots[0], 0, worker.cache[0])
    assert window.video_slots[0].displayed_frame_idx is None
    detached.close()
    qapp.processEvents()
    assert all(not slot.separate_window for slot in window.video_slots)
    assert all(slot.label.isVisible() for slot in window.video_slots)


def test_frame_count_mismatch_rejected(qapp, movie, window, monkeypatch):
    path, time_path, times = movie
    np.save(time_path, times[:-1])
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args[2]) or 0)
    slot = window.video_slots[0]
    window._load_video_data(slot, str(path), str(time_path))
    _wait_for(qapp, lambda: bool(warnings))
    assert "do not match" in warnings[0]
    assert not slot.is_open
    assert slot.frame_times is None
    assert slot.displayed_frame_idx is None


def test_playback_uses_cursor_clock_and_gui_thread(qapp, window, monkeypatch):
    threads = []
    original = window._on_frame_ready

    def record_thread(*args):
        threads.append(QtCore.QThread.currentThread())
        original(*args)

    monkeypatch.setattr(window, "_on_frame_ready", record_thread)
    window.window_start = 10.
    window.window_len = .55
    window._set_cursor_time(10., update_slider=True)
    window._toggle_playback()
    _wait_for(qapp, lambda: all(slot.displayed_frame_idx == 3 for slot in window.video_slots))
    window._stop_playback_if_playing()
    assert threads and all(thread == qapp.thread() for thread in threads)
    for slot in window.video_slots:
        assert abs(slot.displayed_frame_time - window.cursor_time) < .06


def test_playback_does_not_truncate_fractional_milliseconds(window):
    class ElapsedClock:
        elapsed = 0

        def nsecsElapsed(self):
            self.elapsed += 16_800_000
            return self.elapsed

    original = window.playback_elapsed_timer
    window.playback_elapsed_timer = ElapsedClock()
    window._playback_last_elapsed_ns = 0
    window.window_start = 10.
    window.window_len = 1.
    window.cursor_time = 10.
    window.is_playing = True
    for _ in range(10):
        window._advance_playback_frame()
    assert window.cursor_time == pytest.approx(10.168, abs=1e-10)
    window.is_playing = False
    window.playback_elapsed_timer = original


def test_separate_grid_order_independent_of_decoder_open_order(qapp):
    parent = QtWidgets.QWidget()
    window = VideoWindow("Movies", parent)
    for index in [0, 2, 3, 1]:
        window.add_video(index, str(index))
    for index in range(4):
        assert window.grid.itemAtPosition(index // 2, index % 2).widget() is window.panels[index]
    window.close()
    parent.close()


def test_inline_video_pixmap_does_not_force_splitter_width(qapp, window):
    window._set_video_separate_window(0, False)
    window.resize(1400, 900)
    window.splitter.setSizes([350, 1050])
    qapp.processEvents()
    slot = window.video_slots[0]
    image = QtGui.QImage(4096, 2160, QtGui.QImage.Format_RGB888)
    image.fill(QtGui.QColor("gray"))
    slot.requested_frame_idx = 0
    window._on_frame_ready(slot, 0, image)
    qapp.processEvents()
    # A decoded frame sized for a formerly wide panel must not prevent the
    # user (or a saved layout) from making that panel narrow afterward.
    window.splitter.setSizes([1000, 400])
    for _ in range(5):
        window._rescale_all_video_frames()
        qapp.processEvents()
    left, right = window.splitter.sizes()
    assert left >= 950
    assert right <= 430
    assert slot.label.width() <= right
