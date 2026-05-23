"""Tests for VideoConfig list-of-paths support and the _load_video_data
normalize/dispatch logic. Exercises the path through view() + LoupeApp slot
construction rather than calling internal helpers directly."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtWidgets

import loupe.app as loupe_app
from loupe import VideoConfig
from loupe.app import LoupeApp, MultiFileVideoCapture, Series, VideoSlot, VideoWorker
from loupe.state_config import load_state_config


def _state_config():
    pkg_dir = os.path.dirname(loupe_app.__file__)
    return load_state_config(
        path=os.path.join(pkg_dir, "example_state_definitions.json"),
        package_default=False,
    )


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pg.setConfigOptions(useOpenGL=False)
    yield app


@pytest.fixture
def trivial_window(_qapp):
    """A minimal LoupeApp with no videos and one fake trace."""
    series = [Series(name="t", t=np.linspace(0, 1, 10), y=np.zeros(10))]
    w = LoupeApp(xr_series=series, fixed_scale=True, state_config=_state_config())
    yield w
    for slot in w.video_slots:
        QtCore.QMetaObject.invokeMethod(slot.worker, "stop", QtCore.Qt.QueuedConnection)
    for slot in w.video_slots:
        slot.thread.quit()
        if not slot.thread.wait(1000):
            slot.thread.terminate()
            slot.thread.wait(1000)
    w.close()


def _make_slot(
    parent: QtCore.QObject,
    name: str = "Video 1",
    frame_times_correction: float = 0.0,
) -> VideoSlot:
    thread = QtCore.QThread(parent)
    worker = VideoWorker(cache_frames=4)
    worker.moveToThread(thread)
    return VideoSlot(
        index=0, name=name, stretch=3, worker=worker, thread=thread,
        frame_times_correction=frame_times_correction,
    )


@pytest.fixture
def captured_invokes(monkeypatch):
    """Capture QMetaObject.invokeMethod calls to spy on which slot was invoked."""
    calls: list[tuple[object, str, tuple]] = []
    original = QtCore.QMetaObject.invokeMethod

    def spy(target, member, connection_type, *args):
        calls.append((target, member, args))
        # Don't actually invoke (avoids the worker trying to open a fake MP4).
        return True

    monkeypatch.setattr(QtCore.QMetaObject, "invokeMethod", spy)
    return calls


def test_videoconfig_accepts_list_paths():
    cfg = VideoConfig(
        video_path=["a.mp4", "b.mp4"],
        frame_times_path=["a.npy", "b.npy"],
    )
    assert cfg.video_path == ["a.mp4", "b.mp4"]
    assert cfg.frame_times_path == ["a.npy", "b.npy"]


def test_videoconfig_string_paths_still_work():
    cfg = VideoConfig(video_path="a.mp4", frame_times_path="a.npy")
    assert cfg.video_path == "a.mp4"
    assert cfg.frame_times_path == "a.npy"


def test_load_video_data_single_file_uses_open_slot(
    trivial_window, captured_invokes, tmp_path,
):
    vpath = tmp_path / "v.mp4"
    vpath.write_bytes(b"")
    ft_path = tmp_path / "ft.npy"
    np.save(ft_path, np.array([0.0, 0.1, 0.2]))

    slot = _make_slot(trivial_window)
    trivial_window._load_video_data(slot, str(vpath), str(ft_path))

    invocations = [(m, a) for tgt, m, a in captured_invokes if tgt is slot.worker]
    assert len(invocations) == 1
    member, args = invocations[0]
    assert member == "open"
    # Q_ARG carries the path through as the first positional arg.
    assert len(args) == 1
    assert slot.frame_times is not None
    assert slot.frame_times.tolist() == [0.0, 0.1, 0.2]


def test_load_video_data_list_uses_openconcat_slot(
    trivial_window, captured_invokes, tmp_path,
):
    v1 = tmp_path / "v1.mp4"; v1.write_bytes(b"")
    v2 = tmp_path / "v2.mp4"; v2.write_bytes(b"")
    ft1 = tmp_path / "ft1.npy"; np.save(ft1, np.array([0.0, 0.1, 0.2]))
    ft2 = tmp_path / "ft2.npy"; np.save(ft2, np.array([1.0, 1.1]))

    slot = _make_slot(trivial_window)
    trivial_window._load_video_data(
        slot, [str(v1), str(v2)], [str(ft1), str(ft2)],
    )

    invocations = [(m, a) for tgt, m, a in captured_invokes if tgt is slot.worker]
    assert len(invocations) == 1
    member, _ = invocations[0]
    assert member == "openConcat"
    assert slot.frame_times is not None
    assert slot.frame_times.tolist() == [0.0, 0.1, 0.2, 1.0, 1.1]


def test_load_video_data_rejects_length_mismatch(
    trivial_window, captured_invokes, tmp_path, monkeypatch,
):
    v1 = tmp_path / "v1.mp4"; v1.write_bytes(b"")
    v2 = tmp_path / "v2.mp4"; v2.write_bytes(b"")
    ft1 = tmp_path / "ft1.npy"; np.save(ft1, np.array([0.0]))

    warned: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning",
        lambda parent, title, text, *a, **kw: warned.append((title, text)) or 0,
    )

    slot = _make_slot(trivial_window)
    trivial_window._load_video_data(slot, [str(v1), str(v2)], [str(ft1)])

    assert warned and "config error" in warned[0][0]
    assert slot.frame_times is None
    # No worker invocation should have happened.
    assert not [c for c in captured_invokes if c[0] is slot.worker]


def test_load_video_data_missing_file_warns(
    trivial_window, captured_invokes, tmp_path, monkeypatch,
):
    ft1 = tmp_path / "ft.npy"; np.save(ft1, np.array([0.0]))

    warned: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning",
        lambda parent, title, text, *a, **kw: warned.append((title, text)) or 0,
    )

    slot = _make_slot(trivial_window)
    trivial_window._load_video_data(
        slot, str(tmp_path / "does_not_exist.mp4"), str(ft1),
    )

    assert warned and warned[0][0] == "File Not Found"
    assert slot.frame_times is None
    assert not [c for c in captured_invokes if c[0] is slot.worker]


def test_videoconfig_default_correction_is_zero():
    cfg = VideoConfig(video_path="a.mp4", frame_times_path="a.npy")
    assert cfg.frame_times_correction == 0.0


def test_load_video_data_single_with_correction(
    trivial_window, captured_invokes, tmp_path,
):
    vpath = tmp_path / "v.mp4"
    vpath.write_bytes(b"")
    ft_path = tmp_path / "ft.npy"
    np.save(ft_path, np.array([0.0, 0.1, 0.2]))

    slot = _make_slot(trivial_window, frame_times_correction=10.0)
    trivial_window._load_video_data(slot, str(vpath), str(ft_path))

    assert slot.frame_times is not None
    np.testing.assert_allclose(slot.frame_times, [10.0, 10.1, 10.2])


def test_load_video_data_list_with_correction(
    trivial_window, captured_invokes, tmp_path,
):
    v1 = tmp_path / "v1.mp4"; v1.write_bytes(b"")
    v2 = tmp_path / "v2.mp4"; v2.write_bytes(b"")
    ft1 = tmp_path / "ft1.npy"; np.save(ft1, np.array([0.0, 0.1, 0.2]))
    ft2 = tmp_path / "ft2.npy"; np.save(ft2, np.array([1.0, 1.1]))

    slot = _make_slot(trivial_window, frame_times_correction=-5.0)
    trivial_window._load_video_data(
        slot, [str(v1), str(v2)], [str(ft1), str(ft2)],
    )

    assert slot.frame_times is not None
    np.testing.assert_allclose(
        slot.frame_times, [-5.0, -4.9, -4.8, -4.0, -3.9],
    )
