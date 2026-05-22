"""Unit tests for the :class:`MultiFileVideoCapture` adapter — verify that
seeking, reading, frame-count totals, and release behave correctly across the
concatenated underlying captures."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import MagicMock

import pytest

import loupe.app as loupe_app
from loupe.app import MultiFileVideoCapture


pytestmark = pytest.mark.skipif(
    loupe_app.cv2 is None, reason="OpenCV (cv2) is not installed."
)


class _FakeCapture:
    """Minimal stand-in for cv2.VideoCapture used by adapter tests."""

    def __init__(self, n_frames: int, opened: bool = True, file_id: str = ""):
        self._n_frames = int(n_frames)
        self._opened = bool(opened)
        self._file_id = file_id
        self.last_set_idx: int | None = None
        self.set_calls: list[tuple[int, float]] = []
        self.read_calls = 0
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop):
        if prop == loupe_app.cv2.CAP_PROP_FRAME_COUNT:
            return float(self._n_frames)
        return 0.0

    def set(self, prop, value) -> bool:
        self.set_calls.append((prop, float(value)))
        if prop == loupe_app.cv2.CAP_PROP_POS_FRAMES:
            self.last_set_idx = int(value)
        return True

    def read(self):
        self.read_calls += 1
        # Sentinel: identifies which file produced the frame (no actual image).
        return True, self._file_id

    def release(self) -> None:
        self.released = True


@pytest.fixture
def fake_caps(monkeypatch):
    """Replace cv2.VideoCapture with a factory returning _FakeCapture objects.

    The factory maps each path to a pre-configured (n_frames, file_id, opened)
    tuple via the dict yielded by this fixture.
    """
    catalog: dict[str, _FakeCapture] = {}

    def factory(path):
        return catalog[path]

    monkeypatch.setattr(loupe_app.cv2, "VideoCapture", factory)
    yield catalog


def test_multi_file_capture_seeks_into_correct_file(fake_caps):
    # Three files with 10, 20, and 5 frames -> total 35 (global indices 0..34).
    fake_caps["a.mp4"] = _FakeCapture(10, file_id="A")
    fake_caps["b.mp4"] = _FakeCapture(20, file_id="B")
    fake_caps["c.mp4"] = _FakeCapture(5, file_id="C")
    cap = MultiFileVideoCapture(["a.mp4", "b.mp4", "c.mp4"])
    assert cap.isOpened()

    # First frame of file A (global 0).
    cap.set(loupe_app.cv2.CAP_PROP_POS_FRAMES, 0)
    assert cap._active_idx == 0
    assert fake_caps["a.mp4"].last_set_idx == 0
    ok, sentinel = cap.read()
    assert ok and sentinel == "A"

    # Last frame of file A (global 9).
    cap.set(loupe_app.cv2.CAP_PROP_POS_FRAMES, 9)
    assert cap._active_idx == 0
    assert fake_caps["a.mp4"].last_set_idx == 9

    # First frame of file B (global 10) -> local 0 within B.
    cap.set(loupe_app.cv2.CAP_PROP_POS_FRAMES, 10)
    assert cap._active_idx == 1
    assert fake_caps["b.mp4"].last_set_idx == 0
    ok, sentinel = cap.read()
    assert ok and sentinel == "B"

    # Middle of file B (global 25) -> local 15.
    cap.set(loupe_app.cv2.CAP_PROP_POS_FRAMES, 25)
    assert cap._active_idx == 1
    assert fake_caps["b.mp4"].last_set_idx == 15

    # First frame of file C (global 30) -> local 0 within C.
    cap.set(loupe_app.cv2.CAP_PROP_POS_FRAMES, 30)
    assert cap._active_idx == 2
    assert fake_caps["c.mp4"].last_set_idx == 0

    # Last valid global frame (34) -> local 4 in file C.
    cap.set(loupe_app.cv2.CAP_PROP_POS_FRAMES, 34)
    assert cap._active_idx == 2
    assert fake_caps["c.mp4"].last_set_idx == 4


def test_multi_file_capture_clamps_out_of_range_indices(fake_caps):
    fake_caps["a.mp4"] = _FakeCapture(10, file_id="A")
    fake_caps["b.mp4"] = _FakeCapture(20, file_id="B")
    cap = MultiFileVideoCapture(["a.mp4", "b.mp4"])

    # Negative -> 0.
    cap.set(loupe_app.cv2.CAP_PROP_POS_FRAMES, -5)
    assert cap._active_idx == 0
    assert fake_caps["a.mp4"].last_set_idx == 0

    # Beyond total -> last valid frame (global 29 -> local 19 in B).
    cap.set(loupe_app.cv2.CAP_PROP_POS_FRAMES, 9999)
    assert cap._active_idx == 1
    assert fake_caps["b.mp4"].last_set_idx == 19


def test_multi_file_capture_release(fake_caps):
    fake_caps["a.mp4"] = _FakeCapture(10)
    fake_caps["b.mp4"] = _FakeCapture(20)
    cap = MultiFileVideoCapture(["a.mp4", "b.mp4"])

    cap.release()

    assert fake_caps["a.mp4"].released
    assert fake_caps["b.mp4"].released
    assert not cap.isOpened()
    assert cap.get(loupe_app.cv2.CAP_PROP_FRAME_COUNT) == 0.0


def test_multi_file_capture_frame_count(fake_caps):
    fake_caps["a.mp4"] = _FakeCapture(7)
    fake_caps["b.mp4"] = _FakeCapture(13)
    fake_caps["c.mp4"] = _FakeCapture(2)
    cap = MultiFileVideoCapture(["a.mp4", "b.mp4", "c.mp4"])

    assert cap.get(loupe_app.cv2.CAP_PROP_FRAME_COUNT) == 22.0


def test_isopened_false_when_one_underlying_failed(fake_caps):
    fake_caps["a.mp4"] = _FakeCapture(10, opened=True)
    fake_caps["bad.mp4"] = _FakeCapture(20, opened=False)
    cap = MultiFileVideoCapture(["a.mp4", "bad.mp4"])

    assert not cap.isOpened()


def test_isopened_false_when_total_frames_zero(fake_caps):
    fake_caps["empty.mp4"] = _FakeCapture(0, opened=True)
    cap = MultiFileVideoCapture(["empty.mp4"])

    assert not cap.isOpened()
