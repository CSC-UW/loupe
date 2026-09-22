"""Unit tests for the :class:`MultiFileVideoCapture` adapter — verify that
seeking, reading, frame-count totals, and release behave correctly across the
concatenated underlying captures."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import MagicMock

import pytest

import loupe.app as loupe_app
from loupe.app import MultiFileVideoCapture
from loupe.video import FrameCountReport, VideoWorker


pytestmark = pytest.mark.skipif(
    loupe_app.cv2 is None, reason="OpenCV (cv2) is not installed."
)


class _FakeCapture:
    """Minimal stand-in for cv2.VideoCapture used by adapter tests."""

    def __init__(
        self,
        n_frames: int,
        opened: bool = True,
        file_id: str = "",
        n_decodable: int | None = None,
    ):
        self._n_frames = int(n_frames)
        # Frames the container indexes but never wrote (a truncated final
        # fragment) fail to decode; by default every indexed frame decodes.
        self._n_decodable = self._n_frames if n_decodable is None else int(n_decodable)
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
        position = self.last_set_idx or 0
        if position >= self._n_decodable:
            return False, None
        self.last_set_idx = position + 1
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


# ---------------------------------------------------------------------------
# Indexed-vs-timestamp frame count reconciliation
# ---------------------------------------------------------------------------


def test_truncated_tail_is_capped_at_timestamp_count(fake_caps):
    # File B's container indexes 20 frames but only 17 decode; 17 timestamps.
    fake_caps["a.mp4"] = _FakeCapture(10, file_id="A")
    fake_caps["b.mp4"] = _FakeCapture(20, file_id="B", n_decodable=17)
    fake_caps["c.mp4"] = _FakeCapture(5, file_id="C")
    cap = MultiFileVideoCapture(["a.mp4", "b.mp4", "c.mp4"])
    assert cap.header_frame_counts == [10, 20, 5]

    notes = cap.apply_expected_frame_counts([10, 17, 5], slack=120)

    assert cap.frame_counts == [10, 17, 5]
    assert cap.header_frame_counts == [10, 20, 5]
    assert cap.get(loupe_app.cv2.CAP_PROP_FRAME_COUNT) == 32.0
    assert len(notes) == 1 and "file 2" in notes[0] and "3 frame" in notes[0]
    # Probing must leave every capture rewound.
    assert fake_caps["b.mp4"].last_set_idx == 0
    # Global indices past B's usable frames now land in C, not in B's phantom tail.
    cap.set(loupe_app.cv2.CAP_PROP_POS_FRAMES, 27)
    assert cap._active_idx == 2
    assert fake_caps["c.mp4"].last_set_idx == 0
    ok, sentinel = cap.read()
    assert ok and sentinel == "C"


def test_truncated_tail_stops_sequential_reads_at_cap(fake_caps):
    fake_caps["a.mp4"] = _FakeCapture(10, file_id="A", n_decodable=8)
    cap = MultiFileVideoCapture(["a.mp4"])
    cap.apply_expected_frame_counts([8], slack=None)
    assert [cap.read()[0] for _ in range(9)] == [True] * 8 + [False]


def test_matching_counts_need_no_probe(fake_caps):
    fake_caps["a.mp4"] = _FakeCapture(10, file_id="A")
    cap = MultiFileVideoCapture(["a.mp4"])
    assert cap.apply_expected_frame_counts([10], slack=0) == []
    assert fake_caps["a.mp4"].read_calls == 0


@pytest.mark.parametrize(
    "n_frames,n_decodable,expected,slack,reason",
    [
        (12, 12, 11, 120, "decodes frame 11"),          # a complete file with too few timestamps
        (12, 12, 13, 120, "past the end of the video"),  # timestamps outrun the container
        (20, 17, 17, 2, "above the 2-frame allowance"),  # excess beyond the slack
        (20, 15, 17, 120, "does not decode"),            # timestamps outrun the decodable frames
    ],
)
def test_genuine_mismatches_are_rejected(fake_caps, n_frames, n_decodable, expected, slack, reason):
    fake_caps["a.mp4"] = _FakeCapture(n_frames, n_decodable=n_decodable)
    cap = MultiFileVideoCapture(["a.mp4"])
    with pytest.raises(ValueError, match="do not match") as excinfo:
        cap.apply_expected_frame_counts([expected], slack=slack)
    assert reason in str(excinfo.value)
    # A rejected reconciliation leaves the header counts in force.
    assert cap.frame_counts == [n_frames]


def test_wrong_number_of_timestamp_arrays_rejected(fake_caps):
    fake_caps["a.mp4"] = _FakeCapture(10)
    cap = MultiFileVideoCapture(["a.mp4"])
    with pytest.raises(ValueError):
        cap.apply_expected_frame_counts([10, 10], slack=None)


def _open_worker(fake_caps, expected, slack=120):
    worker = VideoWorker(cache_frames=4, frame_count_slack=slack)
    results = {}
    worker.frameCounts.connect(lambda report: results.__setitem__("report", report))
    worker.opened.connect(lambda ok, msg: results.update(ok=ok, msg=msg))
    worker.openConcat(list(fake_caps), expected)
    return worker, results


def test_worker_reports_usable_counts_and_notes(fake_caps):
    fake_caps["a.mp4"] = _FakeCapture(10, file_id="A")
    fake_caps["b.mp4"] = _FakeCapture(20, file_id="B", n_decodable=17)
    worker, results = _open_worker(fake_caps, [10, 17])
    assert results["ok"], results["msg"]
    report = results["report"]
    assert isinstance(report, FrameCountReport)
    assert report.header == [10, 20]
    assert report.usable == [10, 17]
    assert len(report.notes) == 1
    worker.stop()


def test_worker_rejects_mismatch_and_releases_captures(fake_caps):
    fake_caps["a.mp4"] = _FakeCapture(12)
    worker, results = _open_worker(fake_caps, [11])
    assert results["ok"] is False
    assert "do not match" in results["msg"]
    assert "report" not in results
    assert worker.cap is None
    assert fake_caps["a.mp4"].released


def test_worker_without_expected_counts_takes_header_at_face_value(fake_caps):
    fake_caps["a.mp4"] = _FakeCapture(12, n_decodable=9)
    worker, results = _open_worker(fake_caps, None)
    assert results["ok"]
    assert results["report"].usable == [12]
    assert results["report"].notes == []
    worker.stop()
