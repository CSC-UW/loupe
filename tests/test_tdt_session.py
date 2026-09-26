"""Complete extension sessions: configuration, GUI, clocks, annotations/video."""

import copy
import json
import threading
import time

import numpy as np
import polars as pl
import pytest
from PySide6 import QtWidgets
from PySide6.QtTest import QTest

from loupe.extensions import Extension, discover_extensions
from loupe.extensions.tdt.reader import Block, Epoc, Snips, Stream, Cancelled
from loupe.extensions.tdt.session import (
    atomic_json,
    default_config,
    launch_session,
    prepare_session,
    read_config,
    ttl_series,
    validate_config,
)
from loupe.extensions.tdt.launcher import TDTLauncher, _jobs
from loupe.file_series import FileSegment, FileSignal


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def block(tmp_path):
    signal = (np.sin(np.arange(20_000) / 30) * 0.002).astype("<f4")
    path = tmp_path / "samples.bin"
    signal.tofile(path)
    streams = {
        "EEGr": Stream(
            "EEGr",
            1000,
            [1, 2, 4],
            {c: [FileSegment(path, 0, len(signal), 0)] for c in [1, 2, 4]},
            "<f4",
        )
    }
    return Block(
        tmp_path,
        20,
        {"start_utc": "2026-01-01", "scalars": []},
        streams,
        {
            "Bttn": Epoc(
                "Bttn",
                np.array([2.0, 6.0, 8.0]),
                np.array([4.0, 7.0, np.inf]),
                np.ones(3),
            )
        },
        {
            "eSpk": Snips(
                "eSpk",
                np.array([1.0, 3.0, 5.0, 8.0]),
                np.array([1, 4, 1, 2]),
                np.ones(4),
            )
        },
        [],
        [],
    )


def all_config(block):
    config = default_config(block)
    for row in config["rows"]:
        row["enabled"] = True
        if row["kind"] == "epoc":
            row["mode"] = "both"
    return config


def close(window, qapp):
    window.close()
    qapp.processEvents()


def test_ttl_union_preserves_exact_edges():
    series = ttl_series("Pulses", np.array([[1, 3], [2, 5], [5, 6], [8, 9]]), 10)
    np.testing.assert_equal(series.t, [0, 1, 1, 6, 6, 8, 8, 9, 9, 10])
    np.testing.assert_equal(series.y, [0, 0, 1, 1, 0, 0, 1, 1, 0, 0])


def test_prepare_selected_channels_layout_and_spike_ids(block):
    config = all_config(block)
    config["rows"][0].update(channels="4,1", mode="lines")
    config["rows"][2]["channels"] = "4,1"
    config["rows"] = config["rows"][::-1]
    prepared = prepare_session(block, config)
    assert prepared.kwargs["subplot_order"] == [
        ("raster", 0),
        ("ts", 0),
        ("ts", 1),
        ("ts", 2),
    ]
    assert [s.name for s in prepared.kwargs["xr_series"]] == [
        "Bttn TTL",
        "EEGr · Ch 4",
        "EEGr · Ch 1",
    ]
    raster = prepared.kwargs["raster_series_list"][0]
    np.testing.assert_equal(raster.row_keys, [4, 1])
    np.testing.assert_equal(raster.timestamps, [1, 3, 5])
    np.testing.assert_equal(raster.yvals, [1, 0, 1])
    assert isinstance(prepared.kwargs["xr_series"][1].y, FileSignal)
    assert len(prepared.epocs) == 1


def test_saved_runtime_roundtrip_includes_labels_order_style_and_epocs(block, qapp):
    prepared = prepare_session(block, all_config(block))
    window = launch_session(prepared, save=False)
    try:
        window.dense_groups[0].gain = 333
        window.dense_groups[0].hidden_traces = {1}
        window.window_len = 4
        window.window_start = 7
        window.subplot_order = list(reversed(window.subplot_order))
        label_id = window.interval_label_set.add(2, 5, "NREM")
        window.interval_label_set.set_note(label_id, "edited")
        window.extension_session.prepared.epocs[0][0]["color"] = "#123456"
        window._rebuild_all_plots()
        saved_order = window.subplot_order[:]
        window.extension_session.save(window)
        config = read_config(block)
    finally:
        close(window, qapp)
    restored = launch_session(prepare_session(block, config), save=False)
    try:
        assert restored.dense_groups[0].gain == 333
        assert restored.dense_groups[0].hidden_traces == {1}
        assert restored.window_len == 4
        assert restored.window_start == 7
        assert restored.subplot_order == saved_order
        assert restored.interval_label_set.at_time(3).label == "NREM"
        assert restored.interval_label_set.at_time(3).note == "edited"
        assert restored.extension_session.prepared.epocs[0][0]["color"] == "#123456"
        assert restored.update_block_config_action.isEnabled()
        # Shading is independent of overlapping editable labels.
        assert len(restored.extension_session._overlay_items) == 3
        for plot in restored.dense_plots + restored.plots:
            plot.setYRange(-3, 3)
        restored._rebuild_all_plots()
        assert len(restored.extension_session._overlay_items) == 3
    finally:
        close(restored, qapp)


def test_empty_labels_and_epoc_only_session_roundtrip(block, qapp):
    config = default_config(block)
    config["rows"][0]["enabled"] = False
    window = launch_session(prepare_session(block, config), save=False)
    try:
        assert window.t_global_max == block.duration
        window.extension_session.save(window)
    finally:
        close(window, qapp)
    loaded = prepare_session(block, read_config(block))
    assert len(loaded.kwargs["interval_label_set"]) == 0
    assert loaded.kwargs["xr_series"][0].name == "Block timeline"


def test_replaced_data_cannot_overwrite_a_block_config(block, qapp):
    from loupe.series import Series
    from types import SimpleNamespace

    config = default_config(block)
    config["rows"][0]["mode"] = "lines"
    window = launch_session(prepare_session(block, config), save=False)
    try:
        window.extension_session.save(window)
        before = (block.path / ".loupe" / "tdt.json").read_bytes()
        window.video_slots.append(SimpleNamespace(video_path="unrelated-video.avi"))
        with pytest.raises(ValueError, match="data sources changed"):
            window.extension_session.save(window)
        window.video_slots.pop()
        window.set_series([Series("unrelated", np.arange(10), np.arange(10))])
        with pytest.raises(ValueError, match="data sources changed"):
            window.extension_session.save(window)
        assert (block.path / ".loupe" / "tdt.json").read_bytes() == before
    finally:
        close(window, qapp)


def test_atomic_save_does_not_corrupt_prior_config(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    atomic_json(path, {"good": True})
    with pytest.raises(ValueError):
        atomic_json(path, {"bad": float("nan")})
    assert json.loads(path.read_text()) == {"good": True}

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr("loupe.extensions.tdt.session.os.replace", fail)
    with pytest.raises(OSError, match="disk full"):
        atomic_json(path, {"replace": True})
    assert json.loads(path.read_text()) == {"good": True}
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(version=99),
        lambda c: c.update(fingerprint=[{"changed": True}]),
        lambda c: c["rows"][0].update(channels="9"),
        lambda c: c["rows"][0].update(mode="bad"),
        lambda c: c["rows"][0].update(color="bad"),
        lambda c: c["rows"][0].update(height=-1),
        lambda c: c.update(window_len=float("inf")),
        lambda c: c["rows"].append(c["rows"][0]),
    ],
)
def test_bad_configs_fail_before_loading(block, mutation):
    config = default_config(block)
    mutation(config)
    with pytest.raises(ValueError):
        validate_config(block, config)


def test_annotations_custom_csv_columns(block):
    path = block.path / "hypnogram.csv"
    pl.DataFrame(
        {"start_time": [0.0, 10.0], "end_time": [10.0, 20.0], "state": ["Wake", "NREM"]}
    ).write_csv(path)
    config = default_config(block)
    config.update(
        annotation_path=str(path),
        annotation_schema={
            "start_col": "start_time",
            "end_col": "end_time",
            "label_col": "state",
        },
    )
    result = prepare_session(block, config)
    assert result.kwargs["interval_label_set"].at_time(12).label == "NREM"


def test_launcher_editing_and_saved_state_invalidation(block, qapp):
    launcher = TDTLauncher()
    try:
        launcher._inspected(block)
        launcher._set_busy(False)
        assert launcher.table.columnWidth(1) >= 140
        assert launcher.collect_config() == launcher.config
        config = copy.deepcopy(launcher.config)
        config["view_config"] = {"sentinel": True}
        launcher.apply_config(config)
        assert launcher.collect_config()["view_config"] == {"sentinel": True}
        launcher.table.selectRow(0)
        launcher.move_row(1)
        config = launcher.collect_config()
        assert config["rows"][0]["store"] == "Bttn"
        assert "view_config" not in config
        launcher.table.cellWidget(1, 2).setText("4,1")
        assert launcher.collect_config()["rows"][1]["channels"] == "4,1"
    finally:
        close(launcher, qapp)


def test_cancel_job_keeps_gui_alive_without_result_or_orphan(qapp):
    launcher = TDTLauncher()
    started = threading.Event()
    release = threading.Event()
    results = []

    def work(cancel, progress):
        started.set()
        release.wait(2)
        if cancel.is_set():
            raise Cancelled()
        return "unexpected"

    launcher._start(work, results.append)
    assert started.wait(1)
    job = launcher.job
    launcher.reject()
    assert job.cancel.is_set()
    release.set()
    deadline = time.monotonic() + 3
    while job in _jobs and time.monotonic() < deadline:
        QTest.qWait(10)
    assert job not in _jobs
    assert not job.isRunning()
    assert results == []


def test_video_counts_and_cam_timestamps(block, tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "camera.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    if not writer.isOpened():
        pytest.skip("MJPG encoder unavailable")
    for index in range(5):
        writer.write(np.full((24, 32, 3), 20 * index, dtype=np.uint8))
    writer.release()
    block.epocs["Cam1"] = Epoc(
        "Cam1", np.array([1.0, 1.1, 1.2, 1.4, 1.5]), np.arange(5) + 2.0, np.arange(1, 6)
    )
    config = default_config(block)
    config["videos"] = [
        {"path": str(path), "epoc": "Cam1", "enabled": True, "correction": 0.2}
    ]
    prepared = prepare_session(block, config)
    video = prepared.kwargs["video_configs"][0]
    np.testing.assert_equal(np.load(video.frame_times_path), block.epocs["Cam1"].onset)
    assert video.frame_times_correction == 0.2
    assert prepared.temporary is not None
    prepared.temporary.cleanup()
    block.epocs["Cam1"].onset = np.array([1.0, 1.1, 1.2])
    block.epocs["Cam1"].values = np.arange(1, 4)
    with pytest.raises(ValueError, match="no timestamp"):
        prepare_session(block, config)


def test_missing_sdk_message_and_broken_external_extension(monkeypatch):
    monkeypatch.setattr("loupe.extensions.util.find_spec", lambda name: None)
    extension = Extension("tdt", "TDT", "unused:unused", dependency="tdt", extra="tdt")
    with pytest.raises(ImportError, match=r"uv add 'loupe\[tdt\]'"):
        extension.open()

    class Entry:
        name = "broken"

        def load(self):
            raise ImportError("optional dependency absent")

    monkeypatch.setattr(
        "loupe.extensions.metadata.entry_points", lambda **kw: [Entry()]
    )
    extensions, errors = discover_extensions()
    assert [e.id for e in extensions] == ["tdt"]
    assert "broken" in errors[0]
