"""Acquisition clock, binary sample, and bounded-memory reader contracts."""

import struct
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from loupe._decimation import segment_for_window
from loupe.file_series import FileSegment, FileSignal, UniformTimeAxis, PacketTimeAxis
from loupe.extensions.tdt.reader import (
    Epoc,
    Stream,
    _sev_header,
    _sev_segments,
    inspect_block,
    parse_channels,
)


@pytest.mark.parametrize("fs", [152.587891, 1017.252625, 24414.0625])
@pytest.mark.parametrize("side", ["left", "right"])
def test_lazy_clock_matches_numpy_at_and_between_samples(fs, side):
    axis = UniformTimeAxis(1001, fs, 0.117)
    expected = 0.117 + np.arange(1001) / fs
    values = np.r_[
        expected,
        np.nextafter(expected, -np.inf),
        np.nextafter(expected, np.inf),
        -1,
        100,
    ]
    np.testing.assert_array_equal(
        np.searchsorted(axis, values, side), np.searchsorted(expected, values, side)
    )
    for value in values[::37]:
        assert np.searchsorted(axis, value, side) == np.searchsorted(
            expected, value, side
        )
    assert axis[-1] == expected[-1]
    np.testing.assert_equal(axis[::100], expected[::100])
    with pytest.raises(TypeError):
        np.asarray(axis)


def test_file_signal_gaps_chunk_boundaries_and_bounds(tmp_path):
    path = tmp_path / "signal"
    values = np.array([9, 2, 3, 4, 8, 6, 7], dtype="<i2")
    path.write_bytes(b"header" + values.tobytes())
    signal = FileSignal(
        [FileSegment(path, 2, 4, 6), FileSegment(path, 9, 3, 14)], "<i2", size=14
    )
    expected = np.array(
        [np.nan, np.nan, 9, 2, 3, 4, np.nan, np.nan, np.nan, 8, 6, 7, np.nan, np.nan]
    )
    np.testing.assert_equal(signal[:], expected)
    np.testing.assert_equal(signal[::2], expected[::2])
    np.testing.assert_equal(signal[5:10], expected[5:10])
    assert np.isnan(signal[-1])
    with pytest.raises(IndexError):
        signal[14]
    with pytest.raises(TypeError):
        np.asarray(signal)
    with pytest.raises(ValueError, match="Truncated"):
        FileSignal([FileSegment(path, 0, 100, 0)], "<f4")
    with pytest.raises(ValueError, match="Overlapping"):
        FileSignal([FileSegment(path, 0, 4, 6), FileSegment(path, 3, 1, 6)], "<i2")


def test_bounded_decimation_preserves_peaks_and_gaps(tmp_path, monkeypatch):
    path = tmp_path / "samples"
    values = np.zeros(100_000, dtype="<f4")
    values[3111], values[69999] = -321, 987
    values.tofile(path)
    signal = FileSignal(
        [FileSegment(path, 0, 50000, 0), FileSegment(path, 50100, 50000, 200000)], "<f4"
    )
    axis = UniformTimeAxis(len(signal), 1000)
    t, y = segment_for_window(axis, signal, 0, 101, 2000)
    assert len(t) <= 2000
    assert np.nanmin(y) == -321
    assert np.nanmax(y) == 987
    assert np.isnan(y).any()
    assert np.all(np.diff(t) >= 0)
    assert signal.window_envelope(axis, 0, 101, 2000)[1] is y
    # Amplitude previews must stay bounded independently of recording length.
    assert len(signal.preview()) <= 8 * 3 * 2048


def write_sev(path, data, store="EEGr", channel=1, version=3):
    raw = np.asarray(data, dtype="<f4")
    header = struct.pack(
        "<Q3sB4sHHHHBBH12s",
        40 + raw.nbytes,
        b"SEV",
        version,
        store.encode(),
        channel,
        2,
        4,
        0,
        0,
        32,
        2,
        b"\0" * 12,
    )
    path.write_bytes(header + raw.tobytes())


def test_sev_header_rates_and_gap_logs(tmp_path):
    path = tmp_path / "block_EEGr_Ch2.sev"
    write_sev(path, np.arange(8), channel=2, version=2)
    name, channel, fs, dtype, count, hour = _sev_header(path, {"EEGr": 1017.252625})
    assert (name, channel, fs, count, hour) == ("EEGr", 2, 1017.252625, 8, 0)
    (tmp_path / "EEGr_log.txt").write_text(
        "recording started at sample: 3\ngap detected. last saved sample: 6, new saved sample: 10\n"
    )
    segments = _sev_segments(path, name, hour, count, dtype)
    signal = FileSignal(segments, dtype)
    np.testing.assert_equal(
        signal[:], [np.nan, np.nan, 0, 1, 2, 3, np.nan, np.nan, np.nan, 4, 5, 6, 7]
    )


def test_tev_packet_clock_preserves_real_gap(tmp_path):
    path = tmp_path / "block.tev"
    path.write_bytes(np.arange(12, dtype="<f4").tobytes())
    stream = Stream(
        "Wavt",
        1000.00002,
        [3],
        {},
        "<f4",
        tev_path=path,
        timestamps=np.array([0.1, 0.104, 0.110]),
        offsets=np.array([0, 16, 32]),
        chunk_channels=np.array([3, 3, 3]),
        chunk_samples=4,
    )
    axis, signal = stream.signal(3)
    assert len(axis) == 114
    np.testing.assert_equal(
        signal[100:], [0, 1, 2, 3, 4, 5, 6, 7, np.nan, np.nan, 8, 9, 10, 11]
    )
    assert isinstance(axis, PacketTimeAxis)
    assert axis[100] == 0.1
    assert axis[104] == 0.104
    assert axis[110] == 0.110
    # Searching an anchored clock must use those timestamps, not rounded fs.
    full = axis[:]
    for side in ["left", "right"]:
        np.testing.assert_equal(
            np.searchsorted(axis, full, side), np.searchsorted(full, full, side)
        )


@pytest.mark.parametrize(
    "text,expected",
    [("all", [1, 2, 3, 4, 8]), ("3-1,8", [3, 2, 1, 8]), (" 2, 4 ", [2, 4])],
)
def test_channel_order(text, expected):
    assert parse_channels(text, [1, 2, 3, 4, 8]) == expected


@pytest.mark.parametrize("text", ["1,1", "0", "6", "1-100", "a", "1,,2"])
def test_invalid_channels(text):
    with pytest.raises(ValueError):
        parse_channels(text, [1, 2, 3, 4, 8])


def test_epoc_clipping_keeps_intervals_in_block():
    epoc = Epoc(
        "Bttn", np.array([-2, 8, 3, np.nan]), np.array([1, np.inf, 2, 9]), np.ones(4)
    )
    np.testing.assert_equal(epoc.intervals(10), [[0, 1], [8, 10]])


def test_discovery_never_reads_payloads_and_structtype_conversion(
    tmp_path, monkeypatch
):
    # Reproduce the SDK's unusual dict subclass: fields live in __dict__.
    class Struct(dict):
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def items(self):
            return self.__dict__.items()

    (tmp_path / "block.tsq").write_bytes(b"fake headers")
    (tmp_path / "block.Tbk").write_bytes(b"metadata")
    path = tmp_path / "block_EEGr_Ch1.sev"
    write_sev(path, np.arange(10))
    headers = SimpleNamespace(
        start_time=np.array([1000]),
        stop_time=np.array([1010]),
        tev_path=str(tmp_path / "block.tev"),
        stores=Struct(Bttn=Struct(type_str="epocs", onset=[2], offset=[3], data=[1])),
    )
    calls = []

    def read_block(path, **kwargs):
        calls.append(kwargs)
        assert kwargs == {"headers": 1}
        return headers

    sdk = ModuleType("tdt")
    sdk.read_block = read_block
    binary = ModuleType("tdt.TDTbin2py")
    binary.parse_tbk = lambda p: [Struct(StoreName="EEGr", SampleFreq="1017.252625")]
    monkeypatch.setitem(sys.modules, "tdt", sdk)
    monkeypatch.setitem(sys.modules, "tdt.TDTbin2py", binary)
    block = inspect_block(tmp_path)
    assert block.streams["EEGr"].fs == 1017.252625
    assert block.epocs["Bttn"].onset[0] == 2
    assert len(calls) == 1
    np.testing.assert_equal(block.streams["EEGr"].signal(1)[1][:], np.arange(10))


def test_invalid_block_does_not_read_sdk(tmp_path):
    pytest.importorskip("tdt")
    with pytest.raises(ValueError, match="exactly one"):
        inspect_block(tmp_path)
