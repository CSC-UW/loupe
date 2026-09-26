"""Metadata-only TDT indexing. No stream samples or snippet waveforms are read."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re
import struct
import threading

import numpy as np

from loupe.file_series import FileSegment, FileSignal, UniformTimeAxis, PacketTimeAxis

FORMATS = {0: "<f4", 1: "<i4", 2: "<i2", 3: "i1", 4: "<f8"}


class Cancelled(Exception):
    """A user cancelled a metadata/load job."""


def check_cancel(cancel: threading.Event | None):
    if cancel is not None and cancel.is_set():
        raise Cancelled()


@dataclass
class Stream:
    name: str
    fs: float
    channels: list[int]
    sources: dict[int, list[FileSegment]]
    dtype: str
    # TEV stores retain only the small chunk index, not sample payloads.
    tev_path: Path | None = None
    timestamps: np.ndarray | None = None
    offsets: np.ndarray | None = None
    chunk_channels: np.ndarray | None = None
    chunk_samples: int = 0
    error: str | None = None
    packet_sources: dict = field(default_factory=dict)

    def signal(self, channel: int) -> tuple[UniformTimeAxis, FileSignal]:
        if self.error:
            raise ValueError(f"{self.name}: {self.error}")
        if channel not in self.channels:
            raise ValueError(f"{self.name}: unknown channel {channel}")
        segments = self.sources.get(channel, [])
        packet = None
        if channel in self.packet_sources:
            packet = self.packet_sources[channel]
        elif self.tev_path is not None:
            mask = (
                self.chunk_channels == channel
                if len(self.chunk_channels) > 1
                else slice(None)
            )
            ts, offsets = self.timestamps[mask], self.offsets[mask]
            packet = (self.tev_path, ts, offsets, self.chunk_samples)
        if packet is not None:
            path, ts, offsets, samples = packet
            if not len(ts):
                raise ValueError(f"{self.name}: no samples for channel {channel}")
            # The SDK stores fs as float32 in TSQ. Rounding each chunk delta
            # avoids accumulating that quantization into spurious sample gaps.
            starts = np.r_[
                round(ts[0] * self.fs),
                round(ts[0] * self.fs) + np.cumsum(np.rint(np.diff(ts) * self.fs)),
            ].astype(np.int64)
            counts = np.full(len(starts), samples, dtype=np.int64)
            if len(starts) > 1 and np.any(np.diff(starts) < samples):
                raise ValueError(f"{self.name}: overlapping stream packets")
            adjacent = (np.diff(starts) == samples) & (
                np.diff(offsets.astype(np.int64))
                == samples * np.dtype(self.dtype).itemsize
            )
            edges = np.r_[0, np.flatnonzero(~adjacent) + 1]
            lengths = np.add.reduceat(counts, edges)
            segments = [
                FileSegment(path, int(starts[i]), int(n), int(offsets[i]))
                for i, n in zip(edges, lengths)
            ]
        signal = FileSignal(segments, self.dtype)
        if packet is not None:
            return PacketTimeAxis(
                len(signal), self.fs, packet_starts=starts, packet_times=ts
            ), signal
        return UniformTimeAxis(len(signal), self.fs), signal


@dataclass
class Epoc:
    name: str
    onset: np.ndarray
    offset: np.ndarray
    values: np.ndarray

    def intervals(self, duration: float):
        if len(self.onset) != len(self.offset):
            raise ValueError(f"{self.name}: onset/offset counts differ")
        start = np.maximum(self.onset, 0)
        end = np.minimum(self.offset, duration)
        valid = np.isfinite(start) & np.isfinite(end) & (end > start)
        intervals = np.column_stack((start[valid], end[valid]))
        return intervals[np.argsort(intervals[:, 0], kind="stable")]


@dataclass
class Snips:
    name: str
    timestamps: np.ndarray
    channels: np.ndarray
    sortcodes: np.ndarray


@dataclass
class Block:
    path: Path
    duration: float
    info: dict
    streams: dict[str, Stream]
    epocs: dict[str, Epoc]
    snips: dict[str, Snips]
    videos: list[Path]
    fingerprint: list[dict]
    warnings: list[str] = field(default_factory=list)


def fingerprint(path: Path) -> list[dict]:
    suffixes = {".tsq", ".tev", ".sev", ".tbk"}
    return [
        {"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in sorted(path.iterdir())
        if p.is_file() and (p.suffix.lower() in suffixes or p.name.endswith("_log.txt"))
    ]


def _sev_header(path: Path, rates: dict):
    with path.open("rb") as f:
        raw = f.read(40)
    if len(raw) != 40:
        raise ValueError(f"Truncated SEV header: {path.name}")
    _, magic, version, raw_name, channel, _, width, _, fmt, decimate, rate, _ = (
        struct.unpack("<Q3sB4sHHHHBBH12s", raw)
    )
    if magic.lower() != b"sev" or version not in (1, 2, 3):
        raise ValueError(f"Unsupported SEV header/version in {path.name}: {version}")
    match = re.search(r"_(.{4})_[Cc]h\d+(?:-\d+h)?\.sev$", path.name, re.IGNORECASE)
    name = (
        raw_name.decode("ascii")
        if version >= 3
        else (match[1] if match else raw_name.decode("ascii"))
    )
    fmt &= 7
    if fmt not in FORMATS or width != np.dtype(FORMATS[fmt]).itemsize or not decimate:
        raise ValueError(f"Unsupported SEV sample format in {path.name}")
    fs = float(rates.get(name, 0)) or 2.0 ** (rate - 12) * 25_000_000 / decimate
    size = path.stat().st_size - 40
    if size % width:
        raise ValueError(f"Incomplete sample at end of {path.name}")
    hour_match = re.search(r"-(\d+)h(?=\.sev$)", path.name, re.IGNORECASE)
    return (
        name,
        channel,
        fs,
        FORMATS[fmt],
        size // width,
        int(hour_match[1]) if hour_match else 0,
    )


def _sev_segments(path, store, hour, count, dtype, previous_end=0):
    log_name = f"{store}{'-' + str(hour) + 'h' if hour else ''}_log.txt"
    log = path.parent / log_name
    start_sample = previous_end + 1
    gaps = []
    if log.exists():
        text = log.read_text(errors="replace")
        match = re.search(r"recording started at sample:\s*(\d+)", text, re.I)
        if match is None:
            raise ValueError(f"Missing start sample in {log_name}")
        start_sample = int(match[1])
        gaps = [
            (int(a), int(b))
            for a, b in re.findall(
                r"gap detected\.\s*last saved sample:\s*(\d+),\s*new saved sample:\s*(\d+)",
                text,
                re.I,
            )
        ]
    segments = []
    logical, consumed = start_sample - 1, 0
    for last, new in gaps:
        length = last - logical  # logs use one-based inclusive sample numbers
        if length < 0 or new <= last or consumed + length > count:
            raise ValueError(f"Invalid gap records in {log_name}")
        if length:
            segments.append(
                FileSegment(
                    path, logical, length, 40 + consumed * np.dtype(dtype).itemsize
                )
            )
        consumed += length
        logical = new - 1
    if consumed < count:
        segments.append(
            FileSegment(
                path,
                logical,
                count - consumed,
                40 + consumed * np.dtype(dtype).itemsize,
            )
        )
    return segments


def inspect_block(path, *, cancel=None, progress=lambda message: None) -> Block:
    """Read SDK headers once and index SEV/TEV stores without loading samples."""
    try:
        import tdt
        from tdt.TDTbin2py import parse_tbk
    except ImportError as exc:
        raise ImportError(
            "Install the TDT extension with: uv add 'loupe[tdt]'"
        ) from exc
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Not a block directory: {path}")
    files = list(path.iterdir())
    tsq = [p for p in files if p.suffix.lower() == ".tsq"]
    if len(tsq) != 1:
        raise ValueError("Choose one TDT block folder containing exactly one .tsq file")
    progress("Reading TDT store metadata…")
    before = fingerprint(path)
    headers = tdt.read_block(str(path), headers=1)
    check_cancel(cancel)
    # StructType subclasses dict but keeps fields in __dict__; dict(value)
    # and value.get() silently return empty data in SDK 0.7.2.
    stores = {key: dict(value.items()) for key, value in headers.stores.items()}
    rates = {}
    tbk = [p for p in files if p.suffix.lower() == ".tbk"]
    if tbk:
        for record in parse_tbk(str(tbk[0])):
            record = dict(record.items())
            if float(record.get("SampleFreq", 0)) > 0:
                rates[record["StoreName"]] = float(record["SampleFreq"])
    start = float(np.ravel(headers.start_time)[0])
    stop = float(np.ravel(headers.stop_time)[0])
    duration = stop - start if np.isfinite(stop) and stop > start else 0
    streams, epocs, snips, warnings = {}, {}, {}, []
    sev_files = []
    for file in files:
        if file.suffix.lower() == ".sev" and not file.name.startswith("._"):
            sev_files.append((file, _sev_header(file, rates)))
    previous_hours = {}
    for file, (name, channel, fs, dtype, count, hour) in sorted(
        sev_files, key=lambda x: x[1][-1]
    ):
        check_cancel(cancel)
        stream = streams.setdefault(name, Stream(name, fs, [], {}, dtype))
        if stream.fs != fs or stream.dtype != dtype:
            raise ValueError(f"Inconsistent SEV headers for {name}")
        previous = stream.sources.setdefault(channel, [])
        if channel not in stream.channels:
            stream.channels.append(channel)
        if previous and hour == 0:
            raise ValueError(f"Duplicate SEV file for {name} channel {channel}")
        previous_end = previous[-1].start + previous[-1].count if previous else 0
        if (name, channel) in previous_hours and hour != previous_hours[
            (name, channel)
        ] + 1:
            if not (path / f"{name}-{hour}h_log.txt").exists():
                raise ValueError(
                    f"{file.name}: missing hour segment; no log gives its start time"
                )
        if hour and not previous and not (path / f"{name}-{hour}h_log.txt").exists():
            raise ValueError(
                f"{file.name}: first segment is missing; no log gives its start time"
            )
        previous.extend(_sev_segments(file, name, hour, count, dtype, previous_end))
        previous_hours[(name, channel)] = hour
    for name, record in stores.items():
        check_cancel(cancel)
        kind = record["type_str"]
        if kind == "streams" and name in streams:
            # One-SEV-per-channel blocks can use every native packet clock
            # anchor and offset directly. This also reveals gaps even when
            # the optional SEV log was not copied with the recording.
            stream = streams[name]
            packet_channels = np.asarray(record["chan"]).ravel()
            times = np.asarray(record["ts"]).ravel()
            offsets = np.asarray(record["data"]).ravel()
            for channel, segments in stream.sources.items():
                if len(packet_channels) == 1 and packet_channels[0] != channel:
                    continue
                paths = {s.path for s in segments}
                if len(paths) != 1:
                    continue
                selected = (
                    packet_channels == channel
                    if len(packet_channels) > 1
                    else slice(None)
                )
                ts, off = times[selected], offsets[selected]
                samples = (
                    (int(record["size"]) - 10) * 4 // np.dtype(stream.dtype).itemsize
                )
                source_path = next(iter(paths))
                if len(ts) and np.all(np.diff(off.astype(np.int64)) > 0):
                    if (
                        int(off[-1]) + samples * np.dtype(stream.dtype).itemsize
                        != source_path.stat().st_size
                    ):
                        warnings.append(
                            f"{name} Ch {channel}: SEV tail differs from TSQ index; using SEV/log timing"
                        )
                    else:
                        stream.packet_sources[channel] = (source_path, ts, off, samples)
        elif kind == "streams":
            channels = np.asarray(record["chan"]).ravel()
            dtype = FORMATS.get(int(record["dform"]))
            stream = Stream(
                name,
                float(rates.get(name, record["fs"])),
                [int(c) for c in np.unique(channels)],
                {},
                dtype or "<f4",
            )
            if record.get("ucf"):
                stream.error = "Expected SEV files are missing"
            elif dtype is None:
                stream.error = "Unsupported stream sample format"
            else:
                stream.tev_path = Path(headers.tev_path)
                stream.timestamps = np.asarray(record["ts"]).ravel()
                stream.offsets = np.asarray(record["data"]).ravel()
                stream.chunk_channels = channels
                stream.chunk_samples = (
                    (int(record["size"]) - 10) * 4 // np.dtype(dtype).itemsize
                )
            streams[name] = stream
        elif kind == "epocs":
            epocs[name] = Epoc(
                name,
                np.asarray(record["onset"]).ravel(),
                np.asarray(record["offset"]).ravel(),
                np.asarray(record["data"]).ravel(),
            )
        elif kind == "snips":
            times = np.asarray(record["ts"]).ravel()
            channels = np.asarray(record["chan"]).ravel()
            if len(channels) == 1:
                channels = np.full(len(times), channels[0], dtype=np.uint16)
            snips[name] = Snips(
                name, times, channels, np.asarray(record["sortcode"]).ravel()
            )
    for stream in streams.values():
        stream.channels.sort()
        for segments in stream.sources.values():
            if segments:
                duration = max(
                    duration, (segments[-1].start + segments[-1].count) / stream.fs
                )
        if stream.error:
            warnings.append(f"{stream.name}: {stream.error}")
    if duration <= 0:
        raise ValueError("Block has no usable recording duration")
    if before != fingerprint(path):
        raise ValueError(
            "The block changed during discovery; finish acquisition and open it again"
        )
    progress("Block metadata ready")
    info = {
        "block": path.name,
        "start_utc": datetime.fromtimestamp(start, timezone.utc).isoformat(),
        "duration_s": duration,
        "scalars": [n for n, s in stores.items() if s["type_str"] == "scalars"],
    }
    return Block(
        path,
        duration,
        info,
        streams,
        epocs,
        snips,
        sorted(
            p for p in files if p.suffix.lower() in {".avi", ".mp4", ".mkv", ".mov"}
        ),
        before,
        warnings,
    )


def parse_channels(text: str, available: list[int]) -> list[int]:
    """Parse ordered channel IDs such as ``1-4, 8, 6``; never reinterpret IDs."""
    if not text.strip() or text.strip().lower() == "all":
        return list(available)
    result = []
    for item in text.split(","):
        item = item.strip()
        if re.fullmatch(r"\d+", item):
            values = [int(item)]
        elif re.fullmatch(r"\d+\s*-\s*\d+", item):
            lo, hi = [int(v) for v in item.split("-")]
            values = range(lo, hi + (1 if hi >= lo else -1), 1 if hi >= lo else -1)
        else:
            raise ValueError(f"Invalid channel selection: {item!r}. Use 1-4, 8 or all.")
        for value in values:
            if value not in available:
                raise ValueError(
                    f"Channel {value} is unavailable; available IDs: {available}"
                )
            if value in result:
                raise ValueError(f"Duplicate channel {value}")
            result.append(value)
    return result
