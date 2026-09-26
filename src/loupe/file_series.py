"""Bounded reads for disk-backed, regularly sampled time series.

The viewer only needs indexing, time searches and a small amplitude preview.
These objects deliberately refuse implicit full-recording NumPy conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class UniformTimeAxis:
    size: int
    fs: float
    start: float = 0.0

    def __post_init__(self):
        if self.size < 0 or not np.isfinite(self.fs) or self.fs <= 0:
            raise ValueError("Invalid time axis")

    def __len__(self):
        return self.size

    def _time_at(self, index):
        return self.start + index / self.fs

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._time_at(np.arange(*key.indices(self.size), dtype=np.int64))
        index = np.asarray(key)
        index = np.where(index < 0, index + self.size, index)
        if np.any((index < 0) | (index >= self.size)):
            raise IndexError(key)
        result = self._time_at(index)
        return float(result) if result.ndim == 0 else result

    def searchsorted(self, value, side="left", sorter=None):
        if sorter is not None or side not in ("left", "right"):
            raise ValueError("Unsupported time search")
        if np.ndim(value) == 0:
            lo, hi = 0, self.size
            while lo < hi:
                mid = (lo + hi) // 2
                time = self._time_at(mid)
                before = time < value if side == "left" else time <= value
                if before:
                    lo = mid + 1
                else:
                    hi = mid
            return lo
        # Binary search compares the very same floating point timestamps as
        # __getitem__; unlike ceil(t*fs), it is exact at sample boundaries.
        values = np.asarray(value)
        lo = np.zeros(values.shape, dtype=np.int64)
        hi = np.full(values.shape, self.size, dtype=np.int64)
        while np.any(lo < hi):
            mid = (lo + hi) // 2
            times = self._time_at(mid)
            before = times < values if side == "left" else times <= values
            active = lo < hi
            lo = np.where(active & before, mid + 1, lo)
            hi = np.where(active & ~before, mid, hi)
        return int(lo) if lo.ndim == 0 else lo

    def __array_function__(self, func, types, args, kwargs):
        if func is np.searchsorted:
            return self.searchsorted(*args[1:], **kwargs)
        if func in (np.min, np.nanmin):
            return self[0]
        if func in (np.max, np.nanmax):
            return self[-1]
        return NotImplemented

    def __array__(self, *args, **kwargs):
        raise TypeError("Slice a file-backed time axis before converting to NumPy")


@dataclass(frozen=True)
class PacketTimeAxis(UniformTimeAxis):
    """Use recorded packet clock anchors instead of accumulating rounded fs.

    TDT writes fs with limited precision, so a uniform extrapolation can drift
    by samples over a day. Between anchors, samples retain their nominal rate.
    Real gaps are represented by NaNs in the paired FileSignal.
    """

    packet_starts: np.ndarray | None = None
    packet_times: np.ndarray | None = None

    def _time_at(self, index):
        packet = np.maximum(
            0, np.searchsorted(self.packet_starts, index, side="right") - 1
        )
        return (
            self.packet_times[packet] + (index - self.packet_starts[packet]) / self.fs
        )


@dataclass(frozen=True)
class FileSegment:
    """An uninterrupted run of samples at logical index ``start``."""

    path: Path
    start: int
    count: int
    offset: int


class FileSignal:
    """Read-only signal with explicit NaN gaps and memory-mapped sample runs."""

    def __init__(self, segments: list[FileSegment], dtype, size: int | None = None):
        self.dtype = np.dtype(dtype).newbyteorder("<")
        self.segments = sorted(segments, key=lambda s: s.start)
        file_sizes = {
            segment.path: segment.path.stat().st_size
            for segment in {s.path: s for s in self.segments}.values()
        }
        end = 0
        for seg in self.segments:
            if seg.start < end or seg.count <= 0 or seg.offset < 0:
                raise ValueError("Overlapping or invalid stream segments")
            if seg.offset + seg.count * self.dtype.itemsize > file_sizes[seg.path]:
                raise ValueError(f"Truncated stream file: {seg.path.name}")
            end = seg.start + seg.count
        self.size = end if size is None else max(end, size)
        self.shape = (self.size,)
        self._ends = np.array([s.start + s.count for s in self.segments])
        self._maps = {}
        self._preview = None
        self._envelope_cache = None

    def __len__(self):
        return self.size

    def __array__(self, *args, **kwargs):
        raise TypeError("Slice a file-backed signal before converting to NumPy")

    def __getitem__(self, key):
        if not isinstance(key, slice):
            if not np.isscalar(key):
                raise TypeError("FileSignal supports scalar or slice indexing")
            i = int(key)
            i = i + self.size if i < 0 else i
            if not 0 <= i < self.size:
                raise IndexError(key)
            return self[i : i + 1][0]
        first, stop, step = key.indices(self.size)
        if step != 1:
            # Do not allocate the skipped part of the recording.
            return np.array([self[i] for i in range(first, stop, step)])
        result = np.full(
            max(0, stop - first), np.nan, dtype=np.result_type(self.dtype, np.float32)
        )
        si = int(np.searchsorted(self._ends, first, side="right"))
        for index in range(si, len(self.segments)):
            seg = self.segments[index]
            if seg.start >= stop:
                break
            lo, hi = max(first, seg.start), min(stop, seg.start + seg.count)
            if hi <= lo:
                continue
            mapped = self._maps.get(seg.path)
            if mapped is None:
                mapped = self._maps[seg.path] = np.memmap(
                    seg.path, mode="r", dtype=np.uint8
                )
            byte = seg.offset + (lo - seg.start) * self.dtype.itemsize
            result[lo - first : hi - first] = np.ndarray(
                (hi - lo,), dtype=self.dtype, buffer=mapped, offset=byte
            )
        return result

    def preview(self):
        """A bounded amplitude sample for initial scale/centering, cached once."""
        if self._preview is None:
            chunks = []
            # Sample valid spans, not the missing leading/trailing data.
            if self.segments:
                for si in np.unique(
                    np.linspace(0, len(self.segments) - 1, 8, dtype=int)
                ):
                    seg = self.segments[si]
                    for fraction in (0.0, 0.5, 0.95):
                        start = seg.start + int(max(0, seg.count - 2048) * fraction)
                        chunks.append(
                            self[start : min(start + 2048, seg.start + seg.count)]
                        )
            self._preview = np.concatenate(chunks) if chunks else np.array([np.nan])
        return self._preview

    def window_envelope(self, t: UniformTimeAxis, t0, t1, max_pts=4000):
        """Peak envelope with bounded buffers, even for a day-long window.

        Cache the last viewport. Bins intersecting an acquisition gap remain
        blank, so overview decimation never draws through missing samples.
        """
        key = (t.start, t.fs, float(t0), float(t1), int(max_pts))
        if self._envelope_cache is not None and self._envelope_cache[0] == key:
            return self._envelope_cache[1]
        first = max(0, t.searchsorted(t0) - 1)
        stop = min(len(t), t.searchsorted(t1) + 1)
        n = stop - first
        if n <= max_pts:
            result = (t[first:stop], self[first:stop])
        else:
            width = max(1, int(np.ceil(n / max(1, max_pts // 2))))
            xs, ys = [], []
            # Each iteration reads at most ~1M samples (or one enormous bin).
            batch = max(1, 1_000_000 // width)
            for start in range(first, stop, batch * width):
                end = min(stop, start + batch * width)
                values = self[start:end]
                offsets = np.arange(0, len(values), width)
                low = np.fmin.reduceat(values, offsets)
                high = np.fmax.reduceat(values, offsets)
                missing = np.logical_or.reduceat(~np.isfinite(values), offsets)
                low[missing] = high[missing] = np.nan
                mid = (
                    start
                    + (offsets + np.minimum(offsets + width, len(values)) - 1) // 2
                )
                xs.append(np.repeat(t[mid], 2))
                ys.append(np.column_stack((low, high)).ravel())
            result = (np.concatenate(xs), np.concatenate(ys))
        self._envelope_cache = (key, result)
        return result


def amplitude_preview(y):
    """Keep regular arrays exact; use a bounded preview for file-backed data."""
    return y.preview() if isinstance(y, FileSignal) else np.asarray(y)
