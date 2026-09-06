"""Pure utility functions used by the viewer's hot rendering path.

These are pure numpy / stdlib helpers — no Qt, no app state. They live in
their own module so the trace/dense/heatmap refresh paths don't pay any
indirection cost (module-level function calls are essentially free) and so
the rest of ``app.py`` doesn't need to scroll past them.

``segment_for_window`` in particular is called per-paint from
``LoupeApp._refresh_curves`` and must stay vectorised — do not wrap it in
classes or method dispatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg

if TYPE_CHECKING:
    from loupe.series import SampleMarkers


def nice_time_range(t_arrays):
    vals = [(np.nanmin(t), np.nanmax(t)) for t in t_arrays if t is not None and len(t)]
    return (
        (0.0, 1.0)
        if not vals
        else (float(min(v[0] for v in vals)), float(max(v[1] for v in vals)))
    )


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


def _scatter_kwargs_for_marker(marker: "SampleMarkers") -> dict:
    """Map a :class:`SampleMarkers`'s style fields to ``pg.ScatterPlotItem`` kwargs.

    'o' renders a filled circle (no pen) with alpha applied to the fill.
    'x' and other pyqtgraph symbols render as outlined strokes with alpha
    applied to the pen.
    """
    base = pg.mkColor(marker.color)
    tinted = pg.mkColor(base)
    tinted.setAlpha(int(max(0, min(255, marker.alpha))))
    size = float(marker.size)
    if marker.marker == "o":
        return {
            "symbol": "o",
            "pen": None,
            "brush": pg.mkBrush(tinted),
            "size": size,
        }
    return {
        "symbol": marker.marker,
        "pen": pg.mkPen(tinted, width=1.5),
        "brush": None,
        "size": size,
    }


#: ``SampleMarkers.marker`` value that renders as full-height vertical lines
#: (one ``pg.PlotCurveItem`` with ``connect="pairs"`` per series) instead of a
#: scatter symbol at the sample's value.
VLINE_MARKER = "vline"


def is_vline_marker(marker) -> bool:
    """True when a :class:`SampleMarkers` set renders as vertical lines."""
    return getattr(marker, "marker", None) == VLINE_MARKER


def _vline_pen_for_marker(marker) -> "pg.QtGui.QPen":
    """Pen for a ``"vline"`` marker set: its color + alpha, ``size`` as width (px)."""
    tinted = pg.mkColor(marker.color)
    tinted.setAlpha(int(max(0, min(255, marker.alpha))))
    pen = pg.mkPen(tinted, width=float(marker.size))
    pen.setCosmetic(True)  # crisp at every zoom, like the global event lines
    return pen


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
    # assign each timestamp to a bin index (0..bins-1).
    # bi is monotonically non-decreasing because ts is sorted ascending,
    # so we can skip an explicit sort and feed it directly to searchsorted.
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
