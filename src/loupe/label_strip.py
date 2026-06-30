"""Reusable renderer for interval-label color bands.

:class:`LabelBandRenderer` draws each interval label as a translucent shaded
band into a single target ``PlotItem``. It is deliberately *host-agnostic*: the
pinned "label strip" above the plot stack uses one today (see
:class:`loupe.app.LoupeApp`), and an in-grid "label subplot" host can reuse the
same class later by handing it a different ``plot_item`` (and, if the host
should follow the view window, the same windowed ``entries_provider``).

The add / update / remove diff mirrors the hypnogram label rendering on
``LoupeApp`` (``_add/_update/_remove_hypnogram_interval_label_visual``) so the
behavior matches exactly — in particular, a surviving row whose span or name
changed in place (a partial overwrite or ``merge_adjacent``) is repositioned /
recolored rather than left stale, and an unchanged row keeps the *same* region
object across syncs.
"""

from __future__ import annotations

from typing import Callable, Iterable

import pyqtgraph as pg


def _default_region_factory(a: float, b: float, color: tuple) -> pg.LinearRegionItem:
    """Build the standard non-interactive shaded band used for one label."""
    return pg.LinearRegionItem(
        values=(a, b),
        brush=pg.mkBrush(*color),
        pen=pg.mkPen(*color),
        movable=False,
    )


def _default_set_region_color(region: pg.LinearRegionItem, color: tuple) -> None:
    """Apply an RGBA ``color`` to a band's fill and boundary lines.

    Mirrors ``LoupeApp._set_region_color`` so the strip restyles identically to
    the trace / hypnogram regions when the alpha multiplier changes.
    """
    region.setBrush(pg.mkBrush(*color))
    pen = pg.mkPen(*color)
    for line in region.lines:
        line.setPen(pen)


class LabelBandRenderer:
    """Draw interval-label color bands into one target ``PlotItem``.

    Parameters
    ----------
    plot_item:
        The pyqtgraph ``PlotItem`` (or any object exposing ``addItem`` /
        ``removeItem``) the bands are drawn into.
    entries_provider:
        Zero-arg callable returning the rows to draw as ``[(key, row), ...]``,
        where ``key`` is a stable id (the label ``row_id``) and ``row`` exposes
        ``start``, ``end`` and ``label``. Pass a windowed provider (e.g.
        ``LoupeApp._visible_interval_label_entries``) for a view-following strip,
        or a full-set provider for an overview band.
    color_provider:
        Callable mapping a label name to an RGBA tuple (e.g.
        ``LoupeApp._interval_label_brush_color`` — already alpha-scaled).
    region_factory:
        ``(a, b, color) -> region`` building one band item. Injectable so the
        diff logic is unit-testable without a Qt event loop.
    set_region_color:
        ``(region, color) -> None`` restyling an existing band in place.
    z:
        Z-value applied to every band. The strip owns its viewbox, so the
        default ``0`` is fine; exposed for hosts that layer bands against other
        content.
    """

    def __init__(
        self,
        plot_item,
        *,
        entries_provider: Callable[[], Iterable[tuple]],
        color_provider: Callable[[str], tuple],
        region_factory: Callable[[float, float, tuple], object] = _default_region_factory,
        set_region_color: Callable[[object, tuple], None] = _default_set_region_color,
        z: float = 0,
    ) -> None:
        self._plot = plot_item
        self._entries_provider = entries_provider
        self._color_provider = color_provider
        self._region_factory = region_factory
        self._set_region_color = set_region_color
        self._z = z
        # key -> region item currently on the plot.
        self._visuals: dict = {}
        # key -> (start, end, name) last drawn, for in-place stale detection.
        self._drawn: dict = {}

    # ---- internal add / update / remove (mirrors the hypnogram diff) ----

    def _add(self, key, row) -> None:
        a, b, name = float(row.start), float(row.end), str(row.label)
        region = self._region_factory(a, b, self._color_provider(name))
        setz = getattr(region, "setZValue", None)
        if setz is not None:
            setz(self._z)
        self._plot.addItem(region)
        self._visuals[key] = region
        self._drawn[key] = (a, b, name)

    def _update(self, key, row) -> None:
        region = self._visuals.get(key)
        if region is None:
            return
        a, b, name = float(row.start), float(row.end), str(row.label)
        drawn = self._drawn.get(key)
        if drawn == (a, b, name):
            return
        if drawn is None or drawn[0] != a or drawn[1] != b:
            region.setRegion((a, b))
        if drawn is None or drawn[2] != name:
            self._set_region_color(region, self._color_provider(name))
        self._drawn[key] = (a, b, name)

    def _remove(self, key) -> None:
        self._drawn.pop(key, None)
        region = self._visuals.pop(key, None)
        if region is None:
            return
        try:
            self._plot.removeItem(region)
        except Exception:
            pass

    # ---- public API ----

    def sync(self, *, force_rebuild: bool = False) -> None:
        """Reconcile the drawn bands with the current entries.

        Adds bands for new keys, repositions/recolors keys whose row changed in
        place, and removes bands whose key is no longer present. With
        ``force_rebuild`` every band is torn down and recreated first.
        """
        if force_rebuild:
            self.clear()
        entries = list(self._entries_provider())
        new_keys = {key for key, _ in entries}
        for key in list(self._visuals):
            if key not in new_keys:
                self._remove(key)
        for key, row in entries:
            if key in self._visuals:
                self._update(key, row)
            else:
                self._add(key, row)

    def refresh_colors(self) -> None:
        """Re-apply the current color for every drawn band (e.g. after the
        interval-label alpha multiplier changes)."""
        for key, region in self._visuals.items():
            drawn = self._drawn.get(key)
            name = drawn[2] if drawn is not None else ""
            self._set_region_color(region, self._color_provider(name))

    def clear(self) -> None:
        """Remove every band from the target plot."""
        for key in list(self._visuals):
            self._remove(key)
        self._drawn.clear()
