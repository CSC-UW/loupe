"""Shared heatmap helpers used by both the renderer (app.py) and the
heatmap-controls dialog (dialogs.py).

Kept in a tiny module of its own so neither file has to import from the
other (which would be circular: app.py instantiates the dialog, the dialog
needs the cache-token helper, and the renderer in app.py uses the same
helper to key its per-frame cache).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.colors import Colormap


# Build a mip-map lazily for arrays exceeding this size (rows * cols).
ARRAY_MIPMAP_THRESHOLD = 5_000_000
ARRAY_MIPMAP_TARGET_MIN_COLS = 1500


# Built-in colormap suggestions for the Heatmap Plot Controls dialog.
ARRAY_COLORMAP_PRESETS = (
    "magma",
    "viridis",
    "plasma",
    "inferno",
    "cividis",
    "gray",
    "RdBu_r",
    "coolwarm",
    "seismic",
    "hot",
)


def _colormap_display_name(cmap: "str | Colormap") -> str:
    """Return a human-readable string name for a colormap value.

    Used for the GUI combo box (which only displays/sets text). Falls back to
    ``"custom"`` if a Colormap instance has no ``.name`` attribute.
    """
    if isinstance(cmap, str):
        return cmap
    return str(getattr(cmap, "name", "custom"))


def _colormap_cache_token(cmap: "str | Colormap"):
    """Hashable, equality-stable token identifying a colormap value.

    Used as part of the per-frame heatmap-render cache key (which compares by
    ``==``). Strings compare by value; Colormap instances by ``id()`` so a
    new instance forces a re-render.
    """
    if isinstance(cmap, str):
        return cmap
    return ("__cmap_obj__", id(cmap))
