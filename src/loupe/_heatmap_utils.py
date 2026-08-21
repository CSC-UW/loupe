"""Shared heatmap helpers used by both the renderer (app.py) and the
heatmap-controls dialog (dialogs.py).

Kept in a tiny module of its own so neither file has to import from the
other (which would be circular: app.py instantiates the dialog, the dialog
needs the cache-token helper, and the renderer in app.py uses the same
helper to key its per-frame cache).
"""

from __future__ import annotations

import math
from numbers import Real
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


def _normalize_nan_shade(
    value: "bool | str | tuple[str, float]",
) -> tuple[int, int, int, float] | None:
    """Validate and normalize ``HeatmapConfig.shade_nans``.

    ``False`` disables shading. A hex color uses the default 0.7 alpha, while
    ``(hex_color, alpha)`` accepts an explicit alpha in the inclusive 0..1
    range. The returned tuple keeps alpha as a float so View-Config round trips
    do not lose precision through uint8 conversion.
    """
    if value is False:
        return None

    alpha: object = 0.7
    color: object = value
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(
                "shade_nans must be False, '#RRGGBB', or "
                "('#RRGGBB', alpha)."
            )
        color, alpha = value

    if not isinstance(color, str):
        raise ValueError(
            "shade_nans must be False, '#RRGGBB', or ('#RRGGBB', alpha)."
        )
    hex_color = color.strip()
    if (
        len(hex_color) != 7
        or not hex_color.startswith("#")
        or not all(c in "0123456789abcdefABCDEF" for c in hex_color[1:])
    ):
        raise ValueError(
            f"shade_nans color must be a '#RRGGBB' hex string, got {color!r}."
        )

    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise ValueError("shade_nans alpha must be a number between 0 and 1.")
    alpha_value = float(alpha)
    if not math.isfinite(alpha_value) or not 0.0 <= alpha_value <= 1.0:
        raise ValueError("shade_nans alpha must be between 0 and 1.")

    return (
        int(hex_color[1:3], 16),
        int(hex_color[3:5], 16),
        int(hex_color[5:7], 16),
        alpha_value,
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
