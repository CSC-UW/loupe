"""State definitions: keymap (state -> hotkeys) and per-state label colors.

Loaded from a JSON file and/or programmatic kwargs. Multiple hotkeys per state
are supported via either forward (``{key: state}``) or inverse
(``{state: [keys]}``) JSON forms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "LoupeConfigError",
    "StateConfig",
    "load_state_config",
    "normalize_keymap",
    "parse_color",
]

RGBA = tuple[int, int, int, int]


class LoupeConfigError(ValueError):
    """Raised on invalid or missing state-definition configuration."""


# ---------------------------------------------------------------------------
# Color parsing
# ---------------------------------------------------------------------------


def parse_color(value: Any) -> RGBA:
    """Coerce a color to ``(R, G, B, A)`` of ints in 0-255.

    Accepts: ``"#RRGGBB"``, ``"#RRGGBBAA"``, ``[R, G, B]``, ``[R, G, B, A]``,
    or a tuple of the same shapes.
    """
    if isinstance(value, str):
        s = value.strip().lstrip("#")
        if len(s) not in (6, 8) or not all(c in "0123456789abcdefABCDEF" for c in s):
            raise LoupeConfigError(f"Invalid hex color string: {value!r}")
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        a = int(s[6:8], 16) if len(s) == 8 else 255
        return (r, g, b, a)
    if isinstance(value, (list, tuple)):
        if len(value) == 3:
            r, g, b = value
            a = 255
        elif len(value) == 4:
            r, g, b, a = value
        else:
            raise LoupeConfigError(
                f"Color must have 3 or 4 components, got {len(value)}: {value!r}"
            )
        try:
            return (int(r), int(g), int(b), int(a))
        except (TypeError, ValueError) as exc:
            raise LoupeConfigError(f"Color components must be integers: {value!r}") from exc
    raise LoupeConfigError(f"Unrecognized color value: {value!r}")


# ---------------------------------------------------------------------------
# Keymap normalization
# ---------------------------------------------------------------------------


def _looks_like_inverse(raw: dict) -> bool:
    """Heuristic: if any value is a list, treat the dict as state -> [keys]."""
    return any(isinstance(v, (list, tuple)) for v in raw.values())


def normalize_keymap(raw: dict | None) -> dict[str, list[str]]:
    """Return ``state -> [keys]`` from either forward or inverse JSON forms.

    Forward form: ``{"w": "Wake", "1": "NREM"}``
    Inverse form: ``{"Wake": ["w", "W"], "NREM": ["1", "n"]}`` (or single string).

    Raises :class:`LoupeConfigError` if any single key is bound to two different
    states or if the input is malformed.
    """
    if not raw:
        return {}

    state_to_keys: dict[str, list[str]] = {}
    seen_keys: dict[str, str] = {}  # key -> state, for conflict detection

    if _looks_like_inverse(raw):
        for state, keys in raw.items():
            if not isinstance(state, str) or not state:
                raise LoupeConfigError(f"State name must be a non-empty string: {state!r}")
            if isinstance(keys, str):
                key_list = [keys]
            elif isinstance(keys, (list, tuple)):
                key_list = list(keys)
            else:
                raise LoupeConfigError(
                    f"Keys for state {state!r} must be a list or string, got {type(keys).__name__}"
                )
            normalized: list[str] = []
            for k in key_list:
                if not isinstance(k, str) or not k.strip():
                    raise LoupeConfigError(
                        f"Hotkey for state {state!r} must be a non-empty string, got {k!r}"
                    )
                k = k.strip()
                if k in seen_keys and seen_keys[k] != state:
                    raise LoupeConfigError(
                        f"Hotkey {k!r} is bound to both {seen_keys[k]!r} and {state!r}"
                    )
                seen_keys[k] = state
                if k not in normalized:
                    normalized.append(k)
            if normalized:
                state_to_keys.setdefault(state, []).extend(
                    k for k in normalized if k not in state_to_keys.get(state, [])
                )
    else:
        for key, state in raw.items():
            if not isinstance(key, str) or not key.strip():
                raise LoupeConfigError(f"Hotkey must be a non-empty string, got {key!r}")
            if not isinstance(state, str) or not state:
                raise LoupeConfigError(
                    f"State name for hotkey {key!r} must be a non-empty string, got {state!r}"
                )
            key = key.strip()
            if key in seen_keys and seen_keys[key] != state:
                raise LoupeConfigError(
                    f"Hotkey {key!r} is bound to both {seen_keys[key]!r} and {state!r}"
                )
            seen_keys[key] = state
            keys = state_to_keys.setdefault(state, [])
            if key not in keys:
                keys.append(key)

    return state_to_keys


# ---------------------------------------------------------------------------
# StateConfig
# ---------------------------------------------------------------------------


@dataclass
class StateConfig:
    """Resolved keymap + colors for a session.

    Attributes
    ----------
    keys_for_state
        Mapping from state name to its ordered list of hotkeys.
    label_colors
        Mapping from state name to ``(R, G, B, A)``.
    """

    keys_for_state: dict[str, list[str]] = field(default_factory=dict)
    label_colors: dict[str, RGBA] = field(default_factory=dict)

    @property
    def key_to_state(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for state, keys in self.keys_for_state.items():
            for k in keys:
                out[k] = state
        return out

    def hotkeys_label(self, state: str) -> str:
        """Render ``"[w, W]"`` for a state with multiple keys; ``""`` if none."""
        keys = self.keys_for_state.get(state, [])
        if not keys:
            return ""
        return "[" + ", ".join(keys) + "]"

    def state_with_hotkeys(self, state: str) -> str:
        """Render ``"Wake [w, W]"`` for display in tables and dialogs."""
        suffix = self.hotkeys_label(state)
        return f"{state} {suffix}" if suffix else state


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_json_state_definitions(path: Path) -> tuple[dict, dict]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise LoupeConfigError(f"Invalid JSON in {path}: {exc}") from exc
    keymap = data.get("keymap", {}) or {}
    colors = data.get("label_colors", {}) or {}
    if not isinstance(keymap, dict):
        raise LoupeConfigError(f"'keymap' in {path} must be an object")
    if not isinstance(colors, dict):
        raise LoupeConfigError(f"'label_colors' in {path} must be an object")
    return keymap, colors


def _coerce_label_colors(raw: dict) -> dict[str, RGBA]:
    return {state: parse_color(color) for state, color in raw.items()}


def load_state_config(
    *,
    path: str | Path | None = None,
    keymap: dict | None = None,
    label_colors: dict | None = None,
    package_default: bool = True,
) -> StateConfig:
    """Resolve a :class:`StateConfig` from any combination of sources.

    Resolution order:

    1. Start empty.
    2. If ``path`` is given, merge its ``keymap`` and ``label_colors``.
       Otherwise, if ``package_default`` is true and a file named
       ``state_definitions.json`` sits next to ``app.py``, merge it.
    3. ``keymap`` and ``label_colors`` kwargs override per-state on top.
    4. Validate: every state with a hotkey must have a color, and vice versa.
    5. If everything is empty, raise :class:`LoupeConfigError`.

    Parameters
    ----------
    path
        Optional path to a ``.json`` file with ``"keymap"`` and
        ``"label_colors"`` keys.
    keymap
        Optional explicit keymap (forward or inverse form).
    label_colors
        Optional explicit ``state -> color`` mapping.
    package_default
        If True (default), fall back to ``state_definitions.json`` next to
        ``app.py`` when ``path`` is not provided.
    """
    raw_keymap: dict = {}
    raw_colors: dict = {}

    if path is not None:
        p = Path(path)
        if not p.exists():
            raise LoupeConfigError(f"state_definitions file not found: {p}")
        raw_keymap, raw_colors = _read_json_state_definitions(p)
    elif package_default:
        # Look for state_definitions.json next to app.py (this package's dir).
        # Imported lazily so test paths can be controlled by passing path=...
        from pathlib import Path as _P
        pkg_dir = _P(__file__).resolve().parent
        candidate = pkg_dir / "state_definitions.json"
        if candidate.exists():
            raw_keymap, raw_colors = _read_json_state_definitions(candidate)

    keys_for_state = normalize_keymap(raw_keymap)
    if keymap is not None:
        # Kwarg overrides: any hotkey present in the kwarg replaces the
        # file-side binding for that key. We work via the forward
        # ``{key: state}`` form to keep the per-key semantics explicit.
        forward = dict(StateConfig(keys_for_state=keys_for_state).key_to_state)
        kwarg_forward = dict(
            StateConfig(keys_for_state=normalize_keymap(keymap)).key_to_state
        )
        forward.update(kwarg_forward)
        keys_for_state = normalize_keymap(forward)

    if label_colors is not None:
        merged_colors = _coerce_label_colors(raw_colors)
        merged_colors.update(_coerce_label_colors(label_colors))
        colors_resolved = merged_colors
    else:
        colors_resolved = _coerce_label_colors(raw_colors)

    if not keys_for_state and not colors_resolved:
        raise LoupeConfigError(
            "No state definitions found. Pass `state_definitions=<path>`, "
            "`keymap=...`, `label_colors=...`, or place a "
            "`state_definitions.json` file next to `loupe/app.py`. "
            "See `example_state_definitions.json` for the schema."
        )

    # Validate: every state with a hotkey must have a color.
    missing_color = [s for s in keys_for_state if s not in colors_resolved]
    if missing_color:
        raise LoupeConfigError(
            "These states have hotkeys but no color: " + ", ".join(sorted(missing_color))
        )

    return StateConfig(
        keys_for_state=keys_for_state,
        label_colors=colors_resolved,
    )
