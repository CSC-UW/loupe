"""Versioned, data-free persistence for Loupe's runtime view state.

``TraceConfig`` / ``HeatmapConfig`` / ``RasterConfig`` describe data inputs
and how a window is initially constructed.  A :class:`ViewConfig` is a
different thing: it is a portable snapshot of the presentation state owned by
an already-running :class:`loupe.app.LoupeApp`.

This module deliberately has no Qt imports.  JSON parsing and validation can
therefore happen before Loupe constructs a ``QApplication`` or starts video
threads.  The adapter that captures/applies a config lives in
``loupe.view_config_runtime``.
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "LOUPE_VIEW_CONFIG_FORMAT",
    "LOUPE_VIEW_CONFIG_VERSION",
    "PlotRef",
    "ViewConfig",
    "ViewConfigApplyReport",
    "ViewConfigError",
    "coerce_view_config",
]


LOUPE_VIEW_CONFIG_FORMAT = "loupe-view-config"
LOUPE_VIEW_CONFIG_VERSION = 1

_TOP_LEVEL_KEYS = {
    "format",
    "schema_version",
    "metadata",
    "display",
    "plots",
    "sample_markers",
    "global_events",
    "videos",
    "session",
    "tuner",
}


class ViewConfigError(ValueError):
    """Raised when a View-Config cannot be parsed, validated, or applied."""

    def __init__(self, message: str, *, report: "ViewConfigApplyReport | None" = None):
        super().__init__(message)
        self.report = report


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ViewConfigError(
            f"{context} must be a JSON object, got {type(value).__name__}."
        )
    out = dict(value)
    bad_keys = [k for k in out if not isinstance(k, str)]
    if bad_keys:
        raise ViewConfigError(f"{context} contains non-string object keys.")
    return out


def _validate_json_value(value: Any, context: str = "View-Config") -> None:
    """Reject values JSON cannot represent portably, including NaN/Infinity."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ViewConfigError(f"{context} contains a non-finite number.")
        return
    if isinstance(value, list):
        for i, child in enumerate(value):
            _validate_json_value(child, f"{context}[{i}]")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ViewConfigError(f"{context} contains a non-string object key.")
            _validate_json_value(child, f"{context}.{key}")
        return
    raise ViewConfigError(
        f"{context} contains unsupported {type(value).__name__}; "
        "View-Configs may contain JSON values only."
    )


@dataclass(frozen=True)
class PlotRef:
    """Portable identity for one rendered subplot.

    ``source_id`` is user-authored (``Config.view_id``) when
    ``source_explicit`` is true.  Without one, the runtime matcher uses
    ``kind`` + ``name`` + duplicate ``occurrence`` and intentionally avoids
    silently replaying settings by raw list position.
    """

    kind: str
    name: str
    occurrence: int = 0
    source_id: str | None = None
    source_explicit: bool = False
    source_index: int | None = None
    local_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "name": self.name,
            "occurrence": self.occurrence,
        }
        if self.source_id is not None:
            out["source_id"] = self.source_id
        if self.source_explicit:
            out["source_explicit"] = True
        if self.source_index is not None:
            out["source_index"] = self.source_index
        if self.local_index is not None:
            out["local_index"] = self.local_index
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PlotRef":
        data = _require_mapping(raw, "plot.ref")
        kind = data.get("kind")
        name = data.get("name")
        occurrence = data.get("occurrence", 0)
        if kind not in {"ts", "dense", "raster", "heatmap"}:
            raise ViewConfigError(f"plot.ref.kind is invalid: {kind!r}.")
        if not isinstance(name, str):
            raise ViewConfigError("plot.ref.name must be a string.")
        if not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 0:
            raise ViewConfigError("plot.ref.occurrence must be a non-negative integer.")
        source_id = data.get("source_id")
        if source_id is not None and (
            not isinstance(source_id, str) or not source_id.strip()
        ):
            raise ViewConfigError("plot.ref.source_id must be a non-empty string.")
        source_explicit = data.get("source_explicit", False)
        if not isinstance(source_explicit, bool):
            raise ViewConfigError("plot.ref.source_explicit must be a boolean.")
        source_index = data.get("source_index")
        local_index = data.get("local_index")
        for label, value in (("source_index", source_index), ("local_index", local_index)):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ViewConfigError(
                    f"plot.ref.{label} must be a non-negative integer or null."
                )
        return cls(
            kind=kind,
            name=name,
            occurrence=occurrence,
            source_id=source_id,
            source_explicit=source_explicit,
            source_index=source_index,
            local_index=local_index,
        )

    @property
    def label(self) -> str:
        suffix = f"#{self.occurrence + 1}" if self.occurrence else ""
        return f"{self.kind}:{self.name or '(unnamed)'}{suffix}"


@dataclass
class ViewConfigApplyReport:
    """Structured result returned after applying a View-Config."""

    matched: list[str] = field(default_factory=list)
    fallback_matches: list[str] = field(default_factory=list)
    unmatched_saved: list[str] = field(default_factory=list)
    unmatched_current: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_exact(self) -> bool:
        return not (
            self.fallback_matches
            or self.unmatched_saved
            or self.unmatched_current
            or self.skipped
            or self.warnings
        )

    @property
    def applied_count(self) -> int:
        return len(self.matched) + len(self.fallback_matches)

    def summary(self) -> str:
        if self.is_exact:
            return f"Applied View-Config to {self.applied_count} plot(s)."
        parts = [f"applied {self.applied_count} plot(s)"]
        if self.fallback_matches:
            parts.append(f"{len(self.fallback_matches)} fallback match(es)")
        if self.unmatched_saved:
            parts.append(f"{len(self.unmatched_saved)} saved plot(s) unmatched")
        if self.unmatched_current:
            parts.append(f"{len(self.unmatched_current)} current plot(s) unchanged")
        if self.skipped:
            parts.append(f"{len(self.skipped)} setting(s) skipped")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        return "View-Config: " + ", ".join(parts) + "."

    def details(self) -> str:
        lines = [self.summary()]
        for title, values in (
            ("Fallback matches", self.fallback_matches),
            ("Unmatched saved plots", self.unmatched_saved),
            ("Current plots left unchanged", self.unmatched_current),
            ("Skipped settings", self.skipped),
            ("Warnings", self.warnings),
        ):
            if values:
                lines.append("")
                lines.append(f"{title}:")
                lines.extend(f"- {v}" for v in values)
        return "\n".join(lines)


@dataclass
class ViewConfig:
    """Validated representation of a ``.loupe-view.json`` file."""

    metadata: dict[str, Any] = field(default_factory=dict)
    display: dict[str, Any] = field(default_factory=dict)
    plots: list[dict[str, Any]] = field(default_factory=list)
    sample_markers: list[dict[str, Any]] = field(default_factory=list)
    global_events: list[dict[str, Any]] = field(default_factory=list)
    videos: list[dict[str, Any]] = field(default_factory=list)
    session: dict[str, Any] | None = None
    tuner: list[dict[str, Any]] | None = None
    schema_version: int = LOUPE_VIEW_CONFIG_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LOUPE_VIEW_CONFIG_VERSION:
            raise ViewConfigError(
                f"Unsupported View-Config schema_version={self.schema_version}; "
                f"this Loupe supports version {LOUPE_VIEW_CONFIG_VERSION}."
            )
        self.metadata = _require_mapping(self.metadata, "metadata")
        self.display = _require_mapping(self.display, "display")
        for label in ("plots", "sample_markers", "global_events", "videos"):
            value = getattr(self, label)
            if not isinstance(value, list):
                raise ViewConfigError(f"{label} must be a JSON array.")
            for i, item in enumerate(value):
                _require_mapping(item, f"{label}[{i}]")
        if self.session is not None:
            self.session = _require_mapping(self.session, "session")
        if self.tuner is not None:
            if not isinstance(self.tuner, list):
                raise ViewConfigError("tuner must be a JSON array or null.")
            for i, item in enumerate(self.tuner):
                _require_mapping(item, f"tuner[{i}]")

        for i, plot in enumerate(self.plots):
            if "ref" not in plot:
                raise ViewConfigError(f"plots[{i}] is missing ref.")
            PlotRef.from_dict(plot["ref"])
            order = plot.get("order", i)
            if not isinstance(order, int) or isinstance(order, bool) or order < 0:
                raise ViewConfigError(f"plots[{i}].order must be a non-negative integer.")
            if "visible" in plot and not isinstance(plot["visible"], bool):
                raise ViewConfigError(f"plots[{i}].visible must be a boolean.")
            if "height" in plot:
                height = plot["height"]
                if (
                    not isinstance(height, (int, float))
                    or isinstance(height, bool)
                    or not math.isfinite(float(height))
                    or not (0.01 <= float(height) <= 20.0)
                ):
                    raise ViewConfigError(
                        f"plots[{i}].height must be finite and in [0.01, 20.0]."
                    )

        _validate_json_value(self.to_dict())

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ViewConfig":
        data = _require_mapping(raw, "View-Config")
        unknown = set(data) - _TOP_LEVEL_KEYS
        if unknown:
            raise ViewConfigError(
                "Unknown top-level View-Config field(s): "
                + ", ".join(sorted(unknown))
            )
        fmt = data.get("format")
        if fmt != LOUPE_VIEW_CONFIG_FORMAT:
            raise ViewConfigError(
                f"Not a Loupe View-Config: format must be "
                f"{LOUPE_VIEW_CONFIG_FORMAT!r}, got {fmt!r}."
            )
        version = data.get("schema_version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise ViewConfigError("schema_version must be an integer.")
        if version != LOUPE_VIEW_CONFIG_VERSION:
            direction = "newer" if version > LOUPE_VIEW_CONFIG_VERSION else "older"
            raise ViewConfigError(
                f"This is a {direction} View-Config schema (version {version}); "
                f"this Loupe supports version {LOUPE_VIEW_CONFIG_VERSION}."
            )
        return cls(
            metadata=copy.deepcopy(data.get("metadata", {})),
            display=copy.deepcopy(data.get("display", {})),
            plots=copy.deepcopy(data.get("plots", [])),
            sample_markers=copy.deepcopy(data.get("sample_markers", [])),
            global_events=copy.deepcopy(data.get("global_events", [])),
            videos=copy.deepcopy(data.get("videos", [])),
            session=copy.deepcopy(data.get("session")),
            tuner=copy.deepcopy(data.get("tuner")),
            schema_version=version,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ViewConfig":
        p = Path(path).expanduser()
        try:
            with p.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError as exc:
            raise ViewConfigError(f"View-Config file not found: {p}") from exc
        except OSError as exc:
            raise ViewConfigError(f"Could not read View-Config {p}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ViewConfigError(f"Invalid JSON in View-Config {p}: {exc}") from exc
        try:
            return cls.from_dict(raw)
        except ViewConfigError as exc:
            raise ViewConfigError(f"Invalid View-Config {p}: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "format": LOUPE_VIEW_CONFIG_FORMAT,
            "schema_version": self.schema_version,
            "metadata": copy.deepcopy(self.metadata),
            "display": copy.deepcopy(self.display),
            "plots": copy.deepcopy(self.plots),
            "sample_markers": copy.deepcopy(self.sample_markers),
            "global_events": copy.deepcopy(self.global_events),
            "videos": copy.deepcopy(self.videos),
        }
        if self.session is not None:
            out["session"] = copy.deepcopy(self.session)
        if self.tuner is not None:
            out["tuner"] = copy.deepcopy(self.tuner)
        return out

    def save(self, path: str | Path) -> Path:
        p = Path(path).expanduser()
        if not p.name:
            raise ViewConfigError("View-Config path must name a file.")
        if p.suffix == "":
            p = p.with_name(p.name + ".loupe-view.json")
        parent = p.parent
        if not parent.exists():
            raise ViewConfigError(f"View-Config directory does not exist: {parent}")
        payload = self.to_dict()
        _validate_json_value(payload)
        tmp_path: Path | None = None
        try:
            fd, raw_tmp = tempfile.mkstemp(
                prefix=f".{p.name}.", suffix=".tmp", dir=str(parent)
            )
            tmp_path = Path(raw_tmp)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, p)
            tmp_path = None
        except OSError as exc:
            raise ViewConfigError(f"Could not save View-Config {p}: {exc}") from exc
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
        return p

    @staticmethod
    def new_metadata() -> dict[str, Any]:
        """Metadata common to runtime-captured files."""
        return {
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }


def coerce_view_config(
    value: str | Path | Mapping[str, Any] | ViewConfig,
) -> ViewConfig:
    """Normalize every public API input form to a validated ViewConfig."""
    if isinstance(value, ViewConfig):
        # Round-trip through dict so a caller cannot mutate nested structures
        # while a window is applying them.
        return ViewConfig.from_dict(value.to_dict())
    if isinstance(value, (str, Path)):
        return ViewConfig.load(value)
    if isinstance(value, Mapping):
        return ViewConfig.from_dict(value)
    raise TypeError(
        "view_config must be a path, mapping, or ViewConfig, got "
        f"{type(value).__name__}."
    )
