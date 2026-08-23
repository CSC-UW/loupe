"""Capture and apply :mod:`loupe.view_config` state on a live Loupe window.

The main application still owns rendering.  This adapter translates its
runtime registries into portable records and applies records through a small
number of batched refreshes.  Keeping this code outside ``app.py`` prevents
file-format and compatibility logic from spreading across GUI dialogs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from loupe._heatmap_utils import _colormap_display_name, _normalize_nan_shade
from loupe.tuner import BoolParam, ChoiceParam, IntParam
from loupe.view_config import (
    PlotRef,
    ViewConfig,
    ViewConfigApplyReport,
    ViewConfigError,
    coerce_view_config,
)


@dataclass(frozen=True)
class _Target:
    kind: str
    index: int
    name: str
    ref: PlotRef

    @property
    def key(self) -> tuple[str, int]:
        return (self.kind, self.index)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ViewConfigError("Cannot save a non-finite scalar value.")
        return value
    raise ViewConfigError(
        f"Cannot store {type(value).__name__} as a portable View-Config scalar."
    )


def _rgba(value: Any) -> list[int]:
    try:
        c = pg.mkColor(value)
    except Exception as exc:
        raise ViewConfigError(f"Cannot serialize color {value!r}.") from exc
    return [int(c.red()), int(c.green()), int(c.blue()), int(c.alpha())]


def _rgb(value: Any) -> list[int]:
    return _rgba(value)[:3]


def _color_tuple(value: Any, *, alpha: bool = True) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) not in (3, 4):
        raise ViewConfigError(f"Color must contain 3 or 4 channels, got {value!r}.")
    vals = tuple(int(v) for v in value)
    if any(v < 0 or v > 255 for v in vals):
        raise ViewConfigError(f"Color channels must be in 0..255, got {value!r}.")
    if alpha:
        return vals if len(vals) == 4 else (*vals, 255)
    return vals[:3]


def _serialize_colormap(cmap: Any) -> dict[str, Any]:
    if isinstance(cmap, str):
        return {"name": cmap}
    name = _colormap_display_name(cmap)
    try:
        import matplotlib as mpl

        if name in mpl.colormaps:
            return {"name": name}
    except Exception:
        pass
    try:
        rgba = np.asarray(cmap(np.linspace(0.0, 1.0, 256)), dtype=float)
        rgba = np.clip(np.rint(rgba * 255.0), 0, 255).astype(np.uint8)
        return {"name": name, "rgba_lut": rgba.tolist()}
    except Exception as exc:
        raise ViewConfigError(f"Cannot serialize heatmap colormap {cmap!r}.") from exc


def _deserialize_colormap(raw: Any) -> Any:
    if not isinstance(raw, Mapping):
        raise ViewConfigError("heatmap.colormap must be an object.")
    if "rgba_lut" not in raw:
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise ViewConfigError("heatmap.colormap.name must be a non-empty string.")
        return name
    lut = np.asarray(raw["rgba_lut"], dtype=float)
    if lut.ndim != 2 or lut.shape[1] not in (3, 4) or lut.shape[0] < 2:
        raise ViewConfigError("heatmap.colormap.rgba_lut must be an N x 3/4 array.")
    if not np.isfinite(lut).all() or np.any(lut < 0) or np.any(lut > 255):
        raise ViewConfigError("heatmap.colormap.rgba_lut channels must be in 0..255.")
    if lut.shape[1] == 3:
        lut = np.column_stack([lut, np.full(lut.shape[0], 255.0)])
    from matplotlib.colors import ListedColormap

    return ListedColormap(lut / 255.0, name=str(raw.get("name") or "loupe_custom"))


def _plot_name(app, kind: str, index: int) -> str:
    if kind == "ts":
        if app.overlay_mode:
            return str(app.overlay_groups[index].label)
        return str(app.series[index].name)
    if kind == "dense":
        return str(app.dense_groups[index].name)
    if kind == "raster":
        return str(app.raster_series[index].name)
    if kind == "heatmap":
        return str(app.heatmap_series[index].name)
    raise KeyError(kind)


def _counts(app) -> dict[str, int]:
    return {
        "ts": len(app.overlay_groups) if app.overlay_mode else len(app.series),
        "dense": len(app.dense_groups),
        "raster": len(app.raster_series),
        "heatmap": len(app.heatmap_series),
    }


def _inventory(app) -> list[_Target]:
    occurrences: dict[tuple[str, str], int] = {}
    targets: list[_Target] = []
    identities = getattr(app, "_view_plot_identities", {}) or {}
    for kind, count in _counts(app).items():
        meta_list = identities.get(kind, [])
        for index in range(count):
            name = _plot_name(app, kind, index)
            occurrence = occurrences.get((kind, name), 0)
            occurrences[(kind, name)] = occurrence + 1
            meta = meta_list[index] if index < len(meta_list) else {}
            targets.append(_Target(
                kind=kind,
                index=index,
                name=name,
                ref=PlotRef(
                    kind=kind,
                    name=name,
                    occurrence=occurrence,
                    source_id=meta.get("source_id"),
                    source_explicit=bool(meta.get("source_explicit", False)),
                    source_index=meta.get("source_index"),
                    local_index=meta.get("local_index"),
                ),
            ))
    return targets


def _default_order(app) -> list[tuple[str, int]]:
    counts = _counts(app)
    return (
        [("ts", i) for i in range(counts["ts"])]
        + [("dense", i) for i in range(counts["dense"])]
        + [("raster", i) for i in range(counts["raster"])]
        + [("heatmap", i) for i in range(counts["heatmap"])]
    )


def _full_order(app) -> list[tuple[str, int]]:
    valid = set(_default_order(app))
    configured = list(app.subplot_order or [])
    out = [tuple(x) for x in configured if tuple(x) in valid]
    out.extend(x for x in _default_order(app) if x not in out)
    return out


def _factor_and_visible(app, target: _Target) -> tuple[float, bool]:
    if target.kind == "ts":
        factors, visible = app.plot_height_factors, app.trace_visible
    elif target.kind == "dense":
        factors, visible = app.dense_height_factors, app.dense_visible
    elif target.kind == "raster":
        factors, visible = app.raster_height_factors, app.raster_visible
    else:
        factors, visible = app.heatmap_height_factors, app.heatmap_visible
    factor = float(factors[target.index]) if target.index < len(factors) else 1.0
    shown = bool(visible[target.index]) if target.index < len(visible) else True
    return factor, shown


def _plot_item(app, target: _Target):
    if target.kind == "ts":
        return app.plots[target.index]
    if target.kind == "dense":
        return app.dense_plots[target.index]
    if target.kind == "raster":
        return app.raster_plots[target.index]
    return app.heatmap_plots[target.index]


def _y_axis_state(app, target: _Target) -> dict[str, Any]:
    vb = _plot_item(app, target).getViewBox()
    lo, hi = vb.viewRange()[1]
    auto = bool(vb.autoRangeEnabled()[1])
    return {"auto": auto, "range": [float(lo), float(hi)]}


def _overlay_styles(app, index: int) -> list[dict[str, Any]]:
    if index >= len(app.overlay_series):
        return []
    occurrences: dict[str, int] = {}
    out = []
    for overlay in app.overlay_series[index]:
        occurrence = occurrences.get(str(overlay.name), 0)
        occurrences[str(overlay.name)] = occurrence + 1
        out.append({
            "name": str(overlay.name),
            "occurrence": occurrence,
            "color": _rgba(overlay.color),
            "width": float(getattr(overlay, "width", 1.0)),
            "symbol": getattr(overlay, "symbol", None),
            "symbol_size": float(getattr(overlay, "symbol_size", 8.0)),
        })
    return out


def _zip_styles(app, index: int) -> list[dict[str, Any]]:
    group = app.overlay_groups[index]
    identities = getattr(app, "_view_plot_identities", {}) or {}
    plot_meta = (
        identities.get("ts", [])[index]
        if index < len(identities.get("ts", []))
        else {}
    )
    source_ids = plot_meta.get("curve_source_ids", [])
    occurrences: dict[str, int] = {}
    out = []
    for trace in group.traces:
        name = str(trace.name)
        occurrence = occurrences.get(name, 0)
        occurrences[name] = occurrence + 1
        color = (
            app.overlay_colors[trace.source_idx]
            if trace.source_idx < len(app.overlay_colors)
            else (255, 255, 255, 255)
        )
        record = {
            "name": name,
            "occurrence": occurrence,
            "source_index": int(trace.source_idx),
            "color": _rgba(color),
            "width": 1.0,
        }
        if trace.source_idx < len(source_ids) and source_ids[trace.source_idx] is not None:
            record["view_id"] = source_ids[trace.source_idx]
        out.append(record)
    return out


def _plot_state(app, target: _Target) -> dict[str, Any]:
    state: dict[str, Any] = {"y_axis": _y_axis_state(app, target)}
    i = target.index
    if target.kind == "ts":
        if app.overlay_mode:
            state["zip_curves"] = _zip_styles(app, i)
        else:
            color = (
                app.series_colors[i]
                if i < len(getattr(app, "series_colors", []))
                else (255, 255, 255, 255)
            )
            width = (
                app.series_line_widths[i]
                if i < len(getattr(app, "series_line_widths", []))
                else 1.0
            )
            state["trace_style"] = {
                "color": _rgba(color),
                "width": float(width),
            }
            state["overlays"] = _overlay_styles(app, i)
    elif target.kind == "dense":
        group = app.dense_groups[i]
        palette = app._dense_category_map(i)
        state["dense"] = {
            "gain": float(group.gain),
            "step": int(group.step),
            "traces_per_page": (
                int(group.traces_per_page)
                if group.traces_per_page is not None
                else None
            ),
            "palette": (
                {str(k): _rgba(v) for k, v in palette.items()}
                if palette is not None
                else None
            ),
        }
    elif target.kind == "heatmap":
        heat = app.heatmap_series[i]
        state["heatmap"] = {
            "vmin": float(heat.vmin),
            "vmax": float(heat.vmax),
            "colormap": _serialize_colormap(heat.colormap),
            "decim_method": str(heat.decim_method),
            "shade_nans": (
                False
                if heat.shade_nans is None
                else [
                    f"#{heat.shade_nans[0]:02X}{heat.shade_nans[1]:02X}"
                    f"{heat.shade_nans[2]:02X}",
                    float(heat.shade_nans[3]),
                ]
            ),
        }
    else:
        raster = app.raster_series[i]
        state["raster_style"] = {
            "color": _rgb(raster.color),
            "category_colors": (
                [_rgb(x) for x in raster.category_colors]
                if raster.category_colors is not None
                else None
            ),
            "separator_color": (
                _rgba(raster.separator_color)
                if raster.separator_color is not None
                else None
            ),
            "separator_width": (
                float(raster.separator_width)
                if raster.separator_width is not None
                else None
            ),
        }
    return state


def _is_widget_explicitly_visible(widget) -> bool:
    return widget is not None and not widget.isHidden()


def _splitter_ratio(app) -> float:
    sizes = list(app.splitter.sizes()) if app.splitter is not None else []
    total = sum(max(0, int(x)) for x in sizes)
    if len(sizes) >= 2 and total > 0:
        return float(max(0, sizes[0]) / total)
    return 0.6


def _marker_ref(marker: Any, occurrence: int) -> dict[str, Any]:
    out = {"symbol": str(marker.marker), "occurrence": int(occurrence)}
    if getattr(marker, "view_id", None) is not None:
        out["view_id"] = str(marker.view_id)
    return out


def _capture_markers(app, targets: list[_Target]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    occurrences: dict[str, int] = {}
    for marker in app.sample_markers:
        symbol = str(marker.marker)
        occurrence = occurrences.get(symbol, 0)
        occurrences[symbol] = occurrence + 1
        out.append({
            "scope": "stacked",
            "marker": _marker_ref(marker, occurrence),
            "style": {
                "color": _rgba(marker.color),
                "size": float(marker.size),
                "alpha": int(marker.alpha),
            },
        })
    dense_targets = {t.index: t for t in targets if t.kind == "dense"}
    for gi, group in enumerate(app.dense_groups):
        parent = dense_targets.get(gi)
        if parent is None:
            continue
        occurrences = {}
        for marker in group.sample_markers:
            symbol = str(marker.marker)
            occurrence = occurrences.get(symbol, 0)
            occurrences[symbol] = occurrence + 1
            out.append({
                "scope": "dense",
                "parent_ref": parent.ref.to_dict(),
                "marker": _marker_ref(marker, occurrence),
                "style": {
                    "color": _rgba(marker.color),
                    "size": float(marker.size),
                    "alpha": int(marker.alpha),
                },
            })
    return out


def _scalar_identity(value: Any) -> dict[str, Any]:
    if isinstance(value, np.generic):
        value = value.item()
    try:
        portable = _json_scalar(value)
        return {"type": type(value).__name__, "value": portable}
    except ViewConfigError:
        return {"type": type(value).__name__, "repr": repr(value)}


def _scalar_token(record: Mapping[str, Any]) -> tuple[str, str]:
    if "value" in record:
        return (str(record.get("type")), repr(record.get("value")))
    return (str(record.get("type")), str(record.get("repr")))


def _capture_global_events(app) -> list[dict[str, Any]]:
    out = []
    for class_value, style in app._resolved_event_styles.items():
        out.append({
            "class": _scalar_identity(class_value),
            "style": {
                "line_color": _rgb(style["line_color"]),
                "line_style": str(style["line_style"]),
                "line_width": float(style["line_width"]),
                "line_alpha": int(style["line_alpha"]),
            },
        })
    return out


def _capture_videos(app) -> list[dict[str, Any]]:
    occurrences: dict[str, int] = {}
    out = []
    for slot in app.video_slots:
        occurrence = occurrences.get(slot.name, 0)
        occurrences[slot.name] = occurrence + 1
        ref: dict[str, Any] = {"name": slot.name, "occurrence": occurrence}
        if slot.view_id is not None:
            ref["view_id"] = slot.view_id
        out.append({
            "ref": ref,
            "visible": bool(slot.desired_visible),
            "stretch": int(slot.stretch),
            "frame_step_target": slot.index == app.frame_step_source,
        })
    return out


def _capture_session(app) -> dict[str, Any]:
    geometry = app.geometry()
    window_state = (
        "fullscreen" if app.isFullScreen()
        else "maximized" if app.isMaximized()
        else "normal"
    )
    scrollbar = app.plot_scroll_area.verticalScrollBar()
    return {
        "window_start": float(app.window_start),
        "cursor_time": float(app.cursor_time),
        "plot_scroll_value": int(scrollbar.value()),
        "window_geometry": {
            "x": int(geometry.x()),
            "y": int(geometry.y()),
            "width": int(geometry.width()),
            "height": int(geometry.height()),
            "state": window_state,
        },
    }


def _capture_tuner(app) -> list[dict[str, Any]]:
    counts: dict[tuple[str | None, str], int] = {}
    out = []
    for param in app._tuner_params:
        key = (param.name, type(param).__name__)
        occurrence = counts.get(key, 0)
        counts[key] = occurrence + 1
        out.append({
            "name": param.name,
            "type": type(param).__name__,
            "occurrence": occurrence,
            "value": _json_scalar(param.value),
        })
    return out


def capture_view_config(
    app,
    *,
    include_session: bool = False,
    include_tuner: bool = False,
) -> ViewConfig:
    targets = _inventory(app)
    order_positions = {entry: i for i, entry in enumerate(_full_order(app))}
    plots = []
    for target in targets:
        factor, visible = _factor_and_visible(app, target)
        plots.append({
            "ref": target.ref.to_dict(),
            "order": int(order_positions.get(target.key, len(order_positions))),
            "visible": visible,
            "height": factor,
            "state": _plot_state(app, target),
        })

    metadata = ViewConfig.new_metadata()
    try:
        metadata["loupe_version"] = importlib_metadata.version("loupe")
    except importlib_metadata.PackageNotFoundError:
        metadata["loupe_version"] = "unknown"

    display = {
        "window_len": float(app.window_len),
        "smooth_scroll_fraction": float(app.smooth_scroll_fraction),
        "playback_speed": float(app.playback_speed),
        "splitter_ratio": _splitter_ratio(app),
        "scale_raster_proportionally": bool(app.scale_raster_proportionally),
        "raster_brightness": float(app.raster_brightness),
        "raster_event_height": float(app.raster_event_height),
        "raster_event_thickness": int(app.raster_event_thickness),
        "scale_heatmap_proportionally": bool(app.scale_heatmap_proportionally),
        "compact_heatmaps_to_fit": bool(app.compact_heatmaps_to_fit),
        "compact_rasters_to_fit": bool(getattr(app, "compact_rasters_to_fit", False)),
        "interval_labels": {
            "alpha": float(app.interval_label_alpha_multiplier),
            "overlays_visible": bool(app.interval_label_overlays_enabled),
            "strip_visible": bool(app.label_strip_visible),
            "hypnogram_visible": _is_widget_explicitly_visible(app.hypnogram_widget),
            "hypnogram_zoomed": bool(app.hypnogram_zoomed),
        },
        "tuner_visible": bool(
            app._tuner_dock is not None and not app._tuner_dock.isHidden()
        ),
    }
    return ViewConfig(
        metadata=metadata,
        display=display,
        plots=plots,
        sample_markers=_capture_markers(app, targets),
        global_events=_capture_global_events(app),
        videos=_capture_videos(app),
        session=_capture_session(app) if include_session else None,
        tuner=_capture_tuner(app) if include_tuner else None,
    )


def save_view_config(
    app,
    path: str | Path,
    *,
    include_session: bool = False,
    include_tuner: bool = False,
) -> Path:
    return capture_view_config(
        app,
        include_session=include_session,
        include_tuner=include_tuner,
    ).save(path)


def _match_plots(
    saved_plots: list[dict[str, Any]],
    current: list[_Target],
    report: ViewConfigApplyReport,
) -> list[tuple[dict[str, Any], _Target]]:
    consumed: set[tuple[str, int]] = set()
    matches: list[tuple[dict[str, Any], _Target]] = []
    for record in sorted(saved_plots, key=lambda x: int(x.get("order", 0))):
        saved = PlotRef.from_dict(record["ref"])
        available = [t for t in current if t.key not in consumed and t.kind == saved.kind]
        target = None
        fallback = False
        if saved.source_explicit and saved.source_id is not None:
            source_matches = [t for t in available if t.ref.source_id == saved.source_id]
            named = [
                t for t in source_matches
                if t.name == saved.name and t.ref.occurrence == saved.occurrence
            ]
            if len(named) == 1:
                target = named[0]
            elif saved.local_index is not None:
                local = [t for t in source_matches if t.ref.local_index == saved.local_index]
                if len(local) == 1:
                    target = local[0]
                    fallback = True
        else:
            named = [
                t for t in available
                if t.name == saved.name and t.ref.occurrence == saved.occurrence
            ]
            if len(named) == 1:
                target = named[0]

        if target is None:
            report.unmatched_saved.append(saved.label)
            continue
        consumed.add(target.key)
        matches.append((record, target))
        if fallback:
            report.fallback_matches.append(
                f"{saved.label} -> {target.ref.label} by explicit source/local index"
            )
        else:
            report.matched.append(f"{saved.label} -> {target.ref.label}")

    for target in current:
        if target.key not in consumed:
            report.unmatched_current.append(target.ref.label)
    return matches


def _set_action_checked(action, checked: bool) -> None:
    if action is None:
        return
    action.blockSignals(True)
    action.setChecked(bool(checked))
    action.blockSignals(False)


def _number(
    raw: Mapping[str, Any],
    key: str,
    report: ViewConfigApplyReport,
    *,
    lo: float | None = None,
    hi: float | None = None,
    integer: bool = False,
) -> float | int | None:
    if key not in raw:
        return None
    value = raw[key]
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        report.skipped.append(f"{key}: expected a finite number")
        return None
    out: float | int = int(value) if integer else float(value)
    if lo is not None and out < lo or hi is not None and out > hi:
        report.skipped.append(f"{key}: {out} is outside the supported range")
        return None
    return out


def _bool(raw: Mapping[str, Any], key: str, report: ViewConfigApplyReport) -> bool | None:
    if key not in raw:
        return None
    value = raw[key]
    if not isinstance(value, bool):
        report.skipped.append(f"{key}: expected a boolean")
        return None
    return value


def _close_state_dialogs(app) -> None:
    for attr in ("y_axis_dialog", "_dense_ctrl_dialog", "_heatmap_ctrl_dialog"):
        dialog = getattr(app, attr, None)
        if dialog is not None:
            dialog.close()
            dialog.deleteLater()
            setattr(app, attr, None)


def _apply_tuner(app, records: list[dict[str, Any]], report: ViewConfigApplyReport) -> None:
    counts: dict[tuple[str | None, str], int] = {}
    current: dict[tuple[str | None, str, int], Any] = {}
    for param in app._tuner_params:
        base = (param.name, type(param).__name__)
        occurrence = counts.get(base, 0)
        counts[base] = occurrence + 1
        current[(param.name, type(param).__name__, occurrence)] = param

    changed = []
    for record in records:
        key = (record.get("name"), str(record.get("type")), int(record.get("occurrence", 0)))
        param = current.get(key)
        if param is None:
            report.skipped.append(f"Tuner parameter {key!r} is not present")
            continue
        value = record.get("value")
        try:
            if isinstance(param, ChoiceParam) and value not in param.choices:
                raise ValueError(f"not one of {param.choices!r}")
            if isinstance(param, BoolParam) and not isinstance(value, bool):
                raise ValueError("expected a boolean")
            if isinstance(param, IntParam) and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise ValueError("expected an integer")
            if param.min is not None and value < param.min:
                raise ValueError(f"below minimum {param.min}")
            if param.max is not None and value > param.max:
                raise ValueError(f"above maximum {param.max}")
            param.value = value
        except Exception as exc:
            report.skipped.append(f"Tuner parameter {key!r}: {exc}")
            continue
        changed.append(param)

    dock = getattr(app, "_tuner_dock", None)
    if dock is not None and hasattr(dock, "sync_from_params"):
        dock.sync_from_params()
    for param in changed:
        app._pending_dirty_param_ids.add(id(param))
    if changed:
        if app._tuner_refresh_timer.isActive():
            app._tuner_refresh_timer.stop()
        app._flush_tuner()


def _set_factor_and_visible(app, target: _Target, factor: float, visible: bool) -> None:
    if target.kind == "ts":
        factors, flags = app.plot_height_factors, app.trace_visible
    elif target.kind == "dense":
        factors, flags = app.dense_height_factors, app.dense_visible
    elif target.kind == "raster":
        factors, flags = app.raster_height_factors, app.raster_visible
    else:
        factors, flags = app.heatmap_height_factors, app.heatmap_visible
    while len(factors) <= target.index:
        factors.append(1.0)
    while len(flags) <= target.index:
        flags.append(True)
    factors[target.index] = max(0.01, min(20.0, float(factor)))
    flags[target.index] = bool(visible)


def _match_named_records(saved: list[dict[str, Any]], current: list[Any], name_getter) -> list[tuple[dict, int]]:
    occurrences: dict[str, int] = {}
    index: dict[tuple[str, int], int] = {}
    for i, item in enumerate(current):
        name = str(name_getter(item))
        occurrence = occurrences.get(name, 0)
        occurrences[name] = occurrence + 1
        index[(name, occurrence)] = i
    out = []
    for record in saved:
        key = (str(record.get("name", "")), int(record.get("occurrence", 0)))
        if key in index:
            out.append((record, index[key]))
    return out


def _apply_ts_state(app, target: _Target, state: Mapping[str, Any], report) -> None:
    i = target.index
    if app.overlay_mode:
        records = state.get("zip_curves", [])
        if not isinstance(records, list):
            report.skipped.append(f"{target.ref.label}: zip_curves is not a list")
            return
        group = app.overlay_groups[i]
        identities = getattr(app, "_view_plot_identities", {}) or {}
        plot_meta = (
            identities.get("ts", [])[i]
            if i < len(identities.get("ts", []))
            else {}
        )
        source_ids = plot_meta.get("curve_source_ids", [])
        named_matches = _match_named_records(records, group.traces, lambda x: x.name)
        named_by_record = {id(record): trace_i for record, trace_i in named_matches}
        used: set[int] = set()
        matched_records: list[tuple[dict[str, Any], int]] = []
        for record in records:
            trace_i = None
            view_id = record.get("view_id")
            if view_id is not None:
                candidates = [
                    ti
                    for ti, trace in enumerate(group.traces)
                    if ti not in used
                    and trace.source_idx < len(source_ids)
                    and source_ids[trace.source_idx] == view_id
                ]
                if len(candidates) == 1:
                    trace_i = candidates[0]
            elif id(record) in named_by_record:
                candidate = named_by_record[id(record)]
                if candidate not in used:
                    trace_i = candidate
            if trace_i is not None:
                used.add(trace_i)
                matched_records.append((record, trace_i))
            else:
                identity = record.get("view_id") or record.get("name") or "(unnamed)"
                report.skipped.append(
                    f"{target.ref.label}: Zip curve not present or ambiguous: {identity!r}"
                )
        for record, trace_i in matched_records:
            trace = group.traces[trace_i]
            try:
                color = _color_tuple(record["color"])
                width = float(record.get("width", 1.0))
                if not math.isfinite(width) or width <= 0:
                    raise ValueError("width must be positive and finite")
                app.overlay_colors[trace.source_idx] = color
                if i < len(app._plot_to_curves) and trace_i < len(app._plot_to_curves[i]):
                    app._plot_to_curves[i][trace_i].setPen(pg.mkPen(color, width=width))
            except Exception as exc:
                report.skipped.append(f"{target.ref.label} Zip style: {exc}")
        return

    style = state.get("trace_style")
    if isinstance(style, Mapping):
        try:
            color = _color_tuple(style["color"])
            width = float(style.get("width", 1.0))
            if not math.isfinite(width) or width <= 0:
                raise ValueError("width must be positive and finite")
            while len(app.series_colors) <= i:
                app.series_colors.append((255, 255, 255, 255))
            while len(app.series_line_widths) <= i:
                app.series_line_widths.append(1.0)
            app.series_colors[i] = color
            app.series_line_widths[i] = width
            app.curves[i].setPen(pg.mkPen(color, width=width))
        except Exception as exc:
            report.skipped.append(f"{target.ref.label} trace style: {exc}")

    records = state.get("overlays", [])
    if not isinstance(records, list) or i >= len(app.overlay_series):
        return
    overlays = app.overlay_series[i]
    for record, oi in _match_named_records(records, overlays, lambda x: x.name):
        overlay = overlays[oi]
        try:
            color = _color_tuple(record["color"])
            width = float(record.get("width", 1.0))
            symbol = record.get("symbol")
            symbol_size = float(record.get("symbol_size", 8.0))
            if not math.isfinite(width) or width <= 0:
                raise ValueError("width must be positive and finite")
            if not math.isfinite(symbol_size) or symbol_size <= 0:
                raise ValueError("symbol_size must be positive and finite")
            overlay.color = color
            overlay.width = width
            overlay.symbol = symbol
            overlay.symbol_size = symbol_size
            if i < len(app.overlay_curve_items) and oi < len(app.overlay_curve_items[i]):
                item = app.overlay_curve_items[i][oi]
                if symbol:
                    item.setPen(None)
                    item.setSymbol(str(symbol))
                    item.setSymbolBrush(pg.mkBrush(color))
                    item.setSymbolPen(None)
                    item.setSymbolSize(symbol_size)
                else:
                    item.setSymbol(None)
                    item.setPen(pg.mkPen(color, width=width))
        except Exception as exc:
            report.skipped.append(f"{target.ref.label} overlay style: {exc}")


def _apply_dense_state(app, target: _Target, state: Mapping[str, Any], report) -> None:
    raw = state.get("dense")
    if not isinstance(raw, Mapping):
        return
    group = app.dense_groups[target.index]
    rebuild = False
    gain = _number(raw, "gain", report, lo=0.001)
    if gain is not None:
        group.gain = float(gain)
    step = _number(raw, "step", report, lo=1, integer=True)
    if step is not None:
        step = min(int(step), max(1, len(group.series)))
        rebuild = rebuild or step != group.step
        group.step = step
    if "traces_per_page" in raw:
        value = raw["traces_per_page"]
        if value is None:
            group.traces_per_page = None
        elif isinstance(value, int) and not isinstance(value, bool) and value > 0:
            group.traces_per_page = min(value, max(1, len(group.series)))
        else:
            report.skipped.append(f"{target.ref.label}: invalid traces_per_page")
    palette = raw.get("palette")
    if palette is not None:
        if isinstance(palette, Mapping):
            try:
                group.palette = {str(k): _color_tuple(v) for k, v in palette.items()}
                rebuild = True
            except Exception as exc:
                report.skipped.append(f"{target.ref.label} dense palette: {exc}")
        else:
            report.skipped.append(f"{target.ref.label}: dense palette is not an object")
    if rebuild and target.index < len(app.dense_plots):
        app._rebuild_dense_curves(target.index)


def _apply_heatmap_state(app, target: _Target, state: Mapping[str, Any], report) -> None:
    raw = state.get("heatmap")
    if not isinstance(raw, Mapping):
        return
    heat = app.heatmap_series[target.index]
    vmin = _number(raw, "vmin", report)
    vmax = _number(raw, "vmax", report)
    next_min = float(vmin) if vmin is not None else heat.vmin
    next_max = float(vmax) if vmax is not None else heat.vmax
    if next_max <= next_min:
        report.skipped.append(f"{target.ref.label}: heatmap vmax must exceed vmin")
    else:
        heat.vmin, heat.vmax = next_min, next_max
    if "colormap" in raw:
        try:
            heat.colormap = _deserialize_colormap(raw["colormap"])
        except Exception as exc:
            report.skipped.append(f"{target.ref.label} colormap: {exc}")
    method = raw.get("decim_method")
    if method in {"peak", "mean"} and method != heat.decim_method:
        heat.decim_method = method
        if heat.mipmap_levels is not None:
            from loupe.xr_loader import _build_mipmap
            from loupe._heatmap_utils import ARRAY_MIPMAP_TARGET_MIN_COLS

            heat.mipmap_levels = _build_mipmap(
                heat.Y, method, ARRAY_MIPMAP_TARGET_MIN_COLS
            )
    elif method is not None and method not in {"peak", "mean"}:
        report.skipped.append(f"{target.ref.label}: invalid heatmap decim_method")
    if "shade_nans" in raw:
        shade_value = raw["shade_nans"]
        if isinstance(shade_value, list):
            shade_value = tuple(shade_value)
        try:
            heat.shade_nans = _normalize_nan_shade(shade_value)
        except Exception as exc:
            report.skipped.append(f"{target.ref.label} shade_nans: {exc}")
    if target.index < len(app._heatmap_cache_keys):
        app._heatmap_cache_keys[target.index] = None


def _apply_raster_state(app, target: _Target, state: Mapping[str, Any], report) -> None:
    raw = state.get("raster_style")
    if not isinstance(raw, Mapping):
        return
    raster = app.raster_series[target.index]
    try:
        color = raster.color
        category_colors = raster.category_colors
        separator_color = raster.separator_color
        separator_width = raster.separator_width
        if "color" in raw:
            color = _color_tuple(raw["color"], alpha=False)
        if raw.get("category_colors") is not None:
            category_colors = [
                _color_tuple(v, alpha=False) for v in raw["category_colors"]
            ]
            if (
                raster.category_colors is not None
                and len(category_colors) != len(raster.category_colors)
            ):
                raise ValueError("category color count does not match current data")
        if raw.get("separator_color") is not None:
            separator_color = _color_tuple(raw["separator_color"])
        if raw.get("separator_width") is not None:
            separator_width = float(raw["separator_width"])
            if not math.isfinite(separator_width) or separator_width <= 0:
                raise ValueError("separator_width must be positive and finite")
        raster.color = color
        raster.category_colors = category_colors
        raster.separator_color = separator_color
        raster.separator_width = separator_width
        if target.index < len(app.raster_separator_lines):
            pen_color = separator_color or (120, 120, 120)
            pen_width = separator_width if separator_width is not None else 1.0
            for line in app.raster_separator_lines[target.index]:
                line.setPen(pg.mkPen(pen_color, width=pen_width))
    except Exception as exc:
        report.skipped.append(f"{target.ref.label} raster style: {exc}")


def _apply_y_axis(app, target: _Target, raw: Any, report) -> None:
    if not isinstance(raw, Mapping):
        return
    auto = raw.get("auto", False)
    yrange = raw.get("range")
    if not isinstance(auto, bool):
        report.skipped.append(f"{target.ref.label}: y_axis.auto is not boolean")
        return
    plot = _plot_item(app, target)
    if auto:
        plot.enableAutoRange("y", True)
        return
    if (
        not isinstance(yrange, list)
        or len(yrange) != 2
        or not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in yrange)
        or float(yrange[1]) <= float(yrange[0])
    ):
        report.skipped.append(f"{target.ref.label}: invalid y_axis.range")
        return
    plot.enableAutoRange("y", False)
    plot.setYRange(float(yrange[0]), float(yrange[1]), padding=0)


def _apply_markers(app, records, current_targets, report) -> None:
    def find_marker(markers, marker_ref):
        view_id = marker_ref.get("view_id")
        if view_id is not None:
            found = [m for m in markers if getattr(m, "view_id", None) == view_id]
            return found[0] if len(found) == 1 else None
        symbol = str(marker_ref.get("symbol", ""))
        occurrence = int(marker_ref.get("occurrence", 0))
        found = [m for m in markers if str(m.marker) == symbol]
        return found[occurrence] if occurrence < len(found) else None

    for record in records:
        scope = record.get("scope")
        marker_ref = record.get("marker")
        style = record.get("style")
        if not isinstance(marker_ref, Mapping) or not isinstance(style, Mapping):
            report.skipped.append("Malformed sample-marker record")
            continue
        marker = None
        apply_callback = None
        if scope == "stacked":
            marker = find_marker(app.sample_markers, marker_ref)
            if marker is not None:
                mi = app.sample_markers.index(marker)
                apply_callback = lambda i=mi: app._apply_sample_marker_style(i)
        elif scope == "dense" and isinstance(record.get("parent_ref"), Mapping):
            parent = PlotRef.from_dict(record["parent_ref"])
            parent_matches = _match_plots(
                [{"ref": parent.to_dict(), "order": 0}], current_targets,
                ViewConfigApplyReport(),
            )
            if parent_matches:
                target = parent_matches[0][1]
                group = app.dense_groups[target.index]
                marker = find_marker(group.sample_markers, marker_ref)
                if marker is not None:
                    mi = group.sample_markers.index(marker)
                    apply_callback = lambda gi=target.index, i=mi: app._apply_dense_marker_style(gi, i)
        if marker is None:
            report.skipped.append(f"Sample marker not present: {dict(marker_ref)!r}")
            continue
        try:
            color = _color_tuple(style["color"])
            size = float(style["size"])
            alpha = int(style["alpha"])
            if not math.isfinite(size) or not (2.0 <= size <= 40.0):
                raise ValueError("size outside supported range")
            if not (0 <= alpha <= 255):
                raise ValueError("size/alpha outside supported range")
            marker.color = color
            marker.size = size
            marker.alpha = alpha
            apply_callback()
        except Exception as exc:
            report.skipped.append(f"Sample marker {dict(marker_ref)!r}: {exc}")


def _apply_global_events(app, records, report) -> None:
    current = {
        _scalar_token(_scalar_identity(value)): value
        for value in app._resolved_event_styles
    }
    for record in records:
        class_record = record.get("class")
        style = record.get("style")
        if not isinstance(class_record, Mapping) or not isinstance(style, Mapping):
            report.skipped.append("Malformed global-event style record")
            continue
        class_value = current.get(_scalar_token(class_record))
        # None is a valid class key, so membership must be checked separately.
        token = _scalar_token(class_record)
        if token not in current:
            report.skipped.append(f"Global-event class not present: {dict(class_record)!r}")
            continue
        try:
            line_style = str(style["line_style"])
            if line_style not in {"solid", "dashed", "dotted", "dashdot", "dashdotdot"}:
                raise ValueError("invalid line style")
            line_color = _color_tuple(style["line_color"], alpha=False)
            line_width = float(style["line_width"])
            line_alpha = int(style["line_alpha"])
            if not math.isfinite(line_width) or not (0.5 <= line_width <= 8.0):
                raise ValueError("line width outside 0.5..8.0")
            if not (0 <= line_alpha <= 255):
                raise ValueError("line alpha outside 0..255")
            resolved = app._resolved_event_styles[class_value]
            resolved.update({
                "line_color": line_color,
                "line_style": line_style,
                "line_width": line_width,
                "line_alpha": line_alpha,
            })
            app._apply_global_event_class_style(class_value)
        except Exception as exc:
            report.skipped.append(f"Global-event class {dict(class_record)!r}: {exc}")


def _video_ref_matches(saved: Mapping[str, Any], slot, occurrence: int) -> bool:
    view_id = saved.get("view_id")
    if view_id is not None:
        return slot.view_id == view_id
    return saved.get("name") == slot.name and int(saved.get("occurrence", 0)) == occurrence


def _apply_videos(app, records, report) -> None:
    occurrences: dict[str, int] = {}
    current = []
    for slot in app.video_slots:
        occurrence = occurrences.get(slot.name, 0)
        occurrences[slot.name] = occurrence + 1
        current.append((slot, occurrence))
    used: set[int] = set()
    target_index = None
    for record in records:
        ref = record.get("ref")
        if not isinstance(ref, Mapping):
            report.skipped.append("Video record missing ref")
            continue
        matches = [
            (slot, occurrence) for slot, occurrence in current
            if slot.index not in used and _video_ref_matches(ref, slot, occurrence)
        ]
        if len(matches) != 1:
            report.skipped.append(f"Video not present or ambiguous: {dict(ref)!r}")
            continue
        slot, _ = matches[0]
        used.add(slot.index)
        visible = record.get("visible", slot.desired_visible)
        stretch = record.get("stretch", slot.stretch)
        if not isinstance(visible, bool) or not isinstance(stretch, int) or isinstance(stretch, bool):
            report.skipped.append(f"Video {slot.name}: invalid visibility/stretch")
            continue
        slot.stretch = max(0, min(20, int(stretch)))
        if slot.show_action is not None:
            _set_action_checked(slot.show_action, visible)
        app._set_video_visible(slot.index, visible)
        if record.get("frame_step_target") is True:
            target_index = slot.index
    app._apply_video_stretches()
    if target_index is not None:
        app.frame_step_source = target_index
        for slot in app.video_slots:
            if slot.step_action is not None:
                _set_action_checked(slot.step_action, slot.index == target_index)


def _apply_session(app, session: Mapping[str, Any], report) -> None:
    window_start = _number(session, "window_start", report)
    cursor_time = _number(session, "cursor_time", report)
    if window_start is not None:
        app.window_start = max(
            app.t_global_min,
            min(float(window_start), max(app.t_global_min, app.t_global_max - app.window_len)),
        )
    if cursor_time is not None:
        app.cursor_time = max(
            app.window_start,
            min(float(cursor_time), app.window_start + app.window_len),
        )
    app._apply_x_range()
    app._update_nav_slider_from_window()
    scroll_value = session.get("plot_scroll_value")
    if isinstance(scroll_value, int) and not isinstance(scroll_value, bool):
        bar = app.plot_scroll_area.verticalScrollBar()
        QtCore.QTimer.singleShot(0, lambda v=scroll_value, b=bar: b.setValue(v))

    geometry = session.get("window_geometry")
    if isinstance(geometry, Mapping):
        try:
            width = max(400, int(geometry["width"]))
            height = max(300, int(geometry["height"]))
            x, y = int(geometry["x"]), int(geometry["y"])
            screen = app.screen() or QtWidgets.QApplication.primaryScreen()
            if screen is not None:
                area = screen.availableGeometry()
                width = min(width, area.width())
                height = min(height, area.height())
                x = max(area.left(), min(x, area.right() - width + 1))
                y = max(area.top(), min(y, area.bottom() - height + 1))
            app.setGeometry(x, y, width, height)
            state = geometry.get("state", "normal")
            if state == "fullscreen":
                app.showFullScreen()
                _set_action_checked(getattr(app, "action_fullscreen", None), True)
            elif state == "maximized":
                _set_action_checked(getattr(app, "action_fullscreen", None), False)
                app.showMaximized()
            elif state == "normal":
                _set_action_checked(getattr(app, "action_fullscreen", None), False)
                app.showNormal()
            else:
                report.skipped.append(f"Unknown window state {state!r}")
        except Exception as exc:
            report.skipped.append(f"Window geometry: {exc}")


def apply_view_config(
    app,
    config_or_path,
    *,
    strict: bool = False,
) -> ViewConfigApplyReport:
    config = coerce_view_config(config_or_path)
    report = ViewConfigApplyReport()
    current = _inventory(app)
    matches = _match_plots(config.plots, current, report)
    if strict and (report.unmatched_saved or report.unmatched_current):
        raise ViewConfigError(
            "Strict View-Config compatibility check failed.\n" + report.details(),
            report=report,
        )

    app._stop_playback_if_playing()
    _close_state_dialogs(app)
    if config.tuner is not None:
        _apply_tuner(app, config.tuner, report)

    display = config.display
    window_len = _number(display, "window_len", report, lo=0.1, hi=3600.0)
    if window_len is not None:
        app.window_len = float(window_len)
        app.window_spin.blockSignals(True)
        app.window_spin.setValue(app.window_len)
        app.window_spin.blockSignals(False)
    smooth = _number(display, "smooth_scroll_fraction", report, lo=0.001, hi=1.0)
    if smooth is not None:
        app.smooth_scroll_fraction = float(smooth)
    speed = _number(display, "playback_speed", report, lo=0.25, hi=4.0)
    if speed is not None:
        app.playback_speed = float(speed)
    raster_brightness = _number(display, "raster_brightness", report, lo=0.2, hi=3.0)
    if raster_brightness is not None:
        app.raster_brightness = float(raster_brightness)
    raster_height = _number(display, "raster_event_height", report, lo=0.1, hi=0.5)
    if raster_height is not None:
        app.raster_event_height = float(raster_height)
    raster_thickness = _number(
        display, "raster_event_thickness", report, lo=1, hi=10, integer=True
    )
    if raster_thickness is not None:
        app.raster_event_thickness = int(raster_thickness)
    for key, attr, action_attr in (
        ("scale_raster_proportionally", "scale_raster_proportionally", "action_proportional_raster"),
        ("scale_heatmap_proportionally", "scale_heatmap_proportionally", "action_proportional_heatmap"),
        ("compact_heatmaps_to_fit", "compact_heatmaps_to_fit", "action_compact_heatmaps_to_fit"),
        ("compact_rasters_to_fit", "compact_rasters_to_fit", "action_compact_rasters_to_fit"),
    ):
        value = _bool(display, key, report)
        if value is not None:
            setattr(app, attr, value)
            _set_action_checked(getattr(app, action_attr, None), value)

    labels = display.get("interval_labels")
    if isinstance(labels, Mapping):
        alpha = _number(labels, "alpha", report, lo=0.0, hi=1.0)
        if alpha is not None:
            app.interval_label_alpha_multiplier = float(alpha)
        overlays_visible = _bool(labels, "overlays_visible", report)
        if overlays_visible is not None:
            app.interval_label_overlays_enabled = overlays_visible
        strip_visible = _bool(labels, "strip_visible", report)
        if strip_visible is not None:
            app.label_strip_visible = strip_visible
            app.label_strip_widget.setVisible(strip_visible)
        hyp_visible = _bool(labels, "hypnogram_visible", report)
        if hyp_visible is not None and app.hypnogram_widget is not None:
            app.hypnogram_widget.setVisible(hyp_visible)
        hyp_zoomed = _bool(labels, "hypnogram_zoomed", report)
        if hyp_zoomed is not None:
            app.hypnogram_zoomed = hyp_zoomed
        app._refresh_interval_label_alpha()
        app._sync_label_display_actions()

    pending_y: list[tuple[_Target, Any]] = []
    matched_keys: list[tuple[int, tuple[str, int]]] = []
    for record, target in matches:
        factor = float(record.get("height", 1.0))
        visible = bool(record.get("visible", True))
        _set_factor_and_visible(app, target, factor, visible)
        state = record.get("state", {})
        if not isinstance(state, Mapping):
            report.skipped.append(f"{target.ref.label}: state is not an object")
            continue
        if target.kind == "ts":
            _apply_ts_state(app, target, state, report)
        elif target.kind == "dense":
            _apply_dense_state(app, target, state, report)
        elif target.kind == "heatmap":
            _apply_heatmap_state(app, target, state, report)
        else:
            _apply_raster_state(app, target, state, report)
        pending_y.append((target, state.get("y_axis")))
        matched_keys.append((int(record.get("order", 0)), target.key))

    matched_keys.sort(key=lambda x: x[0])
    ordered = [key for _, key in matched_keys]
    ordered.extend(key for key in _full_order(app) if key not in ordered)
    app.subplot_order = ordered
    app._apply_trace_visibility()

    for target, y_axis in pending_y:
        _apply_y_axis(app, target, y_axis, report)
    app._setup_dense_vscrollbars()
    app._sync_dense_vscrollbar_from_yrange()
    app._refresh_dense_curves()
    app._refresh_raster_pen_cache()
    app._refresh_raster_plots()
    app._refresh_heatmap_plots()
    app._refresh_curves()

    _apply_markers(app, config.sample_markers, current, report)
    _apply_global_events(app, config.global_events, report)
    _apply_videos(app, config.videos, report)

    ratio = _number(display, "splitter_ratio", report, lo=0.05, hi=0.95)
    if ratio is not None:
        app.splitter.setSizes([int(float(ratio) * 1000), int((1.0 - float(ratio)) * 1000)])
    tuner_visible = _bool(display, "tuner_visible", report)
    if tuner_visible is not None and app._tuner_params:
        app._toggle_tuner_panel(tuner_visible)
    app._sync_interval_label_visuals(force_rebuild=True)
    app._update_hypnogram_xrange()

    if config.session is not None:
        _apply_session(app, config.session, report)
    else:
        app.window_start = max(
            app.t_global_min,
            min(app.window_start, max(app.t_global_min, app.t_global_max - app.window_len)),
        )
        app.cursor_time = max(
            app.window_start,
            min(app.cursor_time, app.window_start + app.window_len),
        )
        app._apply_x_range()
        app._update_nav_slider_from_window()

    app._update_status(report.summary())
    return report
