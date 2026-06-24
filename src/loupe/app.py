#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loupe: multi-trace viewer with windowed rendering, video scrubbing,
page hotkeys, and cross-page draggable labeling.

pip install PySide6 pyqtgraph opencv-python numpy

Most leaf classes / helpers used by :class:`LoupeApp` live in dedicated
modules and are re-exported below for backward compatibility with code
that imports them from ``loupe.app`` (e.g. ``from loupe.app import
Series, DenseGroup, ...``):

* :mod:`loupe.series` — runtime data containers (Series, RasterSeries,
  HeatmapSeries, DenseGroup, IntervalLabelVisualBundle, SampleMarkers).
* :mod:`loupe._decimation` — pure utilities used in the per-paint
  refresh path (segment_for_window, find_nearest_frame, …).
* :mod:`loupe.viewboxes` — custom pyqtgraph ViewBox / PlotItem subclasses.
* :mod:`loupe.video` — video decode pipeline (VideoWorker, VideoSlot, …).
* :mod:`loupe.label_panel` — interval-label summary widgets.
* :mod:`loupe.dialogs` — subplot-controls dialogs (Y-axis, dense, heatmap).
* :mod:`loupe._heatmap_utils` — shared heatmap helpers used by both the
  renderer here and the heatmap-controls dialog.
"""

import glob
import math
import os
from functools import partial

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from loupe.interval_labels import (
    IntervalLabelIOError,
    IntervalLabelSchema,
    IntervalLabelSchemaError,
    IntervalLabelSet,
)
from loupe.state_config import StateConfig, load_state_config

from loupe._decimation import (
    _scatter_kwargs_for_marker,
    clamp,
    find_nearest_frame,
    next_pow_two,
    nice_time_range,
    segment_for_window,
)
from loupe._heatmap_utils import (
    ARRAY_COLORMAP_PRESETS,
    ARRAY_MIPMAP_TARGET_MIN_COLS,
    ARRAY_MIPMAP_THRESHOLD,
    _colormap_cache_token,
    _colormap_display_name,
)
from loupe.dialogs import (
    DenseViewControlsDialog,
    HeatmapControlsDialog,
    YAxisControlsDialog,
)
from loupe.label_panel import (
    IntervalLabelSummaryBarWidget,
    IntervalLabelSummaryWidget,
    StateComboDelegate,
)
from loupe.series import (
    DenseGroup,
    HeatmapSeries,
    IntervalLabelKey,
    IntervalLabelVisualBundle,
    RasterSeries,
    SampleMarkers,
    Series,
)
from loupe.video import (
    MultiFileVideoCapture,
    VideoSlot,
    VideoWorker,
    cv2,
)
from loupe.viewboxes import (
    DenseViewBox,
    HoverablePlotItem,
    SelectableViewBox,
)


# ---------------- Module-only constants ----------------

RASTER_ALPHA_LEVEL_COUNT = 11

# Maximum number of distinct categorical color values supported by `hue=`.
# Beyond this, the user almost certainly wants colormap-style binning rather
# than per-value colors, so we fail loudly. 32 categories × 11 alpha buckets
# = 352 PlotDataItems per raster subplot — a reasonable upper bound.
RASTER_MAX_CATEGORIES = 32

# Color used for events whose `hue` value is null/None/NaN. Matches the
# dense plot's _CATEGORY_NA_COLOR gray (sans the alpha channel — raster alpha
# is per-event, not part of the base color).
RASTER_NA_COLOR: tuple[int, int, int] = (160, 160, 160)

# Left y-axis gutter (pixels). All subplots share one width so their spines
# form a single vertical line. The width is auto-fit to the widest tick label
# at startup (see LoupeApp._align_left_axes), then locked, so autoscale
# relabeling causes no horizontal jitter. These bound that measured value.
LEFT_AXIS_WIDTH_FLOOR = 50  # never narrower than this
LEFT_AXIS_WIDTH_PAD = 6  # headroom added to the measured max


def _raster_extent(ms) -> float:
    """Total vertical extent of a raster in row-units, including any gaps
    opened by horizontal separators.  Falls back to the logical row count
    ``n_rows`` when no separators are present (the legacy behavior)."""
    return ms.y_extent if ms.y_extent is not None else ms.n_rows


# ---------------- Global event marker styling ----------------

_GLOBAL_EVENT_LINE_STYLE_TO_QT: dict[str, QtCore.Qt.PenStyle] = {
    "solid":      QtCore.Qt.PenStyle.SolidLine,
    "dashed":     QtCore.Qt.PenStyle.DashLine,
    "dotted":     QtCore.Qt.PenStyle.DotLine,
    "dashdot":    QtCore.Qt.PenStyle.DashDotLine,
    "dashdotdot": QtCore.Qt.PenStyle.DashDotDotLine,
}
_GLOBAL_EVENT_STYLE_ORDER: list[str] = [
    "solid", "dashed", "dotted", "dashdot", "dashdotdot",
]
_GLOBAL_EVENT_COLOR_CYCLE: list[tuple[int, int, int]] = [
    (230, 230, 230),  # light gray (most visible on dark bg)
    (180, 230, 255),  # light cyan
    (255, 230, 180),  # light cream
    (200, 255, 200),  # light green
    (255, 200, 200),  # light pink
    (230, 180, 255),  # light lavender
]
_GLOBAL_EVENT_SINGLE_DEFAULT: dict = {
    "line_color": (230, 230, 230),
    "line_style": "solid",
    "line_width": 1.5,
    "line_alpha": 200,
}
_GLOBAL_EVENT_VALID_STYLE_KEYS: frozenset = frozenset(
    {"line_color", "line_style", "line_width", "line_alpha"}
)
# Z-value sits above label regions, curves, images, sample-marker scatters,
# and the hypnogram window marker.
_GLOBAL_EVENT_Z: int = 100


def _parse_global_event_color(c) -> tuple[int, int, int]:
    """Normalize a hex string or RGB(A) tuple to an ``(r, g, b)`` 3-tuple."""
    if isinstance(c, str):
        s = c.strip().lstrip("#")
        if len(s) in (6, 8):
            try:
                return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
            except ValueError:
                pass
        raise ValueError(f"Cannot parse global-event line_color: {c!r}")
    if isinstance(c, (tuple, list)) and len(c) >= 3:
        return (int(c[0]), int(c[1]), int(c[2]))
    raise ValueError(f"Cannot parse global-event line_color: {c!r}")


# ---------------- Main window ----------------


class LoupeApp(QtWidgets.QMainWindow):
    # --- state_config convenience accessors ---
    @property
    def keymap(self) -> dict[str, str]:
        """Forward ``{key: state}`` mapping derived from :attr:`state_config`."""
        return self.state_config.key_to_state

    @property
    def label_colors(self) -> dict[str, tuple[int, int, int, int]]:
        return self.state_config.label_colors

    @property
    def labels_writeback_allowed(self) -> bool:
        return self.interval_label_set.writeback_allowed

    def __init__(
        self,
        data_dir=None,
        data_files=None,
        colors=None,
        video_configs: list | None = None,
        fixed_scale=True,
        window_len: float = 10.0,
        # Raster viewer arguments
        raster_timestamps=None,
        raster_yvals=None,
        raster_alphas=None,
        raster_colors=None,
        # xarray series (pre-converted list[Series])
        xr_series=None,
        # Pre-converted RasterSeries from df_loader
        raster_series_list=None,
        # Overlay mode
        overlay_groups=None,
        overlay_colors=None,
        # Dense mode
        dense_groups=None,
        # Heatmap mode
        heatmap_series=None,
        # Initial layout order — list of ("ts"|"dense"|"raster"|"heatmap", idx)
        # tuples that specifies the visual subplot order top-to-bottom.
        # When None, falls back to the type-segregated default
        # (ts → dense → raster → heatmap). User can still rearrange interactively
        # via the Plot Order dialog after launch.
        subplot_order=None,
        # Sample-aligned marker overlays for stacked-subplots traces.
        sample_markers: list[SampleMarkers] | None = None,
        # Per-stacked-series overlay curves (parallel to xr_series). Each entry
        # is a list[OverlayCurve] drawn on that host series' subplot.
        overlay_series: list | None = None,
        # Per-stacked-series host legend label (parallel to xr_series); None for
        # series without overlays (keeps their legend empty as before).
        overlay_main_names: list | None = None,
        # Per-stacked-trace bool flags: draw a minimal bottom-boundary line.
        bottom_spines: list | None = None,
        # State definitions (keymap + label colors)
        state_config: StateConfig | None = None,
        # Interval-label data
        interval_label_set: IntervalLabelSet | None = None,
        # Initial interval-label-overlay alpha multiplier (0.0 – 1.0). None → 1.0.
        interval_label_alpha: float | None = None,
        # Global event markers: vertical lines drawn across every pane.
        global_events=None,
        # Optional ProgressReporter for launch-time progress.
        reporter=None,
    ):
        super().__init__()
        self.setWindowTitle("Loupe — Multi-Trace + Video + Labeling")
        self.resize(1400, 900)

        if reporter is None:
            from loupe.progress import null_reporter
            reporter = null_reporter()
        self._reporter = reporter

        pg.setConfigOptions(
            antialias=False, useOpenGL=True, background="k", foreground="w"
        )

        # Data & plots
        self.series: list[Series] = []
        self.t_global_min = 0.0
        self.t_global_max = 1.0
        self.plots: list[pg.PlotItem] = []
        self.curves: list[pg.PlotDataItem] = []
        self.plot_cur_lines: list[pg.InfiniteLine] = []
        self.plot_sel_regions: list[pg.LinearRegionItem] = []
        # Optional minimal bottom-boundary line per trace subplot (or None when
        # disabled). Parallel to self.plots / self.series by index.
        self.plot_bottom_spines: list = []
        self.hovered_plot = None  # *** FIX 2: Track which plot is hovered ***

        # Locked uniform left-axis width (px). None until auto-fit measures it
        # on the first painted layout; reset to None on full rebuilds so new
        # content re-fits. See _align_left_axes.
        self._left_axis_width: int | None = None

        # Sample-aligned marker overlays — outer index = series index,
        # inner index = marker-set index (matches self.sample_markers order).
        self.sample_markers: list[SampleMarkers] = list(sample_markers) if sample_markers else []
        self.sample_marker_scatters: list[list[pg.ScatterPlotItem]] = []
        # Per-host-series overlay curves (parallel to self.series). Outer index =
        # host series index; inner = OverlayCurve drawn on that subplot. The
        # *_items list holds the matching pyqtgraph PlotDataItems after build.
        self.overlay_series: list[list] = list(overlay_series) if overlay_series else []
        self.overlay_main_names: list = list(overlay_main_names) if overlay_main_names else []
        self.overlay_curve_items: list[list[pg.PlotDataItem]] = []
        # Per-series add_bottom_spine flags (indexed by stacked-trace index).
        self.series_bottom_spine: list = list(bottom_spines) if bottom_spines else []

        # Overlay mode
        self.overlay_mode: bool = False
        self.overlay_groups: list = []  # list[OverlayGroup]
        self.overlay_colors: list[tuple] = []
        self._plot_to_curves: list[list[pg.PlotDataItem]] = []
        self._plot_to_series: list[list[int]] = []

        # Dense mode (EEG-style stacked traces on a single axis)
        self.dense_groups: list[DenseGroup] = []
        self.dense_plots: list[pg.PlotItem] = []
        self.dense_curves: list[list[pg.PlotDataItem]] = []
        self.dense_cur_lines: list[pg.InfiniteLine] = []
        self.dense_sel_regions: list[pg.LinearRegionItem] = []
        self.dense_interval_label_regions: list[list[pg.LinearRegionItem]] = []
        self.dense_height_factors: list[float] = []
        self.dense_visible: list[bool] = []
        self._dense_means: list[list[float]] = []
        self.dense_vscrollbars: list[QtWidgets.QScrollBar] = []
        self.dense_vscroll_proxies: list[QtWidgets.QGraphicsProxyWidget] = []
        self._dense_vscroll_inverted: list[bool] = []

        # Raster viewer data and plots
        self.raster_series: list[RasterSeries] = []
        self.raster_plots: list[pg.PlotItem] = []
        self.raster_items: list[pg.ScatterPlotItem | None] = []
        self.raster_cur_lines: list[pg.InfiniteLine] = []
        self.raster_sel_regions: list[pg.LinearRegionItem] = []
        # One inner list of horizontal-separator InfiniteLines per raster subplot.
        self.raster_separator_lines: list[list[pg.InfiniteLine]] = []
        # Shape: [raster_idx][category_idx][alpha_level] -> PlotDataItem / QPen.
        # For non-categorical raster series the outer category dim is length 1.
        self._raster_line_items: list[list[list[pg.PlotDataItem]]] = []
        self._raster_pens: list[list[list[QtGui.QPen]]] = []
        # Raster rendering settings
        self.raster_event_height = (
            0.2  # distance from center in each direction (0.1-0.5)
        )
        self.raster_event_thickness = 2  # pen width in pixels
        self.scale_raster_proportionally = True  # toggled via View menu
        self.raster_brightness = (
            1.0  # brightness multiplier for alpha values (0.2 to 3.0)
        )
        # alpha multiplier for interval-label overlay regions (0.0 to 1.0)
        if interval_label_alpha is None:
            self.interval_label_alpha_multiplier = 1.0
        else:
            if (
                not isinstance(interval_label_alpha, (int, float))
                or isinstance(interval_label_alpha, bool)
                or math.isnan(float(interval_label_alpha))
                or not (0.0 <= float(interval_label_alpha) <= 1.0)
            ):
                raise ValueError(
                    f"interval_label_alpha must be a float in [0.0, 1.0], "
                    f"got {interval_label_alpha!r}"
                )
            self.interval_label_alpha_multiplier = float(interval_label_alpha)
        # Custom height factors for individual plot height control (1.0 = default)
        self.plot_height_factors: list[float] = []  # one per time series plot
        self.raster_height_factors: list[float] = []  # one per raster plot
        # Visibility flags for raster plots (similar to trace_visible for time series)
        self.raster_visible: list[bool] = []

        # Heatmap plots
        self.heatmap_series: list[HeatmapSeries] = []
        self.heatmap_plots: list[pg.PlotItem] = []
        self.heatmap_image_items: list[pg.ImageItem] = []
        self.heatmap_cur_lines: list[pg.InfiniteLine] = []
        self.heatmap_sel_regions: list[pg.LinearRegionItem] = []
        self.heatmap_height_factors: list[float] = []
        self.heatmap_visible: list[bool] = []
        # Proportional sizing (mirror raster-plot behaviour). On by default
        # because most users prefer per-row weighting for heatmaps.
        self.scale_heatmap_proportionally = True
        # When True, scale all heatmap plots down uniformly so the total height
        # of every visible subplot fits in the plot-area viewport without
        # vertical scrolling. Toggled via View menu. Re-evaluated on every
        # resize. Has no effect when there are no heatmap plots.
        self.compact_heatmaps_to_fit = True
        # Cache last-rendered key per heatmap plot so identical refreshes return early
        self._heatmap_cache_keys: list[tuple | None] = []
        # Cache uint8 RGBA LUTs by colormap name (built once per name)
        self._lut_cache: dict[str, np.ndarray] = {}
        # Plot order: list of (type, index) tuples, e.g., [("ts", 0), ("ts", 1), ("raster", 0)]
        # None means use default order (all ts first, then all raster)
        self.subplot_order: list[tuple] | None = None

        # Rendering budget (per plot)
        self.max_pts_per_plot = 4000

        # Window/cursor & labels
        self.window_len = float(window_len)
        self.window_start = 0.0
        self.cursor_time = 0.0
        # Vertical paging for stacked-subplots view
        self.trace_height_px = 120  # pixels per stacked subplot

        # State definitions (hotkeys + label colors). If the caller didn't
        # pre-build a StateConfig, resolve one now from the package-default
        # state_definitions.json. With no defaults available, this raises
        # LoupeConfigError.
        if state_config is None:
            state_config = load_state_config()
        self.state_config: StateConfig = state_config

        # Interval-label data (DataFrame-backed). Defaults to an empty legacy schema.
        if interval_label_set is None:
            interval_label_set = IntervalLabelSet.empty()
        self.interval_label_set: IntervalLabelSet = interval_label_set

        # Global event markers (vertical lines drawn across every pane).
        # _resolved_event_styles keys are unique values from
        # global_events.style_events_on (or None for the single-style case).
        # _global_event_lines_by_class maps the same keys to the list of
        # InfiniteLine objects belonging to that class — enables live restyle
        # via the "Style Global Events…" dialog.
        self.global_events = global_events
        self._resolved_event_styles: dict = {}
        self._global_event_lines_by_class: dict = {}
        if self.global_events is not None:
            self._resolve_global_event_styles()

        # Visual bookkeeping keyed by row_id (stable across edits/merges).
        self._interval_label_visuals: dict[IntervalLabelKey, IntervalLabelVisualBundle] = {}
        self._hypnogram_interval_label_visuals: dict[IntervalLabelKey, pg.LinearRegionItem] = {}
        # Drawn (start, end, label) for each hypnogram region, so the sync can
        # detect a surviving row whose geometry/label changed in place (the
        # hypnogram regions are bare LinearRegionItems with no bundle to carry
        # this, unlike the window visuals above).
        self._hypnogram_interval_label_drawn: dict[
            IntervalLabelKey, tuple[float, float, str]
        ] = {}
        # Mirror the IntervalLabelSet's row_ids/starts/ends so visual sync code can
        # index into them before the GUI runs its first _finalize_interval_label_change.
        self._interval_label_keys_in_order: list[IntervalLabelKey] = [
            int(rid) for rid in self.interval_label_set.row_ids
        ]
        self._interval_label_starts = np.asarray(self.interval_label_set.starts, dtype=float)
        self._interval_label_ends = np.asarray(self.interval_label_set.ends, dtype=float)
        self._select_start = None
        self._select_end = None
        self._is_zoom_drag = False
        self.fixed_scale = bool(fixed_scale)

        # Video slots — one VideoSlot per VideoConfig. Workers + threads
        # are owned by the slot; labels/menu actions get attached later
        # during _build_ui / _build_menu. Labels and shortcuts top out
        # gracefully when more than 9 slots are supplied.
        self.video_slots: list[VideoSlot] = []
        configs = list(video_configs) if video_configs else []
        for i, cfg in enumerate(configs):
            thread = QtCore.QThread(self)
            worker = VideoWorker(cache_frames=120)
            worker.moveToThread(thread)
            name = getattr(cfg, "name", None) or f"Video {i + 1}"
            stretch = getattr(cfg, "stretch", None)
            if stretch is None:
                stretch = 3 if i == 0 else 2
            slot = VideoSlot(
                index=i,
                name=name,
                stretch=int(stretch),
                worker=worker,
                thread=thread,
                video_path=getattr(cfg, "video_path", None),
                frame_times_path=getattr(cfg, "frame_times_path", None),
                frame_times_correction=float(
                    getattr(cfg, "frame_times_correction", 0.0) or 0.0
                ),
            )
            self.video_slots.append(slot)

        # Playback
        self.is_playing = False
        self.playback_timer = QtCore.QTimer(self)
        self.playback_timer.timeout.connect(self._advance_playback_frame)
        self.playback_elapsed_timer = QtCore.QElapsedTimer()

        # Smooth scroll settings (fraction of window per wheel step)
        self.smooth_scroll_fraction = 0.10
        self._deferred_view_refresh_timer = QtCore.QTimer(self)
        self._deferred_view_refresh_timer.setSingleShot(True)
        self._deferred_view_refresh_timer.timeout.connect(
            self._flush_deferred_view_refresh
        )
        self._deferred_view_refresh_needs_nav_slider = False
        # Playback speed (1.0 = real time)
        self.playback_speed = 1.0
        # Index into self.video_slots that clocks frame-by-frame stepping
        self.frame_step_source = 0

        # Hypnogram overview
        self.hypnogram_widget = None
        self.hypnogram_plot = None
        self._hypnogram_window_marker_lines: list[pg.InfiniteLine] = []
        self.hypnogram_zoomed = False
        self.hypnogram_zoom_padding = 30.0

        # Right panel layout references (used in _build_ui)
        self.right_layout = None
        self.videos_layout = None
        self.videos_widget = None

        self._build_ui()

        self.y_axis_dialog = None

        for slot in self.video_slots:
            slot.worker.frameReady.connect(partial(self._on_frame_ready, slot))
            slot.worker.opened.connect(partial(self._on_video_opened, slot))
            slot.thread.start()
        # Store dense groups early (before set_series triggers plot creation)
        if dense_groups:
            self.dense_groups = dense_groups
            self.dense_height_factors = [1.0] * len(dense_groups)
            self.dense_visible = [True] * len(dense_groups)
            # Cache per-trace means for display transform
            self._dense_means = [
                [float(np.nanmean(s.y)) for s in g.series]
                for g in dense_groups
            ]

        # Store heatmap series early (before set_series triggers plot creation)
        if heatmap_series:
            self.heatmap_series = list(heatmap_series)
            self.heatmap_height_factors = [1.0] * len(self.heatmap_series)
            self.heatmap_visible = [True] * len(self.heatmap_series)
            self._heatmap_cache_keys = [None] * len(self.heatmap_series)

        # Store raster series early too so set_series's _create_all_plots
        # picks them up in a single pass — otherwise it builds trace-only
        # plots first, then _rebuild_all_plots tears them down to add rasters.
        if raster_series_list:
            self.raster_series = list(raster_series_list)
            self.raster_height_factors = [1.0] * len(self.raster_series)
            self.raster_visible = [True] * len(self.raster_series)
            self.subplot_order = None

        # Prefer overlay groups, then xarray series, then explicit file list, then dir
        if overlay_groups:
            self.set_overlay_series(overlay_groups, colors=overlay_colors)
        elif xr_series or dense_groups or heatmap_series:
            self.set_series(xr_series or [], colors=colors)

        # User-supplied initial layout order overrides the default segregated
        # order produced by set_series. Apply AFTER set_series, then rebuild
        # the layout so the first display reflects the requested order.
        if subplot_order is not None:
            self.subplot_order = list(subplot_order)
            self._rebuild_all_plots()
        elif data_files:
            # Allow flexible formats: list[str], comma-separated strings, or "[a,b]".
            def _normalize_file_list(df):
                if not df:
                    return []
                items = [df] if isinstance(df, str) else list(df)
                out = []
                for it in items:
                    s = (it or "").strip()
                    # Strip surrounding list brackets if present
                    if s.startswith("[") and s.endswith("]"):
                        s = s[1:-1]
                    parts = s.split(",") if "," in s else [s]
                    for p in parts:
                        q = p.strip().strip('"').strip("'")
                        # Remove a lingering trailing comma if passed as a token like "file.npy,"
                        if q.endswith(","):
                            q = q[:-1].rstrip()
                        if q:
                            out.append(q)
                return out

            def _normalize_list(raw_list):
                if not raw_list:
                    return []
                items = [raw_list] if isinstance(raw_list, str) else list(raw_list)
                out = []
                for it in items:
                    s = (it or "").strip()
                    if s.startswith("[") and s.endswith("]"):
                        s = s[1:-1]
                    parts = s.split(",") if "," in s else [s]
                    for p in parts:
                        q = p.strip().strip('"').strip("'")
                        if q.endswith(","):
                            q = q[:-1].rstrip()
                        if q:
                            out.append(q)
                return out

            paths = _normalize_file_list(data_files)
            color_list = _normalize_list(colors) if colors else None
            self._load_series_from_files(paths, colors=color_list)
        elif data_dir:
            self._load_series_from_dir(data_dir)
        for slot in self.video_slots:
            if slot.video_path and slot.frame_times_path:
                self._load_video_data(slot, slot.video_path, slot.frame_times_path)

        # Load raster viewer data if provided
        if raster_timestamps and raster_yvals:
            self._load_raster_data(
                raster_timestamps, raster_yvals, raster_alphas, raster_colors
            )

        # Pre-converted RasterSeries (from df_loader) were stored above so
        # set_series's first _create_all_plots already includes them. Here we
        # only need to handle the cases set_series didn't cover: overlay mode
        # (which doesn't include rasters in _create_overlay_plots) and the
        # raster-only path (no traces/dense/heatmap/overlay at all).
        if raster_series_list:
            self._update_status(
                f"Loaded {len(raster_series_list)} raster series from DataFrame."
            )
            if overlay_groups:
                self._rebuild_all_plots()
            elif not (xr_series or dense_groups or heatmap_series):
                self._update_time_range_from_raster()
                self._create_raster_only_plots()

    def eventFilter(self, obj, ev):
        try:
            if ev.type() == QtCore.QEvent.Type.Resize:
                for slot in self.video_slots:
                    if obj is slot.label:
                        self._rescale_video_frame(slot)
                        break
        except Exception:
            pass
        return super().eventFilter(obj, ev)

    # ---------- UI ----------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(4, 2, 4, 2)
        top.setSpacing(6)
        v.addLayout(top)
        top.addWidget(QtWidgets.QLabel("Window (s):"))
        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(0.1, 3600.0)
        self.window_spin.setDecimals(2)
        self.window_spin.setValue(self.window_len)
        self.window_spin.valueChanged.connect(self._on_window_len_changed)
        top.addWidget(self.window_spin)
        top.addSpacing(20)
        top.addWidget(QtWidgets.QLabel("Navigate:"))
        self.nav_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.nav_slider.setRange(0, 10000)
        self.nav_slider.valueChanged.connect(self._on_nav_slider_changed)
        top.addWidget(self.nav_slider, 1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        v.addWidget(splitter, 1)
        self.splitter = splitter

        # left plots (wrapped in scroll area for vertical paging)
        left = QtWidgets.QWidget()
        leftl = QtWidgets.QVBoxLayout(left)
        leftl.setContentsMargins(0, 0, 0, 0)
        self.plot_area = pg.GraphicsLayoutWidget()
        self.plot_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.plot_area.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.plot_area.ci.layout.setVerticalSpacing(2)
        self.plot_scroll_area = QtWidgets.QScrollArea()
        self.plot_scroll_area.setWidgetResizable(True)
        self.plot_scroll_area.setWidget(self.plot_area)
        self.plot_scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        # Prevent scroll area from stealing PageUp/PageDown key events
        self.plot_scroll_area.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.plot_scroll_area.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.plot_scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        leftl.addWidget(self.plot_scroll_area, 1)
        splitter.addWidget(left)

        # right side
        right = QtWidgets.QWidget()
        right.setMinimumWidth(150)
        rl = QtWidgets.QVBoxLayout(right)
        self.right_layout = rl
        # Group videos into a dedicated container so we can control relative sizes
        self.videos_widget = QtWidgets.QWidget()
        self.videos_layout = QtWidgets.QVBoxLayout(self.videos_widget)
        self.videos_layout.setContentsMargins(0, 0, 0, 0)
        self.videos_layout.setSpacing(4)

        # Per-slot QLabels for video frames. Primary stays visible (shows
        # the "No video" placeholder until a frame arrives); non-primary
        # slots start hidden and reveal themselves on successful open.
        for slot in self.video_slots:
            lbl = QtWidgets.QLabel(f"No {slot.name.lower()}")
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumHeight(240 if slot.index == 0 else 200)
            lbl.setStyleSheet("background-color:#222;border:1px solid #444;")
            if slot.index != 0:
                lbl.hide()
            self.videos_layout.addWidget(lbl, slot.stretch)
            lbl.installEventFilter(self)
            slot.label = lbl
        # When there are no videos at all, keep a placeholder so the area
        # is not empty (matches the prior "No video" look).
        if not self.video_slots:
            placeholder = QtWidgets.QLabel("No video")
            placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            placeholder.setMinimumHeight(240)
            placeholder.setStyleSheet(
                "background-color:#222;border:1px solid #444;"
            )
            self.videos_layout.addWidget(placeholder, 3)
            self._video_placeholder = placeholder

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Cursor:"))
        self.window_cursor_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.window_cursor_slider.setRange(0, 10000)
        self.window_cursor_slider.valueChanged.connect(self._on_window_cursor_changed)
        row.addWidget(self.window_cursor_slider)

        roww = QtWidgets.QWidget()
        roww.setLayout(row)
        # Add videos container before cursor row
        rl.addWidget(self.videos_widget, 1)
        rl.addWidget(roww)

        # Label Summary panel (replaces former static image; hidden when
        # any non-primary video opens — see _on_video_opened).
        self.interval_label_summary_panel = IntervalLabelSummaryWidget(main_window=self)
        rl.addWidget(self.interval_label_summary_panel, 2)

        # Hypnogram overview plot (full-recording labels with moving window box)
        self.hypnogram_widget = pg.PlotWidget()
        self.hypnogram_widget.setMinimumHeight(90)
        hp = self.hypnogram_widget.getPlotItem()
        hp.showGrid(x=False, y=False)
        hp.hideAxis("left")
        hp.setMenuEnabled(False)
        hp.setMouseEnabled(x=False, y=False)
        hp.enableAutoRange("y", False)
        hp.setYRange(0, 1)
        self.hypnogram_plot = hp
        rl.addWidget(self.hypnogram_widget, 1)

        # Two-tone outlined frame marking the current window on the hypnogram.
        # Each edge is a thick dark outer line topped by a thinner bright inner
        # line; the dark stroke dominates against light label fills, the bright
        # stroke against dark ones, keeping the marker visible at any palette.
        # Order: [start outer, start inner, end outer, end inner] — outers are
        # added first so the inners draw on top at the same z-value.
        a = self.window_start
        b = self.window_start + self.window_len
        outer_pen = pg.mkPen(0, 0, 0, 230, width=4)
        inner_pen = pg.mkPen(255, 255, 255, 230, width=1.5)
        self._hypnogram_window_marker_lines = [
            pg.InfiniteLine(pos=a, angle=90, movable=False, pen=outer_pen),
            pg.InfiniteLine(pos=a, angle=90, movable=False, pen=inner_pen),
            pg.InfiniteLine(pos=b, angle=90, movable=False, pen=outer_pen),
            pg.InfiniteLine(pos=b, angle=90, movable=False, pen=inner_pen),
        ]
        for line in self._hypnogram_window_marker_lines:
            line.setZValue(20)
            self.hypnogram_plot.addItem(line)

        # Ensure rescale happens when the splitter is adjusted
        self.splitter.splitterMoved.connect(self._on_splitter_moved)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.status = self.statusBar()
        self._update_status()
        self._build_menu()
        # Apply initial video stretches
        self._apply_video_stretches()

    def _build_menu(self):
        mfile = self.menuBar().addMenu("&File")
        a = QtGui.QAction("Load &Time Series…", self)
        a.triggered.connect(self._on_load_time_series)
        mfile.addAction(a)
        b = QtGui.QAction("Load &Video && Frame Times…", self)
        b.triggered.connect(self._on_load_video)
        mfile.addAction(b)
        m = QtGui.QAction("Load &Raster Data…", self)
        m.triggered.connect(self._on_load_raster_data)
        mfile.addAction(m)
        mfile.addSeparator()

        c = QtGui.QAction("Load Interval &Labels…", self)
        c.triggered.connect(self._on_load_interval_labels)
        mfile.addAction(c)

        d = QtGui.QAction("&Export Interval Labels As…", self)
        d.triggered.connect(self._on_export_interval_labels)
        mfile.addAction(d)

        save_action = QtGui.QAction("&Save Interval Labels (overwrite source)", self)
        save_action.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self._on_save_to_source)
        # Disabled unless the user opted in via interval_labels_writeback=True.
        save_action.setEnabled(self.interval_label_set.writeback_allowed)
        self._save_to_source_action = save_action
        mfile.addAction(save_action)
        mfile.addSeparator()

        q = QtGui.QAction("&Quit", self)
        q.triggered.connect(self.close)
        mfile.addAction(q)

        medit = self.menuBar().addMenu("&Edit")
        clr = QtGui.QAction("Clear current selection", self)
        clr.triggered.connect(self._clear_selection)
        medit.addAction(clr)
        dl = QtGui.QAction("Delete last label", self)
        dl.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Backspace))
        dl.triggered.connect(self._delete_last_label)
        medit.addAction(dl)
        medit.addSeparator()
        note_action = QtGui.QAction("Add/Edit Epoch Note...", self)
        note_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+N"))
        note_action.triggered.connect(self._edit_epoch_note)
        medit.addAction(note_action)

        mview = self.menuBar().addMenu("&View")

        # ----- Group 1: Navigation & playback -----
        play_action = QtGui.QAction("Toggle Playback", self)
        play_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Space))
        play_action.triggered.connect(self._toggle_playback)
        mview.addAction(play_action)

        page_left_action = QtGui.QAction("Page Left", self)
        page_left_action.setShortcuts(
            [
                QtGui.QKeySequence(QtCore.Qt.Key.Key_BracketLeft),
                QtGui.QKeySequence(QtCore.Qt.Key.Key_PageUp),
            ]
        )
        page_left_action.triggered.connect(lambda: self._page(-1))
        mview.addAction(page_left_action)

        page_right_action = QtGui.QAction("Page Right", self)
        page_right_action.setShortcuts(
            [
                QtGui.QKeySequence(QtCore.Qt.Key.Key_BracketRight),
                QtGui.QKeySequence(QtCore.Qt.Key.Key_PageDown),
            ]
        )
        page_right_action.triggered.connect(lambda: self._page(+1))
        mview.addAction(page_right_action)

        step_back_action = QtGui.QAction("Step Frame Back", self)
        step_back_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Left))
        step_back_action.triggered.connect(lambda: self._step_frame(-1))
        mview.addAction(step_back_action)

        step_fwd_action = QtGui.QAction("Step Frame Forward", self)
        step_fwd_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Right))
        step_fwd_action.triggered.connect(lambda: self._step_frame(+1))
        mview.addAction(step_fwd_action)

        next_epoch_action = QtGui.QAction("Next Epoch", self)
        next_epoch_action.setShortcut(QtGui.QKeySequence("N"))
        next_epoch_action.triggered.connect(lambda: self._jump_to_epoch_by_offset(+1))
        mview.addAction(next_epoch_action)

        prev_epoch_action = QtGui.QAction("Previous Epoch", self)
        prev_epoch_action.setShortcut(QtGui.QKeySequence("B"))
        prev_epoch_action.triggered.connect(lambda: self._jump_to_epoch_by_offset(-1))
        mview.addAction(prev_epoch_action)

        jump_epochs_action = QtGui.QAction("Jump to Epochs...", self)
        jump_epochs_action.setShortcut(QtGui.QKeySequence("Ctrl+J"))
        jump_epochs_action.triggered.connect(self._show_jump_to_epochs_dialog)
        mview.addAction(jump_epochs_action)

        playback_speed_action = QtGui.QAction("Set Playback Speed...", self)
        playback_speed_action.triggered.connect(self._adjust_playback_speed)
        mview.addAction(playback_speed_action)

        scroll_speed_action = QtGui.QAction("Adjust Smooth Scroll Speed...", self)
        scroll_speed_action.triggered.connect(self._adjust_scroll_speed)
        mview.addAction(scroll_speed_action)

        self.action_fullscreen = QtGui.QAction("Fullscreen", self)
        self.action_fullscreen.setCheckable(True)
        self.action_fullscreen.setShortcut(QtGui.QKeySequence("F11"))
        self.action_fullscreen.toggled.connect(self._toggle_fullscreen)
        mview.addAction(self.action_fullscreen)

        # ----- Group 2: Hypnogram -----
        mview.addSeparator()

        toggle_hyp_vis_action = QtGui.QAction("Toggle Hypnogram Visibility", self)
        toggle_hyp_vis_action.setShortcut(QtGui.QKeySequence("H"))
        toggle_hyp_vis_action.triggered.connect(self._toggle_hypnogram_visibility)
        mview.addAction(toggle_hyp_vis_action)

        toggle_hyp_zoom_action = QtGui.QAction("Toggle Hypnogram Zoom", self)
        toggle_hyp_zoom_action.setShortcut(QtGui.QKeySequence("Z"))
        toggle_hyp_zoom_action.triggered.connect(self._toggle_hypnogram_zoom)
        mview.addAction(toggle_hyp_zoom_action)

        # ----- Group 3: Trace / time-series plots -----
        mview.addSeparator()

        y_axis_action = QtGui.QAction("Y-Axis Controls...", self)
        y_axis_action.setShortcut(QtGui.QKeySequence("Ctrl+D"))
        y_axis_action.triggered.connect(self._show_y_axis_dialog)
        mview.addAction(y_axis_action)

        zoom_y_in_action = QtGui.QAction("Zoom Y In (hovered plot)", self)
        zoom_y_in_action.setShortcut(QtGui.QKeySequence("Ctrl+1"))
        zoom_y_in_action.triggered.connect(lambda: self._zoom_active_plot_y(0.9))
        mview.addAction(zoom_y_in_action)

        zoom_y_out_action = QtGui.QAction("Zoom Y Out (hovered plot)", self)
        zoom_y_out_action.setShortcut(QtGui.QKeySequence("Ctrl+2"))
        zoom_y_out_action.triggered.connect(lambda: self._zoom_active_plot_y(1.1))
        mview.addAction(zoom_y_out_action)

        dense_ctrl_action = QtGui.QAction("Dense View Controls...", self)
        dense_ctrl_action.setShortcut(QtGui.QKeySequence("Ctrl+G"))
        dense_ctrl_action.triggered.connect(self._show_dense_controls_dialog)
        mview.addAction(dense_ctrl_action)

        interval_label_alpha_action = QtGui.QAction("Adjust Interval Label Alpha...", self)
        interval_label_alpha_action.triggered.connect(self._adjust_interval_label_alpha)
        mview.addAction(interval_label_alpha_action)

        # ----- Group 4: Raster plots -----
        mview.addSeparator()

        self.action_proportional_raster = QtGui.QAction(
            "Proportional Raster Plots", self
        )
        self.action_proportional_raster.setCheckable(True)
        self.action_proportional_raster.setChecked(self.scale_raster_proportionally)
        self.action_proportional_raster.setShortcut(QtGui.QKeySequence("Ctrl+Shift+R"))
        self.action_proportional_raster.toggled.connect(
            self._toggle_proportional_raster
        )
        mview.addAction(self.action_proportional_raster)

        raster_brightness_action = QtGui.QAction("Adjust Raster Brightness...", self)
        raster_brightness_action.triggered.connect(self._adjust_raster_brightness)
        mview.addAction(raster_brightness_action)

        raster_height_action = QtGui.QAction("Raster Event Height...", self)
        raster_height_action.triggered.connect(self._adjust_raster_event_height)
        mview.addAction(raster_height_action)

        raster_thickness_action = QtGui.QAction("Raster Event Thickness...", self)
        raster_thickness_action.triggered.connect(self._adjust_raster_event_thickness)
        mview.addAction(raster_thickness_action)

        adjust_sample_markers_action = QtGui.QAction(
            "Adjust Sample Marker Properties...", self
        )
        adjust_sample_markers_action.triggered.connect(
            self._adjust_sample_marker_properties
        )
        mview.addAction(adjust_sample_markers_action)

        if self.global_events is not None:
            style_global_events_action = QtGui.QAction(
                "Style Global Events...", self
            )
            style_global_events_action.triggered.connect(
                self._adjust_global_events_style
            )
            mview.addAction(style_global_events_action)

        # ----- Group 5: Heatmap plots -----
        mview.addSeparator()

        self.action_proportional_heatmap = QtGui.QAction(
            "Proportional Heatmap Plots", self
        )
        self.action_proportional_heatmap.setCheckable(True)
        self.action_proportional_heatmap.setChecked(self.scale_heatmap_proportionally)
        self.action_proportional_heatmap.toggled.connect(
            self._toggle_proportional_heatmap
        )
        mview.addAction(self.action_proportional_heatmap)

        # Uniformly compress heatmap plots so the entire stack fits on screen
        # (no vertical scrollbar). On by default; users with few heatmaps can
        # turn it off to get the full per-row 12px sizing.
        self.action_compact_heatmaps_to_fit = QtGui.QAction(
            "Compact Heatmap Plots to Fit Screen", self
        )
        self.action_compact_heatmaps_to_fit.setCheckable(True)
        self.action_compact_heatmaps_to_fit.setChecked(self.compact_heatmaps_to_fit)
        self.action_compact_heatmaps_to_fit.toggled.connect(
            self._toggle_compact_heatmaps_to_fit
        )
        mview.addAction(self.action_compact_heatmaps_to_fit)

        heatmap_ctrl_action = QtGui.QAction("Heatmap Plot Controls...", self)
        heatmap_ctrl_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+H"))
        heatmap_ctrl_action.triggered.connect(self._show_heatmap_controls_dialog)
        mview.addAction(heatmap_ctrl_action)

        # ----- Group 6: Subplot layout -----
        mview.addSeparator()

        subplot_control_action = QtGui.QAction("Subplot Control Board...", self)
        subplot_control_action.setShortcut(QtGui.QKeySequence("Ctrl+H"))
        subplot_control_action.triggered.connect(self._show_subplot_control_dialog)
        mview.addAction(subplot_control_action)

        increase_focused_height_action = QtGui.QAction(
            "Increase Focused Subplot Height", self
        )
        increase_focused_height_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+,"))
        increase_focused_height_action.triggered.connect(
            lambda: self._adjust_focused_subplot_height(1.1)
        )
        mview.addAction(increase_focused_height_action)

        decrease_focused_height_action = QtGui.QAction(
            "Decrease Focused Subplot Height", self
        )
        decrease_focused_height_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+."))
        decrease_focused_height_action.triggered.connect(
            lambda: self._adjust_focused_subplot_height(0.9)
        )
        mview.addAction(decrease_focused_height_action)

        reset_focused_height_action = QtGui.QAction(
            "Reset Focused Subplot Height", self
        )
        reset_focused_height_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+0"))
        reset_focused_height_action.triggered.connect(
            self._reset_focused_subplot_height
        )
        mview.addAction(reset_focused_height_action)

        # ----- Group 7: Videos -----
        mview.addSeparator()

        # Show/Hide videos (with Ctrl+Shift+N hotkeys for N=1..9)
        for slot in self.video_slots:
            action = QtGui.QAction(f"Show {slot.name}", self)
            action.setCheckable(True)
            action.setChecked(slot.index == 0)
            if slot.index < 9:
                action.setShortcut(
                    QtGui.QKeySequence(f"Ctrl+Shift+{slot.index + 1}")
                )
            action.toggled.connect(partial(self._set_video_visible, slot.index))
            mview.addAction(action)
            slot.show_action = action

        adjust_video_sizes_action = QtGui.QAction(
            "Adjust Secondary Videos Size...", self
        )
        adjust_video_sizes_action.triggered.connect(self._adjust_secondary_video_sizes)
        mview.addAction(adjust_video_sizes_action)

        # Frame-step target selector
        step_menu = mview.addMenu("Frame Step Target")
        self.step_action_group = QtGui.QActionGroup(self)
        self.step_action_group.setExclusive(True)
        for slot in self.video_slots:
            action = QtGui.QAction(slot.name, self, checkable=True)
            self.step_action_group.addAction(action)
            action.setChecked(slot.index == 0)
            action.triggered.connect(partial(self._set_frame_step_source, slot.index))
            step_menu.addAction(action)
            slot.step_action = action

        mhelp = self.menuBar().addMenu("&Help")
        hh = QtGui.QAction("Shortcuts / Help", self)
        hh.setShortcuts(
            [
                QtGui.QKeySequence(QtCore.Qt.Key.Key_F1),
                QtGui.QKeySequence("?"),
            ]
        )
        hh.triggered.connect(self._show_help)
        mhelp.addAction(hh)

    def _adjust_scroll_speed(self):
        try:
            val, ok = QtWidgets.QInputDialog.getDouble(
                self,
                "Adjust Smooth Scroll Speed",
                "Fraction of window per wheel step (0.001 - 1.0):",
                float(self.smooth_scroll_fraction),
                0.001,
                1.0,
                3,
            )
        except Exception:
            val, ok = (self.smooth_scroll_fraction, False)
        if ok:
            self.smooth_scroll_fraction = float(max(0.001, min(1.0, val)))

    def _adjust_playback_speed(self):
        try:
            val, ok = QtWidgets.QInputDialog.getDouble(
                self,
                "Set Playback Speed",
                "Playback speed (0.25x - 4.0x, step 0.25):",
                float(self.playback_speed),
                0.25,
                4.0,
                2,
            )
        except Exception:
            val, ok = (self.playback_speed, False)
        if ok:
            # Quantize to nearest 0.25
            q = round(float(val) / 0.25) * 0.25
            self.playback_speed = float(max(0.25, min(4.0, q)))

    def _adjust_raster_event_height(self):
        """Adjust the height of raster event lines (distance from center in each direction)."""
        try:
            val, ok = QtWidgets.QInputDialog.getDouble(
                self,
                "Raster Event Height",
                "Event height (0.1 - 0.5, distance from row center):",
                float(self.raster_event_height),
                0.1,
                0.5,
                2,
            )
        except Exception:
            val, ok = (self.raster_event_height, False)
        if ok:
            self.raster_event_height = float(max(0.1, min(0.5, val)))
            self._refresh_raster_plots()

    def _adjust_raster_event_thickness(self):
        """Adjust the pen width of raster event lines (in pixels)."""
        try:
            val, ok = QtWidgets.QInputDialog.getInt(
                self,
                "Raster Event Thickness",
                "Event line thickness (pixels, 1-10):",
                int(self.raster_event_thickness),
                1,
                10,
                1,
            )
        except Exception:
            val, ok = (self.raster_event_thickness, False)
        if ok:
            self.raster_event_thickness = int(max(1, min(10, val)))
            self._refresh_raster_pen_cache()
            self._refresh_raster_plots()

    def _toggle_proportional_raster(self, checked: bool):
        """Toggle proportional raster plot sizing on/off."""
        self.scale_raster_proportionally = checked
        self._apply_trace_visibility()  # Rebuilds layout with new sizing

    def _focused_height_factor_slot(self):
        """Resolve the hovered subplot to its (factors_list, idx, label) triple.

        Returns None if no stacked subplot is hovered.
        """
        if self.hovered_plot is None:
            return None
        targets = (
            (self.plots, self.plot_height_factors, "trace"),
            (self.dense_plots, self.dense_height_factors, "dense"),
            (self.raster_plots, self.raster_height_factors, "raster"),
            (self.heatmap_plots, self.heatmap_height_factors, "heatmap"),
        )
        for plots, factors, label in targets:
            for i, plt in enumerate(plots):
                if plt is self.hovered_plot:
                    while len(factors) <= i:
                        factors.append(1.0)
                    return factors, i, label
        return None

    def _adjust_focused_subplot_height(self, multiplier: float):
        """Scale the focused subplot's height factor (clamped to [0.1, 20])."""
        slot = self._focused_height_factor_slot()
        if slot is None:
            self._update_status("Hover a subplot to resize it")
            return
        factors, idx, label = slot
        new_factor = max(0.1, min(20.0, factors[idx] * multiplier))
        factors[idx] = new_factor
        self._apply_trace_visibility()
        self._update_status(f"{label}[{idx}] height: {new_factor:.2f}")

    def _reset_focused_subplot_height(self):
        """Reset the focused subplot's height factor to 1.0."""
        slot = self._focused_height_factor_slot()
        if slot is None:
            self._update_status("Hover a subplot to resize it")
            return
        factors, idx, label = slot
        factors[idx] = 1.0
        self._apply_trace_visibility()
        self._update_status(f"{label}[{idx}] height: 1.00")

    def _toggle_proportional_heatmap(self, checked: bool):
        """Toggle proportional heatmap plot sizing on/off."""
        self.scale_heatmap_proportionally = checked
        self._apply_trace_visibility()  # Rebuilds layout with new sizing

    def _adjust_raster_brightness(self):
        """Show a dialog to adjust raster event brightness."""
        if not self.raster_series:
            QtWidgets.QMessageBox.information(
                self, "Raster Brightness", "No raster plots loaded."
            )
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Adjust Raster Brightness")
        lay = QtWidgets.QVBoxLayout(dlg)

        label = QtWidgets.QLabel(
            "Adjust brightness multiplier for raster events.\n"
            "1.0 = default, <1.0 = dimmer, >1.0 = brighter"
        )
        lay.addWidget(label)

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(20, 300)  # 0.2 to 3.0
        slider.setValue(int(self.raster_brightness * 100))
        lay.addWidget(slider)

        val_label = QtWidgets.QLabel(f"{self.raster_brightness:.2f}")
        lay.addWidget(val_label)

        def on_change(val):
            brightness = val / 100.0
            val_label.setText(f"{brightness:.2f}")
            self.raster_brightness = brightness
            self._refresh_raster_plots()

        slider.valueChanged.connect(on_change)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)

        dlg.exec()

    def _adjust_sample_marker_properties(self):
        """View-menu handler: live-edit color, size, and opacity per sample-marker set."""
        if not self.sample_markers:
            QtWidgets.QMessageBox.information(
                self,
                "Sample Marker Properties",
                "No sample markers loaded.\n\n"
                "Pass bool_event_arrays= to view() to add markers.",
            )
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Adjust Sample Marker Properties")
        outer = QtWidgets.QVBoxLayout(dlg)

        intro = QtWidgets.QLabel(
            "Changes apply live. Close the dialog when done."
        )
        outer.addWidget(intro)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(12)
        for col, header in enumerate(["#", "Symbol", "Color", "Size", "Opacity"]):
            lbl = QtWidgets.QLabel(header)
            lbl.setStyleSheet("font-weight: bold;")
            grid.addWidget(lbl, 0, col)

        def _make_color_button(marker_idx: int) -> QtWidgets.QPushButton:
            marker = self.sample_markers[marker_idx]
            btn = QtWidgets.QPushButton()
            btn.setFixedWidth(60)

            def _refresh_swatch():
                qc = pg.mkColor(self.sample_markers[marker_idx].color)
                btn.setStyleSheet(
                    f"background-color: {qc.name()}; border: 1px solid #888;"
                )

            def _on_click():
                cur = pg.mkColor(self.sample_markers[marker_idx].color)
                qc0 = QtGui.QColor(cur.red(), cur.green(), cur.blue())
                picked = QtWidgets.QColorDialog.getColor(
                    qc0, dlg, f"Marker {marker_idx} Color"
                )
                if picked.isValid():
                    self.sample_markers[marker_idx].color = (
                        picked.red(), picked.green(), picked.blue()
                    )
                    _refresh_swatch()
                    self._apply_sample_marker_style(marker_idx)

            btn.clicked.connect(_on_click)
            _refresh_swatch()
            return btn

        for li, marker in enumerate(self.sample_markers):
            row = li + 1
            grid.addWidget(QtWidgets.QLabel(str(li)), row, 0)

            symbol_lbl = QtWidgets.QLabel(marker.marker)
            symbol_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(symbol_lbl, row, 1)

            grid.addWidget(_make_color_button(li), row, 2)

            size_spin = QtWidgets.QDoubleSpinBox()
            size_spin.setRange(2.0, 40.0)
            size_spin.setSingleStep(0.5)
            size_spin.setDecimals(1)
            size_spin.setValue(float(marker.size))
            size_spin.valueChanged.connect(
                lambda v, idx=li: (
                    setattr(self.sample_markers[idx], "size", float(v)),
                    self._apply_sample_marker_style(idx),
                )
            )
            grid.addWidget(size_spin, row, 3)

            alpha_row = QtWidgets.QHBoxLayout()
            alpha_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            alpha_slider.setRange(0, 255)
            alpha_slider.setValue(int(marker.alpha))
            alpha_lbl = QtWidgets.QLabel(f"{int(marker.alpha)}")
            alpha_lbl.setFixedWidth(32)
            alpha_slider.valueChanged.connect(
                lambda v, idx=li, lbl=alpha_lbl: (
                    setattr(self.sample_markers[idx], "alpha", int(v)),
                    lbl.setText(str(int(v))),
                    self._apply_sample_marker_style(idx),
                )
            )
            alpha_row.addWidget(alpha_slider)
            alpha_row.addWidget(alpha_lbl)
            wrap = QtWidgets.QWidget()
            wrap.setLayout(alpha_row)
            grid.addWidget(wrap, row, 4)

        outer.addLayout(grid)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        outer.addWidget(btns)

        dlg.exec()

    def _adjust_global_events_style(self):
        """View-menu handler: live-edit color, line style, width, and alpha
        per global-events class."""
        if self.global_events is None or not self._resolved_event_styles:
            QtWidgets.QMessageBox.information(
                self,
                "Style Global Events",
                "No global event markers loaded.\n\n"
                "Pass global_events=GlobalEventsConfig(...) to view() to add markers.",
            )
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Style Global Events")
        outer = QtWidgets.QVBoxLayout(dlg)

        intro = QtWidgets.QLabel("Changes apply live. Close the dialog when done.")
        outer.addWidget(intro)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(12)
        for col, header in enumerate(["Class", "Color", "Style", "Width", "Alpha"]):
            lbl = QtWidgets.QLabel(header)
            lbl.setStyleSheet("font-weight: bold;")
            grid.addWidget(lbl, 0, col)

        def _make_color_button(class_val) -> QtWidgets.QPushButton:
            btn = QtWidgets.QPushButton()
            btn.setFixedWidth(60)

            def _refresh_swatch():
                r, g, b = _parse_global_event_color(
                    self._resolved_event_styles[class_val]["line_color"]
                )
                btn.setStyleSheet(
                    f"background-color: rgb({r},{g},{b}); border: 1px solid #888;"
                )

            def _on_click():
                cur = _parse_global_event_color(
                    self._resolved_event_styles[class_val]["line_color"]
                )
                picked = QtWidgets.QColorDialog.getColor(
                    QtGui.QColor(*cur), dlg, f"Color for {class_val!r}"
                )
                if picked.isValid():
                    self._resolved_event_styles[class_val]["line_color"] = (
                        picked.red(), picked.green(), picked.blue()
                    )
                    _refresh_swatch()
                    self._apply_global_event_class_style(class_val)

            btn.clicked.connect(_on_click)
            _refresh_swatch()
            return btn

        for row_idx, (class_val, style) in enumerate(
            self._resolved_event_styles.items(), start=1,
        ):
            label_text = "(all events)" if class_val is None else repr(class_val)
            grid.addWidget(QtWidgets.QLabel(label_text), row_idx, 0)

            grid.addWidget(_make_color_button(class_val), row_idx, 1)

            style_combo = QtWidgets.QComboBox()
            style_combo.addItems(_GLOBAL_EVENT_STYLE_ORDER)
            style_combo.setCurrentText(style["line_style"])

            def _on_style(text, cv=class_val):
                self._resolved_event_styles[cv]["line_style"] = text
                self._apply_global_event_class_style(cv)

            style_combo.currentTextChanged.connect(_on_style)
            grid.addWidget(style_combo, row_idx, 2)

            width_spin = QtWidgets.QDoubleSpinBox()
            width_spin.setRange(0.5, 8.0)
            width_spin.setSingleStep(0.25)
            width_spin.setDecimals(2)
            width_spin.setValue(float(style["line_width"]))

            def _on_width(v, cv=class_val):
                self._resolved_event_styles[cv]["line_width"] = float(v)
                self._apply_global_event_class_style(cv)

            width_spin.valueChanged.connect(_on_width)
            grid.addWidget(width_spin, row_idx, 3)

            alpha_row = QtWidgets.QHBoxLayout()
            alpha_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            alpha_slider.setRange(0, 255)
            alpha_slider.setValue(int(style["line_alpha"]))
            alpha_lbl = QtWidgets.QLabel(f"{int(style['line_alpha'])}")
            alpha_lbl.setFixedWidth(32)

            def _on_alpha(v, cv=class_val, lbl=alpha_lbl):
                self._resolved_event_styles[cv]["line_alpha"] = int(v)
                lbl.setText(str(int(v)))
                self._apply_global_event_class_style(cv)

            alpha_slider.valueChanged.connect(_on_alpha)
            alpha_row.addWidget(alpha_slider)
            alpha_row.addWidget(alpha_lbl)
            wrap = QtWidgets.QWidget()
            wrap.setLayout(alpha_row)
            grid.addWidget(wrap, row_idx, 4)

        outer.addLayout(grid)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        outer.addWidget(btns)

        dlg.exec()

    def _interval_label_brush_color(self, name: str) -> tuple[int, int, int, int]:
        """Return the effective RGBA for a label, scaled by interval_label_alpha_multiplier."""
        r, g, b, a = self.label_colors.get(name, (150, 150, 150, 80))
        a_scaled = int(round(a * float(self.interval_label_alpha_multiplier)))
        a_scaled = max(0, min(255, a_scaled))
        return (r, g, b, a_scaled)

    def _refresh_interval_label_alpha(self) -> None:
        """Re-apply current interval_label_alpha_multiplier to all existing label regions."""
        # Visuals are keyed by row_id (an int), so the label name lives on the
        # bundle / drawn-state — not in the dict key.
        for bundle in self._interval_label_visuals.values():
            color = self._interval_label_brush_color(bundle.label)
            for regions in (
                bundle.plot_regions,
                bundle.dense_regions,
                bundle.raster_regions,
                bundle.heatmap_regions,
            ):
                for _i, reg in regions:
                    self._set_region_color(reg, color)
        for key, region in self._hypnogram_interval_label_visuals.items():
            drawn = self._hypnogram_interval_label_drawn.get(key)
            name = drawn[2] if drawn is not None else ""
            self._set_region_color(region, self._interval_label_brush_color(name))

    def _adjust_interval_label_alpha(self):
        """Show a dialog to adjust label overlay alpha (transparency)."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Adjust Interval Label Alpha")
        lay = QtWidgets.QVBoxLayout(dlg)

        label = QtWidgets.QLabel(
            "Adjust alpha multiplier for label overlay regions.\n"
            "1.00 = default opacity, 0.00 = fully transparent."
        )
        lay.addWidget(label)

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(0, 100)  # 0.00 to 1.00
        slider.setValue(int(round(self.interval_label_alpha_multiplier * 100)))
        lay.addWidget(slider)

        val_label = QtWidgets.QLabel(f"{self.interval_label_alpha_multiplier:.2f}")
        lay.addWidget(val_label)

        def on_change(val):
            mult = val / 100.0
            val_label.setText(f"{mult:.2f}")
            self.interval_label_alpha_multiplier = mult
            self._refresh_interval_label_alpha()

        slider.valueChanged.connect(on_change)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)

        dlg.exec()

    def _edit_epoch_note(self):
        """Edit the note (and any extra columns) for the current epoch."""
        if len(self.interval_label_set) == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "No Epochs",
                "No scored epochs exist. Please score an epoch first.",
            )
            return

        # Find the epoch at cursor position
        target_row = self.interval_label_set.at_time(self.cursor_time)

        # Otherwise, the most recently created surviving epoch.
        if target_row is None and self.interval_label_set.history:
            for rid in reversed(self.interval_label_set.history):
                row = self.interval_label_set.row_for_id(rid)
                if row is not None:
                    target_row = row
                    break

        if target_row is None:
            target_row = self.interval_label_set.row_at_index(len(self.interval_label_set) - 1)

        editable_cols: list[str] = []
        if self.interval_label_set.schema.note_col:
            editable_cols.append(self.interval_label_set.schema.note_col)
        editable_cols.extend(self.interval_label_set.schema.extra_cols)
        if not editable_cols:
            QtWidgets.QMessageBox.information(
                self,
                "No editable metadata",
                "This IntervalLabelSchema declares no note column or extra columns.",
            )
            return

        title = "Edit epoch metadata" if len(editable_cols) > 1 else "Epoch Note"
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setMinimumWidth(420)
        layout = QtWidgets.QVBoxLayout(dlg)

        info_label = QtWidgets.QLabel(
            f"Epoch: {target_row.label}\n"
            f"Time: {target_row.start:.3f}s - {target_row.end:.3f}s"
        )
        layout.addWidget(info_label)

        editors: dict[str, QtWidgets.QPlainTextEdit | QtWidgets.QLineEdit] = {}
        for col in editable_cols:
            row_label = QtWidgets.QLabel(col)
            layout.addWidget(row_label)
            if col == self.interval_label_set.schema.note_col:
                editor = QtWidgets.QPlainTextEdit()
                value = target_row.note
                editor.setPlainText(value)
                editor.setMinimumHeight(100)
            else:
                editor = QtWidgets.QLineEdit()
                value = target_row.extras.get(col)
                editor.setText("" if value is None else str(value))
            editors[col] = editor
            layout.addWidget(editor)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() == QtWidgets.QDialog.Accepted:
            for col, editor in editors.items():
                if isinstance(editor, QtWidgets.QPlainTextEdit):
                    text = editor.toPlainText().strip()
                else:
                    text = editor.text().strip()
                if col == self.interval_label_set.schema.note_col:
                    self.interval_label_set.set_note(target_row.row_id, text)
                else:
                    self.interval_label_set.update_cell(
                        target_row.row_id, col, text or None
                    )
            self._update_status()
            self._refresh_interval_label_summary()

    def _show_jump_to_epochs_dialog(self):
        """Show a table of all epochs for navigation and filtering."""
        if len(self.interval_label_set) == 0:
            QtWidgets.QMessageBox.information(
                self, "Jump to Epochs", "No scored epochs to display."
            )
            return

        schema = self.interval_label_set.schema
        # Columns: Start (s), End (s), State [hotkeys], (Note?), then extras
        column_specs: list[tuple[str, str]] = [
            ("Start (s)", "__start"),
            ("End (s)", "__end"),
            ("State", "__state"),
        ]
        if schema.note_col:
            column_specs.append(("Notes", schema.note_col))
        for c in schema.extra_cols:
            column_specs.append((c, c))

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Jump to Epochs")
        dlg.setMinimumWidth(700)
        dlg.setMinimumHeight(500)
        layout = QtWidgets.QVBoxLayout(dlg)

        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.addWidget(QtWidgets.QLabel("Filter by State:"))
        state_filter = QtWidgets.QComboBox()
        state_filter.addItem("(All)")
        unique_states = sorted({row.label for row in self.interval_label_set})
        for state in unique_states:
            state_filter.addItem(state)
        filter_layout.addWidget(state_filter)

        filter_layout.addWidget(QtWidgets.QLabel("Filter Notes:"))
        notes_filter = QtWidgets.QLineEdit()
        notes_filter.setPlaceholderText("Enter text to search in notes/extras...")
        filter_layout.addWidget(notes_filter, 1)

        layout.addLayout(filter_layout)

        table = QtWidgets.QTableWidget()
        table.setColumnCount(len(column_specs))
        table.setHorizontalHeaderLabels([h for h, _ in column_specs])
        table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(table, 1)

        def cell_text_for(row, key: str) -> str:
            if key == "__start":
                return f"{row.start:.3f}"
            if key == "__end":
                return f"{row.end:.3f}"
            if key == "__state":
                return self.state_config.state_with_hotkeys(row.label)
            if schema.note_col and key == schema.note_col:
                return row.note
            v = row.extras.get(key)
            return "" if v is None else str(v)

        def populate_table():
            state_val = state_filter.currentText()
            search_val = notes_filter.text().strip().lower()

            table.setRowCount(0)
            r = 0
            for row in self.interval_label_set:
                if state_val != "(All)" and row.label != state_val:
                    continue
                # Build all cell texts so we can filter against any of them
                cells = [cell_text_for(row, key) for _, key in column_specs]
                if search_val and not any(search_val in c.lower() for c in cells):
                    continue
                table.insertRow(r)
                for c_idx, text in enumerate(cells):
                    table.setItem(r, c_idx, QtWidgets.QTableWidgetItem(text))
                table.item(r, 0).setData(
                    QtCore.Qt.ItemDataRole.UserRole, int(row.row_id)
                )
                r += 1

            table.resizeColumnsToContents()

        populate_table()

        state_filter.currentTextChanged.connect(lambda: populate_table())
        notes_filter.textChanged.connect(lambda: populate_table())

        def on_double_click(r, _col):
            item = table.item(r, 0)
            if not item:
                return
            row_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
            row = self.interval_label_set.row_for_id(int(row_id)) if row_id is not None else None
            if row is None:
                return
            center = (row.start + row.end) / 2.0
            new_start = center - self.window_len / 2.0
            new_start = clamp(
                new_start,
                self.t_global_min,
                max(self.t_global_min, self.t_global_max - self.window_len),
            )
            self.window_start = new_start
            self.cursor_time = center
            self._apply_x_range()
            self._update_nav_slider_from_window()

        table.cellDoubleClicked.connect(on_double_click)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dlg.reject)
        layout.addWidget(close_btn)

        dlg.exec()

    def _jump_to_epoch_by_offset(self, direction: int) -> None:
        """Jump to the next (+1) or previous (-1) labelled epoch relative to cursor.

        Centers the view on the epoch without changing window size,
        using the same logic as the Jump to Epochs dialog.
        """
        if len(self.interval_label_set) == 0:
            return

        cursor = self.cursor_time

        if direction > 0:
            # Find first epoch whose center is strictly after cursor
            for row in self.interval_label_set:
                center = (row.start + row.end) / 2.0
                if center > cursor:
                    break
            else:
                return  # no epoch found ahead
        else:
            # Find last epoch whose center is strictly before cursor
            found = None
            for r in self.interval_label_set:
                center = (r.start + r.end) / 2.0
                if center < cursor:
                    found = r
                else:
                    break
            if found is None:
                return
            row = found
            center = (row.start + row.end) / 2.0

        # Center view on the epoch (same as Jump to Epochs dialog)
        new_start = center - self.window_len / 2.0
        new_start = clamp(
            new_start,
            self.t_global_min,
            max(self.t_global_min, self.t_global_max - self.window_len),
        )
        self.window_start = new_start
        self.cursor_time = center
        self._apply_x_range()
        self._update_nav_slider_from_window()

    def _show_subplot_control_dialog(self):
        """Show a comprehensive dialog to control subplot heights, visibility, and order."""
        n_ts_plots = (
            len(self.overlay_groups)
            if self.overlay_mode
            else len(self.series)
        )
        total_subplots = (
            n_ts_plots
            + len(self.dense_groups)
            + len(self.raster_series)
            + len(self.heatmap_series)
        )
        if total_subplots == 0:
            QtWidgets.QMessageBox.information(
                self, "Subplot Control", "No subplots loaded."
            )
            return

        # Ensure all control lists are properly sized
        while len(self.plot_height_factors) < n_ts_plots:
            self.plot_height_factors.append(1.0)
        while len(self.raster_height_factors) < len(self.raster_series):
            self.raster_height_factors.append(1.0)
        while len(self.dense_height_factors) < len(self.dense_groups):
            self.dense_height_factors.append(1.0)
        while len(self.heatmap_height_factors) < len(self.heatmap_series):
            self.heatmap_height_factors.append(1.0)
        if not hasattr(self, "trace_visible") or len(self.trace_visible) != n_ts_plots:
            self.trace_visible = [True] * n_ts_plots
        while len(self.raster_visible) < len(self.raster_series):
            self.raster_visible.append(True)
        while len(self.dense_visible) < len(self.dense_groups):
            self.dense_visible.append(True)
        while len(self.heatmap_visible) < len(self.heatmap_series):
            self.heatmap_visible.append(True)

        # Initialize subplot order if not set
        if self.subplot_order is None:
            self.subplot_order = []
            for i in range(n_ts_plots):
                self.subplot_order.append(("ts", i))
            for i in range(len(self.dense_groups)):
                self.subplot_order.append(("dense", i))
            for i in range(len(self.raster_series)):
                self.subplot_order.append(("raster", i))
            for i in range(len(self.heatmap_series)):
                self.subplot_order.append(("heatmap", i))

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Subplot Control Board")
        dlg.setMinimumWidth(550)
        dlg.setMinimumHeight(400)
        main_lay = QtWidgets.QVBoxLayout(dlg)

        info_label = QtWidgets.QLabel(
            "Control subplot heights, visibility, and order.\n"
            "Drag rows to reorder. Check 'Hide' to hide a subplot."
        )
        main_lay.addWidget(info_label)

        # Create a list widget that supports drag-and-drop reordering
        list_widget = QtWidgets.QListWidget()
        list_widget.setDragDropMode(
            QtWidgets.QAbstractItemView.DragDropMode.InternalMove
        )
        list_widget.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        list_widget.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        list_widget.setMinimumHeight(250)
        main_lay.addWidget(list_widget, 1)

        # Store references to widgets for each row
        row_widgets = []  # List of dicts with plot info and widget references

        def create_row_widget(plot_type, idx):
            """Create a widget for a single row in the list."""
            if plot_type == "ts":
                if self.overlay_mode:
                    name = self.overlay_groups[idx].label
                else:
                    name = self.series[idx].name
                factor = self.plot_height_factors[idx]
                visible = self.trace_visible[idx]
                display_name = f"[TS] {name}"
            elif plot_type == "dense":
                group = self.dense_groups[idx]
                name = group.name
                factor = self.dense_height_factors[idx]
                visible = self.dense_visible[idx]
                n = len(group.series)
                display_name = f"[Dense/{n}] {name}"
            elif plot_type == "heatmap":
                asx = self.heatmap_series[idx]
                name = asx.name
                factor = self.heatmap_height_factors[idx]
                visible = self.heatmap_visible[idx]
                n = asx.Y.shape[0]
                display_name = f"[Heatmap/{n}] {name}"
            else:
                name = self.raster_series[idx].name
                factor = self.raster_height_factors[idx]
                visible = self.raster_visible[idx]
                display_name = f"[Raster] {name}"

            widget = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(widget)
            layout.setContentsMargins(4, 2, 4, 2)

            # Drag handle indicator
            drag_label = QtWidgets.QLabel("≡")
            drag_label.setStyleSheet("color: gray; font-size: 14px;")
            drag_label.setFixedWidth(20)
            layout.addWidget(drag_label)

            # Name label
            name_label = QtWidgets.QLabel(display_name)
            name_label.setMinimumWidth(120)
            name_label.setMaximumWidth(180)
            layout.addWidget(name_label)

            # Height slider
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(1, 2000)  # 0.01x to 20.0x (very wide range)
            slider.setValue(int(factor * 100))
            slider.setMinimumWidth(150)
            layout.addWidget(slider)

            # Value label
            val_label = QtWidgets.QLabel(f"{factor:.2f}x")
            val_label.setMinimumWidth(50)
            layout.addWidget(val_label)

            # Hide checkbox
            hide_check = QtWidgets.QCheckBox("Hide")
            hide_check.setChecked(not visible)
            layout.addWidget(hide_check)

            # Connect slider
            def on_slider_change(val):
                new_factor = val / 100.0
                if plot_type == "ts":
                    self.plot_height_factors[idx] = new_factor
                elif plot_type == "dense":
                    self.dense_height_factors[idx] = new_factor
                elif plot_type == "heatmap":
                    self.heatmap_height_factors[idx] = new_factor
                else:
                    self.raster_height_factors[idx] = new_factor
                val_label.setText(f"{new_factor:.2f}x")
                self._apply_trace_visibility()

            slider.valueChanged.connect(on_slider_change)

            # Connect hide checkbox
            def on_hide_changed(state):
                is_visible = state != QtCore.Qt.CheckState.Checked.value
                if plot_type == "ts":
                    self.trace_visible[idx] = is_visible
                elif plot_type == "dense":
                    self.dense_visible[idx] = is_visible
                elif plot_type == "heatmap":
                    self.heatmap_visible[idx] = is_visible
                else:
                    self.raster_visible[idx] = is_visible
                self._apply_trace_visibility()

            hide_check.stateChanged.connect(on_hide_changed)

            return {
                "widget": widget,
                "type": plot_type,
                "idx": idx,
                "slider": slider,
                "val_label": val_label,
                "hide_check": hide_check,
            }

        # Populate the list widget based on current order
        for plot_type, idx in self.subplot_order:
            # Validate the entry
            valid = False
            if plot_type == "ts" and idx < len(self.series):
                valid = True
            elif plot_type == "dense" and idx < len(self.dense_groups):
                valid = True
            elif plot_type == "raster" and idx < len(self.raster_series):
                valid = True
            elif plot_type == "heatmap" and idx < len(self.heatmap_series):
                valid = True
            if valid:
                row_data = create_row_widget(plot_type, idx)
                row_widgets.append(row_data)
                item = QtWidgets.QListWidgetItem()
                item.setSizeHint(row_data["widget"].sizeHint())
                item.setData(QtCore.Qt.ItemDataRole.UserRole, (plot_type, idx))
                list_widget.addItem(item)
                list_widget.setItemWidget(item, row_data["widget"])

        def update_order_from_list():
            """Update subplot_order based on current list widget order."""
            new_order = []
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                data = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if data:
                    new_order.append(data)
            self.subplot_order = new_order
            self._apply_trace_visibility()

        # Connect model changes to update order
        list_widget.model().rowsMoved.connect(lambda: update_order_from_list())

        # Button row
        btn_row = QtWidgets.QHBoxLayout()

        reset_heights_btn = QtWidgets.QPushButton("Reset Heights")

        def reset_heights():
            for rw in row_widgets:
                rw["slider"].blockSignals(True)
                rw["slider"].setValue(100)
                rw["val_label"].setText("1.00x")
                if rw["type"] == "ts":
                    self.plot_height_factors[rw["idx"]] = 1.0
                elif rw["type"] == "dense":
                    self.dense_height_factors[rw["idx"]] = 1.0
                elif rw["type"] == "heatmap":
                    self.heatmap_height_factors[rw["idx"]] = 1.0
                else:
                    self.raster_height_factors[rw["idx"]] = 1.0
                rw["slider"].blockSignals(False)
            self._apply_trace_visibility()

        reset_heights_btn.clicked.connect(reset_heights)
        btn_row.addWidget(reset_heights_btn)

        show_all_btn = QtWidgets.QPushButton("Show All")

        def show_all():
            for rw in row_widgets:
                rw["hide_check"].blockSignals(True)
                rw["hide_check"].setChecked(False)
                if rw["type"] == "ts":
                    self.trace_visible[rw["idx"]] = True
                elif rw["type"] == "dense":
                    self.dense_visible[rw["idx"]] = True
                elif rw["type"] == "heatmap":
                    self.heatmap_visible[rw["idx"]] = True
                else:
                    self.raster_visible[rw["idx"]] = True
                rw["hide_check"].blockSignals(False)
            self._apply_trace_visibility()

        show_all_btn.clicked.connect(show_all)
        btn_row.addWidget(show_all_btn)

        reset_order_btn = QtWidgets.QPushButton("Reset Order")

        def reset_order():
            # Rebuild the default order
            n_ts_reset = (
                len(self.overlay_groups)
                if self.overlay_mode
                else len(self.series)
            )
            self.subplot_order = []
            for i in range(n_ts_reset):
                self.subplot_order.append(("ts", i))
            for i in range(len(self.dense_groups)):
                self.subplot_order.append(("dense", i))
            for i in range(len(self.raster_series)):
                self.subplot_order.append(("raster", i))
            for i in range(len(self.heatmap_series)):
                self.subplot_order.append(("heatmap", i))
            # Close and reopen dialog to refresh
            dlg.accept()
            QtCore.QTimer.singleShot(50, self._show_subplot_control_dialog)

        reset_order_btn.clicked.connect(reset_order)
        btn_row.addWidget(reset_order_btn)

        btn_row.addStretch()
        main_lay.addLayout(btn_row)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        btns.accepted.connect(dlg.accept)
        main_lay.addWidget(btns)

        dlg.exec()

    def _apply_custom_plot_heights(self):
        """Apply custom height factors to all visible plots based on subplot_order.

        Two-pass: compute each visible row's natural ``(preferred, stretch)``
        first, then — when ``compact_heatmaps_to_fit`` is on and the natural
        total exceeds the plot-area viewport — scale every heatmap row's
        preferred height down by a single uniform ratio so the visible total
        matches the viewport (no scrollbar).  Lines and raster rows keep
        their natural heights so the shrink is borne entirely by the heatmaps.
        """
        try:
            layout = self.plot_area.ci.layout

            # Base height for calculations - allow very small heights
            BASE_HEIGHT = 100
            MIN_HEIGHT = 1  # Very small minimum to allow extreme shrinking

            # Get the ordered list of visible plots
            visible_plots = self._get_visible_subplot_order()

            # ---- Pass 1: compute natural (preferred, stretch) per row ------
            # Each entry: (plot_type, idx, plt, factor, preferred, stretch)
            specs: list[tuple] = []
            for plot_type, idx in visible_plots:
                if plot_type == "ts":
                    factor = (
                        self.plot_height_factors[idx]
                        if idx < len(self.plot_height_factors)
                        else 1.0
                    )
                    plt = self.plots[idx] if idx < len(self.plots) else None

                    preferred = max(MIN_HEIGHT, int(BASE_HEIGHT * factor))
                    stretch = max(
                        1, int(factor * 100)
                    )  # Scale stretch more aggressively

                elif plot_type == "dense":
                    factor = (
                        self.dense_height_factors[idx]
                        if idx < len(self.dense_height_factors)
                        else 1.0
                    )
                    plt = (
                        self.dense_plots[idx] if idx < len(self.dense_plots) else None
                    )
                    # Dense plots get more height by default
                    preferred = max(MIN_HEIGHT, int(BASE_HEIGHT * factor * 3))
                    stretch = max(1, int(factor * 300))

                elif plot_type == "heatmap":
                    factor = (
                        self.heatmap_height_factors[idx]
                        if idx < len(self.heatmap_height_factors)
                        else 1.0
                    )
                    plt = (
                        self.heatmap_plots[idx]
                        if idx < len(self.heatmap_plots)
                        else None
                    )
                    asx = (
                        self.heatmap_series[idx]
                        if idx < len(self.heatmap_series)
                        else None
                    )

                    if self.scale_heatmap_proportionally and asx is not None:
                        # Mirror raster proportional sizing: weight by row count.
                        BASE_HEIGHT_PER_ROW = 12
                        n_rows = max(1, asx.Y.shape[0])
                        preferred = max(
                            MIN_HEIGHT, int(n_rows * BASE_HEIGHT_PER_ROW * factor)
                        )
                        stretch = max(1, int(n_rows * factor * 10))
                    else:
                        # Heatmaps benefit from a bit more vertical room than line
                        # plots; mirror dense's larger default.
                        preferred = max(MIN_HEIGHT, int(BASE_HEIGHT * factor * 2))
                        stretch = max(1, int(factor * 200))

                else:  # raster
                    factor = (
                        self.raster_height_factors[idx]
                        if idx < len(self.raster_height_factors)
                        else 1.0
                    )
                    plt = (
                        self.raster_plots[idx] if idx < len(self.raster_plots) else None
                    )
                    ms = (
                        self.raster_series[idx]
                        if idx < len(self.raster_series)
                        else None
                    )

                    if self.scale_raster_proportionally and ms:
                        BASE_HEIGHT_PER_ROW = 12
                        preferred = max(
                            MIN_HEIGHT,
                            int(_raster_extent(ms) * BASE_HEIGHT_PER_ROW * factor),
                        )
                        stretch = max(1, int(_raster_extent(ms) * factor * 10))
                    else:
                        preferred = max(MIN_HEIGHT, int(BASE_HEIGHT * factor))
                        stretch = max(1, int(factor * 100))

                specs.append((plot_type, idx, plt, factor, preferred, stretch))

            # ---- Pass 1.5: optional uniform heatmap compression ------------
            if self.compact_heatmaps_to_fit and any(s[0] == "heatmap" for s in specs):
                viewport_h = self._available_plot_area_height()
                if viewport_h > 0:
                    natural_total = sum(s[4] for s in specs)
                    heatmap_total = sum(s[4] for s in specs if s[0] == "heatmap")
                    non_heatmap_total = natural_total - heatmap_total
                    target_heatmap_total = max(MIN_HEIGHT, viewport_h - non_heatmap_total)
                    if heatmap_total > target_heatmap_total and heatmap_total > 0:
                        compress = target_heatmap_total / heatmap_total
                        specs = [
                            (
                                pt, idx, plt, factor,
                                max(MIN_HEIGHT, int(round(pref * compress))) if pt == "heatmap" else pref,
                                stretch,
                            )
                            for (pt, idx, plt, factor, pref, stretch) in specs
                        ]

            # ---- Pass 2: apply to layout + per-plot axis tweaks -------------
            for row, (plot_type, idx, plt, factor, preferred, stretch) in enumerate(specs):
                if plt:
                    is_raster_label = (plot_type == "raster")
                    self._configure_plot_for_height(
                        plt, factor, is_raster=is_raster_label
                    )
                layout.setRowPreferredHeight(row, preferred)
                layout.setRowMinimumHeight(row, MIN_HEIGHT)
                layout.setRowStretchFactor(row, stretch)

            # Re-assert the uniform left-axis width now that heights/visibility
            # changed, so widths never diverge between this pass and the next
            # deferred _align_left_axes.
            self._align_left_axes()

        except Exception:
            import traceback

            traceback.print_exc()

    def _available_plot_area_height(self) -> int:
        """Return the height (px) the plot-area viewport can display without scrolling."""
        try:
            vp = self.plot_area.viewport()
            h = int(vp.height()) if vp is not None else 0
            return max(0, h)
        except Exception:
            return 0

    def _toggle_compact_heatmaps_to_fit(self, checked: bool) -> None:
        """View-menu handler for the 'Compact Heatmap Plots to Fit Screen' toggle."""
        self.compact_heatmaps_to_fit = bool(checked)
        self._apply_custom_plot_heights()

    def _configure_plot_for_height(self, plt, factor, is_raster=False):
        """Configure plot axis tick-label visibility based on height factor.

        Left-axis *width* is owned solely by :meth:`_align_left_axes` (one
        uniform width for every subplot), so this method must not touch it —
        otherwise it would re-introduce divergent widths and break spine
        alignment. A very small plot keeps the shared gutter with a blank label
        area, which stays aligned.
        """
        try:
            # For very small plots (below 0.2x), hide axis labels to save space.
            if factor < 0.2:
                plt.getAxis("left").setStyle(showValues=False)
                plt.setLabel("left", "")
            else:
                plt.getAxis("left").setStyle(showValues=True)
        except Exception:
            pass

    def _get_visible_subplot_order(self):
        """Get the list of visible subplots in their current order."""
        # Ensure visibility lists are properly sized
        n_ts = (
            len(self.overlay_groups)
            if self.overlay_mode
            else len(self.series)
        )
        if not hasattr(self, "trace_visible") or len(self.trace_visible) != n_ts:
            self.trace_visible = [True] * n_ts
        while len(self.raster_visible) < len(self.raster_series):
            self.raster_visible.append(True)
        while len(self.dense_visible) < len(self.dense_groups):
            self.dense_visible.append(True)
        while len(self.heatmap_visible) < len(self.heatmap_series):
            self.heatmap_visible.append(True)

        # Use subplot_order if set, otherwise default order
        if self.subplot_order:
            order = self.subplot_order
        else:
            order = [("ts", i) for i in range(n_ts)]
            order += [("dense", i) for i in range(len(self.dense_groups))]
            order += [("raster", i) for i in range(len(self.raster_series))]
            order += [("heatmap", i) for i in range(len(self.heatmap_series))]

        # Filter to only visible plots
        visible = []
        for plot_type, idx in order:
            if plot_type == "ts":
                if idx < len(self.trace_visible) and self.trace_visible[idx]:
                    visible.append((plot_type, idx))
            elif plot_type == "dense":
                if idx < len(self.dense_visible) and self.dense_visible[idx]:
                    visible.append((plot_type, idx))
            elif plot_type == "heatmap":
                if idx < len(self.heatmap_visible) and self.heatmap_visible[idx]:
                    visible.append((plot_type, idx))
            else:  # raster
                if idx < len(self.raster_visible) and self.raster_visible[idx]:
                    visible.append((plot_type, idx))
        return visible

    # ---------- Data ----------
    def _load_series_from_dir(self, folder):
        self._stop_playback_if_playing()
        pairs = []
        for tpath in glob.glob(os.path.join(folder, "*_t.npy")):
            name = os.path.basename(tpath)[:-6]
            ypath = os.path.join(folder, f"{name}_y.npy")
            if os.path.exists(ypath):
                pairs.append((name, tpath, ypath))
        if not pairs:
            QtWidgets.QMessageBox.warning(
                self, "No data", "No *_t.npy / *_y.npy pairs found."
            )
            return

        series = []
        for name, tpath, ypath in sorted(pairs):
            try:
                t = np.load(tpath).astype(float)
                y = np.load(ypath).astype(float)
                if t.ndim != 1 or y.ndim != 1 or len(t) != len(y):
                    raise ValueError("t and y must be 1-D & equal length")
                series.append(Series(name, t, y))
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Load error", f"{name}: {e}")
        if series:
            self.set_series(series)

    def _load_raster_from_dir(self, folder):
        """Load raster series from a directory with *_t.npy, *_y.npy, and optional *_a.npy files."""
        pairs = []
        for tpath in sorted(glob.glob(os.path.join(folder, "*_t.npy"))):
            base = os.path.basename(tpath)[:-6]  # strip "_t.npy"
            ypath = os.path.join(folder, f"{base}_y.npy")
            if not os.path.exists(ypath):
                continue
            apath = os.path.join(folder, f"{base}_a.npy")
            pairs.append((tpath, ypath, apath if os.path.exists(apath) else ""))

        if not pairs:
            QtWidgets.QMessageBox.warning(
                self,
                "No raster data",
                "No *_t.npy / *_y.npy pairs found in the selected directory.",
            )
            return

        ts_paths = [p[0] for p in pairs]
        yv_paths = [p[1] for p in pairs]
        al_paths = [p[2] for p in pairs]
        self._load_raster_data(ts_paths, yv_paths, al_paths, colors=None)

    def _on_load_raster_data(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder with raster *_t.npy and *_y.npy files"
        )
        if folder:
            self._load_raster_from_dir(folder)

    def _load_series_from_files(self, files, colors=None):
        """Load series from an explicit ordered list of *_t.npy / *_y.npy files.

        The display order (top to bottom) follows the order in which distinct
        base names first appear in the provided list. A "base name" is the
        filename without the trailing "_t.npy" or "_y.npy".
        """
        self._stop_playback_if_playing()

        if not files:
            QtWidgets.QMessageBox.warning(self, "No data", "No files provided.")
            return

        # Build mapping base_name -> { 't': path or None, 'y': path or None }
        series_map = {}
        order = []  # first-seen order of base names
        first_index = {}  # base name -> first index in files list

        def base_for(path: str):
            fn = os.path.basename(path)
            if fn.endswith("_t.npy"):
                return fn[:-6], "t"
            if fn.endswith("_y.npy"):
                return fn[:-6], "y"
            return None, None

        for idx, p in enumerate(files):
            if not p:
                continue
            b, kind = base_for(p)
            if b is None:
                QtWidgets.QMessageBox.warning(
                    self, "Skip", f"Not a *_t.npy or *_y.npy file: {p}"
                )
                continue
            if b not in series_map:
                series_map[b] = {"t": None, "y": None}
                order.append(b)
                first_index[b] = idx
            series_map[b][kind] = p

        # Assemble in the order seen; require both t and y
        series = []
        series_colors = []
        for b in order:
            paths = series_map.get(b, {})
            tpath, ypath = paths.get("t"), paths.get("y")
            if not tpath or not ypath:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Missing pair",
                    f"Skipping '{b}': need both {b}_t.npy and {b}_y.npy",
                )
                continue
            if not (os.path.exists(tpath) and os.path.exists(ypath)):
                QtWidgets.QMessageBox.warning(
                    self, "File not found", f"Missing files for '{b}'."
                )
                continue
            try:
                t = np.load(tpath).astype(float)
                y = np.load(ypath).astype(float)
                if t.ndim != 1 or y.ndim != 1 or len(t) != len(y):
                    raise ValueError("t and y must be 1-D & equal length")
                series.append(Series(b, t, y))
                series_colors.append(None)  # placeholder
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Load error", f"{b}: {e}")

        if not series:
            QtWidgets.QMessageBox.warning(
                self, "No data", "No valid *_t.npy / *_y.npy pairs in list."
            )
            return

        # Map provided colors to series order
        def _parse_color(cs: str):
            s = (cs or "").strip()
            try:
                if s.startswith("#"):
                    s = s[1:]
                    if len(s) in (6, 8):
                        r = int(s[0:2], 16)
                        g = int(s[2:4], 16)
                        b = int(s[4:6], 16)
                        a = int(s[6:8], 16) if len(s) == 8 else 255
                        return (r, g, b, a)
                if s.lower().startswith("0x"):
                    v = int(s, 16)
                    r = (v >> 16) & 0xFF
                    g = (v >> 8) & 0xFF
                    b = v & 0xFF
                    return (r, g, b, 255)
                if "," in s:
                    parts = [int(x.strip()) for x in s.split(",") if x.strip()]
                    if len(parts) == 3:
                        return (parts[0], parts[1], parts[2], 255)
                    if len(parts) >= 4:
                        return (parts[0], parts[1], parts[2], parts[3])
            except Exception:
                pass
            return None

        mapped_colors = None
        if colors:
            # If colors count matches series count, map 1:1
            if len(colors) == len(series):
                mapped_colors = [
                    (_parse_color(c) or (255, 255, 255, 255)) for c in colors
                ]
            # If colors matches files count, use first occurrence index mapping
            elif len(colors) == len(files):
                mapped_colors = []
                for b in order:
                    ci = first_index.get(b, 0)
                    col = _parse_color(colors[ci]) or (255, 255, 255, 255)
                    mapped_colors.append(col)
            else:
                QtWidgets.QMessageBox.information(
                    self,
                    "Colors",
                    (
                        f"Ignoring --colors: count {len(colors)} doesn't match series ({len(series)}) "
                        f"or file count ({len(files)})."
                    ),
                )

        # Store colors aligned with series; default to white if not provided
        self.series_colors = mapped_colors or [(255, 255, 255, 255)] * len(series)
        self.set_series(series)

    def _on_load_time_series(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder with *_t.npy and *_y.npy"
        )
        if folder:
            self._load_series_from_dir(folder)

    def set_series(self, series_list, colors=None):
        self.series = series_list

        # Assign per-trace colours (RGBA tuples or default white).
        if colors and len(colors) == len(series_list):
            parsed = []
            for c in colors:
                if isinstance(c, str):
                    c = c.strip().lstrip("#")
                    try:
                        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
                        a = int(c[6:8], 16) if len(c) == 8 else 255
                        parsed.append((r, g, b, a))
                    except Exception:
                        parsed.append((255, 255, 255, 255))
                elif isinstance(c, (tuple, list)) and len(c) >= 3:
                    parsed.append(tuple(c[:4]) if len(c) >= 4 else (*c[:3], 255))
                else:
                    parsed.append((255, 255, 255, 255))
            self.series_colors = parsed
        else:
            self.series_colors = [(255, 255, 255, 255)] * len(series_list)

        self._clear_all_interval_label_visuals()
        self.plot_area.clear()
        # New plot set: re-fit the uniform left-axis width from scratch.
        self._left_axis_width = None
        self.plots.clear()
        self.curves.clear()
        self.plot_bottom_spines.clear()
        self.sample_marker_scatters.clear()
        self.overlay_curve_items.clear()
        self.plot_cur_lines.clear()
        self.plot_sel_regions.clear()
        self.dense_plots.clear()
        self.dense_curves.clear()
        self.dense_cur_lines.clear()
        self.dense_sel_regions.clear()
        self.dense_interval_label_regions.clear()
        self.dense_vscrollbars.clear()
        self.dense_vscroll_proxies.clear()
        self._dense_vscroll_inverted.clear()
        self.raster_plots.clear()
        self.raster_items.clear()
        self.raster_cur_lines.clear()
        self.raster_sel_regions.clear()
        self.raster_separator_lines.clear()
        self._raster_line_items.clear()
        self._raster_pens.clear()
        self.heatmap_plots.clear()
        self.heatmap_image_items.clear()
        self.heatmap_cur_lines.clear()
        self.heatmap_sel_regions.clear()

        # Initialize height factors and visibility for time series plots
        self.plot_height_factors = [1.0] * len(series_list)
        self.trace_visible = [True] * len(series_list)
        # Reset subplot order when loading new data
        self.subplot_order = None

        # Calculate time range from time series, dense groups, raster, and heatmap data
        t_arrays = [s.t for s in self.series]
        for g in self.dense_groups:
            for s in g.series:
                t_arrays.append(s.t)
        for ms in self.raster_series:
            if len(ms.timestamps) > 0:
                t_arrays.append(ms.timestamps)
        for asx in self.heatmap_series:
            if len(asx.t) > 0:
                t_arrays.append(asx.t)
        self.t_global_min, self.t_global_max = nice_time_range(t_arrays)
        self.window_start = self.t_global_min
        self.cursor_time = self.window_start

        # Create all plots (time series, dense, raster, and heatmap)
        self._create_all_plots()

        reporter = getattr(self, "_reporter", None)
        if reporter is not None:
            reporter.phase("Rendering")
        self._apply_x_range()
        self._update_nav_slider_from_window()
        n_dense = sum(len(g.series) for g in self.dense_groups)
        parts = []
        if self.series:
            parts.append(f"{len(self.series)} series")
        if n_dense:
            parts.append(f"{n_dense} dense traces in {len(self.dense_groups)} group(s)")
        self._update_status(f"Loaded {', '.join(parts) or '0 series'}.")
        self._update_hypnogram_extents()
        # Align left axes after layout settles
        QtCore.QTimer.singleShot(0, self._align_left_axes)
        QtCore.QTimer.singleShot(100, self._align_left_axes)
        # Initialize trace visibility state if needed
        if not hasattr(self, "trace_visible") or len(self.trace_visible) != len(
            self.series
        ):
            self.trace_visible = [True] * len(self.series)
        self._apply_trace_visibility()
        if len(self.interval_label_set) > 0:
            self._sync_interval_label_visuals(force_rebuild=True, refresh_summary=False)
        self._sync_global_event_visuals(force_rebuild=True)

    # Default overlay color palette
    _DEFAULT_OVERLAY_COLORS = [
        (100, 200, 255, 255),  # light blue
        (255, 150, 50, 255),  # orange
        (100, 255, 100, 255),  # green
        (255, 100, 255, 255),  # magenta
        (255, 255, 100, 255),  # yellow
        (255, 100, 100, 255),  # red
        (100, 255, 255, 255),  # cyan
        (200, 150, 255, 255),  # lavender
    ]

    # Colorblind-friendly categorical palette (Tol 'light', sans pale grey/black).
    # Light pastel hues picked for visibility against the black plot background.
    _CATEGORY_COLORS = [
        (153, 221, 255, 255),  # light cyan   #99DDFF
        (238, 136, 102, 255),  # orange       #EE8866
        (238, 221, 136, 255),  # light yellow #EEDD88
        (255, 170, 187, 255),  # pink         #FFAABB
        (119, 170, 221, 255),  # light blue   #77AADD
        (68, 187, 153, 255),   # mint         #44BB99
        (187, 204, 51, 255),   # pear         #BBCC33
        (170, 170, 0, 255),    # olive        #AAAA00
    ]
    _CATEGORY_NA_COLOR = (160, 160, 160, 255)

    def set_overlay_series(self, overlay_groups, colors=None):
        """Load overlay groups into the viewer.

        Parameters
        ----------
        overlay_groups : list[OverlayGroup]
            Groups of traces to overlay on shared subplots.
        colors : list or None
            One color per source DataArray. Accepts hex strings or RGB(A) tuples.
        """
        self._stop_playback_if_playing()
        self.overlay_mode = True
        self.overlay_groups = overlay_groups

        # Determine number of source DataArrays
        n_sources = 0
        for g in overlay_groups:
            for tr in g.traces:
                n_sources = max(n_sources, tr.source_idx + 1)

        # Parse overlay colors
        if colors and len(colors) >= n_sources:
            parsed = []
            for c in colors:
                if isinstance(c, str):
                    c = c.strip().lstrip("#")
                    try:
                        r, g_, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
                        a = int(c[6:8], 16) if len(c) == 8 else 255
                        parsed.append((r, g_, b, a))
                    except Exception:
                        parsed.append((255, 255, 255, 255))
                elif isinstance(c, (tuple, list)) and len(c) >= 3:
                    parsed.append(tuple(c[:4]) if len(c) >= 4 else (*c[:3], 255))
                else:
                    parsed.append((255, 255, 255, 255))
            self.overlay_colors = parsed
        else:
            self.overlay_colors = list(self._DEFAULT_OVERLAY_COLORS[:n_sources])
            # Extend with white if more sources than palette entries
            while len(self.overlay_colors) < n_sources:
                self.overlay_colors.append((255, 255, 255, 255))

        # Flatten all traces into self.series for data storage
        self.series = []
        self._plot_to_series = []
        series_idx = 0
        for group in overlay_groups:
            indices = []
            for tr in group.traces:
                self.series.append(Series(tr.name, tr.t, tr.y))
                indices.append(series_idx)
                series_idx += 1
            self._plot_to_series.append(indices)

        # Clear existing plots
        self._clear_all_interval_label_visuals()
        self.plot_area.clear()
        # New plot set: re-fit the uniform left-axis width from scratch.
        self._left_axis_width = None
        self.plots.clear()
        self.curves.clear()
        self.plot_bottom_spines.clear()
        self.sample_marker_scatters.clear()
        self.overlay_curve_items.clear()
        self._plot_to_curves.clear()
        self.plot_cur_lines.clear()
        self.plot_sel_regions.clear()
        self.raster_plots.clear()
        self.raster_items.clear()
        self.raster_cur_lines.clear()
        self.raster_sel_regions.clear()
        self.raster_separator_lines.clear()
        self._raster_line_items.clear()
        self._raster_pens.clear()

        # Height factors and visibility are per-plot (per overlay group)
        self.plot_height_factors = [1.0] * len(overlay_groups)
        self.trace_visible = [True] * len(overlay_groups)
        self.subplot_order = None

        # Calculate time range
        t_arrays = [s.t for s in self.series]
        for ms in self.raster_series:
            if len(ms.timestamps) > 0:
                t_arrays.append(ms.timestamps)
        self.t_global_min, self.t_global_max = nice_time_range(t_arrays)
        self.window_start = self.t_global_min
        self.cursor_time = self.window_start

        # Create all plots
        self._create_all_plots()

        self._apply_x_range()
        self._update_nav_slider_from_window()
        self._update_status(
            f"Loaded {len(overlay_groups)} overlay groups "
            f"({len(self.series)} total traces)."
        )
        self._update_hypnogram_extents()
        QtCore.QTimer.singleShot(0, self._align_left_axes)
        QtCore.QTimer.singleShot(100, self._align_left_axes)
        self._apply_trace_visibility()
        if len(self.interval_label_set) > 0:
            self._sync_interval_label_visuals(force_rebuild=True, refresh_summary=False)
        self._sync_global_event_visuals(force_rebuild=True)

    def set_xarray(self, data, filter_dict=None):
        """Load xarray DataArray(s) into the viewer.

        Parameters
        ----------
        data : xr.DataArray or list[xr.DataArray] or str
            In-memory DataArray(s), or a path to a zarr/netCDF store.
        filter_dict : dict, optional
            Dimension slicing, e.g. ``{"syn_id": slice(3, 6)}``.
        """
        from loupe.xr_loader import (
            convert_xarray_inputs,
            load_xarray_from_path,
        )

        if isinstance(data, str):
            data = load_xarray_from_path(data, filter_dict=filter_dict)
        elif filter_dict is not None and not isinstance(data, list):
            data = data.sel(**filter_dict)

        tuples = convert_xarray_inputs(data)
        self.set_series([Series(n, t, y) for n, t, y in tuples])

    def set_raster_df(
        self,
        data,
        *,
        time_col: str,
        order_by: str,
        split_by: str | list[str] | None = None,
        alpha_by: str | None = None,
        array_name: str = "",
        palette=None,
        alpha_range: tuple[float, float] = (0.3, 1.0),
    ):
        """Load a Polars DataFrame as raster plots.

        Parameters
        ----------
        data : pl.DataFrame, list[pl.DataFrame], or str
            In-memory DataFrame(s), or a path to a parquet file.
        time_col : str
            Column with event timestamps (seconds).  Required.
        order_by : str
            Column for raster row assignment.  Required.
        split_by : str or list[str] or None
            Column(s) to split into separate subplots.
        alpha_by : str or None
            Column for per-event opacity.
        array_name : str
            Array-level prefix for each subplot label.  ``""`` (default)
            uses the raw group value(s); a non-empty string is used as a
            prefix verbatim.
        palette : dict, list, tuple or None
            Per-group color specification.
        alpha_range : tuple[float, float]
            ``(min_alpha, max_alpha)`` for normalizing *alpha_by*.
        """
        from loupe.df_loader import (
            dataframe_to_raster_series,
            load_dataframe_from_parquet,
        )

        if isinstance(data, str):
            data = load_dataframe_from_parquet(data, time_col=time_col)

        if not isinstance(data, list):
            data = [data]

        all_ms: list[RasterSeries] = []
        for mdf in data:
            all_ms.extend(
                dataframe_to_raster_series(
                    mdf,
                    time_col=time_col,
                    order_by=order_by,
                    split_by=split_by,
                    alpha_by=alpha_by,
                    array_name=array_name,
                    palette=palette,
                    alpha_range=alpha_range,
                )
            )

        if not all_ms:
            return

        self.raster_series = all_ms
        self.raster_height_factors = [1.0] * len(all_ms)
        self.raster_visible = [True] * len(all_ms)
        self.subplot_order = None
        if self.series:
            self._rebuild_all_plots()
        else:
            self._update_time_range_from_raster()
            self._create_raster_only_plots()
        self._update_status(
            f"Loaded {len(all_ms)} raster series from DataFrame."
        )

    # ---------- Raster Viewer ----------
    def _load_raster_data(self, timestamps_paths, yvals_paths, alpha_paths, colors):
        """Load raster data from provided file paths."""

        def _normalize_list(raw_list):
            if not raw_list:
                return []
            items = [raw_list] if isinstance(raw_list, str) else list(raw_list)
            out = []
            for it in items:
                s = (it or "").strip()
                if s.startswith("[") and s.endswith("]"):
                    s = s[1:-1]
                parts = s.split(",") if "," in s else [s]
                for p in parts:
                    q = p.strip().strip('"').strip("'")
                    if q.endswith(","):
                        q = q[:-1].rstrip()
                    if q:
                        out.append(q)
            return out

        ts_paths = _normalize_list(timestamps_paths)
        yv_paths = _normalize_list(yvals_paths)
        al_paths = _normalize_list(alpha_paths) if alpha_paths else []
        color_list = _normalize_list(colors) if colors else []

        if len(ts_paths) != len(yv_paths):
            QtWidgets.QMessageBox.warning(
                self,
                "Raster Data",
                f"raster_timestamps ({len(ts_paths)}) and raster_yvals ({len(yv_paths)}) must have same length.",
            )
            return

        def _parse_color(cs: str):
            s = (cs or "").strip()
            try:
                if s.startswith("#"):
                    s = s[1:]
                    if len(s) in (6, 8):
                        r = int(s[0:2], 16)
                        g = int(s[2:4], 16)
                        b = int(s[4:6], 16)
                        return (r, g, b)
                if s.lower().startswith("0x"):
                    v = int(s, 16)
                    r = (v >> 16) & 0xFF
                    g = (v >> 8) & 0xFF
                    b = v & 0xFF
                    return (r, g, b)
                if "," in s:
                    parts = [int(x.strip()) for x in s.split(",") if x.strip()]
                    if len(parts) >= 3:
                        return (parts[0], parts[1], parts[2])
            except Exception:
                pass
            return None

        raster_series = []
        for i, (ts_path, yv_path) in enumerate(zip(ts_paths, yv_paths)):
            if not os.path.exists(ts_path):
                QtWidgets.QMessageBox.warning(
                    self, "Raster Data", f"Timestamps file not found: {ts_path}"
                )
                continue
            if not os.path.exists(yv_path):
                QtWidgets.QMessageBox.warning(
                    self, "Raster Data", f"Yvals file not found: {yv_path}"
                )
                continue

            try:
                timestamps = np.load(ts_path).astype(float).flatten()
                yvals = np.load(yv_path).astype(int).flatten()

                if len(timestamps) != len(yvals):
                    raise ValueError(
                        f"timestamps ({len(timestamps)}) and yvals ({len(yvals)}) must have same length"
                    )

                # Load alphas if provided
                if i < len(al_paths) and al_paths[i] and os.path.exists(al_paths[i]):
                    alphas = np.load(al_paths[i]).astype(float).flatten()
                    if len(alphas) != len(timestamps):
                        raise ValueError(
                            f"alphas ({len(alphas)}) must match timestamps ({len(timestamps)})"
                        )
                    alphas = np.clip(alphas, 0.0, 1.0)
                else:
                    alphas = np.ones(len(timestamps), dtype=float)

                # Parse color
                if i < len(color_list) and color_list[i]:
                    color = _parse_color(color_list[i]) or (255, 255, 255)
                else:
                    color = (255, 255, 255)

                # Determine number of rows
                n_rows = int(np.max(yvals)) + 1 if len(yvals) > 0 else 1

                # Sort by timestamps for efficient windowed rendering
                order = np.argsort(timestamps)
                timestamps = timestamps[order]
                yvals = yvals[order]
                alphas = alphas[order]

                _bn = os.path.basename(ts_path)
                for _suf in ("_timestamps.npy", "_t.npy", ".npy"):
                    if _bn.endswith(_suf):
                        _bn = _bn[: -len(_suf)]
                        break
                name = _bn
                raster_series.append(
                    RasterSeries(
                        name=name,
                        timestamps=timestamps,
                        yvals=yvals,
                        alphas=alphas,
                        color=color,
                        n_rows=n_rows,
                    )
                )
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Raster Load Error", f"Error loading raster {i}: {e}"
                )
                continue

        if raster_series:
            self.raster_series = raster_series
            # Initialize height factors and visibility for raster plots
            self.raster_height_factors = [1.0] * len(raster_series)
            self.raster_visible = [True] * len(raster_series)
            # Reset subplot order to include new raster plots
            self.subplot_order = None
            self._update_status(f"Loaded {len(raster_series)} raster series.")
            # Rebuild plots to include raster series
            if self.series:
                self._rebuild_all_plots()
            else:
                # Update time range and create raster plots if no time series
                self._update_time_range_from_raster()
                self._create_raster_only_plots()

    def _update_time_range_from_raster(self):
        """Update global time range to include raster timestamps."""
        if not self.raster_series:
            return
        for ms in self.raster_series:
            if len(ms.timestamps) > 0:
                t_min = float(np.min(ms.timestamps))
                t_max = float(np.max(ms.timestamps))
                self.t_global_min = min(self.t_global_min, t_min)
                self.t_global_max = max(self.t_global_max, t_max)

    def _create_raster_only_plots(self):
        """Create raster plots when there are no time series."""
        if not self.raster_series:
            return

        self.window_start = self.t_global_min
        self.cursor_time = self.window_start

        # Clear any existing plots
        self._clear_all_interval_label_visuals()
        self.plot_area.clear()
        self.raster_plots.clear()
        self.raster_items.clear()
        self.raster_cur_lines.clear()
        self.raster_sel_regions.clear()
        self.raster_separator_lines.clear()
        self._raster_line_items.clear()
        self._raster_pens.clear()

        master_plot = None
        total_plots = len(self.raster_series)

        for idx, ms in enumerate(self.raster_series):
            vb = SelectableViewBox()
            vb.sigWheelScrolled.connect(self._page)
            vb.sigWheelSmoothScrolled.connect(self._on_smooth_scroll)
            vb.sigWheelCursorScrolled.connect(self._on_cursor_wheel)

            plt = HoverablePlotItem(viewBox=vb)
            plt.sigHovered.connect(self._on_plot_hovered)
            self.plot_area.addItem(plt, row=idx, col=0)

            plt.setLabel("left", ms.name)
            is_last = idx == total_plots - 1
            plt.setLabel("bottom", "Time", units="s" if is_last else None)

            plt.showGrid(x=True, y=False, alpha=0.15)
            plt.enableAutoRange("x", False)
            plt.enableAutoRange("y", False)
            plt.setYRange(-0.5, _raster_extent(ms) - 0.5, padding=0.02)

            left_axis = plt.getAxis("left")
            left_axis.setTicks([[(0, "0"), (_raster_extent(ms) - 1, str(ms.n_rows - 1))]])
            try:
                lf = QtGui.QFont()
                lf.setPointSize(9)
                left_axis.setStyle(tickFont=lf)
            except Exception:
                pass

            if not is_last:
                try:
                    plt.setLabel("bottom", "")
                    plt.showAxis("bottom", False)
                except Exception:
                    pass

            line_items, pens = self._create_raster_render_items(plt, ms)

            cur_line = pg.InfiniteLine(
                angle=90, movable=False, pen=pg.mkPen((255, 255, 255, 120))
            )
            plt.addItem(cur_line)

            sel_region = pg.LinearRegionItem(
                values=(0, 0), brush=pg.mkBrush(100, 200, 255, 40), movable=True
            )
            sel_region.setZValue(-10)
            sel_region.hide()
            plt.addItem(sel_region)
            sel_region.sigRegionChanged.connect(self._on_active_region_dragged)

            self.raster_plots.append(plt)
            self.raster_items.append(None)
            self.raster_cur_lines.append(cur_line)
            self.raster_sel_regions.append(sel_region)
            self.raster_separator_lines.append(
                self._add_raster_separator_lines(plt, ms)
            )
            self._raster_line_items.append(line_items)
            self._raster_pens.append(pens)

            if master_plot is None:
                master_plot = plt
            else:
                plt.setXLink(master_plot)

            vb.sigDragStart.connect(self._on_drag_start)
            vb.sigDragUpdate.connect(self._on_drag_update)
            vb.sigDragFinish.connect(self._on_drag_finish)

        self._apply_custom_plot_heights()
        self._apply_x_range()
        self._update_nav_slider_from_window()
        self._update_hypnogram_extents()
        if len(self.interval_label_set) > 0:
            self._sync_interval_label_visuals(force_rebuild=True, refresh_summary=False)
        self._sync_global_event_visuals(force_rebuild=True)
        QtCore.QTimer.singleShot(0, self._align_left_axes)

    def _rebuild_all_plots(self):
        """Rebuild plots including both time series and raster plots."""
        # Store current state
        old_window_start = self.window_start
        old_cursor = self.cursor_time

        # Clear and rebuild
        self._clear_all_interval_label_visuals()
        self.plot_area.clear()
        # New plot set: re-fit the uniform left-axis width from scratch.
        self._left_axis_width = None
        self.plots.clear()
        self.curves.clear()
        self.plot_bottom_spines.clear()
        self.sample_marker_scatters.clear()
        self.overlay_curve_items.clear()
        self.plot_cur_lines.clear()
        self.plot_sel_regions.clear()
        self.dense_plots.clear()
        self.dense_curves.clear()
        self.dense_cur_lines.clear()
        self.dense_sel_regions.clear()
        self.dense_interval_label_regions.clear()
        self.dense_vscrollbars.clear()
        self.dense_vscroll_proxies.clear()
        self._dense_vscroll_inverted.clear()
        self.raster_plots.clear()
        self.raster_items.clear()
        self.raster_cur_lines.clear()
        self.raster_sel_regions.clear()
        self.raster_separator_lines.clear()
        self._raster_line_items.clear()
        self._raster_pens.clear()
        self.heatmap_plots.clear()
        self.heatmap_image_items.clear()
        self.heatmap_cur_lines.clear()
        self.heatmap_sel_regions.clear()

        # Recalculate time range
        t_arrays = [s.t for s in self.series]
        for g in self.dense_groups:
            for s in g.series:
                t_arrays.append(s.t)
        for ms in self.raster_series:
            if len(ms.timestamps) > 0:
                t_arrays.append(ms.timestamps)
        for asx in self.heatmap_series:
            if len(asx.t) > 0:
                t_arrays.append(asx.t)
        self.t_global_min, self.t_global_max = nice_time_range(t_arrays)

        # Create all plots
        self._create_all_plots()

        # Restore state
        self.window_start = clamp(
            old_window_start,
            self.t_global_min,
            max(self.t_global_min, self.t_global_max - self.window_len),
        )
        self.cursor_time = old_cursor

        self._apply_x_range()
        self._update_nav_slider_from_window()
        self._sync_interval_label_visuals(force_rebuild=True, refresh_summary=False)
        self._sync_global_event_visuals(force_rebuild=True)
        QtCore.QTimer.singleShot(0, self._align_left_axes)

    def _create_all_plots(self):
        """Create time series, dense, raster, and heatmap plots in the layout."""
        if self.overlay_mode:
            self._create_overlay_plots()
            return
        master_plot = None
        total_plots = (
            len(self.series)
            + len(self.dense_groups)
            + len(self.raster_series)
            + len(self.heatmap_series)
        )
        reporter = getattr(self, "_reporter", None)
        if reporter is not None and total_plots > 0:
            reporter.phase("Creating plots")
        built = 0

        # Build a (kind, idx) → display-row lookup. Each entry of
        # subplot_order maps to a sequential row, top-to-bottom. When
        # subplot_order is unset, fall back to the legacy segregated order
        # (ts → dense → raster → heatmap). Each plot builds in type-segregated
        # order regardless, then looks up its row from this map — so the
        # visual layout is decoupled from build order.
        if self.subplot_order:
            order = list(self.subplot_order)
        else:
            order = (
                [("ts", i) for i in range(len(self.series))]
                + [("dense", i) for i in range(len(self.dense_groups))]
                + [("raster", i) for i in range(len(self.raster_series))]
                + [("heatmap", i) for i in range(len(self.heatmap_series))]
            )
        row_for = {entry: row for row, entry in enumerate(order)}
        last_row = total_plots - 1

        def _row(kind: str, idx: int) -> int:
            return row_for.get((kind, idx), 0)

        # Create time series plots first
        for idx, s in enumerate(self.series):
            if reporter is not None:
                reporter.item(built, total_plots, detail=s.name)
            built += 1
            row_idx = _row("ts", idx)
            vb = SelectableViewBox()
            vb.sigWheelScrolled.connect(self._page)
            vb.sigWheelSmoothScrolled.connect(self._on_smooth_scroll)
            vb.sigWheelCursorScrolled.connect(self._on_cursor_wheel)

            plt = HoverablePlotItem(viewBox=vb)
            plt.sigHovered.connect(self._on_plot_hovered)
            self.plot_area.addItem(plt, row=row_idx, col=0)

            plt.setLabel("left", s.name)
            is_last = row_idx == last_row
            plt.setLabel("bottom", "Time", units="s" if is_last else None)
            plt.showGrid(x=True, y=True, alpha=0.15)
            plt.addLegend(offset=(10, 10))
            plt.enableAutoRange("x", False)

            if not is_last:
                try:
                    plt.setLabel("bottom", "")
                    plt.showAxis("bottom", False)
                except Exception:
                    pass

            try:
                lf = QtGui.QFont()
                lf.setPointSize(9)
                plt.getAxis("left").setStyle(tickFont=lf)
            except Exception:
                pass

            if getattr(self, "series_colors", None) and idx < len(self.series_colors):
                pen_color = self.series_colors[idx]
            else:
                pen_color = (255, 255, 255)
            pen = pg.mkPen(pen_color, width=1)
            # Name the host curve only when it carries overlays, so its legend
            # distinguishes host vs. overlay; otherwise leave it unnamed (no
            # legend entry, preserving the pre-overlay look).
            host_name = (
                self.overlay_main_names[idx]
                if idx < len(self.overlay_main_names)
                else None
            )
            curve = pg.PlotDataItem(
                [], [], pen=pen, name=host_name, antialias=False
            )
            plt.addItem(curve)
            # NB: must follow addItem(), which resets these to the PlotItem defaults.
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)

            # Overlay curves drawn on the same subplot (TraceConfig.overlay_arrays).
            overlay_items_for_series: list[pg.PlotDataItem] = []
            if idx < len(self.overlay_series):
                for oc in self.overlay_series[idx]:
                    o_pen = pg.mkPen(oc.color, width=1)
                    o_curve = pg.PlotDataItem(
                        [], [], pen=o_pen, name=oc.name, antialias=False
                    )
                    plt.addItem(o_curve)
                    o_curve.setDownsampling(auto=True, method="peak")
                    o_curve.setClipToView(True)
                    overlay_items_for_series.append(o_curve)
            self.overlay_curve_items.append(overlay_items_for_series)

            scatters_for_series: list[pg.ScatterPlotItem] = []
            for marker in self.sample_markers:
                sk = _scatter_kwargs_for_marker(marker)
                scatter = pg.ScatterPlotItem(**sk, pxMode=True, antialias=False)
                scatter.setZValue(10)
                plt.addItem(scatter)
                scatters_for_series.append(scatter)
            self.sample_marker_scatters.append(scatters_for_series)

            cur_line = pg.InfiniteLine(
                angle=90, movable=False, pen=pg.mkPen((255, 255, 255, 120))
            )
            plt.addItem(cur_line)

            sel_region = pg.LinearRegionItem(
                values=(0, 0), brush=pg.mkBrush(100, 200, 255, 40), movable=True
            )
            sel_region.setZValue(-10)
            sel_region.hide()
            plt.addItem(sel_region)
            sel_region.sigRegionChanged.connect(self._on_active_region_dragged)

            # Optional minimal bottom-boundary line. Skipped on the bottom-most
            # subplot, which already shows the full time axis. Pinned to the
            # current y-min and re-pinned on y-range changes so it stays flush
            # with the bottom under both fixed and auto scaling; adds no layout
            # height, so stacking stays as tight as without it.
            bottom_spine = None
            want_spine = (
                idx < len(self.series_bottom_spine)
                and bool(self.series_bottom_spine[idx])
                and not is_last
            )
            if want_spine:
                bottom_spine = pg.InfiniteLine(
                    angle=0, movable=False, pen=pg.mkPen((255, 255, 255), width=1)
                )
                bottom_spine.setZValue(-4)
                plt.addItem(bottom_spine)
                vb.sigYRangeChanged.connect(
                    lambda *_, _vb=vb, _sp=bottom_spine: self._pin_bottom_spine(_vb, _sp)
                )
                self._pin_bottom_spine(vb, bottom_spine)
            self.plot_bottom_spines.append(bottom_spine)

            self.plots.append(plt)
            self.curves.append(curve)
            self.plot_cur_lines.append(cur_line)
            self.plot_sel_regions.append(sel_region)

            if master_plot is None:
                master_plot = plt
            else:
                plt.setXLink(master_plot)

            vb.sigDragStart.connect(self._on_drag_start)
            vb.sigDragUpdate.connect(self._on_drag_update)
            vb.sigDragFinish.connect(self._on_drag_finish)

            if self.fixed_scale:
                try:
                    y = np.asarray(s.y, dtype=float)
                    if idx < len(self.overlay_series) and self.overlay_series[idx]:
                        y = np.concatenate(
                            [y]
                            + [
                                np.asarray(oc.y, dtype=float)
                                for oc in self.overlay_series[idx]
                            ]
                        )
                    lo = float(np.nanpercentile(y, 1.0))
                    hi = float(np.nanpercentile(y, 99.0))
                    if not np.isfinite(lo) or not np.isfinite(hi):
                        raise ValueError("non-finite percentiles")
                    if hi <= lo:
                        hi = lo + 1.0
                    pad = 0.05 * (hi - lo)
                    plt.enableAutoRange("y", False)
                    plt.setYRange(lo - pad, hi + pad, padding=0)
                except Exception:
                    plt.enableAutoRange("y", False)
            else:
                plt.enableAutoRange("y", True)

        # Create dense plots
        for gi in range(len(self.dense_groups)):
            if reporter is not None:
                reporter.item(
                    built, total_plots, detail=self.dense_groups[gi].name
                )
            built += 1
            row_idx = _row("dense", gi)
            plt = self._create_dense_plot(gi, master_plot=master_plot)
            self.plot_area.addItem(plt, row=row_idx, col=0)
            self.plot_area.addItem(self.dense_vscroll_proxies[gi], row=row_idx, col=1)
            is_last = row_idx == last_row
            plt.setLabel("bottom", "Time", units="s" if is_last else None)
            if not is_last:
                try:
                    plt.setLabel("bottom", "")
                    plt.showAxis("bottom", False)
                except Exception:
                    pass
            if master_plot is None:
                master_plot = plt

        # Create raster plots
        for idx, ms in enumerate(self.raster_series):
            if reporter is not None:
                reporter.item(built, total_plots, detail=ms.name)
            built += 1
            row_idx = _row("raster", idx)
            vb = SelectableViewBox()
            vb.sigWheelScrolled.connect(self._page)
            vb.sigWheelSmoothScrolled.connect(self._on_smooth_scroll)
            vb.sigWheelCursorScrolled.connect(self._on_cursor_wheel)

            plt = HoverablePlotItem(viewBox=vb)
            plt.sigHovered.connect(self._on_plot_hovered)
            self.plot_area.addItem(plt, row=row_idx, col=0)

            plt.setLabel("left", ms.name)
            is_last = row_idx == last_row
            plt.setLabel("bottom", "Time", units="s" if is_last else None)

            # Raster plots: no horizontal grid, minimal y-axis
            plt.showGrid(x=True, y=False, alpha=0.15)
            plt.enableAutoRange("x", False)
            plt.enableAutoRange("y", False)

            # Set Y range to show all rows
            plt.setYRange(-0.5, _raster_extent(ms) - 0.5, padding=0.02)

            # Configure Y-axis: only show min and max tick values
            left_axis = plt.getAxis("left")
            left_axis.setTicks([[(0, "0"), (_raster_extent(ms) - 1, str(ms.n_rows - 1))]])
            try:
                lf = QtGui.QFont()
                lf.setPointSize(9)
                left_axis.setStyle(tickFont=lf)
            except Exception:
                pass

            if not is_last:
                try:
                    plt.setLabel("bottom", "")
                    plt.showAxis("bottom", False)
                except Exception:
                    pass

            line_items, pens = self._create_raster_render_items(plt, ms)

            cur_line = pg.InfiniteLine(
                angle=90, movable=False, pen=pg.mkPen((255, 255, 255, 120))
            )
            plt.addItem(cur_line)

            sel_region = pg.LinearRegionItem(
                values=(0, 0), brush=pg.mkBrush(100, 200, 255, 40), movable=True
            )
            sel_region.setZValue(-10)
            sel_region.hide()
            plt.addItem(sel_region)
            sel_region.sigRegionChanged.connect(self._on_active_region_dragged)

            self.raster_plots.append(plt)
            self.raster_items.append(None)
            self.raster_cur_lines.append(cur_line)
            self.raster_sel_regions.append(sel_region)
            self.raster_separator_lines.append(
                self._add_raster_separator_lines(plt, ms)
            )
            self._raster_line_items.append(line_items)
            self._raster_pens.append(pens)

            if master_plot is None:
                master_plot = plt
            else:
                plt.setXLink(master_plot)

            vb.sigDragStart.connect(self._on_drag_start)
            vb.sigDragUpdate.connect(self._on_drag_update)
            vb.sigDragFinish.connect(self._on_drag_finish)

        # Create heatmap plots
        for idx, asx in enumerate(self.heatmap_series):
            if reporter is not None:
                reporter.item(built, total_plots, detail=asx.name)
            built += 1
            row_idx = _row("heatmap", idx)
            plt = self._create_array_plot(idx, asx, master_plot, row_idx, last_row + 1)
            self.plot_area.addItem(plt, row=row_idx, col=0)
            if master_plot is None:
                master_plot = plt

        # Apply custom plot heights (includes raster row heights logic)
        self._apply_custom_plot_heights()
        self._setup_dense_vscrollbars()
        self._constrain_scrollbar_column()

    def _create_array_plot(
        self,
        heatmap_idx: int,
        asx: HeatmapSeries,
        master_plot,
        row_idx: int,
        total_plots: int,
    ):
        """Build a single heatmap-style plot backed by a pg.ImageItem."""
        vb = SelectableViewBox()
        vb.sigWheelScrolled.connect(self._page)
        vb.sigWheelSmoothScrolled.connect(self._on_smooth_scroll)
        vb.sigWheelCursorScrolled.connect(self._on_cursor_wheel)

        plt = HoverablePlotItem(viewBox=vb)
        plt.sigHovered.connect(self._on_plot_hovered)

        plt.setLabel("left", asx.name)
        is_last = row_idx == total_plots - 1
        plt.setLabel("bottom", "Time", units="s" if is_last else None)

        # Heatmap aesthetic: keep vertical grid lines, no horizontal grid.
        plt.showGrid(x=True, y=False, alpha=0.15)
        plt.enableAutoRange("x", False)
        plt.enableAutoRange("y", False)

        n_rows = asx.Y.shape[0]
        plt.setYRange(0.0, float(n_rows), padding=0)

        # Y-axis ticks: show only the first and last row_label values.
        left_axis = plt.getAxis("left")
        if asx.row_labels is not None and len(asx.row_labels) >= 1:
            first_label = str(asx.row_labels[0])
            last_label = str(asx.row_labels[-1])
        else:
            first_label = "0"
            last_label = str(n_rows - 1) if n_rows > 0 else "0"
        left_axis.setTicks([[(0.5, first_label), (max(0.0, n_rows - 0.5), last_label)]])
        try:
            lf = QtGui.QFont()
            lf.setPointSize(9)
            left_axis.setStyle(tickFont=lf)
        except Exception:
            pass

        if not is_last:
            try:
                plt.setLabel("bottom", "")
                plt.showAxis("bottom", False)
            except Exception:
                pass

        # ImageItem in row-major order: image[row, col] with row=y, col=time.
        image_item = pg.ImageItem(axisOrder="row-major")
        image_item.setZValue(0)
        plt.addItem(image_item)

        # Cursor + selection (parallel to other plot types).
        cur_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen((255, 255, 255, 120))
        )
        plt.addItem(cur_line)

        sel_region = pg.LinearRegionItem(
            values=(0, 0), brush=pg.mkBrush(100, 200, 255, 40), movable=True
        )
        sel_region.setZValue(-10)
        sel_region.hide()
        plt.addItem(sel_region)
        sel_region.sigRegionChanged.connect(self._on_active_region_dragged)

        if master_plot is not None:
            plt.setXLink(master_plot)

        vb.sigDragStart.connect(self._on_drag_start)
        vb.sigDragUpdate.connect(self._on_drag_update)
        vb.sigDragFinish.connect(self._on_drag_finish)

        # Register with parallel registries.
        self.heatmap_plots.append(plt)
        self.heatmap_image_items.append(image_item)
        self.heatmap_cur_lines.append(cur_line)
        self.heatmap_sel_regions.append(sel_region)
        if len(self._heatmap_cache_keys) <= heatmap_idx:
            self._heatmap_cache_keys.append(None)

        return plt

    def _create_overlay_plots(self):
        """Create plots for overlay mode: multiple curves per subplot."""
        master_plot = None
        row_idx = 0
        total_plots = len(self.overlay_groups) + len(self.raster_series)
        self._plot_to_curves = []

        for grp_idx, group in enumerate(self.overlay_groups):
            vb = SelectableViewBox()
            vb.sigWheelScrolled.connect(self._page)
            vb.sigWheelSmoothScrolled.connect(self._on_smooth_scroll)
            vb.sigWheelCursorScrolled.connect(self._on_cursor_wheel)

            plt = HoverablePlotItem(viewBox=vb)
            plt.sigHovered.connect(self._on_plot_hovered)
            self.plot_area.addItem(plt, row=row_idx, col=0)

            plt.setLabel("left", group.label)
            is_last = row_idx == total_plots - 1
            plt.setLabel("bottom", "Time", units="s" if is_last else None)
            plt.showGrid(x=True, y=True, alpha=0.15)
            plt.addLegend(offset=(10, 10))
            plt.enableAutoRange("x", False)

            if not is_last:
                try:
                    plt.setLabel("bottom", "")
                    plt.showAxis("bottom", False)
                except Exception:
                    pass

            try:
                lf = QtGui.QFont()
                lf.setPointSize(9)
                plt.getAxis("left").setStyle(tickFont=lf)
            except Exception:
                pass

            # Create one curve per trace in this group
            group_curves = []
            for tr in group.traces:
                pen_color = self.overlay_colors[tr.source_idx]
                pen = pg.mkPen(pen_color, width=1)
                curve = pg.PlotDataItem([], [], pen=pen, name=tr.name, antialias=False)
                plt.addItem(curve)
                # NB: must follow addItem(), which resets these to the PlotItem defaults.
                curve.setDownsampling(auto=True, method="peak")
                curve.setClipToView(True)
                group_curves.append(curve)
                # Also maintain flat curves list for compat
                self.curves.append(curve)
            self._plot_to_curves.append(group_curves)

            # Cursor line and selection region (one per subplot)
            cur_line = pg.InfiniteLine(
                angle=90, movable=False, pen=pg.mkPen((255, 255, 255, 120))
            )
            plt.addItem(cur_line)

            sel_region = pg.LinearRegionItem(
                values=(0, 0), brush=pg.mkBrush(100, 200, 255, 40), movable=True
            )
            sel_region.setZValue(-10)
            sel_region.hide()
            plt.addItem(sel_region)
            sel_region.sigRegionChanged.connect(self._on_active_region_dragged)

            self.plots.append(plt)
            self.plot_cur_lines.append(cur_line)
            self.plot_sel_regions.append(sel_region)

            if master_plot is None:
                master_plot = plt
            else:
                plt.setXLink(master_plot)

            vb.sigDragStart.connect(self._on_drag_start)
            vb.sigDragUpdate.connect(self._on_drag_update)
            vb.sigDragFinish.connect(self._on_drag_finish)

            # Fixed scale: compute Y range from ALL traces in group
            if self.fixed_scale:
                try:
                    all_y = np.concatenate(
                        [np.asarray(tr.y, dtype=float) for tr in group.traces]
                    )
                    lo = float(np.nanpercentile(all_y, 1.0))
                    hi = float(np.nanpercentile(all_y, 99.0))
                    if not np.isfinite(lo) or not np.isfinite(hi):
                        raise ValueError("non-finite percentiles")
                    if hi <= lo:
                        hi = lo + 1.0
                    pad = 0.05 * (hi - lo)
                    plt.enableAutoRange("y", False)
                    plt.setYRange(lo - pad, hi + pad, padding=0)
                except Exception:
                    plt.enableAutoRange("y", False)
            else:
                plt.enableAutoRange("y", True)

            row_idx += 1

        # Create raster plots (same as _create_all_plots)
        for idx, ms in enumerate(self.raster_series):
            vb = SelectableViewBox()
            vb.sigWheelScrolled.connect(self._page)
            vb.sigWheelSmoothScrolled.connect(self._on_smooth_scroll)
            vb.sigWheelCursorScrolled.connect(self._on_cursor_wheel)

            plt = HoverablePlotItem(viewBox=vb)
            plt.sigHovered.connect(self._on_plot_hovered)
            self.plot_area.addItem(plt, row=row_idx, col=0)

            plt.setLabel("left", ms.name)
            is_last = row_idx == total_plots - 1
            plt.setLabel("bottom", "Time", units="s" if is_last else None)
            plt.showGrid(x=True, y=True, alpha=0.15)
            plt.enableAutoRange("x", False)
            plt.enableAutoRange("y", False)

            unique_y = np.unique(ms.yvals)
            if len(unique_y) > 0:
                y_min = float(unique_y[0]) - 0.5
                y_max = float(unique_y[-1]) + 0.5
                plt.setYRange(y_min, y_max, padding=0)

            line_items, pens = self._create_raster_render_items(plt, ms)

            cur_line = pg.InfiniteLine(
                angle=90, movable=False, pen=pg.mkPen((255, 255, 255, 120))
            )
            plt.addItem(cur_line)

            sel_region = pg.LinearRegionItem(
                values=(0, 0), brush=pg.mkBrush(100, 200, 255, 40), movable=True
            )
            sel_region.setZValue(-10)
            sel_region.hide()
            plt.addItem(sel_region)
            sel_region.sigRegionChanged.connect(self._on_active_region_dragged)

            self.raster_plots.append(plt)
            self.raster_items.append(None)
            self.raster_cur_lines.append(cur_line)
            self.raster_sel_regions.append(sel_region)
            self.raster_separator_lines.append(
                self._add_raster_separator_lines(plt, ms)
            )
            self._raster_line_items.append(line_items)
            self._raster_pens.append(pens)

            if master_plot is None:
                master_plot = plt
            else:
                plt.setXLink(master_plot)

            vb.sigDragStart.connect(self._on_drag_start)
            vb.sigDragUpdate.connect(self._on_drag_update)
            vb.sigDragFinish.connect(self._on_drag_finish)

            row_idx += 1

        self._apply_custom_plot_heights()

    def _add_raster_separator_lines(self, plt, ms) -> list:
        """Draw the horizontal separator lines for one raster subplot.

        Returns the created ``pg.InfiniteLine`` handles (empty list when the
        series has no separators).  The lines sit in the empty gap bands opened
        by the row-position shift applied in ``dataframe_to_raster_series``.
        """
        handles: list = []
        if not ms.separator_lines:
            return handles
        color = ms.separator_color if ms.separator_color is not None else (120, 120, 120)
        width = ms.separator_width if ms.separator_width is not None else 1.0
        for ypos in ms.separator_lines:
            line = pg.InfiniteLine(
                pos=ypos, angle=0, movable=False, pen=pg.mkPen(color, width=width)
            )
            line.setZValue(-5)  # above the background grid, below the event ticks
            plt.addItem(line)
            handles.append(line)
        return handles

    def _build_raster_pens(self, ms: RasterSeries) -> list[list[QtGui.QPen]]:
        """Build one row of 11 alpha-graded pens per category color.

        Returns a nested list shaped ``[n_categories][RASTER_ALPHA_LEVEL_COUNT]``.
        For non-categorical series ``ms.category_colors`` is ``None`` and the
        outer dimension is length 1, reproducing the single-color fast path.
        """
        if ms.category_colors is not None:
            base_colors = ms.category_colors
        else:
            base_colors = [ms.color]
        return [
            [
                pg.mkPen(
                    color=(
                        r,
                        g,
                        b,
                        int((alevel / (RASTER_ALPHA_LEVEL_COUNT - 1)) * 255),
                    ),
                    width=self.raster_event_thickness,
                )
                for alevel in range(RASTER_ALPHA_LEVEL_COUNT)
            ]
            for (r, g, b) in base_colors
        ]

    def _create_raster_render_items(
        self, plt: pg.PlotItem, ms: RasterSeries
    ) -> tuple[list[list[pg.PlotDataItem]], list[list[QtGui.QPen]]]:
        pens_grid = self._build_raster_pens(ms)
        line_items_grid: list[list[pg.PlotDataItem]] = []
        for cat_pens in pens_grid:
            cat_items: list[pg.PlotDataItem] = []
            for pen in cat_pens:
                line_item = pg.PlotDataItem(
                    [], [], pen=pen, connect="pairs", antialias=False
                )
                plt.addItem(line_item)
                cat_items.append(line_item)
            line_items_grid.append(cat_items)
        return line_items_grid, pens_grid

    def _refresh_raster_pen_cache(self) -> None:
        for midx, ms in enumerate(self.raster_series):
            if midx >= len(self._raster_line_items):
                break
            pens_grid = self._build_raster_pens(ms)
            if midx < len(self._raster_pens):
                self._raster_pens[midx] = pens_grid
            else:
                self._raster_pens.append(pens_grid)
            for cat_items, cat_pens in zip(self._raster_line_items[midx], pens_grid):
                for line_item, pen in zip(cat_items, cat_pens):
                    line_item.setPen(pen)

    def _is_trace_plot_visible(self, plot_idx: int) -> bool:
        return (
            not hasattr(self, "trace_visible")
            or plot_idx >= len(self.trace_visible)
            or self.trace_visible[plot_idx]
        )

    def _is_raster_plot_visible(self, plot_idx: int) -> bool:
        return plot_idx >= len(self.raster_visible) or self.raster_visible[plot_idx]

    def _raster_segment_for_window(
        self, ms: RasterSeries, t0: float, t1: float, max_events: int = 10000
    ):
        """
        Return event data for the [t0, t1] window, limited to max_events for
        performance. Returns ``(timestamps, yvals, alphas, cats)`` where
        ``cats`` is the per-event category index (or ``None`` when the
        RasterSeries has no categorical coloring).
        """
        if t1 <= t0 or len(ms.timestamps) == 0:
            empty_cats = None if ms.category_index is None else np.empty(0, dtype=np.int16)
            return np.empty(0), np.empty(0), np.empty(0), empty_cats

        # Binary search for window bounds (timestamps are sorted)
        i0 = np.searchsorted(ms.timestamps, t0, side="left")
        i1 = np.searchsorted(ms.timestamps, t1, side="right")

        if i0 >= i1:
            empty_cats = None if ms.category_index is None else np.empty(0, dtype=np.int16)
            return np.empty(0), np.empty(0), np.empty(0), empty_cats

        # Slice to window
        ts = ms.timestamps[i0:i1]
        ys = ms.yvals[i0:i1]
        als = ms.alphas[i0:i1]
        cats = ms.category_index[i0:i1] if ms.category_index is not None else None

        # Downsample if too many events (uniform sampling)
        if len(ts) > max_events:
            step = len(ts) // max_events
            ts = ts[::step]
            ys = ys[::step]
            als = als[::step]
            if cats is not None:
                cats = cats[::step]

        return ts, ys, als, cats

    # ------------------------------------------------------------------ dense
    def _dense_visible_indices(self, group_idx: int) -> list[int]:
        """Return indices into group.series for visible traces."""
        group = self.dense_groups[group_idx]
        return [
            i
            for i in range(0, len(group.series), group.step)
            if i not in group.hidden_traces
        ]

    def _dense_offsets(self, group_idx: int) -> np.ndarray:
        """Compute Y-offsets for visible traces in a dense group."""
        group = self.dense_groups[group_idx]
        visible = self._dense_visible_indices(group_idx)
        if group.order_values is not None and len(group.order_values) == len(group.series):
            offsets = group.order_values[visible].astype(float)
        else:
            offsets = np.arange(len(visible), dtype=float)
        return offsets

    def _dense_offset_margin(self, offsets: np.ndarray) -> float:
        """Compute a reasonable margin for dense plot Y-range."""
        if len(offsets) < 2:
            return 1.0
        gaps = np.diff(offsets)
        return float(np.median(gaps)) * 0.5 if len(gaps) > 0 else 1.0

    def _dense_category_map(self, group_idx: int) -> dict[str, tuple] | None:
        """Build a stable category -> RGBA mapping for a dense group.

        Returns *None* if the group has no ``color_values``.  When
        ``group.palette`` is provided, it overrides the default cycle:
        ``dict`` mappings are looked up by stringified category key, and
        ``list``/``tuple`` palettes are assigned in first-seen order.
        Categories without a palette entry fall back to the default cycle.
        """
        group = self.dense_groups[group_idx]
        if group.color_values is None or len(group.color_values) != len(group.series):
            return None
        user_palette = group.palette
        user_dict: dict | None = None
        user_list: list | None = None
        if isinstance(user_palette, dict):
            user_dict = {str(k): v for k, v in user_palette.items()}
        elif isinstance(user_palette, (list, tuple)):
            user_list = list(user_palette)
        seen: dict[str, tuple] = {}
        default_palette = self._CATEGORY_COLORS
        for v in group.color_values:
            key = str(v)
            if key in seen:
                continue
            if user_dict is not None and key in user_dict:
                seen[key] = user_dict[key]
            elif user_list is not None and len(user_list) > 0:
                seen[key] = user_list[len(seen) % len(user_list)]
            else:
                seen[key] = default_palette[len(seen) % len(default_palette)]
        return seen

    def _dense_pens(self, group_idx: int) -> list:
        """Return one pen per visible trace in a dense group.

        Uses the categorical ``color_values`` mapping when present;
        otherwise falls back to the default gray pen.
        """
        group = self.dense_groups[group_idx]
        visible = self._dense_visible_indices(group_idx)
        cat_map = self._dense_category_map(group_idx)
        if cat_map is None:
            default_pen = pg.mkPen((200, 200, 200), width=1)
            return [default_pen] * len(visible)
        na_keys = {"nan", "None", "", "NA", "<NA>"}
        pens = []
        for si in visible:
            key = str(group.color_values[si])
            color = self._CATEGORY_NA_COLOR if key in na_keys else cat_map.get(key, self._CATEGORY_NA_COLOR)
            pens.append(pg.mkPen(color, width=1))
        return pens

    def _create_dense_plot(self, group_idx: int, master_plot=None):
        """Create a single dense (EEG-style) PlotItem for a DenseGroup."""
        group = self.dense_groups[group_idx]
        visible = self._dense_visible_indices(group_idx)
        offsets = self._dense_offsets(group_idx)
        visible_labels = [group.trace_labels[i] for i in visible]

        vb = DenseViewBox()
        vb.sigWheelScrolled.connect(self._page)
        vb.sigWheelSmoothScrolled.connect(self._on_smooth_scroll)
        vb.sigWheelCursorScrolled.connect(self._on_cursor_wheel)
        vb.sigWheelGainAdjust.connect(
            lambda d: self._adjust_dense_gain(1.2 if d > 0 else 1 / 1.2)
        )
        vb.sigWheelGainAdjustFocused.connect(
            lambda d: self._adjust_dense_gain_focused(1.2 if d > 0 else 1 / 1.2)
        )
        vb.sigWheelVerticalSmooth.connect(self._on_dense_vertical_smooth)

        plt = HoverablePlotItem(viewBox=vb)
        plt.sigHovered.connect(self._on_plot_hovered)

        plt.setLabel("left", group.name)
        plt.showGrid(x=True, y=False, alpha=0.15)
        plt.enableAutoRange("x", False)
        plt.enableAutoRange("y", False)

        # Y-axis ticks: trace labels at offset positions
        left_axis = plt.getAxis("left")
        tick_list = [(float(o), lbl) for o, lbl in zip(offsets, visible_labels)]
        left_axis.setTicks([tick_list])
        try:
            lf = QtGui.QFont()
            lf.setPointSize(8)
            left_axis.setStyle(tickFont=lf)
        except Exception:
            pass

        # Create curves (one pen per visible trace; may vary by hue category)
        pens = self._dense_pens(group_idx)
        curves: list[pg.PlotDataItem] = []
        for pen in pens:
            curve = pg.PlotDataItem([], [], pen=pen, antialias=False)
            curve.setZValue(5)
            plt.addItem(curve)
            # NB: must follow addItem(), which resets these to the PlotItem defaults.
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)
            curves.append(curve)

        # Cursor line
        cur_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen((255, 255, 255, 120))
        )
        plt.addItem(cur_line)

        # Selection region for labeling
        sel_region = pg.LinearRegionItem(
            values=(0, 0), brush=pg.mkBrush(100, 200, 255, 40), movable=True
        )
        sel_region.setZValue(-10)
        sel_region.hide()
        plt.addItem(sel_region)
        sel_region.sigRegionChanged.connect(self._on_active_region_dragged)

        # Set Y-range: show traces_per_page traces, or all if not set
        if len(offsets) > 0:
            margin = self._dense_offset_margin(offsets)
            tpp = group.traces_per_page
            if tpp is not None and tpp < len(offsets):
                # Show the first tpp traces (lowest offsets)
                page_max = float(offsets[min(tpp, len(offsets)) - 1])
                plt.setYRange(
                    float(offsets[0]) - margin,
                    page_max + margin,
                    padding=0,
                )
            else:
                plt.setYRange(
                    float(offsets.min()) - margin,
                    float(offsets.max()) + margin,
                    padding=0,
                )

        # X-link
        if master_plot is not None:
            plt.setXLink(master_plot)

        # Connect drag signals for labeling
        vb.sigDragStart.connect(self._on_drag_start)
        vb.sigDragUpdate.connect(self._on_drag_update)
        vb.sigDragFinish.connect(self._on_drag_finish)

        self.dense_plots.append(plt)
        self.dense_curves.append(curves)
        self.dense_cur_lines.append(cur_line)
        self.dense_sel_regions.append(sel_region)
        self.dense_interval_label_regions.append([])

        # Per-group vertical scrollbar as a proxy widget for the graphics layout.
        # An explicit stylesheet is required: QScrollBar inside a QGraphicsProxyWidget
        # doesn't reliably paint the native handle, so we draw it via CSS.
        sb = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Vertical, self)
        sb.setFixedWidth(14)
        sb.setStyleSheet(
            """
            QScrollBar:vertical {
                background: #2a2a2a;
                width: 14px;
                margin: 0px;
                border: 1px solid #444;
            }
            QScrollBar::handle:vertical {
                background: #888;
                min-height: 20px;
                border-radius: 3px;
                margin: 1px;
            }
            QScrollBar::handle:vertical:hover {
                background: #aaa;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
                background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            """
        )
        sb.hide()
        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(sb)
        sb.valueChanged.connect(
            lambda val, gi=group_idx: self._on_dense_vscrollbar_changed(gi, val)
        )
        self.dense_vscrollbars.append(sb)
        self.dense_vscroll_proxies.append(proxy)
        self._dense_vscroll_inverted.append(False)

        return plt

    def _rebuild_dense_curves(self, group_idx: int):
        """Rebuild curve items for a dense group (after step/visibility change)."""
        group = self.dense_groups[group_idx]
        plt = self.dense_plots[group_idx]

        # Remove old curves
        for curve in self.dense_curves[group_idx]:
            plt.removeItem(curve)

        visible = self._dense_visible_indices(group_idx)
        offsets = self._dense_offsets(group_idx)
        visible_labels = [group.trace_labels[i] for i in visible]

        pens = self._dense_pens(group_idx)
        curves: list[pg.PlotDataItem] = []
        for pen in pens:
            curve = pg.PlotDataItem([], [], pen=pen, antialias=False)
            curve.setZValue(5)
            plt.addItem(curve)
            # NB: must follow addItem(), which resets these to the PlotItem defaults.
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)
            curves.append(curve)
        self.dense_curves[group_idx] = curves

        # Update Y-axis ticks
        left_axis = plt.getAxis("left")
        tick_list = [(float(o), lbl) for o, lbl in zip(offsets, visible_labels)]
        left_axis.setTicks([tick_list])

        # Keep the same number of traces visible as before the rebuild
        if len(offsets) > 0:
            old_yrange = plt.getViewBox().viewRange()[1]
            old_span = old_yrange[1] - old_yrange[0]
            old_center = (old_yrange[0] + old_yrange[1]) / 2.0
            # Clamp center to valid offset range
            center = max(float(offsets.min()), min(float(offsets.max()), old_center))
            half = old_span / 2.0
            plt.setYRange(center - half, center + half, padding=0)

    def _refresh_dense_curves(self):
        """Update dense plot curves for the current window."""
        if not self.dense_groups:
            return
        t0, t1 = self.window_start, self.window_start + self.window_len
        for gi, group in enumerate(self.dense_groups):
            visible = self._dense_visible_indices(gi)
            offsets = self._dense_offsets(gi)
            curves = self.dense_curves[gi]
            means = self._dense_means[gi]
            for li, (si, offset) in enumerate(zip(visible, offsets)):
                s = group.series[si]
                i0 = max(0, np.searchsorted(s.t, t0) - 1)
                i1 = min(len(s.t), np.searchsorted(s.t, t1) + 1)
                ts = s.t[i0:i1]
                ys_display = (s.y[i0:i1] - means[si]) * group.gain + offset
                curves[li].setData(ts, ys_display, _callSync="off")

    def _vertical_page(self, direction: int):
        """Scroll the plot scroll area up/down by one page."""
        if not hasattr(self, "plot_scroll_area"):
            return
        sb = self.plot_scroll_area.verticalScrollBar()
        page = self.plot_scroll_area.viewport().height()
        sb.setValue(sb.value() + direction * page)

    def _update_plot_area_height(self):
        """Set plot_area minimum height so the scroll area shows a scrollbar when needed."""
        visible = self._get_visible_subplot_order()
        n = len(visible)
        if n == 0:
            return
        # Desired height = n * trace_height_px, but at least the scroll area height
        scroll_h = self.plot_scroll_area.viewport().height() if hasattr(self, "plot_scroll_area") else 600
        desired = n * self.trace_height_px
        if desired > scroll_h:
            self.plot_area.setMinimumHeight(desired)
        else:
            self.plot_area.setMinimumHeight(0)

    # ---- Dense vertical scrollbars ------------------------------------------
    def _constrain_scrollbar_column(self):
        """Set column constraints so scrollbar proxies stay narrow."""
        if not self.dense_vscroll_proxies:
            return
        try:
            layout = self.plot_area.ci.layout
            layout.setColumnStretchFactor(1, 0)
            layout.setColumnMaximumWidth(1, 16)
        except Exception:
            pass

    def _setup_dense_vscrollbars(self):
        """Configure per-group dense vertical scrollbars."""
        for gi in range(len(self.dense_groups)):
            self._setup_dense_vscrollbar_for_group(gi)

    def _setup_dense_vscrollbar_for_group(self, gi: int):
        """Configure the vertical scrollbar for a single dense group."""
        if gi >= len(self.dense_groups) or gi >= len(self.dense_plots) or gi >= len(self.dense_vscrollbars):
            return
        sb = self.dense_vscrollbars[gi]
        offsets = self._dense_offsets(gi)
        if len(offsets) < 2:
            sb.hide()
            return
        group = self.dense_groups[gi]
        tpp = group.traces_per_page
        if tpp is None or tpp >= len(offsets):
            sb.hide()
            return
        margin = self._dense_offset_margin(offsets)
        total_min = float(offsets.min()) - margin
        total_max = float(offsets.max()) + margin

        plt = self.dense_plots[gi]
        y_range = plt.getViewBox().viewRange()[1]
        visible_span = y_range[1] - y_range[0]

        scale = 100.0
        sb.blockSignals(True)
        sb.setMinimum(int(total_min * scale))
        sb.setMaximum(int((total_max - visible_span) * scale))
        sb.setPageStep(int(visible_span * scale))
        sb.setSingleStep(int(scale))
        if group.descending:
            sb.setValue(sb.maximum() - int(y_range[0] * scale) + sb.minimum())
        else:
            sb.setValue(int(y_range[0] * scale))
        sb.blockSignals(False)
        sb.show()
        self._dense_vscroll_inverted[gi] = group.descending

    def _on_dense_vscrollbar_changed(self, gi: int, value: int):
        """Handle dense vertical scrollbar value change for group *gi*."""
        if gi >= len(self.dense_groups) or gi >= len(self.dense_plots) or gi >= len(self.dense_vscrollbars):
            return
        plt = self.dense_plots[gi]
        vb = plt.getViewBox()
        y_range = vb.viewRange()[1]
        visible_span = y_range[1] - y_range[0]
        sb = self.dense_vscrollbars[gi]
        if self._dense_vscroll_inverted[gi]:
            new_min = (sb.maximum() - value + sb.minimum()) / 100.0
        else:
            new_min = value / 100.0
        vb.setYRange(new_min, new_min + visible_span, padding=0)

    def _sync_dense_vscrollbar_from_yrange(self, gi: int | None = None):
        """Update scrollbar position to match dense plot Y-range.  gi=None syncs all."""
        if gi is not None:
            self._sync_one_dense_vscrollbar(gi)
        else:
            for i in range(len(self.dense_vscrollbars)):
                self._sync_one_dense_vscrollbar(i)

    def _sync_one_dense_vscrollbar(self, gi: int):
        if gi >= len(self.dense_vscrollbars) or gi >= len(self.dense_plots):
            return
        sb = self.dense_vscrollbars[gi]
        if not sb.isVisible():
            return
        plt = self.dense_plots[gi]
        y_range = plt.getViewBox().viewRange()[1]
        sb.blockSignals(True)
        if self._dense_vscroll_inverted[gi]:
            sb.setValue(sb.maximum() - int(y_range[0] * 100.0) + sb.minimum())
        else:
            sb.setValue(int(y_range[0] * 100.0))
        sb.blockSignals(False)

    def _dense_vertical_page(self, direction: int):
        """Page a dense plot vertically. Uses hovered plot, or first dense group."""
        if not self.dense_groups or not self.dense_plots:
            return False
        # Find which dense plot to page: hovered, or default to first
        gi = 0
        if self.hovered_plot is not None:
            found = False
            for i, plt in enumerate(self.dense_plots):
                if plt is self.hovered_plot:
                    gi = i
                    found = True
                    break
            if not found and self.hovered_plot in self.plots:
                # Hovered plot is a stacked subplot — let caller handle it
                return False

        plt = self.dense_plots[gi]
        offsets = self._dense_offsets(gi)
        if len(offsets) < 2:
            return True
        vb = plt.getViewBox()
        y_range = vb.viewRange()[1]
        visible_span = y_range[1] - y_range[0]
        scroll_amount = visible_span * direction
        vb.setYRange(
            y_range[0] + scroll_amount,
            y_range[1] + scroll_amount,
            padding=0,
        )
        self._sync_dense_vscrollbar_from_yrange(gi)
        return True

    def _on_dense_vertical_smooth(self, direction: int):
        """Shift+Alt+wheel: smooth vertical scroll on the hovered dense plot."""
        if not self.dense_groups or not self.dense_plots:
            return
        # Find which dense plot to scroll: hovered, or default to first
        gi = 0
        if self.hovered_plot is not None:
            for i, plt in enumerate(self.dense_plots):
                if plt is self.hovered_plot:
                    gi = i
                    break
        plt = self.dense_plots[gi]
        offsets = self._dense_offsets(gi)
        if len(offsets) < 2:
            return
        vb = plt.getViewBox()
        y_range = vb.viewRange()[1]
        # Scroll by ~3 traces per notch
        gap = float(np.median(np.diff(offsets)))
        scroll_amount = gap * 3 * direction
        vb.setYRange(
            y_range[0] + scroll_amount,
            y_range[1] + scroll_amount,
            padding=0,
        )
        self._sync_dense_vscrollbar_from_yrange(gi)

    def _adjust_dense_gain(self, factor: float):
        """Scale gain for all dense groups by *factor*."""
        if not self.dense_groups:
            return
        for group in self.dense_groups:
            group.gain = max(0.001, group.gain * factor)
        self._refresh_dense_curves()
        self._update_status(f"Dense gain: {self.dense_groups[0].gain:.2f}x")

    def _adjust_dense_gain_focused(self, factor: float):
        """Scale gain for the hovered dense group only by *factor*."""
        if not self.dense_groups or self.hovered_plot is None:
            return
        gi = None
        for i, plt in enumerate(self.dense_plots):
            if plt is self.hovered_plot:
                gi = i
                break
        if gi is None:
            return
        group = self.dense_groups[gi]
        group.gain = max(0.001, group.gain * factor)
        self._refresh_dense_curves()
        self._update_status(f"Dense gain [{group.name}]: {group.gain:.2f}x")

    def _refresh_raster_plots(self):
        """Update raster raster plots for current window."""
        if not self.raster_series:
            return

        t0 = self.window_start
        t1 = self.window_start + self.window_len
        height = self.raster_event_height

        for midx, (ms, plt) in enumerate(zip(self.raster_series, self.raster_plots)):
            if not self._is_raster_plot_visible(midx):
                continue

            if midx >= len(self._raster_line_items):
                continue

            ts, ys, als, cats = self._raster_segment_for_window(ms, t0, t1)
            line_items_grid = self._raster_line_items[midx]
            if not line_items_grid:
                continue

            if len(ts) == 0:
                for cat_items in line_items_grid:
                    for line_item in cat_items:
                        line_item.setData([], [], _callSync="off")
                continue

            # Calculate Y positions for each event
            y_centers = ys.astype(float) + 0.5
            y_bottoms = y_centers - height
            y_tops = y_centers + height

            # Apply brightness multiplier to alphas (clamped to 0-1)
            brightness = getattr(self, "raster_brightness", 1.0)
            adjusted_als = np.clip(als * brightness, 0.0, 1.0)

            # Group by alpha levels (quantize to 11 levels 0-10) for efficiency
            alpha_levels = np.round(adjusted_als * (RASTER_ALPHA_LEVEL_COUNT - 1)).astype(
                int
            )

            # Non-categorical series use a single synthetic category (index 0).
            cats_arr = cats if cats is not None else np.zeros(len(ts), dtype=np.int16)

            for cidx, cat_items in enumerate(line_items_grid):
                cat_mask = cats_arr == cidx
                if not np.any(cat_mask):
                    for line_item in cat_items:
                        line_item.setData([], [], _callSync="off")
                    continue
                cat_levels = alpha_levels[cat_mask]
                cat_ts = ts[cat_mask]
                cat_yb = y_bottoms[cat_mask]
                cat_yt = y_tops[cat_mask]
                for alevel, line_item in enumerate(cat_items):
                    amask = cat_levels == alevel
                    if not np.any(amask):
                        line_item.setData([], [], _callSync="off")
                        continue
                    indices = np.where(amask)[0]
                    seg_x = np.repeat(cat_ts[indices], 2)
                    seg_y = np.empty(2 * len(indices))
                    seg_y[0::2] = cat_yb[indices]
                    seg_y[1::2] = cat_yt[indices]
                    line_item.setData(seg_x, seg_y, _callSync="off")

    # ---------- Heatmap plots ----------

    def _is_heatmap_plot_visible(self, plot_idx: int) -> bool:
        return plot_idx >= len(self.heatmap_visible) or self.heatmap_visible[plot_idx]

    def _get_array_lut(self, cmap: "str | Colormap") -> np.ndarray:
        """Return a cached uint8 RGBA LUT for *cmap*.

        Accepts either a matplotlib colormap name (string) or a
        ``matplotlib.colors.Colormap`` instance (e.g. ``cmcrameri.cm.batlow``).
        Strings are resolved via ``matplotlib.colormaps``; Colormap instances
        are used directly. Cached by string name (or by ``id()`` for Colormap
        instances, which are not hashable).
        """
        if isinstance(cmap, str):
            cache_key: object = cmap
        else:
            cache_key = id(cmap)
        cached = self._lut_cache.get(cache_key)
        if cached is not None:
            return cached

        cmap_obj = None
        if isinstance(cmap, str):
            try:
                import matplotlib as mpl  # lazy

                cmap_obj = mpl.colormaps[cmap]
            except Exception:
                try:
                    import matplotlib as mpl  # lazy

                    cmap_obj = mpl.colormaps["magma"]
                except Exception:
                    # Last-ditch fallback: greyscale
                    lut = np.zeros((256, 4), dtype=np.uint8)
                    lut[:, 3] = 255
                    lut[:, 0] = np.arange(256, dtype=np.uint8)
                    lut[:, 1] = np.arange(256, dtype=np.uint8)
                    lut[:, 2] = np.arange(256, dtype=np.uint8)
                    self._lut_cache[cache_key] = lut
                    return lut
        else:
            cmap_obj = cmap

        lut = (cmap_obj(np.linspace(0.0, 1.0, 256)) * 255).astype(np.uint8)
        self._lut_cache[cache_key] = lut
        return lut

    def _decimate_along_time(
        self, Y: np.ndarray, max_cols: int, method: str
    ) -> np.ndarray:
        """Reduce columns of *Y* to at most *max_cols* via peak/mean per bin."""
        if max_cols <= 0 or Y.shape[1] <= max_cols:
            return Y
        factor = Y.shape[1] // max_cols
        if factor <= 1:
            return Y
        n = (Y.shape[1] // factor) * factor
        reshaped = Y[:, :n].reshape(Y.shape[0], -1, factor)
        if method == "mean":
            return reshaped.mean(axis=2)
        return reshaped.max(axis=2)  # peak (sentinel -inf survives)

    def _slice_array_at_window(
        self, asx: HeatmapSeries, i0: int, i1: int, target_w: int
    ) -> tuple[np.ndarray, int, int]:
        """Slice an HeatmapSeries (or its mip-map) to the visible window.

        Returns ``(Y_slice, sliced_i0, sliced_i1)``: the second/third are the
        level-0 column indices the slice covers (used to position the image
        in time-coords).
        """
        if asx.mipmap_levels is None or len(asx.mipmap_levels) <= 1:
            return asx.Y[:, i0:i1], i0, i1
        # Pick the highest level whose factor 2^L is still <= the desired
        # decimation factor (visible_cols / target_w).
        cols = max(1, i1 - i0)
        target = max(1, target_w)
        desired = cols // (target * 2) if target > 0 else 1
        level = 0
        max_level = len(asx.mipmap_levels) - 1
        while level < max_level and (1 << (level + 1)) <= max(1, desired):
            level += 1
        factor = 1 << level
        # Convert level-0 indices to this level's index space.
        l_i0 = i0 // factor
        l_i1 = max(l_i0 + 1, i1 // factor + (1 if (i1 % factor) else 0))
        l_i1 = min(l_i1, asx.mipmap_levels[level].shape[1])
        return asx.mipmap_levels[level][:, l_i0:l_i1], l_i0 * factor, l_i1 * factor

    def _refresh_heatmap_plots(self) -> None:
        """Update heatmap plots for the current window."""
        if not self.heatmap_series:
            return
        t0 = self.window_start
        t1 = self.window_start + self.window_len
        # Make sure cache list keeps pace with the registry length.
        while len(self._heatmap_cache_keys) < len(self.heatmap_series):
            self._heatmap_cache_keys.append(None)

        for i, asx in enumerate(self.heatmap_series):
            if not self._is_heatmap_plot_visible(i):
                continue
            if i >= len(self.heatmap_image_items):
                continue
            image_item = self.heatmap_image_items[i]

            i0 = int(np.searchsorted(asx.t, t0, side="left"))
            i1 = int(np.searchsorted(asx.t, t1, side="right"))
            if i1 - i0 < 2:
                image_item.clear()
                self._heatmap_cache_keys[i] = None
                continue

            target_w = max(
                1, int(self.heatmap_plots[i].getViewBox().width())
            )

            cache_key = (
                i0, i1, target_w,
                float(asx.vmin), float(asx.vmax),
                _colormap_cache_token(asx.colormap), asx.decim_method,
            )
            if self._heatmap_cache_keys[i] == cache_key:
                continue

            Y_slice, sliced_i0, sliced_i1 = self._slice_array_at_window(
                asx, i0, i1, target_w
            )
            Y_disp = self._decimate_along_time(
                Y_slice, target_w * 2, asx.decim_method
            )
            # Manual LUT mapping → uint8 RGBA (Performance Layer 4).
            denom = max(asx.vmax - asx.vmin, 1e-12)
            norm = np.clip((Y_disp - asx.vmin) / denom, 0.0, 1.0)
            # Replace any residual NaNs (from mean decim with all-NaN bins) → 0
            if np.any(np.isnan(norm)):
                norm = np.where(np.isnan(norm), 0.0, norm)
            idx = (norm * 255.0).astype(np.uint8)
            lut = self._get_array_lut(asx.colormap)
            rgba = lut[idx]

            image_item.setImage(rgba, autoLevels=False)

            # Place the image in time coordinates so it co-pans with line plots.
            n_rows = asx.Y.shape[0]
            t_start = float(asx.t[sliced_i0]) if sliced_i0 < len(asx.t) else float(asx.t[0])
            end_idx = min(sliced_i1 - 1, len(asx.t) - 1)
            t_end = float(asx.t[max(end_idx, sliced_i0)])
            width = max(t_end - t_start, 1e-12)
            image_item.setRect(QtCore.QRectF(t_start, 0.0, width, float(n_rows)))

            self._heatmap_cache_keys[i] = cache_key

    # ---------- Video & Static Image ----------
    def _load_video_data(self, slot: VideoSlot, vpath, ft_path):
        self._stop_playback_if_playing()
        slot.is_open = False
        slot.frame_times = None
        slot.requested_frame_idx = None
        slot.video_path = vpath
        slot.frame_times_path = ft_path

        vpaths = [vpath] if isinstance(vpath, str) else list(vpath)
        ft_paths = [ft_path] if isinstance(ft_path, str) else list(ft_path)
        if len(vpaths) != len(ft_paths):
            QtWidgets.QMessageBox.warning(
                self,
                f"{slot.name} config error",
                f"video paths ({len(vpaths)}) and frame_times paths "
                f"({len(ft_paths)}) must be the same length.",
            )
            return
        missing = [p for p in vpaths + ft_paths if not os.path.exists(p)]
        if missing:
            QtWidgets.QMessageBox.warning(
                self,
                "File Not Found",
                f"{slot.name}: missing files: {missing}",
            )
            return

        if len(vpaths) == 1:
            QtCore.QMetaObject.invokeMethod(
                slot.worker,
                "open",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, vpaths[0]),
            )
        else:
            QtCore.QMetaObject.invokeMethod(
                slot.worker,
                "openConcat",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG("QStringList", vpaths),
            )

        try:
            ft_arrays = [np.load(p).astype(float) for p in ft_paths]
            for ft in ft_arrays:
                if ft.ndim != 1:
                    raise ValueError("frame_times.npy must be 1-D")
            ft = ft_arrays[0] if len(ft_arrays) == 1 else np.concatenate(ft_arrays)
            corr = slot.frame_times_correction
            if corr:
                ft = ft + corr
            slot.frame_times = ft
            corr_note = f", {corr:+g}s correction" if corr else ""
            self._update_status(
                f"Loaded frame_times for {slot.name} ({len(ft)} frames{corr_note})."
            )
            self._request_initial_frame()
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, f"{slot.name} frame times error", str(e)
            )
            slot.frame_times = None

    def _on_load_video(self):
        if cv2 is None:
            QtWidgets.QMessageBox.warning(
                self, "Video", "OpenCV (cv2) is not installed."
            )
            return
        if not self.video_slots:
            QtWidgets.QMessageBox.warning(
                self,
                "Video",
                "No video slots configured. Pass videos=[VideoConfig(...)] to view().",
            )
            return
        vpath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select video file",
            filter="Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)",
        )
        if not vpath:
            return

        ft_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select frame_times.npy"
        )
        if not ft_path:
            return

        self._load_video_data(self.video_slots[0], vpath, ft_path)

    def _on_video_opened(self, slot: VideoSlot, ok, msg):
        if not ok:
            slot.is_open = False
            QtWidgets.QMessageBox.warning(self, slot.name, msg or "Failed to open.")
            return
        slot.is_open = True
        slot.requested_frame_idx = None
        if slot.label is not None:
            slot.label.show()
        # Hide the label-summary panel once any non-primary video opens
        # (matches the historical "make room for a second video" behavior).
        if slot.index != 0 and self.interval_label_summary_panel is not None:
            self.interval_label_summary_panel.hide()
        self._request_initial_frame()

    def _request_initial_frame(self):
        for slot in self.video_slots:
            if slot.is_open and slot.frame_times is not None:
                self._set_cursor_time(self.cursor_time, update_slider=(slot.index == 0))

    def _request_video_frame(self, slot: VideoSlot, t: float) -> None:
        ft = slot.frame_times
        if ft is None or len(ft) == 0:
            return

        idx = find_nearest_frame(ft, t)
        if slot.requested_frame_idx == idx:
            return

        slot.requested_frame_idx = idx
        QtCore.QMetaObject.invokeMethod(
            slot.worker,
            "requestFrame",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(int, int(idx)),
        )

    def _schedule_deferred_view_refresh(self, *, update_nav_slider: bool) -> None:
        self._deferred_view_refresh_needs_nav_slider = (
            self._deferred_view_refresh_needs_nav_slider or update_nav_slider
        )
        if not self._deferred_view_refresh_timer.isActive():
            self._deferred_view_refresh_timer.start(0)

    def _flush_deferred_view_refresh(self) -> None:
        update_nav_slider = self._deferred_view_refresh_needs_nav_slider
        self._deferred_view_refresh_needs_nav_slider = False
        self._apply_x_range_core()
        if update_nav_slider:
            self._update_nav_slider_from_window()

    def _on_frame_ready(self, slot: VideoSlot, idx, qimg):
        if idx != slot.requested_frame_idx:
            return
        if qimg is None or qimg.isNull():
            return
        pix = QtGui.QPixmap.fromImage(qimg)
        if pix.isNull():
            return
        slot.last_pixmap = pix
        self._rescale_video_frame(slot)

    def _rescale_video_frame(self, slot: VideoSlot):
        if slot.last_pixmap is None or slot.label is None:
            return
        scaled = slot.last_pixmap.scaled(
            slot.label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        slot.label.setPixmap(scaled)

    def _rescale_all_video_frames(self):
        for slot in self.video_slots:
            self._rescale_video_frame(slot)

    def _on_splitter_moved(self, pos, index):
        self._rescale_all_video_frames()

    # ---------- Selection / labeling ----------
    def _show_y_axis_dialog(self):
        if not self.series:
            QtWidgets.QMessageBox.information(
                self, "Y-Axis Controls", "Load time series data first."
            )
            return
        if self.y_axis_dialog is not None:
            self.y_axis_dialog.deleteLater()

        self.y_axis_dialog = YAxisControlsDialog(self)
        self.y_axis_dialog.show()
        self.y_axis_dialog.raise_()
        self.y_axis_dialog.activateWindow()

    def _show_dense_controls_dialog(self):
        if not self.dense_groups:
            QtWidgets.QMessageBox.information(
                self, "Dense View Controls", "No dense trace groups loaded."
            )
            return
        if hasattr(self, "_dense_ctrl_dialog") and self._dense_ctrl_dialog is not None:
            self._dense_ctrl_dialog.deleteLater()
        self._dense_ctrl_dialog = DenseViewControlsDialog(self)
        self._dense_ctrl_dialog.show()
        self._dense_ctrl_dialog.raise_()
        self._dense_ctrl_dialog.activateWindow()

    def _show_heatmap_controls_dialog(self):
        if not self.heatmap_series:
            QtWidgets.QMessageBox.information(
                self, "Heatmap Plot Controls", "No heatmap plots loaded."
            )
            return
        if hasattr(self, "_heatmap_ctrl_dialog") and self._heatmap_ctrl_dialog is not None:
            self._heatmap_ctrl_dialog.deleteLater()
        self._heatmap_ctrl_dialog = HeatmapControlsDialog(self)
        self._heatmap_ctrl_dialog.show()
        self._heatmap_ctrl_dialog.raise_()
        self._heatmap_ctrl_dialog.activateWindow()

    def _on_plot_hovered(self, plot, is_hovered):
        if is_hovered:
            self.hovered_plot = plot
        else:
            if self.hovered_plot is plot:
                self.hovered_plot = None

    def _on_drag_start(self, x):
        self._stop_playback_if_playing()
        self._select_start = x
        self._select_end = x
        # Determine if this drag is a zoom gesture (Shift held)
        try:
            mods = QtWidgets.QApplication.keyboardModifiers()
        except Exception:
            mods = QtCore.Qt.KeyboardModifier.NoModifier
        self._is_zoom_drag = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
        self._show_active_selection()

    def _on_drag_update(self, x):
        self._select_end = x
        self._show_active_selection()

    def _on_drag_finish(self, x):
        self._select_end = x
        # If this was a Shift+drag, zoom to the selected time range
        if self._is_zoom_drag:
            a = float(min(self._select_start, self._select_end))
            b = float(max(self._select_start, self._select_end))
            if b > a:
                new_len = max(0.1, b - a)
                self.window_len = new_len
                self.window_start = clamp(
                    a,
                    self.t_global_min,
                    max(self.t_global_min, self.t_global_max - self.window_len),
                )
                # Sync UI without triggering change handler
                self.window_spin.blockSignals(True)
                self.window_spin.setValue(self.window_len)
                self.window_spin.blockSignals(False)
                self._apply_x_range()
                self._update_nav_slider_from_window()
            self._is_zoom_drag = False
            self._clear_selection()
            return
        self._show_active_selection(final=True)

    def _on_active_region_dragged(self):
        self._stop_playback_if_playing()
        # Get region from whichever selection region was dragged
        if self.plot_sel_regions:
            a, b = self.plot_sel_regions[0].getRegion()
            self._select_start, self._select_end = float(a), float(b)
        elif self.raster_sel_regions:
            a, b = self.raster_sel_regions[0].getRegion()
            self._select_start, self._select_end = float(a), float(b)
        elif self.heatmap_sel_regions:
            a, b = self.heatmap_sel_regions[0].getRegion()
            self._select_start, self._select_end = float(a), float(b)

    def _show_active_selection(self, final=False):
        if self._select_start is None or self._select_end is None:
            for r in self.plot_sel_regions:
                r.hide()
            for r in self.dense_sel_regions:
                r.hide()
            for r in self.raster_sel_regions:
                r.hide()
            for r in self.heatmap_sel_regions:
                r.hide()
            return
        a = min(self._select_start, self._select_end)
        b = max(self._select_start, self._select_end)
        for r in self.plot_sel_regions:
            r.setRegion((a, b))
            r.show()
        for r in self.dense_sel_regions:
            r.setRegion((a, b))
            r.show()
        for r in self.raster_sel_regions:
            r.setRegion((a, b))
            r.show()
        for r in self.heatmap_sel_regions:
            r.setRegion((a, b))
            r.show()

    def _clear_selection(self):
        self._select_start = None
        self._select_end = None
        for r in self.plot_sel_regions:
            r.hide()
        for r in self.dense_sel_regions:
            r.hide()
        for r in self.raster_sel_regions:
            r.hide()
        for r in self.heatmap_sel_regions:
            r.hide()

    def _interval_label_key(self, row) -> IntervalLabelKey:
        """Stable visual key for a label row (its row_id)."""
        return int(row.row_id)

    def _rebuild_interval_label_index(self) -> None:
        ls = self.interval_label_set
        self._interval_label_keys_in_order = list(int(rid) for rid in ls.row_ids)
        self._interval_label_starts = np.asarray(ls.starts, dtype=float)
        self._interval_label_ends = np.asarray(ls.ends, dtype=float)

    def _visible_interval_label_index_range(self) -> tuple[int, int]:
        if len(self.interval_label_set) == 0 or self.window_len <= 0:
            return (0, 0)
        t0 = float(self.window_start)
        t1 = float(self.window_start + self.window_len)
        return self.interval_label_set.visible_index_range(t0, t1)

    def _visible_interval_label_entries(self) -> list[tuple[IntervalLabelKey, "object"]]:
        start_idx, end_idx = self._visible_interval_label_index_range()
        return [
            (self._interval_label_keys_in_order[idx], self.interval_label_set.row_at_index(idx))
            for idx in range(start_idx, end_idx)
        ]

    def _has_visible_window_interval_label_targets(self) -> bool:
        return (
            any(self._is_trace_plot_visible(idx) for idx in range(len(self.plots)))
            or any(self._is_raster_plot_visible(idx) for idx in range(len(self.raster_plots)))
            or any(self._is_heatmap_plot_visible(idx) for idx in range(len(self.heatmap_plots)))
            or len(self.dense_plots) > 0
        )

    def _remove_graphics_item(self, item) -> None:
        try:
            if item is not None and item.scene():
                item.scene().removeItem(item)
        except Exception:
            pass

    def _add_window_interval_label_visual(self, row) -> None:
        key = self._interval_label_key(row)
        if key in self._interval_label_visuals:
            return

        a, b, name = float(row.start), float(row.end), str(row.label)
        color = self._interval_label_brush_color(name)
        plot_regions: list[tuple[int, pg.LinearRegionItem]] = []
        raster_regions: list[tuple[int, pg.LinearRegionItem]] = []
        dense_regions: list[tuple[int, pg.LinearRegionItem]] = []
        heatmap_regions: list[tuple[int, pg.LinearRegionItem]] = []

        for i, plt in enumerate(self.plots):
            if not self._is_trace_plot_visible(i):
                continue
            reg = pg.LinearRegionItem(
                values=(a, b),
                brush=pg.mkBrush(*color),
                pen=pg.mkPen(*color),
                movable=False,
            )
            reg.setZValue(-20)
            plt.addItem(reg)
            plot_regions.append((i, reg))

        for i, plt in enumerate(self.dense_plots):
            reg = pg.LinearRegionItem(
                values=(a, b),
                brush=pg.mkBrush(*color),
                pen=pg.mkPen(*color),
                movable=False,
            )
            reg.setZValue(-20)
            plt.addItem(reg)
            dense_regions.append((i, reg))

        for i, plt in enumerate(self.raster_plots):
            if not self._is_raster_plot_visible(i):
                continue
            reg = pg.LinearRegionItem(
                values=(a, b),
                brush=pg.mkBrush(*color),
                pen=pg.mkPen(*color),
                movable=False,
            )
            reg.setZValue(-20)
            plt.addItem(reg)
            raster_regions.append((i, reg))

        for i, plt in enumerate(self.heatmap_plots):
            if not self._is_heatmap_plot_visible(i):
                continue
            reg = pg.LinearRegionItem(
                values=(a, b),
                brush=pg.mkBrush(*color),
                pen=pg.mkPen(*color),
                movable=False,
            )
            # Heatmap plots draw their image at z=0; keep label regions on top
            # of the heatmap so they stay visible (-20 would hide them).
            reg.setZValue(20)
            plt.addItem(reg)
            heatmap_regions.append((i, reg))

        if not (plot_regions or raster_regions or dense_regions or heatmap_regions):
            return

        self._interval_label_visuals[key] = IntervalLabelVisualBundle(
            plot_regions=plot_regions,
            raster_regions=raster_regions,
            dense_regions=dense_regions,
            hypnogram_region=None,
            heatmap_regions=heatmap_regions,
            start=a,
            end=b,
            label=name,
        )

    @staticmethod
    def _set_region_color(reg: pg.LinearRegionItem, color: tuple) -> None:
        """Apply an RGBA ``color`` to a region's fill and boundary lines."""
        reg.setBrush(pg.mkBrush(*color))
        pen = pg.mkPen(*color)
        for line in reg.lines:
            line.setPen(pen)

    def _update_window_interval_label_visual(
        self, bundle: IntervalLabelVisualBundle, row
    ) -> None:
        """Reposition/recolor an existing bundle in place if ``row``'s geometry
        or label changed since the bundle was drawn.

        Keyed-by-row_id sync removes vanished rows and adds brand-new ones, but
        a row whose ``row_id`` survives an edit while its span shrinks/moves (a
        partial overwrite splitting an epoch, or ``merge_adjacent`` extending a
        neighbour) keeps its bundle. Without this, the region would keep drawing
        its stale span — the doubled/overlapping overlays. Cheap no-op (just
        three comparisons) in the common navigation case where nothing changed.
        """
        a, b, name = float(row.start), float(row.end), str(row.label)
        geom_changed = bundle.start != a or bundle.end != b
        label_changed = bundle.label != name
        if not (geom_changed or label_changed):
            return
        color = self._interval_label_brush_color(name) if label_changed else None
        for regions in (
            bundle.plot_regions,
            bundle.dense_regions,
            bundle.raster_regions,
            bundle.heatmap_regions,
        ):
            for _i, reg in regions:
                if geom_changed:
                    reg.setRegion((a, b))
                if color is not None:
                    self._set_region_color(reg, color)
        bundle.start, bundle.end, bundle.label = a, b, name

    def _remove_window_interval_label_visual(self, key: IntervalLabelKey) -> None:
        bundle = self._interval_label_visuals.pop(key, None)
        if bundle is None:
            return

        for _i, item in bundle.plot_regions:
            self._remove_graphics_item(item)

        for _i, item in bundle.dense_regions:
            self._remove_graphics_item(item)

        for _i, item in bundle.raster_regions:
            self._remove_graphics_item(item)

        for _i, item in bundle.heatmap_regions:
            self._remove_graphics_item(item)

    def _add_hypnogram_interval_label_visual(self, row) -> None:
        key = self._interval_label_key(row)
        if key in self._hypnogram_interval_label_visuals or self.hypnogram_plot is None:
            return

        a, b, name = float(row.start), float(row.end), str(row.label)
        color = self._interval_label_brush_color(name)
        region = pg.LinearRegionItem(
            values=(a, b),
            brush=pg.mkBrush(*color),
            pen=pg.mkPen(*color),
            movable=False,
        )
        region.setZValue(-10)
        self.hypnogram_plot.addItem(region)
        self._hypnogram_interval_label_visuals[key] = region
        self._hypnogram_interval_label_drawn[key] = (a, b, name)

    def _update_hypnogram_interval_label_visual(self, key: IntervalLabelKey, row) -> None:
        """Reposition/recolor an existing hypnogram region if ``row`` changed
        in place. Mirror of :meth:`_update_window_interval_label_visual` for the
        bare LinearRegionItems tracked in the hypnogram dict."""
        region = self._hypnogram_interval_label_visuals.get(key)
        if region is None:
            return
        a, b, name = float(row.start), float(row.end), str(row.label)
        drawn = self._hypnogram_interval_label_drawn.get(key)
        if drawn == (a, b, name):
            return
        if drawn is None or drawn[0] != a or drawn[1] != b:
            region.setRegion((a, b))
        if drawn is None or drawn[2] != name:
            self._set_region_color(region, self._interval_label_brush_color(name))
        self._hypnogram_interval_label_drawn[key] = (a, b, name)

    def _remove_hypnogram_interval_label_visual(self, key: IntervalLabelKey) -> None:
        self._hypnogram_interval_label_drawn.pop(key, None)
        region = self._hypnogram_interval_label_visuals.pop(key, None)
        if region is None:
            return
        self._remove_graphics_item(region)

    def _clear_window_interval_label_visuals(self) -> None:
        for key in list(self._interval_label_visuals):
            self._remove_window_interval_label_visual(key)

    def _clear_hypnogram_interval_label_visuals(self) -> None:
        for key in list(self._hypnogram_interval_label_visuals):
            self._remove_hypnogram_interval_label_visual(key)
        self._hypnogram_interval_label_drawn.clear()

    def _clear_all_interval_label_visuals(self) -> None:
        self._clear_window_interval_label_visuals()
        self._clear_hypnogram_interval_label_visuals()

    # ---------------- Global event marker visuals ----------------

    def _resolve_global_event_styles(self) -> None:
        """Populate :attr:`_resolved_event_styles` from the user config.

        Keyed by class value (or the sentinel ``None`` for the single-style
        case).  Each value is a fully-resolved style dict
        ``{line_color, line_style, line_width, line_alpha}``.

        Auto-defaults cycle through distinct line styles first, then add
        colors — a 6-color × 5-style palette covers 30 unique classes
        before any combination repeats.
        """
        import warnings

        cfg = self.global_events
        if cfg is None:
            self._resolved_event_styles = {}
            return

        def _validate_style(style: dict, ctx: str) -> dict:
            unknown = set(style) - _GLOBAL_EVENT_VALID_STYLE_KEYS
            if unknown:
                warnings.warn(
                    f"GlobalEventsConfig.style_kwargs[{ctx}] has unknown keys "
                    f"{sorted(unknown)!r}; valid keys: "
                    f"{sorted(_GLOBAL_EVENT_VALID_STYLE_KEYS)!r}.",
                    stacklevel=2,
                )
            if "line_style" in style:
                ls = style["line_style"]
                if ls not in _GLOBAL_EVENT_LINE_STYLE_TO_QT:
                    raise ValueError(
                        f"GlobalEventsConfig.style_kwargs[{ctx}].line_style="
                        f"{ls!r} is not one of "
                        f"{sorted(_GLOBAL_EVENT_LINE_STYLE_TO_QT)!r}."
                    )
            out = {k: v for k, v in style.items() if k in _GLOBAL_EVENT_VALID_STYLE_KEYS}
            if "line_color" in out:
                out["line_color"] = _parse_global_event_color(out["line_color"])
            return out

        if cfg.style_events_on is None:
            self._resolved_event_styles = {
                None: dict(_GLOBAL_EVENT_SINGLE_DEFAULT)
            }
            return

        try:
            uniques = sorted(
                cfg.data[cfg.style_events_on].unique().to_list(),
                key=lambda v: (v is None, repr(v)),
            )
        except Exception:
            uniques = list(cfg.data[cfg.style_events_on].unique().to_list())

        n_styles = len(_GLOBAL_EVENT_STYLE_ORDER)
        n_colors = len(_GLOBAL_EVENT_COLOR_CYCLE)
        user_kwargs = cfg.style_kwargs or {}
        resolved: dict = {}
        for i, val in enumerate(uniques):
            style_idx = i % n_styles
            color_idx = (i // n_styles) % n_colors
            base = {
                "line_color": _GLOBAL_EVENT_COLOR_CYCLE[color_idx],
                "line_style": _GLOBAL_EVENT_STYLE_ORDER[style_idx],
                "line_width": 1.5,
                "line_alpha": 200,
            }
            if val in user_kwargs:
                override = _validate_style(user_kwargs[val], repr(val))
                base.update(override)
            resolved[val] = base
        self._resolved_event_styles = resolved

    def _global_event_pen(self, style: dict) -> QtGui.QPen:
        """Build a :class:`QtGui.QPen` from a resolved style dict."""
        r, g, b = _parse_global_event_color(style["line_color"])
        a = int(max(0, min(255, style["line_alpha"])))
        qcolor = QtGui.QColor(r, g, b, a)
        pen = pg.mkPen(qcolor, width=float(style["line_width"]))
        pen.setStyle(_GLOBAL_EVENT_LINE_STYLE_TO_QT[style["line_style"]])
        # Cosmetic ensures the dash pattern stays visually crisp regardless
        # of view transform; otherwise dashed lines can disappear on zoom.
        pen.setCosmetic(True)
        return pen

    def _global_event_panes(self) -> list[pg.PlotItem]:
        """Return every visible plot pane an event marker should appear on."""
        panes: list[pg.PlotItem] = []
        for i, plt in enumerate(self.plots):
            if self._is_trace_plot_visible(i):
                panes.append(plt)
        panes.extend(self.dense_plots)
        for i, plt in enumerate(self.raster_plots):
            if self._is_raster_plot_visible(i):
                panes.append(plt)
        for i, plt in enumerate(self.heatmap_plots):
            if self._is_heatmap_plot_visible(i):
                panes.append(plt)
        return panes

    def _sync_global_event_visuals(self, *, force_rebuild: bool = False) -> None:
        """Render/refresh vertical event-marker lines across every pane.

        Lines are grouped by class in
        :attr:`_global_event_lines_by_class` so the Style dialog can
        restyle them en masse.  Called after every plot rebuild.
        """
        if self.global_events is None:
            if force_rebuild:
                self._clear_global_event_visuals()
            return

        if force_rebuild or not self._global_event_lines_by_class:
            self._clear_global_event_visuals()
        else:
            return

        cfg = self.global_events
        df = cfg.data
        times_all = df[cfg.event_times_column].to_numpy()

        if cfg.style_events_on is None:
            groups: dict = {None: times_all}
        else:
            classes = df[cfg.style_events_on].to_numpy()
            groups = {}
            for val in self._resolved_event_styles:
                mask = classes == val
                if mask.any():
                    groups[val] = times_all[mask]

        panes = self._global_event_panes()
        if not panes:
            return

        for class_val, times in groups.items():
            style = self._resolved_event_styles.get(class_val)
            if style is None:
                continue
            pen = self._global_event_pen(style)
            bucket = self._global_event_lines_by_class.setdefault(class_val, [])
            for t in times:
                for pane in panes:
                    ln = pg.InfiniteLine(
                        pos=float(t), angle=90, movable=False, pen=pen,
                    )
                    ln.setZValue(_GLOBAL_EVENT_Z)
                    pane.addItem(ln)
                    bucket.append(ln)

    def _clear_global_event_visuals(self) -> None:
        """Remove every InfiniteLine produced by global events."""
        for bucket in self._global_event_lines_by_class.values():
            for ln in bucket:
                self._remove_graphics_item(ln)
        self._global_event_lines_by_class.clear()

    def _apply_global_event_class_style(self, class_val) -> None:
        """Restyle every line for one class after a dialog change."""
        style = self._resolved_event_styles.get(class_val)
        if style is None:
            return
        pen = self._global_event_pen(style)
        for ln in self._global_event_lines_by_class.get(class_val, []):
            ln.setPen(pen)

    def _refresh_interval_label_summary(self, force: bool = False) -> None:
        panel = getattr(self, "interval_label_summary_panel", None)
        if panel is None:
            return
        if force or not panel.isHidden():
            panel.refresh()

    def _sync_hypnogram_interval_label_visuals(self, *, force_rebuild: bool = False) -> None:
        if force_rebuild:
            self._clear_hypnogram_interval_label_visuals()
        else:
            new_key_set = set(self._interval_label_keys_in_order)
            for key in list(self._hypnogram_interval_label_visuals):
                if key not in new_key_set:
                    self._remove_hypnogram_interval_label_visual(key)

        for row in self.interval_label_set:
            key = self._interval_label_key(row)
            if key not in self._hypnogram_interval_label_visuals:
                self._add_hypnogram_interval_label_visual(row)
            else:
                self._update_hypnogram_interval_label_visual(key, row)

    def _sync_window_interval_label_visuals(self, *, force_rebuild: bool = False) -> None:
        if force_rebuild or not self._has_visible_window_interval_label_targets():
            self._clear_window_interval_label_visuals()
            if not self._has_visible_window_interval_label_targets():
                return

        visible_entries = self._visible_interval_label_entries()
        new_key_set = {key for key, _ in visible_entries}
        for key in list(self._interval_label_visuals):
            if key not in new_key_set:
                self._remove_window_interval_label_visual(key)

        for key, row in visible_entries:
            bundle = self._interval_label_visuals.get(key)
            if bundle is None:
                self._add_window_interval_label_visual(row)
            else:
                # Surviving row whose span/label may have changed in place
                # (partial overwrite / merge) — reposition it instead of
                # leaving a stale region behind.
                self._update_window_interval_label_visual(bundle, row)

    def _sync_interval_label_visuals(
        self,
        *,
        force_rebuild: bool = False,
        refresh_summary: bool = True,
        force_rebuild_window: bool = False,
    ) -> None:
        self._sync_hypnogram_interval_label_visuals(force_rebuild=force_rebuild)
        self._sync_window_interval_label_visuals(
            force_rebuild=force_rebuild or force_rebuild_window
        )

        if refresh_summary:
            self._refresh_interval_label_summary()

    def _finalize_interval_label_change(
        self, *, force_rebuild: bool = False, refresh_summary: bool = True
    ) -> None:
        self._merge_adjacent_same_interval_labels()
        self._rebuild_interval_label_index()
        self._sync_interval_label_visuals(
            force_rebuild=force_rebuild, refresh_summary=refresh_summary
        )

    def _add_new_interval_label(self, start, end, label):
        """Adds new label, overwriting/modifying existing ones in the range."""
        try:
            self.interval_label_set.add(float(start), float(end), str(label))
        except ValueError:
            return
        self._finalize_interval_label_change()

    def _clear_interval_labels_in_range(self, start: float, end: float):
        """Remove any labels overlapping [start, end). Preserve non-overlapping parts.

        If an existing labeled epoch partially overlaps the range, it is split
        and only the overlapping part is removed.
        """
        self.interval_label_set.clear_range(float(start), float(end))
        self._finalize_interval_label_change()

    def _merge_adjacent_same_interval_labels(self, adjacency_eps: float = 1e-9):
        self.interval_label_set.merge_adjacent(eps=adjacency_eps)

    def _redraw_all_interval_labels(self):
        """Force a full rebuild of all visual label regions."""
        self._sync_interval_label_visuals(force_rebuild=True)

    def _set_hypnogram_window_marker(self, a: float, b: float) -> None:
        lines = self._hypnogram_window_marker_lines
        if not lines:
            return
        lines[0].setPos(a)
        lines[1].setPos(a)
        lines[2].setPos(b)
        lines[3].setPos(b)

    def _update_hypnogram_extents(self):
        if self.hypnogram_plot is None:
            return
        # Keep current zoom mode when extents change
        if not self.hypnogram_zoomed:
            self.hypnogram_plot.enableAutoRange("x", False)
            self.hypnogram_plot.setXRange(
                self.t_global_min, self.t_global_max, padding=0
            )
        else:
            self._update_hypnogram_xrange()
        # Ensure the window marker reflects current window
        self._set_hypnogram_window_marker(
            self.window_start, self.window_start + self.window_len
        )

    def _delete_last_label(self):
        if len(self.interval_label_set) == 0:
            return
        # Match the legacy "latest by end time" semantic: pick the row whose
        # end is greatest, regardless of creation order.
        ends = self.interval_label_set.ends
        if len(ends) == 0:
            return
        idx = int(np.argmax(ends))
        row = self.interval_label_set.row_at_index(idx)
        self.interval_label_set.delete_row(row.row_id)
        self._finalize_interval_label_change()
        self._update_status(
            f"Deleted label: {row.label} [{row.start:.3f}, {row.end:.3f}]"
        )

    def _zoom_active_plot_y(self, factor):
        """Zooms the Y-axis of the currently hovered plot."""
        if self.hovered_plot is None:
            return

        plot = self.hovered_plot
        plot.enableAutoRange("y", False)
        vb = plot.getViewBox()
        y_range = vb.viewRange()[1]
        center = (y_range[0] + y_range[1]) / 2.0
        height = (y_range[1] - y_range[0]) * factor
        vb.setYRange(center - height / 2.0, center + height / 2.0, padding=0)

    def _toggle_hypnogram_visibility(self):
        if self.hypnogram_widget is not None:
            self.hypnogram_widget.setVisible(not self.hypnogram_widget.isVisible())

    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        # Single-key state hotkeys and `0` clear are intentionally context-
        # dependent (need an active selection), so they're handled here rather
        # than as menu QActions with global shortcuts. Every other binding
        # lives on a QAction in _build_menu so it's discoverable in the menus.
        ktxt = ev.text().lower()
        key = ev.key()

        if (
            ktxt in self.keymap
            and self._select_start is not None
            and self._select_end is not None
        ):
            self._stop_playback_if_playing()
            label = self.keymap[ktxt]
            a = float(min(self._select_start, self._select_end))
            b = float(max(self._select_start, self._select_end))
            if b > a:
                self._add_new_interval_label(a, b, label)
                self._update_status(f"Labeled {label}: [{a:.3f}, {b:.3f}]")
                self._clear_selection()
                return

        if (
            key == QtCore.Qt.Key.Key_0
            and self._select_start is not None
            and self._select_end is not None
        ):
            self._stop_playback_if_playing()
            a = float(min(self._select_start, self._select_end))
            b = float(max(self._select_start, self._select_end))
            if b > a:
                self._clear_interval_labels_in_range(a, b)
                self._update_status(f"Cleared interval labels in [{a:.3f}, {b:.3f}]")
                self._clear_selection()
                return

        super().keyPressEvent(ev)

    def _toggle_hypnogram_zoom(self):
        self.hypnogram_zoomed = not self.hypnogram_zoomed
        self._update_hypnogram_xrange()

    def _set_frame_step_source(self, which: int):
        self.frame_step_source = int(which)

    def _available_frame_times(self, which: int):
        if 0 <= which < len(self.video_slots):
            return self.video_slots[which].frame_times
        return None

    def _fallback_frame_source(self) -> int:
        # First slot with frame_times loaded, else -1.
        for slot in self.video_slots:
            if slot.frame_times is not None:
                return slot.index
        return -1

    def _step_frame(self, direction: int):
        src = self.frame_step_source
        ft = self._available_frame_times(src)
        if ft is None:
            src = self._fallback_frame_source()
            if src < 0:
                return
            ft = self._available_frame_times(src)
            if ft is None:
                return
        idx = find_nearest_frame(ft, self.cursor_time)
        new_idx = int(np.clip(idx + (1 if direction >= 1 else -1), 0, len(ft) - 1))
        new_t = float(ft[new_idx])
        self._set_cursor_time(new_t, update_slider=True)

    def _update_hypnogram_xrange(self):
        if self.hypnogram_plot is None:
            return
        if not self.hypnogram_zoomed:
            # Show full extent
            self.hypnogram_plot.enableAutoRange("x", False)
            self.hypnogram_plot.setXRange(
                self.t_global_min, self.t_global_max, padding=0
            )
        else:
            # Zoom around current window with +/- padding
            pad = float(self.hypnogram_zoom_padding)
            a = max(self.t_global_min, self.window_start - pad)
            b = min(self.t_global_max, self.window_start + self.window_len + pad)
            if b <= a:
                b = min(self.t_global_max, a + 1.0)
            self.hypnogram_plot.enableAutoRange("x", False)
            self.hypnogram_plot.setXRange(a, b, padding=0)

    # ---------- Export / Import ----------

    _INTERVAL_LABEL_LOAD_FILTER = (
        "Label files (*.csv *.htsv *.parquet *.txt);;"
        "CSV (*.csv);;HTSV (*.htsv);;Parquet (*.parquet);;Visbrain (*.txt)"
    )
    _INTERVAL_LABEL_SAVE_FILTER = (
        "CSV (*.csv);;HTSV (*.htsv);;Parquet (*.parquet)"
    )

    def load_interval_labels(
        self,
        path: str,
        *,
        schema: IntervalLabelSchema | None = None,
        writeback_allowed: bool | None = None,
    ) -> None:
        """Load interval labels from a CSV/HTSV/Parquet/Visbrain file.

        Parameters
        ----------
        path
            Path to the file.
        schema
            Optional :class:`IntervalLabelSchema`. If omitted, inferred for ``.csv``
            and ``.txt``; required for ``.htsv``/``.parquet``.
        writeback_allowed
            If given, override the new IntervalLabelSet's writeback flag. If omitted,
            preserves whatever the previous IntervalLabelSet had.
        """
        wb = (
            self.interval_label_set.writeback_allowed
            if writeback_allowed is None
            else bool(writeback_allowed)
        )
        new_set = IntervalLabelSet.from_path(path, schema=schema, writeback_allowed=wb)
        self.interval_label_set = new_set
        self._finalize_interval_label_change(force_rebuild=True)
        self._update_status(
            f"Loaded {len(self.interval_label_set)} interval labels from {os.path.basename(path)}"
        )

    def _on_load_interval_labels(self):
        self._stop_playback_if_playing()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load interval labels", filter=self._INTERVAL_LABEL_LOAD_FILTER
        )
        if not path:
            return
        try:
            self.load_interval_labels(path)
        except (IntervalLabelIOError, IntervalLabelSchemaError) as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Load error",
                f"Failed to load interval-labels file:\n\n{e}\n\n"
                f"For .htsv/.parquet, pass an explicit IntervalLabelSchema via the "
                f"`interval_labels=` and `interval_label_schema=` kwargs of view().",
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Load error", f"Failed to load or parse interval-labels file:\n\n{e}"
            )

    def _on_export_interval_labels(self):
        """Save interval labels to a new file via Save-As dialog (never overwrites source)."""
        self._stop_playback_if_playing()
        if len(self.interval_label_set) == 0:
            QtWidgets.QMessageBox.information(self, "Export", "No interval labels to export.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export interval labels", filter=self._INTERVAL_LABEL_SAVE_FILTER
        )
        if not path:
            return
        try:
            self.interval_label_set.save_as(path)
            self._update_status(f"Exported interval labels to {path}")
        except (IntervalLabelIOError, IntervalLabelSchemaError) as e:
            QtWidgets.QMessageBox.warning(self, "Export error", str(e))

    def _on_save_to_source(self):
        """Overwrite the original interval-labels file. Requires interval_labels_writeback=True."""
        self._stop_playback_if_playing()
        if not self.interval_label_set.writeback_allowed:
            QtWidgets.QMessageBox.warning(
                self,
                "Save",
                "Save-to-source is disabled. Pass `interval_labels_writeback=True` to "
                "view() to opt in to overwriting the original file.",
            )
            return
        if self.interval_label_set.source_path is None:
            QtWidgets.QMessageBox.warning(
                self, "Save", "No source file recorded; use Export Interval Labels As… instead."
            )
            return
        try:
            self.interval_label_set.save_to_source()
            self._update_status(f"Saved interval labels to {self.interval_label_set.source_path}")
        except (IntervalLabelIOError, IntervalLabelSchemaError) as e:
            QtWidgets.QMessageBox.warning(self, "Save error", str(e))

    # ---------- Navigation / rendering ----------

    def _stop_playback_if_playing(self):
        """Stops playback if it is currently active."""
        if self.is_playing:
            self.is_playing = False
            self.playback_timer.stop()
            self._update_status("Playback stopped.")

    def _toggle_fullscreen(self, checked: bool) -> None:
        if checked:
            self.showFullScreen()
        else:
            self.showNormal()

    def _toggle_playback(self):
        """Toggles video playback on or off."""
        if self.is_playing:
            self._stop_playback_if_playing()
        else:
            primary_ft = (
                self.video_slots[0].frame_times if self.video_slots else None
            )
            if primary_ft is None:
                self._update_status("No video loaded to play.")
                return
            self.is_playing = True
            self.playback_elapsed_timer.start()
            self.playback_timer.start(16)
            self._update_status("Playing...")

    def _advance_playback_frame(self):
        """Called by the QTimer to advance the cursor time."""
        if not self.is_playing:
            return

        dt_ms = self.playback_elapsed_timer.restart()
        dt_sec = (dt_ms / 1000.0) * float(self.playback_speed)

        t_start = self.window_start
        t_end = self.window_start + self.window_len
        if t_end <= t_start:
            return

        new_cursor_time = self.cursor_time + dt_sec

        if new_cursor_time >= t_end:
            new_cursor_time = t_start + (new_cursor_time - t_end)
            if new_cursor_time >= t_end:
                new_cursor_time = t_start

        self._set_cursor_time(new_cursor_time, update_slider=True)

    def _page(self, direction: int):
        self._stop_playback_if_playing()
        direction = 1 if direction >= 1 else -1
        total = self.t_global_max - self.t_global_min
        if total <= 0:
            return
        new_start = self.window_start + direction * self.window_len
        new_start = clamp(
            new_start,
            self.t_global_min,
            max(self.t_global_min, self.t_global_max - self.window_len),
        )
        rel = (
            0.0
            if self.window_len <= 0
            else (self.cursor_time - self.window_start) / self.window_len
        )
        self.window_start = new_start
        self.cursor_time = self.window_start + rel * self.window_len

        self._apply_x_range()
        self._update_nav_slider_from_window()

    def _on_smooth_scroll(self, direction: int):
        self._stop_playback_if_playing()
        direction = 1 if direction >= 1 else -1
        total = self.t_global_max - self.t_global_min
        if total <= 0:
            return
        delta = direction * float(self.smooth_scroll_fraction) * float(self.window_len)
        new_start = self.window_start + delta
        new_start = clamp(
            new_start,
            self.t_global_min,
            max(self.t_global_min, self.t_global_max - self.window_len),
        )
        rel = (
            0.0
            if self.window_len <= 0
            else (self.cursor_time - self.window_start) / self.window_len
        )
        self.window_start = new_start
        self.cursor_time = self.window_start + rel * self.window_len
        self._schedule_deferred_view_refresh(update_nav_slider=True)

    def _on_cursor_wheel(self, dy: int):
        # Adjust the in-window cursor position proportionally to wheel delta
        self._stop_playback_if_playing()
        wl = float(self.window_len)
        if wl <= 0:
            return
        # Typical mouse wheel notch is 120 units. Move ~2% of window per notch.
        step_per_notch = 0.02
        frac = (float(dy) / 120.0) * step_per_notch
        dt = frac * wl
        xr0 = self.window_start
        xr1 = self.window_start + wl
        new_t = clamp(self.cursor_time + dt, xr0, xr1)
        self._set_cursor_time(new_t, update_slider=True)

    def _on_window_len_changed(self, v):
        self._stop_playback_if_playing()
        self.window_len = float(v)
        self.window_start = clamp(
            self.window_start,
            self.t_global_min,
            max(self.t_global_min, self.t_global_max - self.window_len),
        )
        self._apply_x_range()
        self._update_nav_slider_from_window()

    def _on_nav_slider_changed(self, value):
        self._stop_playback_if_playing()
        if self.t_global_max <= self.t_global_min:
            return
        total = max(1e-9, self.t_global_max - self.t_global_min)
        span = max(1e-9, total - self.window_len)
        start = self.t_global_min + (value / 10000.0) * span
        self.window_start = clamp(
            start,
            self.t_global_min,
            max(self.t_global_min, self.t_global_max - self.window_len),
        )
        self._schedule_deferred_view_refresh(update_nav_slider=False)

    def _apply_x_range(self):
        if self._deferred_view_refresh_timer.isActive():
            self._deferred_view_refresh_timer.stop()
        self._deferred_view_refresh_needs_nav_slider = False
        self._apply_x_range_core()

    def _apply_x_range_core(self):
        xr = (self.window_start, self.window_start + self.window_len)
        for plt in self.plots:
            plt.enableAutoRange("x", False)
            plt.setXRange(*xr, padding=0.0)

        # Also apply to dense plots
        for plt in self.dense_plots:
            plt.enableAutoRange("x", False)
            plt.setXRange(*xr, padding=0.0)

        # Also apply to raster plots
        for plt in self.raster_plots:
            plt.enableAutoRange("x", False)
            plt.setXRange(*xr, padding=0.0)

        # Also apply to heatmap plots
        for plt in self.heatmap_plots:
            plt.enableAutoRange("x", False)
            plt.setXRange(*xr, padding=0.0)

        new_cursor_time = clamp(self.cursor_time, xr[0], xr[1])
        self._set_cursor_time(new_cursor_time, update_slider=True)

        self._refresh_curves()
        self._sync_window_interval_label_visuals()

        # Update hypnogram window marker to show current window
        self._set_hypnogram_window_marker(float(xr[0]), float(xr[1]))
        # If zoomed, keep hypnogram centered on the current window +/- padding
        if self.hypnogram_zoomed:
            self._update_hypnogram_xrange()

    def _update_nav_slider_from_window(self):
        if self.t_global_max <= self.t_global_min:
            self.nav_slider.setValue(0)
            return
        total = max(1e-9, self.t_global_max - self.t_global_min)
        span = max(1e-9, total - self.window_len)
        frac = (
            0.0
            if span <= 0
            else clamp((self.window_start - self.t_global_min) / span, 0.0, 1.0)
        )
        self.nav_slider.blockSignals(True)
        self.nav_slider.setValue(int(round(frac * 10000)))
        self.nav_slider.blockSignals(False)

    def _update_cursor_lines(self):
        for ln in self.plot_cur_lines:
            ln.setPos(self.cursor_time)
        for ln in self.dense_cur_lines:
            ln.setPos(self.cursor_time)
        for ln in self.raster_cur_lines:
            ln.setPos(self.cursor_time)
        for ln in self.heatmap_cur_lines:
            ln.setPos(self.cursor_time)

    def _set_cursor_time(self, t, update_slider=True):
        self.cursor_time = t

        self._update_cursor_lines()

        if update_slider:
            self._update_window_cursor_from_cursor_time()

        for slot in self.video_slots:
            self._request_video_frame(slot, self.cursor_time)

        if not self.is_playing:
            self._update_status()

    def _on_window_cursor_changed(self, value):
        self._stop_playback_if_playing()
        frac = value / 10000.0
        t = self.window_start + frac * self.window_len
        self._set_cursor_time(t, update_slider=False)

    def _update_window_cursor_from_cursor_time(self):
        frac = (
            0.0
            if self.window_len <= 0
            else clamp(
                (self.cursor_time - self.window_start) / self.window_len, 0.0, 1.0
            )
        )
        self.window_cursor_slider.blockSignals(True)
        self.window_cursor_slider.setValue(int(round(frac * 10000)))
        self.window_cursor_slider.blockSignals(False)

    def _target_pts(self):
        all_plots = self.plots or self.dense_plots
        if not all_plots:
            return self.max_pts_per_plot
        target_plot = None
        for idx, plt in enumerate(self.plots):
            if self._is_trace_plot_visible(idx):
                target_plot = plt
                break
        if target_plot is None:
            target_plot = all_plots[0]
        vb = target_plot.getViewBox()
        px = max(300, int(vb.width()))
        return int(min(2 * px, self.max_pts_per_plot))

    def _refresh_curves(self):
        t0, t1 = self.window_start, self.window_start + self.window_len
        if self.overlay_mode and self._plot_to_series:
            for plot_idx, series_indices in enumerate(self._plot_to_series):
                if not self._is_trace_plot_visible(plot_idx):
                    continue
                for local_idx, si in enumerate(series_indices):
                    s = self.series[si]
                    curve = self._plot_to_curves[plot_idx][local_idx]
                    i0 = max(0, np.searchsorted(s.t, t0) - 1)
                    i1 = min(len(s.t), np.searchsorted(s.t, t1) + 1)
                    curve.setData(s.t[i0:i1], s.y[i0:i1], _callSync="off")
        else:
            for idx, (s, curve) in enumerate(zip(self.series, self.curves)):
                if not self._is_trace_plot_visible(idx):
                    continue
                i0 = max(0, np.searchsorted(s.t, t0) - 1)
                i1 = min(len(s.t), np.searchsorted(s.t, t1) + 1)
                t_slice = s.t[i0:i1]
                y_slice = s.y[i0:i1]
                curve.setData(t_slice, y_slice, _callSync="off")
                # overlay_curve_items is 1:1 with self.series (empty list when a
                # series has no overlays); a non-empty entry implies a matching
                # self.overlay_series[idx], so this never indexes out of range.
                if idx < len(self.overlay_curve_items) and self.overlay_curve_items[idx]:
                    for oc, o_curve in zip(
                        self.overlay_series[idx], self.overlay_curve_items[idx]
                    ):
                        j0 = max(0, np.searchsorted(oc.t, t0) - 1)
                        j1 = min(len(oc.t), np.searchsorted(oc.t, t1) + 1)
                        o_curve.setData(
                            oc.t[j0:j1], oc.y[j0:j1], _callSync="off"
                        )
                if self.sample_markers and idx < len(self.sample_marker_scatters):
                    for marker_idx, marker in enumerate(self.sample_markers):
                        mask = marker.bool_per_series[idx][i0:i1]
                        self.sample_marker_scatters[idx][marker_idx].setData(
                            x=t_slice[mask], y=y_slice[mask], _callSync="off"
                        )

        # Also refresh dense, raster, and heatmap plots
        self._refresh_dense_curves()
        self._refresh_raster_plots()
        self._refresh_heatmap_plots()

    def _apply_sample_marker_style(self, marker_idx: int) -> None:
        """Push the current style of ``self.sample_markers[marker_idx]`` to every
        scatter item already on screen for that marker set. Used by the
        Adjust Sample Marker Properties dialog to update markers live."""
        if not (0 <= marker_idx < len(self.sample_markers)):
            return
        marker = self.sample_markers[marker_idx]
        sk = _scatter_kwargs_for_marker(marker)
        no_pen = pg.mkPen(0, 0, 0, 0)
        no_brush = pg.mkBrush(0, 0, 0, 0)
        for scatters in self.sample_marker_scatters:
            if marker_idx >= len(scatters):
                continue
            scat = scatters[marker_idx]
            scat.setSymbol(sk["symbol"])
            scat.setSize(sk["size"])
            scat.setPen(sk["pen"] if sk["pen"] is not None else no_pen)
            scat.setBrush(sk["brush"] if sk["brush"] is not None else no_brush)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._rescale_all_video_frames()
        QtCore.QTimer.singleShot(50, self._refresh_curves)
        QtCore.QTimer.singleShot(60, self._align_left_axes)
        # Re-fit heatmap plots when compact mode is on, since the available
        # viewport height changed. Deferred so the layout has settled by the
        # time we measure viewport().height().
        if getattr(self, "compact_heatmaps_to_fit", False) and self.heatmap_series:
            QtCore.QTimer.singleShot(70, self._apply_custom_plot_heights)

    def _pin_bottom_spine(self, vb, spine) -> None:
        """Keep a trace subplot's bottom-boundary line flush with the bottom of
        the current y-range. Inset by ~1px (in data units) so the line is not
        clipped by the viewbox's bottom edge. Connected to ``sigYRangeChanged``
        so it tracks both autoscale and fixed-range plots."""
        try:
            y0, y1 = vb.viewRange()[1]
            inset = (y1 - y0) / max(1, vb.height())
            spine.setPos(y0 + inset)
        except Exception:
            pass

    def _align_left_axes(self):
        """Give every subplot the same left-axis width so the y-spines form one
        straight vertical line across the whole stack.

        The width auto-fits the widest tick label across *all* plot types
        (traces, dense, raster, heatmap) once the layout has painted, then locks
        into ``self._left_axis_width``. Later calls only re-apply the locked
        value, so autoscale relabeling never shifts the spines. The lock is
        reset to ``None`` on full rebuilds (``set_series`` / ``_rebuild_all_plots``)
        so new content re-fits.
        """
        try:
            all_plots = (
                list(self.plots)
                + list(self.dense_plots)
                + list(self.raster_plots)
                + list(self.heatmap_plots)
            )
            if not all_plots:
                return
            if self._left_axis_width is None:
                # Auto-fit: measure natural (auto) widths once the layout has
                # painted, then lock. Axes are in pyqtgraph's default auto-width
                # mode at this point because nothing else sets their width.
                try:
                    painted = self.plot_area.viewport().height() > 0
                except Exception:
                    painted = False
                widths = [int(plt.getAxis("left").width()) for plt in all_plots]
                if not painted or not widths or max(widths) < 20:
                    # Layout not settled yet; a later deferred call will lock it.
                    return
                self._left_axis_width = (
                    max(max(widths), LEFT_AXIS_WIDTH_FLOOR) + LEFT_AXIS_WIDTH_PAD
                )
            for plt in all_plots:
                plt.getAxis("left").setWidth(int(self._left_axis_width))
        except Exception:
            pass

    # ---------- Video size allocation (right panel) ----------
    def _apply_video_stretches(self):
        if self.videos_layout is None:
            return
        try:
            for slot in self.video_slots:
                if slot.label is None:
                    continue
                idx = self.videos_layout.indexOf(slot.label)
                if idx >= 0:
                    self.videos_layout.setStretch(idx, max(0, int(slot.stretch)))
        except Exception:
            pass
        if self.videos_widget is not None:
            self.videos_widget.updateGeometry()
            self.videos_widget.adjustSize()

    def _set_video_stretches(self, stretches: list[int]):
        for slot, s in zip(self.video_slots, stretches):
            slot.stretch = max(0, int(s))
        self._apply_video_stretches()
        QtCore.QTimer.singleShot(0, self._rescale_all_video_frames)

    def _adjust_secondary_video_sizes(self):
        visible_slots = [
            s for s in self.video_slots
            if s.label is not None and s.label.isVisible()
        ]
        if len(visible_slots) < 2:
            QtWidgets.QMessageBox.information(
                self,
                "Adjust Sizes",
                "Need at least two visible videos to adjust sizes.",
            )
            return

        # Snapshot original stretches so Cancel can restore them.
        original = [s.stretch for s in self.video_slots]

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Adjust Video Sizes")
        outer = QtWidgets.QVBoxLayout(dlg)
        outer.addWidget(QtWidgets.QLabel(
            "Layout weight per video (relative — larger = bigger):"
        ))
        form = QtWidgets.QFormLayout()
        outer.addLayout(form)

        spinboxes: list[tuple[VideoSlot, QtWidgets.QSpinBox]] = []
        for slot in visible_slots:
            spin = QtWidgets.QSpinBox()
            spin.setRange(0, 20)
            spin.setValue(int(slot.stretch))
            form.addRow(slot.name, spin)
            spinboxes.append((slot, spin))

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        outer.addWidget(btns)

        def apply_preview(*_):
            new_stretches = list(original)
            for slot, spin in spinboxes:
                new_stretches[slot.index] = spin.value()
            self._set_video_stretches(new_stretches)

        for _, spin in spinboxes:
            spin.valueChanged.connect(apply_preview)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        apply_preview()
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            self._set_video_stretches(original)

    # ---------- Trace visibility ----------
    def _apply_trace_visibility(self):
        # Rebuild the graphics layout based on visibility and subplot order
        try:
            # Remove all plots from layout first
            self.plot_area.clear()
            master_plot = None
            row = 0

            # Get visible plots in order
            visible_plots = self._get_visible_subplot_order()

            # Determine which is the last visible plot for axis labeling
            last_idx = len(visible_plots) - 1

            for i, (plot_type, idx) in enumerate(visible_plots):
                is_last = i == last_idx

                if plot_type == "ts":
                    if idx >= len(self.plots):
                        continue
                    plt = self.plots[idx]
                    plt.setVisible(True)
                    self.plot_area.addItem(plt, row=row, col=0)

                    # Update bottom axis visibility
                    if not is_last:
                        try:
                            plt.setLabel("bottom", "")
                            plt.showAxis("bottom", False)
                        except Exception:
                            pass
                    else:
                        try:
                            plt.showAxis("bottom", True)
                            plt.setLabel(
                                "bottom", "Time", units="s" if is_last else None
                            )
                            bax = plt.getAxis("bottom")
                            bax.setStyle(showValues=True, tickLength=-5)
                            bax.setHeight(None)
                        except Exception:
                            pass

                elif plot_type == "dense":
                    if idx >= len(self.dense_plots):
                        continue
                    plt = self.dense_plots[idx]
                    plt.setVisible(True)
                    self.plot_area.addItem(plt, row=row, col=0)
                    if idx < len(self.dense_vscroll_proxies):
                        self.plot_area.addItem(
                            self.dense_vscroll_proxies[idx], row=row, col=1
                        )

                    if not is_last:
                        try:
                            plt.setLabel("bottom", "")
                            plt.showAxis("bottom", False)
                        except Exception:
                            pass
                    else:
                        try:
                            plt.showAxis("bottom", True)
                            plt.setLabel(
                                "bottom", "Time", units="s" if is_last else None
                            )
                            bax = plt.getAxis("bottom")
                            bax.setStyle(showValues=True, tickLength=-5)
                            bax.setHeight(None)
                        except Exception:
                            pass

                elif plot_type == "heatmap":
                    if idx >= len(self.heatmap_plots):
                        continue
                    plt = self.heatmap_plots[idx]
                    plt.setVisible(True)
                    self.plot_area.addItem(plt, row=row, col=0)

                    if not is_last:
                        try:
                            plt.setLabel("bottom", "")
                            plt.showAxis("bottom", False)
                        except Exception:
                            pass
                    else:
                        try:
                            plt.showAxis("bottom", True)
                            plt.setLabel(
                                "bottom", "Time", units="s" if is_last else None
                            )
                            bax = plt.getAxis("bottom")
                            bax.setStyle(showValues=True, tickLength=-5)
                            bax.setHeight(None)
                        except Exception:
                            pass

                else:  # raster
                    if idx >= len(self.raster_plots):
                        continue
                    plt = self.raster_plots[idx]
                    plt.setVisible(True)
                    self.plot_area.addItem(plt, row=row, col=0)

                    # Update bottom axis visibility
                    if not is_last:
                        try:
                            plt.setLabel("bottom", "")
                            plt.showAxis("bottom", False)
                        except Exception:
                            pass
                    else:
                        try:
                            plt.showAxis("bottom", True)
                            plt.setLabel(
                                "bottom", "Time", units="s" if is_last else None
                            )
                            bax = plt.getAxis("bottom")
                            bax.setStyle(showValues=True, tickLength=-5)
                            bax.setHeight(None)
                        except Exception:
                            pass

                if master_plot is None:
                    master_plot = plt
                else:
                    plt.setXLink(master_plot)
                row += 1

            # Hide plots that are not visible
            for idx, plt in enumerate(self.plots):
                if (
                    not self.trace_visible[idx]
                    if idx < len(self.trace_visible)
                    else False
                ):
                    plt.setVisible(False)
            for idx, plt in enumerate(self.dense_plots):
                if (
                    not self.dense_visible[idx]
                    if idx < len(self.dense_visible)
                    else False
                ):
                    plt.setVisible(False)
            for idx, plt in enumerate(self.raster_plots):
                if (
                    not self.raster_visible[idx]
                    if idx < len(self.raster_visible)
                    else False
                ):
                    plt.setVisible(False)
            for idx, plt in enumerate(self.heatmap_plots):
                if (
                    not self.heatmap_visible[idx]
                    if idx < len(self.heatmap_visible)
                    else False
                ):
                    plt.setVisible(False)

            # Apply custom plot heights
            self._apply_custom_plot_heights()
            self._constrain_scrollbar_column()
            self._update_plot_area_height()
            # Re-apply x-range to keep all linked
            self._apply_x_range()
            if len(self.interval_label_set) > 0:
                self._sync_interval_label_visuals(
                    refresh_summary=False,
                    force_rebuild_window=True,
                )
            self._sync_global_event_visuals(force_rebuild=True)
            QtCore.QTimer.singleShot(0, self._align_left_axes)
        except Exception:
            import traceback

            traceback.print_exc()

    def _set_video_visible(self, which: int, visible: bool):
        if not (0 <= which < len(self.video_slots)):
            return
        slot = self.video_slots[which]
        if slot.label is None:
            return
        slot.label.setVisible(bool(visible))
        self._apply_video_stretches()
        QtCore.QTimer.singleShot(0, self._rescale_all_video_frames)

    # ---------- Help/Status & cleanup ----------

    def _show_help(self):
        self._stop_playback_if_playing()

        def _strip_mnemonic(text: str) -> str:
            # "&Save Labels..." -> "Save Labels...", "&&" -> "&"
            out = []
            i = 0
            while i < len(text):
                ch = text[i]
                if ch == "&" and i + 1 < len(text) and text[i + 1] == "&":
                    out.append("&")
                    i += 2
                    continue
                if ch == "&":
                    i += 1
                    continue
                out.append(ch)
                i += 1
            return "".join(out)

        # Walk the menu bar and collect every QAction that has a shortcut.
        rows_by_menu: list[tuple[str, list[tuple[str, str]]]] = []
        for menu_action in self.menuBar().actions():
            menu = menu_action.menu()
            if menu is None:
                continue
            menu_name = _strip_mnemonic(menu_action.text()) or "Menu"
            rows: list[tuple[str, str]] = []
            for act in menu.actions():
                if act.isSeparator():
                    continue
                seqs = act.shortcuts()
                if not seqs:
                    continue
                label = _strip_mnemonic(act.text()).rstrip(".")
                shortcut_text = " / ".join(
                    s.toString(QtGui.QKeySequence.SequenceFormat.NativeText)
                    for s in seqs
                    if not s.isEmpty()
                )
                if shortcut_text:
                    rows.append((shortcut_text, label))
            if rows:
                rows_by_menu.append((menu_name, rows))

        # State hotkeys (configurable, not menu actions).
        keys_for_state = self.state_config.keys_for_state
        if keys_for_state:
            state_rows = [
                (" / ".join(keys), f"Label selection as {state}")
                for state, keys in sorted(keys_for_state.items())
            ]
        else:
            state_rows = [
                (
                    "—",
                    "No states configured — pass keymap=, label_colors=, or "
                    "state_definitions= to view().",
                )
            ]
        state_rows.append(("0", "Clear interval labels in active selection"))

        # Mouse / wheel interactions (not expressible as QActions).
        mouse_rows = [
            ("Click-drag in any plot", "Create / extend selection"),
            ("Mouse wheel", "Page left/right one full window"),
            ("Shift + wheel", "Smooth scroll window"),
            ("Ctrl + wheel", "Cursor scrub within current window"),
            ("Alt + wheel", "Adjust trace gain (Dense view)"),
            ("Shift + Alt + wheel", "Vertical scroll through traces (Dense view)"),
        ]

        def _table(rows: list[tuple[str, str]]) -> str:
            parts = [
                "<table cellspacing='0' cellpadding='4' "
                "style='border-collapse:collapse;'>"
            ]
            for shortcut, label in rows:
                parts.append(
                    "<tr>"
                    f"<td style='padding-right:18px; white-space:nowrap;'>"
                    f"<b>{shortcut}</b></td>"
                    f"<td>{label}</td>"
                    "</tr>"
                )
            parts.append("</table>")
            return "".join(parts)

        html_parts = ["<h3>Keyboard shortcuts</h3>"]
        for menu_name, rows in rows_by_menu:
            html_parts.append(f"<h4>{menu_name}</h4>")
            html_parts.append(_table(rows))
        html_parts.append("<h4>Selection &amp; labeling</h4>")
        html_parts.append(
            "<p style='margin:0 0 6px 0; color:#888;'>"
            "These keys act on the active selection; create one by click-dragging "
            "in any plot. State hotkeys are configurable via "
            "<code>keymap=</code> or <code>state_definitions=</code>."
            "</p>"
        )
        html_parts.append(_table(state_rows))
        html_parts.append("<h3>Mouse &amp; wheel</h3>")
        html_parts.append(_table(mouse_rows))
        html_parts.append(
            "<p style='margin-top:10px; color:#888;'>"
            "See <code>KEYBINDINGS.md</code> in the repo for the canonical reference."
            "</p>"
        )
        html = "".join(html_parts)

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Shortcuts / Help")
        dlg.resize(560, 640)
        layout = QtWidgets.QVBoxLayout(dlg)
        browser = QtWidgets.QTextBrowser(dlg)
        browser.setOpenExternalLinks(True)
        browser.setHtml(html)
        layout.addWidget(browser)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Close, parent=dlg
        )
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        dlg.exec()

    def _update_status(self, msg=None):
        info = []
        if self.series:
            info += [
                f"{len(self.series)} traces",
                f"t=[{self.t_global_min:.2f},{self.t_global_max:.2f}]s",
            ]
        info += [
            f"win={self.window_len:.2f}s @ {self.window_start:.2f}s",
            self._format_cursor_with_state(),
        ]
        if self.is_playing and not msg:
            msg = "Playing..."
        if msg:
            info.append("| " + msg)
        self.status.showMessage("  ".join(info))

    def _format_cursor_with_state(self):
        row = self.interval_label_set.at_time(self.cursor_time)
        if row is None:
            return f"cursor={self.cursor_time:.3f}s, state='Unlabeled'"

        result = f"cursor={self.cursor_time:.3f}s, state='{row.label}'"
        note = row.note or ""
        if note:
            if len(note) > 40:
                note = note[:37] + "..."
            result += f" | Note: {note}"
        return result

    def _get_state_and_epoch_at_time(self, t):
        """Return the IntervalLabelRow at time ``t``, or ``None`` if unlabeled."""
        return self.interval_label_set.at_time(t)

    def _get_state_at_time(self, t):
        row = self._get_state_and_epoch_at_time(t)
        return row.label if row else None

    def closeEvent(self, ev):
        try:
            self._stop_playback_if_playing()
            for slot in self.video_slots:
                QtCore.QMetaObject.invokeMethod(
                    slot.worker, "stop", QtCore.Qt.QueuedConnection
                )
            for slot in self.video_slots:
                slot.thread.quit()
            for slot in self.video_slots:
                if not slot.thread.wait(1000):
                    slot.thread.terminate()
        except Exception as e:
            print(f"ERROR: Exception during closeEvent: {e}")
        super().closeEvent(ev)

