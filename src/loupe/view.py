"""The :func:`view` entry point — the only function most users call.

Takes :class:`TraceConfig` / :class:`HeatmapConfig` / :class:`RasterConfig`
/ :class:`Zip` / :class:`VideoConfig` instances, dispatches each to its
data converter, and hands the result to :class:`loupe.app.LoupeApp`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loupe.configs import (
    GlobalEventsConfig,
    HeatmapConfig,
    RasterConfig,
    TraceConfig,
    VideoConfig,
    Zip,
)
from loupe.interval_labels import IntervalLabelSchema, IntervalLabelSet
from loupe.state_config import load_state_config

if TYPE_CHECKING:
    import polars as pl

    from loupe.app import LoupeApp


def _resolve_marker_size_alpha(marker) -> tuple[float, int]:
    """Resolve a config :class:`SampleMarkers`' ``None`` size/alpha to defaults.

    ``'o'`` (filled circle) → size 8.0, alpha 110 (semi-transparent fill);
    any other symbol → size 9.0, alpha 255 (solid stroke). Shared by the
    stacked and dense conversion paths so the two never drift.
    """
    size = marker.size if marker.size is not None else (8.0 if marker.marker == "o" else 9.0)
    alpha = marker.alpha if marker.alpha is not None else (110 if marker.marker == "o" else 255)
    return float(size), int(alpha)


def view(
    data: "TraceConfig | HeatmapConfig | RasterConfig | Zip | list | None" = None,
    *,
    window_len: float = 10.0,
    # Video sources
    videos: "VideoConfig | list[VideoConfig] | None" = None,
    # Global event markers (vertical lines across every pane)
    global_events: "GlobalEventsConfig | None" = None,
    # Interval-label loading
    interval_labels: "pl.DataFrame | str | Path | None" = None,
    interval_label_schema: IntervalLabelSchema | None = None,
    interval_labels_writeback: bool = False,
    # State definitions
    state_definitions: str | Path | None = None,
    keymap: dict | None = None,
    label_colors: dict | None = None,
    interval_label_alpha: float | None = None,
    interval_label_overlays: bool = True,
    **kwargs,
) -> "LoupeApp":
    """Launch the Loupe viewer.

    Parameters
    ----------
    data : Config or list of Configs, optional
        One or more of :class:`TraceConfig`, :class:`HeatmapConfig`,
        :class:`RasterConfig`, or :class:`Zip`.  A bare scalar Config is
        accepted as shorthand for a one-element list.  List position
        determines top-to-bottom subplot order.

        Bare ``xr.DataArray`` and ``pl.DataFrame`` inputs are **not**
        accepted — wrap them explicitly in ``TraceConfig(da)`` or
        ``RasterConfig(df)``.
    window_len : float
        Initial time window in seconds (default ``10.0``).
    videos : VideoConfig or list[VideoConfig], optional
        Synchronized video sources for the right panel.  A single
        :class:`VideoConfig` is accepted as shorthand for a one-element
        list.  Both ``video_path`` and ``frame_times_path`` may be lists
        of equal length, in which case the files are loaded as one
        continuous (concatenated) video — see :class:`VideoConfig`.
    global_events : GlobalEventsConfig, optional
        Vertical event-marker lines drawn across every plot pane on top
        of all other layers (including label shading).  Per-class styling
        via :attr:`GlobalEventsConfig.style_events_on` /
        :attr:`GlobalEventsConfig.style_kwargs`; live-editable via the
        View → "Style Global Events…" menu entry.
    interval_labels : pl.DataFrame, str, or Path, optional
        Initial interval labels.  Either a polars DataFrame (requires
        ``interval_label_schema``) or a path to a ``.csv``, ``.htsv``,
        ``.parquet``, or Visbrain ``.txt`` file.
    interval_label_schema : IntervalLabelSchema, optional
        Required when ``interval_labels`` is a DataFrame or an ``.htsv``/
        ``.parquet`` file.
    interval_labels_writeback : bool
        If True, the GUI's "Save Interval Labels (overwrite source)" action
        will overwrite the file passed in ``interval_labels``.  Default False.
    state_definitions : str or Path, optional
        Path to a JSON file with ``"keymap"`` and ``"label_colors"``.
    keymap : dict, optional
        Programmatic state hotkeys; overrides ``state_definitions``.
    label_colors : dict, optional
        Programmatic ``state -> color`` map; overrides
        ``state_definitions``.
    interval_label_alpha : float, optional
        Initial interval-label overlay alpha multiplier in ``[0.0, 1.0]``.
    interval_label_overlays : bool, optional
        Whether to shade label spans across the subplots. Default True. Pass
        False to rely on the pinned label strip / hypnogram instead (the
        overlays can also be toggled at runtime with ``Ctrl+Shift+L``).
    **kwargs
        Forwarded to :class:`LoupeApp` (``fixed_scale``, etc.).

    Returns
    -------
    LoupeApp
        The viewer window.  In Jupyter (with ``%gui qt6``) the window stays
        alive after the call returns.  In a script the call blocks until
        the window is closed.

    Examples
    --------
    In-memory traces::

        view(TraceConfig(da))

    From a zarr store::

        view(TraceConfig.from_path("data.zarr", group="dmd_2",
                                   filter_dict={"syn_id": slice(3, 6)}))

    DataFrame raster::

        view(RasterConfig(ev, time_col="time", order_by="source_id",
                          split_by="dmd", alpha_by="snr_denoised"))

    Mixed layout (list order = top-to-bottom)::

        view([
            TraceConfig(traces),
            RasterConfig(events, time_col="time", order_by="source_id",
                         split_by="dmd"),
            HeatmapConfig(dff, split_by="dend-ID", order_by="pos"),
        ])

    Zip (one subplot per shared coord value)::

        view(Zip(
            [TraceConfig(F), TraceConfig(dFF), TraceConfig(denoised)],
            on="syn_id", colors=["#ff0000", "#00ff00", "#0000ff"],
        ))
    """
    from PySide6 import QtWidgets

    from loupe.app import LoupeApp
    from loupe.df_loader import _parse_raster_color, dataframe_to_raster_series
    from loupe.series import (
        DenseGroup,
        HeatmapSeries,
        OverlayCurve,
        SampleMarkers as _RenderedSampleMarkers,
        Series,
    )
    from loupe.xr_loader import (
        convert_event_arrays_aligned_with,
        convert_overlay_arrays_aligned_with,
        convert_xarray_inputs_overlay,
        convert_xarray_inputs_with_order,
        dataarray_to_heatmaps,
    )

    # ---- normalize data into a list of Configs ----------------------------
    _allowed = (TraceConfig, HeatmapConfig, RasterConfig, Zip)
    if data is None:
        data_list: list = []
    elif isinstance(data, _allowed):
        data_list = [data]
    elif isinstance(data, list):
        data_list = list(data)
    else:
        raise TypeError(
            f"view() data= must be a Config or list of Configs (TraceConfig, "
            f"HeatmapConfig, RasterConfig, Zip), got {type(data).__name__}. "
            f"Wrap bare DataArrays in TraceConfig(da) and bare DataFrames "
            f"in RasterConfig(df)."
        )
    for i, item in enumerate(data_list):
        if not isinstance(item, _allowed):
            raise TypeError(
                f"view() data[{i}] must be TraceConfig / HeatmapConfig / "
                f"RasterConfig / Zip, got {type(item).__name__}. Wrap bare "
                f"DataArrays in TraceConfig(da) and bare DataFrames in "
                f"RasterConfig(df)."
            )

    # ---- cross-Config validation ------------------------------------------
    n_zip = sum(1 for x in data_list if isinstance(x, Zip))
    if n_zip > 1:
        raise ValueError("Only one Zip is supported per window in this release.")
    if n_zip == 1:
        if any(isinstance(x, (TraceConfig, HeatmapConfig)) for x in data_list):
            raise ValueError(
                "Zip cannot coexist with TraceConfig or HeatmapConfig in the "
                "same window. Move the traces into the Zip, or remove the Zip."
            )

    # Sample-marker validation is split by mode. Dense carriers render as one
    # aggregated ScatterPlotItem per group (independent of trace count), so they
    # are self-contained: any number may coexist with each other and with
    # heatmaps / rasters / stacked traces. Stacked carriers keep the original
    # single-carrier / no-coexistence rules (their markers share global app
    # state and assume a single TraceConfig).
    stacked_marker_carriers = [
        x for x in data_list
        if isinstance(x, TraceConfig)
        and x.sample_markers is not None
        and x.mode == "stacked-subplots"
    ]
    if len(stacked_marker_carriers) > 1:
        raise ValueError(
            "Only one stacked-subplots TraceConfig may carry sample_markers per "
            "window."
        )
    if stacked_marker_carriers:
        carrier = stacked_marker_carriers[0]
        others = [x for x in data_list if x is not carrier]
        if any(isinstance(x, (HeatmapConfig, RasterConfig, Zip)) for x in others):
            raise ValueError(
                "A stacked-subplots TraceConfig with sample_markers cannot "
                "coexist with HeatmapConfig / RasterConfig / Zip in the same "
                "window. (Dense-mode markers have no such restriction.)"
            )
        if any(isinstance(x, TraceConfig) for x in others):
            raise ValueError(
                "A stacked-subplots TraceConfig with sample_markers must be the "
                "only TraceConfig in the data list. (Dense-mode markers have no "
                "such restriction.)"
            )

    # ---- Qt application + launch-progress reporter -----------------------
    # Bring the QApplication up before the slow conversion loop so the
    # splash screen can show progress during that work.
    app = QtWidgets.QApplication.instance()
    created_app = False
    if app is None:
        _warn_if_ipython_without_qt()
        app = QtWidgets.QApplication([])
        created_app = True

    from loupe.progress import ProgressReporter, make_splash
    splash = make_splash(app) if created_app else None
    reporter = ProgressReporter(splash=splash)

    # ---- dispatch each Config to its converter ----------------------------
    xr_series: list[Series] = []
    stacked_colors_acc: list = []
    any_stacked_color = False
    bottom_spines_acc: list = []
    any_bottom_spine = False
    dense_list: list[DenseGroup] = []
    heatmap_list: list[HeatmapSeries] = []
    raster_list = []
    overlay_groups = None
    overlay_colors: list | None = None
    order_acc: list[tuple[str, int]] = []
    sample_markers_rendered: list | None = None
    # Per-stacked-series overlay curves (parallel to xr_series). Each entry is
    # a list[OverlayCurve] ([] when that series carries no overlays); the
    # parallel name list holds the host's legend label (None when no overlays).
    overlay_series_acc: list[list] = []
    overlay_main_names_acc: list = []
    any_overlays = False

    def _resolve_array_name(cfg) -> str:
        v = cfg.array_name
        if v is False:
            return ""
        if v is True:
            name = getattr(cfg.data, "name", None)
            if not name:
                raise ValueError(
                    f"{type(cfg).__name__}(array_name=True) requires "
                    f"data.name; set the DataArray's name or pass an "
                    f"explicit string."
                )
            return str(name)
        return str(v)

    reporter.phase("Converting input data")
    for i, item in enumerate(data_list):
        reporter.item(i, len(data_list), detail=type(item).__name__)
        if isinstance(item, Zip):
            das = [t.data for t in item.traces]
            overlay_groups = convert_xarray_inputs_overlay(
                das, item.on, name_prefix=_resolve_array_name(item),
                reporter=reporter,
            )
            overlay_colors = item.colors
        elif isinstance(item, RasterConfig):
            new_ms = dataframe_to_raster_series(
                item.data,
                time_col=item.time_col,
                order_by=item.order_by,
                split_by=item.split_by,
                alpha_by=item.alpha_by,
                array_name=item.array_name,
                palette=item.palette,
                alpha_range=item.alpha_range,
                hue=item.hue,
                horizontal_separators=item.horizontal_separators,
                separator_params=item.separator_params,
                reporter=reporter,
            )
            if item.hue is not None:
                if item.color is not None:
                    import warnings
                    warnings.warn(
                        "RasterConfig: hue takes precedence; "
                        "color is ignored.",
                        stacklevel=2,
                    )
            elif item.color is not None:
                if item.palette is not None:
                    import warnings
                    warnings.warn(
                        "RasterConfig: color takes precedence over palette; "
                        "palette is ignored.",
                        stacklevel=2,
                    )
                resolved = _parse_raster_color(item.color)
                for ms in new_ms:
                    ms.color = resolved
            base = len(raster_list)
            raster_list.extend(new_ms)
            for j in range(len(new_ms)):
                order_acc.append(("raster", base + j))
        elif isinstance(item, HeatmapConfig):
            # Callable array_name is plumbed straight through; everything
            # else resolves to a string prefix here so dataarray_to_heatmaps
            # only sees the two cases it knows about.
            resolved_array_name = (
                item.array_name
                if callable(item.array_name)
                else _resolve_array_name(item)
            )
            new_heatmaps = dataarray_to_heatmaps(
                item.data,
                split_by=item.split_by,
                order_by=item.order_by,
                descending=item.descending,
                cmap=item.cmap,
                vmin=item.vmin,
                vmax=item.vmax,
                decim_method=item.decim_method,
                array_name=resolved_array_name,
                reporter=reporter,
            )
            base = len(heatmap_list)
            heatmap_list.extend(new_heatmaps)
            for j in range(len(new_heatmaps)):
                order_acc.append(("heatmap", base + j))
        else:  # TraceConfig
            cfg = item
            prefix = _resolve_array_name(cfg)
            if cfg.overlay_arrays is not None and cfg.mode != "stacked-subplots":
                raise ValueError(
                    "TraceConfig.overlay_arrays require mode='stacked-subplots'."
                )
            if cfg.mode == "dense":
                tuples, order_vals, trace_labels, color_vals = convert_xarray_inputs_with_order(
                    cfg.data,
                    order_by=cfg.order_by,
                    descending=cfg.descending,
                    name_prefix=prefix,
                    hue=cfg.hue,
                    reporter=reporter,
                )
                series_objs = [Series(n, t, y) for n, t, y in tuples]
                group_name = prefix or (str(cfg.data.name) if cfg.data.name else f"dense_{i}")
                # Sample markers attach to the group. The same sort permutation
                # is applied here as for the data (both go through
                # _compute_trace_sort_index), so bool_per_series[si] lines up
                # with series_objs[si]. Rendered as one aggregated scatter per
                # marker set — see LoupeApp.dense_marker_scatters.
                dense_markers: list = []
                if cfg.sample_markers is not None:
                    bool_per_marker = convert_event_arrays_aligned_with(
                        cfg.data,
                        [m.bool_array for m in cfg.sample_markers],
                        order_by=cfg.order_by,
                        descending=cfg.descending,
                    )
                    for marker, bps in zip(cfg.sample_markers, bool_per_marker):
                        size, alpha = _resolve_marker_size_alpha(marker)
                        dense_markers.append(
                            _RenderedSampleMarkers(
                                marker=marker.marker,
                                color=marker.color,
                                bool_per_series=bps,
                                size=size,
                                alpha=alpha,
                            )
                        )
                dense_idx = len(dense_list)
                dense_list.append(DenseGroup(
                    name=str(group_name),
                    series=series_objs,
                    trace_labels=trace_labels,
                    order_values=order_vals,
                    color_values=color_vals,
                    palette=cfg.palette,
                    descending=cfg.descending,
                    gain=cfg.gain,
                    step=cfg.step,
                    traces_per_page=cfg.traces_per_page,
                    sample_markers=dense_markers,
                ))
                order_acc.append(("dense", dense_idx))
            elif cfg.mode == "stacked-subplots":
                tuples, _, _, _ = convert_xarray_inputs_with_order(
                    cfg.data,
                    order_by=cfg.order_by,
                    descending=cfg.descending,
                    name_prefix=prefix,
                    hue=cfg.hue,
                    reporter=reporter,
                )
                new_series = [Series(n, t, y) for n, t, y in tuples]
                base = len(xr_series)
                xr_series.extend(new_series)
                for _s in new_series:
                    stacked_colors_acc.append(cfg.color)
                    if cfg.color is not None:
                        any_stacked_color = True
                    bottom_spines_acc.append(cfg.add_bottom_spine)
                    if cfg.add_bottom_spine:
                        any_bottom_spine = True
                    order_acc.append(("ts", base))
                    base += 1
                if cfg.overlay_arrays:
                    aligned = convert_overlay_arrays_aligned_with(
                        cfg.data,
                        cfg.overlay_arrays,
                        order_by=cfg.order_by,
                        descending=cfg.descending,
                    )  # [n_overlay][n_series] of (t, y)
                    n_overlay = len(cfg.overlay_arrays)
                    ov_names = [
                        str(a.name) if getattr(a, "name", None) else f"overlay {k}"
                        for k, a in enumerate(cfg.overlay_arrays)
                    ]
                    palette = LoupeApp._DEFAULT_OVERLAY_COLORS
                    if cfg.overlay_colors is not None:
                        ov_colors = list(cfg.overlay_colors)
                        ov_colors += [
                            palette[k % len(palette)]
                            for k in range(len(ov_colors), n_overlay)
                        ]
                    else:
                        ov_colors = [
                            palette[k % len(palette)] for k in range(n_overlay)
                        ]
                    main_name = (
                        str(cfg.data.name)
                        if getattr(cfg.data, "name", None)
                        else ""
                    )
                    for si, s in enumerate(new_series):
                        overlay_series_acc.append([
                            OverlayCurve(
                                name=ov_names[k],
                                color=ov_colors[k],
                                t=aligned[k][si][0],
                                y=aligned[k][si][1],
                            )
                            for k in range(n_overlay)
                        ])
                        overlay_main_names_acc.append(main_name or s.name or "trace")
                    any_overlays = True
                else:
                    for _s in new_series:
                        overlay_series_acc.append([])
                        overlay_main_names_acc.append(None)
                if cfg.sample_markers is not None:
                    bool_per_marker = convert_event_arrays_aligned_with(
                        cfg.data,
                        [m.bool_array for m in cfg.sample_markers],
                        order_by=cfg.order_by,
                        descending=cfg.descending,
                    )
                    sample_markers_rendered = []
                    for marker, bps in zip(cfg.sample_markers, bool_per_marker):
                        size, alpha = _resolve_marker_size_alpha(marker)
                        sample_markers_rendered.append(
                            _RenderedSampleMarkers(
                                marker=marker.marker,
                                color=marker.color,
                                bool_per_series=bps,
                                size=size,
                                alpha=alpha,
                            )
                        )
            else:
                raise ValueError(
                    f"Unknown TraceConfig.mode={cfg.mode!r} "
                    f"(expected 'stacked-subplots' or 'dense')."
                )

    # Compute subplot_order, forwarded only if it deviates from the default
    # (ts → dense → raster → heatmap) — matches the default ordering inside
    # LoupeApp so callers without mixed input never carry a no-op list.
    default_order = (
        [("ts", k) for k in range(len(xr_series))]
        + [("dense", k) for k in range(len(dense_list))]
        + [("raster", k) for k in range(len(raster_list))]
        + [("heatmap", k) for k in range(len(heatmap_list))]
    )
    config_subplot_order = order_acc if order_acc != default_order else None

    xr_series_out = xr_series or None
    stacked_colors = stacked_colors_acc if any_stacked_color else None
    dense_groups = dense_list or None
    heatmap_series = heatmap_list or None
    raster_series_list = raster_list or None

    # ---- Build main window ------------------------------------------------
    # Resolve the state config (keymap + label colors) up front so any
    # config error surfaces before we build the GUI.
    state_config = load_state_config(
        path=state_definitions,
        keymap=keymap,
        label_colors=label_colors,
    )

    # Build the initial IntervalLabelSet, if any.
    interval_label_set: IntervalLabelSet | None = None
    if interval_labels is not None:
        try:
            import polars as pl_runtime
        except ImportError:  # pragma: no cover - polars is a hard dep
            pl_runtime = None
        if pl_runtime is not None and isinstance(interval_labels, pl_runtime.DataFrame):
            if interval_label_schema is None:
                raise ValueError(
                    "interval_label_schema= is required when interval_labels is a polars DataFrame."
                )
            interval_label_set = IntervalLabelSet.from_dataframe(
                interval_labels,
                interval_label_schema,
                writeback_allowed=interval_labels_writeback,
            )
        else:
            interval_label_set = IntervalLabelSet.from_path(
                interval_labels,
                schema=interval_label_schema,
                writeback_allowed=interval_labels_writeback,
            )

    if stacked_colors is not None and "colors" not in kwargs:
        kwargs["colors"] = stacked_colors
    if config_subplot_order is not None and "subplot_order" not in kwargs:
        kwargs["subplot_order"] = config_subplot_order
    if any_bottom_spine and "bottom_spines" not in kwargs:
        kwargs["bottom_spines"] = bottom_spines_acc

    if videos is None:
        video_configs = []
    elif isinstance(videos, VideoConfig):
        video_configs = [videos]
    else:
        video_configs = list(videos)
        for v in video_configs:
            if not isinstance(v, VideoConfig):
                raise TypeError(
                    f"videos must contain VideoConfig instances, got {type(v).__name__}"
                )

    if global_events is not None:
        if not isinstance(global_events, GlobalEventsConfig):
            raise TypeError(
                f"global_events= must be a GlobalEventsConfig, "
                f"got {type(global_events).__name__}."
            )
        ge_cols = list(global_events.data.columns)
        if global_events.event_times_column not in ge_cols:
            raise ValueError(
                f"GlobalEventsConfig.event_times_column="
                f"{global_events.event_times_column!r} not found in DataFrame "
                f"columns {ge_cols!r}."
            )
        if global_events.style_events_on is not None:
            if global_events.style_events_on not in ge_cols:
                raise ValueError(
                    f"GlobalEventsConfig.style_events_on="
                    f"{global_events.style_events_on!r} not found in DataFrame "
                    f"columns {ge_cols!r}."
                )
            if global_events.style_kwargs is not None:
                uniques = set(
                    global_events.data[global_events.style_events_on]
                    .unique()
                    .to_list()
                )
                stray = set(global_events.style_kwargs) - uniques
                if stray:
                    import warnings
                    warnings.warn(
                        f"GlobalEventsConfig.style_kwargs has keys not present "
                        f"in data[{global_events.style_events_on!r}].unique(): "
                        f"{sorted(stray, key=repr)!r}.",
                        stacklevel=2,
                    )
        elif global_events.style_kwargs is not None:
            import warnings
            warnings.warn(
                "GlobalEventsConfig.style_kwargs is ignored when "
                "style_events_on is None.",
                stacklevel=2,
            )

    reporter.phase("Building main window")
    w = LoupeApp(
        xr_series=xr_series_out,
        raster_series_list=raster_series_list,
        overlay_groups=overlay_groups,
        overlay_colors=overlay_colors,
        dense_groups=dense_groups,
        heatmap_series=heatmap_series,
        window_len=window_len,
        sample_markers=sample_markers_rendered,
        overlay_series=overlay_series_acc if any_overlays else None,
        overlay_main_names=overlay_main_names_acc if any_overlays else None,
        state_config=state_config,
        interval_label_set=interval_label_set,
        interval_label_alpha=interval_label_alpha,
        interval_label_overlays=interval_label_overlays,
        video_configs=video_configs,
        global_events=global_events,
        reporter=reporter,
        **kwargs,
    )
    w.show()
    if splash is not None:
        splash.finish(w)
    reporter.done()

    if created_app:
        import sys
        sys.exit(app.exec())
    else:
        return w


def _warn_if_ipython_without_qt() -> None:
    """Print a hint if we're inside IPython but no Qt loop is running."""
    try:
        ip = get_ipython()  # type: ignore[name-defined]  # noqa: F821
        loop = getattr(ip, "active_eventloop", None)
        if loop not in ("qt", "qt5", "qt6"):
            import warnings
            warnings.warn(
                "No Qt event loop detected. Run '%gui qt6' before calling "
                "view() for interactive use in Jupyter/IPython.",
                stacklevel=3,
            )
    except NameError:
        pass  # Not in IPython
