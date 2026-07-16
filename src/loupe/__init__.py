"""Loupe: Multi-trace data viewer for neuroscience."""

from __future__ import annotations

from loupe.configs import (
    GlobalEventsConfig,
    HeatmapConfig,
    RasterConfig,
    SampleMarkers,
    TraceConfig,
    VideoConfig,
    Zip,
)
from loupe.interval_labels import IntervalLabelSchema, IntervalLabelSet
from loupe.state_config import StateConfig, load_state_config
from loupe.tuner import (
    BoolParam,
    ChoiceParam,
    IntParam,
    Param,
    Tunable,
    tunable,
)
from loupe.view import view
from loupe.view_config import ViewConfig, ViewConfigApplyReport, ViewConfigError

__all__ = [
    "BoolParam",
    "ChoiceParam",
    "GlobalEventsConfig",
    "HeatmapConfig",
    "IntParam",
    "IntervalLabelSchema",
    "IntervalLabelSet",
    "Param",
    "RasterConfig",
    "SampleMarkers",
    "StateConfig",
    "TraceConfig",
    "Tunable",
    "VideoConfig",
    "ViewConfig",
    "ViewConfigApplyReport",
    "ViewConfigError",
    "Zip",
    "tunable",
    "view",
]
