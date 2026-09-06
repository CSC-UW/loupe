"""Loupe: Multi-trace data viewer for neuroscience."""

from __future__ import annotations

import os as _os

# loupe is written against PySide6. pyqtgraph and IPython's ``%gui qt6`` both
# auto-select a Qt binding and prefer PyQt6 when it is installed alongside
# (it is, as a dependency of napari / cellpose / suite2p in the shared venv);
# once PyQt6 is loaded, PySide6 can no longer be imported in that process.
# Pin both selectors before anything Qt-related is imported. ``setdefault`` so
# an explicit user choice still wins. Import loupe before ``%gui qt6`` /
# ``import pyqtgraph`` for this to take effect.
_os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
_os.environ.setdefault("QT_API", "pyside6")

from loupe.configs import (  # noqa: E402
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
