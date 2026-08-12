"""Live variable viewer for a running IPython/Jupyter kernel.

Opens a kernel-resident Qt window that lists the variables in the user
namespace and shows a rich inspector for the selected one:

* Polars DataFrames render in the virtualized :class:`~loupe.loupeDF.DataFrameViewer`
  (sortable table + per-column summary dock).
* xarray objects render as their monospace text repr.
* NumPy arrays get a shape/dtype/stats summary plus a corner preview.
* Everything else gets a bounded, safe ``repr``.
* A **Plots** tab collects every image the kernel displays (matplotlib
  inline figures etc.) into a browsable gallery.
* An **Open in loupe…** button launches the full Loupe viewer on the
  selected object via a config-builder dialog
  (:mod:`loupe.varview_launcher`).

Usage from a kernel with a Qt event loop::

    %gui qt6
    from loupe.varview import varview
    varview()

The window refreshes automatically after every executed cell (via
IPython's ``post_run_cell`` event) and reads objects directly from the
kernel's namespace — nothing is ever copied or serialized. Because it
lives inside the kernel process, it is only responsive while the kernel
is idle (i.e. between cells), exactly like any ``%gui qt6`` window.

Outside IPython you can still open it on an explicit namespace mapping
(``varview(ns=globals())``); it then refreshes only via the Refresh
button / F5.
"""

from __future__ import annotations

import base64
import reprlib
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping

from PySide6 import QtCore, QtGui, QtWidgets

_WINDOW: "VarViewWindow | None" = None

#: Non-underscore IPython bookkeeping names that are never worth listing.
_HIDDEN_NAMES = frozenset({"In", "Out", "exit", "quit", "get_ipython", "display"})

#: Gallery keeps at most this many captured images (oldest dropped).
_MAX_GALLERY_IMAGES = 80

#: Auto-compute ndarray stats only below this element count.
_NDARRAY_STATS_MAX_ELEMS = 20_000_000

_KIND_LABELS = {
    "polars_df": "DataFrame",
    "polars_series": "Series",
    "xr_dataset": "Dataset",
    "xr_dataarray": "DataArray",
    "ndarray": "ndarray",
    "figure": "Figure",
    "scalar": "scalar",
    "str": "str",
    "sequence": "sequence",
    "mapping": "mapping",
    "other": "other",
}

#: Kinds the "Open in loupe…" launcher supports.
_LAUNCHABLE_KINDS = frozenset(
    {"polars_df", "xr_dataset", "xr_dataarray", "ndarray"}
)


# ---------------------------------------------------------------------------
# Namespace scanning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VarInfo:
    """Cheap, display-ready metadata for one namespace entry.

    Built without calling ``repr`` on arbitrary objects and without
    importing any library the user hasn't already imported.
    """

    name: str
    kind: str
    type_name: str
    info: str
    size_bytes: int | None

    @property
    def token(self) -> str:
        """Fingerprint used to decide whether the inspector needs a rebuild."""
        return f"{self.kind}|{self.type_name}|{self.info}|{self.size_bytes}"


def _safe(func: Callable[[], Any]) -> Any:
    try:
        return func()
    except Exception:
        return None


_BOUNDED_REPR = reprlib.Repr()
_BOUNDED_REPR.maxstring = 120
_BOUNDED_REPR.maxother = 120
_BOUNDED_REPR.maxlevel = 3


def _classify(obj: Any) -> VarInfo | None:
    """Classify *obj* into a :class:`VarInfo` (sans name).

    Uses :data:`sys.modules` lookups so scanning never imports a library
    on its own. Returns a VarInfo with an empty name; the scanner fills
    the name in.
    """
    pl = sys.modules.get("polars")
    xr = sys.modules.get("xarray")
    np = sys.modules.get("numpy")
    mpl_figure = sys.modules.get("matplotlib.figure")
    type_name = type(obj).__name__

    if pl is not None and isinstance(obj, pl.DataFrame):
        size = _safe(lambda: int(obj.estimated_size()))
        return VarInfo("", "polars_df", type_name,
                       f"{obj.height:,} rows × {obj.width} cols", size)
    if pl is not None and isinstance(obj, pl.Series):
        size = _safe(lambda: int(obj.estimated_size()))
        return VarInfo("", "polars_series", type_name,
                       f"len {obj.len():,} ({obj.dtype})", size)
    if xr is not None and isinstance(obj, xr.Dataset):
        dims = ", ".join(f"{k}: {v}" for k, v in obj.sizes.items())
        size = _safe(lambda: int(obj.nbytes))
        return VarInfo("", "xr_dataset", type_name,
                       f"{len(obj.data_vars)} vars ({dims})", size)
    if xr is not None and isinstance(obj, xr.DataArray):
        dims = ", ".join(f"{k}: {v}" for k, v in obj.sizes.items())
        size = _safe(lambda: int(obj.nbytes))
        return VarInfo("", "xr_dataarray", type_name,
                       f"({dims}) {obj.dtype}", size)
    if np is not None and isinstance(obj, np.ndarray):
        return VarInfo("", "ndarray", type_name,
                       f"{obj.shape} {obj.dtype}", int(obj.nbytes))
    if mpl_figure is not None and isinstance(obj, mpl_figure.Figure):
        return VarInfo("", "figure", type_name, "matplotlib figure", None)
    if isinstance(obj, bool) or obj is None:
        return VarInfo("", "scalar", type_name, repr(obj), None)
    if isinstance(obj, (int, float, complex)):
        return VarInfo("", "scalar", type_name, repr(obj)[:80], None)
    if isinstance(obj, str):
        preview = obj[:60].replace("\n", "\\n")
        suffix = "…" if len(obj) > 60 else ""
        return VarInfo("", "str", type_name,
                       f"len {len(obj):,}: '{preview}{suffix}'", None)
    if isinstance(obj, (list, tuple, set, frozenset)):
        return VarInfo("", "sequence", type_name, f"len {len(obj):,}", None)
    if isinstance(obj, Mapping):
        return VarInfo("", "mapping", type_name,
                       f"len {len(obj):,}", None)
    # Anything else: type info only. Never repr() an unknown object during
    # a scan — reprs can be arbitrarily slow.
    mod = type(obj).__module__
    info = mod if mod not in ("builtins", "__main__") else ""
    return VarInfo("", "other", type_name, info, None)


def scan_namespace(
    ns: Mapping[str, Any], *, include_all: bool = False
) -> list[VarInfo]:
    """Return sorted :class:`VarInfo` entries for the interesting names in *ns*.

    Skips underscore names, IPython bookkeeping (``In``/``Out``/…), and —
    unless *include_all* — modules, functions, classes, and other
    callables.
    """
    import types

    out: list[VarInfo] = []
    for name in sorted(ns.keys()):
        if name.startswith("_") or name in _HIDDEN_NAMES:
            continue
        try:
            obj = ns[name]
        except Exception:
            continue
        if isinstance(obj, types.ModuleType):
            if not include_all:
                continue
            out.append(VarInfo(name, "other", "module",
                               getattr(obj, "__name__", ""), None))
            continue
        if not include_all and callable(obj) and _classify_is_plain_callable(obj):
            continue
        try:
            info = _classify(obj)
        except Exception:
            info = VarInfo("", "other", type(obj).__name__, "", None)
        if info is not None:
            out.append(VarInfo(name, info.kind, info.type_name,
                               info.info, info.size_bytes))
    return out


def _classify_is_plain_callable(obj: Any) -> bool:
    """True for functions/classes/methods — callable *data* objects are kept.

    A Tunable, a partial-like object, or any instance with ``__call__``
    that also looks like data would be misclassified by ``callable()``
    alone; only skip the unambiguous cases.
    """
    import types

    return isinstance(
        obj,
        (
            types.FunctionType,
            types.BuiltinFunctionType,
            types.MethodType,
            types.LambdaType,
            type,
        ),
    )


def _format_bytes(n: int | None) -> str:
    if n is None:
        return ""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{n} B"


# ---------------------------------------------------------------------------
# Variable table model
# ---------------------------------------------------------------------------


class VarTableModel(QtCore.QAbstractTableModel):
    """Table model over a list of :class:`VarInfo`."""

    HEADERS = ("Name", "Type", "Info", "Size")

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._rows: list[VarInfo] = []

    def set_rows(self, rows: list[VarInfo]) -> None:
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def rows(self) -> list[VarInfo]:
        return self._rows

    def var_at(self, row: int) -> VarInfo | None:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if (
            orientation == QtCore.Qt.Orientation.Horizontal
            and role == QtCore.Qt.ItemDataRole.DisplayRole
        ):
            return self.HEADERS[section]
        return None

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        v = self._rows[index.row()]
        col = index.column()
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return v.name
            if col == 1:
                return _KIND_LABELS.get(v.kind, v.kind) if v.kind != "other" else v.type_name
            if col == 2:
                return v.info
            return _format_bytes(v.size_bytes)
        if role == QtCore.Qt.ItemDataRole.UserRole:
            # Sort role: numeric size, case-insensitive strings elsewhere.
            if col == 3:
                return -1 if v.size_bytes is None else v.size_bytes
            return str(self.data(index, QtCore.Qt.ItemDataRole.DisplayRole)).lower()
        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            return f"{v.name}: {v.type_name}\n{v.info}"
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole and col == 3:
            return (
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
        return None


# ---------------------------------------------------------------------------
# Detail (inspector) widgets
# ---------------------------------------------------------------------------


def _fixed_font() -> QtGui.QFont:
    return QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)


def _text_detail(text: str, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QPlainTextEdit:
    w = QtWidgets.QPlainTextEdit(parent)
    w.setReadOnly(True)
    w.setFont(_fixed_font())
    w.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
    w.setPlainText(text)
    return w


def _bounded_repr(obj: Any, limit: int = 40_000) -> str:
    try:
        text = repr(obj)
    except Exception as exc:  # user __repr__ can raise anything
        return f"<repr() failed: {type(exc).__name__}: {exc}>"
    if len(text) > limit:
        return text[:limit] + f"\n… [truncated at {limit:,} chars]"
    return text


def _ndarray_detail_text(arr) -> str:
    import numpy as np

    lines = [
        f"shape:  {arr.shape}",
        f"dtype:  {arr.dtype}",
        f"size:   {arr.size:,} elements ({_format_bytes(int(arr.nbytes))})",
    ]
    if arr.size and np.issubdtype(arr.dtype, np.number):
        if arr.size <= _NDARRAY_STATS_MAX_ELEMS:
            stats = _safe(
                lambda: (
                    float(np.nanmin(arr)),
                    float(np.nanmax(arr)),
                    float(np.nanmean(arr)),
                    float(np.nanstd(arr)),
                    int(np.count_nonzero(~np.isfinite(arr.astype(float, copy=False))))
                    if np.issubdtype(arr.dtype, np.floating)
                    else 0,
                )
            )
            if stats is not None:
                mn, mx, mean, std, nonfinite = stats
                lines += [
                    f"min:    {mn:.6g}",
                    f"max:    {mx:.6g}",
                    f"mean:   {mean:.6g}",
                    f"std:    {std:.6g}",
                ]
                if nonfinite:
                    lines.append(f"non-finite: {nonfinite:,}")
        else:
            lines.append(
                f"(stats skipped: > {_NDARRAY_STATS_MAX_ELEMS:,} elements)"
            )
    lines.append("")
    lines.append(
        np.array2string(arr, max_line_width=120, threshold=200, edgeitems=4)
    )
    return "\n".join(lines)


def _figure_detail(fig, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QWidget:
    """Render a matplotlib figure to a pixmap inside a scroll area."""
    import io

    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    except Exception as exc:
        return _text_detail(f"<could not render figure: {exc}>", parent)
    pix = QtGui.QPixmap()
    pix.loadFromData(buf.getvalue(), "PNG")
    label = QtWidgets.QLabel()
    label.setPixmap(pix)
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
    area = QtWidgets.QScrollArea(parent)
    area.setWidget(label)
    area.setWidgetResizable(False)
    return area


def build_detail_widget(
    info: VarInfo, obj: Any, parent: QtWidgets.QWidget | None = None
) -> QtWidgets.QWidget:
    """Build the inspector widget for one variable."""
    if info.kind == "polars_df":
        from loupe.loupeDF import DataFrameViewer

        # A QMainWindow embeds fine as a child widget; we get the
        # virtualized table + column-summary dock + toolbar for free.
        viewer = DataFrameViewer(obj, title=info.name)
        viewer.setParent(parent)
        viewer.setWindowFlags(QtCore.Qt.WindowType.Widget)
        return viewer
    if info.kind == "polars_series":
        from loupe.loupeDF import DataFrameViewer

        frame = _safe(lambda: obj.rename(info.name or "series").to_frame())
        if frame is None:
            return _text_detail(_bounded_repr(obj), parent)
        viewer = DataFrameViewer(frame, title=info.name)
        viewer.setParent(parent)
        viewer.setWindowFlags(QtCore.Qt.WindowType.Widget)
        return viewer
    if info.kind in ("xr_dataset", "xr_dataarray"):
        return _text_detail(_bounded_repr(obj, limit=120_000), parent)
    if info.kind == "ndarray":
        text = _safe(lambda: _ndarray_detail_text(obj))
        return _text_detail(text if text is not None else _bounded_repr(obj), parent)
    if info.kind == "figure":
        return _figure_detail(obj, parent)
    if info.kind in ("sequence", "mapping"):
        return _text_detail(_BOUNDED_REPR.repr(obj), parent)
    return _text_detail(_bounded_repr(obj), parent)


class VarDetailPanel(QtWidgets.QWidget):
    """Header (name / type / actions) plus the per-type inspector body."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_name: str | None = None
        self._current_token: str | None = None
        self._ns_getter: Callable[[], Mapping[str, Any]] | None = None

        self._title = QtWidgets.QLabel("")
        f = self._title.font()
        f.setBold(True)
        f.setPointSizeF(f.pointSizeF() + 1)
        self._title.setFont(f)
        self._title.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self._subtitle = QtWidgets.QLabel("")
        self._subtitle.setStyleSheet("color: gray;")

        self._loupe_btn = QtWidgets.QPushButton("Open in loupe…")
        self._loupe_btn.setEnabled(False)
        self._loupe_btn.clicked.connect(self._open_in_loupe)

        self._refresh_btn = QtWidgets.QToolButton()
        self._refresh_btn.setText("⟳")
        self._refresh_btn.setToolTip("Rebuild this inspector view")
        self._refresh_btn.clicked.connect(self.rebuild)

        header = QtWidgets.QHBoxLayout()
        title_col = QtWidgets.QVBoxLayout()
        title_col.setSpacing(0)
        title_col.addWidget(self._title)
        title_col.addWidget(self._subtitle)
        header.addLayout(title_col, stretch=1)
        header.addWidget(self._refresh_btn)
        header.addWidget(self._loupe_btn)

        self._body = QtWidgets.QStackedWidget()
        placeholder = QtWidgets.QLabel("Select a variable")
        placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: gray;")
        self._body.addWidget(placeholder)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(header)
        layout.addWidget(self._body, stretch=1)

    # -- public API ---------------------------------------------------------

    def set_namespace_getter(
        self, getter: Callable[[], Mapping[str, Any]]
    ) -> None:
        self._ns_getter = getter

    @property
    def current_name(self) -> str | None:
        return self._current_name

    @property
    def current_token(self) -> str | None:
        return self._current_token

    def show_var(self, info: VarInfo, obj: Any) -> None:
        """Display *obj* in the inspector body."""
        self._current_name = info.name
        self._current_token = info.token
        self._title.setText(info.name)
        self._subtitle.setText(f"{info.type_name}   {info.info}")
        self._loupe_btn.setEnabled(info.kind in _LAUNCHABLE_KINDS)
        widget = build_detail_widget(info, obj, parent=self)
        self._swap_body(widget)

    def clear(self) -> None:
        self._current_name = None
        self._current_token = None
        self._title.setText("")
        self._subtitle.setText("")
        self._loupe_btn.setEnabled(False)
        placeholder = QtWidgets.QLabel("Select a variable")
        placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: gray;")
        self._swap_body(placeholder)

    def rebuild(self) -> None:
        """Re-fetch the current variable from the namespace and re-render."""
        if self._current_name is None or self._ns_getter is None:
            return
        ns = self._ns_getter()
        if self._current_name not in ns:
            self.clear()
            return
        obj = ns[self._current_name]
        classified = _classify(obj)
        info = VarInfo(
            self._current_name,
            classified.kind,
            classified.type_name,
            classified.info,
            classified.size_bytes,
        )
        self.show_var(info, obj)

    # -- internals ----------------------------------------------------------

    def _swap_body(self, widget: QtWidgets.QWidget) -> None:
        while self._body.count():
            old = self._body.widget(0)
            self._body.removeWidget(old)
            old.setParent(None)  # detach now; deleteLater is deferred
            old.deleteLater()
        self._body.addWidget(widget)
        self._body.setCurrentWidget(widget)

    def _open_in_loupe(self) -> None:
        if self._current_name is None or self._ns_getter is None:
            return
        ns = self._ns_getter()
        obj = ns.get(self._current_name)
        if obj is None:
            return
        from loupe.varview_launcher import open_launcher

        open_launcher(self._current_name, obj, parent=self.window())


# ---------------------------------------------------------------------------
# Plots gallery
# ---------------------------------------------------------------------------


class PlotsGallery(QtWidgets.QWidget):
    """Thumbnail strip + zoomable display of captured display images."""

    count_changed = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self._list = QtWidgets.QListWidget()
        self._list.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self._list.setMovement(QtWidgets.QListView.Movement.Static)
        self._list.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self._list.setIconSize(QtCore.QSize(148, 100))
        self._list.setGridSize(QtCore.QSize(166, 132))
        self._list.setWordWrap(True)
        self._list.setFixedWidth(190)
        self._list.setSpacing(4)
        self._list.currentItemChanged.connect(self._on_item_changed)

        self._image_label = QtWidgets.QLabel("No plots captured yet")
        self._image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("color: gray;")
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidget(self._image_label)
        self._scroll.setWidgetResizable(True)

        self._fit = QtWidgets.QCheckBox("Fit to window")
        self._fit.setChecked(True)
        self._fit.toggled.connect(self._render_current)

        save_btn = QtWidgets.QPushButton("Save PNG…")
        save_btn.clicked.connect(self._save_current)
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.clicked.connect(self.clear)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self._fit)
        controls.addStretch(1)
        controls.addWidget(save_btn)
        controls.addWidget(clear_btn)

        right = QtWidgets.QVBoxLayout()
        right.addLayout(controls)
        right.addWidget(self._scroll, stretch=1)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self._list)
        layout.addLayout(right, stretch=1)

    def count(self) -> int:
        return self._list.count()

    def add_image(self, image_bytes: bytes, label: str) -> None:
        """Append a PNG/JPEG image (raw bytes) to the gallery."""
        pix = QtGui.QPixmap()
        if not pix.loadFromData(image_bytes):
            return
        item = QtWidgets.QListWidgetItem(label)
        item.setIcon(QtGui.QIcon(pix.scaled(
            self._list.iconSize(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )))
        item.setData(QtCore.Qt.ItemDataRole.UserRole, pix)
        self._list.addItem(item)
        while self._list.count() > _MAX_GALLERY_IMAGES:
            self._list.takeItem(0)
        self._list.setCurrentItem(item)
        self.count_changed.emit(self._list.count())

    def clear(self) -> None:
        self._list.clear()
        self._image_label.setPixmap(QtGui.QPixmap())
        self._image_label.setText("No plots captured yet")
        self.count_changed.emit(0)

    # -- internals ----------------------------------------------------------

    def _current_pixmap(self) -> QtGui.QPixmap | None:
        item = self._list.currentItem()
        if item is None:
            return None
        pix = item.data(QtCore.Qt.ItemDataRole.UserRole)
        return pix if isinstance(pix, QtGui.QPixmap) else None

    def _on_item_changed(self, *args) -> None:
        self._render_current()

    def _render_current(self) -> None:
        pix = self._current_pixmap()
        if pix is None:
            return
        self._image_label.setText("")
        avail = self._scroll.viewport().size() - QtCore.QSize(4, 4)
        if self._fit.isChecked() and avail.width() > 50 and avail.height() > 50:
            shown = pix.scaled(
                avail,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        else:
            shown = pix
        self._image_label.setPixmap(shown)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._fit.isChecked():
            self._render_current()

    def _save_current(self) -> None:
        pix = self._current_pixmap()
        if pix is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save plot", "plot.png", "PNG images (*.png)"
        )
        if path:
            pix.save(path, "PNG")


# ---------------------------------------------------------------------------
# Display-publisher tee (captures inline figures)
# ---------------------------------------------------------------------------


class DisplayPubTee:
    """Wraps ``ip.display_pub.publish`` to mirror published images.

    Every ``display_data`` the kernel publishes flows through here; any
    ``image/png`` / ``image/jpeg`` payload is decoded and forwarded to the
    gallery callback, then the original publisher runs untouched. All
    capture work is inside a broad try/except: a failure here must never
    break the user's display.
    """

    def __init__(
        self,
        original: Callable[..., Any],
        on_image: Callable[[bytes, str], None],
        label_provider: Callable[[], str],
    ) -> None:
        self._original = original
        self._on_image = on_image
        self._label_provider = label_provider

    def __call__(self, *args, **kwargs):
        try:
            data = kwargs.get("data")
            if data is None and args:
                data = args[0]
            if isinstance(data, dict):
                payload = data.get("image/png") or data.get("image/jpeg")
                raw = self._decode(payload)
                if raw is not None:
                    self._on_image(raw, self._label_provider())
        except Exception:
            pass
        return self._original(*args, **kwargs)

    @staticmethod
    def _decode(payload: Any) -> bytes | None:
        if payload is None:
            return None
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, str):
            try:
                return base64.b64decode(payload)
            except Exception:
                return None
        return None


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class VarViewWindow(QtWidgets.QMainWindow):
    """The variable-viewer main window.

    Owns no data: every refresh re-reads the namespace via *ns_getter*.
    """

    def __init__(
        self,
        ns_getter: Callable[[], Mapping[str, Any]],
        *,
        ip: Any | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._ns_getter = ns_getter
        self._ip = ip
        self._hooked = False
        self._display_tee: DisplayPubTee | None = None

        self.setWindowTitle("Loupe VarView")
        self.resize(1250, 780)

        # ---- left: filter + table ----------------------------------------
        self._filter = QtWidgets.QLineEdit()
        self._filter.setPlaceholderText("filter variables…")
        self._filter.setClearButtonEnabled(True)

        self._model = VarTableModel(self)
        self._proxy = QtCore.QSortFilterProxyModel(self)
        self._proxy.setSourceModel(self._model)
        self._proxy.setFilterCaseSensitivity(
            QtCore.Qt.CaseSensitivity.CaseInsensitive
        )
        self._proxy.setFilterKeyColumn(0)
        self._proxy.setSortRole(QtCore.Qt.ItemDataRole.UserRole)
        self._filter.textChanged.connect(self._proxy.setFilterFixedString)

        self._table = QtWidgets.QTableView()
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)
        self._table.sortByColumn(0, QtCore.Qt.SortOrder.AscendingOrder)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._table.setAlternatingRowColors(True)
        self._table.setWordWrap(False)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(22)
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.setColumnWidth(0, 150)
        self._table.setColumnWidth(1, 90)
        self._table.setColumnWidth(2, 210)
        self._table.setColumnWidth(3, 70)
        self._table.selectionModel().currentRowChanged.connect(
            self._on_row_changed
        )
        self._table.doubleClicked.connect(self._on_row_double_clicked)

        self._show_all = QtWidgets.QCheckBox("all objects")
        self._show_all.setToolTip(
            "Include modules, functions, and classes in the list"
        )
        self._show_all.toggled.connect(lambda _=None: self.refresh())

        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_F5))
        refresh_btn.clicked.connect(self.refresh)

        left_bottom = QtWidgets.QHBoxLayout()
        left_bottom.addWidget(self._show_all)
        left_bottom.addStretch(1)
        left_bottom.addWidget(refresh_btn)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 4, 8)
        left_layout.addWidget(self._filter)
        left_layout.addWidget(self._table, stretch=1)
        left_layout.addLayout(left_bottom)

        # ---- right: inspector + plots tabs --------------------------------
        self._detail = VarDetailPanel()
        self._detail.set_namespace_getter(self._ns_getter)

        self._gallery = PlotsGallery()
        self._gallery.count_changed.connect(self._update_plots_tab_label)

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.addTab(self._detail, "Inspector")
        self._tabs.addTab(self._gallery, "Plots")

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([560, 690])
        self._splitter = splitter
        self.setCentralWidget(splitter)

        self.statusBar().showMessage("Ready")
        self._restore_geometry()

    # -- refresh ------------------------------------------------------------

    def refresh(self) -> None:
        """Rescan the namespace and update the table (and stale inspector)."""
        try:
            ns = self._ns_getter()
            rows = scan_namespace(ns, include_all=self._show_all.isChecked())
        except Exception as exc:
            self.statusBar().showMessage(f"refresh failed: {exc}", 8000)
            return

        selected = self._selected_name()
        self._model.set_rows(rows)
        if selected is not None:
            self._reselect(selected)

        # Rebuild the inspector if its variable changed shape/identity or
        # vanished. Same-token objects are left alone (no pointless churn).
        name = self._detail.current_name
        if name is not None:
            row = next((r for r in rows if r.name == name), None)
            if row is None:
                self._detail.clear()
            elif row.token != self._detail.current_token:
                obj = ns.get(name)
                if obj is not None:
                    self._detail.show_var(row, obj)

        stamp = datetime.now().strftime("%H:%M:%S")
        self.statusBar().showMessage(
            f"{len(rows)} variables · refreshed {stamp}"
        )

    def add_gallery_image(self, image_bytes: bytes, label: str) -> None:
        self._gallery.add_image(image_bytes, label)

    @property
    def gallery(self) -> PlotsGallery:
        return self._gallery

    @property
    def detail(self) -> VarDetailPanel:
        return self._detail

    @property
    def model(self) -> VarTableModel:
        return self._model

    def select_variable(self, name: str) -> bool:
        """Programmatically select *name* in the table. Returns success."""
        return self._reselect(name, block=False)

    # -- IPython wiring -----------------------------------------------------

    def attach_ipython(self, ip: Any) -> None:
        """Register the post-cell refresh hook and the display-pub tee."""
        if self._hooked:
            return
        self._ip = ip
        ip.events.register("post_run_cell", self._on_post_run_cell)
        try:
            pub = ip.display_pub
            self._display_tee = DisplayPubTee(
                pub.publish,
                self._on_captured_image,
                lambda: self._execution_label(),
            )
            pub.publish = self._display_tee
        except Exception:
            self._display_tee = None
        self._hooked = True

    def detach_ipython(self) -> None:
        if not self._hooked or self._ip is None:
            return
        try:
            self._ip.events.unregister("post_run_cell", self._on_post_run_cell)
        except Exception:
            pass
        try:
            pub = self._ip.display_pub
            if pub.publish is self._display_tee and self._display_tee is not None:
                pub.publish = self._display_tee._original
        except Exception:
            pass
        self._hooked = False

    def _on_post_run_cell(self, result=None) -> None:
        # Must never raise — IPython would report the error after every cell.
        try:
            self.refresh()
        except Exception:
            pass

    def _on_captured_image(self, image_bytes: bytes, label: str) -> None:
        try:
            self._gallery.add_image(image_bytes, label)
        except Exception:
            pass

    def _execution_label(self) -> str:
        stamp = datetime.now().strftime("%H:%M:%S")
        n = getattr(self._ip, "execution_count", None)
        return f"[{n}] {stamp}" if n is not None else stamp

    # -- internals ----------------------------------------------------------

    def _update_plots_tab_label(self, count: int) -> None:
        label = "Plots" if count == 0 else f"Plots ({count})"
        self._tabs.setTabText(self._tabs.indexOf(self._gallery), label)

    def _selected_name(self) -> str | None:
        idx = self._table.currentIndex()
        if not idx.isValid():
            return None
        src = self._proxy.mapToSource(idx)
        info = self._model.var_at(src.row())
        return info.name if info is not None else None

    def _reselect(self, name: str, *, block: bool = True) -> bool:
        for row, info in enumerate(self._model.rows()):
            if info.name == name:
                src_idx = self._model.index(row, 0)
                proxy_idx = self._proxy.mapFromSource(src_idx)
                if proxy_idx.isValid():
                    if block:
                        with QtCore.QSignalBlocker(self._table.selectionModel()):
                            self._table.setCurrentIndex(proxy_idx)
                    else:
                        self._table.setCurrentIndex(proxy_idx)
                    return True
        return False

    def _on_row_double_clicked(self, index: QtCore.QModelIndex) -> None:
        """Double-click on a launchable variable opens the loupe launcher."""
        if not index.isValid():
            return
        src = self._proxy.mapToSource(index)
        info = self._model.var_at(src.row())
        if info is None or info.kind not in _LAUNCHABLE_KINDS:
            return
        try:
            obj = self._ns_getter()[info.name]
        except Exception:
            return
        from loupe.varview_launcher import open_launcher

        open_launcher(info.name, obj, parent=self)

    def _on_row_changed(
        self, current: QtCore.QModelIndex, previous: QtCore.QModelIndex
    ) -> None:
        del previous
        if not current.isValid():
            return
        src = self._proxy.mapToSource(current)
        info = self._model.var_at(src.row())
        if info is None:
            return
        try:
            ns = self._ns_getter()
            obj = ns[info.name]
        except Exception:
            return
        self._detail.show_var(info, obj)

    def _settings(self) -> QtCore.QSettings:
        return QtCore.QSettings("loupe", "varview")

    def _restore_geometry(self) -> None:
        s = self._settings()
        geo = s.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        split = s.value("splitter")
        if split is not None:
            self._splitter.restoreState(split)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        s = self._settings()
        s.setValue("geometry", self.saveGeometry())
        s.setValue("splitter", self._splitter.saveState())
        self.detach_ipython()
        global _WINDOW
        if _WINDOW is self:
            _WINDOW = None
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def varview(ns: Mapping[str, Any] | None = None) -> VarViewWindow:
    """Open (or raise) the live variable viewer.

    Parameters
    ----------
    ns : mapping, optional
        Namespace to browse. Defaults to the running IPython kernel's
        ``user_ns``; required when called outside IPython.

    Returns
    -------
    VarViewWindow
        The viewer window (a singleton — calling again raises the
        existing window).
    """
    global _WINDOW

    ip = None
    try:
        from IPython import get_ipython

        ip = get_ipython()
    except ImportError:
        pass

    if ns is None:
        if ip is None:
            raise RuntimeError(
                "varview() outside IPython requires an explicit namespace: "
                "varview(ns=globals())"
            )
        def ns_getter() -> Mapping[str, Any]:
            return ip.user_ns
    else:
        def ns_getter() -> Mapping[str, Any]:
            return ns

    app = QtWidgets.QApplication.instance()
    if app is None:
        _warn_if_ipython_without_qt()
        app = QtWidgets.QApplication([])

    if _WINDOW is not None:
        _WINDOW.show()
        _WINDOW.raise_()
        _WINDOW.activateWindow()
        _WINDOW.refresh()
        return _WINDOW

    w = VarViewWindow(ns_getter, ip=ip)
    if ip is not None:
        w.attach_ipython(ip)
    w.show()
    w.refresh()
    _WINDOW = w
    return w


def _warn_if_ipython_without_qt() -> None:
    try:
        ip = get_ipython()  # type: ignore[name-defined]  # noqa: F821
    except NameError:
        return
    loop = getattr(ip, "active_eventloop", None)
    if loop not in ("qt", "qt5", "qt6"):
        import warnings

        warnings.warn(
            "No Qt event loop detected. Run '%gui qt6' before calling "
            "varview() for interactive use in Jupyter/IPython.",
            stacklevel=3,
        )


__all__ = [
    "VarInfo",
    "VarViewWindow",
    "scan_namespace",
    "varview",
]
