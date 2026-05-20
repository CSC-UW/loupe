"""Small Qt table viewer for Polars DataFrames.

Use from a notebook with a Qt event loop enabled::

    %gui qt6
    from loupe.loupeDF import view_df

    viewer = view_df(df)

The viewer intentionally lives outside the main Loupe trace viewer.  It shares
only the package dependencies: Polars for tabular data and PySide6 for the UI.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import polars as pl
from PySide6 import QtCore, QtGui, QtWidgets


_OPEN_WINDOWS: list["DataFrameViewer"] = []


@dataclass(frozen=True)
class ColumnSummary:
    """Computed summary for one Polars column."""

    column: str
    metrics: tuple[tuple[str, str], ...]
    value_counts: pl.DataFrame | None
    value_count_limit: int


class PolarsDataFrameModel(QtCore.QAbstractTableModel):
    """A virtual Qt table model backed directly by a Polars DataFrame."""

    sort_failed = QtCore.Signal(str)

    def __init__(self, df: pl.DataFrame, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        _require_polars_dataframe(df)
        self._df = df
        self._refresh_columns()

    @property
    def dataframe(self) -> pl.DataFrame:
        return self._df

    def rowCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return self._df.height

    def columnCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return self._df.width

    def data(
        self,
        index: QtCore.QModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid():
            return None

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            value = self._series[index.column()][index.row()]
            return format_value(value)

        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            dtype = self._dtypes[index.column()]
            if _is_numeric_dtype(dtype):
                return (
                    QtCore.Qt.AlignmentFlag.AlignRight
                    | QtCore.Qt.AlignmentFlag.AlignVCenter
                )
            return QtCore.Qt.AlignmentFlag.AlignVCenter

        if role == QtCore.Qt.ItemDataRole.ForegroundRole:
            value = self._series[index.column()][index.row()]
            if value is None:
                return QtGui.QBrush(QtGui.QColor(130, 130, 130))
            return None

        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            col = self._columns[index.column()]
            dtype = self._dtypes[index.column()]
            value = self._series[index.column()][index.row()]
            return f"{col}\n{dtype}\n{format_value(value, max_len=1000)}"

        return None

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if orientation == QtCore.Qt.Orientation.Horizontal:
            if section < 0 or section >= len(self._columns):
                return None
            if role == QtCore.Qt.ItemDataRole.DisplayRole:
                return self._columns[section]
            if role == QtCore.Qt.ItemDataRole.ToolTipRole:
                return f"{self._columns[section]}\n{self._dtypes[section]}"
            return None

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return str(section)
        return None

    def sort(
        self,
        column: int,
        order: QtCore.Qt.SortOrder = QtCore.Qt.SortOrder.AscendingOrder,
    ) -> None:
        if column < 0 or column >= len(self._columns):
            return

        column_name = self._columns[column]
        descending = order == QtCore.Qt.SortOrder.DescendingOrder
        self.layoutAboutToBeChanged.emit()
        try:
            self._df = self._sort_dataframe(column_name, descending=descending)
        except Exception as exc:
            self.layoutChanged.emit()
            self.sort_failed.emit(f"Could not sort {column_name!r}: {exc}")
            return

        self._refresh_columns()
        self.layoutChanged.emit()

    def sort_by_name(self, column: str, *, descending: bool = False) -> None:
        if column not in self._columns:
            raise ValueError(f"Unknown column {column!r}.")
        order = (
            QtCore.Qt.SortOrder.DescendingOrder
            if descending
            else QtCore.Qt.SortOrder.AscendingOrder
        )
        self.sort(self._columns.index(column), order)

    def _sort_dataframe(self, column: str, *, descending: bool) -> pl.DataFrame:
        try:
            return self._df.sort(
                column,
                descending=descending,
                nulls_last=True,
                maintain_order=True,
            )
        except TypeError:
            return self._df.sort(column, descending=descending, nulls_last=True)

    def _refresh_columns(self) -> None:
        self._columns = list(self._df.columns)
        self._series = [self._df.get_column(name) for name in self._columns]
        self._dtypes = [series.dtype for series in self._series]


class CopyableTableView(QtWidgets.QTableView):
    """QTableView with spreadsheet-style Ctrl+C copying."""

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.matches(QtGui.QKeySequence.StandardKey.Copy):
            self.copy_selection_to_clipboard()
            return
        super().keyPressEvent(event)

    def copy_selection_to_clipboard(self) -> None:
        indexes = self.selectedIndexes()
        if not indexes:
            return

        rows = sorted({idx.row() for idx in indexes})
        cols = sorted({idx.column() for idx in indexes})
        selected = {(idx.row(), idx.column()) for idx in indexes}
        model = self.model()
        lines = []
        for row in rows:
            parts = []
            for col in cols:
                if (row, col) in selected:
                    value = model.data(
                        model.index(row, col),
                        QtCore.Qt.ItemDataRole.DisplayRole,
                    )
                    parts.append("" if value is None else str(value))
                else:
                    parts.append("")
            lines.append("\t".join(parts))
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))


class DataFrameViewer(QtWidgets.QMainWindow):
    """Qt window for viewing and summarizing a Polars DataFrame."""

    def __init__(
        self,
        df: pl.DataFrame,
        *,
        title: str | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        _require_polars_dataframe(df)

        self._source_df = df
        self._model = PolarsDataFrameModel(df, self)
        self._model.sort_failed.connect(self._show_status)
        self._summary: ColumnSummary | None = None
        self._value_count_dialogs: list[ValueCountsDialog] = []

        self.setWindowTitle(
            title or f"LoupeDF - {df.height:,} rows x {df.width:,} cols"
        )
        self.resize(1200, 800)

        self._table = CopyableTableView(self)
        self._table.setModel(self._model)
        self._table.setSortingEnabled(True)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems
        )
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._table.setWordWrap(False)
        self._table.verticalHeader().setDefaultSectionSize(22)
        self._table.horizontalHeader().setDefaultSectionSize(150)
        self._table.horizontalHeader().setMinimumSectionSize(60)
        self._table.horizontalHeader().setSectionsMovable(True)
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.selectionModel().currentColumnChanged.connect(
            self._on_current_column_changed
        )
        self.setCentralWidget(self._table)

        self._column_combo = QtWidgets.QComboBox(self)
        self._column_combo.addItems(df.columns)
        self._column_combo.currentTextChanged.connect(self._on_combo_column_changed)

        self._build_toolbar()
        self._build_summary_dock()
        self._show_status(f"{df.height:,} rows x {df.width:,} columns")

        if df.width:
            self._update_summary(df.columns[0])

    @property
    def dataframe(self) -> pl.DataFrame:
        """The currently displayed DataFrame, including any active sort order."""

        return self._model.dataframe

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("DataFrame")
        toolbar.setMovable(False)

        toolbar.addWidget(QtWidgets.QLabel("Column:", self))
        toolbar.addWidget(self._column_combo)

        sort_asc = QtGui.QAction("Sort Asc", self)
        sort_asc.triggered.connect(lambda: self._sort_selected_column(False))
        toolbar.addAction(sort_asc)

        sort_desc = QtGui.QAction("Sort Desc", self)
        sort_desc.triggered.connect(lambda: self._sort_selected_column(True))
        toolbar.addAction(sort_desc)

        toolbar.addSeparator()

        unique_action = QtGui.QAction("Value Counts", self)
        unique_action.triggered.connect(self._show_value_counts_dialog)
        toolbar.addAction(unique_action)

        copy_action = QtGui.QAction("Copy", self)
        copy_action.setShortcut(QtGui.QKeySequence.StandardKey.Copy)
        copy_action.triggered.connect(self._table.copy_selection_to_clipboard)
        toolbar.addAction(copy_action)

    def _build_summary_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Column Summary", self)
        dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        panel = QtWidgets.QWidget(dock)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        self._metric_table = QtWidgets.QTableWidget(0, 2, panel)
        self._metric_table.setHorizontalHeaderLabels(["metric", "value"])
        self._metric_table.horizontalHeader().setStretchLastSection(True)
        self._metric_table.verticalHeader().setVisible(False)
        self._metric_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._metric_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        layout.addWidget(self._metric_table, stretch=2)

        top_label = QtWidgets.QLabel("Top values", panel)
        layout.addWidget(top_label)

        self._value_table = QtWidgets.QTableWidget(0, 2, panel)
        self._value_table.setHorizontalHeaderLabels(["value", "count"])
        self._value_table.horizontalHeader().setStretchLastSection(True)
        self._value_table.verticalHeader().setVisible(False)
        self._value_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._value_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        layout.addWidget(self._value_table, stretch=3)

        dock.setWidget(panel)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _sort_selected_column(self, descending: bool) -> None:
        column = self._column_combo.currentText()
        if not column:
            return
        self._model.sort_by_name(column, descending=descending)
        order_label = "descending" if descending else "ascending"
        self._show_status(f"Sorted by {column!r} ({order_label})")

    def _show_value_counts_dialog(self) -> None:
        column = self._column_combo.currentText()
        if not column:
            return
        self._show_status(f"Computing value counts for {column!r}...")
        QtWidgets.QApplication.processEvents()
        try:
            counts = value_counts_for_column(self._model.dataframe, column)
        except Exception as exc:
            self._show_status(f"Could not compute value counts: {exc}")
            return
        dialog = ValueCountsDialog(counts, column=column, parent=self)
        self._keep_value_counts_dialog_alive(dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self._show_status(f"{counts.height:,} unique values in {column!r}")

    def _keep_value_counts_dialog_alive(self, dialog: "ValueCountsDialog") -> None:
        self._value_count_dialogs.append(dialog)

        def _forget_dialog() -> None:
            try:
                self._value_count_dialogs.remove(dialog)
            except ValueError:
                pass

        dialog.destroyed.connect(_forget_dialog)

    def _on_current_column_changed(
        self,
        current: QtCore.QModelIndex,
        previous: QtCore.QModelIndex,
    ) -> None:
        del previous
        if not current.isValid():
            return
        column_name = self._model.dataframe.columns[current.column()]
        if self._column_combo.currentText() != column_name:
            self._column_combo.blockSignals(True)
            self._column_combo.setCurrentText(column_name)
            self._column_combo.blockSignals(False)
        self._update_summary(column_name)

    def _on_combo_column_changed(self, column: str) -> None:
        if not column:
            return
        col_idx = self._model.dataframe.columns.index(column)
        self._table.setCurrentIndex(self._model.index(0, col_idx))
        self._update_summary(column)

    def _update_summary(self, column: str) -> None:
        try:
            self._summary = summarize_column(self._model.dataframe, column)
        except Exception as exc:
            self._show_status(f"Could not summarize {column!r}: {exc}")
            return

        self._metric_table.setRowCount(len(self._summary.metrics))
        for row, (metric, value) in enumerate(self._summary.metrics):
            self._metric_table.setItem(row, 0, QtWidgets.QTableWidgetItem(metric))
            self._metric_table.setItem(row, 1, QtWidgets.QTableWidgetItem(value))
        self._metric_table.resizeColumnsToContents()

        self._populate_value_table(self._summary)

    def _populate_value_table(self, summary: ColumnSummary) -> None:
        counts = summary.value_counts
        if counts is None:
            self._value_table.setRowCount(1)
            self._value_table.setItem(
                0,
                0,
                QtWidgets.QTableWidgetItem(
                    f"Skipped because n unique > {summary.value_count_limit:,}"
                ),
            )
            self._value_table.setItem(0, 1, QtWidgets.QTableWidgetItem(""))
            return

        value_col, count_col = counts.columns
        self._value_table.setRowCount(counts.height)
        for row in range(counts.height):
            self._value_table.setItem(
                row,
                0,
                QtWidgets.QTableWidgetItem(format_value(counts[value_col][row])),
            )
            self._value_table.setItem(
                row,
                1,
                QtWidgets.QTableWidgetItem(format_value(counts[count_col][row])),
            )
        self._value_table.resizeColumnsToContents()

    def _show_status(self, message: str) -> None:
        self.statusBar().showMessage(message, 8000)


class ValueCountsDialog(QtWidgets.QDialog):
    """Dialog showing the full value-counts table for one column."""

    def __init__(
        self,
        counts: pl.DataFrame,
        *,
        column: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Value counts - {column}")
        self.resize(520, 700)

        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(f"{counts.height:,} unique values in {column!r}", self)
        layout.addWidget(label)

        table = CopyableTableView(self)
        table.setModel(PolarsDataFrameModel(counts, table))
        table.setSortingEnabled(True)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.verticalHeader().setDefaultSectionSize(22)
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)


def view_df(
    df: pl.DataFrame,
    *,
    title: str | None = None,
    parent: QtWidgets.QWidget | None = None,
    block: bool | None = None,
) -> DataFrameViewer:
    """Open a Qt viewer for a Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to display.  The table model is virtualized, so all rows
        are available without converting the DataFrame to pandas.
    title : str, optional
        Window title.  Defaults to a shape-aware LoupeDF title.
    parent : QWidget, optional
        Qt parent widget.
    block : bool or None
        Whether to run ``QApplication.exec()`` before returning.  By default,
        scripts block when this function creates the QApplication; notebooks
        return immediately.  In Jupyter, run ``%gui qt6`` first.

    Returns
    -------
    DataFrameViewer
        The live viewer window.
    """

    _require_polars_dataframe(df)
    app = QtWidgets.QApplication.instance()
    created_app = False
    if app is None:
        _warn_if_ipython_without_qt()
        app = QtWidgets.QApplication([])
        created_app = True

    window = DataFrameViewer(df, title=title, parent=parent)
    window.show()
    _keep_window_alive(window)

    if block is None:
        block = created_app and not _in_ipython()
    if block:
        app.exec()
    return window


def summarize_column(
    df: pl.DataFrame,
    column: str,
    *,
    top_n: int = 12,
    value_count_limit: int = 50_000,
) -> ColumnSummary:
    """Summarize one column with Polars-native operations."""

    _require_polars_dataframe(df)
    if column not in df.columns:
        raise ValueError(f"Unknown column {column!r}.")

    series = df.get_column(column)
    dtype = series.dtype
    nulls = int(series.null_count())
    non_nulls = int(series.len() - nulls)
    n_unique = int(series.n_unique()) if series.len() else 0

    metrics: list[tuple[str, str]] = [
        ("column", column),
        ("dtype", str(dtype)),
        ("rows", format_count(series.len())),
        ("non-null", format_count(non_nulls)),
        ("null", format_count(nulls)),
        ("n unique", format_count(n_unique)),
    ]

    estimated_size = _safe_call(lambda: series.estimated_size("mb"))
    if estimated_size is not None:
        metrics.append(("estimated size", f"{float(estimated_size):.3g} MB"))

    nan_count = _nan_count(series)
    if nan_count is not None:
        metrics.append(("NaN", format_count(nan_count)))

    if _is_boolean_dtype(dtype):
        true_count = _safe_call(lambda: int(series.sum()))
        if true_count is not None:
            metrics.append(("true", format_count(true_count)))
            metrics.append(("false", format_count(non_nulls - true_count)))
    elif _is_numeric_dtype(dtype):
        _append_numeric_metrics(metrics, series)
    else:
        _append_min_max_metrics(metrics, series)

    value_counts = None
    if n_unique <= value_count_limit:
        value_counts = value_counts_for_column(df, column).head(top_n)

    return ColumnSummary(
        column=column,
        metrics=tuple(metrics),
        value_counts=value_counts,
        value_count_limit=value_count_limit,
    )


def value_counts_for_column(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Return value counts for *column*, sorted by descending count."""

    _require_polars_dataframe(df)
    if column not in df.columns:
        raise ValueError(f"Unknown column {column!r}.")

    count_name = "__count__"
    while count_name in df.columns:
        count_name = f"_{count_name}"
    return df.get_column(column).value_counts(sort=True, name=count_name)


def format_value(value: Any, *, max_len: int = 240) -> str:
    """Format a scalar value for compact table display."""

    if value is None:
        return "null"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        text = f"{value:.8g}"
    else:
        item = value.item() if hasattr(value, "item") else value
        text = str(item)

    if len(text) > max_len:
        return f"{text[: max_len - 1]}..."
    return text


def format_count(value: int) -> str:
    return f"{int(value):,}"


def _append_numeric_metrics(
    metrics: list[tuple[str, str]],
    series: pl.Series,
) -> None:
    numeric_metrics: tuple[tuple[str, Any], ...] = (
        ("mean", _safe_call(series.mean)),
        ("std", _safe_call(series.std)),
        ("min", _safe_call(series.min)),
        ("q25", _safe_call(lambda: series.quantile(0.25))),
        ("median", _safe_call(series.median)),
        ("q75", _safe_call(lambda: series.quantile(0.75))),
        ("max", _safe_call(series.max)),
    )
    for label, value in numeric_metrics:
        if value is not None:
            metrics.append((label, format_value(value)))


def _append_min_max_metrics(
    metrics: list[tuple[str, str]],
    series: pl.Series,
) -> None:
    min_value = _safe_call(series.min)
    max_value = _safe_call(series.max)
    if min_value is not None:
        metrics.append(("min", format_value(min_value)))
    if max_value is not None:
        metrics.append(("max", format_value(max_value)))


def _safe_call(func) -> Any | None:
    try:
        return func()
    except Exception:
        return None


def _nan_count(series: pl.Series) -> int | None:
    try:
        mask = series.is_nan()
    except Exception:
        return None
    try:
        return int(mask.sum())
    except Exception:
        return None


def _is_numeric_dtype(dtype: pl.DataType) -> bool:
    predicate = getattr(dtype, "is_numeric", None)
    if predicate is not None:
        return bool(predicate())
    return dtype in {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.Int128,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }


def _is_boolean_dtype(dtype: pl.DataType) -> bool:
    return dtype == pl.Boolean


def _require_polars_dataframe(df: Any) -> None:
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"view_df expects a polars DataFrame, got {type(df)!r}.")


def _keep_window_alive(window: DataFrameViewer) -> None:
    _OPEN_WINDOWS.append(window)

    def _forget_window() -> None:
        try:
            _OPEN_WINDOWS.remove(window)
        except ValueError:
            pass

    window.destroyed.connect(_forget_window)


def _in_ipython() -> bool:
    try:
        get_ipython()  # type: ignore[name-defined]  # noqa: F821
    except NameError:
        return False
    return True


def _warn_if_ipython_without_qt() -> None:
    try:
        ip = get_ipython()  # type: ignore[name-defined]  # noqa: F821
    except NameError:
        return
    loop = getattr(ip, "active_eventloop", None)
    if loop not in ("qt", "qt5", "qt6"):
        warnings.warn(
            "No Qt event loop detected. Run '%gui qt6' before calling view_df() "
            "for interactive use in Jupyter/IPython.",
            stacklevel=3,
        )


__all__ = [
    "ColumnSummary",
    "DataFrameViewer",
    "PolarsDataFrameModel",
    "ValueCountsDialog",
    "summarize_column",
    "value_counts_for_column",
    "view_df",
]
