import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import polars as pl
import pytest
from PySide6 import QtCore, QtWidgets

from loupe.loupeDF import (
    DataFrameViewer,
    PolarsDataFrameModel,
    summarize_column,
    value_counts_for_column,
    view_df,
)


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_model_exposes_all_rows_columns_and_values():
    df = pl.DataFrame(
        {
            "syn_id": [10, 11, 12],
            "state": ["wake", "nrem", None],
            "score": [0.1, 0.25, float("nan")],
        }
    )

    model = PolarsDataFrameModel(df)

    assert model.rowCount() == 3
    assert model.columnCount() == 3
    assert (
        model.headerData(
            1,
            QtCore.Qt.Orientation.Horizontal,
            QtCore.Qt.ItemDataRole.DisplayRole,
        )
        == "state"
    )
    assert model.data(model.index(1, 1), QtCore.Qt.ItemDataRole.DisplayRole) == "nrem"
    assert model.data(model.index(2, 1), QtCore.Qt.ItemDataRole.DisplayRole) == "null"
    assert model.data(model.index(2, 2), QtCore.Qt.ItemDataRole.DisplayRole) == "NaN"


def test_model_sort_reorders_dataframe():
    df = pl.DataFrame({"value": [2, 3, 1], "label": ["b", "c", "a"]})
    model = PolarsDataFrameModel(df)

    model.sort(0, QtCore.Qt.SortOrder.DescendingOrder)

    assert model.data(model.index(0, 0), QtCore.Qt.ItemDataRole.DisplayRole) == "3"
    assert model.data(model.index(0, 1), QtCore.Qt.ItemDataRole.DisplayRole) == "c"
    assert model.dataframe["value"].to_list() == [3, 2, 1]


def test_summarize_column_reports_unique_and_numeric_stats():
    df = pl.DataFrame(
        {
            "group": ["a", "b", "a", None],
            "value": [1.0, 2.0, 3.0, None],
        }
    )

    group_summary = summarize_column(df, "group", top_n=5)
    group_metrics = dict(group_summary.metrics)
    assert group_metrics["rows"] == "4"
    assert group_metrics["null"] == "1"
    assert group_metrics["n unique"] == "3"
    assert group_summary.value_counts is not None
    assert group_summary.value_counts.height == 3

    value_summary = summarize_column(df, "value", top_n=5)
    value_metrics = dict(value_summary.metrics)
    assert value_metrics["mean"] == "2"
    assert value_metrics["median"] == "2"


def test_value_counts_handles_column_named_count():
    df = pl.DataFrame({"count": ["x", "x", "y"]})

    counts = value_counts_for_column(df, "count")

    assert counts.columns == ["count", "__count__"]
    assert counts["count"].to_list() == ["x", "y"]
    assert counts["__count__"].to_list() == [2, 1]


def test_viewer_and_view_df_construct_without_running_event_loop(qapp):
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    viewer = DataFrameViewer(df)
    viewer.show()
    qapp.processEvents()
    assert viewer.dataframe.shape == (2, 2)
    viewer.close()

    window = view_df(df, block=False)
    qapp.processEvents()
    assert window.dataframe.shape == (2, 2)
    window.close()
