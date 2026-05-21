from __future__ import annotations

import polars as pl
import pytest

from loupe.labels import (
    LabelIOError,
    LabelSchema,
    LabelSchemaError,
    LabelSet,
    infer_schema_for_path,
)


# ---------------------------------------------------------------------------
# LabelSchema validation
# ---------------------------------------------------------------------------


def test_schema_requires_end_or_duration():
    with pytest.raises(LabelSchemaError):
        LabelSchema(start_col="s", end_col=None, duration_col=None, label_col="l")


def test_schema_legacy_round_trip():
    s = LabelSchema.legacy()
    assert s.start_col == "start_s"
    assert s.end_col == "end_s"
    assert s.label_col == "label"
    assert s.note_col == "note"


def test_schema_validate_missing_column():
    schema = LabelSchema(start_col="a", end_col="b", label_col="state")
    df = pl.DataFrame({"a": [0.0], "b": [1.0]})  # missing 'state'
    with pytest.raises(LabelSchemaError, match="state"):
        schema.validate(df)


def test_schema_end_and_duration_disagree_raises():
    schema = LabelSchema(start_col="s", end_col="e", duration_col="d", label_col="lbl")
    df = pl.DataFrame({"s": [0.0], "e": [1.0], "d": [2.0], "lbl": ["x"]})
    with pytest.raises(LabelSchemaError, match="row 0"):
        schema.validate(df)


def test_schema_end_and_duration_agree_ok():
    schema = LabelSchema(start_col="s", end_col="e", duration_col="d", label_col="lbl")
    df = pl.DataFrame({"s": [0.0, 1.0], "e": [1.0, 3.0], "d": [1.0, 2.0], "lbl": ["x", "y"]})
    schema.validate(df)


def test_schema_duplicate_column_names_raise():
    with pytest.raises(LabelSchemaError, match="duplicate"):
        LabelSchema(start_col="x", end_col="y", label_col="x")


# ---------------------------------------------------------------------------
# Format I/O round-trips
# ---------------------------------------------------------------------------


@pytest.fixture
def legacy_set():
    schema = LabelSchema.legacy()
    df = pl.DataFrame(
        {
            "start_s": [0.0, 5.0, 10.0],
            "end_s": [5.0, 10.0, 15.0],
            "label": ["Wake", "NREM", "REM"],
            "note": ["", "deep", ""],
        }
    )
    return LabelSet.from_dataframe(df, schema)


def test_round_trip_csv(tmp_path, legacy_set):
    path = tmp_path / "labels.csv"
    legacy_set.save_as(path)
    reloaded = LabelSet.from_path(path)
    assert reloaded.df.drop("__loupe_row_id").equals(legacy_set.df.drop("__loupe_row_id"))


def test_round_trip_htsv(tmp_path, legacy_set):
    path = tmp_path / "labels.htsv"
    legacy_set.save_as(path)
    reloaded = LabelSet.from_path(path, schema=LabelSchema.legacy())
    assert reloaded.df.drop("__loupe_row_id").equals(legacy_set.df.drop("__loupe_row_id"))


def test_round_trip_parquet(tmp_path, legacy_set):
    path = tmp_path / "labels.parquet"
    legacy_set.save_as(path)
    reloaded = LabelSet.from_path(path, schema=LabelSchema.legacy())
    assert reloaded.df.drop("__loupe_row_id").equals(legacy_set.df.drop("__loupe_row_id"))


def test_save_preserves_user_column_names(tmp_path):
    schema = LabelSchema(
        start_col="start_time",
        end_col="end_time",
        duration_col="duration",
        label_col="state",
        extra_cols=("scorer", "confidence"),
    )
    df = pl.DataFrame(
        {
            "start_time": [0.0, 5.0],
            "end_time": [5.0, 10.0],
            "duration": [5.0, 5.0],
            "state": ["Wake", "NREM"],
            "scorer": ["alice", "bob"],
            "confidence": [0.9, 0.8],
        }
    )
    ls = LabelSet.from_dataframe(df, schema)
    out = tmp_path / "labels.htsv"
    ls.save_as(out)
    raw = pl.read_csv(out, separator="\t")
    assert set(raw.columns) == {
        "start_time",
        "end_time",
        "duration",
        "state",
        "scorer",
        "confidence",
    }
    assert "__loupe_row_id" not in raw.columns


def test_visbrain_writer_raises(tmp_path, legacy_set):
    with pytest.raises(LabelIOError, match="read-only"):
        legacy_set.save_as(tmp_path / "out.txt")


# ---------------------------------------------------------------------------
# Visbrain reader
# ---------------------------------------------------------------------------


def test_visbrain_reader(tmp_path):
    path = tmp_path / "hyp.txt"
    path.write_text(
        "*Duration_sec\t100.0\n"
        "*Datafile\tW:/example\n"
        "Wake\t10.0\n"
        "N1\t15.5\n"
        "N2\t30.0\n",
        encoding="utf-8",
    )
    ls = LabelSet.from_path(path)
    assert len(ls) == 3
    rows = list(ls)
    assert rows[0].start == 0.0
    assert rows[0].end == 10.0
    assert rows[0].label == "Wake"
    assert rows[1].start == 10.0
    assert rows[1].end == 15.5
    assert rows[2].start == 15.5
    assert rows[2].end == 30.0


def test_visbrain_skips_metadata(tmp_path):
    path = tmp_path / "hyp.txt"
    path.write_text("*hello\n*world\nWake\t1.0\n", encoding="utf-8")
    ls = LabelSet.from_path(path)
    assert len(ls) == 1


# ---------------------------------------------------------------------------
# infer_schema_for_path
# ---------------------------------------------------------------------------


def test_infer_schema_csv():
    s = infer_schema_for_path("foo.csv")
    assert s.start_col == "start_s"


def test_infer_schema_visbrain():
    s = infer_schema_for_path("foo.txt")
    assert s.start_col == "start_time"
    assert s.label_col == "state"


def test_infer_schema_htsv_raises():
    with pytest.raises(LabelIOError, match="explicit"):
        infer_schema_for_path("foo.htsv")


def test_infer_schema_parquet_raises():
    with pytest.raises(LabelIOError, match="explicit"):
        infer_schema_for_path("foo.parquet")


# ---------------------------------------------------------------------------
# LabelSet mechanics
# ---------------------------------------------------------------------------


def test_empty_set():
    ls = LabelSet.empty()
    assert len(ls) == 0
    assert ls.at_time(5.0) is None


def test_add_label_to_empty():
    ls = LabelSet.empty()
    rid = ls.add(0.0, 10.0, "Wake")
    assert len(ls) == 1
    row = ls.at_time(5.0)
    assert row is not None and row.row_id == rid and row.label == "Wake"


def test_add_overlapping_clips_existing():
    ls = LabelSet.empty()
    ls.add(0.0, 10.0, "Wake")
    ls.add(5.0, 15.0, "NREM")
    rows = list(ls)
    assert [r.start for r in rows] == [0.0, 5.0]
    assert [r.end for r in rows] == [5.0, 15.0]
    assert [r.label for r in rows] == ["Wake", "NREM"]


def test_add_contained_splits_existing():
    ls = LabelSet.empty()
    ls.add(0.0, 20.0, "Wake")
    ls.add(8.0, 12.0, "NREM")
    rows = list(ls)
    assert [r.label for r in rows] == ["Wake", "NREM", "Wake"]
    assert [(r.start, r.end) for r in rows] == [(0.0, 8.0), (8.0, 12.0), (12.0, 20.0)]


def test_clear_range_splits_existing():
    ls = LabelSet.empty()
    ls.add(0.0, 20.0, "Wake")
    ls.clear_range(8.0, 12.0)
    rows = list(ls)
    assert [(r.start, r.end) for r in rows] == [(0.0, 8.0), (12.0, 20.0)]


def test_merge_adjacent():
    schema = LabelSchema.legacy()
    df = pl.DataFrame(
        {
            "start_s": [0.0, 5.0, 10.0],
            "end_s": [5.0, 10.0, 15.0],
            "label": ["Wake", "Wake", "NREM"],
            "note": ["", "", ""],
        }
    )
    ls = LabelSet.from_dataframe(df, schema)
    ls.merge_adjacent()
    rows = list(ls)
    assert [(r.start, r.end, r.label) for r in rows] == [
        (0.0, 10.0, "Wake"),
        (10.0, 15.0, "NREM"),
    ]


def test_pop_last_undo():
    ls = LabelSet.empty()
    ls.add(0.0, 5.0, "Wake")
    rid = ls.add(10.0, 15.0, "NREM")
    popped = ls.pop_last()
    assert popped is not None and popped.row_id == rid
    rows = list(ls)
    assert [r.label for r in rows] == ["Wake"]


def test_note_survives_endpoint_edit():
    """row_ids are stable across edits → notes don't go stale."""
    ls = LabelSet.empty()
    rid = ls.add(0.0, 10.0, "Wake")
    ls.set_note(rid, "interesting")
    ls.update_cell(rid, "start_s", 1.0)
    assert ls.get_note(rid) == "interesting"


def test_update_cell_keeps_duration_consistent():
    schema = LabelSchema(
        start_col="s", end_col="e", duration_col="d", label_col="lbl"
    )
    df = pl.DataFrame({"s": [0.0], "e": [10.0], "d": [10.0], "lbl": ["Wake"]})
    ls = LabelSet.from_dataframe(df, schema)
    rid = list(ls)[0].row_id
    ls.update_cell(rid, "e", 20.0)
    row_dict = ls.df.filter(pl.col("__loupe_row_id") == rid).row(0, named=True)
    assert row_dict["d"] == pytest.approx(20.0)


def test_save_to_source_requires_writeback(tmp_path, legacy_set):
    path = tmp_path / "labels.csv"
    legacy_set.save_as(path)
    reloaded = LabelSet.from_path(path)
    with pytest.raises(LabelIOError, match="labels_writeback"):
        reloaded.save_to_source()


def test_save_to_source_with_opt_in(tmp_path, legacy_set):
    path = tmp_path / "labels.csv"
    legacy_set.save_as(path)
    reloaded = LabelSet.from_path(path, writeback_allowed=True)
    reloaded.add(20.0, 25.0, "Wake")
    reloaded.save_to_source()
    again = LabelSet.from_path(path)
    assert len(again) == 4


def test_split_inherits_extras():
    schema = LabelSchema(
        start_col="s", end_col="e", label_col="lbl", extra_cols=("scorer",)
    )
    df = pl.DataFrame({"s": [0.0], "e": [20.0], "lbl": ["Wake"], "scorer": ["alice"]})
    ls = LabelSet.from_dataframe(df, schema)
    ls.add(8.0, 12.0, "NREM")
    rows = list(ls)
    # The new middle row gets a null extra.
    middle = next(r for r in rows if r.label == "NREM")
    assert middle.extras["scorer"] is None
    # Both halves of the split keep the original extras.
    halves = [r for r in rows if r.label == "Wake"]
    assert len(halves) == 2
    assert all(h.extras["scorer"] == "alice" for h in halves)


def test_visible_in():
    ls = LabelSet.empty()
    ls.add(0.0, 5.0, "Wake")
    ls.add(5.0, 10.0, "NREM")
    ls.add(10.0, 15.0, "REM")
    rows = list(ls.visible_in(4.0, 11.0))
    assert [r.label for r in rows] == ["Wake", "NREM", "REM"]
