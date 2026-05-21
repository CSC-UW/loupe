"""Labels: pluggable schema, in-memory `LabelSet`, and format I/O.

The :class:`LabelSchema` describes which user-named columns hold the start
time, end time (or duration), label string, optional note, and optional extra
columns. The :class:`LabelSet` wraps a polars ``DataFrame`` plus the schema
plus session bookkeeping (history, source path, writeback flag).

Format dispatch is by file extension. CSV, HTSV (header-bearing TSV), and
Parquet support both reading and writing; Visbrain ``.txt`` is read-only
because writing it would silently drop the note and extras columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import polars as pl

__all__ = [
    "LabelIOError",
    "LabelRow",
    "LabelSchema",
    "LabelSet",
    "LabelSchemaError",
    "infer_schema_for_path",
]


# Internal column name used to track stable row identity across edits.
# Stripped on save. Chosen to be unlikely to collide with user data.
_ROW_ID_COL = "__loupe_row_id"

_LEGACY_CSV_SCHEMA_KWARGS: dict[str, Any] = dict(
    start_col="start_s",
    end_col="end_s",
    label_col="label",
    note_col="note",
)

_VISBRAIN_SCHEMA_KWARGS: dict[str, Any] = dict(
    start_col="start_time",
    end_col="end_time",
    duration_col="duration",
    label_col="state",
)


class LabelIOError(IOError):
    """Raised on label read/write failure."""


class LabelSchemaError(ValueError):
    """Raised when a DataFrame does not match the declared :class:`LabelSchema`."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelSchema:
    """Describe how a user's DataFrame columns map to label semantics.

    Exactly one of ``end_col`` or ``duration_col`` is required (both is also
    accepted, but the values must agree on every row — see :meth:`validate`).

    Parameters
    ----------
    start_col
        Name of the column holding interval start times in seconds.
    end_col
        Name of the column holding interval end times in seconds.
    duration_col
        Name of the column holding interval durations in seconds. If
        ``end_col`` is omitted, ``end = start + duration`` is used.
    label_col
        Name of the column holding the state name.
    note_col
        Optional name of a single freeform note column. Used by the legacy
        CSV format and by the GUI's status bar / Edit Note dialog.
    extra_cols
        Names of additional columns to display, edit, and save. Order is
        preserved.
    """

    start_col: str = "start_s"
    end_col: str | None = "end_s"
    duration_col: str | None = None
    label_col: str = "label"
    note_col: str | None = None
    extra_cols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.end_col is None and self.duration_col is None:
            raise LabelSchemaError(
                "LabelSchema requires at least one of end_col or duration_col."
            )
        if not self.start_col:
            raise LabelSchemaError("LabelSchema.start_col cannot be empty.")
        if not self.label_col:
            raise LabelSchemaError("LabelSchema.label_col cannot be empty.")
        # Detect column-name collisions in the schema declaration.
        names = [
            self.start_col,
            self.end_col,
            self.duration_col,
            self.label_col,
            self.note_col,
            *self.extra_cols,
        ]
        seen: set[str] = set()
        for n in names:
            if n is None:
                continue
            if n in seen:
                raise LabelSchemaError(
                    f"LabelSchema has duplicate column name {n!r}"
                )
            seen.add(n)

    @classmethod
    def legacy(cls) -> "LabelSchema":
        """Loupe's traditional CSV schema (``start_s,end_s,label,note``)."""
        return cls(**_LEGACY_CSV_SCHEMA_KWARGS)

    @classmethod
    def visbrain(cls) -> "LabelSchema":
        """Schema produced by the Visbrain ``.txt`` reader."""
        return cls(**_VISBRAIN_SCHEMA_KWARGS)

    @property
    def all_cols(self) -> tuple[str, ...]:
        """Columns this schema describes, in display order."""
        cols: list[str] = [self.start_col]
        if self.end_col:
            cols.append(self.end_col)
        if self.duration_col and self.duration_col not in cols:
            cols.append(self.duration_col)
        cols.append(self.label_col)
        if self.note_col and self.note_col not in cols:
            cols.append(self.note_col)
        for c in self.extra_cols:
            if c not in cols:
                cols.append(c)
        return tuple(cols)

    def validate(self, df: pl.DataFrame, *, eps: float = 1e-9) -> None:
        """Check required columns are present and consistent.

        Raises :class:`LabelSchemaError` on any violation.
        """
        missing = [c for c in self.all_cols if c not in df.columns]
        if missing:
            raise LabelSchemaError(
                f"DataFrame is missing schema column(s): {missing}. "
                f"Available columns: {df.columns}"
            )
        if self.end_col and self.duration_col and len(df) > 0:
            diff = (
                df[self.end_col].cast(pl.Float64)
                - df[self.start_col].cast(pl.Float64)
                - df[self.duration_col].cast(pl.Float64)
            ).abs()
            max_diff = float(diff.max() or 0.0)
            if max_diff > eps:
                # Find the first offending row for a useful error message.
                offending = int(diff.arg_max() or 0)
                raise LabelSchemaError(
                    f"end_col and duration_col disagree at row {offending}: "
                    f"|{self.end_col} - ({self.start_col} + {self.duration_col})| "
                    f"= {max_diff} > {eps}"
                )


# ---------------------------------------------------------------------------
# LabelRow view
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelRow:
    """A read-only view of one label row."""

    row_id: int
    start: float
    end: float
    label: str
    note: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start


# ---------------------------------------------------------------------------
# Format readers / writers
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=10000)


def _read_htsv(path: Path) -> pl.DataFrame:
    if path.suffix != ".htsv":
        raise LabelIOError(
            f"HTSV files must use the .htsv extension, got {path.suffix!r}"
        )
    return pl.read_csv(path, separator="\t", infer_schema_length=10000)


def _read_parquet(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def _read_visbrain(path: Path) -> pl.DataFrame:
    """Parse a Visbrain hypnogram .txt file.

    Format::

        *Duration_sec\\t7200.0
        *Datafile\\t...
        N1\\t41.99
        N2\\t89.99
        ...

    Lines beginning with ``*`` are metadata and are skipped. Data lines are
    ``<state>\\t<end_time>`` with the start of each bout equal to the previous
    bout's end (or 0.0 for the first).
    """
    df = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        comment_prefix="*",
        new_columns=["state", "end_time"],
        infer_schema_length=10000,
    )
    df = df.with_columns(pl.col("end_time").cast(pl.Float64))
    end = df["end_time"].to_numpy()
    start = np.empty_like(end)
    start[0] = 0.0
    if len(end) > 1:
        start[1:] = end[:-1]
    duration = end - start
    df = df.with_columns(
        pl.Series("start_time", start),
        pl.Series("duration", duration),
    )
    # Reorder to match LabelSchema.visbrain().all_cols
    return df.select(["start_time", "end_time", "duration", "state"])


def _write_csv(df: pl.DataFrame, path: Path) -> None:
    df.write_csv(path)


def _write_htsv(df: pl.DataFrame, path: Path) -> None:
    if path.suffix != ".htsv":
        raise LabelIOError(
            f"HTSV files must use the .htsv extension, got {path.suffix!r}"
        )
    df.write_csv(path, separator="\t")


def _write_parquet(df: pl.DataFrame, path: Path) -> None:
    df.write_parquet(path)


READERS = {
    "csv": _read_csv,
    "htsv": _read_htsv,
    "parquet": _read_parquet,
    "txt": _read_visbrain,
}

WRITERS = {
    "csv": _write_csv,
    "htsv": _write_htsv,
    "parquet": _write_parquet,
    # No "txt": Visbrain is read-only; writing would silently drop notes/extras.
}


def _format_for_path(path: Path) -> str:
    """Map a file extension to a format key. Raises on unknown formats."""
    ext = path.suffix.lower().lstrip(".")
    if ext in READERS or ext in WRITERS:
        return ext
    raise LabelIOError(
        f"Unrecognized label format for path {path!r}. "
        f"Supported extensions: {sorted(set(READERS) | set(WRITERS))}"
    )


def infer_schema_for_path(path: str | Path) -> LabelSchema:
    """Best-effort default :class:`LabelSchema` for a label-file path.

    Used by ``view()`` when the user passes a path but no schema. Falls back
    to the legacy CSV schema for ``.csv``; uses :meth:`LabelSchema.visbrain`
    for ``.txt``; raises for ``.htsv`` and ``.parquet`` because their column
    names cannot be guessed safely.
    """
    p = Path(path)
    fmt = _format_for_path(p)
    if fmt == "csv":
        return LabelSchema.legacy()
    if fmt == "txt":
        return LabelSchema.visbrain()
    raise LabelIOError(
        f"Cannot infer LabelSchema for {p.name}. Pass an explicit "
        f"`label_schema=LabelSchema(...)` describing the columns."
    )


# ---------------------------------------------------------------------------
# LabelSet
# ---------------------------------------------------------------------------


class LabelSet:
    """In-memory label store backed by a polars DataFrame.

    The DataFrame uses the user's column names as declared in
    :class:`LabelSchema`, plus a private ``__loupe_row_id`` column for stable
    cross-edit references. ``__loupe_row_id`` is stripped on save.

    All methods that mutate state keep the DataFrame sorted by start time
    and refresh the cached numpy ``starts`` / ``ends`` arrays used by
    :meth:`at_time` and :meth:`visible_in`.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        schema: LabelSchema,
        *,
        source_path: Path | None = None,
        source_format: str | None = None,
        writeback_allowed: bool = False,
    ) -> None:
        schema.validate(df)
        # Add row_id column if not present.
        if _ROW_ID_COL not in df.columns:
            df = df.with_columns(
                pl.Series(_ROW_ID_COL, np.arange(len(df), dtype=np.int64))
            )
        self._df: pl.DataFrame = df
        self.schema: LabelSchema = schema
        self.source_path: Path | None = source_path
        self.source_format: str | None = source_format
        self.writeback_allowed: bool = bool(writeback_allowed)
        self.history: list[int] = []
        self._next_id: int = (
            int(df[_ROW_ID_COL].max() or -1) + 1 if len(df) > 0 else 0
        )
        self._invalidate_caches()
        self._sort_inplace()

    # --- construction ----------------------------------------------------

    @classmethod
    def empty(cls, schema: LabelSchema | None = None) -> "LabelSet":
        """Return an empty LabelSet matching ``schema`` (defaults to legacy)."""
        schema = schema or LabelSchema.legacy()
        cols = {schema.start_col: pl.Float64, schema.label_col: pl.Utf8}
        if schema.end_col:
            cols[schema.end_col] = pl.Float64
        if schema.duration_col:
            cols[schema.duration_col] = pl.Float64
        if schema.note_col:
            cols[schema.note_col] = pl.Utf8
        for c in schema.extra_cols:
            cols.setdefault(c, pl.Utf8)
        cols[_ROW_ID_COL] = pl.Int64
        df = pl.DataFrame(schema=cols)
        return cls(df, schema)

    @classmethod
    def from_dataframe(
        cls,
        df: pl.DataFrame,
        schema: LabelSchema,
        *,
        source_path: Path | None = None,
        writeback_allowed: bool = False,
    ) -> "LabelSet":
        """Create a LabelSet from an in-memory DataFrame matching ``schema``."""
        df = _normalize_loaded_df(df, schema)
        return cls(
            df,
            schema,
            source_path=source_path,
            source_format=None,
            writeback_allowed=writeback_allowed,
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        schema: LabelSchema | None = None,
        *,
        format: str | None = None,
        writeback_allowed: bool = False,
    ) -> "LabelSet":
        """Read a labels file. ``schema`` is inferred for ``.csv``/``.txt``."""
        p = Path(path)
        fmt = format or _format_for_path(p)
        if fmt not in READERS:
            raise LabelIOError(f"Format {fmt!r} is read-only-not-supported.")
        if schema is None:
            schema = infer_schema_for_path(p)
        df = READERS[fmt](p)
        df = _normalize_loaded_df(df, schema)
        return cls(
            df,
            schema,
            source_path=p,
            source_format=fmt,
            writeback_allowed=writeback_allowed,
        )

    # --- internals -------------------------------------------------------

    def _next_row_id(self) -> int:
        out = self._next_id
        self._next_id += 1
        return out

    def _invalidate_caches(self) -> None:
        if len(self._df) == 0:
            self._starts: np.ndarray = np.empty(0, dtype=float)
            self._ends: np.ndarray = np.empty(0, dtype=float)
            self._row_ids: np.ndarray = np.empty(0, dtype=np.int64)
        else:
            self._starts = self._df[self.schema.start_col].cast(pl.Float64).to_numpy()
            self._ends = self._effective_end_series().to_numpy()
            self._row_ids = self._df[_ROW_ID_COL].to_numpy()

    def _effective_end_series(self) -> pl.Series:
        if self.schema.end_col:
            return self._df[self.schema.end_col].cast(pl.Float64)
        # duration_col only — derive end on the fly
        return (
            self._df[self.schema.start_col].cast(pl.Float64)
            + self._df[self.schema.duration_col].cast(pl.Float64)
        )

    def _sort_inplace(self) -> None:
        if len(self._df) > 1:
            self._df = self._df.sort(self.schema.start_col)
        self._invalidate_caches()
        # Prune history of any row_ids no longer present.
        if self.history:
            present = set(self._row_ids.tolist())
            self.history = [rid for rid in self.history if rid in present]

    def _set_row_endpoints(
        self,
        row: dict[str, Any],
        *,
        start: float | None = None,
        end: float | None = None,
    ) -> dict[str, Any]:
        """Update ``row`` in place with new start/end, syncing duration."""
        sc = self.schema.start_col
        ec = self.schema.end_col
        dc = self.schema.duration_col
        if start is not None:
            row[sc] = float(start)
        if end is not None and ec:
            row[ec] = float(end)
        if dc:
            cur_start = float(row[sc])
            if ec:
                cur_end = float(row[ec])
            elif end is not None:
                cur_end = float(end)
            else:
                cur_end = cur_start + float(row.get(dc, 0.0))
            row[dc] = cur_end - cur_start
        return row

    def _make_new_row_dict(
        self,
        *,
        start: float,
        end: float,
        label: str,
        inherit_from: int | None = None,
    ) -> dict[str, Any]:
        """Build a fresh row dict matching this LabelSet's columns."""
        sc = self.schema.start_col
        ec = self.schema.end_col
        dc = self.schema.duration_col
        lc = self.schema.label_col
        nc = self.schema.note_col

        row: dict[str, Any] = {sc: float(start), lc: str(label)}
        if ec:
            row[ec] = float(end)
        if dc:
            row[dc] = float(end) - float(start)
        if nc:
            row[nc] = ""
        for col in self.schema.extra_cols:
            row[col] = None

        if inherit_from is not None:
            template = self._df.filter(pl.col(_ROW_ID_COL) == int(inherit_from))
            if len(template) == 1:
                src = template.row(0, named=True)
                if nc:
                    row[nc] = src.get(nc, "") or ""
                for col in self.schema.extra_cols:
                    row[col] = src.get(col)

        row[_ROW_ID_COL] = self._next_row_id()
        return row

    def _dataframe_from_dicts(self, rows: list[dict[str, Any]]) -> pl.DataFrame:
        """Rebuild ``self._df`` from a list of row dicts, preserving its schema."""
        if not rows:
            return self._df.head(0)
        cols = list(self._df.columns)
        data = {col: [r.get(col) for r in rows] for col in cols}
        return pl.DataFrame(data, schema=self._df.schema)

    # --- iteration / lookup ---------------------------------------------

    def __len__(self) -> int:
        return self._df.height

    def __iter__(self) -> Iterator[LabelRow]:
        for src in self._df.iter_rows(named=True):
            yield self._row_from_dict(src)

    @property
    def df(self) -> pl.DataFrame:
        """A view into the underlying DataFrame (includes the private row_id column)."""
        return self._df

    @property
    def starts(self) -> np.ndarray:
        return self._starts

    @property
    def ends(self) -> np.ndarray:
        return self._ends

    @property
    def row_ids(self) -> np.ndarray:
        return self._row_ids

    def _row_from_dict(self, src: dict[str, Any]) -> LabelRow:
        sc = self.schema.start_col
        ec = self.schema.end_col
        dc = self.schema.duration_col
        nc = self.schema.note_col
        start = float(src[sc])
        if ec:
            end = float(src[ec])
        else:
            end = start + float(src[dc])
        label = str(src[self.schema.label_col])
        note = str(src.get(nc, "") or "") if nc else ""
        extras = {c: src.get(c) for c in self.schema.extra_cols}
        return LabelRow(
            row_id=int(src[_ROW_ID_COL]),
            start=start,
            end=end,
            label=label,
            note=note,
            extras=extras,
        )

    def at_time(self, t: float) -> LabelRow | None:
        """Return the row containing ``t`` (start <= t < end), or None."""
        if len(self._df) == 0:
            return None
        idx = int(np.searchsorted(self._starts, t, side="right") - 1)
        if idx < 0:
            return None
        if t < self._starts[idx] or t >= self._ends[idx]:
            return None
        return self.row_at_index(idx)

    def row_at_index(self, idx: int) -> LabelRow:
        return self._row_from_dict(self._df.row(idx, named=True))

    def index_for_row_id(self, row_id: int) -> int | None:
        hits = np.where(self._row_ids == int(row_id))[0]
        return int(hits[0]) if len(hits) else None

    def row_for_id(self, row_id: int) -> LabelRow | None:
        idx = self.index_for_row_id(row_id)
        return self.row_at_index(idx) if idx is not None else None

    def visible_index_range(self, t0: float, t1: float) -> tuple[int, int]:
        """Half-open ``(start_idx, end_idx)`` of rows that overlap ``[t0, t1)``."""
        if len(self._df) == 0 or t1 <= t0:
            return (0, 0)
        start_idx = int(np.searchsorted(self._ends, t0, side="right"))
        end_idx = int(np.searchsorted(self._starts, t1, side="left"))
        if end_idx < start_idx:
            end_idx = start_idx
        return (start_idx, end_idx)

    def visible_in(self, t0: float, t1: float) -> Iterator[LabelRow]:
        a, b = self.visible_index_range(t0, t1)
        for i in range(a, b):
            yield self.row_at_index(i)

    # --- mutation --------------------------------------------------------

    def add(
        self,
        start: float,
        end: float,
        label: str,
        *,
        inherit_from: int | None = None,
    ) -> int:
        """Add a label spanning ``[start, end)``.

        Existing labels overlapping the range are split: portions outside the
        new range are preserved (and keep their extras), portions inside are
        replaced. Returns the row_id of the new label.
        """
        if end <= start:
            raise ValueError(f"end ({end}) must be > start ({start})")

        survivors: list[dict[str, Any]] = []
        for src in self._df.iter_rows(named=True):
            ex_start = float(src[self.schema.start_col])
            ex_end = self._end_value(src)
            if ex_end <= start or ex_start >= end:
                survivors.append(dict(src))
                continue
            # Overlap: keep non-overlapping tails, drop the overlapping middle.
            if ex_start < start:
                left = dict(src)
                left = self._set_row_endpoints(left, end=start)
                survivors.append(left)
            if ex_end > end:
                right = dict(src)
                right = self._set_row_endpoints(right, start=end)
                # If this is a contained-split (we already kept the left tail),
                # the right tail needs a fresh row_id.
                if ex_start < start:
                    right[_ROW_ID_COL] = self._next_row_id()
                survivors.append(right)

        # Build the new row.
        new_row = self._make_new_row_dict(
            start=start, end=end, label=label, inherit_from=inherit_from
        )
        new_row_id = int(new_row[_ROW_ID_COL])
        survivors.append(new_row)

        self._df = self._dataframe_from_dicts(survivors)
        self._sort_inplace()
        self.history.append(new_row_id)
        return new_row_id

    def _end_value(self, src: dict[str, Any]) -> float:
        if self.schema.end_col:
            return float(src[self.schema.end_col])
        return float(src[self.schema.start_col]) + float(src[self.schema.duration_col])

    def clear_range(self, start: float, end: float) -> None:
        """Delete any portions of labels overlapping ``[start, end)``."""
        if end <= start or len(self._df) == 0:
            return
        survivors: list[dict[str, Any]] = []
        for src in self._df.iter_rows(named=True):
            ex_start = float(src[self.schema.start_col])
            ex_end = self._end_value(src)
            if ex_end <= start or ex_start >= end:
                survivors.append(dict(src))
                continue
            if ex_start < start:
                left = dict(src)
                left = self._set_row_endpoints(left, end=start)
                survivors.append(left)
            if ex_end > end:
                right = dict(src)
                right = self._set_row_endpoints(right, start=end)
                if ex_start < start:
                    right[_ROW_ID_COL] = self._next_row_id()
                survivors.append(right)
        self._df = self._dataframe_from_dicts(survivors)
        self._sort_inplace()

    def merge_adjacent(self, eps: float = 1e-9) -> None:
        """Merge consecutive same-label rows whose gap is <= ``eps``.

        Merged rows lose their extras (we cannot decide which neighbor's
        metadata to keep). The first row's extras and note are kept.
        """
        if len(self._df) <= 1:
            return
        rows = list(self._df.iter_rows(named=True))
        rows.sort(key=lambda r: float(r[self.schema.start_col]))

        merged: list[dict[str, Any]] = []
        for src in rows:
            cur_start = float(src[self.schema.start_col])
            cur_end = self._end_value(src)
            cur_label = str(src[self.schema.label_col])
            if not merged:
                merged.append({**src, "_end_value": cur_end})
                continue
            prev = merged[-1]
            prev_end = float(prev["_end_value"])
            prev_label = str(prev[self.schema.label_col])
            if cur_label == prev_label and cur_start <= prev_end + eps:
                # Extend prev's end to max.
                new_end = max(prev_end, cur_end)
                prev["_end_value"] = new_end
                if self.schema.end_col:
                    prev[self.schema.end_col] = new_end
                if self.schema.duration_col:
                    prev[self.schema.duration_col] = new_end - float(
                        prev[self.schema.start_col]
                    )
            else:
                merged.append({**src, "_end_value": cur_end})

        for m in merged:
            m.pop("_end_value", None)
        self._df = self._dataframe_from_dicts(merged)
        self._sort_inplace()

    def delete_row(self, row_id: int) -> None:
        self._df = self._df.filter(pl.col(_ROW_ID_COL) != int(row_id))
        self._sort_inplace()

    def update_cell(self, row_id: int, col: str, value: Any) -> None:
        """Set the value of ``col`` for the row with ``row_id``.

        Updating ``start_col``, ``end_col``, or ``duration_col`` keeps the
        other endpoint-related columns consistent automatically.
        """
        if col not in self._df.columns:
            raise LabelSchemaError(f"Column {col!r} not present in the LabelSet")
        sc = self.schema.start_col
        ec = self.schema.end_col
        dc = self.schema.duration_col

        target = pl.col(_ROW_ID_COL) == int(row_id)
        if col == sc:
            new_start = float(value)
            if ec and dc:
                self._df = self._df.with_columns(
                    pl.when(target).then(pl.lit(new_start)).otherwise(pl.col(sc)).alias(sc),
                ).with_columns(
                    pl.when(target)
                    .then(pl.col(ec) - pl.lit(new_start))
                    .otherwise(pl.col(dc))
                    .alias(dc)
                )
            elif ec:
                self._df = self._df.with_columns(
                    pl.when(target).then(pl.lit(new_start)).otherwise(pl.col(sc)).alias(sc)
                )
            else:
                # duration-only: keep duration constant, end shifts implicitly.
                self._df = self._df.with_columns(
                    pl.when(target).then(pl.lit(new_start)).otherwise(pl.col(sc)).alias(sc)
                )
        elif ec is not None and col == ec:
            new_end = float(value)
            self._df = self._df.with_columns(
                pl.when(target).then(pl.lit(new_end)).otherwise(pl.col(ec)).alias(ec)
            )
            if dc:
                self._df = self._df.with_columns(
                    pl.when(target)
                    .then(pl.col(ec) - pl.col(sc))
                    .otherwise(pl.col(dc))
                    .alias(dc)
                )
        elif dc is not None and col == dc:
            new_dur = float(value)
            self._df = self._df.with_columns(
                pl.when(target).then(pl.lit(new_dur)).otherwise(pl.col(dc)).alias(dc)
            )
            if ec:
                self._df = self._df.with_columns(
                    pl.when(target)
                    .then(pl.col(sc) + pl.col(dc))
                    .otherwise(pl.col(ec))
                    .alias(ec)
                )
        else:
            self._df = self._df.with_columns(
                pl.when(target).then(pl.lit(value)).otherwise(pl.col(col)).alias(col)
            )
        self._sort_inplace()

    # --- notes (back-compat for legacy single-note column) --------------

    def get_note(self, row_id: int) -> str:
        if not self.schema.note_col:
            return ""
        df = self._df.filter(pl.col(_ROW_ID_COL) == int(row_id))
        if len(df) != 1:
            return ""
        v = df[self.schema.note_col][0]
        return str(v) if v is not None else ""

    def set_note(self, row_id: int, note: str) -> None:
        if not self.schema.note_col:
            raise LabelSchemaError(
                "This LabelSchema has no note_col. Use update_cell() to edit "
                "an extra column."
            )
        self.update_cell(row_id, self.schema.note_col, note or "")

    # --- undo ------------------------------------------------------------

    def pop_last(self) -> LabelRow | None:
        """Remove the most recently surviving created row and return it."""
        while self.history:
            rid = self.history.pop()
            idx = self.index_for_row_id(rid)
            if idx is not None:
                row = self.row_at_index(idx)
                self.delete_row(rid)
                return row
        return None

    # --- I/O -------------------------------------------------------------

    def to_savable_df(self) -> pl.DataFrame:
        """Return the DataFrame minus the private row-id column."""
        return self._df.drop(_ROW_ID_COL)

    def save_as(self, path: str | Path, *, format: str | None = None) -> None:
        """Write to ``path``. The format is inferred from the extension if not given."""
        p = Path(path)
        fmt = format or _format_for_path(p)
        if fmt not in WRITERS:
            raise LabelIOError(
                f"Format {fmt!r} is read-only. Export to .csv, .htsv, or .parquet instead."
            )
        WRITERS[fmt](self.to_savable_df(), p)

    def save_to_source(self) -> None:
        """Overwrite the original source file. Requires ``writeback_allowed=True``."""
        if not self.writeback_allowed:
            raise LabelIOError(
                "Save-to-source is disabled. Pass `labels_writeback=True` to "
                "view() to opt in to overwriting the original file."
            )
        if self.source_path is None:
            raise LabelIOError("LabelSet has no source_path; use save_as() instead.")
        self.save_as(self.source_path, format=self.source_format)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _normalize_loaded_df(df: pl.DataFrame, schema: LabelSchema) -> pl.DataFrame:
    """Validate ``df`` against ``schema``, then ensure a usable canonical form.

    - Cast start/end/duration columns to ``Float64``.
    - If only one of end/duration is present and the schema declares both,
      synthesize the missing one.
    - Cast the note column to ``Utf8`` (filling nulls with empty strings).
    - Ensure all extra_cols exist (missing ones become null-filled).
    """
    schema.validate(df)
    sc = schema.start_col
    ec = schema.end_col
    dc = schema.duration_col

    casts: list[pl.Expr] = [pl.col(sc).cast(pl.Float64).alias(sc)]
    if ec and ec in df.columns:
        casts.append(pl.col(ec).cast(pl.Float64).alias(ec))
    if dc and dc in df.columns:
        casts.append(pl.col(dc).cast(pl.Float64).alias(dc))
    df = df.with_columns(casts)

    # Synthesize end from duration or vice versa if the schema declares both
    # but only one is present.
    if ec and dc:
        if ec not in df.columns and dc in df.columns:
            df = df.with_columns(
                (pl.col(sc) + pl.col(dc)).alias(ec)
            )
        elif dc not in df.columns and ec in df.columns:
            df = df.with_columns(
                (pl.col(ec) - pl.col(sc)).alias(dc)
            )

    if schema.note_col and schema.note_col in df.columns:
        df = df.with_columns(
            pl.col(schema.note_col)
            .cast(pl.Utf8)
            .fill_null("")
            .alias(schema.note_col)
        )

    # Final validation pass after coercions (catches end/duration mismatch).
    schema.validate(df)
    return df
