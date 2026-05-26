"""Interval-label summary widgets used by :class:`loupe.app.LoupeApp`.

- :class:`StateComboDelegate` — table dropdown for the State column.
- :class:`IntervalLabelSummaryBarWidget` — horizontal stacked color bar.
- :class:`IntervalLabelSummaryWidget` — editable table panel; receives
  ``main_window`` so it can read ``interval_label_set`` /
  ``label_colors`` / ``t_global_*`` and push edits back via
  ``main_window._finalize_interval_label_change()``.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from loupe.interval_labels import IntervalLabelSchemaError


class StateComboDelegate(QtWidgets.QStyledItemDelegate):
    """Dropdown delegate for the State column in the label summary table."""

    def __init__(self, states: list[str], parent=None):
        super().__init__(parent)
        self._states = list(states)

    def set_states(self, states: list[str]):
        self._states = list(states)

    def createEditor(self, parent, option, index):
        combo = QtWidgets.QComboBox(parent)
        for state in self._states:
            combo.addItem(state)
        return combo

    def setEditorData(self, editor, index):
        current = index.data()
        idx = editor.findText(current or "")
        if idx >= 0:
            editor.setCurrentIndex(idx)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText(), QtCore.Qt.ItemDataRole.EditRole)


class IntervalLabelSummaryBarWidget(QtWidgets.QWidget):
    """Horizontal stacked color bar showing per-state labeling fractions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(24)
        self.segments: list[tuple[float, tuple, str]] = []

    def set_data(self, segments: list[tuple[float, tuple, str]]):
        """Set bar segments: list of (fraction, rgba_tuple, label_text)."""
        self.segments = segments
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        try:
            w, h = self.width(), self.height()
            if not self.segments:
                painter.fillRect(0, 0, w, h, QtGui.QColor(40, 40, 40))
                return
            x = 0.0
            for frac, color, text in self.segments:
                seg_w = frac * w
                int_x = int(round(x))
                int_w = int(round(x + seg_w)) - int_x
                painter.fillRect(int_x, 0, int_w, h, QtGui.QColor(color[0], color[1], color[2]))
                if int_w > 50:
                    painter.setPen(QtGui.QColor(255, 255, 255))
                    font = painter.font()
                    font.setPointSize(8)
                    painter.setFont(font)
                    painter.drawText(
                        int_x, 0, int_w, h,
                        QtCore.Qt.AlignmentFlag.AlignCenter,
                        text,
                    )
                x += seg_w
        finally:
            painter.end()


class IntervalLabelSummaryWidget(QtWidgets.QWidget):
    """Inline panel showing an editable table of all scored interval labels with summary stats."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._refreshing = False
        # Column-index bookkeeping for inline edits; refresh() updates these.
        self._note_col_idx: int | None = None
        self._extra_col_start: int = 4

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # Table; columns are reset dynamically in refresh() based on the
        # active IntervalLabelSchema (so any number of extra columns can be shown).
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Start", "End", "Duration", "State"])
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
        )
        self.table.setStyleSheet(
            "QTableWidget { background-color: #1a1a1a; color: #ddd; gridline-color: #444; }"
            "QHeaderView::section { background-color: #333; color: #ddd; border: 1px solid #444; padding: 2px; }"
        )
        # State column uses a dropdown delegate
        self._state_delegate = StateComboDelegate(
            sorted(main_window.label_colors.keys())
        )
        self.table.setItemDelegateForColumn(3, self._state_delegate)

        self.table.cellChanged.connect(self._on_cell_changed)
        layout.addWidget(self.table, 1)

        # Summary text
        self.summary_label = QtWidgets.QLabel("Fraction of total recording labelled: 0.0%")
        self.summary_label.setStyleSheet("color: #ccc; padding: 2px;")
        layout.addWidget(self.summary_label)

        bar_header = QtWidgets.QLabel("Labelling by state:")
        bar_header.setStyleSheet("color: #ccc; font-weight: bold; padding: 0px 2px;")
        layout.addWidget(bar_header)

        # Color bar
        self.bar_widget = IntervalLabelSummaryBarWidget()
        layout.addWidget(self.bar_widget)

    def refresh(self):
        """Repopulate table and summary from main_window.interval_label_set."""
        if self._refreshing:
            return
        self._refreshing = True
        try:
            self.table.blockSignals(True)
            ls = self.main_window.interval_label_set
            schema = ls.schema
            note_col = schema.note_col
            extra_cols = list(schema.extra_cols)

            # Column layout: Start, End, Duration, State, [Note?], extras...
            headers = ["Start", "End", "Duration", "State"]
            if note_col:
                headers.append("Note")
            headers.extend(extra_cols)
            self.table.setColumnCount(len(headers))
            self.table.setHorizontalHeaderLabels(headers)
            self._note_col_idx = 4 if note_col else None
            self._extra_col_start = 5 if note_col else 4

            self.table.setRowCount(0)
            for i, row in enumerate(ls):
                self.table.insertRow(i)
                self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(f"{row.start:.3f}"))
                self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{row.end:.3f}"))
                self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{row.duration:.3f}"))
                self.table.setItem(i, 3, QtWidgets.QTableWidgetItem(row.label))
                if note_col:
                    self.table.setItem(
                        i, self._note_col_idx, QtWidgets.QTableWidgetItem(row.note)
                    )
                for j, col in enumerate(extra_cols):
                    v = row.extras.get(col)
                    self.table.setItem(
                        i,
                        self._extra_col_start + j,
                        QtWidgets.QTableWidgetItem("" if v is None else str(v)),
                    )

                # Store row_id for mapping back during edits
                self.table.item(i, 0).setData(
                    QtCore.Qt.ItemDataRole.UserRole, int(row.row_id)
                )

            self.table.resizeColumnsToContents()
            self.table.blockSignals(False)
            self._update_summary()
            self._state_delegate.set_states(sorted(self.main_window.label_colors.keys()))
        finally:
            self._refreshing = False

    def _update_summary(self):
        """Recompute summary stats and update the bar widget."""
        total_recording = self.main_window.t_global_max - self.main_window.t_global_min
        if total_recording <= 0:
            self.summary_label.setText("No recording loaded")
            self.bar_widget.set_data([])
            return

        ls = self.main_window.interval_label_set
        total_labelled = float(np.sum(ls.ends - ls.starts)) if len(ls) else 0.0
        pct = (total_labelled / total_recording) * 100.0
        self.summary_label.setText(
            f"Fraction of total recording labelled: {pct:.1f}%"
        )

        state_durations: dict[str, float] = {}
        for row in ls:
            dur = row.duration
            state_durations[row.label] = state_durations.get(row.label, 0.0) + dur

        segments = []
        for state, dur in sorted(state_durations.items(), key=lambda x: -x[1]):
            frac = dur / total_recording
            color = self.main_window.label_colors.get(state, (150, 150, 150, 80))
            bar_color = (color[0], color[1], color[2], 255)
            text = f"{state} {frac * 100:.0f}%"
            segments.append((frac, bar_color, text))
        self.bar_widget.set_data(segments)

    def _on_cell_changed(self, row: int, col: int):
        """Handle inline cell edits — validate and propagate to IntervalLabelSet."""
        if self._refreshing:
            return
        ls = self.main_window.interval_label_set
        if row < 0 or row >= len(ls):
            return

        item_id = self.table.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)
        if item_id is None:
            return
        row_id = int(item_id)
        target_row = ls.row_for_id(row_id)
        if target_row is None:
            return

        item = self.table.item(row, col)
        if item is None:
            return
        new_text = item.text().strip()

        schema = ls.schema
        try:
            if col == 0:  # Start
                new_start = float(new_text)
                if new_start >= target_row.end:
                    raise ValueError("Start must be less than End")
                if new_start < 0:
                    raise ValueError("Start must be non-negative")
                ls.update_cell(row_id, schema.start_col, new_start)

            elif col == 1:  # End
                new_end = float(new_text)
                if new_end <= target_row.start:
                    raise ValueError("End must be greater than Start")
                if schema.end_col is None:
                    raise ValueError(
                        "Cannot edit End directly: schema has duration_col only"
                    )
                ls.update_cell(row_id, schema.end_col, new_end)

            elif col == 2:  # Duration
                new_dur = float(new_text)
                if new_dur <= 0:
                    raise ValueError("Duration must be positive")
                new_end = target_row.start + new_dur
                if schema.end_col:
                    ls.update_cell(row_id, schema.end_col, new_end)
                elif schema.duration_col:
                    ls.update_cell(row_id, schema.duration_col, new_dur)

            elif col == 3:  # State
                if not new_text:
                    raise ValueError("State cannot be empty")
                ls.update_cell(row_id, schema.label_col, new_text)

            elif self._note_col_idx is not None and col == self._note_col_idx:
                ls.set_note(row_id, new_text)
                # Note edits don't need visual redraw of label regions
                return

            else:
                # Extras column.
                extra_idx = col - self._extra_col_start
                if 0 <= extra_idx < len(schema.extra_cols):
                    col_name = schema.extra_cols[extra_idx]
                    ls.update_cell(row_id, col_name, new_text or None)
                return

        except (ValueError, IntervalLabelSchemaError) as e:
            QtWidgets.QMessageBox.warning(self, "Invalid edit", str(e))
            self.refresh()
            return

        # Re-sort, merge, and sync visuals (which also triggers refresh)
        self.main_window._finalize_interval_label_change()
