"""TDT block launcher. Disk discovery and validation run outside the GUI thread."""

from __future__ import annotations

import copy
from pathlib import Path
import threading

import numpy as np
import polars as pl
from PySide6 import QtCore, QtGui, QtWidgets

from .reader import Cancelled, inspect_block, check_cancel
from .session import (
    MODES,
    config_path,
    default_config,
    launch_session,
    prepare_session,
    read_config,
)

# A running QThread must outlive even a parent window being closed. The worker
# owns no widgets, so cancelling and closing any launcher is safe.
_jobs = set()


class Job(QtCore.QThread):
    result = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(str)

    def __init__(self, work):
        super().__init__()
        self.work = work
        self.cancel = threading.Event()

    def run(self):
        try:
            value = self.work(cancel=self.cancel, progress=self.progress.emit)
            check_cancel(self.cancel)
            self.result.emit(value)
        except Cancelled:
            pass
        except Exception as exc:
            self.failed.emit(str(exc))


class ColorButton(QtWidgets.QPushButton):
    changed = QtCore.Signal()

    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.set_color(color)
        self.clicked.connect(self.choose)

    def set_color(self, color):
        self.color = str(color)
        self.setText(self.color)
        swatch = QtGui.QPixmap(14, 14)
        swatch.fill(QtGui.QColor(self.color))
        self.setIcon(QtGui.QIcon(swatch))

    def choose(self):
        color = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(self.color), self, "Plot color"
        )
        if color.isValid():
            self.set_color(color.name())
            self.changed.emit()


class TDTLauncher(QtWidgets.QDialog):
    def __init__(self, parent=None, path=None):
        super().__init__(parent)
        self.setWindowTitle("Loupe · Open TDT block")
        self.resize(1100, 780)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.block = None
        self.saved_config = None
        self.config = None
        self.job = None
        self._close_when_done = False
        self.last_window = None
        layout = QtWidgets.QVBoxLayout(self)
        heading = QtWidgets.QLabel("Open a TDT recording")
        heading.setStyleSheet("font-size: 22px; font-weight: 600; margin: 6px 0;")
        layout.addWidget(heading)
        browse_row = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit(str(path or ""))
        self.path_edit.setPlaceholderText("Choose a TDT block folder")
        self.browse_button = QtWidgets.QPushButton("Browse…")
        self.inspect_button = QtWidgets.QPushButton("Inspect block")
        browse_row.addWidget(self.path_edit, 1)
        browse_row.addWidget(self.browse_button)
        browse_row.addWidget(self.inspect_button)
        layout.addLayout(browse_row)
        self.info_label = QtWidgets.QLabel(
            "Streams are read from disk as you browse. All times are seconds from block start."
        )
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.info_label)
        saved_row = QtWidgets.QHBoxLayout()
        self.existing_button = QtWidgets.QPushButton("Use existing block config")
        self.existing_button.setEnabled(False)
        self.reset_button = QtWidgets.QPushButton("Fresh settings")
        self.reset_button.setEnabled(False)
        self.saved_label = QtWidgets.QLabel("")
        saved_row.addWidget(self.existing_button)
        saved_row.addWidget(self.reset_button)
        saved_row.addWidget(self.saved_label, 1)
        layout.addLayout(saved_row)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setEnabled(False)
        layout.addWidget(self.tabs, 1)

        plots_page = QtWidgets.QWidget()
        plots_layout = QtWidgets.QVBoxLayout(plots_page)
        hint = QtWidgets.QLabel(
            "Check stores to display. Rows run from top to bottom; channel lists also set channel order."
        )
        plots_layout.addWidget(hint)
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Store", "Display", "Channels", "Color", "Height", "Dense gain", "Opacity"]
        )
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        for col in range(1, 7):
            self.table.horizontalHeader().setSectionResizeMode(
                col, QtWidgets.QHeaderView.ResizeMode.Interactive
            )
        # Qt's ResizeToContents ignores some embedded-widget size hints before
        # the first layout. Reserve readable controls even during async load.
        for col, width in enumerate([160, 135, 115, 70, 110, 75], start=1):
            self.table.setColumnWidth(col, width)
        plots_layout.addWidget(self.table, 1)
        move_row = QtWidgets.QHBoxLayout()
        up, down = QtWidgets.QPushButton("Move up"), QtWidgets.QPushButton("Move down")
        up.clicked.connect(lambda: self.move_row(-1))
        down.clicked.connect(lambda: self.move_row(1))
        move_row.addWidget(up)
        move_row.addWidget(down)
        move_row.addStretch()
        move_row.addWidget(
            QtWidgets.QLabel(
                "Gain 0 = automatic scaling · Shading applies across every plot"
            )
        )
        plots_layout.addLayout(move_row)
        self.tabs.addTab(plots_page, "Stores & layout")

        video_page = QtWidgets.QWidget()
        video_layout = QtWidgets.QVBoxLayout(video_page)
        video_hint = QtWidgets.QLabel(
            "Load video by checking its row. Frame times come from the chosen epoc's onsets.\n"
            "Loupe verifies frame counts before opening; timing correction is in seconds."
        )
        video_layout.addWidget(video_hint)
        self.video_table = QtWidgets.QTableWidget(0, 4)
        self.video_table.setHorizontalHeaderLabels(
            ["Load", "Video file", "Frame epoc", "Correction (s)"]
        )
        self.video_table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.video_table.verticalHeader().hide()
        for col, width in [(0, 55), (2, 190), (3, 130)]:
            self.video_table.setColumnWidth(col, width)
        video_layout.addWidget(self.video_table)
        add_video = QtWidgets.QPushButton("Add video file…")
        add_video.clicked.connect(self.add_video)
        video_layout.addWidget(add_video)
        self.tabs.addTab(video_page, "Video")

        annotation_page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(annotation_page)
        annotation_hint = QtWidgets.QLabel(
            "Optional hypnogram or annotation file. Map its columns below.\n"
            "Editing labels is independent of recorded epocs. Saved block configs retain your edits."
        )
        form.addRow(annotation_hint)
        annotation_row = QtWidgets.QHBoxLayout()
        self.annotation_edit = QtWidgets.QLineEdit()
        annotation_browse = QtWidgets.QPushButton("Choose CSV…")
        annotation_browse.clicked.connect(self.browse_annotation)
        clear_labels = QtWidgets.QPushButton("Clear")
        clear_labels.clicked.connect(self.clear_annotation)
        annotation_row.addWidget(self.annotation_edit)
        annotation_row.addWidget(annotation_browse)
        annotation_row.addWidget(clear_labels)
        form.addRow("Annotation file", annotation_row)
        self.schema_edits = {}
        for key, label, default in [
            ("start_col", "Start time", "start_s"),
            ("end_col", "End time", "end_s"),
            ("duration_col", "Duration (instead of end)", ""),
            ("label_col", "State / label", "label"),
            ("note_col", "Note (optional)", ""),
        ]:
            edit = QtWidgets.QLineEdit(default)
            self.schema_edits[key] = edit
            form.addRow(label, edit)
        self.annotation_status = QtWidgets.QLabel(
            "Times must use seconds from this block's start."
        )
        self.annotation_status.setWordWrap(True)
        form.addRow(self.annotation_status)
        self.tabs.addTab(annotation_page, "Annotations")

        options = QtWidgets.QHBoxLayout()
        options.addWidget(QtWidgets.QLabel("Initial window (s)"))
        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(0.1, 3600)
        self.window_spin.setValue(10)
        options.addWidget(self.window_spin)
        self.save_check = QtWidgets.QCheckBox("Save block config")
        self.save_check.setChecked(True)
        self.save_check.setToolTip(
            "Write selections, view settings and annotations to .loupe/tdt.json in this block"
        )
        options.addStretch()
        options.addWidget(self.save_check)
        layout.addLayout(options)
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        buttons = QtWidgets.QHBoxLayout()
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.load_button = QtWidgets.QPushButton("Load block")
        self.load_button.setDefault(True)
        self.load_button.setEnabled(False)
        buttons.addStretch()
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.load_button)
        layout.addLayout(buttons)
        self.browse_button.clicked.connect(self.browse)
        self.inspect_button.clicked.connect(self.inspect)
        self.path_edit.returnPressed.connect(self.inspect)
        self.existing_button.clicked.connect(
            lambda: self.apply_config(self.saved_config)
        )
        self.reset_button.clicked.connect(
            lambda: self.apply_config(default_config(self.block))
        )
        self.load_button.clicked.connect(self.load)
        self.cancel_button.clicked.connect(self.reject)
        if path:
            QtCore.QTimer.singleShot(0, self.inspect)

    def browse(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose a TDT block", self.path_edit.text()
        )
        if path:
            self.path_edit.setText(path)
            self.inspect()

    def _set_busy(self, busy):
        self.progress_bar.setVisible(busy)
        for widget in (self.path_edit, self.browse_button, self.inspect_button):
            widget.setEnabled(not busy)
        self.tabs.setEnabled(not busy and self.block is not None)
        self.load_button.setEnabled(not busy and self.block is not None)
        self.reset_button.setEnabled(not busy and self.block is not None)
        self.existing_button.setEnabled(not busy and self.saved_config is not None)
        self.window_spin.setEnabled(not busy)
        self.save_check.setEnabled(not busy)

    def _start(self, work, result):
        self._set_busy(True)
        self.status_label.setStyleSheet("")
        job = self.job = Job(work)
        _jobs.add(job)
        job.progress.connect(self.status_label.setText)
        job.result.connect(result)
        job.failed.connect(self._failed)
        job.finished.connect(self._finished)
        job.finished.connect(lambda: _jobs.discard(job))
        job.start()

    @QtCore.Slot(str)
    def _failed(self, message):
        self.status_label.setStyleSheet("color: #d66;")
        self.status_label.setText(message)

    @QtCore.Slot()
    def _finished(self):
        self.job = None
        self._set_busy(False)
        if self._close_when_done:
            super().reject()

    def inspect(self):
        if self.job is not None:
            return
        path = self.path_edit.text().strip()
        if not path:
            self.browse()
            return
        self.block = self.saved_config = None
        self.status_label.setText("Inspecting block…")
        self._start(lambda **kw: inspect_block(path, **kw), self._inspected)

    @QtCore.Slot(object)
    def _inspected(self, block):
        if self._close_when_done:
            return
        self.block = block
        self.path_edit.setText(str(block.path))
        self.info_label.setText(
            f"{block.path.name} · {block.duration / 3600:.2f} hours · "
            f"{len(block.streams)} stream stores · {len(block.epocs)} epoc stores · "
            f"{len(block.snips)} spike stores\nStart (UTC): {block.info['start_utc']}"
            + (
                f" · Scalar stores: {', '.join(block.info['scalars'])}"
                if block.info["scalars"]
                else ""
            )
        )
        self.info_label.setToolTip(str(block.info))
        self.saved_label.setText("No saved block config")
        self.apply_config(default_config(block))
        if config_path(block).exists():
            try:
                self.saved_config = read_config(block)
                self.saved_label.setText("Saved configuration available")
            except Exception as exc:
                self.saved_label.setText(f"Saved config unavailable: {exc}")
        self.status_label.setText(
            "\n".join(block.warnings)
            or "Choose your stores, channels and display options, then load the block."
        )

    def apply_config(self, config):
        if config is None:
            return
        self.config = copy.deepcopy(config)
        self._populate_rows(config["rows"])
        self._populate_videos(config.get("videos", []))
        self.window_spin.setValue(config.get("window_len", 10))
        self.annotation_edit.setText(config.get("annotation_path", ""))
        schema = config.get("annotation_schema") or {
            "start_col": "start_s",
            "end_col": "end_s",
            "label_col": "label",
        }
        for key, edit in self.schema_edits.items():
            edit.setText(schema.get(key) or "")
        snapshot = config.get("annotation_snapshot")
        self.annotation_status.setText(
            "Saved annotation edits will be restored. Choose another file or Clear to replace them."
            if snapshot
            else "Times must use seconds from this block's start."
        )
        if config.get("view_config"):
            self.status_label.setText(
                "Using saved view settings and annotations. Changing store settings starts a fresh plot layout."
            )

    def _populate_rows(self, rows):
        self.table.setRowCount(0)
        for i, row in enumerate(rows):
            self.table.insertRow(i)
            kind, name = row["kind"], row["store"]
            if kind == "stream":
                source = self.block.streams[name]
                detail = f"{len(source.channels)} ch · {source.fs:g} Hz"
                tip = f"Channels: {source.channels}. " + (source.error or "")
            elif kind == "epoc":
                detail = f"{len(self.block.epocs[name].onset):,} epocs"
                tip = "Shading is independent of your editable state labels. Open-ended intervals end at the block boundary."
            else:
                source = self.block.snips[name]
                detail = f"{len(source.timestamps):,} spikes"
                tip = f"Channels: {np.unique(source.channels).tolist()}; raster shows channel IDs, not sort codes."
            item = QtWidgets.QTableWidgetItem(f"{name}   {detail}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, copy.deepcopy(row))
            item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsSelectable
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
            )
            item.setCheckState(
                QtCore.Qt.CheckState.Checked
                if row["enabled"]
                else QtCore.Qt.CheckState.Unchecked
            )
            item.setToolTip(tip)
            if kind == "stream" and self.block.streams[name].error:
                item.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(i, 0, item)
            mode = QtWidgets.QComboBox()
            labels = {
                "dense": "Dense plot",
                "lines": "Line per channel",
                "shade": "Global shading",
                "ttl": "TTL trace",
                "both": "Shading + TTL",
                "raster": "Channel raster",
            }
            for option in MODES[kind]:
                mode.addItem(labels[option], option)
            mode.setCurrentIndex(mode.findData(row["mode"]))
            self.table.setCellWidget(i, 1, mode)
            channels = QtWidgets.QLineEdit(row.get("channels", ""))
            channels.setPlaceholderText("all / 1-4, 8")
            channels.setMinimumWidth(120)
            channels.setEnabled(kind != "epoc")
            channels.setToolTip(tip)
            self.table.setCellWidget(i, 2, channels)
            self.table.setCellWidget(i, 3, ColorButton(row["color"]))
            for col, key, minimum, maximum, default in [
                (4, "height", 0.1, 20, 1),
                (5, "gain", 0, 1e12, 0),
                (6, "alpha", 0, 1, 0.22),
            ]:
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(minimum, maximum)
                spin.setDecimals(2 if col != 5 else 3)
                spin.setSingleStep(0.1)
                spin.setValue(row.get(key, default))
                spin.setEnabled(
                    col == 4
                    or (col == 5 and kind == "stream")
                    or (col == 6 and kind == "epoc")
                )
                if col == 5:
                    spin.setSpecialValueText("Auto")
                self.table.setCellWidget(i, col, spin)
            for col in range(1, 7):
                self.table.cellWidget(i, col).setMaximumHeight(34)
            self.table.setRowHeight(i, 38)

    def _read_rows(self):
        rows = []
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            row = copy.deepcopy(item.data(QtCore.Qt.ItemDataRole.UserRole))
            row.update(
                enabled=item.checkState() == QtCore.Qt.CheckState.Checked,
                mode=self.table.cellWidget(i, 1).currentData(),
                channels=self.table.cellWidget(i, 2).text().strip(),
                color=self.table.cellWidget(i, 3).color,
                height=self.table.cellWidget(i, 4).value(),
                gain=self.table.cellWidget(i, 5).value(),
            )
            if row["kind"] == "epoc":
                row["alpha"] = self.table.cellWidget(i, 6).value()
            rows.append(row)
        return rows

    def move_row(self, direction):
        index = self.table.currentRow()
        target = index + direction
        if index < 0 or not 0 <= target < self.table.rowCount():
            return
        rows = self._read_rows()
        rows[index], rows[target] = rows[target], rows[index]
        self._populate_rows(rows)
        self.table.selectRow(target)

    def _populate_videos(self, videos):
        self.video_table.setRowCount(0)
        for i, video in enumerate(videos):
            self.video_table.insertRow(i)
            item = QtWidgets.QTableWidgetItem("")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, copy.deepcopy(video))
            item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
            )
            item.setCheckState(
                QtCore.Qt.CheckState.Checked
                if video.get("enabled")
                else QtCore.Qt.CheckState.Unchecked
            )
            self.video_table.setItem(i, 0, item)
            path = QtWidgets.QLineEdit(video["path"])
            path.setToolTip(video["path"])
            self.video_table.setCellWidget(i, 1, path)
            epoc = QtWidgets.QComboBox()
            epoc.addItem("Choose frame epoc", "")
            for name, source in self.block.epocs.items():
                epoc.addItem(f"{name} ({len(source.onset):,} frames)", name)
            epoc.setCurrentIndex(max(0, epoc.findData(video.get("epoc", ""))))
            self.video_table.setCellWidget(i, 2, epoc)
            correction = QtWidgets.QDoubleSpinBox()
            correction.setRange(-1e6, 1e6)
            correction.setDecimals(6)
            correction.setValue(video.get("correction", 0))
            self.video_table.setCellWidget(i, 3, correction)

    def _read_videos(self):
        videos = []
        for i in range(self.video_table.rowCount()):
            video = copy.deepcopy(
                self.video_table.item(i, 0).data(QtCore.Qt.ItemDataRole.UserRole)
            )
            video.update(
                enabled=self.video_table.item(i, 0).checkState()
                == QtCore.Qt.CheckState.Checked,
                path=self.video_table.cellWidget(i, 1).text().strip(),
                epoc=self.video_table.cellWidget(i, 2).currentData(),
                correction=self.video_table.cellWidget(i, 3).value(),
            )
            videos.append(video)
        return videos

    def add_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose a video",
            str(self.block.path),
            "Videos (*.avi *.mp4 *.mkv *.mov);;All files (*)",
        )
        if path:
            videos = self._read_videos()
            videos.append(dict(path=path, epoc="", enabled=True, correction=0))
            self._populate_videos(videos)

    def browse_annotation(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose annotations",
            str(self.block.path),
            "Annotations (*.csv *.htsv *.parquet *.txt)",
        )
        if not path:
            return
        self.config.pop("annotation_snapshot", None)
        self.annotation_edit.setText(path)
        try:
            suffix = Path(path).suffix.lower()
            if suffix == ".parquet":
                columns = list(pl.read_parquet_schema(path))
            elif suffix == ".txt":
                columns = ["start_time", "end_time", "state"]
            else:
                columns = pl.read_csv(
                    path, separator="\t" if suffix == ".htsv" else ",", n_rows=0
                ).columns
            candidates = {
                "start_col": ["start_s", "start_time", "onset", "start"],
                "end_col": ["end_s", "end_time", "offset", "end"],
                "label_col": ["label", "state", "stage"],
                "note_col": ["note"],
                "duration_col": ["duration", "duration_s"],
            }
            for key, names in candidates.items():
                self.schema_edits[key].setText(
                    next((n for n in names if n in columns), "")
                )
            if self.schema_edits["end_col"].text():
                self.schema_edits["duration_col"].clear()
            self.annotation_status.setText("Columns: " + ", ".join(columns))
        except Exception as exc:
            self.annotation_status.setText(str(exc))

    def clear_annotation(self):
        self.annotation_edit.clear()
        if self.config:
            self.config.pop("annotation_snapshot", None)
        self.annotation_status.setText("No annotations loaded.")

    def collect_config(self):
        config = copy.deepcopy(self.config)
        rows, videos = self._read_rows(), self._read_videos()
        if (
            rows != config["rows"]
            or videos != config.get("videos", [])
            or self.window_spin.value() != config.get("window_len", 10)
        ):
            config.pop("view_config", None)
        annotation_path = self.annotation_edit.text().strip()
        schema = {
            key: edit.text().strip() or None for key, edit in self.schema_edits.items()
        }
        if annotation_path != config.get("annotation_path", "") or (
            annotation_path and schema != config.get("annotation_schema")
        ):
            config.pop("annotation_snapshot", None)
        config.update(
            rows=rows,
            videos=videos,
            window_len=self.window_spin.value(),
            annotation_path=annotation_path,
            annotation_schema=schema if annotation_path else None,
        )
        return config

    def load(self):
        if self.job is not None or self.block is None:
            return
        try:
            config = self.collect_config()
            block = self.block
            self._start(
                lambda **kw: prepare_session(block, config, **kw), self._prepared
            )
        except Exception as exc:
            self._failed(str(exc))

    @QtCore.Slot(object)
    def _prepared(self, prepared):
        if self._close_when_done:
            return
        try:
            self.last_window = launch_session(
                prepared, save=self.save_check.isChecked()
            )
            # Finish the QThread before deleting its signal receiver.
            self._close_when_done = True
        except Exception as exc:
            self._failed(str(exc))

    def reject(self):
        if self.job is not None:
            self.job.cancel.set()
            self._close_when_done = True
            self.status_label.setText("Cancelling…")
            self.cancel_button.setEnabled(False)
        else:
            super().reject()

    def closeEvent(self, event):
        if self.job is not None:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)


class EpocStyleDialog(QtWidgets.QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.setWindowTitle("TDT epoc appearance")
        form = QtWidgets.QFormLayout(self)
        for row, _ in window.extension_session.prepared.epocs:
            controls = QtWidgets.QHBoxLayout()
            color = ColorButton(row["color"])
            alpha = QtWidgets.QDoubleSpinBox()
            alpha.setRange(0, 1)
            alpha.setSingleStep(0.05)
            alpha.setValue(row.get("alpha", 0.22))

            def update(*args, row=row, color=color, alpha=alpha):
                row.update(color=color.color, alpha=alpha.value())
                window.extension_session.refresh_overlays(window)

            color.changed.connect(update)
            alpha.valueChanged.connect(update)
            controls.addWidget(color)
            controls.addWidget(alpha)
            form.addRow(row["store"], controls)
        form.addRow(
            QtWidgets.QLabel("Use Update saved block config to retain these changes.")
        )
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
