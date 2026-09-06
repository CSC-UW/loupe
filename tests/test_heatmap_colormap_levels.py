"""Live bulk heatmap-level controls grouped by current colormap."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from loupe.dialogs import HeatmapControlsDialog
from loupe.series import HeatmapSeries


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _series(name: str, colormap: str, vmin: float, vmax: float) -> HeatmapSeries:
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    return HeatmapSeries(
        name=name,
        t=np.arange(values.shape[1], dtype=float),
        Y=values,
        row_labels=None,
        row_dim_name="row",
        colormap=colormap,
        vmin=vmin,
        vmax=vmax,
    )


class _HeatmapWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.heatmap_series = [
            _series("magma one", "magma", 0.0, 10.0),
            _series("viridis", "viridis", 1.0, 11.0),
            _series("magma two", "magma", 2.0, 12.0),
            _series("plasma", "plasma", 3.0, 13.0),
        ]
        self._heatmap_cache_keys = ["cached-0", "cached-1", "cached-2", "cached-3"]
        self.refresh_count = 0

    def _refresh_heatmap_plots(self):
        self.refresh_count += 1


def _check_colormap(dialog, name: str) -> None:
    for row in range(dialog.colormap_list.count()):
        item = dialog.colormap_list.item(row)
        if item.text().startswith(f"{name} "):
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            return
    raise AssertionError(f"No colormap row named {name!r}")


def test_bulk_levels_select_any_current_colormap_combination_and_update_live(_qapp):
    window = _HeatmapWindow()
    controls = HeatmapControlsDialog(window)

    assert controls.adjust_by_colormap_btn.text() == "Adjust vmin/vmax by colormap"
    controls.adjust_by_colormap_btn.click()
    _qapp.processEvents()
    bulk = controls._colormap_levels_dialog

    rows = [
        bulk.colormap_list.item(row).text()
        for row in range(bulk.colormap_list.count())
    ]
    assert rows == [
        "magma — 2 heatmaps",
        "viridis — 1 heatmap",
        "plasma — 1 heatmap",
    ]
    assert not bulk.vmin_spin.isEnabled()
    assert not bulk.vmax_spin.isEnabled()

    _check_colormap(bulk, "magma")
    _check_colormap(bulk, "plasma")
    assert bulk.vmin_spin.isEnabled()
    assert "3 heatmaps selected" in bulk.status_label.text()

    bulk.vmin_spin.setValue(-7.5)
    assert [heatmap.vmin for heatmap in window.heatmap_series] == [
        -7.5,
        1.0,
        -7.5,
        -7.5,
    ]
    assert window._heatmap_cache_keys == [None, "cached-1", None, None]
    assert window.refresh_count == 1
    assert controls._group_widgets[0]["vmin_spin"].value() == pytest.approx(-7.5)
    assert controls._group_widgets[2]["vmin_spin"].value() == pytest.approx(-7.5)
    assert controls._group_widgets[3]["vmin_spin"].value() == pytest.approx(-7.5)

    bulk.vmax_spin.setValue(25.0)
    assert [heatmap.vmax for heatmap in window.heatmap_series] == [
        25.0,
        11.0,
        25.0,
        25.0,
    ]
    assert window.refresh_count == 2
    assert "Changes apply immediately" in bulk.status_label.text()

    bulk.close()
    controls.close()
    window.close()
    _qapp.processEvents()


def test_bulk_swap_replaces_colormap_of_every_selected_group(_qapp):
    from loupe._heatmap_utils import _colormap_cache_token

    window = _HeatmapWindow()
    controls = HeatmapControlsDialog(window)
    controls._show_colormap_levels_dialog()
    dialog = controls._colormap_levels_dialog
    assert not dialog.swap_btn.isEnabled()

    _check_colormap(dialog, "magma")
    assert dialog.swap_btn.isEnabled()
    dialog.swap_combo.setCurrentText("cividis")
    dialog.swap_btn.click()

    assert [s.colormap for s in window.heatmap_series] == [
        "cividis", "viridis", "cividis", "plasma"
    ]
    assert window.refresh_count == 1
    assert window._heatmap_cache_keys == [None, "cached-1", None, "cached-3"]
    # regrouped: the new cividis group is the checked row, levels stay per-heatmap
    rows = [dialog.colormap_list.item(r).text() for r in range(dialog.colormap_list.count())]
    assert "cividis — 2 heatmaps" in rows and not any(r.startswith("magma") for r in rows)
    assert dialog._selected_tokens == frozenset({_colormap_cache_token("cividis")})
    assert (window.heatmap_series[0].vmin, window.heatmap_series[2].vmin) == (0.0, 2.0)
    # per-heatmap combos followed the swap
    assert controls._group_widgets[0]["cmap_combo"].currentText() == "cividis"

    # unknown name: nothing changes, status explains
    dialog.swap_combo.setCurrentText("not-a-colormap")
    dialog.swap_btn.click()
    assert [s.colormap for s in window.heatmap_series] == [
        "cividis", "viridis", "cividis", "plasma"
    ]
    assert "Unknown colormap" in dialog.status_label.text()
    dialog.close()
    controls.close()
