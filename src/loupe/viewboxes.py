"""Custom pyqtgraph ``ViewBox`` / ``PlotItem`` subclasses used by Loupe.

These are leaf widgets — they own no app state and emit Qt signals that the
main :class:`loupe.app.LoupeApp` wires up at plot-creation time. Kept in a
separate module so the pyqtgraph customisation lives apart from the main
app logic.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets


class SelectableViewBox(pg.ViewBox):
    sigDragStart = QtCore.Signal(float)
    sigDragUpdate = QtCore.Signal(float)
    sigDragFinish = QtCore.Signal(float)
    sigWheelScrolled = QtCore.Signal(int)
    sigWheelSmoothScrolled = QtCore.Signal(int)
    sigWheelCursorScrolled = QtCore.Signal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # *** FIX 1: Explicitly disable default mouse pan/zoom behavior ***
        # This allows our custom event handlers to take full control.
        self.setMouseEnabled(x=False, y=False)
        self._drag = False

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            x = float(self.mapSceneToView(ev.scenePos()).x())
            if ev.isStart():
                self._drag = True
                self.sigDragStart.emit(x)
                ev.accept()
                return
            elif ev.isFinish():
                if self._drag:
                    self.sigDragFinish.emit(x)
                self._drag = False
                ev.accept()
                return
            else:
                if self._drag:
                    self.sigDragUpdate.emit(x)
                ev.accept()
                return
        # Do not call super, to prevent default drag (pan) behavior

    def wheelEvent(self, ev, axis=None):
        dy = 0
        if hasattr(ev, "delta"):
            try:
                dy = ev.delta()
            except Exception:
                dy = 0
        else:
            try:
                ad = ev.angleDelta()
                dy = ad.y() if hasattr(ad, "y") else 0
            except Exception:
                dy = 0
        direction = 1 if dy > 0 else -1
        # Use Shift+wheel for smooth scrolling; otherwise page
        try:
            mods = QtWidgets.QApplication.keyboardModifiers()
        except Exception:
            mods = QtCore.Qt.KeyboardModifier.NoModifier
        # Ctrl: cursor scroll within window (like dragging cursor slider)
        if mods & QtCore.Qt.KeyboardModifier.ControlModifier:
            self.sigWheelCursorScrolled.emit(int(dy))
        # Shift: smooth scroll the window
        elif mods & QtCore.Qt.KeyboardModifier.ShiftModifier:
            self.sigWheelSmoothScrolled.emit(direction)
        else:
            self.sigWheelScrolled.emit(direction)
        ev.accept()


class DenseViewBox(SelectableViewBox):
    """ViewBox for dense plots — Alt+wheel gain (all groups),
    Ctrl+Alt+wheel gain (hovered group only),
    Shift+Alt+wheel vertical scroll."""

    sigWheelGainAdjust = QtCore.Signal(int)
    sigWheelGainAdjustFocused = QtCore.Signal(int)
    sigWheelVerticalSmooth = QtCore.Signal(int)

    def wheelEvent(self, ev, axis=None):
        try:
            mods = QtWidgets.QApplication.keyboardModifiers()
        except Exception:
            mods = QtCore.Qt.KeyboardModifier.NoModifier
        alt = bool(mods & QtCore.Qt.KeyboardModifier.AltModifier)
        shift = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
        # Accept both ControlModifier (Ctrl on Linux/Win, Cmd on macOS) and
        # MetaModifier (the macOS ⌃ Control key with Qt's default swap), so
        # the macOS-keycap "Control" key works as the user expects.
        ctrl = bool(
            mods
            & (
                QtCore.Qt.KeyboardModifier.ControlModifier
                | QtCore.Qt.KeyboardModifier.MetaModifier
            )
        )
        if alt:
            dy = 0
            if hasattr(ev, "delta"):
                try:
                    dy = ev.delta()
                except Exception:
                    dy = 0
            else:
                try:
                    ad = ev.angleDelta()
                    dy = ad.y() if hasattr(ad, "y") else 0
                except Exception:
                    dy = 0
            direction = 1 if dy > 0 else -1
            if shift:
                self.sigWheelVerticalSmooth.emit(direction)
            elif ctrl:
                self.sigWheelGainAdjustFocused.emit(direction)
            else:
                self.sigWheelGainAdjust.emit(direction)
            ev.accept()
        else:
            super().wheelEvent(ev, axis)


# *** FIX 2: Create a PlotItem that signals when the mouse enters/leaves it ***
class HoverablePlotItem(pg.PlotItem):
    sigHovered = QtCore.Signal(
        object, bool
    )  # Emits self, True on enter, False on leave

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Required to receive hover events
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, ev):
        self.sigHovered.emit(self, True)
        super().hoverEnterEvent(ev)

    def hoverLeaveEvent(self, ev):
        self.sigHovered.emit(self, False)
        super().hoverLeaveEvent(ev)
