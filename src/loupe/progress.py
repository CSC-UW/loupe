"""Lightweight launch-progress reporter.

A single :class:`ProgressReporter` fans out updates to a Qt splash screen
(when one is available) and to stderr.  It is designed for the launch path:
``view()`` builds one, threads it through the slow data converters and into
``LoupeApp.__init__``, then dismisses the splash after ``w.show()``.

Per-item updates inside tight loops (thousands of traces) are throttled by
both wall-clock time and item count so the reporter adds negligible overhead.
Phase boundaries are infrequent and always emitted.

Splash updates use ``QSplashScreen.repaint()`` rather than
``QApplication.processEvents()`` — repaint touches only the splash widget,
while processEvents can fire signals on widgets that are still mid-construction
(e.g. inside ``LoupeApp.__init__``).
"""

from __future__ import annotations

import sys
import time
from typing import TextIO

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    _HAS_QT = True
except ImportError:  # pragma: no cover - Qt is a hard dep in practice
    _HAS_QT = False


_SPLASH_WIDTH = 480
_SPLASH_HEIGHT = 160


class ProgressReporter:
    """Report launch progress to a Qt splash screen and stderr."""

    def __init__(
        self,
        splash: "QtWidgets.QSplashScreen | None" = None,
        stream: TextIO | None = None,
        min_splash_interval: float = 0.05,
    ):
        self._splash = splash
        self._stream = stream if stream is not None else sys.stderr
        self._min_splash_interval = float(min_splash_interval)

        self._phase: str = ""
        self._last_splash_time: float = 0.0
        self._last_item_index: int = -1
        self._last_stderr_item: int = -1
        self._last_n: int = -1

    # ------------------------------------------------------------------ phases

    def phase(self, name: str) -> None:
        """Begin a new phase. Always logged + drawn."""
        self._phase = name
        self._reset_loop_state()
        self._stream.write(f"[loupe] {name}\n")
        self._stream.flush()
        self._draw(name)

    def item(self, i: int, n: int, detail: str = "") -> None:
        """Report iterative progress within the current phase. Throttled.

        Each distinct ``n`` is treated as a new loop scope, so the throttle
        counters reset and the loop's first item is never silently swallowed
        by stale state from a previous loop.
        """
        if n <= 0:
            return

        if n != self._last_n or i == 0:
            self._reset_loop_state()
            self._last_n = n

        now = time.perf_counter()
        step = max(1, n // 200)
        time_due = (now - self._last_splash_time) >= self._min_splash_interval
        count_due = (i - self._last_item_index) >= step
        last_item = i >= n - 1

        if not (time_due and count_due) and not last_item:
            return

        self._last_splash_time = now
        self._last_item_index = i

        text = self._format_item(i, n, detail)
        self._draw(text)

        stderr_step = max(1, n // 10)
        if (i - self._last_stderr_item) >= stderr_step or last_item:
            self._last_stderr_item = i
            self._stream.write(f"[loupe]   {i + 1}/{n} {detail}\n".rstrip() + "\n")
            self._stream.flush()

    def done(self) -> None:
        """Final message; splash is dismissed by the caller via ``splash.finish``."""
        self._stream.write("[loupe] Ready\n")
        self._stream.flush()

    # ------------------------------------------------------------------ helpers

    def _reset_loop_state(self) -> None:
        self._last_item_index = -1
        self._last_stderr_item = -1
        self._last_splash_time = 0.0
        self._last_n = -1

    def _format_item(self, i: int, n: int, detail: str) -> str:
        base = f"{self._phase}: {i + 1}/{n}" if self._phase else f"{i + 1}/{n}"
        return f"{base}  {detail}" if detail else base

    def _draw(self, text: str) -> None:
        if self._splash is None:
            return
        try:
            self._splash.showMessage(
                text,
                int(QtCore.Qt.AlignmentFlag.AlignBottom | QtCore.Qt.AlignmentFlag.AlignHCenter),
                QtCore.Qt.GlobalColor.white,
            )
            self._splash.repaint()
        except Exception:
            # The splash may have been closed early; degrade silently.
            self._splash = None


class NullReporter(ProgressReporter):
    """No-op reporter used when callers don't supply one."""

    def __init__(self) -> None:
        super().__init__(splash=None, stream=_DevNull())

    def phase(self, name: str) -> None:  # noqa: D401 - trivial override
        pass

    def item(self, i: int, n: int, detail: str = "") -> None:
        pass

    def done(self) -> None:
        pass


class _DevNull:
    def write(self, _s: str) -> int:
        return 0

    def flush(self) -> None:
        pass


_NULL_REPORTER = NullReporter()


def null_reporter() -> ProgressReporter:
    """Return a shared no-op reporter callees can use unconditionally."""
    return _NULL_REPORTER


def make_splash(app) -> "QtWidgets.QSplashScreen | None":
    """Build a small splash screen suitable for launch-time progress.

    Returns ``None`` when Qt is unavailable, when ``app`` lacks a usable
    display (offscreen/headless), or when splash creation fails for any
    reason — callers always get either a splash or ``None``.
    """
    if not _HAS_QT or app is None:
        return None
    try:
        screens = app.screens()
    except Exception:
        screens = []
    if not screens:
        return None

    try:
        pixmap = QtGui.QPixmap(_SPLASH_WIDTH, _SPLASH_HEIGHT)
        pixmap.fill(QtGui.QColor(20, 20, 24))

        painter = QtGui.QPainter(pixmap)
        try:
            title_font = QtGui.QFont()
            title_font.setPointSize(20)
            title_font.setBold(True)
            painter.setFont(title_font)
            painter.setPen(QtGui.QColor(230, 230, 230))
            painter.drawText(
                pixmap.rect().adjusted(0, 24, 0, 0),
                int(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter),
                "Loupe",
            )

            sub_font = QtGui.QFont()
            sub_font.setPointSize(10)
            painter.setFont(sub_font)
            painter.setPen(QtGui.QColor(170, 170, 170))
            painter.drawText(
                pixmap.rect().adjusted(0, 64, 0, 0),
                int(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter),
                "Loading…",
            )
        finally:
            painter.end()

        splash = QtWidgets.QSplashScreen(
            pixmap, QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        splash.show()
        app.processEvents()
        return splash
    except Exception:
        return None
