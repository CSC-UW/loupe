"""Test-session environment for loupe.

loupe is built on PySide6, but several test modules import ``pyqtgraph``
before ``PySide6``. pyqtgraph auto-selects a Qt binding on import and prefers
PyQt6 when it is installed (it is, as a dependency of napari / cellpose /
suite2p in the shared venv); once PyQt6 is loaded, importing PySide6 fails
("could not import module 'PySide6.QtGui'"). Pin the binding before any test
module is imported. Offscreen rendering keeps the suite headless.
"""

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_API", "pyside6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
