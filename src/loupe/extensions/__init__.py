"""Lazy acquisition extensions; SDKs are imported only when used.

Third-party packages register a factory under the ``loupe.extensions`` entry
point group. The factory returns an :class:`Extension`.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, metadata, util
from typing import Callable


@dataclass(frozen=True)
class Extension:
    id: str
    name: str
    opener: str | Callable
    dependency: str | None = None
    extra: str | None = None
    action: str = "Open folder…"

    @property
    def available(self) -> bool:
        return self.dependency is None or util.find_spec(self.dependency) is not None

    def open(self, parent=None, path=None):
        if not self.available:
            raise ImportError(
                f"Install this extension with: uv add 'loupe[{self.extra or self.id}]'"
            )
        target = self.opener
        if isinstance(target, str):
            module, name = target.split(":", 1)
            target = getattr(import_module(module), name)
        return target(parent=parent, path=path)


def discover_extensions() -> tuple[list[Extension], list[str]]:
    extensions = [
        Extension(
            "tdt",
            "TDT",
            "loupe.extensions.tdt:open_block",
            dependency="tdt",
            extra="tdt",
            action="Open block…",
        )
    ]
    errors = []
    known = {"tdt"}
    for entry in sorted(
        metadata.entry_points(group="loupe.extensions"), key=lambda e: e.name
    ):
        try:
            extension = entry.load()()
            if (
                not isinstance(extension, Extension)
                or not extension.id
                or extension.id in known
            ):
                raise ValueError(
                    "Factory must return an Extension with a unique nonempty id"
                )
            known.add(extension.id)
            extensions.append(extension)
        except Exception as exc:
            errors.append(f"{entry.name}: {exc}")
    return extensions, errors


def install_menu(window, file_menu):
    """Install extension actions on any Loupe window without importing SDKs."""
    from PySide6 import QtWidgets

    menu = file_menu.addMenu("Extensions")
    extensions, errors = discover_extensions()

    def launch(extension):
        try:
            dialog = extension.open(parent=window)
            if dialog is not None:
                # Parent ownership protects dialogs and jobs until completion.
                dialog.show()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(window, extension.name, str(exc))

    for extension in extensions:
        submenu = menu.addMenu(extension.name)
        action = submenu.addAction(extension.action)
        action.triggered.connect(lambda checked=False, ext=extension: launch(ext))
        if not extension.available:
            action.setToolTip(
                f"Install with uv add 'loupe[{extension.extra or extension.id}]'"
            )
    if errors:
        action = menu.addAction("Extension loading errors…")
        action.triggered.connect(
            lambda: QtWidgets.QMessageBox.warning(
                window, "Extensions", "\n".join(errors)
            )
        )
    window.extensions_menu = menu
