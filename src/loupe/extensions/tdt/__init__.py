"""TDT block discovery, lazy streams, and the graphical block launcher."""


def open_block(path=None, *, parent=None):
    """Show the TDT launcher. Run ``%gui qt6`` first when using a notebook."""
    from PySide6 import QtWidgets
    from .launcher import TDTLauncher

    app = QtWidgets.QApplication.instance()
    created = app is None
    if created:
        app = QtWidgets.QApplication([])
    if path is None:
        path = QtWidgets.QFileDialog.getExistingDirectory(parent, "Choose a TDT block")
        if not path:
            return None
    dialog = TDTLauncher(parent=parent, path=path)
    dialog.show()
    # Keep top-level launchers alive when called from a notebook without assignment.
    if not hasattr(app, "_loupe_launchers"):
        app._loupe_launchers = []
    app._loupe_launchers.append(dialog)
    dialog.destroyed.connect(
        lambda: (
            app._loupe_launchers.remove(dialog)
            if dialog in app._loupe_launchers
            else None
        )
    )
    if created:
        app.exec()
    return dialog
