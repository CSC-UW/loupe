"""Standalone Loupe application: ``loupe`` or ``python -m loupe``."""

from __future__ import annotations


def main():
    import argparse
    import sys
    from PySide6 import QtWidgets
    from loupe.app import LoupeApp

    parser = argparse.ArgumentParser(description="Loupe neuroscience data viewer")
    parser.add_argument("--tdt", metavar="BLOCK", help="Open the TDT block launcher")
    args = parser.parse_args()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    if args.tdt:
        from loupe.extensions.tdt import open_block

        window = open_block(args.tdt)
    else:
        window = LoupeApp()
        window.show()
    # Keep a reference for the lifetime of the event loop.
    app._loupe_main_window = window
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
