"""Runnable demo of Loupe's live parameter **Tuner**.

As a script (blocks until you close the window)::

    python examples/tuner_demo.py

In a notebook, enable the Qt loop first, then run the body of ``main()``
without the trailing block::

    %gui qt6

The window opens with three noisy traces, each carrying a matched-filter
*overlay*. A Tuner dock (toggle with Ctrl+T) exposes the filter's ``tau_s``;
drag it and every overlay recomputes live. Afterwards ``tau.value`` holds the
value you settled on, or click **Copy values** to grab it as a dict.

The ``matched_filter`` here is a small self-contained stand-in — in real use you
would wrap any pure function, e.g.
``tunable(wis.scope.pro.ls_to_matched_filter, raw_ls, tau_s=tau)``.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from loupe import Param, TraceConfig, tunable, view


def matched_filter(da: xr.DataArray, tau_s: float = 0.05, n_tau: float = 5.0) -> xr.DataArray:
    """Onset-aligned, unit-energy exponential matched filter (toy, pure)."""
    t = da.coords["time"].values
    fs = 1.0 / float(np.median(np.diff(t)))
    klen = max(1, int(round(n_tau * tau_s * fs)))
    kern = np.exp(-np.arange(klen) / (tau_s * fs))[::-1]  # reversed → peak at onset
    kern = kern / np.sqrt((kern**2).sum())  # unit energy

    def _conv(y: np.ndarray) -> np.ndarray:
        return np.convolve(y, kern, mode="same")

    out = xr.apply_ufunc(
        _conv, da,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
    )
    out.name = "matched_filter"
    return out


def main() -> None:
    n = 4000
    t = np.arange(n) / 1000.0  # 1 kHz
    rng = np.random.default_rng(0)
    y = rng.standard_normal((3, n)).astype("float32")
    for col in (600, 1800, 3000):  # a few transients to filter
        y[:, col] += 8.0
    ls = xr.DataArray(
        y, dims=("syn_id", "time"),
        coords={"syn_id": [10, 11, 12], "time": t}, name="ls",
    )

    tau = Param(0.05, 0.01, 0.3, name="tau_s")  # default, min, max
    view(TraceConfig(
        data=ls,
        overlay_arrays=[tunable(matched_filter, ls, tau_s=tau)],
    ))


if __name__ == "__main__":
    main()
