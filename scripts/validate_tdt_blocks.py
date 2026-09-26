"""Read-only real-block checks; writes reports/screenshots only under --output.

Run in the shared environment with QT_QPA_PLATFORM=offscreen for headless QA.
"""

from pathlib import Path
import argparse
import json
import time
import resource
import numpy as np
import tdt
from loupe.extensions.tdt.reader import inspect_block
from loupe.extensions.tdt.session import default_config, prepare_session, launch_session
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("blocks", nargs="+", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
app = QApplication([])
paths = args.blocks
reports = []
for i, p in enumerate(paths):
    start = time.monotonic()
    b = inspect_block(p)
    report = {
        "block": p.name,
        "discovery_s": time.monotonic() - start,
        "duration_s": b.duration,
        "parity": [],
    }
    for n, s in b.streams.items():
        if s.error:
            continue
        for ch in [s.channels[0], s.channels[-1]][: 2 if len(s.channels) > 1 else 1]:
            t, y = s.signal(ch)
            for sec in (
                [0.0, 10.0, b.duration - 2] if s.tev_path is None else [0.0, 10.0]
            ):
                sdk = tdt.read_block(
                    str(p),
                    evtype=["streams"],
                    store=n,
                    channel=ch,
                    t1=sec,
                    t2=sec + 1.0,
                )
                d = sdk.streams[n]
                data = np.asarray(d.data).ravel()
                offset = int(round(d.start_time * s.fs))
                ours = y[offset : offset + len(data)]
                np.testing.assert_allclose(ours, data, rtol=0, atol=0)
                report["parity"].append(
                    {"store": n, "channel": ch, "t": sec, "samples": len(data)}
                )
    report["native_clock_checks"] = []
    for n, s in b.streams.items():
        if s.error:
            continue
        for ch in s.channels:
            axis, signal = s.signal(ch)
            if not hasattr(axis, "packet_starts"):
                continue
            for k in [0, len(axis.packet_starts) // 2, len(axis.packet_starts) - 1]:
                sample = int(axis.packet_starts[k])
                stamp = float(axis.packet_times[k])
                assert axis[sample] == stamp
                if s.tev_path is None:
                    path, times, offsets, count = s.packet_sources[ch]
                else:
                    path, times, offsets, count = (
                        s.tev_path,
                        s.timestamps,
                        s.offsets,
                        s.chunk_samples,
                    )
                with path.open("rb") as f:
                    f.seek(int(offsets[k]))
                    direct = np.frombuffer(
                        f.read(count * np.dtype(s.dtype).itemsize), dtype=s.dtype
                    )
                np.testing.assert_equal(signal[sample : sample + count], direct)
                report["native_clock_checks"].append(
                    {
                        "store": n,
                        "channel": ch,
                        "packet": k,
                        "sample": sample,
                        "time_s": stamp,
                    }
                )
    print("PARITY", p.name, len(report["parity"]), flush=True)
    c = default_config(b)
    for r in c["rows"]:
        r["enabled"] = r["store"] in (
            ["LFP_", "EEGr", "Bttn", "eSpk"]
            if "LFP_" in b.streams
            else ["EEG_", "EEGr"]
        )
        if r["store"] == "Bttn":
            r["mode"] = "both"
    start = time.monotonic()
    prepared = prepare_session(b, c)
    report["prepare_s"] = time.monotonic() - start
    start = time.monotonic()
    w = launch_session(prepared, save=False)
    app.processEvents()
    report["window_s"] = time.monotonic() - start
    if "Bttn" in b.epocs and len(b.epocs["Bttn"].onset):
        onset = float(b.epocs["Bttn"].onset[0])
        w.window_len = min(45, b.duration)
        w.window_spin.setValue(w.window_len)
        w.window_start = max(0, onset - 10)
        w._apply_x_range()
        w._set_cursor_time(onset + 2)
    else:
        w.window_start = max(0, b.duration / 2 - 5)
        w._apply_x_range()
        w._set_cursor_time(b.duration / 2)
    for _ in range(100):
        QTest.qWait(20)
        if all(
            s.is_open and s.displayed_frame_idx is not None
            for s in w.video_slots
            if s.video_path
        ):
            break
    print(
        "VIDEO", [(s.is_open, s.displayed_frame_idx) for s in w.video_slots], flush=True
    )
    assert all(
        s.is_open and s.displayed_frame_idx is not None
        for s in w.video_slots
        if s.video_path
    ), "Video failed to open/decode"
    w.grab().save(str(args.output / f"viewer-{i}.png"))
    timings = []
    for sec in np.linspace(0, b.duration - 10, 10):
        start = time.monotonic()
        w.window_len = 10
        w.window_start = sec
        w._apply_x_range()
        app.processEvents()
        timings.append(time.monotonic() - start)
    report["navigation_s"] = timings
    report["video"] = [
        {"open": s.is_open, "frames": s.video_frame_counts}
        for s in w.video_slots
        if s.video_path
    ]
    report["max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    reports.append(report)
    w.close()
    app.processEvents()
    print("REPORT", json.dumps(report), flush=True)
(args.output / "results.json").write_text(json.dumps(reports, indent=2))
