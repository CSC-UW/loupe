# TDT validation — 2026-09-26

Implemented on `codex/tdt-extension`. All recording files were read-only during
validation; configuration round trips used disposable fixtures under `/tmp`.

## Automated checks

- Full suite: **364 passed**. Final extension/View-Config checks: **50 passed**.
- Focused tests cover channel order, exact clock searches, gap handling,
  truncated/overlapping sources, SEV metadata and logs, TEV packet clocks,
  peak-preserving bounded reads, epoc clipping/TTL, custom annotation columns,
  video frame counts, worker cancellation, optional SDK errors, invalid and
  changed configs, atomic-save failures, GUI layout and complete state replay.
- Saving/restoring includes label notes, epoc appearance, hidden dense channels,
  gain, plot order and navigation. Rendering was inspected from Qt screenshots.
- The real launcher was also driven through asynchronous discovery and its
  Load block button; video startup and worker cleanup passed.
- Wheel and source distribution built successfully. The wheel includes the
  extension, `tdt` dependency extra, and `loupe` console entry point.
- Ruff and whitespace checks passed on the implementation.

Commands (shared environment):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 QT_QPA_PLATFORM=offscreen \
  /data/slap_analysis/slap_mi_2_sleep/.venv/bin/python -m pytest -q

QT_QPA_PLATFORM=offscreen \
  /data/slap_analysis/slap_mi_2_sleep/.venv/bin/python \
  scripts/validate_tdt_blocks.py /path/to/block-A /path/to/block-B \
  --output /tmp/loupe-tdt-validation
```

The real-block script writes only its JSON report and screenshots to the chosen
output directory. It does not create or update `.loupe` in the input blocks.
The final run's machine-readable results are preserved in
[`validation/tdt-example-blocks.json`](validation/tdt-example-blocks.json).

## Real recordings

| Check | ACRTO_P1-sleep_calibration | HALOMOL_3-251214-101210 |
|---|---:|---:|
| Duration | 14,813.48 s | 85,902.28 s |
| Metadata discovery | 1.05 s | 0.29 s |
| Selected-data preparation | 0.12 s | 0.03 s |
| Window construction/render | 0.19 s | 0.11 s |
| Median 10-s navigation update | 25.7 ms | 17.2 ms |
| Maximum tested navigation update | 32.8 ms | 25.6 ms |
| Exact SDK window comparisons | 22 | 14 |
| Exact native packet checks | 108 | 15 |
| Camera onset count / video frames | 148,133 / 148,133 | 85,902 / 85,902 |

Timings use the workstation's warm filesystem cache and Qt's offscreen renderer.
They are measurements of these selections, not latency guarantees. The first
viewer includes 16 LFP channels, two EEGr lines, Bttn shading plus TTL, the eSpk
raster (4,153,725 events), and video. The second includes both EEG_ and EEGr
channels plus video. Both videos opened, decoded and followed the selected
cursor. Screenshots were inspected for trace/raster layout, shading and video.

A separate raw-data run enabled all 16 NNXr channels at 24,414.0625 Hz:

| Visible window | Measured update times |
|---|---|
| 10 s | 9–22 ms |
| 60 s | 30–33 ms |
| 600 s | 125–149 ms |

Those raw-window checks read visible data through the envelope renderer. No
full NNXr recording was copied into an in-memory sample/timestamp array.

### Timing evidence and scope

SDK comparisons covered the first and last channel of each store, near the
beginning, at 10 seconds, and near the end for SEV stores; TEV comparisons used
the first two windows. Native-file checks covered **every channel** of all
streams, at its first, middle and final indexed packet. They compared the full
packet sample payload against file bytes and verified exact TSQ timestamps.

The day-long block exposes why packet timestamps matter: Wavt's nominal
metadata says 1017.252625 Hz while packet timing corresponds to approximately
1017.25260417 Hz. Uniform extrapolation differs by about 1.76 ms near the end.
The SDK's TEV window rounding can select a different late sample; the extension
keeps the native packet clocks and payloads rather than reproducing that drift.
SEV streams use their packet anchors as well when a complete single-file index
is available.

Camera alignment was verified structurally (onsets, continuous frame counters,
exact counts and decoded cursor frames). No independent physical stimulus or
camera-latency measurement was performed. Validation ran on Linux with Qt
offscreen; macOS, Windows and a separate native OpenGL performance run were
not tested. Split-file and acquisition-gap behavior has synthetic coverage;
the two supplied examples are single-file-per-channel blocks.
