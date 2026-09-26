# TDT extension implementation

Branch: `codex/tdt-extension`. Specification: `Loupe TDT Extenstion.md`.

## Work plan

1. Add an optional `tdt` extra and lazy extension registry, File menu integration,
   and a standalone `loupe` launcher. External extensions use entry points.
2. Inspect TDT metadata without reading stream samples or spike waveforms.
   Read streams from disk by visible window, preserving block-relative timing,
   channel identity, acquisition gaps and segment boundaries.
3. Build a responsive block launcher with stream/channel/display selection,
   epoc shading/TTL/color, channel rasters, video/frame epoc, annotation import,
   plot order, height, color, and saved configuration controls.
4. Persist block selections and runtime view state atomically under `.loupe`.
   Preserve editable annotations independently of immutable acquisition epocs.
5. Validate synthetic edge cases, SDK parity on both supplied blocks, video
   timestamps/counts, saved-state replay, actual rendered GUI, and regressions.

## Acceptance checks

- Core Loupe starts without the optional SDK; missing extras have actionable UI.
- Opening both supplied blocks never materializes the full stream data.
- Selected channels and plots match the requested order and styles.
- Epoc overlay and TTL agree exactly; annotations can overlap epocs.
- Rasters show actual channel IDs, including channels with no spikes in view.
- Video uses recorded epoc timestamps with count validation, never encoded FPS.
- Reopening and updating a saved block restores both choices and presentation.
- Discovery/loading are cancellable without orphaned threads or partial config.
- Focused and full tests pass; real-block coverage and limits are documented.

## Progress

- Read specification, repository guidance, TDT SDK documentation and local SDK.
- Confirmed both example blocks and first-block local video are accessible.
- First-block header discovery: about 0.85 s; 4,153,725 eSpk timestamps,
  21 Bttn intervals, and 148,133 Cam1 timestamps.
- Implemented extension registry/extra/CLI, asynchronous launcher, native SEV
  and TEV window readers, epoc shading/TTL, channel rasters, video, annotations,
  ordering/styles, atomic block configs, and runtime state replay.
- Real-data checks cover both supplied blocks, every stream channel's native
  first/middle/final packets, SDK window comparisons and both videos.
- Full suite passed 364 tests; final focused checks passed 50 tests. Raw
  16-channel NNXr navigation was measured up to
  a 600-second viewport. See `tdt-validation.md` for evidence and test scope.
