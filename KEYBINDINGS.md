## Loupe keybindings

This is the complete reference for keyboard shortcuts and mouse/wheel interactions in Loupe.

A few notes:

- On macOS, use **Cmd** in place of **Ctrl** for any binding listed below.
- **State hotkeys** for labeling (e.g. `w`, `1`, `r`) are user-configurable. The bundled `example_state_definitions.json` provides defaults; see the "State definitions" section of the README to customize them via JSON or the `keymap=` / `state_definitions=` kwargs of `view()`. The Help menu inside the app ("Shortcuts / Help") prints the active state hotkeys at runtime.
- Plot-targeted bindings (Y-zoom, focused subplot resize) act on the plot currently under the mouse cursor.

---

### Navigation & windowing

| Shortcut | Action | Context / Notes |
| --- | --- | --- |
| `[` / PageUp | Page window left (previous window) | |
| `]` / PageDown | Page window right (next window) | |
| Left arrow | Step selected video one frame back | Hold to repeat. Pick the target video via View → Frame Step Target. |
| Right arrow | Step selected video one frame forward | Hold to repeat. |
| Mouse wheel | Page left/right one full window | |
| Shift + wheel | Smooth scroll window | Fraction of window length, configurable via View → Adjust Smooth Scroll Speed…. |
| Ctrl + wheel | Cursor scrub within current window | Like dragging the cursor slider. |
| F11 | Toggle fullscreen | Also available as View → Fullscreen. |

### Playback

| Shortcut | Action | Context / Notes |
| --- | --- | --- |
| Space | Toggle playback | Loops within the current window. Speed is set via View → Set Playback Speed… (0.25× – 4×). |

### Selection & labeling

| Shortcut | Action | Context / Notes |
| --- | --- | --- |
| Click‑drag in any plot | Create / update selection | Drag handles to extend or refine. |
| State hotkey (e.g. `w`, `1`, `r`) | Apply that state's label to the active selection | Bindings come from the active state config — see notes above. |
| `0` | Clear all labels in the selected range | Splits existing intervals as needed. |
| Backspace | Delete the most recently ending label | Also available as Edit → Delete last label. |

### Epoch navigation & notes

| Shortcut | Action | Context / Notes |
| --- | --- | --- |
| `n` | Jump to next epoch | Requires labels. |
| `b` | Jump to previous epoch | Requires labels. |
| Ctrl + Shift + N | Add or edit a note for the epoch at the cursor | Falls back to the most recently labeled epoch if cursor is unlabeled. |
| Ctrl + J | Open "Jump to Epochs" dialog | Filter by state or note text; double-click an epoch to navigate. |

### Zoom & axes

| Shortcut | Action | Context / Notes |
| --- | --- | --- |
| Ctrl + 1 | Zoom Y-axis in on the hovered plot | |
| Ctrl + 2 | Zoom Y-axis out on the hovered plot | |
| Ctrl + D | Open Y-Axis Controls dialog | Per-trace autorange toggle and min/max input. |
| `z` | Toggle hypnogram zoom | Window ± padding vs. full extent. |
| `h` | Toggle hypnogram visibility | Frees vertical space for videos. |
| Ctrl + L | Toggle label strip | Thin color band of the labels, pinned above the plots and aligned to the current window. Stays visible while scrolling channels. |
| Ctrl + Shift + L | Toggle label overlays | Show/hide the translucent label shading drawn across the subplots. The label strip and hypnogram are unaffected. Start with overlays off via `view(interval_label_overlays=False)`. |

### Dense view

| Shortcut | Action | Context / Notes |
| --- | --- | --- |
| Alt + wheel | Adjust trace gain (amplitude scaling) for all dense subplots | Dense mode plots only. |
| Ctrl + Alt + wheel | Adjust trace gain for the hovered dense subplot only | Dense mode plots only. On macOS, ⌘ or ⌃ both work as the Ctrl modifier here. Does nothing when no dense subplot is hovered. |
| Shift + Alt + wheel | Smooth vertical scroll through traces | ~3 traces per notch. Dense mode plots only. |
| Ctrl + G | Open Dense View Controls dialog | Gain slider, step, traces per page. |

### Subplot management

| Shortcut | Action | Context / Notes |
| --- | --- | --- |
| Ctrl + H | Open Subplot Control Board | Height, visibility, and order for all subplots (stacked, dense, raster). |
| Ctrl + Shift + , | Increase height of the focused subplot | Acts on the plot under the cursor. Other plots shrink proportionally. |
| Ctrl + Shift + . | Decrease height of the focused subplot | Acts on the plot under the cursor. |
| Ctrl + Shift + 0 | Reset focused subplot height to 1.0× | |

### Raster plots

| Shortcut | Action | Context / Notes |
| --- | --- | --- |
| Ctrl + Shift + R | Toggle Proportional Raster Plots | On by default. Sizes raster plots by row count. |

Brightness, event height, and event thickness are adjusted via View → Adjust Raster Brightness… / Raster Event Height… / Raster Event Thickness… (no dedicated keybinding).

### Heatmap plots

| Shortcut | Action | Context / Notes |
| --- | --- | --- |
| Ctrl + Shift + H | Open Heatmap Plot Controls dialog | Available when heatmap plots are present. |

### Video controls

| Shortcut | Action | Context / Notes |
| --- | --- | --- |
| Ctrl + Shift + 1 … Ctrl + Shift + 9 | Toggle visibility of the Nth video | Same as View → Show *video name*. |

### File / labels

| Shortcut | Action | Context / Notes |
| --- | --- | --- |
| Ctrl + S | Save Labels (overwrite source) | Only enabled when `view()` was called with `labels_writeback=True`. |

### Tables (dialogs)

| Shortcut | Action | Context / Notes |
| --- | --- | --- |
| Ctrl + C / Cmd + C | Copy selection to clipboard | Inside table dialogs (e.g. Jump to Epochs). |
