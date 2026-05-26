# TODO

## Known issues

- [ ] When labels are provided that cover a superset of the data (e.g. a full 48h hypnogram is provided, but only 1h of trace data are provided), no label overlays appear -- even ones the user creates from within the loupe session.

## Features

- [ ] Add the ability to save and load configurations (gains, hidden subplots).
- [ ] Add multi-window/multi-screen support.
- [ ] Add support for display of larger-than-memory data (e.g. 48h full-neuropixel LFPs), but only if it can be done without risking degradation of performance for in-memory data. Performance must remain top priority.

### Labels

- [ ] Add a way to view multiple, possibly overlapping, sets of labels simultaneously.
  - The user should be able to choose whether 1 or more sets of labels are drawn as overlays. If labels from two distinct sets overlap, they should just be drawn over each other. It is up to the user to set their colors and alphas so that this doesn't look awful, if the user expects it to happen.
  - It should be possible to plot labels as their own narrow subplot, under e.g. a traces subplot.
  - It would be nice if different subplots could use different label overlays, but this may become extremely cumbersome for many subplots and labels, and could degrade performance.
- [ ] Add support for filtering labels (e.g. based on probe and condition).

### TraceConfig

#### Dense mode

- [ ] When dense traces are colored by a key (e.g. anatomy), there should be an easy way to view the legend.

### RasterConfig

- [ ] Consider using `PlotCurveItem` or `PlotScatteritem` instead of `PlotDataItem` for optimum performance.

### LabelStripConfig

- [ ] Add the ability to display labels as a subplot aligned to the data window.

## Other

- [ ] `app.py` is getting huge, consider refactoring for agent friendliness.
- [ ] When displaying very long pieces of data, scientific notation should not be used for the time axis labels. 36000 seconds should display as 36000, not 3.6e4, or 36 kiloseconds.
- [ ] If no video is provided, the space used for the video viewer should be used for something else, e.g. the label / hypnogram views.
- [ ] Include small sample data, for testing. Consider DataLad / GIN / GitLFS, or similar.
- [ ] Add a synthetic data generation utility for testing.
