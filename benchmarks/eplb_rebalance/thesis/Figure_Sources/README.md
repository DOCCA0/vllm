# Editable figure sources

This directory contains the source code and final input data used to generate
the thesis figures, as requested by the thesis-writing guidelines.

- `ILP/` contains the two-stage ILP implementation and illustrative instance
  used for `example_hotspot_before.png`, `example_hotspot_after.png`, and
  `example_hotspot_links.png`.
- `results/serving_nixl_20260911_async_random500/` contains the plotting script,
  final benchmark JSON files, and one-second NIC samples used for
  `e2e_improvement.png` and `nic_timeseries.png`.
- `results/serving_nixl_20260911_async_random500_profile/` contains the plotting
  script and measured profile summary used for
  `scheduler_cost_vs_saved.png`. Its script reads the E2E durations from the
  adjacent final E2E directory.
- The asynchronous EPLB control-loop figure is native TikZ and remains directly
  editable in `Chapters/02_background.tex`.

These files are research artifacts and are not required to compile the thesis.
