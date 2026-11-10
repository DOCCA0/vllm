# Thesis source

Build from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error Thesis.tex
makeglossaries Thesis
latexmk -pdf -interaction=nonstopmode -halt-on-error Thesis.tex
```

The generated `Thesis.pdf` includes front matter, bibliography, appendices,
acronym/list pages, acknowledgements, and two-sided blank pages. Its six main
chapters follow the MSc writing guidelines: introduction; background and
related work; problem statement and formulation; methodology and system
design; experimental evaluation; and conclusions.

`Executive_Summary.tex` builds the separate executive summary required for a
full thesis:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error Executive_Summary.tex
```

`Figure_Sources/` contains editable plotting code and the final input data used
by every raster result figure; the EPLB control-loop source is native TikZ in
`Chapters/02_background.tex`. `GUIDELINE_COMPLIANCE.md` maps the source to the
requirements in the MSc thesis-writing guide.

Two upload archives are generated one directory above this source tree:

- `thesis_overleaf.zip` contains the complete thesis and editable figure
  sources. Its main document is `Thesis.tex`.
- `executive_summary_overleaf.zip` is a standalone executive-summary project.
  Its main document is `Executive_Summary.tex`.

The cover uses the author-provided 2026--2027 academic year. The template class
was adjusted to render the co-advisor field, which the original class accepted
but did not display.

Final reported data come only from:

- `../results/serving_nixl_20260911_async_random500/`
- `../results/serving_nixl_20260911_async_random500_profile/`

The ILP is an offline two-stage lexicographic optimum: it first minimizes the
worst modeled inference time, then minimizes active batches while preserving
that optimum. It does not impose the fixed rank-disjoint-flow limit used by the
merged deterministic greedy implementation.
