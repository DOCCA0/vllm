# Thesis source

Build from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error Thesis.tex
makeglossaries Thesis
latexmk -pdf -interaction=nonstopmode -halt-on-error Thesis.tex
```

The generated `Thesis.pdf` has 60 PDF pages including front matter,
bibliography, appendices, acronym/list pages, acknowledgements, and two-sided
blank pages. Its six main chapters follow the reference thesis structure:
introduction, background, related work, proposed system, experimental
evaluation, and conclusions.

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
