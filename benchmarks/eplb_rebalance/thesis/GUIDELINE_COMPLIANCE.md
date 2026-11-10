# MSc thesis guideline compliance

This checklist maps the submitted source to *Guidelines for MSc thesis
writing* (5 August 2024).

| Guideline | Thesis evidence |
|---|---|
| Official Overleaf-compatible format | `Thesis.tex` uses the PoliMi School of Industrial and Information Engineering class in `Configuration_Files/`. |
| Full thesis and executive summary | `Thesis.pdf` and the standalone `Executive_Summary.pdf` are both provided. |
| English and Italian abstracts | Both appear in the front matter of `Thesis.tex`. |
| Introduction: motivation, contributions, organization | Chapter 1 contains dedicated Motivation, Contributions, and Thesis Structure sections. |
| State of the art and differentiation | Chapter 2 combines background with related work and ends with explicit positioning against prior work. |
| Formal problem statement | Chapter 3 identifies inputs, outputs, constraints, objectives, the network model, and the two-stage ILP. |
| Methodology, pseudocode, and baseline | Chapter 4 gives system design and pseudocode; Chapter 5 defines the legacy baseline and controlled A/B procedure. |
| Numerical evaluation and takeaways | Chapter 5 documents scenario assumptions, metrics, quantitative results, qualitative interpretation, research-question answers, and threats to validity. |
| Conclusions and future work | Chapter 6 summarizes outcomes and lists future directions. |
| Figures and tables | Every numbered figure, table, and algorithm is referenced in the text and has a descriptive caption. Lists of figures and tables are in the front matter. |
| Editable figure sources | `Figure_Sources/` contains plotting code, the final input data, and the ILP generator; the control-loop diagram is native TikZ. |
| At least 30 references | The full thesis renders 31 bibliography entries, all cited in the text. |
| Readability aids | The thesis uses sections, subsections, numbered research questions, lists, tables, figures, and pseudocode. |
| Evaluation assumptions | Chapter 5 states the topology, model, workload, cache setting, baseline, run order, metric definitions, data provenance, and limitations. |

The only guideline that cannot be verified from the source archive is the
workflow rule for replying to supervisors' Overleaf comments without deleting
them. That requirement must be followed in the shared Overleaf project.
