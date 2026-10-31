# Gündem Score v1

[Türkçe](SCORING.tr.md)

A fixed **0–100 composite quality score** for this Turkish headline classification
task. Higher is better. It is a project-defined index, not a standard benchmark,
an accuracy percentage, or a measure of general LLM intelligence.

```text
Gündem Score = 30 × topic_micro_F1
             + 20 × type_accuracy
             + 30 × (1 − mean_score_MAE / 3)
             + 20 × exact_match

mean_score_MAE = (impact_MAE + severity_MAE + urgency_MAE) / 3
```

F1, type accuracy and exact match are fractions in [0, 1]. The three numeric
fields range from 0 to 3, so their MAE ranges from 0 to 3. A perfect result is
100; a result with every response invalid is 0.

| Component | Maximum points | Purpose |
| --- | ---: | --- |
| Topic micro-F1 | 30 | Main news classification task |
| Type accuracy | 20 | Event category agreement |
| Numeric score accuracy | 30 | Partial credit for impact, severity and urgency; 10 points each |
| All-field exact match | 20 | Bonus for getting the entire response correct |

These weights are an explicit editorial choice, not fitted to this leaderboard
or statistically learned. Exact match intentionally overlaps the field metrics
to reward joint correctness. Speed, parameter count and memory are separate;
they do not change the quality score. Invalid outputs are already penalized in
every input metric, so there is no second validity multiplier.

There is no min/max normalization against competitors. Adding a model does not
change anyone else's score. JSON preserves unrounded points and component
contributions; tables show one decimal. Rankings use unrounded scores and then
exact match, F1 and model ID as deterministic tie breakers. A small difference
on this 101-example, single-seed test is not evidence of a meaningful advantage.
Weights/formula changes require a new score version; do not compare versions.

Only compare scores on matching data, prompts and decoding conditions. Current
scores measure teacher-label agreement; the small test was used previously.
All original models failed the requested JSON contract under this experiment's
short prompt, explaining their zero scores. Missing evaluations remain missing.

`bash run_experiments.sh report` adds scores to CSV/JSON/Markdown and creates
`overall-tr.png/svg` and `overall-en.png/svg` alongside the detailed charts.
Old summary JSON files can also be plotted with `plot_experiments.py`; scores
are computed from their existing metrics without running a model again.

[Current results](multimodel-results/README.md) · [Implementation](../lab/scoring.py)
