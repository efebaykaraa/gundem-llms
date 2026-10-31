# Multi-model experiments

[Türkçe](MULTI-MODEL.tr.md)

The suite compares each checkpoint **before and after response distillation** on
the same prepared headline dataset. The registry currently contains ten models:
the two existing LFM3 variants and eight additional requested checkpoints.

The [completed local result snapshot](multimodel-results/README.md) contains all
20 evaluations, bilingual charts, CSV/JSON and recorded provenance.

**[Gündem Score v1](SCORING.md)** ranks results on a fixed 0–100 quality scale, with CSV/JSON/Markdown and bilingual `overall` charts generated automatically. Speed remains separate.

## Layout

```text
configs/models.json            # Model registry and shared recipe
lab/                           # Orchestration, model loading, scoring and plots
run_experiments.sh             # Environment-selecting entry point
workspace/                     # Local artifacts; ignored by Git
  datasets/news-v1/            # Frozen manifest and train/validation/test files
  experiments/<model-id>/
    model.lock.json            # Exact model revision
    state.json                 # Current stage / failure detail
    adapter/                   # LoRA weights, checkpoints and training manifest
    train.log
    original.json              # Completed original-model evaluation
    distilled.json             # Completed distilled-model evaluation
    *.predictions.jsonl        # Actual outputs and per-headline errors
  envs/                        # Links to existing training/converter environments
  reports/
    leaderboard.json
    leaderboard.csv
    leaderboard.md
    comparison-tr.png / .svg
    comparison-en.png / .svg
```

Existing LFM adapters are linked from `data-deepseek` and `data-deepseek-base`.
Their original files and older entry points are preserved. Full HF weights remain
in the Hugging Face cache. Evaluating a distilled adapter merges it in CPU memory,
then moves the result to GPU; it does not write an extra full-weight copy for each
model. Export GGUF or a full HF model separately when needed.

## Install and initialize

Use the existing training environment, or create one with `requirements-train.txt`.
For charts and optional hybrid-model acceleration:

```bash
data-deepseek/venv-train/bin/python -m pip install \
  -r requirements-train.txt -r requirements-report.txt -r requirements-kernels.txt

bash run_experiments.sh init --source data-deepseek/prepared
bash run_experiments.sh preflight
```

The source must be a prepared dataset with all three splits and `manifest.json`.
The snapshot is copied only if absent and must have matching hashes on reuse.
A fresh clone must first collect/label/prepare its own data or obtain the original
dataset; the complete dataset and weights are not shipped in Git.

The shell wrapper selects `DISTILL_PYTHON`, then `workspace/envs/training/bin/python`,
then the existing `data-deepseek/venv-train/bin/python`, then `python3`.
For another environment:

```bash
DISTILL_PYTHON=/absolute/path/to/venv/bin/python bash run_experiments.sh preflight
```

For gated Llama repositories, request access on their model pages and use
`hf auth login` locally. The global CLI and the training environment normally
share the standard HF credential cache. Do not put credentials in configuration.

## Run, inspect and resume

```bash
# All ten models, original + distilled; one GPU job at a time.
bash run_experiments.sh run

# A subset (IDs come from configs/models.json).
bash run_experiments.sh run --models gpt-x3-150m llama-3.2-1b qwen3.5-0.8b-base

# Read current status while a run is active.
bash run_experiments.sh status

# Evaluate existing adapters without training them.
bash run_experiments.sh run --evaluate-only --models lfm3-posttrained lfm3-base

# Rebuild all summary files and both chart languages from completed JSON reports.
bash run_experiments.sh report
```

To regenerate all eight figures from a shared summary JSON, without weights,
datasets, or the training stack, install `requirements-report.txt` and run:

```bash
python3 plot_experiments.py --input workspace/reports/leaderboard.json --out data/charts
```

Rerunning checks compatible manifests and reuses completed training/evaluations.
Training resumes from a complete optimizer checkpoint. An interrupted run before
its first checkpoint is restarted, with its partial directory retained for
diagnosis. Evaluation fingerprints include data, revision, adapter weights,
settings, implementation hashes, and runtime package versions; incompatible
results are remeasured. Interrupted evaluations retain diagnostic partial outputs
but are not presented as completed results.

Failures are recorded per model and do not abort remaining experiments. A failed,
blocked, or pending variant appears as missing, never as an invented zero score.
Actual model outputs that fail the JSON contract do receive the established
invalid-response penalty. Runtime/OOM failures stop that evaluation instead of
converting every headline into a classification failure.

## Interpretation

All variants use the same 825/107/101 splits in the existing local dataset,
three-epoch LoRA recipe, and task instructions. Counts are read from each dataset's
manifest rather than hardcoded. Each checkpoint uses its explicit/native prompt
wrapper; both variants within a pair must have the same template hash.

The suite measures teacher-label agreement, not independent human accuracy. The
test set has already been evaluated in earlier experiments. One training seed and
this small dataset are insufficient for a general model-quality claim. Native
tasks differ: Octen is a retriever and Limite is a mathematical reasoner.

Limite's experimental compatibility path exposes the effective LoRA weight to
its direct `F.linear(module.weight)` calls. Forward, adapter gradients and PEFT
merging are smoke-tested. Those direct calls bypass PEFT's feature dropout, so
the configured 0.05 LoRA dropout is not applied on these paths. Treat Limite as
an architecture-specific transfer experiment rather than an identical dropout
implementation. Spark's pinned remote code also needs tied-weight metadata and
causal-mask argument compatibility with the pinned Transformers version.

All new suite training uses a common shifted causal cross-entropy, normalized by
the assistant target-token count over the accumulated batch (`--uniform-loss`).
This avoids architecture-specific handling of `num_items_in_batch`. Its values
and gradients are checked against HF's standard causal loss. The two legacy LFM
adapters use that standard HF objective at their pinned revisions; their manifests
are explicitly recognized. Native-loss pilot runs are archived and excluded from
the final comparison.

Quality uses exact match, topic micro-F1, type accuracy, three score MAEs, and
invalid-output counts. Invalid responses stay in all denominators and add a
maximum score error of 3. Charts show both variants for every model and preserve
missing entries. JSON also includes original-to-distilled deltas.

Timing uses synchronized GPU batch generation after warmup and excludes model
loading, adapter merging, and training. Milliseconds per headline are amortized
over a batch; they are not serial API latency or phone speed. Runtime versions
must match before results can enter the same table. Optional FLA kernels can
change timings, so installing them invalidates previous cached evaluations.

For existing serial LM Studio Q4 comparisons, continue using `benchmark_lms.sh`.
For deployment artifacts, use `train.py merge` and `export_gguf.py`; a checkpoint's
HF training support does not automatically imply llama.cpp/GGUF support.
