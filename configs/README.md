# Model configuration

Edit [models.json](models.json) to choose models and the shared training/evaluation settings.

Each model has a unique `id`, Hugging Face `repo`, `revision` and `prompt_style`.
Use the IDs with `bash run_experiments.sh run --models <id>`. Gated repositories
require model access and a local `hf auth login`.

| Prompt style | Use |
| --- | --- |
| `native` | Require the checkpoint's supplied chat template |
| `auto` | Use its template, falling back to plain text |
| `plain` | Explicit System / User / Assistant text |
| `task-chatml` | Explicit task ChatML; used for Limite |

Keep prepared data unchanged during a run. Use a separate config/workspace when
changing data or training settings. Adding model entries to the current suite is
supported; compatible completed work is reused.

[Run and resume experiments](../docs/MULTI-MODEL.md) ·
[Architecture compatibility notes](../docs/MULTI-MODEL.md#interpretation) ·
[Gündem Score](../docs/SCORING.md)
