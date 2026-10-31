# Gündem Distillation Lab

Train small models to classify Turkish news headlines, compare original and distilled versions, and generate English/Turkish charts.

[Türkçe](README.tr.md) · [Model results](docs/multimodel-results/README.md)

The workflow uses DeepSeek or a local Qwen model to label headlines, then trains a LoRA adapter on those answers. Model weights and full datasets are downloaded or generated locally.

## Setup

Use Linux/Bash, Python with venv support, and an NVIDIA CUDA GPU for training. Install a CUDA-enabled PyTorch build suitable for your machine. Run these commands from this directory:

```bash
python3 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install -r requirements-train.txt -r requirements-report.txt -r requirements-kernels.txt
export DISTILL_PYTHON="$PWD/.venv-train/bin/python"
```

For gated models such as Llama, obtain access on Hugging Face and run `hf auth login`.

## Prepare training data

Put your DeepSeek API key on a single line in `~/.deepseek_api_key`:

```bash
touch ~/.deepseek_api_key
chmod 600 ~/.deepseek_api_key
nano ~/.deepseek_api_key
```

Collect and label headlines, then prepare the splits:

```bash
bash start_distillation.sh --provider deepseek \
  --work-dir data-deepseek --label-only --progress bar

python3 pipeline.py prepare \
  --input data-deepseek/labels.jsonl --out data-deepseek/prepared
```

DeepSeek API usage is billed to your account. Review the labels before training. To use a local Qwen teacher instead, configure its URL and model in `teacher.json` and omit `--provider deepseek`.

## Train and compare models

Choose models and training settings in [configs/models.json](configs/models.json). The suite supports LFM3, GPT-X3, Octen, Spark, Llama, Limite and Qwen checkpoints.

```bash
bash run_experiments.sh init --source data-deepseek/prepared
bash run_experiments.sh preflight
bash run_experiments.sh run
```

Run a subset, inspect progress, or refresh reports:

```bash
bash run_experiments.sh run --models qwen3.5-0.8b-base llama-3.2-1b
bash run_experiments.sh status
bash run_experiments.sh report
```

Repeat the run command to resume compatible work. Use a separate workspace when changing the experiment settings; see the [multi-model guide](docs/MULTI-MODEL.md).

## Outputs

| Path | Contents |
| --- | --- |
| `workspace/experiments/<model>/adapter/` | Trained LoRA adapter |
| `workspace/experiments/<model>/original.json` / `distilled.json` | Model evaluation results |
| `workspace/reports/leaderboard.csv` / `.json` / `.md` | Ranked results and component metrics |
| `workspace/reports/overall-{tr,en}.png` / `.svg` | Gündem Score ranking |
| `workspace/reports/comparison-{tr,en}.png` / `.svg` | Detailed metric charts |

**Gündem Score** is a 0–100 quality score for this headline task; higher is better. Speed is reported separately. [Score definition](docs/SCORING.md) · [Published results](docs/multimodel-results/README.md).

To render charts from a shared report without loading models:

```bash
python3 plot_experiments.py \
  --input docs/multimodel-results/leaderboard.json --out data/charts
```

## Train one LFM model

With the environment and teacher configured:

```bash
bash start_distillation.sh --provider deepseek \
  --work-dir data-deepseek --training-python "$DISTILL_PYTHON" --progress bar
```

The adapter is written to `data-deepseek/adapter/` and the merged HF model to `data-deepseek/merged/`.

For app use, export a compatible GGUF and import it into Gündem. For LM Studio comparison, install LM Studio and its `lms` CLI, then run `bash benchmark_lms.sh` after training and merging in the full Gündem checkout.

[Manual training, GGUF export and benchmark commands](docs/USAGE.md) · [Model configuration](configs/README.md).
