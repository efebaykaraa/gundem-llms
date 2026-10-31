# Training, export and benchmark commands

[Back to README](../README.md) · [Türkçe](USAGE.tr.md)

Run all commands from `tools/distill`, or the standalone repository root.

## Run each stage manually

```bash
# Collect headlines; repeat on later days to accumulate new items.
python3 pipeline.py collect --count 1000 --out data-deepseek/headlines.jsonl

# Label with the configured teacher.
python3 pipeline.py label --provider deepseek \
  --input data-deepseek/headlines.jsonl --out data-deepseek/labels.jsonl

# Create train, validation, and test splits.
python3 pipeline.py prepare \
  --input data-deepseek/labels.jsonl --out data-deepseek/prepared

# Create a separate training environment.
python3 -m venv .venv-train
.venv-train/bin/python -m pip install -r requirements-train.txt

# Train, then merge the adapter into the base model.
.venv-train/bin/python train.py train \
  --data data-deepseek/prepared --out data-deepseek/adapter \
  --student LiquidAI/LFM3-1.7B --epochs 3
.venv-train/bin/python train.py merge \
  --adapter data-deepseek/adapter --out data-deepseek/merged --cpu
```

The student receives the short prompt in `student-system.txt` and the headline only. Source, date, URL, and article body are not sent to the teacher/student as headline content. Review teacher labels before training; `prepare --reviewed-only` selects records marked `reviewed: true`. Related headlines can be assigned the same `event_group` to keep that group within one split. Similar events with different titles are not automatically detected.

## Benchmark with LM Studio

### Full Gündem checkout

After training has produced `data-deepseek/merged` and `data-deepseek/prepared`:

```bash
bash benchmark_lms.sh
```

The script builds the quantizer from `../../vendor/llama.cpp`, installs converter dependencies in a separate `.venv-convert-lms` environment, and converts both models to Q4_K_M. It locates the original HF model in the local cache using the revision recorded in the training manifest. It then imports the GGUFs into LM Studio using symbolic links, starts or reuses the local API server, and loads, measures, and unloads each benchmark model sequentially. Other loaded models are left in place; unload them yourself for a controlled timing comparison.

Defaults: GPU `max`, context length `2048`, server port `1234`, and 160 output tokens. An existing server's port is reused. If API authentication is enabled, set `STUDENT_API_KEY`.

```bash
bash benchmark_lms.sh --prepare-only
bash benchmark_lms.sh --run-only --gpu max --context-length 2048
```

Existing GGUFs are reused. Choose new output paths if the HF weights change.

To compare two already fine-tuned models starting from different checkpoints, use
`--comparison starting-checkpoint`, with the post-trained+LoRA model as `--old-gguf`
and the Base+LoRA model as `--new-gguf`. This uses separate LM Studio identifiers
and labels the charts accordingly. Both runs should use the same prepared splits
and training settings; this flag changes naming, not experiment validation.

### Standalone repository

The standalone repository does **not** include Gündem's vendored llama.cpp sources. Use already-exported, matching Q4_K_M models with `--run-only`:

```bash
bash benchmark_lms.sh --run-only \
  --old-gguf /absolute/path/original-Q4_K_M.gguf \
  --new-gguf /absolute/path/distilled-Q4_K_M.gguf \
  --data data-deepseek/prepared \
  --new-manifest data-deepseek/merged/distill-manifest.json
```

To export weights yourself, use an LFM3-compatible llama.cpp checkout and its converter dependencies in a separate environment. LFM3 tokenizers used here require Transformers 5; the automatic preparation script installs `transformers==5.20.2` after the converter requirements. Once the converter environment and quantizer are ready:

```bash
/path/to/converter-env/bin/python export_gguf.py \
  --model data-deepseek/merged \
  --llama-dir /path/to/llama.cpp \
  --quantizer /path/to/llama.cpp/build/bin/llama-quantize \
  --out data-deepseek/gguf/distilled-Q4_K_M.gguf
```

Export the original HF model at the manifest's revision in the same quantization format. `export_gguf.py` creates an intermediate F16 file and a SHA-256 sidecar, and refuses to overwrite an existing output. Allow space for both temporary and final files.

### Reports

| Output | Contents |
| --- | --- |
| `data-deepseek/benchmark-lms.json` | Aggregate scores, latency, and new-minus-old differences |
| `data-deepseek/benchmark-lms.json.predictions.jsonl` | Reference and both predictions for each test headline |
| `data-deepseek/benchmark-lms-comparison*.png` / `.svg` | English and Turkish charts |

The benchmark checks dataset manifests, split hashes, and headline/event-group leakage before inference. Reference answers are not sent as model inputs. In the delta report, positive accuracy/F1 changes and negative MAE/latency/invalid-count changes indicate improvement.

## Generate charts without running models

Use a Python environment with Matplotlib:

```bash
python3 -m pip install matplotlib
python3 plot_benchmark.py
python3 plot_benchmark.py --lang en
python3 plot_benchmark.py --lang tr
```

By default, both languages are generated. English filenames end in `-en`; each language produces PNG and SVG. A different report or output prefix can be supplied:

```bash
python3 plot_benchmark.py \
  --input docs/benchmark-example.json --out data/charts/comparison
```

`benchmark_lms.sh` also generates both languages automatically. A plotting failure does not discard the benchmark JSON.

## Other evaluation methods

For two already-running OpenAI-compatible endpoints, update the example config files and run:

```bash
python3 pipeline.py benchmark --data data-deepseek/prepared \
  --new-manifest data-deepseek/merged/distill-manifest.json \
  --old-config benchmark-old.example.json \
  --new-config benchmark-new.example.json \
  --out data-deepseek/benchmark.json
```

The example configs use ports 8001 and 8002. `Connection refused` means no server is listening at the configured address; this command does not start servers.

For sequential local HF inference on CUDA, without API servers:

```bash
data-deepseek/venv-train/bin/python benchmark_local.py
```

The local HF benchmark uses greedy generation without constrained JSON decoding. Its results are a different experiment from the schema-constrained LM Studio benchmark.

