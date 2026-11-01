# Reproducing the mobile catalog

For downloads and scores, see [Mobile models](MOBILE.md) / [Mobil modeller](MOBILE.tr.md).

These publishing scripts run from `tools/distill` inside a recursive [Android checkout](https://github.com/efebaykaraa/gundem-android). They use completed experiments in `workspace/experiments`, the frozen dataset in `workspace/datasets/news-v1`, `workspace/reports/leaderboard.json`, and the existing release index in `workspace/releases/models-v1.0.0`. Train and evaluate first using the main README. Source repositories do not contain model weights or the private dataset.

## GGUF pairs

Build llama.cpp's `llama-quantize` target into `vendor/llama.cpp/build-distill` and prepare `.venv-convert-lms` with the GGUF conversion dependencies. In the training environment:

```bash
python mobile_export.py
```

This creates both variants of the nine supported GGUF families. Adapter merges use FP32; weights are then quantized to Q4_K_M. Existing artifacts are reused only after checking the checkpoint revision, adapter digest and output digest. GPT-X3 is exported separately below.

Prepare the native smoke executable with the Android repository's `docs/run-native-smoke.sh`, then publish using a Hugging Face account with write access:

```bash
python mobile_publish.py
```

The publisher runs both files through the actual native engine, uploads them to the existing model repositories, and verifies remote sizes and SHA-256 before writing receipts under `workspace/mobile/published`. Starting inference verifies runtime support; an original model may still exhaust its output limit or return poor classifications. This check is not a quality benchmark.

## GPT-X3 ONNX pair

Install `requirements-onnx.txt` into the training environment. Build `gundem-tokenizer-smoke` using `docs/native-smoke.CMakeLists.txt` in the Android repository, then:

```bash
python export_gpt_mobile.py
```

The exporter checks native BPE token IDs against HF on 106 cases and compares FP32 ONNX logits against the pinned PyTorch checkpoint before dynamic INT8 MatMul quantization. The standalone ZIP includes inference weights, a vocabulary-only GGUF, the descriptor and the model license. The ONNX wrapper is adapted under Apache-2.0 from [ahmadw/GPT-X3-150M-ONNX](https://huggingface.co/ahmadw/GPT-X3-150M-ONNX/tree/695dd12882449179a75845635adfd4b526e89b64).

Run `GptOnnxRuntimeTest` from the Android root against **each** ZIP using the JNI instructions in `docs/host-jni.md`. These tests exercise production Kotlin/JNI/ONNX inference, JSON constraints, cancellation, reset and model installation. Then, from this directory:

```bash
python publish_gpt_mobile.py
```

This uploads both bundles and verifies their remote hashes. A successful two-test integration report is required. Publishing uses the currently authenticated `efebaykara` account and the existing Gündem repositories; fork maintainers should change the namespace and tag before publishing.

## App and documentation

After every pair has a verified receipt:

```bash
python sync_mobile_catalog.py
```

The command refuses an incomplete 20-variant catalog. It updates Android's catalog and default model, the recommended-model metadata, and the English/Turkish download tables. Run Android unit tests, lint and the APK build before releasing. The regression test requires all 20 downloads, 18 GGUF files and two ONNX bundles.

The displayed scores remain the original HF/BF16 benchmark measurements. Export validation and host inference tests do not establish phone performance or reevaluate quantized-model quality.
