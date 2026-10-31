# Multi-model suite configuration

`models.json` defines the model registry and the common training/evaluation recipe.
The runner resolves every model revision to an immutable commit in
`workspace/experiments/<id>/model.lock.json`. A gated model requires local HF login
and approval on its model page; no token belongs in this file.

All models share the same prepared dataset and short task system prompt. Original
and distilled variants of a model use the same revision, tokenizer, and effective
chat template. Native wrappers differ between architectures, so cross-model
comparisons use identical task content rather than identical input token IDs.

Prompt styles:

- `native`: require the checkpoint's chat template.
- `auto`: preserve its template, or install the explicit plain-text fallback.
- `plain`: `System: ... / User: ... / Assistant: ...`, with the tokenizer's BOS.
- `task-chatml`: explicit ChatML with the common task prompt. Limite needs this
  because its supplied template rejects any noncanonical mathematical system prompt.

Qwen3.5 is loaded through its text-only `Qwen3_5ForCausalLM` path. Vision weights
are not adapted. Octen WebGR is a specialized URL-identifier retriever; this suite
tests transfer to headline JSON classification, not its original retrieval quality.
Unknown identifier tokens, malformed JSON, and token-limit cuts count as invalid.
Limite's experimental LoRA path accounts for direct `F.linear(module.weight)` calls;
the forward, backward, and standard PEFT merge are checked by a CPU smoke test.
Direct-weight calls bypass PEFT feature dropout, so Limite does not apply the
configured LoRA dropout on those paths; this is an experimental architecture
exception, documented in the multi-model guide.

Results use CUDA bfloat16, greedy decoding, no JSON grammar, and batch size 4.
Timing is amortized batch inference time per headline. Do not combine this table
with the earlier LM Studio Q4/grammar/serial-latency measurements.

Changing the training/evaluation recipe requires a separate `--workspace`.
Use a separate config with a different `dataset` path for another frozen dataset.
Never change the prepared split files in place. Adding models to an existing suite
is supported; existing model/revision locks and compatible adapters are reused.

Primary model cards:

- https://huggingface.co/AxiomicLabs/GPT-X3-150M
- https://huggingface.co/Octen/Octen-WebGR-0.6B
- https://huggingface.co/XHToken/Spark-X2.6-1.7B
- https://huggingface.co/meta-llama/Llama-3.2-1B
- https://huggingface.co/meta-llama/Llama-3.2-3B
- https://huggingface.co/paradigma-inc/limite-1b-violetto
- https://huggingface.co/Qwen/Qwen3.5-0.8B-Base
- https://huggingface.co/Qwen/Qwen3.5-2B-Base
