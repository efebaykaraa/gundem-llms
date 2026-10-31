# Multi-model distillation comparison

Completed: **20 / 20** variants.

Gündem Score v1 (0–100): 30% topic F1 + 20% type accuracy + 30% normalized score accuracy + 20% exact match. Score accuracy = 1 − mean MAE / 3. Quality only; hardware speed is separate. Invalid outputs are already penalized. This is a task-specific composite, not an accuracy percentage.

| Model | Variant | Gündem Score / 100 | Exact match | Topic F1 | Type accuracy | Invalid |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Llama 3.2 3B | distilled | 82.1 | 43.6% | 91.9% | 89.1% | 0 |
| Qwen3.5 0.8B Base | distilled | 81.9 | 45.5% | 91.1% | 87.1% | 0 |
| Qwen3.5 2B Base | distilled | 81.8 | 45.5% | 91.4% | 83.2% | 0 |
| Spark X2.6 1.7B | distilled | 81.7 | 41.6% | 92.7% | 89.1% | 0 |
| Llama 3.2 1B | distilled | 81.0 | 43.6% | 91.2% | 85.1% | 1 |
| LFM3 1.7B Base | distilled | 78.1 | 34.7% | 89.4% | 85.1% | 1 |
| LFM3 1.7B | distilled | 77.9 | 36.6% | 87.4% | 84.2% | 0 |
| Octen WebGR 0.6B | distilled | 70.6 | 23.8% | 86.2% | 66.3% | 0 |
| Limite 1B Violetto | distilled | 70.0 | 25.7% | 83.9% | 66.3% | 0 |
| GPT-X3 150M | distilled | 69.6 | 24.8% | 83.3% | 66.3% | 0 |
| GPT-X3 150M | original | 0.0 | 0.0% | 0.0% | 0.0% | 101 |
| LFM3 1.7B Base | original | 0.0 | 0.0% | 0.0% | 0.0% | 101 |
| LFM3 1.7B | original | 0.0 | 0.0% | 0.0% | 0.0% | 101 |
| Limite 1B Violetto | original | 0.0 | 0.0% | 0.0% | 0.0% | 101 |
| Llama 3.2 1B | original | 0.0 | 0.0% | 0.0% | 0.0% | 101 |
| Llama 3.2 3B | original | 0.0 | 0.0% | 0.0% | 0.0% | 101 |
| Octen WebGR 0.6B | original | 0.0 | 0.0% | 0.0% | 0.0% | 101 |
| Qwen3.5 0.8B Base | original | 0.0 | 0.0% | 0.0% | 0.0% | 101 |
| Qwen3.5 2B Base | original | 0.0 | 0.0% | 0.0% | 0.0% | 101 |
| Spark X2.6 1.7B | original | 0.0 | 0.0% | 0.0% | 0.0% | 101 |

## Model status

| Model | Status | Detail |
| --- | --- | --- |
| GPT-X3 150M | complete |  |
| LFM3 1.7B | complete |  |
| LFM3 1.7B Base | complete |  |
| Octen WebGR 0.6B | complete |  |
| Spark X2.6 1.7B | complete |  |
| Llama 3.2 1B | complete |  |
| Llama 3.2 3B | complete |  |
| Limite 1B Violetto | complete |  |
| Qwen3.5 0.8B Base | complete |  |
| Qwen3.5 2B Base | complete |  |

## Interpretation

- Teacher-label agreement on a small, previously evaluated test set; not human accuracy.
- All results use HF/CUDA bfloat16 greedy generation without a JSON grammar.
- Timing is amortized batch inference time per headline, not serial latency or phone speed.
- Native wrappers differ across models. Within each pair, wrapper and checkpoint revision match.
- Failed or pending variants are not assigned zero scores.
