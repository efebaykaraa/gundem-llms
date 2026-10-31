# Model downloads

[Türkçe](RELEASES.tr.md)

**models-v1.0.0** · [GitHub Release](https://github.com/efebaykaraa/Gundem/releases/tag/models-v1.0.0)

Adapter ZIPs require the base model. LFM GGUF files are standalone and require a compatible llama.cpp/LM Studio runtime.

| Model | Hugging Face | Adapter ZIP |
| --- | --- | --- |
| llama-3.2-3b | [efebaykara/Llama-3.2-3B-Gundem-LoRA](https://huggingface.co/efebaykara/Llama-3.2-3B-Gundem-LoRA) | [ZIP](https://github.com/efebaykaraa/Gundem/releases/download/models-v1.0.0/Llama-3.2-3B-Gundem-LoRA.zip) |
| qwen3.5-0.8b-base | [efebaykara/Qwen3.5-0.8B-Base-Gundem-LoRA](https://huggingface.co/efebaykara/Qwen3.5-0.8B-Base-Gundem-LoRA) | [ZIP](https://github.com/efebaykaraa/Gundem/releases/download/models-v1.0.0/Qwen3.5-0.8B-Base-Gundem-LoRA.zip) |
| qwen3.5-2b-base | [efebaykara/Qwen3.5-2B-Base-Gundem-LoRA](https://huggingface.co/efebaykara/Qwen3.5-2B-Base-Gundem-LoRA) | [ZIP](https://github.com/efebaykaraa/Gundem/releases/download/models-v1.0.0/Qwen3.5-2B-Base-Gundem-LoRA.zip) |
| spark-x2.6-1.7b | [efebaykara/Spark-X2.6-1.7B-Gundem-LoRA](https://huggingface.co/efebaykara/Spark-X2.6-1.7B-Gundem-LoRA) | [ZIP](https://github.com/efebaykaraa/Gundem/releases/download/models-v1.0.0/Spark-X2.6-1.7B-Gundem-LoRA.zip) |
| llama-3.2-1b | [efebaykara/Llama-3.2-1B-Gundem-LoRA](https://huggingface.co/efebaykara/Llama-3.2-1B-Gundem-LoRA) | [ZIP](https://github.com/efebaykaraa/Gundem/releases/download/models-v1.0.0/Llama-3.2-1B-Gundem-LoRA.zip) |
| lfm3-base | [efebaykara/LFM3-1.7B-Base-Gundem-LoRA](https://huggingface.co/efebaykara/LFM3-1.7B-Base-Gundem-LoRA) | [ZIP](https://github.com/efebaykaraa/Gundem/releases/download/models-v1.0.0/LFM3-1.7B-Base-Gundem-LoRA.zip) |
| lfm3-posttrained | [efebaykara/LFM3-1.7B-Gundem-LoRA](https://huggingface.co/efebaykara/LFM3-1.7B-Gundem-LoRA) | [ZIP](https://github.com/efebaykaraa/Gundem/releases/download/models-v1.0.0/LFM3-1.7B-Gundem-LoRA.zip) |
| octen-webgr-0.6b | [efebaykara/Octen-WebGR-0.6B-Gundem-LoRA](https://huggingface.co/efebaykara/Octen-WebGR-0.6B-Gundem-LoRA) | [ZIP](https://github.com/efebaykaraa/Gundem/releases/download/models-v1.0.0/Octen-WebGR-0.6B-Gundem-LoRA.zip) |
| limite-1b-violetto | [efebaykara/limite-1b-violetto-Gundem-LoRA](https://huggingface.co/efebaykara/limite-1b-violetto-Gundem-LoRA) | [ZIP](https://github.com/efebaykaraa/Gundem/releases/download/models-v1.0.0/limite-1b-violetto-Gundem-LoRA.zip) |
| gpt-x3-150m | [efebaykara/GPT-X3-150M-Gundem-LoRA](https://huggingface.co/efebaykara/GPT-X3-150M-Gundem-LoRA) | [ZIP](https://github.com/efebaykaraa/Gundem/releases/download/models-v1.0.0/GPT-X3-150M-Gundem-LoRA.zip) |

## GGUF

- [LFM3-1.7B-Base-Gundem-Q4_K_M.gguf](https://github.com/efebaykaraa/Gundem/releases/download/models-v1.0.0/LFM3-1.7B-Base-Gundem-Q4_K_M.gguf)
- [LFM3-1.7B-Gundem-Q4_K_M.gguf](https://github.com/efebaykaraa/Gundem/releases/download/models-v1.0.0/LFM3-1.7B-Gundem-Q4_K_M.gguf)

In Gündem, import through **Model → GGUF dosyası seç**. HF/BF16 benchmark scores do not directly describe GGUF performance.

## Use an adapter

Extract the ZIP and install CUDA-enabled PyTorch and the packaged requirements. From the extracted folder:

```bash
python -m pip install -r requirements.txt
python release_infer.py --model . --headline "İşçiler ücret artışı için greve başladı"
```

The script downloads the pinned base model, merges the adapter and uses CUDA BF16. Llama requires Hugging Face model access and `hf auth login`.

Each package includes a model card, license, evaluation and SHA-256 metadata. Octen is non-commercial; Llama and LFM retain their own terms. Check each model card’s license.

[SHA256SUMS.txt](https://github.com/efebaykaraa/Gundem/releases/download/models-v1.0.0/SHA256SUMS.txt) · [Benchmark results](multimodel-results/README.md)
