# Model indirmeleri

[English](RELEASES.md)

**models-v1.0.0** · [GitHub Release](https://github.com/efebaykaraa/gundem-llms/releases/tag/models-v1.0.0)

Adapter ZIP’leri temel modelle birlikte çalışır. LFM GGUF dosyaları tek başına kullanılabilir; uyumlu llama.cpp/LM Studio gerekir.

| Model | Hugging Face | Adapter ZIP |
| --- | --- | --- |
| llama-3.2-3b | [efebaykara/Llama-3.2-3B-Gundem-LoRA](https://huggingface.co/efebaykara/Llama-3.2-3B-Gundem-LoRA) | [ZIP](https://github.com/efebaykaraa/gundem-llms/releases/download/models-v1.0.0/Llama-3.2-3B-Gundem-LoRA.zip) |
| qwen3.5-0.8b-base | [efebaykara/Qwen3.5-0.8B-Base-Gundem-LoRA](https://huggingface.co/efebaykara/Qwen3.5-0.8B-Base-Gundem-LoRA) | [ZIP](https://github.com/efebaykaraa/gundem-llms/releases/download/models-v1.0.0/Qwen3.5-0.8B-Base-Gundem-LoRA.zip) |
| qwen3.5-2b-base | [efebaykara/Qwen3.5-2B-Base-Gundem-LoRA](https://huggingface.co/efebaykara/Qwen3.5-2B-Base-Gundem-LoRA) | [ZIP](https://github.com/efebaykaraa/gundem-llms/releases/download/models-v1.0.0/Qwen3.5-2B-Base-Gundem-LoRA.zip) |
| spark-x2.6-1.7b | [efebaykara/Spark-X2.6-1.7B-Gundem-LoRA](https://huggingface.co/efebaykara/Spark-X2.6-1.7B-Gundem-LoRA) | [ZIP](https://github.com/efebaykaraa/gundem-llms/releases/download/models-v1.0.0/Spark-X2.6-1.7B-Gundem-LoRA.zip) |
| llama-3.2-1b | [efebaykara/Llama-3.2-1B-Gundem-LoRA](https://huggingface.co/efebaykara/Llama-3.2-1B-Gundem-LoRA) | [ZIP](https://github.com/efebaykaraa/gundem-llms/releases/download/models-v1.0.0/Llama-3.2-1B-Gundem-LoRA.zip) |
| lfm3-base | [efebaykara/LFM3-1.7B-Base-Gundem-LoRA](https://huggingface.co/efebaykara/LFM3-1.7B-Base-Gundem-LoRA) | [ZIP](https://github.com/efebaykaraa/gundem-llms/releases/download/models-v1.0.0/LFM3-1.7B-Base-Gundem-LoRA.zip) |
| lfm3-posttrained | [efebaykara/LFM3-1.7B-Gundem-LoRA](https://huggingface.co/efebaykara/LFM3-1.7B-Gundem-LoRA) | [ZIP](https://github.com/efebaykaraa/gundem-llms/releases/download/models-v1.0.0/LFM3-1.7B-Gundem-LoRA.zip) |
| octen-webgr-0.6b | [efebaykara/Octen-WebGR-0.6B-Gundem-LoRA](https://huggingface.co/efebaykara/Octen-WebGR-0.6B-Gundem-LoRA) | [ZIP](https://github.com/efebaykaraa/gundem-llms/releases/download/models-v1.0.0/Octen-WebGR-0.6B-Gundem-LoRA.zip) |
| limite-1b-violetto | [efebaykara/limite-1b-violetto-Gundem-LoRA](https://huggingface.co/efebaykara/limite-1b-violetto-Gundem-LoRA) | [ZIP](https://github.com/efebaykaraa/gundem-llms/releases/download/models-v1.0.0/limite-1b-violetto-Gundem-LoRA.zip) |
| gpt-x3-150m | [efebaykara/GPT-X3-150M-Gundem-LoRA](https://huggingface.co/efebaykara/GPT-X3-150M-Gundem-LoRA) | [ZIP](https://github.com/efebaykaraa/gundem-llms/releases/download/models-v1.0.0/GPT-X3-150M-Gundem-LoRA.zip) |

## GGUF

- [LFM3-1.7B-Base-Gundem-Q4_K_M.gguf](https://github.com/efebaykaraa/gundem-llms/releases/download/models-v1.0.0/LFM3-1.7B-Base-Gundem-Q4_K_M.gguf)
- [LFM3-1.7B-Gundem-Q4_K_M.gguf](https://github.com/efebaykaraa/gundem-llms/releases/download/models-v1.0.0/LFM3-1.7B-Gundem-Q4_K_M.gguf)

Gündem’de **Model → GGUF dosyası seç** ile içe aktarın. HF/BF16 kıyas puanları doğrudan GGUF performansını göstermez.

GGUF dosyalarıyla birlikte lisans ve atıf metinlerini içeren [LICENSES.zip](https://github.com/efebaykaraa/gundem-llms/releases/download/models-v1.0.0/LICENSES.zip) paketini de indirin.

## Adapter kullanımı

ZIP’i açın; CUDA destekli PyTorch ve paket bağımlılıklarını kurun. Ardından açılan klasörde:

```bash
python -m pip install -r requirements.txt
python release_infer.py --model . --headline "İşçiler ücret artışı için greve başladı"
```

Script sabitlenmiş temel modeli indirir, adapter’ı birleştirir ve CUDA BF16 ile çalıştırır. Llama için Hugging Face erişimi ve `hf auth login` gerekir.

Her pakette model kartı, lisans, değerlendirme ve SHA-256 bilgisi bulunur. Octen ticari olmayan kullanım lisanslıdır; Llama ve LFM kendi koşullarına tabidir. Model kartındaki lisansı kontrol edin.

[SHA256SUMS.txt](https://github.com/efebaykaraa/gundem-llms/releases/download/models-v1.0.0/SHA256SUMS.txt) · [Kıyas sonuçları](multimodel-results/README.tr.md)
