# Mobil modeller

Gündem Android model ekranında 10 ailenin hem orijinal hem fine-tuned sürümünü seçebilirsiniz. Her kartta Gündem Score gösterilir. Aşağıdaki dosyalar tam modeldir; ayrıca adapter uygulamak gerekmez.

[Android APK](https://github.com/efebaykaraa/gundem-android/releases/tag/v0.9.1) · [JSON catalog](mobile-catalog.json) · [Export instructions](MOBILE-EXPORT.md)

Puanlar 101 test başlığında HF/BF16 ile ölçüldü; mobil dosyalar yeniden puanlanmadı. 0,0 ölçülmüş sonuçtur: orijinal modeller bu kısıtlanmamış kıyasta geçerli JSON üretemedi. Uygulama JSON kısıtlaması kullanır. Puanlar telefon hızını veya genel zekâyı ölçmez.

| Model | Sürüm | Gündem Score / 100 | MB | İndir |
| --- | --- | ---: | ---: | --- |
| LFM3 1.7B Base | Fine-tuned | 78,1 | 1100 | [Q4_K_M GGUF](https://huggingface.co/efebaykara/LFM3-1.7B-Base-Gundem-LoRA/resolve/0d3b796d033f01e4ecf33a7cc0545889647c0443/gguf/lfm3-base-distilled-Q4_K_M.gguf) |
| LFM3 1.7B Base | Orijinal | 0,0 | 1100 | [Q4_K_M GGUF](https://huggingface.co/efebaykara/LFM3-1.7B-Base-Gundem-LoRA/resolve/0d3b796d033f01e4ecf33a7cc0545889647c0443/gguf/lfm3-base-original-Q4_K_M.gguf) |
| Llama 3.2 3B | Fine-tuned | 82,1 | 2019 | [Q4_K_M GGUF](https://huggingface.co/efebaykara/Llama-3.2-3B-Gundem-LoRA/resolve/2e19f1f605ed0e97a1628e6dde8291ce6ba9caaf/gguf/llama-3.2-3b-distilled-Q4_K_M.gguf) |
| Llama 3.2 3B | Orijinal | 0,0 | 2019 | [Q4_K_M GGUF](https://huggingface.co/efebaykara/Llama-3.2-3B-Gundem-LoRA/resolve/2e19f1f605ed0e97a1628e6dde8291ce6ba9caaf/gguf/llama-3.2-3b-original-Q4_K_M.gguf) |
| Qwen3.5 0.8B Base | Fine-tuned | 81,9 | 529 | [Q4_K_M GGUF](https://huggingface.co/efebaykara/Qwen3.5-0.8B-Base-Gundem-LoRA/resolve/6d4070587a7eb25270a862a7fb9e2f340a9f71b9/gguf/qwen3.5-0.8b-base-distilled-Q4_K_M.gguf) |
| Qwen3.5 0.8B Base | Orijinal | 0,0 | 529 | [Q4_K_M GGUF](https://huggingface.co/efebaykara/Qwen3.5-0.8B-Base-Gundem-LoRA/resolve/6d4070587a7eb25270a862a7fb9e2f340a9f71b9/gguf/qwen3.5-0.8b-base-original-Q4_K_M.gguf) |
| Qwen3.5 2B Base | Fine-tuned | 81,8 | 1274 | [Q4_K_M GGUF](https://huggingface.co/efebaykara/Qwen3.5-2B-Base-Gundem-LoRA/resolve/807245009d83dbd8c5c8d8516998c574ac82ab36/gguf/qwen3.5-2b-base-distilled-Q4_K_M.gguf) |
| Qwen3.5 2B Base | Orijinal | 0,0 | 1274 | [Q4_K_M GGUF](https://huggingface.co/efebaykara/Qwen3.5-2B-Base-Gundem-LoRA/resolve/807245009d83dbd8c5c8d8516998c574ac82ab36/gguf/qwen3.5-2b-base-original-Q4_K_M.gguf) |
| Spark X2.6 1.7B | Fine-tuned | 81,7 | 1107 | [Q4_K_M GGUF](https://huggingface.co/efebaykara/Spark-X2.6-1.7B-Gundem-LoRA/resolve/cd764da1dfbe09b8845cd0ea6b01d3159aeb2c07/gguf/spark-x2.6-1.7b-distilled-Q4_K_M.gguf) |
| Spark X2.6 1.7B | Orijinal | 0,0 | 1107 | [Q4_K_M GGUF](https://huggingface.co/efebaykara/Spark-X2.6-1.7B-Gundem-LoRA/resolve/cd764da1dfbe09b8845cd0ea6b01d3159aeb2c07/gguf/spark-x2.6-1.7b-original-Q4_K_M.gguf) |
| Llama 3.2 1B | Fine-tuned | 81,0 | 808 | [Q4_K_M GGUF](https://huggingface.co/efebaykara/Llama-3.2-1B-Gundem-LoRA/resolve/fa39fbb6f0c46307229395c29443355484e8cf62/gguf/llama-3.2-1b-distilled-Q4_K_M.gguf) |
| Llama 3.2 1B | Orijinal | 0,0 | 808 | [Q4_K_M GGUF](https://huggingface.co/efebaykara/Llama-3.2-1B-Gundem-LoRA/resolve/fa39fbb6f0c46307229395c29443355484e8cf62/gguf/llama-3.2-1b-original-Q4_K_M.gguf) |
| LFM3 1.7B | Fine-tuned | 77,9 | 1100 | [Q4_K_M GGUF](https://huggingface.co/efebaykara/LFM3-1.7B-Gundem-LoRA/resolve/d991ac33cd6533fd9c4fe9e634b1fd0a8b450fcb/gguf/lfm3-posttrained-distilled-Q4_K_M.gguf) |
| LFM3 1.7B | Orijinal | 0,0 | 1100 | [Q4_K_M GGUF](https://huggingface.co/efebaykara/LFM3-1.7B-Gundem-LoRA/resolve/d991ac33cd6533fd9c4fe9e634b1fd0a8b450fcb/gguf/lfm3-posttrained-original-Q4_K_M.gguf) |
| Octen WebGR 0.6B | Fine-tuned | 70,6 | 510 | [Q4_K_M GGUF](https://huggingface.co/efebaykara/Octen-WebGR-0.6B-Gundem-LoRA/resolve/fda69b5c8709d487d1ac7bbe98465f9b5d3f98f5/gguf/octen-webgr-0.6b-distilled-Q4_K_M.gguf) |
| Octen WebGR 0.6B | Orijinal | 0,0 | 510 | [Q4_K_M GGUF](https://huggingface.co/efebaykara/Octen-WebGR-0.6B-Gundem-LoRA/resolve/fda69b5c8709d487d1ac7bbe98465f9b5d3f98f5/gguf/octen-webgr-0.6b-original-Q4_K_M.gguf) |
| Limite 1B Violetto | Fine-tuned | 70,0 | 670 | [Q4_K_M GGUF](https://huggingface.co/efebaykara/limite-1b-violetto-Gundem-LoRA/resolve/1924b24ad130baa62dae18e507c07025903929ad/gguf/limite-1b-violetto-distilled-Q4_K_M.gguf) |
| Limite 1B Violetto | Orijinal | 0,0 | 670 | [Q4_K_M GGUF](https://huggingface.co/efebaykara/limite-1b-violetto-Gundem-LoRA/resolve/1924b24ad130baa62dae18e507c07025903929ad/gguf/limite-1b-violetto-original-Q4_K_M.gguf) |
| GPT-X3 150M | Fine-tuned | 69,6 | 247 | [ONNX INT8 ZIP](https://huggingface.co/efebaykara/GPT-X3-150M-Gundem-LoRA/resolve/d8672439175b78274f4d2016f2d90d45d7d189a3/onnx/gpt-x3-150m-distilled-ONNX-INT8.zip) |
| GPT-X3 150M | Orijinal | 0,0 | 247 | [ONNX INT8 ZIP](https://huggingface.co/efebaykara/GPT-X3-150M-Gundem-LoRA/resolve/d8672439175b78274f4d2016f2d90d45d7d189a3/onnx/gpt-x3-150m-original-ONNX-INT8.zip) |

İndirme bağlantıları sabit commit’e bağlıdır; boyut ve SHA-256 JSON kataloğunda bulunur. Modellerin RAM ihtiyacı cihaz ve girdiye göre değişir. Her model kendi lisansını korur; Octen ticari olmayan kullanımla sınırlıdır.
