<div align="center">

<h1>Gündem Distillation Lab</h1>

<p>Haber başlıkları için küçük modeller. Eğitimden kıyasa tek akış.</p>

<p>
  <a href="#kurulum"><img src="https://img.shields.io/badge/Python-Training%20tools-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python: Training tools"></a>
  <a href="requirements-train.txt"><img src="https://img.shields.io/badge/PyTorch-LoRA%20SFT-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch: LoRA SFT"></a>
  <a href="#kurulum"><img src="https://img.shields.io/badge/NVIDIA-CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="NVIDIA: CUDA"></a>
  <a href="docs/multimodel-results/README.tr.md"><img src="https://img.shields.io/badge/Reports-TR%20%2B%20EN-0D9488?style=for-the-badge" alt="Reports: TR + EN"></a>
</p>

<p>
  <a href="README.md">English</a> &nbsp;·&nbsp; <a href="#kurulum">Kurulum</a> &nbsp;·&nbsp; <a href="docs/multimodel-results/README.tr.md">Model sonuçları</a> &nbsp;·&nbsp; <a href="docs/SCORING.tr.md">Gündem Score</a>
</p>

</div>

Akış, DeepSeek veya yerel Qwen ile başlıkları etiketler; bu cevaplarla LoRA adapter eğitir. Model ağırlıkları ve tam veri kümesi yerelde indirilir veya oluşturulur.

> **Başlıklar → Öğretmen etiketleri → LoRA eğitimi → Kıyas ve grafikler**

[Eğitilmiş modelleri indir](docs/RELEASES.tr.md): Hugging Face adapter’ları ve GitHub GGUF/ZIP dosyaları.

[20 mobil indirme ve puanları](docs/MOBILE.tr.md): her ailenin orijinal ve fine-tuned sürümü, [Gündem Android](https://github.com/efebaykaraa/gundem-android) için hazır.

## Kurulum

Eğitim için Linux/Bash, venv destekli Python ve NVIDIA CUDA GPU kullanın. Makinenize uygun CUDA destekli PyTorch kurun. Komutları bu klasörde çalıştırın:

```bash
python3 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install -r requirements-train.txt -r requirements-report.txt -r requirements-kernels.txt
export DISTILL_PYTHON="$PWD/.venv-train/bin/python"
```

Llama gibi erişim onaylı modeller için Hugging Face’te erişim alıp `hf auth login` çalıştırın.

## Eğitim verisini hazırlama

DeepSeek API anahtarınızı `~/.deepseek_api_key` dosyasına tek satır olarak yazın:

```bash
touch ~/.deepseek_api_key
chmod 600 ~/.deepseek_api_key
nano ~/.deepseek_api_key
```

Başlıkları toplayıp etiketleyin, ardından veri ayrımını hazırlayın:

```bash
bash start_distillation.sh --provider deepseek \
  --work-dir data-deepseek --label-only --progress bar

python3 pipeline.py prepare \
  --input data-deepseek/labels.jsonl --out data-deepseek/prepared
```

DeepSeek API kullanımı hesabınızdan ücretlendirilir. Eğitimden önce etiketleri inceleyin. Yerel Qwen öğretmeni için `teacher.json` içindeki adresi ve model kimliğini ayarlayıp `--provider deepseek` seçeneğini kaldırın.

## Modelleri eğitme ve kıyaslama

Model ve eğitim ayarlarını [configs/models.json](configs/models.json) içinde seçin. Akış LFM3, GPT-X3, Octen, Spark, Llama, Limite ve Qwen checkpoint’lerini destekler.

```bash
bash run_experiments.sh init --source data-deepseek/prepared
bash run_experiments.sh preflight
bash run_experiments.sh run
```

Belirli modelleri çalıştırmak, ilerlemeyi görmek veya raporları yenilemek için:

```bash
bash run_experiments.sh run --models qwen3.5-0.8b-base llama-3.2-1b
bash run_experiments.sh status
bash run_experiments.sh report
```

Aynı komut uyumlu işleri kaldığı yerden sürdürür. Deney ayarlarını değiştirirken ayrı çalışma alanı kullanın; [çoklu model kılavuzuna](docs/MULTI-MODEL.tr.md) bakın.

## Çıktılar

| Yol | İçerik |
| --- | --- |
| `workspace/experiments/<model>/adapter/` | Eğitilmiş LoRA adapter |
| `workspace/experiments/<model>/original.json` / `distilled.json` | Model ölçüm sonuçları |
| `workspace/reports/leaderboard.csv` / `.json` / `.md` | Sıralama ve bileşen metrikleri |
| `workspace/reports/overall-{tr,en}.png` / `.svg` | Gündem Score sıralaması |
| `workspace/reports/comparison-{tr,en}.png` / `.svg` | Ayrıntılı metrik grafikleri |

**Gündem Score**, bu başlık sınıflandırma görevine özel 0–100 kalite puanıdır; yüksek daha iyi. Hız ayrıca gösterilir. [Puanın tanımı](docs/SCORING.tr.md) · [Yayımlanan sonuçlar](docs/multimodel-results/README.tr.md).

Model yüklemeden paylaşılan rapordan grafik üretmek için:

```bash
python3 plot_experiments.py \
  --input docs/multimodel-results/leaderboard.json --out data/charts
```

## Tek LFM modeli eğitme

Ortamı ve öğretmeni ayarladıktan sonra:

```bash
bash start_distillation.sh --provider deepseek \
  --work-dir data-deepseek --training-python "$DISTILL_PYTHON" --progress bar
```

Adapter `data-deepseek/adapter/`, birleştirilmiş HF modeli `data-deepseek/merged/` altına yazılır.

Uygulamada kullanmak için uyumlu GGUF dışa aktarıp Gündem’e yükleyin. LM Studio ile kıyaslamak için LM Studio ve `lms` CLI kurun; tam Gündem projesinde eğitim ve birleştirme sonrasında `bash benchmark_lms.sh` çalıştırın.

[Elle eğitim, GGUF dönüşümü ve benchmark komutları](docs/USAGE.tr.md) · [Model yapılandırması](configs/README.md).

[Android uygulaması](https://github.com/efebaykaraa/gundem-android)
