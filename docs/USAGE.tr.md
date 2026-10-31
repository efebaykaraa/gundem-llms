# Eğitim, dışa aktarma ve benchmark komutları

[README’ye dön](../README.tr.md) · [English](USAGE.md)

Komutları `tools/distill` içinde veya bağımsız reponun kökünde çalıştırın.

## Eğitim çıktıları

| Yol | İçerik |
| --- | --- |
| `data-deepseek/labels.jsonl` | Öğretmen etiketleri |
| `data-deepseek/prepared/` | Eğitim, doğrulama, test ve veri manifesti |
| `data-deepseek/adapter/` | LoRA adapter ve eğitim manifesti |
| `data-deepseek/merged/` | Temel modelle birleştirilmiş tam HF modeli |
| `data-deepseek/gguf/` | Dönüştürülmüş GGUF dosyaları |

Öğrenci `student-system.txt` içindeki kısa talimatı ve yalnız başlığı alır. Loss yalnız assistant JSON’u ve EOS üzerinde hesaplanır. Öğretmen etiketlerini eğitimden önce inceleyin; yalnız `reviewed: true` kayıtları seçmek için `prepare --reviewed-only` kullanılabilir. Aynı olayın farklı başlıklarına ortak `event_group` verilirse bu grup veri bölümleri arasında ayrılmaz; olay benzerliği otomatik bulunmaz.

Aşamaları elle çalıştırma ve GGUF dışa aktarma için [komut kılavuzuna](USAGE.md) bakın.

## LM Studio ile benchmark

Tam Gündem projesinde, eğitim ve birleştirme tamamlandıktan sonra:

```bash
bash benchmark_lms.sh
```

Script `../../vendor/llama.cpp` içindeki quantizer’ı derler, ayrı `.venv-convert-lms` ortamını kurar ve iki modeli aynı Q4_K_M biçimine dönüştürür. Orijinal modeli eğitim manifestindeki sabit sürümün yerel HF önbelleğinden bulur. Sonra modelleri sembolik bağlantıyla LM Studio’ya ekler, API’yi açar veya mevcut sunucuyu kullanır; modelleri sırayla yükleyip ölçer ve kaldırır.

Varsayılan ayarlar GPU `max`, bağlam `2048`, port `1234` ve 160 çıktı tokenıdır. Mevcut sunucunun portu kullanılır. Diğer yüklü modellere dokunulmaz; kontrollü süre ölçümü için onları ayrıca kaldırın. API kimlik doğrulaması açıksa `STUDENT_API_KEY` ayarlayın.

```bash
bash benchmark_lms.sh --prepare-only
bash benchmark_lms.sh --run-only --gpu max --context-length 2048
```

Hazır GGUF’lar yeniden kullanılır. HF modeli değişirse farklı çıktı yolları seçin.

Farklı başlangıç checkpoint’lerinden eğitilmiş iki modeli kıyaslamak için
`--comparison starting-checkpoint` ekleyin. `--old-gguf` post-trained+LoRA,
`--new-gguf` Base+LoRA modeli olmalıdır. LM Studio kimlikleri ayrılır ve grafikler
uygun adlarla üretilir. İki eğitimde aynı veri ayrımı ve eğitim ayarları
kullanılmalıdır; bu seçenek adlandırmayı değiştirir, deney doğrulamasını değiştirmez.

**Bağımsız repo:** Gündem’in vendored llama.cpp kaynakları burada bulunmaz. İki modeli LFM3 destekli bir converter ile aynı Q4_K_M biçimine dönüştürüp şu komutu kullanın:

```bash
bash benchmark_lms.sh --run-only \
  --old-gguf /tam/yol/original-Q4_K_M.gguf \
  --new-gguf /tam/yol/distilled-Q4_K_M.gguf \
  --data data-deepseek/prepared \
  --new-manifest data-deepseek/merged/distill-manifest.json
```

Veri ve eğitim manifesti, eğitilmiş modele ait olmalıdır. Benchmark bölüm hash’lerini, manifest uyumunu ve başlık/olay grubu sızıntısını denetler. Referans cevaplar modele verilmez. GGUF dosyalarının doğru modelleri içerdiği ayrıca doğrulanmalıdır.

Sonuçlar `data-deepseek/benchmark-lms.json`, başlık bazlı cevaplar `data-deepseek/benchmark-lms.json.predictions.jsonl` dosyasına yazılır. `delta_new_minus_old` alanında doğruluk/F1 için pozitif; MAE, süre ve hata için negatif fark iyileşme demektir.

## Türkçe ve İngilizce grafik

```bash
python3 -m pip install matplotlib
python3 plot_benchmark.py
```

Varsayılan komut iki dilde PNG ve SVG üretir. İngilizce dosyalara `-en` eklenir. Model benchmark’ını tekrar çalıştırmaz.

```bash
python3 plot_benchmark.py --lang en
python3 plot_benchmark.py --lang tr
python3 plot_benchmark.py \
  --input docs/benchmark-example.json --out data/charts/karsilastirma
```

LM Studio benchmark’ı tamamlanınca iki dil otomatik oluşturulur. Grafik üretimi başarısız olursa benchmark JSON’u korunur.

## Diğer değerlendirme yolları

API kıyaslaması için `benchmark-old.example.json` ve `benchmark-new.example.json` ayarlarını çalışan endpoint’lerle eşleştirin:

```bash
python3 pipeline.py benchmark --data data-deepseek/prepared \
  --new-manifest data-deepseek/merged/distill-manifest.json \
  --old-config benchmark-old.example.json \
  --new-config benchmark-new.example.json \
  --out data-deepseek/benchmark.json
```

Örnek portlar 8001 ve 8002’dir. Bu komut sunucuları başlatmaz; `Connection refused`, belirtilen adreste sunucu olmadığını gösterir.

API kullanmadan, modelleri sırayla CUDA’ya yüklemek için eğitim ortamıyla `benchmark_local.py` çalıştırılabilir. Bu yol JSON şemasıyla zorlanmış üretim kullanmaz; LM Studio ölçümüyle aynı deney değildir.

