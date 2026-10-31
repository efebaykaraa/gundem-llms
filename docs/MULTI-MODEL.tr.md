# Çoklu model deneyleri

[English](MULTI-MODEL.md)

[Tamamlanan 20 ölçümün sonuçları](multimodel-results/README.tr.md): tam tablo,
TR/EN grafikler, CSV/JSON ve deney bilgileri.

Her model, aynı başlık veri kümesinde **orijinal ve distilled** olarak ölçülür.
Kayıt listesi önceki iki LFM3 varyantını ve istenen sekiz yeni modeli içerir:
GPT-X3, Octen WebGR, Spark, Llama 1B/3B, Limite ve Qwen3.5 0.8B/2B Base.
Toplam on model ve yirmi varyant vardır.

**[Gündem Score v1](SCORING.tr.md)** sonuçları 0–100 kalite puanıyla sıralar; CSV/JSON/Markdown ve TR/EN `overall` grafikleri otomatik üretilir. Hız ayrı tutulur.

## Klasör düzeni

```text
configs/models.json            # Modeller ve ortak deney ayarları
lab/                           # Eğitim kuyruğu, yükleme, ölçüm ve grafik kodu
run_experiments.sh             # Eğitim ortamını seçen giriş komutu
workspace/                     # Yerel çıktılar; Git'e dahil edilmez
  datasets/news-v1/            # Ortak, hash ile doğrulanan veri ayrımı
  experiments/<model-id>/
    model.lock.json            # Sabitlenen model commit'i
    state.json                 # İşlem durumu / hata
    adapter/                   # LoRA, checkpoint ve eğitim manifesti
    train.log
    original.json              # Orijinal modelin tamamlanan ölçümü
    distilled.json             # Distilled modelin tamamlanan ölçümü
    *.predictions.jsonl        # Başlık bazında gerçek çıktı ve hatalar
  envs/                        # Mevcut eğitim/converter ortamlarına bağlantılar
  reports/
    leaderboard.json
    leaderboard.csv
    leaderboard.md
    comparison-tr.png / .svg
    comparison-en.png / .svg
```

Eski LFM adapter’ları `data-deepseek` ve `data-deepseek-base` içinden bağlantıyla
kullanılır; eski dosyalar ve komutlar korunur. Temel ağırlıklar HF önbelleğinde
kalır. Distilled ölçümünde adapter CPU belleğinde birleştirilip GPU’ya aktarılır;
her model için diske ayrıca tam ağırlık kopyası yazılmaz. GGUF veya tam HF modeli
paylaşmak için ayrıca dışa aktarma yapılabilir.

## Kurulum ve veri hazırlığı

Mevcut eğitim ortamını kullanın veya `requirements-train.txt` ile bir ortam kurun.
Grafik ve isteğe bağlı hibrit model hızlandırması:

```bash
data-deepseek/venv-train/bin/python -m pip install \
  -r requirements-train.txt -r requirements-report.txt -r requirements-kernels.txt

bash run_experiments.sh init --source data-deepseek/prepared
bash run_experiments.sh preflight
```

Kaynak klasörde `manifest.json`, `train.jsonl`, `validation.jsonl` ve `test.jsonl`
bulunmalıdır. Veri ilk seferde kopyalanır; tekrar kullanımda hash’ler eşleşmelidir.
Yeni klonlanan repoda tam veri ve ağırlıklar yoktur; veriyi önce hazırlayın veya
ayrıca edinin. 80 örnek başlık, bu deneyin tam verisi değildir.

Wrapper sırasıyla `DISTILL_PYTHON`, `workspace/envs/training/bin/python`, mevcut
`data-deepseek/venv-train/bin/python`, ardından `python3` seçer. Başka bir ortam:

```bash
DISTILL_PYTHON=/tam/yol/venv/bin/python bash run_experiments.sh preflight
```

Llama için model sayfalarındaki erişim onayından sonra yerelde `hf auth login`
yapılmalıdır. Global CLI ile eğitim ortamı normal HF kimlik bilgisi önbelleğini
paylaşır. Token yapılandırmaya veya repoya yazılmaz.

## Çalıştırma ve devam

```bash
# Bütün modeller; GPU işleri sırayla yürür.
bash run_experiments.sh run

# Yalnız seçilen modeller:
bash run_experiments.sh run --models gpt-x3-150m llama-3.2-1b qwen3.5-0.8b-base

# İş devam ederken durum:
bash run_experiments.sh status

# Eğitimi tekrar yapmadan mevcut adapter'ları değerlendir:
bash run_experiments.sh run --evaluate-only --models lfm3-posttrained lfm3-base

# Kaydedilmiş ölçümlerden tablo ve iki dilde grafikleri yeniden üret:
bash run_experiments.sh report
```

Yalnız paylaşılan özet JSON’dan sekiz grafiği yeniden üretmek için ağırlık, veri
veya eğitim ortamı gerekmez. `requirements-report.txt` kurulumu yeterlidir:

```bash
python3 plot_experiments.py --input workspace/reports/leaderboard.json --out data/charts
```

Tamamlanan uyumlu eğitimler tekrar yapılmaz. Eğitim, optimizer durumu bulunan
checkpoint’ten devam eder. İlk checkpoint’ten önce kesilirse yarım klasör saklanıp
eğitim yeniden başlatılır. Değerlendirme; veri, model sürümü, adapter ağırlıkları,
ayarlar, kod hash’leri ve çalışma ortamı sürümleriyle kimliklendirilir. Uyuşmayan
ölçümler yeniden yapılır. Yarım değerlendirme dosyaları yalnız teşhis içindir;
tamamlanmış sonuç sayılmaz.

Bir modelin hatası diğer modelleri durdurmaz. Erişilemeyen, başarısız veya bekleyen
varyant grafikte boş görünür; uydurma sıfır puan almaz. Modelin gerçekten ürettiği
geçersiz JSON cevapları ise hata sayılır. GPU/bellek/çalışma hatası, bütün
başlıkları başarısız tahmine çevirmek yerine o ölçümü durdurur.

## Adil kıyas ve sınırlar

Yereldeki ortak ayrım 825 eğitim, 107 doğrulama ve 101 test örneğidir; sayılar
manifestten okunur. Üç epoch, LoRA ayarları ve kısa görev talimatı ortaktır.
Her mimarinin şablonu açıkça seçilir; aynı modelin orijinal ve distilled
varyantlarının şablon hash’i aynı olmak zorundadır. Modeller arasında talimat
aynıdır, ancak farklı tokenizer/şablonlar aynı token dizisini üretmez.

Önemli mimari farklar:

- Octen WebGR URL kimliği üreten özel bir arama modelidir. Burada haber JSON’una
  aktarımı denenir; kendi arama kalitesi ölçülmez. Metin tokenizer’ının dışında
  üretilen SID kimlikleri geçersiz cevap sayılır.
- Limite’nin hazır şablonu sabit matematik talimatı zorunlu kılar. İki varyantta da
  ortak görev talimatıyla açık ChatML kullanılır. Deneysel LoRA yolu, doğrudan
  ağırlık okuyan katmanlarda adapter gradyanını korur; ayrı smoke test vardır.
  Bu doğrudan çağrılar PEFT'nin feature dropout yolunu atlar; yapılandırmadaki
  0.05 LoRA dropout bu yollarda uygulanmaz. Limite sonucu bu mimari farkla
  birlikte yorumlanmalıdır.
- Qwen3.5, yalnız metin modeli sınıfıyla çalıştırılır; görsel ağırlıklar eğitilmez.
- Yerel sohbet şablonu olmayan Llama Base modellerinde açık düz metin görev
  şablonu kullanılır.
- Spark'ın sabitlenmiş eski remote code'u için tied-weight metadata ve causal
  mask argümanları, kullanılan Transformers sürümüne uyarlanır.

Yeni eğitimlerde ortak shifted causal cross-entropy, biriktirilen batch’teki
assistant hedef token sayısına bölünerek kullanılır (`--uniform-loss`). Böylece
özel mimarilerin `num_items_in_batch` farkları eğitimi değiştirmez. Loss ve
gradyanlar HF’nin standart causal loss’u ile sayısal olarak doğrulanır. Önceki
iki LFM adapter’ı aynı standart HF hedefini kullanır; sabit sürüm ve manifestleri
açıkça tanınır. Native loss ile yapılan ön denemeler arşivlenir ve son kıyasa alınmaz.

Kalite ölçümleri öğretmen etiketleriyle uyumdur, bağımsız insan doğruluğu değildir.
Test kümesi önceki deneylerde de kullanıldı. Tek seed ve küçük veriyle genel model
üstünlüğü iddia edilmemelidir. Geçersiz cevaplar paydada kalır; her puana maksimum
3 hata eklenir. Rapor tam eşleşme, etiket micro-F1, tür doğruluğu, üç puanın MAE’si,
geçersiz cevaplar ve orijinal→distilled farklarını içerir.

Bütün varyantlar HF/CUDA BF16, greedy, JSON grammar olmadan ve batch=4 ile ölçülür.
Süre GPU senkronizasyonuyla, ısınma sonrasında ölçülür; model yükleme, birleştirme
ve eğitim dahil değildir. Başlık başına süre batch’e bölünmüş amorti süredir;
seri API gecikmesi veya telefon performansı değildir. Çalışma ortamı sürümleri
uyuşmayan sonuçlar aynı tabloya alınmaz. FLA kernel kurulum/değişikliği önceki
önbellekli değerlendirmeleri geçersiz kılar.

Önceki LM Studio Q4/JSON grammar ölçümleri için `benchmark_lms.sh` kullanılmaya
devam edilebilir. HF eğitim desteği, otomatik olarak llama.cpp/GGUF desteği
anlamına gelmez. Dışa aktarma için `train.py merge` ve `export_gguf.py` araçları
korunmuştur.
