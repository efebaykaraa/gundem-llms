# Gündem Score v1

[English](SCORING.md)

Bu Türkçe haber sınıflandırma görevi için **0–100 arasında bileşik kalite puanı**.
Yüksek daha iyi. Projeye özel bir endekstir; standart benchmark, doğruluk yüzdesi
veya genel LLM zekâ puanı değildir.

```text
Gündem Score = 30 × etiket_micro_F1
             + 20 × tür_doğruluğu
             + 30 × (1 − ortalama_puan_MAE / 3)
             + 20 × tam_eşleşme

ortalama_puan_MAE = (etki_MAE + ciddiyet_MAE + aciliyet_MAE) / 3
```

F1, tür doğruluğu ve tam eşleşme 0–1 aralığında kullanılır. Üç puan alanı 0–3
olduğundan MAE de 0–3 aralığındadır. Kusursuz sonuç 100; bütün cevapları geçersiz
olan bir sonuç 0 alır.

| Bileşen | Maksimum puan | Gerekçe |
| --- | ---: | --- |
| Etiket micro-F1 | 30 | Ana haber sınıflandırma görevi |
| Tür doğruluğu | 20 | Olay kategorisiyle uyum |
| Sayısal puan doğruluğu | 30 | Etki, ciddiyet, aciliyette yakın cevaba kısmi kredi; her biri 10 puan |
| Bütün alanlarda tam eşleşme | 20 | Cevabın tamamını doğru üretmeye ek ağırlık |

Ağırlıklar açık bir proje tercihidir; sıralamaya göre optimize edilmedi veya
istatistiksel olarak öğrenilmedi. Tam eşleşme, bütün alanları birlikte doğru
üretmeyi ödüllendirmek için alan metrikleriyle bilinçli olarak örtüşür. Hız,
parametre sayısı ve bellek ayrı gösterilir. Geçersiz cevaplar mevcut metriklerde
zaten cezalıdır; ayrıca geçerlilik çarpanı kullanılmaz.

Rakiplere göre min/max normalizasyonu yoktur; yeni model eklemek diğerlerinin
puanını değiştirmez. JSON ham puanı ve bileşen katkılarını saklar; tablolar bir
ondalık gösterir. Sıralama yuvarlanmamış puanla, eşitlikte tam eşleşme, F1 ve
model kimliğiyle yapılır. 101 örnekli, tek seed'li bu testte küçük puan farkları
anlamlı üstünlük kanıtı değildir. Formül/ağırlık değişirse yeni puan sürümü
gerekir; farklı sürümler kıyaslanmaz.

Yalnız aynı veri, talimat ve üretim koşullarındaki puanlar kıyaslanmalıdır.
Mevcut sonuçlar öğretmen etiketleriyle uyumu ölçer; test daha önce de kullanıldı.
Bu kısa talimatta bütün orijinal modeller JSON sözleşmesini sağlayamadığı için
sıfır aldı. Ölçülmemiş varyantlara puan üretilmez.

`bash run_experiments.sh report` CSV/JSON/Markdown'a puanı ekler; ayrıntılı
grafiklerin yanında `overall-tr.png/svg` ve `overall-en.png/svg` üretir.
Eski özet JSON dosyaları da `plot_experiments.py` ile modelden yeniden çıkarım
yapmadan puanlanıp grafiğe dökülebilir.

[Güncel sonuçlar](multimodel-results/README.tr.md) · [Uygulama](../lab/scoring.py)
