# DataMedX Hackathon 2 — Stage 1 Sunum Notları

## 1. Problem Tanımı
Veri seti: 500 onkoloji hastası, 5 kanser türü (her biri 100 hasta), 54 sütun. Veri seti hem **klinik laboratuvar** değerlerini (HbA1c, kreatinin, ALT/AST/ALP/GGT, albumin, CRP, LDH, elektrolitler) hem **demografik** verileri hem de **epikriz serbest metnini** içeriyor. Çoğu hücre `[v1], [v2], …` formatında çoklu zaman noktası barındırıyor.

Üç yönlü bir çalışma yaptık:
1. **Onkoloji + diyabet** kesişimi: HbA1c üzerinden ADA eşikleriyle diyabet/prediyabet yükünün kanser türlerine göre dağılımı.
2. **NLP**: Türkçe epikriz metninden TF-IDF anahtar terimler ve regex-tabanlı semptom çıkarımı.
3. **ML**: Lab + demografik özelliklerden kanser türü tahmini için baseline sınıflandırıcılar.

## 2. Veri Hazırlama
- `parse_utils.py` ile `[v]` köşeli-parantez listelerini parse ettik; her lab için `_mean`, `_last`, `_n` (ölçüm sayısı) çıkardık.
- Yaş = 2026 − doğum yılı.
- Epikriz metninde `_x000D_` artıkları temizlendi, lower-case'e indirildi.
- `ölüm durumu` sütunu tamamen boş olduğu için kullanılmadı.

## 3. EDA Bulguları (`figures/01–04`)
- Cinsiyet: 291 kadın, 209 erkek. Beklendiği üzere meme/over → kadın, prostat → erkek baskın.
- Yaş ortalaması 67.2; en yüksek prostat (75.2), en düşük over (61.3).
- En çok eksik lab: BUN (~%17), HbA1c (~%20). Çekirdek lab paneli 500/500 hastanın çoğunda mevcut.
- Albumin ve klor değerlerinde kanser türleri arasında belirgin farklılık var (özellikle over kanseri için anormal yüksek albumin sentetik veri bulgusu olabilir).

## 4. HbA1c / Diyabet Analizi (`figures/05`)
ADA eşikleri (normal <5.7, prediyabet 5.7–6.4, diyabet ≥6.5) kullanıldı; her hasta için **maksimum** HbA1c değeri esas alındı.

| Kanser türü | Normal | Prediyabet | Diyabet | Bilinmiyor |
|---|---|---|---|---|
| Karaciğer | 25% | 20% | 34% | 21% |
| Meme | 16% | 32% | 36% | 16% |
| Multipl miyelom | 20% | 17% | 27% | 36% |
| Over | 34% | 31% | 20% | 15% |
| Prostat | 22% | 28% | 39% | 11% |

Ki-kare testi: χ² = 17.03, dof = 8, **p = 0.030** → kanser türleri arasında HbA1c kategorilerinde **istatistiksel olarak anlamlı fark** var. Prostat ve meme kanserinde diyabetik oran (~%36–39) ile en yüksek; over kanserinde en düşük (%20).

> Klinik öneri: Prostat ve meme onkoloji takibi yapan kliniklerde rutin HbA1c taraması ve metabolik ko-morbidite yönetimi (örn. diyetisyen yönlendirmesi) için tetik kuralları tasarlanabilir.

## 5. NLP Bulguları (`figures/06`)
**TF-IDF anahtar terimler** (ilk 10) her kanser türü için anlamlı tıbbi sinyaller verdi: prostat → "psma", over → "over"/"markerleri", meme → meme spesifik tedavi terimleri, miyelom → "kür" (kemoterapi siklusu).

**Semptom regex sıklığı**: 5 türün hepsinde "ağrı" %95+ — beklenen onkolojik tablo. Ayırt edici bulgular:
- Bulantı/kusma → karaciğer ve miyelomda daha yüksek.
- Nefes darlığı → prostat (%47) ve meme (%42) öne çıkıyor (kemik metastazı / efüzyon hipotezi).
- Kabızlık → miyelom ve prostat (opioid kullanımı, hiperkalsemi olası nedenler).

> Uygulama fikri: Epikriz girilirken anlık olarak semptom etiketlerini çıkaran bir **Türkçe NLP eklentisi** klinik dokümantasyonu yapısallaştırabilir.

## 6. Baseline Sınıflandırıcı (`figures/07–09`)
Hedef: 17 lab + yaş + cinsiyet → 5 sınıflı kanser türü. 5-fold stratified CV.

| Model | Accuracy (mean ± std) | Macro F1 |
|---|---|---|
| Logistic Regression | **0.684 ± 0.027** | 0.684 |
| Random Forest | **0.712 ± 0.031** | 0.710 |

Random=%20 — model %71.2 ile bunun **3.5 katından** iyi. En kolay ayrılan: meme (F1=0.90, çünkü cinsiyet+albumin imzası çok güçlü). En zor: multipl miyelom (F1=0.53), karaciğer ile karışıyor (her ikisinde de bozuk hepatik panel).

**RF özellik önemleri (top 5):** klor_mean, is_kadin, ast_mean, ggt_mean, yas.

> Klinik öneri: HBP içine, lab paneli girildikten sonra kanser tipiyle uyumsuz lab paterni varsa **uyarı** veren bir karar destek modeli entegre edilebilir.

## 7. Sınırlamalar
- 500 hasta küçük örneklem; sentetik üretim olası.
- Lab değerlerinde zaman serisi var ama biz `_mean`/`_last` özetleriyle yetindik.
- HbA1c ve "Bilinmiyor" oranı (%11–%36) yüksek; eksik veri kanser türüyle korele olabilir → potansiyel önyargı.
- Türkçe stopword listesi minimal; gerçek üretimde Zemberek/Trnlp gibi morfolojik çözümleyici kullanılmalı.

## 8. Sonraki Adımlar
1. Lab zaman-serisinden trend özellikleri (eğim, varyasyon) çıkarmak.
2. Epikrizden gerçek NER (BERTurk + onkoloji etiket seti).
3. Kanser türü yerine **prognoz / mortalite** modeli (ölüm tarihi sütunu doluysa survival analiz).
4. ATC kod hiyerarşisinden ilaç-temelli kümeleme (terapi profilleri).

## 9. Üretilen Çıktılar
- `src/`: parse_utils, load_data, eda, hba1c_analysis, nlp_epikriz, model_baseline.
- `figures/`: 9 grafik.
- Tekrar üretilebilirlik: `pip install -r requirements.txt && python src/eda.py && python src/hba1c_analysis.py && python src/nlp_epikriz.py && python src/model_baseline.py`.
