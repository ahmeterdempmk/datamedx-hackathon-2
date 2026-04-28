# DataMedX Hackathon 2 — Stage 1

İstinye Üniversitesi DataMedX Hackathon 2 ön değerlendirme görevi.

## Görev
Paylaşılan veri seti (`data/datamedx_veriset_26.xlsx`) üzerinde onkoloji / NLP / veri analizi yönünde basit bir uygulama, EDA ve baseline model geliştirmek.

## Veri Seti
- 500 hasta × 54 sütun
- 5 kanser türü (her biri 100 hasta): Karaciğer, Meme, Multipl miyelom, Over, Prostat
- HbA1c, böbrek/karaciğer/elektrolit lab değerleri, ilaç (ATC kodlu), ICD10, epikriz serbest metni

## Yaklaşım
1. **Veri parse + temizleme** — `[v1], [v2]` formatlı çoklu zaman noktalarını çözen `parse_utils`.
2. **EDA** — demografi, eksik değer haritası, lab dağılımları.
3. **HbA1c diyabet yükü** — ADA eşikleriyle kanser türü × diyabet kategorisi (ki-kare p=0.030, anlamlı).
4. **Türkçe NLP** — epikriz TF-IDF anahtar terim + 10 semptom için regex sıklığı.
5. **Baseline ML** — lab+demografi → kanser türü; Random Forest 5-fold CV **%71.2 accuracy** (random=%20).

Detaylı sunum: [`docs/sunum.md`](docs/sunum.md)

## Kurulum & Çalıştırma
```bash
pip install -r requirements.txt
# Excel dosyasını data/datamedx_veriset_26.xlsx altına koyun (gizlilik sebebiyle repo'da yok)
python src/load_data.py        # smoke test
python src/eda.py              # figures/01–04
python src/hba1c_analysis.py   # figures/05
python src/nlp_epikriz.py      # figures/06
python src/model_baseline.py   # figures/07–09
```

## Ana Sonuçlar
| Metrik | Değer |
|---|---|
| Kanser türü × HbA1c kategorisi (χ²) | p = 0.030 (anlamlı) |
| En diyabetik grup | Prostat (%39) |
| Logistic Regression CV accuracy | 0.684 ± 0.027 |
| Random Forest CV accuracy | **0.712 ± 0.031** |
| RF en önemli 3 özellik | klor, cinsiyet, AST |

## Repo Yapısı
```
src/                  parse_utils, load_data, eda, hba1c_analysis, nlp_epikriz, model_baseline
figures/              9 grafik (commit'lenmiş)
docs/sunum.md         sunum notları
data/                 .gitignore (ham veri commit'lenmez)
```

## Lisans
Yarışma katılımı için hazırlanmıştır.
