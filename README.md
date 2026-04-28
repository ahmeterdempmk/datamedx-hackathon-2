# DataMedX Hackathon 2 — Stage 1

İstinye Üniversitesi DataMedX Hackathon 2 ön değerlendirme görevi.

## Görev
Paylaşılan veri seti (`data/datamedx_veriset_26.xlsx`) üzerinde onkoloji / NLP / veri analizi yönünde basit bir uygulama, EDA ve baseline model geliştirmek.

## Veri Seti
- 500 hasta × 54 sütun
- 5 kanser türü (her biri 100 hasta): Karaciğer, Meme, Multipl miyelom, Over, Prostat
- HbA1c, böbrek/karaciğer/elektrolit lab değerleri, ilaç (ATC kodlu), ICD10, epikriz serbest metni

## Yaklaşım
1. Veri parse + temizleme (`[v1], [v2]` formatlı çoklu zaman noktaları)
2. EDA (demografi, eksik değer, lab dağılımları)
3. HbA1c üzerinden diyabet/prediyabet yükü analizi (ADA eşikleri)
4. Epikriz metninde TF-IDF + semptom regex çıkarımı
5. Lab + demografi → kanser türü baseline classifier

Detaylı sunum: [`docs/sunum.md`](docs/sunum.md)

## Lisans
Yarışma katılımı için hazırlanmıştır.
