from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from load_data import load_dataset

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

TURKISH_STOPWORDS = {
    "ve", "ile", "için", "bir", "bu", "olan", "olarak", "var", "yok", "de", "da",
    "den", "dan", "ki", "mi", "mı", "ne", "ya", "ama", "fakat", "veya", "çok",
    "sonra", "önce", "şu", "her", "tüm", "tedavi", "hasta", "hastanın", "öykü",
    "yakınması", "şikayeti", "kontrol", "izlem", "plan", "no", "belirtilmedi",
    "durumu", "önerilme", "bulunmuyor", "ediliyor", "edildi", "bulunmamaktadır",
    "değerlendirme", "amacıyla", "edilmiştir", "yapılmıştır", "alınmıştır",
}

SYMPTOMS = {
    "ağrı": r"\bağrı\w*",
    "bulantı": r"\bbulantı\w*",
    "kusma": r"\bkusma\w*",
    "yorgunluk": r"\byorgun\w*|halsiz\w*",
    "ateş": r"\bateş\w*",
    "iştahsızlık": r"\biştahsız\w*",
    "kilo kaybı": r"kilo kayb\w*",
    "nefes darlığı": r"nefes darlığ\w*|dispne",
    "kabızlık": r"\bkabızlık\w*",
    "ishal": r"\bishal\w*",
}


def top_terms_per_cancer(df, top_k=10):
    docs, labels = [], []
    for cancer, grp in df.groupby("kanser_turu"):
        text = " ".join(grp["epikriz_clean"].dropna().tolist())
        docs.append(text)
        labels.append(cancer)

    vec = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 1),
        token_pattern=r"(?u)\b[a-zçğıöşü]{3,}\b",
        stop_words=list(TURKISH_STOPWORDS),
    )
    X = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out()

    print("\n=== Kanser türü başına TF-IDF ile öne çıkan terimler ===")
    for i, label in enumerate(labels):
        row = X[i].toarray().ravel()
        top_idx = row.argsort()[::-1][:top_k]
        print(f"\n{label}:")
        for j in top_idx:
            print(f"  {vocab[j]:25s}  {row[j]:.3f}")


def symptom_counts(df):
    rows = []
    for cancer, grp in df.groupby("kanser_turu"):
        texts = grp["epikriz_clean"].dropna()
        row = {"kanser_turu": cancer, "n": len(texts)}
        for name, pattern in SYMPTOMS.items():
            row[name] = sum(bool(re.search(pattern, t)) for t in texts)
        rows.append(row)
    out = pd.DataFrame(rows).set_index("kanser_turu")
    pct = out.drop(columns="n").div(out["n"], axis=0) * 100
    print("\n=== Epikrizde semptom yüzdeleri ===")
    print(pct.round(1))
    return pct


if __name__ == "__main__":
    df = load_dataset()
    top_terms_per_cancer(df, top_k=10)
    pct = symptom_counts(df)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax = pct.plot(kind="bar", figsize=(11, 6), colormap="tab10")
    ax.set_title("Kanser Türüne Göre Epikrizde Semptom Sıklığı (%)")
    ax.set_ylabel("Hasta yüzdesi")
    plt.xticks(rotation=20, ha="right")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_symptom_frequency.png", dpi=120)
    plt.close()
    print(f"\nFigure saved: {FIG_DIR / '06_symptom_frequency.png'}")
