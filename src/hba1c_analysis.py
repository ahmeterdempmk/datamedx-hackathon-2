from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency

from load_data import load_dataset

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)


def hba1c_category(value):
    if value is None or pd.isna(value):
        return "bilinmiyor"
    if value < 5.7:
        return "normal"
    if value < 6.5:
        return "prediyabet"
    return "diyabet"


def run():
    df = load_dataset()
    df["hba1c_max"] = df["hba1c"].apply(lambda v: max(_extract(v), default=None))
    df["hba1c_kategori"] = df["hba1c_max"].apply(hba1c_category)

    table = (
        df.groupby(["kanser_turu", "hba1c_kategori"]).size().unstack(fill_value=0)
        [["normal", "prediyabet", "diyabet", "bilinmiyor"]]
    )
    print("Kanser türü × HbA1c kategorisi (sayı):")
    print(table)

    pct = table.div(table.sum(axis=1), axis=0) * 100
    print("\nYüzde dağılım:")
    print(pct.round(1))

    observed = table[["normal", "prediyabet", "diyabet"]]
    chi2, p, dof, _ = chi2_contingency(observed.values)
    print(f"\nKi-kare: chi2={chi2:.2f} dof={dof} p={p:.4f}")
    print("Yorum:", "anlamlı fark var" if p < 0.05 else "anlamlı fark yok")

    ax = pct[["normal", "prediyabet", "diyabet"]].plot(
        kind="bar", stacked=True, figsize=(9, 5),
        color=["#2ca02c", "#ff7f0e", "#d62728"],
    )
    ax.set_title("Kanser Türüne Göre HbA1c (ADA) Kategorileri (%)")
    ax.set_ylabel("Hasta yüzdesi")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Kategori")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_hba1c_categories.png", dpi=120)
    plt.close()
    print(f"\nFigure saved: {FIG_DIR / '05_hba1c_categories.png'}")


def _extract(v):
    from parse_utils import extract_floats
    return extract_floats(v)


if __name__ == "__main__":
    run()
