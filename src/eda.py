from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_dataset, LAB_COLUMNS

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)


def plot_gender_by_cancer(df):
    ct = df.groupby(["kanser_turu", "cinsiyet_clean"]).size().unstack(fill_value=0)
    ax = ct.plot(kind="bar", stacked=True, figsize=(8, 5), color=["#e377c2", "#1f77b4"])
    ax.set_title("Kanser Türüne Göre Cinsiyet Dağılımı")
    ax.set_ylabel("Hasta Sayısı")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_gender_by_cancer.png", dpi=120)
    plt.close()


def plot_age_distribution(df):
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="kanser_turu", y="yas", palette="Set2")
    plt.title("Kanser Türüne Göre Yaş Dağılımı")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_age_by_cancer.png", dpi=120)
    plt.close()


def plot_missingness(df):
    cols = [f"{c}_mean" for c in LAB_COLUMNS if f"{c}_mean" in df.columns]
    miss = df[cols].isna().mean().sort_values()
    plt.figure(figsize=(8, 6))
    miss.plot(kind="barh", color="#d62728")
    plt.title("Lab Değerlerinde Eksik Oranı")
    plt.xlabel("Eksik oranı")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_lab_missingness.png", dpi=120)
    plt.close()


def plot_lab_boxplots(df):
    key_labs = ["hba1c_mean", "kreatinin_mean", "alt_mean", "albumin_mean", "crp_mean", "ldh_mean"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, col in zip(axes.flat, key_labs):
        if col not in df.columns:
            continue
        sns.boxplot(data=df, x="kanser_turu", y=col, ax=ax, palette="Set3")
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=25)
        ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_key_labs_by_cancer.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    df = load_dataset()
    plot_gender_by_cancer(df)
    plot_age_distribution(df)
    plot_missingness(df)
    plot_lab_boxplots(df)
    print(f"EDA figures written to {FIG_DIR}")
    print(df.groupby("kanser_turu")[["yas", "hba1c_mean", "kreatinin_mean", "albumin_mean"]].mean().round(2))
