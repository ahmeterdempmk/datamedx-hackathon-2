from pathlib import Path
import pandas as pd

from parse_utils import (
    extract_floats,
    latest_float,
    mean_float,
    parse_bracket_list,
    clean_epikriz_text,
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "datamedx_veriset_26.xlsx"

LAB_COLUMNS = [
    "hba1c", "üre", "kreatinin", "bun", "alt", "alp", "ast", "ggt",
    "bilirubin", "potasyum", "kalsiyum", "magnezyum", "klor",
    "albumin", "crp", "ldh", "sodyum",
]


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    df["cinsiyet_clean"] = df["cinsiyet"].apply(
        lambda v: parse_bracket_list(v)[0] if parse_bracket_list(v) else None
    )
    df["dogum_yili"] = pd.to_numeric(df["doğum tarihi"], errors="coerce")
    df["yas"] = 2026 - df["dogum_yili"]

    for col in LAB_COLUMNS:
        if col in df.columns:
            df[f"{col}_mean"] = df[col].apply(mean_float)
            df[f"{col}_last"] = df[col].apply(latest_float)
            df[f"{col}_n"] = df[col].apply(lambda v: len(extract_floats(v)))

    df["epikriz_clean"] = df["epikriz"].apply(clean_epikriz_text)
    df["ilac_list"] = df["ilac"].apply(parse_bracket_list)
    df["atc_list"] = df["atc kod"].apply(parse_bracket_list)
    df["icd10_list"] = df["icd10"].apply(parse_bracket_list)

    return df


if __name__ == "__main__":
    df = load_dataset()
    print(f"Loaded shape: {df.shape}")
    print(f"Cancer types:\n{df['kanser_turu'].value_counts()}")
    print(f"Gender:\n{df['cinsiyet_clean'].value_counts()}")
    print(f"Age (mean / median): {df['yas'].mean():.1f} / {df['yas'].median():.1f}")
    print(f"HbA1c mean parsed (non-null): {df['hba1c_mean'].notna().sum()} / {len(df)}")
    print(df[["kanser_turu", "yas", "cinsiyet_clean", "hba1c_mean", "kreatinin_mean"]].head())
