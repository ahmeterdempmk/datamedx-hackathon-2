from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from load_data import load_dataset, LAB_COLUMNS

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)


def build_features(df):
    feat_cols = [f"{c}_mean" for c in LAB_COLUMNS if f"{c}_mean" in df.columns]
    feat_cols += ["yas"]
    df = df.copy()
    df["is_kadin"] = (df["cinsiyet_clean"] == "kadın").astype(int)
    feat_cols.append("is_kadin")
    X = df[feat_cols].astype(float)
    y = df["kanser_turu"].astype(str)
    return X, y, feat_cols


def evaluate(name, pipe, X, y, cv):
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    y_pred = cross_val_predict(pipe, X, y, cv=cv)
    print(f"\n=== {name} ===")
    print(f"CV accuracy: mean={scores.mean():.3f}  std={scores.std():.3f}  folds={scores.tolist()}")
    print(classification_report(y, y_pred, digits=3))
    return y_pred


def plot_confusion(y_true, y_pred, name, fname):
    labels = sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=120)
    plt.close()


def feature_importance(rf_pipe, feat_cols, X, y):
    rf_pipe.fit(X, y)
    rf = rf_pipe.named_steps["clf"]
    imp = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=True)
    plt.figure(figsize=(8, 7))
    imp.plot(kind="barh", color="#2ca02c")
    plt.title("Random Forest Özellik Önemleri")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_feature_importance.png", dpi=120)
    plt.close()
    print("\nTop 8 önemli özellik:")
    print(imp.sort_values(ascending=False).head(8).round(4))


if __name__ == "__main__":
    df = load_dataset()
    X, y, feat_cols = build_features(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lr_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, multi_class="auto")),
    ])
    rf_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ])

    y_pred_lr = evaluate("Logistic Regression", lr_pipe, X, y, cv)
    plot_confusion(y, y_pred_lr, "Logistic Regression", "07_cm_logreg.png")

    y_pred_rf = evaluate("Random Forest", rf_pipe, X, y, cv)
    plot_confusion(y, y_pred_rf, "Random Forest", "08_cm_rf.png")

    feature_importance(rf_pipe, feat_cols, X, y)
    print(f"\nFigures written to {FIG_DIR}")
