"""Data loading, cleaning, splitting and exploratory data analysis (EDA)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # no GUI needed; works on any machine
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.config import (DATA_PATH, FEATURES, FEATURE_LABELS, FIGURE_DIR,
                        RANDOM_STATE, TARGET, TEST_SIZE)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run:  python src/data_generation.py")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Inspect and clean the dataset (missing values, duplicates, types)."""
    if verbose:
        print(f"Dataset shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head().to_string(index=False))
        print("\nData types:")
        print(df.dtypes.to_string())
        print("\nMissing values per column:")
        print(df.isnull().sum().to_string())
        print(f"\nDuplicate rows: {df.duplicated().sum()}")

    df = df.drop_duplicates().copy()
    # Fill any missing values with the column median (none expected here).
    for col in FEATURES:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    if verbose:
        print(f"Shape after cleaning: {df.shape}")
    return df


def split_data(df: pd.DataFrame):
    """Separate X / y and make a stratified 80/20 train-test split."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=y)


def _save(fig, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=150)
    plt.close(fig)


def run_eda(df: pd.DataFrame) -> None:
    """Create and save the EDA graphs."""
    sns.set_theme(style="whitegrid")
    labelled = df.copy()
    labelled["Outcome"] = labelled[TARGET].map({1: "Pass", 0: "Fail"})
    palette = {"Pass": "#2e8b57", "Fail": "#d9534f"}

    # 1. Pass vs Fail distribution
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.countplot(data=labelled, x="Outcome", order=["Fail", "Pass"],
                  hue="Outcome", palette=palette, legend=False, ax=ax)
    for bar in ax.patches:
        ax.annotate(int(bar.get_height()), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=11)
    ax.set_title("Distribution of Pass vs Fail")
    ax.set_xlabel("Result")
    ax.set_ylabel("Number of Students")
    _save(fig, "pass_fail_distribution.png")

    # 2-4. Feature vs result (box plots)
    for col, fname in [("study_hours", "study_hours_vs_result.png"),
                       ("attendance", "attendance_vs_result.png"),
                       ("previous_score", "previous_score_vs_result.png")]:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(data=labelled, x="Outcome", y=col, order=["Fail", "Pass"],
                    hue="Outcome", palette=palette, legend=False, ax=ax)
        ax.set_title(f"{FEATURE_LABELS[col]} vs Result")
        ax.set_xlabel("Result")
        ax.set_ylabel(FEATURE_LABELS[col])
        _save(fig, fname)

    # 5. Correlation heatmap
    fig, ax = plt.subplots(figsize=(7, 5.5))
    corr = df[FEATURES + [TARGET]].rename(columns={**FEATURE_LABELS, TARGET: "Result"}).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
    ax.set_title("Correlation Heatmap")
    _save(fig, "correlation_heatmap.png")
    print(f"EDA figures saved to: {FIGURE_DIR}")
