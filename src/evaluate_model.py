"""Metrics and evaluation plots (all computed on the unseen TEST set)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)

from src.config import FEATURE_LABELS, FEATURES, FIGURE_DIR


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _save(fig, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=150)
    plt.close(fig)


def compute_metrics(model, X_test, y_test) -> dict:
    """Accuracy, precision, recall, F1 and ROC-AUC on the test set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }


def plot_confusion_matrix(model, name, X_test, y_test) -> None:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Fail (0)", "Pass (1)"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.grid(False)
    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    # Name each cell: TN, FP, FN, TP
    cell_names = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
    for i in range(2):
        for j in range(2):
            dark_cell = cm[i, j] > cm.max() / 2
            ax.text(j, i + 0.28, cell_names[i][j], ha="center", va="center",
                    fontsize=9, color="white" if dark_cell else "dimgray")
    _save(fig, f"confusion_matrix_{_slug(name)}.png")


def plot_roc_curves(models: dict, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, model in models.items():
        prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.500)")
    ax.set_title("ROC Curves - Model Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, "roc_curve.png")


def plot_model_comparison(results: pd.DataFrame) -> None:
    """Grouped bar chart of all metrics for every model."""
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    long_df = results.melt(id_vars="Model", value_vars=metrics,
                           var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=long_df, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Comparison (Test Set)")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)
    ax.legend(loc="lower right")
    _save(fig, "model_comparison.png")


def plot_feature_importance(rf_pipeline) -> pd.DataFrame:
    importances = rf_pipeline.named_steps["model"].feature_importances_
    imp = (pd.DataFrame({"Feature": [FEATURE_LABELS[f] for f in FEATURES],
                         "Importance": importances})
           .sort_values("Importance", ascending=True))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp["Feature"], imp["Importance"], color="#3b7dd8")
    for y, v in enumerate(imp["Importance"]):
        ax.text(v + 0.003, y, f"{v:.3f}", va="center", fontsize=9)
    ax.set_title("Random Forest - Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    _save(fig, "random_forest_feature_importance.png")
    return imp.sort_values("Importance", ascending=False)


def select_best_model(results: pd.DataFrame, tolerance: float = 0.005) -> str:
    """Best = highest ROC-AUC; if within `tolerance` of the top AUC, use F1 as tie-breaker."""
    top_auc = results["ROC-AUC"].max()
    contenders = results[results["ROC-AUC"] >= top_auc - tolerance]
    return contenders.sort_values(["F1 Score", "ROC-AUC"], ascending=False).iloc[0]["Model"]
