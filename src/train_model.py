"""Train Logistic Regression, Decision Tree and Random Forest; evaluate; save the best.

Run:  python src/train_model.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import (COMPARISON_PATH, FEATURES, METADATA_PATH, MODEL_DIR,
                        MODEL_PATH, OUTPUT_DIR, RANDOM_STATE)
from src.evaluate_model import (compute_metrics, plot_confusion_matrix,
                                plot_feature_importance, plot_model_comparison,
                                plot_roc_curves, select_best_model)
from src.preprocessing import clean_data, load_data, run_eda, split_data


def build_models() -> dict:
    """Every model is a Pipeline, so the saved file always includes its own preprocessing.
    The scaler lives INSIDE the pipeline, so it is fitted on training data only (no leakage)."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        # Trees do not need scaling; 'passthrough' keeps the pipeline structure consistent.
        "Decision Tree": Pipeline([
            ("scaler", "passthrough"),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", "passthrough"),
            ("model", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ]),
    }


def main() -> None:
    print("=" * 60, "\n1. LOAD & CLEAN DATA\n" + "=" * 60)
    df = clean_data(load_data())

    print("\n" + "=" * 60, "\n2. EXPLORATORY DATA ANALYSIS\n" + "=" * 60)
    run_eda(df)

    print("\n" + "=" * 60, "\n3. TRAIN / TEST SPLIT (80/20, stratified)\n" + "=" * 60)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    print("\n" + "=" * 60, "\n4. TRAIN & EVALUATE MODELS (on unseen test set)\n" + "=" * 60)
    models = build_models()
    rows = []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        scores = compute_metrics(pipe, X_test, y_test)
        rows.append({"Model": name, **scores})
        plot_confusion_matrix(pipe, name, X_test, y_test)
        print(f"Trained {name}")

    results = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(COMPARISON_PATH, index=False)
    plot_roc_curves(models, X_test, y_test)
    plot_model_comparison(results)
    importance = plot_feature_importance(models["Random Forest"])

    print("\nModel Comparison (test set):")
    print(results.round(4).to_string(index=False))
    print("\nRandom Forest feature importance:")
    print(importance.round(4).to_string(index=False))

    print("\n" + "=" * 60, "\n5. SELECT & SAVE BEST MODEL\n" + "=" * 60)
    best_name = select_best_model(results)
    best_row = results[results["Model"] == best_name].iloc[0]
    print(f"Best Model: {best_name}")
    print(f"Reason: it achieved the highest ROC-AUC ({best_row['ROC-AUC']:.4f}) "
          f"(ties within 0.005 AUC are broken by F1; its F1 = {best_row['F1 Score']:.4f}, "
          f"accuracy = {best_row['Accuracy']:.4f}).")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models[best_name], MODEL_PATH)
    METADATA_PATH.write_text(json.dumps({
        "best_model": best_name,
        "features": FEATURES,
        "metrics": {k: round(float(v), 4) for k, v in best_row.drop("Model").items()},
    }, indent=2))
    print(f"Saved model (with preprocessing) to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
