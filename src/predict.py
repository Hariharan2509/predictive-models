"""Load the saved model and predict for new student data.

Run an example:  python src/predict.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd

from src.config import FEATURES, METADATA_PATH, MODEL_PATH


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run:  python src/train_model.py")
    return joblib.load(MODEL_PATH)


def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}


def predict_student(model, study_hours, attendance, previous_score,
                    assignment_score, sleep_hours) -> dict:
    """Return the label (PASS/FAIL) and the model's real probability of passing."""
    row = pd.DataFrame([[study_hours, attendance, previous_score,
                         assignment_score, sleep_hours]], columns=FEATURES)
    pred = int(model.predict(row)[0])
    prob_pass = float(model.predict_proba(row)[0][1])
    return {"label": "PASS" if pred == 1 else "FAIL",
            "prob_pass": prob_pass, "prob_fail": 1 - prob_pass}


if __name__ == "__main__":
    mdl = load_model()
    examples = {
        "Strong student": (7, 92, 85, 88, 7.5),
        "Average student": (4, 72, 60, 65, 7),
        "Weak student": (1, 45, 35, 40, 5),
    }
    for label, vals in examples.items():
        r = predict_student(mdl, *vals)
        print(f"{label:16s} {vals} -> {r['label']} (P(pass) = {r['prob_pass']:.1%})")
