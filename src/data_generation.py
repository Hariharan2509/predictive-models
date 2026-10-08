"""Generate a reproducible, realistic student-performance dataset.

Run:  python src/data_generation.py
The Pass/Fail label is produced from a noisy logistic relationship with the
features, so models can learn real patterns (but it is not perfectly separable).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import DATA_PATH, RANDOM_STATE


def generate_dataset(n_samples: int = 1500, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    study_hours = np.clip(rng.normal(4.0, 2.0, n_samples), 0, 10)
    # Students who study more tend to attend slightly more
    attendance = np.clip(rng.normal(70, 12, n_samples) + 1.5 * (study_hours - 4), 30, 100)
    previous_score = np.clip(rng.normal(60, 15, n_samples) + 1.0 * (study_hours - 4), 0, 100)
    assignment_score = np.clip(rng.normal(65, 14, n_samples) + 0.4 * (previous_score - 60), 0, 100)
    sleep_hours = np.clip(rng.normal(6.8, 1.2, n_samples), 3, 10)

    # Weighted standardized features -> probability of passing
    z = (
        0.9 * (study_hours - 4.0) / 2.0
        + 0.8 * (attendance - 72) / 12.0
        + 1.1 * (previous_score - 60) / 15.0
        + 0.9 * (assignment_score - 65) / 14.0
        + 0.3 * (sleep_hours - 6.8) / 1.2
        + 0.8                                   # baseline (most students pass)
        + rng.normal(0, 0.8, n_samples)         # real-life randomness
    )
    prob_pass = 1 / (1 + np.exp(-z))
    result = (rng.random(n_samples) < prob_pass).astype(int)

    df = pd.DataFrame({
        "study_hours": study_hours.round(1),
        "attendance": attendance.round(1),
        "previous_score": previous_score.round(1),
        "assignment_score": assignment_score.round(1),
        "sleep_hours": sleep_hours.round(1),
        "result": result,
    })
    return df


def main() -> None:
    df = generate_dataset()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset saved to: {DATA_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Pass rate: {df['result'].mean():.1%}")


if __name__ == "__main__":
    main()
