"""Central configuration: project-relative paths and column names."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "student_performance.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
COMPARISON_PATH = OUTPUT_DIR / "model_comparison.csv"

FEATURES = ["study_hours", "attendance", "previous_score", "assignment_score", "sleep_hours"]
TARGET = "result"  # 1 = PASS, 0 = FAIL

# Nice names for charts and the web app
FEATURE_LABELS = {
    "study_hours": "Study Hours",
    "attendance": "Attendance (%)",
    "previous_score": "Previous Exam Score",
    "assignment_score": "Assignment Score",
    "sleep_hours": "Sleep Hours",
}

RANDOM_STATE = 42
TEST_SIZE = 0.20
