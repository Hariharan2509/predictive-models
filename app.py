"""Streamlit app: Student Performance Prediction Using Machine Learning.

Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from src.config import COMPARISON_PATH, FEATURE_LABELS, FIGURE_DIR
from src.predict import load_metadata, load_model, predict_student

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")


@st.cache_resource
def get_model():
    return load_model()


st.title("🎓 Student Performance Prediction Using Machine Learning")
st.caption("Predict student academic outcome using supervised machine learning")

try:
    model = get_model()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()
except Exception as err:  # e.g. model saved with a different scikit-learn version
    st.error(f"The saved model could not be loaded ({err}). "
             "Re-create it by running:  python src/train_model.py")
    st.stop()

meta = load_metadata()

# ---------------- Sidebar: student inputs ----------------
st.sidebar.header("Student Details")
study_hours = st.sidebar.slider(FEATURE_LABELS["study_hours"] + " (per day)", 0.0, 10.0, 4.0, 0.1)
attendance = st.sidebar.slider(FEATURE_LABELS["attendance"], 0.0, 100.0, 75.0, 0.5)
previous_score = st.sidebar.slider(FEATURE_LABELS["previous_score"], 0.0, 100.0, 60.0, 0.5)
assignment_score = st.sidebar.slider(FEATURE_LABELS["assignment_score"], 0.0, 100.0, 65.0, 0.5)
sleep_hours = st.sidebar.slider(FEATURE_LABELS["sleep_hours"] + " (per day)", 0.0, 12.0, 7.0, 0.1)
predict_clicked = st.sidebar.button("Predict Result", type="primary")

tab_predict, tab_dashboard = st.tabs(["Student Prediction", "Model Performance Dashboard"])

# ---------------- Section 1: prediction ----------------
with tab_predict:
    st.header("Student Prediction")
    st.write("Enter the student's details in the sidebar and click **Predict Result**.")
    if meta:
        st.info(f"Model in use: **{meta['best_model']}** (automatically selected as the best model)")

    summary = pd.DataFrame({
        "Feature": [FEATURE_LABELS[k] for k in FEATURE_LABELS],
        "Value": [study_hours, attendance, previous_score, assignment_score, sleep_hours],
    })
    st.table(summary)

    if predict_clicked:
        result = predict_student(model, study_hours, attendance, previous_score,
                                 assignment_score, sleep_hours)
        if result["label"] == "PASS":
            st.success(f"Prediction: {result['label']}")
        else:
            st.error(f"Prediction: {result['label']}")
        c1, c2 = st.columns(2)
        c1.metric("Probability of Passing", f"{result['prob_pass'] * 100:.1f}%")
        c2.metric("Probability of Failing", f"{result['prob_fail'] * 100:.1f}%")
        st.progress(min(max(result["prob_pass"], 0.0), 1.0))

# ---------------- Section 2: dashboard ----------------
with tab_dashboard:
    st.header("Model Performance Dashboard")
    st.write("All metrics are calculated on the unseen 20% test set.")

    if not COMPARISON_PATH.exists():
        st.warning("Run `python src/train_model.py` first to generate model results.")
    else:
        results = pd.read_csv(COMPARISON_PATH)
        best = results.sort_values("ROC-AUC", ascending=False).iloc[0]
        if meta:
            best = results[results["Model"] == meta["best_model"]].iloc[0]

        st.subheader(f"Best model: {best['Model']}")
        cols = st.columns(5)
        for col, metric in zip(cols, ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]):
            col.metric(metric, f"{best[metric]:.3f}")

        st.subheader("Model Comparison")
        st.dataframe(results.style.format({m: "{:.4f}" for m in results.columns[1:]}),
                     hide_index=True)
        st.bar_chart(results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]])

        def show_image(filename: str, caption: str) -> None:
            path = FIGURE_DIR / filename
            if path.exists():
                st.image(str(path), caption=caption)
            else:
                st.warning(f"{filename} not found. Run the training script.")

        st.subheader("Confusion Matrices")
        cm_cols = st.columns(3)
        for col, (fname, cap) in zip(cm_cols, [
                ("confusion_matrix_logistic_regression.png", "Logistic Regression"),
                ("confusion_matrix_decision_tree.png", "Decision Tree"),
                ("confusion_matrix_random_forest.png", "Random Forest")]):
            with col:
                show_image(fname, cap)

        left, right = st.columns(2)
        with left:
            st.subheader("ROC Curves")
            show_image("roc_curve.png", "ROC curves of all models")
        with right:
            st.subheader("Feature Importance (Random Forest)")
            show_image("random_forest_feature_importance.png", "Which features matter most")

        with st.expander("Exploratory Data Analysis graphs"):
            e1, e2 = st.columns(2)
            with e1:
                show_image("pass_fail_distribution.png", "Pass vs Fail distribution")
                show_image("attendance_vs_result.png", "Attendance vs Result")
                show_image("correlation_heatmap.png", "Correlation heatmap")
            with e2:
                show_image("study_hours_vs_result.png", "Study Hours vs Result")
                show_image("previous_score_vs_result.png", "Previous Score vs Result")

st.markdown("---")
st.caption("College project - Predictive Modeling Using Machine Learning")
