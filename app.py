"""
Spam Filter Lab: Streamlit front end for the Spambase classifiers.

Loads the five pre-trained pipelines from ./model (no retraining), scores an
uploaded test CSV, and shows metrics, a confusion matrix and a classification
report. The CSV must contain all 57 feature columns plus the `is_spam` target.
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
SAMPLE_CSV = os.path.join(BASE_DIR, "test_data.csv")
TARGET = "is_spam"
CLASS_NAMES = ["Not spam", "Spam"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


# ---------------------------------------------------------------------------
# Core logic (kept free of UI calls so it is easy to test)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(model_name):
    """Load a saved pipeline (scaler + classifier) from disk."""
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[model_name]))


def split_features_target(df, model):
    """Validate the uploaded frame against the model and return (X, y).

    Raises ValueError with a readable message if columns are missing or the
    target is not 0/1.
    """
    if TARGET not in df.columns:
        raise ValueError(f"The CSV needs a target column named '{TARGET}' (1 = spam, 0 = not spam).")

    required = list(model.feature_names_in_)
    missing = [c for c in required if c not in df.columns]
    if missing:
        shown = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        raise ValueError(f"{len(missing)} required feature column(s) are missing: {shown}")

    y = df[TARGET]
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"'{TARGET}' must contain only 0 and 1.")
    if y.nunique() < 2:
        raise ValueError("The target column has only one class, so AUC and MCC cannot be computed.")

    X = df[required]  # also puts the columns in the order the model was trained on
    if X.isna().any().any():
        raise ValueError("The feature columns contain missing values.")
    return X, y.astype(int)


def score_model(model, X, y):
    """Return metrics dict, hard predictions and spam probabilities."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred, y_prob


def score_all_models(df):
    """Metrics table and ROC points for every model on the uploaded data."""
    rows, roc_points = [], {}
    for name in MODEL_FILES:
        model = load_model(name)
        X, y = split_features_target(df, model)
        metrics, _, y_prob = score_model(model, X, y)
        rows.append({"Model": name, **metrics})
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_points[name] = (fpr, tpr, metrics["AUC"])
    return pd.DataFrame(rows), roc_points


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def confusion_figure(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def roc_figure(roc_points):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for name, (fpr, tpr, auc) in roc_points.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Spam Filter Lab", page_icon="📬", layout="wide")
    st.title("Spam Classifier")
    st.write(
        "Compare five classifiers trained on the UCI **Spambase** dataset "
        "(57 text-statistics features per email). Upload the test CSV, pick a model, "
        "and see how it performs. Models are loaded from disk and are **not** retrained."
    )

    # ---- Sidebar: data + model choice ----
    with st.sidebar:
        st.header("1. Test data")
        uploaded = st.file_uploader("Upload test CSV", type="csv",
                                    help=f"Needs the 57 feature columns plus '{TARGET}'.")
        st.header("2. Model")
        model_name = st.selectbox("Choose a classifier", list(MODEL_FILES))
        st.caption("Tip: upload the repo's `test_data.csv`.")
        if os.path.exists(SAMPLE_CSV):
            with open(SAMPLE_CSV, "rb") as f:
                st.download_button("Download test_data.csv", f.read(),
                                   file_name="test_data.csv", mime="text/csv")

    if uploaded is None:
        st.info("Upload a test CSV in the sidebar to see the results.")
        return

    df = pd.read_csv(uploaded)
    st.caption(f"Scoring {len(df):,} emails from the uploaded file: {uploaded.name}.")

    model = load_model(model_name)
    try:
        X, y = split_features_target(df, model)
    except ValueError as err:
        st.error(str(err))
        return

    metrics, y_pred, y_prob = score_model(model, X, y)

    tab_model, tab_compare, tab_data = st.tabs(
        [f"{model_name}", "Compare all models", "Data and errors"]
    )

    # ---- Tab 1: selected model ----
    with tab_model:
        st.subheader("Evaluation metrics")
        cols = st.columns(6)
        for col, name in zip(cols, METRIC_NAMES):
            col.metric(name, f"{metrics[name]:.4f}")

        left, right = st.columns([1, 1.3])
        with left:
            st.subheader("Confusion matrix")
            st.pyplot(confusion_figure(y, y_pred))
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            st.caption(
                f"{fp} legitimate email(s) wrongly flagged as spam (false positives) "
                f"and {fn} spam email(s) that slipped through (false negatives)."
            )
        with right:
            st.subheader("Classification report")
            report = classification_report(y, y_pred, target_names=CLASS_NAMES,
                                           output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).T.round(3))

    # ---- Tab 2: all models side by side ----
    with tab_compare:
        table, roc_points = score_all_models(df)
        st.subheader("All five models on this data")
        st.dataframe(table.set_index("Model").round(4))

        best = table.set_index("Model")[METRIC_NAMES].idxmax()
        st.write("**Best model per metric:** " +
                 " | ".join(f"{m}: {best[m]}" for m in METRIC_NAMES))

        chart_col, roc_col = st.columns(2)
        with chart_col:
            metric = st.selectbox("Metric to chart", METRIC_NAMES, index=4)
            st.bar_chart(table.set_index("Model")[metric])
        with roc_col:
            st.pyplot(roc_figure(roc_points))

    # ---- Tab 3: data preview and misclassified emails ----
    with tab_data:
        st.subheader("Data preview")
        st.dataframe(df.head(20))
        st.write(f"Spam share in this file: **{y.mean():.1%}**")

        wrong = df[(y_pred != y.values)].copy()
        wrong["predicted"] = y_pred[y_pred != y.values]
        st.subheader(f"Rows {model_name} got wrong ({len(wrong)})")
        if wrong.empty:
            st.success("No misclassified rows.")
        else:
            st.dataframe(wrong.head(50))


if __name__ == "__main__":
    main()