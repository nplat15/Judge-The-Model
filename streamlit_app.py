"""Streamlit app for the Judge the Model classification activity.

The app reads precomputed results and uses m11.py for all evaluation metrics.
It never trains models in the viewer's session.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import m11


TRAIN = "Training"
TEST = "Test"
SECTIONS = [
    "1. Beyond Accuracy",
    "2. A Threshold for the Job",
    "3. Training vs. Test",
]
JOB_DETAILS = {
    "readmission": {
        "positive": "Readmitted within 30 days",
        "purpose": "Estimate the chance of readmission within 30 days at discharge.",
    },
    "diabetes": {
        "positive": "Doctor diagnosis or HbA1c at least 6.5%",
        "purpose": "Flag adults who may need follow-up testing; finding positives matters most.",
    },
}

st.set_page_config(page_title="Judge the Model", page_icon="⚖️", layout="wide")

st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
      [data-testid="stMetric"] {background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, .28); color: var(--text-color);
        padding: 0.9rem 1rem; border-radius: 0.65rem;}
      [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {color: var(--text-color);}
      .eyebrow {color:#526274; font-size:.85rem; font-weight:700; letter-spacing:.06em;
        text-transform:uppercase;}
    </style>
    """,
    unsafe_allow_html=True,
)


def load(name: str) -> dict | None:
    return m11.load(name)


def fmt_count(value: int) -> str:
    return f"{value:,}"


def metric_rows(res: dict, threshold: float) -> tuple[list[dict], dict[str, dict]]:
    rows: list[dict] = []
    matrices: dict[str, dict] = {}
    for model in m11.MODELS:
        if model not in res["models"]:
            rows.append(
                {
                    "Model": m11.MODEL_LABEL[model],
                    "Status": "Results pending",
                }
            )
            continue
        probabilities, actual = m11.preds(res, model, "test")
        counts = m11.confusion(probabilities, actual, threshold)
        metrics = m11.metrics(counts)
        matrices[model] = counts
        rows.append(
            {
                "Model": m11.MODEL_LABEL[model],
                "Status": "Ready",
                "True positives detected": fmt_count(counts["tp"]),
                "Missed positives": fmt_count(counts["fn"]),
                "False alarms": fmt_count(counts["fp"]),
                "True negatives": fmt_count(counts["tn"]),
                "Accuracy": m11.fmt_pct(metrics["accuracy"]),
                "Sensitivity (recall, TPR)": m11.fmt_pct(metrics["sensitivity"]),
                "Specificity": m11.fmt_pct(metrics["specificity"]),
                "FPR (1 − specificity)": m11.fmt_pct(metrics["fpr"]),
                "Precision": m11.fmt_pct(metrics["precision"]),
                "F1": m11.fmt_pct(metrics["f1"]),
            }
        )
    return rows, matrices


def show_confusion(counts: dict, positive_label: str) -> None:
    matrix = pd.DataFrame(
        [
            [counts["tp"], counts["fn"]],
            [counts["fp"], counts["tn"]],
        ],
        index=[f"Actual: positive · {positive_label}", "Actual: negative · no outcome"],
        columns=["Predicted: positive · flagged", "Predicted: negative · not flagged"],
    ).map(fmt_count)
    st.dataframe(matrix, width="stretch", height=135)


def recommendation_panel() -> None:
    decisions = load("decisions") or {}
    recs = decisions.get("recommendations", {})
    with st.container(border=True):
        st.subheader("Recommendations")
        if not any(recs.get(key) for key in m11.DATASETS):
            st.caption(
                "Recommendations will appear here after you record them in "
                "DECISIONS.md and regenerate the results."
            )
            return
        left, right = st.columns(2)
        for column, key in zip((left, right), m11.DATASETS):
            with column:
                st.markdown(f"**{m11.DATASET_LABEL[key]}**")
                st.write(recs.get(key) or "Recommendation not written yet.")


def overview(res: dict, key: str) -> None:
    test = res["test"]
    train = res["train"]
    st.markdown('<p class="eyebrow">The job</p>', unsafe_allow_html=True)
    st.write(JOB_DETAILS[key]["purpose"])
    st.markdown(f"**Positive class:** {JOB_DETAILS[key]['positive']}")
    a, b, c = st.columns(3)
    a.metric(f"{TRAIN} patients", fmt_count(train["n"]))
    b.metric(f"{TEST} patients", fmt_count(test["n"]))
    c.metric("Positive outcomes in test set", fmt_count(test["n_pos"]))


def beyond_accuracy(res: dict, key: str, selected_model: str) -> None:
    st.subheader("Compare every model at the default threshold")
    st.write(
        "At 0.50, a patient is flagged when the model assigns a probability of at least 50%. "
        "Accuracy alone can hide missed positives, so compare all four patient groups and the other metrics."
    )
    test_y = res["test"]["y"]
    prevalence_counts = {
        "tp": res["test"]["n_pos"],
        "fn": 0,
        "fp": 0,
        "tn": res["test"]["n"] - res["test"]["n_pos"],
    }
    prevalence = m11.metrics(prevalence_counts)["prevalence"]
    baseline_accuracy = m11.majority_accuracy(test_y)
    prevalence_col, baseline_col = st.columns(2)
    prevalence_col.metric("Test-set prevalence", m11.fmt_pct(prevalence))
    baseline_col.metric(
        "Test majority-class baseline accuracy",
        m11.fmt_pct(baseline_accuracy),
        help="Accuracy from predicting the most common test-set class for every patient.",
    )

    st.markdown("**How to read the rates**")
    rate_a, rate_b, rate_c = st.columns(3)
    rate_a.markdown("**Sensitivity (recall, TPR)**  \nOf the patients who had the outcome, the share the model flagged.")
    rate_b.markdown("**Specificity**  \nOf the patients who did not have the outcome, the share the model left unflagged.")
    rate_c.markdown("**Precision**  \nOf the patients the model flagged, the share who had the outcome.")

    rows, matrices = metric_rows(res, 0.50)
    st.caption(f"{TEST} · all five models · threshold 0.50 · counts are patients")
    st.dataframe(pd.DataFrame(rows).set_index("Model"), width="stretch", hide_index=False)

    if selected_model in matrices:
        st.subheader(f"Confusion matrix at 0.50 · {m11.MODEL_LABEL[selected_model]}")
        st.caption(f"{TEST} · rows are actual; columns are predicted. Positive class: {JOB_DETAILS[key]['positive']}.")
        show_confusion(matrices[selected_model], JOB_DETAILS[key]["positive"])

    with st.expander("View a confusion matrix for each of the five models"):
        for model in m11.MODELS:
            st.markdown(f"**{m11.MODEL_LABEL[model]}**")
            if model in matrices:
                st.caption(f"{TEST} · threshold 0.50 · rows are actual; columns are predicted")
                show_confusion(matrices[model], JOB_DETAILS[key]["positive"])
            else:
                st.info(f"{m11.MODEL_LABEL[model]} has no {TEST.lower()} results yet.")
    pending = [m11.MODEL_LABEL[m] for m in m11.MODELS if m not in matrices]
    if pending:
        st.info("Classifier results pending: " + ", ".join(pending) + ".")


def threshold_section(res: dict, key: str, model: str) -> None:
    st.subheader("Choose the cutoff for a positive flag")
    st.write(
        "The model gives each patient a probability. The threshold is a separate decision: "
        "lower cutoffs flag more people, while higher cutoffs flag fewer. The Test metrics and "
        "confusion matrices update when you move the slider."
    )
    if model not in res["models"]:
        st.warning(
            f"{m11.MODEL_LABEL[model]} has no saved results yet. Generate the model results before comparing thresholds."
        )
        return

    threshold_value = st.slider(
        "Probability threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
        format="%.2f",
        help="A patient is flagged when the model's probability is at or above this value.",
    )
    probabilities, actual = m11.preds(res, model, "test")
    reference_counts = m11.confusion(probabilities, actual, 0.50)
    selected_counts = m11.confusion(probabilities, actual, threshold_value)
    reference_metrics = m11.metrics(reference_counts)
    selected_metrics = m11.metrics(selected_counts)
    comparison = []
    paired_metrics = [
        ("Sensitivity (recall, TPR)", "sensitivity", "percent"),
        ("Specificity", "specificity", "percent"),
        ("Precision", "precision", "percent"),
        ("FPR (1 − specificity)", "fpr", "percent"),
        ("Patients flagged", "flagged", "count"),
        ("Flagged per 100 patients", "flagged_per_100", "number"),
    ]
    for label, metric_key, value_type in paired_metrics:
        reference_value = reference_metrics[metric_key]
        selected_value = selected_metrics[metric_key]
        if value_type == "percent":
            reference_display = m11.fmt_pct(reference_value)
            selected_display = m11.fmt_pct(selected_value)
        elif value_type == "count":
            reference_display = fmt_count(reference_value)
            selected_display = fmt_count(selected_value)
        else:
            reference_display = m11.fmt_num(reference_value, 1)
            selected_display = m11.fmt_num(selected_value, 1)
        comparison.append(
            {
                "Metric": label,
                "Test at 0.50": reference_display,
                f"Test at {threshold_value:.2f}": selected_display,
            }
        )
    st.caption(f"{TEST} · {m11.MODEL_LABEL[model]} · reference threshold 0.50 vs. live slider threshold")
    st.dataframe(pd.DataFrame(comparison).set_index("Metric"), width="stretch")
    st.markdown(
        "Sensitivity is the share of patients who had the outcome that were flagged. "
        "Specificity is the share of patients without the outcome who were left unflagged. "
        "Precision is the share of flagged patients who had the outcome."
    )
    reference_col, selected_col = st.columns(2)
    with reference_col:
        st.markdown("**Reference threshold · 0.50**")
        st.caption(f"{TEST} · rows are actual; columns are predicted")
        show_confusion(reference_counts, JOB_DETAILS[key]["positive"])
    with selected_col:
        st.markdown(f"**Slider threshold · {threshold_value:.2f}**")
        st.caption(f"{TEST} · rows are actual; columns are predicted")
        show_confusion(selected_counts, JOB_DETAILS[key]["positive"])


def train_vs_test(res: dict, selected_model: str) -> None:
    st.subheader("Compare scores on seen and unseen patients")
    st.write(
        f"{TRAIN} AUROC is measured on patients used to fit each model. {TEST} AUROC is measured "
        "on the held-out patients the models did not see during fitting. A large gap can suggest "
        "memorisation; test performance is the more relevant estimate for new patients."
    )
    summary = m11.auroc_summary(res)
    train_n = fmt_count(res["train"]["n"])
    test_n = fmt_count(res["test"]["n"])
    train_auroc_label = f"{TRAIN} AUROC (n={train_n})"
    test_auroc_label = f"{TEST} AUROC (n={test_n})"
    if selected_model in summary:
        selected = summary[selected_model]
        a, b, c = st.columns(3)
        a.metric(f"{train_auroc_label} · {m11.MODEL_LABEL[selected_model]}", m11.fmt_num(selected["train"]))
        b.metric(f"{test_auroc_label} · {m11.MODEL_LABEL[selected_model]}", m11.fmt_num(selected["test"]))
        c.metric(f"{TRAIN} − {TEST} AUROC gap", m11.fmt_num(selected["gap"]))
    rows = []
    chart = []
    for model in m11.MODELS:
        if model not in summary:
            rows.append({"Model": m11.MODEL_LABEL[model], "Status": "Results pending"})
            continue
        values = summary[model]
        rows.append(
            {
                "Model": m11.MODEL_LABEL[model],
                "Status": "Ready",
                train_auroc_label: m11.fmt_num(values["train"]),
                test_auroc_label: m11.fmt_num(values["test"]),
                f"{TRAIN} − {TEST} AUROC gap": m11.fmt_num(values["gap"]),
            }
        )
        if values["train"] is not None and values["test"] is not None:
            chart.append(
                {
                    "Model": m11.MODEL_LABEL[model],
                    TRAIN: values["train"],
                    TEST: values["test"],
                }
            )
    st.caption(
        f"{TRAIN} and {TEST} · each AUROC shows its patient count in the column label · "
        "AUROC describes ranking, not accuracy or a chosen threshold"
    )
    st.dataframe(pd.DataFrame(rows).set_index("Model"), width="stretch")
    if chart:
        st.bar_chart(pd.DataFrame(chart).set_index("Model"), y=[TRAIN, TEST])
    st.caption(
        "A higher training score than test score is a warning sign to examine, not proof by itself that a model is unusable."
    )
    pending = [model for model in m11.MODELS if model not in summary]
    if pending:
        st.info(
            "Classifier results pending: "
            + ", ".join(m11.MODEL_LABEL[m] for m in pending)
            + "."
        )


st.title("Judge the Model")
st.caption("Which classifier would you trust for this job, at what threshold, and with what limits?")
recommendation_panel()

with st.sidebar:
    st.header("Explore the results")
    section = st.radio("Section", SECTIONS, label_visibility="collapsed")
    dataset_key = st.selectbox("Clinical job", m11.DATASETS, format_func=m11.DATASET_LABEL.get)
    res = load(dataset_key)
    available_models = [model for model in m11.MODELS if model in (res or {}).get("models", {})]
    selected_model = st.selectbox(
        "Model",
        m11.MODELS,
        index=m11.MODELS.index(available_models[0]) if available_models else 0,
        format_func=lambda model: m11.MODEL_LABEL[model]
        + ("" if model in (res or {}).get("models", {}) else " · results pending"),
    )

if res is None:
    st.error(f"No results file is available for {m11.DATASET_LABEL[dataset_key]} yet.")
else:
    overview(res, dataset_key)
    st.divider()
    st.header(section)
    if section == SECTIONS[0]:
        beyond_accuracy(res, dataset_key, selected_model)
    elif section == SECTIONS[1]:
        threshold_section(res, dataset_key, selected_model)
    else:
        train_vs_test(res, selected_model)
