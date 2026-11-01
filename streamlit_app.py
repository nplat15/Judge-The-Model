"""Streamlit app for the Judge the Model classification activity.

The app reads precomputed results and uses m11.py for all evaluation metrics.
It never trains models in the viewer's session.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import m11


knn_drag_map = components.declare_component(
    "knn_drag_map",
    path=str(Path(__file__).parent / "components" / "knn_drag_map"),
)


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
ILLUSTRATIVE_PATIENTS = [
    {"id": "P1", "feature_1": 1.0, "feature_2": 13.8, "actual": 1},
    {"id": "P2", "feature_1": 2.0, "feature_2": 12.6, "actual": 1},
    {"id": "P3", "feature_1": 3.0, "feature_2": 18.4, "actual": 1},
    {"id": "P4", "feature_1": 4.0, "feature_2": 15.2, "actual": 1},
    {"id": "P5", "feature_1": 5.0, "feature_2": 23.0, "actual": 1},
    {"id": "P6", "feature_1": 6.0, "feature_2": 17.8, "actual": 1},
    {"id": "P7", "feature_1": 1.5, "feature_2": 8.7, "actual": 0},
    {"id": "P8", "feature_1": 2.5, "feature_2": 15.5, "actual": 0},
    {"id": "P9", "feature_1": 3.5, "feature_2": 11.3, "actual": 0},
    {"id": "P10", "feature_1": 4.5, "feature_2": 23.1, "actual": 0},
    {"id": "P11", "feature_1": 5.5, "feature_2": 13.9, "actual": 0},
    {"id": "P12", "feature_1": 6.5, "feature_2": 27.7, "actual": 0},
]
ILLUSTRATION_DEFAULT_SLOPE = 1.8
ILLUSTRATION_DEFAULT_INTERCEPT = 10.0

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


def sigmoid(score: float) -> float:
    clipped = max(-60.0, min(60.0, score))
    return 1 / (1 + math.exp(-clipped))


def logistic_curve_svg(patients: list[dict]) -> str:
    width, height = 520, 310
    left, right, top, bottom = 56, 18, 25, 52
    plot_width, plot_height = width - left - right, height - top - bottom
    scores = [patient["score"] for patient in patients]
    score_min = min(-6.0, min(scores) - 1.0)
    score_max = max(6.0, max(scores) + 1.0)

    def x_position(score: float) -> float:
        return left + (score - score_min) / (score_max - score_min) * plot_width

    def y_position(probability: float) -> float:
        return top + (1 - probability) * plot_height

    samples = [score_min + i * (score_max - score_min) / 120 for i in range(121)]
    curve_points = " ".join(
        f"{x_position(score):.1f},{y_position(sigmoid(score)):.1f}" for score in samples
    )
    grid = []
    for probability in (0.0, 0.5, 1.0):
        y = y_position(probability)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" '
            f'stroke="#64748b" stroke-opacity=".35" stroke-dasharray="4 5" />'
        )
        grid.append(
            f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" '
            f'fill="#aebccc" font-size="12">{probability:g}</text>'
        )
    patient_circles = []
    for patient in patients:
        score = patient["score"]
        probability = patient["probability"]
        actual = patient["actual"]
        correct = patient["correct"]
        fill = "#58a6f5" if actual else "#f3b562"
        stroke = "#f8fafc" if correct else "#ff5c67"
        title = (
            f'{patient["id"]}: actual {"had outcome" if actual else "did not have outcome"}, '
            f'score {score:.2f}, probability {probability:.2f}, '
            f'{"correct side" if correct else "wrong side"}'
        )
        patient_circles.append(
            f'<circle cx="{x_position(score):.1f}" cy="{y_position(probability):.1f}" '
            f'r="5.5" fill="{fill}" stroke="{stroke}" stroke-width="3">'
            f'<title>{title}</title></circle>'
        )
    zero_x = x_position(0)
    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="Sigmoid curve with twelve illustrative patients">
      <rect width="100%" height="100%" rx="12" fill="rgba(127, 139, 154, .08)" />
      {''.join(grid)}
      <line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{height-bottom}" stroke="#aebccc" stroke-opacity=".45" stroke-dasharray="4 5" />
      <polyline points="{curve_points}" fill="none" stroke="#b39df5" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" />
      {''.join(patient_circles)}
      <text x="{left + plot_width/2:.1f}" y="{height-12}" text-anchor="middle" fill="#d9e2ec" font-size="13">Model score z</text>
      <text x="17" y="{top + plot_height/2:.1f}" transform="rotate(-90 17 {top + plot_height/2:.1f})" text-anchor="middle" fill="#d9e2ec" font-size="13">Probability</text>
      <text x="{min(width-118, zero_x+7):.1f}" y="{y_position(.5)-8:.1f}" fill="#cbd5e1" font-size="11">p = 0.5</text>
    </svg>
    """


def decision_boundary_svg(slope: float, intercept: float, patients: list[dict]) -> str:
    width, height = 520, 310
    left, right, top, bottom = 58, 18, 25, 52
    plot_width, plot_height = width - left - right, height - top - bottom
    x_min, x_max = 0.0, 7.0
    line_ends = [slope * x_min + intercept, slope * x_max + intercept]
    feature_2_values = [patient["feature_2"] for patient in patients]
    y_min = min(feature_2_values + line_ends) - 2.0
    y_max = max(feature_2_values + line_ends) + 2.0

    def x_position(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    grid = []
    for index in range(5):
        value = y_min + index * (y_max - y_min) / 4
        y = y_position(value)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" '
            f'stroke="#64748b" stroke-opacity=".3" />'
        )
        grid.append(
            f'<text x="{left-9}" y="{y+4:.1f}" text-anchor="end" '
            f'fill="#aebccc" font-size="11">{value:.1f}</text>'
        )
    circles = []
    for patient in patients:
        actual = patient["actual"]
        correct = patient["correct"]
        fill = "#58a6f5" if actual else "#f3b562"
        stroke = "#f8fafc" if correct else "#ff5c67"
        title = (
            f'{patient["id"]}: Feature 1 {patient["feature_1"]:.1f}, '
            f'Feature 2 {patient["feature_2"]:.1f}; '
            f'{"had outcome" if actual else "did not have outcome"}; '
            f'{"correct side" if correct else "wrong side"}'
        )
        circles.append(
            f'<circle cx="{x_position(patient["feature_1"]):.1f}" '
            f'cy="{y_position(patient["feature_2"]):.1f}" r="6.5" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="3">'
            f'<title>{title}</title></circle>'
        )
    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="Decision boundary with twelve illustrative patients">
      <rect width="100%" height="100%" rx="12" fill="rgba(127, 139, 154, .08)" />
      {''.join(grid)}
      <line x1="{x_position(x_min):.1f}" y1="{y_position(line_ends[0]):.1f}" x2="{x_position(x_max):.1f}" y2="{y_position(line_ends[1]):.1f}" stroke="#b39df5" stroke-width="3.5" />
      {''.join(circles)}
      <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#aebccc" stroke-opacity=".65" />
      <text x="{left + plot_width/2:.1f}" y="{height-12}" text-anchor="middle" fill="#d9e2ec" font-size="13">Feature 1</text>
      <text x="18" y="{top + plot_height/2:.1f}" transform="rotate(-90 18 {top + plot_height/2:.1f})" text-anchor="middle" fill="#d9e2ec" font-size="13">Feature 2</text>
    </svg>
    """


KNN_TRAINING_POINTS = [
    (14, .48, 0), (20, .72, 0), (22, 1.40, 0), (27, 1.62, 0), (29, 1.38, 0),
    (35, .92, 0), (37, 1.32, 0), (38, .22, 0), (42, .62, 0), (45, .84, 0),
    (48, 1.22, 0), (51, 1.18, 0), (52, .60, 0), (55, 1.04, 0), (57, .98, 0),
    (60, .68, 0), (64, .76, 0), (65, 1.86, 1), (68, .32, 0), (72, 1.02, 0),
    (75, 1.52, 1), (80, .74, 0), (83, 1.68, 1), (86, 1.72, 1), (88, 1.24, 0),
    (93, 1.18, 0), (97, .76, 0), (102, .46, 0), (109, .72, 0), (113, 1.68, 1),
    (120, .24, 0), (123, .88, 0), (125, 1.08, 0), (128, 1.30, 1), (130, .72, 1),
    (134, .74, 1), (137, 1.62, 1), (140, .78, 1), (142, 1.48, 1), (145, 1.82, 1),
    (148, 1.44, 1), (151, 1.38, 1), (155, 1.42, 1), (180, 1.20, 1), (186, .32, 1),
    (188, .14, 0), (40, .18, 0), (30, 1.36, 0), (68, 1.88, 1), (118, .82, 0),
]


def knn_prediction_region_svg(
    points: list[tuple[float, float, int]],
    query_x: float,
    query_y: float,
    k: int,
    scale_features: bool,
) -> str:
    width, height = 920, 640
    left, right, top, bottom = 92, 24, 180, 70
    plot_width, plot_height = width - left - right, height - top - bottom
    x_min, x_max, y_min, y_max = 0.0, 200.0, 0.0, 2.0

    def px(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_width

    def py(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_height

    def distance(x: float, y: float, point: tuple[float, float, int]) -> float:
        dx, dy = x - point[0], y - point[1]
        if scale_features:
            dx, dy = dx / 200.0, dy / 2.0
        return dx * dx + dy * dy

    def vote(x: float, y: float) -> float:
        nearest = sorted(points, key=lambda point: distance(x, y, point))[:k]
        return sum(point[2] for point in nearest) / k

    # Shade the feature space according to the local k-neighbour vote.
    cols, rows = 64, 36
    cells = []
    cell_w, cell_h = plot_width / cols + .4, plot_height / rows + .4
    for row in range(rows):
        y = y_max - (row + .5) / rows * (y_max - y_min)
        for col in range(cols):
            x = x_min + (col + .5) / cols * (x_max - x_min)
            positive_region = vote(x, y) >= .5
            color = "#f08c63" if positive_region else "#78aef0"
            opacity = ".19" if positive_region else ".20"
            cells.append(
                f'<rect x="{left + col*plot_width/cols:.1f}" y="{top + row*plot_height/rows:.1f}" '
                f'width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{color}" fill-opacity="{opacity}" />'
            )

    nearest = sorted(points, key=lambda point: distance(query_x, query_y, point))[:k]
    predicted_positive = sum(point[2] for point in nearest) / k >= .5
    lines = []
    for point in nearest:
        lines.append(
            f'<line x1="{px(query_x):.1f}" y1="{py(query_y):.1f}" x2="{px(point[0]):.1f}" y2="{py(point[1]):.1f}" '
            'stroke="#c9d4de" stroke-width="2" stroke-dasharray="6 6" opacity=".8" />'
        )
    marks = []
    for x, y, label in points:
        color = "#f08c63" if label else "#78aef0"
        fill = color if label else "#1d2328"
        marks.append(
            f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="7" fill="{fill}" stroke="{color}" stroke-width="3" />'
        )
    rings = [
        f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="12" fill="none" stroke="#f1f5f9" stroke-width="3" stroke-dasharray="4 3" />'
        for x, y, _ in nearest
    ]
    query = (
        f'<path id="knn-query" d="M {px(query_x):.1f} {py(query_y)-15:.1f} L {px(query_x)+15:.1f} {py(query_y):.1f} '
        f'L {px(query_x):.1f} {py(query_y)+15:.1f} L {px(query_x)-15:.1f} {py(query_y):.1f} Z" '
        'fill="#1d2328" stroke="#f1f5f9" stroke-width="4" />'
    )
    region_color = "#f08c63" if predicted_positive else "#78aef0"
    heading_text = f"kNN predicts {'positive' if predicted_positive else 'negative'} at the diamond"
    scale_text = "Scaled distance: each feature divided by its full range" if scale_features else "Raw distance: original feature units"
    return f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="Interactive k-nearest-neighbours map with prediction regions">
      <rect width="100%" height="100%" rx="12" fill="#1d2328" />
      <text x="30" y="32" fill="#e7edf3" font-size="18">k = {k} nearest neighbours · {scale_text}</text>
      <circle cx="42" cy="68" r="7" fill="#f08c63" /><text x="60" y="74" fill="#b7c7d9" font-size="16">Training patient: had the outcome</text>
      <circle cx="438" cy="68" r="7" fill="#1d2328" stroke="#78aef0" stroke-width="3" /><text x="456" y="74" fill="#b7c7d9" font-size="16">Training patient: didn't</text>
      <path d="M 790 58 L 800 68 L 790 78 L 780 68 Z" fill="#1d2328" stroke="#f1f5f9" stroke-width="3" /><text x="812" y="74" fill="#b7c7d9" font-size="16">New patient</text>
      <circle cx="42" cy="112" r="11" fill="none" stroke="#f1f5f9" stroke-width="3" stroke-dasharray="4 3" /><text x="64" y="118" fill="#b7c7d9" font-size="16">One of the k nearest (outlined; dashed line)</text>
      <rect x="490" y="98" width="22" height="22" fill="#f08c63" fill-opacity=".3" /><text x="522" y="116" fill="#b7c7d9" font-size="16">Predict positive (≥ 0.5)</text>
      <rect x="730" y="98" width="22" height="22" fill="#78aef0" fill-opacity=".3" /><text x="762" y="116" fill="#b7c7d9" font-size="16">Predict negative</text>
      {''.join(cells)}
      <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#52606d" stroke-width="1.5" />
      {''.join(lines)}{''.join(marks)}{''.join(rings)}{query}
      <text x="{left + plot_width/2}" y="{height-18}" text-anchor="middle" fill="#b7c7d9" font-size="17">Feature A (original units, 0 to 200)</text>
      <text x="25" y="{top + plot_height/2}" transform="rotate(-90 25 {top + plot_height/2})" text-anchor="middle" fill="#b7c7d9" font-size="17">Feature B (original units, 0 to 2)</text>
      <text x="{left+15}" y="{top+25}" fill="{region_color}" font-size="14">{heading_text}</text>
    </svg>'''


def logistic_illustration() -> None:
    st.subheader("How logistic regression turns a score into a probability")
    st.markdown(
        "A weighted sum of features plus an intercept gives each patient a score. "
        "The sigmoid curve maps that score to a probability between 0 and 1."
    )
    st.latex(r"p = \frac{1}{1 + e^{-z}}, \qquad z = \beta_0 + \beta_1x_1 + \cdots + \beta_kx_k")
    st.caption(
        "The x values are patient features. The β weights are learned from training patients; "
        "β₀ is the intercept. This demonstration uses two made-up features and lets you place the line by hand."
    )

    if st.button("Reset the line", key="logistic_reset"):
        st.session_state["illustration_slope"] = ILLUSTRATION_DEFAULT_SLOPE
        st.session_state["illustration_intercept"] = ILLUSTRATION_DEFAULT_INTERCEPT
    slope_col, intercept_col = st.columns(2)
    with slope_col:
        slope = st.slider(
            "Slope of the boundary",
            min_value=-1.0,
            max_value=4.0,
            value=ILLUSTRATION_DEFAULT_SLOPE,
            step=0.1,
            key="illustration_slope",
        )
    with intercept_col:
        intercept = st.slider(
            "Intercept (Feature 2 at Feature 1 = 0)",
            min_value=-10.0,
            max_value=30.0,
            value=ILLUSTRATION_DEFAULT_INTERCEPT,
            step=0.5,
            key="illustration_intercept",
        )

    patients = []
    for patient in ILLUSTRATIVE_PATIENTS:
        score = patient["feature_2"] - (slope * patient["feature_1"] + intercept)
        probability = sigmoid(score)
        predicted = int(probability >= 0.5)
        patients.append(
            {
                **patient,
                "score": score,
                "probability": probability,
                "predicted": predicted,
                "correct": predicted == patient["actual"],
            }
        )
    correct_count = sum(patient["correct"] for patient in patients)
    wrong_count = len(patients) - correct_count

    st.markdown(
        f"**Score** = Feature 2 − ({slope:.1f} × Feature 1 + {intercept:.1f}). "
        "Above the line, the score is positive and probability is over 0.5. "
        "Below the line, the probability is under 0.5."
    )
    curve_col, boundary_col = st.columns(2)
    with curve_col:
        st.markdown("**S-shaped curve · score to probability**")
        st.markdown(logistic_curve_svg(patients), unsafe_allow_html=True)
        st.caption("The 0.5 line is the threshold; each dot is one made-up patient at their score.")
    with boundary_col:
        st.markdown("**Decision boundary · two features**")
        st.markdown(decision_boundary_svg(slope, intercept, patients), unsafe_allow_html=True)
        st.caption("The line marks probability 0.5. The red outline marks a patient on the wrong side.")

    positive_color, negative_color, wrong_color = "#58a6f5", "#f3b562", "#ff5c67"
    st.markdown(
        f'<div style="display:flex;gap:1.2rem;flex-wrap:wrap;margin:.2rem 0 .8rem">'
        f'<span style="color:{positive_color}">● Had the outcome</span>'
        f'<span style="color:{negative_color}">● Did not have the outcome</span>'
        f'<span style="color:{wrong_color}">○ On the wrong side</span></div>',
        unsafe_allow_html=True,
    )
    correct_metric, wrong_metric = st.columns(2)
    correct_metric.metric("On the correct side", f"{correct_count} of 12 patients")
    wrong_metric.metric("On the wrong side", f"{wrong_count} of 12 patients")
    st.caption(
        "Illustrative only: all 12 patients and feature values are made up. "
        "The activity model learns its weights from the training data instead of placing this line by hand."
    )
    st.markdown(
        "**Why the boundary stays straight:** with two features, all points with probability 0.5 lie on a straight line. "
        "With more features, that boundary becomes a flat surface in a higher-dimensional feature space. "
        "Logistic regression can miss curved or combined effects unless those patterns are added as features."
    )


def forest_tree_svg(tree_index: int, age: int, bmi: int, bp: int) -> tuple[str, float]:
    """Draw one small teaching tree and highlight the made-up patient's path."""
    tree_specs = [
        {
            "title": "Tree 1",
            "nodes": [
                {"id": "r", "x": 180, "y": 45, "text": "Age ≥ 55?", "feature": "age", "cutoff": 55, "yes": "n1", "no": "l0"},
                {"id": "n1", "x": 220, "y": 135, "text": "BP ≥ 140?", "feature": "bp", "cutoff": 140, "yes": "l2", "no": "l1"},
                {"id": "l0", "x": 55, "y": 228, "probability": .05},
                {"id": "l1", "x": 180, "y": 228, "probability": .20},
                {"id": "l2", "x": 320, "y": 228, "probability": .55},
            ],
            "edges": [("r", "n1", "yes"), ("r", "l0", "no"), ("n1", "l2", "yes"), ("n1", "l1", "no")],
        },
        {
            "title": "Tree 2",
            "nodes": [
                {"id": "r", "x": 180, "y": 45, "text": "BMI ≥ 30?", "feature": "bmi", "cutoff": 30, "yes": "n2", "no": "n1"},
                {"id": "n1", "x": 75, "y": 135, "text": "Age ≥ 45?", "feature": "age", "cutoff": 45, "yes": "l1", "no": "l0"},
                {"id": "n2", "x": 285, "y": 135, "text": "Age ≥ 70?", "feature": "age", "cutoff": 70, "yes": "l3", "no": "l2"},
                {"id": "l0", "x": 25, "y": 228, "probability": .05},
                {"id": "l1", "x": 115, "y": 228, "probability": .15},
                {"id": "l2", "x": 245, "y": 228, "probability": .40},
                {"id": "l3", "x": 335, "y": 228, "probability": .65},
            ],
            "edges": [("r", "n2", "yes"), ("r", "n1", "no"), ("n1", "l1", "yes"), ("n1", "l0", "no"), ("n2", "l3", "yes"), ("n2", "l2", "no")],
        },
        {
            "title": "Tree 3",
            "nodes": [
                {"id": "r", "x": 180, "y": 45, "text": "BMI ≥ 35?", "feature": "bmi", "cutoff": 35, "yes": "l2", "no": "n1"},
                {"id": "n1", "x": 130, "y": 135, "text": "BP ≥ 120?", "feature": "bp", "cutoff": 120, "yes": "l1", "no": "l0"},
                {"id": "l0", "x": 55, "y": 228, "probability": .10},
                {"id": "l1", "x": 195, "y": 228, "probability": .30},
                {"id": "l2", "x": 325, "y": 228, "probability": .70},
            ],
            "edges": [("r", "l2", "yes"), ("r", "n1", "no"), ("n1", "l1", "yes"), ("n1", "l0", "no")],
        },
    ]
    spec = tree_specs[tree_index]
    nodes = {node["id"]: node for node in spec["nodes"]}
    features = {"age": age, "bmi": bmi, "bp": bp}
    active_nodes = set()
    active_edges = set()
    current = "r"
    while "probability" not in nodes[current]:
        node = nodes[current]
        active_nodes.add(current)
        branch = "yes" if features[node["feature"]] >= node["cutoff"] else "no"
        next_node = node[branch]
        active_edges.add((current, next_node))
        current = next_node
    active_nodes.add(current)
    probability = nodes[current]["probability"]

    edge_svg = []
    for parent_id, child_id, label in spec["edges"]:
        parent, child = nodes[parent_id], nodes[child_id]
        active = (parent_id, child_id) in active_edges
        color = "#78c9c3" if active else "#52606a"
        stroke_width = 4 if active else 2
        mid_x = (parent["x"] + child["x"]) / 2
        mid_y = (parent["y"] + child["y"]) / 2 - 7
        edge_svg.append(
            f'<line x1="{parent["x"]}" y1="{parent["y"]+22}" x2="{child["x"]}" y2="{child["y"]-20}" stroke="{color}" stroke-width="{stroke_width}" />'
            f'<text x="{mid_x}" y="{mid_y}" text-anchor="middle" fill="#b8c8d4" font-size="13">{label}</text>'
        )
    node_svg = []
    for node in spec["nodes"]:
        active = node["id"] in active_nodes
        if "probability" in node:
            stroke = "#78c9c3" if active else "#3a454d"
            weight = "700" if active else "400"
            node_svg.append(
                f'<rect x="{node["x"]-33}" y="{node["y"]-18}" width="66" height="36" rx="8" fill="#20272c" stroke="{stroke}" stroke-width="{3 if active else 2}" />'
                f'<text x="{node["x"]}" y="{node["y"]+5}" text-anchor="middle" fill="#eef4f8" font-size="15" font-weight="{weight}">{node["probability"]:.2f}</text>'
            )
        else:
            stroke = "#78c9c3" if active else "#3a454d"
            node_svg.append(
                f'<rect x="{node["x"]-62}" y="{node["y"]-22}" width="124" height="44" rx="9" fill="#20272c" stroke="{stroke}" stroke-width="{4 if active else 2}" />'
                f'<text x="{node["x"]}" y="{node["y"]+5}" text-anchor="middle" fill="#eef4f8" font-size="14" font-weight="600">{node["text"]}</text>'
            )
    svg = f'''<svg viewBox="0 0 360 280" role="img" aria-label="{spec["title"]} decision tree; highlighted path probability {probability:.2f}">
      <path d="M 180 10 L 190 20 L 180 30 L 170 20 Z" fill="#20272c" stroke="#e7edf3" stroke-width="3" />
      {''.join(edge_svg)}{''.join(node_svg)}
      <text x="180" y="272" text-anchor="middle" fill="#e7edf3" font-size="16" font-weight="700">{spec["title"]}: {probability:.2f}</text>
    </svg>'''
    return svg, probability


def boosting_waterfall_svg(start_score: float, corrections: list[float], rounds: int) -> str:
    """Draw sequential made-up score corrections as a waterfall chart."""
    active = corrections[:rounds]
    running = [start_score]
    for correction in active:
        running.append(running[-1] + correction)
    final_score = running[-1]
    width = max(620, 165 * (len(active) + 2))
    left, right, top, bottom = 82, width - 35, 28, 300
    low, high = -2.0, 0.25

    def y(value: float) -> float:
        return top + (high - value) / (high - low) * (bottom - top)

    zero_y = y(0)
    step = (right - left) / (len(active) + 1)
    bar_width = min(72, step * 0.55)
    bars, labels, connectors, axis = [], [], [], []

    def whole_bar(x: float, value: float, label: str) -> None:
        bars.append(f'<rect x="{x-bar_width/2:.1f}" y="{min(zero_y,y(value)):.1f}" width="{bar_width:.1f}" height="{abs(zero_y-y(value)):.1f}" fill="#78c9c3" />')
        labels.append(f'<text x="{x:.1f}" y="{y(value)+22:.1f}" class="value">{value:.2f}</text>')
        labels.append(f'<text x="{x:.1f}" y="{bottom+30}" class="category">{label}</text>')

    whole_bar(left, start_score, "Start")
    for index, correction in enumerate(active):
        previous, current = running[index], running[index + 1]
        x = left + step * (index + 1)
        y_previous, y_current = y(previous), y(current)
        color = "#f08c68" if correction >= 0 else "#78aee8"
        bars.append(f'<rect x="{x-bar_width/2:.1f}" y="{min(y_previous,y_current):.1f}" width="{bar_width:.1f}" height="{max(abs(y_current-y_previous),2):.1f}" fill="{color}" />')
        connectors.append(f'<line x1="{x-bar_width/2:.1f}" y1="{y_previous:.1f}" x2="{x+step-bar_width/2:.1f}" y2="{y_previous:.1f}" stroke="#aab8c2" stroke-dasharray="4 4" />')
        label_y = min(y_previous, y_current) - 9 if correction >= 0 else max(y_previous, y_current) + 20
        labels.append(f'<text x="{x:.1f}" y="{label_y:.1f}" class="value">{correction:+.2f}</text>')
        labels.append(f'<text x="{x:.1f}" y="{bottom+30}" class="category">Tree {index+1}</text>')
    whole_bar(left + step * (len(active) + 1), final_score, "Final")
    for tick in (0, -0.5, -1.0, -1.5, -2.0):
        tick_y = y(tick)
        axis.append(f'<line x1="{left-32}" y1="{tick_y:.1f}" x2="{right+8}" y2="{tick_y:.1f}" stroke="#344149" />')
        axis.append(f'<text x="{left-43}" y="{tick_y+5:.1f}" text-anchor="end" class="tick">{tick:g}</text>')
    return f'''<svg viewBox="0 0 {width} 365" role="img" aria-label="Gradient boosting waterfall: start score {start_score:.2f}, {rounds} tree corrections, final score {final_score:.2f}">
      <style>text{{font-family:Arial,sans-serif}}.tick{{fill:#aebdca;font-size:16px}}.category{{fill:#aebdca;font-size:16px;text-anchor:middle}}.value{{fill:#f0f4f7;font-size:17px;font-weight:700;text-anchor:middle}}</style>
      {''.join(axis)}<line x1="{left-32}" y1="{zero_y:.1f}" x2="{right+8}" y2="{zero_y:.1f}" stroke="#52606a" stroke-width="2" />
      <text x="20" y="165" transform="rotate(-90 20 165)" text-anchor="middle" class="tick">Model score (log-odds)</text>
      {''.join(connectors)}{''.join(bars)}{''.join(labels)}
    </svg>'''


def model_illustration(model: str) -> None:
    """Interactive, made-up demonstrations of each model's central idea."""
    st.divider()
    if model == "baseline":
        st.subheader("How the prevalence baseline makes a prediction")
        st.write(
            "This deliberately simple reference model ignores each patient's features. "
            "It predicts the training-set prevalence as the same probability for everyone, "
            "then flags the majority class at a 0.50 threshold."
        )
        prevalence = st.slider(
            "Made-up positive prevalence", 0, 100, 20, 1, format="%d%%", key="baseline_demo_prevalence"
        )
        positives = round(prevalence)
        negatives = 100 - positives
        cols = st.columns(3)
        cols[0].metric("Probability for every patient", f"{prevalence}%")
        cols[1].metric("Class predicted for everyone at 0.50", "Positive" if prevalence >= 50 else "Negative")
        cols[2].metric("Accuracy on this made-up mix", f"{max(positives, negatives)}%")
        st.caption(
            "Changing the prevalence changes the constant probability and majority class. "
            "This is a reference point, not a patient-specific risk estimate."
        )
        st.markdown("**Made-up group of 100 patients**")
        st.dataframe(pd.DataFrame({"Outcome group": ["Had outcome", "Did not have outcome"], "Patients": [positives, negatives]}), hide_index=True, width="stretch")

    elif model == "knn":
        st.subheader("How k-nearest neighbours uses nearby patients")
        st.write(
            "Move the new patient and change k to see the local prediction region and which training "
            "patients are nearest. KNN predicts from the most common outcome among those neighbours."
        )
        # A drag result arrives in session state on the rerun it triggers. Apply
        # it before creating the sliders so both controls and the map stay in sync.
        drag_state = st.session_state.get("knn_drag_map")
        drag_position = (
            (float(drag_state["x"]), float(drag_state["y"]))
            if isinstance(drag_state, dict) and "x" in drag_state and "y" in drag_state
            else None
        )
        if drag_position is not None and drag_position != st.session_state.get("knn_last_applied_drag"):
            st.session_state["knn_demo_query_x"] = float(drag_state["x"])
            st.session_state["knn_demo_query_y"] = float(drag_state["y"])
            st.session_state["knn_last_applied_drag"] = drag_position
        a, b = st.columns(2)
        query_x = a.slider("New patient's Feature A", 0.0, 200.0, 105.0, 1.0, key="knn_demo_query_x")
        query_y = b.slider("New patient's Feature B", 0.0, 2.0, 1.45, 0.01, key="knn_demo_query_y")
        c, d = st.columns(2)
        k = c.slider("k (number of neighbours)", 1, 15, 7, 2, key="knn_demo_k")
        scale_features = d.checkbox("Scale both features first", value=True, key="knn_demo_scale")
        nearest = sorted(
            KNN_TRAINING_POINTS,
            key=lambda point: ((query_x-point[0]) / 200.0) ** 2 + ((query_y-point[1]) / 2.0) ** 2
            if scale_features else (query_x-point[0]) ** 2 + (query_y-point[1]) ** 2,
        )[:k]
        positives = sum(point[2] for point in nearest)
        probability = positives / k
        knn_drag_map(
            svg=knn_prediction_region_svg(
                KNN_TRAINING_POINTS, query_x, query_y, k, scale_features
            ),
            key="knn_drag_map",
            default=None,
            height=640,
        )
        result_col, nearest_col = st.columns([1, 2])
        result_col.metric(
            "Neighbour vote: predicted positive probability",
            f"{probability:.0%}",
            f"{positives} of {k} nearest had the outcome",
        )
        result_col.caption(
            f"At a 0.50 cutoff, this toy example predicts {'positive' if probability >= .5 else 'negative'}."
        )
        nearest_col.caption("Nearest made-up training patients")
        nearest_col.dataframe(pd.DataFrame([
            {"Feature A": point[0], "Feature B": point[1], "Outcome": "Had outcome" if point[2] else "Didn't", "Distance": round(math.sqrt(
                ((query_x-point[0]) / 200.0) ** 2 + ((query_y-point[1]) / 2.0) ** 2
                if scale_features else (query_x-point[0]) ** 2 + (query_y-point[1]) ** 2
            ), 3)} for point in nearest
        ]), hide_index=True, width="stretch")
        st.caption(
            "Illustrative only: these patients are made up. When scaling is on, both features are divided by their full ranges "
            "before distance is measured; without scaling, Feature A's larger numeric range can dominate."
        )

    elif model == "rf":
        st.subheader("How a random forest combines decision trees")
        st.write(
            "Many decision trees are fit on resamples of the training patients, and each split considers "
            "a random subset of features. The forest's probability is the average of the tree probabilities. "
            "Change this made-up patient's features to follow a different path through the trees."
        )
        st.markdown(
            "◇ **One made-up patient:** change age, BMI, and systolic blood pressure (BP). "
            "The highlighted path shows the route through each tree."
        )
        age_col, bmi_col, bp_col = st.columns(3)
        age = age_col.slider("Age", 18, 90, 62, 1, key="forest_demo_age")
        bmi = bmi_col.slider("BMI", 15, 50, 33, 1, key="forest_demo_bmi")
        bp = bp_col.slider("Systolic blood pressure (BP)", 80, 220, 128, 1, key="forest_demo_bp")
        tree_cols = st.columns(3)
        probabilities = []
        for index, column in enumerate(tree_cols):
            svg, tree_probability = forest_tree_svg(index, age, bmi, bp)
            probabilities.append(tree_probability)
            with column:
                st.markdown(svg, unsafe_allow_html=True)
        probability = sum(probabilities) / len(probabilities)
        st.markdown(
            f"**Illustrative forest probability** = average of the three trees = "
            f"({ ' + '.join(f'{p:.2f}' for p in probabilities) }) ÷ 3 = **{probability:.2f}**. "
            "The activity's forest averages 300 trees."
        )
        st.caption(
            "Made-up trees and thresholds for teaching, not a real patient's prediction. "
            "Random forests can overfit; compare Training and Test AUROC in Section 3. A single prediction is hard to explain."
        )

    elif model == "lgbm":
        st.subheader("How gradient boosting builds a prediction in stages")
        st.write(
            "Trees are added in sequence. Each new tree learns a correction intended to reduce the current model's loss "
            "(how wrong its predictions are on the training patients). The corrections add on the score scale, "
            "and the final score becomes a probability."
        )
        corrections = [0.80, -0.35, 0.45, -0.20, 0.15]
        start_probability = 0.20
        start_score = math.log(start_probability / (1 - start_probability))
        rounds = st.slider("Trees added", 0, len(corrections), 3, 1, key="boost_demo_rounds")
        score = start_score + sum(corrections[:rounds])
        probability = sigmoid(score)
        st.markdown(boosting_waterfall_svg(start_score, corrections, rounds), unsafe_allow_html=True)
        st.markdown(
            '<span style="color:#78c9c3">■</span> starting/final score&nbsp;&nbsp;&nbsp; '
            '<span style="color:#f08c68">■</span> tree raises score&nbsp;&nbsp;&nbsp; '
            '<span style="color:#78aee8">■</span> tree lowers score',
            unsafe_allow_html=True,
        )
        update_text = " ".join(f"{value:+.2f}" for value in corrections[:rounds]) or "no corrections yet"
        st.markdown(
            f"**Illustrative** One made-up patient. Start at the log-odds of a made-up starting prevalence, "
            f"0.20: ln(0.20 / 0.80) = {start_score:.2f}. Final score: {start_score:.2f} {update_text} = {score:.2f}; "
            f"probability = 1 / (1 + e^({-score:.2f})) = **{probability:.0%}**. "
            "Each correction is on the log-odds scale, not a probability, and a tree can lower a patient's prediction as well as raise it."
        )
        st.caption("Made-up corrections for teaching. Adding more trees can improve fit, but too many may overfit; compare Training and Test AUROC in Section 3.")


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
    if selected_model in {"baseline", "logreg", "knn", "rf", "lgbm"}:
        st.divider()
        if selected_model == "logreg":
            logistic_illustration()
        else:
            model_illustration(selected_model)


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
