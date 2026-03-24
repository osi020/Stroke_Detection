"""
NeuroScan AI – Stroke Risk Assessment System
Professional clinical decision-support tool for stroke screening.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# ─── Constants ────────────────────────────────────────────────────────────────
DATA_PATH          = Path("data/raw_data/healthcare-dataset-stroke-data.csv")
THRESHOLD_STANDARD  = 0.72   # LogReg_Balanced  (F2-optimised, higher precision)
THRESHOLD_SENSITIVE = 0.14   # LogReg_Balanced_calibrated (F2-optimised, higher recall)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI – Stroke Risk Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- Base ---------- */
.stApp { background: #EEF2F7; }

/* ---------- Header ---------- */
.app-header {
    background: linear-gradient(135deg, #0A2342 0%, #1565C0 100%);
    border-radius: 14px;
    padding: 22px 32px;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.header-title { color: #ffffff; font-size: 26px; font-weight: 700; margin: 0; }
.header-sub   { color: #90CAF9; font-size: 13px; margin-top: 4px; }
.header-badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 6px 16px;
    color: #ffffff;
    font-size: 12px;
    font-weight: 500;
}

/* ---------- Cards ---------- */
.card {
    background: #ffffff;
    border-radius: 12px;
    padding: 22px 24px;
    margin-bottom: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    border-top: 3px solid #1565C0;
}
.card-neutral { border-top: 3px solid #90CAF9; }
.card-success { border-top: 3px solid #43A047; }
.card-warning { border-top: 3px solid #FB8C00; }
.card-danger  { border-top: 3px solid #E53935; }
.card-critical{ border-top: 3px solid #B71C1C; }

/* ---------- Section titles ---------- */
.section-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #78909C;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-divider {
    border: none;
    border-top: 1px solid #ECEFF1;
    margin: 18px 0;
}

/* ---------- Risk badge ---------- */
.risk-wrap { text-align: center; padding: 10px 0; }
.risk-badge {
    display: inline-block;
    padding: 10px 32px;
    border-radius: 30px;
    font-size: 20px;
    font-weight: 800;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.badge-low      { background: #E8F5E9; color: #1B5E20; }
.badge-moderate { background: #FFF8E1; color: #E65100; }
.badge-high     { background: #FFF3E0; color: #BF360C; }
.badge-critical { background: #FFEBEE; color: #B71C1C; }

.risk-prob {
    font-size: 42px;
    font-weight: 800;
    margin: 8px 0 2px 0;
    text-align: center;
}
.risk-prob-label {
    text-align: center;
    color: #78909C;
    font-size: 13px;
    margin-bottom: 6px;
}

/* ---------- Model decision ---------- */
.decision-positive {
    background: #FFEBEE;
    border: 2px solid #EF9A9A;
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
    color: #C62828;
    font-weight: 700;
    font-size: 15px;
}
.decision-negative {
    background: #E8F5E9;
    border: 2px solid #A5D6A7;
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
    color: #2E7D32;
    font-weight: 700;
    font-size: 15px;
}

/* ---------- Risk factor pills ---------- */
.factor-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.factor-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 13px;
    border-radius: 14px;
    font-size: 13px;
    font-weight: 500;
}
.pill-high   { background: #FFEBEE; color: #C62828; border: 1px solid #FFCDD2; }
.pill-medium { background: #FFF3E0; color: #BF360C; border: 1px solid #FFE0B2; }
.pill-low    { background: #E8F5E9; color: #2E7D32; border: 1px solid #C8E6C9; }
.pill-info   { background: #E3F2FD; color: #1565C0; border: 1px solid #BBDEFB; }

/* ---------- Metric mini-card ---------- */
.metrics-grid { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 10px; }
.mini-metric {
    flex: 1;
    min-width: 90px;
    background: #F5F7FA;
    border-radius: 10px;
    padding: 12px 10px;
    text-align: center;
}
.mini-val   { font-size: 20px; font-weight: 700; color: #0A2342; }
.mini-label { font-size: 10px; text-transform: uppercase; color: #90A4AE;
              letter-spacing: 0.8px; margin-top: 2px; }

/* ---------- Recommendation item ---------- */
.rec-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 9px 0;
    border-bottom: 1px solid #F5F5F5;
    font-size: 14px;
    color: #37474F;
}
.rec-icon { font-size: 16px; margin-top: 1px; }

/* ---------- Disclaimer ---------- */
.disclaimer {
    background: #FFF8E1;
    border-left: 4px solid #FFC107;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    font-size: 12px;
    color: #5D4037;
    margin-top: 10px;
}

/* ---------- Welcome placeholder ---------- */
.welcome-box {
    background: #ffffff;
    border-radius: 14px;
    padding: 60px 40px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    color: #90A4AE;
}
.welcome-icon { font-size: 64px; margin-bottom: 16px; }
.welcome-title { font-size: 20px; font-weight: 600; color: #546E7A; }
.welcome-sub   { font-size: 14px; margin-top: 6px; }

/* ---------- Button ---------- */
.stButton > button {
    background: linear-gradient(135deg, #1565C0 0%, #0A2342 100%);
    color: white !important;
    border: none;
    padding: 14px 0;
    border-radius: 10px;
    font-size: 16px;
    font-weight: 700;
    width: 100%;
    letter-spacing: 0.5px;
    transition: all 0.2s;
    box-shadow: 0 4px 14px rgba(21, 101, 192, 0.35);
}
.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(21, 101, 192, 0.5);
    transform: translateY(-1px);
}

/* ---------- Inputs ---------- */
.stSelectbox label, .stNumberInput label,
.stRadio label, .stCheckbox label {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #37474F !important;
}

/* ---------- Streamlit chrome ---------- */
#MainMenu, footer, .stDeployButton { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING  (cached – trains once per session)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading AI models…")
def load_models():
    df = pd.read_csv(DATA_PATH)
    df = df[df["gender"] != "Other"].copy()
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer",       SimpleImputer(strategy="median")),
        ("log_transform", FunctionTransformer(np.log1p, validate=False)),
        ("scaler",        StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], sparse_threshold=0.0)

    # ── Standard model (LogReg_Balanced, F2 threshold 0.72) ─────────────────
    standard_pipe = ImbPipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=3000, random_state=42, class_weight="balanced"
        )),
    ])
    standard_pipe.fit(X_train, y_train)

    # ── Sensitive model (calibrated, F2 threshold 0.14) ─────────────────────
    sensitive_pipe = CalibratedClassifierCV(standard_pipe, method="isotonic", cv=3)
    sensitive_pipe.fit(X_train, y_train)

    return standard_pipe, sensitive_pipe, num_cols, cat_cols


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def build_input_df(gender, age, hypertension, heart_disease, ever_married,
                   work_type, residence, glucose, bmi, smoking):
    return pd.DataFrame([{
        "gender":            gender,
        "age":               float(age),
        "hypertension":      int(hypertension),
        "heart_disease":     int(heart_disease),
        "ever_married":      ever_married,
        "work_type":         work_type,
        "Residence_type":    residence,
        "avg_glucose_level": float(glucose),
        "bmi":               float(bmi) if bmi else np.nan,
        "smoking_status":    smoking,
    }])


def get_risk_factors(age, hypertension, heart_disease, glucose, bmi, smoking):
    """Return list of (label, severity, icon) tuples based on clinical thresholds."""
    factors = []

    # Age
    if age >= 75:
        factors.append(("Age ≥ 75 (very high risk)",  "high",   "🔴"))
    elif age >= 65:
        factors.append(("Age 65-74 (elderly)",        "high",   "🔴"))
    elif age >= 45:
        factors.append(("Age 45-64 (middle-aged)",    "medium", "🟡"))
    else:
        factors.append(("Age < 45 (low age risk)",    "low",    "🟢"))

    # Hypertension
    if hypertension:
        factors.append(("Hypertension detected",      "high",   "🔴"))
    else:
        factors.append(("No hypertension",            "low",    "🟢"))

    # Heart disease
    if heart_disease:
        factors.append(("Heart disease detected",     "high",   "🔴"))
    else:
        factors.append(("No cardiac history",         "low",    "🟢"))

    # Glucose
    if glucose >= 200:
        factors.append((f"Glucose {glucose} mg/dL (diabetic range)",    "high",   "🔴"))
    elif glucose >= 140:
        factors.append((f"Glucose {glucose} mg/dL (pre-diabetic)",      "medium", "🟡"))
    elif glucose >= 100:
        factors.append((f"Glucose {glucose} mg/dL (borderline)",        "medium", "🟡"))
    else:
        factors.append((f"Glucose {glucose} mg/dL (normal)",            "low",    "🟢"))

    # BMI
    if bmi:
        if bmi >= 35:
            factors.append((f"BMI {bmi:.1f} (obese class II)",          "high",   "🔴"))
        elif bmi >= 30:
            factors.append((f"BMI {bmi:.1f} (obese)",                   "medium", "🟡"))
        elif bmi >= 25:
            factors.append((f"BMI {bmi:.1f} (overweight)",              "medium", "🟡"))
        else:
            factors.append((f"BMI {bmi:.1f} (healthy weight)",          "low",    "🟢"))
    else:
        factors.append(("BMI unknown",                                    "info",   "ℹ️"))

    # Smoking
    smoking_map = {
        "smokes":          ("Active smoker",         "high",   "🔴"),
        "formerly smoked": ("Former smoker",         "medium", "🟡"),
        "never smoked":    ("Non-smoker",            "low",    "🟢"),
        "Unknown":         ("Smoking status unknown","info",   "ℹ️"),
    }
    factors.append(smoking_map.get(smoking, ("Smoking unknown", "info", "ℹ️")))

    return factors


def risk_level(prob, threshold):
    if prob < threshold * 0.5:
        return "Low",      "badge-low",      "card-success", "#43A047"
    elif prob < threshold:
        return "Moderate", "badge-moderate", "card-warning", "#FB8C00"
    elif prob < threshold * 1.5:
        return "High",     "badge-high",     "card-danger",  "#E53935"
    else:
        return "Critical", "badge-critical", "card-critical","#B71C1C"


def gauge_chart(prob, threshold, color):
    pct = prob * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "#90A4AE", "tickfont": {"size": 11}},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  20],  "color": "#E8F5E9"},
                {"range": [20, 40],  "color": "#FFF9C4"},
                {"range": [40, 60],  "color": "#FFE0B2"},
                {"range": [60, 100], "color": "#FFCDD2"},
            ],
            "threshold": {
                "line":      {"color": "#0A2342", "width": 3},
                "thickness": 0.85,
                "value":     threshold * 100,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"},
    )
    return fig


def get_recommendations(level, factors, mode):
    recs = []
    high_factors = [f for f, sev, _ in factors if sev == "high"]
    has_htn      = any("Hypertension"   in f for f in high_factors)
    has_cardiac  = any("Heart disease"  in f for f in high_factors)
    has_glucose  = any("Glucose"        in f for f, s, _ in factors if s in ("high","medium"))
    has_smoking  = any("smoker" in f.lower() and "former" not in f.lower()
                       for f, s, _ in factors if s == "high")

    if level in ("High", "Critical"):
        recs += [
            ("🏥", "Urgent neurology or stroke specialist referral recommended."),
            ("⚡", "Consider immediate brain imaging (CT/MRI) if symptomatic."),
        ]
    elif level == "Moderate":
        recs += [
            ("📋", "Schedule follow-up appointment for comprehensive cardiovascular review."),
            ("🔬", "Consider ordering full lipid panel and HbA1c test."),
        ]
    else:
        recs += [
            ("✅", "Current risk profile is low. Continue routine check-ups."),
        ]

    if has_htn:
        recs.append(("💊", "Review and optimise antihypertensive therapy. Target BP < 130/80 mmHg."))
    if has_cardiac:
        recs.append(("❤️", "Cardiac evaluation and anticoagulation assessment recommended."))
    if has_glucose:
        recs.append(("🩸", "Glucose management review. Consider HbA1c and endocrinology consult."))
    if has_smoking:
        recs.append(("🚭", "Smoking cessation counselling and pharmacotherapy strongly advised."))

    recs.append(("🏃", "Encourage Mediterranean diet and ≥150 min/week moderate physical activity."))

    if mode == "Sensitive":
        recs.append(("ℹ️", "Sensitive mode active: model prioritises recall — some positives may "
                           "be false alarms. Clinical judgement is essential."))
    return recs


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════════════════════
standard_model, sensitive_model, num_cols, cat_cols = load_models()


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
  <div>
    <div class="header-title">🧠 NeuroScan AI</div>
    <div class="header-sub">Stroke Risk Assessment System &nbsp;·&nbsp;
       Clinical Decision Support Tool &nbsp;·&nbsp; v2.0</div>
  </div>
  <div class="header-badge">⚕️ For Medical Professional Use</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
form_col, result_col = st.columns([1, 1.15], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
# LEFT COLUMN – PATIENT FORM
# ─────────────────────────────────────────────────────────────────────────────
with form_col:

    # ── Patient ID (display only) ────────────────────────────────────────────
    st.markdown('<div class="card card-neutral">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🪪 Patient Record</div>', unsafe_allow_html=True)
    pid_col, pname_col = st.columns(2)
    patient_id   = pid_col.text_input("Patient ID", placeholder="e.g. PT-20241001")
    patient_name = pname_col.text_input("Patient Name", placeholder="Optional")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Demographics ─────────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">👤 Demographics</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    gender  = c1.selectbox("Gender", ["Male", "Female"])
    age     = c2.number_input("Age (years)", min_value=1, max_value=120, value=55, step=1)

    c3, c4 = st.columns(2)
    married   = c3.selectbox("Marital Status", ["Yes", "No"], index=0)
    work_type = c4.selectbox("Work Type", [
        "Private", "Self-employed", "Govt_job", "children", "Never_worked"
    ])
    residence = st.radio("Residence Type", ["Urban", "Rural"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Clinical Measurements ────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🩺 Clinical Measurements</div>', unsafe_allow_html=True)

    h1, h2 = st.columns(2)
    hypertension  = h1.checkbox("Hypertension", value=False)
    heart_disease = h2.checkbox("Heart Disease", value=False)

    g1, g2 = st.columns(2)
    glucose = g1.number_input(
        "Avg Glucose Level (mg/dL)",
        min_value=50.0, max_value=400.0, value=100.0, step=0.1,
        help="Normal fasting: 70–99 mg/dL | Pre-diabetic: 100–125 | Diabetic: ≥ 126",
    )
    bmi_raw = g2.number_input(
        "BMI (kg/m²) — leave 0 if unknown",
        min_value=0.0, max_value=70.0, value=26.0, step=0.1,
        help="Underweight: <18.5 | Normal: 18.5–24.9 | Overweight: 25–29.9 | Obese: ≥30",
    )
    bmi = bmi_raw if bmi_raw > 0 else None
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Lifestyle ────────────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🚬 Lifestyle Factors</div>', unsafe_allow_html=True)
    smoking = st.selectbox("Smoking Status", [
        "never smoked", "formerly smoked", "smokes", "Unknown"
    ])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Screening mode ───────────────────────────────────────────────────────
    st.markdown('<div class="card card-neutral">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">⚙️ Screening Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "",
        ["Sensitive  (fewer missed strokes, more false alarms)",
         "Standard  (higher precision, fewer false alarms)"],
        index=0,
        label_visibility="collapsed",
    )
    mode_key = "Sensitive" if mode.startswith("Sensitive") else "Standard"
    threshold = THRESHOLD_SENSITIVE if mode_key == "Sensitive" else THRESHOLD_STANDARD

    st.markdown(
        f'<div class="disclaimer">{"🔍 Sensitive mode: Recall 72% · Precision 20% · Decision threshold 0.14 — optimised to minimise missed strokes." if mode_key=="Sensitive" else "🎯 Standard mode: Recall 68% · Precision 22% · Decision threshold 0.72 — optimised for balanced screening."}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Analyse button ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    analyse = st.button("🔍  Assess Stroke Risk", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# RIGHT COLUMN – RESULTS
# ─────────────────────────────────────────────────────────────────────────────
with result_col:

    if not analyse:
        # Welcome placeholder
        st.markdown("""
        <div class="welcome-box">
          <div class="welcome-icon">🧠</div>
          <div class="welcome-title">Enter Patient Information</div>
          <div class="welcome-sub">Complete the patient form and click<br>
            <strong>Assess Stroke Risk</strong> to generate an AI-powered report.</div>
        </div>
        """, unsafe_allow_html=True)

        # Model performance summary
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📊 Model Performance Summary"):
            perf_df = pd.DataFrame({
                "Mode":         ["Sensitive (calibrated)", "Standard"],
                "Threshold":    [0.14, 0.72],
                "Recall":       ["72%", "68%"],
                "Precision":    ["20%", "22%"],
                "F2 Score":     [0.477, 0.482],
                "PR-AUC":       [0.256, 0.257],
                "ROC-AUC":      [0.819, 0.842],
            })
            st.dataframe(perf_df, hide_index=True, use_container_width=True)
            st.caption(
                "Thresholds selected by F2-score (β=2) — recall weighted 2× over precision, "
                "appropriate for medical screening where missing a case is more costly than a false alarm."
            )
    else:
        # ── Run prediction ───────────────────────────────────────────────────
        input_df = build_input_df(
            gender, age, hypertension, heart_disease, married,
            work_type, residence, glucose, bmi, smoking,
        )

        model    = sensitive_model if mode_key == "Sensitive" else standard_model
        prob     = float(model.predict_proba(input_df)[0, 1])
        decision = int(prob >= threshold)
        level, badge_cls, card_cls, gauge_color = risk_level(prob, threshold)

        # ── Risk gauge + badge ───────────────────────────────────────────────
        st.markdown(f'<div class="card {card_cls}">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">📊 Risk Assessment Result</div>',
                    unsafe_allow_html=True)

        # patient label
        label_parts = []
        if patient_name: label_parts.append(patient_name)
        if patient_id:   label_parts.append(f"ID: {patient_id}")
        if label_parts:
            st.markdown(
                f'<div style="text-align:center;font-size:13px;color:#78909C;margin-bottom:4px">'
                f'{" · ".join(label_parts)}</div>',
                unsafe_allow_html=True,
            )

        st.plotly_chart(gauge_chart(prob, threshold, gauge_color),
                        use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f'<div class="risk-wrap">'
            f'<div class="risk-prob" style="color:{gauge_color}">{prob*100:.1f}%</div>'
            f'<div class="risk-prob-label">AI Risk Score &nbsp;|&nbsp; '
            f'Decision threshold: {threshold:.2f}</div>'
            f'<span class="risk-badge {badge_cls}">{level} Risk</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        if decision == 1:
            st.markdown(
                '<div class="decision-positive">⚠️ MODEL PREDICTION: ELEVATED STROKE RISK DETECTED</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="decision-negative">✅ MODEL PREDICTION: NO SIGNIFICANT STROKE RISK DETECTED</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Risk factor breakdown ────────────────────────────────────────────
        factors = get_risk_factors(age, hypertension, heart_disease, glucose, bmi, smoking)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">🔎 Risk Factor Analysis</div>',
                    unsafe_allow_html=True)

        high_f   = [(f, i) for f, s, i in factors if s == "high"]
        medium_f = [(f, i) for f, s, i in factors if s == "medium"]
        low_f    = [(f, i) for f, s, i in factors if s == "low"]
        info_f   = [(f, i) for f, s, i in factors if s == "info"]

        def pills_html(items, css_class):
            return "".join(
                f'<span class="factor-pill {css_class}">{icon} {label}</span>'
                for label, icon in items
            )

        if high_f:
            st.markdown(
                f'<div style="font-size:12px;font-weight:700;color:#C62828;'
                f'margin-bottom:4px">HIGH RISK FACTORS</div>'
                f'<div class="factor-row">{pills_html(high_f,"pill-high")}</div>'
                f'<br>',
                unsafe_allow_html=True,
            )
        if medium_f:
            st.markdown(
                f'<div style="font-size:12px;font-weight:700;color:#BF360C;'
                f'margin-bottom:4px">MODERATE RISK FACTORS</div>'
                f'<div class="factor-row">{pills_html(medium_f,"pill-medium")}</div>'
                f'<br>',
                unsafe_allow_html=True,
            )
        if low_f:
            st.markdown(
                f'<div style="font-size:12px;font-weight:700;color:#2E7D32;'
                f'margin-bottom:4px">PROTECTIVE FACTORS</div>'
                f'<div class="factor-row">{pills_html(low_f,"pill-low")}</div>'
                f'<br>',
                unsafe_allow_html=True,
            )
        if info_f:
            st.markdown(
                f'<div class="factor-row">{pills_html(info_f,"pill-info")}</div>',
                unsafe_allow_html=True,
            )

        # Risk factor score bar
        n_high   = len(high_f)
        n_medium = len(medium_f)
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        c_h, c_m, c_l = st.columns(3)
        c_h.metric("High Risk Factors",     n_high)
        c_m.metric("Moderate Risk Factors", n_medium)
        c_l.metric("Protective Factors",    len(low_f))
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Clinical interpretation ──────────────────────────────────────────
        interp = {
            "Low":      ("✅ Low stroke probability. Maintain current healthy habits and "
                         "continue routine annual screening."),
            "Moderate": ("⚠️ Moderate stroke risk identified. Several risk factors require "
                         "attention. Enhanced monitoring and lifestyle modification are advised."),
            "High":     ("🚨 High stroke risk detected. Multiple significant risk factors are "
                         "present. Prompt specialist review and targeted intervention are recommended."),
            "Critical": ("🔴 Critical stroke risk profile. Immediate clinical evaluation is "
                         "strongly recommended. Do not delay specialist referral."),
        }

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">📋 Clinical Interpretation</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:14px;line-height:1.7;color:#37474F">'
            f'{interp[level]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # Recommendations
        st.markdown('<div class="section-label">💊 Recommendations</div>',
                    unsafe_allow_html=True)
        recs = get_recommendations(level, factors, mode_key)
        for icon, text in recs:
            st.markdown(
                f'<div class="rec-item"><span class="rec-icon">{icon}</span>'
                f'<span>{text}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Both models side-by-side ─────────────────────────────────────────
        with st.expander("🔬 Second Opinion — Both Models"):
            std_prob  = float(standard_model.predict_proba(input_df)[0, 1])
            sens_prob = float(sensitive_model.predict_proba(input_df)[0, 1])

            std_dec  = "⚠️ Positive" if std_prob  >= THRESHOLD_STANDARD  else "✅ Negative"
            sens_dec = "⚠️ Positive" if sens_prob >= THRESHOLD_SENSITIVE else "✅ Negative"

            comp_df = pd.DataFrame({
                "Model":         ["Standard (LogReg_Balanced)",
                                  "Sensitive (Calibrated)"],
                "Risk Score":    [f"{std_prob*100:.1f}%",   f"{sens_prob*100:.1f}%"],
                "Threshold":     [THRESHOLD_STANDARD,       THRESHOLD_SENSITIVE],
                "Decision":      [std_dec,                  sens_dec],
                "Recall":        ["68%",                    "72%"],
                "Precision":     ["22%",                    "20%"],
            })
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

        # ── Disclaimer ───────────────────────────────────────────────────────
        st.markdown(
            '<div class="disclaimer">'
            '⚕️ <strong>Clinical Disclaimer:</strong> NeuroScan AI is a screening aid '
            'based on a logistic regression model (PR-AUC ≈ 0.257). It does not replace '
            'clinical examination, imaging, or specialist judgement. All results must be '
            'interpreted by a qualified medical professional.'
            '</div>',
            unsafe_allow_html=True,
        )
