from pathlib import Path
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="ChurnPredict | Customer Churn Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent

CLEANED_DATA_FILE = "cleaned_Customer_Churn.csv"

@st.cache_data
def load_cleaned_data():
    path = BASE_DIR / CLEANED_DATA_FILE
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

cleaned_df = load_cleaned_data()



# ============================================================
# Custom UI  (ClientPulse-inspired: near-black, coral accent, italic highlights)
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    :root{ --bg:#0A0A0B; --sur:#111113; --sur2:#16161A; --line:rgba(255,255,255,.08);
           --tx:#F4F4F5; --mut:#8B8B93; --acc:#F0484B; --acc2:#FF6467; }
    .stApp{ background:radial-gradient(1200px 500px at 50% -10%, rgba(240,72,75,.10), transparent 60%), var(--bg); color:var(--tx); }
    html, body, [class*="css"]{ font-family:"Inter","Segoe UI",system-ui,sans-serif; }
    .block-container{ max-width:1180px; padding:1rem 1.5rem 0; }
    #MainMenu, footer, header[data-testid="stHeader"], section[data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"]{ display:none !important; }
    em{ color:var(--acc); font-style:italic; }

    @keyframes fadeUp{ from{opacity:0; transform:translateY(18px)} to{opacity:1; transform:none} }
    @keyframes pop{ 0%{opacity:0; transform:scale(.93)} 100%{opacity:1; transform:scale(1)} }
    @keyframes draw{ to{ stroke-dashoffset:0 } }
    @keyframes glow{ 0%,100%{opacity:.55} 50%{opacity:1} }
    @keyframes blink{ 0%,100%{opacity:1} 50%{opacity:.35} }

    /* ---------- Top nav ---------- */
    .brand{ display:flex; align-items:center; gap:.55rem; font-weight:600; font-size:1.15rem; padding-top:.35rem; }
    .brand-dot{ width:24px; height:24px; border-radius:50%; background:var(--acc); display:grid; place-items:center; font-size:.8rem; color:#fff; }
    /* nav = right-aligned segmented selector (no round radios) */
    .st-key-nav{ display:flex; justify-content:flex-end; }
    .st-key-nav [data-testid="stButtonGroup"], .st-key-nav [role="radiogroup"]{
        background:var(--sur); border:1px solid var(--line); border-radius:12px; padding:4px; gap:2px; flex-wrap:wrap; justify-content:flex-end; }
    .st-key-nav button{ background:transparent !important; border:0 !important; border-radius:9px !important; padding:.45rem .95rem; transition:all .25s ease; }
    .st-key-nav button p, .st-key-nav button span{ color:#C4C4CB !important; font-weight:600 !important; font-size:.9rem; }
    .st-key-nav button:hover{ background:rgba(255,255,255,.07) !important; }
    .st-key-nav button:hover p{ color:#fff !important; }
    .st-key-nav button[kind$="Active"], .st-key-nav button[aria-checked="true"], .st-key-nav button[aria-pressed="true"]{
        background:var(--acc) !important; box-shadow:0 6px 18px rgba(240,72,75,.35); }
    .st-key-nav button[kind$="Active"] p, .st-key-nav button[aria-checked="true"] p, .st-key-nav button[aria-pressed="true"] p{ color:#fff !important; }
    .st-key-nav [role="radiogroup"] > label{ padding:.45rem .9rem; border-radius:9px; cursor:pointer; }
    .st-key-nav [role="radiogroup"] > label > :first-child:not(:last-child){ display:none; }
    .st-key-nav [role="radiogroup"] > label p{ color:#C4C4CB; font-weight:600; }
    .st-key-nav [role="radiogroup"] > label:has(input:checked){ background:var(--acc); }
    .st-key-nav [role="radiogroup"] > label:has(input:checked) p{ color:#fff; }

    /* ---------- Hero ---------- */
    .hero{ text-align:center; padding:3.2rem 0 1rem; animation:fadeUp .7s ease both; }
    .badge{ display:inline-block; padding:.35rem .8rem; border:1px solid var(--line); border-radius:999px; background:var(--sur);
            font:500 .68rem "JetBrains Mono",monospace; letter-spacing:.1em; color:var(--mut); }
    .hero h1, .cta h2{ font-size:clamp(2.4rem,5.6vw,4.4rem); line-height:1.05; font-weight:600; letter-spacing:-.045em; margin:1.2rem 0 1rem; }
    .hero p{ max-width:620px; margin:0 auto; color:#c9c9cf; line-height:1.6; }
    .mono{ font:500 .68rem "JetBrains Mono",monospace; letter-spacing:.1em; text-transform:uppercase; color:#5d5d66; }
    .st-key-hero_primary button, .st-key-cta_primary button{ background:var(--acc); color:#fff; border:0; font-weight:600; border-radius:10px; min-height:2.8rem; }
    .st-key-hero_primary button:hover, .st-key-cta_primary button:hover{ background:var(--acc2); color:#fff; transform:translateY(-3px); box-shadow:0 10px 28px rgba(240,72,75,.4); }
    .st-key-hero_secondary button, .st-key-cta_secondary button{ background:var(--sur2); color:#fff; border:1px solid var(--line); font-weight:500; border-radius:10px; min-height:2.8rem; }
    .st-key-hero_secondary button:hover, .st-key-cta_secondary button:hover{ border-color:rgba(255,255,255,.25); color:#fff; transform:translateY(-3px); }

    /* ---------- Dashboard mock ---------- */
    .dash{ display:grid; grid-template-columns:1fr 2.2fr 1.4fr; gap:1.4rem; padding:1.5rem; margin:2rem auto 0;
           background:#0d0d0f; border:2px solid rgba(255,255,255,.09); border-radius:20px; box-shadow:0 0 80px rgba(240,72,75,.10);
           animation:fadeUp .9s .15s ease both; }
    .stat-l{ margin-top:.9rem; } .stat-v{ font-size:2rem; font-weight:500; letter-spacing:-.03em; }
    .stat-v.red{ color:var(--acc); }
    .legend{ display:flex; justify-content:space-between; }
    .dash svg{ width:100%; height:200px; } .ln{ fill:none; stroke-width:2; stroke-dasharray:1200; stroke-dashoffset:1200; animation:draw 2.8s ease forwards; }
    .act{ background:var(--sur2); border:1px solid var(--line); border-radius:10px; padding:.7rem .8rem; margin-bottom:.7rem; transition:transform .25s; }
    .act:hover{ transform:translateX(6px); border-color:rgba(240,72,75,.4); }
    .act b{ font-size:.85rem; font-weight:500; } .act span{ display:block; font-size:.72rem; color:var(--mut); margin-top:.15rem; }

    /* ---------- Sections / cards ---------- */
    .strip{ background:#08080a; border-top:1px solid var(--line); border-bottom:1px solid var(--line); margin:3rem -1.5rem 0; padding:1.6rem 1.5rem; text-align:center; }
    .strip b{ display:inline-block; margin:.6rem 1.6rem 0; font-size:1.3rem; font-weight:600; color:#6b6b73; transition:color .25s; } .strip b:hover{ color:#fff; }
    .sec{ padding:3.4rem 0 .5rem; } .sec h2{ font-size:2.3rem; font-weight:600; letter-spacing:-.04em; line-height:1.15; margin:.6rem 0; }
    .sec p{ color:var(--mut); max-width:520px; }
    .grid4{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:1rem; margin-top:1.6rem; }
    .card, .metric-card, .result-card, [data-testid="stMetric"], [data-testid="stForm"]{
        background:var(--sur); border:1px solid var(--line); border-radius:14px; transition:transform .25s, border-color .25s, box-shadow .25s; }
    .card{ padding:1.3rem; animation:fadeUp .7s ease both; }
    .card:hover, .metric-card:hover, [data-testid="stMetric"]:hover{ transform:translateY(-6px); border-color:rgba(240,72,75,.45); box-shadow:0 16px 40px rgba(240,72,75,.10); }
    .card i{ font-style:normal; color:var(--acc); font-size:1.2rem; } .card h4{ margin:.7rem 0 .4rem; font-weight:500; font-size:1.1rem; }
    .card span{ color:var(--mut); font-size:.85rem; line-height:1.5; }
    .metric-card{ padding:1rem 1.1rem; min-height:104px; } .metric-label{ font:500 .68rem "JetBrains Mono",monospace; letter-spacing:.08em; text-transform:uppercase; color:var(--mut); }
    .metric-value{ font-size:1.7rem; font-weight:600; margin-top:.35rem; letter-spacing:-.03em; } .metric-note{ font-size:.78rem; color:var(--mut); }
    [data-testid="stMetric"]{ padding:.9rem 1rem; } [data-testid="stForm"]{ padding:1.3rem; }
    .result-card{ padding:1.6rem; text-align:center; margin:1rem 0; animation:pop .5s ease both; }
    .positive{ border-left:6px solid #22c55e; } .negative{ border-left:6px solid var(--acc); }
    .result-label{ font-size:2.2rem; font-weight:600; letter-spacing:-.03em; } .result-sub{ color:var(--mut); }
    .section-kicker{ font:500 .7rem "JetBrains Mono",monospace; letter-spacing:.1em; text-transform:uppercase; color:var(--mut); }
    .pill{ display:inline-block; padding:.28rem .62rem; border:1px solid var(--line); border-radius:999px; font-size:.76rem; }

    /* ---------- Page header (inner pages) ---------- */
    .phead{ padding:2.4rem 0 1.2rem; animation:fadeUp .6s ease both; }
    .phead h1{ font-size:clamp(2rem,4vw,3.2rem); font-weight:600; letter-spacing:-.04em; margin:.8rem 0 .4rem; }
    .phead p{ color:var(--mut); max-width:620px; }

    /* ---------- Bottom CTA + footer ---------- */
    .cta{ position:relative; text-align:center; margin:3.5rem -1.5rem 0; padding:4.5rem 1rem 3.5rem; overflow:hidden; border-radius:24px 24px 0 0; }
    .cta:before{ content:""; position:absolute; inset:-40% 20% 30% 20%; background:radial-gradient(ellipse at top, rgba(240,72,75,.55), transparent 70%); animation:glow 4s ease-in-out infinite; }
    .cta > *{ position:relative; } .cta p{ color:#d3d3d8; }
    .foot{ display:grid; grid-template-columns:2fr repeat(4,1fr); gap:2rem; padding:3rem 0 1.2rem; border-top:1px solid var(--line); }
    .foot h5{ font-size:1rem; font-weight:500; margin:0 0 .8rem; } .foot a, .foot span{ display:block; color:var(--mut); font-size:.88rem; margin:.45rem 0; text-decoration:none; transition:color .2s; } .foot a:hover{ color:#fff; }
    .foot-b{ display:flex; justify-content:space-between; padding:1.2rem 0 1.6rem; border-top:1px solid var(--line); }
    .ok:before{ content:""; display:inline-block; width:7px; height:7px; border-radius:50%; background:#22c55e; margin-right:.5rem; animation:blink 2s infinite; }

    /* ---------- Inputs / widgets ---------- */
    [data-testid="stWidgetLabel"] p, label p{ font-weight:700; }
    [data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"]{ background:var(--sur2) !important; border-radius:10px !important; border:1px solid var(--line) !important; }
    [data-baseweb="select"] > div:hover, [data-baseweb="input"]:focus-within{ border-color:var(--acc) !important; }
    [data-testid="stFormSubmitButton"] button{ background:var(--acc); color:#fff; border:0; border-radius:10px; font-weight:600; min-height:2.8rem; }
    [data-testid="stFormSubmitButton"] button:hover{ background:var(--acc2); color:#fff; transform:translateY(-3px); box-shadow:0 10px 28px rgba(240,72,75,.4); }
    button[data-baseweb="tab"]{ font-weight:600; } [data-baseweb="tab-highlight"]{ background:var(--acc) !important; }
    [data-testid="stExpander"]{ background:var(--sur); border:1px solid var(--line); border-radius:14px; }
    [data-testid="stDataFrame"]{ border-radius:12px; overflow:hidden; border:1px solid var(--line); }
    /* ---------- Readability (works even without the dark config theme) ---------- */
    h1, h2, h3, h4, h5, [data-testid="stHeading"], [data-testid="stHeading"] *{ color:#FFFFFF !important; }
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li{ color:#E4E4E7; }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *{ color:#A1A1AA !important; }
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *{ color:#F4F4F5 !important; }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *{ color:#B4B4BC !important; font-weight:600; }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] *{ color:#FFFFFF !important; font-weight:600; }
    [data-testid="stAlert"]{ background:var(--sur) !important; border:1px solid var(--line); border-radius:12px; }
    [data-testid="stAlert"] *{ color:#E4E4E7 !important; }
    button[data-baseweb="tab"] p{ color:#B4B4BC !important; } button[data-baseweb="tab"][aria-selected="true"] p{ color:#fff !important; }
    [data-baseweb="select"] *, input, textarea{ color:#F4F4F5 !important; -webkit-text-fill-color:#F4F4F5; }
    [data-baseweb="popover"] ul, [data-baseweb="menu"]{ background:#16161A !important; } [data-baseweb="popover"] li{ color:#F4F4F5 !important; }
    [data-testid="stExpander"] summary p, [data-testid="stExpander"] p{ color:#F4F4F5 !important; }
    code, pre{ color:#F4F4F5 !important; background:#16161A !important; }
    .stDownloadButton button, .stButton button{ color:#fff; }
    /* ---------- Dark form controls (override Streamlit's light defaults) ---------- */
    .stApp [data-baseweb="select"] > div,
    .stApp [data-baseweb="select"] > div > div,
    .stApp [data-baseweb="input"],
    .stApp [data-baseweb="input"] > div,
    .stApp [data-baseweb="base-input"],
    .stApp [data-testid="stNumberInputContainer"]{
        background-color:#1A1A1F !important; background:#1A1A1F !important; border-color:rgba(255,255,255,.14) !important; }
    .stApp [data-baseweb="select"] > div, .stApp [data-baseweb="input"]{ border-radius:10px !important; border:1px solid rgba(255,255,255,.14) !important; }
    .stApp [data-baseweb="select"] > div:hover, .stApp [data-baseweb="input"]:focus-within{ border-color:#F0484B !important; box-shadow:0 0 0 3px rgba(240,72,75,.18); }
    .stApp [data-baseweb="select"] *, .stApp [data-baseweb="input"] input, .stApp [data-baseweb="base-input"] input{
        color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important; opacity:1 !important; font-weight:500; }
    .stApp [data-baseweb="select"] svg, .stApp [data-baseweb="input"] svg{ fill:#D4D4D8 !important; color:#D4D4D8 !important; }
    .stApp [data-testid="stNumberInputStepDown"], .stApp [data-testid="stNumberInputStepUp"]{ background:#25252B !important; color:#fff !important; }
    .stApp [data-testid="stNumberInputStepDown"]:hover, .stApp [data-testid="stNumberInputStepUp"]:hover{ background:#F0484B !important; }
    [data-baseweb="popover"], [data-baseweb="popover"] > div, [data-baseweb="popover"] ul, [data-baseweb="menu"]{ background:#1A1A1F !important; }
    [data-baseweb="popover"] li, [data-baseweb="popover"] li *{ color:#FFFFFF !important; background:transparent !important; }
    [data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"]{ background:rgba(240,72,75,.25) !important; }
    [data-testid="stSlider"] *{ color:#F4F4F5 !important; }
    /* ---------- Selectbox fix: dark root, transparent children (same in light or dark Streamlit theme) ---------- */
    .stApp [data-testid="stSelectbox"] [data-baseweb="select"],
    .stApp [data-baseweb="select"]{ background:#1A1A1F !important; background-color:#1A1A1F !important; border-radius:10px !important; }
    .stApp [data-baseweb="select"] div, .stApp [data-baseweb="select"] span, .stApp [data-baseweb="select"] input{
        background:transparent !important; background-color:transparent !important;
        color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important; opacity:1 !important; }
    .stApp [data-baseweb="select"] > div{ border:1px solid rgba(255,255,255,.14) !important; border-radius:10px !important; }
    .stApp [data-baseweb="select"] input{ caret-color:transparent; }
    .stApp [data-baseweb="select"] ::selection{ background:transparent; }
    /* ---------- Selectbox v3: slate field, white text (all descendants forced) ---------- */
    .stApp [data-testid="stSelectbox"] *{
        background-color:transparent !important; color:#FFFFFF !important;
        -webkit-text-fill-color:#FFFFFF !important; opacity:1 !important; }
    .stApp [data-testid="stSelectbox"] [data-baseweb="select"],
    .stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div{
        background-color:#2B2F3D !important; background:#2B2F3D !important; border-radius:10px !important; }
    .stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div{ border:1px solid rgba(255,255,255,.22) !important; }
    .stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div:hover{ border-color:#F0484B !important; }
    .stApp [data-testid="stSelectbox"] svg{ fill:#FFFFFF !important; }
    .stApp [data-testid="stNumberInputContainer"], .stApp [data-testid="stNumberInputContainer"] > div{ background-color:#2B2F3D !important; }
    @media (max-width:900px){ .dash{ grid-template-columns:1fr; } .foot{ grid-template-columns:1fr 1fr; } }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Model files
# ============================================================
MODEL_FILES = {
    "Logistic Regression": "logistic_model.pkl",
    "SVM": "svm_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl",
}


# ============================================================
# Model evaluation results
# ============================================================
MODEL_RESULTS = pd.DataFrame(
    [
        ["Logistic Regression", 0.7984, 0.6339, 0.5695, 0.6000, 0.8481],
        ["SVM", 0.8013, 0.6451, 0.5588, 0.5989, 0.8444],
        ["Random Forest", 0.8020, 0.6557, 0.5348, 0.5800, 0.8511],
        ["XGBoost", 0.8034, 0.6570, 0.5428, 0.5944, 0.8543],
    ],
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
    ],
)


# ============================================================
# Customer input options
# ============================================================
OPTIONS = {
    "Gender": ["Male", "Female"],
    "Senior Citizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "Phone Service": ["No", "Yes"],
    "Multiple Lines": ["No phone service", "No", "Yes"],
    "Internet Service": ["DSL", "Fiber optic", "No"],
    "Online Security": ["No internet service", "No", "Yes"],
    "Online Backup": ["No internet service", "No", "Yes"],
    "Device Protection": ["No internet service", "No", "Yes"],
    "Tech Support": ["No internet service", "No", "Yes"],
    "Streaming TV": ["No internet service", "No", "Yes"],
    "Streaming Movies": ["No internet service", "No", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "Paperless Billing": ["No", "Yes"],
    "Payment Method": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}


# ============================================================
# Load models and preprocessing
# ============================================================
@st.cache_resource
def load_artifacts():

    models = {}
    missing = []

    for name, filename in MODEL_FILES.items():

        path = BASE_DIR / filename

        if path.exists():
            models[name] = joblib.load(path)
        else:
            missing.append(filename)

    preprocessor_path = BASE_DIR / "churn_preprocessor.pkl"

    if not preprocessor_path.exists():
        raise FileNotFoundError(
            "churn_preprocessor.pkl is missing."
        )

    preprocessor = joblib.load(preprocessor_path)

    threshold_path = BASE_DIR / "xgboost_threshold.pkl"

    if threshold_path.exists():
        threshold = float(joblib.load(threshold_path))
    else:
        threshold = 0.3

    return models, preprocessor, threshold, missing


try:

    models, preprocessor, xgb_threshold, missing = load_artifacts()

except Exception as exc:

    st.error(
        f"Unable to load project artifacts: {exc}"
    )
    st.stop()


if not models:

    st.error("No trained model files were found.")
    st.stop()


# ============================================================
# Prediction function
# ============================================================
def predict_customer(model_name, values):

    data = pd.DataFrame([values])
    processed = preprocessor.transform(data)
    model = models[model_name]

    if model_name == "XGBoost":

        probability = float(
            model.predict_proba(processed)[0, 1]
        )

        prediction = "Yes" if probability >= xgb_threshold else "No"

    else:

        raw_prediction = model.predict(processed)[0]

        # Convert 0/1 prediction to Yes/No
        prediction = "Yes" if int(raw_prediction) == 1 else "No"

        if hasattr(model, "predict_proba"):
            probability = float(
                model.predict_proba(processed)[0, 1]
            )
        else:
            probability = None

    return prediction, probability


# ============================================================
# Session state
# ============================================================
if "history" not in st.session_state:

    st.session_state.history = []



# ============================================================
# EDA helpers
# ============================================================
def churn_rate_table(column):
    if cleaned_df.empty or "Churn Label" not in cleaned_df.columns:
        return pd.DataFrame()

    table = pd.crosstab(
        cleaned_df[column],
        cleaned_df["Churn Label"],
        normalize="index"
    ) * 100

    return table.round(2)


def numeric_summary():
    numeric_cols = [
        col for col in
        ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]
        if col in cleaned_df.columns
    ]

    if not numeric_cols:
        return pd.DataFrame()

    return cleaned_df[numeric_cols].describe().T.round(2)


# ============================================================
# Layout: top navigation + pages
# ============================================================
def H(markup):
    """Collapse HTML to one line so Markdown never treats it as a code block."""
    return "".join(line.strip() for line in markup.splitlines())


PAGES = ["Home", "Predict Churn", "Model Lab", "EDA", "History", "Methodology", "About"]
PAGE_INFO = {
    "Predict Churn": ("LIVE PREDICTION", "Will this customer <em>stay?</em>", "Enter a customer profile and estimate churn risk with your chosen model."),
    "Model Lab": ("MODEL LAB", "Compare the <em>four</em> models", "Accuracy, F1 and threshold tuning results from the project."),
    "EDA": ("EXPLORATORY ANALYSIS", "What drives <em>churn?</em>", "Key patterns from the cleaned customer dataset."),
    "History": ("SESSION LOG", "Your <em>predictions</em>", "Every prediction made in this session."),
    "Methodology": ("HOW IT WAS BUILT", "End-to-end <em>methodology</em>", "From data cleaning to deployment."),
    "About": ("PROJECT", "About this <em>project</em>", "Objective, models and deployment notes."),
}

if "page" not in st.session_state:
    st.session_state.page = "Home"


def go(target):
    st.session_state.page = target


n1, n2 = st.columns([1.5, 5.5], vertical_alignment="center")
n1.markdown('<div class="brand"><span class="brand-dot">↗</span>ChurnPredict</div>', unsafe_allow_html=True)
with n2:
    with st.container(key="nav"):
        if st.session_state.get("page") not in PAGES:
            st.session_state.page = st.session_state.get("_last_page", "Home")
        if hasattr(st, "segmented_control"):
            st.segmented_control("Navigation", PAGES, key="page", selection_mode="single", label_visibility="collapsed")
        else:
            st.radio("Navigation", PAGES, key="page", horizontal=True, label_visibility="collapsed")

page = st.session_state.get("page")
if page not in PAGES:  # segmented control was un-clicked: stay on the last page
    page = st.session_state.get("_last_page", "Home")
st.session_state["_last_page"] = page
available_models = [n for n in MODEL_FILES if n in models]
selected_model = "XGBoost" if "XGBoost" in available_models else available_models[0]

if missing:
    st.warning("Missing model files: " + ", ".join(missing))

if page != "Home":
    kicker, headline, sub = PAGE_INFO[page]
    st.markdown(H(f'<div class="phead"><span class="badge">{kicker}</span><h1>{headline}</h1><p>{sub}</p></div>'), unsafe_allow_html=True)
    if page == "Predict Churn":
        selected_model = st.selectbox(
            "Prediction model",
            available_models,
            index=available_models.index(selected_model),
        )


# ============================================================
# Home (landing page)
# ============================================================
if page == "Home":

    if not cleaned_df.empty and "Churn Label" in cleaned_df.columns:
        n_cust = len(cleaned_df)
        n_churn = int((cleaned_df["Churn Label"] == "Yes").sum())
        rate = n_churn / n_cust * 100
        rev = (
            cleaned_df.loc[cleaned_df["Churn Label"] == "Yes", "Monthly Charges"].sum()
            if "Monthly Charges" in cleaned_df.columns else 0
        )
    else:
        n_cust, n_churn, rate, rev = 7043, 1869, 26.54, 139130

    acts = [("Month-to-month contracts", "Highest churn segment"), ("Fiber optic internet", "Elevated churn risk"), ("Electronic check payers", "Above-average churn")]
    try:
        ct = churn_rate_table("Contract")
        if not ct.empty and "Yes" in ct.columns:
            acts = [(f"{c} contracts", f"{ct.loc[c, 'Yes']:.1f}% churn rate") for c in ct.index[:3]]
    except Exception:
        pass
    act_html = "".join(f'<div class="act"><b>{t}</b><span>{d}</span></div>' for t, d in acts)

    st.markdown(H(f"""
    <div class="hero">
      <span class="badge">ML-POWERED CHURN INTELLIGENCE</span>
      <h1>Predict Churn Before<br><em>Cancellations</em> Happen</h1>
      <p>ChurnPredict analyzes contract, billing and service patterns across your telecom customer base to flag high-risk accounts before they leave.</p>
    </div>"""), unsafe_allow_html=True)

    _, c1, _ = st.columns([3, 1.6, 3])
    with c1:
        with st.container(key="hero_primary"):
            st.button("Predict Churn  →", use_container_width=True, on_click=go, args=("Predict Churn",))

    st.markdown(H(f"""
    <div style="text-align:center;margin-top:1.6rem" class="mono">Trained on {n_cust:,} customers · 4 models · 85.4% ROC-AUC</div>
    <div class="dash">
      <div>
        <div class="mono">Customers</div><div class="stat-v">{n_cust:,}</div>
        <div class="mono stat-l">At-risk accounts</div><div class="stat-v red">{n_churn:,}</div>
        <div class="mono stat-l">Churn rate</div><div class="stat-v">{rate:.1f}%</div>
        <div class="mono stat-l">Monthly revenue at risk</div><div class="stat-v">${rev:,.0f}</div>
      </div>
      <div>
        <div class="legend"><span class="mono">Churn risk trend</span><span class="mono">● Baseline &nbsp; <span style="color:#F0484B">● Live activity</span></span></div>
        <svg viewBox="0 0 460 200" preserveAspectRatio="none">
          <path class="ln" stroke="#3a3a40" d="M0,170 L70,150 L120,130 L170,135 L220,110 L260,50 L300,100 L380,140 L460,150"/>
          <path class="ln" stroke="#F0484B" style="animation-delay:.4s" d="M0,150 L60,120 L100,125 L130,70 L170,65 L220,120 L260,160 L330,130 L400,110 L460,105"/>
        </svg>
      </div>
      <div><div class="mono" style="margin-bottom:.8rem">Live activity</div>{act_html}</div>
    </div>
    <div class="strip">
      <div class="mono">Built with</div>
      <b>Python</b><b>Pandas</b><b>Scikit-learn</b><b>XGBoost</b><b>Joblib</b><b>Streamlit</b>
    </div>
    <div class="sec">
      <span class="mono">The problem</span>
      <h2>The High Cost Of <em>Unseen</em> Risk</h2>
      <p>Customers rarely leave without warning. They leave because nobody saw the signals building in their contract, billing and usage data.</p>
      <div class="grid4">
        <div class="card"><i>◎</i><h4>Silent Churn</h4><span>Customers disengage weeks before they cancel. By the time you notice, the decision is made.</span></div>
        <div class="card"><i>▤</i><h4>Revenue Volatility</h4><span>Losing high-value monthly accounts without warning leaves gaps in recurring revenue.</span></div>
        <div class="card"><i>⊘</i><h4>Contract Blindness</h4><span>Month-to-month plans churn far more than long-term contracts, yet are rarely flagged.</span></div>
        <div class="card"><i>↻</i><h4>Reactive Retention</h4><span>Teams spend time on win-backs instead of acting on predicted risk early.</span></div>
      </div>
    </div>
    """), unsafe_allow_html=True)


# ============================================================
# Predict Churn
# ============================================================
if page == "Predict Churn":

    left, right = st.columns(
        [2.6, 1],
        gap="large"
    )

    with left:

        st.markdown("### Predict customer churn")

        st.caption(
            f"Selected model: **{selected_model}**"
        )

        with st.form("customer_form"):

            st.markdown("**Demographic data**")
            d1, d2, d3, d4 = st.columns(4)
            gender = d1.selectbox("Gender", OPTIONS["Gender"])
            senior = d2.selectbox("Senior Citizen", OPTIONS["Senior Citizen"])
            partner = d3.selectbox("Partner", OPTIONS["Partner"])
            dependents = d4.selectbox("Dependents", OPTIONS["Dependents"])

            st.markdown("**Account & charges**")
            b1, b2, b3, b4 = st.columns(4)
            tenure = b1.number_input("Tenure Months", min_value=0, max_value=100, value=12)
            monthly = b2.number_input("Monthly Charges", min_value=0.0, max_value=2000.0, value=70.0, step=1.0)
            total = b3.number_input("Total Charges", min_value=0.0, max_value=100000.0, value=840.0, step=10.0)
            cltv = b4.number_input("CLTV", min_value=0.0, max_value=100000.0, value=4500.0, step=50.0)

            st.markdown("**Services signed up for**")
            s1, s2, s3 = st.columns(3)
            phone = s1.selectbox("Phone Service", OPTIONS["Phone Service"])
            lines = s2.selectbox("Multiple Lines", OPTIONS["Multiple Lines"])
            internet = s3.selectbox("Internet Service", OPTIONS["Internet Service"])
            security = s1.selectbox("Online Security", OPTIONS["Online Security"])
            backup = s2.selectbox("Online Backup", OPTIONS["Online Backup"])
            protection = s3.selectbox("Device Protection", OPTIONS["Device Protection"])
            support = s1.selectbox("Tech Support", OPTIONS["Tech Support"])
            tv = s2.selectbox("Streaming TV", OPTIONS["Streaming TV"])
            movies = s3.selectbox("Streaming Movies", OPTIONS["Streaming Movies"])

            st.markdown("**Contract & payment**")
            p1, p2, p3 = st.columns(3)
            contract = p1.selectbox("Contract", OPTIONS["Contract"])
            paperless = p2.selectbox("Paperless Billing", OPTIONS["Paperless Billing"])
            payment = p3.selectbox("Payment Method", OPTIONS["Payment Method"])

            clicked = st.form_submit_button(
                "Predict Churn",
                type="primary",
                use_container_width=True
            )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------
        if clicked:

            values = {

                "Gender": gender,
                "Senior Citizen": senior,
                "Partner": partner,
                "Dependents": dependents,

                "Tenure Months": tenure,

                "Phone Service": phone,
                "Multiple Lines": lines,
                "Internet Service": internet,
                "Online Security": security,
                "Online Backup": backup,
                "Device Protection": protection,
                "Tech Support": support,
                "Streaming TV": tv,
                "Streaming Movies": movies,

                "Contract": contract,
                "Paperless Billing": paperless,
                "Payment Method": payment,

                "Monthly Charges": monthly,
                "Total Charges": total,
                "CLTV": cltv,
            }

            try:

                prediction, probability = predict_customer(
                    selected_model,
                    values
                )

                if prediction == "Yes":

                    emoji = ""
                    css = "negative"
                    message = (
                        "This customer is predicted to churn."
                    )

                else:

                    emoji = ""
                    css = "positive"
                    message = (
                        "This customer is predicted to stay."
                    )

                st.markdown(
                    f"""
                    <div class="result-card {css}">
                        <div class="result-label">
                            {emoji} {prediction.upper()}
                        </div>
                        <div class="result-sub">
                            {message}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Model Used",
                    selected_model
                )

                col2.metric(
                    "Churn Probability",
                    (
                        f"{probability * 100:.1f}%"
                        if probability is not None
                        else "N/A"
                    )
                )

                st.session_state.history.insert(
                    0,
                    {
                        "Model": selected_model,
                        "Prediction": (
                            "Churn"
                            if prediction == "Yes"
                            else "No Churn"
                        ),
                        "Churn Probability": (
                            f"{probability * 100:.1f}%"
                            if probability is not None
                            else "N/A"
                        ),
                    }
                )

                if selected_model == "XGBoost":

                    st.info(
                        f"XGBoost uses the project threshold "
                        f"of {xgb_threshold:.1f}."
                    )

            except Exception as exc:

                st.error(
                    f"Prediction failed: {exc}"
                )


    # --------------------------------------------------------
    # How it works
    # --------------------------------------------------------
    with right:

        st.subheader("How it works")

        st.markdown(
            """
            **1. Customer input**  
            Enter demographic, service and billing information.

            **2. Preprocessing**  
            The saved project preprocessing pipeline transforms the features.

            **3. Model selection**  
            Choose one of the four trained classifiers.

            **4. Prediction**  
            The selected model predicts customer churn.

            **5. Result**  
            The app displays the prediction and probability.
            """
        )

        st.info(
            "Tip: Try the same customer with different models "
            "to compare predictions."
        )


# ============================================================
# Model Lab
# ============================================================
if page == "Model Lab":

    st.subheader("Model Lab")

    st.write(
        "Compare the four tuned models using the evaluation "
        "results from the project."
    )

    display = MODEL_RESULTS.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
    ]:

        display[column] = display[column].apply(
            lambda x:
            "N/A"
            if pd.isna(x)
            else f"{x * 100:.2f}%"
        )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Performance comparison")

    chart_df = (
        MODEL_RESULTS
        .set_index("Model")
        [["Accuracy", "F1-Score"]]
        * 100
    )

    st.bar_chart(chart_df)

    st.subheader("XGBoost threshold tuning")

    threshold_df = pd.DataFrame(
        [
            [0.3, 76.72, 54.29, 77.81, 63.96],
            [0.4, 79.28, 59.67, 67.65, 63.41],
            [0.5, 80.34, 65.70, 54.28, 59.44],
            [0.6, 80.06, 75.14, 37.17, 49.73],
            [0.7, 78.57, 80.51, 25.40, 38.62],
        ],
        columns=[
            "Threshold",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
        ],
    )

    st.dataframe(
        threshold_df.style.format(
            {
                "Threshold": "{:.1f}",
                "Accuracy": "{:.2f}%",
                "Precision": "{:.2f}%",
                "Recall": "{:.2f}%",
                "F1-Score": "{:.2f}%",
            }
        ),
        use_container_width=True,
        hide_index=True
    )



# ============================================================
# EDA
# ============================================================
if page == "EDA":

    st.subheader("Exploratory Data Analysis")
    st.caption(
        "Key patterns from the cleaned customer dataset used in the project."
    )

    if cleaned_df.empty:
        st.warning(
            "cleaned_Customer_Churn.csv was not found in the app folder."
        )
    else:
        total_customers = len(cleaned_df)
        churned = int((cleaned_df["Churn Label"] == "Yes").sum())
        retained = int((cleaned_df["Churn Label"] == "No").sum())
        churn_rate = churned / total_customers * 100

        e1, e2, e3, e4 = st.columns(4)

        e1.metric("Customers", f"{total_customers:,}")
        e2.metric("Churned", f"{churned:,}")
        e3.metric("Stayed", f"{retained:,}")
        e4.metric("Churn Rate", f"{churn_rate:.2f}%")

        st.divider()

        st.subheader("Churn distribution")

        churn_counts = (
            cleaned_df["Churn Label"]
            .value_counts()
            .rename_axis("Churn Label")
            .to_frame("Customers")
        )

        st.bar_chart(churn_counts)

        st.subheader("Important churn patterns")

        pattern_tabs = st.tabs([
            "Contract",
            "Tenure",
            "Monthly Charges",
            "Internet Service",
            "Payment Method",
            "Senior Citizen",
        ])

        with pattern_tabs[0]:
            table = churn_rate_table("Contract")
            if not table.empty:
                st.bar_chart(table)
                st.dataframe(
                    table,
                    use_container_width=True
                )

        with pattern_tabs[1]:
            tenure_stats = (
                cleaned_df.groupby("Churn Label")["Tenure Months"]
                .agg(["mean", "median"])
                .round(2)
            )

            st.dataframe(
                tenure_stats,
                use_container_width=True
            )

            st.bar_chart(
                cleaned_df.groupby("Churn Label")["Tenure Months"].mean()
            )

        with pattern_tabs[2]:
            monthly_stats = (
                cleaned_df.groupby("Churn Label")["Monthly Charges"]
                .agg(["mean", "median"])
                .round(2)
            )

            st.dataframe(
                monthly_stats,
                use_container_width=True
            )

            st.bar_chart(
                cleaned_df.groupby("Churn Label")["Monthly Charges"].mean()
            )

        with pattern_tabs[3]:
            table = churn_rate_table("Internet Service")
            if not table.empty:
                st.bar_chart(table)
                st.dataframe(
                    table,
                    use_container_width=True
                )

        with pattern_tabs[4]:
            table = churn_rate_table("Payment Method")
            if not table.empty:
                st.bar_chart(table)
                st.dataframe(
                    table,
                    use_container_width=True
                )

        with pattern_tabs[5]:
            table = churn_rate_table("Senior Citizen")
            if not table.empty:
                st.bar_chart(table)
                st.dataframe(
                    table,
                    use_container_width=True
                )

        st.divider()

        st.subheader("Numerical feature summary")

        summary = numeric_summary()

        if not summary.empty:
            st.dataframe(
                summary,
                use_container_width=True
            )

        st.divider()

        st.subheader("Cleaned customer data")

        st.caption(
            f"Reference dataset: {total_customers:,} rows × "
            f"{cleaned_df.shape[1]} columns"
        )

        sample_size = st.slider(
            "Rows to preview",
            min_value=5,
            max_value=min(100, total_customers),
            value=min(10, total_customers),
            step=5,
        )

        st.dataframe(
            cleaned_df.head(sample_size),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download cleaned customer data",
            data=cleaned_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_Customer_Churn.csv",
            mime="text/csv",
        )


# ============================================================
# History
# ============================================================
if page == "History":

    st.subheader(
        "Current-session prediction history"
    )

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button(
            "Clear prediction history"
        ):

            st.session_state.history = []

            st.rerun()

    else:

        st.info(
            "Prediction history will appear here "
            "after predictions."
        )


# ============================================================
# Methodology
# ============================================================
if page == "Methodology":

    st.subheader(
        "End-to-end methodology"
    )

    steps = [

        (
            "01",
            "Data Cleaning",
            "Removed IDs, leakage columns and location columns."
        ),

        (
            "02",
            "Missing Values",
            "Converted Total Charges to numeric and treated "
            "11 tenure-zero blanks as 0."
        ),

        (
            "03",
            "EDA",
            "Analyzed churn by contract, tenure, internet "
            "service, payment method and other features."
        ),

        (
            "04",
            "Train-Test Split",
            "80:20 stratified split."
        ),

        (
            "05",
            "Preprocessing",
            "StandardScaler for numerical features and "
            "OneHotEncoder for categorical features."
        ),

        (
            "06",
            "Model Training",
            "Logistic Regression, SVM, Random Forest and XGBoost."
        ),

        (
            "07",
            "Hyperparameter Tuning",
            "5-fold GridSearchCV using F1-score."
        ),

        (
            "08",
            "Threshold Tuning",
            "XGBoost threshold tested from 0.3 to 0.7; "
            "0.3 used for higher churn recall."
        ),

        (
            "09",
            "Deployment",
            "Interactive Streamlit application."
        ),
    ]

    for number, title, description in steps:

        st.markdown(
            f"**{number} · {title}**"
        )

        st.caption(description)


    st.subheader("Pipeline")

    st.code(
        """Customer Data
      ↓
Data Preprocessing
      ↓
Feature Transformation
      ↓
4 Machine Learning Models
      ↓
Evaluation & Hyperparameter Tuning
      ↓
XGBoost Threshold Tuning
      ↓
Interactive Model Selection
      ↓
Churn Prediction"""
    )


# ============================================================
# About
# ============================================================
if page == "About":

    st.subheader(
        "About this project"
    )

    st.write(
        "Customer Churn Prediction is a machine-learning "
        "project designed to predict whether a customer is "
        "likely to leave a service using demographic, service "
        "and billing information."
    )

    st.subheader("Objective")

    st.write(
        "Build a practical churn prediction system with "
        "multiple trained machine-learning models."
    )

    st.subheader("Models")

    st.write(
        ", ".join(MODEL_FILES.keys())
    )

    st.subheader("Deployment note")

    st.write(
        "The prediction path uses the same saved preprocessing "
        "pipeline used during training."
    )


# ============================================================
# Footer
# ============================================================
st.markdown(
    H("""
    <div class="foot">
      <div><div class="brand"><span class="brand-dot">↗</span>ChurnPredict</div>
        <span style="max-width:300px;margin-top:1rem">Predictive churn intelligence for telecom customers. Spot at-risk accounts early with data-driven models.</span></div>
      <div><h5>Product</h5><a>Predict Churn</a><a>Model Lab</a><a>EDA</a><a>History</a></div>
      <div><h5>Project</h5><a>Methodology</a><a>About</a><a>Dataset</a></div>
      <div><h5>Stack</h5><a>Python</a><a>XGBoost</a><a>Scikit-learn</a><a>Streamlit</a></div>
      <div><h5>Models</h5><a>Logistic Regression</a><a>SVM</a><a>Random Forest</a><a>XGBoost</a></div>
    </div>
    <div class="foot-b"><span class="mono">2026 ChurnPredict · Data Science Major Project</span><span class="mono ok">All systems operational</span></div>
    """),
    unsafe_allow_html=True,
)
