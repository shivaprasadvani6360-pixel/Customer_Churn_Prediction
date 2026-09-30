from pathlib import Path
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="ChurnGuard | Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# Custom UI
# ============================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 22px;
        border: 1px solid rgba(128,128,128,.20);
        background: linear-gradient(
            135deg,
            rgba(99,102,241,.13),
            rgba(14,165,233,.08)
        );
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        font-size: 2.6rem;
        margin: 0;
        font-weight: 800;
    }

    .hero p {
        font-size: 1.05rem;
        margin: .5rem 0 0;
        opacity: .82;
    }

    .result-card {
        padding: 1.4rem;
        border-radius: 18px;
        text-align: center;
        border: 1px solid rgba(128,128,128,.22);
        margin: .8rem 0 1rem;
    }

    .positive {
        border-left: 7px solid #16a34a;
    }

    .negative {
        border-left: 7px solid #dc2626;
    }

    .result-label {
        font-size: 2rem;
        font-weight: 800;
    }

    .result-sub {
        opacity: .78;
        margin-top: .3rem;
    }

    .pill {
        display: inline-block;
        padding: .25rem .7rem;
        border-radius: 999px;
        border: 1px solid rgba(128,128,128,.25);
        font-size: .85rem;
        margin-right: .35rem;
    }

    .footer {
        text-align: center;
        opacity: .65;
        padding: 1.5rem 0 .5rem;
        font-size: .85rem;
    }
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
# Sidebar
# ============================================================
with st.sidebar:

    st.markdown("## 📉 ChurnGuard")

    st.caption("Customer Churn Prediction")

    st.divider()

    st.markdown("### 🔮 Prediction model")

    available_models = [
        name
        for name in MODEL_FILES
        if name in models
    ]

    selected_model = st.selectbox(
        "Choose a trained model",
        available_models,
        index=(
            available_models.index("XGBoost")
            if "XGBoost" in available_models
            else 0
        ),
    )

    st.divider()

    st.markdown("### 🧰 Technology")

    for item in [
        "Python",
        "Pandas",
        "Scikit-learn",
        "XGBoost",
        "Joblib",
        "Streamlit",
    ]:

        st.markdown(
            f"<span class='pill'>{item}</span>",
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("### 📌 Project")

    st.write("**Task:** Customer churn classification")

    st.write("**Target:** Churn Label")

    st.write("**Classes:** No Churn / Churn")

    if missing:

        st.warning(
            "Missing model files: "
            + ", ".join(missing)
        )


# ============================================================
# Hero
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>📉 Customer Churn Prediction</h1>
        <p>
        Predict whether a customer is likely to churn
        using four trained machine-learning models.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Overview metrics
# ============================================================
metrics = st.columns(5)

metrics[0].metric(
    "🤖 Models",
    str(len(models))
)

metrics[1].metric(
    "🎯 Best Accuracy",
    "80.34%"
)

metrics[2].metric(
    "📈 Best ROC-AUC",
    "85.43%"
)

metrics[3].metric(
    "🔎 XGB Threshold",
    f"{xgb_threshold:.1f}"
)

metrics[4].metric(
    "👥 Test Samples",
    "1,409"
)

st.divider()


# ============================================================
# Main tabs
# ============================================================
tab_predict, tab_compare, tab_history, tab_method, tab_about = st.tabs(
    [
        "🔮 Predict Churn",
        "📊 Model Lab",
        "📜 History",
        "🔬 Methodology",
        "ℹ️ About",
    ]
)


# ============================================================
# Predict Churn
# ============================================================
with tab_predict:

    left, right = st.columns(
        [1.55, 1],
        gap="large"
    )

    with left:

        st.subheader("Predict customer churn")

        st.caption(
            f"Selected model: **{selected_model}**"
        )

        with st.form("customer_form"):

            c1, c2 = st.columns(2)

            # ------------------------------------------------
            # Numerical + basic customer information
            # ------------------------------------------------
            with c1:

                gender = st.selectbox(
                    "Gender",
                    OPTIONS["Gender"]
                )

                senior = st.selectbox(
                    "Senior Citizen",
                    OPTIONS["Senior Citizen"]
                )

                partner = st.selectbox(
                    "Partner",
                    OPTIONS["Partner"]
                )

                dependents = st.selectbox(
                    "Dependents",
                    OPTIONS["Dependents"]
                )

                tenure = st.number_input(
                    "Tenure Months",
                    min_value=0,
                    max_value=100,
                    value=12
                )

                monthly = st.number_input(
                    "Monthly Charges",
                    min_value=0.0,
                    max_value=2000.0,
                    value=70.0,
                    step=1.0
                )

                total = st.number_input(
                    "Total Charges",
                    min_value=0.0,
                    max_value=100000.0,
                    value=840.0,
                    step=10.0
                )

                cltv = st.number_input(
                    "CLTV",
                    min_value=0.0,
                    max_value=100000.0,
                    value=4500.0,
                    step=50.0
                )

            # ------------------------------------------------
            # Service information
            # ------------------------------------------------
            with c2:

                phone = st.selectbox(
                    "Phone Service",
                    OPTIONS["Phone Service"]
                )

                lines = st.selectbox(
                    "Multiple Lines",
                    OPTIONS["Multiple Lines"]
                )

                internet = st.selectbox(
                    "Internet Service",
                    OPTIONS["Internet Service"]
                )

                security = st.selectbox(
                    "Online Security",
                    OPTIONS["Online Security"]
                )

                backup = st.selectbox(
                    "Online Backup",
                    OPTIONS["Online Backup"]
                )

                protection = st.selectbox(
                    "Device Protection",
                    OPTIONS["Device Protection"]
                )

                support = st.selectbox(
                    "Tech Support",
                    OPTIONS["Tech Support"]
                )

                tv = st.selectbox(
                    "Streaming TV",
                    OPTIONS["Streaming TV"]
                )

                movies = st.selectbox(
                    "Streaming Movies",
                    OPTIONS["Streaming Movies"]
                )

                contract = st.selectbox(
                    "Contract",
                    OPTIONS["Contract"]
                )

                paperless = st.selectbox(
                    "Paperless Billing",
                    OPTIONS["Paperless Billing"]
                )

                payment = st.selectbox(
                    "Payment Method",
                    OPTIONS["Payment Method"]
                )

            clicked = st.form_submit_button(
                "🚀 Predict Churn",
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

                    emoji = "⚠️"
                    css = "negative"
                    message = (
                        "This customer is predicted to churn."
                    )

                else:

                    emoji = "✅"
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
with tab_compare:

    st.subheader("📊 Model Lab")

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
# History
# ============================================================
with tab_history:

    st.subheader(
        "📜 Current-session prediction history"
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
with tab_method:

    st.subheader(
        "🔬 End-to-end methodology"
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
with tab_about:

    st.subheader(
        "ℹ️ About this project"
    )

    st.write(
        "Customer Churn Prediction is a machine-learning "
        "project designed to predict whether a customer is "
        "likely to leave a service using demographic, service "
        "and billing information."
    )

    st.subheader("🎯 Objective")

    st.write(
        "Build a practical churn prediction system with "
        "multiple trained machine-learning models."
    )

    st.subheader("🧠 Models")

    st.write(
        ", ".join(MODEL_FILES.keys())
    )

    st.subheader("📌 Deployment note")

    st.write(
        "The prediction path uses the same saved preprocessing "
        "pipeline used during training."
    )


# ============================================================
# Footer
# ============================================================
st.markdown(
    """
    <div class="footer">
        Customer Churn Prediction · Data Science Major Project
    </div>
    """,
    unsafe_allow_html=True,
)