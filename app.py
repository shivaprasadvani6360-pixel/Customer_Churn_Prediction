    numeric_cols = [
        col for col in
        ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]
        if col in cleaned_df.columns
    ]

    if not numeric_cols:
        return pd.DataFrame()

    return cleaned_df[numeric_cols].describe().T.round(2)


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:

    st.markdown("##  ChurnPredict")

    st.caption("Customer Churn Prediction")

    st.divider()

    st.markdown("### Prediction model")

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
        <h1> Customer Churn Prediction</h1>
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
    "Models",
    str(len(models))
)

metrics[1].metric(
    "Best Accuracy",
    "80.34%"
)

metrics[2].metric(
    "Best ROC-AUC",
    "85.43%"
)

metrics[3].metric(
    "XGB Threshold",
    f"{xgb_threshold:.1f}"
)

metrics[4].metric(
    "Test Samples",
    "1,409"
)

st.divider()


# ============================================================
# Main tabs
# ============================================================
tab_predict, tab_compare, tab_eda, tab_history, tab_method, tab_about = st.tabs(
    [
        "Predict Churn",
        "Model Lab",
        "EDA",
        "History",
        "Methodology",
        "About",
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
