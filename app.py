
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

bundle = joblib.load("risk_model.pkl")
model = bundle["model"]
features = bundle["features"]

st.title("Credit Risk Prediction Dashboard")
st.caption("Trained on the Give Me Some Credit dataset (150,000 loan applicants)")

tab1, tab2 = st.tabs(["Portfolio Overview", "Predict Applicant Risk"])

with tab1:
    st.subheader("Feature Importance")
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    fig, ax = plt.subplots()
    importances.head(10).plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    st.pyplot(fig)

with tab2:
    st.subheader("Enter Applicant Details")
    inputs = {}
    cols = st.columns(3)
    for i, feat in enumerate(features):
        with cols[i % 3]:
            inputs[feat] = st.number_input(feat, value=0.0)

    if st.button("Predict Risk"):
        input_df = pd.DataFrame([inputs])[features]
        prob = model.predict_proba(input_df)[0][1]
        st.metric("Default Risk Probability", f"{prob*100:.1f}%")
        if prob > 0.5:
            st.error("High Risk Applicant")
        else:
            st.success("Low Risk Applicant")
