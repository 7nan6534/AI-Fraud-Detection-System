import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="💳",
    layout="centered"
)

# Load model
model = joblib.load("xgboost_fraud_model.pkl")

# App title
st.title("💳 AI Fraud Detection System")
st.write("This app predicts whether a financial transaction is fraudulent using an XGBoost machine learning model.")

st.info("Enter the transaction details below, then click Predict.")

# Transaction type mapping
type_mapping = {
    "CASH_OUT": 1,
    "PAYMENT": 2,
    "CASH_IN": 3,
    "TRANSFER": 4,
    "DEBIT": 5
}

st.subheader("Transaction Details")

step = st.number_input("Step", min_value=1, value=1)
type_name = st.selectbox("Transaction Type", list(type_mapping.keys()))
amount = st.number_input("Amount", min_value=0.0, value=1000.0)

oldbalanceOrg = st.number_input("Old Balance - Sender", min_value=0.0, value=5000.0)
newbalanceOrig = st.number_input("New Balance - Sender", min_value=0.0, value=4000.0)

oldbalanceDest = st.number_input("Old Balance - Receiver", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance - Receiver", min_value=0.0, value=0.0)

isFlaggedFraud = st.selectbox("Flagged by System?", [0, 1])

input_data = pd.DataFrame({
    "step": [step],
    "type": [type_mapping[type_name]],
    "amount": [amount],
    "oldbalanceOrg": [oldbalanceOrg],
    "newbalanceOrig": [newbalanceOrig],
    "oldbalanceDest": [oldbalanceDest],
    "newbalanceDest": [newbalanceDest],
    "isFlaggedFraud": [isFlaggedFraud]
})

st.divider()

if st.button("🔍 Predict Transaction"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 Fraud Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

    st.metric("Fraud Probability", f"{probability:.2%}")

    with st.expander("View Input Data"):
        st.dataframe(input_data)

st.divider()
st.caption("Built with Python, XGBoost, SHAP Explainability, and Streamlit.")