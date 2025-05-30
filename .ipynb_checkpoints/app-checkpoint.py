import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("random_forest_model.pkl")

# Category mappings (must match training time)
home_mapping = {"Own": 2, "Mortgage": 1, "Rent": 0}
intent_mapping = {
    "education": 0, "home_improvement": 1, "medical": 2,
    "personal": 3, "venture": 4
}
default_mapping = {"Yes": 1, "No": 0}

# App title
st.title("Credit Risk Prediction App")
st.markdown("Enter the applicant details to predict the loan status.")

# Sidebar form
with st.sidebar:
    st.header("Applicant Information")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income", min_value=0.0, value=50000.0)
    home = st.selectbox("Home Ownership", ["Own", "Mortgage", "Rent"])
    emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
    intent = st.selectbox("Loan Purpose", ["education", "home_improvement", "medical", "personal", "venture"])
    amount = st.number_input("Loan Amount", min_value=0.0, value=10000.0)
    rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0)
    default = st.selectbox("Previous Default", ["Yes", "No"])
    cred_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=10)

# Derived feature
percent_income = amount / income if income != 0 else 0

# Map categories to match model input
input_df = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Home": [home_mapping[home]],
    "Emp_length": [emp_length],
    "Intent": [intent_mapping[intent]],
    "Amount": [amount],
    "Rate": [rate],
    "Percent_income": [percent_income],
    "Default": [default_mapping[default]],
    "Cred_length": [cred_length]
})

# Predict
if st.button("Predict Credit Risk"):
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)[0]  # Get probabilities for both classes

    # Map prediction to labels
    status = "🟢 Fully Paid" if prediction[0] == 0 else "🔴 Charged Off"

    # Show prediction
    st.success(f"Predicted Loan Status: {status}")

    # Show both class probabilities
    st.write("### Prediction Probabilities")
    st.write(f"🟢 Fully Paid: **{proba[0] * 100:.2f}%**")
    st.write(f"🔴 Charged Off: **{proba[1] * 100:.2f}%**")



