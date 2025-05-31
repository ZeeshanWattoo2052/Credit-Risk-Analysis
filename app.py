import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Credit Risk App", page_icon="💼", layout="centered")

# ✅ Load model from Hugging Face Hub using huggingface_hub
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="ZeeshanWattoo/random-forest-credit-model", 
        filename="random_forest_model1.pkl"                   
    )
    return joblib.load(model_path)

model = load_model()

# Category mappings
home_mapping = {"Own": 2, "Mortgage": 1, "Rent": 0}
intent_mapping = {
    "education": 0, "home_improvement": 1, "medical": 2,
    "personal": 3, "venture": 4
}
default_mapping = {"Yes": 1, "No": 0}

st.title("💼 Credit Risk Prediction App")
st.markdown("Enter applicant details below to predict the likelihood of loan repayment or default.")

st.header("📝 Applicant Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income", min_value=0.0, value=50000.0, step=1000.0)
    emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
    intent = st.selectbox("Loan Purpose", ["education", "home_improvement", "medical", "personal", "venture"])

with col2:
    home = st.selectbox("Home Ownership", ["Own", "Mortgage", "Rent"])
    amount = st.number_input("Loan Amount", min_value=0.0, value=10000.0, step=500.0)
    rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
    default = st.selectbox("Previous Default", ["Yes", "No"])
    cred_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=10)

if income == 0:
    st.warning("Annual income must be greater than 0.")
    st.stop()

# Derived feature
percent_income = amount / income

# Create input dataframe
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

# Predict and show results
if st.button("🔍 Predict Credit Risk"):
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)[0]

    status = "🟢 Fully Paid" if prediction[0] == 0 else "🔴 Charged Off"
    st.success(f"**Predicted Loan Status:** {status}")

    # Risk metric
    st.metric(label="Loan Risk Score (Default)", value=f"{proba[1]*100:.1f}%")

    # Probabilities
    st.subheader("📊 Prediction Probabilities")
    st.write(f"🟢 Fully Paid: **{proba[0] * 100:.2f}%**")
    st.write(f"🔴 Charged Off: **{proba[1] * 100:.2f}%**")

    # Prepare downloadable result
    result_df = input_df.copy()
    result_df["Predicted Status"] = status
    result_df["Fully Paid Probability (%)"] = round(proba[0] * 100, 2)
    result_df["Charged Off Probability (%)"] = round(proba[1] * 100, 2)

    # Timestamped filename
    filename = f"credit_risk_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Download button
    st.download_button(
        label="💾 Download Prediction as CSV",
        data=result_df.to_csv(index=False),
        file_name=filename,
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.caption("Made with ❤️ by Zeeshan Ahmad Wattoo | BS Software Engineering, Semester 6")
