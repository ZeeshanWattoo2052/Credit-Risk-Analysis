import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Credit Risk App",
    page_icon="💼",
    layout="centered"
)

# Load model
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

st.markdown("<h1 style='color: #007acc;'>💼 Credit Risk Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 16px;'>🔎 <strong>Enter applicant details to predict the likelihood of loan repayment or default.</strong></p>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        "<h2 style='color: #ff6347;'>ℹ️ About This App</h2>"
        "<p>This app predicts the risk of default for loan applicants using a trained <strong>Random Forest</strong> model.</p>"
        "<p><strong>Created By:</strong> Zeeshan Ahmad Wattoo<br>"
        "<strong>Framework:</strong> Streamlit<br>"
        "<strong>Deployed on:</strong> Hugging Face 🤗</p>"
        "<a href='https://github.com/ZeeshanWattoo2052/credit-risk-analysis' target='_blank' style='color: #4caf50; text-decoration: none;'>🔗 GitHub Repo</a>",
        unsafe_allow_html=True
    )

st.header("📝 Applicant Information")
st.markdown(
    "<p style='color: #4caf50; font-weight: bold;'>💡 Tips: Use the sliders and inputs below to enter applicant details.</p>",
    unsafe_allow_html=True
)

# Name input with placeholder and persistent state
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

user_name = st.text_input(
    "👤 Enter Your Name", 
    value=st.session_state.user_name, 
    placeholder="Enter your name"
)

st.session_state.user_name = user_name


# Input form
col1, col2 = st.columns(2)

with col1:
    age = st.slider("🎂 Age", 18, 100, 30)
    income = st.number_input("💰 Annual Income ($)", min_value=0.0, value=50000.0, step=1000.0)
    emp_length = st.slider("👔 Employment Length (years)", 0, 50, 5)
    intent = st.selectbox("🏦 Loan Purpose", ["education", "home_improvement", "medical", "personal", "venture"])

with col2:
    home = st.radio("🏠 Home Ownership", ["Own", "Mortgage", "Rent"])
    amount = st.number_input("💵 Loan Amount ($)", min_value=0.0, value=10000.0, step=500.0)
    rate = st.slider("📈 Interest Rate (%)", 0.0, 100.0, 10.0, 0.5)
    default = st.radio("⚠️ Previous Default", ["Yes", "No"])
    cred_length = st.slider("📜 Credit History Length (years)", 0, 50, 10)

if income == 0:
    st.warning("⚠️ Annual income must be greater than 0.")
    st.stop()

# Derived feature
percent_income = amount / income

# Input DataFrame
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

# Prediction button
if st.button("🔍 Predict Credit Risk"):
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)[0]

    if prediction[0] == 0:
        # Fully Paid
        st.balloons()
        st.success(f"🎉 **Congratulations, {user_name or 'Applicant'}!** You are eligible to get the loan. 💰")
        st.markdown("✅ **Prediction:** 🟢 Fully Paid")
        st.metric(label="Loan Risk Score (Default)", value=f"{proba[1]*100:.1f}%")
    else:
        # Charged Off
        st.warning(
            f"⚠️ **Sorry, {user_name or 'Applicant'}.** Based on our analysis, there is a high risk of default for this application. "
            f"Unfortunately, you are currently **not eligible for the loan**."
        )
        st.markdown("❌ **Prediction:** 🔴 Charged Off")
        st.metric(label="Loan Risk Score (Default)", value=f"{proba[1]*100:.1f}%")

    # Visualize progress
    st.progress(int(proba[1]*100))

    # Probability details
    st.subheader("📊 Prediction Probabilities")
    st.write(f"🟢 Fully Paid: **{proba[0] * 100:.2f}%**")
    st.write(f"🔴 Charged Off: **{proba[1] * 100:.2f}%**")

    # Show data preview
    st.subheader("📄 Input Data Overview")
    st.dataframe(input_df.style.highlight_max(axis=1))

    # Prepare downloadable results
    result_df = input_df.copy()
    status = "Fully Paid" if prediction[0] == 0 else "Charged Off"
    result_df["Predicted Status"] = status
    result_df["Fully Paid Probability (%)"] = round(proba[0] * 100, 2)
    result_df["Charged Off Probability (%)"] = round(proba[1] * 100, 2)

    filename = f"credit_risk_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    st.download_button(
        label="💾 Download Prediction as CSV",
        data=result_df.to_csv(index=False),
        file_name=filename,
        mime="text/csv"
    )

# Centered Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "Made with ❤️ by Zeeshan Ahmad Wattoo"
    "</div>",
    unsafe_allow_html=True
)
