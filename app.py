import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------------------------------
# Page Config
# ----------------------------------------------------
st.set_page_config(
    page_title="EMI Decision Engine",
    layout="centered"
)

st.title("EMI Eligibility & Maximum EMI Predictor")
st.markdown("---")

# ----------------------------------------------------
# Load Models (cached)
# ----------------------------------------------------
ELIGIBILITY_MODEL_PATH = os.path.join("models", "eligibility.pkl")
EMI_MODEL_PATH = os.path.join("models", "maxemi.pkl")

@st.cache_resource
def load_models():
    eligibility_model = joblib.load(ELIGIBILITY_MODEL_PATH)
    emi_model = joblib.load(EMI_MODEL_PATH)
    return eligibility_model, emi_model

eligibility_model, emi_model = load_models()

# ----------------------------------------------------
# Applicant Inputs
# ----------------------------------------------------
st.subheader("Applicant Details")

monthly_salary = st.number_input("Monthly Salary (₹)", 10000, 500000, 80000)
years_of_employment = st.number_input("Years of Employment", 0.0, 40.0, 6.0)
existing_loans = st.selectbox("Existing Loans", [0, 1])
credit_score = st.number_input("Credit Score", 300, 900, 750)
bank_balance = st.number_input("Bank Balance (₹)", 0, 10_000_000, 500_000)
requested_amount = st.number_input("Requested Loan Amount (₹)", 50_000, 5_000_000, 300_000)
requested_tenure = st.number_input("Requested Tenure (months)", 6, 120, 36)
expenses = st.number_input("Monthly Expenses (₹)", 0, 300_000, 25_000)
house_type_rented = st.checkbox("House Type: Rented")
emergency_fund = st.number_input("Emergency Fund (₹)", 0, 5_000_000, 150_000)

# ----------------------------------------------------
# Feature Engineering
# ----------------------------------------------------
input_df = pd.DataFrame([{
    "monthly_salary": monthly_salary,
    "years_of_employment": years_of_employment,
    "existing_loans": existing_loans,
    "credit_score": credit_score,
    "bank_balance": bank_balance,
    "emergency_fund": emergency_fund,
    "requested_amount": requested_amount,
    "requested_tenure": requested_tenure,
    "house_type_rented": int(house_type_rented),
    "expenses": expenses
}])

st.markdown("---")

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------
if st.button("Evaluate EMI"):

    eligibility_pred = eligibility_model.predict(input_df)[0]

    if eligibility_pred == 1:
        st.success("✅ Applicant is **ELIGIBLE** for EMI")

        predicted_emi = emi_model.predict(input_df)[0]
        predicted_emi = max(500, round(predicted_emi))

        st.subheader("Maximum EMI Amount")
        st.success(f"₹ {predicted_emi:,.0f}")

    else:
        st.error("❌ Applicant is **NOT ELIGIBLE** for EMI")
        st.info("EMI amount prediction is skipped.")

st.markdown("---")
st.markdown("Developed by **Padhma S**")
