import streamlit as st
import pandas as pd
import mlflow
import mlflow.pyfunc
import numpy as np


# MLflow Configuration

mlflow.set_tracking_uri("http://127.0.0.1:5000")

#ELIGIBILITY_MODEL_URI = "models:/EMIELIGIBILITY_XGB_Classifier@champion"
#EMI_MODEL_URI = "models:/MAX_EMI_XGB_Regressor@champion"

ELIGIBILITY_MODEL_URI = "models\eligibility"
EMI_MODEL_URI = "models\maxemi"


# Load Models

@st.cache_resource
def load_models():
    eligibility_model = mlflow.pyfunc.load_model(ELIGIBILITY_MODEL_URI)
    emi_model = mlflow.pyfunc.load_model(EMI_MODEL_URI)
    return eligibility_model, emi_model

eligibility_model, emi_model = load_models()


# Streamlit UI

st.set_page_config(page_title="EMI Decision Engine", layout="centered")
st.title("EMI Eligibility & Maximum EMI Predictor")
st.markdown("---")


# Get Applicant Details

st.subheader("Applicant Details")

monthly_salary = st.number_input("Monthly Salary (₹)", 10000, 500000, 80000)
years_of_employment = st.number_input("Years of Employment", 0.0, 40.0, 6.0)

existing_loans = st.selectbox("Existing Loans", [0, 1])
credit_score = st.number_input("Credit Score", 300, 900, 750)
bank_balance = st.number_input("Bank Balance (₹)", 0, 10000000, 500000)
requested_amount = st.number_input("Requested Loan Amount (₹)", 50000, 5000000, 300000)
requested_tenure = st.number_input("Requested Tenure (months)", 6, 120, 36)

expenses = st.number_input("Monthly Expenses (₹)", 0, 300000, 25000)
house_type_rented = st.checkbox("House Type: Rented 0 or 1")
emergency_fund = st.number_input(
    "Emergency Fund (₹)", 0, 5000000, 150000
)



# ----------------------------------------------------
# Use model to Predict
# ----------------------------------------------------
disposable_funds = monthly_salary - expenses

input_df = pd.DataFrame([{
    "monthly_salary": monthly_salary,
    "years_of_employment": years_of_employment,
    "existing_loans": existing_loans,
    "credit_score": credit_score,
    "bank_balance": bank_balance,
    "emergency_fund": emergency_fund,
    "requested_amount": requested_amount,
    "requested_tenure": requested_tenure,
    "house_type_rented": house_type_rented,
    "expenses": expenses


}])

st.markdown("---")

if st.button("Evaluate EMI"):

    eligibility_pred = eligibility_model.predict(input_df)[0]

    if eligibility_pred == 1:
        st.success(" Applicant is **ELIGIBLE** for EMI")

        predicted_emi = emi_model.predict(input_df)[0]
        predicted_emi = max(500, round(predicted_emi))

        st.subheader("Maximum EMI Amount")
        st.success(f"₹ {predicted_emi:,.0f}")

    else:
        st.error("Applicant is **NOT ELIGIBLE** for EMI")
        st.info("EMI amount prediction is skipped.")


st.markdown("---")
st.markdown("Developed by Padhma S ")
