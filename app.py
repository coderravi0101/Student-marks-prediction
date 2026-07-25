import streamlit as st
import pandas as pd
import joblib

model = joblib.load("logistic_regression_model.pkl")

st.set_page_config(page_title="Logistic Regression Prediction", page_icon="📊")
st.title("📊 Logistic Regression Prediction App")

study_hours = st.number_input("Study Hours",0.0,24.0,5.0,0.5)

if st.button("Predict"):
    X = pd.DataFrame({"Study_Hours":[study_hours]})
    pred = model.predict(X)[0]
    st.write("### Prediction")
    st.success(f"Predicted Class: {pred}")
    if hasattr(model,"predict_proba"):
        p=model.predict_proba(X)[0]
        st.write(f"Class 0: {p[0]*100:.2f}%")
        st.write(f"Class 1: {p[1]*100:.2f}%")
