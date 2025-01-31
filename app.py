
import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
knn = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app
st.title("Churn Prediction using KNN")

# Input fields for the top 5 features
day_charge = st.number_input("Day Charge")
day_mins = st.number_input("Day Mins")
intl_plan = st.selectbox("International Plan", [0, 1])  # Assuming 0 = No, 1 = Yes
eve_charge = st.number_input("Evening Charge")
eve_mins = st.number_input("Evening Mins")

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_data = np.array([[day_charge, day_mins, intl_plan, eve_charge, eve_mins]])
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = knn.predict(input_data_scaled)
    st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
