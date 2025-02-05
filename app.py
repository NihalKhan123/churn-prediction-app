import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model and scaler
knn = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit app title
st.set_page_config(page_title="Churn Prediction App", layout="wide")
st.title("📊 Churn Prediction using KNN")
st.markdown("### Enter customer details to predict churn and see the feature distribution")

# Sidebar for user input
st.sidebar.header("📝 Input Features")
day_charge = st.sidebar.number_input("💰 Day Charge", min_value=0.0, step=0.1)
day_mins = st.sidebar.number_input("⏳ Day Minutes", min_value=0.0, step=0.1)
intl_plan = st.sidebar.selectbox("🌍 International Plan", ["No", "Yes"])  # Convert to binary
eve_charge = st.sidebar.number_input("🌙 Evening Charge", min_value=0.0, step=0.1)
eve_mins = st.sidebar.number_input("🌆 Evening Minutes", min_value=0.0, step=0.1)

# Convert input to binary for model
intl_plan = 1 if intl_plan == "Yes" else 0

# Centered prediction button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Predict Churn"):
        # Prepare input data
        input_data = np.array([[day_charge, day_mins, intl_plan, eve_charge, eve_mins]])
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = knn.predict(input_data_scaled)
        result = "❌ Churn" if prediction[0] == 1 else "✅ No Churn"

        # Display result
        st.success(f"### Prediction: {result}")

        # Visualization of input values
        st.subheader("📊 Input Feature Distribution")

        feature_names = ["Day Charge", "Day Minutes", "Intl Plan", "Evening Charge", "Evening Minutes"]
        input_values = [day_charge, day_mins, intl_plan, eve_charge, eve_mins]

        # Create a bar chart to visualize the input values
        fig, ax = plt.subplots()
        ax.bar(feature_names, input_values, color=['blue', 'blue', 'red', 'blue', 'blue'])
        ax.set_ylabel("Feature Values")
        ax.set_title("Entered Customer Features")

        # Display the plot
        st.pyplot(fig)
