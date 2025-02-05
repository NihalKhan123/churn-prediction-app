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
st.markdown("### Enter customer details to predict churn")

# Sidebar for user input
st.sidebar.header("📝 Input Features")
day_charge = st.sidebar.number_input("💰 Day Charge", min_value=0.0, step=0.1)
day_mins = st.sidebar.number_input("⏳ Day Minutes", min_value=0.0, step=0.1)
intl_plan = st.sidebar.selectbox("🌍 International Plan", ["No", "Yes"])  # Convert to binary
eve_charge = st.sidebar.number_input("🌙 Evening Charge", min_value=0.0, step=0.1)
eve_mins = st.sidebar.number_input("🌆 Evening Minutes", min_value=0.0, step=0.1)

# Convert input to binary for model
intl_plan_1 = 1 if intl_plan == "Yes" else 0  # Adjusted to match transformed_cleaned_data

# Centered prediction button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Predict Churn"):
        # Prepare input data
        input_data = np.array([[day_charge, day_mins, intl_plan_1, eve_charge, eve_mins]])
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = knn.predict(input_data_scaled)
        result = "❌ Churn" if prediction[0] == 1 else "✅ No Churn"

        # Display result
        st.success(f"### Prediction: {result}")

# File Upload for Batch Predictions
st.sidebar.header("📂 Batch Prediction")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data", df.head())

    # Ensure the features match the ones in the uploaded file
    batch_features = ['day.charge', 'day.mins', 'intl.plan_1', 'eve.charge', 'eve.mins']
    if all(feature in df.columns for feature in batch_features):
        if st.button("📊 Predict for Uploaded Data"):
            df_scaled = scaler.transform(df[batch_features])  # Only scale the relevant features
            df["Prediction"] = knn.predict(df_scaled)
            st.write("### Predictions", df)
            st.download_button("📥 Download Results", df.to_csv(index=False), "results.csv", "text/csv")
    else:
        st.error("Uploaded data does not contain the required features.")

# Visualization
# Visualization
st.subheader("📈 Churn Distribution")

# Ensure predictions have been made and the 'Prediction' column exists
if uploaded_file:
    if 'Prediction' in df.columns:
        # Group the data by predictions and count occurrences
        churn_data = pd.DataFrame({
            'Churn': ['Yes', 'No'], 
            'Count': [
                df[df['Prediction'] == 1].shape[0],  # Count of churn (1)
                df[df['Prediction'] == 0].shape[0]   # Count of no churn (0)
            ]
        })

        # Ensure the chart has data
        if churn_data['Count'].sum() > 0:
            fig, ax = plt.subplots()
            ax.bar(churn_data['Churn'], churn_data['Count'], color=['red', 'green'])
            ax.set_ylabel('Count')
            ax.set_title('Churn Distribution')
            st.pyplot(fig)
        else:
            st.warning("No predictions to display.")
    else:
        st.warning("Make sure predictions are made before displaying churn distribution.")

