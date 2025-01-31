import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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

# Adding Insight Graphs
st.subheader("Insight Graphs")
# Creating a dummy dataset (replace with your real data)
data = {
    'Day Minutes': np.random.rand(100) * 100,
    'Evening Minutes': np.random.rand(100) * 100
}
df = pd.DataFrame(data)

# Plot a Line Chart using Plotly
fig = px.line(df, x="Day Minutes", y="Evening Minutes", title="Day Minutes vs Evening Minutes")
st.plotly_chart(fig)

# Seaborn Bar Plot showing churn distribution
st.subheader("Churn Distribution")
churn_data = {'Churn': ['Yes', 'No'], 'Count': [45, 55]}  # Example data
churn_df = pd.DataFrame(churn_data)
fig2 = sns.barplot(x='Churn', y='Count', data=churn_df)
st.pyplot(fig2.figure)
