
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Example dataset (replace with your own dataset)
# Generating a random dataset for demonstration
np.random.seed(42)
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = np.random.randint(0, 2, size=100)  # Binary target (0 or 1, e.g., churn)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Save the model and scaler using joblib
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Streamlit app title
st.title("Churn Prediction using KNN")

# Input fields for prediction (replace with your actual feature names)
st.sidebar.subheader("Enter Feature Values")
day_charge = st.sidebar.number_input("Day Charge")
day_mins = st.sidebar.number_input("Day Mins")
intl_plan = st.sidebar.selectbox("International Plan", [0, 1])  # 0 = No, 1 = Yes
eve_charge = st.sidebar.number_input("Evening Charge")
eve_mins = st.sidebar.number_input("Evening Mins")

# Predict button
if st.sidebar.button("Predict"):
    # Prepare input data
    input_data = np.array([[day_charge, day_mins, intl_plan, eve_charge, eve_mins]])
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = knn.predict(input_data_scaled)
    st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")

# Visualization Section
st.subheader("Data Visualizations")

# 1. Histogram of Day Mins and Evening Mins
st.subheader("Distribution of Day and Evening Minutes")
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Day Mins Histogram
ax[0].hist(X[:, 0], bins=20, color='skyblue', edgecolor='black')
ax[0].set_title("Day Minutes Distribution")
ax[0].set_xlabel("Day Minutes")
ax[0].set_ylabel("Frequency")

# Evening Mins Histogram
ax[1].hist(X[:, 1], bins=20, color='lightgreen', edgecolor='black')
ax[1].set_title("Evening Minutes Distribution")
ax[1].set_xlabel("Evening Minutes")
ax[1].set_ylabel("Frequency")

st.pyplot(fig)

# 2. Scatter plot of Day vs Evening Minutes
st.subheader("Scatter Plot: Day Minutes vs Evening Minutes")
fig2 = px.scatter(x=X[:, 0], y=X[:, 1], title="Day Minutes vs Evening Minutes", labels={"x": "Day Minutes", "y": "Evening Minutes"})
st.plotly_chart(fig2)

# 3. Pie chart of Churn Distribution
st.subheader("Churn Distribution")
churn_data = pd.DataFrame({'Churn': ['Yes', 'No'], 'Count': [45, 55]})  # Example data
fig3 = px.pie(churn_data, names='Churn', values='Count', title="Churn Distribution")
st.plotly_chart(fig3)

# 4. Line chart showing the relationship between Day Charge and Evening Charge
st.subheader("Line Chart: Day Charge vs Evening Charge")
fig4 = px.line(x=X[:, 0], y=X[:, 2], title="Day Charge vs Evening Charge", labels={"x": "Day Charge", "y": "Evening Charge"})
st.plotly_chart(fig4)

# 5. Correlation Heatmap
st.subheader("Correlation Heatmap")
df = pd.DataFrame(X, columns=['Day Charge', 'Day Mins', 'Intl Plan', 'Evening Charge', 'Evening Mins'])
corr_matrix = df.corr()

fig5, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig5)

# Adding some extra charts for better understanding of the dataset
st.subheader("More Insights")

# Bar plot showing the distribution of 'Churn' labels
st.subheader("Bar Plot: Churn Distribution")
churn_counts = pd.DataFrame({'Churn': ['Yes', 'No'], 'Count': [45, 55]})  # Example data
fig6 = sns.barplot(x='Churn', y='Count', data=churn_counts, palette="Blues_d")
st.pyplot(fig6.figure)

# Show the feature importance (dummy example)
st.subheader("Feature Importance (Dummy Example)")
feature_importance = np.random.rand(5)  # Example feature importance values
features = ['Day Charge', 'Day Mins', 'Intl Plan', 'Evening Charge', 'Evening Mins']
fig7, ax = plt.subplots()
ax.barh(features, feature_importance, color='lightcoral')
ax.set_xlabel("Importance")
ax.set_title("Feature Importance")
st.pyplot(fig7)
