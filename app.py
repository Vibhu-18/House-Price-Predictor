import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title("🏠 House Price Prediction App")

# Load dataset
data = pd.read_csv("House Price India.csv")

# Remove unnecessary columns
data = data.drop(["id", "Date"], axis=1)

# Split features and target
X = data.drop("Price", axis=1)
y = data["Price"]

# Train model
model = RandomForestRegressor()
model.fit(X, y)

st.write("Model trained successfully ✅")

# Create input fields dynamically
st.header("Enter House Details")

input_data = []

for column in X.columns:
    value = st.number_input(f"Enter {column}", value=float(X[column].mean()))
    input_data.append(value)

# Predict button
if st.button("Predict Price"):
    prediction = model.predict([input_data])
    st.success(f"Estimated House Price: ₹ {prediction[0]:,.2f}")
