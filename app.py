import streamlit as st
import pickle
import pandas as pd

# Load trained model
model = pickle.load(open("house_price_model.pkl", "rb"))

# Load dataset to get feature columns
data = pd.read_csv("House Price India.csv")
feature_columns = data.drop(["Price", "id", "Date"], axis=1).columns

st.title("🏠 House Price Predictor")

st.write("Enter house details:")

input_data = {}

for col in feature_columns:
    input_data[col] = st.number_input(f"Enter {col}")

if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    st.success(f"Predicted Price: ₹ {prediction[0]:,.2f}")
