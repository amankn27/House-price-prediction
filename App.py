import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("🏠 California House Price Prediction App")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("housing_model.pkl")

model = load_model()

# Sidebar Inputs
st.sidebar.header("Enter House Details")

longitude = st.sidebar.number_input("Longitude", value=-122.23)
latitude = st.sidebar.number_input("Latitude", value=37.88)
housing_median_age = st.sidebar.slider("Housing Median Age", 1, 100, 35)
total_rooms = st.sidebar.number_input("Total Rooms", value=4)
total_bedrooms = st.sidebar.number_input("Total Bedrooms", value=2)
population = st.sidebar.number_input("Population", value=2)
households = st.sidebar.number_input("Households", value=1)
median_income = st.sidebar.number_input("Median Income", value=8.3252)
ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity",
    ["NEAR BAY", "INLAND", "NEAR OCEAN", "<1H OCEAN", "ISLAND"]
)

# Create DataFrame
input_data = pd.DataFrame({
    "longitude": [longitude],
    "latitude": [latitude],
    "housing_median_age": [housing_median_age],
    "total_rooms": [total_rooms],
    "total_bedrooms": [total_bedrooms],
    "population": [population],
    "households": [households],
    "median_income": [median_income],
    "ocean_proximity": [ocean_proximity]
})

# Feature Engineering
input_data["rooms_per_household"] = input_data["total_rooms"] / input_data["households"]
input_data["bedrooms_per_room"] = input_data["total_bedrooms"] / input_data["total_rooms"]
input_data["population_per_household"] = input_data["population"] / input_data["households"]

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    prediction_inr = prediction * 91.6

    st.success(f"Predicted Price (USD): ${round(prediction, 2)}")
    st.success(f"Predicted Price (INR): ₹ {round(prediction_inr, 2)}")

# ------------------------------
# Visualization Section
# ------------------------------

st.subheader("📊 Dataset Visualization")

data = pd.read_csv("Housing.csv")

col1, col2 = st.columns(2)

with col1:
    fig1 = plt.figure(figsize=(6,4))
    sns.histplot(data["median_house_value"], bins=50)
    st.pyplot(fig1)

with col2:
    fig2 = plt.figure(figsize=(6,4))
    sns.scatterplot(
        x="longitude",
        y="latitude",
        data=data,
        hue="median_house_value",
        palette="coolwarm"
    )
    st.pyplot(fig2)
