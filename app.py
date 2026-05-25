import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Page Configuration
st.set_page_config(
    page_title="House Price Prediction",
    layout="centered"
)

# Title
st.title("House Price Prediction using Linear Regression")

st.write(
    "Predict House Price based on Area using a Linear Regression Model"
)

# Load Dataset
@st.cache_data
def load_data():
    # Kaggle Housing Dataset CSV
    return pd.read_csv("Housing.csv")

df = load_data()

# Dataset Preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Select Required Columns
x = df[["area"]]
y = df["price"]

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train Model
model = LinearRegression()

model.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x_test)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

adj_r2 = 1 - (1 - r2) * (len(y_test)-1) / (len(y_test)-2)

# Visualization
st.subheader("Area vs Price")

fig, ax = plt.subplots()

ax.scatter(df["area"], df["price"])

ax.plot(
    df["area"],
    model.predict(
        scaler.transform(x)
    ),
    color="red"
)

ax.set_xlabel("Area")

ax.set_ylabel("Price")

st.pyplot(fig)

# Model Performance
st.subheader("Model Performance")

c1, c2 = st.columns(2)

c1.metric(
    "Mean Absolute Error",
    f"{mae:.2f}"
)

c2.metric(
    "Root Mean Squared Error",
    f"{rmse:.2f}"
)

c3, c4 = st.columns(2)

c3.metric(
    "R2 Score",
    f"{r2:.3f}"
)

c4.metric(
    "Adjusted R2",
    f"{adj_r2:.3f}"
)

# Coefficient and Intercept
st.subheader("Model Details")

st.write(f"Coefficient : {model.coef_[0]:.3f}")

st.write(f"Intercept : {model.intercept_:.3f}")

# Prediction Section
st.subheader("Predict House Price")

area = st.slider(
    "Select Area",
    int(df.area.min()),
    int(df.area.max())
)

predicted_price = model.predict(
    scaler.transform([[area]])
)[0]

st.success(
    f"Predicted House Price : ₹ {predicted_price:,.2f}"
)