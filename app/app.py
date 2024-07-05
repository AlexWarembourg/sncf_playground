import streamlit as st
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from src.project_utils import load_data


# Function to load your data
@st.cache
def load_data():
    # Replace with the actual data loading logic
    historical_data, forecast_data = load_data()
    return historical_data, forecast_data


# Function to plot time series
def plot_time_series(historical, forecast, unique_id):
    plt.figure(figsize=(10, 6))
    plt.plot(historical["date"], historical["y"], label="Historical")
    plt.plot(forecast["date"], forecast["y"], label="Forecast", linestyle="--")
    plt.title(f"Time Series for {unique_id}")
    plt.xlabel("Date")
    plt.ylabel("y")
    plt.legend()
    st.pyplot(plt)


# Function to plot the most affluent series
def plot_most_affluent_series(forecast):
    top_series = forecast.groupby("station_id")["y"].sum().idxmax()
    top_series_data = forecast[forecast["station_id"] == top_series]
    plt.figure(figsize=(10, 6))
    plt.plot(
        top_series_data["date"],
        top_series_data["y"],
        label="Forecast",
        color="orange",
    )
    plt.title(f"Most Affluent Series: {top_series}")
    plt.xlabel("Date")
    plt.ylabel("y")
    plt.legend()
    st.pyplot(plt)


# Function to plot top 5 and flop 5 stations
def plot_top_flop_stations(forecast):
    forecast_sum = forecast.groupby("station_id")["y"].sum().reset_index()
    top_5 = forecast_sum.nlargest(5, "y")
    flop_5 = forecast_sum.nsmallest(5, "y")

    plt.figure(figsize=(10, 6))
    plt.barh(top_5["station_id"], top_5["y"], color="green", label="Top 5")
    plt.barh(flop_5["station_id"], flop_5["y"], color="red", label="Flop 5")
    plt.xlabel("Total Predicted Affluence")
    plt.title("Top 5 and Flop 5 Predicted Stations")
    plt.legend()
    st.pyplot(plt)


# Load data
historical_data, forecast_data = load_data()

# Streamlit application layout
st.title("Station Affluence Prediction Dashboard")

# Unique ID selector
unique_id = st.selectbox("Select Station ID:", historical_data["station_id"].unique())

# Filter data for the selected unique_id
historical_series = historical_data[historical_data["station_id"] == unique_id]
forecast_series = forecast_data[forecast_data["station_id"] == unique_id]

# Plot time series
plot_time_series(historical_series, forecast_series, unique_id)

# Most affluent series
st.subheader("Most Affluent Predicted Series")
plot_most_affluent_series(forecast_data)

# Top 5 and Flop 5 stations
st.subheader("Top 5 and Flop 5 Predicted Stations")
plot_top_flop_stations(forecast_data)
