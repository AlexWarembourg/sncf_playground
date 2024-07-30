import streamlit as st
import pandas as pd
import numpy as np
import polars as pl
import sys
from pathlib import Path
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import AutoDateFormatter, AutoDateLocator

# Ensure the project root is in the PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set page layout as the first Streamlit command
st.set_page_config(layout="wide")

from attendance.app.utils import load_image, load_dataset

data = load_dataset(project_root)
forecast_data = data.filter(pl.col("type") == "forecast")
number_of_timeseries = int(data["station"].n_unique())
number_of_total_anomalies = int(forecast_data.filter(pl.col("anomaly") == "anomaly").shape[0])
number_of_timeseries_in_anomalies = int(
    forecast_data.filter(pl.col("anomaly") == "anomaly")["station"].n_unique()
)

# CSS styling for cards with gradient colors and bold numbers
st.markdown(
    """
    <style>
    body {
        background-color: #92B5D4;
    }
    .card {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        padding: 20px;
        margin: 10px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .card:nth-child(2) {
        background: linear-gradient(135deg, #fbc2eb 0%, #a18cd1 100%);
    }
    .card:nth-child(3) {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
    }
    .card:nth-child(4) {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    }
    .card:nth-child(5) {
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
    }
    .card-container {
        display: flex;
        flex-wrap: wrap;
    }
    .card-title {
        font-size: 24px;
        font-weight: bold;
        color: #333;
    }
    .card-metric {
        font-size: 32px;
        font-weight: bold;
        color: #333;
    }
    .overdue-table {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Sidebar content
st.sidebar.image(load_image("logo.png"), use_column_width=True)  # Add your logo file
st.sidebar.markdown("## Webapp Goal")
st.sidebar.markdown(
    "This dashboard provides an overview of y and usage statistics for different product IDs. Select a unique ID from the dropdown below to filter the data and view overdue items."
)
unique_id = st.sidebar.selectbox("Select Unique ID", data["station"].unique())

# Header
st.title("Dashboard")

# Metrics in cards
st.markdown('<div class="card-container">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f'<div class="card"><div class="card-title">Monitored</div><div class="card-metric">{number_of_timeseries}</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="card"><div class="card-title">Number of day in Anomaly</div><div class="card-metric">{number_of_total_anomalies}</div></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f'<div class="card"><div class="card-title">Number of Station in Anomaly</div><div class="card-metric">{number_of_timeseries_in_anomalies}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")


# ==========================================================
st.markdown('<div class="card overdue-table">', unsafe_allow_html=True)
st.header("anomaly Report")
anomaly_data = data.filter((pl.col("anomaly") == "anomaly") & (pl.col("type") != "historical"))
if anomaly_data.shape[0] > 0:
    st.write(anomaly_data)
else:
    st.write("No overdue items for the selected ID.")
st.markdown("</div>", unsafe_allow_html=True)
# ==========================================================

st.markdown("---")

# Filter data based on selected unique ID
filtered_data = data.filter(pl.col("station") == unique_id)
historical_data = filtered_data.filter(pl.col("type") == "historical")
length_ts = historical_data.shape[0]
show_n = 120 if length_ts > 120 else length_ts
historical_data = historical_data.filter(
    pl.col("date").cast(pl.String).str.to_datetime()
    >= pl.col("date").cast(pl.String).str.to_datetime().max() - timedelta(days=show_n)
)
forecast_data = filtered_data.filter(pl.col("type") == "forecast")

# Create a figure and axis
fig, ax = plt.subplots(1, 1, figsize=(18, 6))
# Plot historical data
ax.plot(
    historical_data["date"],
    historical_data["y"],
    label="Historical y",
    color="royalblue",
    linewidth=2,
)
# Plot forecast data
ax.plot(
    forecast_data["date"],
    forecast_data["y_hat"],
    label="Forecast",
    color="green",
    marker="x",
    linewidth=2,
)
ax.fill_between(
    forecast_data["date"],
    forecast_data["lower_bound"],
    forecast_data["upper_bound"],
    color="red",
    alpha=0.3,
    label="Confidence Interval",
)

# Plot anomalies
anomalies = forecast_data.filter(pl.col("anomaly") == "anomaly")
ax.scatter(anomalies["date"], anomalies["y_hat"], color="red", s=50, label="Anomalies", zorder=5)
# Customize the plot
ax.set_title("Forecast Monitoring", fontsize=20, fontweight="bold")
ax.set_xlabel("Date", fontsize=16)
ax.set_ylabel("y", fontsize=16)
ax.legend(fontsize=14)
ax.xaxis(historical_data["date"].to_numpy()[::-14])
locator = AutoDateLocator()
ax.xaxis.set_major_formatter(AutoDateFormatter(locator=locator))
fig.autofmt_xdate()
st.pyplot(fig, use_container_width=True)

st.markdown("---")
