import streamlit as st
import pandas as pd
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Ensure the project root is in the PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set page layout as the first Streamlit command
st.set_page_config(layout="wide")

from attendance.app.utils import load_image, load_dataset

data = load_dataset(project_root)
print("dataset has been loaded")

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

# Filter data based on selected unique ID
filtered_data = data.filter(pl.col("station") == unique_id)
overdue_data = data.filter(pl.col("anomaly") == "anomaly")

# Header
st.title("Dashboard")

# Metrics in cards
st.markdown('<div class="card-container">', unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(
        '<div class="card"><div class="card-title">New Orders</div><div class="card-metric">12</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        '<div class="card"><div class="card-title">In Progress</div><div class="card-metric">4</div></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        '<div class="card"><div class="card-title">Pending Reviews</div><div class="card-metric">8</div></div>',
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        '<div class="card"><div class="card-title">Overdue</div><div class="card-metric">3</div></div>',
        unsafe_allow_html=True,
    )
with col5:
    st.markdown(
        '<div class="card"><div class="card-title">Tickets Resolved</div><div class="card-metric">5</div></div>',
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Plotting with Matplotlib and Seaborn
fig, ax = plt.subplots(1, 2, figsize=(18, 4))

# y Report
historical_data = filtered_data.filter(pl.col("type") == "historical")
forecast_data = filtered_data.filter(pl.col("type") == "forecast")

ax[0].plot(
    historical_data["date"],
    historical_data["y"],
    label="Historical y",
    color="blue",
)
ax[0].plot(
    forecast_data["date"],
    forecast_data["y_hat"],
    label="Forecast",
    color="orange",
)
ax[0].fill_between(
    forecast_data["date"],
    forecast_data["lower_bound"],
    forecast_data["upper_bound"],
    color="red",
    alpha=0.5,
    label="Confidence Interval",
)
anomalies = forecast_data.filter(pl.col("anomaly") == "anomaly")
ax[0].scatter(anomalies["date"], anomalies["y_hat"], color="red", s=100, label="Anomalies")
ax[0].set_title("y Report", fontsize=18, fontweight="bold")
ax[0].set_xlabel("date", fontsize=14)
ax[0].set_ylabel("y", fontsize=14)
ax[0].legend(fontsize=12)

# Monthly Usage Stats (Density Plot)
sns.histplot(filtered_data["y"], kde=True, stat="density", ax=ax[1], color="blue", bins=20)
mean_y = filtered_data["y"].mean()
ax[1].axvline(mean_y, color="red", linestyle="--", label=f"Mean: {mean_y:.2f}")
ax[1].set_title("Monthly Usage Stats", fontsize=18, fontweight="bold")
ax[1].set_xlabel("y", fontsize=14)
ax[1].set_ylabel("Density", fontsize=14)
ax[1].legend(fontsize=12)

plt.tight_layout()
st.pyplot(fig)

# Table for overdue items
st.markdown('<div class="card overdue-table">', unsafe_allow_html=True)
st.header("anomaly Report")
if not overdue_data.empty:
    st.write(overdue_data)
else:
    st.write("No overdue items for the selected ID.")
st.markdown("</div>", unsafe_allow_html=True)
# Footer
st.markdown("---")
