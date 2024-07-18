import streamlit as st
import pandas as pd
import numpy as np
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

train_data, test_data, submission = load_dataset(project_root)

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

# Sample data
np.random.seed(42)
historical_data = pd.DataFrame(
    {
        "Date": pd.date_range(start="1/1/2023", periods=180, freq="D"),
        "Sales": np.random.randint(100, 500, 180),
        "Quantity": np.random.randint(20, 100, 180),
        "ID": np.random.choice(["A", "B", "C"], 180),
        "Status": np.random.choice(["On Time", "Overdue"], 180),
    }
)
historical_data["Type"] = "Historical"
historical_data["Lower CI"] = np.nan
historical_data["Upper CI"] = np.nan
historical_data["Anomaly"] = "Normal"

# Forecast data
forecast_dates = pd.date_range(start="7/1/2023", periods=30, freq="D")
forecast_sales = np.random.randint(100, 500, 30)
forecast_confidence_interval = np.random.randint(50, 100, 30)

forecast_data = pd.DataFrame(
    {
        "Date": forecast_dates,
        "Sales": forecast_sales,
        "Lower CI": forecast_sales - forecast_confidence_interval,
        "Upper CI": forecast_sales + forecast_confidence_interval,
        "Type": "Forecast",
    }
)

# Flag anomalies in the forecast data
forecast_data["Anomaly"] = np.where(
    (forecast_data["Sales"] > forecast_data["Upper CI"])
    | (forecast_data["Sales"] < forecast_data["Lower CI"]),
    "Anomaly",
    "Normal",
)

# Combine historical and forecast data
data = pd.concat([historical_data, forecast_data])

# Sidebar content
st.sidebar.image(load_image("logo.png"), use_column_width=True)  # Add your logo file
st.sidebar.markdown("## Webapp Goal")
st.sidebar.markdown(
    "This dashboard provides an overview of sales and usage statistics for different product IDs. Select a unique ID from the dropdown below to filter the data and view overdue items."
)
unique_id = st.sidebar.selectbox("Select Unique ID", data["ID"].unique())

# Filter data based on selected unique ID
filtered_data = data[data["ID"] == unique_id]
overdue_data = filtered_data[filtered_data["Status"] == "Overdue"]

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

# Sales Report
historical_data = filtered_data[filtered_data["Type"] == "Historical"]
forecast_data = filtered_data[filtered_data["Type"] == "Forecast"]

ax[0].plot(
    historical_data["Date"],
    historical_data["Sales"],
    label="Historical Sales",
    color="blue",
)
ax[0].plot(
    forecast_data["Date"],
    forecast_data["Sales"],
    label="Forecast Sales",
    color="orange",
)
ax[0].fill_between(
    forecast_data["Date"],
    forecast_data["Lower CI"],
    forecast_data["Upper CI"],
    color="red",
    alpha=0.5,
    label="Confidence Interval",
)
anomalies = forecast_data[forecast_data["Anomaly"] == "Anomaly"]
ax[0].scatter(anomalies["Date"], anomalies["Sales"], color="red", s=100, label="Anomalies")
ax[0].set_title("Sales Report", fontsize=18, fontweight="bold")
ax[0].set_xlabel("Date", fontsize=14)
ax[0].set_ylabel("Sales", fontsize=14)
ax[0].legend(fontsize=12)

# Monthly Usage Stats (Density Plot)
sns.histplot(filtered_data["Quantity"], kde=True, stat="density", ax=ax[1], color="blue", bins=20)
mean_quantity = filtered_data["Quantity"].mean()
ax[1].axvline(mean_quantity, color="red", linestyle="--", label=f"Mean: {mean_quantity:.2f}")
ax[1].set_title("Monthly Usage Stats", fontsize=18, fontweight="bold")
ax[1].set_xlabel("Quantity", fontsize=14)
ax[1].set_ylabel("Density", fontsize=14)
ax[1].legend(fontsize=12)

plt.tight_layout()
st.pyplot(fig)

# Table for overdue items
st.markdown('<div class="card overdue-table">', unsafe_allow_html=True)
st.header("Anomaly Report")
if not overdue_data.empty:
    st.write(overdue_data)
else:
    st.write("No overdue items for the selected ID.")
st.markdown("</div>", unsafe_allow_html=True)
# Footer
st.markdown("---")
