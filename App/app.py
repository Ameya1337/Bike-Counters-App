import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px


# Load your dataset
@st.cache_data
def load_data():
    data = pd.read_csv("Data/predictions.csv") 
    data["date"] = pd.to_datetime(
        data["date"]
    ) 
    return data


df = load_data()

# Title and description
st.title("Bike Traffic in Paris")
st.write("Visualization of actual and predicted bike traffic at counters in Paris.")

# Sidebar filters
st.sidebar.header("Filters")

# Convert min and max date for the filter to datetime.date
min_date = df["date"].min().date()
max_date = df["date"].max().date()

# Use date_input with the correct date format
counter = st.sidebar.selectbox("Select Counter", df["counter_name"].unique())
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

# Convert date_range values to datetime
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

# Filter data using the datetime-compatible date range
filtered_data = df[
    (df["counter_name"] == counter) & (df["date"].between(start_date, end_date))
]

# Select only numeric columns for resampling (i.e., averaging)
numeric_cols = filtered_data.select_dtypes(include=np.number).columns

# Aggregate by day (only numeric columns)
filtered_data_daily = (
    filtered_data.resample("D", on="date")[numeric_cols].mean().reset_index()
)

# Time Series Plot using Matplotlib
st.subheader(f"Time Series (Daily Average) for {counter}")

# Create Matplotlib figure
fig, ax = plt.subplots(figsize=(10, 5))

# Plot actual and predicted bike counts (daily average)
filtered_data_daily.plot(
    x="date",
    y="bike_count",
    ax=ax,
    label="Actual (Daily Avg)",
    color="#1f77b4",
    linewidth=1.5,
    alpha=0.8,
)
filtered_data_daily.plot(
    x="date",
    y="predicted_bike_count",
    ax=ax,
    label="Predicted (Daily Avg)",
    linestyle="--",
    color="#ff7f0e",
    linewidth=1.5,
    alpha=0.8,
)

# Customize the plot
ax.set_title(
    f"Actual vs Predicted Bike Count (Daily Average) at {counter}", fontsize=14
)
ax.set_ylabel("Bike Count", fontsize=12)
ax.set_xlabel("Date", fontsize=12)
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)

# Scatter Plot for Log Comparison using Plotly
st.subheader(f"Log Bike Count Comparison for {counter}")

# Check if log_bike_count columns exist
if "log_bike_count" in filtered_data and "predicted_log_bike_count" in filtered_data:
    fig_scatter = px.scatter(
        filtered_data,
        x="log_bike_count",
        y="predicted_log_bike_count",
        labels={
            "log_bike_count": "Log Actual",
            "predicted_log_bike_count": "Log Predicted",
        },
        title=f"Log Actual vs Predicted Bike Counts at {counter}",
        trendline="ols",
    )
    st.plotly_chart(fig_scatter)
else:
    st.warning("Log bike count data is not available for this counter.")

# Map of Bike Counters in Paris
st.subheader("Map of Bike Counters in Paris")

map_data = df[["latitude", "longitude", "site_name"]].drop_duplicates()
st.map(map_data)

# Model Performance Metrics
st.sidebar.subheader("Model Performance")

actual = filtered_data["bike_count"]
predicted = filtered_data["predicted_bike_count"]

# Compute and display Mean Absolute Error and R² Score
mae = mean_absolute_error(actual, predicted)
r2 = r2_score(actual, predicted)

st.sidebar.metric("Mean Absolute Error", f"{mae:.2f}")
st.sidebar.metric("R² Score", f"{r2:.2f}")
