import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="India Weather Dashboard", layout="wide")

st.title("🌦️ India Weather Analysis & Forecasting System")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/weather.csv")
df.columns = df.columns.str.strip()

# -----------------------------
# CLEAN DATA
# -----------------------------
df['Temperature (°C)'] = pd.to_numeric(df['Temperature (°C)'], errors='coerce')
df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')

df = df.dropna(subset=['Temperature (°C)', 'Last Updated'])
df = df.sort_values('Last Updated')

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("🎛️ Controls")

states = sorted(df['State/UT'].dropna().unique())
selected_state = st.sidebar.selectbox("Select State/UT", states)

df = df[df['State/UT'] == selected_state]

districts = sorted(df['District'].dropna().unique())
selected_district = st.sidebar.selectbox("Select District", districts)

df = df[df['District'] == selected_district]

forecast_days = st.sidebar.slider("Forecast Days", 7, 30, 10)

show_data = st.sidebar.checkbox("Show Raw Data")

# -----------------------------
# SHOW DATA
# -----------------------------
if show_data:
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

# -----------------------------
# TREND GRAPH
# -----------------------------
st.subheader("📈 Temperature Trend")

fig, ax = plt.subplots()
ax.plot(df['Last Updated'], df['Temperature (°C)'])
ax.set_xlabel("Date")
ax.set_ylabel("Temperature (°C)")
st.pyplot(fig)

# -----------------------------
# FEATURE ENGINEERING (FIXED)
# -----------------------------
df['DayNumber'] = (df['Last Updated'] - df['Last Updated'].min()).dt.days

X = df[['DayNumber']]
y = df['Temperature (°C)']

# -----------------------------
# MODEL
# -----------------------------
model = LinearRegression()
model.fit(X, y)

# -----------------------------
# FUTURE PREDICTION
# -----------------------------
future_days_array = np.array(
    range(df['DayNumber'].max(), df['DayNumber'].max() + forecast_days)
).reshape(-1, 1)

future_pred = model.predict(future_days_array)

future_dates = pd.date_range(df['Last Updated'].max(), periods=forecast_days)

# -----------------------------
# OUTPUT TABLE
# -----------------------------
st.subheader("🔮 Forecast Results")

result = pd.DataFrame({
    "Date": future_dates,
    "Predicted Temperature (°C)": future_pred
})

st.dataframe(result)

# -----------------------------
# DOWNLOAD
# -----------------------------
csv = result.to_csv(index=False)
st.download_button("⬇️ Download Forecast", csv, "forecast.csv", "text/csv")

# -----------------------------
# FORECAST GRAPH
# -----------------------------
st.subheader("📉 Forecast Visualization")

fig2, ax2 = plt.subplots()
ax2.plot(df['Last Updated'], df['Temperature (°C)'], label="Actual")
ax2.plot(future_dates, future_pred, label="Forecast", color='red')
ax2.legend()

st.pyplot(fig2)

# -----------------------------
# INFO
# -----------------------------
st.subheader("ℹ️ Details")
st.write("State:", selected_state)
st.write("District:", selected_district)
st.write("Data Points:", len(df))
st.write("Forecast Days:", forecast_days)