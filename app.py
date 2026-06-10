
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

model_1d = tf.keras.models.load_model("tesla_best_model_1d.keras")
model_5d = tf.keras.models.load_model("tesla_best_model_5d.keras")
model_10d = tf.keras.models.load_model("tesla_best_model_10d.keras")

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="Tesla Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------------------------
# Title & Description
# -------------------------------------------------------
st.title("📈 Tesla Stock Price Prediction")
st.markdown("""
This app predicts Tesla (TSLA) stock closing prices
using Deep Learning models — **SimpleRNN** and **LSTM**.

**Prediction Horizons Available:**
- 1 Day Ahead
- 5 Days Ahead  
- 10 Days Ahead
""")

st.markdown("---")

# -------------------------------------------------------
# Load Data & Models
# -------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("TSLA.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df

@st.cache_resource
def load_models_and_scaler():
    scaler     = joblib.load("tesla_scaler.pkl")
    model_1d   = load_model("tesla_best_model_1d.keras")
    model_5d   = load_model("tesla_best_model_5d.keras")
    model_10d  = load_model("tesla_best_model_10d.keras")
    return scaler, model_1d, model_5d, model_10d

# Load everything
try:
    df = load_data()
    scaler, model_1d, model_5d, model_10d = load_models_and_scaler()
    st.success("✅ Data and Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

# -------------------------------------------------------
# Sidebar — User Controls
# -------------------------------------------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

# Model selection
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["LSTM Tuned (Best)", "SimpleRNN", "LSTM"]
)

# Prediction horizon
horizon = st.sidebar.selectbox(
    "Prediction Horizon",
    ["1 Day Ahead", "5 Days Ahead", "10 Days Ahead"]
)

# Date range for chart
st.sidebar.markdown("---")
st.sidebar.subheader("📅 Chart Date Range")

min_date = df.index.min().date()
max_date = df.index.max().date()

start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime("2018-01-01").date(),
    min_value=min_date,
    max_value=max_date
)

end_date = st.sidebar.date_input(
    "End Date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Built with TensorFlow & Streamlit\n"
    "Dataset: TSLA 2010-2020"
)

# -------------------------------------------------------
# Main Dashboard — Key Metrics
# -------------------------------------------------------
st.subheader("📊 Tesla Stock Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Latest Price",
        value=f"${df['Adj Close'].iloc[-1]:.2f}",
        delta=f"{df['Adj Close'].iloc[-1] - df['Adj Close'].iloc[-2]:.2f}"
    )

with col2:
    st.metric(
        label="All-Time High",
        value=f"${df['Adj Close'].max():.2f}"
    )

with col3:
    st.metric(
        label="All-Time Low",
        value=f"${df['Adj Close'].min():.2f}"
    )

with col4:
    total_return = ((df["Adj Close"].iloc[-1] /
                     df["Adj Close"].iloc[0]) - 1) * 100
    st.metric(
        label="Total Return",
        value=f"{total_return:.0f}%"
    )

st.markdown("---")

# -------------------------------------------------------
# Historical Price Chart
# -------------------------------------------------------
st.subheader("📉 Historical Price Chart")

df_filtered = df[
    (df.index >= pd.to_datetime(start_date)) &
    (df.index <= pd.to_datetime(end_date))
]

fig1, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(df_filtered.index,
         df_filtered["Adj Close"],
         color="steelblue",
         linewidth=1.5,
         label="Adj Close")

# Add moving averages
if len(df_filtered) >= 30:
    ma30 = df_filtered["Adj Close"].rolling(30).mean()
    ax1.plot(df_filtered.index, ma30,
             color="orange", linewidth=1.5,
             linestyle="--", label="30-Day MA")

if len(df_filtered) >= 90:
    ma90 = df_filtered["Adj Close"].rolling(90).mean()
    ax1.plot(df_filtered.index, ma90,
             color="green", linewidth=1.5,
             linestyle="-.", label="90-Day MA")

ax1.set_title("Tesla Adj Close Price", fontsize=14)
ax1.set_xlabel("Date")
ax1.set_ylabel("Price ($)")
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig1)

st.markdown("---")

# -------------------------------------------------------
# Prediction Section
# -------------------------------------------------------
st.subheader("🤖 Model Prediction")

WINDOW_SIZE = 60

# Select correct model based on horizon
horizon_map = {
    "1 Day Ahead"  : model_1d,
    "5 Days Ahead" : model_5d,
    "10 Days Ahead": model_10d
}
selected_model = horizon_map[horizon]

# Prepare data for prediction
scaled_data  = scaler.transform(df[["Adj Close"]])
train_size   = int(len(scaled_data) * 0.80)
test_data    = scaled_data[train_size - WINDOW_SIZE:]

# Create test sequences
X_test, y_test = [], []
forecast_days  = int(horizon.split()[0])

for i in range(WINDOW_SIZE, len(test_data) - forecast_days + 1):
    X_test.append(test_data[i - WINDOW_SIZE:i, 0])
    y_test.append(test_data[i + forecast_days - 1, 0])

X_test = np.array(X_test).reshape(-1, WINDOW_SIZE, 1)
y_test = np.array(y_test)

# Make predictions
y_pred      = selected_model.predict(X_test, verbose=0)
y_pred_act  = scaler.inverse_transform(
    y_pred.reshape(-1, 1)
).flatten()
y_test_act  = scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()

# Evaluation metrics
from sklearn.metrics import (mean_squared_error,
                              mean_absolute_error,
                              r2_score)

mse  = mean_squared_error(y_test_act, y_pred_act)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test_act, y_pred_act)
r2   = r2_score(y_test_act, y_pred_act)

# Display metrics
st.subheader("📐 Model Performance Metrics")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("MSE",  f"{mse:.4f}")
with m2:
    st.metric("RMSE", f"${rmse:.4f}")
with m3:
    st.metric("MAE",  f"${mae:.4f}")
with m4:
    st.metric("R²",   f"{r2:.4f}")

st.markdown("---")

# Prediction chart
st.subheader(f"📈 Actual vs Predicted — {horizon}")

fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.plot(y_test_act,
         label="Actual Price",
         color="steelblue",
         linewidth=1.5)
ax2.plot(y_pred_act,
         label=f"Predicted ({model_choice})",
         color="coral",
         linewidth=1.5,
         linestyle="--")
ax2.set_title(
    f"Tesla Actual vs Predicted — {horizon}",
    fontsize=14
)
ax2.set_xlabel("Time Steps (Test Period)")
ax2.set_ylabel("Adj Close Price ($)")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)

st.markdown("---")

# -------------------------------------------------------
# Next Day Forecast
# -------------------------------------------------------
st.subheader("🔮 Next Price Forecast")

# Use last 60 days to predict next price
last_60_days   = scaled_data[-WINDOW_SIZE:]
last_60_reshaped = last_60_days.reshape(1, WINDOW_SIZE, 1)

next_pred_scaled = model_1d.predict(
    last_60_reshaped, verbose=0
)
next_pred_actual = scaler.inverse_transform(
    next_pred_scaled
)[0][0]

current_price = df["Adj Close"].iloc[-1]
change        = next_pred_actual - current_price
change_pct    = (change / current_price) * 100

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric(
        "Current Price",
        f"${current_price:.2f}"
    )
with col_b:
    st.metric(
        "Predicted Next Day",
        f"${next_pred_actual:.2f}",
        delta=f"{change:.2f} ({change_pct:.2f}%)"
    )
with col_c:
    direction = "📈 BUY Signal" if change > 0 else "📉 SELL Signal"
    st.metric("Signal", direction)

st.markdown("---")

# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.markdown("""
---
⚠️ **Disclaimer:** This app is for educational purposes only.
Stock price predictions are not financial advice.
Always consult a qualified financial advisor before investing.

Built as part of Data Science Internship Project 🎓
""")
