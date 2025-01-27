import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model # type: ignore
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the model
try:
    model = load_model(r'C:\Users\anurag verma\OneDrive\Desktop\ProjectS5\StockPredictionsModel.keras')
except Exception as e:
    st.error(f"Error loading model: {e}")

st.header('Stock Price Predictor')

# Get stock symbol input
stock = st.text_input('Enter Stock Symbol', 'AAPL')

# Get date input from user
start_date = st.date_input("Start Date", value=pd.to_datetime('2015-01-20'))
end_date = st.date_input("End Date", value=pd.to_datetime('2025-01-20'))

# Download stock data
data = yf.download(stock, start=start_date, end=end_date)

st.subheader('Stock Data') 
st.write(data)

# Split data into training and testing sets
data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
past_100_days = data_train.tail(100)
data_test_combined = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test_combined)

# Plot 50-day Moving Average vs Close Price
st.subheader('50-day Moving Average VS Close Price')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='50-day Moving Average')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
st.pyplot(fig1)

# Plot 50-day, 100-day, and Close Price
st.subheader('50-day Moving Average VS 100-day Moving Average VS Close Price')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='50-day Moving Average')
plt.plot(ma_100_days, 'g', label='100-day Moving Average')
plt.plot(data.Close, 'b', label='Close Price')
plt.legend()
st.pyplot(fig2)

# Plot 100-day, 200-day, and Close Price
st.subheader('100-day Moving Average VS 200-day Moving Average VS Close Price')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r', label='100-day Moving Average')
plt.plot(ma_200_days, 'g', label='200-day Moving Average')
plt.plot(data.Close, 'b', label='Close Price')
plt.legend()
st.pyplot(fig3)

# Prepare data for prediction
x_test = []
y_test = []

for i in range(100, data_test_scaled.shape[0]):
    x_test.append(data_test_scaled[i - 100:i])       #last 100 days data 
    y_test.append(data_test_scaled[i, 0])            #corresponding actual price

x_test, y_test = np.array(x_test), np.array(y_test)

# Make predictions
if 'model' in locals():
    predictions = model.predict(x_test)

    # Inverse scaling for predictions and actual values (back to their original values)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot Original Price vs Predicted Price
    st.subheader('Original Price VS Predicted Price')
    fig4 = plt.figure(figsize=(8, 6))
    plt.plot(predictions, 'r', label='Predicted Price')
    plt.plot(y_test, 'g', label='Original Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig4)

    # Calculate and display MSE, RMSE, and R²
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    st.subheader(f"Mean Squared Error (MSE): {mse:.2f}")
    st.subheader(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.subheader(f"R-squared (R²): {r2:.2f}")
else:
    st.warning("Model not loaded. Please check the model path or format.")