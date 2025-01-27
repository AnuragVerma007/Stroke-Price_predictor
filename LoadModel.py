import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dense, Dropout, LSTM # type: ignore
from keras.models import Sequential # type: ignore
import os

# Disable oneDNN optimizations for TensorFlow to prevent potential issues
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Define date range and stock symbol
start = '2014-10-18'
end = '2024-10-18'
stock = 'GOOG'

# Download stock data from Yahoo Finance
try:
    data = yf.download(stock, start, end)
    data.reset_index(inplace=True)   # Reset index for convenience
    data.dropna(inplace=True)        # Remove any missing values
except Exception as e:
    print("Error downloading data:", e)
    exit()

# Plot 100-day moving average alongside the close price
try:
    ma_100_days = data.Close.rolling(100).mean()  # Calculate 100-day moving average
    plt.figure(figsize=(8, 6))
    plt.plot(ma_100_days, 'r', label='100-day Moving Average')
    plt.plot(data.Close, 'g', label='Close Price')
    plt.title('100-day Moving Average and Close Price')
    plt.legend()
    plt.show()

    # Plot both 100-day and 200-day moving averages
    ma_200_days = data.Close.rolling(200).mean()  # Calculate 200-day moving average
    plt.figure(figsize=(8, 6))
    plt.plot(ma_100_days, 'r', label='100-day Moving Average')
    plt.plot(ma_200_days, 'b', label='200-day Moving Average')
    plt.plot(data.Close, 'g', label='Close Price')
    plt.title('100 & 200-day Moving Averages and Close Price')
    plt.legend()
    plt.show()
except Exception as e:
    print("Error in plotting moving averages:", e)

# Split data into training (80%) and testing (20%) sets
try:
    data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])

    # Print the lengths of training and testing sets
    print("Training data length:", data_train.shape[0])
    print("Testing data length:", data_test.shape[0])
    print("Total data length:", len(data))
except Exception as e:
    print("Error in splitting data:", e)

# Scaling the data between 0 and 1 for LSTM compatibility
try:
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_train_scale = scaler.fit_transform(data_train)  # Fit and transform training data
except Exception as e:
    print("Error in scaling data:", e)
    exit()

# Preparing training data with 100-day time windows
try:
    x_train, y_train = [], []
    for i in range(100, data_train_scale.shape[0]):
        x_train.append(data_train_scale[i-100:i])  # Last 100 days as features
        y_train.append(data_train_scale[i, 0])     # Current day as target

    # Convert lists to numpy arrays and reshape x_train for LSTM input shape
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)  # Reshape to (samples, timesteps, features)
except Exception as e:
    print("Error preparing training data:", e)
    exit()

# Building the LSTM model
try:
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(units=80, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
except Exception as e:
    print("Error building model:", e)
    exit()

# Train the model
try:
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)
    model.summary()
except Exception as e:
    print("Error training model:", e)
    exit()

# Prepare test data by combining last 100 days of training with test data
try:
    past_100_days = data_train.tail(100)
    data_test = pd.concat([past_100_days, data_test], ignore_index=True)

    # Scale the test data using the same scaler fitted on training data
    data_test_scale = scaler.transform(data_test)

    # Prepare x_test and y_test using 100-day time windows
    x_test, y_test = [], []
    for i in range(100, data_test_scale.shape[0]):
        x_test.append(data_test_scale[i-100:i])  # Last 100 days as features
        y_test.append(data_test_scale[i, 0])     # Current day as target

    # Convert lists to numpy arrays and reshape for LSTM input
    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
except Exception as e:
    print("Error preparing test data:", e)
    exit()

# Make predictions on test data
try:
    y_predict = model.predict(x_test)

    # Inverse scaling to get predictions and actual values back to original scale
    y_predict = scaler.inverse_transform(y_predict)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
except Exception as e:
    print("Error during prediction:", e)
    exit()

# Plot predicted prices against actual prices
try:
    plt.figure(figsize=(8,6))
    plt.plot(y_predict, 'r', label='Predicted Price')
    plt.plot(y_test, 'g', label='Original Price')
    plt.title('Predicted Price and Original Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
except Exception as e:
    print("Error in plotting predictions:", e)

# Save the model
try:
    model.save('StockPredictionsModel.keras')
except Exception as e:
    print("Error saving model:", e)