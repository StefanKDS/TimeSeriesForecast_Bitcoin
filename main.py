import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from Helper import evaluate_preds, plot_time_series
import tensorflow as tf

# LOAD DATA #########################################################################

# Download Bitcoin historical data from GitHub
#os.system('wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv')
#os.mkdir('Data')
#shutil.copy('BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv', 'Data')
#os.remove('BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv')

# Parse dates and set date column to index
df = pd.read_csv("Data/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv",
                 parse_dates=["Date"],
                 index_col=["Date"]) # parse the date column (tell pandas column 1 is a datetime)

# Only want closing price for each day
bitcoin_prices = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})

# Plot bitcoin_prices
bitcoin_prices.plot(figsize=(10, 7))
plt.ylabel("BTC Price")
plt.title("Price of Bitcoin from 1 Oct 2013 to 18 May 2021", fontsize=16)
plt.legend(fontsize=14);
plt.show()

# PREPARE DATA ########################################################################

# Get bitcoin date array
timesteps = bitcoin_prices.index.to_numpy()
prices = bitcoin_prices["Price"].to_numpy()

# Create train and test splits the right way for time series data
split_size = int(0.8 * len(prices)) # 80% train, 20% test
# Create train data splits (everything before the split)
X_train, y_train = timesteps[:split_size], prices[:split_size]
# Create test data splits (everything after the split)
X_test, y_test = timesteps[split_size:], prices[split_size:]

plt.figure(figsize=(10, 7))
plot_time_series(timesteps=X_train, values=y_train, plot=plt, label="Train data")
plot_time_series(timesteps=X_test, values=y_test, plot=plt, label="Test data")
plt.show()

# NAIVE FORECAST ######################################################################

# Create a naïve forecast
naive_forecast = y_test[:-1] # Naïve forecast equals every value excluding the last value
naive_forecast[:10], naive_forecast[-10:] # View frist 10 and last 10

# Plot naive forecast
plt.figure(figsize=(10, 7))
plot_time_series(timesteps=X_train, values=y_train, plot=plt, label="Train data")
plot_time_series(timesteps=X_test, values=y_test, plot=plt, label="Test data")
plot_time_series(timesteps=X_test[1:], values=naive_forecast, plot = plt, format="-", label="Naive forecast");
plt.show()

#plt.figure(figsize=(10, 7))
#offset = 300 # offset the values by 300 timesteps
#plot_time_series(timesteps=X_test, values=y_test, plot=plt,start=offset, label="Test data")
#plot_time_series(timesteps=X_test[1:], values=naive_forecast, plot=plt,format="-", start=offset, label="Naive forecast");
#plt.show()

naive_results = evaluate_preds(y_true=y_test[1:],
                               y_pred=naive_forecast)
print(naive_results)

# Find average price of Bitcoin in test dataset
tf.reduce_mean(y_test).numpy()