from Helper import make_train_test_splits, make_windows, make_preds, evaluate_preds
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import layers

HORIZON = 1  # predict 1 step at a time
WINDOW_SIZE = 7  # use a week worth of timesteps to predict the horizon

# LOAD DATA #########################################################################

# Parse dates and set date column to index
df = pd.read_csv("Data/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv",
                 parse_dates=["Date"],
                 index_col=["Date"])  # parse the date column (tell pandas column 1 is a datetime)

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

full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)

# View the first 3 windows/labels
for i in range(3):
    print(f"Window: {full_windows[i]} -> Label: {full_labels[i]}")

# Train test split
train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)


# Create a function to implement a ModelCheckpoint callback with a specific filename
def create_model_checkpoint(model_name, save_path="model_experiments"):
    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name),
                                              # create filepath to save model
                                              verbose=0,  # only output a limited amount of text
                                              save_best_only=True)  # save only the best model to file

# CREATE MODEL ########################################################################

# Set random seed for as reproducible results as possible
tf.random.set_seed(42)

# Construct model
model_1 = tf.keras.Sequential([
  layers.Dense(128, activation="relu"),
  layers.Dense(HORIZON, activation="linear") # linear activation is the same as having no activation
], name="model_1_dense") # give the model a name so we can save it

# Compile model
model_1.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["mae"]) # we don't necessarily need this when the loss function is already MAE

# Fit model
model_1.fit(x=train_windows, # train windows of 7 timesteps of Bitcoin prices
            y=train_labels, # horizon value of 1 (using the previous 7 timesteps to predict next day)
            epochs=100,
            verbose=1,
            batch_size=128,
            validation_data=(test_windows, test_labels),
            callbacks=[create_model_checkpoint(model_name=model_1.name)]) # create ModelCheckpoint callback to save best model

# Evaluate model on test data
print("Evaluate model on test data")
model_1.evaluate(test_windows, test_labels)

from Helper import make_preds, evaluate_preds, plot_time_series
import tensorflow as tf

print("Load in saved best performing model_1 and evaluate on test data")
# Load in saved best performing model_1 and evaluate on test data
model_1 = tf.keras.models.load_model("model_experiments/model_1_dense")
model_1.evaluate(test_windows, test_labels)

# Make predictions using model_1 on the test dataset and view the results
model_1_preds = make_preds(model_1, test_windows)

# Evaluate preds
model_1_results = evaluate_preds(y_true=tf.squeeze(test_labels), # reduce to right shape
                                 y_pred=model_1_preds)
print(model_1_results)

plt.figure(figsize=(10, 7))
# Account for the test_window offset and index into test_labels to ensure correct plotting
plot_time_series(timesteps=test_windows[-len(test_windows):], values=test_labels[:, 0], plot=plt, label="Test_data")
plot_time_series(timesteps=test_windows[-len(test_windows):], values=model_1_preds, plot=plt, format="-", label="model_1_preds")
plt.show()