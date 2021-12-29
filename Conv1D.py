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

# Check data sample shapes
print(train_windows[0].shape) # returns (WINDOW_SIZE, )

# Before we pass our data to the Conv1D layer, we have to reshape it in order to make sure it works
x = tf.constant(train_windows[0])
expand_dims_layer = layers.Lambda(lambda x: tf.expand_dims(x, axis=1)) # add an extra dimension for timesteps
print(f"Original shape: {x.shape}") # (WINDOW_SIZE)
print(f"Expanded shape: {expand_dims_layer(x).shape}") # (WINDOW_SIZE, input_dim)
print(f"Original values with expanded shape:\n {expand_dims_layer(x)}")

# Create a function to implement a ModelCheckpoint callback with a specific filename
def create_model_checkpoint(model_name, save_path="model_experiments"):
    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name),
                                              # create filepath to save model
                                              verbose=0,  # only output a limited amount of text
                                              save_best_only=True)  # save only the best model to file

# CREATE MODEL ########################################################################

# Create model
model_4 = tf.keras.Sequential([
  # Create Lambda layer to reshape inputs, without this layer, the model will error
  layers.Lambda(lambda x: tf.expand_dims(x, axis=1)), # resize the inputs to adjust for window size / Conv1D 3D input requirements
  layers.Conv1D(filters=128, kernel_size=5, padding="causal", activation="relu"),
  layers.Dense(HORIZON)
], name="model_4_conv1D")

# Compile model
model_4.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam())

# Fit model
model_4.fit(train_windows,
            train_labels,
            batch_size=128,
            epochs=100,
            verbose=0,
            validation_data=(test_windows, test_labels),
            callbacks=[create_model_checkpoint(model_name='model_4_cnn')])

# Evaluate model on test data
print("Evaluate model on test data")
model_4.evaluate(test_windows, test_labels)

from Helper import make_preds, evaluate_preds, plot_time_series
import tensorflow as tf

print("Load in saved best performing model_4 and evaluate on test data")
# Load in saved best performing model_1 and evaluate on test data
model_4 = tf.keras.models.load_model("model_experiments/model_4_cnn")
model_4.evaluate(test_windows, test_labels)

# Make predictions using model_1 on the test dataset and view the results
model_4_preds = make_preds(model_4, test_windows)

# Evaluate preds
model_4_results = evaluate_preds(y_true=tf.squeeze(test_labels), # reduce to right shape
                                 y_pred=model_4_preds)
print(model_4_results)

plt.figure(figsize=(10, 7))
# Account for the test_window offset and index into test_labels to ensure correct plotting
plot_time_series(timesteps=test_windows[-len(test_windows):], values=test_labels[:, 0], plot=plt, label="Test_data")
plot_time_series(timesteps=test_windows[-len(test_windows):], values=model_4_preds, plot=plt, format="-", label="model_4_preds")
plt.show()