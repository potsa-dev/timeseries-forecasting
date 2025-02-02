import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import torch
from prophet import Prophet
from models import LSTMModel  # Import the LSTM model class

# Set folders
data_folder = "data_collection"
model_folder = "models"
diagram_folder = "diagrams"

# Ensure the diagrams folder exists
os.makedirs(diagram_folder, exist_ok=True)

# Function to load data
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df = df[["timestamp", "close"]].rename(columns={"timestamp": "ds", "close": "y"})
    return df

# Function to make predictions with Prophet
def predict_prophet(coin, test_df):
    model_path = os.path.join(model_folder, f"prophet_{coin}.pkl")
    if not os.path.exists(model_path):
        print(f"No saved Prophet model found for {coin}")
        return None

    prophet_model = joblib.load(model_path)
    future = prophet_model.make_future_dataframe(periods=len(test_df), freq="min")
    forecast = prophet_model.predict(future)

    return forecast["yhat"].iloc[-len(test_df):].values

# Function to make predictions with LSTM
def predict_lstm(coin, test_series):
    model_path = os.path.join(model_folder, f"lstm_{coin}.pth")
    if not os.path.exists(model_path):
        print(f"No saved LSTM model found for {coin}")
        return None

    model = LSTMModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_data = torch.tensor(test_series.values, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        predictions = model(test_data.unsqueeze(1)).squeeze()

    return predictions.numpy()

# Function to plot and save results
def plot_predictions(df, test_df, prophet_preds, lstm_preds, coin):
    plt.figure(figsize=(12, 6))

    # Plot historical data
    plt.plot(df["ds"], df["y"], label="Historical Data", color="black", alpha=0.5)

    # Plot true future values
    plt.plot(test_df["ds"], test_df["y"], label="True Values", color="blue")

    # Plot Prophet predictions
    if prophet_preds is not None:
        plt.plot(test_df["ds"], prophet_preds, label="Prophet Prediction", color="green", linestyle="dashed")

    # Plot LSTM predictions
    if lstm_preds is not None:
        plt.plot(test_df["ds"], lstm_preds, label="LSTM Prediction", color="red", linestyle="dashed")

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"Predictions for {coin}")
    plt.legend()

    # Save the plot inside the `diagrams/` folder
    save_path = os.path.join(diagram_folder, f"{coin}_prediction.png")
    plt.savefig(save_path)
    print(f"Graph saved: {save_path}")

    plt.show()

# Main function to load models and predict
def main():
    csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

    for csv_file in csv_files:
        coin = csv_file.replace("_last_12_months_data.csv", "")
        file_path = os.path.join(data_folder, csv_file)

        # Load data
        df = load_data(file_path)

        # Train-Test Split (80% train, 20% test)
        train_size = int(len(df) * 0.8)
        train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]

        print(f"\n--- Predicting future values for {coin} ---")

        # Prophet Predictions
        prophet_preds = predict_prophet(coin, test_df)

        # LSTM Predictions
        lstm_preds = predict_lstm(coin, test_df["y"])

        # Plot results and save
        plot_predictions(df, test_df, prophet_preds, lstm_preds, coin)

if __name__ == "__main__":
    main()
