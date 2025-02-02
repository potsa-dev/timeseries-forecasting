import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

# Set data folder
data_folder = "data_collection"
model_folder = "models"

# Ensure model folder exists
os.makedirs(model_folder, exist_ok=True)

# Get list of CSV files (top 10 volatile coins)
csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

# Function to load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df = df[["timestamp", "close"]].rename(columns={"timestamp": "ds", "close": "y"})
    return df

# Function to train Prophet model and save it
def train_prophet(train_df, test_df, coin):
    model_path = os.path.join(model_folder, f"prophet_{coin}.pkl")

    # Check if the model already exists
    if os.path.exists(model_path):
        print(f"Loading existing Prophet model for {coin}")
        prophet_model = joblib.load(model_path)
    else:
        print(f"Training new Prophet model for {coin}")
        prophet_model = Prophet()
        prophet_model.fit(train_df)
        joblib.dump(prophet_model, model_path)  # Save the model

    future = prophet_model.make_future_dataframe(periods=len(test_df), freq="min")
    forecast = prophet_model.predict(future)
    return forecast["yhat"].iloc[-len(test_df):]

# Function to evaluate model performance
def evaluate_model(y_true, y_pred, model_name, coin):
    y_pred = y_pred[:len(y_true)]  # Ensure predictions match true values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"RMSE for {model_name} on {coin}: {rmse:.5f}")
    return rmse

# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Function to train LSTM model and save it
def train_lstm(train, test, coin):
    model_path = os.path.join(model_folder, f"lstm_{coin}.pth")
    
    # Convert data to PyTorch tensors
    train_data = torch.tensor(train.values, dtype=torch.float32).view(-1, 1)
    test_data = torch.tensor(test.values, dtype=torch.float32).view(-1, 1)
    train_dataset = TensorDataset(train_data[:-1], train_data[1:])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    # Check if the model already exists
    model = LSTMModel()
    if os.path.exists(model_path):
        print(f"Loading existing LSTM model for {coin}")
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"Training new LSTM model for {coin}")
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(10):  # Train for 10 epochs
            for inputs, targets in train_loader:
                inputs, targets = inputs.unsqueeze(1), targets.view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        torch.save(model.state_dict(), model_path)  # Save the trained model

    model.eval()
    with torch.no_grad():
        test_inputs = test_data.unsqueeze(1)
        predictions = model(test_inputs).squeeze()

    return predictions.numpy()[:len(test)]

# Main loop to train & evaluate models for each coin
results = {}

for csv_file in csv_files:
    coin = csv_file.replace("_last_12_months_data.csv", "")
    file_path = os.path.join(data_folder, csv_file)

    # Load data
    df = load_data(file_path)

    # Train-Test Split (80% train, 20% test)
    train_size = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]
    train_series, test_series = train_df["y"], test_df["y"]

    print(f"\n--- Training models for {coin} ---")

    # Train & Evaluate Prophet
    prophet_preds = train_prophet(train_df, test_df, coin)
    results[(coin, "Prophet")] = evaluate_model(test_series, prophet_preds, "Prophet", coin)

    # Train & Evaluate LSTM
    lstm_preds = train_lstm(train_series, test_series, coin)
    results[(coin, "LSTM")] = evaluate_model(test_series, lstm_preds, "LSTM", coin)

    print(f"Finished training for {coin}")

# Save results
results_df = pd.DataFrame.from_dict(results, orient="index", columns=["RMSE"])
results_df.to_csv("model_performance_results.csv")

print("\nModel evaluation completed. Results saved to 'model_performance_results.csv'.")
