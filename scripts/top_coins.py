import os
from binance.client import Client
import pandas as pd
import numpy as np
import time

# Initialize Binance API (public access, no API key required)
client = Client()

# Ensure the 'data_collection' folder exists
output_folder = "data_collection"
os.makedirs(output_folder, exist_ok=True)

# Step 1: Get the top 50 USDT trading pairs by volume
tickers = client.get_ticker()
df_tickers = pd.DataFrame(tickers)
df_tickers["quoteVolume"] = df_tickers["quoteVolume"].astype(float)

# Filter only USDT pairs and sort by highest volume
df_tickers = df_tickers[df_tickers["symbol"].str.endswith("USDT")]
df_tickers = df_tickers.sort_values(by="quoteVolume", ascending=False).head(50)

# Extract symbols
top_symbols = df_tickers["symbol"].tolist()

# Function to fetch last 12 months of historical data
def fetch_recent_historical_data(symbol, interval=Client.KLINE_INTERVAL_1MINUTE):
    try:
        print(f"Fetching last 12 months of data for {symbol}...")
        # Fetch data from 12 months ago to now
        start_str = "12 months ago UTC"
        klines = client.get_historical_klines(symbol, interval, start_str, "now")

        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades", "taker_base", "taker_quote", "ignore"
        ])

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

# Step 2: Fetch historical data for each coin and calculate volatility
volatility_list = []
coin_data = {}

for symbol in top_symbols:
    df = fetch_recent_historical_data(symbol)

    if df is not None:
        # Calculate log returns
        df["returns"] = np.log(df["close"] / df["close"].shift(1))

        # Compute standard deviation of returns (volatility)
        volatility = df["returns"].std()

        # Store in dictionary
        coin_data[symbol] = df

        # Append to list for ranking
        volatility_list.append((symbol, volatility))

    time.sleep(0.5)  # Sleep to avoid rate limits

# Step 3: Rank coins by volatility and select the top 10
volatility_list = sorted(volatility_list, key=lambda x: x[1], reverse=True)[:10]
top_volatile_symbols = [x[0] for x in volatility_list]

print("\nTop 10 Most Volatile Coins:")
for symbol, vol in volatility_list:
    print(f"{symbol}: {vol:.5f}")

# Step 4: Fetch and save data for the top 10 volatile coins
final_coin_data = {}

for symbol in top_volatile_symbols:
    df = coin_data[symbol]
    final_coin_data[symbol] = df

    # Save CSV under 'data_collection/' folder
    file_path = os.path.join(output_folder, f"{symbol}_last_12_months_data.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved {symbol} data to {file_path}")

print("Data collection and saving completed.")
