from binance.client import Client
import pandas as pd
import time

# Initialize Binance API client (no API key needed for public data)
client = Client()

# Define the trading pair and timeframe
symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_1MINUTE  # 1-minute candles (adjust if needed)

# Binance API only allows 1000 candles per request, so we paginate
start_str = "1 Jan 2017"  # Binance started BTCUSDT trading in 2017
end_str = "now"

# Fetch all historical data
print(f"Fetching historical data for {symbol}...")

klines = client.get_historical_klines(symbol, interval, start_str, end_str)

# Convert to DataFrame
btc_usdt_df = pd.DataFrame(klines, columns=[
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "trades", "taker_base", "taker_quote", "ignore"
])

# Convert timestamp to readable date
btc_usdt_df["timestamp"] = pd.to_datetime(btc_usdt_df["timestamp"], unit="ms")

# Save as CSV
btc_usdt_df.to_csv("BTCUSDT_historical_data.csv", index=False)

print(f"Data saved to BTCUSDT_historical_data.csv with {len(btc_usdt_df)} rows.")

