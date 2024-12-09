import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone

# Initialize Binance API
binance = ccxt.binance({
    'apiKey': 'Qe7IW111ZxNXTgzBm9WGijDNvt4nkIZ9E1zJEX8UKTFtucblZTV4lIFCHUNXkbVa',
    'secret': '50uLXUExoUPyTBKx2UR2yBMp7qRqRUeWVMwpfIA38O9aI0fWc0XNfao5lWcccnjb',
    'enableRateLimit': True,  # Optional, enables rate limiting
    'timeout': 30000
})

# Define symbol and timeframe
symbols = ['BTC/USDT', 'ETH/USDT', 'PEPE/USDT', 'WIF/USDT', 'SOL/USDT', 'BONK/USDT', 'DOGE/USDT', 'SHIB/USDT', 'BOME/USDT', 'PYTH/USDT', 'JTO/USDT', 'JUP/USDT', 'W/USDT', 'TIA/USDT', 'FLOKI/USDT', 'PENDLE/USDT', 'RNDR/USDT', 'WLD/USDT', 'LPT/USDT', 'ORDI/USDT', 'TAO/USDT']  # Example symbol (Bitcoin to USDT)
timeframe = '1m'     # Minute-level data

# Set end time to now and start time to January 1, 2022
end_time = datetime.now(timezone.utc)  # Ensure end_time is timezone-aware
start_time = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

for symbol in symbols:
    data = []
    current_start_time = start_time

    # Fetch data in chunks until we reach the end time
    while current_start_time < end_time:
        since = int(current_start_time.timestamp() * 1000)
        print(f"Fetching data for {symbol} from {current_start_time} to {end_time}...")
        ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if len(ohlcv) == 0:
            break
        data.extend(ohlcv)
        last_timestamp = ohlcv[-1][0]
        current_start_time = datetime.fromtimestamp(last_timestamp / 1000, tz=timezone.utc) + timedelta(minutes=1)  # Move to the next minute after the last timestamp

    # Convert data to DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Save data to CSV file
    filename = f'binance_historical_data_{symbol.replace("/", "_")}.csv'
    df.to_csv(filename, index=False)
    print(f"Data retrieval for {symbol} complete. Total data points retrieved: {len(data)}. Saved to {filename}")
