import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import time

# Initialize Binance API
binance = ccxt.binance({
    'enableRateLimit': True  # Enable rate limit for safe API calls
})

# Define symbols and timeframes 【'PEPE/USDT', 'WIF/USDT', 'SOL/USDT', 'BONK/USDT', 'DOGE/USDT', 'SHIB/USDT', 'BOME/USDT', 'PYTH/USDT', 'JTO/USDT', 'JUP/USDT', 'W/USDT', 'TIA/USDT', 'FLOKI/USDT', 'PENDLE/USDT', 'RNDR/USDT', 'WLD/USDT', 'LPT/USDT']
symbols = ['BTC/USDT', 'ETH/USDT', 'PEPE/USDT', 'WIF/USDT', 'SOL/USDT', 'BONK/USDT', 'DOGE/USDT', 'SHIB/USDT', 'BOME/USDT', 'PYTH/USDT', 'JTO/USDT', 'JUP/USDT', 'W/USDT', 'TIA/USDT', 'FLOKI/USDT', 'PENDLE/USDT', 'RNDR/USDT', 'WLD/USDT', 'LPT/USDT', 'ORDI/USDT', 'TAO/USDT']
timeframes = ['5m', '15m', '30m', '1h', '4h', '8h', '1d', '1w']

# Set an early start date (the API will adjust to the earliest available date)
start_date = '2022-01-01' 

def timeframe_to_milliseconds(timeframe):
    """Convert a Binance timeframe string to milliseconds."""
    amount = int(timeframe[:-1])
    unit = timeframe[-1]
    if unit == 'm':
        return amount * 60000  # minutes to milliseconds
    elif unit == 'h':
        return amount * 3600000  # hours to milliseconds
    elif unit == 'd':
        return amount * 86400000  # days to milliseconds
    elif unit == 'w':
        return amount * 604800000  # weeks to milliseconds
    else:
        raise ValueError("Unsupported timeframe unit")
    
# Loop through each symbol and each timeframe
for symbol in symbols:
    for timeframe in timeframes:
        print(f"Fetching data for {symbol} at timeframe {timeframe}")
        start_time = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_time = binance.milliseconds()
        
        data = []
        limit = 1000
        retries = 5  # Maximum number of retries

        while start_time < end_time and retries > 0:
            try:
                ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=start_time, limit=limit)
                if not ohlcv:
                    break
                data.extend(ohlcv)
                start_time = ohlcv[-1][0] + (timeframe_to_milliseconds(timeframe))  # Move to the next interval
            except ccxt.RequestTimeout as e:
                print("Request timed out. Trying again...")
                retries -= 1
                if retries <= 0:
                    print("Maximum retries exceeded. Exiting.")
                    break
                time.sleep(10)  # Wait for 10 seconds before retrying
        
        # Convert data to DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Save data to CSV file, include token pair in the file name
        filename = f'bn_hisdata_spot_{symbol.replace("/", "_")}_{timeframe}.csv'
        df.to_csv(filename, index=False)

        print(f"Data retrieval complete for {symbol} at {timeframe}. Total data points retrieved: {len(data)}. Data saved to {filename}")
