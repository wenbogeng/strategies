import ccxt
import csv
from datetime import datetime, timezone, timedelta


def fetch_oi_history(exchange, symbol, start_date, timeframe='1h'):
    # Converting start date to milliseconds since epoch
    start_time = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)

    # Ensure start_time is within the last 30 days
    current_time = exchange.milliseconds()
    thirty_days_ago = current_time - 30 * 24 * 60 * 60 * 1000
    if start_time < thirty_days_ago:
        start_time = thirty_days_ago  # Set start_time to 30 days ago if it is older

    end_time = current_time  # Current time in milliseconds

    data = []  # Initialize data as an empty list
    try:
        # Fetch open interest history
        fetched_data = exchange.fetch_open_interest_history(symbol, timeframe, start_time, limit=500,
                                                            params={'endTime': end_time})
        if fetched_data:
            data.extend(fetched_data)
        else:
            print(f"No more data returned for {symbol}.")
    except Exception as e:
        print(f"Failed to fetch data for {symbol}: {e}")
        return None  # Ensure returning None or an empty list here

    if data:  # Check if data is not empty
        return data
    return None  # Return None if no data was added to the list


def save_to_csv(data, filename):
    if not data:
        print(f"No data to save for {filename}")
        return

    # Define the CSV column headers
    headers = ['timestamp', 'datetime', 'symbol', 'baseVolume', 'quoteVolume', 'openInterestAmount',
               'openInterestValue']

    # Open file in write mode
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)  # Write the header row

        # Write data rows
        for entry in data:
            csvwriter.writerow([
                entry['timestamp'],
                entry['datetime'],
                entry['symbol'],
                entry['baseVolume'],
                entry['quoteVolume'],
                entry['openInterestAmount'],
                entry['openInterestValue']
            ])


def main():
    # Initialize CCXT Binance connection with rate limit enabled
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',  # Ensure to use the correct type (futures)
        }
    })

    # List of symbols to fetch data for
    symbols = ['BTC/USDT', 'ETH/USDT', 'PEPE/USDT', 'WIF/USDT', 'SOL/USDT', 'BONK/USDT', 'DOGE/USDT', 'SHIB/USDT',
               'BOME/USDT', 'PYTH/USDT', 'JTO/USDT', 'JUP/USDT', 'W/USDT', 'TIA/USDT', 'FLOKI/USDT', 'PENDLE/USDT',
               'RNDR/USDT', 'WLD/USDT', 'LPT/USDT', 'ORDI/USDT', 'TAO/USDT']

    # Define start date
    start_date = '2020-01-01'  # Define start date

    # Fetch open interest history for each symbol
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        data = fetch_oi_history(exchange, symbol, start_date, '1h')
        if data:
            print(f"Data retrieved successfully for {symbol}.")
            for entry in data:
                print(entry)

            # Save data to CSV
            filename = f'oi_history_{symbol.replace("/", "_")}.csv'
            save_to_csv(data, filename)
            print(f"Data saved to {filename}")
        else:
            print(f"No data retrieved for {symbol}.")


if __name__ == '__main__':
    main()
