import requests
import pandas as pd
from datetime import datetime


def fetch_open_interest(symbol, start_date, end_date):
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    data = pd.DataFrame()
    period = '1d'
    limit = 500  # Max limit per API call
    while start_date < end_date:
        # Ensuring the end time does not exceed the current time
        current_end_time = min(end_date, start_date + pd.DateOffset(days=limit - 1))
        params = {
            'symbol': symbol,
            'period': period,
            'limit': limit,
            'startTime': int(start_date.timestamp() * 1000),
            'endTime': int(current_end_time.timestamp() * 1000)
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            batch_data = pd.DataFrame(response.json())
            if not batch_data.empty:
                data = pd.concat([data, batch_data], ignore_index=True)
                last_date = pd.to_datetime(batch_data['timestamp'].iloc[-1], unit='ms')
                start_date = last_date + pd.DateOffset(days=1)
            else:
                break
        else:
            print(f"Failed to fetch data for {symbol}, HTTP status: {response.status_code}, Params: {params}")
            break
    return data


def main():
    # List of symbols to fetch data for
    symbols = ['BTCUSDT', 'ETHUSDT', 'PEPEUSDT', 'WIFUSDT', 'SOLUSDT', 'BONKUSDT',
               'DOGEUSDT', 'SHIBUSDT', 'BOMEUSDT', 'PYTHUSDT', 'JTOUSDT', 'JUPUSDT',
               'WUSDT', 'TIAUSDT', 'FLOKIUSDT', 'PENDLEUSDT', 'RNDRUSDT', 'WLDUSDT',
               'LPTUSDT', 'ORDIUSDT', 'TAOUSDT']

    # Define the date range
    start_date = datetime.strptime('2020-01-01', '%Y-%m-%d')
    end_date = datetime.now()

    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        oi_data = fetch_open_interest(symbol, start_date, end_date)

        if not oi_data.empty:
            filename = f'{symbol}_open_interest_data.csv'
            oi_data.to_csv(filename, index=False)
            print(f"Data fetch complete and saved to {filename}")
        else:
            print(f"No data fetched for {symbol}. Please check the API response and parameters.")


if __name__ == '__main__':
    main()
