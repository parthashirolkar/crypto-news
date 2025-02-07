import yfinance as yf



def get_data():
    ticker = 'BTC-USD'

    # Specify the date range
    start_date = '2025-01-22'
    end_date = '2025-01-29'

    # Download historical Bitcoin data
    btc_data = yf.download(ticker, start=start_date, end=end_date, progress=False, multi_level_index=False)
    # btc_data.reset_index(drop=True, inplace=True)

    return btc_data[['Open', 'Close']]
