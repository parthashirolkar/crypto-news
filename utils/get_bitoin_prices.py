from datetime import datetime, timedelta
import yfinance as yf


def get_data(end_date: str):
    ticker = "BTC-USD"

    end_date = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S")
    start_date = end_date - timedelta(7)

    end_date_str = end_date.strftime("%Y-%m-%d")
    start_date_str = start_date.strftime("%Y-%m-%d")

    # Download historical Bitcoin data
    btc_data = yf.download(
        ticker,
        start=start_date_str,
        end=end_date_str,
        progress=False,
        multi_level_index=False,
    )

    return btc_data[["Open", "Close"]]
