"""Module 1: Fetches live stock data from Yahoo Finance API."""
import yfinance as yf


def _flatten_columns(data):
    """Flatten MultiIndex columns returned by newer yfinance versions."""
    if isinstance(data.columns, __import__('pandas').MultiIndex):
        data.columns = [col[0] if col[1] == "" else col[0] for col in data.columns]
    return data


def fetch_stock_data(ticker="AAPL", period="2y", interval="1d"):
    data = yf.download(ticker, period=period, interval=interval)
    data = _flatten_columns(data)
    data = data.dropna()
    data = data.reset_index()
    print(f"Ticker: {ticker}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"Rows: {len(data)}")
    return data


def fetch_multiple_stocks(tickers, period="2y", interval="1d"):
    results = {}
    for ticker in tickers:
        results[ticker] = fetch_stock_data(ticker, period, interval)
    return results


def get_latest_price(ticker="AAPL"):
    data = yf.download(ticker, period="5d", interval="1d")
    data = _flatten_columns(data)
    latest = data.iloc[-1]
    result = {
        "ticker": ticker,
        "date": str(data.index[-1].date()),
        "open": round(float(latest["Open"]), 2),
        "high": round(float(latest["High"]), 2),
        "low": round(float(latest["Low"]), 2),
        "close": round(float(latest["Close"]), 2),
        "volume": int(latest["Volume"]),
    }
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result


if __name__ == "__main__":
    df = fetch_stock_data("AAPL")
    print(df.head())
    print()
    stocks = fetch_multiple_stocks(["AAPL", "MSFT"])
    for t, d in stocks.items():
        print(f"{t}: {len(d)} rows")
    print()
    get_latest_price("AAPL")
