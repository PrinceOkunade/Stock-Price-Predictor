import yfinance as yf

def fetch_stock_data(ticker="AAPL", period="2y", interval="1d"):
    data = yf.download(ticker, period=period, interval=interval)
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

if __name__ == "__main__":
      stocks = fetch_multiple_stocks(["AAPL", "MSFT"])
      for ticker, df in stocks.items():
          print(f"{ticker}: {len(df)} rows")





