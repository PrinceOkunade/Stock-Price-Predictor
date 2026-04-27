End-to-end Stock Price Direction Prediction Machine Learning project


This project demonstrates working with LIVE data via APIs.
The model pulls real-time stock market data, engineers technical indicator features,
trains a classifier to predict whether tomorrow's closing price goes UP or DOWN,
and serves predictions through a Streamlit dashboard that refreshes with live data.

IMPORTANT — Before starting, make sure:
1. All dependencies are installed by running: pip install -r requirements.txt
2. No API key is needed — yfinance is free and keyless.

My project folder structure is:
stock-price-predictor/
├── notebooks/
│   └── stock_prediction.ipynb
├── src/
│   ├── data_fetcher.py
│   ├── feature_engineer.py
│   ├── trainer.py
│   └── app.py
├── models/
├── images/
└── requirements.txt

requirements.txt must contain:
yfinance>=0.2.30
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
shap>=0.42.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.11.0
streamlit>=1.25.0
joblib>=1.2.0
ta>=0.11.0

========================================================
STEP 1 — BUILD THE DATA FETCHER MODULE (src/data_fetcher.py)
========================================================

Build a module that fetches live stock data from yfinance.

FUNCTION 1 — fetch_stock_data(ticker, period, interval):
  - Uses yfinance.download() to pull OHLCV data.
  - Parameters:
      ticker: str — stock symbol e.g. "AAPL" (default "AAPL")
      period: str — how far back e.g. "2y" (default "2y")
      interval: str — candle size e.g. "1d" (default "1d")
  - Drops any rows with NaN.
  - Resets index so Date becomes a column.
  - Prints: ticker, date range, number of rows fetched.
  - Returns: pandas DataFrame with columns Date, Open, High, Low, Close, Volume.

FUNCTION 2 — fetch_multiple_stocks(tickers, period, interval):
  - Calls fetch_stock_data for each ticker in a list.
  - Returns: dict of {ticker: DataFrame}.

FUNCTION 3 — get_latest_price(ticker):
  - Fetches the most recent 5 days of data for a single ticker.
  - Returns: dict with keys: ticker, date, open, high, low, close, volume.
  - Print the latest price info.

Add if __name__ == "__main__" block that demos all 3 functions with AAPL.

========================================================
STEP 2 — BUILD THE FEATURE ENGINEERING MODULE (src/feature_engineer.py)
========================================================

Build a module that creates technical indicator features from OHLCV data.
Use the `ta` library (Technical Analysis) for standard indicators.

FUNCTION — engineer_features(df):
  Takes a DataFrame with Date, Open, High, Low, Close, Volume.
  Adds the following features:

  PRICE FEATURES:
    - returns_1d: (Close - Close.shift(1)) / Close.shift(1) — daily return
    - returns_5d: 5-day rolling return
    - returns_10d: 10-day rolling return
    - log_return: np.log(Close / Close.shift(1))

  MOVING AVERAGES:
    - sma_5: 5-day simple moving average of Close
    - sma_20: 20-day SMA
    - sma_50: 50-day SMA
    - ema_12: 12-day exponential moving average
    - ema_26: 26-day EMA
    - sma_cross: 1 if sma_5 > sma_20, else 0 (golden/death cross signal)

  MOMENTUM INDICATORS (use ta library):
    - rsi_14: 14-day Relative Strength Index — ta.momentum.RSIIndicator(close, window=14)
    - macd: MACD line — ta.trend.MACD(close).macd()
    - macd_signal: MACD signal line — ta.trend.MACD(close).macd_signal()
    - macd_diff: MACD histogram — ta.trend.MACD(close).macd_diff()
    - stoch_k: Stochastic oscillator %K — ta.momentum.StochasticOscillator(high, low, close)

  VOLATILITY INDICATORS:
    - bb_high: Bollinger Band upper — ta.volatility.BollingerBands(close).bollinger_hband()
    - bb_low: Bollinger Band lower — ta.volatility.BollingerBands(close).bollinger_lband()
    - bb_width: (bb_high - bb_low) / sma_20
    - atr_14: Average True Range 14-day — ta.volatility.AverageTrueRange(high, low, close)

  VOLUME INDICATORS:
    - volume_sma_20: 20-day SMA of Volume
    - volume_ratio: Volume / volume_sma_20
    - obv: On-Balance Volume — ta.volume.OnBalanceVolumeIndicator(close, volume)

  TARGET VARIABLE:
    - target: 1 if next day's Close > today's Close, else 0
      Calculated as: (df['Close'].shift(-1) > df['Close']).astype(int)

  After adding all features:
    - Drop rows with NaN (from rolling windows and shift).
    - Drop the last row (target is NaN for the latest day).
    - Print shape, list of all feature columns, and target distribution.
    - Return: DataFrame with all original + engineered columns + target.

Add if __name__ == "__main__" block that fetches AAPL data and prints
engineered features shape and sample.

========================================================
STEP 3 — BUILD THE TRAINING MODULE (src/trainer.py)
========================================================

Build a module that trains and evaluates models.

FUNCTION 1 — prepare_data(df, test_ratio=0.2):
  - Feature columns = all engineered features (NOT Date, Open, High, Low, Close, Volume, target).
  - IMPORTANT: Use a TIME-BASED split, NOT random.
    The most recent test_ratio fraction of rows becomes the test set.
    This prevents look-ahead bias (future data leaking into training).
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx], test = df.iloc[split_idx:]
  - Separate X_train, y_train, X_test, y_test.
  - StandardScaler fit on X_train only, transform both.
  - Print: train size, test size, train target distribution, test target distribution.
  - Return: X_train, X_test, y_train, y_test, scaler, feature_names.

FUNCTION 2 — train_and_evaluate(X_train, y_train, X_test, y_test):
  - Train 3 models:
      Logistic Regression (max_iter=1000, random_state=42)
      Random Forest (n_estimators=200, random_state=42, n_jobs=-1)
      XGBoost (n_estimators=200, max_depth=5, learning_rate=0.1,
               random_state=42, eval_metric='logloss')
  - For each model, print:
      Accuracy, F1-Score, ROC-AUC, classification_report.
  - Return: dict of {name: {'model': model, 'acc': ..., 'f1': ..., 'auc': ...}}

FUNCTION 3 — tune_best_model(X_train, y_train, X_test, y_test):
  - GridSearchCV on XGBoost with:
      param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
      }
    scoring='roc_auc', cv=TimeSeriesSplit(n_splits=5).
    IMPORTANT: Use TimeSeriesSplit, NOT StratifiedKFold,
    because this is time-series data — future must never leak into past.
  - Print best_params_, best_score_.
  - Evaluate best_estimator_ on test set.
  - Return: best_model.

FUNCTION 4 — save_artifacts(model, scaler, feature_names, path='../models/'):
  - joblib.dump model, scaler, feature_names.
  - Print saved paths.

FUNCTION 5 — retrain_pipeline(ticker='AAPL', period='2y'):
  - Calls fetch_stock_data, engineer_features, prepare_data,
    train_and_evaluate, tune_best_model, save_artifacts
    all in sequence — a one-call retraining pipeline.
  - Print timestamp and confirmation.
  - Return: best_model, scaler, feature_names, results_dict.

Add if __name__ == "__main__" block that runs retrain_pipeline('AAPL').

========================================================
STEP 4 — RUN THE NOTEBOOK (notebooks/stock_prediction.ipynb)
========================================================

Execute every cell in order. Use random_state=42 everywhere.

CELL 1 — IMPORTS:
  Import all necessary libraries: pandas, numpy, matplotlib, seaborn, plotly,
  yfinance, ta, sklearn, xgboost, shap, joblib.
  Also import from the src modules: data_fetcher, feature_engineer, trainer.
  Add sys.path.insert(0, '../src') so imports work.
  Print "All libraries loaded successfully."

CELL 2 — DATA FETCHING (LIVE API CALL):
  Fetch 2 years of daily AAPL data using fetch_stock_data.
  Print shape, head(5), tail(5).
  Print: "Data fetched LIVE from Yahoo Finance API at {datetime.now()}".
  This line is the key differentiator from static datasets — highlight it.

CELL 3 — STOCK PRICE VISUALISATION:
  Plot 1 — stock_price_history.png:
    Line chart of Close price over time, with 20-day and 50-day SMA overlaid.
    Use plotly for interactivity, save static version with matplotlib.
  Plot 2 — volume_history.png:
    Bar chart of daily Volume, coloured green (up day) / red (down day).
  Plot 3 — daily_returns_distribution.png:
    Histogram of daily returns with KDE overlay. Mark mean and ±2 std.
  Print one insight below each plot.

CELL 4 — FEATURE ENGINEERING:
  Call engineer_features(df).
  Print shape, all feature names, target distribution.
  Plot 4 — feature_correlation.png:
    Heatmap of top 15 most correlated features with target.
    Use seaborn, figsize=(12, 10), annotate with values.

CELL 5 — TECHNICAL INDICATORS VISUALISATION (save to ../images/):
  Plot 5 — technical_indicators.png:
    3-row subplot:
      Row 1: Close price with Bollinger Bands shaded
      Row 2: RSI with overbought (70) / oversold (30) lines
      Row 3: MACD line, signal line, and histogram bars
    Share x-axis. Show most recent 120 trading days only.

CELL 6 — DATA PREPARATION:
  Call prepare_data(df_featured, test_ratio=0.2).
  Print train/test sizes, date ranges for each split.
  IMPORTANT: Emphasise that the split is TIME-BASED:
  print("TRAIN: {first_train_date} to {last_train_date}")
  print("TEST:  {first_test_date}  to {last_test_date}")
  print("No future data leaks into training — time-based split enforced.")

CELL 7 — MODEL TRAINING AND COMPARISON:
  Call train_and_evaluate(X_train, y_train, X_test, y_test).
  For each model print full classification_report.
  Save to ../images/:
    - model_comparison.png: grouped bar chart of Accuracy, F1, AUC for all 3.
    - roc_curves.png: ROC curves for all 3 models + diagonal.

CELL 8 — HYPERPARAMETER TUNING:
  Call tune_best_model(X_train, y_train, X_test, y_test).
  Print best_params_ and test set metrics.

CELL 9 — EVALUATION:
  Save to ../images/evaluation_detailed.png:
    Left: confusion matrix heatmap with labels Down / Up.
    Right: precision-recall curve.

CELL 10 — SHAP EXPLAINABILITY:
  Use shap.TreeExplainer(best_model). Compute shap_values on X_test.
  Save to ../images/:
    - shap_feature_importance.png: bar summary plot max_display=15
    - shap_beeswarm.png: dot summary plot max_display=15
    - shap_waterfall.png: waterfall for the prediction with highest UP probability.
  Print the prediction details for that sample.

CELL 11 — SAVE ARTIFACTS:
  Call save_artifacts(best_model, scaler, feature_names).
  Print confirmation.

CELL 12 — LIVE PREDICTION DEMO:
  Fetch the latest AAPL data using the API.
  Engineer features for the most recent complete row.
  Scale and predict using the saved model.
  Print:
    "LIVE PREDICTION ({today's date}):"
    "Stock: AAPL"
    "Latest Close: ${close_price}"
    "Predicted Direction: UP/DOWN"
    "Confidence: {probability}%"
    "This prediction was made using data fetched LIVE from Yahoo Finance."

CELL 13 — NOTEBOOK SUMMARY:
  Print a formatted summary block:
    - Data source: Yahoo Finance API (LIVE)
    - Ticker analysed
    - Date range of training data
    - Number of features engineered
    - Best model name and test metrics
    - Top 5 features by SHAP importance
    - Number of images saved
    - Number of model artifacts saved
    - Timestamp of this run

========================================================
STEP 5 — BUILD THE STREAMLIT APP (src/app.py)
========================================================

Build a fully working Streamlit app that fetches LIVE data and makes predictions.

PAGE CONFIG:
  st.set_page_config(page_title="Stock Direction Predictor",
                     page_icon="📈", layout="wide")

LOAD ARTIFACTS:
  Use @st.cache_resource to load model, scaler, feature_names from ../models/.
  Also create shap.TreeExplainer.

HEADER:
  Title: "Stock Price Direction Predictor"
  Subtitle: "Predicts whether tomorrow's closing price goes UP or DOWN using
  live market data from Yahoo Finance + XGBoost + SHAP explanations."
  Show a badge/info: "🟢 LIVE DATA — predictions update with real market data"

SIDEBAR INPUTS:
  Stock Selection:
    - Ticker input: text_input defaulting to "AAPL"
    - Quick-pick buttons for popular stocks: AAPL, GOOGL, MSFT, AMZN, TSLA, NVDA
    - Training period: selectbox ["1y", "2y", "5y"] default "2y"
  
  Model Options:
    - Checkbox: "Retrain model with latest data" (default unchecked)
    - If checked, show a warning: "Retraining takes 2-5 minutes"
  
  A primary "Predict" button at the bottom.

MAIN AREA — shown when Predict is clicked:

  SECTION 1 — LIVE MARKET DATA:
    - Use fetch_stock_data to pull live data for the selected ticker.
    - Show 3 metrics in a row: Latest Close, Day Change (%), 52-Week Range.
    - Show an interactive plotly candlestick chart of the last 60 trading days
      with volume bars underneath.
    - st.caption("Data fetched live from Yahoo Finance at {timestamp}")

  SECTION 2 — FEATURE ENGINEERING:
    - Call engineer_features on the fetched data.
    - Show a collapsible expander "Technical Indicators" with:
        Current RSI, MACD, Bollinger Band position, SMA cross status.

  SECTION 3 — PREDICTION:
    If "Retrain" is checked:
      - Call retrain_pipeline(ticker, period) with a st.spinner.
      - Reload the new model artifacts.
    Else:
      - Use the pre-trained model.
    
    - Build the feature vector for the latest day.
    - Scale with the loaded scaler, using reindex for column alignment.
    - Predict probability.
    
    Row of 3 metrics:
      Prediction: UP or DOWN (with 🟢 or 🔴)
      Confidence: probability as percentage
      Signal Strength: STRONG (>70%) / MODERATE (55-70%) / WEAK (<55%)

  SECTION 4 — SHAP EXPLANATION:
    st.subheader("Why this prediction?")
    - Compute SHAP values for the single input.
    - Display waterfall_plot via st.pyplot(), max_display=10.
    - Markdown: "Red bars push toward UP, blue bars push toward DOWN."

  SECTION 5 — HISTORICAL PERFORMANCE (BACKTEST):
    st.subheader("Model Backtest")
    - Using the test set predictions, show:
      - Accuracy, F1, ROC-AUC as metrics.
      - A line chart comparing: actual cumulative return vs
        model-guided strategy cumulative return over the test period.
        (Model-guided: go long if predicted UP, stay flat if predicted DOWN.)
    - st.caption("Past performance does not guarantee future results.")

  SECTION 6 — FEATURE DASHBOARD:
    st.subheader("Current Technical Indicators")
    - Display a clean st.dataframe() with columns:
      Indicator, Value, Signal (Bullish/Bearish/Neutral).
    - Examples: RSI < 30 → Bullish, RSI > 70 → Bearish, SMA Cross Up → Bullish.

FOOTER:
  st.divider()
  st.caption("Built with yfinance · XGBoost · SHAP · Streamlit")
  st.caption("⚠️ This is an educational project, not financial advice.")

ERROR HANDLING:
  - If model files not found: st.error asking to run notebook or check Retrain.
  - If ticker not found: st.error("Invalid ticker symbol").
  - If market is closed and no new data: st.warning with last available date.

========================================================
STEP 6 — LOCAL DEPLOYMENT & VERIFICATION
========================================================

1. Verify models/ folder has all 3 pkl files.
2. Launch: streamlit run src/app.py
3. Smoke test:
   - Test with AAPL — confirm live price shows, prediction renders.
   - Test with MSFT — confirm switching tickers works.
   - Test with an invalid ticker like "XXXYZ" — confirm error handling.
   - Check the Retrain option with a 1y period — confirm it completes.
   - Confirm SHAP waterfall renders.
   - Confirm backtest chart renders.
4. Print: "Deployment verified. App running at http://localhost:8501"

========================================================
STEP 7 — FINAL FOLDER VERIFICATION
========================================================

Print the complete folder tree:
stock-price-predictor/
├── notebooks/stock_prediction.ipynb   ✓
├── src/data_fetcher.py                ✓
├── src/feature_engineer.py            ✓
├── src/trainer.py                     ✓
├── src/app.py                         ✓
├── models/stock_model.pkl             ✓
├── models/scaler.pkl                  ✓
├── models/feature_names.pkl           ✓
├── images/ (list all .png files)      ✓
└── requirements.txt                   ✓

If any file is missing, rebuild it before finishing.
Do not stop until every file above is confirmed present.
