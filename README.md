# Stock Price Direction Predictor

An end-to-end machine learning project that predicts whether a stock's **next-day closing price will go UP or DOWN** using **live market data** from Yahoo Finance, technical indicators, and an XGBoost classifier, served through an interactive Streamlit dashboard with SHAP explanations.

## Highlights

- **Live data** — pulls real-time OHLCV data via `yfinance` (no API key required)
- **20+ engineered features** — returns, moving averages, RSI, MACD, Bollinger Bands, ATR, OBV, and more
- **Time-based train/test split** — prevents look-ahead bias (no future data leaks into training)
- **3 models compared** — Logistic Regression, Random Forest, XGBoost
- **Hyperparameter tuning** — `GridSearchCV` with `TimeSeriesSplit` cross-validation
- **Explainability** — SHAP values show why each prediction was made
- **Interactive dashboard** — Streamlit app with candlestick charts, predictions, and backtest

## Tech Stack

`Python` · `yfinance` · `pandas` · `scikit-learn` · `XGBoost` · `ta` · `SHAP` · `Streamlit` · `Plotly`

## Project Structure

```
stock-price-predictor/
├── notebooks/
│   └── stock_prediction.ipynb     # Full analysis notebook
├── src/
│   ├── data_fetcher.py            # Live data from Yahoo Finance
│   ├── feature_engineer.py        # Technical indicators
│   ├── trainer.py                 # Model training pipeline
│   └── app.py                     # Streamlit dashboard
├── models/                        # Saved model, scaler, feature names
├── images/                        # Charts and SHAP visualisations
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/stock-price-predictor.git
cd stock-price-predictor

# (Optional) create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Run the notebook
```bash
jupyter notebook notebooks/stock_prediction.ipynb
```
Walks through data fetching, feature engineering, model training, tuning, evaluation, and SHAP explanations end-to-end.

### 2. Train a model from the CLI
```bash
cd src
python trainer.py
```
Runs the full retraining pipeline on AAPL and saves artifacts to `models/`.

### 3. Launch the Streamlit dashboard
```bash
streamlit run src/app.py
```
Then open http://localhost:8501. You can:
- Pick any stock ticker (AAPL, MSFT, NVDA, etc.)
- See live price, candlestick chart, and technical indicators
- Get an UP/DOWN prediction with confidence
- View the SHAP explanation for the prediction
- See a backtest of the model's strategy vs buy-and-hold

## Sample Output

The notebook generates visualisations in `images/`:
- `stock_price_history.png` — price with moving averages
- `technical_indicators.png` — Bollinger Bands, RSI, MACD
- `model_comparison.png` — accuracy / F1 / AUC across models
- `roc_curves.png` — ROC curves for all 3 models
- `shap_beeswarm.png` — feature impact on predictions
- `shap_waterfall.png` — explanation for a single prediction

## How It Works

1. **Fetch** — `yfinance` pulls daily OHLCV data for the chosen ticker
2. **Engineer** — the `ta` library computes ~20 technical indicators
3. **Label** — target = 1 if tomorrow's close > today's close, else 0
4. **Split** — most recent 20% of rows held out as the test set (time-based, no shuffle)
5. **Scale** — `StandardScaler` fit on training data only
6. **Train** — three models trained, XGBoost tuned via `GridSearchCV` + `TimeSeriesSplit`
7. **Explain** — SHAP `TreeExplainer` attributes each prediction to its top features
8. **Serve** — Streamlit app loads the saved model and serves live predictions

## Disclaimer

This is a **project**, not financial advice. Past performance does not guarantee future results. Do not use these predictions to make real trading decisions.

## License

MIT
