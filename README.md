# Stock Price Direction Predictor

An end-to-end machine learning pipeline that tries to predict whether a stock's **next-day closing price will go UP or DOWN** using **live market data** from Yahoo Finance, technical indicators, and an XGBoost classifier, served through an interactive Streamlit dashboard with SHAP explanations.

**Headline result: the models do not beat a coin flip.** That is the expected outcome for next-day direction from public price/volume indicators, and the pipeline is built to measure it honestly rather than hide it. See [Results](#results).

## Highlights

- **Live data** — pulls daily OHLCV data via `yfinance`
- **18 scale-free features** — returns, distance from moving averages, RSI, Stochastic, MACD (% of price), Bollinger %B and width, ATR (% of price), volume ratio, OBV trend
- **Time-based train/test split** — the most recent 20% is held out; the scaler is fit on training data only
- **3 models compared** — Logistic Regression, Random Forest, XGBoost
- **Hyperparameter tuning** — `GridSearchCV` with `TimeSeriesSplit` cross-validation
- **Explainability** — SHAP values show what drove each prediction
- **Interactive dashboard** — Streamlit app with candlestick charts, a live prediction, and an out-of-sample backtest against buy-and-hold

## Results

AAPL, 2 years of daily data, held-out test set of the most recent 91 trading days (run on 2026-09-25; exact numbers change as new data arrives):

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.549 | 0.543 |
| Random Forest | 0.538 | 0.545 |
| XGBoost (default) | 0.549 | 0.533 |
| XGBoost (tuned, saved model) | 0.538 | 0.499 |
| Always predict UP (baseline) | 0.549 | 0.500 |

No model clearly beats the always-UP baseline, and tuning on ~360 training rows makes the test score slightly worse (best CV ROC-AUC 0.526 → test 0.499), which is what fitting noise looks like. The scores are also unstable: Yahoo's adjusted prices differ slightly from one download to the next, and that alone moves the tuned model's test ROC-AUC between roughly 0.50 and 0.53. The model's reported probabilities are also poorly calibrated: the model can output a P(UP) of 85–95% despite having no demonstrated edge. Treat the dashboard as a demo of the engineering, not as a signal.

Design choices that keep the evaluation honest:
- **Scale-free features only.** Raw price levels (moving averages, Bollinger bands, ATR, OBV) change with the stock's price regime and differ by orders of magnitude between tickers, so they are converted to ratios/percentages before modelling.
- **Live prediction uses the latest bar.** Rows whose next-day outcome is unknown are excluded from training, but the prediction is made from the most recent close, not the last labelled row.
- **Backtest is out-of-sample and correctly aligned.** A fresh copy of the model is fit on the first 80% of the chosen ticker's history, and each day-*t* signal is paired with the close-to-close return from *t* to *t+1*.

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
├── tests/
│   └── test_features.py           # Offline feature/target tests
├── models/                        # Saved model, scaler, feature names
├── images/                        # Charts and SHAP visualisations
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone the repo
git clone https://github.com/PrinceOkunade/Stock-Price-Predictor.git
cd Stock-Price-Predictor

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
- Get an UP/DOWN prediction for the next trading day (the saved model is trained on AAPL; tick **Retrain** to fit the chosen ticker)
- View the SHAP explanation for the prediction
- See an out-of-sample backtest (accuracy, ROC-AUC, always-UP baseline, and cumulative return vs buy-and-hold)

### 4. Run the tests
```bash
pytest tests/
```
Offline checks (synthetic data, no network) that the target is next-day direction, the live prediction uses the latest bar, and features are unchanged when prices are rescaled.

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
2. **Engineer** — the `ta` library computes technical indicators, converted to 18 scale-free features
3. **Label** — target = 1 if tomorrow's close > today's close, else 0 (the latest bar has no label and is used only for the live prediction)
4. **Split** — most recent 20% of rows held out as the test set (time-based, no shuffle)
5. **Scale** — `StandardScaler` fit on training data only
6. **Train** — three models trained, XGBoost tuned via `GridSearchCV` + `TimeSeriesSplit`
7. **Explain** — SHAP `TreeExplainer` attributes each prediction to its top features
8. **Serve** — Streamlit app loads the saved model, predicts from the latest close, and backtests on held-out data

## Disclaimer

This is a **project**, not financial advice. Past performance does not guarantee future results. Do not use these predictions to make real trading decisions.

## License

MIT, see [LICENSE](LICENSE).
