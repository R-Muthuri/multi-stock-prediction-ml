# Multi-Stock Prediction Dashboard

A complete end-to-end machine learning project
analyzing and predicting stock price direction
for 6 major technology stocks using Python.

## Stocks Analyzed
- AAPL — Apple Inc.
- MSFT — Microsoft
- GOOGL — Alphabet
- TSLA — Tesla
- AMZN — Amazon
- NVDA — Nvidia

## Project Structure
- aapl_stock_prediction.ipynb — Single stock analysis
- multi_stock_notebook.ipynb  — Multi stock analysis
- aapl_dashboard.py           — Single stock dashboard
- multi_stock_dashboard.py    — Multi stock dashboard

## Analysis Performed
- Exploratory Data Analysis
- Feature Engineering (20 technical indicators)
- Machine Learning Prediction (XGBoost)
- Correlation Analysis
- Volatility Analysis
- Portfolio Optimization (Efficient Frontier)

## Model Performance
| Stock | Accuracy |
|-------|----------|
| NVDA  | 53.11%   |
| AMZN  | 52.01%   |
| AAPL  | 50.92%   |
| GOOGL | 46.89%   |
| TSLA  | 46.89%   |
| MSFT  | 46.52%   |

## Portfolio Optimization Results
| Portfolio      | Return | Volatility | Sharpe |
|----------------|--------|------------|--------|
| Max Sharpe     | 44.61% | 38.44%     | 1.03   |
| Min Volatility | 17.89% | 24.31%     | 0.53   |
| Max Return     | 49.47% | 43.81%     | 1.02   |

## How to Run

### Single Stock Dashboard

streamlit run aapl_dashboard.py


### Multi Stock Dashboard

streamlit run multi_stock_dashboard.py


## Requirements

pip install yfinance ta xgboost scikit-learn
pip install seaborn matplotlib pandas numpy streamlit


## Disclaimer
For educational purposes only — Not financial advice

Data sourced from Yahoo Finance via yfinance
