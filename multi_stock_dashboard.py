# ============================================
# MULTI STOCK PREDICTION DASHBOARD
# ============================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ta
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Page Configuration ──────────────────────
st.set_page_config(
    page_title="Multi-Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# ── Title ────────────────────────────────────
st.title("Multi-Stock Prediction Dashboard")
st.markdown("*Powered by Machine Learning | For educational purposes only*")
st.divider()

# ── Stock Configuration ──────────────────────
STOCKS = {
    'AAPL' : 'Apple Inc.',
    'MSFT' : 'Microsoft',
    'GOOGL': 'Alphabet',
    'TSLA' : 'Tesla',
    'AMZN' : 'Amazon',
    'NVDA' : 'Nvidia',
}

FEATURES = [
    'MA50', 'MA200', 'Returns',
    'Daily_Range', 'MA50_Distance', 'MA200_Distance',
    'Volume_Change', 'Returns_Lag1', 'Returns_Lag2',
    'Returns_Lag3', 'Volatility_5d', 'RSI',
    'MACD', 'MACD_Signal', 'MACD_Diff',
    'BB_High', 'BB_Low', 'BB_Width',
    'ATR', 'Day_of_Week'
]

# ── Load & Cache Data ────────────────────────
@st.cache_data
def load_all_data():
    all_data = {}
    for ticker in STOCKS.keys():
        data = yf.download(ticker, start="2020-01-01")
        if data.columns.nlevels > 1:
            data.columns = [col[0] for col in data.columns]

        # Basic indicators
        data['MA50']    = data['Close'].rolling(50).mean()
        data['MA200']   = data['Close'].rolling(200).mean()
        data['Returns'] = data['Close'].pct_change()

        # Technical indicators
        data['RSI']         = ta.momentum.RSIIndicator(
                              data['Close']).rsi()
        macd                = ta.trend.MACD(data['Close'])
        data['MACD']        = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Diff']   = macd.macd_diff()
        bb                  = ta.volatility.BollingerBands(
                              data['Close'])
        data['BB_High']     = bb.bollinger_hband()
        data['BB_Low']      = bb.bollinger_lband()
        data['BB_Width']    = bb.bollinger_wband()
        data['ATR']         = ta.volatility.AverageTrueRange(
                              data['High'],
                              data['Low'],
                              data['Close']
                              ).average_true_range()

        # Feature engineering
        data['Target']         = (data['Close'].shift(-1) >
                                  data['Close']).astype(int)
        data['Daily_Range']    = data['High'] - data['Low']
        data['MA50_Distance']  = data['Close'] - data['MA50']
        data['MA200_Distance'] = data['Close'] - data['MA200']
        data['Volume_Change']  = data['Volume'].pct_change()
        data['Returns_Lag1']   = data['Returns'].shift(1)
        data['Returns_Lag2']   = data['Returns'].shift(2)
        data['Returns_Lag3']   = data['Returns'].shift(3)
        data['Volatility_5d']  = data['Returns'].rolling(5).std()
        data['Day_of_Week']    = data.index.dayofweek

        all_data[ticker] = data.dropna()
    return all_data

# ── Train & Cache Models ─────────────────────
@st.cache_resource
def train_all_models(all_data):
    results = {}
    for ticker, data in all_data.items():
        X = data[FEATURES]
        y = data['Target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        scaler         = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train_scaled, y_train)
        preds    = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, preds)

        latest        = data[FEATURES].iloc[-1:]
        latest_scaled = scaler.transform(latest)
        prediction    = model.predict(latest_scaled)[0]
        probability   = model.predict_proba(latest_scaled)[0]

        results[ticker] = {
            'model'     : model,
            'scaler'    : scaler,
            'accuracy'  : accuracy,
            'prediction': prediction,
            'prob_up'   : probability[1],
            'prob_down' : probability[0],
            'last_close': data['Close'].iloc[-1],
            'y_test'    : y_test,
            'preds'     : preds
        }
    return results

# ── Load Data & Train ────────────────────────
with st.spinner("Loading data and training models for all stocks..."):
    all_data = load_all_data()
    results  = train_all_models(all_data)

# ── Build Rankings ───────────────────────────
rankings = []
for ticker, result in results.items():
    rankings.append({
        'Ticker'    : ticker,
        'Company'   : STOCKS[ticker],
        'Last Close': f"${result['last_close']:.2f}",
        'Direction' : 'UP' if result['prediction'] == 1
                      else 'DOWN',
        'UP Prob'   : f"{result['prob_up']:.2%}",
        'DOWN Prob' : f"{result['prob_down']:.2%}",
        'Accuracy'  : f"{result['accuracy']:.2%}",
        'Confidence': result['prob_up']
                      if result['prediction'] == 1
                      else result['prob_down']
    })

rankings_df = pd.DataFrame(rankings).sort_values(
              'Confidence', ascending=False
              ).reset_index(drop=True)

# ════════════════════════════════════════════
# SECTION 1: KEY METRICS
# ════════════════════════════════════════════
st.subheader("Key Metrics")

cols = st.columns(6)
for col, (ticker, result) in zip(cols, results.items()):
    col.metric(
        label=ticker,
        value=f"${result['last_close']:.2f}",
        delta=f"{'UP' if result['prediction'] == 1 else 'DOWN'} "
              f"{result['prob_up']:.2%}"
    )

st.divider()

# ════════════════════════════════════════════
# SECTION 2: RANKINGS TABLE
# ════════════════════════════════════════════
st.subheader("Stock Rankings by Prediction Confidence")

st.dataframe(
    rankings_df.drop(columns='Confidence'),
    use_container_width=True,
    hide_index=True
)

st.divider()

# ════════════════════════════════════════════
# SECTION 3: PREDICTION CONFIDENCE CHART
# ════════════════════════════════════════════
st.subheader("Prediction Confidence Comparison")

fig1, ax1 = plt.subplots(figsize=(12, 5))
colors = ['green' if results[t]['prediction'] == 1
          else 'red'
          for t in rankings_df['Ticker']]

bars = ax1.bar(
    rankings_df['Ticker'],
    rankings_df['Confidence'],
    color=colors, alpha=0.7,
    edgecolor='black'
)
ax1.axhline(
    y=0.55, color='blue',
    linestyle='--', alpha=0.7,
    label='55% benchmark'
)
ax1.set_title(
    'Prediction Confidence by Stock',
    fontweight='bold'
)
ax1.set_ylabel('Confidence')
ax1.set_ylim(0, 1)
ax1.legend()

for bar, (_, row) in zip(bars, rankings_df.iterrows()):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        row['Direction'],
        ha='center', fontsize=11,
        fontweight='bold'
    )

st.pyplot(fig1)
st.divider()

# ════════════════════════════════════════════
# SECTION 4: PRICE CHARTS
# ════════════════════════════════════════════
st.subheader("Price Charts with Moving Averages")

fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (ticker, data) in enumerate(all_data.items()):
    axes[idx].plot(
        data.index, data['Close'],
        label='Close', color='steelblue',
        linewidth=1
    )
    axes[idx].plot(
        data.index, data['MA50'],
        label='MA50', color='orange',
        linewidth=1.5
    )
    axes[idx].plot(
        data.index, data['MA200'],
        label='MA200', color='green',
        linewidth=1.5
    )
    axes[idx].set_title(
        f"{ticker} — {STOCKS[ticker]}",
        fontweight='bold'
    )
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig2)
st.divider()

# ════════════════════════════════════════════
# SECTION 5: CORRELATION HEATMAP
# ════════════════════════════════════════════
st.subheader("Returns Correlation Matrix")

returns_all = pd.DataFrame({
    ticker: data['Returns']
    for ticker, data in all_data.items()
})
returns_corr = returns_all.corr()

fig3, ax3 = plt.subplots(figsize=(10, 7))
sns.heatmap(
    returns_corr,
    annot=True, fmt='.2f',
    cmap='RdYlGn',
    vmin=-1, vmax=1,
    center=0, square=True,
    linewidths=0.5, ax=ax3
)
ax3.set_title(
    'Stock Returns Correlation Matrix',
    fontweight='bold'
)
st.pyplot(fig3)
st.divider()

# ════════════════════════════════════════════
# SECTION 6: VOLATILITY CHART
# ════════════════════════════════════════════
st.subheader("Annualized Volatility Comparison")

volatility = returns_all.std() * (252 ** 0.5) * 100
vol_df = pd.DataFrame({
    'Ticker'    : volatility.index,
    'Annual Vol': volatility.values
}).sort_values('Annual Vol', ascending=False)

fig4, ax4 = plt.subplots(figsize=(10, 5))
colors_vol = [
    'red' if v > 50 else
    'orange' if v > 30 else
    'green'
    for v in vol_df['Annual Vol']
]
bars_vol = ax4.bar(
    vol_df['Ticker'],
    vol_df['Annual Vol'],
    color=colors_vol, alpha=0.7,
    edgecolor='black'
)
ax4.axhline(
    y=30, color='green',
    linestyle='--', alpha=0.7,
    label='Low volatility (30%)'
)
ax4.axhline(
    y=50, color='red',
    linestyle='--', alpha=0.7,
    label='High volatility (50%)'
)
for bar, val in zip(bars_vol, vol_df['Annual Vol']):
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f'{val:.1f}%',
        ha='center', fontweight='bold'
    )
ax4.set_title(
    'Annualized Volatility by Stock',
    fontweight='bold'
)
ax4.set_ylabel('Volatility (%)')
ax4.legend()
st.pyplot(fig4)
st.divider()

# ════════════════════════════════════════════
# SECTION 7: PORTFOLIO OPTIMIZATION
# ════════════════════════════════════════════
st.subheader("Portfolio Optimization — Efficient Frontier")

returns_matrix = returns_all.dropna()
mean_returns   = returns_matrix.mean()
cov_matrix     = returns_matrix.cov()
n_days         = 252
n_portfolios   = 3000
n_stocks       = len(returns_matrix.columns)
risk_free_rate = 0.05

np.random.seed(42)
port_returns    = np.zeros(n_portfolios)
port_volatility = np.zeros(n_portfolios)
port_sharpe     = np.zeros(n_portfolios)
port_weights    = np.zeros((n_portfolios, n_stocks))

for i in range(n_portfolios):
    weights            = np.random.random(n_stocks)
    weights            = weights / np.sum(weights)
    port_weights[i]    = weights
    p_return           = np.sum(mean_returns * weights) * n_days
    p_vol              = np.sqrt(np.dot(
                         weights.T,
                         np.dot(cov_matrix * n_days, weights)
                         ))
    port_returns[i]    = p_return
    port_volatility[i] = p_vol
    port_sharpe[i]     = (p_return - risk_free_rate) / p_vol

sim_df          = pd.DataFrame({
    'Return'    : port_returns,
    'Volatility': port_volatility,
    'Sharpe'    : port_sharpe
})
max_sharpe_port = sim_df.iloc[sim_df['Sharpe'].idxmax()]
min_vol_port    = sim_df.iloc[sim_df['Volatility'].idxmin()]

fig5, ax5 = plt.subplots(figsize=(10, 6))
scatter = ax5.scatter(
    sim_df['Volatility'],
    sim_df['Return'],
    c=sim_df['Sharpe'],
    cmap='viridis',
    alpha=0.5, s=10
)
plt.colorbar(scatter, ax=ax5, label='Sharpe Ratio')
ax5.scatter(
    max_sharpe_port['Volatility'],
    max_sharpe_port['Return'],
    color='red', marker='*',
    s=300, zorder=5,
    label='Max Sharpe'
)
ax5.scatter(
    min_vol_port['Volatility'],
    min_vol_port['Return'],
    color='blue', marker='*',
    s=300, zorder=5,
    label='Min Volatility'
)
ax5.set_title(
    'Efficient Frontier',
    fontweight='bold'
)
ax5.set_xlabel('Annual Volatility')
ax5.set_ylabel('Annual Return')
ax5.legend()
ax5.grid(True, alpha=0.3)
st.pyplot(fig5)

# Portfolio metrics
col1, col2 = st.columns(2)
col1.metric(
    label="Max Sharpe Return",
    value=f"{max_sharpe_port['Return']:.2%}",
    delta=f"Volatility: {max_sharpe_port['Volatility']:.2%}"
)
col2.metric(
    label="Min Volatility Return",
    value=f"{min_vol_port['Return']:.2%}",
    delta=f"Volatility: {min_vol_port['Volatility']:.2%}"
)

st.divider()

# ════════════════════════════════════════════
# SECTION 8: MODEL ACCURACY COMPARISON
# ════════════════════════════════════════════
st.subheader("Model Accuracy Comparison")

accuracy_df = pd.DataFrame({
    'Ticker'  : list(results.keys()),
    'Accuracy': [r['accuracy'] for r in results.values()]
}).sort_values('Accuracy', ascending=False)

fig6, ax6 = plt.subplots(figsize=(10, 5))
colors_acc = [
    'green' if a > 0.55 else
    'orange' if a > 0.50 else
    'red'
    for a in accuracy_df['Accuracy']
]
bars_acc = ax6.bar(
    accuracy_df['Ticker'],
    accuracy_df['Accuracy'],
    color=colors_acc, alpha=0.7,
    edgecolor='black'
)
ax6.axhline(
    y=0.55, color='blue',
    linestyle='--', alpha=0.7,
    label='55% benchmark'
)
ax6.axhline(
    y=0.50, color='red',
    linestyle='--', alpha=0.7,
    label='50% random chance'
)
for bar, val in zip(bars_acc, accuracy_df['Accuracy']):
    ax6.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.002,
        f'{val:.2%}',
        ha='center', fontweight='bold'
    )
ax6.set_title(
    'XGBoost Model Accuracy by Stock',
    fontweight='bold'
)
ax6.set_ylabel('Accuracy')
ax6.set_ylim(0, 0.75)
ax6.legend()
st.pyplot(fig6)
st.divider()

# ════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════
st.markdown("""
---
### Disclaimer
This dashboard is for **educational purposes only**

It is **not financial advice**

Always do your own research before making investment decisions

*Data sourced from Yahoo Finance via yfinance*

*Models: XGBoost trained on 20 technical indicators*
""")