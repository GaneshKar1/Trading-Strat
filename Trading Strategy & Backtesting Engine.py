import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller

start_date = '2020-01-01'
end_date = '2024-01-01'
tickers = ['WMT', 'TGT']
data = yf.download(tickers, start=start_date, end=end_date)['Close'] # Changed 'Adj Close' to 'Close'

x = data['TGT']
y = data['WMT']
x_const = sm.add_constant(x)

# Use a 60-day rolling window for the hedge ratio to remove forward-looking bias
window_ols = 60
model = RollingOLS(y, x_const, window=window_ols)
rolling_res = model.fit()
hedge_ratios = rolling_res.params['TGT']

spread = data['WMT'] - hedge_ratios * data['TGT']

window = 30
spread_mean = spread.rolling(window=window).mean()
spread_std = spread.rolling(window=window).std()
z_score = (spread - spread_mean) / spread_std

entry_threshold = 2.0
exit_threshold = 0.0

positions = pd.DataFrame(index=z_score.index)
positions['WMT'] = 0
positions['TGT'] = 0

long_spread = z_score < -entry_threshold
short_spread = z_score > entry_threshold
exit_spread = abs(z_score) < 0.5

positions['signal'] = np.select(
    [long_spread, short_spread],
    [1, -1],
    default=0
)

positions['signal'] = positions['signal'].replace(0, np.nan).ffill().fillna(0)

daily_returns = data.pct_change()

# Calculate raw strategy returns
strategy_returns = positions['signal'].shift(1) * (daily_returns['WMT'] - hedge_ratios.shift(1) * daily_returns['TGT'])

# --- ADD TRANSACTION COSTS ---
# A trade occurs whenever the signal changes (from 0 to 1, 1 to 0, -1 to 0, etc.)
cost_per_trade = 0.001  # 0.1% per trade
trades = positions['signal'].diff().fillna(0).abs()
transaction_costs = trades * cost_per_trade

# Subtract costs from returns
strategy_returns = strategy_returns - transaction_costs
# ------------------------------

plt.figure(figsize=(12, 6))
plt.title("Z-Score of WMT/TGT Spread")
z_score.plot(label='Z-Score')
plt.axhline(entry_threshold, color='r', linestyle='--', label='Sell Threshold')
plt.axhline(-entry_threshold, color='g', linestyle='--', label='Buy Threshold')
plt.axhline(0, color='black', linestyle='-', label='Mean')
plt.legend()
plt.show()

cumulative_strategy_returns = (1 + strategy_returns).cumprod()
cumulative_wmt_returns = (1 + daily_returns['WMT']).cumprod()

plt.figure(figsize=(12, 6))
plt.title("Cumulative Returns: Strategy vs Holding WMT")
cumulative_strategy_returns.plot(label='Pairs Trading Strategy', color='blue')
cumulative_wmt_returns.plot(label='Buy & Hold WMT', color='gray', alpha=0.5)
plt.legend()
plt.show()

sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
print(f"Strategy Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Latest Hedge Ratio: {hedge_ratios.iloc[-1]:.4f}")



