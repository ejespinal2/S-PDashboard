import yfinance as yf
import pandas as pd

sp500_ticker = "^GSPC"
sp500_data = yf.download(sp500_ticker, start="2015-01-01", end="2025-07-01")

# Calculate daily returns
sp500_data['Return'] = sp500_data['Close'].pct_change()

# Calculate monthly returns
monthly_returns = sp500_data['Return'].resample('ME' , label='right').sum()

# Calculate yearly returns
yearly_returns = sp500_data['Return'].resample('YE').sum()

# Calculate yearly volatility
yearly_volatility = sp500_data['Return'].resample('YE').std() * (252 ** 0.5)

# Create a new DataFrame with monthly and yearly returns, and yearly volatility
returns_df = pd.DataFrame({'Monthly Returns': monthly_returns, 'Yearly Returns': yearly_returns, 'Yearly Volatility': yearly_volatility})

# Save the returns to am excel spreadsheet
returns_df.to_excel("sp500_returns.xlsx", index=True)
print("Successfully saved returns to sp500_returns.csv")