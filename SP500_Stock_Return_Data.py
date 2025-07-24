import pandas as pd
import yfinance as yf

# Get the list of individual stocks in the S&P 500
sp500_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
data_table = pd.read_html(sp500_URL)[0]
sp500_tickers = data_table['Symbol'].tolist()
sp500_sectors = data_table['GICS Sector'].tolist()


# Create an empty list to store the results
stock_results = []

# Loop through each stock and calculate the returns and volatility
for stock, sector in zip(sp500_tickers, sp500_sectors):
    try:
        # Download the historical price data for the stock
        stock_data = yf.download(stock, start="2015-01-01", end="2025-07-01")

        # Calculate the daily returns
        stock_data['Return'] = stock_data['Close'].pct_change()

        # Resample the data by month and year
        monthly_returns = stock_data['Return'].resample('M').sum()
        yearly_returns = stock_data['Return'].resample('Y').sum()
        yearly_volatility = stock_data['Return'].resample('Y').std() * (252 ** 0.5)

        # Create a new DataFrame with the individual returns and volatility for each month and year
        monthly_returns_df = pd.DataFrame({'Stock': [stock] * len(monthly_returns), 'Date': monthly_returns.index, 'Monthly Return': monthly_returns.values, 'Sector': [sector] * len(monthly_returns)})
        yearly_returns_df = pd.DataFrame({'Stock': [stock] * len(yearly_returns), 'Date': yearly_returns.index, 'Yearly Return': yearly_returns.values, 'Sector': [sector] * len(yearly_returns)})
        yearly_volatility_df = pd.DataFrame({'Stock': [stock] * len(yearly_volatility), 'Date': yearly_volatility.index, 'Yearly Volatility': yearly_volatility.values, 'Sector': [sector] * len(yearly_volatility)})

        # Append the results to the list
        stock_results.append(pd.concat([monthly_returns_df, yearly_returns_df, yearly_volatility_df], ignore_index=True))
    except Exception as e:
        print(f"Error processing stock {stock}: {e}")

# Concatenate the results into a single DataFrame
stock_results_df = pd.concat(stock_results, ignore_index=True)

# Save the results to a new Excel file
stock_results_df.to_excel("sp500_stock_returns.xlsx", index=False)